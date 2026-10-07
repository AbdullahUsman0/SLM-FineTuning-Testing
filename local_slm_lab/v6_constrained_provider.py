"""Research-only original-prompt v6 decoding with an enforced structural schema.

The model stays on CUDA. XGrammar's CPU matcher produces the token bitmask;
its explicit torch_native backend applies that mask on CUDA without Triton.
The existing strict application validator remains the acceptance gate.
"""
from __future__ import annotations

import importlib.metadata
import json
from time import perf_counter

from forecasting_assistant.domain.models import ExtractorResult
from local_slm_lab.structured_output_providers import _attempt, _combined_trace
from local_slm_lab.v5_provider import V5TransformersProvider

BACKEND = "xgrammar==0.2.7/torch_native"
COMPILE_OPTIONS = {"strict_mode": True, "any_order": False, "any_whitespace": True}


class NativeGrammarLogitsProcessor:
    """One batch-one grammar matcher per generation; mask the full model vocabulary."""

    def __init__(self, compiled):
        import xgrammar as xgr
        self.xgr = xgr
        self.vocab_size = compiled.tokenizer_info.vocab_size
        self.matcher = xgr.GrammarMatcher(compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        self.prefilled = False
        self.steps = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1 or scores.shape != (1, self.vocab_size):
            raise RuntimeError("Constrained study requires batch one and full vocabulary coverage")
        if self.prefilled:
            if self.matcher.is_terminated():
                raise RuntimeError("Generation continued after the grammar stop token")
            if not self.matcher.accept_token(input_ids[0, -1].item()):
                raise RuntimeError("A generated token was outside the grammar")
        else:
            self.prefilled = True
        self.matcher.fill_next_token_bitmask(self.bitmask)
        self.xgr.apply_token_bitmask_inplace(
            scores, self.bitmask.to(scores.device), vocab_size=self.vocab_size,
            backend="torch_native",
        )
        self.steps += 1
        return scores


class V6ConstrainedProvider(V5TransformersProvider):
    """Toggle only grammar masking; both arms share the original prompt and model."""

    name = "v6_original_prompt_optional_xgrammar"
    constraints_enabled = False

    def initialize_constraints(self, output_schema):
        import xgrammar as xgr
        if importlib.metadata.version("xgrammar") != "0.2.7":
            raise RuntimeError("The study pins xgrammar==0.2.7")
        started = perf_counter()
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        # Qwen3.5 is multimodal: top-level config.vocab_size is not authoritative.
        self.constraint_vocab_size = self._model.get_output_embeddings().weight.shape[0]
        stop_ids = self._model.generation_config.eos_token_id
        if stop_ids is None:
            stop_ids = tokenizer.eos_token_id
        self.constraint_stop_ids = stop_ids
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.constraint_vocab_size, stop_token_ids=stop_ids,
        )
        compiler = xgr.GrammarCompiler(info, max_threads=4)
        self.compiled_extractor = compiler.compile_json_schema(
            json.dumps(output_schema), **COMPILE_OPTIONS,
        )
        self.schema_compile_ms = (perf_counter() - started) * 1000

    def _structured(self, operation, system, user, output_type):
        if not self.constraints_enabled:
            try:
                return super()._structured(operation, system, user, output_type)
            finally:
                if self._traces:
                    self._traces[-1].update(model_calls=1, retry_count=0, retry_success=False,
                                            constraint_backend="none", schema_compile_ms=0)
        if operation != "extract" or output_type is not ExtractorResult:
            raise RuntimeError("This study constrains only extraction")
        processor = NativeGrammarLogitsProcessor(self.compiled_extractor)
        result, trace = _attempt(self, system, user, output_type, logits_processor=[processor])
        self._traces.append(_combined_trace(
            operation, [trace], result, constraint_backend=BACKEND,
            schema_compile_ms=self.schema_compile_ms,
            mask_steps=processor.steps, mask_backend="torch_native",
        ))
        if result is None:
            raise RuntimeError("Constrained extraction failed the unchanged validator")
        return result

    async def extract(self, message, state):
        self._torch.cuda.synchronize()
        try:
            return await super().extract(message, state)
        finally:
            self._torch.cuda.synchronize()
