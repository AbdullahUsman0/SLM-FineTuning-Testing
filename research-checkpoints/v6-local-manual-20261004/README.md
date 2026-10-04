# V6 transfer and local manual smoke — October 4, 2026

The release ZIP was downloaded and its SHA-256 independently matched
`0f1133277540017e0c440d6dd1f55c3eeb9692cc3e9787c088a1412c98a41827`.
The ZIP contents and the actual adapter files pulled in commit `91960bc` both
matched the original completion record's size and SHA-256 values.

The adapter was converted to F16 GGUF with the existing llama.cpp b11026
converter and pinned Qwen base configuration. All 372 LoRA tensors converted;
the local CPU server loaded the adapter successfully alongside the existing
Q4_K_M 2B stock model. See [runtime-manifest.json](runtime-manifest.json) for
artifact hashes. This records the local base GGUF hash; it does not establish
exact underlying HF weight-revision equivalence.

Hardware: Intel Core Ultra 5 125U, 12 cores / 14 logical processors, 32 GB RAM.
Backend: llama.cpp CPU, eight threads, one server slot, context 8192,
greedy generation, repetition penalty 1, seed 42, maximum 1024 output tokens,
thinking disabled. The NPU and Intel GPU were not used.
The pinned fpy commit was exported to an ignored local snapshot without
changing the sibling checkout. All 31 Python source-file hashes matched the
training launch record byte for byte.

[transcript.json](transcript.json) retains two illustrative manual turns:

1. An explicitly phrased weather request returned valid JSON and valid output
   schema, with intent plus nine separate non-intent updates: target meaning,
   Celsius unit, daily frequency, seven-day horizon, target/time columns,
   upload mode, CSV format and filename.
2. A follow-up correction returned valid JSON/schema, marked
   `correction_detected: true`, and changed the horizon to 14 days. The
   pipeline's displayed state retained the earlier facts.

Server-reported first-call time was 52.642 seconds: 24.933 seconds prompt
processing plus 27.710 seconds generation, approximately 17.47 generated
tokens/second. The correction took 35.671 seconds. These are single-call
observations with a cached local base and adapter, not latency distributions.
The chat launcher was also checked with `/quit`; it verified adapter hashes
and saved a transcript successfully.

This is quantized local smoke evidence, not full behavioral validation, a
matched BF16 model comparison, or a generalization score. Neither informal
phrasing nor the whole weather/economics mixture was evaluated in this check.
No sealed final labels or paid API were used. The previously recorded corpus
review/overlap limitations still apply.

Run on the prepared laptop with `scripts/chat-v6-local.ps1`. If its background
server is stopped, start `scripts/start-v6-local.ps1` in another terminal first.
See [the test guide](../../V6_MANUAL_TEST_GUIDE.md) for cases and setup context.
