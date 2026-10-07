# Frozen v6 generation-time schema study

384 extractions compare original and XGrammar constrained decoding across the same 32 three-turn validation conversations and two context tracks. The original prompt is identical for both arms. Weights, user script, gold labels, strict validator and greedy settings are unchanged.

The grammar uses the existing exported schema, full model vocabulary and actual generation EOS IDs. XGrammar produces CPU token masks; the explicit torch_native kernel applies masks to CUDA logits, avoiding a Triton dependency on Windows. One fresh matcher per request. Object keys follow schema order; this restriction was selected during unscored contract checks, before this protocol was frozen. Duplicate slot IDs and JSON encoded inside candidate_value remain strict application checks. No postprocessing repair or retry.

See protocol.json for hashes, criteria and limitations. This is an exploratory development screen. No training, sealed final labels, paid APIs or default inference changes.
