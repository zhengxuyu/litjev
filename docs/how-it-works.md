# How it works

LitJev reads candidate scores from the model's output head and builds typed
responses in Python: no generated JSON, no answer-text parsing, and no generated
answer tokens.

1. **Shared prefill:** encode only the state and generic instructions once.
2. **Cached branches:** replicate KV/recurrent states and batch each question's
   instructions and criteria in a second forward pass, ending with `Answer:`.
   Original option keys map to internal letter codes (A–Z, then eligible
   AA…ZZ/AAA…ZZZ), verified as single tokens at this boundary. Unsupported
   tokenizers are rejected, never truncated.
3. **Readout:** take `output.logits[i, len(suffix_ids[i]) - 1, candidate_ids[i]]`.
   For Qwen, these are vocabulary-head (`lm_head`) logits after the final decoder
   normalization, at the final input position (the colon for Qwen), predicting
   a space-prefixed internal code token. These codes are not generated.
4. **Typed response:** normalize candidate logits, select the maximum, and build
   JSON in code. No autoregressive answer generation is performed.

![Pass one prefills the shared state; pass two reads each answer at Answer:](images/prefill-readout.svg)

With `--backend sglang` the same two passes run on SGLang: each question is
sent as the shared prefix plus its own suffix, the engine's prefix cache skips
the prefix after the first question, and the candidate-code logprobs come back
through `token_ids_logprob`. See [the README](../README.md#serving-through-sglang).

Branches cannot see one another's questions or results. Renaming a question ID
does not change its model input. This schema migration changes the prompt compared
with the original catalog-based version, so historical accuracy, calibration profiles
and latency numbers must not be applied to it without re-evaluation.

## Relationship to Jev

LitJev is a hypothesis-based reproduction of Jev's decision layer from public
information. It is not a connection to the official Jev API, and it does not
reproduce Jev's training or proprietary internals.

- `/v1/systemone` follows the public `model / state / questions` request and typed
  response shapes, including structured criteria. See the [migration matrix](jev-schema.md).
- Jev's exact confidence formula is not published in the referenced docs. LitJev
  uses normalized Gini concentration: `(K * sum(p_i²) - 1) / (K - 1)` for K > 1,
  and 1 for a single option. Uniform distributions yield 0, point masses yield 1.
  This is **not max probability**, not an accuracy estimate, and not a claim of
  numerical parity with Jev.
- No RLCD training, learned correctness head, or one-forward guarantee is provided.

References: [TypeSafe documentation](https://docs.typesafe.ai/introduction) and the
[RLCD reference implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).
These are independent external projects, not endorsements.

## Calibration

Probabilities are **not calibrated by default**. Debug diagnostics report
`calibration_fitted=false`, and normalization does not establish calibration.
Optional post-hoc temperature fitting is available:

```bash
# With the uncalibrated server running:
uv run litjev-mmlu --split validation --limit 70 --export-logits validation-logits.jsonl
uv run litjev-calibrate validation-logits.jsonl --model Qwen/Qwen3.8-27B --output calibration.json
# Stop the old server, then restart with the profile:
uv run litjev --model Qwen/Qwen3.8-27B --calibration calibration.json
```

Never fit on test labels. Keep model revision, prompt, option order, and evaluation
protocol fixed. A fitted temperature does not guarantee calibration on another
population or safe out-of-distribution routing. Do not use this preview as the sole
authority for consequential actions.

## Requirements and hardware

- Python 3.11+ and uv.
- Tested hardware: one NVIDIA H100 80 GB, BF16. Budget roughly 54 GB for 27B BF16
  weights alone, plus runtime overhead and branch caches. An 80 GB GPU is the tested
  starting point for 27B, not a general minimum; smaller Qwen checkpoints need less.
- Linux uses the locked PyTorch CUDA 12.8 distribution and needs a compatible driver.
  Model downloads need additional disk space and network access.
- `--device-map auto` is the default. Multi-GPU/offload configurations are not
  benchmarked here. CPU-only 27B inference is not a supported performance target.

Server options: `--model`, `--revision`, `--dtype` (`bfloat16`, `float16`,
`float32`), `--device-map`, `--port`, `--calibration`, `--backend` (`transformers`,
`sglang`). Run `uv run litjev --help`.
