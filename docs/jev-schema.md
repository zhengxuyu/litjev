# Jev schema migration

Prompt format `user_question_codes_v3` closes the shared state user turn without
opening an assistant turn. Each cached branch then supplies its question/options
in a new user turn, followed by the assistant `Answer:` prefix. Questions remain
independent; inference still uses two forwards and zero generated tokens.
Calibration profiles from `isolated_question_codes_v2` must be refitted.

Contract checked against the public [HTTP reference](https://docs.typesafe.ai/api),
[structured entries](https://docs.typesafe.ai/primitives/advanced), and
[Score reference](https://docs.typesafe.ai/primitives/score) on 2026-09-17.
The advanced reference and SDK permit structured/null entries beyond the shorter
examples in the HTTP reference.

| Layer | Before | Current |
| --- | --- | --- |
| HTTP | `/v1/calibrated-schema`, `/v1/batch-mcq` | `/v1/systemone` |
| Input | `state`, `schema` | `model`, `state`, `questions` |
| Question | `enum/boolean`, `description`, `choices` | `choice/score/noul`, `instructions`, `criteria` |
| Option IDs | Required a one-token label | Arbitrary keys; single-token letter-code mapping stays internal |
| Question IDs | Included in model prompt | Only used to associate answers |
| Isolation | All questions in shared prefix | State-only prefix, one question per cached branch |
| Result | `value`, `gamma`, logits mixed together | Type-specific answers and token usage |
| Diagnostics | In standard answer | Separate `/v1/systemone/debug` envelope |

The public Python entry points use the same question objects: `Choice`, `Score`,
`Noul`, `DecisionSchema`, `SystemOneRequest`. No legacy schema parser remains.
MMLU-Pro preparation, games, frontend and benchmark scripts use this contract.
Dataset labels remain outside requests.

## Deliberate differences from the hosted service

- Model: `litjev` resolves to the loaded local checkpoint. Unknown model names fail;
  we do not impersonate TypeSafe weights or expose their model catalog.
- Confidence: normalized Gini concentration, explicitly a LitJev approximation.
  Exact Jev numerical confidence and its learned calibration are not reproduced.
- Inference: two top-level forwards, zero generated output tokens. This is not a
  reproduction of proprietary training or a one-forward architecture.
- Images, timings and raw logits: only in the debug extension. Standard output
  has exactly `model`, `answers`, `usage`.
- Deployment: no TypeSafe auth, quota, hosted billing or overload service. Keep local
  or behind an authenticated proxy. Context limits still apply; no silent truncation.

## Breaking changes

Old endpoints return 404; old fields at the standard endpoint return 422.
Re-prepare MMLU request files with `litjev-mmlu --prepare-only`; archived reports are
not silently rewritten. Refit calibration after this prompt change. Old performance
measurements remain historical and do not validate the new prompt.

The playground example is the user's ten-question Choice example, also available
as `examples/request.json`. Contract tests keep the packaged frontend copy aligned.

## Migration verification

- 36 Python tests: contract validation, all three answer types, ten-question batching,
  ID isolation, arbitrary option keys, cached/full-input equivalence on tiny Llama
  and Qwen models, image M-RoPE, games, calibration version checks.
- Two Playwright tests: the actual playground assets send the new request and render
  Choice/Score/Noul/timing correctly with a fixture response; film replay still works.
- Actual Qwen checkpoint tokenizer, no weights loaded: 255 distinct single-token
  letter codes verified against the complete 255-option suffix (5,527 tokens in the
  test prompt). A–J token IDs: `[357, 417, 351, 414, 458, 426, 469, 462, 353, 604]`.
  Decimal numbers 10 and above split into tokens, which is why letter codes are used.
- Wheel and source distribution build successfully. These checks do not measure
  pretrained 27B answer accuracy or GPU latency; a new cluster benchmark is still needed.
