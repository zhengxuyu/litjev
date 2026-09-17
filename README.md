# LitJev

**An implementation of our hypothesis about Jev.** Built with Hugging Face
Transformers, LitJev implements a JEV-like decision layer that adapts off-the-shelf
LLMs into typed decision APIs, with a browser playground and HTTP server included.

这是我们对 Jev 实现方式的猜想实现：基于 Hugging Face Transformers 构建
JEV-like 决策层，将 off-the-shelf 大模型快速接入 JEV-like API。
这一猜想来自公开信息，不代表 Jev 的真实内部架构；当前首先保证 Qwen 可用。

Evaluate up to ten multiple-choice or boolean fields per request. LitJev reads
candidate scores from the model's output head and builds typed responses in Python:
no generated JSON, no answer-text parsing, and no generated answer tokens.

> Independent research project, not affiliated with or endorsed by TypeSafe AI.
> Not the official Jev implementation or an API-compatible replacement today.
> First supported target: `Qwen/Qwen3.8-27B`. Probabilities are **not calibrated by
> default**. Other checkpoints are not guaranteed to work.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), obtain this
repository, and run from its root:

```bash
git clone https://github.com/zhengxuyu/litjev.git
cd litjev
uv run --locked litjev --model Qwen/Qwen3.8-27B
```

Open **http://127.0.0.1:8000/**. This command installs dependencies and starts both
the playground and API. Load the example, then submit. The first inference request
downloads/loads the model and can take several minutes. Later requests reuse it.
Opening the page does not load weights.

For an existing checkpoint:

```bash
uv run --locked litjev --model /path/to/qwen-checkpoint --device-map cuda:0
```

**Not yet published to PyPI.** The commands above run this checkout. Do not assume
`uvx litjev` or `pip install litjev` installs this project before a release.

### Requirements

- Python 3.11+ and uv.
- Tested hardware: one NVIDIA H100 80 GB, BF16. Budget roughly 54 GB for 27B BF16
  weights alone, plus runtime overhead and branch caches. An 80 GB GPU is the tested
  starting point, not a general minimum.
- Linux uses the locked PyTorch CUDA 12.8 distribution and needs a compatible driver.
  Model downloads need additional disk space and network access.
- `--device-map auto` is the default. Multi-GPU/offload configurations are not
  benchmarked here. CPU-only 27B inference is not a supported performance target.

Options: `--model`, `--revision`, `--dtype` (`bfloat16`, `float16`, `float32`),
`--device-map`, `--port`, `--calibration`. Run `uv run litjev --help` for usage.
The server binds to loopback only, with one process owning the model.

## HTTP API

Interactive API documentation: **http://127.0.0.1:8000/docs**.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/calibrated-schema \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

Example request:

```json
{
  "state": "Answer each question using its listed options.",
  "schema": {
    "math": {
      "type": "enum",
      "description": "What is 2 + 3?\nA. 4\nB. 5\nC. 6",
      "choices": ["A", "B", "C"]
    },
    "planet": {
      "type": "enum",
      "description": "Which planet is closest to the Sun?\nA. Venus\nB. Mars\nC. Mercury",
      "choices": ["A", "B", "C"]
    }
  }
}
```

This is a **field mapping, not JSON Schema**. Requests contain 1–10 fields.
`enum` requires a description and unique choices; `boolean` requires a description
and uses `true`/`false` candidates. Every choice must encode as a single token at the
answer boundary. Multi-token labels are rejected, not truncated. Use A–J and put
option meanings in the description.

Each entry in `answers` includes:

| Field | Meaning |
| --- | --- |
| `value` | Selected choice string, or boolean |
| `probabilities` | Softmax distribution over supplied candidates |
| `gamma` | Maximum candidate probability, not a validated correctness estimate |
| `logits` | Raw candidate logits in choice order |
| `provenance` | Output-head position, token IDs, suffix, and temperature |

The response also includes `model`, `usage`, `calibration_fitted`, and `timing`.
This backend reports `usage.forward_calls=2` and `usage.output_tokens=0`.

Other endpoints:

- `GET /health`: service status and whether weights have loaded.
- `POST /v1/batch-mcq`: exactly ten `questions`, each with `question_id`, `prompt`,
  and `options` (a label-to-description mapping). Used by the benchmark CLI.

The playground displays probability bars, JSON, logit provenance, and timing.
`model_setup_seconds` includes first-use loading; `decision_seconds` includes
tokenization, inference-lock waiting, and inference. `total_seconds` sums these
server phases. Browser round-trip time also includes transport. These are not
GPU-kernel-only timings.

## Python usage

```python
from dataclasses import asdict

from litjev.backend import ModelSettings, TransformersScorer
from litjev.decision import SchemaDecisionEngine
from litjev.schema import DecisionSchema

model = "Qwen/Qwen3.8-27B"
scorer = TransformersScorer.load(ModelSettings(model_id=model))
engine = SchemaDecisionEngine(scorer, model_id=model)
schema = DecisionSchema.from_mapping({
    "math": {
        "type": "enum",
        "description": "What is 2 + 3? A. 4 B. 5",
        "choices": ["A", "B"],
    }
})
result = engine.decide("Choose the correct answer.", schema)
print(result.answers["math"].value)
print(asdict(result))
```

## How it works

1. **Shared prefill:** encode the state and complete question catalog once.
2. **Cached branches:** replicate KV/recurrent states and batch each field's short
   `Field "key"\nAnswer:` suffix in a second forward pass.
3. **Readout:** take `output.logits[i, len(suffix_ids[i]) - 1, candidate_ids[i]]`.
   For Qwen, these are vocabulary-head (`lm_head`) logits after the final decoder
   normalization, at the colon position predicting the next token.
4. **Typed response:** normalize candidate logits, select the maximum, and build
   JSON in code. No autoregressive answer generation is performed.

Branches cannot see one another's suffixes or results. **Each still sees all
questions in the shared catalog**; batch composition can affect answers. Question
IDs enter the prompt. This is not fully isolated question evaluation.

### Relationship to Jev

LitJev is inspired by typed decision APIs, but does not reproduce Jev's training
or proprietary internals. In this release:

- Requests use `state` + `schema`, with `enum`/`boolean` fields.
- Jev-style `questions`, `criteria`, Choice/Score/Noul, and `/v1/systemone` are
  **not implemented yet**.
- `gamma` is max softmax probability, not Jev's confidence statistic.
- No RLCD training, learned correctness head, or one-forward guarantee is provided.

References: [TypeSafe documentation](https://docs.typesafe.ai/introduction) and the
[RLCD reference implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).
These are independent external projects, not endorsements.

## Calibration

Default responses have `calibration_fitted=false`. Normalization does not establish
calibration. Optional post-hoc temperature fitting is available:

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

## Evaluation

With the server running:

```bash
uv run litjev-mmlu --split test --limit 10 --output mmlu-results.json
```

Downloads `TIGER-Lab/MMLU-Pro`, sends ten questions per request, and scores labels
locally. Gold answers and CoT explanations never enter inference prompts. This is
**direct-answer scoring, not the standard CoT benchmark protocol**. `--prepare-only`
exports requests without inference; `--revision COMMIT` pins the dataset snapshot.
Incomplete final batches use duplicate padding excluded from metrics. Invalid or
non-MCQ records are skipped with reasons.

A ten-question smoke experiment on one H100 80 GB achieved 9/10 and approximately
0.472 s per warm ten-question request (mean of three repeats). This is not a full
MMLU-Pro score or a latency guarantee. Repeated questions are not extra test examples;
loading, queueing, and HTTP overhead are excluded from this measurement.

See [Slurm usage](docs/slurm.md) for configurable cluster launchers. Raw local
experiment outputs are excluded from the public distribution because they contain
machine paths, hostnames, and dataset text.

## Doom demo

`litjev-doom` plays [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) through the same
decision path as the playground. **The agent never sees the screen.** Each step reads the
engine's symbolic state — health, ammunition, and the name, bearing and distance of every
visible object — renders it as one text block, and submits a schema with a single `action`
field. Its choices are single-token letters standing for the moves the scenario's buttons
allow; the chosen letter maps back to a button vector. No answer tokens are generated.

```bash
uv sync --locked --extra doom
uv run --locked litjev-doom --model Qwen/Qwen3.8-27B
```

Open **http://127.0.0.1:8000/doom** and press start. The page shows the frame each decision was
made on, the exact text the model received, and the probability the output head assigned to
every move. The playground and the `/v1/*` endpoints stay available on the same port and share
the one loaded model.

Options: `--scenario` (default `defend_the_center`), `--tics` per decision, `--resolution`,
`--max-actors`, `--record DIR`, `--window`, plus every option `litjev` accepts. Scenarios come
from the installed ViZDoom package, which ships Freedoom assets; no original Doom WAD is needed.

The game runs in ViZDoom's synchronous `PLAYER` mode, so it waits for each decision and slow
inference never drops frames. `--record DIR` writes one `.lmp` per episode. Those replay at full
speed and at any resolution, which is the practical way to capture video of a run that was
played slower than real time:

```python
game.replay_episode("recordings/episode-000.lmp")
while not game.is_episode_finished():
    game.advance_action()
```

Small checkpoints run the same path, which is enough to develop against without a GPU host.
On an Apple M-series laptop (48 GB unified memory, `--device-map mps --dtype float16`),
`Qwen/Qwen3-4B` produced warm decisions in roughly 0.17–0.18 s from prompts of about 235 input
tokens in `defend_the_center`. That is one observation of this loop on one machine, not a latency
guarantee, and small checkpoints are not a supported accuracy target.

Limits: input is text only, so nothing in this demo reads pixels; `labels` reports what the
engine considers visible, which is not identical to what a human would notice on screen; and
play quality is not evaluated here against any baseline.

## Development

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check .
uv build
```

Tests use fake engines and tiny randomly initialized Transformers models, not 27B
weights. They verify two-forward cache scoring, schemas, API behavior, and calibration
utilities. Browser assets are bundled in the wheel.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[release checklist](docs/releasing.md). No authentication, request quota, or public
hosting service is included. Keep the server local or use an authenticated reverse
proxy before granting remote access.

## License

[Apache License 2.0](LICENSE) for this repository's original code. Model weights,
datasets, and third-party dependencies retain their own licenses and are not bundled
here.

When redistributing LitJev or derivative works, comply with Apache-2.0 Section 4:
provide a copy of the license, mark modified files, retain applicable source notices,
and reproduce applicable attribution from [NOTICE](NOTICE) in a permitted location.
These requirements concern redistribution; they do not require every private use or
hosted API response to display a citation. The full license controls.

## Citation

If you use LitJev in research, benchmarks, publications, or a project, please cite
it and acknowledge its contribution using the entry below. Please carry this
attribution into your project's documentation or acknowledgments. This citation
request is not an additional condition of the Apache-2.0 license; redistribution
must still preserve the applicable notices described above.

如果你在研究、评测、论文或项目中使用 LitJev，请携带以下引用，在文档或致谢中
注明来源。再分发时必须按 Apache-2.0 保留适用的许可证和 NOTICE 归属声明；
学术引用是项目请求，不是对 Apache-2.0 另加的限制。

```bibtex
@software{litjev2026,
  author  = {{ZhengxuYu}},
  title   = {LitJev: A JEV-like Decision Layer for Off-the-Shelf LLMs},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/zhengxuyu/litjev}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
