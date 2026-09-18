# LitJev

**A reproduction of Jev: turn any Qwen model into a fast decision model.**

LitJev reproduces the decision layer of [Jev](https://docs.typesafe.ai/introduction)
on top of off-the-shelf Hugging Face checkpoints. Define questions and options, load a
Qwen model, and get typed choices with probability distributions from a single API
call. No training, no generated answer text: scores are read directly from the
model's output head.

**LitJev 是对 Jev 的复现：把 Qwen 全系列模型变成像 Jev 一样的快速决策模型。**
定义问题和选项，加载模型，即可通过 API 或浏览器前端获取选择结果与概率分布，
无需训练，也无需生成回答文本。

> Independent research project, not affiliated with or endorsed by TypeSafe AI.
> This is a hypothesis-based reproduction from public information, not the official
> Jev implementation or a connection to its API. Request/response JSON follows the
> public Jev schema; internals, confidence values and performance are not identical.
> Probabilities are not calibrated by default.

## Supported models

The full Qwen family is supported (Qwen3.x text and vision checkpoints, any size).
The default and most-tested checkpoint is `Qwen/Qwen3.8-27B` on one H100 80 GB.
Vision checkpoints are required for screenshot decisions. Other model families are
not guaranteed to work.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/zhengxuyu/litjev.git
cd litjev
uv run --locked litjev --model Qwen/Qwen3.8-27B
```

Open **http://127.0.0.1:8000/**, load the example, and submit. The first request
downloads and loads the model, which can take several minutes. Use
`--model /path/to/checkpoint` for a local checkpoint and `--device-map cuda:0` to pin
a GPU. Not yet published to PyPI; the commands above run this checkout.

## Using the API

LitJev uses **exactly the schema defined in the Jev documentation**: the same
`POST /v1/systemone` path, the same `model / state / questions` request body, the
same `choice`, `score` and `noul` question types, and the same `answers` and `usage`
response. Code written for one can talk to the other by changing the base URL.

Send a request with `curl` (the server must be running):

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

The `state` is what the model looks at; each entry in `questions` is one decision
about that state:

```json
{
  "model": "litjev",
  "state": "Customer writes: my order arrived broken and I need it replaced today.",
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "Money back", "replace": "A replacement",
                            "info": "Just information"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent", "Urgent", "Critical"]},
    "escalate": {"type": "noul", "instructions": "Should a human take over?"}
  }
}
```

The response returns one typed answer per question ID:

```json
{
  "model": "Qwen/Qwen3.8-27B",
  "answers": {
    "intent": {"type": "choice", "choice": "replace",
               "probabilities": {"refund": 0.12, "replace": 0.85, "info": 0.03},
               "confidence": 0.71},
    "urgency": {"type": "score", "score": 1.6,
                "legend": {"0": "Not urgent", "1": "Urgent", "2": "Critical"},
                "probabilities": {"0": 0.05, "1": 0.30, "2": 0.65}, "confidence": 0.42},
    "escalate": {"type": "noul", "noul": 0.78}
  },
  "usage": {"input_tokens": 96, "output_tokens": 0}
}
```

`choice` returns the winning option key, `score` the probability-weighted level
index, and `noul` the probability of yes. Values above are illustrative. Up to ten
questions fit in one request. From Python:

```python
import requests

response = requests.post("http://127.0.0.1:8000/v1/systemone", json=request_body)
answers = response.json()["answers"]
if answers["escalate"]["noul"] > 0.5:
    hand_off_to_human()
```

### Switching between LitJev and the Jev API

Because the schema is identical, the same request body works against the hosted
Jev service. Only three things differ:

| | LitJev (local) | Jev (hosted) |
| --- | --- | --- |
| URL | `http://127.0.0.1:8000/v1/systemone` | `https://api.typesafe.ai/v1/systemone` |
| Auth header | none | `Authorization: Bearer <API_KEY>` |
| `model` | `litjev` or the loaded checkpoint ID | `jev-latest` |

Point your client at the other URL, set the header and model name, and everything
else stays the same. Prototype locally on your own GPU, then switch to Jev, or the
other way round. The numeric probabilities and confidence values will differ
between the two, since LitJev runs a different model; only the contract is shared.

## Modules

- **Decision API** — `POST /v1/systemone`, described above. Interactive docs at
  `/docs`, full reference in [HTTP API](docs/api.md).
- **Playground** — browser UI at `/` with probability bars, logits and timings.
- **Benchmarking** — MMLU-Pro direct-answer scoring plus Doom and chess played from
  screenshots, with a `/film` replay viewer. See [benchmarking](docs/benchmarking.md).
- **Calibration** — optional post-hoc temperature fitting. See
  [how it works](docs/how-it-works.md#calibration).

## Documentation

- [HTTP API and Python usage](docs/api.md)
- [How it works, relationship to Jev, calibration, hardware](docs/how-it-works.md)
- [Benchmarking: MMLU-Pro, Doom, chess](docs/benchmarking.md)
- [Visual games interface](docs/visual-games.md)
- [Jev schema migration](docs/jev-schema.md)
- [Slurm usage](docs/slurm.md)
- [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Releasing](docs/releasing.md)

## License

[Apache License 2.0](LICENSE) for original code; adapted jevlike examples retain
their [MIT license](THIRD_PARTY_LICENSES/jevlike-MIT.txt), see
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights and datasets keep their
own licenses. When redistributing, follow Apache-2.0 Section 4 and reproduce
[NOTICE](NOTICE).

## Citation

If you use LitJev in research or a project, please cite it
([CITATION.cff](CITATION.cff)):

```bibtex
@software{litjev2026,
  author  = {{ZhengxuYu}},
  title   = {LitJev: A Jev Decision Layer for Off-the-Shelf LLMs},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/zhengxuyu/litjev}
}
```
