"""The two modes over an SGLang server, so the router need not hold the GPU.

`SGLangScorer` runs the engine in this process, which pins the router to the
machine serving the model. The server accepts `return_hidden_states` and returns
them under `meta_info`, so the split works: the engine on whatever holds the GPU,
and the head, a few hundred kilobytes on a CPU, beside the browser or wherever
the decision has to be made.

Same interface as the in-process scorer and as the transformers one, so
`SchemaDecisionEngine` and `StepScorer` take any of the three.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from inference.sglang_backend import SGLangScorer


@dataclass(frozen=True)
class SGLangServerSettings:
    """Where the server is, and what the head needs from it.

    `feature_layers` must be `(-1,)`: the server returns the last hidden layer and
    no other, so a head that reads deeper cannot be served here. Saying so when
    the scorer is built beats reading one layer's vector as though it were
    another's, which fails silently.
    """

    base_url: str
    model_id: str
    revision: str = "main"
    feature_layers: tuple[int, ...] = (-1,)
    batch_questions: bool = False
    max_input_tokens: int = 16384
    timeout: float = 600.0
    engine_kwargs: dict = field(default_factory=dict)


class _HttpEngine:
    """The slice of the Engine API the scorer uses, over HTTP."""

    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt, sampling_params=None, return_logprob=False,
                 token_ids_logprob=None, return_hidden_states=False):
        payload = {"text": prompt, "sampling_params": sampling_params,
                   "return_logprob": return_logprob,
                   "return_hidden_states": return_hidden_states}
        if token_ids_logprob is not None:
            payload["token_ids_logprob"] = token_ids_logprob
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"SGLang server at {self.base_url} refused the request: "
                f"{error.code} {error.read()[:300].decode('utf-8', 'replace')}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"no SGLang server at {self.base_url}: {error.reason}. Start one with "
                f"python -m sglang.launch_server --model-path <model> "
                f"--enable-return-hidden-states") from error


class SGLangServerScorer(SGLangScorer):
    """`SGLangScorer` with the engine on the far end of a socket."""

    @classmethod
    def load(cls, settings):
        from transformers import AutoTokenizer

        if tuple(settings.feature_layers) not in ((), (-1,)):
            raise ValueError(
                f"the server returns only the last hidden layer; this head reads "
                f"{list(settings.feature_layers)}. Train one with "
                f"litjev-train-head --candidate layer_-1.")
        engine = _HttpEngine(settings.base_url, settings.timeout)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id,
                                                  revision=settings.revision)
        hidden_size = cls._hidden_size(settings.model_id, settings.revision)
        return cls(engine, tokenizer, settings, hidden_size)

    def health(self):
        """Ask the server for one token, so a misconfiguration surfaces at startup."""
        output = self.engine.generate("hello", sampling_params={"max_new_tokens": 1,
                                                                "temperature": 0.0},
                                      return_hidden_states=bool(self.settings.feature_layers))
        meta = output.get("meta_info", {})
        if self.settings.feature_layers and "hidden_states" not in meta:
            raise RuntimeError(
                "the server answered but returned no hidden states. Launch it with "
                "--enable-return-hidden-states.")
        return {"prompt_tokens": meta.get("prompt_tokens"),
                "hidden_states": "hidden_states" in meta}
