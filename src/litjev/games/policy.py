"""One visual policy for any discrete environment with textual action descriptions."""

import json
import time
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from litjev.schema import DecisionSchema
from litjev.vision import VisualState, encode_image


class LocalDecisionClient:
    def __init__(self, engine):
        self.engine = engine
        self.model_id = engine.model_id

    def decide(self, state, schema, image):
        return asdict(
            self.engine.evaluate(
                VisualState(state, Image.fromarray(image)) if image is not None else state,
                DecisionSchema.from_mapping(schema),
            )
        )


class HttpDecisionClient:
    def __init__(self, url="http://127.0.0.1:8000", timeout=600):
        self.url = url.rstrip("/") + "/v1/systemone/debug"
        self.timeout = timeout
        self.model_id = None

    def decide(self, state, schema, image):
        payload = {"state": state, "model": "litjev", "questions": schema}
        if image is not None:
            payload["image"] = encode_image(image)
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self.model_id = result["result"]["model"]
        return result


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    probabilities: tuple[float, ...]
    confidence: float
    logits: tuple[float, ...]
    provenance: dict
    usage: dict
    latency_seconds: float
    calibration_fitted: bool


class LitJevPolicy:
    def __init__(self, client, action_names, instructions):
        if not 2 <= len(action_names) <= 255 or len(set(action_names)) != len(action_names):
            raise ValueError("Visual policy requires 2–255 unique action names")
        self.client = client
        self.action_names = tuple(action_names)
        self.instructions = instructions
        self.labels = self.action_names
        self.schema = {
            "action": {
                "type": "choice",
                "instructions": "Which controller button should be pressed next?",
                "criteria": {name: None for name in self.action_names},
            }
        }

    def decide(self, observation):
        if isinstance(observation, str):
            state = {"task": self.instructions, "observation": observation}
            image = None
        else:
            if observation.dtype != np.uint8 or observation.ndim != 3 or observation.shape[-1] != 3:
                raise ValueError("Policy requires an RGB uint8 observation or scene text")
            state, image = self.instructions, observation
        started = time.perf_counter()
        evaluation = self.client.decide(state, self.schema, image)
        result = evaluation["result"]
        diagnostics = evaluation["diagnostics"]
        elapsed = time.perf_counter() - started
        answer = result["answers"]["action"]
        if answer["choice"] not in self.labels:
            raise ValueError("Model returned an action outside the runtime action set")
        probabilities = answer["probabilities"]
        if set(probabilities) != set(self.labels):
            raise ValueError("Model returned a mismatched action distribution")
        values = tuple(float(probabilities[label]) for label in self.labels)
        if not all(np.isfinite(p) and 0 <= p <= 1 for p in values) or not np.isclose(
            sum(values), 1
        ):
            raise ValueError("Invalid action probabilities")
        return PolicyDecision(
            self.labels.index(answer["choice"]),
            values,
            float(answer["confidence"]),
            tuple(diagnostics["fields"]["action"]["logits"]),
            diagnostics["fields"]["action"]["provenance"],
            {**result["usage"], "forward_calls": diagnostics["forward_calls"]},
            elapsed,
            bool(diagnostics["calibration_fitted"]),
        )
