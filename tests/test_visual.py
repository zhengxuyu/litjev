import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from test_api import FakeEngine
from test_backend import TinyTokenizer
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from litjev.api import create_app
from litjev.backend import TransformersScorer
from litjev.decision import SchemaDecisionEngine
from litjev.schema import Choice, DecisionSchema
from litjev.vision import VisualState, decode_image, encode_image


class ImageTokenizer(TinyTokenizer):
    def encode(self, text, **kwargs):
        return super().encode(text.replace(": ", ":"), **kwargs)


class TinyProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is False
        assert messages[-1]["content"][1]["type"] == "image"
        pixels = np.asarray(messages[-1]["content"][1]["image"])
        return {
            "input_ids": torch.tensor([[10, 250, 250, 250, 250, 11]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0]]),
            "pixel_values": torch.full((16, 24), float(pixels.mean()) / 255),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


def tiny_visual_model():
    torch.manual_seed(7)
    return Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config={
                "vocab_size": 256,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 32,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
            },
            image_token_id=250,
            video_token_id=251,
        )
    ).eval()


def test_image_two_forwards_match_full_input_and_pixels_affect_logits():
    model = tiny_visual_model()
    scorer = TransformersScorer(model, ImageTokenizer(), processor=TinyProcessor())
    schema = DecisionSchema(
        {
            name: Choice(instructions=f"Pick {name}", criteria={"A": None, "B": None})
            for name in ["a", "longer"]
        }
    )
    state = VisualState("Look at the screen", Image.new("RGB", (8, 8), "white"))
    calls, vision_calls = [], []
    hook = model.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
    vision_hook = model.model.visual.register_forward_hook(lambda *args: vision_calls.append(1))
    scores = scorer.score(state, schema)
    hook.remove()
    vision_hook.remove()
    assert len(calls) == 2 and len(vision_calls) == 1
    assert "pixel_values" in calls[0] and "pixel_values" not in calls[1]
    assert calls[1]["position_ids"].shape[0] == 3
    assert calls[1]["position_ids"][0, 0, 0].item() == 4  # prefix 6 + M-RoPE delta -2
    with torch.inference_mode():
        for i, score in enumerate(scores):
            suffix = torch.tensor([score.provenance["slot_token_ids"]])
            inputs = {k: v for k, v in calls[0].items() if k not in {"use_cache", "logits_to_keep"}}
            inputs["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(suffix)], dim=1
            )
            expected = model(**inputs, use_cache=False).logits[
                0, -1, score.provenance["candidate_token_ids"]
            ]
            np.testing.assert_allclose(score.logits, expected.numpy(), atol=1e-6)
    other = scorer.score(VisualState(state.text, Image.new("RGB", (8, 8), "black")), schema)
    assert not np.allclose(scores[0].logits, other[0].logits)
    # An image request must not contaminate subsequent text-only cache positions.
    text_after = scorer.score("text", schema)
    fresh = TransformersScorer(model, ImageTokenizer()).score("text", schema)
    np.testing.assert_array_equal(text_after[0].logits, fresh[0].logits)
    assert SchemaDecisionEngine(scorer).decide(state, schema).usage.output_tokens == 0


def test_image_api_and_validation_happen_before_model_load():
    seen = []

    class RecordingEngine(FakeEngine):
        def evaluate(self, state, schema):
            seen.append(state)
            return super().evaluate(state, schema)

    client = TestClient(create_app(lambda: RecordingEngine()))
    payload = {
        "state": "Look",
        "model": "litjev",
        "questions": {
            "a": {"type": "choice", "instructions": "Pick", "criteria": {"A": None, "B": None}}
        },
        "image": encode_image(np.zeros((8, 8, 3), np.uint8)),
    }
    assert client.post("/v1/systemone/debug", json=payload).status_code == 200
    assert client.post("/v1/systemone", json=payload).status_code == 422
    assert isinstance(seen[0], VisualState)
    assert seen[0].image.size == (8, 8)
    for invalid in [
        "not base64",
        "https://example.com/image.png",
        "data:image/svg+xml;base64,AAAA",
    ]:
        payload["image"] = invalid
        assert client.post("/v1/systemone/debug", json=payload).status_code == 422
    assert len(seen) == 1


def test_image_codec_checks_shape_size_and_format():
    frame = np.zeros((12, 16, 3), np.uint8)
    np.testing.assert_array_equal(np.asarray(decode_image(encode_image(frame))), frame)
    with pytest.raises(ValueError):
        encode_image(np.zeros((12, 16, 4), np.uint8))
    with pytest.raises(ValueError):
        decode_image("A" * 6_000_000)
