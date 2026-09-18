import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from litjev.backend import TransformersScorer
from litjev.schema import Choice, DecisionSchema


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        return text + ("<assistant>" if kwargs.get("add_generation_prompt") else "")

    def encode(self, text, **kwargs):
        # A simple boundary-preserving character tokenizer, no downloads.
        return [ord(char) + 2 for char in text]


@pytest.mark.parametrize("architecture", ["llama", "qwen3_5"])
def test_two_forwards_match_original_independent_branches(architecture):
    torch.manual_seed(4)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()
    if architecture == "qwen3_5":
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
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
            },
            image_token_id=250,
            video_token_id=251,
        )
        model = Qwen3_5ForConditionalGeneration(config).eval()

    # Use labels without a leading space for this artificial character tokenizer.
    class BoundaryTokenizer(TinyTokenizer):
        def encode(self, text, **kwargs):
            return super().encode(text.replace(": ", ":"), **kwargs)

    scorer = TransformersScorer(model, BoundaryTokenizer())
    schema = DecisionSchema(
        {
            f"q{i}": Choice(instructions=f"Pick {i}", criteria={"A": None, "B": None})
            for i in range(1, 11)
        }
    )
    calls = []
    hook = model.register_forward_pre_hook(
        lambda m, a, kw: calls.append(kw["input_ids"].shape), with_kwargs=True
    )
    scores = scorer.score("state", schema)
    hook.remove()
    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 10
    compiled = scorer._compile("state", schema)
    assert all(text.endswith("</end><assistant>Answer:") for text in compiled.slot_texts)
    assert all('"q1"' not in text for text in compiled.slot_texts)
    evidence = scores[0].provenance
    assert evidence["module"] == "lm_head"
    assert evidence["last_decoder_layer_index"] == 1
    assert evidence["absolute_position"] == compiled.positions[0]
    assert evidence["candidate_token_ids"] == compiled.candidates[0]
    assert len(compiled.input_ids) == 10
    with torch.inference_mode():
        for i, row in enumerate(compiled.input_ids):
            # Each batch row must match its complete independent input, despite padding.
            full = model(torch.tensor([row]), use_cache=False).logits
            expected = full[0, -1, compiled.candidates[i]]
            np.testing.assert_allclose(scores[i].logits, expected.numpy(), atol=1e-6)
            assert row[compiled.prefix_length :] == compiled.slot_ids[i]
            assert compiled.positions[i] == len(row) - 1
    assert len({len(row) for row in compiled.input_ids}) > 1
