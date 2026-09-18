import pytest

from litjev.schema import Choice, DecisionSchema
from litjev.slots import compile_prefix_slots


class CodeTokenizer:
    """Model a tokenizer whose spaced letter codes are single vocabulary entries."""

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        return text + ("<assistant>" if kwargs.get("add_generation_prompt") else "")

    def encode(self, text, **kwargs):
        head, marker, tail = text.rpartition("Answer:")
        prefix = [ord(c) for c in head + marker]
        if tail:
            if tail.startswith(" ") and tail[1:].isalpha():
                return prefix + [10000 + sum(ord(c) * 26**i for i, c in enumerate(tail[1:]))]
            return prefix + [ord(c) for c in tail]
        return prefix


def test_255_external_keys_get_distinct_single_token_codes():
    schema = DecisionSchema(
        {"secret": Choice(criteria={f"multi token {i}": None for i in range(255)})}
    )
    compiled = compile_prefix_slots(CodeTokenizer(), schema, [1], max_input_tokens=50000)
    assert len(set(compiled.candidates[0])) == 255
    assert compiled.candidate_codes[0][:3] == ["A", "B", "C"]
    assert "secret" not in compiled.slot_texts[0]


def test_tokenizer_without_supported_boundary_fails_closed():
    class Unsupported:
        def encode(self, text, **kwargs):
            return list(text.encode())

    schema = DecisionSchema({"q": Choice(criteria={"A": None})})
    with pytest.raises(ValueError, match="single-token"):
        compile_prefix_slots(Unsupported(), schema, [1])
