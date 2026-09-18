"""Compile independent question branches sharing only the state prefix."""

from dataclasses import dataclass
from itertools import product
from string import ascii_uppercase

from litjev.prompting import build_decision_messages, render_question_branch

SLOT_FORMAT = "user_question_codes_v3"


@dataclass(frozen=True)
class CompiledSlots:
    input_ids: list[list[int]]
    positions: list[int]
    candidates: list[list[int]]
    slot_texts: list[str]
    slot_ids: list[list[int]]
    prefix_text: str
    prefix_length: int
    candidate_codes: list[list[str]]


def candidate_codes(tokenizer, count):
    """Deterministic letter codes, accepting only exact one-token continuations."""
    boundary = "Answer:"
    prefix = tokenizer.encode(boundary, add_special_tokens=False)
    codes, seen = [], set()
    for width in range(1, 4):
        for letters in product(ascii_uppercase, repeat=width):
            code = "".join(letters)
            ids = tokenizer.encode(boundary + " " + code, add_special_tokens=False)
            if len(ids) == len(prefix) + 1 and ids[:-1] == prefix and ids[-1] not in seen:
                codes.append(code)
                seen.add(ids[-1])
                if len(codes) == count:
                    return codes
    raise ValueError(f"Tokenizer cannot supply {count} distinct single-token answer codes")


def compile_slots(tokenizer, state, schema, max_input_tokens=16384):
    prefix = tokenizer.apply_chat_template(
        build_decision_messages(state, schema),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return compile_prefix_slots(tokenizer, schema, prefix_ids, prefix, max_input_tokens)


def compile_prefix_slots(tokenizer, schema, prefix_ids, prefix="", max_input_tokens=16384):
    """Share label-boundary checks between text and processor-expanded image prefixes."""
    prefix_length = len(prefix_ids)
    rows = []
    positions, candidates, texts, slot_ids = [], [], [], []
    codes = candidate_codes(tokenizer, max(len(field.choices) for field in schema.values()))
    row_codes = []
    for field in schema.values():
        labels = codes[: len(field.choices)]
        text = render_question_branch(tokenizer, field, labels)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        choices = []
        for code in labels:
            appended = tokenizer.encode(text + " " + code, add_special_tokens=False)
            if appended[:-1] != tokens or len(appended) != len(tokens) + 1:
                raise ValueError(
                    f"Tokenizer must encode internal code {code!r} as one token at the answer boundary"
                )
            choices.append(appended[-1])
        if len(set(choices)) != len(choices):
            raise ValueError("Candidate token collision")
        # Exactly the original prefix IDs followed by this branch's suffix IDs.
        row = prefix_ids + tokens
        rows.append(row)
        positions.append(len(row) - 1)
        candidates.append(choices)
        texts.append(text)
        slot_ids.append(tokens)
        row_codes.append(labels)
    if max(map(len, rows)) > max_input_tokens:
        raise ValueError("Request exceeds input token limit; no truncation performed")
    return CompiledSlots(
        rows, positions, candidates, texts, slot_ids, prefix, prefix_length, row_codes
    )
