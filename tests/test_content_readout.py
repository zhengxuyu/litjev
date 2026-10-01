"""Scoring an option by its own text, and the three ways of normalising it.

The coded readout scores single-token letters, which is why it has a preference
over positions at all: a code is a token, a token carries a prior, and the prior
lands on whichever option sits in that slot. Scoring content removes the label
rather than correcting for it.

The three normalisations are kept apart because they fail differently, and these
plant each failure rather than asserting that they differ: sum prefers short
options, mean prefers long ones, and PMI is the only one that is not a length
artefact.
"""
import math

import pytest

from litjev.content_readout import (
    OptionScore,
    as_record,
    chosen_index,
    option_token_ids,
    rank,
    summed_logprob,
)


def score(index, total, tokens, prior=None):
    return OptionScore(index=index, text=f"o{index}", tokens=tokens, total=total, prior=prior)


class Tokenizer:
    """Characters as tokens, so a boundary merge can be arranged deliberately."""

    def __init__(self, merges=()):
        self.merges = set(merges)

    def encode(self, text, add_special_tokens=False):
        for merge in self.merges:
            if merge in text:
                # One token stands for the merged pair, so the prefix's own ids
                # stop being a prefix of the joint ids -- which is the fault.
                text = text.replace(merge, "\x00")
        return [ord(character) for character in text]


# ---------------------------------------------------------------- normalisation


def test_sum_prefers_the_shorter_option():
    """Every extra token subtracts, so a long right answer loses to a short wrong one."""
    short, long = score(0, -6.0, 2), score(1, -9.0, 6)
    assert chosen_index([short, long], "sum") == 0
    assert long.mean > short.mean, "which is exactly what mean is for"


def test_mean_prefers_the_longer_option():
    """A long option's improbable first token is averaged away by its tail."""
    short, long = score(0, -4.0, 1), score(1, -9.0, 6)
    assert chosen_index([short, long], "sum") == 0
    assert chosen_index([short, long], "mean") == 1


def test_pmi_is_the_one_that_is_not_about_length():
    """Two options of very different lengths whose scores are entirely their
    priors: PMI calls them level, and sum and mean do not."""
    scores = [score(0, -4.0, 1, prior=-4.0), score(1, -40.0, 10, prior=-40.0)]
    assert scores[0].pmi == scores[1].pmi == 0.0
    assert chosen_index(scores, "sum") == 0
    assert chosen_index(scores, "mean") == 0, "mean is not length-free either"


def test_pmi_without_the_content_free_pass_is_refused():
    """`None` rather than zero, because zero is a score and absence is not."""
    scores = [score(0, -4.0, 1), score(1, -9.0, 6)]
    assert scores[0].pmi is None
    with pytest.raises(ValueError, match="content-free"):
        rank(scores, "pmi")


def test_an_unknown_normalisation_is_named():
    with pytest.raises(ValueError, match="sum, mean or pmi"):
        rank([score(0, -1.0, 1)], "median")


def test_the_record_keeps_all_three_and_not_the_winner():
    """Which normalisation wins over a benchmark is a weaker fact than where they
    disagree, and the disagreement is not recoverable from a file that kept the
    choice alone."""
    scores = [score(1, -9.0, 6, prior=-8.0), score(0, -4.0, 1, prior=-3.0)]
    rows = as_record(scores)
    assert [row["index"] for row in rows] == [0, 1], "in option order, not score order"
    for row in rows:
        assert set(row) == {"index", "tokens", "sum", "mean", "prior", "pmi"}


# ---------------------------------------------------------------- the boundary


def test_an_option_is_tokenised_after_the_prompt_and_not_alone():
    tokenizer = Tokenizer()
    ids = option_token_ids(tokenizer, "Answer: ", "42")
    assert ids == [ord("4"), ord("2")]


def test_a_merge_across_the_boundary_is_refused():
    """A tokenizer that joins the prompt's last character to the option's first
    makes the option's token count unknowable, so every normalisation would
    divide by the wrong number and the first token's probability would be folded
    into a token that is partly prompt."""
    tokenizer = Tokenizer(merges=[" 4"])
    with pytest.raises(ValueError, match="merges across the boundary"):
        option_token_ids(tokenizer, "Answer: ", "42")


def test_an_option_that_adds_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to score"):
        option_token_ids(Tokenizer(), "Answer: ", "")


# ---------------------------------------------------------------- the sum itself


def test_only_the_options_own_tokens_are_counted():
    """Counting the prompt's would score the question, identically for every
    option, which hides the error in a constant that cancels in the ranking and
    not in the numbers."""
    vocabulary = 4
    # Position i predicts token i+1, so a prefix of length 3 has its first
    # option token predicted from index 2.
    logprobs = [[math.log(0.1)] * vocabulary for _ in range(6)]
    logprobs[2][1] = math.log(0.5)
    logprobs[3][2] = math.log(0.25)
    total = summed_logprob(logprobs, prefix_length=3, option_ids=[1, 2])
    assert total == pytest.approx(math.log(0.5) + math.log(0.25))


# ---------------------------------------------------------------- end to end


def test_the_scorer_returns_one_score_per_option(tmp_path):
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backend import TransformersScorer
    from litjev.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="What is 2+2?",
                      criteria={"A": "three", "B": "four", "C": "five point one"})
    scores, rewritten = scorer.score_options("Answer.", question)
    assert rewritten == 0
    assert [s.index for s in scores] == [0, 1, 2]
    assert [s.text for s in scores] == ["three", "four", "five point one"]
    assert [s.tokens for s in scores] == [5, 4, 14], "its own characters, none of the prompt"
    assert all(s.total < 0 for s in scores)
    assert all(s.prior is not None for s in scores), "the content-free pass ran"
    # Nothing was generated: the scores come from a forward pass over given text.
    assert all(isinstance(s.total, float) for s in scores)


def test_the_content_free_pass_can_be_turned_off():
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backend import TransformersScorer
    from litjev.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="q", criteria={"A": "aa", "B": "bb"})
    scores, _ = scorer.score_options("Answer.", question, content_free=False)
    assert all(s.prior is None for s in scores)


def test_the_prompt_carries_no_label_for_an_option():
    """The whole point: there is no code for a prior to attach to."""
    from litjev.prompting import unlabelled_suffix
    from litjev.schema import Choice

    question = Choice(instructions="What is 2+2?", criteria={"A": "3", "B": "4"})
    rendered = unlabelled_suffix(question)
    assert rendered == "Question: What is 2+2?\nOptions:\n3\n4\nAnswer:"
    assert '"code"' not in rendered and '"option"' not in rendered
    # The criteria keys are A and B and neither reaches the prompt.
    assert "A" not in rendered.replace("Answer:", "") and "B" not in rendered


# ---------------------------------------------------------------- a real tokenizer

REAL = ["Oxygen", "Carbon dioxide", "42", r"\frac{1}{2}", r"(3, \pi/2)", "x = 2", "-5",
        "The answer is B"]


def bpe():
    """A real subword tokenizer, since the stub above is character-level and a
    character-level tokenizer cannot have the fault this guards against."""
    from transformers import AutoTokenizer

    for name in ("Qwen/Qwen2.5-0.5B", "gpt2", "Qwen/Qwen3.5-4B"):
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001, S112  any failure here means "not cached"
            continue
    return None


@pytest.mark.parametrize("option", REAL)
def test_a_real_tokenizer_separates_the_option_from_the_prompt(option):
    """The fault a character-level stub cannot show. BPE binds a separating space
    to the word after it, so a prompt ending in one stops being a prefix of the
    joint encoding: measured on gpt2, none of these options was separable with the
    trailing space and all of them are without it."""
    tokenizer = bpe()
    if tokenizer is None:
        pytest.skip("no tokenizer cached locally")
    ids = option_token_ids(tokenizer, "Question: q\nOptions:\na\nb\nAnswer:", " " + option)
    assert ids, option
    assert tokenizer.decode(ids).strip() == option.strip()


def test_a_prompt_ending_in_a_space_is_what_would_have_broken_it():
    """Asserted so the fix cannot be quietly undone by re-adding the space."""
    tokenizer = bpe()
    if tokenizer is None:
        pytest.skip("no tokenizer cached locally")
    broken = sum(
        tokenizer.encode("Answer: " + option, add_special_tokens=False)[
            : len(tokenizer.encode("Answer: ", add_special_tokens=False))]
        != tokenizer.encode("Answer: ", add_special_tokens=False)
        for option in REAL)
    assert broken, "this tokenizer would not have shown the fault; the test guards nothing"

    from litjev.prompting import unlabelled_suffix
    from litjev.schema import Choice

    rendered = unlabelled_suffix(Choice(instructions="q", criteria={"A": "a", "B": "b"}))
    assert not rendered.endswith(" "), "the space belongs to the option"


def test_the_prompt_shows_the_text_that_is_scored():
    r"""Through json.dumps the model reads "\\frac{1}{2}", escaped and quoted,
    while the score is for \frac{1}{2}: the system prompt asks for an option
    exactly as written and the thing scored is not what was written."""
    from litjev.prompting import unlabelled_suffix
    from litjev.schema import Choice

    option = r"\frac{1}{2}"
    rendered = unlabelled_suffix(Choice(instructions="q", criteria={"A": option, "B": "3"}))
    assert f"\n{option}\n" in rendered, rendered
    assert '\\\\' not in rendered, "no JSON escaping between the prompt and the score"
    assert '"' not in rendered


def test_the_listing_is_one_line_per_option_however_the_option_arrived():
    """The count the prompt shows and the count being scored have to agree. A
    carriage return does it as surely as a newline."""
    from litjev.prompting import listable_options, unlabelled_body
    from litjev.schema import Choice

    fine = Choice(instructions="q", criteria={"A": "3", "B": "4"})
    assert unlabelled_body(fine).count("\n") == 3, "instructions, Options:, two options"

    for bad in ("two\nlines", "carriage\rreturn", "many\n\nbreaks"):
        question = Choice(instructions="q", criteria={"A": bad, "B": "4"})
        texts, rewritten = listable_options(question)
        assert rewritten == 1 and "\n" not in texts[0] and "\r" not in texts[0], bad
        assert unlabelled_body(question).count("\n") == 3, bad


def test_the_count_of_rewritten_options_is_returned_not_left_on_the_scorer():
    """As instance state it was overwritten by every later call, including by the
    content-free pass two lines down, so what a caller read afterwards was
    whatever the last question happened to need."""
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backend import TransformersScorer
    from litjev.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    assert not hasattr(scorer, "rewritten_options"), "it must not be instance state"

    broken = Choice(instructions="q",
                    criteria={"A": "two" + chr(10) + "lines", "B": "fine"})
    scores, rewritten = scorer.score_options("Answer.", broken, content_free=False)
    assert rewritten == 1
    assert [s.text for s in scores] == ["two lines", "fine"], "the listed form is scored"
    # The count is the rewritten option's. This stub eats the space at the
    # boundary the way BPE merges there, so it is nine and not ten -- what
    # matters is that no newline was ever tokenised.
    assert scores[0].tokens == len("two lines")

    # A later question does not inherit the earlier one's count.
    _, after = scorer.score_options(
        "Answer.", Choice(instructions="q", criteria={"A": "a", "B": "b"}),
        content_free=False)
    assert after == 0


def test_the_content_free_pass_does_not_report_its_own_rewriting():
    """It runs on a copy of the question, so its count is about the same options
    -- but the number returned must be the real question's, not the last pass's."""
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backend import TransformersScorer
    from litjev.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="q",
                      criteria={"A": "two" + chr(10) + "lines", "B": "fine"})
    scores, rewritten = scorer.score_options("Answer.", question, content_free=True)
    assert rewritten == 1
    assert all(s.prior is not None for s in scores)


def test_a_screenshot_is_read_through_the_coded_readout_and_says_so():
    """The content prompt is text only, so an image state cannot go through it.
    It is read out by codes instead, and the diagnostics name the readout that
    ran rather than the one the engine was set to."""
    import numpy as np
    from PIL import Image

    from litjev.decision import RawFieldScores, SchemaDecisionEngine
    from litjev.schema import Choice, DecisionSchema
    from litjev.vision import VisualState

    class Both:
        def score(self, state, schema):
            return tuple(RawFieldScores(name, np.array([2.0, 0.0]), 7)
                         for name in schema.names)

        def score_options(self, state, question, content_free=True):
            raise AssertionError("an image state reached the content readout")

    engine = SchemaDecisionEngine(Both())
    assert engine.readout == "content"
    schema = DecisionSchema({"q": Choice(instructions="q", criteria={"a": "x", "b": "y"})})
    state = VisualState("look", Image.new("RGB", (8, 8)))
    found = engine.evaluate(state, schema)
    assert found.diagnostics["readout"] == "coded"
    assert found.diagnostics["normalisation"] is None
    assert found.result.answers["q"].choice == "a"
