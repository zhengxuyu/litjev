"""Score an option by its own text, so that no label stands between the two.

The labelled readout gives every option a single-token code and scores the
codes. That is what gives it a preference over positions: a code is a token, a
token carries a prior, and the prior lands on whatever option happens to sit in
that slot, so answers drift towards some positions whatever they contain. Scoring the content removes the label rather than
correcting for it.

Nothing is generated. Each option is teacher-forced after the same prefix and
its own tokens' log-probabilities are summed, so the cost is one prefill plus k
short continuations that share it -- more than the single pass the coded readout
needs, and still no sampled token.

Three normalisations, kept apart rather than chosen between here, because they
fail differently rather than merely differing:

    sum     the joint log-probability of the option. Favours short options,
            since every extra token subtracts.
    mean    per token. Favours long ones, since a long option's improbable
            first token is averaged away.
    pmi     sum minus what the same option scores after a content-free
            question. The only one of the three that is not a length artefact,
            and the noisiest, since it is a difference of two estimates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionScore:
    """One option's scores, with the token count the normalisations divide by."""

    index: int
    text: str
    tokens: int
    total: float
    prior: float | None = None

    @property
    def mean(self):
        return self.total / self.tokens if self.tokens else float("-inf")

    @property
    def pmi(self):
        """`None` when no content-free pass was run, rather than 0, which is a score."""
        return None if self.prior is None else self.total - self.prior


def option_token_ids(tokenizer, prefix, option):
    """The option's own tokens, as the model will see them after this prefix.

    Tokenised jointly and checked, not tokenised alone and appended. A tokenizer
    merges across a boundary -- the prefix's last character and the option's
    first becoming one token -- and then the option's token count is not what
    counting it alone says, so every normalisation divides by the wrong number
    and the first token's probability is folded into a token that is partly
    prefix.

    Raises on a merge instead of guessing, because the alternative is a score
    that is wrong by an amount nothing reports.
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    joint = tokenizer.encode(prefix + option, add_special_tokens=False)
    if joint[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"the tokenizer merges across the boundary between the prompt and "
            f"{option!r}, so this option's tokens are not separable from the prompt's "
            f"and its score would be normalised by the wrong count")
    found = joint[len(prefix_ids):]
    if not found:
        raise ValueError(f"{option!r} adds no tokens after the prompt, so there is "
                         f"nothing to score")
    return found


def summed_logprob(logprobs, prefix_length, option_ids):
    """Sum the log-probabilities the model gave this option's own tokens.

    `logprobs[i]` is the distribution over the token at position `i + 1`, so the
    first option token is predicted from the last prefix position. Counting the
    prefix's own tokens would score the question rather than the answer, and
    would do it identically for every option, which hides the error in a constant.
    """
    total = 0.0
    for offset, token in enumerate(option_ids):
        total += float(logprobs[prefix_length - 1 + offset][token])
    return total


def rank(scores, by="sum"):
    """The options in order, best first, under one normalisation."""
    key = {"sum": lambda s: s.total, "mean": lambda s: s.mean,
           "pmi": lambda s: s.pmi}.get(by)
    if key is None:
        raise ValueError(f"unknown normalisation {by!r}; use sum, mean or pmi")
    if by == "pmi" and any(score.pmi is None for score in scores):
        raise ValueError("pmi needs the content-free pass, which was not run")
    return sorted(scores, key=key, reverse=True)


def chosen_index(scores, by="sum"):
    return rank(scores, by)[0].index


def as_record(scores):
    """Everything the three normalisations are computed from, for the record.

    All of it, not the winner: which of the three is best over a benchmark is a
    much weaker fact than where they disagree, and the disagreement cannot be
    recovered from a file that kept only the choice.
    """
    return [
        {"index": score.index, "tokens": score.tokens, "sum": score.total,
         "mean": score.mean, "prior": score.prior, "pmi": score.pmi}
        for score in sorted(scores, key=lambda s: s.index)
    ]
