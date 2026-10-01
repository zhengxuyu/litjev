"""SGLang-backed inference for LitJev's fast and slow modes.

`litjev.backend.TransformersScorer` runs the two modes through transformers. It
is slow where it matters most: the slow-thinking mode decodes one sequence at a
time, and a reasoning step on a 27B backbone costs minutes.

This package serves the same two modes through SGLang and satisfies the same
interface, so `SchemaDecisionEngine` and everything above it are unchanged.

Two things make the mapping fit rather than merely work.

The fast mode reads its answer at one position from a set of single-token
candidate codes, which is a request for the logprobs of chosen token ids and no
generation at all. The questions of one request share a state, and the shared
prefill with branched readout is what SGLang's prefix cache does by itself, so
the branching needs no cache surgery here.

The decision head reads a hidden state at the readout position. SGLang can
return that, with caveats serious enough to check rather than trust: see
`verify.py`.

Two scorers, one interface. `SGLangScorer` runs the engine in this process.
`SGLangServerScorer` talks to one over HTTP and receives the hidden states, which
is how `verify.py` compares the two backends from a machine that holds no GPU.

Resolved on first use, so importing the package does not import SGLang.
"""

import importlib

_EXPORTS = {
    "SGLangServerScorer": "inference.http_backend",
    "SGLangServerSettings": "inference.http_backend",
    "SGLangScorer": "inference.sglang_backend",
    "SGLangSettings": "inference.sglang_backend",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = ["SGLangScorer", "SGLangServerScorer", "SGLangServerSettings", "SGLangSettings"]
