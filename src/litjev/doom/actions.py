"""Letter-labelled action menu compiled from whichever buttons a scenario exposes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Single-token labels; every choice must append exactly one token after "Answer:".
LABELS = "ABCDEFGHIJ"

# Canonical moves in menu order. A move is offered only if the scenario has all its buttons.
MOVES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attack", ("ATTACK",)),
    ("turn left", ("TURN_LEFT",)),
    ("turn right", ("TURN_RIGHT",)),
    ("turn left while attacking", ("TURN_LEFT", "ATTACK")),
    ("turn right while attacking", ("TURN_RIGHT", "ATTACK")),
    ("move forward", ("MOVE_FORWARD",)),
    ("move backward", ("MOVE_BACKWARD",)),
    ("strafe left", ("MOVE_LEFT",)),
    ("strafe right", ("MOVE_RIGHT",)),
    ("hold still", ()),
)

QUESTION = "Choose the single best action for this instant."


@dataclass(frozen=True, slots=True)
class ActionMenu:
    labels: tuple[str, ...]
    moves: tuple[str, ...]
    vectors: tuple[tuple[int, ...], ...]

    @classmethod
    def compile(cls, buttons: Sequence[str]) -> ActionMenu:
        index = {name: position for position, name in enumerate(buttons)}
        labels: list[str] = []
        moves: list[str] = []
        vectors: list[tuple[int, ...]] = []
        for move, required in MOVES:
            if len(labels) == len(LABELS) or any(button not in index for button in required):
                continue
            vector = [0] * len(buttons)
            for button in required:
                vector[index[button]] = 1
            labels.append(LABELS[len(labels)])
            moves.append(move)
            vectors.append(tuple(vector))
        if len(labels) < 2:
            raise ValueError("Scenario exposes too few supported buttons")
        return cls(tuple(labels), tuple(moves), tuple(vectors))

    @property
    def description(self) -> str:
        options = "\n".join(
            f"{label}. {move}" for label, move in zip(self.labels, self.moves, strict=True)
        )
        return f"{QUESTION}\n{options}"

    def field(self) -> dict:
        return {"type": "enum", "description": self.description, "choices": list(self.labels)}

    def catalog(self) -> dict[str, str]:
        return dict(zip(self.labels, self.moves, strict=True))

    def move(self, label: str) -> str:
        return self.moves[self.labels.index(label)]

    def vector(self, label: str) -> list[int]:
        return list(self.vectors[self.labels.index(label)])
