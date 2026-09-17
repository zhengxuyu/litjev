"""Render the engine's symbolic view of the level as the text block the model reads."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# The agent itself and short-lived render effects carry no decision value.
HIDDEN_OBJECTS = frozenset({"DoomPlayer", "BulletPuff", "Blood", "Puff"})

# Half-open bearing sectors, in degrees away from where the player is aiming.
SECTORS: tuple[tuple[float, str], ...] = (
    (20.0, "dead ahead"),
    (65.0, "ahead-{side}"),
    (115.0, "to your {side}"),
    (160.0, "behind-{side}"),
)


@dataclass(frozen=True, slots=True)
class PlayerStatus:
    x: float
    y: float
    angle: float
    health: float
    ammo: float
    kills: float

    def summary(self) -> dict[str, float]:
        return {"health": self.health, "ammo": self.ammo, "kills": self.kills}


@dataclass(frozen=True, slots=True)
class VisibleActor:
    name: str
    bearing: float
    distance: float


def relative_bearing(player_angle: float, dx: float, dy: float) -> float:
    """Degrees from the player's aim to a point; positive is left, negative is right."""
    absolute = math.degrees(math.atan2(dy, dx))
    return (absolute - player_angle + 180.0) % 360.0 - 180.0


def describe_bearing(bearing: float) -> str:
    side = "left" if bearing > 0 else "right"
    magnitude = abs(bearing)
    for limit, template in SECTORS:
        if magnitude <= limit:
            return template.format(side=side)
    return "behind you"


def visible_actors(
    labels: Iterable,
    status: PlayerStatus,
    hidden: frozenset[str] = HIDDEN_OBJECTS,
) -> tuple[VisibleActor, ...]:
    actors = []
    for label in labels:
        if label.object_name in hidden:
            continue
        dx = label.object_position_x - status.x
        dy = label.object_position_y - status.y
        bearing = relative_bearing(status.angle, dx, dy)
        actors.append(VisibleActor(label.object_name, bearing, math.hypot(dx, dy)))
    actors.sort(key=lambda actor: actor.distance)
    return tuple(actors)


def render_state(
    briefing: str,
    status: PlayerStatus,
    actors: Sequence[VisibleActor],
    max_actors: int = 6,
) -> str:
    lines = [
        briefing,
        f"Health {status.health:.0f}. Ammo {status.ammo:.0f}. Kills so far {status.kills:.0f}.",
    ]
    if actors:
        lines.append("You can see, nearest first:")
        for actor in actors[:max_actors]:
            lines.append(
                f"- {actor.name}, {describe_bearing(actor.bearing)}, "
                f"{abs(actor.bearing):.0f} degrees off your aim, {actor.distance:.0f} units away"
            )
        if len(actors) > max_actors:
            lines.append(f"- and {len(actors) - max_actors} further away")
    else:
        lines.append("Nothing is visible from where you are facing.")
    return "\n".join(lines)
