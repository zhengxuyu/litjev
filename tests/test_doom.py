import threading
import time

from litjev.decision import DecisionAnswer, DecisionResponse, Usage
from litjev.doom.actions import ActionMenu
from litjev.doom.runtime import Observation
from litjev.doom.session import DoomSession, DoomSettings
from litjev.doom.state import (
    PlayerStatus,
    describe_bearing,
    relative_bearing,
    render_state,
    visible_actors,
)
from litjev.schema import DecisionSchema

ARENA_BUTTONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]


class FakeLabel:
    def __init__(self, name, x, y):
        self.object_name = name
        self.object_position_x = x
        self.object_position_y = y


class FakeEngine:
    def decide(self, state, schema):
        del state
        field = schema["action"]
        # Always picks the last offered move so tests can tell it apart from index zero.
        return DecisionResponse(
            "fake",
            {
                "action": DecisionAnswer(
                    field.field_type,
                    field.choices[-1],
                    {choice: 1.0 / len(field.choices) for choice in field.choices},
                    0.5,
                )
            },
            Usage(11, 0),
        )


class FakeRuntime:
    """Stands in for ViZDoom. One Demon straight ahead, one step per released permit."""

    def __init__(self, settings):
        self.settings = settings
        self.gate = threading.Semaphore(0)
        self.actions = []
        self.episodes = []
        self.closed = False

    def buttons(self):
        return list(ARENA_BUTTONS)

    def restart(self, recording=None):
        self.episodes.append(recording)

    def finished(self):
        return False

    def observe(self):
        if not self.gate.acquire(timeout=0.05):
            return None  # nothing new yet, exactly as an unfinished tic would report
        status = PlayerStatus(x=0.0, y=0.0, angle=0.0, health=71.0, ammo=26.0, kills=2.0)
        return Observation(status, visible_actors([FakeLabel("Demon", 100.0, 0.0)], status), "jpeg")

    def act(self, vector, tics):
        self.actions.append((tuple(vector), tics))

    def close(self):
        self.closed = True


def test_menu_offers_only_moves_the_scenario_supports() -> None:
    menu = ActionMenu.compile(ARENA_BUTTONS)

    assert menu.labels == ("A", "B", "C", "D", "E", "F")
    assert menu.moves[0] == "attack"
    assert menu.vector("A") == [0, 0, 1]
    assert menu.vector("D") == [1, 0, 1]
    assert menu.vector("F") == [0, 0, 0]
    assert "move forward" not in menu.moves


def test_menu_field_is_a_valid_single_token_schema() -> None:
    schema = DecisionSchema.from_mapping({"action": ActionMenu.compile(ARENA_BUTTONS).field()})

    assert schema["action"].choices == ("A", "B", "C", "D", "E", "F")
    assert all(len(choice) == 1 for choice in schema["action"].choices)


def test_menu_rejects_a_scenario_without_supported_buttons() -> None:
    try:
        ActionMenu.compile(["JUMP"])
    except ValueError as error:
        assert "too few supported buttons" in str(error)
    else:
        raise AssertionError("Unsupported button sets must be rejected")


def test_bearing_is_positive_to_the_left_and_negative_to_the_right() -> None:
    assert relative_bearing(0.0, 1.0, 1.0) == 45.0
    assert relative_bearing(0.0, 1.0, -1.0) == -45.0
    assert relative_bearing(90.0, 1.0, 0.0) == -90.0
    assert describe_bearing(5.0) == "dead ahead"
    assert describe_bearing(45.0) == "ahead-left"
    assert describe_bearing(-45.0) == "ahead-right"
    assert describe_bearing(175.0) == "behind you"


def test_state_text_hides_the_player_and_orders_by_distance() -> None:
    status = PlayerStatus(x=0.0, y=0.0, angle=0.0, health=64.0, ammo=26.0, kills=3.0)
    labels = [
        FakeLabel("MarineChainsawVzd", 300.0, 0.0),
        FakeLabel("DoomPlayer", 0.0, 0.0),
        FakeLabel("Demon", 100.0, 0.0),
    ]

    actors = visible_actors(labels, status)
    text = render_state("Hold the centre.", status, actors, max_actors=1)

    assert [actor.name for actor in actors] == ["Demon", "MarineChainsawVzd"]
    assert "Health 64. Ammo 26. Kills so far 3." in text
    assert "Demon, dead ahead" in text
    assert "and 1 further away" in text
    assert "DoomPlayer" not in text


def test_state_text_reports_an_empty_view() -> None:
    status = PlayerStatus(x=0.0, y=0.0, angle=0.0, health=100.0, ammo=26.0, kills=0.0)

    assert "Nothing is visible" in render_state("Hold the centre.", status, ())


def run_session(steps, settings=None):
    """Drive the loop one decision at a time so every payload is observed in order."""
    runtimes = []

    def factory(created):
        runtimes.append(FakeRuntime(created))
        return runtimes[-1]

    session = DoomSession(lambda: FakeEngine(), settings, runtime_factory=factory)
    stream = session.watch()
    session.start()
    deadline = time.monotonic() + 10.0
    while not runtimes and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtimes, "the session never built its runtime"

    collected = []
    for _ in range(steps):
        runtimes[0].gate.release()
        for payload in stream:
            if payload is not None and payload["event"] == "step":
                collected.append(payload)
                break
    session.stop()
    return runtimes[0], collected


def test_session_publishes_one_payload_per_decision_and_presses_the_chosen_buttons() -> None:
    runtime, steps = run_session(2, DoomSettings(tics=3))

    assert [payload["step"] for payload in steps] == [1, 2]
    assert steps[0]["action"] == "F"
    assert steps[0]["move"] == "hold still"
    assert steps[0]["frame"] == "jpeg"
    assert steps[0]["forward_calls"] == 2
    assert steps[0]["output_tokens"] == 0
    assert steps[0]["status"] == {"health": 71.0, "ammo": 26.0, "kills": 2.0}
    assert "Demon, dead ahead" in steps[0]["state_text"]
    assert steps[0]["menu"]["A"] == "attack"
    assert runtime.actions == [((0, 0, 0), 3), ((0, 0, 0), 3)]
    assert runtime.episodes == [None]
    assert runtime.closed


def test_session_writes_one_recording_per_episode_when_asked() -> None:
    settings = DoomSettings(recording="/tmp/litjev-doom-test")

    assert settings.episode_path(0) == "/tmp/litjev-doom-test/episode-000.lmp"
    assert settings.episode_path(12) == "/tmp/litjev-doom-test/episode-012.lmp"
    assert DoomSettings().episode_path(0) is None


def test_session_reports_a_failure_instead_of_dying_silently() -> None:
    def explode(settings):
        del settings
        raise RuntimeError("no display")

    session = DoomSession(lambda: FakeEngine(), runtime_factory=explode)
    stream = session.watch()
    session.start()

    stopped = [payload for payload in stream if payload is not None][-1]
    assert stopped["event"] == "stopped"
    assert "no display" in stopped["error"]
