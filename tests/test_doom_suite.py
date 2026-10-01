import numpy as np

from litjev.games.suite.specs import TASKS, audio_description, outcome


def test_exactly_twelve_distinct_tasks():
    assert len(TASKS) == 12
    assert len({t.name for t in TASKS}) == 12


def test_survival_time_limit_is_not_goal_success():
    assert (
        outcome("take_cover", dead=False, timed_out=True, finished=True, kills=3, peak_reward=1)
        == "survived_horizon"
    )


def test_wrong_notification_target_is_not_success():
    assert (
        outcome(
            "basic_notifications",
            dead=False,
            timed_out=False,
            finished=True,
            kills=1,
            peak_reward=-1,
        )
        == "failed_objective"
    )
    assert (
        outcome(
            "basic_notifications",
            dead=False,
            timed_out=False,
            finished=True,
            kills=1,
            peak_reward=100,
        )
        == "success"
    )


def test_missed_rocket_is_not_success():
    assert (
        outcome(
            "predict_position", dead=False, timed_out=False, finished=True, kills=0, peak_reward=0
        )
        == "failed_objective"
    )


def test_audio_uses_samples_not_hidden_monster_coordinates():
    sound = np.zeros((1260, 2), dtype=np.int16)
    sound[:, 0] = 1000
    result = audio_description(sound)
    assert result["louder_channel"] == "left"
    assert audio_description(np.zeros((1260, 2)))["louder_channel"] == "silent"
