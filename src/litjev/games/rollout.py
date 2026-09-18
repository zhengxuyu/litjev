"""Unified measured trace: pre-action RGB frame, decision, then transition outcome."""

import time
from dataclasses import asdict

from litjev.games.doom_text import OBSERVATION_KIND
from litjev.vision import encode_image

TRACE_VERSION = "litjev.trace.v1"


def record_episode(env, policy, *, seed=0, episode=1):
    observation, _ = env.reset(seed=seed)
    decisions = []
    running_reward = 0.0
    started = time.perf_counter()
    text_mode = getattr(env.unwrapped, "observation_mode", "rgb") == "engine_text"
    while True:
        picture = encode_image(observation)
        description_start = time.perf_counter()
        description = env.unwrapped.describe_observation() if text_mode else None
        description_ms = (time.perf_counter() - description_start) * 1000
        decision = policy.decide(description if text_mode else observation)
        next_observation, reward, terminated, truncated, info = env.step(decision.action)
        running_reward += reward
        row = asdict(decision)
        if text_mode:
            row.update(observation_text=description, description_ms=description_ms)
        row.update(
            episode=episode,
            step=len(decisions),
            frame=picture,
            action_index=decision.action,
            action=policy.action_names[decision.action],
            latency_ms=decision.latency_seconds * 1000,
            timestamp=time.perf_counter() - started,
            reward=reward,
            running_reward=running_reward,
            terminated=terminated,
            truncated=truncated,
            forward_calls=decision.usage.get("forward_calls"),
            output_tokens=decision.usage.get("output_tokens"),
            info=info,
        )
        decisions.append(row)
        observation = next_observation
        if terminated or truncated:
            break
    return {
        "schema_version": TRACE_VERSION,
        "environment": env.spec.id if env.spec else type(env).__name__,
        "model": policy.client.model_id,
        "training": False,
        "observation": OBSERVATION_KIND if text_mode else "rgb_pixels_only",
        "seed": seed,
        "actions": list(policy.action_names),
        "decisions": decisions,
        "episode_reward": running_reward,
        "terminated": terminated,
        "truncated": truncated,
        "wall_seconds": time.perf_counter() - started,
    }
