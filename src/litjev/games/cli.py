"""Play with pretrained Qwen weights, locally or through the LitJev HTTP service."""

import argparse
import json
from pathlib import Path


def play():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=["doom", "chess"])
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--url", help="LitJev server base URL (default: http://127.0.0.1:8000)")
    backend.add_argument("--model", help="Use a local Transformers model instead of HTTP")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--observation", choices=["rgb", "engine_text"], default="rgb")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/trace.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    if args.environment != "doom" and args.observation != "rgb":
        parser.error("engine_text observations are only available for Doom")
    from litjev.games import make_env
    from litjev.games.policy import HttpDecisionClient, LitJevPolicy, LocalDecisionClient
    from litjev.games.rollout import record_episode

    if args.model:
        from litjev.backend import ModelSettings, TransformersScorer
        from litjev.decision import SchemaDecisionEngine

        settings = ModelSettings(args.model, args.revision, args.device_map, args.dtype)
        client = LocalDecisionClient(
            SchemaDecisionEngine(TransformersScorer.load(settings), model_id=args.model)
        )
    else:
        client = HttpDecisionClient(args.url or "http://127.0.0.1:8000")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = None
    env_options = {"observation_mode": args.observation} if args.environment == "doom" else {}
    with make_env(args.environment, max_steps=args.max_steps, **env_options) as env:
        policy = LitJevPolicy(client, env.unwrapped.action_names, env.unwrapped.instructions)
        for index in range(args.episodes):
            trace = record_episode(env, policy, seed=args.seed + index, episode=index + 1)
            if report is None:
                report = {**trace, "episode_summaries": [], "decisions": []}
                for key in ("episode_reward", "terminated", "truncated", "wall_seconds"):
                    report.pop(key)
            report["decisions"].extend(trace["decisions"])
            summary = {
                key: trace[key]
                for key in ("episode_reward", "terminated", "truncated", "wall_seconds", "seed")
            }
            summary.update(episode=index + 1, steps=len(trace["decisions"]))
            summary["outcome"] = trace["decisions"][-1]["info"].get("outcome")
            report["episode_summaries"].append(summary)
            args.output.write_text(json.dumps(report, allow_nan=False) + "\n")
            print(json.dumps({**summary, "trace": str(args.output)}), flush=True)


def film():
    from litjev.games.film import build_film

    parser = argparse.ArgumentParser(
        description="Build a standalone trace replay, with no model load"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/replay.html"))
    args = parser.parse_args()
    try:
        build_film(args.trace, args.output)
    except ValueError as error:
        parser.error(str(error))
    print(args.output)
