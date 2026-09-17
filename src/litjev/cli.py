import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from litjev.calibration import CalibrationProfile, TemperatureCalibrator


def add_model_arguments(parser):
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--calibration")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def engine_factory(args):
    from litjev.backend import ModelSettings, TransformersScorer
    from litjev.decision import SchemaDecisionEngine

    def factory():
        profile = CalibrationProfile.load(args.calibration) if args.calibration else None
        if profile and profile.model_id != args.model:
            raise ValueError("Calibration profile model does not match serving model")
        settings = ModelSettings(args.model, args.revision, args.device_map, args.dtype)
        return SchemaDecisionEngine(
            TransformersScorer.load(settings),
            profile.temperature if profile else 1.0,
            args.model,
            profile is not None,
        )

    return factory


def serve():
    import uvicorn

    from litjev.api import create_app

    args = add_model_arguments(argparse.ArgumentParser()).parse_args()
    # One process owns one model; concurrent forwards are serialized in the backend.
    uvicorn.run(create_app(engine_factory(args)), host="127.0.0.1", port=args.port, workers=1)


def doom():
    """Serve the playground plus the Doom demo from one process and one loaded model."""
    import uvicorn

    from litjev.api import create_app
    from litjev.doom.routes import attach_doom
    from litjev.doom.session import DoomSettings

    parser = add_model_arguments(
        argparse.ArgumentParser(description="Play Doom through the decision layer")
    )
    parser.add_argument("--scenario", default="defend_the_center")
    parser.add_argument("--tics", type=int, default=4, help="Game tics advanced per decision")
    parser.add_argument("--resolution", default="RES_640X480")
    parser.add_argument("--max-actors", type=int, default=6)
    parser.add_argument("--record", help="Directory for .lmp episode recordings")
    parser.add_argument("--window", action="store_true", help="Also open the native Doom window")
    args = parser.parse_args()

    app = create_app(engine_factory(args))
    attach_doom(
        app,
        app.state.provide_engine,
        DoomSettings(
            scenario=args.scenario,
            tics=args.tics,
            resolution=args.resolution,
            max_actors=args.max_actors,
            window=args.window,
            recording=args.record,
        ),
    )
    print(f"Doom demo on http://127.0.0.1:{args.port}/doom")
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


def mmlu():
    from datasets import load_dataset

    from litjev.mmlu import DATASET_ID, convert_rows, ten_question_batches

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/batch-mcq")
    parser.add_argument("--output", default="mmlu-results.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-logits", help="Validation-only calibration JSONL")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.export_logits and (args.split != "validation" or args.prepare_only):
        parser.error("--export-logits requires validation inference")
    dataset = load_dataset(DATASET_ID, revision=args.revision, split=args.split)
    questions, labels, skipped = convert_rows(dataset)
    questions = questions[: args.limit]
    runs = []
    calibration_records = []
    for batch, valid_ids in ten_question_batches(questions):
        request = batch.model_dump()
        run = {"request": request, "scored_ids": valid_ids}
        if not args.prepare_only:
            wire = urllib.request.Request(
                args.url,
                data=json.dumps(request).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter()
            with urllib.request.urlopen(wire, timeout=1800) as response:
                result = json.load(response)
            run.update(response=result, elapsed_seconds=time.perf_counter() - started)
            run["correct"] = sum(
                result["answers"][key]["value"] == labels[key] for key in valid_ids
            )
        runs.append(run)
        if args.export_logits:
            for key in valid_ids:
                answer = result["answers"][key]
                choices = list(next(q for q in batch.questions if q.question_id == key).options)
                calibration_records.append(
                    {
                        "split": "validation",
                        "question_id": key,
                        "logits": answer["logits"],
                        "label": choices.index(labels[key]),
                    }
                )
    report = {
        "dataset": DATASET_ID,
        "revision": args.revision,
        "split": args.split,
        "count": len(questions),
        "skipped": skipped,
        "runs": runs,
        "labels": {q.question_id: labels[q.question_id] for q in questions},
    }
    if runs and not args.prepare_only:
        report["accuracy"] = sum(r["correct"] for r in runs) / len(questions)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if args.export_logits:
        Path(args.export_logits).write_text(
            "".join(json.dumps(row) + "\n" for row in calibration_records)
        )
    print(f"Saved {len(questions)} questions in {len(runs)} ten-question batches: {args.output}")


def calibrate():
    parser = argparse.ArgumentParser(description="Fit temperature on held-out logits JSONL")
    parser.add_argument("input")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Calibration input is empty")
    if any(row.get("split") != "validation" for row in rows):
        parser.error("Every calibration record must declare split=validation")
    max_choices = max(len(row["logits"]) for row in rows)
    logits = np.full((len(rows), max_choices), -1e30)
    for i, row in enumerate(rows):
        if not 0 <= row["label"] < len(row["logits"]):
            parser.error("Label outside candidate range")
        logits[i, : len(row["logits"])] = row["logits"]
    profile = TemperatureCalibrator.fit(logits, np.array([r["label"] for r in rows]), args.model)
    profile.save(args.output)
    print(json.dumps({"temperature": profile.temperature, "nll": profile.nll_after}))
