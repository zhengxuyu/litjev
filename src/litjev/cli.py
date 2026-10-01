import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from litjev.calibration import CalibrationProfile, TemperatureCalibrator
from litjev.slots import SLOT_FORMAT


def serve():
    import uvicorn

    from litjev.api import create_app
    from litjev.backend import ModelSettings, TransformersScorer
    from litjev.decision import DEFAULT_READOUT, READOUTS, SchemaDecisionEngine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--calibration")
    parser.add_argument("--decision-head", help="Trained head (.safetensors) for /v1/systemtwo")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    # Under 'coded', every option is given a letter code (A, B, C, ...) before it
    # reaches the model, and the answer is read off those codes. Option text must
    # therefore carry no numbering of its own -- no "A.", "1)", "(b)" in front of
    # it. The model would see two labels per option, the code and the caller's,
    # and the caller's can pull the answer towards the wrong code.
    parser.add_argument(
        "--readout", choices=sorted(READOUTS), default=DEFAULT_READOUT,
        help="how the fast mode reads its answer. The default scores each option's own "
             "text; 'coded' scores single-token letter codes, and a head trained under it "
             "has to be served under it. Under 'coded' the codes are added for you, so "
             "options must not carry their own numbering (A., 1), (b)); it can mislead "
             "the readout")
    parser.add_argument("--think-budget", type=int, default=0, help="0 disables default routing")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend", choices=("transformers", "sglang"), default="transformers",
        help="sglang is far faster at the slow mode; verify it first with inference.verify")
    parser.add_argument("--sglang-batch-questions", action="store_true",
                        help="send a request's questions to the engine together")
    args = parser.parse_args()

    def factory():
        from litjev.heads import DecisionHead
        from litjev.routing import RoutingPolicy

        profile = CalibrationProfile.load(args.calibration) if args.calibration else None
        if profile and profile.model_id != args.model:
            raise ValueError("Calibration profile model does not match serving model")
        head = DecisionHead.load(args.decision_head) if args.decision_head else None
        layers = ()
        if head is not None:
            head.metadata.check_serving(args.model, args.revision)
            layers = head.metadata.feature_layers
        if args.backend == "sglang":
            from inference import SGLangScorer, SGLangSettings

            scorer = SGLangScorer.load(SGLangSettings(
                model_id=args.model, revision=args.revision, feature_layers=layers,
                batch_questions=args.sglang_batch_questions))
            device = "cpu"
        else:
            settings = ModelSettings(
                args.model, args.revision, args.device_map, args.dtype, feature_layers=layers
            )
            scorer = TransformersScorer.load(settings)
            device = scorer.model.get_input_embeddings().weight.device
        if head is not None:
            if head.metadata.hidden_size != scorer.hidden_size:
                raise ValueError("Decision head hidden size does not match the model")
            head.to(device)
        return SchemaDecisionEngine(
            scorer,
            profile.temperature if profile else 1.0,
            args.model,
            profile is not None,
            head=head,
            readout=args.readout,
            routing=RoutingPolicy(args.lambda_, args.think_budget),
        )

    # One process owns one model; concurrent forwards are serialized in the backend.
    uvicorn.run(create_app(factory), host="127.0.0.1", port=args.port, workers=1)


def mmlu():
    from datasets import load_dataset

    from litjev.mmlu import DATASET_ID, convert_rows, ten_question_batches

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone/debug")
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
                envelope = json.load(response)
                result = envelope["result"]
            run.update(response=result, elapsed_seconds=time.perf_counter() - started)
            run["correct"] = sum(
                result["answers"][key]["choice"] == labels[key] for key in valid_ids
            )
        runs.append(run)
        if args.export_logits:
            for key in valid_ids:
                answer = envelope["diagnostics"]["fields"][key]
                choices = list(batch.questions[key].criteria)
                calibration_records.append(
                    {
                        "split": "validation",
                        "slot_format": SLOT_FORMAT,
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
    if any(row.get("slot_format") != SLOT_FORMAT for row in rows):
        parser.error(
            "Calibration logits must use the current slot_format; re-export validation logits"
        )
    max_choices = max(len(row["logits"]) for row in rows)
    logits = np.full((len(rows), max_choices), -1e30)
    for i, row in enumerate(rows):
        if not 0 <= row["label"] < len(row["logits"]):
            parser.error("Label outside candidate range")
        logits[i, : len(row["logits"])] = row["logits"]
    profile = TemperatureCalibrator.fit(logits, np.array([r["label"] for r in rows]), args.model)
    profile.save(args.output)
    print(json.dumps({"temperature": profile.temperature, "nll": profile.nll_after}))


def collect():
    from datasets import load_dataset

    from litjev.backend import ModelSettings, TransformersScorer
    from litjev.collect import collect_batch, save_records
    from litjev.heads import parse_feature_layers
    from litjev.mmlu import DATASET_ID, convert_rows, ten_question_batches

    parser = argparse.ArgumentParser(description="Collect fast/slow records for the decision head")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every k-th question so a small sample spans all categories",
    )
    parser.add_argument(
        "--layers",
        default="-1",
        help="hidden_states indices; write --layers=-1,40,48 (the value starts with '-')",
    )
    parser.add_argument("--budget", type=int, default=512, help="Thinking tokens per question")
    parser.add_argument("--output", default="head-records.npz")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Rewrite the output file every k batches so a timeout keeps what was collected",
    )
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0 or args.budget <= 0 or args.stride <= 0:
        parser.error("--limit, --budget and --stride must be positive, --offset non-negative")
    layers = parse_feature_layers(args.layers)
    dataset = load_dataset(DATASET_ID, revision=args.dataset_revision, split=args.split)
    questions, labels, skipped = convert_rows(dataset)
    categories = {str(row.get("question_id", "")): str(row.get("category", "")) for row in dataset}
    questions = questions[args.offset :: args.stride][: args.limit]
    if not questions:
        parser.error("No questions selected")
    settings = ModelSettings(
        args.model, args.revision, args.device_map, args.dtype, feature_layers=layers
    )
    scorer = TransformersScorer.load(settings)
    metadata = {
        "model_id": args.model,
        "revision": args.revision,
        "hidden_size": scorer.hidden_size,
        "feature_layers": list(layers),
        "budget": args.budget,
        "dataset": DATASET_ID,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "offset": args.offset,
        "stride": args.stride,
        "skipped": len(skipped),
    }
    records = []
    started = time.perf_counter()
    for batches, (batch, valid_ids) in enumerate(ten_question_batches(questions), start=1):
        records.extend(collect_batch(scorer, batch, valid_ids, labels, args.budget, categories))
        print(f"{len(records)}/{len(questions)} records, {time.perf_counter() - started:.0f}s")
        if args.checkpoint_every > 0 and batches % args.checkpoint_every == 0:
            save_records(args.output, records, {**metadata, "partial": True})
    save_records(args.output, records, metadata)
    fast = sum(r.fast_correct for r in records) / len(records)
    slow = sum(r.slow_correct for r in records) / len(records)
    print(json.dumps({"records": len(records), "fast_accuracy": fast, "slow_accuracy": slow}))


def train_head():
    from litjev.collect import load_records
    from litjev.training import DEFAULT_LAMBDAS, run_training

    parser = argparse.ArgumentParser(description="Train the decision head on collected records")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="decision-head.safetensors")
    parser.add_argument("--report", default="decision-head-report.json")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--select-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be in (0, 1)")
    records, metadata = load_records(args.inputs)
    head, report = run_training(
        records,
        metadata,
        args.holdout_fraction,
        args.select_layers,
        args.epochs,
        args.probe_epochs,
        args.seed,
        tuple(float(x) for x in args.lambdas.split(",")),
    )
    head.save(args.output)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "chosen": report["chosen"],
                "chosen_layers": report["chosen_layers"],
                "auroc_head": report["head"]["auroc_fast_correct"],
                "auroc_stats_probe": report["baseline_stats_probe"]["auroc_fast_correct"],
                "auroc_concentration": report["baseline_auroc_concentration"],
                "auroc_max_probability": report["baseline_auroc_max_probability"],
                "split_level": report["split_level"],
                "fast_accuracy": report["fast_accuracy"],
                "slow_accuracy": report["slow_accuracy"],
                "output": args.output,
            }
        )
    )
