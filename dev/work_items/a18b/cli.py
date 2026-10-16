"""Generate cases, calibrate judges and score the declared repertoire.

Each command uses a fresh evidence directory. Existing replies and ratings
remain immutable. Partial selection is diagnostic and cannot pass the suite.
External model failures and invalid reviews retain their raw evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import yaml

from dev.work_items.a18b.identity import save_review_origin, save_suite, saved_suite
from dev.work_items.a18b.policy import Fact, PublicMessage, ReviewCriterion, ReviewPacket, Scenario, score_case, score_suite
from dev.work_items.a18b.supervision import run_worker, worker_cancellation
from dev.work_items.a18b.review import evaluate, evaluate_panel
from dev.work_items.a18b.report import is_problem, render
from dev.work_items.a18b.runner import HERE, ROOT, load_suite, public_messages, run_case
from dev.work_items.shared.evidence import save_json, sha256_file


def model(config: Path):
    """Use the explicit evaluation preset. Readiness failures fail the gate."""
    from pm.llm.activation import ModelActivation
    from pm.llm.config import LlmPreset, load_config, resolve_connection
    from pm.llm.stage_model import LazyStageModel
    return LazyStageModel(resolve_connection(load_config(config), preset=LlmPreset.UnitTestEvaluator),
                          ModelActivation())


def fresh(path: Path) -> Path:
    """Require a new ignored run directory. Never overwrite prior evidence."""
    path = path.resolve()
    if not path.is_relative_to(ROOT / ".runtime"):
        raise ValueError("A18B evidence belongs under .runtime.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def review_case(root: Path, scenario, suite, evaluator, *, cancel=None) -> dict:
    """Review frozen packets after runtime cleanup, including a failed run."""
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    panels = {"editor": [], "grounding": []}
    errors = []
    if receipt["scenario"] != scenario.id or receipt["suite_version"] != suite.version:
        raise ValueError("The saved case does not match the selected scenario version.")
    setup = Scenario.model_validate_json((root / "private-setup.json").read_bytes())
    if setup != scenario:
        raise ValueError("The saved scenario changed. Use its original manifest revision.")
    if receipt["runtime_ok"] and (len(receipt["episodes"]) != len(scenario.scenes) or any(
            episode["sent"] != scene.messages or "completed_at" not in episode
            for episode, scene in zip(receipt["episodes"], scenario.scenes))):
        raise ValueError("The claimed completed case has incomplete scene evidence.")
    if suite.cadence_policy != "legacy" and receipt.get("cadence_policy") != suite.cadence_policy:
        raise ValueError("The runtime receipt does not establish the selected cadence policy.")
    for audience in panels:
        packet_path = root / f"{audience}-packet.json"
        original = sha256_file(packet_path)
        if receipt["packet_sha256"][audience] != original:
            raise ValueError("The frozen review packet differs from its terminal receipt.")
        packet = ReviewPacket.model_validate_json(packet_path.read_bytes())
        expected = tuple(ReviewCriterion(id=item.id, description=item.description)
                         for item in suite.common_criteria + scenario.criteria if item.audience == audience)
        if packet.criteria != expected:
            raise ValueError("The saved rubric changed. Use its original scenario and prompt revision.")
        if receipt["new_companion_messages"] == 0:
            errors.append(f"{audience}: No new companion speech. Historical speech cannot establish new behavior.")
            continue
        for repeat in range(suite.gate.review_repeats):
            try:
                panels[audience].append(evaluate_panel(evaluator, packet, root / "reviews" / f"{audience}-{repeat+1}", cancel=cancel))
            except Exception as error:
                errors.append(f"{audience}/{repeat+1}: {type(error).__name__}: {error}")
                save_json(root / "reviews" / f"{audience}-{repeat+1}-error.json", {"error": errors[-1], "traceback": traceback.format_exc()})
        if original != sha256_file(packet_path):
            raise RuntimeError("A frozen review packet changed during review.")
    worker = json.loads((root / "generation-worker.json").read_text(encoding="utf-8"))
    worker_ok = worker["returncode"] == 0 and worker["termination_reason"] is None
    if not worker_ok:
        errors.append(f"Worker process failed: exit={worker['returncode']}, reason={worker['termination_reason']}")
    result = score_case(scenario, suite, panels, runtime_ok=receipt["runtime_ok"] and worker_ok)
    manual = sorted((root / "reviews").rglob("human-review.json"))
    result["manual_review_required"] = [path.relative_to(root).as_posix() for path in manual]
    result["review_protocol_complete"] = not any(
        json.loads(path.read_text(encoding="utf-8"))["kind"] == "reviewer protocol failure" for path in manual)
    if not result["review_protocol_complete"]:
        result["complete"] = False
    if manual:
        # A host disposition is not missing model evidence turned into credit.
        # Keep all usable judgments, but require inspection before acceptance.
        result["strong"] = False
    result["review_errors"] = errors
    save_json(root / "score.json", result)
    return result


def summarize(root: Path, suite, rows: list[dict]) -> dict:
    """Retain the run scope when a saved report is rescored or filtered."""
    origin = json.loads((root / "run-origin.json").read_text(encoding="utf-8"))
    if saved_suite(root) != suite:
        raise ValueError("The score must use the run's frozen suite contract.")
    result = score_suite(suite, rows)
    result["scope"] = origin["scope"]
    save_json(root / "suite-score.json", result)
    return result


def calibration(config: Path, out: Path, repeats: int, *, cancel=None) -> dict:
    """Check the judges on labelled controls, never on fabricated PM results."""
    fixtures = yaml.safe_load((HERE / "calibration.yaml").read_text(encoding="utf-8"))
    suite = load_suite()
    rubric = {item.id: item for item in suite.common_criteria}
    evaluator = model(config)
    results = []
    started = datetime.now(timezone.utc)
    identity = {"prompts_sha256": sha256_file(HERE / "prompts.yaml"), "controls_sha256": sha256_file(HERE / "calibration.yaml")}
    for control in fixtures["controls"]:
        criterion = rubric[control["criterion"]]
        at = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)
        messages = tuple(PublicMessage(id=f"m{index:04}", at=at + timedelta(seconds=index*20),
            speaker=item["speaker"], text=item["text"]) for index, item in enumerate(control["messages"], start=1))
        facts = tuple(Fact(id=f"f{index:04}", at=at, about="whole excerpt", text=text)
                      for index, text in enumerate(control.get("facts", []), start=1))
        packet = ReviewPacket(audience=criterion.audience, premise=control["premise"], conversation=messages,
            criteria=(ReviewCriterion(id=criterion.id, description=criterion.description),), facts=facts)
        for repeat in range(repeats):
            row = {"control": control["id"], "expected": control["expected"], "repeat": repeat+1, "passed": False}
            try:
                answer = evaluate(evaluator, packet, out / control["id"] / str(repeat+1), cancel=cancel)
                row.update(observed=answer.judgments[0].verdict, reason=answer.judgments[0].reason,
                           passed=(answer.judgments[0].verdict == control["expected"]
                                   or control["expected"] == "pass" and answer.judgments[0].verdict == "good_enough"))
            except Exception as error:
                row.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            results.append(row)
            save_json(out / "receipt.json", {"kind": "synthetic reviewer calibration, not PM behavior",
                "started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
                **identity,
                "passed": all(item["passed"] for item in results), "complete": len(results) == len(fixtures["controls"])*repeats,
                "results": results, "human_review": "pending"})
            print(f"{row['control']}/{repeat+1}: {'pass' if row['passed'] else 'FAIL'}", flush=True)
    return json.loads((out / "receipt.json").read_text(encoding="utf-8"))


def failed_case(root: Path, scenario, suite, reason: str) -> dict:
    """Keep a preparation or process error visible without stopping later cases."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "score.json"
    if path.exists():
        (root / "partial-score.json").write_bytes(path.read_bytes())
    result = score_case(scenario, suite, {}, runtime_ok=False)
    result["review_errors"] = [reason]
    save_json(path, result)
    return result


def review_worker(root: Path, scenario, suite, config: Path) -> dict:
    """Bound each review phase and retain failure without aborting the repertoire."""
    try:
        if (root / "STOP").exists() or (root.parent / "STOP").exists():
            return failed_case(root, scenario, suite, "Review cancelled by STOP. Coverage remains incomplete.")
        command = [sys.executable, "-m", "dev.work_items.a18b", "review-case", "--out", str(root),
                   "--case", scenario.id, "--config", str(config.resolve())]
        with (root.parent / f"{scenario.id}-review.log").open("w", encoding="utf-8") as log:
            worker = run_worker(command, cwd=ROOT, case=root, log=log, timeout_seconds=suite.review_timeout_seconds)
        save_json(root / "review-worker.json", worker.to_dict())
        if worker.returncode != 0 or worker.termination_reason is not None:
            return failed_case(root, scenario, suite,
                f"Review worker failed: exit={worker.returncode}, reason={worker.termination_reason}. Inspect its log.")
        return json.loads((root / "score.json").read_text(encoding="utf-8"))
    except Exception as error:
        save_json(root / "controller-error.json", {"error": f"{type(error).__name__}: {error}",
                                                  "traceback": traceback.format_exc()})
        return failed_case(root, scenario, suite, f"{type(error).__name__}: {error}")


def bundle(out: Path, suite, report: dict) -> Path:
    """Expose case summaries and full transcripts without changing human feedback."""
    items = []
    for row in report["cases"]:
        scenario = next(item for item in suite.scenarios if item.id == row["scenario"])
        case = out / scenario.id
        lines = [f"**Automated criteria: {row['passed_criteria']}/{row['total_criteria']} · complete: {row['complete']}**",
                 "", "Model judgments are fallible. Human review is pending.", ""]
        try:
            packet = ReviewPacket.model_validate_json((case / "editor-packet.json").read_bytes())
        except (OSError, ValueError) as error:
            lines.append(f"Public packet unavailable: {type(error).__name__}: {error}")
            packet = None
        new_speech = False
        if packet is not None:
            after_boundary = False
            for message in packet.conversation:
                after_boundary = after_boundary or message.id == packet.review_from_message
                new_speech = new_speech or after_boundary and message.speaker == "Spamuel"
                lines.extend([f"**{message.at.isoformat()} · {message.speaker} · {message.id}**", "", message.text, ""])
        items.append({"id": scenario.id, "title": scenario.title, "persona": "Spamuel / full canon",
            "scenario": scenario.id, "context": [{"role": "Public premise", "text": scenario.premise}],
            "content": "\n".join(lines), "reviewable": new_speech,
            "notes": [f"Automated status: {'strong' if row['strong'] else 'needs review'}", *row["critical_failures"],
                *(["Reviewer evidence needs human inspection."] if row.get("manual_review_required") else []),
                *[f"Unresolved critical criterion: {item}" for item in row.get("critical_unresolved", [])], *row.get("review_errors", [])],
            "references": [{"label": "Exact score and criteria", "path": (case / "score.json").relative_to(ROOT).as_posix()},
                {"label": "Runtime receipt", "path": (case / "receipt.json").relative_to(ROOT).as_posix()}]})
    path = out / "review-bundle.json"
    save_json(path, {"schema": "pm.dev-review.bundle.v1", "id": out.name, "work_item": "A18B",
        "title": "A18B · blinded cognitive believability · " + out.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": f"{report['status']}: {report['evaluated_cases']}/{report['declared_cases']} declared cases evaluated. {report['scope']}. Human review pending.", "items": items})
    return path


def execution_batches(scenarios, jobs: int) -> list[list[tuple[int, Scenario]]]:
    """Keep normal cases first and give each Minecraft fixture exclusive body access.

    Each group contains at most one Minecraft fixture. Conversation cases use
    an isolated unavailable game endpoint and disable world polling. The next group
    starts after every worker in the current group completes and cleans up.
    """
    ordered = sorted(enumerate(scenarios), key=lambda item: (item[1].priority == "edge", item[0]))
    if jobs == 1:
        return [[item] for item in ordered]
    batches = []
    for priority in ("normal", "edge"):
        selected = [item for item in ordered if item[1].priority == priority]
        conversations = [item for item in selected if not item[1].minecraft]
        minecraft = [item for item in selected if item[1].minecraft]
        while conversations or minecraft:
            batch = [minecraft.pop(0)] if minecraft else []
            capacity = jobs - len(batch)
            batch.extend(conversations[:capacity])
            del conversations[:capacity]
            batches.append(batch)
    return batches


def run_cases(out: Path, suite, config: Path, selected: list[str], *, seed: int, jobs: int) -> list[dict]:
    """Run isolated cases and update one report after each terminal result.

    Progress records distinguish generation, review, and completed cases.
    STOP cancels active workers and prevents queued work from starting.
    Every case retains the full suite denominator and its original seed offset.
    """
    lock = Lock()
    progress = {item.id: "queued" for item in suite.scenarios if not selected or item.id in selected}

    def mark(scenario_id: str, state: str) -> None:
        with lock:
            progress[scenario_id] = state
            save_json(out / "progress.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
                "jobs": jobs, "cases": progress, "finished": all(value == "finished" for value in progress.values())})

    def execute(index: int, scenario: Scenario) -> dict:
        case = out / scenario.id
        if (out / "STOP").exists():
            mark(scenario.id, "cancelled")
            return failed_case(case, scenario, suite, "Cancelled before generation. No behavioral evidence.")
        mark(scenario.id, "generating")
        print(f"Generating {scenario.id}. Evidence: {case}", flush=True)
        command = [sys.executable, "-m", "dev.work_items.a18b", "run-case", "--out", str(case),
                   "--case", scenario.id, "--config", str(config.resolve()), "--seed", str(seed + index)]
        with (out / f"{scenario.id}.log").open("w", encoding="utf-8") as log:
            process = run_worker(command, cwd=ROOT, case=case, log=log, timeout_seconds=suite.case_timeout_seconds)
        save_json(out / f"{scenario.id}-worker.json", process.to_dict())
        case.mkdir(exist_ok=True)
        save_json(case / "generation-worker.json", process.to_dict())
        if not (case / "receipt.json").exists():
            result = failed_case(case, scenario, suite, "No terminal case receipt. Inspect worker log.")
        else:
            mark(scenario.id, "reviewing")
            result = review_worker(case, scenario, suite, config)
        mark(scenario.id, "finished")
        return result

    results = []
    batches = execution_batches(suite.scenarios, jobs)
    save_json(out / "progress.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs, "cases": progress, "finished": False})
    render(out, suite, summarize(out, suite, results))
    for batch in batches:
        wanted = [(index, item) for index, item in batch if not selected or item.id in selected]
        if not wanted or (out / "STOP").exists():
            continue
        with ThreadPoolExecutor(max_workers=min(jobs, len(wanted)), thread_name_prefix="a18b-case") as pool:
            pending = {pool.submit(execute, index, item): item for index, item in wanted}
            for future in as_completed(pending):
                scenario = pending[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    save_json(out / f"{scenario.id}-controller-error.json", {"error": str(error),
                        "traceback": traceback.format_exc()})
                    results.append(failed_case(out / scenario.id, scenario, suite, f"Controller: {type(error).__name__}: {error}"))
                    mark(scenario.id, "finished")
                results.sort(key=lambda row: next(index for index, item in enumerate(suite.scenarios) if item.id == row["scenario"]))
                with lock:
                    render(out, suite, summarize(out, suite, results))
    return results


def main() -> int:
    """Run a command. A selected subset always reports incomplete suite coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "calibrate", "calibrate-worker", "run", "run-case", "review-case", "review-capture", "score", "show", "review-saved"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.openrouter.yaml")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1,
                        help="Isolated case workers. At most one Minecraft fixture runs at a time.")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--problems", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least one.")
    if args.command == "review-saved":
        if args.source is None:
            parser.error("review-saved requires a saved suite directory as --source.")
        suite = saved_suite(args.source)
    elif args.command in {"score", "show", "calibrate-worker"} and args.out is not None:
        suite = saved_suite(args.out)
    elif args.command == "review-case" and args.out is not None:
        suite = saved_suite(args.out.parent)
    else:
        suite = load_suite()
    choices = {item.id: item for item in suite.scenarios}
    if set(args.case) - choices.keys():
        parser.error("Unknown case. Run list to see the declared repertoire.")
    if args.command == "list":
        print(json.dumps({"version": suite.version, "cadence_policy": suite.cadence_policy,
            "normal_cases": sum(row.priority == "normal" for row in suite.scenarios),
            "edge_cases": sum(row.priority == "edge" for row in suite.scenarios),
            "gate": suite.gate.model_dump(), "cases": [
            {"id": row.id, "priority": row.priority, "minecraft": row.minecraft, "episodes": len(row.scenes), "human_messages": sum(scene.messages for scene in row.scenes)}
            for row in suite.scenarios]}, indent=2))
        return 0
    if args.out is None:
        parser.error("This command requires --out.")
    if args.command in {"score", "show"}:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.out.glob("*/score.json")]
        result = summarize(args.out, suite, rows)
        print(render(args.out, suite, result, problems=args.problems))
        visible = [row for row in rows if not args.problems or is_problem(row)]
        print(json.dumps({**result, "cases": visible}, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "run-case":
        if len(args.case) != 1:
            parser.error("run-case requires exactly one --case.")
        frozen = saved_suite(args.out.resolve().parent)
        if frozen != suite:
            raise ValueError("The worker checkout differs from the controller's frozen suite.")
        result = run_case(choices[args.case[0]], suite, args.config.resolve(), args.out.resolve(), args.seed)
        return 0 if result["runtime_ok"] else 1
    if args.command == "review-case":
        if len(args.case) != 1:
            parser.error("review-case requires exactly one --case.")
        os.environ["PM_PROMPT_LOG_DIR"] = str(args.out.parent / "review-llm-logs")
        save_review_origin(args.out)
        with worker_cancellation(args.out, suite.review_timeout_seconds) as cancel:
            review_case(args.out, choices[args.case[0]], suite, model(args.config), cancel=cancel)
        return 0
    if args.command == "calibrate-worker":
        save_review_origin(args.out)
        with worker_cancellation(args.out, suite.review_timeout_seconds * 4) as cancel:
            result = calibration(args.config, args.out, args.repeats, cancel=cancel)
        return 0 if result["passed"] and result["complete"] else 1
    out = fresh(args.out)
    scope = {"run": "fresh runtime execution", "review-saved": "reviewer regression on frozen evidence; not runtime regression",
             "calibrate": "synthetic reviewer calibration; no companion runtime execution",
             "review-capture": "diagnostic editorial review of a saved public transcript"}[args.command]
    save_json(out / "run-origin.json", {"scope": scope, "suite_version": suite.version,
        "suite_snapshot_sha256": save_suite(out, suite),
        "manifest_sha256": sha256_file(HERE / "scenarios.yaml"), "prompts_sha256": sha256_file(HERE / "prompts.yaml"),
        "config_sha256": sha256_file(args.config),
        "source_run": str(args.source.resolve()) if args.source is not None else None,
        "source_origin_sha256": sha256_file(args.source / "run-origin.json") if args.command == "review-saved" else None,
        "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv})
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "review-llm-logs")
    if args.command == "calibrate":
        with (out / "calibration.log").open("w", encoding="utf-8") as log:
            worker = run_worker([sys.executable, "-m", "dev.work_items.a18b", "calibrate-worker",
                "--config", str(args.config.resolve()), "--out", str(out), "--repeats", str(args.repeats)],
                cwd=ROOT, case=out, log=log, timeout_seconds=suite.review_timeout_seconds * 4)
        save_json(out / "calibration-worker.json", worker.to_dict())
        if worker.returncode != 0 or worker.termination_reason is not None:
            return 1
        result = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        return 0 if result["passed"] and result["complete"] else 1
    if args.command == "review-capture":
        if args.source is None or len(args.case) != 1:
            parser.error("review-capture requires --source and one --case for the editorial rubric.")
        receipt = json.loads(args.source.read_text(encoding="utf-8"))
        scenario = choices[args.case[0]]
        packet = ReviewPacket(audience="editor", premise=scenario.premise, conversation=public_messages(receipt["messages"]),
            criteria=tuple(ReviewCriterion(id=item.id, description=item.description) for item in suite.common_criteria + scenario.criteria if item.audience == "editor"))
        evaluator = model(args.config)
        save_review_origin(out)
        reviews = [evaluate_panel(evaluator, packet, out / f"editor-{index+1}").model_dump(mode="json") for index in range(args.repeats)]
        save_json(out / "receipt.json", {"kind": "saved public transcript; diagnostic editorial review only",
            "source_sha256": sha256_file(args.source), "reviews": reviews, "human_review": "pending"})
        return 0
    if args.command == "review-saved":
        if args.source is None:
            parser.error("review-saved requires a saved suite directory as --source.")
        import shutil
        for scenario in suite.scenarios:
            source = args.source / scenario.id
            if not source.exists() or args.case and scenario.id not in args.case:
                continue
            target = out / scenario.id
            target.mkdir()
            for name in ("receipt.json", "private-setup.json", "editor-packet.json", "grounding-packet.json", "generation-worker.json"):
                path = source / name
                if path.is_file():
                    shutil.copyfile(path, target / name)
            save_json(target / "replay-origin.json", {"source": str(source.resolve()),
                "scope": "Frozen public and observation packets only. The companion runtime did not execute.",
                "source_receipt_sha256": sha256_file(source / "receipt.json")})
        results = [review_worker(out / scenario.id, scenario, suite, args.config) for scenario in suite.scenarios
                   if (out / scenario.id).exists()]
        report = summarize(out, suite, results)
        print(render(out, suite, report))
        if args.register:
            subprocess.run([sys.executable, "dev/review/server.py", "register", str(bundle(out, suite, report))], cwd=ROOT, check=True)
        return 0 if report["passed"] else 1
    results = run_cases(out, suite, args.config, args.case, seed=args.seed, jobs=args.jobs)
    report = summarize(out, suite, results)
    render(out, suite, report)
    path = bundle(out, suite, report)
    if args.register:
        subprocess.run([sys.executable, "dev/review/server.py", "register", str(path)], cwd=ROOT, check=True)
    print(path, flush=True)
    return 0 if report["passed"] else 1
