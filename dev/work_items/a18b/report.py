"""Render saved judgments and transcripts without another model call.

Every inserted story string is HTML-escaped. Filtering changes the view, not
the score or declared case count. Missing cases remain visible as incomplete.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

from dev.work_items.a18b.policy import Review, ReviewPacket


def is_problem(score: dict) -> bool:
    """Include incomplete, disputed and failed criteria in the problem view."""
    return not score["complete"] or any(value != "pass" for value in score["criteria"].values())


def render(root: Path, suite, report: dict, *, problems: bool = False) -> Path:
    """Write a local report. Filtering changes its initial view, not its score."""
    rows = {item["scenario"]: item for item in report["cases"]}
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>A18B review</title>",
        "<style>body{font:16px system-ui;max-width:1050px;margin:3rem auto;padding:1rem;background:#f6f5ef;color:#172b2b}"
        "article{background:white;border:1px solid #cbd8cf;padding:1rem;margin:1rem 0}summary{cursor:pointer}"
        "pre{white-space:pre-wrap}button{padding:.5rem;margin:.3rem}blockquote{border-left:3px solid #739b8b;padding-left:1rem}"
        ".muted{color:#52655e}</style>", "<h1>A18B cognitive believability</h1>",
        f"<p>{escape(report['status'])}: {report['evaluated_cases']}/{report['declared_cases']} cases evaluated; "
        f"{report['strong_cases']} strong cases. Human review is pending.</p>",
        f"<p><strong>Scope: {escape(report.get('scope', 'mechanism check only'))}</strong></p>",
        "<p>" + ("Legacy timing: not a normal-cadence acceptance run." if suite.cadence_policy == "legacy"
                    else "Primary cadence: wait for actual companion speech. Explicit edge cases contain bursts.") + "</p>",
        "<p class='muted'>Model judgments are fallible. Saved-case review does not execute changed runtime code.</p>",
        "<button onclick=\"filterCases(false)\">All cases</button><button onclick=\"filterCases(true)\">Problematic cases</button>"]
    for scenario in suite.scenarios:
        score = rows.get(scenario.id)
        problem = score is None or is_problem(score)
        parts.append(f"<article data-problem='{str(problem).lower()}'><h2>{escape(scenario.title)}</h2>")
        if score is None:
            parts.append("<p>No terminal score. This case remains in the suite denominator.</p>")
            failure = root / f"{scenario.id}-worker-error.json"
            if failure.exists():
                parts.append("<pre>" + escape(failure.read_text(encoding="utf-8")) + "</pre>")
            parts.append("</article>")
            continue
        parts.append(f"<p>{score['passed_criteria']}/{score['total_criteria']} criteria; complete: {score['complete']}.</p>")
        try:
            receipt = json.loads((root / scenario.id / "receipt.json").read_text(encoding="utf-8"))
            failures = receipt["failures"]
            if receipt.get("cadence_policy"):
                parts.append(f"<p>Cadence: {escape(receipt['cadence_policy'])}; priority: {escape(receipt['priority'])}.</p>")
        except (OSError, ValueError, KeyError) as error:
            failures = [f"Runtime receipt unavailable: {type(error).__name__}: {error}"]
        for failure in [*failures, *score.get("review_errors", [])]:
            parts.append(f"<p><strong>Failure:</strong> {escape(failure)}</p>")
        parts.append("<pre>" + escape(json.dumps(score["criteria"], indent=2)) + "</pre>")
        review_root = root / scenario.id / "reviews"
        review_paths = list(review_root.glob("*/review.json"))
        review_paths.extend(path for path in review_root.glob("*/part-*/review.json")
                            if not (path.parent.parent / "review.json").exists())
        for path in sorted(review_paths):
            try:
                review = Review.model_validate_json(path.read_bytes())
            except (OSError, ValueError) as error:
                parts.append(f"<p>Review unavailable: {escape(str(error))}</p>")
                continue
            label = path.parent.relative_to(review_root).as_posix()
            if path.parent.name.startswith("part-"):
                label += " (incomplete panel)"
            parts.append(f"<details><summary>{escape(label)}: {escape(review.summary)}</summary>")
            for judgment in review.judgments:
                parts.append(f"<p><strong>{escape(judgment.criterion_id)}: {escape(judgment.verdict)}</strong> {escape(judgment.reason)}</p>")
                for citation in judgment.citations:
                    parts.append(f"<blockquote>{escape(citation.source_id)}: {escape(citation.quote)}</blockquote>")
            parts.append("</details>")
        try:
            packet = ReviewPacket.model_validate_json((root / scenario.id / "editor-packet.json").read_bytes())
        except (OSError, ValueError) as error:
            parts.append(f"<p>Public packet unavailable: {escape(str(error))}</p></article>")
            continue
        parts.append("<details><summary>Full supplied public story</summary>")
        for message in packet.conversation:
            if packet.review_from_message == message.id:
                parts.append("<hr><p><strong>New case messages start here. Earlier messages supply public background.</strong></p>")
            parts.append(f"<p><strong>{escape(message.id)} · {escape(message.at.isoformat())} · {message.speaker}</strong></p><p>{escape(message.text)}</p>")
        parts.append("</details></article>")
    parts.append("<script>function filterCases(problemOnly){document.querySelectorAll('article').forEach(el=>"
                 "el.hidden=problemOnly&&el.dataset.problem!=='true')}"
                 + ("filterCases(true);" if problems else "") + "</script></html>")
    path = root / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
