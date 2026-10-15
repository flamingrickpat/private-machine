"""Review captured development results through a standalone local server.

The register owns immutable bundle references. Ignored JSON files own captured
content and human feedback. This tool imports no companion code and runs no
workflows. Invalid input receives a plain error. Code errors keep tracebacks.
Windows file locks retry the same atomic replacement until it succeeds.
Stop the server with Ctrl+C. Each successful save reaches disk before its reply.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "pm.dev-review.bundle.v1"
SCORES = ("naturalness", "self_explanation", "immersion", "usefulness", "groundedness")


def now() -> str:
    """Return the UTC time for registration and human feedback history."""
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict:
    """Read one UTF-8 JSON object. Invalid files retain their parse error."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    """Replace JSON atomically. A Windows lock retries without dropping feedback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            print(f"{error}. Waiting for the JSON file lock.", flush=True)
            time.sleep(0.2)


def identifier(value: str) -> str:
    """Validate an external identifier before using it as a filename."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ValueError("Use letters, numbers, dots, underscores, or hyphens in an id.")
    return value


def relative(path: Path) -> str:
    """Return a repository-relative reference for this local register."""
    return path.resolve().relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    """Return the digest of a captured source without changing that source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(bundle: dict, registry_path: Path, data: Path) -> str:
    """Validate and copy a bundle, then register its immutable content digest.

    Existing ids cannot acquire different content. Use a new id for a new run.
    Re-registration preserves all feedback. A failed register write leaves the
    captured bundle on disk and can be retried with the same input.
    """
    required = {"schema", "id", "title", "work_item", "created_at", "summary", "items"}
    if not required <= bundle.keys():
        raise ValueError("The bundle is missing required fields. See dev/review/README.md.")
    if bundle["schema"] != SCHEMA:
        raise ValueError(f"Use bundle schema {SCHEMA}.")
    bundle_id = identifier(bundle["id"])
    seen = set()
    for item in bundle["items"]:
        required_item = {"id", "title", "persona", "scenario", "context", "content", "reviewable", "notes", "references"}
        if not required_item <= item.keys():
            raise ValueError("A review item is missing required fields. See dev/review/README.md.")
        item_id = identifier(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate review item: {item_id}")
        seen.add(item_id)
        if not isinstance(item["content"], str):
            raise ValueError("Review content must be text.")
        for context in item["context"]:
            if not isinstance(context["role"], str) or not isinstance(context["text"], str):
                raise ValueError("Context roles and content must be text.")
        for ref in item["references"]:
            reference_path(ref["path"])
    target = data / "bundles" / f"{bundle_id}.json"
    encoded = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise ValueError("This bundle id already has different content. Use a new id.")
    write(target, bundle)
    registry = load(registry_path) if registry_path.exists() else {
        "schema": "pm.dev-review.register.v1", "bundles": []}
    row = {key: bundle[key] for key in ("id", "title", "work_item", "created_at")}
    row.update(path=relative(target), sha256=digest(target), items=len(bundle["items"]))
    row["reviewable"] = sum(item["reviewable"] for item in bundle["items"])
    registry["bundles"] = [old for old in registry["bundles"] if old["id"] != bundle_id] + [row]
    write(registry_path, registry)
    return bundle_id


def reference_path(value: str) -> Path:
    """Allow explicit review references under the repo, excluding credentials.

    References are selected when a bundle is registered. The server exposes no
    directory browser and no route that accepts an arbitrary filesystem path.
    """
    path = (ROOT / value).resolve()
    name = path.relative_to(ROOT).as_posix()
    if any(part in {".git", ".venv"} or part.startswith(".env") for part in path.parts):
        raise ValueError("Credential and environment files are not review references.")
    if path.name == "ghost.json":
        raise ValueError("Use a captured receipt, not a live checkpoint, as a review reference.")
    if not name.startswith((".runtime/", "dev/", "docs/", "pm/", "workflows/", "tests/", "test_data/")):
        raise ValueError("Use an evidence, documentation, code, or test reference.")
    return path


def registered(registry_path: Path, bundle_id: str) -> dict:
    """Load a registered bundle and verify its immutable digest."""
    row = next(row for row in load(registry_path)["bundles"] if row["id"] == bundle_id)
    path = ROOT / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError("The captured bundle changed after registration.")
    return load(path)


def feedback(data: Path, bundle_id: str) -> dict:
    """Read feedback separately from the captured result and its source receipt."""
    path = data / "feedback" / f"{identifier(bundle_id)}.json"
    return load(path) if path.exists() else {"schema": "pm.dev-review.feedback.v1", "items": {}}


def save_feedback(data: Path, bundle: dict, item_id: str, proposal: dict) -> dict:
    """Validate browser input and preserve each prior revision before replacement.

    The single-threaded HTTP server serializes writes. Revision checks reject
    a stale browser tab instead of overwriting newer feedback. Draft ratings
    remain distinct from submitted human review.
    """
    item = next(item for item in bundle["items"] if item["id"] == item_id)
    if not item["reviewable"]:
        raise ValueError("This case has no captured reply to rate.")
    for key in SCORES:
        value = proposal[key]
        if value is not None and (type(value) is not int or not 1 <= value <= 5):
            raise ValueError("Ratings must be empty or integers from 1 to 5.")
    if proposal["regenerate"] not in (None, "yes", "no"):
        raise ValueError("Choose yes, no, or undecided for regeneration.")
    if proposal["status"] not in {"draft", "reviewed"}:
        raise ValueError("Feedback status must be draft or reviewed.")
    for key in ("reviewer", "reason", "unsupported_claims", "intended_deception", "details"):
        if not isinstance(proposal[key], str):
            raise ValueError(f"Feedback {key} must be text.")
    if proposal["status"] == "reviewed" and not (proposal["reason"].strip() or proposal["details"].strip()
                                                or any(proposal[key] is not None for key in SCORES)):
        raise ValueError("Add a rating or feedback before marking this item reviewed.")
    document = feedback(data, bundle["id"])
    previous = document["items"].get(item_id)
    revision = previous["revision"] if previous else 0
    if proposal["revision"] != revision:
        raise FileExistsError("This feedback changed in another tab. Reload before saving.")
    accepted = {key: proposal[key] for key in (*SCORES, "regenerate", "status", "reviewer", "reason",
                                             "unsupported_claims", "intended_deception", "details")}
    accepted.update(updated_at=now(), revision=revision + 1,
                    history=[*previous["history"], {key: value for key, value in previous.items() if key != "history"}]
                    if previous else [])
    document["items"][item_id] = accepted
    write(data / "feedback" / f"{bundle['id']}.json", document)
    return accepted


def source_turns(work_item: str, run_root: Path) -> tuple[dict[str, str], Path]:
    """Read literal gate prompts through the AST without importing runtime code."""
    filename = "alpha_a18_self_model_gate.py" if work_item == "A18" else "alpha_a17_private_loop_gate.py"
    captured = run_root / "source" / filename
    source = captured if captured.is_file() else ROOT / "dev/work_items" / filename
    values = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"TURNS", "REOPEN_TURN"}:
                    values[target.id] = ast.literal_eval(node.value)
    return dict([*values["TURNS"], values["REOPEN_TURN"]]), source


def import_receipt(path: Path, work_item: str, title: str) -> dict:
    """Convert a completed A17/A18 or console receipt into review cards.

    This reads receipts only. It never opens a companion checkpoint. Historical
    ratings are not inferred from model scores or a successful gate result.
    """
    receipt = load(path)
    refs = [{"label": "Captured receipt", "path": relative(path)}]
    guides = {"A18": "a18-self-model.md", "A17": "a17-private-loop.md",
              "A09R": "a09r-openrouter-delivery.md"}
    if work_item in guides:
        refs.append({"label": f"{work_item} verification", "path": "docs/work_items/verification/" + guides[work_item]})
    bundle_id = identifier((path.parent.parent.name + "-" + path.parent.name + "-personas")
                           if "results" in receipt else path.parent.name + "-conversation")
    items = []
    prior = []
    notes = ["Captured application evidence. Review the reply and its fit to the situation."]
    if "ok" in receipt:
        notes.append(f"Mechanical gate: {'passed' if receipt['ok'] else 'failed'}. This is separate from human quality.")
    notes.extend(str(value) for value in receipt.get("failures", []))
    if "results" in receipt:
        for result in receipt["results"]:
            persona, case = result["persona"], result["case"]
            packet_path = path.parent / "matched-evidence.json"
            packet = load(packet_path)
            states = ", ".join(f"{name}: {value['strongest']}" for name, value in packet["overlay_initial"]["findings"].items())
            context = [{"role": "Shared evidence", "text": states},
                       {"role": "Limits", "text": "\n".join(packet["limits"])},
                       {"role": "Jesse", "text": result["question"]}]
            items.append(dict(id=f"{persona}-{case}", title=case.replace("_", " ").title(),
                              persona=persona, scenario=case, context=context, content=result["reply"],
                              reviewable=True, notes=[receipt["scope"], "Both personas received the same facts and question.",
                                                     "Evidence digest: " + result["matched_evidence_sha256"]],
                              references=[*refs, {"label": "Matched evidence", "path": relative(packet_path)},
                                          {"label": "Exact request and persona card", "path": relative(path.parent / f"{persona}-{case}-request.json")}]))
    else:
        if "facts" in receipt:
            if work_item not in {"A17", "A18"}:
                raise ValueError("Use a generic review bundle for this gate's context and prompts.")
            prompts, prompt_source = source_turns(work_item, path.parent)
            captured_prompts = prompt_source.resolve().is_relative_to(path.parent.resolve())
            refs.append({"label": "Captured gate prompts" if captured_prompts else "Gate prompts (current source)",
                         "path": relative(prompt_source)})
            if not captured_prompts:
                notes.append("Scripted prompts come from the current gate source; this run has no captured gate source file.")
            prior = [{"role": "Run setup", "text": "A fresh application instance loads the preserved persona history. "
                      "The gate waits for private activity before sending the first user turn. "
                      "Seed: " + receipt["seed"]}]
            idle = receipt["facts"].get("idle", {})
            if idle.get("main_reaction_summary"):
                prior.append({"role": "Recorded initial reaction", "text": idle["main_reaction_summary"]})
            overlay = receipt["facts"].get("overlay_initial")
            if overlay:
                states = ", ".join(f"{name}: {value['strongest']}" for name, value in overlay["findings"].items())
                prior.append({"role": "Initial inspected capabilities", "text": states})
            rows = []
            for case, question in prompts.items():
                result = receipt["facts"].get("turns", {}).get(case)
                if case == "reopen":
                    result = receipt["facts"].get("restart")
                    replies = result.get("reply", []) if result else []
                else:
                    replies = result.get("replies", []) if result else []
                if isinstance(replies, str):
                    replies = [replies] if replies else []
                rows.append((case, question, "\n\n".join(replies), "", result is not None))
            notes.append("The gate groups replies by arrival order. Compare the content with the actual question.")
        else:
            prior = [{"role": "Run setup", "text": "Captured console conversation. Persona: "
                      + str(receipt.get("persona", "not recorded"))
                      + ". The run's source and mechanical facts are in the attached receipt."}]
            rows = [(row["name"], row.get("input", "[Original prompt not retained in this receipt; this is a recovered pending reply.]"),
                     row["reply"]["text"] if row.get("reply") else "",
                     row.get("correlation_id", ""), True) for row in receipt["exchanges"]]
        for case, question, reply, correlation, attempted in rows:
            context = [*prior, {"role": "Jesse · current prompt", "text": question}]
            item_notes = [*notes]
            if case == "reopen":
                item_notes.append("After shutdown and checkpoint reopen. Check continuity and claims about stopped time.")
            if not reply:
                item_notes.append("No captured reply. This case cannot receive a quality rating.")
            if correlation:
                item_notes.append("Exchange correlation: " + correlation)
            items.append(dict(id=identifier(case), title=case.replace("_", " ").replace("-", " ").title(),
                              persona=receipt.get("persona", "Spamuel / dramatic"), scenario=case,
                              context=context, content=reply, reviewable=bool(reply), notes=item_notes,
                              references=refs))
            if attempted:
                prior = [*context, {"role": "Companion · captured reply", "text": reply or "[No reply]"}]
    return dict(schema=SCHEMA, id=bundle_id, title=title, work_item=work_item,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                summary=f"{len(items)} captured cases. Human review remains separate from mechanical checks.",
                source={"path": relative(path), "sha256": digest(path)}, items=items)


class ReviewHandler(BaseHTTPRequestHandler):
    """Serve the review UI, registered evidence, and serialized feedback saves."""

    def log_message(self, format: str, *args) -> None:
        """Keep the local request log readable in the launch terminal."""
        print(format % args, flush=True)

    def respond(self, status: int, value, content_type: str = "application/json") -> None:
        """Send a noncached response. User text is rendered as text in the UI."""
        body = value.encode("utf-8") if content_type.startswith("text/") else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Read the catalog, a bundle, or one explicitly registered reference."""
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.respond(403, {"error": "Open the review page at its printed 127.0.0.1 address."})
            return
        route = urlsplit(self.path)
        query = parse_qs(route.query)
        registry_path, data = self.server.registry_path, self.server.data_path
        try:
            if route.path == "/":
                self.respond(200, (HERE / "app.html").read_text(encoding="utf-8"), "text/html")
            elif route.path == "/api/catalog":
                rows = []
                for row in load(registry_path)["bundles"]:
                    path = ROOT / row["path"]
                    saved = feedback(data, row["id"])
                    reviewed = sum(value["status"] == "reviewed" for value in saved["items"].values())
                    rows.append({**row, "available": path.is_file(), "reviewed": reviewed,
                                 "drafts": sum(value["status"] == "draft" for value in saved["items"].values())})
                self.respond(200, {"bundles": rows})
            elif route.path == "/api/bundle":
                bundle = registered(registry_path, query["id"][0])
                self.respond(200, {"bundle": bundle, "feedback": feedback(data, bundle["id"])})
            elif route.path == "/api/reference":
                bundle = registered(registry_path, query["bundle"][0])
                item = next(item for item in bundle["items"] if item["id"] == query["item"][0])
                ref = item["references"][int(query["ref"][0])]
                path = reference_path(ref["path"])
                raw = path.read_bytes()
                encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
                self.respond(200, {"label": ref["label"], "path": ref["path"],
                                   "content": raw.decode(encoding)})
            else:
                self.respond(404, {"error": "No review route exists here."})
        except (ValueError, KeyError, StopIteration, FileNotFoundError) as error:
            self.respond(400, {"error": str(error) or "The requested review result is not available."})

    def do_POST(self) -> None:
        """Save feedback from this local UI. Other browser origins cannot save."""
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if (self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}"
                or self.headers.get("X-Review-Request") != "1" or (origin is not None and origin != expected)):
            self.respond(403, {"error": "Save feedback from the local review page."})
            return
        if self.path != "/api/feedback":
            self.respond(404, {"error": "No review route exists here."})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bundle = registered(self.server.registry_path, request["bundle_id"])
            accepted = save_feedback(self.server.data_path, bundle, request["item_id"], request["feedback"])
            self.respond(200, accepted)
        except FileExistsError as error:
            self.respond(409, {"error": str(error)})
        except (ValueError, KeyError, StopIteration) as error:
            self.respond(400, {"error": str(error) or "The review item is not registered."})


def main() -> None:
    """Register captured runs, print feedback, or serve the standalone UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=HERE / "registry.json")
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/dev-review")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local review desk. Stop it with Ctrl+C.")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    capture = sub.add_parser("import-receipt", help="Register a completed A17, A18, or console receipt.")
    capture.add_argument("receipt", type=Path)
    capture.add_argument("--work-item", required=True)
    capture.add_argument("--title", required=True)
    capture.add_argument("--id", help="Use a new id when registering a different context selection for the same receipt.")
    add = sub.add_parser("register", help="Register a generic review bundle from JSON.")
    add.add_argument("bundle", type=Path)
    sub.add_parser("feedback", help="Print saved human feedback for the next agent session.")
    args = parser.parse_args()
    if args.command == "import-receipt":
        bundle = import_receipt(args.receipt, args.work_item, args.title)
        if args.id:
            bundle["id"] = args.id
        print(register(bundle, args.registry, args.data))
    elif args.command == "register":
        print(register(load(args.bundle), args.registry, args.data))
    elif args.command == "feedback":
        print(json.dumps([{ "bundle_id": row["id"], "work_item": row["work_item"],
                            "feedback": feedback(args.data, row["id"])["items"]}
                          for row in load(args.registry)["bundles"]
                          if feedback(args.data, row["id"])["items"]], indent=2, ensure_ascii=False))
    else:
        server = HTTPServer(("127.0.0.1", args.port), ReviewHandler)
        server.registry_path, server.data_path = args.registry, args.data
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Development review: {url}\nFeedback: {args.data / 'feedback'}\nStop with Ctrl+C.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Review server stopped.", flush=True)
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
