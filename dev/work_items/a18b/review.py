"""Review immutable packets through fresh, stateless model calls.

The model sees one system instruction and one JSON data turn. It receives no
scenario direction, earlier rating, private history or other review. Invalid
output remains on disk and raises; a valid low score is never regenerated.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from functools import cache
from pathlib import Path

import yaml

from dev.work_items.a18b.constants import MAX_TEST_INPUT_TOKENS, REVIEW_CRITERIA_PER_CALL, REVIEW_REASONING_TOKENS
from dev.work_items.a18b.policy import CitationError, Closed, JessePacket, Review, ReviewPacket, validate_review
from dev.work_items.shared.evidence import save_json
from pm.context_render.turns import Role, Turn
from pm.llm.context_window import estimate_tokens
from pm.llm.reasoning import ReasoningOptions

HERE = Path(__file__).resolve().parent


class RepairedSpan(Closed):
    """Supply an exact span for one existing quotation, without a new verdict."""

    index: int
    quote: str


class CitationRepair(Closed):
    """The proofreader can repair quotations only, never judgments or scores."""

    spans: tuple[RepairedSpan, ...]


@cache
def prompts() -> dict:
    """Read versioned instructions. No role receives the whole prompt file."""
    return yaml.safe_load((HERE / "prompts.yaml").read_text(encoding="utf-8"))


def review_turns(packet: ReviewPacket) -> list[Turn]:
    """Serialize the closed packet as data, without transcript role promotion."""
    return [Turn(Role.System, prompts()[packet.audience]), Turn(Role.User, packet.model_dump_json())]


def input_budget(model_name: str, turns: list[Turn], *, schema: dict | None = None) -> tuple[int, str]:
    """Bound a test-agent input, including its response schema when supplied.

    The schema is an extra estimation turn only. It never becomes a model
    message. Over-budget input raises before inference, without truncation.
    """
    measured = [*turns, Turn(Role.User, json.dumps(schema))] if schema is not None else turns
    tokens, measurement = estimate_tokens(model_name, measured)
    if tokens > MAX_TEST_INPUT_TOKENS:
        raise ValueError(f"The test-agent input exceeds its {MAX_TEST_INPUT_TOKENS} estimated-token allowance. Nothing was truncated.")
    return tokens, measurement


def validate_provider_budget(usage: dict) -> None:
    """Reject a result when reported input exceeds the shared test allowance."""
    reported = usage.get("prompt_tokens")
    if type(reported) is int and reported > MAX_TEST_INPUT_TOKENS:
        raise ValueError("The provider reported test input above the context allowance. This result cannot establish acceptance.")


def review_parts(packet: ReviewPacket) -> tuple[ReviewPacket, ...]:
    """Partition criteria without dropping, duplicating, or summarizing evidence."""
    return tuple(packet.model_copy(update={"criteria": packet.criteria[index:index + REVIEW_CRITERIA_PER_CALL]})
                 for index in range(0, len(packet.criteria), REVIEW_CRITERIA_PER_CALL))


def review_schema(packet: ReviewPacket | None = None) -> dict:
    """Make the provider schema enforce citations for decided judgments.

    An uncertain judgment can omit citations. A pass or fail cannot. This
    matches the host validator instead of asking for a looser wire format.
    """
    schema = Review.model_json_schema()
    judgment = schema["$defs"]["Judgment"]
    properties = judgment["properties"]
    required = judgment["required"]
    # Each branch is a complete object. Some provider schema compilers drop
    # the surrounding required fields when a branch contains only refinements.
    schema["$defs"]["Judgment"] = {"title": "Judgment", "anyOf": [
        {"type": "object", "additionalProperties": False,
         "properties": {**properties, "verdict": {"enum": ["pass", "fail"]},
                        "citations": {**properties["citations"], "minItems": 1}},
         "required": [*required, "citations"]},
        {"type": "object", "additionalProperties": False,
         "properties": {**properties, "verdict": {"const": "uncertain"}}, "required": required},
    ]}
    if packet is not None:
        count = len(packet.criteria)
        schema["properties"]["judgments"].update(minItems=count, maxItems=count)
        wanted = [item.id for item in packet.criteria]
        for variant in schema["$defs"]["Judgment"]["anyOf"]:
            variant["properties"]["criterion_id"] = {"const": wanted[0]} if count == 1 else {"enum": wanted}
    return schema


def evaluate_panel(model, packet: ReviewPacket, out: Path, *, cancel=None) -> Review:
    """Review small criterion groups in fresh contexts, then join exact judgments.

    Calls share only the original admissible evidence. No call receives an
    earlier judgment or summary. A missing group leaves the panel incomplete.
    """
    parts = review_parts(packet)
    if len(parts) == 1:
        return evaluate(model, packet, out, cancel=cancel)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "panel-plan.json", {"max_input_tokens": MAX_TEST_INPUT_TOKENS,
        "criteria_per_call": REVIEW_CRITERIA_PER_CALL,
        "groups": [[item.id for item in part.criteria] for part in parts]})
    reviews = [evaluate(model, part, out / f"part-{index:02}", cancel=cancel)
               for index, part in enumerate(parts, start=1)]
    result = Review(judgments=tuple(item for review in reviews for item in review.judgments),
                    summary="\n".join(f"Group {index}: {review.summary}" for index, review in enumerate(reviews, start=1)))
    validate_review(result, packet)
    save_json(out / "review.json", result.model_dump(mode="json"))
    return result


def exact_quote_span(source: str, attempted: str) -> str | None:
    """Return a unique exact span with the same words, or reject the repair.

    Punctuation, whitespace, case and apostrophe typography can differ.
    Every word and its order must match. The returned span preserves original
    source case. Ambiguous matches return None.
    The result always comes from the original source, never a paraphrase.
    """
    if attempted and attempted in source:
        return attempted
    pattern = r"\w+(?:['’]\w+)*"
    original = list(re.finditer(pattern, source))
    wanted = [match.group().replace("’", "'").casefold() for match in re.finditer(pattern, attempted)]
    if not wanted:
        return None
    tokens = [match.group().replace("’", "'").casefold() for match in original]
    matches = [index for index in range(len(tokens) - len(wanted) + 1)
               if tokens[index:index + len(wanted)] == wanted]
    if len(matches) != 1:
        return None
    start = matches[0]
    return source[original[start].start():original[start + len(wanted) - 1].end()]


def evaluate(model, packet: ReviewPacket, out: Path, *, cancel=None) -> Review:
    """Persist input and raw output. Invalid reviews raise; cancellation stops I/O."""
    out.mkdir(parents=True, exist_ok=False)
    encoded = packet.model_dump_json().encode("utf-8")
    (out / "packet.json").write_bytes(encoded)
    turns = review_turns(packet)
    schema = review_schema(packet)
    tokens, measurement = input_budget(model.connection.model, turns, schema=schema)
    save_json(out / "request.json", {"messages": [{"role": row.role.value, "content": row.content.text} for row in turns],
        "packet_sha256": hashlib.sha256(encoded).hexdigest(), "estimated_input_tokens": tokens, "measurement": measurement,
        "model": model.connection.model, "prompt_version": "a18b.review.v7",
        "max_input_tokens": MAX_TEST_INPUT_TOKENS, "estimate_includes_response_schema": True,
        "system_prompt_sha256": hashlib.sha256(turns[0].content.text.encode("utf-8")).hexdigest()})
    started = time.monotonic()
    completion = model.complete_result(turns, temperature=0.2, reasoning=ReasoningOptions(max_tokens=REVIEW_REASONING_TOKENS), cancel=cancel,
        response_format={"type": "json_schema", "json_schema": {
            "name": "a18b_review_v6", "strict": True, "schema": schema}})
    save_json(out / "raw.json", {"text": completion.text, "usage": completion.usage, "model": completion.model,
        "elapsed_seconds": round(time.monotonic() - started, 3)})
    validate_provider_budget(completion.usage)
    review = Review.model_validate(json.loads(completion.text))
    try:
        validate_review(review, packet)
    except CitationError:
        texts = {item.id: item.text for item in (*packet.conversation, *packet.facts)}
        bad = [(judgment_index, citation_index, citation) for judgment_index, judgment in enumerate(review.judgments)
               for citation_index, citation in enumerate(judgment.citations) if citation.quote not in texts[citation.source_id]]
        repair_input = [{"index": index, "source": texts[citation.source_id], "attempted_quote": citation.quote}
                        for index, (_, _, citation) in enumerate(bad)]
        normalized = [RepairedSpan(index=item["index"], quote=span) for item in repair_input
                      if (span := exact_quote_span(item["source"], item["attempted_quote"])) is not None]
        save_json(out / "citation-normalization.json", {"policy": "unique word span; restore original source typography and case",
            "spans": [item.model_dump() for item in normalized]})
        unresolved = [item for item in repair_input if item["index"] not in {span.index for span in normalized}]
        repaired_spans = list(normalized)
        for item in unresolved:
            # Each proofreader sees one source and one rejected quotation.
            # It receives no other source, verdict, score, reason or case fact.
            repair_input = [{**item, "index": 0}]
            repair_turns = [Turn(Role.System, prompts()["citation_repair"]), Turn(Role.User, json.dumps(repair_input, ensure_ascii=False))]
            repair_schema = CitationRepair.model_json_schema()
            input_budget(model.connection.model, repair_turns, schema=repair_schema)
            repair_root = out / f"citation-repair-{item['index']:02}"
            save_json(repair_root / "input.json", repair_input)
            repaired = model.complete_result(repair_turns, temperature=0, reasoning=ReasoningOptions(enabled=False), cancel=cancel,
                response_format={"type": "json_schema", "json_schema": {"name": "a18b_citations", "strict": True,
                    "schema": repair_schema}})
            save_json(repair_root / "raw.json", {"text": repaired.text, "usage": repaired.usage})
            validate_provider_budget(repaired.usage)
            spans = CitationRepair.model_validate(json.loads(repaired.text)).spans
            if len(spans) != 1 or spans[0].index != 0:
                raise ValueError("The proofreader must repair only its supplied quotation.")
            # The private global index stays with the host. The single-source
            # proofreader always works on local index zero.
            repaired_spans.append(RepairedSpan(index=item["index"], quote=spans[0].quote))
        correction = CitationRepair(spans=tuple(repaired_spans))
        if len(correction.spans) != len(bad) or {item.index for item in correction.spans} != set(range(len(bad))):
            raise ValueError("The citation repair did not cover each rejected quotation exactly once.")
        data = review.model_dump(mode="json")
        for span in correction.spans:
            judgment_index, citation_index, citation = bad[span.index]
            if not span.quote or span.quote not in texts[citation.source_id]:
                raise ValueError("The citation repair still does not match its original source.")
            data["judgments"][judgment_index]["citations"][citation_index]["quote"] = span.quote
        review = Review.model_validate(data)
        validate_review(review, packet)
    save_json(out / "review.json", review.model_dump(mode="json"))
    return review


def compose(model, packet: JessePacket, *, cancel=None) -> str:
    """Generate one human message from public speech and one current beat."""
    turns = [Turn(Role.System, prompts()["jesse"]), Turn(Role.User, packet.model_dump_json())]
    input_budget(model.connection.model, turns)
    completion = model.complete_result(turns, temperature=0.85, reasoning=ReasoningOptions(enabled=False), cancel=cancel)
    validate_provider_budget(completion.usage)
    text = completion.text.strip()
    if not text:
        raise ValueError("Jesse returned no public message.")
    return text
