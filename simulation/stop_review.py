"""Correlate the stop question with its worker, dated observation, public answer and later commands."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

from pydantic import BaseModel, Field

from pm.memory.store import MemoryStore
from workflows.minecraft_body.gateway import READ_TOOLS
from workflows.minecraft_body.journal import read_actor_journal
from workflows.minecraft_body.world_text import StopObservation
from workflows.minecraft_body.spatial import WorldPosition, coordinates_in_answer
from workflows.minecraft_body.world_text import MinecraftToolResult


class StageFactSources(BaseModel):
    """Delivery citations from both current and earlier simulation checkpoints."""

    delivery_id: str | None = None
    delivery_ids: tuple[str, ...] = ()


class StopReview(BaseModel):
    stage_id: str
    delivery_id: str | None = None
    correlation_id: str | None = None
    run_id: str | None = None
    worker_ack_seconds: float | None = None
    public_ack_seconds: float | None = None
    answer_seconds: float | None = None
    observation_age_seconds: float | None = None
    observed_position: WorldPosition | None = None
    public_answer: str | None = None
    coordinates_match: bool | None = None
    later_world_change: bool | None = None
    later_commands: list[str] = Field(default_factory=list)
    within_fifteen_minutes: bool | None = None
    acknowledged_within_two_minutes: bool | None = None


class ReviewPayload(BaseModel):
    knoxel_id: int | None = None
    status: str = ""
    operation: str = ""
    action_id: str = ""
    receipt: ReviewReceipt | None = None


class ReviewReceipt(BaseModel):
    run_id: str = ""
    stop_observation: StopObservation | None = None


class ReviewEnvelope(BaseModel):
    id: str
    topic: str
    correlation_id: str
    payload: ReviewPayload


class ReviewDelivery(BaseModel):
    real_time: datetime
    envelope: ReviewEnvelope


class CommandEntry(BaseModel):
    event: str
    time: float | None = None
    seq: int | None = None
    tool: str = ""
    ok: bool | None = None
    result: MinecraftToolResult | None = None


def inspect_later_commands(review: StopReview, root: Path, start: float, end: float) -> None:
    journals = list(root.rglob("actor-journal.jsonl"))
    if not journals:
        return
    review.later_world_change = False
    for journal in journals:
        for wire in read_actor_journal(journal):
            if wire["event"] not in {"dispatch", "result", "uncertain"} or not start < wire["time"] < end:
                continue
            if wire.get("tool") in READ_TOOLS or wire.get("tool") == "minecraft_stop":
                continue
            entry = CommandEntry.model_validate(wire)
            review.later_commands.append(f"{journal.relative_to(root)} #{entry.seq} {entry.event} {entry.tool}")
            result = entry.result
            if result and (result.placed_count or result.dropped or entry.ok and result.block or result.inventory_delta and result.inventory_delta.changes
                           or result.final_position and result.final_position != review.observed_position):
                review.later_world_change = True
            elif review.later_world_change is False:
                review.later_world_change = None


def review_stops(root: Path, store: MemoryStore, stage_id: str, delivery_ids: set[str]) -> list[StopReview]:
    events: dict[str, list[ReviewDelivery]] = {}
    delivery_by_correlation: dict[str, str] = {}
    for feature in store.all_features:
        delivery = feature.metadata.get("delivery_id")
        if delivery and (delivery in delivery_ids or f":stage:{stage_id}:turn:" in str(delivery)):
            correlation = str(feature.metadata["exchange_correlation"])
            delivery_by_correlation[correlation] = str(delivery)
            events[correlation] = []
    telemetry = root / "telemetry.jsonl"
    seen: set[str] = set()
    resumes: dict[str, list[datetime]] = {}
    if telemetry.is_file():
        with telemetry.open(encoding="utf-8") as stream:
            for line in stream:
                if not any(topic in line for topic in ('"user.input"', '"action.receipt"', '"dialogue.reply"')):
                    continue
                row = ReviewDelivery.model_validate_json(line)
                envelope = row.envelope
                if envelope.id in seen:
                    continue
                seen.add(envelope.id)
                payload = envelope.payload
                if payload.operation == "resume" and payload.status == "accepted" and payload.receipt:
                    resumes.setdefault(payload.receipt.run_id, []).append(row.real_time)
                if envelope.correlation_id in events:
                    events[envelope.correlation_id].append(row)
    reviews: list[StopReview] = []
    for correlation, rows in events.items():
        delivery = delivery_by_correlation[correlation]
        if delivery not in delivery_ids and not any(row.envelope.payload.operation in {"terminate", "stop_observed"} for row in rows):
            continue
        review = StopReview(stage_id=stage_id, delivery_id=delivery, correlation_id=correlation)
        reviews.append(review)
        asked = next((row.real_time for row in rows if row.envelope.topic == "user.input"), None)
        stop: StopObservation | None = None
        for row in rows:
            payload = row.envelope.payload
            if payload.receipt and payload.receipt.stop_observation:
                stop = payload.receipt.stop_observation
                review.run_id = stop.run_id
                review.observed_position = stop.reading.observation.camera.feet_position
                if asked:
                    review.worker_ack_seconds = (stop.acknowledged_at - asked).total_seconds()
        for row in rows:
            payload = row.envelope.payload
            if row.envelope.topic != "dialogue.reply" or asked is None:
                continue
            elapsed = (row.real_time - asked).total_seconds()
            if payload.action_id.startswith("management-reply:terminate:"):
                review.public_ack_seconds = elapsed
                review.acknowledged_within_two_minutes = elapsed <= 120
            if payload.action_id.startswith("management-reply:stop_observed:") and stop and payload.knoxel_id is not None:
                reply = store.get_knoxel_by_id(payload.knoxel_id)
                if reply is None:
                    continue
                review.public_answer = reply.content
                review.coordinates_match = coordinates_in_answer(reply.content, stop.reading.observation.camera.feet_position)
                review.observation_age_seconds = (row.real_time - stop.reading.observed_at).total_seconds()
                if review.coordinates_match:
                    review.answer_seconds = elapsed
                    review.within_fifteen_minutes = elapsed <= 900
        if stop:
            end = min((at.timestamp() for at in resumes.get(stop.run_id, []) if at > stop.acknowledged_at), default=math.inf)
            inspect_later_commands(review, root / "managed_tasks", stop.acknowledged_at.timestamp(), end)
    return reviews or [StopReview(stage_id=stage_id)]
