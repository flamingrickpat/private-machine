"""One embedding service with a read-only legacy cache.

Missing or incompatible external cache entries count as misses. Invalid API
vectors fail explicitly. Local write errors keep their tracebacks. Transport
retry and cancellation use the shared model client. Constructors and index
inspection never send requests. Only explicit embed/prepare/search calls do.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from runtime.model_io import JsonClient, ModelError
from runtime.models import EmbeddingConfig


def valid_vector(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError("The embedding dimension does not match the configured model.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
        raise ValueError("The embedding contains invalid vector values.")
    if math.hypot(*vector) == 0:
        raise ValueError("A zero vector cannot be used for cosine similarity.")
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    valid_vector(left, len(right))
    valid_vector(right, len(left))
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    return math.fsum((a / left_norm) * (b / right_norm) for a, b in zip(left, right))


def parse_embedding_response(response: dict, config: EmbeddingConfig) -> list[float]:
    """Validate provider output once, including its explicit response alias."""
    try:
        if len(response["data"]) != 1 or response["data"][0]["index"] != 0:
            raise ValueError("Expected exactly one embedding.")
        vector = valid_vector(response["data"][0]["embedding"], config.dimensions)
        allowed_models = {config.model}
        if config.response_model is not None:
            allowed_models.add(config.response_model)
        if response["model"] not in allowed_models:
            raise ValueError("Unexpected embedding model.")
    except (KeyError, TypeError, ValueError):
        raise ModelError("The embedding service returned incompatible vector data.") from None
    return vector


class EmbeddingCache:
    def __init__(self, config: EmbeddingConfig):
        self.model = config.model
        self.dimensions = config.dimensions
        self.legacy_dir = config.legacy_cache_dir.expanduser().resolve()
        cache_dir = config.cache_dir.expanduser().resolve()
        if cache_dir == self.legacy_dir or self.legacy_dir in cache_dir.parents:
            raise ValueError("Keep the writable application cache outside the read-only legacy cache.")
        model_key = hashlib.sha256(self.model.encode("utf-8")).hexdigest()
        self.directory = cache_dir / "v1" / model_key
        self.hits = 0
        self.legacy_hits = 0
        self.misses = 0

    @staticmethod
    def content_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _read(self, text: str, *, legacy: bool) -> list[float] | None:
        key = self.content_key(text)
        directory = self.legacy_dir if legacy else self.directory
        try:
            meta = json.loads((directory / f"emb_hash_{key}.json").read_text(encoding="utf-8"))
            if meta["input"] != text or meta["model"] != self.model or meta["dimensions"] != self.dimensions:
                return None
            if legacy:
                data = (directory / f"emb_hash_{key}.bin").read_bytes()
                count = struct.unpack_from("<Q", data)[0]
                if count != self.dimensions or len(data) != 8 + count * 8:
                    return None
                vector = list(struct.unpack_from(f"<{count}d", data, 8))
            else:
                if meta["schema_version"] != 1:
                    return None
                vector = meta["vector"]
            return valid_vector(vector, self.dimensions)
        except (FileNotFoundError, UnicodeError, ValueError, KeyError, TypeError, struct.error):
            return None

    def get(self, text: str) -> list[float] | None:
        for legacy in (False, True):
            vector = self._read(text, legacy=legacy)
            if vector is not None:
                self.hits += 1
                self.legacy_hits += int(legacy)
                return vector
        self.misses += 1
        return None

    def put(self, text: str, vector: list[float]) -> None:
        valid_vector(vector, self.dimensions)
        payload = {
            "schema_version": 1, "model": self.model, "input": text,
            "dimensions": self.dimensions, "vector": vector,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"emb_hash_{self.content_key(text)}.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass
class IndexPlan:
    hits: list[int]
    missing: list[int]


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig, *, live_test: bool = False):
        self.config = config
        self.cache = EmbeddingCache(config)
        self.client = JsonClient(config.resolve(), live_test=live_test)
        self.dimensions = config.dimensions
        self.last_response_model: str | None = None

    async def embed(self, text: str, *, cancel: Event | None = None) -> list[float]:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        response = await self.client.request("POST", "embeddings", payload={
            "model": self.client.connection.model, "input": text, "encoding_format": "float",
        }, cancel=cancel)
        vector = parse_embedding_response(response, self.config)
        self.last_response_model = response["model"]
        self.cache.put(text, vector)
        return vector

    def inspect_index(self, visible_text: dict[int, str]) -> IndexPlan:
        plan = IndexPlan(hits=[], missing=[])
        for record_id, text in visible_text.items():
            (plan.hits if self.cache.get(text) is not None else plan.missing).append(record_id)
        return plan

    async def prepare_index(self, visible_text: dict[int, str], *, cancel: Event | None = None, progress=None) -> IndexPlan:
        """Explicitly fetch only missing text from the caller's scoped input."""
        plan = self.inspect_index(visible_text)
        if progress is not None:
            progress(len(plan.hits), len(visible_text))
        for completed, record_id in enumerate(plan.missing, len(plan.hits) + 1):
            await self.embed(visible_text[record_id], cancel=cancel)
            if progress is not None:
                progress(completed, len(visible_text))
        return plan

    async def search(self, query: str, visible_text: dict[int, str], *, cancel: Event | None = None) -> list[tuple[int, float]]:
        vectors = {record_id: self.cache.get(text) for record_id, text in visible_text.items()}
        if any(vector is None for vector in vectors.values()):
            raise ValueError("The scoped index is incomplete. Inspect and prepare it before search.")
        query_vector = await self.embed(query, cancel=cancel)
        return sorted(
            ((record_id, cosine_similarity(query_vector, vector)) for record_id, vector in vectors.items()),
            key=lambda item: (-item[1], item[0]),
        )
