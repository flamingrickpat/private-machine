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
from urllib.parse import urlsplit, urlunsplit

import httpx

from pm.llm.http_client import JsonClient, ModelError, cancellable
from pm.llm.cache import model_directory
from pm.llm.config import Connection, EmbeddingConfig
from pm.llm.workflow_sessions import serving_context
from pm.records.media import alt_text_view, image_ids, media_lookup


# A token cannot represent less than one UTF-8 byte. Inputs below this byte
# length cannot fill a practical serving context, so they skip the tokenizer
# probe. Servers with a context below this size are not supported targets.
_SHORT_TEXT_BYTES = 128


def embedding_text(text: str) -> str:
    """Return the text that is embedded for ``text``: each image tag becomes its alt text.

    The base64 data of an image is never embedded. A tag whose image has no
    alt text yet reads ``[image: no description yet]``; ``embed`` completes
    the alt text first, so a stored vector always uses the description.
    """
    return alt_text_view(text) if image_ids(text) else text


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


async def fit_embedding_text(connection: Connection, text: str, cancel: Event | None = None) -> tuple[str, dict]:
    """Cut text to the serving context before an embedding request.

    llama.cpp rejects an embedding when the tokenized input is larger than
    ``n_ctx``. The server's own tokenizer decides the length, so the count
    matches inference. A server can add special tokens around the input; the
    raw prefix keeps only the tokens that belong to the caller's text.

    The return value is the text to send and a receipt with the discovered
    context, original count, sent count, and truncation flag. If the endpoint
    has no serving context or tokenizer route, the text stays unchanged. A
    transport or protocol failure raises ModelError without prompt text.
    """
    try:
        context, source = await serving_context(connection, cancel)
    except ModelError:
        return text, {"context_window": None, "context_source": None,
                      "original_tokens": None, "input_tokens": None, "truncated": False}
    parts = urlsplit(connection.base_url)
    root = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1"), "", ""))
    key = connection.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            special = await cancellable(client.post(root + "/tokenize", headers=headers, json={
                "content": text, "add_special": True, "parse_special": True}), cancel)
            if special.status_code in {404, 405, 501}:
                return text, {"context_window": context, "context_source": source,
                              "original_tokens": None, "input_tokens": None, "truncated": False}
            special.raise_for_status()
            special_tokens = special.json()["tokens"]
            if not isinstance(special_tokens, list):
                raise ValueError("Invalid token sequence.")
            if len(special_tokens) <= context:
                return text, {"context_window": context, "context_source": source,
                              "original_tokens": len(special_tokens),
                              "input_tokens": len(special_tokens), "truncated": False}
            raw = await cancellable(client.post(root + "/tokenize", headers=headers, json={
                "content": text, "add_special": False, "parse_special": True}), cancel)
            raw.raise_for_status()
            raw_tokens = raw.json()["tokens"]
            if not isinstance(raw_tokens, list) or any(type(token) is not int for token in raw_tokens):
                raise ValueError("Invalid token sequence.")
            keep = context - (len(special_tokens) - len(raw_tokens))
            if keep <= 0:
                raise ValueError("The serving context cannot hold embedding input.")
            response = await cancellable(client.post(root + "/detokenize", headers=headers,
                json={"tokens": raw_tokens[:keep]}), cancel)
            response.raise_for_status()
            clipped = response.json()["content"]
            if not isinstance(clipped, str):
                raise ValueError("Invalid detokenized text.")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError("Embedding context fit failed. Check the selected embedding endpoint and tokenizer.") from None
    return clipped, {"context_window": context, "context_source": source,
                     "original_tokens": len(special_tokens), "input_tokens": keep, "truncated": True}


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
        self.endpoint = config.base_url
        self.dimensions = config.dimensions
        self.legacy_dir = config.legacy_cache_dir.expanduser().resolve()
        cache_dir = config.cache_dir.expanduser().resolve()
        if cache_dir == self.legacy_dir or self.legacy_dir in cache_dir.parents:
            raise ValueError("Keep the writable application cache outside the read-only legacy cache.")
        self.directory = model_directory(self.model, cache_dir)
        self.hits = 0
        self.legacy_hits = 0
        self.misses = 0

    def content_key(self, text: str) -> str:
        """Identify the source record text, endpoint, model, and dimensions.

        A long record keeps its cache entry under the full text. The stored
        vector comes from the prefix that fits the serving context.
        """
        return hashlib.sha256(json.dumps([self.endpoint, self.model, self.dimensions, text],
                                        ensure_ascii=False).encode("utf-8")).hexdigest()

    def _read(self, text: str, *, legacy: bool) -> list[float] | None:
        key = hashlib.md5(text.encode("utf-8")).hexdigest() if legacy else self.content_key(text)
        directory = self.legacy_dir if legacy else self.directory
        try:
            filename = f"emb_hash_{key}.json" if legacy else f"cache_emb_{key}.json"
            meta = json.loads((directory / filename).read_text(encoding="utf-8"))
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
                if meta["endpoint"] != self.endpoint:
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
        """Publish a validated cache entry, or keep another valid writer's entry.

        A model and input identify immutable cached content. An atomic hard link
        publishes new content once without replacing an open Windows file.
        A concurrent valid publication wins. Invalid old entries can be replaced.
        The cache filesystem must support hard links within one directory.
        Other filesystem errors propagate. This method sends no model request.
        """
        valid_vector(vector, self.dimensions)
        if self._read(text, legacy=False) is not None:
            return
        payload = {
            "schema_version": 1, "model": self.model, "input": text,
            "dimensions": self.dimensions, "vector": vector, "endpoint": self.endpoint,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                target = self.directory / f"cache_emb_{self.content_key(text)}.json"
                os.link(temporary, target)
            except FileExistsError:
                if self._read(text, legacy=False) is None:
                    os.replace(temporary, target)
            except PermissionError:
                # Another writer can publish after the initial cache miss.
                # Keep its complete entry instead of replacing an open file.
                if self._read(text, legacy=False) is None:
                    raise
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
        """Return the vector of ``text``, from the cache or from the endpoint.

        Every embedding of record text passes here. If the text holds image
        tags, the installed media lookup first writes the missing alt texts
        (one multimodal model call per new image). The vector then belongs to
        the text with the alt texts in place of the tags. Text that exceeds
        the serving context is cut to the longest prefix that fits. The cache
        entry keeps the full source text as its key.
        """
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        ids = image_ids(text)
        if ids:
            lookup = media_lookup()
            if lookup is not None:
                await lookup.complete_alt_text(ids, cancel=cancel)
            text = embedding_text(text)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        # Keep the cache key on the full record text. The vector then belongs
        # to that record even when the endpoint sees only its prefix.
        input_text = text
        if len(text.encode("utf-8")) > _SHORT_TEXT_BYTES:
            input_text, _ = await fit_embedding_text(self.client.connection, text, cancel)
        response = await self.client.request("POST", "embeddings", payload={
            "model": self.client.connection.model, "input": input_text, "encoding_format": "float",
        }, cancel=cancel)
        vector = parse_embedding_response(response, self.config)
        self.last_response_model = response["model"]
        self.cache.put(text, vector)
        return vector

    def inspect_index(self, visible_text: dict[int, str]) -> IndexPlan:
        plan = IndexPlan(hits=[], missing=[])
        for record_id, text in visible_text.items():
            (plan.hits if self.cache.get(embedding_text(text)) is not None else plan.missing).append(record_id)
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
        vectors = {record_id: self.cache.get(embedding_text(text)) for record_id, text in visible_text.items()}
        if any(vector is None for vector in vectors.values()):
            raise ValueError("The scoped index is incomplete. Inspect and prepare it before search.")
        query_vector = await self.embed(query, cancel=cancel)
        return sorted(
            ((record_id, cosine_similarity(query_vector, vector)) for record_id, vector in vectors.items()),
            key=lambda item: (-item[1], item[0]),
        )
