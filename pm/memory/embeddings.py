"""Embed text directly. Records own durable vectors and FAISS owns search."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pm.records.base import RecordBase
from pm.memory.vector_index import VectorIndex
from threading import Event
from urllib.parse import urlsplit, urlunsplit

import httpx

from pm.llm.http_client import JsonClient, ModelError, cancellable, wait_until_ready
from pm.llm.config import Connection, EmbeddingConfig
from pm.llm.workflow_sessions import serving_context
from pm.records.media import alt_text_view, image_ids, media_lookup


# A token cannot represent less than one UTF-8 byte. Inputs below this byte
# length cannot fill a practical serving context, so they skip the tokenizer
# probe. Servers with a context below this size are not supported targets.
_SHORT_TEXT_BYTES = 128

# The startup probe. Five plain sentences separate a working encoder from a
# broken one. The first two say the same thing, so a real encoder must score
# that pair far above the unrelated pairs.
SMOKE_TEST_TEXTS: tuple[str, ...] = (
    "The cat sat on the mat near the window.",
    "A small feline rested beside the open window.",
    "Quantum chromodynamics binds quarks inside a proton.",
    "She folded the laundry and put it away.",
    "The referee sent off the striker in the sixty-third minute.",
)
SMOKE_PARAPHRASE_PAIR = (0, 1)
# Unrelated sentences must not look alike. A mean above this bound means the
# endpoint returns one shared direction for every text. The failed run of
# 2026-10-24 measured 0.93 here.
SMOKE_MAX_UNRELATED_MEAN = 0.60
SMOKE_MAX_UNRELATED_SINGLE = 0.85
# A real encoder must also separate the paraphrase pair. A small margin is the
# random-vector case: the vectors carry no meaning.
SMOKE_MIN_PARAPHRASE_MARGIN = 0.05
# The length of the mean of the unit probe vectors. One shared component keeps
# this near 1.0, because all the vectors point the same way. The failed run of
# 2026-10-24 measured 0.97 here. Qwen3-Embedding-0.6B measures 0.68.
SMOKE_MAX_MEAN_VECTOR_NORM = 0.90


class EmbeddingQualityError(RuntimeError):
    """The endpoint returns vectors that cannot support similarity work.

    Store vectors are durable and every similarity rule reads them, so the
    runtime stops before it writes one.
    """


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


def embedding_quality_report(vectors: list[list[float]]) -> dict[str, float]:
    """Measure how far apart the vectors of one :data:`SMOKE_TEST_TEXTS` batch are.

    ``unrelated_mean``, ``unrelated_min``, and ``unrelated_max`` cover every
    pair except the paraphrase pair. ``margin`` is the paraphrase cosine minus
    the unrelated mean. ``mean_vector_norm`` is the length of the mean of the
    unit vectors, so it measures one shared direction rather than length.
    """
    if len(vectors) != len(SMOKE_TEST_TEXTS):
        raise ValueError(f"the quality probe needs {len(SMOKE_TEST_TEXTS)} vectors, got {len(vectors)}")
    unrelated: list[float] = []
    paraphrase = 0.0
    for left, right in combinations(range(len(vectors)), 2):
        score = cosine_similarity(vectors[left], vectors[right])
        if (left, right) == SMOKE_PARAPHRASE_PAIR:
            paraphrase = score
        else:
            unrelated.append(score)
    mean = math.fsum(unrelated) / len(unrelated)
    # Cosine ignores length, so the shared-direction measure must too. It runs
    # on unit vectors, and stays comparable across servers that do not return
    # normalized vectors.
    units = [[value / math.hypot(*vector) for value in vector] for vector in vectors]
    average = [math.fsum(column) / len(units) for column in zip(*units, strict=True)]
    return {
        "unrelated_mean": mean,
        "unrelated_min": min(unrelated),
        "unrelated_max": max(unrelated),
        "paraphrase": paraphrase,
        "margin": paraphrase - mean,
        "mean_vector_norm": math.hypot(*average),
    }


def check_embedding_quality(vectors: list[list[float]], *, model: str = "", dimensions: int = 0) -> dict[str, float]:
    """Return :func:`embedding_quality_report`, or raise :class:`EmbeddingQualityError`.

    The failure this catches is a degenerate endpoint: every text scores about
    0.9 against every other text, so all memories look alike and every merge,
    retrieval, and clustering rule quietly misfires. The error text carries
    the numbers and the settings, because the operator has to fix the server.
    """
    report = embedding_quality_report(vectors)
    problems = []
    if report["unrelated_mean"] > SMOKE_MAX_UNRELATED_MEAN:
        problems.append(f"unrelated sentences average {report['unrelated_mean']:.3f} cosine, over the "
                        f"{SMOKE_MAX_UNRELATED_MEAN:.2f} limit: every text looks like every other text")
    if report["unrelated_max"] > SMOKE_MAX_UNRELATED_SINGLE:
        problems.append(f"one unrelated pair reaches {report['unrelated_max']:.3f} cosine, over the "
                        f"{SMOKE_MAX_UNRELATED_SINGLE:.2f} limit")
    if report["margin"] < SMOKE_MIN_PARAPHRASE_MARGIN:
        problems.append(f"the paraphrase pair scores {report['paraphrase']:.3f}, only "
                        f"{report['margin']:.3f} above the unrelated mean, under the "
                        f"{SMOKE_MIN_PARAPHRASE_MARGIN:.2f} margin: the vectors carry no meaning")
    if report["mean_vector_norm"] > SMOKE_MAX_MEAN_VECTOR_NORM:
        problems.append(f"the batch mean vector keeps {report['mean_vector_norm']:.3f} of unit length, over "
                        f"the {SMOKE_MAX_MEAN_VECTOR_NORM:.2f} limit: one shared direction dominates")
    if not problems:
        return report
    raise EmbeddingQualityError(
        f"Embedding quality check failed for model '{model}' at {dimensions} dimensions with "
        f"{len(SMOKE_TEST_TEXTS)} probe sentences:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
        + "\n  report: " + ", ".join(f"{key}={value:.4f}" for key, value in report.items())
        + "\nThe embedding endpoint must be fixed before this runtime starts. Store vectors are"
        " durable, so it refuses to write them. Usual causes: the served checkpoint lost its"
        " sentence-embedding head (a GGUF converted without --sentence-transformers-dense-modules),"
        " or a --pooling setting that does not match the model."
    )


async def fit_embedding_text(connection: Connection, text: str, cancel: Event | None = None, *,
                             live_test: bool = False) -> tuple[str, dict]:
    """Cut text to the serving context before an embedding request.

    llama.cpp rejects an embedding when the tokenized input is larger than
    ``n_ctx``. The server's own tokenizer decides the length, so the count
    matches inference. A server can add special tokens around the input; the
    raw prefix keeps only the tokens that belong to the caller's text.

    The return value is the text to send and a receipt with the discovered
    context, original count, sent count, and truncation flag. If the endpoint
    has no serving context or tokenizer route, the text stays unchanged. A
    transport or protocol failure raises ModelError without prompt text.
    A silent endpoint is waited out: the tokenizer requests repeat every 30
    seconds until the endpoint answers. ``live_test=True`` keeps the fail-fast
    error for diagnostics and tests.
    """
    try:
        context, source = await serving_context(connection, cancel, live_test=live_test)
    except ModelError:
        return text, {"context_window": None, "context_source": None,
                      "original_tokens": None, "input_tokens": None, "truncated": False}
    parts = urlsplit(connection.base_url)
    root = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1"), "", ""))
    key = connection.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    async def count_with_server() -> tuple[str, dict]:
        """Ask the endpoint for its own token count and clipped text. Retried until it answers."""
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

    return await wait_until_ready(count_with_server, cancel, live_test=live_test,
                                  label=f"Embedding token count at {connection.base_url}")


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


@dataclass
class IndexPlan:
    hits: list[int]
    missing: list[int]


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig, *, live_test: bool = False):
        self.config = config
        self.client = JsonClient(config.resolve(), live_test=live_test)
        self.dimensions = config.dimensions
        self.last_response_model: str | None = None

    async def embed(self, text: str, *, cancel: Event | None = None) -> list[float]:
        """Embed text without a cache. Complete image alt text before inference.

        The caller stores record vectors on their knoxels. Query vectors are
        temporary. Provider errors and cancellation propagate.
        """
        return (await self.embed_many([text], cancel=cancel))[0]

    async def embed_many(self, texts: list[str], *, cancel: Event | None = None) -> list[list[float]]:
        """Embed a batch in input order. Complete image descriptions before inference."""
        if not texts:
            return []
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        ids = [key for text in texts for key in image_ids(text)]
        if ids:
            lookup = media_lookup()
            if lookup is not None:
                await lookup.complete_alt_text(ids, cancel=cancel)
            texts = [embedding_text(text) for text in texts]
        inputs = []
        for text in texts:
            if len(text.encode("utf-8")) > _SHORT_TEXT_BYTES:
                text, _ = await fit_embedding_text(self.client.connection, text, cancel,
                                                   live_test=self.client.live_test)
            inputs.append(text)
        response = await self.client.request("POST", "embeddings", payload={
            "model": self.client.connection.model, "input": inputs[0] if len(inputs) == 1 else inputs,
            "encoding_format": "float",
        }, cancel=cancel)
        rows = sorted(response["data"], key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(inputs))):
            raise ModelError("The embedding service returned an incomplete batch.")
        vectors = [parse_embedding_response({"model": response["model"], "data": [{**row, "index": 0}]}, self.config)
                   for row in rows]
        self.last_response_model = response["model"]
        return vectors

    async def smoke_test(self) -> dict[str, float]:
        """Embed the fixed probe sentences and check that they stay separable.

        Returns the quality report. Raises :class:`EmbeddingQualityError` when
        the endpoint returns vectors that cannot support retrieval, merging,
        or clustering. Each call sends real requests, so callers run it once
        per process, before any stored vector is written.
        """
        vectors = await self.embed_many(list(SMOKE_TEST_TEXTS))
        return check_embedding_quality(vectors, model=self.config.model, dimensions=self.dimensions)

    def inspect_index(self, records: dict[int, RecordBase]) -> IndexPlan:
        """Inspect vectors on records. This method performs no disk or network work."""
        return IndexPlan(
            hits=[key for key, record in records.items() if record.embedding],
            missing=[key for key, record in records.items() if not record.embedding],
        )

    async def record_embedding(self, record: RecordBase, *, cancel: Event | None = None) -> list[float]:
        """Return the record's vector. A missing vector is stored and auto-indexed."""
        if not record.embedding:
            record.embedding = await self.embed(record.content or record.get_story_element(), cancel=cancel)
        return record.embedding

    async def prepare_index(self, records: dict[int, RecordBase], *, cancel: Event | None = None,
                            progress: Callable[[int, int], None] | None = None) -> IndexPlan:
        """Embed all missing records in batches of 32, with up to four requests at once."""
        plan = self.inspect_index(records)
        completed = len(plan.hits)
        if progress is not None:
            progress(completed, len(records))
        gate = asyncio.Semaphore(4)

        async def one(keys: list[int]) -> None:
            nonlocal completed
            async with gate:
                vectors = await self.embed_many([records[key].content or records[key].get_story_element() for key in keys],
                                                cancel=cancel)
                for key, vector in zip(keys, vectors, strict=True):
                    records[key].embedding = vector
                completed += len(keys)
                if progress is not None:
                    progress(completed, len(records))

        tasks = [asyncio.create_task(one(plan.missing[start:start + 32])) for start in range(0, len(plan.missing), 32)]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Stop unfinished requests before a caller can save or discard this work.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return self.inspect_index(records)

    async def search(self, query: str | list[float], records: dict[int, RecordBase], *, cancel: Event | None = None,
                     index: VectorIndex | None = None) -> list[tuple[int, float]]:
        """Search stored record vectors with FAISS after embedding missing records."""
        if not records:
            return []
        await self.prepare_index(records, cancel=cancel)
        if index is None:
            index = VectorIndex()
            index.rebuild(records.values())
        if isinstance(query, str):
            query_vector = next((record.embedding for record in records.values() if record.content == query), None)
            if query_vector is None:
                query_vector = await self.embed(query, cancel=cancel)
        else:
            query_vector = query
        return index.search(query_vector, records)


T = TypeVar("T")


def run_embedding_work(work: Coroutine[Any, Any, T]) -> T:
    """Run embedding work from a synchronous host, including hosts with an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(work)
    with ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(asyncio.run, work).result()
