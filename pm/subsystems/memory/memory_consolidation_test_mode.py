import hashlib
import math
import random
import re
from typing import List, Sequence

from pm.agents.definitions.agent_categorize_fact import FactCategorization
from pm.subsystems.memory.cause_effect import CauseEffectItem
from pm.subsystems.memory.memory_consolidation import DynamicMemoryConsolidator, MemoryConsolidationConfig


class _SyntheticLlmProxy:
    def __init__(self, embedding_dim: int, seed: int):
        self.embedding_dim = embedding_dim
        self.seed = seed

    def get_embedding(self, text: str) -> List[float]:
        digest = hashlib.sha256(f"{self.seed}|{text}".encode("utf-8")).digest()
        values = []
        for index in range(self.embedding_dim):
            byte = digest[index % len(digest)]
            angle = ((byte / 255.0) * (2.0 * math.pi)) + index
            values.append(math.sin(angle) + (0.5 * math.cos(angle * 0.5)))
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class SyntheticMemoryConsolidator(DynamicMemoryConsolidator):
    def __init__(
        self,
        ghost,
        config: MemoryConsolidationConfig,
        *,
        seed: int = 17,
        embedding_dim: int = 32,
    ):
        self.synthetic_seed = seed
        self.synthetic_embedding_dim = embedding_dim
        self.synthetic_call_index = 0
        super().__init__(_SyntheticLlmProxy(embedding_dim=embedding_dim, seed=seed), ghost, config)

    def refine_narratives(self):
        # Narrative refinement is intentionally skipped in test mode so the
        # synthetic consolidator only validates episodic memory generation.
        return

    def _generate_embedding(self, text: str) -> List[float]:
        return self.llm.get_embedding(text)

    def _summarize_text(self, context: str, content: str) -> str:
        self.synthetic_call_index += 1
        rng = self._rng_for("summary", context, content, self.synthetic_call_index)
        begin, end = self._extract_span(context, content)
        mode = "temporal" if "between " in context.lower() else "topical"
        keywords = self._extract_keywords(content, limit=6)
        motifs = " ".join(self._make_gibberish_word(rng) for _ in range(5))
        detail = ", ".join(keywords[:4]) if keywords else self._make_gibberish_word(rng)
        return (
            f"im a {mode} summary from {begin} to {end}. "
            f"i condense {max(1, len(self._extract_story_lines(content)))} story lines around {detail}. "
            f"gibberish tail: {motifs}."
        )

    def _extract_declarative_facts_text(self, context: str, content: str) -> str:
        self.synthetic_call_index += 1
        rng = self._rng_for("facts", context, content, self.synthetic_call_index)
        begin, end = self._extract_span(context, content)
        lines = self._extract_story_lines(content)
        if not lines:
            lines = [content.strip()] if content.strip() else []
        keywords = self._extract_keywords(content, limit=8)

        facts: List[str] = []
        facts.append(f"- This synthetic memory says it spans {begin} to {end}.")
        for index, line in enumerate(lines[:2]):
            snippet = self._normalize_snippet(line, 14)
            if snippet:
                facts.append(f"- The memory records {snippet}.")
            if index == 0 and keywords:
                focus = keywords[index % len(keywords)]
                facts.append(f"- The recurring focus is {focus}.")

        while len(facts) < 3:
            facts.append(
                f"- The cluster preserves {self._make_gibberish_word(rng)} {self._make_gibberish_word(rng)} continuity."
            )
        return "\n".join(facts[:5])

    def _categorize_fact_text(self, fact: str) -> FactCategorization:
        lowered = fact.lower()
        if any(word in lowered for word in ("trust", "support", "repair", "apology", "boundary", "conflict")):
            category = ["relationships_bad", "people_interactions"]
            importance = 0.82
            time_dependent = 0.55
        elif any(word in lowered for word in ("music", "garden", "travel", "robot", "project")):
            category = ["world_events", "people_preferences"]
            importance = 0.68
            time_dependent = 0.62
        else:
            category = ["people_interactions"]
            importance = 0.58
            time_dependent = 0.48

        return FactCategorization(
            reason="Synthetic categorizer maps repeated lexical cues into a compact category set.",
            category=category,
            importance=importance,
            time_dependent=time_dependent,
        )

    def _extract_cause_effect_items(self, block: str) -> List[CauseEffectItem]:
        self.synthetic_call_index += 1
        rng = self._rng_for("cause_effect", block, self.synthetic_call_index)
        lines = self._extract_story_lines(block)
        if len(lines) < 2:
            return []

        items: List[CauseEffectItem] = []
        for index in range(0, min(len(lines) - 1, 4), 2):
            cause = f"synthetic cause {index // 2}: {self._normalize_snippet(lines[index], 10)}"
            effect = f"synthetic effect {index // 2}: {self._normalize_snippet(lines[index + 1], 10)}"
            if not cause or not effect:
                continue
            items.append(
                CauseEffectItem(
                    cause=cause,
                    effect=effect,
                    temporality=round(rng.uniform(0.2, 0.8), 2),
                    importance=rng.choice(["low", "medium", "high"]),
                    duration=round(rng.uniform(0.2, 0.9), 2),
                )
            )
        return items[:3]

    def _refine_narrative_text(self, prompt):
        text = " ".join(part for _, part in prompt)
        return self._summarize_text("narrative", text)

    def _integrate_graph_memory(self, memories):
        # Graph memory has its own LLM-heavy extraction stack. Test mode keeps the
        # consolidation focus narrow and deterministic.
        return

    def _get_optimal_clusters(self, event_knoxels, contextual_embeddings):
        # Test mode does not need expensive embedding clustering. We want stable,
        # deterministic topical groups so the planner can be tested against a
        # memory hierarchy that actually exists.
        labels = {}
        current_label = 0
        current_topic = None
        current_block_size = 0
        max_block_size = max(4, self.config.min_split_up_cluster_size)

        for knoxel in sorted(event_knoxels, key=lambda item: (item.timestamp_world_begin, item.id)):
            topic = (knoxel.metadata or {}).get("topic", "general")
            if current_topic is None:
                current_topic = topic
            if topic != current_topic or current_block_size >= max_block_size:
                current_label += 1
                current_topic = topic
                current_block_size = 0
            labels[knoxel.id] = current_label
            current_block_size += 1
        return labels

    def _rng_for(self, *parts: object) -> random.Random:
        joined = "|".join(str(part) for part in parts)
        digest = hashlib.sha256(f"{self.synthetic_seed}|{joined}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _make_gibberish_word(self, rng: random.Random) -> str:
        syllables = (
            "ka", "lo", "mi", "re", "ta", "su", "ven", "dor",
            "shi", "nal", "qua", "zin", "tor", "lek", "pha", "mur",
        )
        return "".join(rng.choice(syllables) for _ in range(rng.randint(2, 4)))

    def _extract_story_lines(self, text: str) -> List[str]:
        cleaned = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Conversation start:") or line.startswith("Conversation end:"):
                continue
            if line.startswith("Summary start:") or line.startswith("Summary end:"):
                continue
            cleaned.append(line)
        return cleaned

    def _extract_keywords(self, text: str, *, limit: int) -> List[str]:
        words = re.findall(r"[a-zA-Z_]{4,}", text.lower())
        stop_words = {
            "this", "that", "with", "from", "they", "them", "into", "were", "have",
            "would", "there", "their", "about", "conversation", "summary", "start", "end",
            "user", "companion", "memory", "records", "stable", "block",
        }
        unique = []
        for word in words:
            if word in stop_words or word in unique:
                continue
            unique.append(word)
            if len(unique) >= limit:
                break
        return unique

    def _normalize_snippet(self, text: str, limit: int) -> str:
        words = re.findall(r"[a-zA-Z0-9_']+", text.lower())
        return " ".join(words[:limit]).strip()

    def _extract_span(self, context: str, content: str) -> tuple[str, str]:
        between_match = re.search(r"between ([0-9:\- ]+) and ([0-9:\- ]+)\.", context)
        if between_match:
            return between_match.group(1).strip(), between_match.group(2).strip()

        start_match = re.search(r"Conversation start:\s*([0-9:\- ]+)", content)
        end_match = re.search(r"Conversation end:\s*([0-9:\- ]+)", content)
        if start_match and end_match:
            return start_match.group(1).strip(), end_match.group(1).strip()

        summary_start = re.search(r"Summary start:\s*([0-9:\- ]+)", content)
        summary_end = re.search(r"Summary end:\s*([0-9:\- ]+)", content)
        if summary_start and summary_end:
            return summary_start.group(1).strip(), summary_end.group(1).strip()

        return "unknown-begin", "unknown-end"
