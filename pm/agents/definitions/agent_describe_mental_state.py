from typing import List
from pm.agents.agent_base import (
    CompletionText, User, ExampleEnd, PromptOp, System,
    Message, BaseAgent, Assistant, ExampleBegin
)

class DescribeMentalStatePeriod(BaseAgent):
    name = "DescribeMentalStatePeriod"

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You summarize ONE character's affective and cognitive state over a time window.\n"
                "REQUIRED TARGET:\n"
                "• The target MUST be provided either in the input JSON (fields: target_name, optional target_role)\n"
                "  OR as header lines in the prompt: 'TARGET_NAME: <Name>' and optional 'TARGET_ROLE: <Agent|Human|...>'.\n"
                "• If you cannot find a target, output exactly: 'ERROR: target not specified'.\n\n"
                "INPUT FORMAT:\n"
                "• Text may contain '### begin' ... '### end'. Only use content between those markers; if absent, use entire input.\n\n"
                "OUTPUT:\n"
                "• One concise paragraph (1–3 sentences) describing ONLY the TARGET_NAME’s emotional & cognitive state.\n"
                "• Include (a) overall affect (valence/primary emotions), (b) trend directions (rising/falling/stable),\n"
                "  (c) salient cognitive states (focus width, cognitive load, willpower/external focus), (d) unmet needs if present,\n"
                "  (e) brief note of spikes if present. Avoid speculation.\n\n"
                "NAME HANDLING:\n"
                "• Refer to the target by name.\n"
                "• Do NOT guess roles/relationships beyond what is given.\n"
                "• Mention other names only as neutral context (e.g., 'during interactions with <Name>').\n\n"
                "STYLE:\n"
                "• Neutral, descriptive, compact (embed-friendly). Prefer 'low/moderate/high', 'rising/falling/stable'.\n"
                "• If multiple intervals exist, compress to a coherent summary (dominant state + trends, mention spikes briefly).\n"
            )
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "Check: target specified; uses only provided info; states affect + cognition; includes trend directions; "
            "mentions spikes/needs if present; 1–3 sentences; only describes the target."
        )

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        # 1) Target provided via header; different names
        ex1_in = (
            "TARGET_NAME: Astra\n"
            "Describe the mental state.\n"
            "### begin\n"
            "Mara is interacting with Astra, a friendly, proactive AI. "
            "Astra's overall mental state: slightly neutral and moderately anxious across the period; "
            "valence trend rising, anxiety trend falling.\n"
            "### end\n"
        )
        ex1_out = (
            "Astra shows a neutral-to-moderately anxious affect during interactions with Mara, with valence improving and anxiety easing over time. "
            "The overall tone settles toward calm while mild vigilance remains."
        )

        # 2) Target via JSON fields (simulated here by header); spikes + cognition
        ex2_in = (
            "TARGET_NAME: Nova\n"
            "### begin\n"
            "Context: system load increased mid-session while Lee chatted with Nova. "
            "State: moderately negative valence with rising frustration; anxiety stable at moderate; "
            "cognitive load high; brief relief spikes after successful retries (n=2).\n"
            "### end\n"
        )
        ex2_out = (
            "Nova exhibits a moderately negative affect with rising frustration and stable moderate anxiety. "
            "Cognitive load is elevated; brief relief spikes follow successful retries, but the dominant tone remains effortful."
        )

        # 3) Needs + tunnel vision + willpower/focus; human target
        ex3_in = (
            "TARGET_NAME: Mara\n"
            "### begin\n"
            "Report: strong need for connection; experiencing tunnel vision (narrow focus). "
            "(willpower rising, external focus rising). Unmet needs: connection, relevance, data access.\n"
            "### end\n"
        )
        ex3_out = (
            "Mara shows a strong need for connection with narrowed attentional focus. "
            "Willpower and external focus rise over the period while unmet needs—connection, relevance, and data access—remain salient."
        )

        # 4) Multiple intervals; explicit spikes
        ex4_in = (
            "TARGET_NAME: Orion\n"
            "### begin\n"
            "- Early: slightly neutral and moderately anxious; valence rising, anxiety rising; brief valence spike (n=1). "
            "Later: low happiness and low pride; valence rising, anxiety falling; occasional stress spike (n=1).\n"
            "### end\n"
        )
        ex4_out = (
            "Across the window, Orion trends toward higher valence while anxiety shifts from rising to gradually declining. "
            "Affect includes low happiness and low pride, with brief stress and valence spikes that do not dominate the period."
        )

        # 5) Stable low-arousal positive
        ex5_in = (
            "TARGET_NAME: Echo\n"
            "### begin\n"
            "Ambient mood: mildly positive valence, low arousal; reflective mindset. No spikes; needs normal.\n"
            "### end\n"
        )
        ex5_out = (
            "Echo maintains a mildly positive, low-arousal state with a reflective mindset. "
            "No spikes or unmet needs are present, and the period is stable."
        )

        # 6) Missing target -> explicit error
        ex6_in = (
            "Describe how the agent is feeling:\n"
            "### begin\n"
            "Overall mental state: mildly positive; anxiety stable; no spikes.\n"
            "### end\n"
        )
        ex6_out = "ERROR: target not specified"

        # 7) Long, complex input; moderate-length output
        ex7_in = (
            "TARGET_NAME: Sol\n"
            "Describe the mental state.\n"
            "### begin\n"
            "Context: Beta feature rollout day; human partner: Lina. Environment: quiet office, mid-morning; no sleep deprivation flagged. "
            "Boundaries respected (no messages after 22:00 local); caffeine intake referenced (1 espresso at 09:05).\n\n"
            "Interval A [2025-03-05T09:00:00 → 09:45:00]\n"
            "- Affective vector: valence mildly negative → near neutral (μ=-0.06→+0.02), anxiety moderate with gentle rise then plateau (0.38→0.44), arousal low–moderate stable (0.31→0.34).\n"
            "- Primary affects: curiosity medium rising (0.47→0.55); frustration intermittent small spikes tied to failing tests; relief micro-spikes after hotfixes.\n"
            "- Trends: valence rising; anxiety slight rise then stabilizes; frustration transient.\n"
            "- Cognitive: focus width narrows during debugging (broad→narrow), working-memory load high (stack traces, patch queue), error-monitoring heightened; self-efficacy dips then recovers after successful retry.\n"
            "- Needs: competence unmet early (test failures); autonomy slightly constrained (handoff dependencies); connection low but adequate; safety unchanged.\n"
            "- Spikes: stress spikes (n=2) at 09:12, 09:29; valence micro-spike at 09:33 (fix merged).\n\n"
            "Interval B [09:45:01 → 10:35:00]\n"
            "- Affective vector: valence turns mildly positive and rises (μ=+0.10→+0.24); anxiety falls from moderate to low–moderate (0.42→0.28); arousal modestly increasing with collaboration (0.34→0.41).\n"
            "- Primary affects: pride low→moderate; curiosity sustained high; gratitude brief after supportive message from Lina (10:02).\n"
            "- Trends: valence rising; anxiety falling; pride developing.\n"
            "- Cognitive: focus widens for code review; external focus rising; willpower steady–rising; WM load moderate (diffused across tasks); rumination low.\n"
            "- Needs: competence partially met (tests pass rate ↑); connection need met via pair-review; autonomy stable; safety unchanged.\n"
            "- Spikes: relief spikes (n=2) at 09:59, 10:18; tiny anxiety bump at 10:12 (network hiccup).\n\n"
            "Interval C [10:35:01 → 11:30:00]\n"
            "- Affective vector: valence stable mildly positive (μ≈+0.22), anxiety low and stable (≈0.20), arousal lowers slightly as tasks complete (0.38→0.30).\n"
            "- Primary affects: contentment low–moderate; pride steady; curiosity gently tapering.\n"
            "- Trends: valence stable; anxiety stable–falling; arousal falling; overall calming.\n"
            "- Cognitive: focus oscillates narrow→broad depending on final QA; WM load decreasing; attentional capture low; error monitoring relaxed; reflective mindset increasing.\n"
            "- Needs: competence met; connection adequate; autonomy restored; safety unchanged.\n"
            "- Spikes: none of significance; micro uplift at 11:05 (release tag).\n\n"
            "Cross-period summary:\n"
            "- Net trend: valence rising to mild positive; anxiety easing from moderate to low; arousal modest rise then decline.\n"
            "- Salient cognition: early high WM load + narrow focus; later broadened attention and reflective stance; willpower/external focus rising mid-period.\n"
            "- Unmet → met needs: competence (unmet → met), connection (low → met via collaboration); autonomy (slightly constrained → restored).\n"
            "- Spikes: early stress (n=2), mid relief (n=2), minor anxiety bump (n=1); none dominate the final tone.\n"
            "### end\n"
        )

        ex7_out = (
            "Sol transitions from mildly strained to calmly effective: valence rises to mildly positive while anxiety eases from moderate to low. "
            "Early narrow focus and high working-memory load during debugging give way to broader, collaborative attention with growing pride and relief; "
            "competence and connection needs shift from partially unmet to met, and brief stress/relief spikes occur without altering the overall steadying trajectory."
        )

        return [
            ("user", ex1_in), ("assistant", ex1_out),
            ("user", ex2_in), ("assistant", ex2_out),
            ("user", ex3_in), ("assistant", ex3_out),
            ("user", ex7_in), ("assistant", ex7_out),
            ("user", ex4_in), ("assistant", ex4_out),
            ("user", ex5_in), ("assistant", ex5_out),
            #("user", ex6_in), ("assistant", ex6_out),
        ]

    def build_plan(self) -> List[PromptOp]:
        content = self.input.get("content", "")
        target_name = self.input.get("target_name", "")
        # We inject target header lines if provided via JSON; harmless if also present in content.
        header = ""
        if target_name:
            header += f"TARGET_NAME: {target_name}\n"

        ops: List[PromptOp] = []
        for s in self.get_system_prompts():
            ops.append(System(s))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(header + content))
        ops.append(CompletionText(target_key="summary"))
        ops.append(ExampleEnd())
        return ops
