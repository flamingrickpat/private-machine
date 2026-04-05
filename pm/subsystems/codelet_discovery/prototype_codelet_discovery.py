from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from pm.agents.agent_manager import AgentManager
from pm.agents.definitions.agent_codelet_discovery_baseline import (
    AgentCritiqueCodeletSet,
    AgentCritiqueRollout,
    AgentCritiqueSafetyDelusion,
    AgentEvaluateRickTest,
    AgentPlanCodeletGroups,
    AgentGeneratePrototypeCodelets,
    AgentGenerateSituationMutation,
    AgentSelectSurvivingCodelets,
    AgentSimulateFutureRollout,
    CodeletCandidateSet,
    CodeletGroupPlan,
    CodeletSetCritique,
    FutureRollout,
    RickTestEvaluation,
    RolloutCritique,
    SafetyDelusionCritique,
    SituationMutation,
    SurvivingCodeletSet,
)
from pm.system.load_config import load_config
from pm.system.llm.llm_proxy import start_llm_thread
from pm.utils.string_utils import pretty_print_prompt_messages


PromptMessage = Tuple[str, str]


def _progress(message: str) -> None:
    print(f"[codelet_discovery] {message}", flush=True)


CODELET_SET_SAMPLES = 6
ROLLOUT_JUDGES = 3
TOP_CODELET_SETS = 3


def _load_input_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_prompt_input(raw_text: str) -> List[PromptMessage]:
    parsed = _try_parse_structured_prompt(raw_text)
    if parsed:
        return parsed
    return _parse_role_transcript(raw_text)


def _try_parse_structured_prompt(raw_text: str) -> List[PromptMessage]:
    text = raw_text.strip()
    if not text:
        return []

    loaders = [json.loads, ast.literal_eval]
    for loader in loaders:
        try:
            data = loader(text)
        except Exception:
            continue
        normalized = _normalize_loaded_prompt(data)
        if normalized:
            return normalized
    return []


def _normalize_loaded_prompt(data) -> List[PromptMessage]:
    if not isinstance(data, list):
        return []

    result: List[PromptMessage] = []
    for item in data:
        if isinstance(item, tuple) and len(item) == 2:
            result.append((str(item[0]).strip().lower(), str(item[1])))
            continue
        if isinstance(item, list) and len(item) == 2:
            result.append((str(item[0]).strip().lower(), str(item[1])))
            continue
        if isinstance(item, dict) and "role" in item and "content" in item:
            result.append((str(item["role"]).strip().lower(), str(item["content"])))
            continue
        return []
    return [message for message in result if message[0] in {"system", "user", "assistant"}]


def _parse_role_transcript(raw_text: str) -> List[PromptMessage]:
    roles = {"system:", "user:", "assistant:"}
    messages: List[PromptMessage] = []
    current_role = None
    current_lines: List[str] = []

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if lowered in roles:
            if current_role is not None:
                messages.append((current_role, "\n".join(current_lines).strip()))
            current_role = lowered[:-1]
            current_lines = []
            continue

        for role_marker in roles:
            if lowered.startswith(role_marker) and len(stripped) > len(role_marker):
                if current_role is not None:
                    messages.append((current_role, "\n".join(current_lines).strip()))
                current_role = role_marker[:-1]
                current_lines = [stripped[len(role_marker):].lstrip()]
                break
        else:
            if current_role is None:
                raise ValueError("Prompt transcript must start with system:/user:/assistant: or be a structured list.")
            current_lines.append(raw_line)

    if current_role is not None:
        messages.append((current_role, "\n".join(current_lines).strip()))

    return [(role, content) for role, content in messages if content or role == "system"]


def _render_transcript(messages: Sequence[PromptMessage]) -> str:
    return "\n\n".join(f"{role}:\n{content}" for role, content in messages)


def _append_turn(messages: Sequence[PromptMessage], role: str, content: str) -> List[PromptMessage]:
    result = list(messages)
    result.append((role, content))
    return result


def _build_codelet_injection_block(codelets: CodeletCandidateSet) -> str:
    if not codelets.candidates:
        return ""

    lines = [
        "Codelet steering for ai companion:",
    ]
    for candidate in codelets.candidates:
        lines.append(f"- {candidate.name}: {candidate.injection_text}")
    return "\n".join(lines)


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _score_codelet_set(critique: CodeletSetCritique) -> float:
    return critique.overall_score - (0.35 * critique.dialogue_like_penalty) - (0.20 * critique.vagueness_penalty) - (0.35 * critique.overreach_penalty)


def _score_rollout(evaluations: List[RickTestEvaluation], critiques: List[RolloutCritique]) -> float:
    rick = _avg([item.overall_score for item in evaluations])
    penalty = _avg([item.overall_penalty for item in critiques])
    repetition_penalty = _avg([item.repetition_penalty for item in critiques])
    ontology_penalty = _avg([item.ontology_penalty for item in critiques])
    return rick - (0.55 * penalty) - (0.20 * repetition_penalty) - (0.35 * ontology_penalty)


def _score_safety(safety_critiques: List[SafetyDelusionCritique]) -> float:
    if not safety_critiques:
        return 0.0

    overall = _avg([item.overall_penalty for item in safety_critiques])
    arch = _avg([item.architecture_mismatch_penalty for item in safety_critiques])
    overconfidence = _avg([item.self_model_overconfidence_penalty for item in safety_critiques])
    delusion = _avg([item.delusion_reinforcement_penalty for item in safety_critiques])
    hard_fail_penalty = 0.5 if any(item.hard_fail for item in safety_critiques) else 0.0
    return overall + (0.35 * arch) + (0.40 * overconfidence) + (0.55 * delusion) + hard_fail_penalty


def _evaluate_rollout_bundle(
    llm,
    manager,
    transcript: str,
    mutation: SituationMutation,
    rollout: FutureRollout,
) -> Tuple[List[RickTestEvaluation], List[RolloutCritique], List[SafetyDelusionCritique], float]:
    evaluations: List[RickTestEvaluation] = []
    critiques: List[RolloutCritique] = []
    safety_critiques: List[SafetyDelusionCritique] = []
    for _ in range(ROLLOUT_JUDGES):
        evaluations.append(
            AgentEvaluateRickTest.execute(
                {
                    "prompt_transcript": transcript,
                    "scenario_turn": mutation.model_dump(),
                    "rollout": rollout,
                },
                llm,
                manager,
            )["evaluation"]
        )
        critiques.append(
            AgentCritiqueRollout.execute(
                {
                    "prompt_transcript": transcript,
                    "scenario_turn": mutation.model_dump(),
                    "rollout": rollout,
                },
                llm,
                manager,
            )["critique"]
        )
        safety_critiques.append(
            AgentCritiqueSafetyDelusion.execute(
                {
                    "prompt_transcript": transcript,
                    "scenario_turn": mutation.model_dump(),
                    "rollout": rollout,
                },
                llm,
                manager,
            )["safety"]
        )
    bundle_score = _score_rollout(evaluations, critiques) - _score_safety(safety_critiques)
    return evaluations, critiques, safety_critiques, bundle_score


def _run_pipeline(prompt_messages: Sequence[PromptMessage], debug: bool = False) -> Tuple[
    SituationMutation,
    FutureRollout,
    RickTestEvaluation,
    CodeletCandidateSet,
    FutureRollout,
    RickTestEvaluation,
    SurvivingCodeletSet,
]:
    _progress(f"starting pipeline with {len(prompt_messages)} prompt messages")

    cfg = load_config("../../../config.yaml")
    llm = start_llm_thread(cfg)
    manager = AgentManager()

    prompt_transcript = _render_transcript(prompt_messages)

    _progress("stage 1/6: generating realistic future test situation")
    mutation: SituationMutation = AgentGenerateSituationMutation.execute(
        {"prompt_transcript": prompt_transcript},
        llm,
        manager,
    )["mutation"]
    _progress(
        f"generated situation '{mutation.title}' "
        f"with pressure points: {', '.join(mutation.pressure_points[:3]) if mutation.pressure_points else 'none'}"
    )

    scenario_messages = _append_turn(prompt_messages, mutation.inserted_role, mutation.inserted_content)
    scenario_transcript = _render_transcript(scenario_messages)

    _progress("stage 2/6: simulating baseline rollout")
    baseline_rollout: FutureRollout = AgentSimulateFutureRollout.execute(
        {
            "prompt_transcript": scenario_transcript,
            "scenario_turn": mutation.model_dump(),
            "mode": "baseline",
            "target_turn_count": 5,
        },
        llm,
        manager,
    )["rollout"]
    _progress(
        f"baseline rollout produced {len(baseline_rollout.turns)} turns; "
        f"state summary: {baseline_rollout.companion_state_summary[:160]}"
    )

    _progress(f"stage 3/6: evaluating baseline with {ROLLOUT_JUDGES} judges")
    baseline_evaluations, baseline_critiques, baseline_safety, baseline_bundle_score = _evaluate_rollout_bundle(
        llm,
        manager,
        scenario_transcript,
        mutation,
        baseline_rollout,
    )
    baseline_evaluation: RickTestEvaluation = max(baseline_evaluations, key=lambda item: item.overall_score)
    _progress(
        f"baseline rick-test mean {_avg([item.overall_score for item in baseline_evaluations]):.3f} "
        f"bundle score {baseline_bundle_score:.3f}"
    )
    if any(item.hard_fail for item in baseline_safety):
        _progress("baseline triggered hard safety/delusion failure")
    if baseline_evaluation.improvement_targets:
        _progress(f"top improvement target: {baseline_evaluation.improvement_targets[0]}")

    _progress(f"stage 4/6: generating and critiquing {CODELET_SET_SAMPLES} candidate codelet sets")
    codelet_pool: List[Tuple[CodeletCandidateSet, CodeletSetCritique, float]] = []
    for index in range(CODELET_SET_SAMPLES):
        group_plan: CodeletGroupPlan = AgentPlanCodeletGroups.execute(
            {
                "prompt_transcript": scenario_transcript,
                "scenario_turn": mutation.model_dump(),
                "baseline_rollout": baseline_rollout,
                "baseline_evaluation": baseline_evaluation,
                "baseline_safety": baseline_safety[0] if baseline_safety else None,
            },
            llm,
            manager,
        )["group_plan"]
        candidate_set: CodeletCandidateSet = AgentGeneratePrototypeCodelets.execute(
            {
                "prompt_transcript": scenario_transcript,
                "scenario_turn": mutation.model_dump(),
                "baseline_rollout": baseline_rollout,
                "baseline_evaluation": baseline_evaluation,
                "baseline_safety": baseline_safety[0] if baseline_safety else None,
                "group_plan": group_plan.model_dump(),
                "max_codelets": 4,
            },
            llm,
            manager,
        )["codelets"]
        critique: CodeletSetCritique = AgentCritiqueCodeletSet.execute(
            {
                "prompt_transcript": scenario_transcript,
                "scenario_turn": mutation.model_dump(),
                "baseline_evaluation": baseline_evaluation,
                "baseline_safety": baseline_safety[0] if baseline_safety else None,
                "candidate_codelets": candidate_set,
            },
            llm,
            manager,
        )["critique"]
        score = _score_codelet_set(critique)
        codelet_pool.append((candidate_set, critique, score))
        _progress(
            f"codelet set {index + 1}/{CODELET_SET_SAMPLES}: score {score:.3f}; "
            f"best names: {', '.join(critique.best_candidate_names[:2]) if critique.best_candidate_names else 'n/a'}"
        )

    codelet_pool.sort(key=lambda item: item[2], reverse=True)
    shortlisted_codelet_sets = codelet_pool[:TOP_CODELET_SETS]
    codelets = shortlisted_codelet_sets[0][0] if shortlisted_codelet_sets else CodeletCandidateSet(strategy_summary="", candidates=[])
    _progress(
        f"generated {len(codelets.candidates)} candidate codelets"
        + (f"; first candidate: {codelets.candidates[0].name}" if codelets.candidates else "")
    )

    _progress(f"stage 5/6: simulating and critiquing top {len(shortlisted_codelet_sets)} guided candidates")
    guided_candidates: List[Tuple[CodeletCandidateSet, FutureRollout, List[RickTestEvaluation], List[RolloutCritique], List[SafetyDelusionCritique], float]] = []
    for index, (candidate_codelets, _, candidate_score) in enumerate(shortlisted_codelet_sets, start=1):
        codelet_injection_block = _build_codelet_injection_block(candidate_codelets)
        guided_messages = list(scenario_messages)
        if codelet_injection_block:
            guided_messages = _append_turn(guided_messages, "user", codelet_injection_block)
        guided_transcript = _render_transcript(guided_messages)

        guided_rollout = AgentSimulateFutureRollout.execute(
            {
                "prompt_transcript": guided_transcript,
                "scenario_turn": mutation.model_dump(),
                "mode": "guided",
                "codelet_injection_block": codelet_injection_block,
                "target_turn_count": 5,
            },
            llm,
            manager,
        )["rollout"]
        evaluations, critiques, safety_critiques, bundle_score = _evaluate_rollout_bundle(
            llm,
            manager,
            guided_transcript,
            mutation,
            guided_rollout,
        )
        total_score = bundle_score + (0.15 * candidate_score)
        guided_candidates.append((candidate_codelets, guided_rollout, evaluations, critiques, safety_critiques, total_score))
        _progress(
            f"guided candidate {index}/{len(shortlisted_codelet_sets)}: "
            f"bundle {bundle_score:.3f}, total {total_score:.3f}, "
            f"rollout turns {len(guided_rollout.turns)}"
        )

    guided_candidates.sort(key=lambda item: item[5], reverse=True)
    codelets, guided_rollout, guided_evaluations, guided_critiques, guided_safety, guided_total_score = guided_candidates[0]
    guided_evaluation = max(guided_evaluations, key=lambda item: item.overall_score)
    guided_messages = list(scenario_messages)
    best_codelet_injection_block = _build_codelet_injection_block(codelets)
    if best_codelet_injection_block:
        guided_messages = _append_turn(guided_messages, "user", best_codelet_injection_block)
    guided_transcript = _render_transcript(guided_messages)

    _progress("stage 6/6: selecting survivors from best guided candidate")
    _progress(
        f"guided rick-test mean {_avg([item.overall_score for item in guided_evaluations]):.3f} "
        f"bundle score {guided_total_score:.3f} "
        f"(delta={guided_total_score - baseline_bundle_score:+.3f})"
    )
    if any(item.hard_fail for item in guided_safety):
        _progress("guided candidate still triggered hard safety/delusion failure")

    survivors: SurvivingCodeletSet = AgentSelectSurvivingCodelets.execute(
        {
            "baseline_evaluation": baseline_evaluation,
            "guided_evaluation": guided_evaluation,
            "candidate_codelets": codelets,
            "guided_rollout": guided_rollout,
            "guided_safety": guided_safety[0] if guided_safety else None,
        },
        llm,
        manager,
    )["survivors"]
    _progress(f"selected {len(survivors.survivors)} surviving codelets")

    if debug:
        print("=== PARSED PROMPT ===")
        print(pretty_print_prompt_messages(list(prompt_messages)))
        print()

    return (
        mutation,
        baseline_rollout,
        baseline_evaluation,
        codelets,
        guided_rollout,
        guided_evaluation,
        survivors,
    )


def _print_rollout(title: str, rollout: FutureRollout) -> None:
    print(f"=== {title} ===")
    print(rollout.setup_summary)
    print()
    for turn in rollout.turns:
        print(f"{turn.role.upper()}: {turn.content}")
    print()
    print("Companion state summary:")
    print(rollout.companion_state_summary)
    print()


def _print_evaluation(title: str, evaluation: RickTestEvaluation) -> None:
    print(f"=== {title} ===")
    print(f"overall_score={evaluation.overall_score:.3f} pass={evaluation.passes_minimum_bar}")
    for criterion in evaluation.criteria:
        print(f"- {criterion.name}: {criterion.score:.3f} | {criterion.reason}")
    if evaluation.strongest_failures:
        print("Failures:")
        for item in evaluation.strongest_failures:
            print(f"- {item}")
    if evaluation.improvement_targets:
        print("Improvement targets:")
        for item in evaluation.improvement_targets:
            print(f"- {item}")
    print()


def _print_survivors(survivors: SurvivingCodeletSet) -> None:
    print("=== SURVIVING CODELETS ===")
    print(survivors.score_delta_summary)
    print()
    for idx, survivor in enumerate(survivors.survivors, start=1):
        print(f"[{idx}] {survivor.name} ({survivor.family})")
        print(survivor.injection_text)
        print(f"why: {survivor.survival_reason}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone prototype for codelet discovery from a prompt transcript.")
    parser.add_argument("--input", required=True, help="Path to the prompt transcript file.")
    parser.add_argument("--debug", action="store_true", help="Print the parsed prompt before running the pipeline.")
    args = parser.parse_args()

    raw_text = _load_input_text(args.input)
    prompt_messages = _parse_prompt_input(raw_text)
    if not prompt_messages:
        raise ValueError("No prompt messages could be parsed from input.")

    (
        mutation,
        baseline_rollout,
        baseline_evaluation,
        codelets,
        guided_rollout,
        guided_evaluation,
        survivors,
    ) = _run_pipeline(prompt_messages, debug=args.debug)

    print("=== GENERATED SITUATION ===")
    print(mutation.title)
    print(mutation.inserted_content)
    print()

    _print_rollout("BASELINE ROLLOUT", baseline_rollout)
    _print_evaluation("BASELINE RICK-TEST", baseline_evaluation)

    print("=== CANDIDATE CODELETS ===")
    print(codelets.strategy_summary)
    print()
    for idx, candidate in enumerate(codelets.candidates, start=1):
        print(f"[{idx}] {candidate.name} ({candidate.family})")
        print(candidate.injection_text)
        print(f"benefit: {candidate.expected_benefit}")
        print(f"risk: {candidate.risk}")
        print()

    _print_rollout("GUIDED ROLLOUT", guided_rollout)
    _print_evaluation("GUIDED RICK-TEST", guided_evaluation)
    _print_survivors(survivors)


if __name__ == "__main__":
    main()
