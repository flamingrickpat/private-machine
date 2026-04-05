from typing import List, Tuple

from pm.data.prompts.character_card_addendum import architecture_description_story_detailed
from pm.data.prompts.render_story_prompts import RenderStoryPromptsV1
from pm.ghost.ghost_base import BaseGhost
from pm.model.knoxel_common import CauseEffectKnoxel, DeclarativeFactKnoxel, Narrative
from pm.subsystems.context.context_planner_base import VirtualKnoxel
from pm.subsystems.context.context_planner_base import ContextPlannerOutput
from pm.subsystems.context.render_common import render_context_messages_common

def context_render_story_messages_assistant_for_dialoge(
    ghost: BaseGhost,
    planner_output: ContextPlannerOutput
) -> List[Tuple[str, str]]:
    architecture_text = architecture_description_story_detailed.format(
        companion_name=ghost.get_companion_name(),
        user_name=ghost.get_user_name(),
        available_tools=", ".join(ghost.system_config.supported_capabilities + ghost.system_config.unsupported_capabilities),
    ).strip()
    first_user_turn_start = (
        RenderStoryPromptsV1.FIRST_USER_TURN_START
        + "\n\nCharacter Card\n"
        + ghost.get_simple_character_story_block().strip()
        + "\n\nArchitecture\n"
        + architecture_text
    )
    return render_context_messages_common(
        ghost=ghost,
        planner_output=planner_output,
        system_prompt=RenderStoryPromptsV1.SYSTEM_PROMPT,
        first_user_turn_start=first_user_turn_start,
        first_user_turn_end=RenderStoryPromptsV1.FIRST_USER_TURN_END,
        first_user_lane_names=[],
        conversation_lane_names=["historical", "recent"]
    )


def context_render_story_messages_assistant_for_dialoge_static_info(
    ghost: BaseGhost,
    planner_output: ContextPlannerOutput
) -> List[Tuple[str, str]]:
    architecture_text = architecture_description_story_detailed.format(
        companion_name=ghost.get_companion_name(),
        user_name=ghost.get_user_name(),
        available_tools=", ".join(ghost.system_config.supported_capabilities + ghost.system_config.unsupported_capabilities),
    ).strip()

    first_user_parts = [
        RenderStoryPromptsV1.FIRST_USER_TURN_START,
        "",
        "Character Card",
        ghost.get_simple_character_story_block().strip(),
        "",
        "Architecture",
        architecture_text,
    ]

    first_user_parts.extend(_build_static_info_sections(planner_output))

    first_user_turn_start = "\n".join(part for part in first_user_parts if part is not None)

    return render_context_messages_common(
        ghost=ghost,
        planner_output=planner_output,
        system_prompt=RenderStoryPromptsV1.SYSTEM_PROMPT,
        first_user_turn_start=first_user_turn_start,
        first_user_turn_end=RenderStoryPromptsV1.FIRST_USER_TURN_END,
        first_user_lane_names=[],
        conversation_lane_names=["historical", "recent"]
    )


def _build_static_info_sections(planner_output: ContextPlannerOutput) -> List[str]:
    sections: List[str] = []

    sections.extend(
        _render_static_lane_section(
            planner_output=planner_output,
            lane_name="narratives_ai",
            title="Companion Narrative Priors",
            description="These are distilled long-horizon self- and relationship priors for the assistant. Treat them as interpretive background, not literal quoted dialogue.",
        )
    )
    sections.extend(
        _render_static_lane_section(
            planner_output=planner_output,
            lane_name="narratives_user",
            title="User Narrative Priors",
            description="These are distilled priors about the user and the relationship. Use them as soft but important continuity constraints.",
        )
    )
    sections.extend(
        _render_static_lane_section(
            planner_output=planner_output,
            lane_name="facts",
            title="Relevant Facts",
            description="These are compact factual priors extracted from prior history. Prefer them for stable continuity unless the recent timeline clearly supersedes them.",
        )
    )
    sections.extend(
        _render_static_lane_section(
            planner_output=planner_output,
            lane_name="rules",
            title="Learned Interaction Patterns",
            description="These are observed cause/effect patterns from prior interactions. Let them shape expectations and response style implicitly.",
        )
    )

    return sections


def _render_static_lane_section(
    planner_output: ContextPlannerOutput,
    lane_name: str,
    title: str,
    description: str,
) -> List[str]:
    items = planner_output.lane_data.get(lane_name, [])
    if not items:
        return []

    rendered_items = [_render_static_info_item(item) for item in items]
    rendered_items = [item for item in rendered_items if item.strip()]
    if not rendered_items:
        return []

    return [
        "",
        title,
        description,
        "\n".join(rendered_items),
    ]


def _render_static_info_item(item: object) -> str:
    if isinstance(item, VirtualKnoxel):
        return item.content.strip()
    if isinstance(item, Narrative):
        return item.get_story_element().strip()
    if isinstance(item, DeclarativeFactKnoxel):
        return item.get_story_element().strip()
    if isinstance(item, CauseEffectKnoxel):
        return item.get_story_element().strip()
    if hasattr(item, "get_story_element"):
        return item.get_story_element().strip()
    return str(item).strip()
