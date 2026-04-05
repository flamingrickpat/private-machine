from typing import List, Sequence, Tuple

from pm.ghost.ghost_base import BaseGhost
from pm.model.knoxel_common import MemoryClusterKnoxel
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_enums import FeatureType, KnoxelSubtypeBase, ClusterType
from pm.model.knoxel_feature import Feature
from pm.subsystems.context.context_planner_base import ContextPlannerOutput, VirtualKnoxel


DEFAULT_ASSISTANT_TURN_SUB_TYPES: tuple[KnoxelSubtypeBase, ...] = (
    FeatureType.Dialogue,
    FeatureType.Thought,
    FeatureType.ExternalThought,
    ClusterType.Topical
)


def render_context_messages_common(
    ghost: BaseGhost,
    planner_output: ContextPlannerOutput,
    system_prompt: str,
    first_user_turn_start: str,
    first_user_turn_end: str,
    first_user_lane_names: Sequence[str],
    conversation_lane_names: Sequence[str],
    assistant_turn_types: Sequence[FeatureType] = DEFAULT_ASSISTANT_TURN_SUB_TYPES,
) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = [("system", system_prompt)]

    first_user_parts: List[str] = [first_user_turn_start]
    for lane_name in first_user_lane_names:
        first_user_parts.extend(_render_lane_items(planner_output.lane_data[lane_name]))
    first_user_parts.append(first_user_turn_end)
    messages.append(("user", "\n".join(part for part in first_user_parts if part)))

    conversation_items = []
    for lane_name in conversation_lane_names:
        conversation_items.extend(planner_output.lane_data[lane_name])

    messages.extend(
        _render_conversation_messages(
            items=conversation_items,
            assistant_turn_types=assistant_turn_types,
        )
    )
    messages.append(("assistant", f'{ghost.get_companion_name()} says: "'))
    return messages


def _render_lane_items(items: Sequence[object]) -> List[str]:
    return [_render_item(item) for item in items]


def _render_conversation_messages(
    items: Sequence[object],
    assistant_turn_types: Sequence[FeatureType],
) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = []
    current_role = ""
    current_parts: List[str] = []

    for item in items:
        role = _get_item_role(item, assistant_turn_types)
        content = _render_item(item)
        if current_role != role:
            if current_parts:
                messages.append((current_role, "\n".join(current_parts)))
            current_role = role
            current_parts = [content]
            continue
        current_parts.append(content)

    if current_parts:
        messages.append((current_role, "\n".join(current_parts)))
    return messages


def _get_item_role(item: KnoxelBase | VirtualKnoxel, assistant_turn_types: Sequence[KnoxelSubtypeBase]) -> str:
    if item.type in assistant_turn_types:
        return "assistant"
    if isinstance(item, KnoxelBase):
        if item.subtype in assistant_turn_types:
            return "assistant"

    return "user"


def _render_item(item: KnoxelBase | VirtualKnoxel) -> str:
    if isinstance(item, VirtualKnoxel):
        return item.content
    return item.get_story_element()
