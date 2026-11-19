from __future__ import annotations

from dataclasses import dataclass
from pm.context_render.prompt import RenderSettings, RenderedPrompt, require_rendered_prompt
from pm.context_render.turns import Turn, Role


@dataclass(frozen=True, slots=True)
class NoteRenderInput:
    instruction: str


def render_note_selection(input: NoteRenderInput, settings: RenderSettings) -> RenderedPrompt:
    system = (
        "Determine whether this original message requests a new note-file write. "
        "An apology, affection, frustration, remembered task, or workflow status is not a new note request. "
        "Do not treat the message itself as note content unless it explicitly requests that. "
        "Return only JSON with requested (boolean), path, and text. If no new note is requested, "
        "set requested false and text empty. Otherwise extract the exact requested text and path; "
        "default path is outbox/note.txt. Do not add a newline or change the requested text.")
    return require_rendered_prompt("note selection", (Turn(Role.System, system), Turn(Role.User, input.instruction)), settings)
