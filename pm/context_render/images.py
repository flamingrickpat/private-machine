from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from pm.context_render.turns import Turn
from pm.records.media import RecordImage


@dataclass(frozen=True, slots=True)
class ImageCatalog:
    records: tuple[RecordImage, ...] = ()

    def find_image(self, image_id: str) -> RecordImage | None:
        for image in self.records:
            if image.id == image_id:
                return image
        return None


def resolve_media(turns: Sequence[Turn], catalog: ImageCatalog) -> tuple[Turn, ...]:
    from pm.llm.multimodal import resolve_turns
    return tuple(resolve_turns(turns, catalog))
