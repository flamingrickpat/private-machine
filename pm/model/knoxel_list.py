from typing import Optional, List
import numpy as np
from py_linq import Enumerable

from pm.model.knoxel_core import KnoxelBase, KnoxelHaver
from pm.utils.token_utils import get_token_count


class KnoxelList:
    def __init__(self, knoxels: Optional[List[KnoxelBase]] = None):
        self._list: List[KnoxelBase] = list(knoxels) if knoxels else []

    def to_json(self):
        lst = ",".join(self._list)
        return f"[{lst}]"

    def get_token_count(self):
        s = self.get_story(None)
        return get_token_count(s)

    def get_story(self, max_tokens: Optional[int] = None, target_knoxel_ids: List[int] = None) -> str:
        last_timestamp = None

        if not self._list:
            return ""

        ghost: KnoxelHaver = self._list[0]._owner

        for item in self._list:
            if last_timestamp is None:
                last_timestamp = item.timestamp_world_begin

            if item.timestamp_world_begin < last_timestamp:
                raise Exception("Linear history mixed up!")
            last_timestamp = item.timestamp_world_begin

        content_list = [(k.id, k.get_story_element()) for k in self._list]
        current_tokens = 0
        buffer = []

        for _id, content in reversed(content_list):
            tc = get_token_count(content)
            current_tokens += tc
            if max_tokens is not None and current_tokens > max_tokens:
                break
            buffer.insert(0, content)
            if target_knoxel_ids is not None:
                target_knoxel_ids.insert(0, _id)

        narrative = "\n".join(buffer)
        return narrative

    def get_embeddings_np(self) -> Optional[np.ndarray]:
        embeddings = [k.embedding for k in self._list if k.embedding is not None]
        if not embeddings:
            return None
        return np.array(embeddings)

    def reverse(self) -> 'KnoxelList':
        return KnoxelList(list(reversed(self._list)))

    def where(self, predicate) -> 'KnoxelList':
        return KnoxelList([x for x in self._list if predicate(x)])

    def order_by(self, key_selector) -> 'KnoxelList':
        return KnoxelList(sorted(self._list, key=key_selector))

    def order_by_descending(self, key_selector) -> 'KnoxelList':
        return KnoxelList(sorted(self._list, key=key_selector, reverse=True))

    def take(self, count: int) -> 'KnoxelList':
        return KnoxelList(self._list[:count])

    def take_last(self, count: int) -> 'KnoxelList':
        return KnoxelList(self._list[-count:])

    def add(self, knoxel: KnoxelBase):
        self._list.append(knoxel)

    def to_list(self) -> List[KnoxelBase]:
        return list(self._list)

    def last_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]:
        if predicate is None:
            return self._list[-1] if self._list else default
        for item in reversed(self._list):
            if predicate(item):
                return item
        return default

    def first_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]:
        if predicate is None:
            return self._list[0] if self._list else default
        for item in self._list:
            if predicate(item):
                return item
        return default

    def any(self, predicate=None):
        if predicate is None:
            return bool(self._list)
        return any(predicate(x) for x in self._list)

    def as_enumerable(self) -> Enumerable:
        return Enumerable(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self):
        return iter(self._list)