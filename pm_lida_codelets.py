# do not remove! i dont want to see warnings
import shutup
shutup.please()

import logging
import queue
import re
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Optional

from pm.config_loader import companion_name, user_name
from pm.data_structures import FeatureType, Stimulus, StimulusType
from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_codelets import GhostCodelets
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.ghosts.procedures.inner_dialogue import InnerDialogueProc
from pm.llm.llm_proxy import LlmManagerProxy, start_llm_thread

logger = logging.getLogger(__name__)
MAIN_USER_ACTOR_ID = 1


def get_queue() -> Queue:
    return queue.Queue()


def _init_ghost(llm: LlmManagerProxy, cfg: GhostConfig, db_path: str | None = None, load_from_db: bool = False) -> GhostCodelets:
    ghost = GhostCodelets(llm, cfg)
    create_new_persona = True
    if load_from_db and db_path:
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(db_path):
            create_new_persona = False
    ghost._initialize_actors()
    if not hasattr(ghost, "runtime_started_at"):
        ghost.runtime_started_at = datetime.now()
    if not hasattr(ghost, "runtime_now_override"):
        ghost.runtime_now_override = None
    if create_new_persona:
        ghost.init_character()
    return ghost


def _persist_ghost(ghost: GhostCodelets, db_path: str | None):
    if not db_path:
        return
    pers = PersistSqlite(ghost)
    pers.save_state_sqlite(db_path)


def _parse_partner_metadata(raw_text: str) -> tuple[str, str, int]:
    """
    Parse optional shell metadata for multi-partner input.
    Accepted prefixes:
    - [id=2] hello
    - [id=2,name=alice] hello
    - @2 hello
    """
    text = str(raw_text or "").strip()
    if not text:
        return "", user_name, MAIN_USER_ACTOR_ID

    m = re.match(r"^\[id=(\d+)(?:,name=([^\]]+))?\]\s*(.*)$", text, flags=re.IGNORECASE)
    if m:
        actor_id = int(m.group(1))
        actor_name = (m.group(2) or f"partner_{actor_id}").strip()
        content = (m.group(3) or "").strip()
        return content, actor_name, actor_id

    m2 = re.match(r"^@(\d+)\s+(.*)$", text)
    if m2:
        actor_id = int(m2.group(1))
        content = (m2.group(2) or "").strip()
        return content, f"partner_{actor_id}", actor_id

    return text, user_name, MAIN_USER_ACTOR_ID


def _build_stimulus_from_user(ghost: GhostCodelets, msg_text: str, msg_source: str, msg_actor_id: int, sub_tick: int) -> Stimulus:
    base_tick = int(getattr(ghost, "current_tick_id", 0) or 0)
    return Stimulus(
        content=msg_text,
        stimulus_type=StimulusType.UserMessage,
        source=msg_source,
        source_actor_id=msg_actor_id,
        based_on_tick=base_tick,
        async_sub_tick=sub_tick,
        async_tick_insert_begin=(sub_tick == 1),
        async_tick_source_order=1,
    )


def _build_inner_stimuli(ghost: GhostCodelets, sub_tick: int) -> tuple[list[Stimulus], str]:
    packet = InnerDialogueProc.run(ghost) or {}
    based_on_tick = int(getattr(ghost, "current_tick_id", 0) or 0)
    stimuli: list[Stimulus] = []

    thought_text = str(packet.get("inner_dialogue", "") or "").strip()
    lida_context = str(packet.get("lida_context", "") or thought_text).strip()
    if lida_context:
        stimuli.append(
            Stimulus(
                content=lida_context,
                stimulus_type=StimulusType.SystemMessage,
                source="InnerDialogueThread",
                based_on_tick=based_on_tick,
                async_sub_tick=sub_tick,
                async_tick_insert_begin=False,
                async_tick_source_order=1,
            )
        )

    opener = str(packet.get("proactive_opener", "") or "").strip()
    if opener:
        stimuli.append(
            Stimulus(
                content=opener,
                stimulus_type=StimulusType.CompanionMessage,
                source=companion_name,
                based_on_tick=based_on_tick,
                async_sub_tick=sub_tick,
                async_tick_insert_begin=False,
                async_tick_source_order=2,
            )
        )

    return stimuli, opener


def worker_user_facing_input(queue_text_input: Queue, queue_central_events: Queue):
    while True:
        raw = queue_text_input.get(True, timeout=None)
        if raw is None:
            queue_central_events.put({"kind": "shutdown"})
            return

        msg_text, msg_source, msg_actor_id = _parse_partner_metadata(str(raw))
        if not msg_text:
            continue

        queue_central_events.put(
            {
                "kind": "user_message",
                "text": msg_text,
                "source": msg_source,
                "actor_id": int(msg_actor_id),
            }
        )


def worker_inner_thoughts(queue_central_events: Queue, interval_s: float = 45.0):
    while True:
        time.sleep(max(1.0, interval_s))
        queue_central_events.put({"kind": "inner_tick"})


def worker_central_lida(
    ghost: GhostCodelets,
    queue_central_events: Queue,
    queue_text_output: Queue,
    db_path: Optional[str],
    lida_interval_s: float = 180.0,
):
    sub_tick = 0
    next_idle_due = time.time() + max(1.0, lida_interval_s)

    while True:
        timeout = max(0.05, next_idle_due - time.time())
        try:
            event = queue_central_events.get(True, timeout=timeout)
        except queue.Empty:
            event = {"kind": "idle_tick"}

        kind = str((event or {}).get("kind", "")).strip().lower()
        if kind == "shutdown":
            return

        try:
            if kind == "user_message":
                sub_tick += 1
                stim = _build_stimulus_from_user(
                    ghost=ghost,
                    msg_text=str(event.get("text", "") or "").strip(),
                    msg_source=str(event.get("source", user_name) or user_name),
                    msg_actor_id=int(event.get("actor_id", MAIN_USER_ACTOR_ID) or MAIN_USER_ACTOR_ID),
                    sub_tick=sub_tick,
                )
                ghost.cognitive_cycle([stim])
                _persist_ghost(ghost, db_path)

                reply = str(getattr(ghost, "simulated_reply", "") or "").strip()
                if reply:
                    queue_text_output.put(reply)

                next_idle_due = time.time() + max(1.0, lida_interval_s)
                continue

            if kind == "inner_tick":
                sub_tick += 1
                stimuli, opener = _build_inner_stimuli(ghost, sub_tick=sub_tick)
                if stimuli:
                    ghost.cognitive_cycle(stimuli)
                    _persist_ghost(ghost, db_path)
                if opener:
                    queue_text_output.put(opener)
                next_idle_due = time.time() + max(1.0, lida_interval_s)
                continue

            # Central autonomous LIDA update tick.
            ghost.cognitive_cycle([])
            _persist_ghost(ghost, db_path)
            next_idle_due = time.time() + max(1.0, lida_interval_s)
        except Exception:
            logger.exception("Central LIDA worker event failed: %s", kind)


def main(_db_path: str | None = None):
    main_llm = start_llm_thread()

    config = GhostConfig()
    ghost_lida = _init_ghost(main_llm, config, db_path=_db_path, load_from_db=True)

    for f in ghost_lida.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")

    queue_text_input = get_queue()
    queue_text_output = get_queue()
    queue_central_events = get_queue()

    t_user = threading.Thread(
        target=worker_user_facing_input,
        args=(queue_text_input, queue_central_events),
        daemon=True,
    )
    t_user.start()

    t_inner = threading.Thread(
        target=worker_inner_thoughts,
        args=(queue_central_events,),
        daemon=True,
    )
    t_inner.start()

    t_central = threading.Thread(
        target=worker_central_lida,
        args=(ghost_lida, queue_central_events, queue_text_output, _db_path),
        daemon=True,
    )
    t_central.start()

    while True:
        user_input_cmd = input(f"{config.user_name}: ")
        if str(user_input_cmd or "").strip().lower() in {"quit", "exit"}:
            queue_text_input.put(None)
            break

        queue_text_input.put(user_input_cmd)

        # Print first available output (user-turn reply or proactive opener).
        res = queue_text_output.get()
        print(f"{companion_name}: {res}")


if __name__ == '__main__':
    main("./data/main.db")
