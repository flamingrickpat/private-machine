# do not remove! i dont want to see warnings
import shutup
shutup.please()

import logging
import random
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import PydanticDeprecatedSince20

from pm.config_loader import companion_name, user_name
from pm.data_structures import FeatureType, Stimulus, StimulusType
from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_codelets import GhostCodelets
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.ghosts.procedures.inner_dialogue import InnerDialogueProc
from pm.llm.llm_proxy import start_llm_thread

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

logger = logging.getLogger(__name__)
MAIN_USER_ACTOR_ID = 1


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


def _init_or_load_ghost(db_path: str | None) -> GhostCodelets:
    llm = start_llm_thread()
    ghost = GhostCodelets(llm, GhostConfig())

    ghost.enable_knoxel_trace = True
    ghost.knoxel_trace_output_dir = str(Path("data/knoxel_traces").resolve())

    create_new_persona = True
    if db_path:
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(db_path):
            create_new_persona = False

    ghost._initialize_actors()
    if create_new_persona:
        ghost.init_character()
    return ghost


def _run_cycle(ghost: GhostCodelets, stimuli: list[Stimulus], print_reply: bool = True):
    ccq = ghost.cognitive_cycle(stimuli)
    print(f"\n--- TICK {ghost.current_tick_id} ---")
    reply_text = str(getattr(ghost, "simulated_reply", "") or "").strip()
    if print_reply and reply_text:
        print(f"{companion_name}: {reply_text}")
    return ccq


def _cmd_thought(ghost: GhostCodelets):
    packet = InnerDialogueProc.run(ghost)
    print("[thought] topic:", packet.get("topic", ""))
    print("[thought] monologue:", packet.get("inner_dialogue", ""))
    opener = str(packet.get("proactive_opener", "") or "").strip()
    if opener:
        print("[thought] opener:", opener)


def _cmd_lida(ghost: GhostCodelets):
    _run_cycle(ghost, [], print_reply=False)
    print("[lida] tick complete")


def _get_runtime_now(ghost: GhostCodelets) -> datetime:
    raw = getattr(ghost, "runtime_now_override", None)
    if isinstance(raw, datetime):
        return raw
    if raw:
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            pass
    ts = getattr(getattr(ghost, "current_state", None), "timestamp", None)
    return ts or datetime.now()


def _cmd_time(ghost: GhostCodelets, raw: str):
    parts = raw.strip().split()
    if len(parts) < 2:
        now_dt = _get_runtime_now(ghost)
        walk = bool(getattr(ghost, "runtime_time_walk_enabled", False))
        max_step = int(getattr(ghost, "runtime_time_walk_max_step_min", 20) or 20)
        print(f"[time] now={now_dt.isoformat(timespec='minutes')} walk={walk} max_step_min={max_step}")
        return

    sub = parts[1].lower()
    if sub == "set" and len(parts) >= 3:
        val = " ".join(parts[2:]).strip()
        try:
            ghost.runtime_now_override = datetime.fromisoformat(val)
            print(f"[time] set -> {ghost.runtime_now_override.isoformat(timespec='minutes')}")
        except Exception as e:
            print(f"[time] invalid datetime: {e}")
        return

    if sub == "add" and len(parts) >= 3:
        try:
            delta_min = int(parts[2])
            ghost.runtime_now_override = _get_runtime_now(ghost) + timedelta(minutes=delta_min)
            print(f"[time] add {delta_min}m -> {ghost.runtime_now_override.isoformat(timespec='minutes')}")
        except Exception as e:
            print(f"[time] invalid minutes: {e}")
        return

    if sub == "walk" and len(parts) >= 3:
        mode = parts[2].lower()
        if mode == "on":
            step = 20
            if len(parts) >= 4:
                try:
                    step = max(1, int(parts[3]))
                except Exception:
                    pass
            ghost.runtime_time_walk_enabled = True
            ghost.runtime_time_walk_max_step_min = step
            print(f"[time] random walk enabled (max_step_min={step})")
        elif mode == "off":
            ghost.runtime_time_walk_enabled = False
            print("[time] random walk disabled")
        else:
            print("[time] usage: /time walk on [max_step_min] | /time walk off")
        return

    now_dt = _get_runtime_now(ghost)
    print(f"[time] now={now_dt.isoformat(timespec='minutes')}")


def main(db_path: str | None = "data/main.db"):
    ghost = _init_or_load_ghost(db_path)
    ghost.runtime_now_override = None
    ghost.runtime_started_at = datetime.now()
    ghost.runtime_time_walk_enabled = False
    ghost.runtime_time_walk_max_step_min = 20

    for f in ghost.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")

    while True:
        text = input(f"{user_name}: ").strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.startswith("/thought"):
            _cmd_thought(ghost)
            continue
        if text.startswith("/lida"):
            _cmd_lida(ghost)
            continue
        if text.startswith("/time"):
            _cmd_time(ghost, text)
            continue

        if bool(getattr(ghost, "runtime_time_walk_enabled", False)):
            step_max = int(getattr(ghost, "runtime_time_walk_max_step_min", 20) or 20)
            step = random.randint(1, max(1, step_max))
            ghost.runtime_now_override = _get_runtime_now(ghost) + timedelta(minutes=step)

        msg_text, msg_source, msg_actor_id = _parse_partner_metadata(text)
        if not msg_text:
            continue

        stim = Stimulus(
            content=msg_text,
            stimulus_type=StimulusType.UserMessage,
            source=msg_source,
            source_actor_id=msg_actor_id,
            based_on_tick=max(0, ghost.current_tick_id - 1),
            async_sub_tick=1,
            async_tick_insert_begin=True,
            async_tick_source_order=1,
        )
        _run_cycle(ghost, [stim], print_reply=True)


if __name__ == "__main__":
    main("data/main.db")
