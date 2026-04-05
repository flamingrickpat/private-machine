import logging
import json

from pm.ghost.ghost_config import GhostConfig
from pm.ghost.ghost_r2 import GhostR2
from pm.model.knoxel_common import Stimulus
from pm.model.knoxel_enums import StimulusType
from pm.persist.persist_sqlite import PersistSqlite
from pm.subsystems.codelet_discovery.prototype_codelet_discovery import _parse_prompt_input, _run_pipeline
from pm.subsystems.context.common_context import plan_context_story_simple
from pm.subsystems.context.render_presets import context_render_story_messages_assistant_for_dialoge_static_info
from pm.system.llm.llm_proxy import start_llm_thread
from pm.system.load_config import load_config

logger = logging.getLogger(__name__)


def _debug_run_codelet_discovery(real_inp) -> None:
    prompt_messages = _parse_prompt_input(
        json.dumps(real_inp)
    )

    (
        mutation,
        baseline_rollout,
        baseline_evaluation,
        codelets,
        guided_rollout,
        guided_evaluation,
        survivors,
    ) = _run_pipeline(prompt_messages, debug=True)

    print("=== GENERATED SITUATION ===")
    print(mutation.title)
    print(mutation.inserted_content)
    print()

    print("=== BASELINE SCORE ===")
    print(baseline_evaluation.overall_score)
    print(baseline_evaluation.improvement_targets)
    print()

    print("=== CANDIDATE CODELETS ===")
    for candidate in codelets.candidates:
        print(candidate.name)
        print(candidate.injection_text)
        print()

    print("=== GUIDED SCORE ===")
    print(guided_evaluation.overall_score)
    print()

    print("=== SURVIVORS ===")
    for survivor in survivors.survivors:
        print(survivor.name)
        print(survivor.injection_text)
        print()

    print("=== BASELINE ROLLOUT ===")
    for turn in baseline_rollout.turns:
        print(f"{turn.role}: {turn.content}")
    print()

    print("=== GUIDED ROLLOUT ===")
    for turn in guided_rollout.turns:
        print(f"{turn.role}: {turn.content}")
    print()


def main(_db_path: str | None = None):
    cfg = load_config("config.yaml")
    main_llm = start_llm_thread(cfg)

    config = GhostConfig(companion_name=cfg.companion_name, user_name=cfg.user_name, universal_character_card=cfg.character_card_story)
    ghost = GhostR2(main_llm, config)
    ghost.system_config = cfg

    class LlamaTest:
        name: str
        message: str

    #a = main_llm.completion_text(LlmPreset.Fast, [("system", "You are a helpful assistant."), ("user", "Hi :)")])
    #print(a)

    create_new_persona = True
    if _db_path is not None and _db_path != "":
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(_db_path):
            create_new_persona = False

    ghost.initialize_basic_knoxels()

    #res = plan_context_story_simple(ghost, main_llm.get_embedding("cats, kittens and other felines"), 8000)
    #print(res)

    #cnt = 0
    #while True:
    #    try:
    #        clear_derived_memory_knoxels(ghost)
    #        ghost.memory_consolidator.consolidate_memory_if_needed()
    #    except Exception as e:
    #        print(f"explol {e}")
    #    pers = PersistSqlite(ghost)
    #    pers.save_state_sqlite(f"tmp{cnt}.db")
    #    try:
    #        pers = PersistSqlite(ghost)
    #        pers.save_state_sqlite(f"F:/pm/tmp{cnt}.db")
    #    except:
    #        pass
    #    cnt += 1
    #    time.sleep(60)
    #    exit(1)

    planner_output, _, _ = plan_context_story_simple(ghost, main_llm.get_embedding("cats, kittens and other felines, animals"), 16000)
    turns = context_render_story_messages_assistant_for_dialoge_static_info(ghost, planner_output)
    #print(pretty_print_prompt_messages(turns))
    _debug_run_codelet_discovery(turns)
    exit(1)

    #_debug_run_codelet_discovery()
    #exit(1)

    #b = main_llm.completion_text(LlmPreset.Default, turns, comp_settings=CommonCompSettings(max_tokens=1000, stop_on_eot=True))
    #print(b)
    #exit(1)
    #if False:
    #    for f in ghost.all_features:
    #        if f.feature_type == FeatureType.Dialogue:
    #            if f.source == cfg.user_name:
    #                print(f"{cfg.user_name}: {f.content}")
    #            elif f.source == cfg.companion_name:
    #                print(f"{cfg.companion_name}: {f.content}")

    #graph_mem = CognitiveMemoryManager(ghost.llm, ghost)
    #res = graph_mem.answer_question("what is the relationship between user and ai companion?")
    #print(res.content)
    #exit(1)

    #inp = "heyo how are you?"
    #sub_tick = 1
    #stimulus_from_user = Stimulus(
    #    content=inp,
    #    stimulus_type=StimulusType.UserMessage,
    #    source=ghost.ghost_config.user_name,
    #    based_on_tick=ghost.current_tick_id,
    #    async_sub_tick=sub_tick,
    #    async_tick_insert_begin=sub_tick == 1,
    #    async_tick_source_order=1
    #)
    #ccq = ghost.cognitive_cycle([stimulus_from_user])
    ##print(ccq.as_story)
    #pers = PersistSqlite(ghost)
    #pers.save_state_sqlite("tmp.db")

if __name__ == '__main__':
    main("./data/main.db")
