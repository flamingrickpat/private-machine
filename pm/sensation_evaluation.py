from pm.character import companion_name
from pm.controller import controller
from pm.database.tables import Event
from pm.embedding.token import get_token
from pm.subsystems.subsystem_emotion import execute_subsystem_emotion
from pm.subsystems.subsystem_id import execute_subsystem_id
from pm.subsystems.subsystem_superego import execute_subsystem_superego
from pm.system_utils import get_recent_messages_block, add_cognitive_event


def generate_sensation_evaluation():
    ctx = get_recent_messages_block(12)
    lm = get_recent_messages_block(1)
    #thought_id = execute_subsystem_id(ctx, lm)
    thought_emotion = execute_subsystem_emotion(ctx, lm)
    #thought_superego = execute_subsystem_superego(ctx, lm)

    thought = thought_emotion #thought_id + "\n" + thought_emotion + "\n" + thought_superego
    controller.cache_emotion = thought

    event = Event(
        source=f"{companion_name}",
        content=thought,
        embedding=controller.get_embedding(thought),
        token=get_token(thought),
        timestamp=controller.get_timestamp(),
        interlocus=-2
    )
    add_cognitive_event(event)
    return event
