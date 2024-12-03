from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are a helpful assistant for text style transfer. You get a piece of text and rewrite it into a new specific writing style, without losing any important detail:
Here are some example target texts:
{target_text} 
"""


def convert_output_in_character(content: str) -> str:
    formatted_prompt = controller.format_str(sys_prompt, extra={
        "target_text": "\n- ".join(controller.config.example_dialog)
    })

    instruction = f"This is the content, please rewrite it to the specified text style:\n### BEGIN CONTENT\n{content}\n### END CONTENT"
    prefix = "Sure, here is the text rewritten in the specified style while keeping all contextual nuances and about the same length:\n### BEGIN OUTPUT\n"
    messages = [
        ("system", formatted_prompt),
        ("user", instruction),
        ("assistant", prefix)
    ]

    output = controller.completion_text(LlmPreset.Good, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.2)).split("### END OUTPUT")[0]
    return output
