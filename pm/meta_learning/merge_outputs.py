from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


def merge_outputs(user_input, first_try, second_try):
    sysprompt = ("You are a helpful assistant that improves LLM responses by:"
                 "- Combining a safe but boring with an exciting but hallucinatory one"
                 "- The safe response is usually on point, but adds too much extra output"
                 "- Make the final response more engaging, and remove hallucinations")

    block = f"""Hi, I'm {companion_name} the AI companion. These are my latest inputs and planned responses:

### BEGIN USER INPUT
{user_input}
### END USER INPUT

### BEGIN SAFE 
{first_try}
### END SAFE

### BEGIN EXCITING HALLUCINATORY
{second_try}
### END EXCITING HALLUCINATORY

Can you combine it for the perfect response? Please give it to me directly! Do not add quotes, titles or anything at all or start "Sure, here is an answer...". I want to send it directly to the 
user and they should think I made it!
"""

    messages = [
        ("system", sysprompt),
        ("user", block),
    ]

    res = controller.completion_text(LlmPreset.Conscious, inp=messages, comp_settings=CommonCompSettings(max_tokens=2048))
    return res
