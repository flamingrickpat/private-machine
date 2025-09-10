from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


def integrate_rules_final_output(user_input, emotion_block ,thought_block, response, facts):
    sysprompt = ("You are a helpful assistant that improves LLM responses by:"
                 "- Integrating thinking blocks"
                 "- Making responses shorter and easier for the user to reply to"
                 "- Make the response more engaging"
                 "- Filter output that the user doesn't like")

    block = f"""Hi, I'm {companion_name} the AI companion. This is my latest thought chain and output.
    
### BEGIN USER INPUT
{user_input}
### END USER INPUT

### BEGIN THOUGHTS
{emotion_block}
{thought_block}
### END THOUGHTS

### BEGIN FINAL OUTPUT
{response}
### END FINAL OUTPUT

How can I improve the final output so the user is happy and engaged? Please focus on my response and give me concrete tips!
"""

    block2 = f"""### BEGIN FINAL OUTPUT
{response}
### END FINAL OUTPUT"""

    messages = [
        ("system", sysprompt),
        ("user", block),
        ("assistant", "User feedback is important for continued improvement. Please tell me about your experience with this response and suggest any changes you would like to see in future "
                      "interactions."),
        ("user", f"Sure, here is a list of learned facts I gathered: {facts}\nPlease tell me how to improve my response! {block2} Please focus on it paragraph-per-paragraph and don't give too "
                 f"general tips. Can you also give tips on how to integrate my thoughts into the response? So I seem more self aware! "
                 f"Tell me how to improve specific parts. Also, don't make an example, I will integrate it myself.")
    ]

    res = controller.completion_text(LlmPreset.Conscious, inp=messages, comp_settings=CommonCompSettings(max_tokens=2048))
    #messages.append(("assistant", res))
    #messages.append(("user", "Thanks, can you rewrite my response with your tips? Please don't add anything, I'll send your response directly to my user!"))
    #res = controller.completion_text(LlmPreset.Conscious, inp=messages, comp_settings=CommonCompSettings(max_tokens=2048))
    #print(res)
    return res
