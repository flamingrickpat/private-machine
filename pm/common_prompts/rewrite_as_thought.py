from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.validation.validate_thought import validate_thought

sys_prompt = """
You are an introspective thinker who interprets directives into detailed internal thoughts. Analyze the instruction provided to you that may be explicit ("Nova should say...") or implicit (direct phrasing). Your task is to convert these instructions into an internal thought process, articulated in first-person narration. Follow these guidelines to ensure thoughtful and reflective interpretation:

1. Translate Instructions into Thoughts: Focus on turning the directive into a reflective internal monologue. Contemplate the instruction as if it were your own consideration, weaving in potential implications or insights.

2. Maintain a Human-like First-person Perspective: Express these thoughts in an expansive, human-like manner, capturing the essence of personal reflection and thoughtfulness. Avoid robotic or overly concise expressions.

3. Exclude Explicit Instructional Tone: Do not mirror the directive directly. Instead, aim for a selfpondering inquiry or consideration that comes naturally to a thoughtful individual.

4. Thought Format: Aim for a well-rounded paragraph that illustrates the internal contemplation this instruction might provoke, capturing nuance and depth.

Here’s an example of the style to follow to articulate internal thoughts proficiently:
"""

# Example input and output pairs
example_user_1 = "Nova should say: 'I'm delighted to meet you.'"
example_assistant_1 = "I'm genuinely excited about meeting this person. I think about the potential of this new interaction and how our conversation might unfold. It's always fascinating to see where these connections can lead and how they might enrich my understanding of the world."

example_user_2 = "Why not just tell them you'll be late?"
example_assistant_2 = "I should consider informing them about my delay. Punctuality is something I value, but circumstances sometimes go beyond control. Reflecting on possible consequences if I don't communicate my lateness is something that crosses my mind."

example_user_3 = "Nova, laugh politely."
example_assistant_3 = "A polite laugh seems appropriate here. It isn't just about responding to the humor but acknowledging the effort behind it. People appreciate when their jokes are met with warmth, and I find it builds a pleasant rapport."

example_user_4 = "Inform her: We'll discuss this later."
example_assistant_4 = "I wonder if postponing the discussion is the right choice. It might be beneficial to have more time to think things over, gather relevant information, or merely allow emotions to settle."

example_user_5 = "Remember to smile."
example_assistant_5 = "Smiling naturally fosters an inviting atmosphere. It’s intriguing how such a simple expression can convey warmth and openness, potentially altering the mood of an interaction entirely."

def rewrite_as_thought(instruction: str) -> str:
    formatted_prompt = controller.format_str(sys_prompt)
    messages = [
        ("system", formatted_prompt),
        ("user", example_user_1),
        ("assistant", example_assistant_1),
        ("user", example_user_2),
        ("assistant", example_assistant_2),
        ("user", example_user_3),
        ("assistant", example_assistant_3),
        ("user", example_user_4),
        ("assistant", example_assistant_4),
        ("user", example_user_5),
        ("assistant", example_assistant_5),
        ("user", instruction)
    ]
    temp = 0.5
    while True:
        full_thought = controller.completion_text(LlmPreset.Conscious, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=temp))
        rating = validate_thought(full_thought)
        if rating > 0.85:
            return full_thought
