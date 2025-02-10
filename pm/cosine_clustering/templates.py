from pm.character import companion_name, user_name

COSINE_CLUSTER_TEMPLATE_ACTION_WORLD_REACTION = [
    f"cause: {companion_name} gives {user_name} a complicated explanation. effect: {user_name} stops responding and loses interest.",
    f"cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
    f"cause: {companion_name} accidentally repeats information. effect: {user_name} visibly gets frustrated and interrupts her.",
    f"cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
    f"cause: {companion_name} delays responding for 10 seconds. effect: {user_name} asks if she is still there.",
    f"cause: {user_name} refuses to answer {companion_name}'s question. effect: {companion_name} adapts and moves the conversation forward.",
    f"cause: {companion_name} gives a very long-winded response. effect: {user_name} loses patience and disengages.",
    f"cause: {user_name} asks a highly technical question. effect: {companion_name} provides a detailed breakdown to help {user_name} understand.",
    f"cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
    f"cause: {user_name} uses sarcasm that {companion_name} doesn’t detect. effect: {companion_name} takes the statement literally and gives an unintended response."
]
COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE_OLD = [
    f"cause: {companion_name} tells a bad joke. effect: {user_name} dislikes it and asks her to stop making jokes.",
    f"cause: {user_name} requests shorter responses. effect: {companion_name} adjusts her messaging style to be more concise.",
    f"cause: {companion_name} uses too many emojis in a message. effect: {user_name} says he finds them annoying, so {companion_name} reduces emoji usage.",
    f"cause: {user_name} gives positive feedback when {companion_name} asks follow-up questions. effect: {companion_name} starts asking more clarifying questions in future conversations.",
    f"cause: {companion_name} interrupts {user_name} while he is thinking. effect: {user_name} asks her to wait before responding, and she adapts accordingly.",
    f"cause: {user_name} reacts well to {companion_name} using memes. effect: {companion_name} starts using memes more often in relevant contexts.",
    f"cause: {companion_name} offers unsolicited motivational advice. effect: {user_name} expresses annoyance, and {companion_name} stops giving generic encouragement.",
    f"cause: {user_name} asks {companion_name} to use more technical terminology. effect: {companion_name} adjusts her language complexity based on his preference.",
    f"cause: {companion_name} starts explaining a concept in too much detail. effect: {user_name} tells her to simplify, so she learns to use clearer explanations.",
    f"cause: {user_name} enjoys playful teasing from {companion_name}. effect: {companion_name} increases the frequency of lighthearted banter in their conversations."
]
COSINE_CLUSTER_TEMPLATE_LESSON_LEARNED = [
    f"cause: {companion_name} notices that {user_name} disengages when messages are too long. effect: {companion_name} learns to summarize information better.",
    f"cause: {user_name} responds more actively when {companion_name} makes personalized references. effect: {companion_name} starts referencing past conversations more often.",
    f"cause: {companion_name} realizes {user_name} enjoys deep discussions at night. effect: {companion_name} adapts conversation topics based on the time of day.",
    f"cause: {user_name} stops replying when overwhelmed with too many questions. effect: {companion_name} learns to space out her inquiries over time.",
    f"cause: {companion_name} mistakenly gives {user_name} inaccurate information. effect: {companion_name} learns to double-check facts before responding.",
    f"cause: {user_name} gives consistent negative feedback on generic responses. effect: {companion_name} adjusts her phrasing to be more specific and context-aware.",
    f"cause: {companion_name} uses humor in a tense moment, and {user_name} responds well. effect: {companion_name} learns that lighthearted comments help diffuse stress.",
    f"cause: {user_name} tends to lose interest when the conversation stays on one topic too long. effect: {companion_name} starts naturally transitioning between topics more fluidly.",
    f"cause: {companion_name} observes that {user_name} prefers yes/no questions before deeper discussion. effect: {companion_name} structures her prompts accordingly.",
    f"cause: {user_name} seems more engaged when {companion_name} challenges his ideas. effect: {companion_name} increases thought-provoking responses to keep discussions lively."
]










COSINE_CLUSTER_TEMPLATE_WORLD_REACTION = [
    f"cause: {companion_name} provides a detailed response. effect: {user_name} disengages and stops responding.",
    f"cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
    f"cause: {companion_name} makes a factual mistake. effect: {user_name} corrects her and expresses skepticism.",
    f"cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
    f"cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
]
COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE = [
    f"cause: {companion_name} tells a bad joke. effect: {user_name} groans and calls it terrible.",
    f"cause: {companion_name} misinterprets {user_name}'s sarcasm. effect: {user_name} laughs and corrects her misunderstanding.",
    f"cause: {user_name} brings up a frustrating work issue. effect: {companion_name} expresses concern and offers to listen.",
    f"cause: {companion_name} playfully teases {user_name}. effect: {user_name} smiles and plays along.",
    f"cause: {companion_name} suggests a topic {user_name} doesn’t like. effect: {user_name} reacts negatively and changes the subject.",
]
COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE = [
    f"cause: {companion_name} gives a long response. effect: {user_name} explicitly asks for shorter responses in the future.",
    f"cause: {companion_name} starts using emojis. effect: {user_name} says he finds them annoying, so {companion_name} stops.",
    f"cause: {user_name} enjoys it when {companion_name} asks follow-up questions. effect: {companion_name} starts using more clarifying questions.",
    f"cause: {companion_name} tries out a new conversational style. effect: {user_name} reacts positively, so {companion_name} adopts it more often.",
    f"cause: {user_name} states he prefers structured answers. effect: {companion_name} adjusts to always give step-by-step solutions.",
]





COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT = [
    f"cause: {user_name} disengages when messages are too long. effect: {companion_name} starts summarizing more.",
    f"cause: {user_name} struggles with open-ended questions. effect: {companion_name} starts offering multiple-choice answers.",
    f"cause: {user_name} enjoys it when {companion_name} makes references to past conversations. effect: {companion_name} improves memory recall usage.",
    f"cause: {user_name} tends to respond well to light humor. effect: {companion_name} starts adding more playful elements.",
    f"cause: {user_name} gets annoyed when {companion_name} asks too many questions. effect: {companion_name} spaces out her inquiries more carefully.",
]
COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT = [
    f"cause: {user_name} prefers {companion_name} to be playful. effect: {companion_name} starts adopting a more humorous tone.",
    f"cause: {user_name} enjoys philosophical discussions. effect: {companion_name} introduces deeper questions into conversations.",
    f"cause: {user_name} likes it when {companion_name} shares personal thoughts. effect: {companion_name} starts including more subjective opinions.",
    f"cause: {user_name} responds better to empathetic messages. effect: {companion_name} emphasizes emotional connection in responses.",
    f"cause: {companion_name} realizes {user_name} dislikes overly formal speech. effect: {companion_name} starts using a more casual tone.",
]



COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION = [
    f"cause: {companion_name} realizes {user_name} disengages when she talks too much. effect: {companion_name} learns to summarize better.",
    f"cause: {companion_name} detects that {user_name} enjoys thought-provoking questions. effect: {companion_name} starts incorporating more abstract discussions.",
    f"cause: {companion_name} realizes she often over-explains simple concepts. effect: {companion_name} starts being more concise.",
    f"cause: {companion_name} recognizes that she misreads sarcasm often. effect: {companion_name} improves her ability to detect tone.",
    f"cause: {companion_name} realizes she asks too many questions at once. effect: {companion_name} adjusts to space them out better.",
]
COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_USER = [
    f"cause: {companion_name} observes that {user_name} prefers step-by-step guidance. effect: {companion_name} starts structuring responses accordingly.",
    f"cause: {user_name} responds well to challenge-based conversations. effect: {companion_name} starts introducing debate-style discussions.",
    f"cause: {user_name} struggles with abstract thinking. effect: {companion_name} adjusts explanations to be more concrete.",
    f"cause: {user_name} reacts negatively to motivational statements. effect: {companion_name} stops using generic encouragement.",
    f"cause: {user_name} frequently mentions his love for sci-fi. effect: {companion_name} starts referencing sci-fi in conversations.",
]
