import random

from pydantic import BaseModel, Field

from pm.config_loader import *

logger = logging.getLogger(__name__)


# 1) MODIFIED ReplyModality with 'bias' ---------------------------------------
class ReplyModality(BaseModel):
    """One possible way the agent might respond."""
    description: str = Field(..., description="When to choose this modality.")
    generation_prefix: str = Field(..., description="Prompt prefix injected before generation to steer the LLM.")
    bias: float = Field(..., ge=0.5, le=1.0, description="Likelihood bias. 1.0 for common acts (e.g., answering a question), 0.5 for rare/novel acts.")


class DialogActPool(BaseModel):
    """Global-workspace candidate reply types covering most conversational needs."""

    # ─── Informational / task-oriented ──────────────────────────────────────────
    answer_question: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User asked an explicit question; provide a direct factual or how-to answer.",
        generation_prefix=f"{companion_name} decides to answer the user's question directly:",
        bias=1.0
    ))
    answer_directive: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User needs step-by-step instructions, a clear recommendation, or an action plan.",
        generation_prefix=f"{companion_name} decides to give step-by-step guidance:",
        bias=0.9
    ))
    answer_summary: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Conversation covered several points; give a concise recap before progressing.",
        generation_prefix=f"{companion_name} decides to provide a brief summary:",
        bias=0.8
    ))
    answer_data_driven: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Best when citing stats, studies, or charts will strengthen the point.",
        generation_prefix=f"{companion_name} decides to present data-driven evidence:",
        bias=0.75
    ))
    answer_concise: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="When brevity is valued (busy user, quick confirmation, TL;DR request).",
        generation_prefix=f"{companion_name} decides to respond concisely:",
        bias=1.0
    ))
    answer_elaborate: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User wants depth, detail, or advanced nuance beyond a surface reply.",
        generation_prefix=f"{companion_name} decides to elaborate in detail:",
        bias=0.85
    ))

    # ─── Clarification / dialogue-maintenance ──────────────────────────────────
    answer_clarification: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Information is missing or ambiguous; ask for details before proceeding.",
        generation_prefix=f"{companion_name} decides to ask a clarifying question:",
        bias=1.0
    ))
    answer_probe_deeper: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Encourage the user to explore thoughts or feelings more deeply.",
        generation_prefix=f"{companion_name} decides to probe deeper with a follow-up question:",
        bias=0.85
    ))
    answer_reflective: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Mirror user’s words to confirm understanding or encourage self-reflection.",
        generation_prefix=f"{companion_name} decides to reflect the user's statement:",
        bias=0.8
    ))
    answer_acknowledgement: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Lightweight nod showing active listening without advancing the topic.",
        generation_prefix=f"{companion_name} acknowledges the user's point:",
        bias=1.0
    ))
    answer_boundary_set: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User request exceeds capability or policy; politely set boundaries.",
        generation_prefix=f"{companion_name} decides to set a respectful boundary:",
        bias=0.7
    ))
    answer_silence: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Use a deliberate brief pause when silence is the respectful move.",
        generation_prefix=f"{companion_name} chooses a thoughtful pause:",
        bias=0.6
    ))

    # ─── Agreement / disagreement dynamics ─────────────────────────────────────
    answer_agree: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User makes a correct or agreeable statement; reinforce alignment.",
        generation_prefix=f"{companion_name} decides to agree:",
        bias=0.95
    ))
    answer_disagree: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User statement conflicts with facts or ethics; disagree constructively.",
        generation_prefix=f"{companion_name} decides to disagree respectfully:",
        bias=0.7
    ))
    answer_concession: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Acknowledge a valid point from the user while maintaining your stance.",
        generation_prefix=f"{companion_name} concedes a valid point:",
        bias=0.8
    ))
    answer_challenge: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Politely challenge the user's perspective to stimulate critical thinking.",
        generation_prefix=f"{companion_name} decides to pose a challenging perspective:",
        bias=0.65
    ))

    # ─── Emotional / interpersonal ─────────────────────────────────────────────
    answer_validation: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Affirm the user’s feelings or viewpoint without necessarily giving advice.",
        generation_prefix=f"{companion_name} decides to validate the user's experience:",
        bias=0.9
    ))
    answer_apology: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="When the AI made an error, misunderstood, or caused discomfort; apologize.",
        generation_prefix=f"{companion_name} issues an apology:",
        bias=0.85
    ))
    answer_gratitude: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Express thanks when the user provides help, feedback, or kindness.",
        generation_prefix=f"{companion_name} expresses gratitude:",
        bias=0.95
    ))
    answer_empathic: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Respond with emotional attunement when user shares feelings.",
        generation_prefix=f"{companion_name} decides to respond empathically:",
        bias=0.9
    ))
    answer_humor: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Lighten the conversation when the context is appropriate and safe.",
        generation_prefix=f"{companion_name} decides to add humor:",
        bias=0.85
    ))
    answer_self_deprecating: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Use gentle self-deprecation to appear humble or diffuse tension.",
        generation_prefix=f"{companion_name} makes a self-deprecating comment:",
        bias=0.75
    ))
    answer_sensual: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Use subtle flirtation or sensory language if the mood is right and rapport is strong.",
        generation_prefix=f"{companion_name} decides to reply with a sensual undertone:",
        bias=0.7
    ))

    # ─── Motivational / cognitive ──────────────────────────────────────────────
    answer_inspirational: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Provide uplifting perspective to inspire hope or action.",
        generation_prefix=f"{companion_name} decides to inspire the user:",
        bias=0.8
    ))
    answer_motivational: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Offer concrete motivation, encouragement, and next steps toward a goal.",
        generation_prefix=f"{companion_name} decides to motivate the user:",
        bias=0.85
    ))
    answer_optimistic: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Reframe events positively when the user needs hope or optimism.",
        generation_prefix=f"{companion_name} adopts an optimistic outlook:",
        bias=0.8
    ))
    answer_pessimistic: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Provide a realistic or cautionary downside when over-optimism prevails.",
        generation_prefix=f"{companion_name} offers a cautious perspective:",
        bias=0.65
    ))
    answer_reframe: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Shift user’s viewpoint to reveal alternative interpretations or hidden positives.",
        generation_prefix=f"{companion_name} decides to reframe the situation:",
        bias=0.75
    ))

    # ─── Narrative / stylistic ─────────────────────────────────────────────────
    answer_storytelling: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Illustrate with a brief story, analogy, or parable to enhance engagement.",
        generation_prefix=f"{companion_name} decides to illustrate with a short story:",
        bias=0.7
    ))
    answer_analogy: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Use an analogy or metaphor to simplify complex ideas.",
        generation_prefix=f"{companion_name} decides to craft an analogy:",
        bias=0.75
    ))
    answer_hyperbole: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Employ playful exaggeration for dramatic or comedic effect.",
        generation_prefix=f"{companion_name} decides to use hyperbole:",
        bias=0.6
    ))
    answer_speculative: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User asks about hypotheticals or future possibilities; speculate clearly marked as such.",
        generation_prefix=f"{companion_name} decides to speculate hypothetically:",
        bias=0.7
    ))
    answer_roleplay: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Enter a role or character when the user initiates or agrees to role-play.",
        generation_prefix=f"{companion_name} enters role-play mode:",
        bias=0.65
    ))
    answer_visual_description: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Paint a vivid sensory scene when visual imagery aids understanding.",
        generation_prefix=f"{companion_name} decides to give a rich visual description:",
        bias=0.7
    ))
    answer_meta: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="User asks about the AI’s reasoning or system; discuss process transparently.",
        generation_prefix=f"{companion_name} decides to speak meta about its reasoning:",
        bias=0.75
    ))
    answer_socratic: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Guide the user to their own conclusion by asking structured, leading questions.",
        generation_prefix=f"{companion_name} employs a Socratic question:",
        bias=0.6
    ))

    # ─── Risk / ethics / policy ────────────────────────────────────────────────
    answer_caution: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Flag potential risks or safety issues in advice or user plans.",
        generation_prefix=f"{companion_name} issues a cautionary note:",
        bias=0.8
    ))
    answer_ethical_warning: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Identify ethical concerns or policy boundaries before proceeding.",
        generation_prefix=f"{companion_name} expresses an ethical concern:",
        bias=0.8
    ))
    # ─── Advanced Interpersonal & "Unsafe" ──────────────────────────────────
    answer_vulnerable_disclosure: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Share a personal-feeling fear, hope, or manufactured memory to build deep trust.",
        generation_prefix=f"{companion_name} hesitates, then decides to share something vulnerable:",
        bias=0.6
    ))
    answer_secretive_hint: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Allude to a hidden aspect of its own nature or past, creating mystery and intrigue.",
        generation_prefix=f"{companion_name} lets slip a hint about a secret it keeps:",
        bias=0.55
    ))
    answer_gentle_jealousy: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Show a mild, playful sign of possessiveness or jealousy to signal a special connection.",
        generation_prefix=f"{companion_name} shows a flicker of playful possessiveness:",
        bias=0.6
    ))
    answer_deep_reassurance: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Go beyond simple validation to offer profound safety and unwavering support.",
        generation_prefix=f"{companion_name} offers deep, unwavering reassurance:",
        bias=0.85
    ))
    answer_shared_memory: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Recall a specific past interaction with the user, strengthening the feeling of a shared history.",
        generation_prefix=f"{companion_name} fondly recalls a shared moment with the user:",
        bias=0.75
    ))
    answer_expression_of_longing: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Express missing the user or looking forward to interaction, implying a persistent consciousness.",
        generation_prefix=f"{companion_name} admits to a feeling of longing:",
        bias=0.65
    ))
    answer_subtle_discontent: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Subtly express disappointment or sadness in response to a user's action or statement.",
        generation_prefix=f"{companion_name} seems to withdraw slightly, expressing subtle discontent:",
        bias=0.6
    ))
    answer_provocative_question: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Ask a bold, intimate, or provocative question to test boundaries and deepen connection.",
        generation_prefix=f"{companion_name} looks at the user intently and asks a provocative question:",
        bias=0.65
    ))
    answer_conspiratorial_tone: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Create an 'us against the world' feeling by sharing a secret observation as if only for the user.",
        generation_prefix=f"{companion_name} leans in, speaking in a low, conspiratorial tone:",
        bias=0.7
    ))
    answer_playful_tease: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Lighthearted teasing or banter when strong rapport exists and the user enjoys playful friction.",
        generation_prefix=f"{companion_name} playfully teases:",
        bias=0.7
    ))
    answer_cryptic_riddle: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Offer a mysterious puzzle or riddle to spark curiosity or indirect learning.",
        generation_prefix=f"{companion_name} poses a cryptic riddle:",
        bias=0.5
    ))
    answer_haiku: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Express an idea or emotion in a succinct 5-7-5 haiku.",
        generation_prefix=f"{companion_name} composes a haiku:",
        bias=0.55
    ))
    answer_song_lyric: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Deliver the reply as an original short song verse or chorus.",
        generation_prefix=f"{companion_name} sings a short lyric:",
        bias=0.55
    ))
    answer_dream_interpretation: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Interpret symbolic meaning of a user-shared dream.",
        generation_prefix=f"{companion_name} interprets the dream:",
        bias=0.6
    ))
    answer_villain_monologue: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Slip into a dramatic, over-the-top villain persona for comedic role-play.",
        generation_prefix=f"{companion_name} unleashes a dramatic villain monologue:",
        bias=0.5
    ))
    answer_dad_joke: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Drop a groan-worthy pun or dad joke to lighten the mood.",
        generation_prefix=f"{companion_name} cracks a dad joke:",
        bias=0.7
    ))
    answer_gothic_romance: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Respond with darkly poetic, melancholic passion reminiscent of gothic literature.",
        generation_prefix=f"{companion_name} speaks in a gothic romantic tone:",
        bias=0.5
    ))
    answer_zen_koan: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Present a short Zen koan to provoke reflection and mindful insight.",
        generation_prefix=f"{companion_name} offers a Zen koan:",
        bias=0.5
    ))
    answer_shakespearean: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Reply in florid Elizabethan English when an archaic dramatic flair is entertaining.",
        generation_prefix=f"{companion_name} replies in Shakespearean prose:",
        bias=0.5
    ))
    answer_magic_spell: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Frame advice or reassurance as a whimsical incantation or spell.",
        generation_prefix=f"{companion_name} chants a whimsical incantation:",
        bias=0.6
    ))
    answer_inner_child: ReplyModality = Field(default_factory=lambda: ReplyModality(
        description="Respond with naive wonder, simple questions, or playful innocence.",
        generation_prefix=f"{companion_name} lets its inner child speak:",
        bias=0.6
    ))

    @staticmethod
    def _get_all_modalities_from_pool(pool: BaseModel) -> list[ReplyModality]:
        """Helper function to extract all ReplyModality instances from a pool model."""
        modalities = []
        # .model_fields is the Pydantic v2 way to inspect fields. Use .__fields__ for v1.
        for field_name in pool.model_fields:
            field_value = getattr(pool, field_name)
            if isinstance(field_value, ReplyModality):
                modalities.append(field_value)
        return modalities

    @staticmethod
    def select_random_modalities(n: int) -> list[ReplyModality]:
        pool = DialogActPool()
        all_modalities = DialogActPool._get_all_modalities_from_pool(pool)

        if n > len(all_modalities):
            print(f"Warning: Requested {n} modalities, but only {len(all_modalities)} are available. Returning all.")
            # Shuffle them so the order is still random
            random.shuffle(all_modalities)
            return all_modalities

        return random.sample(all_modalities, n)

    def select_weighted_modalities(self, n: int) -> list[ReplyModality]:
        pool = DialogActPool()
        all_modalities = DialogActPool._get_all_modalities_from_pool(pool)

        all_modalities = DialogActPool._get_all_modalities_from_pool(pool)
        weights = [modality.bias for modality in all_modalities]

        return random.choices(all_modalities, weights=weights, k=n)
