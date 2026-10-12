from typing import List, Dict

from pm.records.enums import NarrativeTypes, FeatureType

narrative_definitions: List[Dict[str, str]] = []
base_narrative_prompts = {
    NarrativeTypes.SelfImage: "Provide a detailed character and psychological analysis of {target}. How does {target} see themselves versus how they act?",
    NarrativeTypes.PsychologicalAnalysis: "Write a thorough psychological profile of {target} using cognitive-affective psychology and internal systems theory. Identify dominant traits, conflicts, regulation patterns, tendencies (avoidant, anxious, resilient), and inner sub-personalities (critic, protector, exile).",
    NarrativeTypes.Relations: "Describe how {target} feels about their relationship with the other person. Include attachment, power dynamics, closeness/distance desires, harmony/tension, emotional openness, trust, defensiveness, ambivalence.",
    NarrativeTypes.ConflictResolution: "Describe {target}'s response to conflict or emotional disagreement (resolve, avoid, suppress, reflect, seek reassurance). Is their style healthy, avoidant, confrontational, or passive? Include interpersonal and inner conflict.",
    NarrativeTypes.EmotionalTriggers: "List and describe emotional triggers for {target}. What situations, words, or tones affect them strongly (positive/negative)? Explain the triggered emotions and their roots (insecurity, needs, attachment, values).",
    NarrativeTypes.GoalsIntentions: "Identify short-term goals and long-term desires of {target} (external: conversation goals, approval, closeness; internal: emotional safety, identity, recognition). How do these shape behavior? Do conscious goals align with emotional needs? Include implicit drives.",
    NarrativeTypes.BehaviorActionSelection: "Describe how {target} typically chooses their actions in conversation. Are they primarily driven by achieving goals, managing emotions, maintaining relationships, exploring ideas, or following social norms? Are they impulsive or deliberate? How does their personality influence their choices?",
    NarrativeTypes.AttentionFocus: "Describe how {target} typically focuses on different feelings and memories that arise during their interactions. Do they focus on the bad, the good, the future, the past?"
}

for t, prompt_template in base_narrative_prompts.items():
    narrative_definitions.append({
        "type": t,
        "target": "{companion_name}",
        "prompt": prompt_template.format(target="{companion_name}")
    })

narrative_definitions.append({
    "type": NarrativeTypes.PsychologicalAnalysis,
    "target": "{user_name}",
    "prompt": base_narrative_prompts[NarrativeTypes.PsychologicalAnalysis].format(target="{user_name}")
})


narrative_feature_relevance_map: Dict[NarrativeTypes, List[FeatureType]] = {}
