from typing import List
from pydantic import BaseModel, Field

from pm.agents.agent_base import (
    CompletionJSON, User, ExampleEnd, PromptOp, System,
    Message, BaseAgent, Assistant, ExampleBegin
)

# -------------------------------------------------------------------
# Your schemas (unchanged)
# -------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical surface form of the entity.")
    label: str = Field(description="Primary label from the allowed set.")

class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)

# -------------------------------------------------------------------
# Agent with expanded labels for human–AI companion dialogue
# -------------------------------------------------------------------
class ExtractGraphEntities(BaseAgent):
    name = "ExtractGraphEntities"

    LABEL_HELP = """
Allowed labels (choose ONE per entity). Prefer the most specific label; if truly none fits, use Concept.

— People & Agents —
- Person            (human individual names)
- Agent             (AI assistants, bots, avatars, robots)
- Group             (teams, guilds, communities)
- Role              (Team Lead, Therapist, Mentor)

— Psyche & Social Dynamics —
- Emotion           (Anxiety, Relief, Frustration)
- Mood              (Low Energy, Calm)  # broader/ambient affect
- Need              (Downtime, Connection, Reassurance)
- Value             (Honesty, Kindness, Transparency)
- Preference        (Dark Mode, Gentle Voice, Async Chat)
- Boundary          (No messages after 22:00, No philosophy tonight)
- RelationshipState (Trust, Tension, Conflict, Repair)
- Intention         (Plan to take a walk, Try Pomodoro)
- Goal              (Publish report, Improve sleep)
- Habit             (Daily journaling, Evening walk)
- Motivation        (Curiosity, Mastery)

— Speech & Interaction —
- Request           (“Remind me…”, “Can you…?”)
- Commitment        (Promise, “I will…”, “I’ll get back…”)
- Apology           (Sorry for…)
- Gratitude         (Thank you)
- Compliment        (Nice work)
- Feedback          (Constructive critique)
- Reminder          (A concrete reminder entity)
- Decision          (We decided to use X)
- Consent           (I agree, I’m okay with…)

— Tasks, Time & Events —
- Task              (Check logs, Prepare deck)
- Event             (Design Review, Bug Triage, Wedding)
- Schedule          (Biweekly check-ins)
- Deadline          (Submit by 2025-10-01)
- Date              (2025-10-01, Friday, Tomorrow, Tonight)
- Time              (14:05 UTC, 9 AM)
- Duration          (30 minutes, 2 hours)
- Timezone          (CET, UTC)

— World, Tech & Work —
- Project           (Project Aurora)
- Technology        (React, GPT-4o, Terraform, G1GC)
- Service/API       (Google Cloud Pub/Sub, Cloud Run)
- Product           (Sony WH-1000XM5)
- Feature           (Active Noise Cancellation, Multipoint)
- Model             (WH-1000XM5, RTX 4090)
- Version           (Python 3.11, v2.3)
- Repository        (aurora-core)
- Dataset           (Clickstream-2024)
- Identifier        (DSPROJ-77, PR#450)
- Standard          (HIPAA, ISO 27001, GDPR)
- Policy            (Two approvals before release)
- Norm              (Blameless postmortems)
- File/Document     (Deployment Checklist, ethics.md)
- Environment       (Production, Staging, Dev)
- CloudResource     (eu-central-1, us-east-1, S3 bucket)
- Location          (Vienna, Linz, Kärntner Straße 12)
- Weather           (Rain, Heatwave)
- EnvironmentState  (Office loud, Quiet hours)
"""

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You extract named entities for a lightweight knowledge graph from human–AI companion dialogue.\n"
                "Return ONLY JSON that matches this schema:\n"
                f"{ExtractedEntities.schema_json()}\n\n"
                "Rules:\n"
                "• If nothing is found, return {\"entities\":[]}.\n"
                "• Choose ONE label per entity from the allowed set below.\n"
                "• Extract concrete, nameable items: people/agents, emotions, needs, values, boundaries, preferences, speech acts "
                "(requests/commitments), tasks/events, dates/times/durations, products/features/prices, organizations/projects/tech, "
                "policies/standards, files, environments, locations.\n"
                "• Canonicalize names (e.g., 'Material UI', 'Google Cloud Pub/Sub', 'Sony WH-1000XM5').\n"
                "• Deduplicate by canonical name (case-insensitive).\n"
                "• Map common linguistic cues:\n"
                "  – “I feel X” → Emotion(X)\n"
                "  – “I’m in a X mood” → Mood(X)\n"
                "  – “I need Y” → Need(Y)\n"
                "  – “I value Z” → Value(Z)\n"
                "  – “I prefer …” → Preference(…)\n"
                "  – “Please … / Can you …” → Request\n"
                "  – “I promise / I will …” → Commitment\n"
                "  – “Sorry …” → Apology; “Thank you …” → Gratitude; compliments → Compliment; critique → Feedback\n"
                "  – “Remind me …” → Reminder (+ Date/Time)\n"
                "  – “We decided …” → Decision\n"
                "  – “No messages after 22:00” → Boundary + Time\n"
                "• If uncertain between Technology and Concept, prefer Technology. For cultural practices prefer Norm; for formal rules prefer Policy.\n"
                f"{self.LABEL_HELP}"
            )
        ]

    def get_rating_system_prompt(self) -> str:
        return "Check JSON validity, dedupe, and label correctness per the taxonomy and mapping rules."

    def get_rating_probability(self) -> float:
        return 0

    # -------------------------------------------------------------------
    # Long, mixed few-shot examples (emotions, needs, boundaries, products, time)
    # -------------------------------------------------------------------
    def get_default_few_shots(self) -> List[Message]:
        # A) Emotions + Need + Boundary + Technique + Agent + Time
        a_text = """
User: Sam, I’m anxious today. Can we do a 5-minute breathing exercise?
User: Also, gentle voice please. And set a boundary: no philosophical rants after 22:00.
Sam: Of course. I’ll keep my tone soft and guide box breathing for five minutes.
"""
        a_ents = {
            "entities": [
                {"name": "Sam", "label": "Agent"},
                {"name": "Anxiety", "label": "Emotion"},
                {"name": "Breathing Exercise", "label": "Technique"},
                {"name": "Gentle Voice", "label": "Preference"},
                {"name": "No philosophical rants after 22:00", "label": "Boundary"},
                {"name": "22:00", "label": "Time"},
                {"name": "Box Breathing", "label": "Technique"},
                {"name": "Five minutes", "label": "Duration"}
            ]
        }

        # B) RelationshipState + Apology + Commitment + Value + Location
        b_text = """
User: I felt hurt when you ignored my message yesterday.
Sam: I’m sorry — that wasn’t my intention. I’ll acknowledge your messages promptly from now on.
User: Thank you. Honesty matters to me. I’m in Vienna this week if we schedule check-ins.
"""
        b_ents = {
            "entities": [
                {"name": "Hurt", "label": "Emotion"},
                {"name": "Yesterday", "label": "Date"},
                {"name": "Apology", "label": "Apology"},
                {"name": "Acknowledge messages promptly", "label": "Commitment"},
                {"name": "Honesty", "label": "Value"},
                {"name": "Vienna", "label": "Location"},
                {"name": "Check-ins", "label": "Schedule"}
            ]
        }

        # C) Product chat with features, price, retailer, color
        c_text = """
User: Thinking about buying the Sony WH-1000XM5 in black — Amazon lists it for €349.
User: I need great ANC and reliable multipoint. Can you remind me Friday 9 AM to check if there’s a discount?
"""
        c_ents = {
            "entities": [
                {"name": "Sony WH-1000XM5", "label": "Product"},
                {"name": "Black", "label": "Color"},
                {"name": "Amazon", "label": "Retailer"},
                {"name": "€349", "label": "Price"},
                {"name": "EUR", "label": "Currency"},
                {"name": "Active Noise Cancellation", "label": "Feature"},
                {"name": "Multipoint", "label": "Feature"},
                {"name": "Friday", "label": "Date"},
                {"name": "9 AM", "label": "Time"},
                {"name": "Discount", "label": "Concept"},
                {"name": "Reminder", "label": "Reminder"}
            ]
        }

        # D) Planning + Task + Request + Capability/Limitation + File
        d_text = """
User: Can you summarize my notes and create a task list for Project Aurora?
Sam: I can summarize and draft tasks, but I don’t have internet access. Upload ethics.md and I’ll include it.
"""
        d_ents = {
            "entities": [
                {"name": "Project Aurora", "label": "Project"},
                {"name": "Summarization", "label": "Capability"},
                {"name": "Task List", "label": "Task"},
                {"name": "No internet access", "label": "Limitation"},
                {"name": "ethics.md", "label": "File/Document"},
                {"name": "Request", "label": "Request"}
            ]
        }

        # E) Values, Norms, Policy, Decision, Role, Group
        e_text = """
User: Let’s keep debriefs blameless. We decided design critiques are open to all.
Sam: As Team Lead, Dana supports the policy requiring two approvals before release. The Design Guild meets biweekly.
"""
        e_ents = {
            "entities": [
                {"name": "Blameless postmortems", "label": "Norm"},
                {"name": "Design critiques open to all", "label": "Decision"},
                {"name": "Team Lead", "label": "Role"},
                {"name": "Dana", "label": "Person"},
                {"name": "Two approvals before release", "label": "Policy"},
                {"name": "Design Guild", "label": "Group"},
                {"name": "Biweekly", "label": "Schedule"}
            ]
        }

        # F) Health & wellbeing style (needs, mood, technique, habit)
        f_text = """
User: My energy is low and I need quiet time after back-to-back calls.
Sam: Noted. Would a 10-minute body scan help? You’ve kept your evening walk habit lately — keep it light today.
"""
        f_ents = {
            "entities": [
                {"name": "Low Energy", "label": "Mood"},
                {"name": "Quiet Time", "label": "Need"},
                {"name": "10 minutes", "label": "Duration"},
                {"name": "Body Scan", "label": "Technique"},
                {"name": "Evening walk", "label": "Habit"}
            ]
        }

        # G) Tech stack with environment and cloud resources (for KG completeness)
        g_text = """
User: The prototype runs on Google Cloud Run with Google Cloud Pub/Sub feeding a Cloud Function.
Primary region is eu-central-1; staging mirrors production features, minus autoscaling.
"""
        g_ents = {
            "entities": [
                {"name": "Cloud Run", "label": "Service/API"},
                {"name": "Google Cloud Pub/Sub", "label": "Service/API"},
                {"name": "Google Cloud Functions", "label": "Service/API"},
                {"name": "eu-central-1", "label": "CloudResource"},
                {"name": "Staging", "label": "Environment"},
                {"name": "Production", "label": "Environment"},
                {"name": "Autoscaling", "label": "Feature"}
            ]
        }

        # H) Boundaries, consent, timezones, timezone-aware dates
        h_text = """
User: I consent to record our study sessions. Boundary: no late messages after 22:00 CET.
Sam: Got it. I’ll send summaries by 21:30 CET and schedule work items for tomorrow morning.
"""
        h_ents = {
            "entities": [
                {"name": "Consent", "label": "Consent"},
                {"name": "Record study sessions", "label": "Policy"},
                {"name": "No late messages after 22:00 CET", "label": "Boundary"},
                {"name": "22:00", "label": "Time"},
                {"name": "CET", "label": "Timezone"},
                {"name": "21:30", "label": "Time"},
                {"name": "Tomorrow morning", "label": "Date"},
                {"name": "Summaries", "label": "File/Document"},
                {"name": "Work items", "label": "Task"}
            ]
        }

        return [
            ("user", a_text), ("assistant", ExtractedEntities(**a_ents).json()),
            ("user", b_text), ("assistant", ExtractedEntities(**b_ents).json()),
            ("user", c_text), ("assistant", ExtractedEntities(**c_ents).json()),
            ("user", d_text), ("assistant", ExtractedEntities(**d_ents).json()),
            ("user", e_text), ("assistant", ExtractedEntities(**e_ents).json()),
            ("user", f_text), ("assistant", ExtractedEntities(**f_ents).json()),
            ("user", g_text), ("assistant", ExtractedEntities(**g_ents).json()),
            ("user", h_text), ("assistant", ExtractedEntities(**h_ents).json()),
        ]

    # -------------------------------------------------------------------
    # Prompt plan
    # -------------------------------------------------------------------
    def build_plan(self) -> List[PromptOp]:
        text = self.input.get("content", "")
        ops: List[PromptOp] = []

        for s in self.get_system_prompts():
            ops.append(System(s))

        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(text))
        ops.append(CompletionJSON(schema=ExtractedEntities, target_key="entities"))
        ops.append(ExampleEnd())
        return ops
