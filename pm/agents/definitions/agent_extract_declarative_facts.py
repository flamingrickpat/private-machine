from typing import List, Type
from pydantic import BaseModel

from pm.agents.agent_base import (
    CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System,
    Message, BaseAgent, Assistant, ExampleBegin
)

class ExtractDeclarativeFacts(BaseAgent):
    name = "ExtractDeclarativeFacts"

    # system prompts
    def get_system_prompts(self) -> List[str]:
        return [
            "You are an expert at extracting atomic, source-grounded facts from dialogue.",
            (
                "Task: Read the dialogue (and optional context) and extract ONLY facts explicitly stated by a speaker "
                "or agreed upon by the participants. Be neutral, concise, and avoid speculation. "
                "Prefer concrete, verifiable items (who/what/when/where/decisions/requests/promises/configs). "
                "Represent feelings as facts only when attributed (e.g., 'Anna says she feels unsupported'). "
                "Normalize names and entities consistently. If dates/times are relative, keep them as stated. "
                "Do not invent details or compress separate facts into one. "
                "Output format: a markdown bullet list where each line starts with '- ' (dash + space). No preamble."
            ),
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "You rate extracted fact lists. Criteria: "
            "(1) Every item is an atomic fact grounded in the dialogue; "
            "(2) No speculation or synthesis beyond stated content; "
            "(3) Clear, concise wording; "
            "(4) Consistent naming; "
            "(5) Output is a bullet list starting with '- '."
        )

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        # ───── Short Example 1: Ethics in Healthcare (concise) ─────
        text_s1 = """
        User: I’d like your thoughts on the ethics of AI assistance in healthcare applications.
        AI: Ethical AI in healthcare involves fairness, transparency, patient privacy, and accountability.
        User: How can we ensure fairness when algorithms influence treatment recommendations?
        AI: Use diverse training data representative of patient demographics and continuously audit model outcomes.
        User: For transparency, should we implement explainable AI techniques?
        AI: Yes, techniques like LIME or SHAP can provide feature-level explanations.
        User: How do we protect patient privacy while using their data to train models?
        AI: Apply de-identification, encryption in transit and at rest, and strict access controls with audit logging.
        User: Let’s document these ethical guidelines in our project spec.
        """
        facts_s1 = """\
- the user asks for ethical guidance for ai assistance in healthcare
- the ai lists four principles: fairness, transparency, patient privacy, accountability
- the ai recommends diverse training data and outcome audits to support fairness
- the ai recommends explainable ai methods such as lime or shap for transparency
- the ai recommends de-identification, encryption, and strict access controls with audit logs for privacy
- the user decides to document the ethical guidelines in the project specification
"""

        # ───── Short Example 2: Scheduling (concise) ─────
        text_s2 = """
        User: Can you remind me about each milestone on Mondays at 9 AM?
        AI: I will schedule milestone reminders for Mondays at 9 AM.
        User: Also send me a weekly summary on Fridays at 3 PM.
        AI: A weekly summary will go out Fridays at 3 PM.
        User: If a critical alert fires, notify me immediately by email.
        AI: I will notify you immediately via email for critical alerts.
        """
        facts_s2 = """\
- the user requests milestone reminders on mondays at 9 am
- the ai will schedule milestone reminders on mondays at 9 am
- the user requests a weekly summary on fridays at 3 pm
- the ai will send a weekly summary on fridays at 3 pm
- the user requests immediate email notifications for critical alerts
- the ai will send immediate email notifications for critical alerts
"""

        # ───── Long Example 1: GCP timeline & milestones (matches your style) ─────
        text_l1 = """
        User: Thanks for all the help, Chatty. What are the next steps summary?
        Chatty: Summarizing: GCP stack chosen, UI with React/Material UI/Recharts, ethical guidelines outlined, monitoring strategies set.
        User: Exactly. Now, can you outline the milestones for the next eight weeks?
        Chatty: Week 1: Provision GCP environment, configure IAM and basic networking.
                Week 2: Set up Pub/Sub topics and Cloud Functions.
                Week 3: Design BigQuery schemas and ETL pipelines.
                Week 4: Scaffold React frontend with Material UI components.
                Week 5: Integrate frontend with backend via Cloud Run endpoints.
                Week 6: Develop alerting and notification system using Cloud Monitoring.
                Week 7: Conduct performance/load testing and optimize.
                Week 8: Finalize documentation, ethics review, and plan rollout.
        User: Perfect timeline. Please store these milestones and remind me at each week’s start.
        Chatty: Milestones stored. I’ll notify you weekly.
        """
        facts_l1 = """\
- the user requests information about project next steps from chatty
- chatty states the stack is on gcp and mentions ui with react material ui and recharts
- chatty states ethical guidelines are outlined and monitoring strategies are set
- the user asks for an eight-week milestone plan
- chatty outlines a week-by-week plan from week 1 to week 8
- week 1 includes gcp environment provisioning iam and networking
- week 2 includes pub sub topics and cloud functions
- week 3 includes bigquery schemas and etl pipelines
- week 4 includes scaffolding a react frontend with material ui components
- week 5 includes integrating frontend with backend via cloud run
- week 6 includes alerting and notifications using cloud monitoring
- week 7 includes performance and load testing with optimization
- week 8 includes final documentation ethics review and rollout planning
- the user asks chatty to store milestones and send weekly reminders
- chatty confirms milestones are stored and weekly notifications will be sent
"""

        # ───── Long Example 2: Personal conflict (attributed facts) ─────
        text_l2 = """
        Context: A quiet café in the afternoon.
        Anna: Mark, I need to be honest. I’ve been struggling with the project at work.
        Anna: I feel unsupported by you as my team lead and overlooked in meetings.
        Mark: I didn’t realize you felt that way. I thought independence would be empowering.
        Anna: I appreciate independence, but I need recognition for my contributions.
        Mark: I’m sorry. I will acknowledge your work openly going forward.
        Anna: Thank you. I was worried honesty might harm our trust.
        Mark: I value our friendship and will work on this with you.
        """
        facts_l2 = """\
- the setting is a quiet cafe in the afternoon
- anna says she is struggling with the project at work
- anna says she feels unsupported by mark as her team lead
- anna says she feels overlooked in meetings
- mark says he believed giving independence would be empowering
- anna says she appreciates independence but needs recognition
- mark apologizes to anna
- mark states he will acknowledge anna's work openly going forward
- anna says she was worried honesty might harm their trust
- mark states he values their friendship and will work on the issue with anna
"""

        # ───── Short Example 3: Cloud migration vs design review (concise) ─────
        text_s3 = """
        Team A: We’ll use AWS as the primary provider and roll out in phases starting with internal tools.
        Team A: Identity management and data encryption are our main security concerns.
        Team A: We aim to complete migration in six months with check-ins every two weeks.
        Team B: From user feedback, we’ll simplify navigation, improve accessibility, and refine color contrast.
        Team B: We’ll test with a broader user base next sprint and target final design approval by quarter end.
        """
        facts_s3 = """\
- team a selects aws as the primary provider
- team a plans a phased rollout starting with internal tools
- team a identifies identity management and data encryption as main security concerns
- team a targets completion in six months with biweekly check-ins
- team b decides to simplify navigation
- team b decides to improve accessibility
- team b decides to refine color contrast
- team b will test with a broader user base in the next sprint
- team b targets final design approval by the end of the quarter
"""

        # ───── Long Example 3: Mixed ops & reminders (richer schedule) ─────
        text_l3 = """
        User: Please schedule a design review on Wednesday at 2 PM with Alex and Priya.
        AI: Design review scheduled Wednesday 2 PM with Alex and Priya.
        User: Add a follow-up bug triage Thursday at 11 AM.
        AI: Bug triage set for Thursday at 11 AM.
        User: For the production deploy, we lock the main branch on Friday morning.
        AI: Understood. Main branch will be locked Friday morning.
        User: Also, send me a reminder Friday at 9 AM to confirm the deployment checklist.
        AI: Reminder set for Friday 9 AM to confirm the deployment checklist.
        """
        facts_l3 = """\
- the user requests a design review on wednesday at 2 pm with alex and priya
- the ai schedules a design review on wednesday at 2 pm with alex and priya
- the user requests a bug triage on thursday at 11 am
- the ai schedules a bug triage on thursday at 11 am
- the user states the main branch will be locked on friday morning for production deploy
- the ai acknowledges the main branch lock on friday morning
- the user requests a reminder on friday at 9 am to confirm the deployment checklist
- the ai sets a reminder on friday at 9 am to confirm the deployment checklist
"""

        return [
            ("user", text_s1),
            ("assistant", facts_s1),
            ("user", text_s2),
            ("assistant", facts_s2),
            ("user", text_l1),
            ("assistant", facts_l1),
            ("user", text_l2),
            ("assistant", facts_l2),
            ("user", text_s3),
            ("assistant", facts_s3),
            ("user", text_l3),
            ("assistant", facts_l3),
        ]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input.get("content", "")
        context = self.input.get("context", "")
        ops: List[PromptOp] = []

        for s in self.get_system_prompts():
            ops.append(System(s))
        for role, text in self.get_default_few_shots():
            if role == "user":
                ops.append(User(text))
            elif role == "assistant":
                ops.append(Assistant(text))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Context: {context}\n"
            f"{dialog}\n\n"
            "Extract only atomic, explicitly stated facts as a markdown bullet list. "
            "Each line must start with '- ' and contain exactly one fact."
        ))
        ops.append(CompletionText(target_key="facts"))
        ops.append(ExampleEnd())

        return ops
