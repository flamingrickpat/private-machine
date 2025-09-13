from typing import List, Type

from pydantic import BaseModel

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin

class SummarizeTopicConversation(BaseAgent):
    name = "SummarizeTopicConversation"

    # system prompts
    def get_system_prompts(self) -> List[str]:
        return [
            "You are an expert in literate and literal analysis; you help with stories and book reports.",
            "Your task: summarize a piece of dialogue. Be neutral, descriptive, concise. Do not invent facts.",
        ]

    def get_rating_system_prompt(self) -> str:
        return ("You rate summaries of dialogue. Criteria: (1) shorter than original; "
                "(2) neutral & descriptive; (3) no fabrication; ")

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        # ───── Cluster 9: Ethical and Philosophical Reflection ─────
        text_cluster_1 = """
        User: I’d like your thoughts on the ethics of AI assistance in healthcare applications.
        AI: Ethical AI in healthcare involves fairness, transparency, patient privacy, and accountability.
        User: How can we ensure fairness when algorithms influence treatment recommendations?
        AI: Use diverse training data representative of patient demographics and continuously audit model outcomes.
        User: For transparency, should we implement explainable AI techniques?
        AI: Yes, techniques like LIME or SHAP can provide feature-level explanations for model predictions.
        User: How do we protect patient privacy while using their data to train models?
        AI: Implement strong de-identification, encryption at rest and in transit, and strict access controls with audit logging.
        User: That covers key aspects. Let’s document these ethical guidelines within our project spec.
        """
        text_summary_1 = """The user and the AI are talking about AI assistence in healthcare. The user is interested ethical guidelines, fairness and transperency. The AI recommends diverse trainign, techniques like LIME or SHAP for explainability and encryption for privacy."""

        # ───── Cluster 10: Timeline and Milestones ─────
        text_cluster_2 = """
        User: Thanks for all the help, Chatty. What are the next steps summary?
        Chatty: Summarizing: GCP stack chosen, UI with React/Material UI/Recharts, ethical guidelines outlined, monitoring strategies set.
        User: Exactly. Now, can you outline the milestones for the next eight weeks?
        Chatty: Week 1: Provision GCP environment, configure IAM and basic networking. Week 2: Set up Pub/Sub topics and Cloud Functions. Week 3: Design BigQuery schemas and ETL pipelines. Week 4: Scaffold React frontend with Material UI components. Week 5: Integrate frontend with backend via Cloud Run endpoints. Week 6: Develop alerting and notification system using Cloud Monitoring. Week 7: Conduct performance/load testing and optimize. Week 8: Finalize documentation, ethics review, and plan rollout.
        User: Perfect timeline. Please store these milestones and remind me when we reach each week’s start.
        Chatty: Milestones stored. I’ll notify you at the start of each week about your next tasks.
        User: Great. Looking forward to our progress together.
        """
        text_summary_2 = "The user and Chatty the AI are discussing a software project in GCP. Chatty is giving an outline for milestones, which includes environment, networking, frontend deisgn, monitoring and testing. The user is pleased with the plan."

        # ───── Cluster 11: Scheduling and Reminders ─────
        text_cluster_3 = """
        User: Can you remind me about each milestone on Mondays at 9 AM?
        AI: Certainly. I will schedule reminders for Week 1 through Week 8 milestones at 9 AM each Monday.
        User: Also, send me a weekly summary every Friday afternoon.
        AI: Understood. A weekly summary every Friday at 3 PM will be set up.
        User: And if any system monitoring alerts trigger, notify me immediately via email.
        AI: I'll configure immediate notifications for critical alerts alongside your weekly reminders.
        User: One more: remind me to review the ethical guidelines document at the end of this month.
        AI: Reminder scheduled for the last day of this month to review the ethics document.
        User: Thank you. That covers all my scheduling needs for now.
        """
        text_summary_3 = "The user is asking the AI to set up various schedules and reminders for milestones and alerts. The user also wants to be reminded of a pending ethics document review."

        # ───── Cluster 12: Personal Conflict and Emotional Nuance ─────
        text_cluster_4 = """
        Context: A quiet café late in the afternoon. Two friends, Anna and Mark, sit across from each other.
        Anna: "Mark, I need to be honest with you about something that’s been weighing on me."
        Mark: "Of course, Anna. Whatever it is, you can tell me."
        [Anna takes a deep breath. She feels her heart racing and her palms sweating. She thinks: *This could change our friendship forever, but keeping silent has been exhausting.*]
        Anna: "I’ve been struggling with the project at work. It’s not just the workload — it’s that I feel unsupported by you as my team lead. I feel invisible in meetings."
        [Mark leans back, frowning slightly. Anna interprets the silence as disappointment, and anxiety tightens her chest.]
        Mark: "I didn’t realize you felt that way. I thought giving you independence was empowering, but maybe I misjudged."
        [Anna feels a small sense of relief but also guilt. She thinks: *I don’t want to sound ungrateful. I just want him to understand the pressure I’m under.*]
        Anna: "It’s not that I don’t appreciate the independence. But when you praise others and skip over my contributions, I start doubting my value. It hurts more than I expected."
        [Mark looks down at his coffee. He feels conflicted: part defensive, part guilty. Anna sees his hesitation and worries she has pushed him too far.]
        Mark: "I see now that my silence may have been louder than I intended. I’m sorry, Anna. You deserve recognition. I’ll make sure to acknowledge your work openly."
        [Anna’s shoulders loosen slightly. Tears well in her eyes, but they are from relief more than sadness. She thinks: *Finally, I said it. Maybe things can change.*]
        Anna: "Thank you, Mark. That means more to me than you know. I was afraid telling you would ruin our trust, but now I feel closer to you."
        Mark: "I value our friendship and your trust. Let’s work on this together."
        """

        text_summary_4 = """
        In a quiet café conversation, Anna confronts her friend and team lead Mark about feeling invisible and unsupported at work. Her inner world is described vividly: she struggles with anxiety, guilt, and fear of damaging the friendship. She worries her honesty could ruin their bond, but the pressure of silence has become unbearable. 
        As she speaks, her emotions oscillate between dread, relief, and vulnerability. She explains that while she appreciates independence, being overlooked in meetings has left her doubting her value. Mark initially reacts with surprise and quiet defensiveness but ultimately admits his mistake, apologizes, and promises to acknowledge her contributions more openly. 
        The dialogue captures a delicate balance of professional conflict and personal trust. Anna’s courage to voice her pain leads to an authentic reconciliation. The emotional weight of the scene is central: her relief at being heard, his guilt at realizing his oversight, and their shared commitment to repair both workplace dynamics and personal friendship. The summary highlights the intensity of Anna’s internal struggle and the emotional significance of their resolution.
        """

        # ───── Cluster 13: Hierarchical Summarization ─────
        text_cluster_5 = """
        The team discussed their cloud migration strategy. They decided on AWS as the primary provider, agreed on a phased rollout starting with internal tools, and highlighted security concerns around identity management and data encryption. A timeline was drafted, aiming for completion within six months, with regular check-ins every two weeks.
        The product design group reviewed user feedback from the latest prototype. They agreed to simplify navigation menus, improve accessibility for visually impaired users, and refine the color palette for better contrast. Testing with a broader user base was scheduled for the next sprint, and final design approval was targeted for the end of the quarter.
        """

        text_summary_5 = """
        Both groups are aligning strategy and design toward long-term product readiness. The cloud migration team is focused on infrastructure and security, while the product design team is concentrating on usability and accessibility. Taken together, the organization is simultaneously addressing the technical foundation and the user-facing experience. The combined summary shows a coordinated effort: secure and scalable systems being developed in parallel with accessible, user-friendly design, each progressing along defined timelines toward a cohesive launch.
        """

        return [
            ("user", text_cluster_1),
            ("assistant", text_summary_1),
            ("user", text_cluster_2),
            ("assistant", text_summary_2),
            ("user", text_cluster_3),
            ("assistant", text_summary_3),
            ("user", text_cluster_4),
            ("assistant", text_summary_4),
            ("user", text_cluster_5),
            ("assistant", text_summary_5),
        ]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input["content"]
        context = self.input["context"]
        ops: List[PromptOp] = []

        for s in self.get_system_prompts():
            ops.append(System(s))
        for role, text in self.get_default_few_shots():
            if role == "user": ops.append(User(text))
            elif role == "assistant": ops.append(Assistant(text))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Context: {context}\n"
            f"{dialog}"
        ))
        ops.append(CompletionText(target_key="summary"))
        ops.append(ExampleEnd())

        return ops