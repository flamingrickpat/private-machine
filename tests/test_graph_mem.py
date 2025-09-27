import datetime as dt
import os
import sys

sys.path.append("..")
os.chdir("..")

import threading

from pm.config_loader import model_map
from pm.ghosts.base_ghost import GhostConfig
from pm.graph_memory import *
from pm.llm.llm_proxy import llm_worker
from pm_lida import task_queue, main_llm

ALL = False

# --- NEW TEST SUITE ---
def test_graph_mem():
    logger.info("--- Starting Cognitive Test Scenario ---")

    # Cleanup previous test DB if it exists
    init_db = True

    worker_thread = threading.Thread(target=llm_worker, args=(model_map, task_queue), daemon=True)
    worker_thread.start()

    # 1. Setup: Initialize logging, LLM manager, and the CognitiveMemoryManager.
    # For testing, we use the mock SmallLlmManagerProxy defined earlier.
    ghost = BaseGhost(main_llm, GhostConfig())
    manager = CognitiveMemoryManager(llm_manager=main_llm, ghost=ghost)
    graph_manager = manager

    if init_db:
    # ───── Cluster 1: Greetings and Session Setup ─────
        text_cluster_1 = """
        AI: Hello! I'm Chatty, a conversational AI built to assist you with any questions or tasks you have today. It's nice to meet you.
        User: Hi Chatty! Nice to meet you too. I'm Alex, a software developer working on a new project. How's your performance today?
        AI: I'm running at peak efficiency. Thank you for asking, Alex. I'm ready to help you brainstorm, troubleshoot code, plan events, or just chat.
        User: Wonderful, I could use some brainstorming help. But first, I'd like to share a bit about my current project scope.
        AI: Absolutely, please go ahead and describe your project. I'm listening.
        User: Before we dive into specifics, could you tell me what capabilities you have? For instance, can you recall our past conversations or suggestions?
        AI: I can integrate information you provide, remember details within our session, and use that to answer follow-up questions, diagnose issues, and offer personalized recommendations.
        User: That's impressive. Do you also have knowledge of various programming languages and frameworks?
        AI: Yes, I have familiarity with common languages like Python, JavaScript, Java, C++, and frameworks such as React, Django, and TensorFlow.
        User: Perfect. Let's start with an overview of my project timeline and milestones.
        AI: Sure thing. I'll keep track of our discussion and recall any details when you need them.
        """
        graph_manager.integrate_text(text_cluster_1, group_id="story_test")

    if ALL:
        # ───── Cluster 2: Personal and Professional Background ─────
        text_cluster_2 = """
        User: So, Chatty, here's some background: I graduated with a degree in computer science five years ago. Since then, I've worked mostly in backend systems.
        AI: Understood. Do you have specific industries you've been involved with, such as finance, healthcare, or e-commerce?
        User: Yes, I spent two years at FinSecure, building microservices for secure transactions. After that, I joined HealthNet to develop APIs for patient data management.
        AI: I see. That gives me context on your experience with security and compliance. How has that shaped your current project requirements?
        User: My new undertaking, Project Aurora, requires both robust security measures and high throughput, combining my previous experiences.
        AI: Excellent. Are there particular compliance standards you need to adhere to, like HIPAA or PCI DSS?
        User: Definitely HIPAA compliance is crucial, along with encryption at rest and in transit, and detailed audit logging.
        AI: Got it. Any preferences for technology stacks or cloud providers given your history?
        User: I've primarily used AWS with services like Lambda, API Gateway, DynamoDB, and S3, but I'm open to alternatives if they offer better cost efficiency.
        AI: That helps paint a clear picture of your background and priorities. Ready to proceed to feature brainstorming.
        """
        graph_manager.integrate_text(text_cluster_2, group_id="story_test")

        # ───── Cluster 3: Project Scope and Data Flow ─────
        text_cluster_3 = """
        User: Let's dive into Project Aurora. The goal is to build a real-time analytics dashboard for healthcare metrics.
        AI: What specific metrics are you tracking, and what sources feed into the dashboard?
        User: We're aggregating patient vital stats, appointment rates, and medication adherence from several hospital systems via secure APIs.
        AI: For real-time analytics, do you plan to use streaming platforms like Kafka or AWS Kinesis?
        User: I'm considering Kafka for its mature ecosystem, but Kinesis could simplify integration with AWS Lambda functions.
        AI: Both are viable. Kafka offers greater flexibility, whereas Kinesis integrates seamlessly with other AWS services. Would you like a detailed pros and cons list?
        User: Yes, please list the pros and cons of using Kafka versus Kinesis for our use case.
        AI: Sure. Kafka pros: high throughput, fault tolerance, rich ecosystem; cons: self-managed infrastructure, operational overhead. Kinesis pros: fully managed, auto-scaling; cons: vendor lock-in, cost unpredictability.
        User: That analysis is helpful. Let's consider cost modeling next, including data retention and scaling costs.
        AI: Understood. I'll keep track of cost comparison metrics as we proceed.
        """
        graph_manager.integrate_text(text_cluster_3, group_id="story_test")

        # ───── Cluster 4: Feature Brainstorming ─────
        text_cluster_4 = """
        User: For the dashboard features, I want user authentication, role-based access, customizable views, and alerting.
        AI: Do you need multi-tenant isolation for different hospitals or departments?
        User: Yes, each hospital should only see its own data, but admins can oversee multiple tenants.
        AI: Role-based access can map to hospital IDs and department IDs. Would you like recommendations on implementing RBAC?
        User: Absolutely, outline a simple RBAC model for our microservices architecture.
        AI: We can define roles like admin, clinician, and auditor. Each role maps to permissions such as readMetrics, configureAlerts, and manageUsers.
        User: Should permissions be stored in a central identity service or within each microservice?
        AI: Best practice is to have a centralized identity provider like AWS Cognito or Auth0. Microservices validate JWT tokens issued by the provider.
        User: That makes sense. Let's discuss alert thresholds and notification channels next.
        AI: Sure. What metrics require alerting, and what severity levels are you considering?
        """
        graph_manager.integrate_text(text_cluster_4, group_id="story_test")

        # ───── Cluster 5: Troubleshooting Performance ─────
        text_cluster_5 = """
        User: I'm encountering a latency spike when ingesting data from our Kafka cluster into the aggregation service.
        AI: What is the current throughput, and how many partitions are you using for the topic?
        User: We're pushing fifteen thousand events per second with four partitions. Latency jumps above two seconds intermittently.
        AI: Two seconds is high. Have you monitored consumer lag or GC pauses in your Java application?
        User: The consumer lag rarely exceeds a few hundred messages. GC logs show occasional full GC lasting close to 500 milliseconds.
        AI: Full GC of 500 milliseconds every few seconds could cause spikes. Consider tuning the JVM flags or switching to G1GC.
        User: I’m using CMS by default. Should I configure G1GC with specific heap region sizes?
        AI: Yes. Try `-XX:+UseG1GC` with a `MaxGCPauseMillis` target of `200ms` and adjust `-XX:MaxGCPauseMillis=200` accordingly.
        User: I'll test those settings and increase partitions to six to distribute load more evenly.
        AI: Good plan. Monitor after changes and let me know if latency improves or if new issues arise.
        """
        graph_manager.integrate_text(text_cluster_5, group_id="story_test")

        # ───── Cluster 6: Memory and Preferences Recall ─────
        text_cluster_6 = """
        User: Earlier, I mentioned my AWS preference. Can you recall what AWS services I commonly use?
        AI: You specified AWS Lambda, API Gateway, DynamoDB, S3, and CloudWatch for monitoring.
        User: Correct. Also, I use CloudWatch logs streamed via Kinesis Firehose into S3, then query with Athena.
        AI: Right. Do you plan to automate Athena queries for regular reporting?
        User: Yes, I’ll schedule them daily and send summaries via email.
        AI: And for compliance, you mentioned using AWS Config and strict IAM roles?
        User: Exactly. IAM roles for least privilege and AWS Config rules for automated audits.
        AI: Noted. I'll remember all your AWS preferences, data pipelines, and compliance settings.
        """
        graph_manager.integrate_text(text_cluster_6, group_id="story_test")

        # ───── Cluster 7: Contradictory Update ─────
        text_cluster_7 = """
        User: Actually, we decided to switch from AWS to Google Cloud Platform for cost and performance benefits.
        AI: Understood. Which GCP services will you use instead of AWS Lambda and API Gateway?
        User: We'll use Cloud Functions for serverless, Cloud Run for container endpoints, and Cloud Pub/Sub for messaging.
        AI: And BigQuery in place of Athena, plus Cloud Storage instead of S3?
        User: Yes, exactly. BigQuery for analytics, Cloud Storage for object data, and Stackdriver for monitoring.
        AI: That’s a significant shift. Shall I remove AWS configurations from memory and overwrite with GCP choices?
        User: Yes, overwrite the AWS choices and disregard prior cost estimates.
        AI: Done. I'll remember Cloud Functions, Cloud Run, Pub/Sub, BigQuery, Cloud Storage, and Stackdriver.
        User: Perfect. Moving forward, use GCP service names and pricing for any cost analysis.
        """
        graph_manager.integrate_text(text_cluster_7, group_id="story_test")

        # ───── Cluster 8: UI/UX Preferences ─────
        text_cluster_8 = """
        User: Let's talk about user interface preferences for the dashboard.
        AI: Do you prefer a web-based single-page application or a desktop client?
        User: A web SPA is better for accessibility. I like React for the frontend framework.
        AI: Any CSS framework preferences, like Material UI or Bootstrap?
        User: I prefer Material UI for its design consistency and built-in theming.
        AI: And charting libraries for real-time graphs? D3.js, Chart.js, or Recharts?
        User: Recharts aligns well with React’s component model and provides the interactivity we need.
        AI: Should dark mode be supported by default, with a toggle for users?
        User: Yes, default to system preference and allow manual override for dark/light themes.
        """
        graph_manager.integrate_text(text_cluster_8, group_id="story_test")

        # ───── Cluster 9: Ethical and Philosophical Reflection ─────
        text_cluster_9 = """
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
        graph_manager.integrate_text(text_cluster_9, group_id="story_test")

        # ───── Cluster 10: Timeline and Milestones ─────
        text_cluster_10 = """
        User: Thanks for all the help, Chatty. What are the next steps summary?
        AI: Summarizing: GCP stack chosen, UI with React/Material UI/Recharts, ethical guidelines outlined, monitoring strategies set.
        User: Exactly. Now, can you outline the milestones for the next eight weeks?
        AI: Week 1: Provision GCP environment, configure IAM and basic networking. Week 2: Set up Pub/Sub topics and Cloud Functions. Week 3: Design BigQuery schemas and ETL pipelines. Week 4: Scaffold React frontend with Material UI components. Week 5: Integrate frontend with backend via Cloud Run endpoints. Week 6: Develop alerting and notification system using Cloud Monitoring. Week 7: Conduct performance/load testing and optimize. Week 8: Finalize documentation, ethics review, and plan rollout.
        User: Perfect timeline. Please store these milestones and remind me when we reach each week’s start.
        AI: Milestones stored. I’ll notify you at the start of each week about your next tasks.
        User: Great. Looking forward to our progress together.
        """
        graph_manager.integrate_text(text_cluster_10, group_id="story_test")

        # ───── Cluster 11: Scheduling and Reminders ─────
        text_cluster_11 = """
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
        graph_manager.integrate_text(text_cluster_11, group_id="story_test")

        # ───── Cluster 12: Reflection and Farewell ─────
        text_cluster_12 = """
        AI: It’s been great helping you shape Project Aurora, Alex. Do you have any final questions before we wrap up?
        User: Just one: how would you describe our collaboration so far?
        AI: Our collaboration has been structured, efficient, and adaptive. You provided clear goals, and I responded with detailed analyses, troubleshooting advice, and scheduling support.
        User: I appreciate your thoroughness. I feel confident moving forward.
        AI: I'm glad to hear that. Remember, you can revisit any of these details anytime, and I'll recall them accurately.
        User: Thanks, Chatty. Talk to you tomorrow when the Week 1 milestone reminder arrives.
        AI: I’ll be here and ready. Have a productive day, Alex!
        User: You too. Goodbye for now.
        AI: Goodbye, Alex. Looking forward to our next session.
        """
        graph_manager.integrate_text(text_cluster_12, group_id="story_test")

        print("Extended dialogue story test integration complete.")

    # Assuming `manager` is your GraphMemoryManager instance
    qa_expected = [
        # Cluster 1
        ("What name does the AI introduce itself with?", "Chatty"),
        ("How does the user introduce themself?", "Alex"),
        ("Which capabilities does the AI list when the user asks?", "integrate"),
        ("How does the AI describe its ability to recall past conversation details?", "remember"),
        ("What programming domains does the AI claim familiarity with?", "Python"),
    ]
    if ALL:
        qa_expected.extend([
            # Cluster 2
            ("How many years ago did the user graduate?", "five"),
            ("Which two previous employers does the user mention?", "FinSecure"),
            ("What compliance standard is crucial for the user’s new project?", "HIPAA"),
            ("When asked about cloud preferences, which provider does the user say they’ve primarily used?", "AWS"),
            ("Which services on AWS does the user list as their go-to?", "Lambda"),

            # Cluster 3
            ("What is the name of the user’s project?", "Aurora"),
            ("Which two streaming platforms does the AI compare for real-time analytics?", "Kafka"),
            ("What pros and cons does the AI give for Kafka?", "throughput"),
            ("What pros and cons does the AI give for Kinesis?", "managed"),
            ("What cost-modeling factors does the AI say it will track next?", "retention"),

            # Cluster 4
            ("Name three key dashboard features the user requests.", "authentication"),
            ("What roles are defined in the simple RBAC model?", "admin"),
            ("Which identity provider options does the AI recommend?", "Cognito"),
            ("What notification channels are discussed?", "alerting"),
            ("How does the AI plan to remember the user’s RBAC preferences?", "remember"),

            # Cluster 5
            ("What throughput and partition count does the user report?", "partitions"),
            ("Which garbage collector is the user using by default?", "CMS"),
            ("What JVM flag does the AI suggest to switch to G1GC?", "-XX:+UseG1GC"),
            ("What `MaxGCPauseMillis` target does the AI recommend?", "200ms"),
            ("After the tuning suggestions, what two actions does the user plan to take?", "increase"),

            # Cluster 6
            ("Which AWS service for monitoring does the AI recall?", "CloudWatch"),
            ("How does the user stream CloudWatch logs into S3?", "Kinesis Firehose"),
            ("What tool does the user plan to use to query S3 data?", "Athena"),
            ("What compliance tools does the user mention alongside IAM roles?", "Config"),
            ("What schedule does the user set for Athena queries?", "daily"),

            # Cluster 7
            ("To which cloud provider does the user switch?", "Google"),
            ("Which GCP service replaces AWS Lambda?", "Cloud Functions"),
            ("What does the AI ask before overwriting memory?", "remove"),
            ("What new GCP monitoring service does the user choose?", "Stackdriver"),
            ("After overwriting, what six GCP services does the AI now remember?", "Pub/Sub"),

            # Cluster 8
            ("Which frontend framework does the user prefer?", "React"),
            ("Which component library does the AI note for theming?", "Material UI"),
            ("What charting library does the user select?", "Recharts"),
            ("How should the theme default be determined?", "system"),
            ("How will the AI recall the dark-mode preference?", "default"),

            # Cluster 9
            ("Name two explainable AI techniques the AI mentions.", "LIME"),
            ("Which four ethical principles does the AI list?", "transparency"),
            ("What method does the AI recommend for de-identification?", "de-identification"),
            ("Where does the AI say it will document these guidelines?", "project spec"),

            # Cluster 10
            ("What is the Week 3 milestone?", "schemas"),
            ("Which week is dedicated to performance/load testing?", "load"),
            ("What tasks are planned for Week 5?", "frontend"),
            ("How many total weeks are in the timeline?", "8"),
            ("Which milestone does the AI promise to remind at the start of each week?", "tasks"),

            # Cluster 11
            ("On which day and time should the weekly summary go out?", "Friday"),
            ("When should milestone reminders fire each week?", "Monday"),
            ("What triggers immediate notifications?", "critical"),
            ("When is the ethics-document review scheduled?", "last"),
            ("How many distinct reminder schedules did the user request?", "reminder"),

            # Cluster 12
            ("How does the AI describe the collaboration?", "structured"),
            ("When will Alex next hear from Chatty?", "tomorrow"),
            ("What phrase does the AI use to express readiness for the next session?", "ready"),
            ("What final well-wish does the AI give Alex?", "productive"),
        ])

    cnt_good = 0
    cnt_bad = 0
    for question, expected in qa_expected:
        answer_str = manager.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer_str}")
        if expected.lower() in answer_str.lower():
            cnt_good += 1
            print(f"✅ Found '{expected}' in answer_str.\n")
        else:
            cnt_bad += 1
            print(f"❌ '{expected}' not found in answer_str.\n")

    assert cnt_good > cnt_bad
