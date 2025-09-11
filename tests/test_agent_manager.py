import datetime
import sys
import time

sys.path.append("..")

from pm.agents.definitions.agent_summarize_topic_conversation import SummarizeTopicConversation, SummaryCategoryBaseModel
from pm_lida import main_llm, start_llm_thread

start_llm_thread()

context = "A group of university students — Anna, Ben, Clara, Daniel, Emma, Felix, Grace, and Hannah — share an apartment. They often chat in the kitchen while managing studies, chores, and social life."

example = [
    """Anna: Hey Ben, the sink is overflowing again.  
Ben: I thought it was your turn to wash dishes.  
Anna: No, we agreed I’d handle the trash this week.  
Ben: Oh right, but I had a late lab yesterday.  
Anna: I get it, just please rinse your stuff.  
Ben: Fair enough, I’ll do them now.  """,

    """Clara: I’m seriously panicking about tomorrow’s math exam.  
Daniel: You’ve been studying all week though.  
Clara: Yeah, but every practice test I fail the integrals.  
Daniel: Remember the tutoring session? You got most of them right.  
Clara: Still, I feel like my brain shuts down under pressure.  
""",


    """Emma: What’s everyone doing on Saturday night?  
Felix: I’m going to that indie concert downtown.  
Grace: Oh, I wanted to go too but tickets are sold out.  
Hannah: We could just have a movie marathon here.  
Emma: That sounds fun, and it’s easier on the budget.  
Grace: Yeah, I’ll bring popcorn.  
Felix: Alright, I might skip the concert then.  
""",

    """Ben: We’re out of milk again.  
Clara: I thought Daniel bought some yesterday.  
Daniel: I did, but Emma used it for pancakes.  
Emma: Guilty… sorry, I didn’t realize it was the last carton.  
Grace: Can we make a shopping list together?  
Hannah: Good idea, otherwise we always forget basics.  
Anna: I’ll go tomorrow after class.  
Felix: I’ll join you, I need coffee anyway.  
""",

    """Hannah: Felix, could you please lower the volume at night?  
Felix: Was it too loud yesterday?  
Hannah: Yeah, I couldn’t sleep before my early lecture.  
Felix: Sorry, I’ll use headphones next time.  
"""
]


for ex in example:
    t = time.time()
    inp = {"context": context, "content": ex}
    res = SummarizeTopicConversation.execute(inp, main_llm, None)
    bm: SummaryCategoryBaseModel = res["category"]
    print(res["summary"])
    print(bm.topic)
    print(f"Time: {time.time() - t}")
    for _ in range(5):
        print()
