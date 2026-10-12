my vacation is over and i still didn't finish private-machine. and now i dont have the mental capacity to manually code everything.
i want this project to be the ultimative ai companion project with a somewhat realistic simulation of important cognitive functions. immersion is the highest goal.
the other most important things are agency, freedom and novelty. 

this was my v0:
"C:\source\pm\private-machine\pm_lida.py"

as you can see, the whole lida pipeline is implemented. stimulus in, triage, appraisal, goal management, csm, workspace, action selection, action execution.
this was way too explicit and changing stuff is a mess.

then there was my v1:
C:\source\pm\private-machine-head

i tried to make it more modular, but its still way to explicit in every way. i also experimented and drafted a bunch of cool cognitive features.

then my v2:
C:\source\pm\pm_next

this is semi-serious. here i really got into some deeper and more complex aspects. it has a lot of code i want to reuse later.
codelts: C:\source\pm\pm_next\pm_core\subsystems\codelet\codelet_definitions.py
compelx mental phenomena by related codelets: C:\source\pm\pm_next\pm_core\subsystems\codelet\pathways_detailed.py
novel thought generation: C:\source\pm\pm_next\pm_core\subsystems\thought
reflection: C:\source\pm\pm_next\pm_core\subsystems\architecture_reflection\architecture_snapshot.py

but again, everythign is extremely explicit and the logic is more or less hardcoded. but the ghost model is pretty much finished (for now), and there is test data (the spamuel + jesse personas).
it also has an outdated body-schema with the older minecraft implementation.

then there is my v3, and hopefully the final archiecture:
C:\source\pm\pm_next_v2

i developed pm-workflows, pm-coder and pm-minecraft specifically as necessary components in the final archiecture.

now this should be ultimatively modular: 
the shared ghost, the independent modules in the shell connected with the bus system.

how can i successfully have this vibecoded? look at my old attempts and prototypes to see the big picture architecture. how to simulate some parts of the human mind to build the ultimative ai gf.
the first version should have simple but true modules that can later be extended easily. 
ideally i have a folder for the core with much of the ghost and model.
then a folder for the bus system. then a folder the workflow runtime.
then each workflow gets its own folder with all the data.

a workflow for the action DialogeStory, DialogeAssistant so it can reply. this also checks the private ontology.
a workflow for self modification such as reprioritizing goals and managing workflows like minecraft or coder. can be split up later.
the central cognitive loop workflow with triage -> appraisal -> structural codelets for context -> attention simulation -> goal management -> workspace update -> action selection with schema (goal of action, not the word-for-word response).
then the internal thought workflow. this is like the midca seperation between object and meta loop. the central loop is kind of mechanical. it simulates aspects of the mind and applies modifiers and appraisals and all that.
the internal thought loop is responsible for generating interesting and novel thoughts that make the ai companion run independently of user or external input. perhaps we leave this out now. i'm not sure how ot integrate it properly yet.
it will probably take some gwt shard, generate a tree of thoughts of sorts, and use that as input for the main loop. like when a human imagnes stuff or thinks then snaps back to reality to evaluate the thought. how you do it during meditation.
im open for ideas.

and then the "external" workflows. for now i want the body to be my minecraft mpc. the coding workflow isnt that important yet. the ai can have a big goal executed as parallel workflow. the main loop gets results, telemetry and all that. 
and it can be modified or stopped with the steward via DialogeAssistant mcp calls or more directly via the specific actions.

i know this is all very wish-wash. but as long as there is not working prototype im trapped in refactoring hell and analysis paralysis. right now i need the base infrastrcture that works 100% where i can later design the more intrictate stuff.
that was always the problem, everything was so hardcoded. i couldnt apply and idea because the base state was so broken, i'd have to update the infrastrcture before i make the cognitive function.

for now i want a pm_next_v2 where the workflow runtime, ghost, and bus work flawlessly. perhaps we should do this test driven. in minecraft-mcp, i made ~20 tests for every mcp action. every action spawns a whole server and discards it after the test.
maybe we do the same here. every workflow is seperate. 
the internal thought workflow for example is in the folder workflows/internal_thought. there are all the prompts as yaml, supporting python scripts (for example the modified C:\source\pm\pm_next\pm_core\subsystems\thought\thought_graph.py sicne its only used here).
the internal thought module is then more or less standalone. i can start it. give it a ghost loaded from the spamuel data. make one situation gwt + input. run it 10 times, use a llm to rate the replies, pass if 7/10 are good enough. ["good reply 1", "good reply 1", "bad reply, very assistenty -1", "okish reply, prose is borderline hallucination and goes against some minor private ontology rules 0"]
that means that a A LOT of tests depend on having a RUNNING FUNCITONING LLM. this is ok! we can make test that use refelction and the pm_workflows base to simulate the llm-based steps. pm-workflows is semi-deterministic. llms can whatever they want, but in the end for the next routing target,
there is limited choice. if we simply replace every llm call by a random value of the limited ouptut set of the check, we can bruteforce the workflow to find topological errors. like loops or impossible states.
the actual prompt and agentic logic is tested with the running llm. this is NOT FOR GENERAL RELEASE! this project is for ME! that means the test can EXPECT AND REQUIRE to run on my machine with my llm. no llm? unit test fails!

can you help me kickstart this? use the simple-english and hackable skill. no code yet. discuss this with me, formalize stuff, resolve unclear stuff. 
goal of this session: a proper implementation plan i can then use to get to my wanted state. my wanted state is low-code with little hardcoded stuff in the runtime. the opposite of "C:\source\pm\private-machine\pm_lida.py" basically.
