# System One decision uses

## The idea

The System One decision classifier is delivered but unused.
A02 found poor quality, including confidently wrong answers.
The model policy keeps the LLM as the default.
Beta can look for a short, explicit decision that System One does well.

A good use is a short input, explicit classes, and a simple fact-based decision.
The classifier is faster and cheaper than the chat model.
It is not a general replacement.

## Rough scope

- List candidate decisions from the live loop.
- For each, define the input schema and the classes.
- Write live tests with independently defined results.
- Compare each candidate with the normal LLM on the same cases.
- Adopt only the candidates that pass and give a real benefit.

## Candidate decisions

- Attention urgency band (low, medium, high).
- Public-speech presentation choice (candid, withhold, deceive).
- A simple routing choice between two workflows.
- A short content classification with a small class set.

## Open questions

- Is the quality good enough for any live use?
- Does the decision head fit the input?
- Is the latency gain worth a second service?
- How does a wrong answer reach the INVALID fallback?

## Notes

Read `docs/subsystems/decisions/` and the model policy.
Do not adopt a use just to justify the delivered client.