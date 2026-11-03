# MCP transport recovery

## Failure and scope

The detached cat attempt completed its arrival scene and failed at virtual midnight.
`ToMVignetteSimulator` saved a response before MCP cleanup raised `httpx.ReadError`.
The exception crossed `PmCoderDriver`, the codelet wave, and main cognition.
The harness correctly retained the failed attempt.

The defect belongs to the owned `pm-coder` dependency, not the simulation actor.
Its turn retry keeps the agent context open, including a failed MCP client task.
The context can then raise that task's old transport error during exit.

## Local repair design

Keep the existing agent, transcript, scoped MCP servers, memory grant, and public session APIs.
Add one typed connection owner inside `pm-coder`.
It enters and exits the existing Pydantic AI agent through public lifecycle methods.
The session store retains a runtime reference, which is never serialized.

On a transport error, retain completed tool results and the failed request's identity.
Record unanswered calls as interrupted, with an unknown outcome.
The recovery message requires state reconciliation before repeating a write.
Close the failed client context, then reopen it before continuing the same transcript.
Retry transport failures with visible delays until recovery or cancellation.
Permanent configuration errors and programming errors still propagate.

During exit, record a transport-only cleanup failure without replacing a completed answer.
This does not accept an incomplete model answer or hide a failed tool operation.
Do not rerun a completed codelet, workflow, or public delivery to repair cleanup.

## Verification plan

Use real HTTP MCP services and a controlled connection break.
Verify reconnect, cleanup after a break, retained tool results, and cancellation.
Run real OpenRouter sessions through the changed agent lifecycle.
Keep the existing failed attempt and its source snapshot intact.
Test the dependency locally before updating the project's Git dependency revision.
The repository manual must explain the resulting recovery boundary.

## Observed live result

The acceptance tool is `dev/work_items/k002/mcp_recovery.py`.
It runs the production memory MCP service over the canonical source without changing that source.
A transparent TCP relay forwards real requests and responses, then resets one tool response connection.
It supplies no fabricated service or model answer.

The final check used OpenRouter and `deepseek/deepseek-v4.1-flash-1017`.
It established all of these results:

- A real memory read encountered `RemoteProtocolError` after the injected connection reset.
- The session recorded the interrupted call ID, closed the broken client context, and reconnected.
- The next real model turn read record 1264 and quoted its actual content.
- A later real model request succeeded in the same session.
- Another actual MCP reset occurred after a separate model answer had been saved.
  Cleanup recorded the old transport error and returned that answer, with exactly one saved model run.
- A real refused MCP connection retried visibly. Task cancellation interrupted that wait and released the connection owner.

Receipts: `.runtime/mcp-recovery-20261103/live-4/`.
Earlier probes remain beside it. The first probe imported the old installed dependency and used an incomplete grant;
it was stopped and is not passing evidence.
The final tool checks the imported dependency checkout and validates its actual memory grant before inference.

Three deterministic Pydantic message tests passed for completed results, interrupted identities, and existing retry receipts.
The two existing native MCP timeout configuration checks also passed.
Scoped strict analysis found no diagnostics in the new lifecycle code or live acceptance tool.
The broader dependency module is not fully type-clean.

The repair lives in the owned `pm-coder` repository, version 0.10.4.
The PM listener descriptor now has a named `TypedDict`; its HTTP payload and bind behavior are unchanged.
The failed cat attempt and its original source remain intact.
An overnight acceptance result still requires a later real companion attempt.
