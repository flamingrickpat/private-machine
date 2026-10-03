# Laya decisions and explicit workflow inputs

## Summary

Laya becomes a reusable decision service for Python and workflows.
The [A02 client](../../subsystems/decisions/index.md) now supplies the public Python boundary.
Its use is optional, and existing workflows keep their model choices by default.
Later experiments can retain Laya without adopting it in any workflow.
A classification phase chooses declared routes from concrete inputs.
Laya can select a relevant file from folder metadata before reading its contents.
Exact files and named results remain available.
INVALID routes exceptional cases to a coding agent.

Parent: [alpha plan](index.md).
Delivery: [A02](work-items/a02.md) through [A05](work-items/a05.md).

## Verified local material

The downloaded file is `C:\source\ai\Laya-BF16.gguf`.
It has architecture `modern-bert`, decision type `laya`, and a declared context of 8192.
Its decision head stores a 192-token budget.
The file size is 844,026,720 bytes.

The local binary reports build 11351, revision `631109b34`.
`C:\workspace\llama13` contains binaries, rather than a checked-out source tree.
A02 must verify actual inference with the selected running binary and GGUF.
A01 later observed the running decision service from build 11374, revision `b92761a51`, on port 8082.
The [baseline receipts](baseline/receipts.md) record its artifact hash and separate it from the earlier chat/embedding binary.

The endpoint was merged into llama.cpp on 2026-10-02.
See [upstream pull request 29818](https://github.com/ggml-org/llama.cpp/pull/29818).

The publisher distinguishes Laya checkpoints with different trained context lengths.
The GGUF's architectural maximum does not prove reliable classification of 8192-token reports.
Measure this artifact with short, long, and evidence-at-the-end inputs.
See the [Laya model card](https://huggingface.co/convaiinnovations/laya).

## API boundary

The endpoint accepts a state and independently answered typed questions.
Its question kinds are `choice`, `score`, and `noul`.
Choice returns probabilities and a selected label.
Score returns an expected level index.
Noul returns a probability of true.
These probabilities need calibration on PM inputs.

Long Laya questions and options can be truncated.
Images require a compatible vision decision model and projector.
The downloaded Laya must not be treated as a Minecraft vision model.
See the [llama.cpp System One API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api).

## Reusable Python library

Confirmed location: the existing `pm-coder` distribution, exposed through a public Python API.
Keep the client module independent of Ghost, the agent loop, and the workflow engine.
Use a small typed HTTP client.
Inspect available SDK compatibility before choosing one.

The current distribution installs single Python modules.
A light client module such as `pm_decision.py` can ship with it without reorganizing the entire package.
Choose and document public imports in A02.

Provide synchronous and asynchronous requests, cancellation, and explicit transport/schema errors.
Keep the provider's typed answers intact.
Expose ordinary Python convenience methods for choice, score, and probability decisions.

The client accepts concrete text or structured values.
The workflow adapter owns folder listing, file reads, route selection, and coding-agent fallback.
Workflow policy decides what to do with uncertainty and outages.

A receipt records input digest, question definitions, provider/build identity, answer, and timing.
Credentials remain outside receipts.
Diagnostic source text follows the existing instance-local evidence policy.

Keep decision connections separate from chat and embedding connections.
Do not require a chat-capable decision model.
Allow independent service activation and concurrent chat/body work.

Upgrade only affected packages.
Record tested versions or Git revisions and update lock files in each consuming repository.
Publishing to a remote package registry is outside this plan.

## Workflow contract

Read [input selection and classification phases](workflow-inputs.md) for the proposed manifest and artifact contracts.

## Candidate cognitive uses

The user wants the technology available without forced adoption in existing workflows.
A05 can compare narrow candidates such as triage category or finite attention classifications outside the live path.
Use the same frozen input with human labels, Laya, and the current LLM stage.
These are experiments, rather than planned stage replacements.

New workflows can choose classification when the task calls for a finite decision over concrete evidence.
Keep generative planning, rich appraisal explanations, dialogue, and source reflection on a generative model.
Keep an LLM for interpretive decisions when it produces better results.
A05 can succeed without converting any existing cognitive stage.
