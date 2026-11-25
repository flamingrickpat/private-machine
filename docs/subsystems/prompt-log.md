# Prompt logs

Use prompt logs to see what a model received and why a call failed.
Set `inference_io.prompt_log` in the [model YAML](../models.md).

Each request attempt has a Markdown view and a JSON receipt.
The receipt records the request, response, route, and workflow invocation.
A usage stream reports timing and tokens. Embedding receipts omit response vectors.

A persona normally writes logs under `logs/prompts`. Set
`inference_io.prompt_log_directory` to choose another location. Simulation
invocations can select a dedicated evidence directory.

- [Request log][pm.llm.prompt_log.start_prompt_log]
- [Response log][pm.llm.prompt_log.finish_prompt_log]
- [Execution timeline](timeline.md)
