# Provider rejection stopped private cognition

## Summary

The `tease_then_truth` case stopped after an external provider rejected private-trigger output.
PM correctly refused the unfinished JSON fragment.
The missing boundary was above the transport: the private thought host let that external failure stop the entire application.
The existing JSON repair did not apply to this response.

Parent: [A18B error analysis](../a18b-errors-20261016.md).
Related: [output recovery](../../../subsystems/cognitive-output-recovery.md).

## Actual provider result

The case ran for `1039.266` seconds and published three companion replies.
The failed request owner was `inner_thought`.
The request model was `qwen/qwen3.8-flash`; OpenRouter reported provider `Alibaba`.
Its response arrived after `9.187` seconds.

The response contained:

```text
finish_reason: error
choice.error.code: 502
Upstream error from Alibaba: <400> InternalError.Algo.DataInspectionFailed:
Output data may contain inappropriate content.
```

The provider also attached an unfinished JSON string for `rumination_notice`.
Its metadata used `error_type: provider_unavailable`, despite the content-inspection message.
The generic PM exception therefore obscured useful differences between provider failure and provider filtering.
The raw response retained both details.
The provider's exact filtering rule remains unknown.

This error arrived through the successful HTTP response path.
The nested `502` was a completion error code, not a top-level HTTP 502 status.
PM's saved response does not retain the numeric HTTP success status.

## Exact propagation path

| Source at `f7eb226` | Current logic |
| --- | --- |
| [pm/llm/http_client.py:28](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/http_client.py#L28) | `completion_error` recognizes `choice.error`, `finish_reason=error`, and `content_filter`. It creates `ProviderCompletionError`. |
| [pm/llm/http_client.py:148](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/http_client.py#L148) | The transient retry branch handles top-level HTTP 429 and HTTP 5xx statuses. |
| [pm/llm/http_client.py:159](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/http_client.py#L159) | A completion error inside an HTTP success response raises immediately. It does not enter that transient branch. |
| [pm/llm/stage_model.py:28](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/stage_model.py#L28) | `complete_with_retry` retries only an error containing `no content`, once. This error does not match. |
| [workflows/main_cognition/stage_support.py:67](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/stage_support.py#L67) | The tool call occurs before the JSON parsing handler. Provider errors propagate unchanged. |
| [workflows/inner_thought/host.py:78](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/inner_thought/host.py#L78) | The private host isolates `ModelProposalError` and `ModelFormatError` only. It does not isolate `ProviderCompletionError`. |
| [pm/host/application.py:81](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/host/application.py#L81) and `:157` | The module thread sends any uncaught exception to `Application.on_error`. That method stores the first error and stops all modules and workers. |
| [dev/work_items/a18b/runner.py:149](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/runner.py#L149) | The harness sees the application error while it waits for a reply. It records a failed case. |

Main cognition also explicitly propagates provider errors at [workflows/main_cognition/host.py:295](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/host.py#L295).
The separate tool-calling completion path retries broader `ModelError` values at [pm/llm/stage_model.py:164](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/stage_model.py#L164).
Thus, recovery policy differs between plain cognitive calls and tool-calling calls.
That difference did not protect this private-trigger request.

## What worked and what failed

The transport preserved a real failure and rejected partial text.
It did not invent a completed private thought or silently accept malformed data.
The failure was the application-wide consequence of a rejected external completion.
A private cycle's unavailability ended unrelated conversation and cancelled other activity.

The existing partial-provider test at [tests/unit/pm/llm/test_http_retries.py:83](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/tests/unit/pm/llm/test_http_retries.py#L83) expects this transport exception.
It does not establish private-loop recovery after that exception.

## Proposed repair

Provider completion failures need typed reasons with retained raw evidence.
Temporary provider outages and content inspection need separate policies.
A content-inspection rejection must not retry indefinitely as a temporary network failure.
A private rejected cycle can remain uncommitted, emit a diagnostic, and leave the application available.
Main-loop recovery must also retain the original user causes.
No repair can accept the partial JSON as a valid thought.

Evidence:
`.runtime/a18b-current/fresh/tease_then_truth/llm-logs/b04b9fa45db242cb909f2fe291ac311b.response.json`,
the matching request, `failure.txt`, and `receipt.json`.

Code:
[`pm.llm.http_client.completion_error`][],
[`pm.llm.http_client.JsonClient.request`][],
[`pm.llm.stage_model.complete_with_retry`][],
[`workflows.inner_thought.host.PrivateThoughtLoop.handle_batch`][],
[`pm.host.application.Application.on_error`][].
