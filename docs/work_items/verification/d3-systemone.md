# D3 System One comparison

## Summary

The large D3 Q4 model loads on the RTX 3090 with its projector on the CPU.
An 8K decision context covers the measured small A14B inputs.
Short decisions are faster than either tested Qwen route.
Long inputs and several questions can remove that advantage.
This experiment changes no workflow, chat, or embedding route.

Parent: [optional typed decisions](../../subsystems/decisions/index.md).
Policy: [model selection](../alpha-plan/model-policy.md).
Earlier evidence: [A02](a02-decisions.md) and [A14B](a14b-agent-memory.md).

## Model and machine

The [publisher model card](https://huggingface.co/vllm-sr/d3) describes a decision model that returns choice, probability, and score answers.
This comparison tests choice answers only.
It does not test a replacement chat or embedding model.

| Component | Observed identity |
| --- | --- |
| Source revision | `9729f82ea2998bc207971e7a06039163f378a226` |
| Server | `C:/workspace/llama13/llama-server.exe`, build `b11592-f851e244d` |
| D3 model | `C:/source/ai/d3/vllm-d3-Q4_K_M.gguf`, 19,403,686,112 bytes |
| Projector | `C:/source/ai/d3/mmproj-vllm-d3-BF16.gguf`, 931,146,240 bytes |
| Decision endpoint | `http://127.0.0.1:8082/v1/systemone`, alias `d3-big` |
| GPU | RTX 3090, 24,576 MiB |
| CPU and RAM | Ryzen 9 5950X, 16 physical cores, approximately 128 GiB RAM |
| Local comparison | Strata, `http://127.0.0.1:8080/v1`, alias `qwen3.8-flash-next-iq3_s` |
| Hosted comparison | OpenRouter, `qwen/qwen3.8-flash`, through `config.alpha.openrouter.yaml` |

The local Qwen model uses the operator's `C:/source/strata/run-iq3_s.bat` launcher.
The two Qwen routes use different model variants and serving systems.
Their timings compare deployment paths rather than identical weights.
No package or lock revision changes.

## Launcher and context

The maintained launcher is `dev/tools/start_d3_big_systemone.ps1`.
The machine copy is `C:/source/ai/start_d3_big_systemone.ps1`.
Run the machine copy in PowerShell:

```powershell
& C:/source/ai/start_d3_big_systemone.ps1
```

The launcher selects the RTX 3090 by its NVIDIA UUID.
Its process sees that GPU as `CUDA0`.
The RTX 5090 stays outside the process's visible devices.
The projector uses the CPU backend.
Automatic fitting can move model layers to RAM, while the explicit context limit stays fixed.
The default memory reserve is 1,536 MiB.
Other active GPU consumers can change the resulting layer placement.
The delivered profile uses FP16 cache, one slot, and the input token table on the GPU.
Its observed GPU memory use is approximately 21.3–21.5 GiB.
This table belongs to D3.
It is separate from the embedding model.
`-CpuTokenTable` retains approximately 682 MiB more GPU space for other consumers.

D3 must process each decision input in one physical batch.
A 512-token physical batch rejected a 2,179-token request with HTTP 500.
The preserved failure reports that the physical batch is too small.
A large context declaration alone does not repair this limit.
The launcher derives both batch sizes from the context size unless the operator supplies them.
Its validation rejects a physical batch smaller than one decision context.
The context includes state, question descriptions, options, and internal tokens.

The earlier A14B small chat decisions used 2,841 through 3,523 prompt tokens.
Placement probes test evidence at both ends of larger inputs.
The 8K profile passed six probes through 7,929 actual decision tokens.
The 16K profile passed six probes through 14,484 actual decision tokens.
That larger profile moved nine model layers to RAM.
Its longest probes took approximately 12.2 seconds.

For a larger context, restart the server with this command:

```powershell
& C:/source/ai/start_d3_big_systemone.ps1 -ContextSize 16384 -CacheType q8_0 -CpuTokenTable
```

The launcher does not retry a failed server.
The caller owns shutdown.
Closing a client connection does not establish that server inference stopped.
The launcher restores its process environment after the server exits.
It does not change PM checkpoints or accepted memory.

## Decision quality

The frozen comparison contains 22 cases, repeated twice.
Six cases started as reserved cases in the earlier Flash comparison.
They are exposed cases for this follow-up, with unchanged prompts and expected labels.
The original A01 holdout remains untouched.
Expected labels follow source facts and explicit class definitions.
They do not come from either model's answer.

| Route | Correct trials | Median complete decision |
| --- | ---: | ---: |
| D3 Q4, delivered profile | 42/44 | 0.188 seconds |
| Local Qwen IQ3_S | 40/44 | 0.610 seconds |
| OpenRouter Qwen | 42/44 | 1.125 seconds |

Both large D3 and the expanded OpenRouter repeat selected `FAIL` for two conflicting current reports.
The expected route is `INVALID` because the report order is unknown.
The `FAIL` description also includes any failed check.
This overlap requires review before adoption.
The earlier OpenRouter run selected `INVALID` on that case.
Neither response supplies a human rating.
The original expected label and failed receipts remain visible.

Local Qwen also selected `PASS` for missing test evidence in both repeats.
Large D3 fixed that error and the stale-run error from D3 Flash.
D3 Flash's earlier development result was 28/32 at a 0.078-second median.
Large D3 passed all 32 development trials.

Three captured A14B retrieval inputs add six trials per route.
Each includes the preserved retrieved memory records and the original note request.
The earlier experiment introduced those note preferences as synthetic user inputs.
They are not existing persona facts.
The requests cover a paraphrase, a new exact-wording constraint, and an absent preference.
All three routes passed all six label trials.
D3's two-question median was 4.26 seconds with the delivered profile.
Local Qwen's median was 1.35 seconds.
OpenRouter's median was 1.76 seconds.
Local Qwen reused most of the prompt on identical repeats.
Those medians do not represent fresh prefill alone.
The comparison checks decision fields, not the full A14B explanation or evidence-ID contract.

## Fresh prefill comparison

Each fresh trial places a new nonce before the input.
The nonce prevents local conversation-prefix reuse.
The same state and question go to all three routes.
Requests run sequentially, with two trials per input budget.
Both Qwen routes report zero cached tokens in these probes.

The chat probes disable thinking and request one output token.
Their measurement is time to first content token, including transport and server overhead.
D3 returns a complete decision without text sampling.
Its measurement includes transport and decision-head work.
OpenRouter does not expose the engine's pure prefill time here.

| Approximate state budget | D3 FP16 decision | Local Qwen first token | OpenRouter first token |
| --- | ---: | ---: | ---: |
| 2K tokens | 1.76 seconds | 2.35 seconds | 2.69 seconds |
| 4K tokens | 3.09 seconds | 3.35 seconds | 2.70 seconds |
| 7.8K tokens | 5.92 seconds | 3.20 seconds | 1.09 seconds |

The initial Strata run reports approximately 958–973 tokens/second for 2K prefill.
It reports 1,785–1,796 tokens/second at 4K and 2,525–2,710 near 8K.
These rates exclude cached tokens and differ from sampling speed.
Only two fresh samples support each timing median.
They do not establish a stable service latency target.
The table uses the final matched repeat with the delivered D3 settings.
No documentation build or other owned test ran during that repeat.
An earlier repeat overlapped the documentation build and remains excluded from this table.
The initial fresh comparison gave a lower OpenRouter median of 1.17 seconds at 4K.
It also gave a lower local Qwen median of 2.42 seconds at that budget.
This variation limits claims about a stable crossover point.
Near 8K, both clean comparisons favor the Qwen routes.

## Tuning observations

The 16K physical batch requires RAM layers and increases short-decision latency to 0.422 seconds.
The 8K physical batch keeps all 65 model layers on the 3090.
Q8 cache used approximately 20.7 GiB of GPU memory.
FP16 cache modestly improved fresh prefill and retained the tested answers.
CUDA graph optimization gave no useful improvement.
Moving the model's input token table to the GPU gave approximately a 1% prefill improvement.
That setting used another 682 MiB of GPU memory.
It does not involve the separate embedding model.
Two slots loaded with an 8K limit each, but they gave no useful two-question speed improvement.
A later large-input probe triggered CUDA out of memory and server exit code `-1073740791`.
Windows also temporarily refused a new PowerShell process because its memory commit limit was exhausted.
The failed server exited, and the stable single-slot profile restored the endpoint.
The delivered launcher excludes the failed two-slot option.
The operator-owned Strata server remains running.

## Evidence and checks

Evidence stays under `.runtime/alpha/20261014-d3-big/`.
The earlier Flash comparison stays under `.runtime/alpha/20261014-d3-comparison/`.
The preserved scripts, inputs, and receipts permit local repetition.

| Evidence | Content |
| --- | --- |
| `batch-gpu-token/`, `local-short/`, `openrouter-short/` | Frozen cases, exact responses, expected labels, and timing |
| `captured-8192/`, `captured-gpu-token/` | Preserved A14B retrieval inputs and paired label trials |
| `fresh-prefill-clean/` | Final matched fresh-input digests, stream chunks, usage, and native Strata timings |
| `fresh-prefill-q8/`, `fresh-prefill-gpu-token/` | Initial comparison and tuning repeats |
| `fresh-prefill-delivered/` | Excluded repeat that overlapped the documentation build |
| `prefill-512-probe/` | Physical-batch rejection receipt |
| `prefill-8192/`, `prefill-16384/` | Placement tests with exact inputs and responses |
| `server-*.log`, `launcher-*.ps1` | Allocation, CPU projector, visible GPU identity, and launch settings |
| `local-models.json`, `local-props.json` | Operator-owned Strata endpoint identity |
| `unit-delivered.log`, `unit-delivered-receipts/` | Final real-endpoint unit checks, including the unresolved class failure |
| `launcher-two-slot.stderr.log`, `fresh-prefill-two-slot/` | Failed two-slot allocation and subsequent failed requests |

[`pm_decision.DecisionClient`][] validates the actual provider responses.
[`pm_decision.DecisionReceipt`][] retains provider answers, request digests, and timing.
The diagnostic scripts store expected labels separately.
Their exit status records script execution, rather than semantic acceptance.
Inspect their `correct` fields before claiming a pass.

The live unit script runs three test methods with individual cases as subtests.
The development and captured-input methods pass.
The conflicting-report method fails with `FAIL` instead of `INVALID`.
Root foundation checks pass: 571 unit tests and 63 contract tests.
Those suites establish engineering boundaries rather than model quality.
No Minecraft gate runs in this experiment.
Workflow adoption and a larger jointly designed test remain pending.
