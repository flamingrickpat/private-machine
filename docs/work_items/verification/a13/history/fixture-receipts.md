# Earlier A13 fixture receipts

## Summary

These original receipts preserve failed setups, native failures, and earlier behavioral passes.
Later passing evidence does not rewrite them.
The latest lifecycle and argument-validation results appear on the parent page.

Parent: [A13 verification](../index.md).

These records describe earlier source revisions. Pending statements apply to those named runs.
The parent page records current acceptance.

## Retained live failures

| Receipt | Observed result |
| --- | --- |
| `.runtime/alpha/a13-m3-01/receipt.json` | Site survey accepted. Execution stopped with `KeyError: service_wait` after source code changed during the run. M3 failed. |
| `.runtime/alpha/a13-m3-02/receipt.json` | The planner retained real terrain reads but repeatedly proposed invalid certificate references. The operator stopped the run before source edits. No leaf or physical work was accepted. |
| `.runtime/alpha/a13-m3-03/receipt.json` | Site survey and 40-log gathering passed. Stone gathering retained materials but exposed a client-prefixed diagnosis permit that could not authorize dispatch. The operator cancelled before source edits. M3 remains failed. |
| `.runtime/alpha/a13-m3-04/receipt.json` | Explicit resume preserved the original task and materials. A planner input exceeded 120,000 characters after fresh review. The worker stopped before a new actor dispatch; M3 remains failed. |
| `.runtime/alpha/a13-m3-05/receipt.json` | The compact history reached inference. The proposed refinement removed stable `h_build`, so validation rejected it. Copying the full rejected proposal exceeded the repair budget; no new actor ran. |
| `.runtime/alpha/a13-m3-06/receipt.json` | The planner restored `h_build` but changed unrelated nodes. Its next proposal had invalid JSON. The repair path confused raw output with a typed tree and stopped before actor dispatch. |
| `.runtime/alpha/a13-m3-07/receipt.json` | Further full-tree copies changed unrelated fields. A ready recovery leaf also conflicted with the pending house-only refinement. The next input exceeded its budget; no new actor ran. |
| `.runtime/alpha/a13-m3-08/receipt.json` | Explicit updates added a recovery leaf in revision 4 and preserved all three accepted nodes. Independent review blocked its occupied head-cell predicate. No new actor ran. |
| `.runtime/alpha/a13-m3-09/receipt.json` | Recovery predicates were repaired in revisions 5–7. A ready recovery leaf still suspended optional refinement scope, and repeated splits broke the retained `h_build` parent reference. The operator cancelled before source edits; no new actor ran. |
| `.runtime/alpha/a13-m3-10/receipt.json` | The planner retained `h_build` as a story but changed the home story's criteria outside that scope. Validation rejected the change. Its repair input exceeded budget; no new actor ran. |
| `.runtime/alpha/a13-m3-11/receipt.json` | Repeated proposals claimed a house split but omitted the `h_build` update and detached its parent reference. The operator cancelled before adding an explicit replacement contract. No new actor ran. |
| `.runtime/alpha/a13-m3-12/receipt.json` | Revision 10 preserved earlier certificates and independently accepted recovery and stone gathering. The house worker repeatedly changed destinations while walking stalled. The exact-method guard missed that shared local problem. The operator cancelled normally before repairing the guard; M3 remains failed. |
| `.runtime/alpha/a13-m3-13/receipt.json` | Fresh observation reported food at five, and review prioritized recovery at the current location. Accumulated history exceeded the planner budget at 154,287 characters. No new actor ran. |
| `.runtime/alpha/a13-failures-01/receipt.json` | Fixture used an outdated movement argument schema. Setup failed. |
| `.runtime/alpha/a13-failures-02/receipt.json` | Six protocol boundaries passed. Native actor repeated previews without escape; the player drowned. Recovery failed. |
| `.runtime/alpha/a13-failures-03/receipt.json` | Retained fixture blocks prevented empty-cell setup. No native recovery acceptance. |
| `.runtime/alpha/a13-failures-04/receipt.json` | Selected setup cells were not empty. No native recovery acceptance. |
| `.runtime/alpha/a13-failures-05/receipt.json` | Six boundaries passed. Native actor diagnosed and executed an escape. Three later observations showed full air, stable health, supported dry ground. The strict air-only geometry check kept recovery active, so the aggregate failed. |
| `.runtime/alpha/a13-failures-06/receipt.json` | Five failure boundaries passed. Water moved the player away without reducing oxygen. The requested hazard fixture failed before native recovery. |
| `.runtime/alpha/a13-failures-07/receipt.json` | Five failure boundaries passed. Tall grass occupied the requested source cell, so setup stopped before native recovery. |
| `.runtime/alpha/a13-failures-08/receipt.json` | Six boundaries passed. Native recovery timed out. qwen restarted during the run; pm-coder resumed and compacted its context. No recovery pass. |
| `.runtime/alpha/a13-failures-09/receipt.json` | Nearby cells at the initial height did not fit the obstruction fixture. Setup failed. |
| `.runtime/alpha/a13-failures-10/receipt.json` | Setup selected a higher landing but inspected an upper cell at the wrong height. Setup failed. |
| `.runtime/alpha/a13-failures-11/receipt.json` | Five boundaries passed. Reusing a released gateway correctly failed its epoch check. Setup failed. |
| `.runtime/alpha/a13-failures-12/receipt.json` | All failure boundaries and fresh native water recovery passed. Thirteen calls and one diagnosis established a supported dry landing, full air, and stable health. |
| `.runtime/alpha/a13-failures-13/receipt.json` | Flow carried the body out of water before the first native observation. The operator stopped the inconclusive run; no new recovery acceptance is claimed. |
| `.runtime/alpha/a13-failures-14/receipt.json` | A contained shallow pit kept the body in water with full air. The policy omitted this observed water state, so setup failed before native startup. |
| `.runtime/alpha/a13-failures-15/receipt.json` | All 11 checks passed on the repaired body and policy. The actor began wet, diagnosed its escape, and left the pit. Final health, food, and air were 20; no server deaths occurred. |
| `.runtime/alpha/a13-lava-01/receipt.json` | Five boundaries and optional-work priority passed. Head-space lava killed the player before native startup. Later respawn observations did not establish recovery. The aggregate failed. |
| `.runtime/alpha/a13-adaptation-01/receipt.json` | All repeated protocol boundaries passed. The native actor gathered wood and crafted tools, then fell from a pillar. A later body timeout retained an uncertain search. No completed native adaptation report exists. |
| `.runtime/alpha/a13-adaptation-02/receipt.json` | Repeated boundaries passed. Recipe search supplied incorrect oak-only costs. The actor gathered birch and made workbenches, but qwen restarted and the worker failed with a model connection error. |
| `.runtime/alpha/a13-adaptation-03/receipt.json` | Correct recipe search supported real planks, sticks, and a workbench. The actor spent too many planks and repeated mining without a pickaxe. A model connection failure ended the run before the requested operator cancellation took effect. |
| `.runtime/alpha/a13-adaptation-04/receipt.json` | The first separate case retained real sticks and a workbench. A failed pickaxe recipe reported oak costs instead of the smaller birch deficit. The operator stop used a restart loader and marked the case interrupted. The aggregate failed. |
| `.runtime/alpha/a13-adaptation-05/receipt.json` | The actor harvested two birch logs. A wildcard block scan exceeded the body's two-second HTTP timeout. The uncertain result was retained; the cancelled case did not establish adaptation. |
| `.runtime/alpha/a13-adaptation-06/receipt.json` | The actor retained its two logs and harvested six more. It searched for oak instead of converting owned birch. A surface walk exceeded its 125-second HTTP wait because walking phases received separate budgets. The aggregate failed. |
| `.runtime/alpha/a13-adaptation-07/receipt.json` | The actor corrected its tool, mined the original stone, navigated by a surface alternative, and mined replacement stone. Incorrect global oxygen triggered recovery while server air remained full. The operator stopped the third case; no aggregate acceptance is claimed. |
| `.runtime/alpha/a13-adaptation-08/receipt.json` | All 13 boundary, native adaptation, safety, and survival checks passed on the repaired body. Four native cases retained actual material work and used three diagnoses. Every case finished dry with full health and air; the server recorded no deaths. |
| `.runtime/alpha/a13-lava-edge-02/receipt.json` | All boundaries, optional-work priority, native lava-edge retreat, and survival passed. Seventeen calls and one diagnosis established dry supported ground with full health and air. |
| `.runtime/alpha/a13-lava-edge-03/receipt.json` | All 10 checks passed on the repaired body and wet-body policy. Twenty-three calls and one diagnosis established actual retreat, dry support, full health and air, and no server deaths. |
