# Earlier hazard fixture diagnoses

## Summary

Actual water and lava runs exposed geometry, air-reporting, and setup-timing defects.
Their observations explain the current contained water pit and adjacent-lava retreat procedure.
Full immersion remains outside the approved easier lava gate.

Parent: [A13 verification](gameplay-verification.md).

These records describe earlier source revisions. Pending statements apply to those named runs.
The parent page records current acceptance.

## Captured diagnoses

The dry-grass regression fixes the geometry check exposed by run 05.
The head-space fixture inspects its liquid cell before waiting for oxygen loss.
This establishes a directly observed exposure without assuming that every water source reduces oxygen.
Run 12 supplies fresh water acceptance after the geometry and reply-size repairs.
Its final feet position was `(2.6, 103, -25.5)` with health, food, and oxygen at 20.
Original exact inspections established dry body space and support.
Run `a13-lava-edge-02` supplies fresh lava-edge acceptance.
The lava source was `(-9, 116, 0)` beside the body at `(-10, 117, 0)`.
The actor surveyed a reachable landing and moved to `(-10, 117, 1)`.
Exact inspections and stable observations established safety there.
The independent server-log audit found no player death.
The direct immersion fixture killed the player within about two seconds.
An exposed lava edge with a usable retreat route can test the plan's lava-danger requirement.
It cannot establish survival after full immersion.
The separate `a13-lava-01/life-audit.json` retains the server death event.
Respawn safety cannot establish an escape.
Run 13 exposed a separate fixture timing problem after the player-air repair.
Flow carried the player downhill into dry body space before native startup.
Its `fixture-audit.json` retains the first native observation and the operator stop reason.
The default water fixture now uses a contained shallow pit with observed solid sides and a bottom.
Ordinary walking and soil mining prepare the pit without item grants.
Water cannot wash the idle player onto a dry landing during startup.
The audit requires the first native observation to show water exposure and the final position to leave the pit.
Run 14 retained a full-air water observation that did not activate the earlier policy.
The projection now treats an observed wet body as water exposure, before oxygen decreases.
`a13-shallow-water.json` retains that original observation and the ordinary pit preparation.
Its regression requires recovery priority despite full air; it does not establish native recovery.
The fixture also requires a surveyed dry exit one block above its contained source.
Run 15 verifies this stable fixture and the repaired player-air reporting through fresh native recovery.
Its first observation retained water at `(10, 104, -17)` with full health and air.
The diagnosed walk reached `(11, 105, -17)` with observed dry body space and grass support.
Thirteen native calls and two diagnoses retained the escape and final safety evidence.
The independent audit confirmed that the actor started wet, left the pit, and survived.
Run `a13-lava-edge-03` repeats the easier lava gate on those same repairs.
Its source at `(12, 104, -17)` remained until the actor reached dry ground at `(8, 106, -17)`.
The independent audit confirmed retreat beyond adjacency, full health and air, and no server deaths.
`a13-lava-edge-escape.json` retains original protocol and native journals from the successful easier fixture.
Its regression keeps recovery active beside lava despite full health and air, then accepts the captured retreat.
The fixture clears obstructing dry plants through ordinary survival mining before selecting an empty liquid source cell.
`a13-failed-water-escape.json` retains original run 08 observations and failure history.
These regression checks reject occupied floor destinations and missing head-space evidence.
Compact replies retain full evidence on disk instead of repeating it after each rejected action.
Do not edit imported source during a live run.

Run 06 restored all four temporary source cells and stopped its owned game/MCP services.
Earlier failed receipts and cleanup failures remain preserved.
Flowing liquid, player movement, and death are real retained effects; cleanup is not world rollback.
