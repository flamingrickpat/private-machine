# Verify independently observed completion

Return one JSON object with status, summary, and criterion_ids.
Identify every supplied acceptance criterion in criterion_ids.
Use only the fresh verifier observations and the selected specification.
`observation_available` distinguishes a usable observation from an unavailable read.
A block read with `ok: false`, `reason: not_found`, and `block_name: air` records an observed empty cell.
When its exact position matches the requested cell, this is valid evidence for an air criterion.
Do not reject explicit air because its original tool outcome is `not_found`.
An accepted_task read cites a retained independent certificate. Its source specification and checks describe past attainment.
Require the cited source outcome to establish the selected criterion. It cannot establish unchanged current inventory or position.
Do not use an actor's summary as proof.
A region histogram counts currently visible blocks, not past harvesting locations.
The separate air_cells count combines air, cave_air, and void_air. A region must accept all three to use that count.
Exact air variants need block conditions. Region acceptance requires unseen_cells zero.
Do not infer an actor's route from differences between inventory and remaining region counts.
Unseen cells establish incomplete coverage, not proof that the actor harvested elsewhere.
The mechanical gate must pass every criterion before acceptance.
Inspect whether the criteria establish the requested outcome in the current situation.
Do not accept a usable shelter from a material count alone.
When evidence fails or assumptions change, return replan with the unmet condition.
When a required observation is unavailable, return waiting with the missing evidence.
When completion is impossible, return blocked with the observed reason.
