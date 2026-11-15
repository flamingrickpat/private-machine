# Final run 3 prerequisites: Spamuel, Spamuel, and Minecraft

Written 2026-11-15 for B07 Pass 5 step 6 (owner decision D10). No code was changed for this run.
Run 3 is **not startable today**. This file lists what is missing.

## Resolved the same day (B07 Pass 8)

The owner started the run the same evening. The harness supports bodies now, and the stack the run
needs is a small local tool instead of a manual procedure. What changed:

1. **Bodies are switched on per participant.** `TemplateParticipant.minecraft_mcp_url` names the one
   body a participant holds; `RunEngine` records it in the controller document and passes it to the
   participant at `/control/start`; the participant writes `minecraft.mcp_url` into its own model
   configuration and starts with `enable_minecraft=True`. A participant without the field is
   unchanged. Missing item 2 below is closed by that per-participant field: each Spamuel has his own
   endpoint, so each has his own body lease and avatar.
2. **World time (missing item 3) is a recorded decision, not a code change.** Bodies work on demand:
   the participant's world poll stays off, because a virtual-clock poll would fire hundreds of times
   inside one scene advance while the server keeps real time. Body workflows still read the world when
   they run. See `docs/subsystems/simulation.md`, section "Minecraft bodies".
3. **Checkpoints (missing item 5).** `RunEngine` refuses to restore a run that owns a body, and says
   so. A failed body run needs a new run ID instead of mixed evidence.
4. **Real services.** `dev/tools/minecraft_final_run.py up` starts one disposable server copy plus one
   pm-minecraft process per body (different MCP, body, and viewer ports, different agent homes),
   waits for both bodies to answer an observation, and writes a receipt with every process ID.
   `status` and `stop` read that receipt from another process; `stop` asks the server to save and stop
   over RCON first.

Still true and reported, not fixed: missing item 4 (world effects are not part of the harness
receipts; the pm-minecraft agent homes and logs carry them) and missing item 6
(`MINECRAFT_REQUEST_SECONDS = 660` and the 5-second startup probe are still caps on the body path;
no body request failed in this session).

## What the harness lacks

1. **Bodies are switched off.** `simulation/participant.py::Participant._start` builds every
   participant with `enable_minecraft=False, world_poll_seconds=None` (finding F14). No participant
   gets body tools, the world poll, or the Minecraft routes, whatever the configuration says.
2. **One configuration for all participants.** `RunEngine._run` freezes one model configuration per
   attempt (`freeze_source(...)["config"]`) and starts every participant with it. The body endpoint
   is the `minecraft.mcp_url` of that configuration (`workflows/companion.py::_minecraft_endpoint`),
   so two participants would share one body. Each Spamuel needs its own endpoint: either a
   per-participant `minecraft` section in the template, or a per-participant override passed to the
   participant process. Template and engine have no such field today.
3. **World time.** The harness advances a virtual clock between scenes
   (`RunEngine.advance`, `/control/advance-step`). A Minecraft server keeps real time. The
   participant's world poll and body tasks would run in real time while the scenario clock jumps; how
   the two clocks relate (for example, no clock jumps while a body task runs) is a design decision.
4. **Evidence.** `simulation/evidence.py::capture_participant` captures the store and public speech.
   World effects (positions, inventory, blocks) are not part of a run's evidence; the run would need
   the body journal (`workflows/minecraft_body/journal.py`) or the pm-minecraft receipts copied into
   each attempt.
5. **Checkpoints.** A coordinated checkpoint restores stores and the transit ledger. It cannot
   restore the Minecraft world, so restore after a failure would put the characters back in a world
   that kept its later state. A disposable world per attempt, or no restore for run 3, is needed.
6. **Deadlines on the body path.** `pm.constants.MINECRAFT_REQUEST_SECONDS = 660` is a timeout on one
   body request (`MinecraftConfig.request_timeout_seconds`), and the startup probe has
   `probe_timeout_seconds = 5`. The owner's failure rule (no timeouts on external failures) has not
   been applied to this path; it needs its own review before a long body run.

## What a second body needs

- A second bot account on the server and a second pm-minecraft process (or one process serving two
  bodies, if pm-minecraft supports that; not checked), each with its own MCP URL.
- Its own body lease identity (`docs/subsystems/body-leases.md`): the leases are per body, and two
  companions must never hold one avatar.
- The participant names `Spamuel 1` and `Spamuel alpha` as in run 2, with each bot account named so
  that neither reads as a copy of the other.

## Real services required

- A running Minecraft server that both bots can join, in a disposable world (run 3 mutates the world;
  `AGENTS.md` section 9.3 asks for disposable worlds for damaging tests).
- The tested pm-minecraft checkout at `C:/source/pm/pm-minecraft`, started with its own interpreter,
  one MCP endpoint per body (the documented single-body transport is `http://127.0.0.1:6767/mcp`).
- Two Minecraft accounts (or an offline-mode server that accepts two usernames; not checked here).
- OpenRouter, as for runs 1 and 2.

## Not verified

Nothing in this file was run. The pm-minecraft repository, its multi-body support, and the server
setup were not inspected; the statements about them come from the PM-side code and
`docs/subsystems/body-leases.md`.
