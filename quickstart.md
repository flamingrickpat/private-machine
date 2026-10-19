I should give the owner the actual file paths so they can run the test themselves from scratch: the Spamuel export CSV, the generated card.md, and personality.json. The perf912 home was already imported without reconsolidation, but since they want to run the real test personally,
I'll provide complete commands rather than relying on that existing state.

Character quick start is ready for your real test, committed as aba03d9 (workflows/tool_call/ left out). Nothing has run with real cognition yet. The small tests and a dummy-cognition smoke run passed.

Your test with the Spamuel chat (912 rows, May 1–30)

```powershell
  $T = ".runtime/character-import-test"
  # 1. set up the character
  uv run --locked python -m pm.app.character_cli init --home C:/pm-homes/spamuel --companion-name Spamuel --user-name Jesse --card $T/20261018-203028/card.md --personality $T/20261018-203028/personality.json
  # 2. import + reconsolidate (the slow part)
  uv run --locked python -m pm.app.character_cli import --home C:/pm-homes/spamuel --csv $T/spamuel-full.csv --config config.alpha.openrouter.yaml
  # 3. chat
  uv run --locked python main.py --home C:/pm-homes/spamuel --config config.alpha.openrouter.yaml
```

Then open http://127.0.0.1:8770/ and log in as Jesse.

- Inputs:
    - The CSV holds only the user and companion messages from the canon. I made it with dev/tools/export_chat_csv.py.
    - The card is the Spamuel card text, and the personality file is a hand-written test file. Check it, or swap in your own.
- Card options: --assistant-card and --capabilities are optional.
- Restarting: you can stop and restart step 3 at any time. Every commit is saved to ghost.json.

What the import does
- Records: every exchange gets the two dialogue messages and a cycle state with a neutral mood. No thoughts, feelings or appraisals are invented, and every imported record is marked as imported.

- Reconsolidation in pm/memory/reconsolidation/: this ports your old consolidator's principles:
    - neighbour-weighted embeddings;
    - topic clustering with the best cluster count;
    - day, week and month summaries (with a fix for the old key collision at night);
    - facts, each linked to a long-lived semantic record;
    - a mood for each cluster;
    - 9 narrative types;
    - the old trigger rule.

  Every derived record cites its sources, and a rerun only processes what is new.

- Transfer records: a memory, a self-belief and a self-narrative line: "Spamuel moved from the earlier chat harness into pm by a memory transfer on 2026-10-18. The earlier conversations with Jesse were imported from a chat log; …"

- On failure: if anything fails, the import saves nothing.

- Companion access: the companion can start reconsolidation itself through the managed workflow memory_reconsolidation.

Web chat
- Login: you log in with a name. Other names get their own entity and their own stimulus group.
- History: the full history is shown, including the imported messages.
- Pending replies: a "thinking…" indicator shows while a reply is pending. It is approximate, and a spontaneous companion message can end it early.
- Keys: Enter sends and Shift+Enter makes a new line.
- No password: the chat binds to localhost only, but anyone on this machine can post as anyone.

What to expect
- Import speed: the import without reconsolidation took 165 s for all 912 rows.
- Reconsolidation speed: this is the slow part.
    - With 16 rows it took 247 s and 30 model calls.
    - OpenRouter currently answers about 76% of requests to qwen3.8-flash with HTTP 429 (rate-limited), and each retry waits 30 s.
    - At that rate, all 912 rows could take hours.
    - You can import with --no-reconsolidate first and run character_cli reconsolidate later. Without it, though, the companion has no clusters, facts or narratives.
- Many facts: fact extraction is very granular, giving 53 facts from 16 rows. That is about the same ratio as the canon.
- Fixed after the test: in the 16-row test, one summary misread the date 01.05.26 as January. The prompts now use an unambiguous date format, but no run has used it yet.