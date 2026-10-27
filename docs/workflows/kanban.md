# Local Kanban board

## Summary

The development board stores cards in `dev/kanban/board.json`.
A local browser page edits the same file that an agent can edit directly.
The board uses Python's standard library and local assets.
It needs no account, external service, package installation, or companion instance.

Parent: [development loop](development.md).
Implementation: [`dev.kanban.server.board_handler`][], [`dev.kanban.server.read_board`][], and [`dev.kanban.server.write_board`][].

## Open the board

From the repository root, run:

```powershell
.\dev\kanban\start.ps1
```

From `dev/kanban`, run `.\start.ps1`.
The launcher finds the repository from its own location.
It uses `uv` and the locked repository environment, regardless of the current directory.
The command opens `http://127.0.0.1:8767` in your browser.
The server binds only to the local loopback address.
Stop the server with Ctrl+C in its terminal.

If port 8767 is occupied, select another port:

```powershell
.\dev\kanban\start.ps1 -Port 8768
```

Use `-NoBrowser` to print the address without opening a browser.
The equivalent Python command is `uv run --locked python dev/kanban/server.py`.
The script also runs directly with Python 3.12.
It imports no runtime or workflow code.

## Use the board

1. Press **Add card**, or the plus button in a column.
2. Enter a title and optional notes.
3. Select a column.
4. Press **Save card**.

Drag a card to another column to move it.
For keyboard use, open the card and change its **Column** field.
Open a card to edit its title or notes.
Use **Delete card** to remove a card after confirmation.
Cancel or Escape closes an edit without a save.
During a save, the dialog stays open until the request completes.

The columns are **Backlog**, **To do**, **In progress**, and **Done**.
Cards keep their insertion order within each column.
Dragging changes the column, not the order within a column.
Notes remain plain text, including file paths and work-item references.
The board starts empty and does not infer work-item status from older plans.

## Edit cards through an agent

Ask the agent to read or edit `dev/kanban/board.json`.
For example, ask: “Add a card to investigate the latest console error.”
After a direct file edit, press **Reload** in the board.
The server reads the file on every request.

The file contains `next_id` and `cards`.
Each card contains `id`, `title`, `description`, and `status`.
The status values are `backlog`, `todo`, `doing`, and `done`.
Each id is a unique positive integer less than `next_id`.
New cards use `next_id`, then increase it by one.
Deleted ids stay unused.
[`dev.kanban.server.validate_board`][] defines the accepted file contents.

The file belongs in Git with the board code.
A browser save changes the working tree.
It does not create a Git commit or push anything.
Existing work-item plans, acceptance gates, and verification records retain their purpose.
A card in **Done** records a development choice, not a verified companion result.

## Save and recovery behavior

One HTTP server serializes browser saves.
Each browser save sends the revision of the loaded file.
A changed revision receives HTTP 409 instead of overwriting the newer file.
The dialog retains the draft after an error.

If a conflict occurs, copy your draft before closing the dialog.
Then press **Reload** and apply your changes to the current card.
Use one server per board file.
Complete direct file edits between browser saves.
The revision check does not lock out a simultaneous external file writer.

The server flushes a temporary file before it replaces the board file.
Windows replacement locks retry every 0.2 seconds.
Other file errors appear in the browser.
An interruption before replacement leaves the previous board intact.
An interruption after replacement can leave the browser uncertain about a completed save.
After a lost reply, reload the board before repeating an add or delete.
A forced termination can leave an unused `.tmp` file beside the board.

Invalid JSON or card fields cause an error without replacing the file.
If the file is invalid at startup, the server exits with a plain error.
Correct the JSON before restarting or reloading.
The server serves only its page, two assets, and the board API.
It rejects browser writes from a different origin.

For an isolated board, pass an existing JSON file:

```powershell
.\dev\kanban\start.ps1 -Data .runtime/kanban-check/board.json -Port 8768 -NoBrowser
```

This option changes the data file only.
The page and assets still come from `dev/kanban/`.
[`dev.kanban.server.main`][] defines the command options and server lifecycle.
