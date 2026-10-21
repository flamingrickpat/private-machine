# Multi-persona deployment

## The idea

The project runs as one container with one persona.
Run many containers, one persona each.
Point them at one shared Minecraft world.
Add a shared chatroom for public messages and private messages between personas.

Ten containers can hold ten Spamuel-like personas.
Each keeps its own Ghost, memory, and identity.
They meet in the same world and the same chat.

This is a deployment topology, not a multi-Ghost process.
The single-process Ghost does not change.

## Rough scope

- One container image with the runtime and its configuration.
- One seed and one checkpoint per persona.
- One shared Minecraft server and one shared MCP body farm.
- A shared message service for the public chatroom.
- A private message path between two personas.
- Per-persona credentials and endpoints.
- A supervisor for start, stop, and restart of the whole set.

## Open questions

- How does each persona get a distinct Minecraft character?
- Who owns the world state when two personas act at once?
- How does the body-lease protocol work across containers?
- How does the shared chatroom reach each cognition loop?
- How do private messages stay private from the other containers?
- How do we record causal evidence across containers?

## Notes

Read the Minecraft body-lease and recovery subsystems.
The body lease is process-local today. Beta must make it cross-container.
This idea is the base for [companions and factions](companions-and-factions.md).