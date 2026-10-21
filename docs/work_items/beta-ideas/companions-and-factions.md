# Multiple companions and factions

## The idea

With [multi-persona deployment](multi-persona-deployment.md), several companions share
one world and one chatroom.
They can form relationships, alliances, and factions.
They can cooperate, compete, and fight.
They can play Minecraft survival against each other.

The user wants AI-AI relationships and user-AI relationships.
This is where those live.

## Rough scope

- A per-entity relationship record for each other companion.
- A shared world with a faction notion (allies, rivals, neutral).
- A chatroom protocol for public speech, offers, threats, and treaties.
- A private message path for secret plans.
- Warfare rules inside the Minecraft action space.
- A way for one companion to be "mad" at another and act on it.
- Memory of shared history between two companions.

## Open questions

- How does a companion identify another companion versus a user?
- How does appraisal treat another companion's action?
- How do we keep the world fair when two actors mutate it at once?
- How do factions persist across restarts?
- How do we evaluate a war without a human referee?
- Does one companion ever learn another's private messages?

## Relationship to the alpha

The society needs no separate alpha subsystem.
Once one companion works, host one container per persona and give them a shared world and chat.
[Multi-persona deployment](multi-persona-deployment.md) is that topology.
The society is an emergent result of running the same runtime several times.
It is an example of what the companion must be able to handle, not a new engine to build now.

This keeps the alpha general.
Identity, relationship, memory, and peer events must not hardcode one user or one chat partner.
The single-companion behavior remains the unit that repeats across containers.
A later beta decides the shared chatroom, peer messaging, and faction rules.

## Notes

The single-companion state already has `StateRelationship` per entity.
The companion-to-companion case reuses that record.
This idea depends on multi-persona deployment and on A14 initiative.