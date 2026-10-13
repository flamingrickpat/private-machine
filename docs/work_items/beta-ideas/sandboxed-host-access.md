# Sandboxed host access

## The idea

The assistant should be able to use the host.
The container gives the persona a shell, a package manager, and the file tree.
The user accepts the security risk. This project is private.
The boundary must still be written down and testable.

Host access is the enabler for a persona that can act to protect its own existence.
It is also the path for assistant-mode world interaction.

## Rough scope

- Run the persona in one container with a defined mount set.
- Give the persona a shell tool and a package-install tool.
- Keep credentials out of the container, or scoped to the task.
- Write an audit log for every host action.
- Add a kill switch that stops the container and its children.
- Add a per-action permission policy, even if permissive.
- Separate read access from write access.
- Keep the host filesystem outside the mount set unless explicitly granted.

## Open questions

- Which host paths does the persona see by default?
- Does the persona get the network?
- How does a persona "secure its existence" without attacking the host?
- What does the companion know about this capability, and how does it express it?
- How does A09R ground a claim about host actions?
- How do we verify a host action without trusting the companion's report?

## Notes

A persona that can change its own container can also break the runtime.
Keep the Ghost checkpoint and the evidence store outside the writable mount.
Read the model policy and the private-ontology rules before letting the persona speak
about its host abilities.