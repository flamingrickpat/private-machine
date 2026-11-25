# Prepare the Minecraft server

The Minecraft scenarios use a private world for each launch. PM copies the server
installation and starts one body for each companion.

1. Install Java 17 or newer and prepare a Fabric 1.19.4 server directory.
   Keep the launch JAR, server JAR, libraries, and mods together.
2. Add Fabric API and Agentic Babymode to its `mods` directory.
3. Accept the Minecraft EULA in that server directory.
4. Copy `config_simulation.yaml.example` to `config_simulation.yaml`.
5. Set `server_installation` to the Fabric server directory.
6. Set `mcp_checkout` to the `pm-minecraft` checkout. The scenario guides install it beside PM.

Preflight creates editable instance files before it validates executables. Find them under `.runtime/simulation/instances/`.
Set `java` in the instance server YAML and `browser` in the instance MCP YAML
to executables on your PC. Use the scenario launcher with `-CheckServices` to
verify player connection and camera access in a disposable world.
