# Install Minecraft mods and MCP for a simulation

PM starts Fabric and the Minecraft MCP when you launch a Minecraft scenario.
Each launch gets its own world, game port, and player. The launcher connects
Spamuel to the MCP automatically.

## Put the mods in your Fabric server folder

Start with your installed Fabric 1.19.4 server and Java 17 or newer. Keep
`fabric-server-launch.jar`, `server.jar`, `fabric-server-launcher.properties`,
and `libraries` in the server folder. Use the mod JARs for Minecraft 1.19.4.
The overnight baseline used [Fabric API 0.87.2](https://github.com/FabricMC/fabric-api/releases/tag/0.87.2%2B1.19.4)
and [Agentic Babymode 0.6.0](https://github.com/flamingrickpat/pm-agentic-babymode/tree/99fe2aa428bee9d833693989585f3cf728abb18d).
The Agentic Babymode link opens the source and its build instructions.

Open PowerShell in the `private-machine` folder. Paste the commands and enter
the three paths when PowerShell asks for them. Use a Fabric folder with one
version of each mod.

```powershell
$fabric = (Resolve-Path -LiteralPath (Read-Host 'Path to the installed Fabric server folder')).Path
$fabricApi = (Resolve-Path -LiteralPath (Read-Host 'Path to the Fabric API JAR')).Path
$babymode = (Resolve-Path -LiteralPath (Read-Host 'Path to the Agentic Babymode JAR')).Path
$mods = Join-Path $fabric 'mods'
New-Item -ItemType Directory -Force $mods | Out-Null
foreach ($jar in $fabricApi, $babymode) {
    $installed = Join-Path $mods (Split-Path $jar -Leaf)
    if ($jar -ne $installed) { Copy-Item -LiteralPath $jar -Destination $mods -ErrorAction Stop }
}
notepad (Join-Path $fabric 'eula.txt')
```

If you accept the Minecraft EULA, save the line `eula=true` in `eula.txt`.

## Install PM and the MCP

Install [Node.js 20 or newer](https://nodejs.org/en/download). Paste the next block
in the same PowerShell window. It installs PM and builds the Minecraft MCP in a
sibling folder.

```powershell
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'PM dependency installation failed.' }
if (-not (Test-Path -LiteralPath '../pm-minecraft')) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
    if ($LASTEXITCODE -ne 0) { throw 'Minecraft MCP clone failed.' }
}
& ../pm-minecraft/setup.ps1
```

The [Minecraft MCP project](https://github.com/flamingrickpat/pm-minecraft#setup)
explains its standalone setup. For a simulation, the PM launcher starts it for you.

## Link the Fabric folder to PM

Paste this block in the same PowerShell window. If the configuration already
exists, update `server_installation` in Notepad. Keep `mcp_checkout: ../pm-minecraft`
if you installed the MCP with the preceding block.

```powershell
if (-not (Test-Path -LiteralPath config_simulation.yaml)) {
    $fabricYamlPath = $fabric.Replace('\', '/').Replace("'", "''")
    $example = Get-Content -LiteralPath config_simulation.yaml.example -Raw
    $example.Replace('../minecraft-server-real', "'$fabricYamlPath'") |
        Set-Content -LiteralPath config_simulation.yaml -Encoding UTF8
}
notepad config_simulation.yaml
```

Choose [Jesse and Spamuel](simulation-jesse-spamuel-minecraft.md) or
[two Spamuels](simulation-two-spamuels-minecraft.md). Each guide runs a preflight
and a service check before it starts the simulation.

## Fix a preflight error about Java or the browser

Preflight makes two configuration files in the instance folder that it prints.
For the Jesse and Spamuel guide's `first` run, open them with these commands.

```powershell
$instance = '.runtime/simulation/instances/first-jesse-spamuel-minecraft'
notepad "$instance/config_minecraft_server.yaml"
notepad "$instance/config_pm_minecraft.yaml"
```

In `config_minecraft_server.yaml`, replace the value of `java` with the full
path to your `java.exe`. In `config_pm_minecraft.yaml`, replace the value of
`browser` with the full path to your Chromium browser. Use forward slashes in
both paths. Keep the other settings and rerun preflight with the same run ID.
