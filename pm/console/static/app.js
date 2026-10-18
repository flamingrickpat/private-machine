"use strict";
const $ = (id) => document.getElementById(id);
let instanceId = "",
  events = [],
  snapshot = {},
  nodes = [],
  cursor = 0,
  selectedNode = "",
  selectedEdge = null,
  memoryId = null,
  memoryRevision = null,
  creating = false,
  tab = "chat",
  pollBusy = false,
  unread = 0;
let chatRenderKey = "",
  currentInstance = null,
  commandBusy = false,
  searchBusy = false,
  vectorResults = null;
const backgroundCursors = new Map();
const sessions = JSON.parse(localStorage.getItem("pm.sessions") || "{}");
const pretty = (value) => JSON.stringify(value, null, 2);
function text(tag, value, className = "") {
  const element = document.createElement(tag);
  element.textContent = value;
  element.className = className;
  return element;
}
function error(exc) {
  $("error").textContent = exc.message || String(exc);
}
async function api(path, body) {
  const response = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok) throw Error(data.error);
  return data;
}
async function command(action, fields = {}) {
  if (!instanceId) throw Error("Select an instance.");
  $("error").textContent = "";
  commandBusy = true;
  updateControls();
  try {
    return await api("/api/command", {
      instance: instanceId,
      action,
      ...fields,
    });
  } finally {
    commandBusy = false;
    updateControls();
  }
}
// The persona names come from the worker snapshot. The console has none.
const userName = () => snapshot.user_name || "user";
const companionName = () => snapshot.companion_name || "companion";
const HISTORY_SESSION = "ghost-history";
function sessionList() {
  if (!sessions[instanceId])
    sessions[instanceId] = [
      { id: crypto.randomUUID(), user: userName(), label: `${userName()} / 1` },
    ];
  return sessions[instanceId];
}
function saveSessions() {
  localStorage.setItem("pm.sessions", JSON.stringify(sessions));
}
function renderSessions() {
  const old = $("session").value;
  $("session").replaceChildren();
  for (const s of sessionList()) {
    const o = text("option", s.label);
    o.value = s.id;
    $("session").append(o);
  }
  if (sessionList().some((s) => s.id === old)) $("session").value = old;
  const current = sessionList().find((s) => s.id === $("session").value);
  $("user").value = current.user;
  saveSessions();
}
async function listInstances() {
  const list = await api("/api/instances");
  $("instances").replaceChildren();
  for (const i of list) {
    const element = text(
      "div",
      `${i.name} Â· ${i.state}\n${i.cognition} Â· ${i.pid || "â€”"}`,
      `instance ${i.id === instanceId ? "selected" : ""}`,
    );
    element.onclick = () => select(i.id);
    $("instances").append(element);
  }
  return list;
}
async function select(id) {
  instanceId = id;
  events = [];
  cursor = 0;
  snapshot = {};
  currentInstance = null;
  vectorResults = null;
  searchBusy = false;
  updateControls();
  nodes = [];
  selectedNode = "";
  selectedEdge = null;
  memoryId = null;
  creating = false;
  $("memory-editor").value = "";
  renderSessions();
  await listInstances();
  await poll();
}
$("deploy").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const i = await api(
      "/api/deploy",
      Object.fromEntries(new FormData(event.target)),
    );
    await select(i.id);
  } catch (exc) {
    error(exc);
  }
};
document.querySelectorAll("[data-action]").forEach(
  (button) =>
    (button.onclick = async () => {
      try {
        if (
          button.dataset.action === "kill" &&
          !confirm(
            "Kill this worker and its child processes? Accepted checkpoint data remains. In-flight work can be interrupted.",
          )
        )
          return;
        await command(button.dataset.action);
        await listInstances();
      } catch (exc) {
        error(exc);
      }
    }),
);
$("refresh").onclick = () => command("snapshot").catch(error);
$("notify").onclick = async () => {
  if (!("Notification" in window)) {
    error(Error("This browser does not support desktop notifications."));
    return;
  }
  const permission = await Notification.requestPermission();
  $("notify").textContent = `Notifications: ${permission}`;
};
document.querySelectorAll("[data-tab]").forEach(
  (button) =>
    (button.onclick = () => {
      tab = button.dataset.tab;
      document
        .querySelectorAll(".panel")
        .forEach((p) => (p.hidden = p.id !== tab));
      document
        .querySelectorAll("[data-tab]")
        .forEach((b) => b.classList.toggle("active", b === button));
      if (tab === "chat") {
        unread = 0;
        $("unread").textContent = "";
      }
      render();
    }),
);
$("new-session").onclick = () => {
  if (!instanceId) return;
  const user = $("user").value.trim() || userName();
  const entry = {
    id: crypto.randomUUID(),
    user,
    label: `${user} / ${sessionList().length + 1}`,
  };
  sessionList().push(entry);
  renderSessions();
  $("session").value = entry.id;
  $("user").value = entry.user;
  renderChat();
};
$("session").onchange = () => {
  const current = sessionList().find((s) => s.id === $("session").value);
  $("user").value = current.user;
  renderChat();
};
$("send").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const user = $("user").value.trim();
    if (!user) throw Error("Enter a user name.");
    const s = sessionList().find((s) => s.id === $("session").value);
    s.user = user;
    s.label = `${user} / ${sessionList().indexOf(s) + 1}`;
    saveSessions();
    await command("chat", { session: s.id, user, text: $("message").value });
    $("message").value = "";
    await poll();
  } catch (exc) {
    error(exc);
  }
};
function renderChat() {
  if (currentInstance?.worker_state !== "ready") {
    $("transcript").replaceChildren(
      text("p", "Start the instance to read accepted dialogue from Ghost."),
    );
    chatRenderKey = "";
    return;
  }
  if ($("session").value === HISTORY_SESSION) {
    const history = (snapshot.history || []).filter((row) =>
      [userName(), companionName()].includes(row.speaker),
    );
    const key = `${instanceId}/history/${history.length}/${history.at(-1)?.id}`;
    if (key === chatRenderKey) return;
    chatRenderKey = key;
    $("transcript").replaceChildren();
    for (const row of history) {
      const entry = text(
        "div",
        "",
        `chat-entry ${row.speaker === companionName() ? "assistant" : "user"}`,
      );
      entry.append(
        text(
          "div",
          `${row.speaker} / ${row.timestamp} / knoxel ${row.id}`,
          "meta",
        ),
        text("div", row.text, "chat-text"),
      );
      $("transcript").append(entry);
    }
    $("transcript").scrollTop = $("transcript").scrollHeight;
    return;
  }
  const chatEvents = events.filter((e) =>
    ["chat.input", "chat.reply"].includes(e.kind),
  );
  const key = `${instanceId}/${$("session").value}/${chatEvents.at(-1)?.seq || 0}`;
  if (key === chatRenderKey) return;
  chatRenderKey = key;
  $("transcript").replaceChildren();
  for (const e of events.filter(
    (e) =>
      ["chat.input", "chat.reply"].includes(e.kind) &&
      (e.session === $("session").value || e.session === "unsolicited"),
  )) {
    const entry = text(
      "div",
      "",
      `chat-entry ${e.kind === "chat.reply" ? "assistant" : "user"}`,
    );
    entry.append(
      text(
        "div",
        `${e.user || companionName()} Â· ${new Date(e.time * 1000).toLocaleTimeString()} Â· ${e.correlation_id || ""}`,
        "meta",
      ),
      text("div", e.text, "chat-text"),
    );
    entry.onclick = () => {
      $("correlation").value = e.correlation_id || "";
      document.querySelector('[data-tab="network"]').click();
    };
    $("transcript").append(entry);
  }
  $("transcript").scrollTop = $("transcript").scrollHeight;
}
function filtered() {
  const correlation = $("correlation").value.trim(),
    search = $("event-search").value.toLowerCase();
  return events.filter(
    (e) =>
      (!correlation ||
        e.correlation_id === correlation ||
        e.run_id?.includes(correlation)) &&
      (!search || JSON.stringify(e).toLowerCase().includes(search)),
  );
}
const svgNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs) {
  const e = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function renderGraph() {
  const list = filtered();
  const bus = list.filter((e) => e.kind === "bus");
  const ids = [
    ...new Set([
      ...nodes.map((n) => n.id),
      ...bus.flatMap((e) => [e.source, e.recipient]),
    ]),
  ];
  const coords = new Map(
    ids.map((id, i) => [
      id,
      {
        user_input: { x: 20, y: 180 },
        main_cog: { x: 240, y: 180 },
        dialogue: { x: 470, y: 40 },
        assistant: { x: 470, y: 170 },
        self_management: { x: 470, y: 300 },
        managed_run: { x: 720, y: 300 },
        inner_thought: { x: 20, y: 40 },
        scheduler: { x: 720, y: 40 },
        status: { x: 720, y: 150 },
        tool_dialogue: { x: 720, y: 240 },
        steward: { x: 20, y: 300 },
      }[id] || { x: 720, y: 360 + i * 40 },
    ]),
  );
  $("graph").setAttribute("viewBox", "0 0 920 400");
  $("graph").replaceChildren();
  const defs = svg("defs", {});
  const marker = svg("marker", {
    id: "arrow",
    viewBox: "0 0 10 10",
    refX: 8,
    refY: 5,
    markerWidth: 6,
    markerHeight: 6,
    orient: "auto-start-reverse",
  });
  marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#87958b" }));
  defs.append(marker);
  $("graph").append(defs);
  const edges = new Map();
  for (const e of bus) {
    const key = `${e.source}|${e.recipient}`;
    if (!edges.has(key)) edges.set(key, e);
  }
  for (const e of edges.values()) {
    const a = coords.get(e.source),
      b = coords.get(e.recipient);
    const dx = b.x - a.x,
      dy = b.y - a.y;
    const scale = 1 / Math.max(Math.abs(dx) / 75, Math.abs(dy) / 27);
    const x1 = a.x + 70 + dx * scale,
      y1 = a.y + 22 + dy * scale;
    const x2 = b.x + 70 - dx * scale,
      y2 = b.y + 22 - dy * scale;
    const length = Math.hypot(dx, dy) || 1;
    const cx = (x1 + x2) / 2 - (dy / length) * 35;
    const cy = (y1 + y2) / 2 + (dx / length) * 35;
    const line = svg("path", {
      d: `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`,
      fill: "none",
      class: "edge",
      "marker-end": "url(#arrow)",
    });
    const tooltip = svg("title", {});
    tooltip.textContent = `${e.source} â†’ ${e.recipient}`;
    line.append(tooltip);
    line.onclick = () => {
      selectedNode = "";
      selectedEdge = [e.source, e.recipient];
      renderTrace();
    };
    $("graph").append(line);
  }
  for (const id of ids) {
    const p = coords.get(id);
    const last = events
      .filter((e) => e.node === id && e.kind.startsWith("module."))
      .at(-1);
    const active = last?.kind === "module.start";
    const group = svg("g", { class: `node ${active ? "active" : ""}` });
    group.append(
      svg("rect", { x: p.x, y: p.y, width: 140, height: 44, rx: 2 }),
    );
    const title = svg("text", { x: p.x + 7, y: p.y + 18 });
    title.textContent = id;
    const state = svg("text", { x: p.x + 7, y: p.y + 35 });
    state.textContent = active ? "ACTIVE" : "idle";
    group.append(title, state);
    group.onclick = () => {
      selectedNode = id;
      selectedEdge = null;
      showEvidence(nodes.find((node) => node.id === id) || { id });
      renderTrace();
    };
    $("graph").append(group);
  }
  renderTrace();
}
function renderTrace() {
  renderWorkflowGraph();
  let list = filtered();
  if (selectedEdge)
    list = list.filter(
      (e) =>
        e.kind === "bus" &&
        e.source === selectedEdge[0] &&
        e.recipient === selectedEdge[1],
    );
  if (selectedNode)
    list = list.filter(
      (e) =>
        e.node === selectedNode ||
        e.owner_module === selectedNode ||
        e.source === selectedNode ||
        e.recipient === selectedNode ||
        e.run_id?.startsWith(selectedNode),
    );
  $("detail-title").textContent = selectedNode
    ? `Node: ${selectedNode}`
    : selectedEdge
      ? `${selectedEdge[0]} â†’ ${selectedEdge[1]}`
      : "Trace";
  $("trace").replaceChildren();
  if (selectedNode) {
    const definition = nodes.find((n) => n.id === selectedNode);
    $("trace").append(text("pre", pretty(definition || { id: selectedNode })));
  }
  for (const e of list.slice(-200).reverse()) {
    const element = text(
      "div",
      `${e.seq} ${new Date(e.time * 1000).toLocaleTimeString()} ${e.kind} ${e.topic || e.node || ""} ${e.seconds !== undefined ? e.seconds.toFixed(3) + "s" : ""}`,
      "event",
    );
    element.onclick = () => {
      showEvidence(e);
    };
    $("trace").append(element);
  }
}
for (const id of ["correlation", "event-search"]) $(id).oninput = renderGraph;
$("clear-filter").onclick = () => {
  $("correlation").value = "";
  renderGraph();
};
// Keep machine vectors out of the editor without deleting accepted fields.
const vectorField = key => /embedding|(?:_vec|_vector|_vectors)$/.test(key);
function editableMemory(value) {
  if (Array.isArray(value)) return value.map(editableMemory);
  if (value && typeof value === "object") return Object.fromEntries(
    Object.entries(value).filter(([key]) => !vectorField(key))
      .map(([key, item]) => [key, editableMemory(item)]));
  return value;
}
function restoreVectors(edited, original) {
  if (!original || typeof original !== "object") return edited;
  for (const [key, value] of Object.entries(original)) {
    if (vectorField(key)) edited[key] = value;
    else if (edited[key] && typeof edited[key] === "object") restoreVectors(edited[key], value);
  }
  return edited;
}
function memorySubtype(record) {
  return Object.entries(record.data).filter(([key]) => key.endsWith("_type"))
    .map(([key, value]) => `${key}: ${value}`);
}
function memoryMatches(record) {
  return (!$("memory-type").value || record.__type__ === $("memory-type").value) &&
    (!$("memory-subtype").value || memorySubtype(record).includes($("memory-subtype").value));
}
function memoryOptions() {
  const records = Object.values(snapshot.records || {});
  for (const [id, values] of [
    ["memory-types", records.map(record => record.__type__)],
    ["memory-subtypes", records.filter(record => !$("memory-type").value || record.__type__ === $("memory-type").value).flatMap(memorySubtype)]]) {
    $(id).replaceChildren(...[...new Set(values)].sort().map(value => {
      const option = document.createElement("option"); option.value = value; return option;
    }));
  }
}
function renderMemory() {
  memoryOptions();
  if (currentInstance?.worker_state !== "ready") {
    $("memory-list").replaceChildren(
      text("p", "Start the instance to inspect its Ghost memory."),
    );
    return;
  }
  if (!snapshot.records) {
    $("memory-list").replaceChildren(
      text("p", "Ghost is loading accepted memoryâ€¦"),
    );
    return;
  }
  if ($("search-mode").value === "vector" && vectorResults) {
    $("memory-list").replaceChildren();
    for (const match of vectorResults.matches) {
      if (!memoryMatches(match.record)) continue;
      const e = text(
        "div",
        `${match.id} / ${match.record.__type__} / similarity ${match.score.toFixed(4)}\n${match.record.data.content}`,
        "record",
      );
      e.onclick = () => {
        memoryId = String(match.id);
        memoryRevision = match.revision;
        creating = false;
        $("memory-editor").value = pretty(editableMemory(match.record));
        updateControls();
      };
      $("memory-list").append(e);
    }
    return;
  }

  const query = $("memory-search").value.toLowerCase();
  $("memory-list").replaceChildren();
  let shown = 0,
    total = 0;
  for (const [id, record] of Object.entries(snapshot.records || {})) {
    if (!memoryMatches(record)) continue;
    if (
      !`${id} ${record.__type__} ${record.data.content || ""}`
        .toLowerCase()
        .includes(query)
    )
      continue;
    total++;
    if (shown++ >= 250) continue;
    const element = text(
      "div",
      `${id} Â· ${record.__type__} Â· r${snapshot.revisions[id]}\n${String(record.data.content || "").slice(0, 220)}`,
      "record",
    );
    element.onclick = () => {
      memoryId = id;
      memoryRevision = snapshot.revisions[id];
      creating = false;
      updateControls();
      $("memory-editor").value = pretty(editableMemory(record));
    };
    $("memory-list").append(element);
  }
  $("memory-list").prepend(
    text(
      "p",
      `${total} matches ? first 250 shown. Refine the search for more records.`,
    ),
  );
}
for (const id of ["memory-type", "memory-subtype"]) $(id).oninput = () => {
  renderMemory(); updateControls();
};
$("memory-search").oninput = () => {
  renderMemory();
  updateControls();
};
$("new-memory").onclick = () => {
  try {
    const record = JSON.parse($("memory-editor").value);
    record.data.id = -1;
    record.data.tick_id = -1;
    creating = true;
    memoryId = null;
    $("memory-editor").value = pretty(editableMemory(record));
  } catch {
    error(Error("Select a record as the template first."));
  }
};
$("save-memory").onclick = async () => {
  try {
    if (!creating && !memoryId) throw Error("Select a record first.");
    await command("memory", {
      record: creating ? JSON.parse($("memory-editor").value) : restoreVectors(
        JSON.parse($("memory-editor").value), snapshot.records[memoryId]),
      create: creating,
      revision: creating ? null : memoryRevision,
    });
    await poll();
    if (creating) {
      const receipt = events.findLast(
        (e) => e.kind === "memory.commit",
      )?.receipt;
      memoryId = String(receipt?.added_ids[0] || "");
    }
    creating = false;
    if (memoryId && snapshot.records?.[memoryId]) {
      memoryRevision = snapshot.revisions[memoryId];
      $("memory-editor").value = pretty(editableMemory(snapshot.records[memoryId]));
    }
  } catch (exc) {
    error(exc);
  }
};
function renderTiming() {
  const aggregates = new Map();
  for (const e of events.filter((e) =>
    ["model.end", "model.error", "workflow.end", "workflow.error"].includes(
      e.kind,
    ),
  )) {
    const key = `${e.kind.split(".")[0]} / ${e.node}`;
    const row = aggregates.get(key) || {
      count: 0,
      total: 0,
      max: 0,
      errors: 0,
    };
    row.count++;
    row.total += e.seconds;
    row.max = Math.max(row.max, e.seconds);
    if (e.kind.endsWith("error")) row.errors++;
    aggregates.set(key, row);
  }
  const table = document.createElement("table");
  const heading = document.createElement("tr");
  for (const name of ["Work", "Calls", "Total s", "Mean s", "Max s", "Errors"])
    heading.append(text("th", name));
  table.append(heading);
  for (const [key, r] of [...aggregates].sort(
    (a, b) => b[1].total - a[1].total,
  )) {
    const row = document.createElement("tr");
    for (const value of [
      key,
      r.count,
      r.total.toFixed(3),
      (r.total / r.count).toFixed(3),
      r.max.toFixed(3),
      r.errors,
    ])
      row.append(text("td", value));
    row.onclick = () => {
      selectedNode = key.split(" / ")[1];
      document.querySelector('[data-tab="network"]').click();
    };
    table.append(row);
  }
  $("timing").replaceChildren(table);
}
function renderRuns() {
  $("run-list").replaceChildren();
  for (const [id, run] of Object.entries(snapshot.runs || {})) {
    const block = text("pre", pretty(run));
    block.onclick = () => {
      $("run-control").elements.run_id.value = id;
      $("run-control").elements.question_id.value = run.question_id || "";
    };
    $("run-list").append(block);
  }
}
$("run-control").onsubmit = async (event) => {
  event.preventDefault();
  try {
    await command(
      "run_control",
      Object.fromEntries(new FormData(event.target)),
    );
    await command("snapshot");
  } catch (exc) {
    error(exc);
  }
};
// One record line: id, type, and the start of its content.
function recordLine(id) {
  const wrapped = (snapshot.records || {})[id];
  if (!wrapped) return `${id} (not in snapshot)`;
  const content = String(wrapped.data.content || "").replace(/\s+/g, " ");
  return `${id} ${wrapped.__type__}: ${content.slice(0, 160)}`;
}
function cycleSection(title, lines) {
  const block = text("details", "");
  block.append(text("summary", `${title} (${lines.length})`));
  block.append(text("pre", lines.join("\n") || "(none)"));
  return block;
}
// The newest cycle state and its turn record, as the worker snapshot holds them.
function renderCycle() {
  const view = $("cycle-view");
  view.replaceChildren();
  const cycle = snapshot.cycle;
  if (!cycle) {
    view.append(text("p", "No accepted cycle yet."));
    return;
  }
  const turn = (snapshot.gwt || {})[cycle.tick_id] || {};
  const workspace = cycle.workspace || {};
  const shards = [
    workspace.self_shard,
    ...Object.values(workspace.entity_shards || {}),
    ...Object.values(workspace.workflow_shards || {}),
  ].filter(Boolean);
  view.append(text("h3", `Tick ${cycle.tick_id}`));
  for (const shard of shards) {
    const name = `${shard.key.kind}${shard.key.ref ? ":" + shard.key.ref : ""}`;
    view.append(
      cycleSection(`Shard ${name} items`, shard.items.map(
        (item) => `${recordLine(item.record_id)}  [${item.required_by || item.reason}]`)),
      cycleSection(`Shard ${name} tail`, shard.tail_ids.map(recordLine)),
    );
  }
  const csm = Object.values((cycle.csm || {}).items || {}).sort((a, b) => b.activation - a.activation);
  view.append(
    cycleSection("CSM items", csm.map((item) =>
      `${item.activation.toFixed(2)}  ${recordLine(item.record_id)}  <- ${item.endorsers.join(", ")}`)),
    cycleSection("Structures", (cycle.structure_ids || []).map(recordLine)),
    cycleSection("Codelet runs", ((turn.trace || {}).codelet_runs || []).map((run) =>
      `wave ${run.wave} ${run.codelet} (${run.mode}) ${run.status} score ${Number(run.score).toFixed(2)} ${run.reason}`)),
    cycleSection("Stimulus groups", (cycle.stimulus_groups || []).map((group) =>
      `${group.key.kind}:${group.key.ref} ${group.category} ${group.insignificant ? "insignificant" : ""} ` +
      `stimuli ${group.stimulus_ids.join(", ")}`)),
    cycleSection("Decisions", (cycle.decisions || []).map((item) => `${item.group.kind}:${item.group.ref}  ${item.text}`)),
    cycleSection("Routed actions", (cycle.dispatches || []).map((item) =>
      `${item.group.kind}:${item.group.ref} -> ${item.workflow} ${item.action_type} action ${item.action_id}`)),
  );
}
function render() {
  updateControls();
  if (tab === "chat") renderChat();
  if (tab === "cycle") renderCycle();
  if (tab === "network") renderGraph();
  if (tab === "memory") renderMemory();
  if (tab === "performance") renderTiming();
  if (tab === "runs") renderRuns();
}
async function poll() {
  if (pollBusy || !instanceId) return;
  pollBusy = true;
  try {
    const requestedId = instanceId;
    const data = await api(`/api/instances/${requestedId}?after=${cursor}`);
    if (requestedId !== instanceId) return;
    const initial = cursor === 0;
    const fresh = data.events;
    events.push(...fresh);
    events = events.slice(-10000);
    cursor = fresh.at(-1)?.seq || cursor;
    let restored = false;
    for (const entry of fresh.filter((e) => e.kind === "chat.input")) {
      if (!sessionList().some((s) => s.id === entry.session)) {
        sessionList().push({
          id: entry.session,
          user: entry.user,
          label: `${entry.user} / recovered`,
        });
        restored = true;
      }
    }
    if (restored) {
      renderSessions();
    }
    currentInstance = data.instance;
    if (data.snapshot && data.instance.worker_state === "ready")
      snapshot = data.snapshot;
    if (
      snapshot.history &&
      !sessionList().some((s) => s.id === HISTORY_SESSION)
    ) {
      sessionList().unshift({
        id: HISTORY_SESSION,
        user: userName(),
        label: `${userName()} / accepted history`,
      });
      renderSessions();
      $("session").value = HISTORY_SESSION;
      $("user").value = userName();
    }
    nodes = data.nodes;
    $("selected").textContent =
      `${data.instance.name} / ${data.instance.state} (${data.instance.worker_state}) / ${data.instance.cognition} / PID ${data.instance.pid || "â€”"}`;
    $("log").textContent = data.log;
    $("connection").textContent = "Connected Â· 1s poll";
    for (const e of fresh) {
      if (e.kind === "ready") {
        searchBusy = false;
        vectorResults = null;
      }
      if (e.kind === "memory.index") {
        $("search-status").textContent =
          `Embedding index: ${e.indexed} / ${e.total}`;
      }
      if (e.kind === "memory.search") {
        searchBusy = false;
        vectorResults = e.result || null;
        $("search-status").textContent =
          e.error ||
          `${e.result.indexed} / ${e.result.total} records indexed (${e.result.missing} missing)`;
        if (e.error && !initial) error(Error(e.error));
      }
      if (e.kind === "history") {
        snapshot.history = e.history;
        chatRenderKey = "";
      }
      if (!initial && (e.kind === "application.error" || e.kind === "rejected"))
        error(Error(e.error));
      if (e.kind === "chat.reply" && !initial) {
        if (
          tab !== "chat" ||
          e.session !== $("session").value ||
          document.hidden
        ) {
          unread++;
          $("unread").textContent = `${unread} new replies`;
          if ("Notification" in window && Notification.permission === "granted")
            new Notification(`${data.instance.name} sent a message`, {
              body: e.text.slice(0, 180),
            });
        }
      }
    }
    render();
  } catch (exc) {
    $("connection").textContent = "Connection error";
    error(exc);
  } finally {
    pollBusy = false;
  }
}
setInterval(poll, 1000);
setInterval(backgroundPoll, 5000);
listInstances()
  .then((list) => {
    if (list.length) select(list[0].id);
  })
  .catch(error);

function renderWorkflowGraph() {
  const container = $("workflow-graph");
  container.replaceChildren();
  const starts = filtered().filter(
    (e) =>
      e.kind === "workflow.start" &&
      (!selectedNode ||
        e.owner_module === selectedNode ||
        e.node === selectedNode),
  );
  const latest = new Map();
  for (const e of starts) latest.set(e.node, e);
  const items = [...latest.values()];
  const roots = items.filter(
    (e) => !e.parent_node || !latest.has(e.parent_node),
  );
  const children = items.filter(
    (e) => e.parent_node && latest.has(e.parent_node),
  );
  const positions = new Map();
  roots.forEach((e, i) => positions.set(e.node, { x: 20 + i * 200, y: 20 }));
  children.forEach((e, i) =>
    positions.set(e.node, { x: 20 + i * 200, y: 145 }),
  );
  const width = Math.max(1000, Math.max(roots.length, children.length) * 200);
  container.setAttribute(
    "viewBox",
    `0 0 ${width} ${children.length ? 220 : 100}`,
  );
  for (const e of children) {
    const parent = positions.get(e.parent_node),
      child = positions.get(e.node);
    const link = svg("path", {
      d: `M ${parent.x + 82} ${parent.y + 52} V 110 H ${child.x + 82} V ${child.y}`,
      fill: "none",
      class: "edge",
    });
    link.onclick = () => {
      $("detail").textContent = pretty({
        parent_run: e.parent_run,
        child_run: e.run_id,
        input: e.input,
      });
    };
    container.append(link);
  }
  items.forEach((e) => {
    const { x, y } = positions.get(e.node);
    const end = events.findLast(
      (v) =>
        v.run_id === e.run_id &&
        ["workflow.end", "workflow.error"].includes(v.kind),
    );
    const group = svg("g", { class: `node ${end ? "" : "active"}` });
    group.append(svg("rect", { x, y, width: 165, height: 52 }));
    const name = svg("text", { x: x + 7, y: y + 19 });
    name.textContent = e.node;
    const status = svg("text", { x: x + 7, y: y + 38 });
    status.textContent = end
      ? `${end.status || "error"} / ${end.seconds.toFixed(3)}s`
      : "ACTIVE";
    group.append(name, status);
    group.onclick = () => {
      showEvidence({ start: e, result: end || null });
    };
    container.append(group);
  });
}
async function backgroundPoll() {
  try {
    const list = await listInstances();
    for (const i of list) {
      if (i.id === instanceId) {
        backgroundCursors.set(i.id, cursor);
        continue;
      }
      if (i.state !== "running") continue;
      const previous = backgroundCursors.get(i.id);
      const data = await api(
        `/api/instances/${i.id}?after=${previous || 0}&light=1`,
      );
      backgroundCursors.set(i.id, data.events.at(-1)?.seq || previous || 0);
      if (previous === undefined) continue;
      for (const e of data.events.filter((e) => e.kind === "chat.reply")) {
        unread++;
        $("unread").textContent = `${unread} new replies across instances`;
        if ("Notification" in window && Notification.permission === "granted")
          new Notification(`${i.name} sent a message`, {
            body: e.text.slice(0, 180),
          });
      }
    }
  } catch (exc) {
    error(exc);
  }
}

function updateControls() {
  const live = currentInstance?.state === "running",
    ready = currentInstance?.worker_state === "ready";
  document.querySelector("[data-action=start]").disabled =
    !instanceId || live || commandBusy;
  document.querySelector("[data-action=stop]").disabled =
    !live || currentInstance?.worker_state === "stopping" || commandBusy;
  document.querySelector("[data-action=kill]").disabled = !live || commandBusy;
  for (const id of ["refresh", "new-session", "vector-search", "prepare-index"])
    $(id).disabled =
      !ready ||
      commandBusy ||
      ((id.includes("search") || id === "prepare-index") &&
        (searchBusy || !$("memory-search").value.trim()));
  $("save-memory").disabled = !ready || commandBusy || (!memoryId && !creating);
  $("new-memory").disabled = !ready || commandBusy || !memoryId;
  $("cancel-search").disabled = !ready || !searchBusy || commandBusy;
  $("send").querySelector("button").disabled =
    !ready || commandBusy || !$("message").value.trim() || exchangePending();
  $("run-control").querySelector("button").disabled =
    !ready || commandBusy || !$("run-control").elements.run_id.value;
}
function exchangePending() {
  const inputs = events.filter((e) => e.kind === "chat.input");
  return inputs.some(
    (e) =>
      !events.some(
        (r) => r.kind === "chat.reply" && r.correlation_id === e.correlation_id,
      ) && e.seq > (events.findLast((e) => e.kind === "ready")?.seq || 0),
  );
}
$("message").oninput = updateControls;
$("run-control").oninput = updateControls;
$("search-mode").onchange = () => {
  vectorResults = null;
  renderMemory();
  updateControls();
};
async function vectorSearch(prepare = false) {
  try {
    $("search-mode").value = "vector";
    searchBusy = true;
    $("search-status").textContent = prepare
      ? "Embedding missing records. External requests can take time."
      : "Searching cached vectors?";
    await command("search", { query: $("memory-search").value, prepare,
      record_type: $("memory-type").value, subtype: $("memory-subtype").value });
  } catch (e) {
    searchBusy = false;
    error(e);
  }
  updateControls();
}
$("vector-search").onclick = () => vectorSearch();
$("prepare-index").onclick = () => vectorSearch(true);
$("cancel-search").onclick = () => command("cancel_search").catch(error);
$("check-connections").onclick = async () => {
  const button = $("check-connections");
  button.disabled = true;
  try {
    const data = await api("/api/connections", { instance: instanceId });
    $("connection-status").replaceChildren(
      text("p", `Configuration: ${data.source}`),
    );
    for (const row of data.connections)
      $("connection-status").append(text("pre", pretty(row)));
  } catch (e) {
    error(e);
  } finally {
    button.disabled = false;
  }
};
function showEvidence(value) {
  $("detail").textContent = pretty(value);
  $("source-links").replaceChildren();
  const paths = new Set();
  function find(v, key = "") {
    if (
      typeof v === "string" &&
      ["workflow", "skill"].includes(key) &&
      /\.(md|py|yaml|yml)$/.test(v)
    )
      paths.add(v);
    else if (v && typeof v === "object")
      for (const [k, child] of Object.entries(v)) find(child, k);
  }
  find(value);
  for (const path of paths) {
    const link = text("a", `Open ${path.split(/[\\/]/).at(-1)}`);
    link.href = "#";
    link.onclick = async (e) => {
      e.preventDefault();
      try {
        await api("/api/open", { path });
      } catch (e) {
        error(e);
      }
    };
    $("source-links").append(link, text("span", " / "));
  }
}
updateControls();
