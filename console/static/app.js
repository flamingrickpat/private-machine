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
let chatRenderKey = "";
const backgroundCursors = new Map();
const sessions = JSON.parse(localStorage.getItem("spamuel.sessions") || "{}");
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
  return api("/api/command", { instance: instanceId, action, ...fields });
}
function sessionList() {
  if (!sessions[instanceId])
    sessions[instanceId] = [
      { id: crypto.randomUUID(), user: "Rick", label: "Rick / 1" },
    ];
  return sessions[instanceId];
}
function saveSessions() {
  localStorage.setItem("spamuel.sessions", JSON.stringify(sessions));
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
      `${i.name} · ${i.state}\n${i.cognition} · ${i.pid || "—"}`,
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
  const user = $("user").value.trim() || "Rick";
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
        `${e.user || "Spamuel"} · ${new Date(e.time * 1000).toLocaleTimeString()} · ${e.correlation_id || ""}`,
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
        x: 430 + 340 * Math.cos((2 * Math.PI * i) / ids.length - Math.PI / 2),
        y: 250 + 215 * Math.sin((2 * Math.PI * i) / ids.length - Math.PI / 2),
      },
    ]),
  );
  $("graph").setAttribute("viewBox", "0 0 1000 550");
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
    tooltip.textContent = `${e.source} → ${e.recipient}`;
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
      ? `${selectedEdge[0]} → ${selectedEdge[1]}`
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
      $("detail").textContent = pretty(e);
    };
    $("trace").append(element);
  }
}
for (const id of ["correlation", "event-search"]) $(id).oninput = renderGraph;
$("clear-filter").onclick = () => {
  $("correlation").value = "";
  renderGraph();
};
function renderMemory() {
  const query = $("memory-search").value.toLowerCase();
  $("memory-list").replaceChildren();
  let shown = 0,
    total = 0;
  for (const [id, record] of Object.entries(snapshot.records || {})) {
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
      `${id} · ${record.__type__} · r${snapshot.revisions[id]}\n${String(record.data.content || "").slice(0, 220)}`,
      "record",
    );
    element.onclick = () => {
      memoryId = id;
      memoryRevision = snapshot.revisions[id];
      creating = false;
      $("memory-editor").value = pretty(record);
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
$("memory-search").oninput = renderMemory;
$("new-memory").onclick = () => {
  try {
    const record = JSON.parse($("memory-editor").value);
    record.data.id = -1;
    record.data.tick_id = -1;
    creating = true;
    memoryId = null;
    $("memory-editor").value = pretty(record);
  } catch {
    error(Error("Select a record as the template first."));
  }
};
$("save-memory").onclick = async () => {
  try {
    if (!creating && !memoryId) throw Error("Select a record first.");
    await command("memory", {
      record: JSON.parse($("memory-editor").value),
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
      $("memory-editor").value = pretty(snapshot.records[memoryId]);
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
function render() {
  if (tab === "chat") renderChat();
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
      if (
        initial &&
        !events.some(
          (e) => e.kind === "chat.input" && e.session === $("session").value,
        )
      ) {
        const latest = fresh.findLast((e) => e.kind === "chat.input");
        if (latest) {
          $("session").value = latest.session;
          $("user").value = latest.user;
        }
      }
    }
    if (data.snapshot) snapshot = data.snapshot;
    nodes = data.nodes;
    $("selected").textContent =
      `${data.instance.name} / ${data.instance.state} (${data.instance.worker_state}) / ${data.instance.cognition} / PID ${data.instance.pid || "—"}`;
    $("log").textContent = data.log;
    $("connection").textContent = "Connected · 1s poll";
    for (const e of fresh) {
      if (e.kind === "application.error" || e.kind === "rejected")
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
      $("detail").textContent = pretty({ start: e, result: end || null });
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
