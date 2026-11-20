/* K003 execution timeline. One file, no build step, no external script.

   Routes are hash paths so the browser back button works and a view can be
   linked: #/run/<run>/<attempt>/<tab> and #/run/<run>/<attempt>/<tab>/<detail>.
   The page reads only evidence that a completed simulation attempt wrote. */
"use strict";

const THEME_KEY = "pm-workspace-theme";
const EMBEDDED = new URLSearchParams(location.search).get("embedded") === "1";
const API_ROOT = location.pathname.startsWith("/debugger/") ? "/debugger/api" : "/api";
const $ = (selector) => document.querySelector(selector);
const cache = new Map();
let renderToken = 0;
let state = {
  runs: [],
  run: null,
  attemptPage: null,
  participant: "",
  memorySource: "ghost.pickle",
  timelineKinds: new Set(["speech", "actor", "delivery", "call", "workflow", "cycle", "fact", "failure"]),
  selectedEvent: null,
};

/* ------------------------------------------------------------------ dom */

function h(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function short(value, limit) {
  const text = String(value === null || value === undefined ? "" : value);
  const max = limit || 180;
  return text.length <= max ? text : text.slice(0, max) + "\u2026";
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "-";
  return Number(value).toLocaleString("en-US");
}

function time(value) {
  if (!value) return "-";
  return String(value).replace("T", " ").replace("+00:00", "Z").replace(/(\.\d{3})\d+/, "$1");
}

function badge(label, tone) {
  return h("span", { class: "badge " + (tone || "") }, label);
}

function kv(rows) {
  const cells = [];
  for (const [name, value] of rows) {
    cells.push(h("div", {}, name), h("div", {}, value === null || value === undefined || value === "" ? "-" : String(value)));
  }
  return h("div", { class: "kv" }, cells);
}

function jsonBlock(value, cls) {
  return h("pre", { class: cls || "" }, JSON.stringify(value, null, 2));
}

function notice(message) {
  const box = $("#notice");
  box.textContent = String(message);
  box.classList.remove("hidden");
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => box.classList.add("hidden"), 9000);
}

async function api(path, cacheResponse = true) {
  if (cacheResponse && cache.has(path)) return cache.get(path);
  const response = await fetch(API_ROOT + path);
  let data;
  try {
    data = await response.json();
  } catch (error) {
    data = { error: "no answer from the timeline server" };
  }
  if (!response.ok || data.error) throw new Error(data.error || "HTTP " + response.status);
  if (cacheResponse) cache.set(path, data);
  return data;
}

/* --------------------------------------------------------------- routing */

function route() {
  const raw = location.hash.replace(/^#\/?/, "");
  if (!raw) return { page: "runs" };
  const parts = raw.split("/").filter(Boolean).map((part) => decodeURIComponent(part));
  if (parts[0] !== "run" || !parts[1]) return { page: "runs" };
  return {
    page: "run",
    run: parts[1],
    attempt: Number(parts[2]) || 0,
    tab: parts[3] || "overview",
    detail: parts.slice(4),
  };
}

function go(path) {
  location.hash = "#/" + path;
}

function runPath(extra) {
  const current = route();
  return `run/${current.run}/${current.attempt}/${current.tab}${extra ? "/" + extra : ""}`;
}

function openRun(runId, attempt) {
  go(`run/${runId}/${attempt || 1}/overview`);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme").textContent = theme === "dark" ? "light" : "dark";
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch (error) {
    if (error instanceof DOMException && ["SecurityError", "QuotaExceededError"].includes(error.name)) return;
    throw error;
  }
}

function savedTheme() {
  try {
    return window.localStorage.getItem(THEME_KEY);
  } catch (error) {
    if (error instanceof DOMException && error.name === "SecurityError") return null;
    throw error;
  }
}

/* ---------------------------------------------------------------- render */

async function render(refresh = false) {
  const token = ++renderToken;
  const current = route();
  document.body.dataset.tab = current.tab || "";
  try {
    if (refresh === true) cache.clear();
    state.runs = await api("/runs");
    if (token !== renderToken) return;
    renderRunSelect();
    if (current.page === "runs") {
      $("#meta").textContent = "Simulation runs";
      return renderRunList();
    }
    if (refresh === true || state.run?.summary.id !== current.run || state.attemptPage?.attempt.number !== current.attempt) {
      const summary = state.runs.find(row => row.id === current.run);
      $("#meta").textContent = summary ? `${summary.title} · ${summary.status}` : current.run;
      clear($("#detail"));
      clear($("#sidebar"));
      clear($("#tab")).append(h("h1", {}, summary?.title || current.run),
        h("p", {}, `Loading attempt ${current.attempt || "latest"} records, calls and workflows…`));
    }
    if (refresh === true) await api("/refresh", false);
    if (token !== renderToken) return;
    const run = await api(`/run/${encodeURIComponent(current.run)}`);
    if (token !== renderToken) return;
    const attempt = current.attempt || run.summary.latest_attempt || 1;
    if (attempt !== current.attempt) {
      go(`run/${current.run}/${attempt}/${current.tab}`);
      return;
    }
    renderAttemptSelect(run);
    renderTabs(current);
    const attemptPage = await api(`/attempt?run=${encodeURIComponent(current.run)}&attempt=${attempt}`);
    if (token !== renderToken) return;
    state.run = run;
    state.attemptPage = attemptPage;
    if (state.participant && !attemptPage.overviews.some((row) => row.id === state.participant)) state.participant = "";
    if (!state.participant && attemptPage.overviews.length === 1) state.participant = attemptPage.overviews[0].id;
    renderParticipantSelect(attemptPage);
    $("#attempt-select").disabled = ["chat", "records"].includes(current.tab) && state.memorySource !== "capture";
    renderHeaderMeta();
    renderSidebar(run, attemptPage);
    renderTabs(current);
    await renderTab(current, token);
    if (token !== renderToken) return;
    await renderDetail(current, token);
  } catch (error) {
    if (token !== renderToken) return;
    $("#meta").textContent = "Could not load debugger data";
    renderRunList(error.message);
  }
}

function renderRunSelect() {
  const select = $("#run-select");
  clear(select);
  select.append(h("option", { value: "" }, "choose a run"));
  for (const run of state.runs) {
    select.append(h("option", { value: run.id }, `${run.id} · ${short(run.title, 44)}`));
  }
  select.value = route().run || "";
  select.onchange = () => {
    const run = state.runs.find((row) => row.id === select.value);
    if (run) openRun(run.id, run.latest_attempt || 1);
  };
}

function renderAttemptSelect(run) {
  const select = $("#attempt-select");
  clear(select);
  for (const attempt of run.attempts) {
    select.append(h("option", { value: String(attempt.number) },
      `attempt ${attempt.number}${attempt.status ? " · " + attempt.status : ""}${attempt.failures ? " · " + attempt.failures + " fail" : ""}`));
  }
  select.value = String(route().attempt);
  select.onchange = () => go(`run/${route().run}/${select.value}/${route().tab}`);
}

function renderParticipantSelect(attemptPage) {
  const select = $("#participant-select");
  clear(select);
  select.append(h("option", { value: "" }, "all"));
  for (const row of attemptPage.overviews) select.append(h("option", { value: row.id }, row.id));
  select.value = state.participant;
  select.onchange = () => {
    state.participant = select.value;
    state.selectedEvent = null;
    render();
  };
}

function renderHeaderMeta() {
  const page = state.attemptPage;
  $("#meta").textContent = `${state.run.summary.title} · attempt ${page.attempt.number} · ${page.attempt.status || "unknown"}`;
}

function renderRunList(error) {
  clear($("#sidebar"));
  clear($("#detail"));
  $("#tabs").replaceChildren();
  const tab = clear($("#tab"));
  if (error) tab.append(h("p", { class: "noticeline" }, error));
  tab.append(h("h1", {}, "Choose a simulation run"));
  tab.append(h("p", { class: "dim" }, "Each row is a run folder under .runtime/simulation/runs. Open an attempt to read its conversation, calls, workflows, cycles, and records."));
  const list = h("div", {});
  for (const run of state.runs) {
    const card = h("div", { class: "run-card", onclick: () => openRun(run.id, run.latest_attempt || 1) },
      h("div", { class: "name" }, run.title || run.id),
      h("div", { class: "dim tiny" }, `${run.id} · ${run.status || "unknown"} · ${run.attempt_count} attempt(s) · ${run.participant_count} participant(s) · ${run.failure_count} failure(s)`));
    list.append(card);
  }
  tab.append(list);
}

function renderSidebar(run, page) {
  const side = clear($("#sidebar"));
  side.append(h("div", { class: "pad" },
    h("h1", {}, run.summary.title || run.summary.id),
    h("div", { class: "tiny dim" }, `${run.summary.id} · ${run.summary.status || "unknown"}`),
    h("div", { class: "tiny dim" }, `created ${time(run.summary.created_at)}`),
    h("div", { class: "tiny dim mono" }, run.summary.config || ""),
    h("p", {}, run.plan.brief || ""),
  ));
  side.append(h("div", { class: "pad" },
    h("h2", {}, "Attempts"),
    ...run.attempts.map((attempt) => h("div", { class: "card", onclick: () => go(`run/${run.summary.id}/${attempt.number}/${route().tab}`) },
      h("b", {}, `attempt ${attempt.number}`), " ", badge(attempt.status || "unknown", attempt.status === "completed" ? "" : "warning"),
      h("div", { class: "tiny dim" }, `${attempt.evidence_participants.length} captured participant(s) · ${attempt.failures} failure(s)`))),
  ));
  side.append(h("div", { class: "pad" },
    h("h2", {}, "Participants"),
    ...page.overviews.map((row) => h("div", { class: "card", onclick: () => { state.participant = row.id; $("#participant-select").value = row.id; render(); } },
      h("b", {}, row.id),
      h("div", { class: "tiny dim" }, `${fmt(row.records)} records · ${fmt(row.calls)} calls · ${fmt(row.workflows)} workflows · ${fmt(row.cycles)} cycles`),
      h("div", { class: "tiny dim" }, `${fmt(row.prompt_tokens)} prompt tok · ${fmt(row.completion_tokens)} completion tok · ${fmt(row.errors)} call errors`))),
  ));
  if (page.failures.length) {
    side.append(h("div", { class: "pad" }, h("h2", {}, "Failures"),
      ...page.failures.slice(0, 12).map((failure) => h("div", { class: "card" },
        h("div", { class: "tiny mono" }, `attempt ${failure.attempt || "?"}`),
        h("div", { class: "tiny" }, short(failure.error || failure.cleanup_error || JSON.stringify(failure), 500))))));
  }
  if (Object.keys(page.facts).length) {
    side.append(h("div", { class: "pad" }, h("h2", {}, "Facts"),
      ...Object.entries(page.facts).map(([id, value]) => h("div", { class: "card" },
        h("b", {}, id), h("div", { class: "tiny dim" }, short(value.reason || "", 360))))));
  }
}

const TABS = [
  ["overview", "Overview"],
  ["timeline", "Timeline"],
  ["profile", "Profiler"],
  ["chat", "Chat"],
  ["calls", "Calls"],
  ["workflows", "Workflows"],
  ["cycles", "Cycles"],
  ["records", "Records"],
];

function renderTabs(current) {
  const tabs = clear($("#tabs"));
  for (const [id, label] of TABS) {
    tabs.append(h("a", { href: `#/run/${current.run}/${current.attempt}/${id}`, class: id === current.tab ? "active" : "" }, label));
  }
}

async function renderTab(current, token) {
  const tab = clear($("#tab"));
  if (current.tab === "overview") return renderOverview(tab);
  if (current.tab === "profile") return renderProfile(tab, current);
  if (current.tab === "timeline") return renderTimeline(tab, current);
  if (current.tab === "chat") return renderChat(tab, current);
  if (current.tab === "calls") return renderCalls(tab, current);
  if (current.tab === "workflows") return renderWorkflows(tab, current);
  if (current.tab === "cycles") return renderCycles(tab, current);
  if (current.tab === "records") return renderRecords(tab, current);
  tab.append(h("p", { class: "dim" }, `Unknown tab: ${current.tab}`));
}

function renderOverview(tab) {
  const page = state.attemptPage;
  tab.append(h("h1", {}, `Attempt ${page.attempt.number}`));
  tab.append(h("p", { class: "dim" }, `${page.run.title} · ${page.run.status} · ${page.attempt.status || "unknown"}`));
  const stats = h("div", { class: "statgrid" });
  for (const row of page.overviews) {
    stats.append(h("div", { class: "stat" }, h("b", {}, row.id),
      h("span", {}, `${fmt(row.records)} records · ${fmt(row.calls)} calls`)));
  }
  tab.append(stats);
  if (page.plan) {
    tab.append(h("h2", {}, "Plan"));
    tab.append(h("p", {}, page.plan.brief || ""));
    const rows = (page.plan.steps || []).map((step) => h("tr", {},
      h("td", { class: "mono" }, step.at), h("td", {}, step.id), h("td", {}, step.instruction),
      h("td", {}, (step.fact_ids || []).join(", ")), h("td", {}, step.target || "")));
    tab.append(h("table", {}, h("thead", {}, h("tr", {},
      h("th", {}, "at"), h("th", {}, "id"), h("th", {}, "instruction"), h("th", {}, "facts"), h("th", {}, "target"))),
      h("tbody", {}, rows)));
  }
  if (page.failures.length) {
    tab.append(h("h2", {}, "Failures"));
    for (const failure of page.failures) tab.append(h("pre", {}, JSON.stringify(failure, null, 2)));
  }
  if (page.report) {
    tab.append(h("h2", {}, "Attempt report"));
    tab.append(h("details", {}, h("summary", {}, "open full report.json"), jsonBlock(page.report)));
  }
}

async function renderProfile(tab, current) {
  const query = new URLSearchParams({run: current.run, attempt: String(current.attempt)});
  if (state.participant) query.set("participant", state.participant);
  const receipts = await api(`/calls?${query}`);
  const embeddings = new Set(receipts.filter(call => call.route === "embeddings").map(call => call.call_id));
  const recorded = await api(`/profile?${query}`);
  const bars = recorded.filter(bar => bar.operation !== "embeddings" && !(bar.kind === "call" && embeddings.has(bar.target_id)));
  tab.append(h("h1", {}, "Cognition profiler"));
  if (!bars.length) {
    tab.append(h("p", {}, "This selection has no recorded workflow spans or timed model receipts."));
    return;
  }
  const children = new Map();
  const spanIds = new Set(bars.map(bar => bar.span_id).filter(Boolean));
  const workflows = new Map();
  for (const bar of bars) {
    if (bar.evidence?.kind === "workflow" && bar.invocation_id) {
      workflows.set(bar.participant + "/" + bar.invocation_id, bar.span_id);
    }
  }
  for (const bar of bars) {
    let parent = spanIds.has(bar.parent_span_id) ? bar.parent_span_id : "";
    if (!parent && bar.kind === "call" && bar.invocation_id) {
      parent = workflows.get(bar.participant + "/" + bar.invocation_id) || "";
    }
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(bar);
  }
  const roots = children.get("") || [];
  const search = h("input", {type:"search", placeholder:"Find an operation or participant", "aria-label":"Find an operation or participant"});

  const tree = h("div", {class:"profile-tree"});
  const timeline = h("section", {class:"profile-time"}, h("h1", {}, "Cognition timeline"));
  const viewport = h("div", {class:"profile-viewport"});
  const chart = h("div", {class:"profile-chart"});
  const zoom = h("input", {type:"range", min:1, max:100, step:0.01, value:1});
  zoom.oninput = () => { chart.style.width = Number(zoom.value) * 100 + "%"; };
  // Keep the instant under the pointer stationary while changing horizontal scale.
  viewport.addEventListener("wheel", event => {
    if (event.shiftKey || !event.deltaY) return;
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaMode === 2 ? event.deltaY * viewport.clientHeight : event.deltaY;
    const before = Number(zoom.value);
    const after = Math.min(100, Math.max(1, before * Math.exp(-delta * 0.002)));
    if (after === before) return;
    event.preventDefault();
    const pointer = event.clientX - viewport.getBoundingClientRect().left;
    const fraction = (viewport.scrollLeft + pointer) / chart.offsetWidth;
    zoom.value = String(after);
    zoom.oninput();
    viewport.scrollLeft = fraction * chart.offsetWidth - pointer;
  }, {passive:false});
  viewport.append(chart);
  timeline.append(h("label", {}, "Zoom ", zoom), viewport);
  const operations = h("section", {class:"profile-operations"}, h("h2", {}, "Operations"), search, tree);
  tab.replaceChildren(timeline, profileDivider("horizontal"), operations, profileDivider("vertical"));

  function inspect(bar) {
    for (const selected of tab.querySelectorAll("[aria-pressed=true]")) selected.setAttribute("aria-pressed", "false");
    for (const button of tab.querySelectorAll("[data-operation-id]")) {
      if (button.dataset.operationId === (bar.span_id || bar.target_id)) button.setAttribute("aria-pressed","true");
    }
    const operation = tree.querySelector('[aria-pressed="true"]');
    if (operation) {
      for (let parent = operation.parentElement; tree.contains(parent); parent = parent.parentElement) {
        if (parent.tagName === "DETAILS") parent.open = true;
      }
      operation.scrollIntoView({block:"nearest"});
    }
    profileInspector(bar, children, current);
  }
  function buttonFor(bar, cls) {
    const button = h("button", {type:"button", class:cls, title:profileLabel(bar),
      "data-kind":bar.evidence?.kind || bar.kind, "data-operation-id":bar.span_id || bar.target_id,
      "aria-label":profileLabel(bar), "aria-pressed":"false"}, bar.operation.replaceAll("_"," "));
    button.onclick = () => inspect(bar);
    return button;
  }
  function paint() {
    clear(tree); clear(chart);
    const visible = [];
    const needle = search.value.trim().toLowerCase();
    const lanes = new Map();
    function branch(bar, owner, depth) {

      const nested = (bar.span_id ? children.get(bar.span_id) || [] : []).map(child => branch(child, owner, depth + 1)).filter(Boolean);
      if (!nested.length && !profileLabel(bar).toLowerCase().includes(needle)) return null;
      visible.push(bar);
      if (!lanes.has(owner)) lanes.set(owner, []);
      lanes.get(owner).push({bar, depth, start:Date.parse(bar.start), end:Date.parse(bar.end || bar.start)});
      const row = h("div", {class:"operation-row"}, buttonFor(bar,"operation"), h("small",{},profileDuration(bar)));
      if (!nested.length) return row;
      const group = h("details", {}, h("summary",{}, "Steps (" + nested.length + ")"), nested);
      group.open = Boolean(needle);
      return h("div", {class:"operation-group"}, row, group);
    }
    for (const root of roots) {
      const thread = root.lane.split("/")[1] || root.lane;
      const owner = root.evidence?.kind === "workflow" ? root.operation :
        thread.startsWith("ThreadPoolExecutor") ? "Unattributed work" : thread;
      const node = branch(root, root.participant + " / " + owner, 0);
      if (node) tree.append(node);
    }
    if (!visible.length) {tree.append(h("p",{class:"empty"},"No matching operations.")); return;}
    let first = Infinity;
    let last = -Infinity;
    for (const bar of visible) {
      first = Math.min(first, Date.parse(bar.start));
      last = Math.max(last, Date.parse(bar.end || bar.start));
    }
    const duration = Math.max(1,last-first);
    chart.append(h("p",{class:"dim"},time(new Date(first).toISOString())+" UTC to "+time(new Date(last).toISOString())+" UTC"));
    for (const [name, events] of lanes) {
      chart.append(h("small",{class:"profile-label"}, name));
      const lane = h("div",{class:"profile-lane"});
      // Reserve a band per nesting depth. Overlapping siblings occupy separate rows.
      events.sort((a,b) => a.depth - b.depth || a.start - b.start || b.end - a.end);
      const rowEnds = [];
      let depth = -1;
      let firstRow = 0;
      for (const event of events) {
        if (event.depth !== depth) {
          depth = event.depth;
          firstRow = rowEnds.length;
        }
        let row = firstRow;
        while (row < rowEnds.length && rowEnds[row] > event.start) row++;
        rowEnds[row] = Math.max(event.end,event.start + 1);
        const box = buttonFor(event.bar,"profile-bar");
        box.title += " | Thread " + event.bar.lane;
        box.style.left = (event.start-first)/duration*100+"%";
        box.style.width = (event.end-event.start)/duration*100+"%";
        box.style.top = row * 22 + "px";
        lane.append(box);
      }
      lane.style.height = rowEnds.length * 22 + 3 + "px";
      chart.append(lane);
    }
  }
  search.oninput = paint;

  paint();
}

function profileDivider(orientation) {
  const vertical = orientation === "vertical";
  const property = vertical ? "--inspector-width" : "--timeline-height";
  const main = $("main");
  const divider = h("div", {class:"profile-divider "+orientation, role:"separator", tabindex:0,
    "aria-orientation":orientation, "aria-label":vertical ? "Resize inspection panels" : "Resize timeline",
    "aria-valuemin":15, "aria-valuemax":85});
  function resize(percent) {
    const value = Math.round(Math.max(15, Math.min(85, percent)));
    main.style.setProperty(property, value + "%");
    divider.setAttribute("aria-valuenow", String(value));
    localStorage.setItem(property, String(value));
  }
  resize(Number(localStorage.getItem(property)) || 50);
  divider.onpointerdown = event => { divider.setPointerCapture(event.pointerId); event.preventDefault(); };
  divider.onpointermove = event => {
    if (!divider.hasPointerCapture(event.pointerId)) return;
    const start = vertical ? $("#detail").getBoundingClientRect().left : $(".profile-time").getBoundingClientRect().top;
    const end = vertical ? $(".profile-operations").getBoundingClientRect().right : $("#detail").getBoundingClientRect().bottom;
    resize(100 * ((vertical ? event.clientX : event.clientY) - start) / (end - start));
  };
  divider.onkeydown = event => {
    const keys = vertical ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    resize(Number(divider.getAttribute("aria-valuenow")) + (event.key === keys[0] ? -2 : 2));
  };
  return divider;
}

function profileDuration(bar) {
  return bar.elapsed_seconds == null ? bar.outcome : bar.elapsed_seconds < 1
    ? Math.round(bar.elapsed_seconds * 1000) + " ms" : bar.elapsed_seconds.toFixed(1) + " s";
}

function profileLabel(bar) {
  return [bar.operation.replaceAll("_"," "), bar.evidence?.kind || bar.kind, bar.participant, profileDuration(bar)].join(" \u00b7 ");
}

// Evidence is untrusted text. Build Markdown elements directly and keep raw HTML literal.
function inlineMarkdown(line) {
  const pieces = [];
  const tokens = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  let end = 0;
  for (const match of line.matchAll(tokens)) {
    pieces.push(line.slice(end,match.index));
    const token = match[0];
    if (token.startsWith("`")) pieces.push(h("code",{},token.slice(1,-1)));
    else if (token.startsWith("**")) pieces.push(h("strong",{},token.slice(2,-2)));
    else if (token.startsWith("*")) pieces.push(h("em",{},token.slice(1,-1)));
    else {
      const split = token.indexOf("](");
      pieces.push(h("a",{href:token.slice(split+2,-1),target:"_blank",rel:"noreferrer"},token.slice(1,split)));
    }
    end = match.index + token.length;
  }
  pieces.push(line.slice(end));
  return pieces;
}

function markdownEvidence(text) {
  const article = h("div",{class:"evidence-text"});
  const lines = text.split(/\r?\n/);
  let paragraph = null;
  let list = null;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const fence = line.match(/^\s*(`{3,}|~{3,})/);
    const heading = line.match(/^(#{1,6})\s+(.+)/);
    const item = line.match(/^\s*([-*+]|\d+\.)\s+(.+)/);
    if (fence) {
      const code = [];
      while (++index < lines.length && !lines[index].trimStart().startsWith(fence[1])) code.push(lines[index]);
      article.append(h("pre",{},h("code",{},code.join("\n"))));
    } else if (heading) article.append(h("h"+heading[1].length,{},inlineMarkdown(heading[2])));
    else if (line.startsWith("> ")) article.append(h("blockquote",{},inlineMarkdown(line.slice(2))));
    else if (item) {
      const tag = /^\d/.test(item[1]) ? "ol" : "ul";
      if (!list || list.localName !== tag) {list=h(tag); article.append(list);}
      list.append(h("li",{},inlineMarkdown(item[2])));
      paragraph=null;
      continue;
    } else if (line.trim()) {
      if (!paragraph) {paragraph=h("p"); article.append(paragraph);}
      else paragraph.append("\n");
      paragraph.append(...inlineMarkdown(line));
      list=null;
      continue;
    }
    paragraph=null;
    list=null;
  }
  return article;
}

function evidenceText(value) {
  if (typeof value === "string") {
    try { if (/^\s*[\[{]/.test(value)) return evidenceText(JSON.parse(value)); } catch { /* Ordinary prose. */ }
  }
  if (value !== null && typeof value === "object") {
    return h("div",{class:"evidence-fields"}, Object.entries(value).map(([key, entry]) =>
      h("details",{},h("summary",{},Array.isArray(value) ? "Item " + (Number(key)+1) : key.replaceAll("_"," ")), evidenceText(entry))));
  }
  return markdownEvidence(String(value ?? "No value recorded"));
}

async function profileInspector(selected, children, current) {
  const panel = clear($("#detail"));
  const filter = h("input",{type:"search", placeholder:"Search inputs, outputs and memory", "aria-label":"Search evidence"});
  const content = h("div",{class:"evidence-tree"});
  const status = h("p",{class:"dim","aria-live":"polite"},"Reading evidence...");
  panel.append(h("h1",{},selected.operation.replaceAll("_"," ")),h("p",{class:"dim"},profileLabel(selected)), filter, status, content);
  function match() {
    const term = filter.value.trim().toLowerCase();
    let hits = 0;
    for (const section of content.querySelectorAll("details")) {
      const found = section.textContent.toLowerCase().includes(term);
      section.hidden = !found;
      if (term && found) {section.open = true; hits++;}
    }
    if (term) status.textContent = hits ? "Matching sections expanded" : "No matching text";
    else status.textContent = content.childElementCount ? "" : "No natural-language evidence recorded for this operation.";
  }
  filter.oninput = match;
  const query = new URLSearchParams({run:current.run, attempt:String(current.attempt), participant:selected.participant});
  // Group only explicitly recorded links from the selected operation and its children.
  const inputs = new Set();
  const retrieved = new Set();
  const outputs = new Set();
  function collectRecords(bar) {
    for (const id of bar.evidence?.knoxel_ids || []) inputs.add(id);
    for (const id of bar.evidence?.retrieved_record_ids || []) retrieved.add(id);
    for (const id of bar.evidence?.outputs || []) outputs.add(id);
    if (bar.span_id) for (const child of children.get(bar.span_id) || []) collectRecords(child);
  }
  collectRecords(selected);
  async function showRecords(label, ids) {
    if (!ids.size || !content.isConnected) return;
    const records = h("details",{},h("summary",{},label+" ("+ids.size+")"));
    content.append(records);
    for (const id of ids) {
      if (!content.isConnected) return;
      const recordQuery = new URLSearchParams(query);
      recordQuery.set("id",String(id));
      try {
        const data = await api("/record?"+recordQuery);
        records.append(h("details",{},h("summary",{}, "#"+id+" "+data.card.kind+" "+short(data.card.text,64)),
          h("p",{class:"dim"}, "Tick "+data.card.tick+" | Story "+time(data.record.data.timestamp_world_begin)+" | Created "+time(data.record.data.timestamp_creation)),
          evidenceText(data.card.text),h("a",{href:"#/"+runPath("record/"+selected.participant+"/"+id)},"Open memory record")));
      } catch (error) {records.append(h("p",{}, "#"+id+" "+error.message));}
    }
  }
  const seen = new Set();
  async function read(bar, host) {
    if (!content.isConnected) return;
    const body = h("div",{class:"evidence-body"});
    const section = h("details",{},h("summary",{},profileLabel(bar)),body);
    section.open = bar === selected;
    host.append(section);
    try {
      if (bar.kind === "call") {
        const callQuery = new URLSearchParams(query);
        callQuery.set("call_id",bar.target_id);
        const call = await api("/call?"+callQuery);
        for (const msg of call.messages || []) body.append(h("details",{},h("summary",{},"Input \u00b7 "+msg.role),evidenceText(msg.content)));
        if (call.input_text && !call.messages?.length) body.append(h("details",{},h("summary",{},"Input"),evidenceText(call.input_text)));
        if (call.response_text) body.append(h("details",{open:true},h("summary",{},"Output"),evidenceText(call.response_text)));
        if (call.reasoning_text) body.append(h("details",{},h("summary",{},"Reasoning"),evidenceText(call.reasoning_text)));
        if (call.error) body.append(h("p",{},call.error));
        if (!call.response_text && call.route === "embeddings") body.append(h("p",{class:"dim"},"Output \u00b7 embedding vector"));
      }
      if (bar.span_id && !seen.has(bar.span_id)) {
        seen.add(bar.span_id);
        for (const child of children.get(bar.span_id) || []) await read(child,body);
      }
      if (!body.childElementCount) body.append(h("p",{class:"dim"},"No input or output text was captured for this step."));
    } catch (error) {body.append(h("p",{},error.message));}
  }
  await Promise.all([showRecords("Input knoxels", inputs), showRecords("Retrieved knoxels", retrieved),
    showRecords("Output knoxels", outputs), read(selected,content)]);
  if (!inputs.size && !retrieved.size && !outputs.size) content.append(h("p",{class:"dim"},"No knoxel links recorded for this selection."));
  if (content.isConnected) match();
}

async function renderTimeline(tab, current) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number) });
  if (state.participant) query.set("participant", state.participant);
  const data = await api(`/timeline?${query}`);
  const filters = h("div", { class: "filters" });
  const search = h("input", { type: "search", placeholder: "filter timeline text", value: state.timelineSearch || "" });
  search.oninput = () => { state.timelineSearch = search.value; paintTimeline(); };
  for (const [kind, count] of Object.entries(data.counts).sort()) {
    const box = h("input", { type: "checkbox" });
    box.checked = state.timelineKinds.has(kind);
    box.onchange = () => {
      if (box.checked) state.timelineKinds.add(kind); else state.timelineKinds.delete(kind);
      paintTimeline();
    };
    filters.append(h("label", {}, box, `${kind} (${count})`));
  }
  filters.append(search);
  const list = h("div", {});
  let shownEvents = 600;
  function paintTimeline() {
    clear(list);
    const needle = (state.timelineSearch || "").toLowerCase();
    const rows = data.events.filter((event) => state.timelineKinds.has(event.kind)
      && (!needle || `${event.title} ${event.preview} ${event.participant} ${event.kind}`.toLowerCase().includes(needle)));
    if (!rows.length) list.append(h("p", { class: "empty" }, "No events match this filter."));
    list.append(h("p", { class: "dim" }, `Showing ${Math.min(shownEvents, rows.length)} of ${rows.length} matching events (${data.total} total).`));
    for (const event of rows.slice(0, shownEvents)) {
      const row = h("div", { class: "timeline-row " + (event.severity === "error" ? "error" : event.severity === "warning" ? "warning" : "") },
        h("div", { class: "when" }, `${time(event.at)}${event.clock === "wall" ? " wall" : ""}`),
        h("div", {}, badge(event.kind, event.kind)),
        h("div", { class: "what" }, h("div", { class: "title" }, event.title), h("div", { class: "preview" }, event.preview)));
      row.onclick = () => {
        state.selectedEvent = event;
        const target = event.target || {};
        if (target.kind === "call") go(runPath(`call/${target.participant}/${target.call_id}`));
        else if (target.kind === "record") go(runPath(`record/${target.participant}/${target.record_id}`));
        else if (target.kind === "cycle") go(runPath(`cycle/${target.participant}/${target.tick}`));
        else if (target.kind === "workflow") go(runPath(`workflow/${target.participant}/${target.invocation_id}`));
        else renderSelectedEvent(event);
      };
      list.append(row);
    }
    if (shownEvents < rows.length) list.append(h("button", { type: "button", onclick: () => { shownEvents += 600; paintTimeline(); } }, "Show 600 more"));
  }
  tab.append(h("h1", {}, "Timeline"));
  tab.append(h("p", { class: "dim" }, "Merged by UTC instant. A row marked wall has no recorded virtual time."));
  tab.append(filters, list);
  paintTimeline();
}

async function selectMemorySource(tab, query) {
  const sources = await api(`/memory-sources?${query}`);
  const picker = h("select", {"aria-label":"Memory source"});
  for (const source of [...sources, "capture"]) {
    const title = source === "ghost.pickle" ? "Current accepted memory" : source === "capture" ? "Captured attempt evidence" : source;
    picker.append(h("option", {value:source}, title));
  }
  if (![...picker.options].some(option => option.value === state.memorySource)) {
    state.memorySource = sources.includes("ghost.pickle") ? "ghost.pickle" : "capture";
  }
  picker.value = state.memorySource;
  picker.onchange = () => { state.memorySource = picker.value; render(); };
  tab.append(h("label", {}, "Memory source ", picker));
  query.set("memory", state.memorySource);
  return sources;
}
async function renderChat(tab, current) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number) });
  if (state.participant) query.set("participant", state.participant);
  const sources = await selectMemorySource(tab, query);
  const rows = [...await api(`/chat?${query}`)];
  if (state.memorySource === "ghost.pickle") {
    // The controller logs speech before the participant saves the corresponding memory.
    const deliveries = new Set(rows.map(row => row.delivery_id).filter(Boolean));
    const records = new Set(rows.map(row => row.participant + "/" + row.record_id));
    for (const event of state.run.state.timeline) {
      if (event.kind !== "speech" && event.kind !== "actor") continue;
      const participant = event.kind === "speech" ? event.source : event.target;
      if (state.participant && state.participant !== participant) continue;
      if (deliveries.has(event.delivery_id) || records.has(participant + "/" + event.output_id)) continue;
      rows.push({speaker:event.speaker, participant, text:event.content, at:event.timestamp,
        wall_at:event.wall_at, delivery_id:event.delivery_id, controller:true});
    }
    rows.sort((left,right) => time(left.at).localeCompare(time(right.at)));
  }
  const history = h("input", {type:"checkbox"});
  const hasCheckpoint = sources.includes("initial.pickle");
  history.disabled = !hasCheckpoint;
  history.checked = !hasCheckpoint || localStorage.getItem("pm-show-checkpoint-chat") === "true";
  const checkpointIds = new Set();
  function speechKey(row) {
    return row.delivery_id || row.participant + "/" + row.record_id;
  }
  if (hasCheckpoint) {
    const baseline = new URLSearchParams(query);
    baseline.set("memory","initial.pickle");
    for (const row of await api(`/chat?${baseline}`)) checkpointIds.add(speechKey(row));
  }
  let previousDay = "";
  tab.append(h("h1", {}, "Public conversation"));
  const search = h("input", { type: "search", placeholder: "Find speech or speaker" });
  const previous = h("button", {type:"button"}, "Previous");
  const next = h("button", {type:"button"}, "Next");
  const count = h("span", {class:"dim", "aria-live":"polite"});
  const controls = h("div", {class:"chat-search"}, search, previous, next, count);
  const empty = h("p",{class:"dim"});
  tab.append(h("label",{},history,hasCheckpoint ? "Show ghost checkpoint history" : "Checkpoint history filter unavailable (no initial snapshot)"),controls,empty);
  history.onchange = () => {
    localStorage.setItem("pm-show-checkpoint-chat", String(history.checked));
    search.oninput();
  };
  let matches = [];
  let position = -1;
  function move(direction) {
    if (!matches.length) return;
    position = (position + direction + matches.length) % matches.length;
    matches[position].scrollIntoView({block:"center"});
    count.textContent = `${position + 1} / ${matches.length}`;
  }
  previous.onclick = () => move(-1);
  next.onclick = () => move(1);
  search.oninput = () => {
    const needle = search.value.trim().toLocaleLowerCase();
    matches = [];
    position = -1;
    let visibleRows = 0;
    for (const bubble of tab.querySelectorAll(".chat-row")) {
      bubble.hidden = (!history.checked && bubble.dataset.checkpoint === "true") || !bubble.textContent.toLocaleLowerCase().includes(needle);
      if (!bubble.hidden) visibleRows++;
      if (!bubble.hidden && needle) matches.push(bubble);
    }
    for (const day of tab.querySelectorAll("[data-chat-day]")) {
      let sibling = day.nextElementSibling;
      let visible = false;
      while (sibling?.classList.contains("chat-row")) {
        if (!sibling.hidden) { visible = true; break; }
        sibling = sibling.nextElementSibling;
      }
      day.hidden = !visible;
    }
    previous.disabled = next.disabled = !matches.length;
    empty.hidden = visibleRows > 0;
    empty.textContent = needle ? "No matching messages." : history.checked ? "No conversation in this memory source." : "No conversation after the initial checkpoint in this memory source.";
    count.textContent = needle ? `${matches.length} matches` : "";
  };
  for (const row of rows) {
    const day = row.at.slice(0, 10) || "Undated";
    if (day !== previousDay) { tab.append(h("h2", {"data-chat-day":day}, day)); previousDay = day; }
    const companion = row.speaker !== "Jesse";
    const speaker = companion && row.participant !== "primary" ? row.speaker + " \u00b7 " + row.participant : row.speaker;
    const block = h(row.controller ? "article" : "a", {class:"chat-row", "data-checkpoint":String(!row.controller && checkpointIds.has(speechKey(row))), href:row.controller ? null : "#/"+runPath("record/"+row.participant+"/"+row.record_id)},
      h("div",{class:"chat-byline"},h("strong",{class:"speaker"},speaker), row.controller ? h("small",{class:"dim"},"Controller") : null,
        h("time",{class:"when",title:"Story time: "+time(row.at)+(row.wall_at ? " | Wall time: "+time(row.wall_at) : "")},time(row.at).slice(11,16))),
      h("div",{class:"text"},row.text));
    if (companion) {
      block.classList.add("companion-bubble");
      const hue = Array.from(row.participant).reduce((n,ch) => (31*n+ch.codePointAt(0))>>>0,0)%360;
      block.style.setProperty("--bubble-hue",String(hue));
    }
    tab.append(block);
  }
  search.oninput();
}

async function renderCalls(tab, current) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number) });
  if (state.participant) query.set("participant", state.participant);
  const data = await api(`/calls?${query}`);
  tab.append(h("h1", {}, "Model calls"));
  tab.append(h("p", { class: "dim" }, `${data.length} prompt-log records. Open a row for the exact request and response.`));
  const body = h("tbody");
  for (const call of data) {
    const tr = h("tr", { class: "link" },
      h("td", { class: "mono" }, time(call.world_time || call.started_at)),
      h("td", {}, call.participant),
      h("td", {}, call.module || call.route),
      h("td", { class: "mono" }, call.model || ""),
      h("td", {}, call.route),
      h("td", { class: "num" }, fmt(call.total_tokens)),
      h("td", { class: "num" }, call.elapsed_seconds == null ? "-" : `${call.elapsed_seconds.toFixed(1)}s`),
      h("td", {}, call.error ? badge("error", "failure") : ""));
    tr.onclick = () => go(runPath(`call/${call.participant}/${call.call_id}`));
    body.append(tr);
  }
  tab.append(h("table", {}, h("thead", {}, h("tr", {},
    h("th", {}, "time"), h("th", {}, "participant"), h("th", {}, "module"), h("th", {}, "model"),
    h("th", {}, "route"), h("th", { class: "num" }, "tokens"), h("th", { class: "num" }, "elapsed"), h("th", {}, ""))), body));
}

async function renderWorkflows(tab, current) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number) });
  if (state.participant) query.set("participant", state.participant);
  const rows = await api(`/workflows?${query}`);
  tab.append(h("h1", {}, "Workflow invocations"));
  tab.append(h("p", { class: "dim" }, `${rows.length} kernel folders. Open a row for phases and artifacts.`));
  const body = h("tbody");
  for (const run of rows) {
    const tr = h("tr", { class: "link" },
      h("td", { class: "mono" }, time(run.world_time || run.started_at)),
      h("td", {}, run.participant), h("td", {}, run.workflow),
      h("td", { class: "mono" }, run.invocation_id.slice(0, 10)), h("td", {}, run.status),
      h("td", { class: "num" }, run.phase_count), h("td", { class: "num" }, run.call_count),
      h("td", { class: "num" }, fmt(run.total_tokens)));
    tr.onclick = () => go(runPath(`workflow/${run.participant}/${run.invocation_id}`));
    body.append(tr);
  }
  tab.append(h("table", {}, h("thead", {}, h("tr", {},
    h("th", {}, "time"), h("th", {}, "participant"), h("th", {}, "workflow"), h("th", {}, "invocation"),
    h("th", {}, "status"), h("th", { class: "num" }, "phases"), h("th", { class: "num" }, "calls"),
    h("th", { class: "num" }, "tokens"))), body));
}

async function renderCycles(tab, current) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number) });
  if (state.participant) query.set("participant", state.participant);
  const rows = await api(`/cycles?${query}`);
  tab.append(h("h1", {}, "GWT cycles"));
  tab.append(h("p", { class: "dim" }, `${rows.length} captured turns. Open a row for the full cycle layout.`));
  const body = h("tbody");
  for (const cycle of rows) {
    const tr = h("tr", { class: "link" },
      h("td", { class: "mono" }, time(cycle.at)), h("td", {}, cycle.participant),
      h("td", { class: "num" }, cycle.tick), h("td", { class: "num" }, cycle.member_count),
      h("td", { class: "num" }, cycle.stimulus_count), h("td", { class: "num" }, cycle.codelet_runs),
      h("td", {}, cycle.summary));
    tr.onclick = () => go(runPath(`cycle/${cycle.participant}/${cycle.tick}`));
    body.append(tr);
  }
  tab.append(h("table", {}, h("thead", {}, h("tr", {},
    h("th", {}, "time"), h("th", {}, "participant"), h("th", { class: "num" }, "tick"),
    h("th", { class: "num" }, "members"), h("th", { class: "num" }, "stimuli"),
    h("th", { class: "num" }, "codelets"), h("th", {}, "summary"))), body));
}

async function renderRecords(tab, current) {
  tab.append(h("h1", {}, "Captured records"));
  if (!state.participant) {
    tab.append(h("p", { class: "empty" }, "Select one participant in the header to list its records."));
    return;
  }
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant: state.participant });
  await selectMemorySource(tab, query);
  const search = h("input", { type: "search", placeholder: "search record text, type, role, or id", value: state.recordSearch || "" });
  const list = h("div", {});
  let shownRecords = 600;
  async function paint() {
    clear(list);
    const local = new URLSearchParams(query);
    if (state.recordSearch) local.set("q", state.recordSearch);
    local.set("memory", state.memorySource);
    const rows = await api(`/records?${local}`);
    if (!rows.length) list.append(h("p", { class: "empty" }, "No records match."));
    list.append(h("p", { class: "dim" }, `Showing ${Math.min(shownRecords, rows.length)} of ${rows.length} matching records.`));
    for (const card of rows.slice(0, shownRecords)) {
      const row = h("div", { class: "timeline-row" },
        h("div", { class: "when" }, `#${card.id}`),
        h("div", {}, badge(card.role || card.kind, card.kind === "Feature" ? "speech" : "")),
        h("div", { class: "what" }, h("div", { class: "title" }, `${card.role || card.kind} · tick ${card.tick}`),
          h("div", { class: "preview" }, short(card.text, 260))));
      row.onclick = () => go(runPath(`record/${state.participant}/${card.id}`));
      list.append(row);
    }
    if (shownRecords < rows.length) list.append(h("button", { type: "button", onclick: () => { shownRecords += 600; paint().catch((error) => notice(error.message)); } }, "Show 600 more"));
  }
  search.oninput = () => { state.recordSearch = search.value; paint(); };
  tab.append(h("div", { class: "filters" }, search), list);
  await paint();
}

/* ---------------------------------------------------------------- detail */

async function renderDetail(current, token) {
  const panel = clear($("#detail"));
  if (!current.detail.length) {
    if (current.tab === "profile") {panel.append(h("h2", {}, "Operation details"), h("p",{class:"dim"},"Select an operation to inspect its inputs, outputs and memory.")); return;}
    if (state.selectedEvent) return renderSelectedEvent(state.selectedEvent);
    panel.append(h("p", { class: "empty" }, "Select a timeline row, a call, a workflow, a cycle, or a record."));
    return;
  }
  const kind = current.detail[0];
  try {
    if (kind === "call") await detailCall(panel, current.detail[1], decodeURIComponent(current.detail[2] || ""));
    else if (kind === "record") await detailRecord(panel, current.detail[1], Number(current.detail[2]));
    else if (kind === "cycle") await detailCycle(panel, current.detail[1], Number(current.detail[2]));
    else if (kind === "workflow") await detailWorkflow(panel, current.detail[1], decodeURIComponent(current.detail[2] || ""));
    else if (kind === "artifact") await detailArtifact(panel, current.detail[1], decodeURIComponent(current.detail[2] || ""));
    else if (kind === "agent") await detailAgent(panel, decodeURIComponent(current.detail[1] || ""));
    else panel.append(h("p", { class: "empty" }, `Unknown detail: ${kind}`));
  } catch (error) {
    if (token === renderToken) {
      panel.append(h("p", { class: "empty" }, error.message));
      notice(error.message);
    }
  }
}

function menu(title, subtitle) {
  return h("div", {}, h("h1", {}, title), subtitle ? h("p", { class: "dim" }, subtitle) : null);
}

function renderSelectedEvent(event) {
  const panel = clear($("#detail"));
  panel.append(menu(event.title, `${event.kind} · ${event.participant || "run"} · ${time(event.at)} ${event.clock === "wall" ? "(wall)" : "(virtual)"}`));
  panel.append(h("p", {}, event.preview || ""));
  panel.append(h("h2", {}, "Details"));
  panel.append(jsonBlock(event.details || {}));
}

async function detailCall(panel, participant, callId) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant, call_id: callId });
  const call = await api(`/call?${query}`);
  panel.append(menu(`${call.module || call.route || "model call"}`, `${participant} · ${call.call_id}`));
  panel.append(kv([
    ["route", call.route], ["model", call.model], ["module", call.module],
    ["invocation", call.invocation_id || "-"], ["world time", time(call.world_time)],
    ["started", time(call.started_at)], ["elapsed", call.elapsed_seconds == null ? "-" : `${call.elapsed_seconds.toFixed(2)}s`],
    ["prompt tokens", fmt(call.prompt_tokens)], ["completion tokens", fmt(call.completion_tokens)],
    ["total tokens", fmt(call.total_tokens)], ["finish", call.finish_reason || "-"],
    ["error", call.error || "-"], ["json log", call.json_log || "-"], ["markdown", call.markdown_log || "-"],
  ]));
  if (call.error) panel.append(h("pre", { class: "response" }, call.error));
  if (call.messages.length) {
    panel.append(h("h2", {}, `Request messages (${call.messages.length})`));
    for (const message of call.messages) {
      panel.append(h("details", {}, h("summary", {}, message.role || "message"),
        evidenceText(message.content)));
    }
  } else if (call.input_text) {
    panel.append(h("h2", {}, "Input"));
    panel.append(evidenceText(call.input_text));
  }
  if (call.response_text) {
    panel.append(h("h2", {}, "Response"));
    panel.append(evidenceText(call.response_text));
  }
  if (call.reasoning_text) {
    panel.append(h("h2", {}, "Reasoning"));
    panel.append(evidenceText(call.reasoning_text));
  }
  if (call.usage_events && call.usage_events.length) {
    panel.append(h("h2", {}, "Usage events"));
    const body = h("tbody");
    for (const event of call.usage_events) {
      body.append(h("tr", {}, h("td", {}, event.event), h("td", { class: "mono" }, time(event.time)),
        h("td", { class: "num" }, event.elapsed_seconds == null ? "-" : `${event.elapsed_seconds.toFixed(2)}s`),
        h("td", {}, JSON.stringify(event.usage || {})), h("td", {}, event.error || "")));
    }
    panel.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "event"), h("th", {}, "time"),
      h("th", { class: "num" }, "elapsed"), h("th", {}, "usage"), h("th", {}, "error"))), body));
  }
  panel.append(h("details", {}, h("summary", {}, "raw prompt-log payload"), jsonBlock(call.payload)));
  panel.append(h("details", {}, h("summary", {}, "raw provider response"), jsonBlock(call.response)));
}

async function detailRecord(panel, participant, recordId) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant, id: String(recordId) });
  if (["records", "chat"].includes(route().tab)) query.set("memory", state.memorySource);
  const data = await api(`/record?${query}`);
  const card = data.card;
  panel.append(menu(`${card.role || card.kind} · #${card.id}`, `${participant} · tick ${card.tick} · ${card.kind}`));
  panel.append(evidenceText(card.text || "(no content)"));
  if (card.details && Object.keys(card.details).length) {
    panel.append(h("h2", {}, "Stored details"));
    panel.append(kv(Object.entries(card.details)));
  }
  if (card.links && card.links.length) {
    panel.append(h("h2", {}, "Recorded links"));
    const body = h("tbody");
    for (const link of card.links) {
      const tr = h("tr", { class: "link" }, h("td", {}, link.relation), h("td", { class: "num" }, link.id));
      tr.onclick = () => go(runPath(`record/${participant}/${link.id}`));
      body.append(tr);
    }
    panel.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "relation"), h("th", { class: "num" }, "record"))), body));
  }
  panel.append(h("details", {}, h("summary", {}, "raw captured record"), jsonBlock(data.record)));
  if (data.branch_version) panel.append(h("details", {}, h("summary", {}, "Version from raw branch history"), jsonBlock(data.branch_version)));
}

async function detailCycle(panel, participant, tick) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant, tick: String(tick) });
  const data = await api(`/cycle?${query}`);
  const view = data.presentation;
  panel.append(menu(`Cycle ${view.tick}`, `${participant} · ${short(view.summary || "", 200)}`));
  for (const section of view.sections || []) {
    panel.append(h("h2", {}, section.title));
    if (section.note) panel.append(h("p", { class: "tiny dim" }, section.note));
    for (const row of section.rows || []) {
      if (row.kind === "record") {
        const card = data.cards[String(row.id)];
        const block = h("div", { class: "card" },
          h("b", {}, `#${row.id}`), " ", badge(card ? (card.role || card.kind) : "uncaptured"),
          h("div", { class: "tiny dim" }, row.label || ""),
          card ? h("div", { class: "tiny" }, short(card.text, 260)) : h("div", { class: "tiny dim" }, "record not captured in this package"));
        if (card) block.onclick = () => go(runPath(`record/${participant}/${row.id}`));
        panel.append(block);
      } else {
        panel.append(h("div", { class: "card" }, h("div", { class: "tiny" }, row.text || ""),
          row.data && Object.keys(row.data).length ? h("details", {}, h("summary", {}, "stored values"), jsonBlock(row.data)) : null));
      }
    }
  }
}

async function detailWorkflow(panel, participant, invocationId) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant, invocation: invocationId });
  const run = await api(`/workflow?${query}`);
  panel.append(menu(`${run.workflow} · ${run.invocation_id.slice(0, 10)}`, `${participant} · ${run.status}`));
  panel.append(kv([
    ["invocation", run.invocation_id], ["workflow", run.workflow], ["status", run.status],
    ["world time", time(run.world_time)], ["started", time(run.started_at)], ["ended", time(run.ended_at)],
    ["calls", fmt(run.call_count)], ["prompt tok", fmt(run.prompt_tokens)],
    ["completion tok", fmt(run.completion_tokens)], ["total tok", fmt(run.total_tokens)],
  ]));
  panel.append(h("h2", {}, "Phases"));
  const body = h("tbody");
  for (const phase of run.phases) {
    body.append(h("tr", {}, h("td", {}, phase.phase), h("td", {}, phase.role || phase.kind),
      h("td", {}, phase.status || phase.verdict), h("td", { class: "tiny" }, short(phase.summary, 240))));
  }
  panel.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "phase"), h("th", {}, "role"),
    h("th", {}, "status"), h("th", {}, "summary"))), body));
  panel.append(h("h2", {}, "Artifacts"));
  const artifacts = h("tbody");
  for (const artifact of run.artifacts) {
    const tr = h("tr", { class: "link" }, h("td", { class: "mono" }, artifact.name),
      h("td", {}, artifact.kind), h("td", { class: "num" }, fmt(artifact.size)));
    if (artifact.relative) tr.onclick = () => go(runPath(`artifact/${participant}/${encodeURIComponent(artifact.relative)}`));
    artifacts.append(tr);
  }
  panel.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "file"), h("th", {}, "kind"),
    h("th", { class: "num" }, "bytes"))), artifacts));
}

async function detailArtifact(panel, participant, path) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), participant, path });
  const data = await api(`/artifact?${query}`);
  panel.append(menu(path.split("/").pop() || "artifact", `${participant} · ${fmt(data.size)} bytes`));
  panel.append(h("pre", {}, data.text));
}

async function detailAgent(panel, path) {
  const query = new URLSearchParams({ run: state.run.summary.id, attempt: String(state.attemptPage.attempt.number), path });
  const data = await api(`/agent?${query}`);
  panel.append(menu("Harness agent receipt", path));
  panel.append(jsonBlock(data.receipt));
}

/* ------------------------------------------------------------------ boot */

function boot() {
  const stored = savedTheme();
  setTheme(!EMBEDDED && stored === "dark" ? "dark" : "light");
  if (EMBEDDED) document.documentElement.dataset.embedded = "true";
  $("#theme").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  $("#reload").onclick = () => { state.selectedEvent = null; render(true); };
  window.addEventListener("hashchange", render);
  let polling = false;
  setInterval(async () => {
    const current = route();
    if (polling || document.hidden || current.page !== "run" || current.run !== state.run?.summary.id) return;
    polling = true;
    try {
      const controller = await api(`/controller-status/${encodeURIComponent(current.run)}`, false);
      if (current.run !== route().run || current.attempt !== route().attempt) return;
      const selected = state.attemptPage.attempt;
      const status = selected.number === controller.attempt ? controller.status : selected.status;
      $("#meta").textContent = `${state.run.summary.title} · attempt ${selected.number} · ${status}`;
    } catch (error) {
      if (current.run === route().run) $("#meta").textContent = `Status unavailable · ${error.message}`;
    } finally {
      polling = false;
    }
  }, 10000);
  render(true);
}

boot();
