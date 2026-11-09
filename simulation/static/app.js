"use strict";
const $ = id => document.getElementById(id);
let catalog = {homes: [], runs: []};
let selectedHome = null;
let selectedRun = null;
let selectedCapture = null;
let historyRows = [];
let shownHistory = 100;
let runState = null;
let packages = [];
let polling = false;
let reportKey = "";
let recordCheckpoint = "";
let recordFeedback = {};
let actorSummary = null;
let profileHome = null;
let factsRenderKey = "";
const renderedConversations = new WeakMap();
const RUN_TABS = ["conversation", "facts", "evidence", "script", "analysis", "debugger"];

function savedView() {
  try {
    return window.sessionStorage.getItem("pm-simulation-view");
  } catch (error) {
    if (error instanceof DOMException && error.name === "SecurityError") return null;
    throw error;
  }
}
function rememberView(route) {
  try {
    const storage = window.sessionStorage;
    if (route === null) storage.removeItem("pm-simulation-view");
    else storage.setItem("pm-simulation-view", route);
  } catch (error) {
    if (error instanceof DOMException && ["SecurityError", "QuotaExceededError"].includes(error.name)) return;
    throw error;
  }
}

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = String(text);
  if (className) result.className = className;
  return result;
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
function notice(error) {
  $("notice").hidden = !error;
  $("notice").textContent = error ? String(error.message || error) : "";
}
function action(id, callback) {
  $(id).addEventListener("click", async () => {
    notice(null);
    $(id).disabled = true;
    try { await callback(); } catch (error) { notice(error); }
    finally { $(id).disabled = false; }
  });
}
function show(section, title, status = "Ready") {
  for (const id of ["setup", "history", "run", "capture"]) $(id).hidden = id !== section;
  $("title").textContent = title;
  $("status").textContent = status;
}
function message(row) {
  const article = node("article", undefined, "message");
  const head = node("div", undefined, "message-head");
  const name = row.speaker && row.source ? `${row.source}: ${row.speaker}` : row.speaker || row.source || "Participant";
  let timestamp = row.timestamp || "";
  if (/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?$/.test(timestamp)) timestamp += "Z";
  head.append(node("span", name, "speaker"), node("time", timestamp.replace("T", " ")));
  article.append(head, node("div", row.text ?? row.content ?? "", "message-body"));
  return article;
}
function renderChat(target, rows) {
  const signatures = rows.map(row => JSON.stringify(row));
  const previous = renderedConversations.get(target);
  let unchanged = 0;
  if (previous) {
    while (unchanged < previous.length && unchanged < signatures.length && previous[unchanged] === signatures[unchanged]) unchanged++;
    if (unchanged === previous.length && unchanged === signatures.length) return;
  }
  while (target.children.length > unchanged) target.lastElementChild.remove();
  for (let index = unchanged; index < rows.length; index++) {
    const article = message(rows[index]);
    if (target.id === "conversation") article.id = `delivery-${index}`;
    target.append(article);
  }
  if (!rows.length) target.append(node("div", "No messages yet.", "empty"));
  renderedConversations.set(target, signatures);
}
async function refreshCatalog() {
  catalog = await api("/api/catalog");
  $("captures").replaceChildren();
  for (const capture of catalog.captures || []) {
    const button = node("button", capture.title);
    button.disabled = !capture.available;
    button.addEventListener("click", () => openCapture(capture.id).catch(notice));
    $("captures").append(button);
  }
  $("homes").replaceChildren();
  for (const home of catalog.homes) {
    const button = node("button", home.folder);
    button.append(node("small", `${home.name} · ${home.user}`));
    button.addEventListener("click", () => openHome(home).catch(notice));
    $("homes").append(button);
  }
  $("runs").replaceChildren();
  for (const run of catalog.runs) {
    const button = node("a", run.title, "run-link");
    button.href = `#run/${encodeURIComponent(run.id)}/conversation`;
    button.append(node("small", `${run.status} · attempt ${run.attempt}`));
    $("runs").append(button);
  }
  for (const id of ["primary", "secondary"]) {
    const current = $(id).value;
    $(id).replaceChildren();
    if (id === "secondary") $(id).append(new Option("No connection", ""));
    for (const home of catalog.homes) $(id).append(new Option(`${home.folder} (${home.name})`, home.id));
    if ([...$(id).options].some(option => option.value === current)) $(id).value = current;
  }
}
async function refreshActorProfile() {
  const home = $("primary").value;
  if (!home) return;
  const summary = await api(`/api/actor-profile?home=${encodeURIComponent(home)}`);
  if (home !== $("primary").value) return;
  actorSummary = summary;
  profileHome = home;
  $("plan-card").hidden = true;
  $("start-time").value = summary.suggested_start_time;
  $("actor-summary").textContent = `${summary.speaker_name}: ${summary.speaker_turns} historical messages with ${summary.companion_name}, their AI companion. History ends ${summary.continuity_at}.`;
  showTimeGap();
}
function showTimeGap() {
  if (!actorSummary) return;
  const hours = (Date.parse($("start-time").value) - Date.parse(actorSummary.continuity_at)) / 3600000;
  $("time-gap").textContent = hours > 24 ? `This starts ${Math.floor(hours / 24)} days after saved history. That absence becomes part of the scene.` : "Starts after the selected history, with no long absence added.";
}
$("primary").addEventListener("change", () => refreshActorProfile().catch(notice));
$("start-time").addEventListener("input", showTimeGap);

async function job(kind, body) {
  const started = await api(kind, body);
  for (;;) {
    const result = await api(`/api/jobs/${started.job_id}`);
    if (result.status === "failed") throw new Error(result.error);
    if (result.status === "completed") return result;
    $("job-status").textContent = result.progress ? `Planned ${result.progress.completed_days} / ${result.progress.total_days} days…` : "Working with the configured model…";
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
function previewPlan() {
  const plan = JSON.parse($("plan-json").value);
  $("plan-preview").replaceChildren(node("h2", plan.title));
  const required = node("details");
  required.append(node("summary", `${plan.facts.length} required scenario facts`));
  for (const fact of plan.facts) required.append(node("p", `${fact.id} · ${fact.text}`, "muted"));
  $("plan-preview").append(required);
  renderSceneEditor(plan);
}
async function openHome(home) {
  selectedRun = null;
  selectedCapture = null;
  selectedHome = home;
  const route = `home/${encodeURIComponent(home.id)}`;
  if (location.hash !== `#${route}`) location.hash = route;
  rememberView(route);
  show("history", home.folder);
  const response = await api(`/api/history?home=${encodeURIComponent(home.id)}`);
  if (selectedHome !== home || selectedRun !== null) return;
  $("snapshot").replaceChildren(...response.snapshots.map(name => new Option(name, name)));
  $("snapshot").value = "ghost.pickle";
  historyRows = response.messages;
  shownHistory = 100;
  showHistory();
}
function showHistory() {
  renderChat($("history-chat"), historyRows.slice(-shownHistory));
  $("earlier").hidden = shownHistory >= historyRows.length;
}
async function openRun(id, tab = "conversation") {
  selectedRun = id;
  selectedHome = null;
  selectedCapture = null;
  reportKey = "";
  factsRenderKey = "";
  $("record-reader").hidden = true;
  show("run", "Loading run…");
  selectRunTab(tab);
  const feedback = await api(`/api/runs/${id}/feedback`);
  if (selectedRun !== id) return;
  recordFeedback = feedback.review?.records || {};
  $("review-notes").value = feedback.review?.notes || "";
  $("review-rating").value = feedback.review?.rating ?? "";
  await updateRun();
  await loadWorkspaceRun(id);
}
async function updateRun() {
  if (!selectedRun) return;
  const id = selectedRun;
  const value = await api(`/api/runs/${id}`);
  if (id !== selectedRun) return;
  runState = value.state;
  const labels = {title: runState.title,
    status: runState.requested_status ? `${runState.requested_status} requested · draining` : runState.status,
    "run-progress": `Stage ${Math.min(runState.step_index + 1, value.plan.steps.length)} / ${value.plan.steps.length} · ${runState.clock}`};
  for (const [name, text] of Object.entries(labels)) if ($(name).textContent !== text) $(name).textContent = text;
  renderChat($("conversation"), runState.timeline);
  await updateAnalysis(id);
  if (id !== selectedRun) return;
  const factsKey = JSON.stringify([id, value.plan.facts, runState.facts]);
  if (factsKey !== factsRenderKey) {
    factsRenderKey = factsKey;
    $("facts").replaceChildren(node("h2", "Required facts"));
    for (const fact of value.plan.facts) {
      const line = node("div", undefined, "fact");
      line.append(node("span", runState.facts[fact.id] ? "✓" : "○", "check"), node("span", fact.text));
      if (runState.facts[fact.id]) line.append(node("p", runState.facts[fact.id].reason, "muted"));
      $("facts").append(line);
    }
  }
  const key = JSON.stringify([id, runState.status, runState.reports, runState.failures.length]);
  if (key !== reportKey) {
    reportKey = key;
    $("reports").replaceChildren(node("h2", "Captured attempts"));
    for (const failure of runState.failures) $("reports").append(node("pre", JSON.stringify(failure, null, 2)));
    for (const path of runState.reports) {
      const button = node("button", path, "secondary");
      button.addEventListener("click", () => openReport(path).catch(notice));
      $("reports").append(button);
    }
    if (!runState.reports.length) $("reports").append(node("p", "Evidence appears after this attempt drains its running work.", "muted"));
  }
}
async function artifact(path) {
  const response = await fetch(`/api/runs/${selectedRun}/artifact?path=${encodeURIComponent(path)}`);
  if (!response.ok) throw new Error(await response.text());
  return path.endsWith(".json") ? response.json() : response.text();
}
async function openReport(path) {
  // Existing Windows captures contain filesystem separators in relative paths.
  path = path.replaceAll("\\", "/");
  const report = await artifact(path);
  const base = path.slice(0, path.lastIndexOf("/"));
  const panel = $("reports");
  panel.replaceChildren(node("h2", `Attempt ${report.run.attempt} · ${report.run.status}`), node("p", `Human review: ${report.human_review}. Fact checks are automated.`, "muted"));
  if (report.coordinated_checkpoint) {
    const restore = node("button", "Restore all participants to this checkpoint", "secondary");
    restore.disabled = ["running", "starting"].includes(runState.status);
    restore.addEventListener("click", async () => {
      try { await api(`/api/runs/${selectedRun}/restore`, {attempt: report.run.attempt}); await updateRun(); }
      catch (error) { notice(error); }
    });
    panel.append(restore, node("p", "Restore archives the later branch and retains its reports.", "muted"));
  }
  const links = node("div", undefined, "report-links");
  for (const [id, participant] of Object.entries(report.participants)) {
    const prefix = `${base}/evidence/${id}/`;
    const historyButton = node("button", `${id}: History and forks`, "secondary");
    historyButton.addEventListener("click", () => openHome({id: participant.home, folder: `${report.run.title} / ${id}`}).catch(notice));
    links.append(historyButton);
    for (const [label, source] of [["Chat", participant.chat], ["Private activity", participant.activity], ["Transactions", participant.transactions], ...participant.cycles.map(cycle => [`${cycle.slot} · tick ${cycle.tick_id}`, cycle.path])]) {
      const button = node("button", `${id} · ${label}`, "secondary");
      button.addEventListener("click", () => readPackage(prefix + source, `${id} · ${label}`).catch(notice));
      links.append(button);
    }
    const details = node("details");
    details.append(node("summary", `${id} · ${participant.prompt_files.length} captured prompt files`));
    for (const source of participant.prompt_files) {
      const button = node("button", source, "quiet");
      button.addEventListener("click", () => readPackage(prefix + source, source).catch(notice));
      details.append(button);
    }
    panel.append(details);
  }
  panel.append(links);
}
async function readPackage(path, title) {
  path = path.replaceAll("\\", "/");
  const value = await artifact(path);
  const checkpoint = path.match(/^(attempts\/\d+\/evidence\/[^/]+)\//);
  recordCheckpoint = checkpoint ? checkpoint[1] + "/ghost.pickle" : "";
  $("record-reader").hidden = false;
  $("reader-title").textContent = title;
  packages = typeof value === "string" ? [{content: value}] : value.records || value.transactions || [value];
  $("record-filter").value = "";
  let display = null;
  if (value.schema_id === "pm.record-package.v1") {
    display = await api("/api/present", {package: value});
    for (const wrapped of packages) wrapped.links = display.cognition.records[String(wrapped.data.id)]?.links || [];
  }
  renderRecords();
  if (display) renderCycles(display, $("records"));
  $("record-reader").scrollIntoView({behavior: "smooth"});
}
function renderRecords() {
  const query = $("record-filter").value.toLocaleLowerCase();
  const records = packages.filter(value => JSON.stringify(value).toLocaleLowerCase().includes(query));
  $("records").replaceChildren();
  for (const wrapped of records) {
    const data = wrapped.data || wrapped;
    const card = node("details", undefined, "record");
    card.open = records.length < 8;
    card.append(node("summary", `${data.id ?? data.commit_id ?? ""} · ${wrapped.__type__ || "Record"} · ${(data.content || data.manifest?.kind || "").slice(0, 100)}`));
    if (data.content) card.append(node("div", data.content, "message-body"));
    card.append(node("div", `Tick ${data.tick_id ?? "—"} · ${data.timestamp_world_begin || ""}`, "record-meta"));
    const full = node("details");
    full.append(node("summary", "All captured fields"), node("pre", JSON.stringify(wrapped, null, 2)));
    card.append(full);
    card.title = JSON.stringify({tick: data.tick_id, source: data.provenance, created: data.timestamp_creation, metadata: data.metadata});
    if (data.id !== undefined && selectedRun && recordCheckpoint) {
      const feedbackKey = `${recordCheckpoint}:${data.id}`;
      const feedback = node("details");
      feedback.append(node("summary", "Owner notes and rating"));
      const notes = node("textarea"); notes.rows = 3;
      notes.value = recordFeedback[feedbackKey]?.notes || "";
      const rating = node("input"); rating.type = "number"; rating.min = "1"; rating.max = "5";
      rating.placeholder = "Unrated"; rating.value = recordFeedback[feedbackKey]?.rating ?? "";
      const save = node("button", "Save record feedback", "secondary");
      save.addEventListener("click", async () => {
        try {
          const value = {notes: notes.value, rating: rating.value ? Number(rating.value) : null};
          await api(`/api/runs/${selectedRun}/feedback`, {records: {[feedbackKey]: value}});
          recordFeedback[feedbackKey] = value; save.textContent = "Saved separately";
        } catch (error) { notice(error); }
      });
      feedback.append(notes, rating, save); card.append(feedback);
    }
    if (data.id !== undefined) {
      card.id = `record-${data.id}`;
      {
        const links = wrapped.links || [];
        const strip = node("div", undefined, "report-links");
        for (const link of links) {
          const button = node("button", `${link.relation}: ${link.id}`, "quiet");
          button.addEventListener("click", () => followRecord(link.id).catch(notice));
          strip.append(button);
        }
        card.append(strip);
      }
    }
    $("records").append(card);
  }
}
action("new", async () => { selectedRun = null; selectedHome = null; selectedCapture = null; rememberView(null); history.replaceState(null, "", location.pathname); show("setup", "New simulation"); if (profileHome !== $("primary").value) await refreshActorProfile(); });
action("reload", refreshCatalog);
action("load-canon", async () => { const name = prompt("New home folder name", "spamuel-new"); if (name) { await job("/api/homes/canon", {name}); await refreshCatalog(); } });
action("generate", async () => { const template = readTemplate(false); const result = await job("/api/template-plans", template); $("plan-json").value = JSON.stringify(result.plan, null, 2); $("plan-card").hidden = false; $("job-status").textContent = "Plan ready for review."; previewPlan(); });
action("preview", async () => previewPlan());
action("create-run", async () => {
  const primary = $("primary").value;
  if (!primary) throw new Error("Choose a primary character.");
  const participants = [{id: "primary", home: primary}];
  const edges = [];
  if ($("secondary").value) {
    participants.push({id: "other", home: $("secondary").value});
    edges.push({source: "primary", target: "other", name: "Other Spamuel"}, {source: "other", target: "primary", name: "Other Spamuel"});
  }
  const result = await api("/api/runs", {template: readTemplate(true)});
  await refreshCatalog(); await openRun(result.run_id);
});
action("fork", async () => { const name = prompt("New home folder name", selectedHome.folder + "-fork"); if (name) { await job("/api/homes/fork", {source: selectedHome.id, snapshot: $("snapshot").value, name}); await refreshCatalog(); } });
action("earlier", async () => { shownHistory += 100; showHistory(); });
$("snapshot").addEventListener("change", async () => { try { const result = await api(`/api/history?home=${encodeURIComponent(selectedHome.id)}&snapshot=${encodeURIComponent($("snapshot").value)}`); historyRows = result.messages; shownHistory = 100; showHistory(); } catch (error) { notice(error); } });
for (const button of document.querySelectorAll("[data-command]")) button.addEventListener("click", async () => { try { await api(`/api/runs/${selectedRun}/${button.dataset.command}`, {}); await updateRun(); } catch (error) { notice(error); } });
for (const button of document.querySelectorAll("[data-tab]")) button.addEventListener("click", () => selectRunTab(button.dataset.tab));
$("record-filter").addEventListener("input", renderRecords);
action("save-review", async () => { const rating = $("review-rating").value; await api(`/api/runs/${selectedRun}/feedback`, {notes: $("review-notes").value, rating: rating ? Number(rating) : null}); $("review-saved").textContent = "Saved separately from captured evidence."; });
setInterval(async () => { if (!selectedRun || polling) return; polling = true; try { await updateRun(); } catch (error) { notice(error); } finally { polling = false; } }, 2000);
async function openRoute() {
  const run = location.hash.match(/^#run\/([A-Za-z0-9_-]+)(?:\/([a-z]+))?$/);
  if (run) {
    const tab = run[2] || "conversation";
    if (!RUN_TABS.includes(tab)) throw new Error(`Unknown run tab: ${tab}`);
    if (run[1] !== selectedRun) return openRun(run[1], tab);
    return selectRunTab(tab);
  }
  const capture = location.hash.match(/^#capture\/([A-Za-z0-9_-]+)$/);
  if (capture) {
    if (capture[1] !== selectedCapture) return openCapture(capture[1]);
    return;
  }
  const home = location.hash.match(/^#home\/(.+)$/);
  if (home) {
    const id = decodeURIComponent(home[1]);
    const selected = catalog.homes.find(item => item.id === id);
    if (!selected) throw new Error("This character home is not registered.");
    if (selectedHome?.id !== id) return openHome(selected);
    return;
  }
  if (location.hash) throw new Error("Unknown simulation view.");
  selectedRun = null; selectedHome = null; selectedCapture = null;
  rememberView(null);
  show("setup", "New simulation");
  if (profileHome !== $("primary").value) await refreshActorProfile();
}
refreshCatalog().then(() => {
  if (!location.hash) {
    const saved = savedView();
    if (saved) location.hash = saved;
  }
  return openRoute();
}).catch(notice);
window.addEventListener("hashchange", () => openRoute().catch(notice));

async function openCapture(id) {
  selectedRun = null;
  selectedHome = null;
  selectedCapture = id;
  const capture = await api(`/api/captures/${id}`);
  if (selectedCapture !== id) return;
  show("capture", capture.title || id, "Captured evidence");
  rememberView(`capture/${id}`);
  location.hash = `capture/${id}`;
  const panel = $("capture-items");
  panel.replaceChildren();
  for (const item of capture.items || []) {
    const section = node("section");
    section.append(node("h2", item.title || item.id));
    const seen = new Set();
    for (const pack of item.packages || []) {
      section.append(node("h3", pack.title));
      for (const wrapped of pack.records || []) {
        const data = wrapped.data || wrapped;
        if (seen.has(data.id)) continue;
        seen.add(data.id);
        const card = node("details", undefined, "record");
        card.append(node("summary", `${data.id}: ${wrapped.__type__ || "Record"}`));
        if (data.content) card.append(node("div", data.content, "message-body"));
        card.append(node("pre", JSON.stringify(wrapped, null, 2)));
        section.append(card);
      }
    }
    if (!(item.packages || []).length) {
      const details = node("details");
      details.append(node("summary", "All captured fields"), node("pre", JSON.stringify(item, null, 2)));
      section.append(details);
    }
    panel.append(section);
  }
  const feedback = await api(`/api/captures/${id}/feedback`);
  $("capture-notes").value = feedback.review?.notes || "";
}
action("save-capture-review", async () => {
  await api(`/api/captures/${selectedCapture}/feedback`, {notes: $("capture-notes").value});
});

async function followRecord(id) {
  let selected = document.getElementById(`record-${id}`);
  if (!selected && recordCheckpoint) {
    const record = await api(`/api/runs/${selectedRun}/record?checkpoint=${encodeURIComponent(recordCheckpoint)}&id=${id}`);
    record.links = await api("/api/record-links", {record: record.data});
    packages.unshift(record);
    $("record-filter").value = "";
    renderRecords();
    selected = document.getElementById(`record-${id}`);
  }
  if (!selected) throw new Error(`Record ${id} is outside this captured collection.`);
  selected.open = true;
  selected.scrollIntoView({behavior: "smooth", block: "center"});
}
function renderCycles(display, target) {
  for (const cycle of display.cycles || []) {
    const group = node("details");
    group.open = true;
    group.append(node("summary", cycle.title));
    for (const section of cycle.sections) {
      group.append(node("h3", section.title));
      if (section.note) group.append(node("p", section.note, "muted"));
      for (const row of section.rows || []) {
        const card = display.cognition.records[String(row.record_id ?? row.id)];
        const details = node("details", undefined, "record");
        details.append(node("summary", row.label || card?.role || (row.text || "Recorded state").slice(0,140)));
        details.append(node("div", card?.text || row.text || "", "message-body"));
        details.append(node("pre", JSON.stringify(card || row, null, 2)));
        group.append(details);
      }
    }
    target.prepend(group);
  }
}
