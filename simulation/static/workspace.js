/* Template editing, ratings, and the mounted debugger share the simulation shell. */
"use strict";
let importedTemplate = null;
let analysisLoadedRun = null;
let analysisRenderKey = "";
let activeTab = "conversation";
let manualAnalysisRunning = false;

const directorDefaults = {
  duration_days: 1, scenes_per_day: 3, minimum_turns_per_scene: 2,
  first_hour: 8, last_hour: 22,
  daily_rhythm: "Ordinary work, meals, sleep, and time away from the phone.",
  story_threads: "", scheduled_events: "",
  guidance: "Let conversations breathe. Mix ordinary chat with consequential events.",
  actor_guidance: "Respond naturally and preserve the user's established voice."
};
const analysisDefaults = {
  automatic: true,
  context: "The character is an AI companion. Knowing this is part of the intended experience. Consensual adult romance is allowed.",
  messages_per_section: 40,
  criteria: [
    {id: "fluency", name: "Conversational sense", enabled: true, instructions: "From Jesse's viewpoint, assess whether the reply makes sense, understands context, and responds to what Jesse meant. Jokes, teasing, refusal, and disagreement can be appropriate. Penalize incoherence, repetitive moves, and irrelevant replies."},
    {id: "immersion", name: "AI companion immersion", enabled: true, instructions: "Assess Spamuel as a continuous AI companion with personality, emotions, memory, and initiative. Knowing he is an AI is intended. Consensual adult romance is allowed. Do not reward generic assistant compliance. Assess introspective reports when present, without requiring them or assuming that unseen private activity occurred."}
  ]
};

function field(label, id, value, type = "text") {
  const wrapper = node("label", label);
  const input = node(type === "textarea" ? "textarea" : "input");
  input.id = id;
  if (type === "textarea") input.rows = 3;
  else input.type = type;
  if (type === "checkbox") { input.checked = value; wrapper.className = "check-label"; }
  else input.value = value;
  wrapper.append(input);
  return wrapper;
}

function installWorkspace() {
  for (const id of ["primary", "secondary"]) $(id).addEventListener("change", () => { importedTemplate = null; });
  const scenarioCard = $("brief").closest(".card");
  const settings = node("details");
  settings.open = true;
  settings.append(node("summary", "Duration, pacing, and director"));
  settings.append(field("Scenario title", "scenario-title", "A month with Spamuel"));
  const grid = node("div", undefined, "settings-grid");
  for (const [key, label, min, max] of [
    ["duration_days", "Days", 1, 366], ["scenes_per_day", "Conversations per day", 1, 24],
    ["minimum_turns_per_scene", "Minimum user messages per conversation", 1, null],
    ["first_hour", "First conversation hour", 0, 23], ["last_hour", "Last conversation hour", 0, 23]
  ]) {
    const control = field(label, `director-${key}`, directorDefaults[key], "number");
    control.lastChild.min = min;
    if (max !== null) control.lastChild.max = max;
    grid.append(control);
  }
  settings.append(grid);
  for (const [key, label] of [["daily_rhythm", "Daily rhythm and availability"], ["story_threads", "Ongoing story threads"],
    ["scheduled_events", "Events on particular days"], ["guidance", "Director guidance"], ["actor_guidance", "Jesse / user actor guidance"]]) {
    settings.append(field(label, `director-${key}`, directorDefaults[key], "textarea"));
  }
  scenarioCard.insertBefore(settings, $("generate"));
  scenarioCard.insertBefore(node("p", "The director creates an editable daily outline. The companion chooses its own responses.", "muted"), $("generate"));
  const tools = node("div", undefined, "template-tools");
  const exportButton = node("button", "Export template", "secondary");
  exportButton.id = "export-template";
  const importInput = node("input"); importInput.type = "file"; importInput.accept = ".json,application/json"; importInput.id = "import-template";
  tools.append(exportButton, field("Import template JSON", "template-upload-label", ""));
  tools.lastChild.lastChild.replaceWith(importInput);
  scenarioCard.prepend(tools);
  const ratingCard = node("div", undefined, "card");
  ratingCard.append(node("h2", "Automatic analysis"));
  ratingCard.append(field("Run ratings when the simulation finishes", "automatic-analysis", true, "checkbox"));
  ratingCard.append(field("Context for the rating agents", "analysis-context", analysisDefaults.context, "textarea"));
  const advanced = node("details");
  advanced.append(node("summary", "Edit rating agents"), field("Criteria JSON: edit, add, or disable agents", "criteria-json", JSON.stringify(analysisDefaults.criteria, null, 2), "textarea"));
  $("criteria-json")?.setAttribute("rows", "15");
  ratingCard.append(advanced);
  $("setup").insertBefore(ratingCard, $("plan-card"));

  const tabs = $("run").querySelector(".tabs");
  for (const [id, label] of [["script", "Script"], ["analysis", "Analysis"], ["debugger", "Memory debugger"]]) {
    const button = node("button", label); button.dataset.tab = id;
    button.addEventListener("click", () => selectRunTab(id)); tabs.append(button);
    const panel = node("div"); panel.id = id; panel.hidden = true; $("run").append(panel);
  }
  $("analysis").append(node("div", undefined, "card"));
  const analysisCard = $("analysis").firstChild;
  analysisCard.append(node("h2", "Rating agents"), field("Analysis settings JSON", "run-analysis-settings", "", "textarea"));
  $("run-analysis-settings").rows = 16;
  const save = node("button", "Save settings", "secondary"); save.id = "save-analysis-settings";
  const rerun = node("button", "Analyze completed run"); rerun.id = "rerun-analysis";
  const actions = node("div", undefined, "actions"); actions.append(save, rerun); analysisCard.append(actions);
  const results = node("div", undefined, "card"); results.id = "analysis-results"; $("analysis").append(results);
  const help = node("p", "The debugger reads captured attempts. Full evidence appears after an attempt pauses or finishes.", "muted");
  const standalone = node("a", "Open debugger in its own tab"); standalone.id = "debugger-link"; standalone.target = "_blank";
  const frame = node("iframe"); frame.id = "debugger-frame"; frame.title = "Memory debugger";
  $("debugger").append(help, standalone, frame);
  const addScene = node("button", "Add conversation", "secondary"); addScene.id = "add-scene";
  $("plan-preview").after(addScene);

  action("export-template", async () => downloadTemplate(readTemplate(!$("plan-card").hidden)));
  importInput.addEventListener("change", async () => {
    try {
      if (!importInput.files.length) return;
      await applyTemplate(JSON.parse(await importInput.files[0].text()));
    } catch (error) { notice(error); }
  });
  action("add-scene", async () => {
    const plan = JSON.parse($("plan-json").value);
    const last = plan.steps.at(-1);
    plan.steps.push({id: `scene-${crypto.randomUUID().slice(0,8)}`, at: last?.at || $("start-time").value,
      instruction: "An ordinary conversation.", actor_intent: "Share something and respond naturally.",
      completion: "The conversation has had room to develop.", fact_ids: [], target: last?.target || "primary",
      speaker: last?.speaker || actorSummary.speaker_name, minimum_turns: 2});
    $("plan-json").value = JSON.stringify(plan, null, 2); previewPlan();
  });
  action("save-analysis-settings", async () => {
    await api(`/api/runs/${selectedRun}/analysis-settings`, JSON.parse($("run-analysis-settings").value));
  });
  action("rerun-analysis", async () => {
    manualAnalysisRunning = true;
    try {
      await api(`/api/runs/${selectedRun}/analysis-settings`, JSON.parse($("run-analysis-settings").value));
      $("analysis-results").replaceChildren(node("p", "Analysis started. Results update here."));
      analysisRenderKey = "";
      await job(`/api/runs/${selectedRun}/analyze`, {});
    } finally { manualAnalysisRunning = false; }
  });
}

function readTemplate(withPlan) {
  let participants = [{id: "primary", home: $("primary").value}];
  let edges = [];
  if ($("secondary").value) {
    participants.push({id: "other", home: $("secondary").value});
    edges = [{source: "primary", target: "other", name: "Other companion"}, {source: "other", target: "primary", name: "Other companion"}];
  }
  // Preserve an imported graph, including participants the simple picker cannot display.
  if (importedTemplate && importedTemplate.participants[0].home === $("primary").value) {
    participants = importedTemplate.participants; edges = importedTemplate.edges || [];
  }
  const director = {};
  for (const key of Object.keys(directorDefaults)) {
    const control = $(`director-${key}`);
    director[key] = control.type === "number" ? Number(control.value) : control.value;
  }
  const plan = withPlan ? JSON.parse($("plan-json").value) : null;
  if (plan) { plan.director = director; plan.title = $("scenario-title").value; plan.brief = $("brief").value; }
  return {schema_id: "pm.simulation-template.v1", title: $("scenario-title").value,
    brief: $("brief").value, start_time: $("start-time").value || null, participants, edges, director,
    analysis: {...analysisDefaults, ...(importedTemplate?.analysis || {}), automatic: $("automatic-analysis").checked,
      context: $("analysis-context").value, criteria: JSON.parse($("criteria-json").value)},
    plan};
}

function downloadTemplate(template) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(template, null, 2)], {type: "application/json"}));
  const anchor = node("a"); anchor.href = url; anchor.download = "simulation-template.json"; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function applyTemplate(template) {
  if (template.schema_id !== "pm.simulation-template.v1") throw new Error("Choose a pm.simulation-template.v1 JSON file.");
  template = await api("/api/templates/validate", template);
  if (!catalog.homes.some(home => home.id === template.participants[0].home)) throw new Error("The template's source home is not registered here. Edit its home path first.");
  $("primary").value = template.participants[0].home;
  $("secondary").value = template.participants[1]?.home || "";
  await refreshActorProfile();
  importedTemplate = template;
  $("scenario-title").value = template.title;
  $("brief").value = template.brief;
  if (template.start_time) $("start-time").value = template.start_time;
  for (const [key, value] of Object.entries({...directorDefaults, ...template.director})) $(`director-${key}`).value = value;
  const settings = {...analysisDefaults, ...template.analysis};
  $("automatic-analysis").checked = settings.automatic;
  $("analysis-context").value = settings.context;
  $("criteria-json").value = JSON.stringify(settings.criteria, null, 2);
  showTimeGap();
  if (template.plan) {
    $("plan-json").value = JSON.stringify(template.plan, null, 2); $("plan-card").hidden = false; previewPlan();
  }
}

function renderSceneEditor(plan) {
  let previousDay = "";
  let dayGroup = null;
  plan.steps.forEach((step, index) => {
    const day = step.at.slice(0, 10);
    if (day !== previousDay) {
      dayGroup = node("details", undefined, "scene-day"); dayGroup.open = index === 0;
      dayGroup.append(node("summary", `${day} · ${plan.steps.filter(scene => scene.at.startsWith(day)).length} conversations`));
      $("plan-preview").append(dayGroup); previousDay = day;
    }
    const details = node("details", undefined, "scene-editor");
    details.append(node("summary", `${step.at.slice(11,16)} · ${step.instruction.slice(0,100)} · at least ${step.minimum_turns || 1} messages`));
    for (const [key, label, type] of [["at", "Time with UTC offset", "text"], ["instruction", "Current circumstances", "textarea"],
      ["actor_intent", "Private user motive", "textarea"], ["completion", "Completion condition", "textarea"],
      ["minimum_turns", "Minimum user messages", "number"], ["target", "Recipient participant ID", "text"], ["speaker", "User name", "text"]]) {
      const control = field(label, `scene-${index}-${key}`, step[key] ?? 1, type);
      control.lastChild.addEventListener("change", event => {
        const current = JSON.parse($("plan-json").value);
        const oldTime = current.steps[index].at;
        current.steps[index][key] = type === "number" ? Number(event.target.value) : event.target.value;
        if (key === "at") {
          for (const fact of current.facts) if (current.steps[index].fact_ids.includes(fact.id) && fact.at === oldTime) fact.at = event.target.value;
        }
        $("plan-json").value = JSON.stringify(current, null, 2);
      });
      details.append(control);
    }
    const remove = node("button", "Remove conversation and its exclusive facts", "secondary");
    remove.addEventListener("click", () => {
      const current = JSON.parse($("plan-json").value);
      const removed = current.steps.splice(index, 1)[0];
      const retained = new Set(current.steps.flatMap(scene => scene.fact_ids));
      current.facts = current.facts.filter(fact => !removed.fact_ids.includes(fact.id) || retained.has(fact.id));
      $("plan-json").value = JSON.stringify(current, null, 2); previewPlan();
    });
    details.append(remove); dayGroup.append(details);
  });
}

function selectRunTab(id) {
  if (!RUN_TABS.includes(id)) throw new Error(`Unknown run tab: ${id}`);
  activeTab = id;
  for (const panel of RUN_TABS) $(panel).hidden = panel !== id;
  for (const tab of document.querySelectorAll("[data-tab]")) tab.classList.toggle("selected", tab.dataset.tab === id);
  if (selectedRun) {
    const route = `run/${encodeURIComponent(selectedRun)}/${id}`;
    if (location.hash !== `#${route}`) location.hash = route;
    rememberView(route);
  }
  if (id === "debugger") mountDebugger();
}

function mountDebugger() {
  if (!runState || !runState.reports.length) return;
  const url = `/debugger/?embedded=1#/run/${selectedRun}/${runState.attempt}/overview`;
  if ($("debugger-frame").getAttribute("src") !== url) $("debugger-frame").src = url;
  $("debugger-link").href = `/debugger/#/run/${selectedRun}/${runState.attempt}/overview`;
}

async function loadWorkspaceRun(id) {
  if (id !== selectedRun) return;
  const template = await api(`/api/runs/${id}/template`);
  if (id !== selectedRun) return;
  $("script").replaceChildren();
  const card = node("div", undefined, "card");
  card.append(node("h2", "Saved simulation template"));
  const download = node("button", "Download template", "secondary"); download.addEventListener("click", () => downloadTemplate(template));
  const reuse = node("button", "Use for a new independent run", "secondary");
  reuse.addEventListener("click", async () => {
    try { selectedRun = null; rememberView(null); history.replaceState(null, "", location.pathname); show("setup", "New simulation"); await applyTemplate(template); }
    catch (error) { notice(error); }
  });
  card.append(download, reuse, node("pre", JSON.stringify(template, null, 2))); $("script").append(card);
  mountDebugger();
}

async function updateAnalysis(id) {
  if (!$("analysis-results")) return;
  const data = await api(`/api/runs/${id}/analysis`);
  if (id !== selectedRun) return;
  if (analysisLoadedRun !== id) {
    analysisLoadedRun = id; analysisRenderKey = "";
    $("run-analysis-settings").value = JSON.stringify(data.settings, null, 2);
  }
  $("rerun-analysis").disabled = runState.status !== "completed" || manualAnalysisRunning;
  const key = JSON.stringify([id, data.report?.id, data.report?.status, data.report?.sections.length, data.failure]);
  if (key === analysisRenderKey) return;
  analysisRenderKey = key;
  const panel = $("analysis-results"); panel.replaceChildren(node("h2", "Automatic findings"));
  if (data.failure && !data.report) panel.append(node("pre", data.failure.error));
  if (!data.report) { panel.append(node("p", "Ratings appear after the simulation completes. Your ratings remain separate.", "muted")); return; }
  const report = data.report;
  panel.append(node("p", `${report.status} · ${report.delivery_count} public deliveries · ${report.sections.length} section ratings. Historical context: ${report.historical_reference_count} sampled reference lines.`, "muted"));
  if (report.error) panel.append(node("pre", report.error));
  const summary = node("div", undefined, "rating-summary");
  for (const criterion of report.settings.criteria.filter(item => item.enabled)) {
    const scored = report.sections.filter(section => section.criterion_id === criterion.id && section.result.score !== null);
    const mean = scored.length ? (scored.reduce((sum, section) => sum + section.result.score, 0) / scored.length).toFixed(2) : "unrated";
    summary.append(node("div", `${criterion.name}: ${mean}${scored.length ? " / 5" : ""} · ${scored.length} sections`, "rating-score"));
  }
  panel.append(summary);
  for (const section of report.sections) {
    const details = node("details", undefined, "rating-finding");
    details.append(node("summary", `${section.criterion_id} · deliveries ${section.first_index + 1}–${section.last_index + 1} · ${section.result.score ?? "insufficient evidence"}`), node("p", section.result.summary));
    for (const citation of section.result.citations) {
      const button = node("button", `#${citation.index + 1}: ${citation.quote}`, "secondary");
      button.addEventListener("click", () => { selectRunTab("conversation"); const target = $(`delivery-${citation.index}`); target?.scrollIntoView({block: "center"}); target?.animate([{background: "#d9eafa"}, {background: "transparent"}], {duration: 2500}); });
      details.append(button, node("p", citation.explanation, "muted"));
    }
    panel.append(details);
  }
  if (activeTab === "debugger") mountDebugger();
}

installWorkspace();
