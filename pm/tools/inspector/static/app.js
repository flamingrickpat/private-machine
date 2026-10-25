/* The inspector page: one file, no build step, no external script.

Every route is a hash route (``#/cycle/295/shard/entity/3``), so the browser
back button and middle-click work, and the server still serves only three
files. Each route is its own page: the cycle list, one cycle, one shard, one
record, one cluster. Nothing is buried under a long list, and every id in a
table is a link to the page of that id.

Every call reads one loaded checkpoint, so reload, back, and forward are
always safe. The theme is light by default, dark on request, remembered per
browser. */
"use strict";

const THEME_KEY = "pm-inspector-theme";
const PAGE_SIZES = [50, 100, 250];
const $ = (selector, root) => (root || document).querySelector(selector);
const page = $("#page");

/* ------------------------------------------------------------------ dom */

function h(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const pair of Object.entries(attrs || {})) {
    const name = pair[0];
    const value = pair[1];
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

function text(nodes) {
  /* Return a comma-separated list of nodes, so an empty list stays empty. */
  if (!nodes.length) return "";
  const flat = [];
  nodes.forEach((node, index) => {
    if (index) flat.push(", ");
    flat.push(node);
  });
  return flat;
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "-";
  return Number(value).toLocaleString("en-US");
}

function short(value, limit) {
  const value2 = String(value === null || value === undefined ? "" : value);
  const max = limit || 160;
  return value2.length <= max ? value2 : value2.slice(0, max) + "\u2026";
}

function time(value) {
  if (!value) return "-";
  return String(value).replace("T", " ").slice(0, 16);
}

function tokens(value) {
  return h("span", { class: "tokens" }, fmt(value) + " tk");
}

function cell(content, cls) {
  return h("td", { class: cls || "" }, content);
}

function num(content) {
  return h("td", { class: "num" }, content);
}

function row(...cells) {
  const tr = h("tr");
  for (const item of cells) tr.append(item);
  return tr;
}

function linkRow(route, ...cells) {
  /* One table row that opens ``route`` when clicked anywhere but on a link. */
  const tr = row(...cells);
  tr.className = "link";
  tr.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    location.hash = route;
  });
  return tr;
}

function table(headers, rows) {
  const head = row();
  for (const name of headers) head.append(h("th", {}, name));
  const body = h("tbody");
  for (const item of rows) body.append(item);
  return h("table", {}, h("thead", {}, head), body);
}

function stat(label, value, route) {
  const box = h("div", { class: "stat" }, h("b", {}, fmt(value)), h("span", {}, label));
  return route ? h("a", { class: "statlink", href: route }, box) : box;
}

function badge(label, tone) {
  return h("span", { class: tone ? "badge " + tone : "badge" }, label);
}

function flag(label, clean) {
  return h("span", { class: clean ? "flag ok" : "flag" }, label);
}

function kv(name, value) {
  const shown = value === "" || value === null || value === undefined ? "-" : String(value);
  return [h("div", {}, name), h("div", {}, shown)];
}

function fieldsBlock(fields) {
  const flat = [];
  for (const pair of Object.entries(fields || {})) flat.push(...kv(pair[0], short(pair[1], 4000)));
  if (!flat.length) return h("p", { class: "meta" }, "no stored fields");
  return h("div", { class: "kv" }, flat);
}

function costTable(rows) {
  return table(["part", "tokens"], rows.map((pair) => row(cell(pair[0]), num(tokens(pair[1])))));
}

function noteList(notes) {
  if (!notes || !notes.length) return null;
  return h("div", {}, h("h2", {}, "Reconstruction notes"), h("ul", { class: "tight" }, notes));
}

function notice(message) {
  const box = $("#notice");
  box.textContent = String(message);
  box.classList.remove("hidden");
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => box.classList.add("hidden"), 8000);
}

async function api(path) {
  const response = await fetch("/api" + path);
  let data;
  try {
    data = await response.json();
  } catch (error) {
    data = { error: "no answer from the inspector" };
  }
  if (!response.ok || data.error) throw new Error(data.error || "HTTP " + response.status);
  return data;
}

/* --------------------------------------------------------------- routing */

function current() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [pathPart, queryPart] = raw.split("?");
  return { parts: pathPart.split("/").filter(Boolean), query: new URLSearchParams(queryPart || "") };
}

function href(...parts) {
  return "#/" + parts.filter((part) => part !== "" && part !== undefined && part !== null).join("/");
}

function queryHref(base, params) {
  const value = params.toString();
  return base + (value ? "?" + value : "");
}

function crumbs(entries) {
  const bar = clear($("#crumbs"));
  entries.forEach((entry, index) => {
    if (index) bar.append(h("span", { class: "sep" }, "/"));
    bar.append(entry.length > 1 ? h("a", { href: entry[1] }, entry[0]) : h("span", {}, entry[0]));
  });
}

function pageHead(...kids) {
  return h("div", { class: "pagehead" }, kids);
}

function siblings(entries) {
  return h("div", { class: "rownav" }, entries.map((entry) => h("a", { href: entry[1] }, entry[0])));
}

function loading(message) {
  clear(page).append(h("p", { class: "meta" }, message || "loading\u2026"));
}

function failed(error) {
  const message = error.message || String(error);
  crumbs([["Did not open"]]);
  clear(page).append(h("h1", {}, "This page did not open"), h("p", { class: "waste" }, message),
                     h("div", { class: "rownav" }, h("a", { href: "#/" }, "overview"),
                                              h("a", { href: "#/cycles" }, "cycles"),
                                              h("a", { href: "#/records" }, "records")));
  notice(message);
}

function show(promise, render) {
  promise.then(render).catch(failed);
}

function pageParams(query, keep) {
  /* Copy the reader's own filters out of the current hash into a new query. */
  const params = new URLSearchParams();
  for (const key of keep) {
    if (query.get(key)) params.set(key, query.get(key));
  }
  return params;
}

function pager(query, pageData, base, keep) {
  const size = pageData.items.length || Number(query.get("size") || 100);
  const last = pageData.offset + pageData.items.length;
  const before = pageParams(query, keep);
  before.set("offset", Math.max(0, pageData.offset - size));
  const after = pageParams(query, keep);
  after.set("offset", last);
  return h("div", { class: "pager" },
           pageData.offset > 0 ? h("a", { href: queryHref(base, before) }, "previous") : h("span", { class: "meta" }, "\u2014"),
           h("span", { class: "meta here" },
             fmt(pageData.offset + 1) + "\u2013" + fmt(last) + " of " + fmt(pageData.total)),
           last < pageData.total ? h("a", { href: queryHref(base, after) }, "next") : h("span", { class: "meta" }, "end"));
}

function jumpForm(label, base) {
  const input = h("input", { type: "number", size: "9", placeholder: "tick number" });
  const submit = () => {
    const value = input.value.trim();
    if (value) location.hash = base + value;
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submit();
  });
  const form = h("form", { class: "bar" }, h("label", {}, label, input), h("button", { type: "submit" }, "go"));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  return form;
}

/* ---------------------------------------------------------------- links */

function aRecord(id, label) {
  return h("a", { href: href("record", id) }, label === undefined ? "#" + id : String(label));
}

function aCycle(tick, label) {
  return h("a", { href: href("cycle", tick) }, label === undefined ? "cycle " + tick : String(label));
}

function aCluster(id, label) {
  return h("a", { href: href("cluster", id) }, label === undefined ? "cluster #" + id : String(label));
}

function shardHref(tick, key) {
  return href("cycle", tick, "shard", key.kind, key.ref || "-");
}

function aShard(tick, key, label) {
  const name = key.kind === "self" ? "self" : key.kind + " " + (key.ref || "");
  return h("a", { href: shardHref(tick, key) }, label === undefined ? name : String(label));
}

function aRecords(params, label) {
  return h("a", { href: queryHref("#/records", params) }, label || "records");
}

/* ------------------------------------------------------------ table rows */

function cycleRow(item) {
  return linkRow(href("cycle", item.tick_id),
                 cell(aCycle(item.tick_id, item.tick_id), "num"),
                 cell(time(item.timestamp), "dim"),
                 cell(item.slot_id),
                 num(fmt(item.shards)),
                 num(fmt(item.shard_items)),
                 num(fmt(item.tail_items)),
                 num(fmt(item.csm_items)),
                 num(tokens(item.token_estimate)),
                 num(fmt(item.llm_calls)),
                 cell(short(item.summary, 90), "dim"));
}

function shardRow(tick, shard) {
  return linkRow(shardHref(tick, shard.key),
                 cell(aShard(tick, shard.key, shard.name)),
                 cell(shard.title, "mono dim"),
                 badge(shard.in_story ? "in story" : "out of story", shard.in_story ? "on" : ""),
                 num(fmt(shard.items.length)),
                 num(fmt(shard.tail.length)),
                 num(tokens(shard.items_tokens)),
                 num(tokens(shard.tail_tokens)),
                 num(tokens(shard.framing_tokens)),
                 num(tokens(shard.block_tokens)),
                 cell(short(shard.focus, 70), "dim"));
}

function itemRow(item) {
  const drift = item.stored_tokens && item.stored_tokens !== item.tokens;
  return linkRow(href("record", item.record_id),
                 cell(aRecord(item.record_id, item.record_id), "num"),
                 cell(item.line, "line"),
                 num(tokens(item.tokens)),
                 num(item.stored_tokens ? h("span", { class: drift ? "waste" : "" }, fmt(item.stored_tokens)) : "-"),
                 num(fmt(item.entered_tick)),
                 num(item.activation_at_entry === undefined ? "-" : Number(item.activation_at_entry).toFixed(2)),
                 cell(short(item.reason || item.required_by || "", 70), "dim"));
}

function lineRow(item) {
  return linkRow(href("record", item.record_id),
                 cell(aRecord(item.record_id, item.kind || "#" + item.record_id)),
                 cell(item.line, "line"),
                 num(tokens(item.tokens)),
                 num(item.content_tokens === undefined ? "-" : fmt(item.content_tokens)));
}

function csmRow(item) {
  return linkRow(href("record", item.record_id),
                 cell(aRecord(item.record_id, item.record_id), "num"),
                 num(Number(item.activation).toFixed(3)),
                 num(Number(item.base_activation).toFixed(3)),
                 num(fmt(item.ticks_present)),
                 num(fmt(item.entered_tick)),
                 num(tokens(item.tokens)),
                 badge(item.in_workspace ? "in workspace" : "csm only", item.in_workspace ? "on" : ""),
                 cell(item.endorsers.join(", ") || "-", "dim"));
}

function relationTable(relations) {
  if (!relations.length) return h("p", { class: "meta" }, "nothing points here");
  return table(["field", "record", "kind", "content"], relations.map((relation) => linkRow(
    href("record", relation.record_id),
    cell(relation.field, "mono"),
    cell(aRecord(relation.record_id, relation.record_id), "num"),
    cell(relation.target.kind + (relation.target.subtype ? " \u00b7 " + relation.target.subtype : ""), "dim"),
    cell(short(relation.target.preview, 110)))));
}

function duplicateTable(groups) {
  if (!groups || !groups.length) return h("p", { class: "meta" }, "no repeated content in this cycle");
  return table(["line", "records", "tokens", "wasted", "exact copy"], groups.map((group) => row(
    cell(short(group.line, 120), "line"),
    cell(text(group.record_ids.map((id) => aRecord(id)))),
    num(tokens(group.tokens)),
    num(h("span", { class: "waste" }, fmt(group.wasted_tokens))),
    cell(group.identical_text ? "yes" : "case or spacing", "dim"))));
}

/* ------------------------------------------------------------------ pages */

function pageOverview() {
  crumbs([["Overview"]]);
  loading();
  show(api("/summary").then((summary) => {
    clear(page).append(
      pageHead(h("h1", {}, summary.companion_name + " \u00b7 memory"),
               h("span", { class: "meta" }, "read-only view of one checkpoint")),
      h("div", { class: "grid" },
        stat("records", summary.records, "#/records"),
        stat("cycles", summary.cycles, "#/cycles"),
        stat("entities", summary.entities),
        stat("kinds", summary.kinds.length, "#/types"),
        stat("card chars", summary.character_card_chars, "#/prompts"),
        stat("last cycle", summary.last_tick, href("cycle", summary.last_tick))),
      h("h2", {}, "Start here"),
      h("ul", { class: "tight" },
        h("li", {}, h("a", { href: "#/cycles" }, "cycles"), " \u2014 every accepted cycle; one click opens its shards"),
        h("li", {}, h("a", { href: "#/types" }, "types"), " \u2014 every kind and subtype, each one a filtered list"),
        h("li", {}, h("a", { href: "#/clusters" }, "clusters"), " \u2014 the episodic hierarchy, member by member"),
        h("li", {}, h("a", { href: "#/relations" }, "relations"), " \u2014 which stored fields carry the edges"),
        h("li", {}, h("a", { href: "#/prompts" }, "prompts"), " \u2014 the templates these pages render with")),
      h("h2", {}, "Kinds"),
      table(["kind", "records"], summary.kinds.map((kind) => linkRow(
        queryHref("#/records", new URLSearchParams({ kind: kind.kind })),
        cell(kind.kind), num(fmt(kind.count))))),
      h("h2", {}, "This checkpoint"),
      fieldsBlock({ checkpoint: summary.checkpoint, companion: summary.companion_name, user: summary.user_name,
                    first_tick: summary.first_tick, last_tick: summary.last_tick }));
  }));
}

function pageCycles(query) {
  crumbs([["Cycles"]]);
  const size = Number(query.get("size") || PAGE_SIZES[1]);
  const offset = Number(query.get("offset") || 0);
  const slot = query.get("slot") || "";
  loading("loading cycles\u2026");
  show(api(`/timeline?offset=${offset}&limit=${size}&slot=${encodeURIComponent(slot)}`).then((data) => {
    const slotInput = h("input", { type: "text", size: "10", value: slot, placeholder: "main" });
    const sizeSelect = h("select", {}, PAGE_SIZES.map((value) => {
      const option = h("option", { value }, String(value));
      if (value === size) option.setAttribute("selected", "selected");
      return option;
    }));
    const apply = () => {
      const params = new URLSearchParams();
      if (slotInput.value.trim()) params.set("slot", slotInput.value.trim());
      params.set("size", sizeSelect.value);
      location.hash = queryHref("#/cycles", params);
    };
    const filter = h("form", { class: "bar" }, h("label", {}, "slot", slotInput),
                     h("label", {}, "per page", sizeSelect), h("button", { type: "submit" }, "apply"),
                     slot ? h("a", { href: "#/cycles" }, "clear slot") : null);
    filter.addEventListener("submit", (event) => {
      event.preventDefault();
      apply();
    });
    const keep = ["slot", "size"];
    clear(page).append(
      pageHead(h("h1", {}, "Cycles"), h("span", { class: "meta" },
                                        fmt(data.total) + " accepted cycles, newest first. one click opens a cycle.")),
      filter, jumpForm("jump to tick", "#/cycle/"),
      pager(query, data, "#/cycles", keep),
      table(["tick", "time", "slot", "shards", "items", "tail", "csm", "prompt tk", "llm", "gwt summary"],
            data.items.map(cycleRow)),
      pager(query, data, "#/cycles", keep));
  }));
}

function pageCycle(tick) {
  crumbs([["Cycles", "#/cycles"], ["tick " + tick]]);
  loading("loading cycle " + tick + "\u2026");
  show(api(`/ticks/${tick}`).then((data) => {
    const totals = data.totals;
    const shared = Object.keys(data.shared_records || {}).length;
    const outside = totals.shards - totals.shards_in_story;
    clear(page).append(
      pageHead(h("h1", {}, "cycle " + tick),
               h("span", { class: "meta" }, time(data.timestamp) + " \u00b7 slot " + data.slot_id),
               siblings([["earlier cycle", href("cycle", tick - 1)], ["later cycle", href("cycle", tick + 1)],
                         ["story prefix", href("cycle", tick, "story")],
                         ["gwt turn", href("cycle", tick, "turn")],
                         ["cycle trace", href("cycle", tick, "trace")],
                         ["all cycles", "#/cycles"]])),
      h("p", {}, h("span", { class: "meta" }, "gwt summary: "),
        data.turn_summary || h("span", { class: "dim" }, "(none)")),
      h("div", { class: "grid" },
        stat("story tk (here)", totals.story_tokens, href("cycle", tick, "story")),
        stat("prompt tk (cycle)", data.token_estimate, href("cycle", tick, "trace")),
        stat("block tk", totals.block_tokens),
        stat("framing tk", totals.framing_tokens),
        stat("content tk", totals.content_tokens),
        stat("story lines", totals.item_lines),
        stat("shards", totals.shards),
        stat("out of story", outside),
        stat("csm tk", totals.csm_tokens),
        stat("new tk", totals.new_tokens),
        stat("duplicate tk", totals.duplicate_wasted_tokens),
        stat("llm calls", data.llm_calls, href("cycle", tick, "trace"))),
      h("div", { class: "flags" },
        flag(shared ? shared + " records held by more than one shard" : "no record is held twice", shared === 0),
        flag(totals.duplicate_wasted_tokens ? tokens(totals.duplicate_wasted_tokens).textContent
                                               + " spent on repeated content" : "no repeated content",
             totals.duplicate_wasted_tokens === 0),
        flag(outside ? outside + " shards outside the story prefix" : "every shard reached the story",
             outside === 0),
        flag(totals.framing_tokens ? tokens(totals.framing_tokens).textContent + " of block labels and headers"
                                   : "no framing cost", totals.framing_tokens === 0)),
      h("h2", {}, "Shards of this cycle"),
      table(["shard", "block title", "story", "items", "tail", "items tk", "tail tk", "framing tk", "block tk",
             "what it is about"], data.shards.map((shard) => shardRow(tick, shard))),
      h("h2", {}, "Story prefix"),
      h("p", {}, tokens(data.story.tokens), " \u00b7 ", h("a", { href: href("cycle", tick, "story") },
                                                          "open the whole prefix"), " \u00b7 shown: ",
        text(data.story.shards.map((key) => aShard(tick, key))) || h("span", { class: "dim" }, "nothing")),
      h("h2", {}, "Situational model"),
      table(["record", "activation", "base", "ticks", "entered", "line tk", "workspace"], data.csm.map(csmRow)),
      h("h2", {}, "New stimuli"),
      h("p", {}, tokens(data.new_items.tokens), " \u00b7 records: ",
        text(data.new_items.record_ids.map((id) => aRecord(id))) || h("span", { class: "dim" }, "none")),
      h("pre", {}, data.new_items.text || "(nothing new)"),
      h("h2", {}, "Situational-model text"),
      h("pre", {}, data.csm_text.text || "(empty)"),
      h("h2", {}, "Stimulus groups"),
      table(["key", "stimuli", "category", "flag"], data.groups.map((group) => row(
        cell(aShard(tick, { kind: group.kind, ref: group.ref })),
        cell(text(group.stimulus_ids.map((id) => aRecord(id))) || "-"),
        cell(group.category || "-", "dim"),
        cell(group.insignificant ? "insignificant" : "-", "waste")))),
      h("h2", {}, "Repeated content in this cycle"),
      duplicateTable(data.duplicate_content),
      h("h2", {}, "Records held by more than one shard"),
      shared ? fieldsBlock(data.shared_records) : h("p", { class: "meta" }, "none"));
  }));
}

function pageShard(tick, key) {
  const name = key.kind === "self" ? "self" : key.kind + " " + key.ref;
  crumbs([["Cycles", "#/cycles"], ["tick " + tick, href("cycle", tick)], ["shard " + name]]);
  loading("loading shard " + name + "\u2026");
  show(api(`/ticks/${tick}`).then((data) => {
    const found = data.shards.findIndex((shard) => shard.key.kind === key.kind && (shard.key.ref || "") === key.ref);
    if (found < 0) {
      failed(new Error("cycle " + tick + " holds no shard " + key.kind + ":" + (key.ref || "-")));
      return;
    }
    const shard = data.shards[found];
    const around = [];
    if (found > 0) around.push([data.shards[found - 1].name, shardHref(tick, data.shards[found - 1].key)]);
    around.push(["all shards of cycle " + tick, href("cycle", tick)]);
    if (found + 1 < data.shards.length) around.push([data.shards[found + 1].name, shardHref(tick, data.shards[found + 1].key)]);
    const drift = Object.keys(shard.token_drift || {});
    clear(page).append(
      pageHead(h("h1", {}, "shard \u00b7 " + shard.name),
               badge(shard.in_story ? "in the story prefix" : "not in the story prefix",
                     shard.in_story ? "on" : "off"),
               h("span", { class: "meta" }, "of "), aCycle(tick, "cycle " + tick)),
      siblings(around),
      fieldsBlock({ key: key.kind + ":" + (key.ref || "-"), block_title: shard.title, tail_label: shard.tail_label,
                    focus: shard.focus }),
      h("div", { class: "flags" },
        flag(shard.in_items_and_tail.length
             ? shard.in_items_and_tail.length + " records paid for twice (item list and tail list)"
             : "no record sits in both lists", shard.in_items_and_tail.length === 0),
        flag(shard.duplicate_content.length ? shard.duplicate_content.length + " repeated content groups"
                                            : "no repeated content", shard.duplicate_content.length === 0),
        flag(drift.length ? "stored token counts differ from the rendered cost" : "stored counts match the render",
             drift.length === 0),
        badge(shard.in_story ? "this text reached the model" : "this text did not reach the model",
              shard.in_story ? "on" : "off")),
      h("h2", {}, "What this block cost"),
      costTable([["item lines", shard.items_tokens], ["tail lines", shard.tail_tokens],
                 ["framing (labels and headers)", shard.framing_tokens], ["whole block", shard.block_tokens]]),
      h("h3", {}, "Token counts the cycle stored, against the render"),
      costTable([["stored item tokens", shard.stored_item_tokens], ["stored tail tokens", shard.stored_tail_tokens]]),
      h("h2", {}, "Rendered block"),
      h("pre", { class: "shard" }, shard.block_text),
      h("h2", {}, "Items"),
      table(["record", "story line", "tokens", "stored", "entered", "activation", "why it is here"],
            shard.items.map(itemRow)),
      h("h2", {}, "Tail (recent lines, no context)"),
      table(["record", "story line", "tokens", "content"], shard.tail.map(lineRow)),
      h("h2", {}, "Repeated content in this block"),
      shard.duplicate_content.length
      ? h("div", {}, shard.duplicate_content.map((group) => h("p", {},
          text(group.map((id) => aRecord(id))), " render the same words here")))
      : h("p", { class: "meta" }, "no repeated content in this block"),
      h("h2", {}, "Stored counts that disagree with the render"),
      drift.length ? fieldsBlock(Object.fromEntries(drift.map((id) => ["record " + id,
                                                                       shard.token_drift[id] + " tk different"])))
                   : h("p", { class: "meta" }, "every stored count matches"));
  }));
}

function pageStory(tick) {
  crumbs([["Cycles", "#/cycles"], ["tick " + tick, href("cycle", tick)], ["story prefix"]]);
  loading("loading story prefix\u2026");
  show(api(`/ticks/${tick}/story`).then((story) => {
    clear(page).append(
      pageHead(h("h1", {}, "story prefix \u00b7 cycle " + tick), tokens(story.tokens),
               h("a", { href: href("cycle", tick) }, "back to the cycle")),
      h("p", {}, "other characters: ", story.other_characters || h("span", { class: "dim" }, "nobody"),
        " \u00b7 shards in this order: ",
        text(story.shards.map((key) => aShard(tick, key))) || h("span", { class: "dim" }, "none")),
      h("pre", { class: "story" }, story.text || "(nothing to render)"),
      noteList(story.notes));
  }));
}

function pageCycleJson(tick, view) {
  crumbs([["Cycles", "#/cycles"], ["tick " + tick, href("cycle", tick)], [view]]);
  loading("loading " + view + "\u2026");
  show(api(`/ticks/${tick}/${view}`).then((data) => {
    clear(page).append(pageHead(h("h1", {}, view + " \u00b7 cycle " + tick),
                                h("a", { href: href("cycle", tick) }, "back to the cycle"),
                                h("a", { href: href("cycle", tick, "story"), }, "story prefix")),
                       h("pre", {}, JSON.stringify(data, null, 1)));
  }));
}

function pageRecords(query) {
  crumbs([["Records"]]);
  const size = Number(query.get("size") || 100);
  const offset = Number(query.get("offset") || 0);
  const parts = new URLSearchParams();
  for (const key of ["text", "kind", "subtype"]) {
    if (query.get(key)) parts.set(key, query.get(key));
  }
  loading("loading records\u2026");
  show(Promise.all([api(`/records?${parts.toString()}&offset=${offset}&limit=${size}`), api("/counts")])
    .then(([data, counts]) => {
      const textInput = h("input", { type: "search", value: query.get("text") || "",
                                     placeholder: "search content and speaker" });
      const kindSelect = selectOptions(counts.kinds.map((kind) => kind.kind), query.get("kind"), "all kinds");
      const subtypeSelect = selectOptions(subtypeNames(counts, query.get("kind")), query.get("subtype"),
                                          "all subtypes");
      const filter = h("form", { class: "bar" }, textInput, kindSelect, subtypeSelect,
                       h("button", { type: "submit" }, "search"),
                       query.get("text") || query.get("kind") || query.get("subtype")
                       ? h("a", { href: "#/records" }, "clear") : null,
                       h("a", { href: "#/types" }, "all types"));
      filter.addEventListener("submit", (event) => {
        event.preventDefault();
        const params = new URLSearchParams();
        if (textInput.value.trim()) params.set("text", textInput.value.trim());
        if (kindSelect.value) params.set("kind", kindSelect.value);
        if (subtypeSelect.value) params.set("subtype", subtypeSelect.value);
        location.hash = queryHref("#/records", params);
      });
      const keep = ["text", "kind", "subtype"];
      clear(page).append(
        pageHead(h("h1", {}, "Records"), h("span", { class: "meta" },
                                           fmt(data.total) + " matching of " + fmt(counts.kinds.reduce((sum, item) => sum + item.count, 0)))),
        filter,
        pager(query, data, "#/records", keep),
        table(["record", "kind", "subtype", "tick", "created", "content"], data.items.map((item) => linkRow(
          href("record", item.record_id),
          cell(aRecord(item.record_id, item.record_id), "num"),
          cell(item.kind), cell(item.subtype || "-", "dim"), num(fmt(item.tick_id)), cell(time(item.timestamp), "dim"),
          cell(short(item.preview, 130))))),
        pager(query, data, "#/records", keep));
    }));
}

function selectOptions(names, selected, placeholder) {
  const options = [h("option", { value: "" }, placeholder)];
  for (const name of names) {
    const option = h("option", { value: name }, name);
    if (name === selected) option.setAttribute("selected", "selected");
    options.push(option);
  }
  return h("select", {}, options);
}

function subtypeNames(counts, kind) {
  const found = counts.kinds.find((item) => item.kind === kind);
  if (found) return found.subtypes.map((item) => item.subtype);
  return counts.subtypes.slice(0, 60).map((item) => item.name);
}

function pageRecord(id) {
  crumbs([["Records", "#/records"], ["record " + id]]);
  loading("loading record " + id + "\u2026");
  show(Promise.all([api(`/record/${id}`), api(`/record/${id}/render`)]).then(([detail, render]) => {
    const label = detail.label;
    clear(page).append(
      pageHead(h("h1", {}, "record " + id),
               h("span", { class: "meta" }, (label.kind || "?") + (label.subtype ? " \u00b7 " + label.subtype : "")),
               badge(label.in_checkpoint ? "in checkpoint" : "absent from checkpoint",
                     label.in_checkpoint ? "on" : "off"),
               label.tick_id ? h("span", { class: "meta" }, "born in ",
                                        h("a", { href: href("cycle", label.tick_id) }, "cycle " + label.tick_id)) : null),
      h("div", { class: "grid" },
        stat("story line tk", render.line_tokens),
        stat("content tk", render.content_tokens),
        stat("framing tk", render.framing_tokens),
        stat("context tk", render.context_tokens),
        stat("shards held it", render.in_shards),
        stat("content chars", label.content_chars)),
      h("h2", {}, "Story line (the words the model reads)"),
      h("pre", {}, render.story_line),
      h("h3", {}, "with its id"),
      h("pre", {}, render.story_line_with_id),
      h("h2", {}, "Typed context (the other rendering)"),
      render.context_text ? h("pre", { class: "context" }, render.context_text)
                          : h("p", { class: "meta" }, "nothing to render"),
      h("h2", {}, "Shards that held this record"),
      detail.shard_hits.length
      ? table(["cycle", "shard", "where"], detail.shard_hits.map((hit) => row(
          cell(aCycle(hit.tick_id)), cell(aShard(hit.tick_id, hit.shard)), cell(hit.where))))
      : h("p", { class: "meta" }, "no shard in this checkpoint held it"),
      h("h2", {}, "This record points at"),
      relationTable(detail.outgoing),
      h("h2", {}, "Records that point at this one"),
      relationTable(detail.incoming),
      h("h2", {}, "Stored fields"),
      fieldsBlock(detail.fields));
  }));
}

function pageTypes() {
  crumbs([["Records", "#/records"], ["types"]]);
  loading();
  show(api("/tree").then((data) => {
    clear(page).append(
      pageHead(h("h1", {}, "Types"), h("span", { class: "meta" },
                                        "every kind and subtype; each one opens a filtered record list")),
      table(["kind", "records", "subtypes"], data.tree.map((kind) => row(
        cell(aRecords(new URLSearchParams({ kind: kind.kind }), kind.kind)),
        num(fmt(kind.count)),
        h("td", {}, kind.subtypes.map((sub) => {
          const params = new URLSearchParams({ kind: kind.kind });
          if (sub.subtype !== "-") params.set("subtype", sub.subtype);
          return h("div", {}, aRecords(params, sub.subtype + " (" + fmt(sub.count) + ")"));
        }))))));
  }));
}

function pageClusters(query) {
  crumbs([["Clusters"]]);
  const offset = Number(query.get("offset") || 0);
  const size = Number(query.get("size") || 100);
  loading("loading clusters\u2026");
  show(api(`/clusters?offset=${offset}&limit=${size}`).then((data) => {
    clear(page).append(
      pageHead(h("h1", {}, "Clusters"), h("span", { class: "meta" },
                                          fmt(data.total) + " episodic clusters, highest level first")),
      pager(query, data, "#/clusters", []),
      table(["cluster", "type", "level", "temporal key", "events", "children", "summary"],
            data.items.map((item) => linkRow(href("cluster", item.record_id),
                                             cell(aCluster(item.record_id, item.record_id), "num"),
                                             cell(item.cluster_type), num(fmt(item.level)),
                                             cell(item.temporal_key || "-", "dim"),
                                             num(fmt(item.events.length)), num(fmt(item.clusters.length)),
                                             cell(short(item.preview, 120))))),
      pager(query, data, "#/clusters", []));
  }));
}

function pageCluster(id) {
  crumbs([["Clusters", "#/clusters"], ["cluster " + id]]);
  loading("loading cluster " + id + "\u2026");
  show(Promise.all([api("/clusters?limit=1000"), api(`/record/${id}`)]).then(([all, detail]) => {
    const found = all.items.find((item) => item.record_id === id);
    if (!found) {
      failed(new Error("record " + id + " is not an episodic cluster in this checkpoint"));
      return;
    }
    const parents = all.items.filter((item) => item.clusters.includes(id));
    clear(page).append(
      pageHead(h("h1", {}, "cluster " + id),
               h("span", { class: "meta" }, found.cluster_type + " \u00b7 level " + found.level
                                       + (found.temporal_key ? " \u00b7 " + found.temporal_key : ""))),
      h("div", { class: "grid" },
        stat("member events", found.events.length),
        stat("child clusters", found.clusters.length),
        stat("part of", parents.length),
        stat("summary chars", found.content_chars),
        stat("stored token", found.token)),
      h("h2", {}, "Member events"),
      h("p", {}, text(found.events.map((member) => aRecord(member))) || h("span", { class: "dim" }, "none")),
      h("h2", {}, "Child clusters"),
      h("p", {}, text(found.clusters.map((member) => aCluster(member))) || h("span", { class: "dim" }, "none")),
      h("h2", {}, "Part of"),
      h("p", {}, text(parents.map((item) => aCluster(item.record_id))) || h("span", { class: "dim" }, "top of the hierarchy")),
      h("h2", {}, "Summary"),
      h("pre", {}, detail.content || "(empty)"),
      h("h2", {}, "Stored fields"),
      fieldsBlock(detail.fields));
  }));
}

function pageRelations() {
  crumbs([["Relations"]]);
  loading();
  show(api("/census").then((data) => {
    clear(page).append(
      pageHead(h("h1", {}, "Relations"), h("span", { class: "meta" },
                                           "how many stored references each field carries")),
      h("p", { class: "meta" }, "a record page lists the edges of one record in both directions"),
      table(["field", "edges"], data.edges.map((edge) => row(cell(edge.name, "mono"), num(fmt(edge.count))))));
  }));
}

function pagePrompts() {
  crumbs([["Prompts"]]);
  loading();
  show(api("/prompts").then((prompts) => {
    clear(page).append(
      pageHead(h("h1", {}, "Prompts"), h("span", { class: "meta" }, "the words these pages render with")),
      h("p", { class: "meta" }, "read as data from ", prompts.story_prompts, " and ", prompts.attention_prompts),
      h("h2", {}, "Story template"),
      h("pre", { class: "story" }, prompts.story_template),
      h("h2", {}, "Shard template"),
      h("pre", { class: "shard" }, prompts.shard_template),
      h("h2", {}, "Block labels"),
      fieldsBlock(prompts.labels),
      h("h2", {}, "Focus labels"),
      fieldsBlock(prompts.focus_labels),
      h("h2", {}, "System prompt"),
      h("pre", {}, prompts.system_prompt),
      h("h2", {}, "Story acknowledgement"),
      h("pre", {}, prompts.story_ack),
      prompts.missing_labels.length ? h("p", { class: "waste" },
                                        "labels with no text: " + prompts.missing_labels.join(", ")) : null);
  }));
}

function pageErrors() {
  crumbs([["Errors"]]);
  loading();
  show(api("/errors").then((data) => {
    clear(page).append(pageHead(h("h1", {}, "Errors"), h("span", { class: "meta" },
                                                         fmt(data.errors.length) + " failed views")),
                       data.errors.length ? null : h("p", { class: "good" }, "no view has failed since the start"),
                       data.errors.map((error) => h("div", { class: "card" },
                                                    h("h3", {}, error.path),
                                                    h("p", { class: "waste" }, error.error),
                                                    h("pre", {}, error.trace))));
  }));
}

function pageMissing(what) {
  crumbs([["No such page"]]);
  clear(page).append(h("h1", {}, "No such page"), h("p", { class: "waste" }, what),
                     h("ul", { class: "tight" },
                       h("li", {}, h("a", { href: "#/" }, "overview")),
                       h("li", {}, h("a", { href: "#/cycles" }, "cycles")),
                       h("li", {}, h("a", { href: "#/records" }, "records")),
                       h("li", {}, h("a", { href: "#/types" }, "types")),
                       h("li", {}, h("a", { href: "#/clusters" }, "clusters")),
                       h("li", {}, h("a", { href: "#/relations" }, "relations")),
                       h("li", {}, h("a", { href: "#/prompts" }, "prompts"))));
}

/* ----------------------------------------------------------------- router */

function integer(text2) {
  const value = Number(text2);
  if (!Number.isInteger(value) || value < 0) throw new Error("not a number: " + text2);
  return value;
}

function shardKey(parts) {
  return { kind: parts[0], ref: parts[1] === "-" || !parts[1] ? "" : parts[1] };
}

function render() {
  const state = current();
  const parts = state.parts;
  const query = state.query;
  markTab(parts[0] || "");
  try {
    if (!parts.length) return pageOverview();
    if (parts[0] === "cycles") return pageCycles(query);
    if (parts[0] === "records") return pageRecords(query);
    if (parts[0] === "clusters") return pageClusters(query);
    if (parts[0] === "types") return pageTypes();
    if (parts[0] === "relations") return pageRelations();
    if (parts[0] === "prompts") return pagePrompts();
    if (parts[0] === "errors") return pageErrors();
    if (parts[0] === "record") return pageRecord(integer(parts[1]));
    if (parts[0] === "cluster") return pageCluster(integer(parts[1]));
    if (parts[0] === "cycle") {
      const tick = integer(parts[1]);
      if (parts.length === 2) return pageCycle(tick);
      if (parts[2] === "story") return pageStory(tick);
      if (parts[2] === "turn" || parts[2] === "trace") return pageCycleJson(tick, parts[2]);
      if (parts[2] === "shard" && parts.length >= 4) return pageShard(tick, shardKey(parts.slice(3)));
      return pageMissing("no cycle view: " + parts.join("/"));
    }
    return pageMissing("no page at /" + parts.join("/"));
  } catch (error) {
    failed(error);
  }
}

function markTab(name) {
  const top = { cycle: "cycles", record: "records", cluster: "clusters", types: "records" }[name] || name;
  for (const link of document.querySelectorAll("#tabs a")) {
    const target = link.getAttribute("href").replace(/^#\/?/, "").split("/")[0];
    link.className = target === top ? "active" : "";
  }
}

/* ------------------------------------------------------------------ start */

function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
  } catch (error) {
    return "light";
  }
}

function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  $("#theme").textContent = name === "dark" ? "light" : "dark";
  try {
    localStorage.setItem(THEME_KEY, name);
  } catch (error) {
    void error;                     // storage can be closed; this page still gets a theme
  }
}

$("#theme").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
applyTheme(storedTheme());
window.addEventListener("hashchange", render);
if (!location.hash) location.hash = "#/";
render();

api("/summary").then((summary) => {
  $("#meta").textContent = summary.companion_name + " \u00b7 " + fmt(summary.records) + " records \u00b7 "
    + fmt(summary.cycles) + " cycles \u00b7 read-only";
}).catch((error) => {
  $("#meta").textContent = String(error.message || error);
});
