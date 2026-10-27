"use strict";

const $ = (selector) => document.querySelector(selector);
let snapshot;
let editingId = null;
let busy = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function report(error) {
  $("#notice").textContent = error.message;
  $("#notice").className = "notice error";
}

async function request(options) {
  const response = await fetch("/api/board", options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}

async function reload() {
  if (busy) return;
  try {
    snapshot = await request();
    render();
    $("#add").disabled = false;
    $("#notice").textContent = "";
  } catch (error) {
    report(error);
  }
}

async function save(board) {
  busy = true;
  $("#save").disabled = true;
  $("#delete").disabled = true;
  $("#cancel").disabled = true;
  $("#close").disabled = true;
  try {
    // Keep the loaded board until disk confirms the complete replacement.
    snapshot = await request({
      method: "PUT",
      headers: { "Content-Type": "application/json", "If-Match": snapshot.revision },
      body: JSON.stringify(board),
    });
    render();
    $("#notice").className = "notice";
    $("#notice").textContent = "Saved to the repo.";
  } finally {
    busy = false;
    for (const id of ["save", "delete", "cancel", "close"]) $("#" + id).disabled = false;
  }
}

async function move(id, status) {
  if (busy) return;
  const board = structuredClone(snapshot.board);
  board.cards.find((card) => card.id === id).status = status;
  try {
    await save(board);
  } catch (error) {
    report(error);
    render();
  }
}

function render() {
  $("#board").replaceChildren();
  const count = snapshot.board.cards.length;
  $("#total").textContent = `${count} ${count === 1 ? "card" : "cards"}`;
  for (const [status, label] of Object.entries(snapshot.columns)) {
    const cards = snapshot.board.cards.filter((card) => card.status === status);
    const column = element("section", `column ${status}`);
    column.dataset.status = status;
    column.setAttribute("aria-label", label);
    const heading = element("div", "column-heading");
    heading.append(element("span", "column-dot"), element("h2", "", label), element("span", "count", cards.length));
    const add = element("button", "column-add", "+");
    add.type = "button";
    add.setAttribute("aria-label", `Add card to ${label}`);
    add.addEventListener("click", () => openEditor(null, status));
    heading.append(add);
    column.append(heading);
    for (const card of cards) {
      const article = element("article", "card");
      article.draggable = true;
      article.dataset.id = card.id;
      article.addEventListener("dragstart", (event) => {
        event.dataTransfer.setData("text/plain", String(card.id));
        event.dataTransfer.effectAllowed = "move";
        article.classList.add("dragging");
      });
      article.addEventListener("dragend", () => {
        article.classList.remove("dragging");
        document.querySelectorAll(".drop-target").forEach((node) => node.classList.remove("drop-target"));
      });
      const edit = element("button", "card-edit");
      edit.type = "button";
      edit.append(element("span", "card-id", `K${String(card.id).padStart(3, "0")}`), element("span", "card-title", card.title));
      if (card.description) edit.append(element("span", "card-notes", card.description));
      edit.addEventListener("click", () => openEditor(card, card.status));
      article.append(edit);
      column.append(article);
    }
    if (!cards.length) column.append(element("p", "empty", status === "done" ? "Good things take a little work." : "A little space for what comes next."));
    column.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      column.classList.add("drop-target");
    });
    column.addEventListener("dragleave", (event) => {
      if (!column.contains(event.relatedTarget)) column.classList.remove("drop-target");
    });
    column.addEventListener("drop", (event) => {
      event.preventDefault();
      column.classList.remove("drop-target");
      const id = Number(event.dataTransfer.getData("text/plain"));
      const card = snapshot.board.cards.find((item) => item.id === id);
      if (card && card.status !== status) move(id, status);
    });
    $("#board").append(column);
  }
}

function openEditor(card, status) {
  if (busy) return;
  editingId = card ? card.id : null;
  $("#editor-title").textContent = card ? `Edit K${String(card.id).padStart(3, "0")}` : "New card";
  $("#title").value = card ? card.title : "";
  $("#description").value = card ? card.description : "";
  $("#status").replaceChildren();
  for (const [value, label] of Object.entries(snapshot.columns)) {
    const option = element("option", "", label);
    option.value = value;
    $("#status").append(option);
  }
  $("#status").value = status;
  $("#delete").hidden = card === null;
  $("#editor-error").textContent = "";
  $("#editor").showModal();
  $("#title").focus();
}

$("#add").addEventListener("click", () => openEditor(null, "backlog"));
$("#reload").addEventListener("click", reload);
for (const id of ["cancel", "close"]) $("#" + id).addEventListener("click", () => $("#editor").close());
$("#editor").addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
$("#card-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const title = $("#title").value.trim();
  if (!title) {
    $("#editor-error").textContent = "Enter a title for this card.";
    $("#title").focus();
    return;
  }
  const board = structuredClone(snapshot.board);
  const card = { id: editingId ?? board.next_id++, title, description: $("#description").value, status: $("#status").value };
  if (editingId === null) board.cards.push(card);
  else board.cards[board.cards.findIndex((item) => item.id === editingId)] = card;
  try {
    await save(board);
    $("#editor").close();
  } catch (error) {
    $("#editor-error").textContent = error.message;
  }
});
$("#delete").addEventListener("click", async () => {
  if (busy || !confirm("Delete this card?")) return;
  const board = structuredClone(snapshot.board);
  board.cards = board.cards.filter((card) => card.id !== editingId);
  try {
    await save(board);
    $("#editor").close();
  } catch (error) {
    $("#editor-error").textContent = error.message;
  }
});

reload();
