// Web chat page. It polls the loopback API and renders message text only
// through textContent, so a message can never become markup or script.
"use strict";

const NAME_KEY = "pm.web_chat.name";
// gen changes at each login, so a poll that started before it is discarded.
const state = { session: null, name: "", lastId: 0, lastDate: "", typing: null, timer: null,
                polling: false, again: false, gen: 0, images: [] };
// Matches WEB_CHAT_MAX_IMAGES and WEB_CHAT_MAX_IMAGE_BYTES in pm/constants.py.
// The server checks the same limits; these give an early, clear message.
const MAX_IMAGES = 6;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function dateLabel(date) {
  return date.toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

function nearBottom() {
  const box = $("messages");
  return box.scrollHeight - box.scrollTop - box.clientHeight < 80;
}

function scrollDown() {
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

function addMessage(message) {
  const box = $("messages");
  const when = new Date(message.timestamp);
  const day = when.toDateString();
  if (day !== state.lastDate) {
    box.insertBefore(el("div", "date", dateLabel(when)), state.typing);
    state.lastDate = day;
  }
  const mine = !message.from_companion && message.speaker === state.name;
  const row = el("div", "row " + (mine ? "mine" : "other") + (message.from_companion ? " companion" : ""));
  const bubble = el("div", "bubble");
  if (!mine) bubble.appendChild(el("div", "speaker", message.speaker));
  if (message.images.length) {
    const strip = el("div", "images");
    message.images.forEach((imageId) => {
      const img = el("img");
      img.src = "/api/media/" + message.id + "/" + encodeURIComponent(imageId);
      img.alt = "image";
      img.addEventListener("click", () => openViewer(img.src));
      strip.appendChild(img);
    });
    bubble.appendChild(strip);
  }
  if (message.text) bubble.appendChild(el("div", "text", message.text));
  bubble.appendChild(el("div", "time", when.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })));
  bubble.title = when.toLocaleString();
  row.appendChild(bubble);
  box.insertBefore(row, state.typing);
}

function openViewer(src) {
  const viewer = el("div", "viewer");
  const img = el("img");
  img.src = src;
  viewer.appendChild(img);
  viewer.addEventListener("click", () => viewer.remove());
  document.body.appendChild(viewer);
}

function renderAttachments() {
  const box = $("attachments");
  box.replaceChildren();
  state.images.forEach((image, index) => {
    const chip = el("div", "chip");
    const img = el("img");
    img.src = image.url;
    img.alt = image.name;
    img.title = image.name;
    const remove = el("button", "", "×");
    remove.type = "button";
    remove.title = "Remove";
    remove.addEventListener("click", () => { state.images.splice(index, 1); renderAttachments(); });
    chip.append(img, remove);
    box.appendChild(chip);
  });
  box.hidden = state.images.length === 0;
}

function addFiles(files) {
  for (const file of files) {
    if (!IMAGE_TYPES.includes(file.type)) {
      $("status").textContent = file.name + ": only PNG, JPEG, WebP, or GIF images.";
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      $("status").textContent = file.name + ": larger than " + (MAX_IMAGE_BYTES >> 20) + " MB.";
      continue;
    }
    if (state.images.length >= MAX_IMAGES) {
      $("status").textContent = "At most " + MAX_IMAGES + " images per message.";
      break;
    }
    const reader = new FileReader();
    reader.onload = () => {
      // A data URL is "data:<type>;base64,<data>". The server gets the data part.
      const url = String(reader.result);
      state.images.push({ name: file.name || "pasted image", media_type: file.type, url: url,
                          data_b64: url.slice(url.indexOf(",") + 1) });
      renderAttachments();
    };
    reader.readAsDataURL(file);
  }
}

function setThinking(thinking) {
  const companion = state.session.companion_name;
  state.typing.hidden = !thinking;
  $("status").textContent = thinking ? companion + " is thinking…" : "logged in as " + state.name;
  $("status").classList.toggle("thinking", thinking);
}

async function poll() {
  clearTimeout(state.timer);
  if (state.polling) { state.again = true; return; }
  state.polling = true;
  const gen = state.gen;
  try {
    const response = await fetch("/api/messages?after=" + state.lastId, { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const page = await response.json();
    if (gen !== state.gen) return;
    const stick = state.lastId === 0 || nearBottom();
    page.messages.forEach(addMessage);
    state.lastId = page.last_id;
    setThinking(page.thinking);
    if (stick && (page.messages.length || page.thinking)) scrollDown();
  } catch (error) {
    $("status").textContent = "Connection lost. Retrying…";
  } finally {
    state.polling = false;
    if (!$("chat").hidden) {
      if (state.again) { state.again = false; poll(); }
      else state.timer = setTimeout(poll, state.session.poll_ms);
    }
  }
}

async function send(event) {
  event.preventDefault();
  const box = $("text");
  const text = box.value.trim();
  if (!text && !state.images.length) return;
  $("send").disabled = true;
  try {
    const images = state.images.map((image) => ({ media_type: image.media_type, data_b64: image.data_b64, name: image.name }));
    const response = await fetch("/api/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: state.name, text: text, images: images }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    box.value = "";
    state.images = [];
    renderAttachments();
    resize();
    scrollDown();
    poll();
  } catch (error) {
    $("status").textContent = "Not sent: " + error.message;
  } finally {
    $("send").disabled = false;
    box.focus();
  }
}

function resize() {
  const box = $("text");
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 160) + "px";
}

function showChat() {
  $("login").hidden = true;
  $("chat").hidden = false;
  const companion = state.session.companion_name;
  $("companion-name").textContent = companion;
  $("avatar").textContent = companion.slice(0, 1).toUpperCase();
  document.title = companion;
  $("text").focus();
  poll();
}

function showLogin() {
  $("chat").hidden = true;
  $("login").hidden = false;
  $("login-hint").textContent = "Chat with " + state.session.companion_name + ".";
  $("login-name").value = state.name || state.session.user_name;
  $("login-name").focus();
}

async function start() {
  state.session = await (await fetch("/api/session", { cache: "no-store" })).json();
  state.typing = el("div", "typing", state.session.companion_name + " is thinking…");
  state.typing.hidden = true;
  $("messages").appendChild(state.typing);

  $("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const name = $("login-name").value.trim();
    if (name === state.session.companion_name) {
      $("login-error").textContent = "This name belongs to the companion.";
      return;
    }
    if (!name) return;
    localStorage.setItem(NAME_KEY, name);
    state.name = name;
    // Rebuild the list, so the left and right sides follow the new name.
    $("messages").replaceChildren(state.typing);
    state.lastId = 0;
    state.lastDate = "";
    state.gen += 1;
    showChat();
  });
  $("logout").addEventListener("click", () => {
    clearTimeout(state.timer);
    showLogin();
  });
  $("composer").addEventListener("submit", send);
  $("attach").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", () => { addFiles($("file").files); $("file").value = ""; });
  $("text").addEventListener("paste", (event) => {
    const files = [...event.clipboardData.items].filter((item) => item.kind === "file").map((item) => item.getAsFile());
    if (files.length) { event.preventDefault(); addFiles(files); }
  });
  const chat = $("chat");
  chat.addEventListener("dragover", (event) => { event.preventDefault(); chat.classList.add("dropping"); });
  chat.addEventListener("dragleave", (event) => { if (event.target === chat || !chat.contains(event.relatedTarget)) chat.classList.remove("dropping"); });
  chat.addEventListener("drop", (event) => {
    event.preventDefault();
    chat.classList.remove("dropping");
    addFiles(event.dataTransfer.files);
  });
  $("text").addEventListener("input", resize);
  $("text").addEventListener("keydown", (event) => {
    // Enter sends. Shift+Enter adds a line. An IME composition keeps Enter.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) send(event);
  });

  state.name = localStorage.getItem(NAME_KEY) || "";
  if (state.name) showChat(); else showLogin();
}

start();
