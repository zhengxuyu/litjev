"use strict";
const el = (id) => document.getElementById(id);
const TIMELINE_LENGTH = 48;
const LATENCY_LENGTH = 90;
const RING_CIRCUMFERENCE = 238.76;

const style = getComputedStyle(document.documentElement);
const PALETTE = Array.from({ length: 10 }, (_, index) =>
  style.getPropertyValue(`--a${index}`).trim()
);
const ACCENT = style.getPropertyValue("--accent").trim();
const MUTED = style.getPropertyValue("--muted").trim();
const LINE = style.getPropertyValue("--line").trim();

const rows = new Map();
const colours = new Map();
let rowsKey = "";
let stateLines = [];
let lastAction = null;
const latency = [];

function showError(message) {
  el("error").textContent = message;
  el("error").hidden = false;
}

// Rows are built once per menu so the bars can animate instead of being replaced each frame.
function ensureRows(menu) {
  const key = Object.keys(menu).join("");
  if (key === rowsKey) return;
  rowsKey = key;
  rows.clear();
  colours.clear();
  const container = el("actions");
  container.replaceChildren();
  Object.entries(menu).forEach(([label, move], index) => {
    const colour = PALETTE[index % PALETTE.length];
    colours.set(label, colour);
    const row = document.createElement("div");
    row.className = "row";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = move;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.background = colour;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = "0.00%";
    const letter = document.createElement("span");
    letter.className = "key";
    letter.textContent = label;
    row.append(letter, name, track, value);
    container.append(row);
    rows.set(label, { row, fill, value });
  });
}

function updateBars(payload) {
  for (const [label, entry] of rows) {
    const probability = payload.probabilities[label] ?? 0;
    entry.fill.style.width = `${(probability * 100).toFixed(2)}%`;
    entry.value.textContent = `${(probability * 100).toFixed(2)}%`;
    entry.row.classList.toggle("picked", label === payload.action);
  }
}

function pushTimeline(payload) {
  const timeline = el("timeline");
  const block = document.createElement("i");
  block.style.background = colours.get(payload.action) ?? ACCENT;
  block.title = `第 ${payload.step} 步 · ${payload.move}`;
  timeline.append(block);
  while (timeline.childElementCount > TIMELINE_LENGTH) timeline.firstElementChild.remove();
}

function updateState(text) {
  const lines = text.split("\n");
  const container = el("state-text");
  if (stateLines.length !== lines.length) {
    container.replaceChildren();
    stateLines = lines.map((line) => {
      const row = document.createElement("div");
      row.textContent = line;
      container.append(row);
      return row;
    });
    return;
  }
  lines.forEach((line, index) => {
    const row = stateLines[index];
    if (row.textContent === line) return;
    row.textContent = line;
    row.classList.remove("changed");
    void row.offsetWidth; // restart the highlight animation
    row.classList.add("changed");
  });
}

function drawLatency() {
  const canvas = el("latency");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (latency.length < 2) return;

  const ceiling = Math.max(0.3, ...latency) * 1.25;
  // The line always spans the full width and compresses as the window fills.
  const step = width / (latency.length - 1);
  const offset = 0;
  const y = (value) => height - 14 - (value / ceiling) * (height - 24);

  context.strokeStyle = LINE;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, y(0));
  context.lineTo(width, y(0));
  context.stroke();

  const path = new Path2D();
  latency.forEach((value, index) => {
    const x = offset + index * step;
    if (index === 0) path.moveTo(x, y(value));
    else path.lineTo(x, y(value));
  });
  const area = new Path2D(path);
  area.lineTo(width, y(0));
  area.lineTo(offset, y(0));
  area.closePath();
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(255, 106, 61, .28)");
  gradient.addColorStop(1, "rgba(255, 106, 61, 0)");
  context.fillStyle = gradient;
  context.fill(area);
  context.strokeStyle = ACCENT;
  context.lineWidth = 1.6;
  context.lineJoin = "round";
  context.stroke(path);

  const mean = latency.reduce((total, value) => total + value, 0) / latency.length;
  context.setLineDash([3, 4]);
  context.strokeStyle = MUTED;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, y(mean));
  context.lineTo(width, y(mean));
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = MUTED;
  context.font = "11px ui-monospace, SFMono-Regular, monospace";
  context.fillText(`均值 ${mean.toFixed(3)} s`, 4, y(mean) - 5);
}

function renderStep(payload) {
  el("error").hidden = true;
  ensureRows(payload.menu);
  const frame = el("frame");
  frame.src = `data:image/jpeg;base64,${payload.frame}`;
  el("placeholder").hidden = true;

  el("episode-count").textContent = payload.episode + 1;
  el("step-count").textContent = payload.step;
  el("input-tokens").textContent = payload.input_tokens;
  el("forward-calls").textContent = payload.forward_calls;
  el("output-tokens").textContent = payload.output_tokens;
  el("decision-time").textContent = `${payload.decision_seconds.toFixed(3)} s`;
  el("run-status").textContent = `运行中 · 生命 ${payload.status.health.toFixed(0)} · 弹药 ${payload.status.ammo.toFixed(0)} · 击杀 ${payload.status.kills.toFixed(0)}`;

  const chosen = el("chosen");
  chosen.textContent = `${payload.action} · ${payload.menu[payload.action] ?? ""}`;
  if (payload.action !== lastAction) {
    lastAction = payload.action;
    chosen.classList.remove("pulse");
    void chosen.offsetWidth;
    chosen.classList.add("pulse");
  }
  el("gamma").textContent = `${(payload.gamma * 100).toFixed(0)}%`;
  el("ring").style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - payload.gamma);

  updateBars(payload);
  pushTimeline(payload);
  updateState(payload.state_text);
  latency.push(payload.decision_seconds);
  while (latency.length > LATENCY_LENGTH) latency.shift();
  el("latency-note").textContent = `最近 ${latency.length} 步 · 当前 ${payload.decision_seconds.toFixed(3)} s`;
  drawLatency();
}

function listen() {
  const source = new EventSource("/doom/stream");
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.event === "step") {
      renderStep(payload);
      return;
    }
    if (payload.event === "stopped") {
      el("run-status").textContent = payload.error ? "已停止 · 出错" : "已停止";
      if (payload.error) showError(payload.error);
      el("start").disabled = false;
      source.close();
      setTimeout(listen, 1500);
    }
  };
  source.onerror = () => {
    source.close();
    setTimeout(listen, 3000);
  };
}

async function control(path, pending, settled) {
  el("start").disabled = true;
  el("stop").disabled = true;
  el("run-status").textContent = pending;
  el("error").hidden = true;
  try {
    const response = await fetch(path, { method: "POST" });
    if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}`);
    const status = await response.json();
    el("run-status").textContent = status.running ? settled : "已停止";
    el("start").disabled = status.running;
  } catch (error) {
    showError(error.message);
    el("run-status").textContent = "请求失败";
    el("start").disabled = false;
  } finally {
    el("stop").disabled = false;
  }
}

async function health() {
  try {
    const response = await fetch("/doom/status");
    if (!response.ok) throw new Error("服务不可用");
    const status = await response.json();
    el("scenario").textContent = status.scenario;
    el("health").textContent = "● 已连接";
    el("health").classList.add("live");
    el("start").disabled = status.running;
    if (status.running) el("run-status").textContent = "运行中";
  } catch {
    el("health").textContent = "○ 服务未连接";
    el("health").classList.remove("live");
  }
}

el("start").addEventListener("click", () => control("/doom/start", "启动中 · 首次需要加载模型…", "运行中"));
el("stop").addEventListener("click", () => control("/doom/stop", "停止中…", "运行中"));
window.addEventListener("resize", drawLatency);
health();
listen();
