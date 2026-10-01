"use strict";

const DATA_URL = "data/adaptive-failover.json";
const SUPPORTED_SCHEMA = 1;
const $ = (selector) => document.querySelector(selector);

const state = {
  replay: null,
  frames: [],
  transitions: [],
  index: 0,
  playing: false,
  speed: 1,
  timer: null,
};

document.addEventListener("DOMContentLoaded", loadReplay);

async function loadReplay() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`Replay request returned HTTP ${response.status}.`);
    const replay = await response.json();
    validateReplay(replay);
    state.replay = replay;
    state.frames = replay.events.filter((event) => event.event_type === "inference");
    state.transitions = replay.events.filter((event) => event.event_type === "profile_change");
    initializeViewer();
  } catch (error) {
    showError(error instanceof Error ? error.message : "The experiment artifact could not be loaded.");
  }
}

function validateReplay(replay) {
  if (!replay || typeof replay !== "object") throw new Error("Replay artifact is not a JSON object.");
  if (replay.schema_version !== SUPPORTED_SCHEMA) {
    throw new Error(`Unsupported replay schema v${String(replay.schema_version)}. This viewer supports v${SUPPORTED_SCHEMA}.`);
  }
  if (!replay.run || replay.run.event_type !== "run_start" || !Array.isArray(replay.events)) {
    throw new Error("Replay artifact is missing run metadata or events.");
  }
  const frames = replay.events.filter((event) => event.event_type === "inference");
  if (!frames.length) throw new Error("Replay artifact contains no inference events.");
  for (const event of frames) {
    if (!event.decision || !event.timing || !event.workers || !Number.isFinite(event.frame_index)) {
      throw new Error(`Inference event ${String(event.sequence)} is malformed.`);
    }
  }
}

function initializeViewer() {
  $("#loading-state").hidden = true;
  $("#viewer").hidden = false;
  const run = state.replay.run;
  const summary = state.replay.events.find((event) => event.event_type === "run_end")?.summary || {};
  const synthetic = run.notes?.synthetic_fixture === true;
  $("#fixture-notice").hidden = !synthetic;
  $("#capture-notice").hidden = synthetic;
  $("#evidence-kind").textContent = synthetic ? "sample fixture" : "captured run";
  $("#top-scenario").textContent = run.scenario || run.run_id;
  $("#top-scheduler").textContent = run.scheduler;
  $("#schema-label").textContent = `schema v${run.schema_version}`;
  $("#run-properties").innerHTML = [
    ["Scenario", run.scenario || "none"],
    ["Scheduler", run.scheduler],
    ["EWMA α", formatNumber(run.latency_alpha, 2)],
    ["Model", displayModel(run.model)],
    ["Frames", summary.total_frames ?? state.frames.length],
    ["Schema", `v${run.schema_version}`],
    ["Run ID", run.run_id],
  ].map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd title="${escapeHtml(value)}">${escapeHtml(value)}</dd></div>`).join("");
  $("#mini-summary").innerHTML = [
    ["Fallbacks", summary.fallback_count ?? countFallbacks()],
    ["Route switches", summary.worker_switches ?? countSwitches()],
    ["Mean observed", formatMs(summary.mean_observed_latency_ms)],
    ["Duration", formatMs(summary.scenario_duration_ms)],
  ].map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");

  $("#scrubber").max = String(state.frames.length - 1);
  $("#scrubber").addEventListener("input", (event) => selectFrame(Number(event.target.value)));
  $("#first-button").addEventListener("click", () => selectFrame(0));
  $("#next-button").addEventListener("click", () => selectFrame(state.index + 1));
  $("#play-button").addEventListener("click", togglePlay);
  document.querySelectorAll("[data-speed]").forEach((button) => {
    button.addEventListener("click", () => setSpeed(Number(button.dataset.speed)));
  });
  document.addEventListener("keydown", handleKeyboard);
  $("#latency-chart").addEventListener("click", selectChartPosition);
  renderTimeline();
  renderEventList();
  renderChart();
  selectFrame(0);
}

function showError(message) {
  $("#loading-state").hidden = true;
  $("#error-state").hidden = false;
  $("#error-message").textContent = message;
}

function selectFrame(index) {
  state.index = Math.max(0, Math.min(state.frames.length - 1, index));
  $("#scrubber").value = String(state.index);
  renderCurrentFrame();
  updateSelection();
  if (state.playing && state.index === state.frames.length - 1) stopPlayback();
}

function renderCurrentFrame() {
  const event = state.frames[state.index];
  const maxFrame = state.frames[state.frames.length - 1].frame_index;
  $("#frame-label").textContent = `frame ${event.frame_index} / ${maxFrame}`;
  $("#sequence-value").textContent = `#${String(event.sequence).padStart(3, "0")}`;
  $("#current-profile").textContent = event.profile.replaceAll("_", " ");
  $("#selected-worker").textContent = event.decision.selected_worker || "—";
  $("#executed-worker").textContent = event.decision.executed_worker || "—";
  $("#fallback-used").textContent = event.fallback.used ? "yes" : "no";
  $("#decision-reason").textContent = event.decision.reason || "unavailable";
  $("#fallback-reason").textContent = event.fallback.reason || "";
  $(".decision-panel").classList.toggle("has-fallback", event.fallback.used);
  renderDecisionBars(event);
  renderWorker("edge", event);
  renderWorker("remote", event);
  updateChartPlayhead(event.frame_index);
  updateActivePhase(event.profile);
  $("#chart-description").textContent = `Frame ${event.frame_index}: edge estimate ${formatMs(event.decision.edge_estimate_ms)}, remote estimate ${formatMs(event.decision.remote_estimate_ms)}, observed request ${formatMs(event.timing.observed_request_ms)}. Selected ${event.decision.selected_worker}; executed ${event.decision.executed_worker}.`;
}

function renderDecisionBars(event) {
  const values = [event.decision.edge_estimate_ms, event.decision.remote_estimate_ms].filter(Number.isFinite);
  const max = Math.max(...values, 1);
  $("#decision-bars").innerHTML = ["edge", "remote"].map((worker) => {
    const estimate = event.decision[`${worker}_estimate_ms`];
    const width = Number.isFinite(estimate) ? Math.max(1, estimate / max * 100) : 0;
    const selected = event.decision.selected_worker === worker;
    return `<div class="decision-row ${worker} ${selected ? "selected" : ""}">
      <span class="worker-name">${worker}</span><span class="selection">${selected ? "selected" : ""}</span>
      <span class="bar-track"><span class="bar-value" style="width:${width}%"></span></span>
      <span class="estimate">${formatMs(estimate)}</span></div>`;
  }).join("");
}

function renderWorker(worker, event) {
  const data = event.workers[worker];
  const executed = event.decision.executed_worker === worker;
  const estimate = event.decision[`${worker}_estimate_ms`];
  const observed = executed ? event.timing.observed_request_ms : null;
  const model = executed ? event.timing.model_inference_ms : null;
  const rtt = worker === "remote" && executed ? event.timing.round_trip_ms : null;
  const rows = [
    ["EWMA estimate", formatMs(estimate), ""],
    ["Observed request", formatMs(observed), ""],
    ["Model inference", formatMs(model), ""],
    ...(worker === "remote" ? [["HTTP round trip", formatMs(rtt), ""]] : []),
    ["Controlled delay", data.injected_delay_ms ? `+${formatMs(data.injected_delay_ms)}` : "0.0 ms", "injected"],
  ];
  $(`#${worker}-worker`).innerHTML = `<div class="worker-header"><strong>${worker}</strong><span class="worker-state ${data.available ? "" : "down"}">${data.available ? "ready" : "unavailable"}</span></div>
    <dl>${rows.map(([key, value, className]) => `<dt>${key}</dt><dd class="${className}">${value}</dd>`).join("")}</dl>`;
}

function renderTimeline() {
  const maxFrame = Math.max(...state.frames.map((event) => event.frame_index));
  const starts = [{ frame_index: 0, to_profile: state.replay.run.initial_profile }, ...state.transitions];
  $("#phase-timeline").innerHTML = starts.map((phase, index) => {
    const end = index + 1 < starts.length ? starts[index + 1].frame_index - 1 : maxFrame;
    const span = Math.max(1, end - phase.frame_index + 1);
    const condition = phaseCondition(phase.frame_index);
    return `<button class="phase" type="button" data-frame="${phase.frame_index}" data-profile="${phase.to_profile}" style="flex:${span}">
      <strong>${escapeHtml(phase.to_profile.replaceAll("_", " "))}</strong><span>frames ${phase.frame_index}–${end}</span><span>${escapeHtml(condition)}</span></button>`;
  }).join("");
  document.querySelectorAll(".phase").forEach((phase) => phase.addEventListener("click", () => selectNearestFrame(Number(phase.dataset.frame))));
}

function phaseCondition(startFrame) {
  const event = state.frames.find((frame) => frame.frame_index >= startFrame) || state.frames[state.frames.length - 1];
  const conditions = [];
  for (const worker of ["edge", "remote"]) {
    const data = event.workers[worker];
    if (!data.available) conditions.push(`${worker} unavailable`);
    if (data.injected_delay_ms > 0) conditions.push(`+${formatNumber(data.injected_delay_ms, 0)} ms ${worker} delay`);
  }
  return conditions.length ? conditions.join(" · ") : "no controls";
}

function renderEventList() {
  let priorWorker = null;
  const rows = [];
  for (const event of state.replay.events) {
    if (event.event_type === "profile_change") {
      rows.push(`<li><button type="button" data-frame="${event.frame_index}"><span class="event-row"><span>#${String(event.frame_index).padStart(3, "0")}</span><span class="profile-tag">PROFILE CHANGE · ${escapeHtml(event.from_profile)} → ${escapeHtml(event.to_profile)}</span></span></button></li>`);
    }
    if (event.event_type === "inference") {
      const worker = event.decision.executed_worker || "error";
      const routeChange = priorWorker && priorWorker !== worker;
      priorWorker = worker;
      const suffix = event.fallback.used ? " · FALLBACK" : routeChange ? " · ROUTE CHANGE" : "";
      rows.push(`<li><button type="button" data-sequence="${event.sequence}" data-index="${state.frames.indexOf(event)}"><span class="event-row">
        <span>#${String(event.frame_index).padStart(3, "0")}</span><span class="route-${worker}">${escapeHtml(worker)}</span><span>${formatMs(event.timing.observed_request_ms)}</span>
        <span class="${event.fallback.used ? "fallback-tag" : ""}">${escapeHtml(event.decision.reason || "error")}${suffix}</span></span></button></li>`);
    }
  }
  $("#event-list").innerHTML = rows.join("");
  $("#event-list").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.index !== undefined) selectFrame(Number(button.dataset.index));
    else selectNearestFrame(Number(button.dataset.frame));
  }));
}

function renderChart() {
  const svg = $("#latency-chart");
  const width = 900, height = 300, left = 52, right = 16, top = 28, bottom = 35;
  const maxFrame = Math.max(...state.frames.map((event) => event.frame_index), 1);
  const timingValues = state.frames.flatMap((event) => [event.decision.edge_estimate_ms, event.decision.remote_estimate_ms, event.timing.observed_request_ms]).filter(Number.isFinite);
  const maxValue = Math.max(...timingValues, 1) * 1.1;
  const x = (frame) => left + frame / maxFrame * (width - left - right);
  const y = (value) => top + (1 - value / maxValue) * (height - top - bottom);
  const grid = [0, .25, .5, .75, 1].map((ratio) => `<line class="grid-line" x1="${left}" y1="${y(maxValue * ratio)}" x2="${width - right}" y2="${y(maxValue * ratio)}"/><text class="axis-label" x="${left - 8}" y="${y(maxValue * ratio) + 3}" text-anchor="end">${Math.round(maxValue * ratio)}</text>`).join("");
  const series = [
    ["edge_estimate_ms", "edge-series", (event) => event.decision.edge_estimate_ms],
    ["remote_estimate_ms", "remote-series", (event) => event.decision.remote_estimate_ms],
    ["observed_request_ms", "observed-series", (event) => event.timing.observed_request_ms],
  ].map(([, className, accessor]) => `<path class="series ${className}" d="${linePath(state.frames, accessor, x, y)}"/>`).join("");
  const transitionLines = state.transitions.map((event) => `<g><line class="phase-marker" x1="${x(event.frame_index)}" y1="${top}" x2="${x(event.frame_index)}" y2="${height - bottom}"/><text class="phase-label" x="${x(event.frame_index) + 4}" y="${top - 8}">${escapeHtml(event.to_profile.toUpperCase())}</text></g>`).join("");
  const markers = state.frames.map((event, index) => `<circle class="route-marker ${event.fallback.used ? "fallback-marker" : ""}" data-index="${index}" cx="${x(event.frame_index)}" cy="${y(event.timing.observed_request_ms)}" r="${event.fallback.used ? 5 : 3}" fill="${event.decision.executed_worker === "edge" ? "var(--edge)" : "var(--remote)"}"/>`).join("");
  svg.innerHTML = `<title id="chart-title">Latency estimates and observed request latency by frame</title><desc id="chart-description">Experiment latency trace.</desc>${grid}${transitionLines}${series}${markers}<line id="playhead" class="playhead" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"/><text class="axis-label" x="${left}" y="${height - 8}">frame 0</text><text class="axis-label" x="${width - right}" y="${height - 8}" text-anchor="end">frame ${maxFrame}</text>`;
  svg.dataset.left = String(left); svg.dataset.right = String(right); svg.dataset.maxFrame = String(maxFrame);
  svg.querySelectorAll(".route-marker").forEach((marker) => marker.addEventListener("click", (event) => { event.stopPropagation(); selectFrame(Number(marker.dataset.index)); }));
}

function linePath(events, accessor, x, y) {
  let drawing = false;
  return events.map((event) => {
    const value = accessor(event);
    if (!Number.isFinite(value)) { drawing = false; return ""; }
    const command = drawing ? "L" : "M";
    drawing = true;
    return `${command}${x(event.frame_index).toFixed(2)},${y(value).toFixed(2)}`;
  }).join(" ");
}

function updateChartPlayhead(frame) {
  const svg = $("#latency-chart");
  const left = Number(svg.dataset.left), right = Number(svg.dataset.right), maxFrame = Number(svg.dataset.maxFrame);
  const x = left + frame / maxFrame * (900 - left - right);
  $("#playhead").setAttribute("x1", String(x));
  $("#playhead").setAttribute("x2", String(x));
}

function selectChartPosition(event) {
  const svg = $("#latency-chart");
  const bounds = svg.getBoundingClientRect();
  const frame = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)) * Number(svg.dataset.maxFrame);
  selectNearestFrame(frame);
}

function selectNearestFrame(frame) {
  let best = 0;
  state.frames.forEach((event, index) => {
    if (Math.abs(event.frame_index - frame) < Math.abs(state.frames[best].frame_index - frame)) best = index;
  });
  selectFrame(best);
}

function updateSelection() {
  const sequence = state.frames[state.index].sequence;
  $("#event-list").querySelectorAll("button").forEach((button) => {
    const selected = Number(button.dataset.sequence) === sequence;
    button.classList.toggle("selected", selected);
    if (selected) button.scrollIntoView({ block: "nearest" });
  });
}

function updateActivePhase(profile) {
  document.querySelectorAll(".phase").forEach((phase) => phase.classList.toggle("active", phase.dataset.profile === profile));
}

function togglePlay() {
  if (state.playing) stopPlayback(); else startPlayback();
}

function startPlayback() {
  if (state.index === state.frames.length - 1) state.index = -1;
  state.playing = true;
  $("#play-button").textContent = "Ⅱ";
  $("#play-button").setAttribute("aria-label", "Pause replay");
  $("#play-button").setAttribute("aria-pressed", "true");
  scheduleTick();
}

function scheduleTick() {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    selectFrame(state.index + 1);
    if (state.playing) scheduleTick();
  }, 800 / state.speed);
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  $("#play-button").textContent = "▶";
  $("#play-button").setAttribute("aria-label", "Play replay");
  $("#play-button").setAttribute("aria-pressed", "false");
}

function setSpeed(speed) {
  state.speed = speed;
  document.querySelectorAll("[data-speed]").forEach((button) => {
    const active = Number(button.dataset.speed) === speed;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.playing) scheduleTick();
}

function handleKeyboard(event) {
  if (event.target.matches("input, button, a")) return;
  if (event.code === "Space") { event.preventDefault(); togglePlay(); }
  if (event.key === "ArrowLeft") { event.preventDefault(); selectFrame(state.index - 1); }
  if (event.key === "ArrowRight") { event.preventDefault(); selectFrame(state.index + 1); }
  if (event.key === "Home") { event.preventDefault(); selectFrame(0); }
  if (event.key === "End") { event.preventDefault(); selectFrame(state.frames.length - 1); }
}

function countFallbacks() { return state.frames.filter((event) => event.fallback.used).length; }
function countSwitches() { return state.frames.slice(1).filter((event, index) => event.decision.executed_worker !== state.frames[index].decision.executed_worker).length; }
function displayModel(model) { return model === "ssdlite320_mobilenet_v3_large" ? "SSDLite320 MobileNet V3" : model; }
function formatNumber(value, digits = 1) { return Number.isFinite(value) ? Number(value).toFixed(digits) : "—"; }
function formatMs(value) { return Number.isFinite(value) ? `${formatNumber(value)} ms` : "—"; }
function escapeHtml(value) { return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
