import glob
import json
import os
import time
from typing import Any, Dict, List

from flask import Flask, jsonify

from common import (
    DEVICE_LAYOUT,
    default_base_dir,
    normalize_health_state,
    normalize_network_state,
    parse_timestamp,
    read_csv_tail_rows,
    safe_float,
    safe_int,
    safe_read_json,
)

app = Flask(__name__)
BASE_DIR = os.environ.get("HEALTH_DATA_BASE_DIR", default_base_dir())


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def resolve_sender_file(device_id: int, device_type: str) -> str:
    exact = os.path.join(BASE_DIR, f"sender_log_dev{device_id}_{device_type}.csv")
    if os.path.exists(exact):
        return exact
    matches = glob.glob(os.path.join(BASE_DIR, f"sender_log_dev{device_id}_*.csv"))
    return matches[0] if matches else ""


def dashboard_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Ward Monitor</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap\" rel=\"stylesheet\">
  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js\"></script>
  <style>
    :root {
      --bg: #f3f6fb;
      --surface: #ffffff;
      --border: #dbe3ef;
      --text: #0f172a;
      --muted: #64748b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, sans-serif;
      background: radial-gradient(circle at 10% 0%, #ffffff 0%, #eaf0f8 35%, var(--bg) 100%), var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    .container { max-width: 1360px; margin: 0 auto; padding: 16px; display: grid; gap: 16px; overflow-x: clip; }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
      overflow: hidden;
      transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
    }
    .top-bar { min-height: 72px; display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 14px; }
    .title-main { font-size: 22px; font-weight: 800; }
    .title-sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
    .state-wrap { text-align: center; }
    .state-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 220px;
      padding: 11px 14px;
      border-radius: 999px;
      font-size: 22px;
      font-weight: 800;
      transition: background-color 0.3s ease, color 0.3s ease;
      background: #e2e8f0;
      color: #475569;
    }
    .active-command { margin-top: 8px; font-size: 13px; color: var(--muted); transition: color 0.3s ease; }
    .stats { text-align: right; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: transparent; margin-left: 6px; vertical-align: middle; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; align-items: start; }
    .metrics .card {
      height: 220px;
      display: flex;
      flex-direction: column;
    }
    .metric-head { display: flex; align-items: start; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .metric-title { color: var(--muted); font-size: 13px; }
    .metric-value { font-size: 30px; font-weight: 800; line-height: 1; }
    .metrics canvas {
      width: 100% !important;
      height: 92px !important;
      max-height: 92px !important;
      margin-top: auto;
    }
    .badge {
      font-size: 11px;
      border-radius: 999px;
      padding: 4px 8px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
      transition: background-color 0.3s ease, color 0.3s ease;
    }
    .badge-normal { background: rgba(22, 163, 74, 0.14); color: #15803d; border-color: rgba(22, 163, 74, 0.28); }
    .badge-warning { background: rgba(217, 119, 6, 0.14); color: #b45309; border-color: rgba(217, 119, 6, 0.28); }
    .badge-critical { background: rgba(220, 38, 38, 0.14); color: #b91c1c; border-color: rgba(220, 38, 38, 0.28); }
    .mode-pill { background: #eef2f7; color: #334155; border-color: #d2dbe8; }
    .section-title { font-size: 17px; font-weight: 700; margin: 2px 0 10px; }
    .device-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
    .device-card {
      position: relative;
      height: 140px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 8px;
      overflow: hidden;
      transition: opacity 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .device-top { display: flex; align-items: center; justify-content: space-between; color: var(--muted); font-size: 13px; }
    .device-value { display: flex; align-items: end; gap: 8px; font-size: 28px; font-weight: 800; line-height: 1; }
    .device-unit { color: var(--muted); font-size: 14px; font-weight: 600; padding-bottom: 2px; }
    .device-bottom { display: flex; gap: 8px; align-items: center; }
    .stale { opacity: 0.4; }
    .stale-overlay {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #dc2626;
      letter-spacing: 0.06em;
      background: rgba(255, 255, 255, 0.78);
      pointer-events: none;
    }
    .stale .stale-overlay { display: flex; }
    @keyframes pulse-border {
      0%   { border-color: #da3633; box-shadow: 0 0 0 0 rgba(218,54,51,0.4); }
      70%  { border-color: #da3633; box-shadow: 0 0 0 6px rgba(218,54,51,0); }
      100% { border-color: #da3633; box-shadow: 0 0 0 0 rgba(218,54,51,0); }
    }
    .critical-pulse { animation: pulse-border 1.5s infinite; }
    .bottom-split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .architecture-flow { display: grid; justify-items: center; gap: 8px; padding-top: 8px; }
    .arch-box { width: min(360px, 100%); border: 1px solid var(--border); border-radius: 12px; padding: 10px; text-align: center; background: #f8fafc; }
    .arch-title { font-weight: 800; font-size: 14px; }
    .arch-sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .arch-arrow { font-size: 12px; color: var(--muted); display: grid; justify-items: center; line-height: 1.2; }
    .history { max-height: 300px; overflow-y: auto; margin-top: 8px; border: 1px solid var(--border); border-radius: 10px; background: #f8fafc; }
    .history-row {
      display: grid;
      grid-template-columns: 70px 108px 16px minmax(0, 1fr) 62px;
      align-items: center;
      gap: 8px;
      padding: 10px;
      border-top: 1px solid #e2e8f0;
      font-size: 13px;
    }
    .history-row .badge {
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: middle;
    }
    .history-row:first-child { border-top: none; }
    .history-row.newest { background: #edf3ff; }
    .history-empty { color: var(--muted); text-align: center; padding: 26px 10px; }
    .footer { text-align: center; color: var(--muted); font-size: 12px; padding: 2px 4px 10px; }
    @media (max-width: 1024px) {
      .top-bar { grid-template-columns: 1fr; text-align: left; }
      .state-wrap { text-align: left; }
      .stats { text-align: left; }
      .metrics { grid-template-columns: 1fr; }
      .metrics .card { height: 210px; }
      .bottom-split { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .container { padding: 10px; gap: 12px; }
      .history-row {
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .history-row > div { min-width: 0; }
      .state-badge { width: 100%; max-width: 260px; }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"card top-bar\">
      <div>
        <div class=\"title-main\">Ward Network Monitor</div>
        <div class=\"title-sub\">AI-Native Closed-Loop Control</div>
      </div>
      <div class=\"state-wrap\">
        <div id=\"stateBadge\" class=\"state-badge\">● UNKNOWN</div>
        <div id=\"activeCommand\" class=\"active-command\">Active command: N/A</div>
      </div>
      <div class=\"stats\">
        <div id=\"latencyStat\">Latency: -- ms avg</div>
        <div id=\"updatedStat\">Updated: --<span id=\"fetchDot\" class=\"dot\"></span></div>
        <div id=\"devicesStat\">Devices: 0/8 online</div>
      </div>
    </div>

    <div class=\"metrics\">
      <div class=\"card\"><div class=\"metric-head\"><div><div class=\"metric-title\">Packet Loss</div><div id=\"packetLossValue\" class=\"metric-value\">0.0%</div></div><span id=\"packetLossBadge\" class=\"badge badge-normal\">NORMAL</span></div><canvas id=\"packetLossChart\" height=\"80\"></canvas></div>
      <div class=\"card\"><div class=\"metric-head\"><div><div class=\"metric-title\">Average Delay</div><div id=\"delayValue\" class=\"metric-value\">0 ms</div></div><span id=\"delayBadge\" class=\"badge badge-normal\">NORMAL</span></div><canvas id=\"delayChart\" height=\"80\"></canvas></div>
      <div class=\"card\"><div class=\"metric-head\"><div><div class=\"metric-title\">Jitter</div><div id=\"jitterValue\" class=\"metric-value\">0 ms</div></div><span id=\"jitterBadge\" class=\"badge badge-normal\">NORMAL</span></div><canvas id=\"jitterChart\" height=\"80\"></canvas></div>
    </div>

    <div class=\"card\">
      <div class=\"section-title\">Patient Device Fleet</div>
      <div id=\"deviceGrid\" class=\"device-grid\"></div>
    </div>

    <div class=\"bottom-split\">
      <div class=\"card\">
        <div class=\"section-title\">System Architecture</div>
        <div class=\"architecture-flow\">
          <div class=\"arch-box\" style=\"border-color:#238636\"><div class=\"arch-title\">8 SENDERS</div><div class=\"arch-sub\">ECGx2 SpO2x2 BPx2 Temp Resp</div></div>
          <div class=\"arch-arrow\">UDP :9000<br>▼</div>
          <div class=\"arch-box\" style=\"border-color:#58a6ff\"><div class=\"arch-title\">RECEIVER</div><div class=\"arch-sub\">ML Inference | 9-Case Policy</div><div style=\"margin-top:6px\"><span id=\"receiverState\" class=\"badge mode-pill\">UNKNOWN</span></div></div>
          <div class=\"arch-arrow\">UDP :5006<br>▼</div>
          <div class=\"arch-box\" style=\"border-color:#8957e5\"><div class=\"arch-title\">WARD CONTROLLER</div><div class=\"arch-sub\">Fleet Manager</div><div style=\"margin-top:6px\"><span id=\"controllerCommand\" class=\"badge mode-pill\">N/A</span></div></div>
          <div class=\"arch-arrow\">ward_mode_state.json<br>▼</div>
          <div class=\"arch-box\" style=\"border-color:#6e7681\"><div class=\"arch-title\">ALL 8 SENDERS</div><div class=\"arch-sub\">Adaptive Mode</div></div>
        </div>
      </div>
      <div class=\"card\"><div class=\"section-title\">Command History</div><div id=\"commandHistory\" class=\"history\"><div class=\"history-empty\">Awaiting first command...</div></div></div>
    </div>

    <div class=\"footer\">AI-Native Network Control - Ward Monitor  |  Polling 1s  |  XGBoost F1=0.865  |  Latency ~519ms  |  9-Case Adaptive Policy</div>
  </div>

  <script>
    const DEVICE_LAYOUT = [
      { id: 0, type: "ECG", emoji: "❤️" }, { id: 1, type: "ECG", emoji: "❤️" },
      { id: 2, type: "SpO2", emoji: "🫁" }, { id: 3, type: "SpO2", emoji: "🫁" },
      { id: 4, type: "BloodPressure", emoji: "🩸" }, { id: 5, type: "BloodPressure", emoji: "🩸" },
      { id: 6, type: "Temperature", emoji: "🌡️" }, { id: 7, type: "Respiration", emoji: "💨" }
    ];
    const STATE_COLORS = {
      Stable: { fg: "#166534", bg: "#dcfce7", dot: "🟢" },
      Unstable: { fg: "#92400e", bg: "#fef3c7", dot: "🟡" },
      Critical: { fg: "#991b1b", bg: "#fee2e2", dot: "🔴" },
      UNKNOWN: { fg: "#475569", bg: "#e2e8f0", dot: "⚪" }
    };
    const HEALTH_COLORS = { NORMAL: "badge-normal", ALERT: "badge-warning", CRITICAL: "badge-critical", UNKNOWN: "mode-pill" };
    const COMMAND_COLORS = {
      FULL_ECG: "#15803d", FULL_ECG_PRIORITY: "#15803d", DOWNSAMPLED_ECG: "#b45309",
      SEMANTIC_ALERT: "#b45309", SEMANTIC_CRITICAL: "#b91c1c", SEMANTIC_SUMMARY: "#64748b"
    };

    let charts = {};
    let currentCommand = "N/A";
    const fetchStatus = { state: false, devices: false, telemetry: false, commands: false };
    const CHART_WINDOW = 60;

    function setFetchFailure(endpoint, failed) {
      fetchStatus[endpoint] = !!failed;
      const anyFailed = Object.values(fetchStatus).some(Boolean);
      document.getElementById("fetchDot").style.backgroundColor = anyFailed ? "#da3633" : "transparent";
    }

    function setBadgeClass(el, cls) {
      el.classList.remove("badge-normal", "badge-warning", "badge-critical", "mode-pill");
      el.classList.add(cls);
    }

    function threshold(metric, value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return { label: "NORMAL", cls: "badge-normal", color: "#15803d" };
      if (metric === "packet_loss_rate") { if (v >= 8) return { label: "CRITICAL", cls: "badge-critical", color: "#b91c1c" }; if (v >= 3) return { label: "WARNING", cls: "badge-warning", color: "#b45309" }; }
      if (metric === "avg_delay") { if (v >= 130) return { label: "CRITICAL", cls: "badge-critical", color: "#b91c1c" }; if (v >= 40) return { label: "WARNING", cls: "badge-warning", color: "#b45309" }; }
      if (metric === "jitter") { if (v >= 35) return { label: "CRITICAL", cls: "badge-critical", color: "#b91c1c" }; if (v >= 8) return { label: "WARNING", cls: "badge-warning", color: "#b45309" }; }
      return { label: "NORMAL", cls: "badge-normal", color: "#15803d" };
    }

    function makeChart(canvasId, lineColor) {
      return new Chart(document.getElementById(canvasId), {
        type: "line",
        data: {
          labels: Array(CHART_WINDOW).fill(""),
          datasets: [{
            data: Array(CHART_WINDOW).fill(null),
            borderColor: lineColor,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
            fill: true,
            backgroundColor: "rgba(88,166,255,0.2)"
          }]
        },
        options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { x: { display: false }, y: { display: false } } }
      });
    }

    function buildDevices() {
      const grid = document.getElementById("deviceGrid");
      grid.innerHTML = DEVICE_LAYOUT.map(d => `
        <div class="card device-card stale" id="device-${d.id}">
          <div class="device-top"><span>${d.emoji} ${d.type} · ID ${d.id}</span><span id="dev-age-${d.id}">--</span></div>
          <div class="device-value"><span id="dev-value-${d.id}">--</span><span class="device-unit" id="dev-unit-${d.id}">--</span></div>
          <div class="device-bottom"><span class="badge mode-pill" id="dev-label-${d.id}">UNKNOWN</span><span class="badge mode-pill" id="dev-mode-${d.id}">AUTO</span></div>
          <div class="stale-overlay">NO SIGNAL</div>
        </div>
      `).join("");
    }

    async function updateState() {
      try {
        const r = await fetch("/api/state", { cache: "no-store" });
        if (!r.ok) throw new Error("state");
        const d = await r.json();
        const s = STATE_COLORS[d.network_state] ? d.network_state : "UNKNOWN";
        const c = STATE_COLORS[s];
        const badge = document.getElementById("stateBadge");
        badge.textContent = `● ${s.toUpperCase()}`;
        badge.style.backgroundColor = c.bg;
        badge.style.color = c.fg;

        currentCommand = d.command || "N/A";
        const cmdEl = document.getElementById("activeCommand");
        cmdEl.textContent = `Active command: ${currentCommand}`;
        cmdEl.style.color = COMMAND_COLORS[currentCommand] || "#8b949e";

        document.getElementById("receiverState").textContent = s;
        document.getElementById("controllerCommand").textContent = currentCommand;

        const secondsAgo = Number(d.seconds_ago);
        const updatedEl = document.getElementById("updatedStat");
        const dot = document.getElementById("fetchDot").outerHTML;
        if (Number.isFinite(secondsAgo)) {
          const color = secondsAgo > 8 ? "#da3633" : secondsAgo > 3 ? "#d29922" : "#8b949e";
          updatedEl.style.color = color;
          updatedEl.innerHTML = `Updated: ${secondsAgo.toFixed(1)}s ago ${dot}`;
        }
        document.title = `Ward Monitor — ${s}`;
        setFetchFailure("state", false);
      } catch (_) { setFetchFailure("state", true); }
    }

    async function updateDevices() {
      try {
        const r = await fetch("/api/devices", { cache: "no-store" });
        if (!r.ok) throw new Error("devices");
        const arr = await r.json();
        let online = 0;
        arr.forEach(d => {
          const id = Number(d.device_id);
          const card = document.getElementById(`device-${id}`);
          if (!card) return;
          const value = Number(d.value);
          document.getElementById(`dev-value-${id}`).textContent = Number.isFinite(value) ? (Math.abs(value % 1) < 0.001 ? value.toFixed(0) : value.toFixed(1)) : "--";
          document.getElementById(`dev-unit-${id}`).textContent = d.unit || "--";
          const label = String(d.label || "UNKNOWN").toUpperCase();
          const labelEl = document.getElementById(`dev-label-${id}`);
          labelEl.textContent = label;
          setBadgeClass(labelEl, HEALTH_COLORS[label] || "mode-pill");
          const modeEl = document.getElementById(`dev-mode-${id}`);
          modeEl.textContent = d.mode || "AUTO";
          setBadgeClass(modeEl, "mode-pill");
          const age = Number(d.seconds_ago);
          document.getElementById(`dev-age-${id}`).textContent = Number.isFinite(age) ? `${age.toFixed(1)}s` : "--";
          const stale = !!d.stale;
          card.classList.toggle("stale", stale);
          card.classList.toggle("critical-pulse", !stale && label === "CRITICAL");
          if (!stale) online += 1;
        });
        document.getElementById("devicesStat").textContent = `Devices: ${online}/8 online`;
        setFetchFailure("devices", false);
      } catch (_) { setFetchFailure("devices", true); }
    }

    function updateMetric(kind, value, unit) {
      const t = threshold(kind, value);
      const ids = { packet_loss_rate: ["packetLossValue", "packetLossBadge", "packetLoss"], avg_delay: ["delayValue", "delayBadge", "delay"], jitter: ["jitterValue", "jitterBadge", "jitter"] }[kind];
      const vEl = document.getElementById(ids[0]);
      vEl.textContent = `${Number(value || 0).toFixed(1)}${unit}`;
      vEl.style.color = t.color;
      const bEl = document.getElementById(ids[1]);
      bEl.textContent = t.label;
      setBadgeClass(bEl, t.cls);
      const chart = charts[ids[2]];
      const y = Number(value) || 0;
      const series = chart.data.datasets[0].data;
      const hasSeed = series.some(v => v !== null);
      if (!hasSeed) {
        chart.data.datasets[0].data = Array(CHART_WINDOW).fill(y);
      } else {
        series.shift();
        series.push(y);
      }
      chart.data.datasets[0].borderColor = t.color;
      chart.data.datasets[0].backgroundColor = `${t.color}22`;
      chart.update("none");
    }

    async function updateTelemetry() {
      try {
        const r = await fetch("/api/telemetry", { cache: "no-store" });
        if (!r.ok) throw new Error("telemetry");
        const d = await r.json();
        const cur = d.current || {};
        updateMetric("packet_loss_rate", cur.packet_loss_rate || 0, "%");
        updateMetric("avg_delay", cur.avg_delay || 0, " ms");
        updateMetric("jitter", cur.jitter || 0, " ms");
        setFetchFailure("telemetry", false);
      } catch (_) { setFetchFailure("telemetry", true); }
    }

    async function updateCommands() {
      try {
        const r = await fetch("/api/commands", { cache: "no-store" });
        if (!r.ok) throw new Error("commands");
        const rows = await r.json();
        const container = document.getElementById("commandHistory");
        if (!Array.isArray(rows) || rows.length === 0) {
          container.innerHTML = '<div class="history-empty">Awaiting first command...</div>';
          return;
        }
        const latencies = rows.map(x => Number(x.latency_ms)).filter(x => Number.isFinite(x));
        if (latencies.length) {
          const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
          document.getElementById("latencyStat").textContent = `Latency: ${avg.toFixed(0)} ms avg`;
        }
        container.innerHTML = rows.map((r, i) => {
          const ns = STATE_COLORS[r.network_state] ? r.network_state : "UNKNOWN";
          const cmdColor = COMMAND_COLORS[r.command] || "#8b949e";
          return `
            <div class="history-row ${i === 0 ? "newest" : ""}">
              <div>${r.time_str || "--:--:--"}</div>
              <div><span class="badge mode-pill" style="color:${STATE_COLORS[ns].fg};border-color:${STATE_COLORS[ns].fg}66">${STATE_COLORS[ns].dot} ${ns.toUpperCase()}</span></div>
              <div>→</div>
              <div><span class="badge mode-pill" style="color:${cmdColor};border-color:${cmdColor}66">${r.command || "N/A"}</span></div>
              <div>${Number(r.latency_ms || 0).toFixed(0)}ms</div>
            </div>
          `;
        }).join("");
        setFetchFailure("commands", false);
      } catch (_) { setFetchFailure("commands", true); }
    }

    function bootstrap() {
      buildDevices();
      charts.packetLoss = makeChart("packetLossChart", "#15803d");
      charts.delay = makeChart("delayChart", "#15803d");
      charts.jitter = makeChart("jitterChart", "#15803d");
      updateState(); updateDevices(); updateTelemetry(); updateCommands();
      setInterval(updateState, 1000);
      setInterval(updateDevices, 1000);
      setInterval(updateTelemetry, 1000);
      setInterval(updateCommands, 2000);
    }
    window.addEventListener("DOMContentLoaded", bootstrap);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return dashboard_html()


@app.route("/api/state")
def api_state():
    result: Dict[str, Any] = {
        "network_state": "UNKNOWN",
        "command": "N/A",
        "health_state": "UNKNOWN",
        "timestamp": 0.0,
        "consecutive_windows": 0,
        "seconds_ago": None,
    }

    try:
        path = os.path.join(BASE_DIR, "ward_mode_state.json")
        raw = safe_read_json(path, fallback={})
        ts = parse_timestamp(raw.get("timestamp"))
        now = time.time()
        result.update(
            {
                "network_state": normalize_network_state(raw.get("network_state")),
                "command": str(raw.get("command") or "N/A"),
                "health_state": normalize_health_state(raw.get("health_state")),
                "timestamp": ts if ts is not None else 0.0,
                "consecutive_windows": safe_int(raw.get("consecutive_windows"), 0),
                "seconds_ago": max(0.0, now - ts) if ts is not None else None,
            }
        )
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/devices")
def api_devices():
    now = time.time()
    command = "N/A"

    try:
        state = safe_read_json(os.path.join(BASE_DIR, "ward_mode_state.json"), fallback={})
        command = str(state.get("command") or "N/A")
    except Exception:
        pass

    def command_mode_short(raw: str) -> str:
        c = (raw or "").upper()
        if c.startswith("DOWNSAMPLED"):
            return "DOWNSAMPLED"
        if c.startswith("FULL_ECG"):
            return "FULL"
        if c.startswith("SEMANTIC_CRITICAL"):
            return "CRITICAL"
        if c.startswith("SEMANTIC_ALERT"):
            return "ALERT"
        if c.startswith("SEMANTIC_SUMMARY"):
            return "SUMMARY"
        return "AUTO"

    rows: List[Dict[str, Any]] = []

    for device_id, device_type in DEVICE_LAYOUT:
        item: Dict[str, Any] = {
            "device_id": device_id,
            "device_type": device_type,
            "value": None,
            "unit": "",
            "label": "UNKNOWN",
            "seconds_ago": 9999.0,
            "stale": True,
            "mode": command_mode_short(command),
        }

        try:
            p = resolve_sender_file(device_id, device_type)
            if p:
                tail_rows = read_csv_tail_rows(p, 1)
                if tail_rows:
                    last = tail_rows[-1]
                    ts = parse_timestamp(last.get("timestamp"))
                    secs = max(0.0, now - ts) if ts is not None else 9999.0
                    val = safe_float(last.get("value"))
                    if val is not None and abs(val - round(val)) < 1e-6:
                        val = int(round(val))
                    item.update(
                        {
                            "device_type": str(last.get("device_type") or device_type),
                            "value": val,
                            "unit": str(last.get("unit") or ""),
                            "label": normalize_health_state(last.get("label")),
                            "seconds_ago": secs,
                            "stale": secs > 3.0,
                        }
                    )
        except Exception:
            pass

        rows.append(item)

    return jsonify(rows)


@app.route("/api/telemetry")
def api_telemetry():
    payload = {
        "packet_loss_rate": [],
        "avg_delay": [],
        "jitter": [],
        "throughput_bps": [],
        "timestamps": [],
        "current": {
            "packet_loss_rate": 0.0,
            "avg_delay": 0.0,
            "jitter": 0.0,
            "throughput_bps": 0.0,
            "active_devices": 0,
        },
    }

    try:
        rows = read_csv_tail_rows(os.path.join(BASE_DIR, "network_telemetry.csv"), 120)
        for row in rows:
            payload["packet_loss_rate"].append((safe_float(row.get("packet_loss_rate"), 0.0) or 0.0) * 100.0)
            payload["avg_delay"].append(safe_float(row.get("avg_delay"), 0.0) or 0.0)
            payload["jitter"].append(safe_float(row.get("jitter"), 0.0) or 0.0)
            payload["throughput_bps"].append(safe_float(row.get("throughput_bps"), 0.0) or 0.0)
            payload["timestamps"].append(parse_timestamp(row.get("timestamp")) or 0.0)

        if rows:
            last = rows[-1]
            payload["current"] = {
                "packet_loss_rate": (safe_float(last.get("packet_loss_rate"), 0.0) or 0.0) * 100.0,
                "avg_delay": safe_float(last.get("avg_delay"), 0.0) or 0.0,
                "jitter": safe_float(last.get("jitter"), 0.0) or 0.0,
                "throughput_bps": safe_float(last.get("throughput_bps"), 0.0) or 0.0,
                "active_devices": safe_int(last.get("active_devices"), 0),
            }
    except Exception:
        pass

    return jsonify(payload)


@app.route("/api/commands")
def api_commands():
    out: List[Dict[str, Any]] = []
    try:
        rows = read_csv_tail_rows(os.path.join(BASE_DIR, "command_log.csv"), 20)
        for row in reversed(rows):
            ts = parse_timestamp(row.get("timestamp"))
            out.append(
                {
                    "time_str": time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "--:--:--",
                    "network_state": normalize_network_state(row.get("network_state")),
                    "health_state": normalize_health_state(row.get("health_state")),
                    "command": str(row.get("command") or "N/A"),
                    "latency_ms": safe_float(row.get("latency_ms"), 0.0) or 0.0,
                }
            )
    except Exception:
        pass

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
