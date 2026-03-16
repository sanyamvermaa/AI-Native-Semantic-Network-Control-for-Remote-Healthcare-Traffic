# Requirements: pip install flask
# Run: python dashboard.py
# Then open: http://localhost:5050

import csv
import json
import os
import time
from collections import deque
from datetime import datetime

from flask import Flask, jsonify, make_response

app = Flask(__name__)

BASE_DIR = os.environ.get("HEALTH_DATA_BASE_DIR", "./csv")

DEVICE_FILES = [
    (0, "ECG"),
    (1, "ECG"),
    (2, "SpO2"),
    (3, "SpO2"),
    (4, "BloodPressure"),
    (5, "BloodPressure"),
    (6, "Temperature"),
    (7, "Respiration"),
]

STATE_FILE = "ward_mode_state.json"
COMMAND_LOG_FILE = "command_log.csv"
TELEMETRY_FILE = "network_telemetry.csv"


def _full_path(filename):
    return os.path.join(BASE_DIR, filename)


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_str(value, default=""):
    if value is None:
        return default
    return str(value)


def _read_last_csv_rows(filename, max_rows):
    rows = []
    try:
        path = _full_path(filename)
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = f.readline()
            if not header:
                return rows
            tail = deque(f, maxlen=max_rows)
            if not tail:
                return rows
            reader = csv.DictReader([header] + list(tail))
            for row in reader:
                if row:
                    rows.append(row)
    except Exception:
        return []
    return rows


def _read_last_csv_row(filename):
    try:
        path = _full_path(filename)
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = f.readline()
            if not header:
                return None
            tail = deque(f, maxlen=1)
            if not tail:
                return None
            reader = csv.DictReader([header] + list(tail))
            for row in reader:
                if row:
                    return row
    except Exception:
        return None
    return None


def _read_state():
    defaults = {
        "network_state": "UNKNOWN",
        "command": "FULL_ECG",
        "health_state": "UNKNOWN",
        "timestamp": None,
        "consecutive_windows": 0,
        "last_updated_ago_sec": None,
    }

    try:
        path = _full_path(STATE_FILE)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        now = time.time()
        try:
            mtime = os.path.getmtime(path)
            defaults["last_updated_ago_sec"] = max(0.0, now - mtime)
        except Exception:
            defaults["last_updated_ago_sec"] = None

        defaults["network_state"] = _safe_str(data.get("network_state"), "UNKNOWN")
        defaults["command"] = _safe_str(data.get("command"), "FULL_ECG")
        defaults["health_state"] = _safe_str(data.get("health_state"), "UNKNOWN")
        defaults["timestamp"] = _safe_float(data.get("timestamp"), None)
        defaults["consecutive_windows"] = _safe_int(data.get("consecutive_windows"), 0)
    except Exception:
        pass

    return defaults


def _device_default(device_id, device_type):
    return {
        "device_id": device_id,
        "device_type": device_type,
        "value": None,
        "unit": "",
        "label": "UNKNOWN",
        "last_seen_sec": None,
        "stale": True,
    }


def _read_devices():
    devices = []
    now = time.time()

    for device_id, device_type in DEVICE_FILES:
        filename = f"sender_log_dev{device_id}_{device_type}.csv"
        device = _device_default(device_id, device_type)

        try:
            row = _read_last_csv_row(filename)
            if not row:
                devices.append(device)
                continue

            ts = _safe_float(row.get("timestamp"), None)
            last_seen = (now - ts) if ts is not None else None

            device["value"] = _safe_float(row.get("value"), row.get("value"))
            device["unit"] = _safe_str(row.get("unit"), "")
            device["label"] = _safe_str(row.get("label"), "UNKNOWN")
            device["last_seen_sec"] = round(last_seen, 3) if last_seen is not None else None
            device["stale"] = True if (last_seen is None or last_seen > 3.0) else False
        except Exception:
            pass

        devices.append(device)

    return devices


def _read_telemetry():
    payload = {
        "timestamps": [],
        "packet_loss_rate": [],
        "avg_delay": [],
        "jitter": [],
        "throughput_bps": [],
        "current": {
            "packet_loss_rate": None,
            "avg_delay": None,
            "jitter": None,
            "throughput_bps": None,
        },
    }

    try:
        rows = _read_last_csv_rows(TELEMETRY_FILE, 120)
        if not rows:
            return payload

        for row in rows:
            payload["timestamps"].append(_safe_float(row.get("timestamp"), None))
            payload["packet_loss_rate"].append(_safe_float(row.get("packet_loss_rate"), None))
            payload["avg_delay"].append(_safe_float(row.get("avg_delay"), None))
            payload["jitter"].append(_safe_float(row.get("jitter"), None))
            payload["throughput_bps"].append(_safe_float(row.get("throughput_bps"), None))

        current = rows[-1]
        payload["current"] = {
            "packet_loss_rate": _safe_float(current.get("packet_loss_rate"), None),
            "avg_delay": _safe_float(current.get("avg_delay"), None),
            "jitter": _safe_float(current.get("jitter"), None),
            "throughput_bps": _safe_float(current.get("throughput_bps"), None),
        }
    except Exception:
        return payload

    return payload


def _format_time(epoch_ts):
    ts = _safe_float(epoch_ts, None)
    if ts is None:
        return "--:--:--"
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


def _read_commands():
    entries = []
    try:
        rows = _read_last_csv_rows(COMMAND_LOG_FILE, 15)
        if not rows:
            return []

        for row in reversed(rows):
            ts = _safe_float(row.get("timestamp"), None)
            entries.append(
                {
                    "timestamp": ts,
                    "time_str": _format_time(ts),
                    "network_state": _safe_str(row.get("network_state"), "UNKNOWN"),
                    "health_state": _safe_str(row.get("health_state"), "UNKNOWN"),
                    "command": _safe_str(row.get("command"), "FULL_ECG"),
                    "latency_ms": _safe_float(row.get("latency_ms"), None),
                }
            )
    except Exception:
        return []

    return entries


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/")
def index():
    html = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Ward Monitor - UNKNOWN</title>
  <style>
    :root {
      --bg: #f3f5f8;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --unknown: #6b7280;
      --line: #dbe3ee;
      --ink-soft: #475569;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
    }
    .container {
      width: min(1320px, 96vw);
      margin: 0 auto;
      padding: 14px;
      display: grid;
      gap: 14px;
    }
    .header {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .title {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .state-badge {
      font-size: 1.05rem;
      font-weight: 700;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid transparent;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .state-stable { color: #14532d; background: #dcfce7; border-color: #86efac; }
    .state-unstable { color: #78350f; background: #fef3c7; border-color: #fcd34d; }
    .state-critical { color: #7f1d1d; background: #fee2e2; border-color: #fca5a5; }
    .state-unknown { color: #374151; background: #e5e7eb; border-color: #d1d5db; }
    .right {
      justify-self: end;
      text-align: right;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pill {
      font-size: 0.78rem;
      border-radius: 999px;
      padding: 6px 10px;
      background: #eef3fa;
      color: #1e293b;
      border: 1px solid #cdd9ea;
    }
    .ago { color: var(--muted); font-size: 0.85rem; }
    .kpi {
      color: var(--ink-soft);
      font-size: 0.82rem;
      width: 100%;
      text-align: right;
    }

    .devices {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .device-card {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      min-height: 132px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    .device-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.94rem;
      color: #334155;
      margin-bottom: 8px;
    }
    .device-value {
      font-size: 1.55rem;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .device-mode { color: var(--muted); font-size: 0.8rem; }
    .health {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      border-radius: 999px;
      padding: 4px 9px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .health-normal { color: #14532d; background: #dcfce7; border: 1px solid #86efac; }
    .health-alert { color: #92400e; background: #fef3c7; border: 1px solid #fcd34d; }
    .health-critical {
      color: #7f1d1d;
      background: #fee2e2;
      border: 1px solid #fca5a5;
      animation: pulseGlow 1.2s infinite;
    }
    @keyframes pulseGlow {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.65); }
      70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .stale {
      position: absolute;
      inset: 0;
      background: rgba(127, 29, 29, 0.58);
      color: #fee2e2;
      font-size: 1.4rem;
      font-weight: 800;
      display: none;
      align-items: center;
      justify-content: center;
      letter-spacing: 1px;
    }
    .stale.show { display: flex; }

    .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    .section-title {
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 10px;
    }

    .metric {
      display: grid;
      grid-template-columns: 280px 1fr 220px;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .metric:last-child { margin-bottom: 0; }
    .metric-name { color: #334155; font-size: 0.92rem; }
    .metric-name .v { color: #0f172a; font-weight: 700; }
    .bar-wrap {
      width: 100%;
      height: 14px;
      background: #f8fafc;
      border: 1px solid #d6dfeb;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      transition: width 0.4s ease;
      border-radius: 999px;
    }
    .spark { width: 200px; height: 40px; justify-self: end; }

    .history-list {
      max-height: 260px;
      overflow-y: auto;
      border-top: 1px solid #dbe3ee;
      margin-top: 8px;
    }
    .history-row {
      display: grid;
      grid-template-columns: 80px 1fr 100px;
      gap: 10px;
      padding: 8px 2px;
      border-bottom: 1px solid #e7edf5;
      font-size: 0.88rem;
    }
    .cmd-full { color: #93c5fd; }
    .cmd-downsampled { color: #fcd34d; }
    .cmd-minimal { color: #fca5a5; }
    .cmd-other { color: #d1d5db; }

    .footer {
      text-align: center;
      color: var(--muted);
      font-size: 0.86rem;
      padding: 8px;
    }

    @media (max-width: 1100px) {
      .devices { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric { grid-template-columns: 1fr; }
      .spark { justify-self: start; }
      .header { grid-template-columns: 1fr; text-align: center; }
      .right { justify-self: center; justify-content: center; }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"header\">
      <div class=\"title\">Ward Network Monitor</div>
      <div id=\"networkStateBadge\" class=\"state-badge state-unknown\">UNKNOWN</div>
      <div class=\"right\">
        <span id=\"commandBadge\" class=\"pill\">FULL_ECG</span>
        <span id=\"lastUpdated\" class=\"ago\">Last update --s ago</span>
        <div id=\"perfSummary\" class=\"kpi\">Loop latency -- ms | UI -- ms</div>
      </div>
    </div>

    <div>
      <div class=\"section-title\">Devices</div>
      <div id=\"devicesGrid\" class=\"devices\"></div>
    </div>

    <div class=\"section\">
      <div class=\"section-title\">Live Metrics</div>
      <div id=\"metrics\"></div>
    </div>

    <div class=\"section\">
      <div class=\"section-title\">Command History</div>
      <div id=\"history\" class=\"history-list\"></div>
    </div>

    <div id="footerInfo" class="footer">Polling every 0.4s | Pipeline: ward_controller + receiver</div>
  </div>

  <script>
    const POLL_MS = 400;

    const deviceIcons = {
      ECG: '❤️',
      SpO2: '🫁',
      BloodPressure: '🩸',
      Temperature: '🌡️',
      Respiration: '💨'
    };

    function stateClass(state) {
      const s = (state || 'UNKNOWN').toUpperCase();
      if (s === 'STABLE') return 'state-stable';
      if (s === 'UNSTABLE') return 'state-unstable';
      if (s === 'CRITICAL') return 'state-critical';
      return 'state-unknown';
    }

    function healthClass(label) {
      const l = (label || 'UNKNOWN').toUpperCase();
      if (l === 'NORMAL') return 'health-normal';
      if (l === 'ALERT') return 'health-alert';
      if (l === 'CRITICAL') return 'health-critical';
      return 'health-alert';
    }

    function commandClass(cmd) {
      const c = (cmd || '').toUpperCase();
      if (c.includes('FULL')) return 'cmd-full';
      if (c.includes('DOWNSAMPLED')) return 'cmd-downsampled';
      if (c.includes('MINIMAL')) return 'cmd-minimal';
      return 'cmd-other';
    }

    function metricColor(type, v) {
      if (v == null || Number.isNaN(v)) return '#6b7280';
      if (type === 'packet_loss_rate') {
        if (v < 3) return '#22c55e';
        if (v < 8) return '#f59e0b';
        return '#ef4444';
      }
      if (type === 'avg_delay') {
        if (v < 40) return '#22c55e';
        if (v < 130) return '#f59e0b';
        return '#ef4444';
      }
      if (type === 'jitter') {
        if (v < 8) return '#22c55e';
        if (v < 35) return '#f59e0b';
        return '#ef4444';
      }
      return '#93c5fd';
    }

    function metricMax(type) {
      if (type === 'packet_loss_rate') return 15;
      if (type === 'avg_delay') return 220;
      if (type === 'jitter') return 60;
      return 100;
    }

    function toNum(v) {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }

    function fmt(v, digits = 1, suffix = '') {
      const n = toNum(v);
      if (n == null) return '--';
      return n.toFixed(digits) + suffix;
    }

    function sparkline(values, color) {
      const width = 200;
      const height = 40;
      const cleaned = (values || []).map(toNum).filter(v => v != null).slice(-60);
      if (!cleaned.length) {
        return `<svg class=\"spark\" viewBox=\"0 0 ${width} ${height}\" width=\"200\" height=\"40\"></svg>`;
      }

      let min = Math.min(...cleaned);
      let max = Math.max(...cleaned);
      if (max === min) {
        max = min + 1;
      }

      const points = cleaned.map((v, i) => {
        const x = (i / Math.max(cleaned.length - 1, 1)) * (width - 2) + 1;
        const y = height - 1 - ((v - min) / (max - min)) * (height - 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      return `
        <svg class=\"spark\" viewBox=\"0 0 ${width} ${height}\" width=\"200\" height=\"40\">\n
          <polyline fill=\"none\" stroke=\"${color}\" stroke-width=\"2\" points=\"${points}\"/>\n
        </svg>
      `;
    }

    function median(nums) {
      const arr = (nums || []).filter(v => v != null).slice().sort((a, b) => a - b);
      if (!arr.length) return null;
      const mid = Math.floor(arr.length / 2);
      if (arr.length % 2 === 0) return (arr[mid - 1] + arr[mid]) / 2;
      return arr[mid];
    }

    function renderDevices(devices, command) {
      const grid = document.getElementById('devicesGrid');
      grid.innerHTML = devices.map(d => {
        const icon = deviceIcons[d.device_type] || '📟';
        const value = d.value == null ? '--' : d.value;
        const unit = d.unit || '';
        const label = (d.label || 'UNKNOWN').toUpperCase();
        const staleClass = d.stale ? 'show' : '';

        return `
          <div class=\"device-card\">
            <div class=\"device-head\">
              <span>${icon} Device ${d.device_id} (${d.device_type})</span>
              <span class=\"health ${healthClass(label)}\">${label}</span>
            </div>
            <div class=\"device-value\">${value} <span style=\"font-size:0.95rem;color:#9fb0cc\">${unit}</span></div>
            <div class=\"device-mode\">Mode: ${command || 'FULL_ECG'} | Last seen: ${fmt(d.last_seen_sec, 1, 's')}</div>
            <div class=\"stale ${staleClass}\">STALE</div>
          </div>
        `;
      }).join('');
    }

    function renderMetrics(t) {
      const holder = document.getElementById('metrics');
      const specs = [
        { key: 'packet_loss_rate', label: 'Packet Loss', unit: '%', arr: t.packet_loss_rate },
        { key: 'avg_delay', label: 'Avg Delay', unit: 'ms', arr: t.avg_delay },
        { key: 'jitter', label: 'Jitter', unit: 'ms', arr: t.jitter }
      ];

      holder.innerHTML = specs.map(s => {
        const current = toNum(t.current?.[s.key]);
        const max = metricMax(s.key);
        const pct = current == null ? 0 : Math.max(0, Math.min((current / max) * 100, 100));
        const color = metricColor(s.key, current);

        return `
          <div class=\"metric\">
            <div class=\"metric-name\">${s.label}: <span class=\"v\">${fmt(current, 1, s.unit)}</span></div>
            <div class=\"bar-wrap\"><div class=\"bar-fill\" style=\"width:${pct}%;background:${color};\"></div></div>
            ${sparkline(s.arr || [], color)}
          </div>
        `;
      }).join('');
    }

    function renderHistory(items) {
      const el = document.getElementById('history');
      el.innerHTML = (items || []).map(i => `
        <div class=\"history-row\">
          <div>${i.time_str || '--:--:--'}</div>
          <div class=\"${commandClass(i.command)}\">${i.network_state || 'UNKNOWN'} -> ${i.command || 'FULL_ECG'}</div>
          <div>${fmt(i.latency_ms, 1, ' ms')}</div>
        </div>
      `).join('');
    }

    function renderPerfSummary(commands, uiMs) {
      const latencies = (commands || []).map(c => toNum(c.latency_ms)).filter(v => v != null);
      const med = median(latencies);
      const summary = document.getElementById('perfSummary');
      summary.textContent = `Loop latency ${med == null ? '--' : med.toFixed(1)} ms | UI ${uiMs.toFixed(1)} ms`;
    }

    function applyState(s) {
      const badge = document.getElementById('networkStateBadge');
      const state = (s.network_state || 'UNKNOWN').toUpperCase();
      badge.className = `state-badge ${stateClass(state)}`;
      badge.textContent = state;

      document.getElementById('commandBadge').textContent = s.command || 'FULL_ECG';
      const ago = toNum(s.last_updated_ago_sec);
      document.getElementById('lastUpdated').textContent = `Last update ${ago == null ? '--' : ago.toFixed(1)}s ago`;

      document.title = `Ward Monitor — ${state}`;
    }

    async function fetchJson(url, fallback) {
      try {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) return fallback;
        return await r.json();
      } catch (_) {
        return fallback;
      }
    }

    async function fetchSnapshot() {
      return fetchJson('/api/snapshot', {
        state: { network_state: 'UNKNOWN', command: 'FULL_ECG', last_updated_ago_sec: null },
        devices: [],
        telemetry: { timestamps: [], packet_loss_rate: [], avg_delay: [], jitter: [], throughput_bps: [], current: {} },
        commands: []
      });
    }

    let polling = false;

    async function tick() {
      if (polling) return;
      polling = true;
      const t0 = performance.now();

      const snapshot = await fetchSnapshot();

      const state = snapshot?.state || {};
      const devices = snapshot?.devices || [];
      const telemetry = snapshot?.telemetry || {};
      const commands = snapshot?.commands || [];

      applyState(state);
      renderDevices(devices, state?.command || 'FULL_ECG');
      renderMetrics(telemetry);
      renderHistory(commands);
      renderPerfSummary(commands, performance.now() - t0);

      polling = false;
    }

    tick();
    setInterval(tick, POLL_MS);
    document.getElementById('footerInfo').textContent = `Polling every ${(POLL_MS / 1000).toFixed(1)}s | Pipeline: ward_controller + receiver`;
  </script>
</body>
</html>
    """
    return make_response(html)


@app.route("/api/state")
def api_state():
    return jsonify(_read_state())


@app.route("/api/devices")
def api_devices():
    return jsonify(_read_devices())


@app.route("/api/telemetry")
def api_telemetry():
    return jsonify(_read_telemetry())


@app.route("/api/commands")
def api_commands():
    return jsonify(_read_commands())


@app.route("/api/snapshot")
def api_snapshot():
  return jsonify(
    {
      "state": _read_state(),
      "devices": _read_devices(),
      "telemetry": _read_telemetry(),
      "commands": _read_commands(),
    }
  )


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
