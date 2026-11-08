(() => {
  const params = new URLSearchParams(window.location.search);
  const configuredApi = params.get("api") || localStorage.getItem("epishield-api-base") || "";
  const API_BASE = configuredApi.replace(/\/$/, "");
  const POLL_INTERVAL_MS = 2000;
  const els = {
    runButton: document.getElementById("run-button"),
    runButtonLabel: document.getElementById("run-button-label"),
    refreshButton: document.getElementById("refresh-button"),
    statusText: document.getElementById("status-text"),
    statusDot: document.getElementById("status-dot"),
    statusCaption: document.getElementById("status-caption"),
    rowsRead: document.getElementById("rows-read"),
    rowsLoaded: document.getElementById("rows-loaded"),
    lastUpdated: document.getElementById("last-updated"),
    lastExecution: document.getElementById("last-execution"),
    lastExecutionCaption: document.getElementById("last-execution-caption"),
    sourceCounts: document.getElementById("source-counts"),
    tableCounts: document.getElementById("table-counts"),
    sourceTotal: document.getElementById("source-total"),
    tableTotal: document.getElementById("table-total"),
    runIdLabel: document.getElementById("run-id-label"),
    activityDot: document.getElementById("activity-dot"),
    activityTitle: document.getElementById("activity-title"),
    activityDescription: document.getElementById("activity-description"),
    activityTime: document.getElementById("activity-time"),
    activityLive: document.getElementById("activity-live"),
    errorBox: document.getElementById("error-box"),
    errorMessage: document.getElementById("error-message"),
    connectionLabel: document.getElementById("connection-label"),
    systemDot: document.querySelector(".system-dot"),
    apiOrigin: document.getElementById("api-origin"),
    toast: document.getElementById("toast"),
  };
  let pollingTimer = null;
  let toastTimer = null;
  let lastState = null;

  els.apiOrigin.textContent = API_BASE || "this server";

  function number(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return new Intl.NumberFormat().format(Number(value));
  }

  function date(value, options = {}) {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return null;
    return new Intl.DateTimeFormat(undefined, options).format(parsed);
  }

  function relative(value) {
    if (!value) return null;
    const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
    if (!Number.isFinite(seconds)) return null;
    if (seconds < 10) return "Just now";
    if (seconds < 60) return `${seconds} seconds ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hr ago`;
    return date(value, { month: "short", day: "numeric", year: "numeric" });
  }

  function showToast(message, isError = false) {
    els.toast.textContent = message;
    els.toast.classList.toggle("error", isError);
    els.toast.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => els.toast.classList.remove("show"), 3500);
  }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = payload && typeof payload === "object" ? payload.detail : payload;
      const message = typeof detail === "string" ? detail : detail?.message || `Request failed (${response.status})`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function renderCounts(container, values, totalElement) {
    container.replaceChildren();
    if (!values || typeof values !== "object" || !Object.keys(values).length) {
      const empty = document.createElement("div");
      empty.className = "empty-row";
      empty.textContent = "Counts will appear after a successful run.";
      container.append(empty);
      totalElement.textContent = "—";
      return;
    }
    const entries = Object.entries(values);
    const total = entries.reduce((sum, [, value]) => sum + (Number(value) || 0), 0);
    totalElement.textContent = number(total);
    for (const [label, value] of entries) {
      const row = document.createElement("div");
      row.className = "count-row";
      const name = document.createElement("span");
      name.textContent = label;
      const count = document.createElement("strong");
      count.textContent = number(value);
      row.append(name, count);
      container.append(row);
    }
  }

  function setStatus(status) {
    const normalized = ["idle", "running", "success", "failed"].includes(status) ? status : "idle";
    els.statusText.textContent = normalized.charAt(0).toUpperCase() + normalized.slice(1);
    els.statusDot.className = `status-dot ${normalized}`;
    els.runButton.disabled = normalized === "running" || normalized === "success";
    els.runButtonLabel.textContent = normalized === "running" ? "Pipeline is running…" : normalized === "success" ? "Pipeline completed" : "Run data pipeline";
    els.statusCaption.textContent = {
      idle: "Ready when you are",
      running: "Processing your data now",
      success: "This database has already completed a run",
      failed: "The latest run needs attention",
    }[normalized];
    els.activityLive.classList.toggle("polling", normalized === "running");
    if (normalized === "running") {
      els.activityTitle.textContent = "Pipeline is running";
      els.activityDescription.textContent = "Extracting, transforming, and loading your data.";
      els.activityDot.className = "timeline-dot current";
    } else if (normalized === "success") {
      els.activityTitle.textContent = "Pipeline completed";
      els.activityDescription.textContent = "Your data was processed and loaded successfully.";
      els.activityDot.className = "timeline-dot current";
    } else if (normalized === "failed") {
      els.activityTitle.textContent = "Pipeline failed";
      els.activityDescription.textContent = "The pipeline stopped before completing.";
      els.activityDot.className = "timeline-dot failed";
    } else {
      els.activityTitle.textContent = "Ready to run";
      els.activityDescription.textContent = "Start the pipeline to process your latest data.";
      els.activityDot.className = "timeline-dot";
    }
  }

  function render(state) {
    if (!state || typeof state !== "object") throw new Error("The backend returned an unexpected status response.");
    lastState = state;
    const status = String(state.status || "idle").toLowerCase();
    setStatus(status);
    const updatedAt = state.completed_at || state.started_at;
    els.lastUpdated.textContent = relative(updatedAt) || "Not yet";
    els.lastExecution.textContent = state.started_at ? (date(state.started_at, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) || "—") : "No runs yet";
    els.lastExecutionCaption.textContent = state.completed_at ? `Finished ${relative(state.completed_at)}` : state.started_at ? `Started ${relative(state.started_at)}` : "Waiting for the first run";
    els.activityTime.textContent = updatedAt ? date(updatedAt, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) || "—" : "No activity yet";
    els.runIdLabel.textContent = state.run_id ? `RUN · ${String(state.run_id).slice(0, 8).toUpperCase()}` : "LATEST RUN";
    els.rowsRead.textContent = number(state.total_rows_read);
    els.rowsLoaded.textContent = number(state.total_rows_loaded);
    renderCounts(els.sourceCounts, state.rows_read_by_file, els.sourceTotal);
    renderCounts(els.tableCounts, state.rows_loaded_by_table, els.tableTotal);
    const error = status === "failed" ? state.error : null;
    els.errorBox.hidden = !error;
    els.errorMessage.textContent = error || "";
    if (status === "running") startPolling(); else stopPolling();
  }

  function startPolling() {
    if (pollingTimer) return;
    pollingTimer = window.setInterval(() => loadStatus({ quiet: true }), POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (!pollingTimer) return;
    window.clearInterval(pollingTimer);
    pollingTimer = null;
  }

  async function loadStatus({ quiet = false } = {}) {
    try {
      const state = await request("/pipeline/status");
      render(state);
      els.systemDot.className = "system-dot online";
      els.connectionLabel.textContent = "Backend connected";
      return state;
    } catch (error) {
      els.systemDot.className = "system-dot offline";
      els.connectionLabel.textContent = "Backend unavailable";
      if (!quiet) showToast(`Could not load pipeline status: ${error.message}`, true);
      throw error;
    }
  }

  async function runPipeline() {
    els.runButton.disabled = true;
    els.runButtonLabel.textContent = "Starting pipeline…";
    try {
      const state = await request("/pipeline/run", { method: "POST" });
      render(state);
      showToast("Pipeline started. Status will refresh automatically.");
      await loadStatus({ quiet: true });
    } catch (error) {
      if (error.status === 409) {
        try {
          const latest = await loadStatus({ quiet: true });
          render(latest);
        } catch { /* Keep the request error visible if the status endpoint is unavailable too. */ }
      } else {
        setStatus(lastState?.status || "idle");
      }
      showToast(error.message, true);
    }
  }

  els.runButton.addEventListener("click", runPipeline);
  els.refreshButton.addEventListener("click", () => loadStatus());
  loadStatus().catch(() => {});
})();
