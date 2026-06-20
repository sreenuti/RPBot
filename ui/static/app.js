const state = {
  records: [],
  filename: null,
  selectedIndex: 0,
  runResult: null,
};

const PHASE_ICONS = {
  ingest: "📥",
  prompt: "📝",
  llm: "🧠",
  parse: "🔍",
  validate: "✅",
  retry: "🔄",
  evaluate: "📊",
  threshold: "🎯",
  complete: "🏁",
  error: "❌",
};

const els = {
  mockToggle: document.getElementById("mockToggle"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  runBtn: document.getElementById("runBtn"),
  uploadZone: document.getElementById("uploadZone"),
  fileInput: document.getElementById("fileInput"),
  recordList: document.getElementById("recordList"),
  inputCount: document.getElementById("inputCount"),
  timeline: document.getElementById("timeline"),
  pipelineEmpty: document.getElementById("pipelineEmpty"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  previewEmpty: document.getElementById("previewEmpty"),
  previewContent: document.getElementById("previewContent"),
  decisionBanner: document.getElementById("decisionBanner"),
  messagePreview: document.getElementById("messagePreview"),
  reasoningText: document.getElementById("reasoningText"),
  qualityGrid: document.getElementById("qualityGrid"),
  outputJson: document.getElementById("outputJson"),
  inputJson: document.getElementById("inputJson"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  loadingSub: document.getElementById("loadingSub"),
  toastContainer: document.getElementById("toastContainer"),
  metricRecords: document.getElementById("metricRecords"),
  metricSent: document.getElementById("metricSent"),
  metricSuppressed: document.getElementById("metricSuppressed"),
  metricPersonalization: document.getElementById("metricPersonalization"),
  metricLatency: document.getElementById("metricLatency"),
  metricThreshold: document.getElementById("metricThreshold"),
};

function toast(message, type = "success") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  els.toastContainer.appendChild(node);
  setTimeout(() => node.remove(), 4000);
}

function setLoading(show, subtext) {
  els.loadingOverlay.hidden = !show;
  if (subtext) els.loadingSub.textContent = subtext;
}

async function fetchJson(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out. Is the server running on port 8080?");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function setRecords(records, filename) {
  state.records = records;
  state.filename = filename;
  state.selectedIndex = 0;
  state.runResult = null;
  els.runBtn.disabled = records.length === 0;
  els.inputCount.textContent = `${records.length} loaded`;
  renderRecordList();
  resetPipeline();
  resetPreview();
  resetMetrics();
}

function resetMetrics() {
  ["metricRecords", "metricSent", "metricSuppressed", "metricPersonalization", "metricLatency", "metricThreshold"].forEach((id) => {
    document.getElementById(id).textContent = "—";
  });
}

function resetPipeline() {
  els.pipelineEmpty.hidden = false;
  els.timeline.hidden = true;
  els.timeline.innerHTML = "";
  els.pipelineStatus.textContent = "Idle";
  els.pipelineStatus.classList.remove("running");
}

function resetPreview() {
  els.previewEmpty.hidden = false;
  els.previewContent.hidden = true;
}

function renderRecordList() {
  els.recordList.innerHTML = "";
  state.records.forEach((record, index) => {
    const li = document.createElement("li");
    li.className = `record-item${index === state.selectedIndex ? " active" : ""}`;
    li.innerHTML = `
      <div class="task-id">${escapeHtml(record.task_id)}</div>
      <div class="meta">${escapeHtml(record.persona || "—")} · ${escapeHtml(record.lifecycle_stage || "—")}</div>
    `;
    li.addEventListener("click", () => selectRecord(index));
    els.recordList.appendChild(li);
  });
}

function selectRecord(index) {
  state.selectedIndex = index;
  renderRecordList();
  if (state.runResult) {
    renderPipeline();
    renderPreview();
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatData(data) {
  if (!data || Object.keys(data).length === 0) return "";
  const copy = { ...data };
  if (copy.full_prompt && copy.full_prompt.length > 800) {
    copy.full_prompt = copy.full_prompt.slice(0, 800) + "… [truncated in UI — expand via API for full prompt]";
  }
  return JSON.stringify(copy, null, 2);
}

function renderPipeline() {
  if (!state.runResult?.trace) return;

  const recordTrace = state.runResult.trace.records[state.selectedIndex];
  if (!recordTrace) return;

  els.pipelineEmpty.hidden = true;
  els.timeline.hidden = false;
  els.timeline.innerHTML = "";

  recordTrace.steps.forEach((step, i) => {
    const div = document.createElement("div");
    div.className = "timeline-step";
    div.style.animationDelay = `${i * 0.05}s`;

    const cot =
      step.phase === "llm" && step.data?.chain_of_thought
        ? `<div class="cot-highlight"><strong>Chain of thought:</strong> ${escapeHtml(step.data.chain_of_thought)}</div>`
        : "";

    const elapsed = step.elapsed_ms ? `<span class="step-elapsed">${step.elapsed_ms} ms</span>` : "";

    div.innerHTML = `
      <div class="step-icon ${step.status}">${PHASE_ICONS[step.phase] || "•"}</div>
      <div class="step-body">
        <div class="step-header">
          <span class="step-title">${escapeHtml(step.title)}</span>
          <span class="step-phase">${step.phase}</span>
        </div>
        ${elapsed}
        ${step.message ? `<p class="step-message">${escapeHtml(step.message)}</p>` : ""}
        ${cot}
        ${Object.keys(step.data || {}).length ? `<pre class="step-data">${escapeHtml(formatData(step.data))}</pre>` : ""}
      </div>
    `;
    els.timeline.appendChild(div);
  });
}

function renderPreview() {
  if (!state.runResult) return;

  const output = state.runResult.outputs[state.selectedIndex];
  const input = state.records[state.selectedIndex];
  const recordTrace = state.runResult.trace?.records[state.selectedIndex];

  if (!output) return;

  els.previewEmpty.hidden = true;
  els.previewContent.hidden = false;

  const send = output.should_send;
  els.decisionBanner.className = `decision-banner ${send ? "send" : "suppress"}`;
  els.decisionBanner.textContent = send
    ? `✓ Send via ${output.next_message.channel?.toUpperCase() || "?"} at ${output.next_message.send_at || "TBD"}`
    : "⊘ Communication suppressed";

  els.messagePreview.innerHTML = buildMessagePreview(output);
  els.reasoningText.textContent = output.reasoning || recordTrace?.steps?.find((s) => s.phase === "complete")?.message || "—";

  const q = output.quality || {};
  const thresholdStep = recordTrace?.steps?.find((s) => s.phase === "threshold");
  const thresholdPassed = thresholdStep?.data?.passed ?? true;

  els.qualityGrid.innerHTML = `
    <div class="quality-item"><span class="label">Latency</span><div class="value">${q.latency_ms ?? "—"} ms</div></div>
    <div class="quality-item ${q.personalization_score >= 0.8 ? "pass" : "fail"}"><span class="label">Personalization</span><div class="value">${q.personalization_score ?? "—"}</div></div>
    <div class="quality-item ${q.safety_violations === 0 ? "pass" : "fail"}"><span class="label">Safety violations</span><div class="value">${q.safety_violations ?? 0}</div></div>
    <div class="quality-item ${thresholdPassed ? "pass" : "fail"}"><span class="label">Thresholds</span><div class="value">${thresholdPassed ? "PASS" : "FAIL"}</div></div>
  `;

  els.outputJson.textContent = JSON.stringify(output, null, 2);
  els.inputJson.textContent = JSON.stringify(input, null, 2);
}

function buildMessagePreview(output) {
  const msg = output.next_message;
  if (!output.should_send || !msg.body) {
    return `<p style="color:var(--muted);font-size:0.85rem;text-align:center;padding:1rem;">No message — send suppressed</p>`;
  }

  if (msg.channel === "sms") {
    return `
      <div class="sms-phone">
        <div class="sms-header">SMS · ${escapeHtml(msg.send_at || "")}</div>
        <div class="sms-bubble">${escapeHtml(msg.body)}</div>
      </div>`;
  }

  if (msg.channel === "email") {
    return `
      <div class="email-card">
        <div class="email-header-bar">To: prospect · ${escapeHtml(msg.send_at || "")}</div>
        <div class="email-subject">${escapeHtml(msg.subject || "(no subject)")}</div>
        <div class="email-body">${escapeHtml(msg.body)}</div>
      </div>`;
  }

  return `<pre class="step-data">${escapeHtml(msg.body)}</pre>`;
}

function renderSummary() {
  const summary = state.runResult?.trace?.summary;
  if (!summary) return;

  els.metricRecords.textContent = summary.total_records;
  els.metricSent.textContent = summary.sent;
  els.metricSuppressed.textContent = summary.suppressed;
  els.metricPersonalization.textContent = summary.average_personalization_score.toFixed(2);
  els.metricLatency.textContent = `${Math.round(summary.average_latency_ms)} ms`;
  els.metricThreshold.textContent = `${Math.round(summary.threshold_pass_rate * 100)}%`;
}

async function loadSample({ blocking = false } = {}) {
  if (blocking) {
    setLoading(true, "Loading sample dataset…");
  } else {
    els.inputCount.textContent = "Loading sample…";
  }

  try {
    let data;
    try {
      data = await fetchJson("/static/sample-data.json", {}, 5000);
    } catch {
      data = await fetchJson("/api/sample", {}, 10000);
    }
    setRecords(data.records, data.filename);
    toast(`Loaded ${data.record_count} sample records`);
  } catch (err) {
    toast(err.message, "error");
    els.inputCount.textContent = "Load failed — click Load sample";
  } finally {
    if (blocking) setLoading(false);
  }
}

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".jsonl")) {
    toast("Please upload a .jsonl file", "error");
    return;
  }

  setLoading(true, "Parsing uploaded JSONL…");
  const form = new FormData();
  form.append("file", file);

  try {
    const data = await fetchJson("/api/upload", { method: "POST", body: form });
    setRecords(data.records, data.filename);
    toast(`Uploaded ${data.record_count} records from ${data.filename}`);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    setLoading(false);
  }
}

async function runAgent() {
  if (!state.records.length) return;

  setLoading(true, "Running autonomous agent pipeline…");
  els.pipelineStatus.textContent = "Running";
  els.pipelineStatus.classList.add("running");

  try {
    const data = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        records: state.records,
        mock: els.mockToggle.checked,
      }),
    }, 120000);

    state.runResult = data;
    els.pipelineStatus.textContent = "Complete";
    els.pipelineStatus.classList.remove("running");

    renderSummary();
    renderPipeline();
    renderPreview();
    toast(`Processed ${data.outputs.length} record(s) in ${data.trace?.total_latency_ms ?? "?"} ms`);
  } catch (err) {
    els.pipelineStatus.textContent = "Error";
    els.pipelineStatus.classList.remove("running");
    toast(err.message, "error");
  } finally {
    setLoading(false);
  }
}

els.loadSampleBtn.addEventListener("click", () => loadSample({ blocking: true }));
els.runBtn.addEventListener("click", runAgent);

els.uploadZone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});

els.uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.uploadZone.classList.add("dragover");
});
els.uploadZone.addEventListener("dragleave", () => els.uploadZone.classList.remove("dragover"));
els.uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

loadSample({ blocking: false });
