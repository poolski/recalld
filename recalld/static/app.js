// Drag-and-drop for upload zone
document.addEventListener("DOMContentLoaded", () => {
  const zone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const form = document.getElementById("upload-form");

  if (!zone) return;

  zone.addEventListener("click", () => fileInput.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      updateZoneLabel(files[0].name);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) updateZoneLabel(fileInput.files[0].name);
  });

  function updateZoneLabel(name) {
    const p = zone.querySelector("p");
    if (p) p.textContent = name;
  }
});

// SSE-based stage progress updates
function connectSSE(jobId) {
  const evtSource = new EventSource(`/jobs/${jobId}/events`);
  const stages = ["ingest", "transcribe", "diarise", "align", "postprocess", "vault"];

  evtSource.onmessage = (e) => {
    if (e.data === '"done"') { evtSource.close(); return; }
    const event = JSON.parse(e.data);
    const { stage, status, message, preview, topic_count, strategy,
            obsidian_uri, summary, focus_points, can_skip, can_write_transcript_only } = event;

    updateStage(stage, status, message);

    if (preview) showPreview(preview);
    if (topic_count) showChunkInfo(topic_count, strategy);
    if (obsidian_uri) showResults(obsidian_uri, summary, focus_points);
    if (can_skip) showDiariseSkip(stage);
    if (can_write_transcript_only) showPostprocessFallback(stage);

    appendLog(`[${stage}] ${status}${message ? ': ' + message : ''}`);
  };
}

function updateStage(stage, status, message) {
  const el = document.getElementById(`stage-${stage}`);
  if (!el) return;
  const icon = el.querySelector(".stage-icon");
  const msg = el.querySelector(".stage-msg");
  if (status === "running") icon.innerHTML = '<span class="spinner"></span>';
  if (status === "done") icon.textContent = "✓";
  if (status === "failed") icon.textContent = "✗";
  if (msg && message) msg.textContent = message;
}

function showPreview(text) {
  const el = document.getElementById("transcript-preview");
  if (el) { el.textContent = text; el.style.display = "block"; }
}

function showChunkInfo(count, strategy) {
  const el = document.getElementById("chunk-info");
  if (el) {
    el.textContent = strategy === "map_reduce"
      ? `Detected ${count} topics — summarising in sections`
      : "Summarising full transcript";
    el.style.display = "block";
  }
}

function showResults(uri, summary, focusPoints) {
  const resultsEl = document.getElementById("results-section");
  if (!resultsEl) return;
  const summaryEl = document.getElementById("result-summary");
  const focusEl = document.getElementById("result-focus");
  const linkEl = document.getElementById("obsidian-link");
  if (summaryEl) summaryEl.textContent = summary;
  if (focusEl) focusEl.innerHTML = focusPoints.map(p => `<li>${p}</li>`).join("");
  if (linkEl) linkEl.href = uri;
  resultsEl.style.display = "block";
}

function showDiariseSkip(stage) {
  const el = document.getElementById("diarise-skip-btn");
  if (el) el.style.display = "inline-block";
}

function showPostprocessFallback(stage) {
  const el = document.getElementById("postprocess-fallback-btn");
  if (el) el.style.display = "inline-block";
}

function appendLog(msg) {
  const el = document.getElementById("debug-log");
  if (!el) return;
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

function copyLog() {
  const el = document.getElementById("debug-log");
  if (el) navigator.clipboard.writeText(el.textContent);
}

function saveLog() {
  const el = document.getElementById("debug-log");
  if (!el) return;
  const blob = new Blob([el.textContent], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "recalld-log.txt";
  a.click();
}
