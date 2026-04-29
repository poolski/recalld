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
async function connectSSE(jobId, initialStages = {}) {
  if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true });
  }
  applyStageStatuses(initialStages);
  await hydrateJobState(jobId);
  const evtSource = new EventSource(`/jobs/${jobId}/events`);

  evtSource.onmessage = (e) => {
    if (e.data === '"done"') { evtSource.close(); return; }
    const event = JSON.parse(e.data);
    const { stage, status, message, preview, topic_count, strategy,
            obsidian_uri, summary, focus_points, can_skip, can_write_transcript_only,
            can_confirm_vault, can_confirm_speakers, can_swap_speakers, vault_preview, filename } = event;

    updateStage(stage, status, message);

    if (preview) showPreview(preview);
    if (topic_count) showChunkInfo(topic_count, strategy);
    if (summary !== undefined && summary !== null) showPartialSummary(summary);
    if (focus_points || obsidian_uri) showResults(obsidian_uri, summary, focus_points);
    if (can_skip) showDiariseSkip(stage);
    if (can_write_transcript_only) showPostprocessFallback(stage);
    if (can_confirm_vault) showVaultConfirm(stage, filename);
    if (vault_preview) showVaultPreview(vault_preview);
    if (can_confirm_speakers || can_swap_speakers) showSpeakerConfirm(stage);

    appendStageLog(stage, `[${stage}] ${status}${message ? ': ' + message : ''}`);
  };
}

async function hydrateJobState(jobId) {
  try {
    const resp = await fetch(`/jobs/${jobId}/state`, {
      headers: { "Accept": "application/json" },
    });
    if (!resp.ok) return;
    const state = await resp.json();
    applyStageStatuses(state.stage_statuses || {});
    if (state.preview) showPreview(state.preview);
    if (state.topic_count) showChunkInfo(state.topic_count, state.strategy);
    if (state.summary !== undefined && state.summary !== null) showPartialSummary(state.summary);
    if (state.focus_points || state.obsidian_uri) showResults(state.obsidian_uri, state.summary || "", state.focus_points || []);
    if (state.error) updateStage(state.current_stage, state.stage_statuses[state.current_stage], state.error);
    if (state.can_confirm_vault) showVaultConfirm("vault", state.filename);
    if (state.vault_preview) showVaultPreview(state.vault_preview);
    if (state.can_confirm_speakers || state.can_swap_speakers) showSpeakerConfirm("align");
  } catch (_) {
    // Fall back to template-provided state if the refresh request fails.
  }
}

function applyStageStatuses(stageStatuses) {
  Object.entries(stageStatuses).forEach(([stage, status]) => {
    if (status && status !== "pending") updateStage(stage, status, "");
  });
  if (stageStatuses.vault === "awaiting_confirmation") showVaultConfirm("vault");
  if (stageStatuses.align === "awaiting_confirmation") showSpeakerConfirm("align");
}

function updateStage(stage, status, message) {
  const el = document.getElementById(`stage-${stage}`);
  if (!el) return;
  const pill = document.getElementById(`stage-pill-${stage}`);
  const msg = el.querySelector(".stage-msg");
  const header = el.querySelector(".stage-toggle");
  if (pill) {
    pill.textContent = status.replaceAll("_", " ");
    pill.className = `stage-pill status-${status}`;
  }
  if (status !== "awaiting_confirmation") disableStageConfirmation(stage);
  if (msg && message) msg.textContent = message;

  if (header && (status === "running" || status === "failed" || status === "awaiting_confirmation")) {
    setStageExpanded(stage, true);
  }
}

function clearStageResults(stage) {
  if (stage === "postprocess") {
    const summaryEl = document.getElementById("result-summary");
    const focusEl = document.getElementById("result-focus");
    if (summaryEl) summaryEl.innerHTML = "";
    if (focusEl) focusEl.innerHTML = "";
    const vaultPreviewEl = document.getElementById("vault-preview");
    if (vaultPreviewEl) {
      vaultPreviewEl.innerHTML = "";
      vaultPreviewEl.style.display = "none";
    }
  }
  if (stage === "align") {
    const previewEl = document.getElementById("align-preview");
    if (previewEl) {
      previewEl.innerHTML = "";
      previewEl.style.display = "none";
    }
  }
  if (stage === "vault") {
    const previewEl = document.getElementById("vault-preview");
    if (previewEl) {
      previewEl.innerHTML = "";
      previewEl.style.display = "none";
    }
  }
}

function showPreview(text) {
  const el = document.getElementById("align-preview");
  if (el) { el.innerHTML = marked.parse(text); el.style.display = "block"; }
  setStageExpanded("align", true);
}

function showChunkInfo(count, strategy) {
  const el = document.getElementById("postprocess-chunk-info");
  if (el) {
    el.textContent = strategy === "map_reduce"
      ? `Detected ${count} topics — summarising in sections`
      : "Summarising full transcript";
    el.style.display = "block";
  }
  setStageExpanded("postprocess", true);
}

function showPartialSummary(summary) {
  const resultsEl = document.getElementById("postprocess-results");
  const summaryEl = document.getElementById("result-summary");
  if (summaryEl) summaryEl.innerHTML = marked.parse(summary);
  if (resultsEl) resultsEl.style.display = "block";
  setStageExpanded("postprocess", true);
}

function showResults(uri, summary, focusPoints) {
  const resultsEl = document.getElementById("postprocess-results");
  if (!resultsEl) return;
  const summaryEl = document.getElementById("result-summary");
  const focusEl = document.getElementById("result-focus");
  const linkEl = document.getElementById("obsidian-link");
  if (summary && summaryEl) summaryEl.innerHTML = marked.parse(summary);
  if (focusPoints && focusEl) focusEl.innerHTML = focusPoints.map(p => `<li>${marked.parseInline(p)}</li>`).join("");
  if (linkEl) {
    if (uri) {
      linkEl.href = uri;
      linkEl.style.display = "inline-block";
    } else {
      linkEl.removeAttribute("href");
      linkEl.style.display = "none";
    }
  }
  resultsEl.style.display = "block";
  setStageExpanded("postprocess", true);
  if (uri) setStageExpanded("vault", true);
}

function showDiariseSkip(stage) {
  const el = document.getElementById("diarise-skip-btn");
  if (el) el.style.display = "inline-block";
  setStageExpanded("diarise", true);
}

function showPostprocessFallback(stage) {
  const el = document.getElementById("postprocess-fallback-btn");
  if (el) el.style.display = "inline-block";
  setStageExpanded("postprocess", true);
}

function showVaultConfirm(stage, filename) {
  const el = document.getElementById("vault-confirm-btn");
  const controls = document.getElementById("vault-confirm-controls");
  const input = document.getElementById("vault-filename");
  if (input && filename) input.value = filename;
  if (controls) controls.style.display = "block";
  if (el) {
    el.style.display = "inline-block";
    el.disabled = false;
  }
  setStageExpanded("vault", true);
}

function showVaultPreview(text) {
  const el = document.getElementById("vault-preview");
  if (!el || !text) return;
  el.innerHTML = marked.parse(text);
  el.style.display = "block";
  setStageExpanded("vault", true);
}

function showSpeakerConfirm(stage) {
  const el = document.getElementById("speaker-confirm-controls");
  if (el) {
    el.style.display = "flex";
    const confirm = document.getElementById("speaker-confirm-btn");
    const swap = document.getElementById("speaker-swap-btn");
    if (confirm) confirm.disabled = false;
    if (swap) swap.disabled = false;
  }
  setStageExpanded("align", true);
}

function disableStageConfirmation(stage) {
  if (stage === "diarise") {
    const el = document.getElementById("diarise-skip-btn");
    if (el) el.style.display = "none";
  }
  if (stage === "postprocess") {
    const el = document.getElementById("postprocess-fallback-btn");
    if (el) el.style.display = "none";
  }
  if (stage === "align") {
    const controls = document.getElementById("speaker-confirm-controls");
    const confirm = document.getElementById("speaker-confirm-btn");
    const swap = document.getElementById("speaker-swap-btn");
    if (controls) controls.style.display = "none";
    if (confirm) confirm.disabled = true;
    if (swap) swap.disabled = true;
  }
  if (stage === "vault") {
    const controls = document.getElementById("vault-confirm-controls");
    const el = document.getElementById("vault-confirm-btn");
    if (controls) controls.style.display = "none";
    if (el) {
      el.style.display = "none";
      el.disabled = true;
    }
  }
}

function appendStageLog(stage, msg) {
  const el = document.getElementById(`stage-log-${stage}`);
  if (!el) return;
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

function toggleStage(stage) {
  const body = document.getElementById(`stage-body-${stage}`);
  const header = document.querySelector(`#stage-${stage} .stage-toggle`);
  if (!body || !header) return;
  setStageExpanded(stage, body.hidden);
}

function setStageExpanded(stage, expanded) {
  const body = document.getElementById(`stage-body-${stage}`);
  const header = document.querySelector(`#stage-${stage} .stage-toggle`);
  if (!body || !header) return;
  body.hidden = !expanded;
  header.classList.toggle("expanded", expanded);
  header.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function copyLog() {
  // no-op: per-stage logs live inside the stage accordions
}

function saveLog() {
  // no-op: per-stage logs live inside the stage accordions
}
