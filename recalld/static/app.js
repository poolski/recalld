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

let statusPopoverHideTimer = null;

document.addEventListener("pointerover", (event) => {
  if (event.target.closest("#status-popover")) {
    clearHideStatusPopover();
    return;
  }
  const button = event.target.closest("[data-status-kind]");
  if (!button) return;
  void openStatusPopover(button);
});

document.addEventListener("pointerout", (event) => {
  if (event.target.closest("#status-popover")) {
    const related = event.relatedTarget;
    if (related && related.closest("#status-popover")) return;
    scheduleHideStatusPopover();
    return;
  }
  const button = event.target.closest("[data-status-kind]");
  if (!button) return;
  const related = event.relatedTarget;
  if (related && (button.contains(related) || related.closest("#status-popover"))) return;
  scheduleHideStatusPopover();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideStatusPopover();
});

async function openStatusPopover(button) {
  const popover = document.getElementById("status-popover");
  if (!popover) return;

  const kind = button.dataset.statusKind;
  if (!kind) return;

  clearHideStatusPopover();

  if (!popover.hidden && popover.dataset.kind === kind) {
    hideStatusPopover();
    return;
  }

  popover.dataset.kind = kind;
  popover.innerHTML = "";
  popover.hidden = false;
  renderStatusPopoverSkeleton(popover, button.dataset.statusTitle || "Status");
  positionStatusPopover(button, popover);

  try {
    const resp = await fetch(`/settings/status/details?kind=${encodeURIComponent(kind)}`, {
      headers: { "Accept": "application/json" },
    });
    if (!resp.ok) {
      renderStatusPopoverError(popover, "Unable to load status details.");
      return;
    }
    const data = await resp.json();
    renderStatusPopover(popover, button, data);
  } catch (_) {
    renderStatusPopoverError(popover, "Unable to load status details.");
  }
}

function scheduleHideStatusPopover() {
  clearHideStatusPopover();
  statusPopoverHideTimer = window.setTimeout(() => {
    hideStatusPopover();
  }, 160);
}

function clearHideStatusPopover() {
  if (statusPopoverHideTimer !== null) {
    window.clearTimeout(statusPopoverHideTimer);
    statusPopoverHideTimer = null;
  }
}

function positionStatusPopover(button, popover) {
  const rect = button.getBoundingClientRect();
  const width = 320;
  const top = Math.round(rect.bottom + 10);
  const left = Math.min(
    Math.max(12, Math.round(rect.right - width)),
    Math.max(12, window.innerWidth - width - 12),
  );
  popover.style.top = `${top}px`;
  popover.style.left = `${left}px`;
  popover.style.width = `${width}px`;
}

function hideStatusPopover() {
  clearHideStatusPopover();
  const popover = document.getElementById("status-popover");
  if (!popover) return;
  popover.hidden = true;
  popover.dataset.kind = "";
  popover.innerHTML = "";
}

function renderStatusPopoverSkeleton(popover, title) {
  popover.innerHTML = "";
  const heading = document.createElement("div");
  heading.className = "status-popover-title";
  heading.textContent = title;
  const body = document.createElement("div");
  body.className = "status-popover-empty";
  body.textContent = "Loading details…";
  popover.append(heading, body);
}

function renderStatusPopoverError(popover, message) {
  popover.innerHTML = "";
  const heading = document.createElement("div");
  heading.className = "status-popover-title";
  heading.textContent = "Status";
  const body = document.createElement("div");
  body.className = "status-popover-empty";
  body.textContent = message;
  popover.append(heading, body);
}

function renderStatusPopover(popover, button, data) {
  popover.innerHTML = "";
  const heading = document.createElement("div");
  heading.className = "status-popover-title";
  heading.textContent = data.title || button.dataset.statusTitle || "Status";
  popover.appendChild(heading);

  const list = document.createElement("dl");
  list.className = "status-popover-list";
  (data.items || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "status-popover-row";

    const label = document.createElement("dt");
    label.className = "status-popover-label";
    label.textContent = item.label;

    const value = document.createElement("dd");
    value.className = "status-popover-value";
    value.textContent = item.value;

    row.append(label, value);
    list.appendChild(row);
  });
  popover.appendChild(list);
  popover.hidden = false;
  positionStatusPopover(button, popover);
}

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
            can_confirm_vault, can_confirm_speakers, can_swap_speakers, can_confirm_themes, can_skip_themes,
            vault_preview, filename, themes,
            vault_conflict_path, can_overwrite_vault_note, can_append_vault_note } = event;

    updateStage(stage, status, message);

    if (preview) showPreview(preview);
    if (topic_count) showChunkInfo(topic_count, strategy);
    if (summary !== undefined && summary !== null) showPartialSummary(summary);
    if (focus_points || obsidian_uri) showResults(obsidian_uri, summary, focus_points);
    if (can_skip) showDiariseSkip(stage);
    if (can_write_transcript_only) showPostprocessFallback(stage);
    if (can_confirm_vault) showVaultConfirm(stage, filename);
    if (vault_conflict_path || can_overwrite_vault_note || can_append_vault_note) showVaultConflict(vault_conflict_path);
    if (vault_preview) showVaultPreview(vault_preview);
    if (can_confirm_speakers || can_swap_speakers) showSpeakerConfirm(stage);
    if (can_confirm_themes) showThemeConfirm(stage, themes || []);
    if (can_skip_themes) showThemeFallback(stage);

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
    if (state.vault_conflict_path || state.can_overwrite_vault_note || state.can_append_vault_note) showVaultConflict(state.vault_conflict_path);
    if (state.vault_preview) showVaultPreview(state.vault_preview);
    if (state.can_confirm_speakers || state.can_swap_speakers) showSpeakerConfirm("align");
    if (state.can_confirm_themes) showThemeConfirm("themes", state.themes || []);
    if (state.can_skip_themes) showThemeFallback("themes");
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
  if (stageStatuses.themes === "awaiting_confirmation") showThemeConfirm("themes");
  if (stageStatuses.themes === "failed") showThemeFallback("themes");
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
  updateStageRunningIndicator(stage, status);
  if (status !== "awaiting_confirmation") disableStageConfirmation(stage);
  if (msg) {
    if (message && (status === "failed" || status === "awaiting_confirmation")) {
      msg.textContent = message;
    } else {
      msg.textContent = "";
    }
  }

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
  if (stage === "themes") {
    const form = document.getElementById("theme-confirm-form");
    const editor = document.getElementById("theme-editor");
    const skipBtn = document.getElementById("theme-skip-btn");
    if (editor) editor.innerHTML = "";
    if (form) form.style.display = "none";
    if (skipBtn) skipBtn.style.display = "none";
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

const STAGE_ORDER = ["ingest", "transcribe", "diarise", "align", "themes", "postprocess", "vault"];

function resetPipelineFromStage(stage) {
  const startIndex = STAGE_ORDER.indexOf(stage);
  if (startIndex === -1) return;

  for (let i = startIndex; i < STAGE_ORDER.length; i += 1) {
    const s = STAGE_ORDER[i];
    updateStage(s, "pending", "");
    clearStageResults(s);
    clearStageLog(s);
    disableStageConfirmation(s);
  }
}

function clearStageLog(stage) {
  const el = document.getElementById(`stage-log-${stage}`);
  if (!el) return;
  el.textContent = "";
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

function showThemeConfirm(stage, themes = []) {
  const form = document.getElementById("theme-confirm-form");
  const editor = document.getElementById("theme-editor");
  const skipBtn = document.getElementById("theme-skip-btn");
  const addBtn = document.getElementById("theme-add-btn");
  if (form) form.style.display = "block";
  if (skipBtn) skipBtn.style.display = "none";
  if (editor) renderThemeEditor(editor, themes);
  if (addBtn) {
    addBtn.onclick = () => {
      if (!editor) return;
      appendThemeRow(editor, {
        id: `custom-${Date.now()}`,
        title: "",
        notes: "",
        enabled: true,
        order: editor.querySelectorAll("[data-theme-row]").length + 1,
        source: "manual",
      });
    };
  }
  setStageExpanded("themes", true);
}

function showThemeFallback(stage) {
  const skipBtn = document.getElementById("theme-skip-btn");
  const form = document.getElementById("theme-confirm-form");
  if (form) form.style.display = "none";
  if (skipBtn) skipBtn.style.display = "inline-flex";
  setStageExpanded("themes", true);
}

function renderThemeEditor(editor, themes = []) {
  editor.innerHTML = "";
  const normalized = Array.isArray(themes) ? themes : [];
  if (normalized.length === 0) {
    appendThemeRow(editor, {
      id: `theme-${Date.now()}`,
      title: "",
      notes: "",
      enabled: true,
      order: 1,
      source: "transcript",
    });
  } else {
    normalized.forEach((theme, index) => {
      appendThemeRow(editor, {
        id: theme.id || `theme-${index + 1}`,
        title: theme.title || "",
        notes: theme.notes || "",
        enabled: theme.enabled !== false,
        order: theme.order || (index + 1),
        source: theme.source || "transcript",
      });
    });
  }
}

function appendThemeRow(editor, theme) {
  const row = document.createElement("div");
  row.className = "theme-row";
  row.dataset.themeRow = "true";
  row.dataset.themeId = theme.id;
  const isTranscriptTheme = theme.source !== "manual";

  row.innerHTML = `
    <button type="button" class="theme-drag-handle" aria-label="Drag to reorder" draggable="true">⋮⋮</button>
    <div class="theme-row-body">
      <div class="theme-row-head">
        <input type="hidden" name="theme_id" value="${escapeHtml(theme.id)}">
        <input type="text" name="theme_title" class="theme-title-input" value="${escapeHtml(theme.title)}" placeholder="Theme heading">
        <label class="theme-enabled">
          <input type="checkbox" name="theme_enabled" value="${escapeHtml(theme.id)}" ${theme.enabled ? "checked" : ""}>
          Use
        </label>
      </div>
      <textarea name="theme_notes" class="theme-notes-input" rows="2" placeholder="Short note or cue">${escapeHtml(theme.notes)}</textarea>
      <div class="theme-row-meta" ${isTranscriptTheme ? "" : 'style="display:none"'}>Suggested from transcript</div>
    </div>
    <div class="theme-row-actions">
      <button type="button" class="btn-ghost theme-move-up" onclick="moveThemeRow(this, -1)">↑</button>
      <button type="button" class="btn-ghost theme-move-down" onclick="moveThemeRow(this, 1)">↓</button>
      <button type="button" class="btn-ghost theme-remove" onclick="removeThemeRow(this)">Remove</button>
    </div>
  `;

  const handle = row.querySelector(".theme-drag-handle");
  const dropTargets = [handle, row];

  handle.addEventListener("dragstart", (event) => {
    event.dataTransfer.effectAllowed = "move";
    row.classList.add("dragging");
    window.__draggedThemeRow = row;
  });
  handle.addEventListener("dragend", () => {
    row.classList.remove("dragging");
    window.__draggedThemeRow = null;
  });
  dropTargets.forEach((target) => {
    target.addEventListener("dragover", (event) => {
      event.preventDefault();
    });
    target.addEventListener("drop", (event) => {
      event.preventDefault();
      const dragged = window.__draggedThemeRow;
      if (!dragged || dragged === row || !dragged.parentElement) return;
      const rect = row.getBoundingClientRect();
      const before = event.clientY < rect.top + rect.height / 2;
      row.parentElement.insertBefore(dragged, before ? row : row.nextSibling);
    });
  });

  editor.appendChild(row);
  return row;
}

function moveThemeRow(button, direction) {
  const row = button.closest("[data-theme-row]");
  if (!row || !row.parentElement) return;
  const sibling = direction < 0 ? row.previousElementSibling : row.nextElementSibling;
  if (!sibling) return;
  if (direction < 0) {
    row.parentElement.insertBefore(row, sibling);
  } else {
    row.parentElement.insertBefore(sibling, row);
  }
}

function removeThemeRow(button) {
  const row = button.closest("[data-theme-row]");
  const editor = document.getElementById("theme-editor");
  if (!row || !editor) return;
  row.remove();
  if (!editor.querySelector("[data-theme-row]")) {
    appendThemeRow(editor, {
      id: `theme-${Date.now()}`,
      title: "",
      notes: "",
      enabled: true,
      order: 1,
      source: "manual",
    });
  }
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
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

function showVaultConflict(path) {
  const box = document.getElementById("vault-conflict-box");
  const pathEl = document.getElementById("vault-conflict-path");
  if (pathEl && path) pathEl.textContent = path;
  if (box) box.style.display = "block";
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
  if (stage === "themes") {
    const form = document.getElementById("theme-confirm-form");
    const skipBtn = document.getElementById("theme-skip-btn");
    const addBtn = document.getElementById("theme-add-btn");
    if (form) form.style.display = "none";
    if (skipBtn) skipBtn.style.display = "none";
    if (addBtn) addBtn.onclick = null;
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
    const conflict = document.getElementById("vault-conflict-box");
    if (controls) controls.style.display = "none";
    if (conflict) conflict.style.display = "none";
    if (el) {
      el.style.display = "none";
      el.disabled = true;
    }
  }
}

function updateStageRunningIndicator(stage, status) {
  const el = document.getElementById(`stage-running-${stage}`);
  if (!el) return;
  el.style.display = status === "running" ? "inline-flex" : "none";
}

function appendStageLog(stage, msg) {
  const normalized = (msg || "").trim();
  if (/^\[[^\]]+\]\s+running$/.test(normalized)) return;
  const el = document.getElementById(`stage-log-${stage}`);
  if (!el) return;
  el.textContent += normalized + "\n";
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
