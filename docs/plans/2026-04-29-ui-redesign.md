# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernise the frontend with light/dark palette switching, a proper two-row header (brand + status pills + nav), surface cards on content areas, and polished controls.

**Architecture:** Pure CSS and Jinja2 template changes — no new routes, no backend changes. `style.css` gains a light-mode palette block and new component classes; `base.html` is restructured so the header contains both the status row and nav; four content templates gain `.card` wrappers.

**Tech Stack:** CSS custom properties, Jinja2, HTMX (unchanged)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `recalld/static/style.css` | Modify | Light palette, `--accent-rgb`, header styles, `.card`, control polish |
| `recalld/templates/base.html` | Modify | Two-row header; nav moved out of `<main>` |
| `recalld/templates/index.html` | Modify | Upload form + job list wrapped in `.card` |
| `recalld/templates/settings.html` | Modify | Settings form wrapped in `.card` |
| `recalld/templates/processing.html` | Modify | Stage list wrapped in `.card` |

---

## Task 1: Update `style.css` — palette, header, card, controls

**Files:**
- Modify: `recalld/static/style.css`

- [ ] **Step 1: Add `--accent-rgb` to the existing `:root` block**

In the `:root` block, add one new variable after `--accent`:

```css
--accent-rgb: 124, 106, 247;
```

The full `:root` block should now read:

```css
:root {
  --bg: #1a1a1a;
  --surface: #252525;
  --border: #333;
  --text: #e8e8e8;
  --muted: #888;
  --accent: #7c6af7;
  --accent-rgb: 124, 106, 247;
  --green: #4caf80;
  --red: #e06c75;
  --yellow: #e5c07b;
}
```

- [ ] **Step 2: Add light-mode palette block directly after `:root`**

```css
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f5f5f7;
    --surface: #ffffff;
    --border: #e0e0e0;
    --text: #1a1a1a;
    --muted: #6e6e73;
    --accent: #5b4fe8;
    --accent-rgb: 91, 79, 232;
  }
}
```

- [ ] **Step 3: Replace `.status-bar` with `.app-header` styles**

Remove the existing `.status-bar` rule block:

```css
.status-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 24px;
  display: flex;
  gap: 24px;
  font-size: 13px;
  align-items: center;
}
```

Replace with:

```css
.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-row {
  max-width: 720px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  align-items: center;
}

.header-top {
  height: 48px;
  justify-content: space-between;
}

.brand {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text);
}

.header-nav {
  height: 36px;
  border-top: 1px solid var(--border);
}
```

- [ ] **Step 4: Update `.status-indicator` for pill style**

Replace the existing `.status-indicator` rule:

```css
.status-indicator { display: flex; align-items: center; gap: 6px; cursor: pointer; }
```

With:

```css
.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 12px;
  margin-left: 6px;
}
.status-indicator:first-child { margin-left: 0; }
```

- [ ] **Step 5: Update `nav` styles**

Replace the existing `nav` rule:

```css
nav { display: flex; gap: 20px; margin-bottom: 32px; font-size: 14px; }
nav a { color: var(--muted); }
nav a.active, nav a:hover { color: var(--text); }
```

With:

```css
nav {
  display: flex;
  height: 100%;
}

nav a {
  color: var(--muted);
  padding: 0 14px;
  height: 100%;
  display: inline-flex;
  align-items: center;
  border-bottom: 2px solid transparent;
  font-size: 13px;
  transition: color 0.15s, border-color 0.15s;
  text-decoration: none;
}

nav a.active {
  color: var(--text);
  border-bottom-color: var(--accent);
}

nav a:hover {
  color: var(--text);
  text-decoration: none;
}
```

- [ ] **Step 6: Update `main` top margin and add `.card`**

Change the `main` rule from `margin: 40px auto` to `margin: 24px auto`:

```css
main { max-width: 720px; margin: 24px auto; padding: 0 24px; }
```

Add the `.card` class after `main`:

```css
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 16px;
}
```

- [ ] **Step 7: Polish button and input styles**

Replace the existing `button, .btn` rule:

```css
button, .btn {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 8px 18px;
  font-size: 14px;
  cursor: pointer;
  display: inline-block;
}
button:hover { opacity: 0.85; }
.btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
```

With:

```css
button, .btn {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 8px 18px;
  font-size: 14px;
  cursor: pointer;
  display: inline-block;
  transition: opacity 0.15s, box-shadow 0.15s, transform 0.1s;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
button:hover, .btn:hover { opacity: 0.85; }
button:active, .btn:active { transform: scale(0.98); }

.btn-ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
  box-shadow: none;
}
.btn-ghost:hover { border-color: var(--muted); opacity: 1; }
```

Replace the existing focus rule:

```css
select:focus, input:focus { outline: 2px solid var(--accent); border-color: transparent; }
```

With:

```css
select:focus, input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), 0.2);
}
```

- [ ] **Step 8: Commit**

```bash
git add recalld/static/style.css
git commit -m "feat: update CSS for light/dark palette, header, card, and control polish"
```

---

## Task 2: Restructure `base.html`

**Files:**
- Modify: `recalld/templates/base.html`

- [ ] **Step 1: Replace the header and nav**

Replace the entire contents of `recalld/templates/base.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>recalld</title>
  <link rel="stylesheet" href="/static/style.css">
  <script src="/static/htmx.min.js"></script>
  <script src="/static/app.js"></script>
</head>
<body>

<header class="app-header">
  <div class="header-row header-top">
    <span class="brand">recalld</span>
    <div id="status-bar-indicators"
         hx-get="/settings/status"
         hx-trigger="load, every 30s"
         hx-swap="innerHTML">
      <span class="status-indicator"><span class="dot yellow"></span> checking…</span>
    </div>
  </div>
  <div class="header-row header-nav">
    <nav>
      <a href="/" {% if request.url.path == "/" %}class="active"{% endif %}>Upload</a>
      <a href="/settings/" {% if request.url.path.startswith("/settings") %}class="active"{% endif %}>Settings</a>
    </nav>
  </div>
</header>

<main>
  {% block content %}{% endblock %}
</main>

</body>
</html>
```

- [ ] **Step 2: Start the dev server and verify the header renders correctly**

```
uv run recalld
```

Open `http://localhost:8765` in a browser. Check:
- Two-row header: brand name top-left, status pills top-right
- Nav row below with Upload/Settings links
- Active link has an accent-coloured underline

- [ ] **Step 3: Commit**

```bash
git add recalld/templates/base.html
git commit -m "feat: restructure base.html with two-row header"
```

---

## Task 3: Add card to `index.html`

**Files:**
- Modify: `recalld/templates/index.html`

- [ ] **Step 1: Wrap the upload form in a card**

In `recalld/templates/index.html`, wrap the `<form>` element (and its surrounding `{% if not cfg.categories %}` alert if present) in a `.card` div.

Replace:

```html
{% if not cfg.categories %}
<div class="alert error">
  No categories configured. <a href="/settings/">Add one in Settings</a> before uploading.
</div>
{% else %}
<form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
  ...
</form>
{% endif %}
```

With:

```html
<div class="card">
  {% if not cfg.categories %}
  <div class="alert error">
    No categories configured. <a href="/settings/">Add one in Settings</a> before uploading.
  </div>
  {% else %}
  <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
    <div class="drop-zone" id="drop-zone">
      <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="opacity:0.4">
        <path d="M12 16V4m0 0L8 8m4-4 4 4M4 20h16"/>
      </svg>
      <p>Drop a recording here, or click to browse</p>
      <input type="file" id="file-input" name="file" accept=".m4a,.mp4,.wav,.mov,.mkv,.webm" style="display:none" required>
    </div>

    <div class="field" style="margin-top:16px">
      <label for="category_id">Category</label>
      <select name="category_id" id="category_id">
        {% for cat in cfg.categories %}
        <option value="{{ cat.id }}" {% if cfg.last_used_category == cat.id %}selected{% endif %}>
          {{ cat.name }}
        </option>
        {% endfor %}
      </select>
    </div>

    <button type="submit" style="margin-top:8px">Process recording</button>
  </form>
  {% endif %}
</div>
```

- [ ] **Step 2: Wrap the incomplete jobs section in a card**

Replace:

```html
{% if incomplete_jobs %}
<h2>Incomplete jobs</h2>
<ul class="job-list">
  {% for job in incomplete_jobs %}
  {% include "partials/job_row.html" %}
  {% endfor %}
</ul>
{% endif %}
```

With:

```html
{% if incomplete_jobs %}
<div class="card">
  <h2>Incomplete jobs</h2>
  <ul class="job-list">
    {% for job in incomplete_jobs %}
    {% include "partials/job_row.html" %}
    {% endfor %}
  </ul>
</div>
{% endif %}
```

*(If Task 2 of the delete-job plan has not been applied yet, the loop body will still be the raw `<li>` — wrap that instead.)*

- [ ] **Step 3: Remove the `<h1>` top margin now covered by card padding**

Change:

```html
<h1>New recording</h1>
```

to remove any explicit bottom margin override if one exists; the `.card` padding provides sufficient spacing.

- [ ] **Step 4: Commit**

```bash
git add recalld/templates/index.html
git commit -m "feat: wrap index page sections in cards"
```

---

## Task 4: Add card to `settings.html`

**Files:**
- Modify: `recalld/templates/settings.html`

- [ ] **Step 1: Wrap the settings form in a card**

Read `recalld/templates/settings.html`. Wrap the entire `<form>` (from `<form ...>` to `</form>`) in a `<div class="card">`:

```html
<div class="card">
  <form method="post">
    ... (existing form contents unchanged) ...
  </form>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add recalld/templates/settings.html
git commit -m "feat: wrap settings form in card"
```

---

## Task 5: Add card to `processing.html`

**Files:**
- Modify: `recalld/templates/processing.html`

- [ ] **Step 1: Wrap the stage list and result area in a card**

In `recalld/templates/processing.html`, wrap the `<ul class="stage-list">`, transcript preview div, chunk-info div, and results section in a single `.card`:

```html
<div class="card">
  <ul class="stage-list">
    {% for stage in ["ingest", "transcribe", "diarise", "align", "postprocess", "vault"] %}
    <li class="stage-item" id="stage-{{ stage }}">
      <span class="stage-icon">○</span>
      <span class="stage-name">{{ stage | capitalize }}</span>
      <span class="stage-msg"></span>
    </li>
    {% endfor %}
  </ul>

  <div id="transcript-preview" class="preview-box" style="display:none"></div>
  <div id="chunk-info" style="display:none; color: var(--muted); font-size:13px; margin:8px 0;"></div>

  <div id="results-section" style="display:none; margin-top:24px">
    <h2>Summary</h2>
    <p id="result-summary"></p>
    <h2>Focus</h2>
    <ul id="result-focus" style="padding-left:20px; margin-top:8px"></ul>
    <a id="obsidian-link" href="#" class="btn" style="margin-top:16px; display:inline-block" target="_blank">
      Open in Obsidian
    </a>
  </div>
</div>
```

Leave the skip/fallback buttons and debug panel outside the card (they are secondary controls).

- [ ] **Step 2: Commit**

```bash
git add recalld/templates/processing.html
git commit -m "feat: wrap processing page stage list in card"
```

---

## Task 6: Visual verification

- [ ] **Step 1: Start the dev server**

```
uv run recalld
```

- [ ] **Step 2: Check light mode**

Open `http://localhost:8765` in a browser with light mode active (System Preferences → Appearance → Light, or DevTools → Rendering → Emulate CSS prefers-color-scheme: light).

Verify:
- Background is `#f5f5f7`, cards are white, text is dark
- Header is white with visible border
- Status pills render correctly
- Nav active link has accent underline
- Upload form and incomplete-jobs section each sit in a distinct card

- [ ] **Step 3: Check dark mode**

Switch to dark mode. Verify:
- Original dark palette restored
- No visual regressions on any page

- [ ] **Step 4: Run tests**

```
uv run pytest -v
```

Expected: all tests pass (no backend changes were made).
