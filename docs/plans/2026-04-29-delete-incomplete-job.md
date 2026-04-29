# Delete Incomplete Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Remove button with inline HTMX confirmation to each incomplete job row; confirming deletes the job's scratch directory and removes the row from the DOM without a page reload.

**Architecture:** New `delete_job()` helper in `recalld/jobs.py`; three new endpoints in `recalld/routers/jobs.py` returning HTML fragments; two Jinja2 partial templates for the job row and confirmation prompt; `index.html` updated to `{% include %}` the row partial so the HTML is defined in one place.

**Tech Stack:** FastAPI, Jinja2, HTMX (already in use), pytest + `starlette.testclient.TestClient`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `recalld/jobs.py` | Modify | Add `delete_job()` function |
| `recalld/routers/jobs.py` | Modify | Add `GET /{id}/row`, `GET /{id}/confirm-delete`, `DELETE /{id}` |
| `recalld/templates/partials/job_row.html` | Create | `<li>` fragment: filename, stage, Resume + Remove buttons |
| `recalld/templates/partials/job_confirm_delete.html` | Create | `<li>` fragment: "Remove filename?" with Delete/Cancel |
| `recalld/templates/index.html` | Modify | Replace inline `<li>` with `{% include "partials/job_row.html" %}` |
| `tests/test_jobs.py` | Modify | Add `test_delete_job` |
| `tests/test_routers_jobs.py` | Create | HTTP tests for the 3 new endpoints |

---

## Task 1: Add `delete_job()` to `recalld/jobs.py`

**Files:**
- Modify: `tests/test_jobs.py`
- Modify: `recalld/jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
def test_delete_job(tmp_path):
    from recalld.jobs import delete_job
    job = create_job(category_id="test", original_filename="x.m4a", scratch_root=tmp_path)
    assert (tmp_path / job.id).exists()
    delete_job(job.id, scratch_root=tmp_path)
    assert not (tmp_path / job.id).exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_jobs.py::test_delete_job -v
```

Expected: `ImportError` — `delete_job` is not yet defined.

- [ ] **Step 3: Implement `delete_job` in `recalld/jobs.py`**

Add `import shutil` to the existing imports at the top of `recalld/jobs.py`.

Add this function after `load_job`:

```python
def delete_job(job_id: str, scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> None:
    shutil.rmtree(_job_dir(job_id, scratch_root), ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_jobs.py::test_delete_job -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add recalld/jobs.py tests/test_jobs.py
git commit -m "feat: add delete_job helper"
```

---

## Task 2: Create partial templates

**Files:**
- Create: `recalld/templates/partials/job_row.html`
- Create: `recalld/templates/partials/job_confirm_delete.html`
- Modify: `recalld/templates/index.html`

- [ ] **Step 1: Create `recalld/templates/partials/job_row.html`**

```html
<li class="job-item">
  <span class="filename">{{ job.original_filename }}</span>
  <span class="stage">{{ job.current_stage.value }}</span>
  <form action="/jobs/{{ job.id }}/resume" method="post" style="display:inline">
    <button type="submit" class="btn-ghost" style="padding:4px 12px;font-size:12px">Resume</button>
  </form>
  <button class="btn-ghost" style="padding:4px 12px;font-size:12px;color:var(--red);border-color:var(--red)"
    hx-get="/jobs/{{ job.id }}/confirm-delete"
    hx-target="closest li"
    hx-swap="outerHTML">Remove</button>
</li>
```

- [ ] **Step 2: Create `recalld/templates/partials/job_confirm_delete.html`**

```html
<li class="job-item">
  <span class="filename">Remove <strong>{{ job.original_filename }}</strong>?</span>
  <button class="btn-ghost" style="padding:4px 12px;font-size:12px;color:var(--red);border-color:var(--red)"
    hx-delete="/jobs/{{ job.id }}"
    hx-target="closest li"
    hx-swap="outerHTML">Delete</button>
  <button class="btn-ghost" style="padding:4px 12px;font-size:12px"
    hx-get="/jobs/{{ job.id }}/row"
    hx-target="closest li"
    hx-swap="outerHTML">Cancel</button>
</li>
```

- [ ] **Step 3: Update `index.html` to use the row partial**

In `recalld/templates/index.html`, replace the `<li>...</li>` block inside the `{% for job in incomplete_jobs %}` loop:

Old:
```html
  <li class="job-item">
    <span class="filename">{{ job.original_filename }}</span>
    <span class="stage">{{ job.current_stage.value }}</span>
    <form action="/jobs/{{ job.id }}/resume" method="post" style="display:inline">
      <button type="submit" class="btn-ghost" style="padding:4px 12px;font-size:12px">Resume</button>
    </form>
  </li>
```

New:
```html
  {% include "partials/job_row.html" %}
```

- [ ] **Step 4: Commit**

```bash
git add recalld/templates/
git commit -m "feat: add job row and confirm-delete partial templates"
```

---

## Task 3: Add three new router endpoints

**Files:**
- Create: `tests/test_routers_jobs.py`
- Modify: `recalld/routers/jobs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_routers_jobs.py`:

```python
import pytest
from fastapi.testclient import TestClient
from recalld.app import create_app
from recalld.jobs import create_job


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.jobs.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    return tmp_path


@pytest.fixture
def client(scratch):
    return TestClient(create_app())


def test_delete_removes_directory(scratch, client):
    job = create_job(category_id="test", original_filename="x.m4a", scratch_root=scratch)
    assert (scratch / job.id).exists()
    resp = client.delete(f"/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.text == ""
    assert not (scratch / job.id).exists()


def test_confirm_delete_returns_html(scratch, client):
    job = create_job(category_id="test", original_filename="session.m4a", scratch_root=scratch)
    resp = client.get(f"/jobs/{job.id}/confirm-delete")
    assert resp.status_code == 200
    assert "session.m4a" in resp.text
    assert "Delete" in resp.text
    assert "Cancel" in resp.text


def test_job_row_returns_html(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    resp = client.get(f"/jobs/{job.id}/row")
    assert resp.status_code == 200
    assert "audio.m4a" in resp.text
    assert "Resume" in resp.text
    assert "Remove" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_routers_jobs.py -v
```

Expected: all three tests fail with 404 — endpoints not yet defined.

- [ ] **Step 3: Update imports in `recalld/routers/jobs.py`**

Change the jobs import line at the top of `recalld/routers/jobs.py`:

Old:
```python
from recalld.jobs import DEFAULT_SCRATCH_ROOT, load_job
```

New:
```python
from recalld.jobs import DEFAULT_SCRATCH_ROOT, delete_job, load_job
```

- [ ] **Step 4: Add the three new endpoints to `recalld/routers/jobs.py`**

Insert these three routes after the existing `job_detail` route and before `job_events`:

```python
@router.get("/{job_id}/row", response_class=HTMLResponse)
async def job_row(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return templates.TemplateResponse(request, "partials/job_row.html", {"job": job})


@router.get("/{job_id}/confirm-delete", response_class=HTMLResponse)
async def confirm_delete(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return templates.TemplateResponse(request, "partials/job_confirm_delete.html", {"job": job})


@router.delete("/{job_id}", response_class=HTMLResponse)
async def delete_job_route(job_id: str):
    delete_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return HTMLResponse("")
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_routers_jobs.py -v
```

Expected: all three PASS.

- [ ] **Step 6: Run full test suite**

```
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add recalld/routers/jobs.py tests/test_routers_jobs.py
git commit -m "feat: add delete job router endpoints"
```
