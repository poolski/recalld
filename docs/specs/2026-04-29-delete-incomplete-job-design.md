# Delete Incomplete Job

## Overview

Add a Remove button to each incomplete job row on the index page. Clicking it triggers an inline HTMX confirmation prompt; confirming deletes the job's scratch directory and removes the row from the page.

## Backend

Three new endpoints in `recalld/routers/jobs.py`:

### `GET /jobs/{job_id}/row`
Returns the `<li>` HTML fragment for a single job row (filename, stage, Resume button, Remove button). Used by the Cancel action to restore the row after a confirmation prompt.

### `GET /jobs/{job_id}/confirm-delete`
Returns a `<li>` HTML fragment containing:
- "Remove *{original_filename}*?" label
- **Delete** button — `hx-delete="/jobs/{job_id}"`
- **Cancel** button — `hx-get="/jobs/{job_id}/row"`

### `DELETE /jobs/{job_id}`
Calls `shutil.rmtree` on `DEFAULT_SCRATCH_ROOT / job_id`. Returns HTTP 200 with an empty body. HTMX swaps the row's `outerHTML` with the empty response, removing it from the DOM.

## Frontend

Each `<li>` in the incomplete jobs list (`index.html`) gains a **Remove** button alongside the existing **Resume** button:

```html
<button class="btn-ghost" style="..."
  hx-get="/jobs/{{ job.id }}/confirm-delete"
  hx-target="closest li"
  hx-swap="outerHTML">Remove</button>
```

The confirmation fragment mirrors this pattern — Delete and Cancel both target `closest li` with `outerHTML` swap. No JavaScript required.

## Testing

New test cases (unit/integration):

1. `DELETE /jobs/{job_id}` removes the scratch directory from disk and returns 200.
2. `GET /jobs/{job_id}/confirm-delete` returns HTML containing the job's filename and both Delete and Cancel buttons.
3. `GET /jobs/{job_id}/row` returns HTML containing the job's filename, Resume button, and Remove button.
