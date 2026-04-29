# UI Redesign

## Overview

Modernise the frontend across layout, header/status bar, and controls. Scope: `style.css`, `base.html`, `index.html`, `settings.html`, `processing.html`. The visual redesign itself does not require new backend behavior, though the current `processing.html` implementation now also depends on lightweight job-state hydration from the backend.

## 1. Colour Palette — Light Mode

Add `@media (prefers-color-scheme: light)` overriding CSS variables. Dark palette is unchanged.

| Variable   | Light value |
|------------|-------------|
| `--bg`     | `#f5f5f7`   |
| `--surface`| `#ffffff`   |
| `--border` | `#e0e0e0`   |
| `--text`   | `#1a1a1a`   |
| `--muted`  | `#6e6e73`   |
| `--accent` | `#5b4fe8`   |

Green/red/yellow and all spacing/radius values are unchanged.

## 2. Header & Navigation

Replace `.status-bar` with a two-row header:

**Row 1 — brand + status indicators**
- Full-width, `--surface` background, `1px solid var(--border)` bottom border
- Left: "recalld" in 15px semibold
- Right: status indicators as compact pills — `1px solid var(--border)` border, `4px` border-radius, `6px 10px` padding, coloured dot + label inline. Tooltip hint on hover (unchanged behaviour).

**Row 2 — nav**
- Left-aligned nav links inside the header below row 1, separated by a subtle border or padding
- Active state: `2px solid var(--accent)` bottom border on the link
- Hover: smooth colour transition from `--muted` to `--text`

`<main>` top margin reduces from `40px` to `24px` since the header provides visual separation.

Nav markup moves from inside `<main>` to inside the header element in `base.html`.

## 3. Cards

Key content areas are wrapped in surface cards: `background: var(--surface)`, `border: 1px solid var(--border)`, `border-radius: 12px`, `padding: 24px`.

Cards applied to:
- **index.html** — upload form (drop zone + category + submit); incomplete jobs section (when present)
- **settings.html** — settings form
- **processing.html** — stage list plus recovery and confirmation controls

No change to `max-width` (720px) or column structure.

## 4. Controls

**Primary buttons**
- Add `transition: opacity 0.15s, box-shadow 0.15s`
- Subtle `box-shadow` at rest (e.g. `0 1px 3px rgba(0,0,0,0.2)`)
- `transform: scale(0.98)` on `:active`

**Ghost buttons**
- Border colour shifts to `--muted` on hover (from `--border`)

**Inputs / selects**
- Replace `outline: 2px solid var(--accent)` focus style with `box-shadow: 0 0 0 3px rgba(accent-rgb, 0.25); border-color: var(--accent)`

**Drop zone**
- No changes needed.

**Nav links**
- See Section 2.

## Files Changed

| File | Changes |
|------|---------|
| `recalld/static/style.css` | Light palette, header styles, card class, control polish |
| `recalld/templates/base.html` | Header restructure, nav moved into header |
| `recalld/templates/index.html` | Upload form and job list wrapped in cards |
| `recalld/templates/settings.html` | Form wrapped in card |
| `recalld/templates/processing.html` | Stage list wrapped in card, with room for recovery and confirmation controls |
