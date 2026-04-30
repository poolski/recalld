# Go Layout Restructure Plan

> Completed: the package move described in this plan has been implemented.
> Application code now lives under `app/`, startup wiring under `cmd/`, and `main.go` only delegates to `cmd.Execute()`.

**Current State**

- The app has already been rewritten in Go and currently passes `go test ./...`.
- The executable entrypoint is split between a top-level `main.go` and an internal Cobra wrapper in `internal/cli`.
- Most application logic lives under `internal/recalld`, including HTTP handlers, persistence, pipeline orchestration, templates, and test coverage.
- The current structure is functional, but it is not the layout you want for the final app.

**Target Shape**

- No `internal/` tree for application code.
- All reusable packages live at the repository root as top-level packages.
- All code related to starting, configuring, and running the app lives under `cmd/`.
- A single top-level `main.go` remains as the binary entrypoint and delegates into `cmd/`.
- Cobra remains the CLI framework for startup and flags.

**Planned Changes**

- Move the current application package out of `internal/recalld` into a top-level package tree.
- Move the Cobra command wiring out of `internal/cli` into `cmd/`, alongside the app bootstrap logic.
- Keep `main.go` minimal: it should only invoke the Cobra command.
- Preserve behavior, HTTP routes, templates, SSE events, job persistence, and tests during the move.
- Update imports, package names, and test references after the package move.

**Implementation Notes**

- The app should keep the same runtime behavior while the package boundaries are cleaned up.
- The repo should still build and test after the move, with no change to the user-facing flow.
- The next pass should be mostly mechanical package relocation, not feature work.
