package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func (a *App) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", a.indexHandler)
	mux.HandleFunc("POST /upload", a.uploadHandler)
	mux.HandleFunc("GET /settings/", a.settingsGetHandler)
	mux.HandleFunc("POST /settings/", a.settingsPostHandler)
	mux.HandleFunc("GET /settings/status", a.statusBarHandler)
	mux.HandleFunc("GET /settings/status/details", a.statusDetailsHandler)
	mux.HandleFunc("POST /categories/", a.addCategoryHandler)
	mux.HandleFunc("POST /categories/{id}/delete", a.deleteCategoryHandler)
	mux.HandleFunc("POST /categories/{id}/speakers", a.updateSpeakersHandler)
	mux.HandleFunc("GET /jobs/{id}", a.jobDetailHandler)
	mux.HandleFunc("GET /jobs/{id}/row", a.jobRowHandler)
	mux.HandleFunc("GET /jobs/{id}/confirm-delete", a.confirmDeleteHandler)
	mux.HandleFunc("DELETE /jobs/{id}", a.deleteJobHandler)
	mux.HandleFunc("GET /jobs/{id}/state", a.jobStateHandler)
	mux.HandleFunc("GET /jobs/{id}/events", a.jobEventsHandler)
	mux.HandleFunc("GET /jobs/{id}/open-in-obsidian", a.openInObsidianHandler)
	mux.HandleFunc("POST /jobs/{id}/rerun-from-failed", a.rerunFromFailedHandler)
	mux.HandleFunc("POST /jobs/{id}/rerun-from-start", a.rerunFromStartHandler)
	mux.HandleFunc("POST /jobs/{id}/restart-from/{stage}", a.restartFromStageHandler)
	mux.HandleFunc("POST /jobs/{id}/confirm-speakers", a.confirmSpeakersHandler)
	mux.HandleFunc("POST /jobs/{id}/swap-speakers", a.swapSpeakersHandler)
	mux.HandleFunc("POST /jobs/{id}/confirm-vault-write", a.confirmVaultWriteHandler)
	mux.HandleFunc("POST /jobs/{id}/skip-diarise", a.skipDiariseHandler)
	mux.HandleFunc("POST /jobs/{id}/write-transcript-only", a.writeTranscriptOnlyHandler)
	mux.Handle("GET /static/{path...}", http.StripPrefix("/static/", http.FileServer(http.Dir(a.StaticDir))))
	return mux
}

func pathSegment(r *http.Request, name string) string {
	return r.PathValue(name)
}

func (a *App) jobRowHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	a.render(w, "job_row", job)
}

func (a *App) confirmDeleteHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	a.render(w, "job_confirm_delete", job)
}

func (a *App) deleteJobHandler(w http.ResponseWriter, r *http.Request) {
	if err := a.deleteJob(pathSegment(r, "id")); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	jobs, err := ListIncompleteJobs(a.ScratchRoot)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = fmt.Fprintf(w, `<strong id="queued-jobs-count" hx-swap-oob="true">%d</strong>`, len(jobs))
}

func (a *App) jobDetailHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	a.render(w, "processing", ProcessingPageData{
		ActivePath:      r.URL.Path,
		Job:             job,
		Stages:          buildStageViews(job),
		StageStatusesJS: mustJSONJS(job.StageStatuses),
	})
}

func (a *App) jobStateHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	resp := JobStateResponse{
		ID:                 job.ID,
		Status:             string(job.Status),
		CurrentStage:       string(job.CurrentStage),
		StageStatuses:      job.StageStatuses,
		Filename:           job.Filename,
		Error:              job.Error,
		CanConfirmVault:    job.StageStatuses[string(JobStageVault)] == "awaiting_confirmation",
		CanConfirmSpeakers: job.StageStatuses[string(JobStageAlign)] == "awaiting_confirmation",
		CanSwapSpeakers:    job.StageStatuses[string(JobStageAlign)] == "awaiting_confirmation",
	}
	resp.Preview = a.loadAlignedPreview(job, 5)
	if job.StageStatuses[string(JobStagePostprocess)] == "done" || job.StageStatuses[string(JobStageVault)] == "awaiting_confirmation" {
		if pp, _ := a.loadPostprocessState(job); pp != nil {
			resp.Summary, _ = pp["summary"].(string)
			resp.FocusPoints = anySliceToStrings(pp["focus_points"])
			if s, ok := pp["strategy"].(string); ok {
				resp.Strategy = s
			}
			if n, ok := pp["topic_count"].(int); ok {
				resp.TopicCount = n
			}
		}
	}
	category := findCategory(cfg.Categories, job.CategoryID)
	if category != nil && job.StageStatuses[string(JobStageVault)] == "awaiting_confirmation" {
		resp.VaultPreview = a.loadVaultPreview(job, *category)
		if resp.VaultPreview != "" {
			resp.CanConfirmVault = true
		}
	}
	if job.StageStatuses[string(JobStageVault)] == "done" {
		resp.ObsidianURI = "/jobs/" + job.ID + "/open-in-obsidian"
	}
	a.json(w, http.StatusOK, resp)
}

func (a *App) jobEventsHandler(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	ch, cancel := a.Bus.Subscribe(pathSegment(r, "id"))
	defer cancel()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	for msg := range ch {
		_, _ = fmt.Fprintf(w, "data: %s\n\n", msg)
		flusher.Flush()
		if msg == "\"done\"" {
			return
		}
	}
}

func (a *App) openInObsidianHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	category := findCategory(cfg.Categories, job.CategoryID)
	if category == nil {
		http.NotFound(w, r)
		return
	}
	notePath := a.vaultNotePath(job, *category)
	if notePath == "" {
		http.NotFound(w, r)
		return
	}
	uri := a.vaultURI(job, *category, cfg.VaultName)
	if uri == "" {
		http.NotFound(w, r)
		return
	}
	http.Redirect(w, r, uri, http.StatusFound)
}

func (a *App) rerunFromFailedHandler(w http.ResponseWriter, r *http.Request) {
	a.schedulePipeline(w, r, false, nil)
}

func (a *App) rerunFromStartHandler(w http.ResponseWriter, r *http.Request) {
	a.schedulePipeline(w, r, true, nil)
}

func (a *App) restartFromStageHandler(w http.ResponseWriter, r *http.Request) {
	stage, ok := parseJobStage(pathSegment(r, "stage"))
	if !ok {
		http.Error(w, "invalid stage", http.StatusBadRequest)
		return
	}
	a.schedulePipeline(w, r, false, &stage)
}

func (a *App) confirmSpeakersHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	if job.StageStatuses[string(JobStageAlign)] == "awaiting_confirmation" {
		job.StageStatuses[string(JobStageAlign)] = "done"
		job.CurrentStage = JobStagePostprocess
		job.Status = JobStatusRunning
	}
	_ = a.saveJob(&job)
	_ = a.Bus.Publish(job.ID, map[string]any{"stage": "align", "status": "done"})
	cfg, _ := a.LoadConfig()
	a.launchPipeline(job.ID, a.sourcePath(job), cfg)
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) swapSpeakersHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	_ = a.swapAlignedSpeakers(&job)
	_ = a.saveJob(&job)
	_ = a.Bus.Publish(job.ID, map[string]any{
		"stage":                "align",
		"status":               "awaiting_confirmation",
		"preview":              a.loadAlignedPreview(job, 5),
		"can_confirm_speakers": true,
	})
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) confirmVaultWriteHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	if err := r.ParseForm(); err == nil {
		if filename := strings.TrimSpace(r.FormValue("filename")); filename != "" {
			job.Filename = filename
		}
	}
	job.Status = JobStatusRunning
	job.StageStatuses[string(JobStageVault)] = "pending"
	_ = a.saveJob(&job)
	cfg, _ := a.LoadConfig()
	a.launchPipeline(job.ID, a.sourcePath(job), cfg)
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) skipDiariseHandler(w http.ResponseWriter, r *http.Request) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	words := []WordSegment{}
	if data, err := os.ReadFile(job.TranscriptPath); err == nil {
		_ = json.Unmarshal(data, &words)
	}
	var turns []SpeakerTurn
	if len(words) > 0 {
		turns = []SpeakerTurn{{Start: words[0].Start, End: words[len(words)-1].End, Speaker: "SPEAKER_00"}}
	}
	labelled := Align(words, turns, nil)
	alignedPath := filepath.Join(a.jobsDir(job.ID), "aligned.json")
	_ = os.WriteFile(alignedPath, mustJSON(labelled), 0o600)
	job.AlignedPath = alignedPath
	job.StageStatuses[string(JobStageDiarise)] = "failed"
	job.StageStatuses[string(JobStageAlign)] = "done"
	job.CurrentStage = JobStagePostprocess
	job.Status = JobStatusRunning
	_ = a.saveJob(&job)
	_ = a.Bus.Publish(job.ID, map[string]any{"stage": "align", "status": "done"})
	cfg, _ := a.LoadConfig()
	a.launchPipeline(job.ID, a.sourcePath(job), cfg)
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) writeTranscriptOnlyHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	category := findCategory(cfg.Categories, job.CategoryID)
	if category == nil {
		http.NotFound(w, r)
		return
	}
	var labelled []LabelledTurn
	if data, err := os.ReadFile(job.AlignedPath); err == nil {
		_ = json.Unmarshal(data, &labelled)
	}
	writer := VaultWriter{APIURL: cfg.ObsidianAPIURL, APIKey: cfg.ObsidianAPIKey}
	sessionDate := job.CreatedAt
	filename := job.Filename
	if filename == "" {
		filename = fmt.Sprintf("%s %s.md", sessionDate.Format("2006-01-02"), category.Name)
	}
	content := RenderSessionNote(sessionDate, category.Name, []string{category.SpeakerA, category.SpeakerB}, nil, labelled)
	if err := writer.WriteNote(category.VaultPath, filename, content); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	job.Status = JobStatusComplete
	job.StageStatuses[string(JobStageVault)] = "done"
	_ = a.saveJob(&job)
	_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "done", "obsidian_uri": "/jobs/" + job.ID + "/open-in-obsidian", "summary": "", "focus_points": []string{}})
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) statusDetailsHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	kind := r.URL.Query().Get("kind")
	payload, _ := a.statusPayload(cfg)
	if _, ok := payload[kind]; !ok {
		a.json(w, http.StatusBadRequest, map[string]any{"error": "Unknown status kind: " + kind})
		return
	}
	switch kind {
	case "llm":
		llm := payload[kind].(map[string]any)
		current, _ := llm["current_model"].(*ProviderModel)
		var loaded, max int
		if current != nil {
			loaded = current.LoadedContextLength
			max = current.MaxContextLength
		}
		items := []map[string]any{
			{"label": "Selected model", "value": llm["selected_model"]},
			{"label": "Base URL", "value": llm["base_url"]},
			{"label": "Loaded", "value": ifElse(current != nil && current.IsLoaded(), "Yes", "No")},
			{"label": "Loaded context length", "value": formatInt(loaded)},
			{"label": "Maximum context length", "value": formatInt(max)},
			{"label": "Available models", "value": fmt.Sprint(len(llm["provider_models"].([]ProviderModel)))},
		}
		a.json(w, http.StatusOK, map[string]any{"title": "LLM", "ok": llm["ok"], "items": items})
	case "obsidian":
		obsidian := payload[kind].(map[string]any)
		items := []map[string]any{
			{"label": "Vault name", "value": obsidian["vault_name"]},
			{"label": "API URL", "value": obsidian["url"]},
			{"label": "Health", "value": ifElse(obsidian["ok"].(bool), "Healthy", "Unhealthy")},
			{"label": "HTTP status", "value": fmt.Sprint(obsidian["status_code"])},
			{"label": "Auth key", "value": ifElse(cfg.ObsidianAPIKey != "", "Set", "Not set")},
		}
		if errMsg, _ := obsidian["error"].(string); errMsg != "" {
			items = append(items, map[string]any{"label": "Error", "value": errMsg})
		}
		a.json(w, http.StatusOK, map[string]any{"title": "Obsidian API", "ok": obsidian["ok"], "items": items})
	case "ffmpeg":
		ffmpeg := payload[kind].(map[string]any)
		items := []map[string]any{
			{"label": "Binary", "value": ffmpeg["path"]},
			{"label": "Version", "value": ffmpeg["version"]},
		}
		a.json(w, http.StatusOK, map[string]any{"title": "ffmpeg", "ok": ffmpeg["ok"], "items": items})
	case "diarizer":
		a.json(w, http.StatusOK, map[string]any{"title": "Diarizer", "ok": true, "items": []map[string]any{{"label": "Mode", "value": "Local heuristic"}}})
	default:
		a.json(w, http.StatusBadRequest, map[string]any{"error": "Unknown status kind: " + kind})
	}
}

func (a *App) addCategoryHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/categories/" {
		http.NotFound(w, r)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	cat := Category{
		ID:            newID(),
		Name:          r.FormValue("name"),
		VaultPath:     r.FormValue("vault_path"),
		FocusNotePath: r.FormValue("focus_note_path"),
		SpeakerA:      valueOrDefault(r.FormValue("speaker_a"), "You"),
		SpeakerB:      valueOrDefault(r.FormValue("speaker_b"), "Coach"),
	}
	cfg.Categories = append(cfg.Categories, cat)
	_ = a.SaveConfig(cfg)
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (a *App) deleteCategoryHandler(w http.ResponseWriter, r *http.Request) {
	id := pathSegment(r, "id")
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var cats []Category
	for _, cat := range cfg.Categories {
		if cat.ID != id {
			cats = append(cats, cat)
		}
	}
	cfg.Categories = cats
	if cfg.LastUsedCategory == id {
		if len(cfg.Categories) > 0 {
			cfg.LastUsedCategory = cfg.Categories[0].ID
		} else {
			cfg.LastUsedCategory = ""
		}
	}
	_ = a.SaveConfig(cfg)
	http.Redirect(w, r, "/settings/", http.StatusSeeOther)
}

func (a *App) updateSpeakersHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	id := pathSegment(r, "id")
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for i := range cfg.Categories {
		if cfg.Categories[i].ID == id {
			cfg.Categories[i].SpeakerA = r.FormValue("speaker_a")
			cfg.Categories[i].SpeakerB = r.FormValue("speaker_b")
			break
		}
	}
	_ = a.SaveConfig(cfg)
	http.Redirect(w, r, "/settings/", http.StatusSeeOther)
}

func (a *App) schedulePipeline(w http.ResponseWriter, r *http.Request, fromStart bool, restartStage *JobStage) {
	job, err := a.loadJob(pathSegment(r, "id"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	ResetJobForRerun(&job, fromStart, restartStage)
	job.Status = JobStatusRunning
	_ = a.saveJob(&job)
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	a.launchPipeline(job.ID, a.sourcePath(job), cfg)
	a.render(w, "processing", ProcessingPageData{ActivePath: r.URL.Path, Job: job, Stages: buildStageViews(job), StageStatusesJS: mustJSONJS(job.StageStatuses)})
}

func (a *App) loadAlignedPreview(job Job, limit int) string {
	if job.AlignedPath == "" {
		return ""
	}
	var turns []LabelledTurn
	if data, err := os.ReadFile(job.AlignedPath); err == nil {
		_ = json.Unmarshal(data, &turns)
	}
	if len(turns) > limit {
		turns = turns[:limit]
	}
	lines := make([]string, 0, len(turns))
	for _, turn := range turns {
		lines = append(lines, fmt.Sprintf("**%s:** %s", turn.Speaker, turn.Text))
	}
	return strings.Join(lines, "\n")
}

func (a *App) loadPostprocessState(job Job) (map[string]any, error) {
	if job.PostprocessPath == "" {
		return nil, nil
	}
	data, err := os.ReadFile(job.PostprocessPath)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func (a *App) loadVaultPreview(job Job, category Category) string {
	if job.PostprocessPath == "" || job.AlignedPath == "" {
		return ""
	}
	pp, err := a.loadPostprocessState(job)
	if err != nil || pp == nil {
		return ""
	}
	var turns []LabelledTurn
	if data, err := os.ReadFile(job.AlignedPath); err == nil {
		_ = json.Unmarshal(data, &turns)
	}
	result := &PostProcessResult{
		Summary:     stringValue(pp["summary"]),
		FocusPoints: stringSliceValue(pp["focus_points"]),
		Strategy:    stringValue(pp["strategy"]),
		TopicCount:  intValue(pp["topic_count"]),
	}
	return RenderSessionNotePreview(job.CreatedAt, category.Name, []string{category.SpeakerA, category.SpeakerB}, result, turns, 1200)
}

func (a *App) vaultNotePath(job Job, category Category) string {
	filename := job.Filename
	if filename == "" {
		filename = fmt.Sprintf("%s %s.md", job.CreatedAt.Format("2006-01-02"), category.Name)
	}
	return filepath.ToSlash(filepath.Join(category.VaultPath, filename))
}

func (a *App) vaultURI(job Job, category Category, vaultName string) string {
	notePath := a.vaultNotePath(job, category)
	if notePath == "" {
		return ""
	}
	return "obsidian://open?vault=" + urlQueryEscape(vaultName) + "&file=" + urlQueryEscape(notePath)
}

func (a *App) swapAlignedSpeakers(job *Job) error {
	if job.AlignedPath == "" {
		return nil
	}
	var turns []LabelledTurn
	data, err := os.ReadFile(job.AlignedPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &turns); err != nil {
		return err
	}
	var speakers []string
	for _, turn := range turns {
		seen := false
		for _, sp := range speakers {
			if sp == turn.Speaker {
				seen = true
				break
			}
		}
		if !seen {
			speakers = append(speakers, turn.Speaker)
		}
		if len(speakers) == 2 {
			break
		}
	}
	if len(speakers) != 2 {
		return nil
	}
	swap := map[string]string{speakers[0]: speakers[1], speakers[1]: speakers[0]}
	for i := range turns {
		if v, ok := swap[turns[i].Speaker]; ok {
			turns[i].Speaker = v
		}
	}
	return os.WriteFile(job.AlignedPath, mustJSON(turns), 0o600)
}

func buildStageViews(job Job) []StageView {
	stages := make([]StageView, 0, len(StageNames))
	for _, name := range StageNames {
		status := job.StageStatuses[name]
		expanded := status == "running" || status == "failed" || status == "awaiting_confirmation"
		if name == string(job.CurrentStage) && job.Error != "" {
			expanded = true
		}
		msg := ""
		if name == string(job.CurrentStage) && job.Error != "" {
			msg = job.Error
		}
		stages = append(stages, StageView{Name: name, Status: status, Expanded: expanded, Message: msg})
	}
	return stages
}

func parseJobStage(s string) (JobStage, bool) {
	switch JobStage(s) {
	case JobStageIngest, JobStageTranscribe, JobStageDiarise, JobStageAlign, JobStagePostprocess, JobStageVault:
		return JobStage(s), true
	default:
		return "", false
	}
}

func valueOrDefault(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func stringSliceValue(v any) []string {
	switch t := v.(type) {
	case []string:
		return t
	case []any:
		var out []string
		for _, item := range t {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func anySliceToStrings(v any) []string {
	switch t := v.(type) {
	case []string:
		return t
	case []any:
		out := make([]string, 0, len(t))
		for _, item := range t {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func intValue(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case float64:
		return int(t)
	default:
		return 0
	}
}

func formatInt(v int) string {
	if v <= 0 {
		return "Not loaded"
	}
	return fmt.Sprintf("%d", v)
}

func urlQueryEscape(s string) string {
	r := strings.NewReplacer(" ", "%20", "/", "%2F")
	return r.Replace(s)
}
