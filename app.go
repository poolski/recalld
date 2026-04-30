package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type App struct {
	ConfigPath       string
	ScratchRoot      string
	StaticDir        string
	templates        *template.Template
	Bus              *Bus
	PipelineLauncher func(jobID, sourcePath string, cfg Config)
}

func NewApp(configPath, scratchRoot, staticDir string) (*App, error) {
	app := &App{
		ConfigPath:  configPath,
		ScratchRoot: scratchRoot,
		StaticDir:   staticDir,
		Bus:         NewBus(),
	}
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(scratchRoot, 0o755); err != nil {
		return nil, err
	}
	tmpl, err := parseTemplates()
	if err != nil {
		return nil, err
	}
	app.templates = tmpl
	if cfg, err := app.LoadConfig(); err == nil {
		if err := app.SaveConfig(cfg); err != nil {
			return nil, err
		}
	}
	return app, nil
}

func (a *App) LoadConfig() (Config, error) {
	data, err := os.ReadFile(a.ConfigPath)
	if err != nil {
		if os.IsNotExist(err) {
			cfg := DefaultConfig()
			cfg.Normalize()
			return cfg, nil
		}
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("config file at %s is invalid: %w", a.ConfigPath, err)
	}
	cfg.Normalize()
	return cfg, nil
}

func (a *App) SaveConfig(cfg Config) error {
	cfg.Normalize()
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(a.ConfigPath), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(a.ConfigPath, data, 0o600); err != nil {
		return err
	}
	return nil
}

func (a *App) loadJob(jobID string) (Job, error) {
	path := filepath.Join(a.ScratchRoot, jobID, "job.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return Job{}, err
	}
	var job Job
	if err := json.Unmarshal(data, &job); err != nil {
		return Job{}, err
	}
	job.ensureDefaults()
	return job, nil
}

func (a *App) saveJob(job *Job) error {
	job.ensureDefaults()
	path := filepath.Join(a.ScratchRoot, job.ID, "job.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(job, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func (a *App) deleteJob(jobID string) error {
	return os.RemoveAll(filepath.Join(a.ScratchRoot, jobID))
}

func (a *App) createJob(categoryID, originalFilename string) (Job, error) {
	job := NewJob(categoryID, originalFilename)
	if err := os.MkdirAll(filepath.Join(a.ScratchRoot, job.ID), 0o755); err != nil {
		return Job{}, err
	}
	if err := a.saveJob(&job); err != nil {
		return Job{}, err
	}
	return job, nil
}

func (a *App) sourcePath(job Job) string {
	return filepath.Join(a.ScratchRoot, job.ID, job.OriginalFilename)
}

func (a *App) launchPipeline(jobID, sourcePath string, cfg Config) {
	if a.PipelineLauncher != nil {
		a.PipelineLauncher(jobID, sourcePath, cfg)
		return
	}
	go a.runPipeline(jobID, sourcePath, cfg)
}

func (a *App) jobsDir(jobID string) string {
	return filepath.Join(a.ScratchRoot, jobID)
}

func (a *App) render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	switch name {
	case "index", "settings", "processing":
		bodyName := name + "_body"
		var body bytes.Buffer
		if err := a.templates.ExecuteTemplate(&body, bodyName, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		wrapped := struct {
			ActivePath string
			Body       template.HTML
		}{
			ActivePath: activePathFromData(data),
			Body:       template.HTML(body.String()),
		}
		if err := a.templates.ExecuteTemplate(w, "base", wrapped); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	default:
		if err := a.templates.ExecuteTemplate(w, name, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func activePathFromData(data any) string {
	switch v := data.(type) {
	case IndexPageData:
		return v.ActivePath
	case SettingsPageData:
		return v.ActivePath
	case ProcessingPageData:
		return v.ActivePath
	default:
		return ""
	}
}

func (a *App) renderHTML(w http.ResponseWriter, status int, name string, data any) {
	w.WriteHeader(status)
	a.render(w, name, data)
}

func (a *App) json(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func (a *App) indexHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	incomplete, _ := ListIncompleteJobs(a.ScratchRoot)
	a.render(w, "index", IndexPageData{
		ActivePath:     r.URL.Path,
		Config:         cfg,
		IncompleteJobs: incomplete,
	})
}

func (a *App) uploadHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()
	categoryID := r.FormValue("category_id")
	if categoryID == "" {
		http.Error(w, "category_id required", http.StatusBadRequest)
		return
	}

	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	job, err := a.createJob(categoryID, filepath.Base(header.Filename))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	dest := filepath.Join(a.jobsDir(job.ID), filepath.Base(header.Filename))
	out, err := os.Create(dest)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	out.Close()

	cfg.LastUsedCategory = categoryID
	_ = a.SaveConfig(cfg)

	a.launchPipeline(job.ID, dest, cfg)
	job, _ = a.loadJob(job.ID)
	a.render(w, "processing", ProcessingPageData{
		ActivePath:      r.URL.Path,
		Job:             job,
		Stages:          buildStageViews(job),
		StageStatusesJS: mustJSONJS(job.StageStatuses),
	})
}

func (a *App) settingsPage(w http.ResponseWriter, r *http.Request, saved bool) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var providerModels []ProviderModel
	ctxLen := fallbackContextLength
	if cfg.LLMModel != "" {
		providerModels, _ = ListAvailableModels(cfg.LLMBaseURL, cfg.LLMModel)
		ctxLen, _ = EnsureLoadedContextLength(cfg.LLMBaseURL, cfg.LLMModel)
	}
	var current *ProviderModel
	for i := range providerModels {
		if providerModels[i].Selected {
			current = &providerModels[i]
			break
		}
	}
	a.render(w, "settings", SettingsPageData{
		ActivePath:              r.URL.Path,
		Config:                  cfg,
		ProviderModels:          providerModels,
		CurrentModel:            current,
		DetectedContextLength:   ctxLen,
		HeadroomBudget:          TokenBudget(ctxLen, cfg.LLMContextHeadroom),
		ModelRefreshUnavailable: cfg.LLMModel != "" && len(providerModels) == 0,
		Saved:                   saved,
		ObsidianAPIKeySet:       cfg.ObsidianAPIKey != "",
		HuggingFaceTokenSet:     cfg.HuggingFaceToken != "",
	})
}

func (a *App) settingsGetHandler(w http.ResponseWriter, r *http.Request) {
	a.settingsPage(w, r, false)
}

func (a *App) settingsPostHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	cfg.VaultName = r.FormValue("vault_name")
	cfg.ObsidianAPIURL = r.FormValue("obsidian_api_url")
	if v := strings.TrimSpace(r.FormValue("obsidian_api_key")); v != "" {
		cfg.ObsidianAPIKey = v
	}
	cfg.LLMBaseURL = r.FormValue("llm_base_url")
	cfg.LLMModel = r.FormValue("llm_model")
	if v := strings.TrimSpace(r.FormValue("llm_context_headroom")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.LLMContextHeadroom)
	}
	cfg.LogLevel = r.FormValue("log_level")
	cfg.WhisperModel = r.FormValue("whisper_model")
	cfg.WhisperBaseURL = r.FormValue("whisper_base_url")
	if v := strings.TrimSpace(r.FormValue("huggingface_token")); v != "" {
		cfg.HuggingFaceToken = v
	}
	if v := strings.TrimSpace(r.FormValue("scratch_retention_days")); v != "" {
		fmt.Sscanf(v, "%d", &cfg.ScratchRetentionDays)
	}
	if err := a.SaveConfig(cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	a.settingsPage(w, r, true)
}

func (a *App) statusBarHandler(w http.ResponseWriter, r *http.Request) {
	cfg, err := a.LoadConfig()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	payload, _ := a.statusPayload(cfg)
	indicators := []struct {
		Kind string
		Name string
		OK   bool
		Hint string
	}{
		{"obsidian", "Obsidian API", payload["obsidian"].(map[string]any)["ok"].(bool), "Start Obsidian with Local REST API plugin enabled"},
		{"llm", "LLM", payload["llm"].(map[string]any)["ok"].(bool), "Start your LLM server"},
		{"ffmpeg", "ffmpeg", payload["ffmpeg"].(map[string]any)["ok"].(bool), "Run: brew install ffmpeg"},
		{"diarizer", "Diarizer", payload["diarizer"].(map[string]any)["ok"].(bool), "Local diarization runs in-process"},
	}

	var b strings.Builder
	for _, item := range indicators {
		dotClass := "red"
		if item.OK {
			dotClass = "green"
		}
		hintAttr := ""
		if !item.OK {
			hintAttr = fmt.Sprintf(` title="%s"`, template.HTMLEscapeString(item.Hint))
		}
		fmt.Fprintf(&b, `<button type="button" class="status-indicator" data-status-kind="%s" data-status-title="%s"%s><span class="dot %s"></span>%s</button>`,
			template.HTMLEscapeString(item.Kind),
			template.HTMLEscapeString(item.Name),
			hintAttr,
			dotClass,
			template.HTMLEscapeString(item.Name),
		)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, b.String())
}

func (a *App) statusPayload(cfg Config) (map[string]any, error) {
	obsidian := map[string]any{
		"ok":          false,
		"status_code": nil,
		"error":       "",
		"url":         cfg.ObsidianAPIURL,
		"vault_name":  cfg.VaultName,
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(strings.TrimRight(cfg.ObsidianAPIURL, "/") + "/")
	if err == nil {
		defer resp.Body.Close()
		obsidian["ok"] = resp.StatusCode < 500
		obsidian["status_code"] = resp.StatusCode
	} else {
		obsidian["error"] = err.Error()
	}

	var models []ProviderModel
	if cfg.LLMModel != "" {
		models, _ = ListAvailableModels(cfg.LLMBaseURL, cfg.LLMModel)
	}
	current := (*ProviderModel)(nil)
	for i := range models {
		if models[i].Selected {
			current = &models[i]
			break
		}
	}
	llmOK := cfg.LLMModel != "" && len(models) > 0
	if current == nil && len(models) > 0 {
		current = &models[0]
	}

	ffmpegPath, _ := execLookPath("ffmpeg")
	ffmpeg := map[string]any{
		"ok":      ffmpegPath != "",
		"path":    ifElse(ffmpegPath != "", ffmpegPath, "Not found"),
		"version": "Unavailable",
	}
	if ffmpegPath != "" {
		if out, err := execVersion(ffmpegPath); err == nil {
			ffmpeg["version"] = out
		}
	}

	diarizer := map[string]any{
		"ok": true,
	}
	return map[string]any{
		"obsidian": obsidian,
		"llm": map[string]any{
			"ok":                      llmOK,
			"provider_models":         models,
			"current_model":           current,
			"detected_context_length": func() int { n, _ := DetectContextLength(cfg.LLMBaseURL, cfg.LLMModel); return n }(),
			"base_url":                cfg.LLMBaseURL,
			"selected_model":          ifElse(cfg.LLMModel != "", cfg.LLMModel, "Not set"),
		},
		"ffmpeg":   ffmpeg,
		"diarizer": diarizer,
	}, nil
}

func execLookPath(name string) (string, error) {
	return exec.LookPath(name)
}

func execVersion(path string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, "-version")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return "", nil
	}
	return lines[0], nil
}

func ifElse(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}
