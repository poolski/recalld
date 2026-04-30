package main

import (
	"encoding/json"
	"html/template"
)

type StageView struct {
	Name     string
	Status   string
	Expanded bool
	Message  string
}

type IndexPageData struct {
	ActivePath     string
	Config         Config
	IncompleteJobs []Job
}

type SettingsPageData struct {
	ActivePath              string
	Config                  Config
	ProviderModels          []ProviderModel
	CurrentModel            *ProviderModel
	DetectedContextLength   int
	HeadroomBudget          int
	ModelRefreshUnavailable bool
	Saved                   bool
	ObsidianAPIKeySet       bool
	HuggingFaceTokenSet     bool
}

type ProcessingPageData struct {
	ActivePath      string
	Job             Job
	Stages          []StageView
	StageStatusesJS template.JS
}

type JobStateResponse struct {
	ID                     string            `json:"id"`
	Status                 string            `json:"status"`
	CurrentStage           string            `json:"current_stage"`
	StageStatuses          map[string]string `json:"stage_statuses"`
	Preview                string            `json:"preview"`
	Filename               string            `json:"filename"`
	ObsidianURI            string            `json:"obsidian_uri"`
	Error                  string            `json:"error"`
	CanConfirmVault        bool              `json:"can_confirm_vault"`
	CanConfirmSpeakers     bool              `json:"can_confirm_speakers"`
	CanSwapSpeakers        bool              `json:"can_swap_speakers"`
	CanWriteTranscriptOnly bool              `json:"can_write_transcript_only"`
	VaultPreview           string            `json:"vault_preview"`
	TopicCount             int               `json:"topic_count,omitempty"`
	Strategy               string            `json:"strategy,omitempty"`
	Summary                string            `json:"summary,omitempty"`
	FocusPoints            []string          `json:"focus_points,omitempty"`
}

func mustJSONJS(v any) template.JS {
	data, _ := json.Marshal(v)
	return template.JS(data)
}
