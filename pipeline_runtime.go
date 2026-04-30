package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (a *App) runPipeline(jobID, sourcePath string, cfg Config) {
	job, err := a.loadJob(jobID)
	if err != nil {
		return
	}
	job.Status = JobStatusRunning
	_ = a.saveJob(&job)

	scratch := a.jobsDir(job.ID)
	defer func() {
		if r := recover(); r != nil {
			job.Status = JobStatusFailed
			job.Error = fmt.Sprint(r)
			if job.CurrentStage != "" {
				job.StageStatuses[string(job.CurrentStage)] = "failed"
			}
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": string(job.CurrentStage), "status": "failed", "message": job.Error})
		}
	}()

	if job.CurrentStage == JobStageIngest {
		job.StageStatuses[string(JobStageIngest)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "ingest", "status": "running"})
		wav, err := Ingest(sourcePath, scratch)
		if err != nil {
			job.Status = JobStatusFailed
			job.Error = err.Error()
			job.StageStatuses[string(JobStageIngest)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "ingest", "status": "failed", "message": err.Error()})
			return
		}
		job.WavPath = wav
		job.StageStatuses[string(JobStageIngest)] = "done"
		job.CurrentStage = JobStageTranscribe
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "ingest", "status": "done"})
	}

	if job.CurrentStage == JobStageTranscribe {
		job.StageStatuses[string(JobStageTranscribe)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "transcribe", "status": "running"})
		transcriber := Transcriber{
			Model:            cfg.WhisperModel,
			HuggingFaceToken: cfg.HuggingFaceToken,
			Log: func(level int, text string) {
				if text == "" {
					return
				}
				_ = a.Bus.Publish(job.ID, map[string]any{
					"stage":   "transcribe",
					"status":  "running",
					"message": fmt.Sprintf("[whisper/%d] %s", level, text),
				})
			},
		}
		words, err := transcriber.Transcribe(job.WavPath)
		if err != nil {
			job.Status = JobStatusFailed
			job.Error = err.Error()
			job.StageStatuses[string(JobStageTranscribe)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "transcribe", "status": "failed", "message": err.Error()})
			return
		}
		transcriptPath := filepath.Join(scratch, "transcript.json")
		_ = os.WriteFile(transcriptPath, mustJSON(words), 0o600)
		job.TranscriptPath = transcriptPath
		job.StageStatuses[string(JobStageTranscribe)] = "done"
		job.CurrentStage = JobStageDiarise
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "transcribe", "status": "done"})
	}

	if job.CurrentStage == JobStageDiarise {
		job.StageStatuses[string(JobStageDiarise)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "diarise", "status": "running"})
		turns, err := Diarise(job.WavPath)
		if err != nil {
			job.Status = JobStatusFailed
			job.Error = err.Error()
			job.StageStatuses[string(JobStageDiarise)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "diarise", "status": "failed", "message": err.Error(), "can_skip": true})
			return
		}
		diarPath := filepath.Join(scratch, "diarisation.json")
		_ = os.WriteFile(diarPath, mustJSON(turns), 0o600)
		job.DiarisationPath = diarPath
		job.StageStatuses[string(JobStageDiarise)] = "done"
		job.CurrentStage = JobStageAlign
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "diarise", "status": "done"})
	}

	if job.CurrentStage == JobStageAlign {
		job.StageStatuses[string(JobStageAlign)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "align", "status": "running"})

		var words []WordSegment
		var turns []SpeakerTurn
		if data, err := os.ReadFile(job.TranscriptPath); err == nil {
			_ = json.Unmarshal(data, &words)
		}
		if data, err := os.ReadFile(job.DiarisationPath); err == nil {
			_ = json.Unmarshal(data, &turns)
		}

		var speakerMap map[string]string
		var category *Category
		for i := range cfg.Categories {
			if cfg.Categories[i].ID == job.CategoryID {
				category = &cfg.Categories[i]
				break
			}
		}
		if category != nil {
			speakerMap = buildSpeakerMap(turns, category.SpeakerA, category.SpeakerB)
			job.Speaker00 = category.SpeakerA
			job.Speaker01 = category.SpeakerB
		}
		labelled := Align(words, turns, speakerMap)
		alignedPath := filepath.Join(scratch, "aligned.json")
		_ = os.WriteFile(alignedPath, mustJSON(labelled), 0o600)
		job.AlignedPath = alignedPath
		job.StageStatuses[string(JobStageAlign)] = "awaiting_confirmation"
		job.Status = JobStatusPending
		job.CurrentStage = JobStageAlign
		_ = a.saveJob(&job)
		preview := renderAlignedPreview(labelled, 5)
		_ = a.Bus.Publish(job.ID, map[string]any{
			"stage":                "align",
			"status":               "awaiting_confirmation",
			"preview":              preview,
			"can_confirm_speakers": true,
			"can_swap_speakers":    true,
		})
		return
	}

	if job.CurrentStage == JobStagePostprocess {
		job.StageStatuses[string(JobStagePostprocess)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "postprocess", "status": "running"})

		var labelled []LabelledTurn
		if data, err := os.ReadFile(job.AlignedPath); err == nil {
			_ = json.Unmarshal(data, &labelled)
		}
		category := findCategory(cfg.Categories, job.CategoryID)
		speakerA, speakerB := "You", "Coach"
		if category != nil {
			speakerA = category.SpeakerA
			speakerB = category.SpeakerB
		}
		ctxLen, _ := EnsureLoadedContextLength(cfg.LLMBaseURL, cfg.LLMModel)
		budget := TokenBudget(ctxLen, cfg.LLMContextHeadroom)
		client := &LLMClient{BaseURL: cfg.LLMBaseURL, Model: cfg.LLMModel}
		result, err := PostProcess{Client: client}.Run(context.Background(), labelled, budget, speakerA, speakerB, func(msg string) {
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "postprocess", "status": "running", "message": msg})
		}, func(summary string) {
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "postprocess", "status": "running", "summary": summary})
		}, func(eventType string, data map[string]any) {
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "postprocess", "status": "running", "lmstudio_event": eventType})
		})
		if err != nil {
			job.Status = JobStatusFailed
			job.Error = err.Error()
			job.StageStatuses[string(JobStagePostprocess)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "postprocess", "status": "failed", "message": err.Error(), "can_write_transcript_only": true})
			return
		}
		ppPath := filepath.Join(scratch, "postprocess.json")
		_ = os.WriteFile(ppPath, mustJSON(map[string]any{
			"summary":      result.Summary,
			"focus_points": result.FocusPoints,
			"strategy":     result.Strategy,
			"topic_count":  result.TopicCount,
		}), 0o600)
		job.PostprocessPath = ppPath
		job.TopicCount = result.TopicCount
		job.ChunkStrategy = result.Strategy
		if job.Filename == "" && category != nil {
			job.Filename = fmt.Sprintf("%s %s.md", job.CreatedAt.Format("2006-01-02"), category.Name)
		}
		job.StageStatuses[string(JobStagePostprocess)] = "done"
		job.CurrentStage = JobStageVault
		job.Status = JobStatusPending
		job.StageStatuses[string(JobStageVault)] = "awaiting_confirmation"
		_ = a.saveJob(&job)
		preview := ""
		if category != nil {
			preview = RenderSessionNotePreview(job.CreatedAt, category.Name, []string{category.SpeakerA, category.SpeakerB}, result, labelled, 1200)
		}
		_ = a.Bus.Publish(job.ID, map[string]any{
			"stage":        "postprocess",
			"status":       "done",
			"topic_count":  result.TopicCount,
			"strategy":     result.Strategy,
			"summary":      result.Summary,
			"focus_points": result.FocusPoints,
		})
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "awaiting_confirmation", "can_confirm_vault": true, "filename": job.Filename, "vault_preview": preview})
		return
	}

	if job.CurrentStage == JobStageVault {
		category := findCategory(cfg.Categories, job.CategoryID)
		if job.StageStatuses[string(JobStageVault)] == "awaiting_confirmation" {
			var labelled []LabelledTurn
			var result *PostProcessResult
			if data, err := os.ReadFile(job.AlignedPath); err == nil {
				_ = json.Unmarshal(data, &labelled)
			}
			if data, err := os.ReadFile(job.PostprocessPath); err == nil {
				var pp struct {
					Summary     string   `json:"summary"`
					FocusPoints []string `json:"focus_points"`
					Strategy    string   `json:"strategy"`
					TopicCount  int      `json:"topic_count"`
				}
				_ = json.Unmarshal(data, &pp)
				result = &PostProcessResult{Summary: pp.Summary, FocusPoints: pp.FocusPoints, Strategy: pp.Strategy, TopicCount: pp.TopicCount}
			}
			if category != nil && result != nil {
				preview := RenderSessionNotePreview(job.CreatedAt, category.Name, []string{category.SpeakerA, category.SpeakerB}, result, labelled, 1200)
				_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "awaiting_confirmation", "can_confirm_vault": true, "filename": job.Filename, "vault_preview": preview})
			}
			return
		}

		job.StageStatuses[string(JobStageVault)] = "running"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "running"})

		if category == nil {
			job.Status = JobStatusFailed
			job.Error = "Category not found"
			job.StageStatuses[string(JobStageVault)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "failed", "message": "Category not found"})
			return
		}

		var labelled []LabelledTurn
		var result *PostProcessResult
		if data, err := os.ReadFile(job.AlignedPath); err == nil {
			_ = json.Unmarshal(data, &labelled)
		}
		if data, err := os.ReadFile(job.PostprocessPath); err == nil {
			var pp struct {
				Summary     string   `json:"summary"`
				FocusPoints []string `json:"focus_points"`
				Strategy    string   `json:"strategy"`
				TopicCount  int      `json:"topic_count"`
			}
			_ = json.Unmarshal(data, &pp)
			result = &PostProcessResult{Summary: pp.Summary, FocusPoints: pp.FocusPoints, Strategy: pp.Strategy, TopicCount: pp.TopicCount}
		}
		writer := VaultWriter{APIURL: cfg.ObsidianAPIURL, APIKey: cfg.ObsidianAPIKey}
		sessionDate := job.CreatedAt
		filename := job.Filename
		if filename == "" {
			filename = fmt.Sprintf("%s %s.md", sessionDate.Format("2006-01-02"), category.Name)
		}
		noteContent := RenderSessionNote(sessionDate, category.Name, []string{category.SpeakerA, category.SpeakerB}, result, labelled)
		if err := writer.WriteNote(category.VaultPath, filename, noteContent); err != nil {
			job.Status = JobStatusFailed
			job.Error = err.Error()
			job.StageStatuses[string(JobStageVault)] = "failed"
			_ = a.saveJob(&job)
			_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "failed", "message": err.Error()})
			return
		}
		if category.FocusNotePath != "" && result != nil {
			focusSection := RenderFocusSection(sessionDate, result.FocusPoints)
			exists, _ := writer.NoteExists(category.FocusNotePath)
			if !exists {
				_ = writer.WriteNote("", category.FocusNotePath, "# "+category.Name+" Focus\n"+focusSection)
			} else {
				_ = writer.AppendToNote(category.FocusNotePath, focusSection)
			}
		}
		job.Status = JobStatusComplete
		job.StageStatuses[string(JobStageVault)] = "done"
		_ = a.saveJob(&job)
		_ = a.Bus.Publish(job.ID, map[string]any{"stage": "vault", "status": "done", "obsidian_uri": "/jobs/" + job.ID + "/open-in-obsidian", "summary": ifEmpty(result.Summary), "focus_points": ifResultFocus(result)})
		_ = a.Bus.Publish(job.ID, "done")
		return
	}
}

func buildSpeakerMap(turns []SpeakerTurn, speakerA, speakerB string) map[string]string {
	var speakers []string
	for _, turn := range turns {
		found := false
		for _, existing := range speakers {
			if existing == turn.Speaker {
				found = true
				break
			}
		}
		if !found {
			speakers = append(speakers, turn.Speaker)
		}
		if len(speakers) == 2 {
			break
		}
	}
	m := map[string]string{}
	if len(speakers) > 0 {
		m[speakers[0]] = speakerA
	}
	if len(speakers) > 1 {
		m[speakers[1]] = speakerB
	}
	return m
}

func findCategory(categories []Category, id string) *Category {
	for i := range categories {
		if categories[i].ID == id {
			return &categories[i]
		}
	}
	return nil
}

func renderAlignedPreview(turns []LabelledTurn, limit int) string {
	if limit <= 0 {
		limit = 5
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

func mustJSON(v any) []byte {
	data, _ := json.Marshal(v)
	return data
}

func ifEmpty(s string) string {
	return s
}

func ifResultFocus(result *PostProcessResult) []string {
	if result == nil {
		return nil
	}
	return result.FocusPoints
}
