package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type JobStage string

const (
	JobStageIngest      JobStage = "ingest"
	JobStageTranscribe  JobStage = "transcribe"
	JobStageDiarise     JobStage = "diarise"
	JobStageAlign       JobStage = "align"
	JobStagePostprocess JobStage = "postprocess"
	JobStageVault       JobStage = "vault"
)

var StageNames = []string{
	string(JobStageIngest),
	string(JobStageTranscribe),
	string(JobStageDiarise),
	string(JobStageAlign),
	string(JobStagePostprocess),
	string(JobStageVault),
}

var stageOrder = map[string]int{
	string(JobStageIngest):      0,
	string(JobStageTranscribe):  1,
	string(JobStageDiarise):     2,
	string(JobStageAlign):       3,
	string(JobStagePostprocess): 4,
	string(JobStageVault):       5,
}

type JobStatus string

const (
	JobStatusPending  JobStatus = "pending"
	JobStatusRunning  JobStatus = "running"
	JobStatusFailed   JobStatus = "failed"
	JobStatusComplete JobStatus = "complete"
)

type Category struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	VaultPath     string `json:"vault_path"`
	FocusNotePath string `json:"focus_note_path,omitempty"`
	SpeakerA      string `json:"speaker_a"`
	SpeakerB      string `json:"speaker_b"`
}

type Config struct {
	VaultName            string     `json:"vault_name"`
	ObsidianAPIURL       string     `json:"obsidian_api_url"`
	ObsidianAPIKey       string     `json:"obsidian_api_key"`
	LLMBaseURL           string     `json:"llm_base_url"`
	LLMModel             string     `json:"llm_model"`
	LLMContextHeadroom   float64    `json:"llm_context_headroom"`
	LogLevel             string     `json:"log_level"`
	LastUsedCategory     string     `json:"last_used_category,omitempty"`
	Categories           []Category `json:"categories"`
	WhisperModel         string     `json:"whisper_model"`
	WhisperBaseURL       string     `json:"whisper_base_url"`
	HuggingFaceToken     string     `json:"huggingface_token"`
	ScratchRetentionDays int        `json:"scratch_retention_days"`
}

func DefaultConfig() Config {
	return Config{
		VaultName:            "Personal",
		ObsidianAPIURL:       "https://127.0.0.1:27124",
		LLMBaseURL:           "http://localhost:1234/v1",
		LLMContextHeadroom:   0.8,
		LogLevel:             "info",
		WhisperModel:         "small",
		WhisperBaseURL:       "http://localhost:8081",
		ScratchRetentionDays: 30,
	}
}

func (c *Config) Normalize() {
	if c.VaultName == "" {
		c.VaultName = "Personal"
	}
	if c.ObsidianAPIURL == "" {
		c.ObsidianAPIURL = "https://127.0.0.1:27124"
	}
	if c.LLMBaseURL == "" {
		c.LLMBaseURL = "http://localhost:1234/v1"
	}
	if c.LLMContextHeadroom <= 0 || c.LLMContextHeadroom > 1 {
		c.LLMContextHeadroom = 0.8
	}
	if c.LogLevel == "" {
		c.LogLevel = "info"
	}
	if c.WhisperModel == "" {
		c.WhisperModel = "small"
	}
	if c.WhisperBaseURL == "" {
		c.WhisperBaseURL = "http://localhost:8081"
	}
	if c.ScratchRetentionDays <= 0 {
		c.ScratchRetentionDays = 30
	}
}

type Job struct {
	ID               string            `json:"id"`
	CategoryID       string            `json:"category_id"`
	OriginalFilename string            `json:"original_filename"`
	CreatedAt        time.Time         `json:"created_at"`
	CurrentStage     JobStage          `json:"current_stage"`
	Status           JobStatus         `json:"status"`
	Error            string            `json:"error,omitempty"`
	StageStatuses    map[string]string `json:"stage_statuses"`
	WavPath          string            `json:"wav_path,omitempty"`
	TranscriptPath   string            `json:"transcript_path,omitempty"`
	DiarisationPath  string            `json:"diarisation_path,omitempty"`
	AlignedPath      string            `json:"aligned_path,omitempty"`
	PostprocessPath  string            `json:"postprocess_path,omitempty"`
	Speaker00        string            `json:"speaker_00,omitempty"`
	Speaker01        string            `json:"speaker_01,omitempty"`
	TopicCount       int               `json:"topic_count,omitempty"`
	ChunkStrategy    string            `json:"chunk_strategy,omitempty"`
	Filename         string            `json:"filename,omitempty"`
}

func DefaultStageStatuses() map[string]string {
	statuses := make(map[string]string, len(StageNames))
	for _, stage := range StageNames {
		statuses[stage] = "pending"
	}
	return statuses
}

func inferStageStatuses(currentStage JobStage, status JobStatus) map[string]string {
	statuses := DefaultStageStatuses()
	currentIndex := stageOrder[string(currentStage)]
	for _, stage := range StageNames[:currentIndex] {
		statuses[stage] = "done"
	}
	switch status {
	case JobStatusComplete:
		for _, stage := range StageNames {
			statuses[stage] = "done"
		}
	case JobStatusFailed:
		statuses[string(currentStage)] = "failed"
	case JobStatusRunning:
		statuses[string(currentStage)] = "running"
	}
	return statuses
}

func (j *Job) ensureDefaults() {
	if j.CreatedAt.IsZero() {
		j.CreatedAt = time.Now().UTC()
	}
	if j.CurrentStage == "" {
		j.CurrentStage = JobStageIngest
	}
	if j.Status == "" {
		j.Status = JobStatusPending
	}
	if j.StageStatuses == nil || len(j.StageStatuses) == 0 {
		j.StageStatuses = inferStageStatuses(j.CurrentStage, j.Status)
	}
	if j.TopicCount < 0 {
		j.TopicCount = 0
	}
}

func (j *Job) UnmarshalJSON(data []byte) error {
	type alias Job
	var v alias
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	*j = Job(v)
	if j.StageStatuses == nil || len(j.StageStatuses) == 0 {
		j.StageStatuses = inferStageStatuses(j.CurrentStage, j.Status)
	}
	j.ensureDefaults()
	return nil
}

func (j Job) MarshalJSON() ([]byte, error) {
	type alias Job
	return json.Marshal(alias(j))
}

func NewJob(categoryID, originalFilename string) Job {
	job := Job{
		ID:               newID(),
		CategoryID:       categoryID,
		OriginalFilename: originalFilename,
		CreatedAt:        time.Now().UTC(),
		CurrentStage:     JobStageIngest,
		Status:           JobStatusPending,
		StageStatuses:    DefaultStageStatuses(),
	}
	return job
}

func CanRestartFromStage(job Job, stage JobStage) bool {
	switch stage {
	case JobStageIngest:
		return true
	case JobStageTranscribe, JobStageDiarise:
		return job.WavPath != ""
	case JobStageAlign:
		return job.TranscriptPath != "" && job.DiarisationPath != ""
	case JobStagePostprocess:
		return job.AlignedPath != ""
	case JobStageVault:
		return job.AlignedPath != "" && job.PostprocessPath != ""
	default:
		return false
	}
}

func ResetJobForRerun(job *Job, fromStart bool, restartStage *JobStage) {
	job.Status = JobStatusPending
	job.Error = ""

	if fromStart {
		job.CurrentStage = JobStageIngest
		job.StageStatuses = DefaultStageStatuses()
		job.WavPath = ""
		job.TranscriptPath = ""
		job.DiarisationPath = ""
		job.AlignedPath = ""
		job.PostprocessPath = ""
		job.Speaker00 = ""
		job.Speaker01 = ""
		job.TopicCount = 0
		job.ChunkStrategy = ""
		return
	}

	if restartStage != nil {
		job.CurrentStage = *restartStage
		for _, stage := range StageNames {
			if stageOrder[stage] >= stageOrder[string(*restartStage)] {
				job.StageStatuses[stage] = "pending"
			}
		}
		switch *restartStage {
		case JobStageIngest:
			job.WavPath = ""
			job.TranscriptPath = ""
			job.DiarisationPath = ""
			job.AlignedPath = ""
			job.PostprocessPath = ""
			job.Speaker00 = ""
			job.Speaker01 = ""
			job.TopicCount = 0
			job.ChunkStrategy = ""
		case JobStageTranscribe:
			job.TranscriptPath = ""
			job.DiarisationPath = ""
			job.AlignedPath = ""
			job.PostprocessPath = ""
			job.Speaker00 = ""
			job.Speaker01 = ""
			job.TopicCount = 0
			job.ChunkStrategy = ""
		case JobStageDiarise:
			job.DiarisationPath = ""
			job.AlignedPath = ""
			job.PostprocessPath = ""
			job.Speaker00 = ""
			job.Speaker01 = ""
			job.TopicCount = 0
			job.ChunkStrategy = ""
		case JobStageAlign:
			job.AlignedPath = ""
			job.PostprocessPath = ""
			job.Speaker00 = ""
			job.Speaker01 = ""
			job.TopicCount = 0
			job.ChunkStrategy = ""
		case JobStagePostprocess:
			job.PostprocessPath = ""
			job.TopicCount = 0
			job.ChunkStrategy = ""
		}
		return
	}

	if job.StageStatuses == nil {
		job.StageStatuses = DefaultStageStatuses()
	}
	job.StageStatuses[string(job.CurrentStage)] = "pending"
}

func ListIncompleteJobs(root string) ([]Job, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	var jobs []Job
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		jobPath := filepath.Join(root, entry.Name(), "job.json")
		data, err := os.ReadFile(jobPath)
		if err != nil {
			continue
		}
		var job Job
		if err := json.Unmarshal(data, &job); err != nil {
			continue
		}
		job.ensureDefaults()
		if job.Status != JobStatusComplete {
			jobs = append(jobs, job)
		}
	}

	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].CreatedAt.After(jobs[j].CreatedAt)
	})
	return jobs, nil
}

func newID() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), os.Getpid())
}
