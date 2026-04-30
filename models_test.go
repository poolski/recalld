package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gotest.tools/v3/assert"
	"gotest.tools/v3/assert/cmp"
)

func TestConfigSaveAndLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	app, err := NewApp(filepath.Join(dir, "config.json"), filepath.Join(dir, "jobs"), filepath.Join("recalld", "static"))
	assert.NilError(t, err)

	cfg := DefaultConfig()
	cfg.VaultName = "Work"
	cfg.LLMModel = "qwen/qwen3-4b"
	cfg.Categories = []Category{{ID: "cat-1", Name: "Coaching", VaultPath: "Notes/Sessions", SpeakerA: "You", SpeakerB: "Coach"}}

	assert.NilError(t, app.SaveConfig(cfg))
	loaded, err := app.LoadConfig()
	assert.NilError(t, err)
	assert.Assert(t, cmp.Equal(loaded.VaultName, "Work"))
	assert.Assert(t, cmp.Equal(loaded.LLMModel, "qwen/qwen3-4b"))
	assert.Assert(t, cmp.Equal(len(loaded.Categories), 1))
	assert.Assert(t, cmp.Equal(loaded.Categories[0].Name, "Coaching"))
}

func TestLoadJobInfersLegacyStageStatuses(t *testing.T) {
	dir := t.TempDir()
	job := Job{
		ID:               "job-1",
		CategoryID:       "cat-1",
		OriginalFilename: "session.m4a",
		CreatedAt:        time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC),
		CurrentStage:     JobStageDiarise,
		Status:           JobStatusFailed,
	}
	data, err := json.Marshal(job)
	assert.NilError(t, err)
	jobDir := filepath.Join(dir, job.ID)
	assert.NilError(t, os.MkdirAll(jobDir, 0o755))
	assert.NilError(t, os.WriteFile(filepath.Join(jobDir, "job.json"), data, 0o600))

	app, err := NewApp(filepath.Join(dir, "config.json"), dir, filepath.Join("..", "..", "recalld", "static"))
	assert.NilError(t, err)
	loaded, err := app.loadJob(job.ID)
	assert.NilError(t, err)
	assert.Assert(t, cmp.Equal(loaded.StageStatuses[string(JobStageIngest)], "done"))
	assert.Assert(t, cmp.Equal(loaded.StageStatuses[string(JobStageTranscribe)], "done"))
	assert.Assert(t, cmp.Equal(loaded.StageStatuses[string(JobStageDiarise)], "failed"))
}

func TestResetJobForRerunInvalidatesDownstreamStages(t *testing.T) {
	cases := []struct {
		name                   string
		stage                  JobStage
		wantWavPath            bool
		wantTranscriptPath     bool
		wantDiarisationPath    bool
		wantAlignedPath        bool
		wantPostprocessPath    bool
		wantSpeakerFieldsEmpty bool
		wantTopicCountZero     bool
		wantChunkStrategyEmpty bool
	}{
		{
			name:                   "ingest",
			stage:                  JobStageIngest,
			wantWavPath:            false,
			wantTranscriptPath:     false,
			wantDiarisationPath:    false,
			wantAlignedPath:        false,
			wantPostprocessPath:    false,
			wantSpeakerFieldsEmpty: true,
			wantTopicCountZero:     true,
			wantChunkStrategyEmpty: true,
		},
		{
			name:                   "transcribe",
			stage:                  JobStageTranscribe,
			wantWavPath:            true,
			wantTranscriptPath:     false,
			wantDiarisationPath:    false,
			wantAlignedPath:        false,
			wantPostprocessPath:    false,
			wantSpeakerFieldsEmpty: true,
			wantTopicCountZero:     true,
			wantChunkStrategyEmpty: true,
		},
		{
			name:                   "diarise",
			stage:                  JobStageDiarise,
			wantWavPath:            true,
			wantTranscriptPath:     true,
			wantDiarisationPath:    false,
			wantAlignedPath:        false,
			wantPostprocessPath:    false,
			wantSpeakerFieldsEmpty: true,
			wantTopicCountZero:     true,
			wantChunkStrategyEmpty: true,
		},
		{
			name:                   "align",
			stage:                  JobStageAlign,
			wantWavPath:            true,
			wantTranscriptPath:     true,
			wantDiarisationPath:    true,
			wantAlignedPath:        false,
			wantPostprocessPath:    false,
			wantSpeakerFieldsEmpty: true,
			wantTopicCountZero:     true,
			wantChunkStrategyEmpty: true,
		},
		{
			name:                   "postprocess",
			stage:                  JobStagePostprocess,
			wantWavPath:            true,
			wantTranscriptPath:     true,
			wantDiarisationPath:    true,
			wantAlignedPath:        true,
			wantPostprocessPath:    false,
			wantSpeakerFieldsEmpty: false,
			wantTopicCountZero:     true,
			wantChunkStrategyEmpty: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			job := NewJob("cat-1", "session.m4a")
			job.CurrentStage = JobStageVault
			job.Status = JobStatusFailed
			job.WavPath = "audio.wav"
			job.TranscriptPath = "transcript.json"
			job.DiarisationPath = "diarisation.json"
			job.AlignedPath = "aligned.json"
			job.PostprocessPath = "postprocess.json"
			job.Speaker00 = "You"
			job.Speaker01 = "Coach"
			job.TopicCount = 3
			job.ChunkStrategy = "single"
			job.StageStatuses[string(JobStageIngest)] = "done"
			job.StageStatuses[string(JobStageTranscribe)] = "done"
			job.StageStatuses[string(JobStageDiarise)] = "done"
			job.StageStatuses[string(JobStageAlign)] = "done"
			job.StageStatuses[string(JobStagePostprocess)] = "done"
			job.StageStatuses[string(JobStageVault)] = "failed"

			stage := tc.stage
			ResetJobForRerun(&job, false, &stage)

			assert.Assert(t, cmp.Equal(job.CurrentStage, tc.stage))
			assert.Assert(t, cmp.Equal(job.WavPath != "", tc.wantWavPath))
			assert.Assert(t, cmp.Equal(job.TranscriptPath != "", tc.wantTranscriptPath))
			assert.Assert(t, cmp.Equal(job.DiarisationPath != "", tc.wantDiarisationPath))
			assert.Assert(t, cmp.Equal(job.AlignedPath != "", tc.wantAlignedPath))
			assert.Assert(t, cmp.Equal(job.PostprocessPath != "", tc.wantPostprocessPath))
			if tc.wantSpeakerFieldsEmpty {
				assert.Assert(t, cmp.Equal(job.Speaker00, ""))
				assert.Assert(t, cmp.Equal(job.Speaker01, ""))
			} else {
				assert.Assert(t, cmp.Equal(job.Speaker00, "You"))
				assert.Assert(t, cmp.Equal(job.Speaker01, "Coach"))
			}
			if tc.wantTopicCountZero {
				assert.Assert(t, cmp.Equal(job.TopicCount, 0))
			}
			if tc.wantChunkStrategyEmpty {
				assert.Assert(t, cmp.Equal(job.ChunkStrategy, ""))
			}
			for _, stageName := range StageNames {
				expected := "done"
				if stageOrder[stageName] >= stageOrder[string(tc.stage)] {
					expected = "pending"
				}
				assert.Assert(t, cmp.Equal(job.StageStatuses[stageName], expected))
			}
		})
	}
}
