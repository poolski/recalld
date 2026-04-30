package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"gotest.tools/v3/assert"
	"gotest.tools/v3/assert/cmp"
)

func newTestApp(t *testing.T) *App {
	t.Helper()
	dir := t.TempDir()
	app, err := NewApp(filepath.Join(dir, "config.json"), filepath.Join(dir, "jobs"), filepath.Join("recalld", "static"))
	assert.NilError(t, err)
	app.PipelineLauncher = func(jobID, sourcePath string, cfg Config) {}
	cfg := DefaultConfig()
	cfg.Categories = []Category{
		{ID: "cat-1", Name: "Coaching", VaultPath: "Notes/Sessions", SpeakerA: "You", SpeakerB: "Coach"},
	}
	assert.NilError(t, app.SaveConfig(cfg))
	return app
}

func TestIndexRouteRendersUploadPage(t *testing.T) {
	app := newTestApp(t)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()

	app.Handler().ServeHTTP(rec, req)

	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	assert.Assert(t, strings.Contains(rec.Body.String(), "Start a new recording"))
	assert.Assert(t, strings.Contains(rec.Body.String(), "Coaching"))
}

func TestJobRowAndConfirmDeleteRoutes(t *testing.T) {
	app := newTestApp(t)
	job, err := app.createJob("cat-1", "session.m4a")
	assert.NilError(t, err)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/jobs/"+job.ID+"/row", nil)
	app.Handler().ServeHTTP(rec, req)
	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	assert.Assert(t, strings.Contains(rec.Body.String(), "View details"))

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/jobs/"+job.ID+"/confirm-delete", nil)
	app.Handler().ServeHTTP(rec, req)
	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	assert.Assert(t, strings.Contains(rec.Body.String(), "Delete"))
}

func TestJobStateReturnsVaultPreview(t *testing.T) {
	app := newTestApp(t)
	job, err := app.createJob("cat-1", "session.m4a")
	assert.NilError(t, err)
	alignedPath := filepath.Join(app.jobsDir(job.ID), "aligned.json")
	postPath := filepath.Join(app.jobsDir(job.ID), "postprocess.json")
	assert.NilError(t, os.WriteFile(alignedPath, mustJSON([]LabelledTurn{
		{Speaker: "You", Start: 0, End: 1, Text: "Hello"},
		{Speaker: "Coach", Start: 1, End: 2, Text: "Hi"},
	}), 0o600))
	assert.NilError(t, os.WriteFile(postPath, mustJSON(map[string]any{
		"summary":      "A productive session.",
		"focus_points": []string{"Follow up"},
		"strategy":     "single",
		"topic_count":  1,
	}), 0o600))
	job.CurrentStage = JobStageVault
	job.AlignedPath = alignedPath
	job.PostprocessPath = postPath
	job.StageStatuses[string(JobStagePostprocess)] = "done"
	job.StageStatuses[string(JobStageVault)] = "awaiting_confirmation"
	assert.NilError(t, app.saveJob(&job))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/jobs/"+job.ID+"/state", nil)
	app.Handler().ServeHTTP(rec, req)

	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	var payload map[string]any
	assert.NilError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	assert.Assert(t, cmp.Equal(payload["can_confirm_vault"], true))
	preview, _ := payload["vault_preview"].(string)
	assert.Assert(t, strings.Contains(preview, "A productive session."))
}

func TestDeleteJobRoute(t *testing.T) {
	app := newTestApp(t)
	job, err := app.createJob("cat-1", "session.m4a")
	assert.NilError(t, err)
	req := httptest.NewRequest(http.MethodDelete, "/jobs/"+job.ID, nil)
	rec := httptest.NewRecorder()
	app.Handler().ServeHTTP(rec, req)
	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	_, err = os.Stat(filepath.Join(app.ScratchRoot, job.ID))
	assert.Assert(t, os.IsNotExist(err))
}

func TestDeleteJobRouteUpdatesQueuedCount(t *testing.T) {
	app := newTestApp(t)
	job1, err := app.createJob("cat-1", "session-a.m4a")
	assert.NilError(t, err)
	job2, err := app.createJob("cat-1", "session-b.m4a")
	assert.NilError(t, err)

	req := httptest.NewRequest(http.MethodDelete, "/jobs/"+job1.ID, nil)
	rec := httptest.NewRecorder()
	app.Handler().ServeHTTP(rec, req)

	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	assert.Assert(t, strings.Contains(rec.Body.String(), `id="queued-jobs-count"`))
	assert.Assert(t, strings.Contains(rec.Body.String(), ">1<"))

	remaining, err := app.loadJob(job2.ID)
	assert.NilError(t, err)
	assert.Assert(t, cmp.Equal(remaining.ID, job2.ID))
}

func TestConfirmVaultWriteUpdatesJob(t *testing.T) {
	app := newTestApp(t)
	job, err := app.createJob("cat-1", "session.m4a")
	assert.NilError(t, err)
	job.CurrentStage = JobStageVault
	job.StageStatuses[string(JobStageVault)] = "awaiting_confirmation"
	assert.NilError(t, app.saveJob(&job))

	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/confirm-vault-write", strings.NewReader("filename=custom.md"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	app.Handler().ServeHTTP(rec, req)
	assert.Assert(t, cmp.Equal(rec.Code, http.StatusOK))
	updated, err := app.loadJob(job.ID)
	assert.NilError(t, err)
	assert.Assert(t, cmp.Equal(updated.Filename, "custom.md"))
	assert.Assert(t, cmp.Equal(updated.Status, JobStatusRunning))
}

func TestTranscriberUsesWhisperBinding(t *testing.T) {
	origOpen := openWhisperModel
	origSearchDirs := whisperModelSearchDirs
	origDownload := whisperModelDownload
	defer func() {
		openWhisperModel = origOpen
		whisperModelSearchDirs = origSearchDirs
		whisperModelDownload = origDownload
	}()

	fakeCtx := &fakeWhisperContext{
		segments: []whisper.Segment{
			{
				Num:   0,
				Start: 0,
				End:   2 * time.Second,
				Text:  "hello world",
				Tokens: []whisper.Token{
					{Id: 10, Text: "hello", Start: 0, End: 500 * time.Millisecond},
					{Id: 11, Text: "world", Start: 500 * time.Millisecond, End: time.Second},
				},
			},
		},
	}
	fakeModel := &fakeWhisperModel{ctx: fakeCtx}
	var openedPath string
	openWhisperModel = func(path string) (whisperModel, error) {
		openedPath = path
		return fakeModel, nil
	}

	modelDir := t.TempDir()
	whisperModelSearchDirs = func() []string { return []string{modelDir} }
	whisperModelDownload = func(model, token string) (string, error) {
		assert.Assert(t, cmp.Equal(model, "medium"))
		assert.Assert(t, cmp.Equal(token, ""))
		modelPath := filepath.Join(modelDir, "ggml-medium.bin")
		assert.NilError(t, os.WriteFile(modelPath, []byte("fake model"), 0o600))
		return modelPath, nil
	}

	wavPath := filepath.Join("tests", "fixtures", "conversation.wav")
	_, err := os.Stat(wavPath)
	assert.NilError(t, err)

	words, err := Transcriber{Model: "medium"}.Transcribe(wavPath)
	assert.NilError(t, err)
	assert.Assert(t, cmp.Equal(openedPath, filepath.Join(modelDir, "ggml-medium.bin")))
	assert.Assert(t, cmp.Equal(fakeCtx.processed, true))
	assert.Assert(t, len(fakeCtx.samples) > 0)
	assert.Assert(t, cmp.Equal(len(words), 2))
	assert.Assert(t, cmp.Equal(words[0].Word, "hello"))
	assert.Assert(t, cmp.Equal(words[1].Word, "world"))
	assert.Assert(t, cmp.Equal(words[0].Start, 0.0))
	assert.Assert(t, cmp.Equal(words[1].End, 2.0))
}

func TestTranscriberPrefersSegmentTextOverFragmentedTokens(t *testing.T) {
	origOpen := openWhisperModel
	origSearchDirs := whisperModelSearchDirs
	origDownload := whisperModelDownload
	defer func() {
		openWhisperModel = origOpen
		whisperModelSearchDirs = origSearchDirs
		whisperModelDownload = origDownload
	}()

	fakeCtx := &fakeWhisperContext{
		segments: []whisper.Segment{
			{
				Num:   0,
				Start: 0,
				End:   2 * time.Second,
				Text:  "Recording, there we go.",
				Tokens: []whisper.Token{
					{Id: 10, Text: "Rec", Start: 0, End: 200 * time.Millisecond},
					{Id: 11, Text: " ording", Start: 200 * time.Millisecond, End: 400 * time.Millisecond},
					{Id: 12, Text: ",", Start: 400 * time.Millisecond, End: 500 * time.Millisecond},
					{Id: 13, Text: " there", Start: 500 * time.Millisecond, End: 700 * time.Millisecond},
					{Id: 14, Text: " we", Start: 700 * time.Millisecond, End: 900 * time.Millisecond},
					{Id: 15, Text: " go", Start: 900 * time.Millisecond, End: 1100 * time.Millisecond},
					{Id: 16, Text: ".", Start: 1100 * time.Millisecond, End: 1200 * time.Millisecond},
				},
			},
		},
	}
	fakeModel := &fakeWhisperModel{ctx: fakeCtx}
	openWhisperModel = func(path string) (whisperModel, error) {
		return fakeModel, nil
	}

	modelDir := t.TempDir()
	whisperModelSearchDirs = func() []string { return []string{modelDir} }
	whisperModelDownload = func(model, token string) (string, error) {
		modelPath := filepath.Join(modelDir, "ggml-medium.bin")
		assert.NilError(t, os.WriteFile(modelPath, []byte("fake model"), 0o600))
		return modelPath, nil
	}

	wavPath := filepath.Join("tests", "fixtures", "conversation.wav")
	words, err := Transcriber{Model: "medium"}.Transcribe(wavPath)
	assert.NilError(t, err)
	assert.Assert(t, cmp.DeepEqual([]string{words[0].Word, words[1].Word, words[2].Word, words[3].Word}, []string{"Recording,", "there", "we", "go."}))
	assert.Assert(t, words[0].Start <= words[0].End)
	assert.Assert(t, words[3].End <= 2.01)
}

func TestSegmentTextToWordSegmentsWeightsLongerWordsMoreHeavily(t *testing.T) {
	segment := whisper.Segment{
		Start: 0,
		End:   2 * time.Second,
		Text:  "Recording, there we go.",
	}

	words := segmentTextToWordSegments(segment)

	assert.Assert(t, cmp.Equal(len(words), 4))
	assert.Assert(t, words[0].End > 0.5)
	assert.Assert(t, cmp.Equal(words[3].End, 2.0))
}

func TestAlignRespectsSpeakerChangesWithinSegment(t *testing.T) {
	words := []WordSegment{
		{Segment: 1, Start: 0.0, End: 0.4, Word: "Hello"},
		{Segment: 1, Start: 0.4, End: 0.8, Word: "there"},
		{Segment: 1, Start: 0.8, End: 1.2, Word: "General"},
		{Segment: 1, Start: 1.2, End: 1.6, Word: "Kenobi"},
	}
	turns := []SpeakerTurn{
		{Start: 0.0, End: 1.0, Speaker: "SPEAKER_00"},
		{Start: 1.0, End: 2.0, Speaker: "SPEAKER_01"},
	}

	labelled := Align(words, turns, nil)

	assert.Assert(t, cmp.Equal(len(labelled), 2))
	assert.Assert(t, cmp.Equal(labelled[0].Speaker, "SPEAKER_00"))
	assert.Assert(t, cmp.Equal(labelled[0].Text, "Hello there General"))
	assert.Assert(t, cmp.Equal(labelled[1].Speaker, "SPEAKER_01"))
	assert.Assert(t, cmp.Equal(labelled[1].Text, "Kenobi"))
}

func TestAlignTreatsDashSeparatedChunksAsTurnBoundaries(t *testing.T) {
	words := []WordSegment{
		{Segment: 1, Start: 0.0, End: 0.2, Word: "-"},
		{Segment: 1, Start: 0.2, End: 0.6, Word: "Recording,"},
		{Segment: 1, Start: 0.6, End: 1.0, Word: "there"},
		{Segment: 1, Start: 1.0, End: 1.4, Word: "we"},
		{Segment: 1, Start: 1.4, End: 1.8, Word: "go."},
		{Segment: 1, Start: 1.8, End: 2.0, Word: "-"},
		{Segment: 1, Start: 2.0, End: 2.4, Word: "Beautiful."},
	}
	turns := []SpeakerTurn{
		{Start: 0.0, End: 3.0, Speaker: "SPEAKER_00"},
		{Start: 3.0, End: 5.0, Speaker: "SPEAKER_01"},
	}

	labelled := Align(words, turns, nil)

	assert.Assert(t, cmp.Equal(len(labelled), 2))
	assert.Assert(t, cmp.Equal(labelled[0].Speaker, "SPEAKER_00"))
	assert.Assert(t, cmp.Equal(labelled[0].Text, "- Recording, there we go."))
	assert.Assert(t, cmp.Equal(labelled[1].Speaker, "SPEAKER_01"))
	assert.Assert(t, cmp.Equal(labelled[1].Text, "- Beautiful."))
}

func TestAlignSplitsMixedSpeakersWithinSegment(t *testing.T) {
	words := []WordSegment{
		{Segment: 1, Start: 0.0, End: 0.5, Word: "Recording,"},
		{Segment: 1, Start: 0.5, End: 1.0, Word: "there"},
		{Segment: 1, Start: 1.0, End: 1.5, Word: "we"},
		{Segment: 1, Start: 1.5, End: 2.0, Word: "go."},
		{Segment: 1, Start: 2.0, End: 2.5, Word: "Beautiful."},
		{Segment: 1, Start: 2.5, End: 3.0, Word: "Okay,"},
		{Segment: 1, Start: 3.0, End: 3.5, Word: "let's"},
		{Segment: 1, Start: 3.5, End: 4.0, Word: "hope"},
		{Segment: 1, Start: 4.0, End: 4.5, Word: "I"},
		{Segment: 1, Start: 4.5, End: 5.0, Word: "don't"},
		{Segment: 1, Start: 5.0, End: 5.5, Word: "run"},
		{Segment: 1, Start: 5.5, End: 6.0, Word: "out"},
		{Segment: 1, Start: 6.0, End: 6.5, Word: "of"},
		{Segment: 1, Start: 6.5, End: 7.0, Word: "space."},
	}
	turns := []SpeakerTurn{
		{Start: 0.0, End: 2.0, Speaker: "SPEAKER_00"},
		{Start: 2.0, End: 7.0, Speaker: "SPEAKER_01"},
	}

	labelled := Align(words, turns, nil)

	assert.Assert(t, cmp.Equal(len(labelled), 2))
	assert.Assert(t, cmp.Equal(labelled[0].Speaker, "SPEAKER_00"))
	assert.Assert(t, cmp.Equal(labelled[0].Text, "Recording, there we go."))
	assert.Assert(t, cmp.Equal(labelled[1].Speaker, "SPEAKER_01"))
	assert.Assert(t, cmp.Equal(labelled[1].Text, "Beautiful. Okay, let's hope I don't run out of space."))
}

func TestResolveWhisperModelPathMissing(t *testing.T) {
	origSearchDirs := whisperModelSearchDirs
	origDownload := whisperModelDownload
	defer func() {
		whisperModelSearchDirs = origSearchDirs
		whisperModelDownload = origDownload
	}()

	whisperModelSearchDirs = func() []string { return []string{t.TempDir()} }
	whisperModelDownload = func(model, token string) (string, error) {
		return filepath.Join(t.TempDir(), "ggml-"+model+".bin"), nil
	}

	_, err := resolveWhisperModelPath("medium", "")
	assert.Assert(t, err != nil)
	assert.Assert(t, strings.Contains(err.Error(), "whisper model \"medium\" not found"))
}

type fakeWhisperModel struct {
	ctx *fakeWhisperContext
}

func (m *fakeWhisperModel) Close() error {
	return nil
}

func (m *fakeWhisperModel) NewContext() (whisperContext, error) {
	return m.ctx, nil
}

type fakeWhisperContext struct {
	segments  []whisper.Segment
	next      int
	processed bool
	samples   []float32
}

func (c *fakeWhisperContext) SetTranslate(bool) {}

func (c *fakeWhisperContext) SetThreads(uint) {}

func (c *fakeWhisperContext) SetSplitOnWord(bool) {}

func (c *fakeWhisperContext) SetTokenTimestamps(bool) {}

func (c *fakeWhisperContext) Process(samples []float32, _ whisper.EncoderBeginCallback, _ whisper.SegmentCallback, _ whisper.ProgressCallback) error {
	c.processed = true
	c.samples = append([]float32(nil), samples...)
	return nil
}

func (c *fakeWhisperContext) NextSegment() (whisper.Segment, error) {
	if c.next >= len(c.segments) {
		return whisper.Segment{}, io.EOF
	}
	segment := c.segments[c.next]
	c.next++
	return segment, nil
}

func (c *fakeWhisperContext) IsText(token whisper.Token) bool {
	return token.Text != ""
}
