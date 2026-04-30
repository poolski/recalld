package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
	"unicode"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type WordSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Word    string  `json:"word"`
	Segment int     `json:"segment,omitempty"`
}

type SpeakerTurn struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}

type LabelledTurn struct {
	Speaker string  `json:"speaker"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
}

type IngestError struct{ Err error }

func (e IngestError) Error() string { return e.Err.Error() }

type DiariseError struct{ Err error }

func (e DiariseError) Error() string { return e.Err.Error() }

type TranscribeError struct{ Err error }

func (e TranscribeError) Error() string { return e.Err.Error() }

func Ingest(source, jobDir string) (string, error) {
	srcInfo, err := os.Stat(source)
	if err != nil {
		return "", IngestError{Err: err}
	}
	if srcInfo.IsDir() {
		return "", IngestError{Err: fmt.Errorf("source is a directory")}
	}

	if strings.EqualFold(filepath.Ext(source), ".wav") {
		dest := filepath.Join(jobDir, filepath.Base(source))
		in, err := os.Open(source)
		if err != nil {
			return "", IngestError{Err: err}
		}
		defer in.Close()
		out, err := os.Create(dest)
		if err != nil {
			return "", IngestError{Err: err}
		}
		defer out.Close()
		if _, err := io.Copy(out, in); err != nil {
			return "", IngestError{Err: err}
		}
		return dest, nil
	}

	dest := filepath.Join(jobDir, strings.TrimSuffix(filepath.Base(source), filepath.Ext(source))+".wav")
	cmd := exec.Command("ffmpeg", "-y", "-i", source, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dest)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", IngestError{Err: fmt.Errorf("ffmpeg failed: %w: %s", err, strings.TrimSpace(string(output)))}
	}
	return dest, nil
}

type Transcriber struct {
	Model            string
	HuggingFaceToken string
	Log              func(level int, text string)
}

func (t Transcriber) Transcribe(wavPath string) ([]WordSegment, error) {
	if t.Log != nil {
		prev := whisper.SetLogSink(func(level int, text string) {
			t.Log(level, strings.TrimSpace(text))
		})
		defer whisper.SetLogSink(prev)
	}

	modelPath, err := resolveWhisperModelPath(t.Model, t.HuggingFaceToken)
	if err != nil {
		return nil, TranscribeError{Err: err}
	}

	model, err := openWhisperModel(modelPath)
	if err != nil {
		return nil, TranscribeError{Err: err}
	}
	defer model.Close()

	context, err := model.NewContext()
	if err != nil {
		return nil, TranscribeError{Err: err}
	}
	context.SetTranslate(false)
	context.SetThreads(uint(runtime.NumCPU()))
	context.SetSplitOnWord(true)
	context.SetTokenTimestamps(true)

	whisperPath, cleanup, err := ensurePCM16MonoWav(wavPath)
	if err != nil {
		return nil, TranscribeError{Err: err}
	}
	defer cleanup()

	samples, _, err := readPCM16MonoWav(whisperPath)
	if err != nil {
		return nil, TranscribeError{Err: err}
	}
	if err := context.Process(samples, nil, nil, nil); err != nil {
		return nil, TranscribeError{Err: err}
	}
	return wordSegmentsFromContext(context)
}

type whisperModel interface {
	Close() error
	NewContext() (whisperContext, error)
}

type whisperContext interface {
	SetTranslate(bool)
	SetThreads(uint)
	SetSplitOnWord(bool)
	SetTokenTimestamps(bool)
	Process([]float32, whisper.EncoderBeginCallback, whisper.SegmentCallback, whisper.ProgressCallback) error
	NextSegment() (whisper.Segment, error)
	IsText(whisper.Token) bool
}

type whisperModelAdapter struct {
	model whisper.Model
}

func (m whisperModelAdapter) Close() error {
	return m.model.Close()
}

func (m whisperModelAdapter) NewContext() (whisperContext, error) {
	ctx, err := m.model.NewContext()
	if err != nil {
		return nil, err
	}
	return ctx, nil
}

var openWhisperModel = func(path string) (whisperModel, error) {
	model, err := whisper.New(path)
	if err != nil {
		return nil, err
	}
	return whisperModelAdapter{model: model}, nil
}

var whisperModelSearchDirs = func() []string {
	dirs := []string{
		defaultWhisperModelDir(),
		filepath.Join("third_party", "whispercpp", "models"),
		"models",
	}
	return dirs
}

var whisperModelDownload = downloadWhisperModel

func resolveWhisperModelPath(model, token string) (string, error) {
	if model == "" {
		return "", fmt.Errorf("whisper model is not configured")
	}

	candidates := whisperModelCandidates(model)
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate, nil
		}
		for _, dir := range whisperModelSearchDirs() {
			path := filepath.Join(dir, candidate)
			if fileExists(path) {
				return path, nil
			}
		}
	}

	if !strings.ContainsAny(model, "/\\") {
		path, err := whisperModelDownload(model, token)
		if err != nil {
			return "", err
		}
		if fileExists(path) {
			return path, nil
		}
	}

	return "", fmt.Errorf("whisper model %q not found; download a ggml model and place it in one of: %s", model, strings.Join(whisperModelSearchDirs(), ", "))
}

func whisperModelCandidates(model string) []string {
	candidates := []string{model}
	if filepath.Ext(model) != ".bin" {
		candidates = append(candidates,
			model+".bin",
			"ggml-"+model+".bin",
			"ggml-"+model+".en.bin",
		)
	}
	return candidates
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func defaultWhisperModelDir() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".local", "share", "recalld", "models")
	}
	return filepath.Join(".", "models")
}

func downloadWhisperModel(model, token string) (string, error) {
	outDir := defaultWhisperModelDir()
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", err
	}
	filename := whisperModelFilename(model)
	target := filepath.Join(outDir, filename)
	if fileExists(target) {
		return target, nil
	}

	url := whisperModelURL(filename)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download model %q: unexpected HTTP %d", model, resp.StatusCode)
	}

	tmp, err := os.CreateTemp(outDir, filename+".*.tmp")
	if err != nil {
		return "", err
	}
	tmpPath := tmp.Name()
	defer func() {
		tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(tmpPath, target); err != nil {
		return "", err
	}
	return target, nil
}

func whisperModelFilename(model string) string {
	if filepath.Ext(model) == ".bin" && strings.HasPrefix(model, "ggml-") {
		return model
	}
	if filepath.Ext(model) == ".bin" {
		return "ggml-" + filepath.Base(model)
	}
	return "ggml-" + model + ".bin"
}

func whisperModelURL(filename string) string {
	return "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + filename
}

func ensurePCM16MonoWav(wavPath string) (string, func(), error) {
	if canonicalWav(wavPath) {
		return wavPath, func() {}, nil
	}

	tempDir, err := os.MkdirTemp(filepath.Dir(wavPath), "whisper-wav-")
	if err != nil {
		return "", nil, err
	}
	outputPath := filepath.Join(tempDir, "input.wav")
	cmd := exec.Command("ffmpeg", "-y", "-i", wavPath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", outputPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		_ = os.RemoveAll(tempDir)
		return "", nil, fmt.Errorf("ffmpeg failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return outputPath, func() { _ = os.RemoveAll(tempDir) }, nil
}

func canonicalWav(wavPath string) bool {
	f, err := os.Open(wavPath)
	if err != nil {
		return false
	}
	defer f.Close()

	header := make([]byte, 44)
	if _, err := io.ReadFull(f, header); err != nil {
		return false
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return false
	}
	audioFormat := binary.LittleEndian.Uint16(header[20:22])
	channels := binary.LittleEndian.Uint16(header[22:24])
	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])
	return audioFormat == 1 && channels == 1 && bitsPerSample == 16
}

func readPCM16MonoWav(wavPath string) ([]float32, int, error) {
	f, err := os.Open(wavPath)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	header := make([]byte, 44)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, 0, fmt.Errorf("invalid wav header: %w", err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("unsupported audio format")
	}

	audioFormat := binary.LittleEndian.Uint16(header[20:22])
	channels := binary.LittleEndian.Uint16(header[22:24])
	sampleRate := binary.LittleEndian.Uint32(header[24:28])
	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])
	if audioFormat != 1 || channels != 1 || bitsPerSample != 16 {
		return nil, 0, fmt.Errorf("expected 16-bit mono PCM wav")
	}

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, 0, err
	}
	if len(data)%2 != 0 {
		return nil, 0, fmt.Errorf("invalid wav data length")
	}
	samples := make([]int16, len(data)/2)
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &samples); err != nil {
		return nil, 0, err
	}
	frames := make([]float32, len(samples))
	for i, sample := range samples {
		frames[i] = float32(sample) / 32768.0
	}
	return frames, int(sampleRate), nil
}

func wordSegmentsFromContext(ctx whisperContext) ([]WordSegment, error) {
	var words []WordSegment
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			return words, nil
		}
		if err != nil {
			return nil, err
		}
		words = append(words, segmentToWordSegments(ctx, segment)...)
	}
}

func segmentToWordSegments(ctx whisperContext, segment whisper.Segment) []WordSegment {
	if text := strings.TrimSpace(segment.Text); text != "" {
		return segmentTextToWordSegments(segment)
	}

	if len(segment.Tokens) > 0 {
		words := make([]WordSegment, 0, len(segment.Tokens))
		for _, token := range segment.Tokens {
			if !ctx.IsText(token) {
				continue
			}
			word := strings.TrimSpace(token.Text)
			if word == "" {
				continue
			}
			start := token.Start.Seconds()
			end := token.End.Seconds()
			if end <= start {
				end = start + 0.4
			}
			words = append(words, WordSegment{
				Start:   start,
				End:     end,
				Word:    word,
				Segment: segment.Num + 1,
			})
		}
		if len(words) > 0 {
			return words
		}
	}

	return segmentTextToWordSegments(segment)
}

func segmentTextToWordSegments(segment whisper.Segment) []WordSegment {
	parts := strings.Fields(segment.Text)
	if len(parts) == 0 {
		return nil
	}
	start := segment.Start.Seconds()
	end := segment.End.Seconds()
	if end <= start {
		end = start + float64(len(parts))*0.4
	}
	weights := make([]float64, len(parts))
	totalWeight := 0.0
	for i, part := range parts {
		weight := wordWeight(part)
		weights[i] = weight
		totalWeight += weight
	}
	if totalWeight <= 0 {
		totalWeight = float64(len(parts))
		for i := range weights {
			weights[i] = 1
		}
	}
	span := end - start
	words := make([]WordSegment, 0, len(parts))
	current := start
	for i, part := range parts {
		wordEnd := current + (span * weights[i] / totalWeight)
		if i == len(parts)-1 {
			wordEnd = end
		}
		if wordEnd < current {
			wordEnd = current
		}
		words = append(words, WordSegment{Start: current, End: wordEnd, Word: part, Segment: segment.Num + 1})
		current = wordEnd
	}
	return words
}

func wordWeight(word string) float64 {
	var count int
	for _, r := range word {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			count++
		}
	}
	if count == 0 {
		return 1
	}
	return float64(count)
}

func Diarise(wavPath string) ([]SpeakerTurn, error) {
	turns, err := turnsFromWav(wavPath)
	if err != nil {
		return nil, DiariseError{Err: err}
	}
	return turns, nil
}

func turnsFromWav(wavPath string) ([]SpeakerTurn, error) {
	f, err := os.Open(wavPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 44)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("invalid wav header: %w", err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("unsupported audio format")
	}

	audioFormat := binary.LittleEndian.Uint16(header[20:22])
	channels := binary.LittleEndian.Uint16(header[22:24])
	sampleRate := binary.LittleEndian.Uint32(header[24:28])
	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])
	if audioFormat != 1 || channels != 1 || bitsPerSample != 16 {
		return nil, fmt.Errorf("expected 16-bit mono PCM wav")
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	samples := make([]int16, len(data)/2)
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &samples); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return []SpeakerTurn{{Start: 0, End: 0, Speaker: "SPEAKER_00"}}, nil
	}

	window := int(float64(sampleRate) * 0.03)
	if window < 1 {
		window = 1
	}
	gapMerge := int(float64(sampleRate) * 0.18)
	if gapMerge < 1 {
		gapMerge = 1
	}
	threshold := float64(2400)

	type segment struct {
		start int
		end   int
	}
	var voiced []segment
	inSpeech := false
	segStart := 0
	for i := 0; i < len(samples); i += window {
		end := i + window
		if end > len(samples) {
			end = len(samples)
		}
		energy := 0.0
		for _, sample := range samples[i:end] {
			if sample < 0 {
				energy += float64(-sample)
			} else {
				energy += float64(sample)
			}
		}
		energy /= float64(end - i)
		if energy >= threshold {
			if !inSpeech {
				segStart = i
				inSpeech = true
			}
			continue
		}
		if inSpeech {
			voiced = append(voiced, segment{start: segStart, end: i})
			inSpeech = false
		}
	}
	if inSpeech {
		voiced = append(voiced, segment{start: segStart, end: len(samples)})
	}

	if len(voiced) == 0 {
		return []SpeakerTurn{{Start: 0, End: float64(len(samples)) / float64(sampleRate), Speaker: "SPEAKER_00"}}, nil
	}

	merged := []segment{voiced[0]}
	for _, seg := range voiced[1:] {
		last := &merged[len(merged)-1]
		if seg.start-last.end <= gapMerge {
			last.end = seg.end
			continue
		}
		merged = append(merged, seg)
	}

	turns := make([]SpeakerTurn, 0, len(merged))
	for i, seg := range merged {
		speaker := "SPEAKER_00"
		if i%2 == 1 {
			speaker = "SPEAKER_01"
		}
		turns = append(turns, SpeakerTurn{
			Start:   float64(seg.start) / float64(sampleRate),
			End:     float64(seg.end) / float64(sampleRate),
			Speaker: speaker,
		})
	}
	return turns, nil
}

func Align(words []WordSegment, turns []SpeakerTurn, speakerMap map[string]string) []LabelledTurn {
	if len(turns) == 0 {
		texts := make([]string, 0, len(words))
		for _, w := range words {
			texts = append(texts, w.Word)
		}
		start, end := 0.0, 0.0
		if len(words) > 0 {
			start = words[0].Start
			end = words[len(words)-1].End
		}
		return []LabelledTurn{{Speaker: "UNKNOWN", Start: start, End: end, Text: strings.Join(texts, " ")}}
	}

	speakerAt := func(t float64) string {
		for _, turn := range turns {
			if turn.Start <= t && t < turn.End {
				return turn.Speaker
			}
		}
		nearest := turns[0]
		best := math.Abs(nearest.Start - t)
		for _, turn := range turns[1:] {
			if d := math.Abs(turn.Start - t); d < best {
				best = d
				nearest = turn
			}
		}
		return nearest.Speaker
	}

	mapSpeaker := func(label string) string {
		if speakerMap != nil {
			if mapped, ok := speakerMap[label]; ok {
				return mapped
			}
		}
		return label
	}

	alternateSpeaker := func(prev string, labels []string) string {
		if len(labels) != 2 {
			return prev
		}
		if labels[0] == prev {
			return labels[1]
		}
		if labels[1] == prev {
			return labels[0]
		}
		return prev
	}

	groups := groupWordSegments(words)
	var labelled []LabelledTurn
	var prevRawSpeaker string
	speakers := uniqueSpeakerLabels(turns)
	for _, group := range groups {
		if len(group.words) == 0 {
			continue
		}
		groupTurns := labelWordGroup(group.words, speakerAt)
		if len(groupTurns) == 0 {
			continue
		}
		if group.hardBreak {
			rawSpeaker := groupTurns[0].Speaker
			if len(labelled) > 0 && rawSpeaker == prevRawSpeaker {
				rawSpeaker = alternateSpeaker(rawSpeaker, speakers)
			}
			labelled = append(labelled, LabelledTurn{
				Speaker: mapSpeaker(rawSpeaker),
				Start:   group.words[0].Start,
				End:     group.words[len(group.words)-1].End,
				Text:    strings.Join(wordTexts(group.words), " "),
			})
			prevRawSpeaker = rawSpeaker
			continue
		}
		for _, turn := range groupTurns {
			rawSpeaker := turn.Speaker
			turn.Speaker = mapSpeaker(rawSpeaker)
			if len(labelled) > 0 && labelled[len(labelled)-1].Speaker == turn.Speaker {
				labelled[len(labelled)-1].Text += " " + turn.Text
				labelled[len(labelled)-1].End = turn.End
				continue
			}
			labelled = append(labelled, turn)
			prevRawSpeaker = rawSpeaker
		}
	}
	return labelled
}

type wordGroup struct {
	words     []WordSegment
	hardBreak bool
}

func groupWordSegments(words []WordSegment) []wordGroup {
	if len(words) == 0 {
		return nil
	}
	groups := make([]wordGroup, 0)
	current := make([]WordSegment, 0)
	currentSegment := 0
	currentHardBreak := false
	flush := func() {
		if len(current) == 0 {
			return
		}
		group := make([]WordSegment, len(current))
		copy(group, current)
		groups = append(groups, wordGroup{words: group, hardBreak: currentHardBreak})
		current = current[:0]
		currentHardBreak = false
	}
	for i, word := range words {
		segment := word.Segment
		if segment <= 0 {
			segment = -(i + 1)
		}
		if len(current) > 0 && segment != currentSegment {
			flush()
		}
		if isDashBoundaryWord(word.Word) && len(current) > 0 {
			flush()
			currentHardBreak = true
		}
		current = append(current, word)
		currentSegment = segment
	}
	flush()
	return groups
}

func labelWordGroup(words []WordSegment, speakerAt func(float64) string) []LabelledTurn {
	if len(words) == 0 {
		return nil
	}
	var labelled []LabelledTurn
	for _, word := range words {
		speaker := speakerAt(word.Start)
		if len(labelled) > 0 && labelled[len(labelled)-1].Speaker == speaker {
			labelled[len(labelled)-1].Text += " " + word.Word
			labelled[len(labelled)-1].End = word.End
			continue
		}
		labelled = append(labelled, LabelledTurn{
			Speaker: speaker,
			Start:   word.Start,
			End:     word.End,
			Text:    word.Word,
		})
	}
	return labelled
}

func uniqueSpeakerLabels(turns []SpeakerTurn) []string {
	labels := make([]string, 0, 2)
	seen := make(map[string]struct{})
	for _, turn := range turns {
		if _, ok := seen[turn.Speaker]; ok {
			continue
		}
		seen[turn.Speaker] = struct{}{}
		labels = append(labels, turn.Speaker)
	}
	return labels
}

func wordTexts(words []WordSegment) []string {
	texts := make([]string, 0, len(words))
	for _, word := range words {
		texts = append(texts, word.Word)
	}
	return texts
}

func isDashBoundaryWord(word string) bool {
	switch strings.TrimSpace(word) {
	case "-", "–", "—":
		return true
	default:
		return false
	}
}

type ChunkStrategy struct {
	Strategy   string
	Chunks     [][]LabelledTurn
	TopicCount int
}

func turnsToText(turns []LabelledTurn) string {
	var b strings.Builder
	for i, turn := range turns {
		if i > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "%s: %s", turn.Speaker, turn.Text)
	}
	return b.String()
}

func chunkTranscript(turns []LabelledTurn, tokenBudget int) ChunkStrategy {
	if EstimateTokens(turnsToText(turns)) <= tokenBudget {
		return ChunkStrategy{Strategy: "single", Chunks: [][]LabelledTurn{turns}, TopicCount: 1}
	}

	// Keep chunks on turn boundaries and split conservatively.
	var chunks [][]LabelledTurn
	var current []LabelledTurn
	for _, turn := range turns {
		next := append(append([]LabelledTurn{}, current...), turn)
		if len(current) > 0 && EstimateTokens(turnsToText(next)) > tokenBudget {
			chunks = append(chunks, current)
			current = []LabelledTurn{turn}
			continue
		}
		current = next
	}
	if len(current) > 0 {
		chunks = append(chunks, current)
	}
	if len(chunks) == 0 {
		chunks = [][]LabelledTurn{turns}
	}
	return ChunkStrategy{Strategy: "map_reduce", Chunks: chunks, TopicCount: len(chunks)}
}

var (
	summaryHeadingRe = regexp.MustCompile(`(?s)## Summary\s*(.*?)(?:\n##\s|\z)`)
	focusRe          = regexp.MustCompile(`(?m)^- \[ \] (.+)$`)
)

func parseSummary(markdown string) string {
	if m := summaryHeadingRe.FindStringSubmatch(markdown); len(m) == 2 {
		return strings.TrimSpace(m[1])
	}
	return strings.TrimSpace(markdown)
}

func parseFocusPoints(markdown string) []string {
	matches := focusRe.FindAllStringSubmatch(markdown, -1)
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) == 2 {
			out = append(out, strings.TrimSpace(match[1]))
		}
	}
	return out
}

const singleSystemPrompt = `You are a session notes assistant. Given a transcript, produce:
1. A summary under the heading "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Use 3-5 concise paragraphs to keep the text readable.
   - Paragraphs should be at most 3 sentences long.
   - Avoid long, dense blocks of text.
   - Separate distinct themes into their own paragraphs.
   - Put a blank line between each paragraph.
   - Do not use any other headings or formatting.
2. A short list of focus points or action items under "## Focus" using markdown checkboxes (- [ ] item)
Write clearly and concisely. Do not add extra headings or commentary.`

const mapSystemPrompt = `You are summarising one section of a longer session transcript.
Produce a brief, concise summary of the key themes and any action items mentioned.
Keep it short and avoid dense blocks of text.
Format: plain prose, no headings.`

const reduceSystemPrompt = `You are combining partial summaries of a coaching session transcript into final notes.
Produce:
1. A summary under "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Use 3-5 concise paragraphs to keep the text readable.
   - Separate distinct themes into their own paragraphs.
   - Put a blank line between each paragraph.
   - Do not use any other headings or formatting.
2. A focused list of action items under "## Focus" using markdown checkboxes (- [ ] item)`

type PostProcess struct {
	Client *LLMClient
}

func (p PostProcess) Run(ctx context.Context, turns []LabelledTurn, tokenBudget int, speakerAName, speakerBName string, progressCB func(string), streamCB func(string), eventCB func(string, map[string]any)) (*PostProcessResult, error) {
	if p.Client == nil {
		return nil, errors.New("llm client not configured")
	}
	singlePrompt := strings.ReplaceAll(strings.ReplaceAll(singleSystemPrompt, "{speaker_a_name}", speakerAName), "{speaker_b_name}", speakerBName)
	reducePrompt := strings.ReplaceAll(strings.ReplaceAll(reduceSystemPrompt, "{speaker_a_name}", speakerAName), "{speaker_b_name}", speakerBName)
	effectiveBudget := tokenBudget - 512
	if effectiveBudget < 1 {
		effectiveBudget = 1
	}
	strategy := chunkTranscript(turns, effectiveBudget)

	if strategy.Strategy == "single" {
		raw := ""
		out, errs := p.Client.Stream(ctx, singlePrompt, turnsToText(turns), eventCB)
		for out != nil || errs != nil {
			select {
			case token, ok := <-out:
				if !ok {
					out = nil
					continue
				}
				raw += token
				if streamCB != nil {
					streamCB(parseSummary(raw))
				}
			case err, ok := <-errs:
				if ok && err != nil {
					return nil, err
				}
				errs = nil
			}
		}
		return &PostProcessResult{
			Summary:     parseSummary(raw),
			FocusPoints: parseFocusPoints(raw),
			RawResponse: raw,
			Strategy:    "single",
			TopicCount:  strategy.TopicCount,
		}, nil
	}

	if progressCB != nil {
		progressCB(fmt.Sprintf("Transcript is too large for one summary pass; splitting into %d chunks.", len(strategy.Chunks)))
	}
	var partials []string
	for i, chunk := range strategy.Chunks {
		if progressCB != nil {
			progressCB(fmt.Sprintf("Summarising chunk %d/%d.", i+1, len(strategy.Chunks)))
		}
		partial, err := p.Client.Complete(ctx, mapSystemPrompt, turnsToText(chunk))
		if err != nil {
			return nil, err
		}
		partials = append(partials, partial)
		if progressCB != nil {
			progressCB(fmt.Sprintf("Completed chunk %d/%d.", i+1, len(strategy.Chunks)))
		}
	}
	combined := strings.Join(partials, "\n\n---\n\n")
	raw := ""
	out, errs := p.Client.Stream(ctx, reducePrompt, combined, eventCB)
	for out != nil || errs != nil {
		select {
		case token, ok := <-out:
			if !ok {
				out = nil
				continue
			}
			raw += token
			if streamCB != nil {
				streamCB(parseSummary(raw))
			}
		case err, ok := <-errs:
			if ok && err != nil {
				return nil, err
			}
			errs = nil
		}
	}
	return &PostProcessResult{
		Summary:     parseSummary(raw),
		FocusPoints: parseFocusPoints(raw),
		RawResponse: raw,
		Strategy:    "map_reduce",
		TopicCount:  strategy.TopicCount,
	}, nil
}
