package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const previewMaxChars = 1200

type PostProcessResult struct {
	Summary     string   `json:"summary"`
	FocusPoints []string `json:"focus_points"`
	RawResponse string   `json:"raw_response,omitempty"`
	Strategy    string   `json:"strategy"`
	TopicCount  int      `json:"topic_count"`
}

func renderSessionNoteBody(sessionDate time.Time, category string, speakers []string, result *PostProcessResult, turns []LabelledTurn) string {
	postProcessingStatus := "failed"
	if result != nil {
		postProcessingStatus = "ok"
	}
	speakersYAML := "[" + strings.Join(speakers, ", ") + "]"
	dateStr := sessionDate.Format("2006-01-02")

	var transcriptLines []string
	for _, turn := range turns {
		transcriptLines = append(transcriptLines, fmt.Sprintf("> **%s:** %s", turn.Speaker, turn.Text))
	}

	body := "_Post-processing failed. Transcript preserved below._\n"
	focusSection := ""
	if result != nil {
		var focusItems []string
		for _, p := range result.FocusPoints {
			focusItems = append(focusItems, "- [ ] "+p)
		}
		body = result.Summary + "\n\n[Full transcript ↓](#transcript)\n"
		if len(focusItems) > 0 {
			focusSection = "\n## Focus\n\n" + strings.Join(focusItems, "\n") + "\n"
		} else {
			focusSection = "\n## Focus\n\n"
		}
	}

	return fmt.Sprintf(`---
date: %s
category: %s
speakers: %s
post_processing: %s
---

## Summary

%s%s## Transcript

> [!note]- Full transcript
%s
`, dateStr, category, speakersYAML, postProcessingStatus, body, focusSection, strings.Join(transcriptLines, "\n"))
}

func RenderSessionNote(sessionDate time.Time, category string, speakers []string, result *PostProcessResult, turns []LabelledTurn) string {
	return renderSessionNoteBody(sessionDate, category, speakers, result, turns)
}

func stripFrontmatter(markdown string) string {
	if strings.HasPrefix(markdown, "---\n") {
		parts := strings.SplitN(markdown, "\n---\n", 2)
		if len(parts) == 2 {
			return strings.TrimLeft(parts[1], "\n")
		}
	}
	return strings.TrimSpace(markdown)
}

func truncatePreview(markdown string, maxChars int) string {
	text := strings.TrimSpace(markdown)
	if len(text) <= maxChars {
		return text
	}

	cutoff := strings.LastIndex(text[:maxChars], "\n\n")
	if cutoff < maxChars/2 {
		cutoff = strings.LastIndex(text[:maxChars], "\n")
	}
	if cutoff == -1 {
		cutoff = maxChars
	}
	return strings.TrimRight(text[:cutoff], "\n ") + "\n\n..."
}

func RenderSessionNotePreview(sessionDate time.Time, category string, speakers []string, result *PostProcessResult, turns []LabelledTurn, maxChars int) string {
	if maxChars <= 0 {
		maxChars = previewMaxChars
	}
	note := renderSessionNoteBody(sessionDate, category, speakers, result, turns)
	return truncatePreview(stripFrontmatter(note), maxChars)
}

func RenderFocusSection(sessionDate time.Time, focusPoints []string) string {
	items := make([]string, 0, len(focusPoints))
	for _, p := range focusPoints {
		items = append(items, "- [ ] "+p)
	}
	return fmt.Sprintf("\n## %s\n\n%s\n", sessionDate.Format("2006-01-02"), strings.Join(items, "\n"))
}

type VaultWriter struct {
	APIURL string
	APIKey string
	Client *http.Client
}

func (w VaultWriter) client() *http.Client {
	if w.Client != nil {
		return w.Client
	}
	return &http.Client{Timeout: 15 * time.Second}
}

func (w VaultWriter) headers() http.Header {
	h := make(http.Header)
	if w.APIKey != "" {
		h.Set("Authorization", "Bearer "+w.APIKey)
	}
	return h
}

func (w VaultWriter) WriteNote(vaultPath, filename, content string) error {
	encoded := url.PathEscape(vaultPath + "/" + filename)
	encoded = strings.ReplaceAll(encoded, "%2F", "/")
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(w.APIURL, "/")+"/vault/"+encoded, bytes.NewReader([]byte(content)))
	if err != nil {
		return err
	}
	req.Header = w.headers()
	req.Header.Set("Content-Type", "text/markdown")
	resp, err := w.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("obsidian api error %d", resp.StatusCode)
	}
	return nil
}

func (w VaultWriter) AppendToNote(vaultPath, content string) error {
	encoded := url.PathEscape(vaultPath)
	encoded = strings.ReplaceAll(encoded, "%2F", "/")
	req, err := http.NewRequest(http.MethodPatch, strings.TrimRight(w.APIURL, "/")+"/vault/"+encoded, bytes.NewReader([]byte(content)))
	if err != nil {
		return err
	}
	req.Header = w.headers()
	req.Header.Set("Content-Type", "text/markdown")
	resp, err := w.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("obsidian api error %d", resp.StatusCode)
	}
	return nil
}

func (w VaultWriter) NoteExists(vaultPath string) (bool, error) {
	encoded := url.PathEscape(vaultPath)
	encoded = strings.ReplaceAll(encoded, "%2F", "/")
	req, err := http.NewRequest(http.MethodGet, strings.TrimRight(w.APIURL, "/")+"/vault/"+encoded, nil)
	if err != nil {
		return false, err
	}
	req.Header = w.headers()
	resp, err := w.client().Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK, nil
}

func (w VaultWriter) OpenNote(vaultPath string) error {
	encoded := url.PathEscape(vaultPath)
	encoded = strings.ReplaceAll(encoded, "%2F", "/")
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(w.APIURL, "/")+"/open/"+encoded, nil)
	if err != nil {
		return err
	}
	req.Header = w.headers()
	resp, err := w.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("obsidian api error %d", resp.StatusCode)
	}
	return nil
}
