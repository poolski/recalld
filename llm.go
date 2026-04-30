package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const fallbackContextLength = 6000

type ProviderModel struct {
	ID                  string `json:"id"`
	MaxContextLength    int    `json:"max_context_length,omitempty"`
	LoadedContextLength int    `json:"loaded_context_length,omitempty"`
	Selected            bool   `json:"selected"`
}

func (m ProviderModel) ContextLength() int {
	if m.LoadedContextLength > 0 {
		return m.LoadedContextLength
	}
	return m.MaxContextLength
}

func (m ProviderModel) IsLoaded() bool {
	return m.LoadedContextLength > 0
}

type LLMClient struct {
	BaseURL string
	Model   string
	Client  *http.Client
}

func (c LLMClient) client() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return &http.Client{Timeout: 120 * time.Second}
}

func (c LLMClient) base() string {
	base := strings.TrimRight(c.BaseURL, "/")
	if strings.HasSuffix(base, "/api/v1") {
		return base
	}
	if strings.HasSuffix(base, "/v1") {
		return strings.TrimSuffix(base, "/v1") + "/api/v1"
	}
	return base + "/api/v1"
}

func (c LLMClient) headers() http.Header {
	h := make(http.Header)
	if token := strings.TrimSpace(os.Getenv("LM_API_TOKEN")); token != "" {
		h.Set("Authorization", "Bearer "+token)
	}
	return h
}

func (c LLMClient) chatURL() string {
	return c.base() + "/chat"
}

func (c LLMClient) modelsURL() []string {
	base := strings.TrimRight(c.BaseURL, "/")
	if strings.HasSuffix(base, "/api/v1") {
		root := strings.TrimSuffix(base, "/api/v1")
		return []string{root + "/api/v1/models", root + "/v1/models"}
	}
	if strings.HasSuffix(base, "/v1") {
		root := strings.TrimSuffix(base, "/v1")
		return []string{root + "/api/v1/models", base + "/models"}
	}
	return []string{base + "/api/v1/models", base + "/v1/models"}
}

func parseModelList(data map[string]any) []ProviderModel {
	if raw, ok := data["data"].([]any); ok {
		var models []ProviderModel
		for _, item := range raw {
			entry, ok := item.(map[string]any)
			if !ok {
				continue
			}
			id, _ := entry["id"].(string)
			if id == "" {
				continue
			}
			models = append(models, ProviderModel{
				ID:               id,
				MaxContextLength: asInt(entry["max_context_length"]),
			})
		}
		return models
	}

	if raw, ok := data["models"].([]any); ok {
		var models []ProviderModel
		for _, item := range raw {
			entry, ok := item.(map[string]any)
			if !ok || entry["type"] != "llm" {
				continue
			}
			id, _ := entry["key"].(string)
			if id == "" {
				id, _ = entry["id"].(string)
			}
			if id == "" {
				continue
			}
			models = append(models, ProviderModel{
				ID:                  id,
				MaxContextLength:    asInt(entry["max_context_length"]),
				LoadedContextLength: contextLengthFromLoadedInstances(entry),
			})
		}
		return models
	}

	return nil
}

func contextLengthFromLoadedInstances(entry map[string]any) int {
	raw, ok := entry["loaded_instances"].([]any)
	if !ok {
		return 0
	}
	for _, item := range raw {
		instance, ok := item.(map[string]any)
		if !ok {
			continue
		}
		config, ok := instance["config"].(map[string]any)
		if !ok {
			continue
		}
		if n := asInt(config["context_length"]); n > 0 {
			return n
		}
	}
	return 0
}

func asInt(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case float32:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case json.Number:
		n, _ := t.Int64()
		return int(n)
	case string:
		n, _ := strconv.Atoi(t)
		return n
	default:
		return 0
	}
}

func ListAvailableModels(baseURL, selectedModel string) ([]ProviderModel, error) {
	var data map[string]any
	client := &http.Client{Timeout: 5 * time.Second}
	for _, u := range (LLMClient{BaseURL: baseURL}).modelsURL() {
		req, _ := http.NewRequest(http.MethodGet, u, nil)
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			continue
		}
		if err := json.Unmarshal(body, &data); err != nil {
			continue
		}
		break
	}
	if data == nil {
		return nil, nil
	}

	models := parseModelList(data)
	for i := range models {
		models[i].Selected = models[i].ID == selectedModel
	}
	return models, nil
}

func DetectContextLength(baseURL, model string) (int, error) {
	models, _ := ListAvailableModels(baseURL, model)
	for _, m := range models {
		if m.Selected && m.ContextLength() > 0 {
			return m.ContextLength(), nil
		}
	}
	for _, m := range models {
		if m.ContextLength() > 0 {
			return m.ContextLength(), nil
		}
	}
	return fallbackContextLength, nil
}

func EnsureLoadedContextLength(baseURL, model string) (int, error) {
	models, _ := ListAvailableModels(baseURL, model)
	var selected *ProviderModel
	for i := range models {
		if models[i].ID == model {
			selected = &models[i]
			break
		}
	}
	if selected == nil {
		return DetectContextLength(baseURL, model)
	}
	if selected.LoadedContextLength > 0 {
		return selected.LoadedContextLength, nil
	}
	if selected.MaxContextLength <= 0 {
		selected.MaxContextLength = fallbackContextLength
	}

	client := LLMClient{BaseURL: baseURL, Model: model}
	payload := map[string]any{
		"model":            model,
		"echo_load_config": true,
		"context_length":   selected.MaxContextLength,
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, client.base()+"/models/load", bytes.NewReader(body))
	if err != nil {
		return selected.MaxContextLength, nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.client().Do(req)
	if err != nil {
		return selected.MaxContextLength, nil
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if len(models) == 0 {
		return selected.MaxContextLength, nil
	}
	refreshed, _ := ListAvailableModels(baseURL, model)
	for _, m := range refreshed {
		if m.ID == model && m.ContextLength() > 0 {
			return m.ContextLength(), nil
		}
	}
	return selected.MaxContextLength, nil
}

func TokenBudget(contextLength int, headroom float64) int {
	if headroom <= 0 || headroom > 1 {
		headroom = 0.8
	}
	return int(float64(contextLength) * headroom)
}

func EstimateTokens(text string) int {
	words := len(strings.Fields(text))
	if words == 0 {
		return 0
	}
	return int(float64(words)/0.75 + 0.5)
}

func (c LLMClient) Complete(ctx context.Context, system, user string) (string, error) {
	payload := map[string]any{
		"model":         c.Model,
		"system_prompt": system,
		"input":         user,
		"stream":        false,
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL(), bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, vals := range c.headers() {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llm error %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	return parseLLMOutput(data), nil
}

func (c LLMClient) Stream(ctx context.Context, system, user string, eventCB func(string, map[string]any)) (<-chan string, <-chan error) {
	out := make(chan string)
	errCh := make(chan error, 1)
	go func() {
		defer close(out)
		payload := map[string]any{
			"model":         c.Model,
			"system_prompt": system,
			"input":         user,
			"stream":        true,
		}
		body, _ := json.Marshal(payload)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL(), bytes.NewReader(body))
		if err != nil {
			errCh <- err
			close(errCh)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		for k, vals := range c.headers() {
			for _, v := range vals {
				req.Header.Add(k, v)
			}
		}
		resp, err := c.client().Do(req)
		if err != nil {
			errCh <- err
			close(errCh)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			data, _ := io.ReadAll(resp.Body)
			errCh <- fmt.Errorf("llm error %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
			close(errCh)
			return
		}
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		var eventType string
		var dataLines []string
		yieldedMessage := false
		flush := func() {
			if eventType == "" || len(dataLines) == 0 {
				return
			}
			if strings.Join(dataLines, "\n") == "[DONE]" {
				return
			}
			var data map[string]any
			if err := json.Unmarshal([]byte(strings.Join(dataLines, "\n")), &data); err != nil {
				return
			}
			if eventCB != nil {
				eventCB(eventType, data)
			}
			switch eventType {
			case "message.delta":
				if token, _ := data["content"].(string); token != "" {
					yieldedMessage = true
					out <- token
				}
			case "chat.end":
				if !yieldedMessage {
					if token := parseLLMOutput(data); token != "" {
						out <- token
					}
				}
			default:
				if token := parseLLMOutput(data); token != "" && eventType == "message.start" {
					out <- token
				}
			}
		}

		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				flush()
				eventType = ""
				dataLines = nil
				continue
			}
			if strings.HasPrefix(line, "event: ") {
				eventType = strings.TrimSpace(strings.TrimPrefix(line, "event: "))
				continue
			}
			if strings.HasPrefix(line, "data: ") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data: ")))
			}
		}
		if err := scanner.Err(); err != nil {
			errCh <- err
			close(errCh)
			return
		}
		close(errCh)
	}()
	return out, errCh
}

func parseLLMOutput(data map[string]any) string {
	if data == nil {
		return ""
	}
	if typ, _ := data["type"].(string); typ == "chat.end" {
		if result, ok := data["result"].(map[string]any); ok {
			return parseLLMOutput(result)
		}
	}
	if choices, ok := data["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			if delta, ok := choice["delta"].(map[string]any); ok {
				if content, _ := delta["content"].(string); content != "" {
					return content
				}
			}
			if message, ok := choice["message"].(map[string]any); ok {
				if content, _ := message["content"].(string); content != "" {
					return content
				}
			}
		}
	}
	if output, ok := data["output"].(string); ok {
		return output
	}
	if output, ok := data["text"].(string); ok {
		return output
	}
	if output, ok := data["content"].(string); ok {
		return output
	}
	if output, ok := data["output"].([]any); ok {
		var parts []string
		for _, item := range output {
			if m, ok := item.(map[string]any); ok && m["type"] == "message" {
				if content, _ := m["content"].(string); content != "" {
					parts = append(parts, content)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}
