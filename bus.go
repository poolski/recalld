package main

import (
	"encoding/json"
	"errors"
	"sync"
)

type Bus struct {
	mu   sync.RWMutex
	subs map[string]map[chan string]struct{}
}

func NewBus() *Bus {
	return &Bus{subs: make(map[string]map[chan string]struct{})}
}

func (b *Bus) Subscribe(jobID string) (<-chan string, func()) {
	ch := make(chan string, 32)

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.subs[jobID] == nil {
		b.subs[jobID] = make(map[chan string]struct{})
	}
	b.subs[jobID][ch] = struct{}{}

	cancel := func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if subs := b.subs[jobID]; subs != nil {
			delete(subs, ch)
			if len(subs) == 0 {
				delete(b.subs, jobID)
			}
		}
		close(ch)
	}

	return ch, cancel
}

func (b *Bus) Publish(jobID string, payload any) error {
	data, err := encodeEvent(payload)
	if err != nil {
		return err
	}

	b.mu.RLock()
	subs := b.subs[jobID]
	for ch := range subs {
		select {
		case ch <- data:
		default:
		}
	}
	b.mu.RUnlock()
	return nil
}

func encodeEvent(payload any) (string, error) {
	switch v := payload.(type) {
	case string:
		return jsonString(v), nil
	case []byte:
		return string(v), nil
	case json.RawMessage:
		return string(v), nil
	default:
		data, err := json.Marshal(payload)
		if err != nil {
			return "", err
		}
		return string(data), nil
	}
}

func jsonString(s string) string {
	data, _ := json.Marshal(s)
	return string(data)
}

var errBusClosed = errors.New("event bus closed")
