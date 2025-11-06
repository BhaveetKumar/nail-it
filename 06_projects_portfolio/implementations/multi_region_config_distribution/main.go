package main

import (
	"sync"
	"time"
)

type ConfigDoc struct {
	Key     string
	Value   string
	Version int64
	TS      time.Time
}

type ConfigService struct {
	mu    sync.RWMutex
	data  map[string]ConfigDoc
	audit []ConfigDoc
}

func NewConfigService() *ConfigService {
	return &ConfigService{data: make(map[string]ConfigDoc)}
}

func (cs *ConfigService) Set(key, value string) ConfigDoc {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	old := cs.data[key]
	doc := ConfigDoc{Key: key, Value: value, Version: old.Version + 1, TS: time.Now()}
	cs.data[key] = doc
	cs.audit = append(cs.audit, doc)
	return doc
}

func (cs *ConfigService) Get(key string) (ConfigDoc, bool) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	d, ok := cs.data[key]
	return d, ok
}

func (cs *ConfigService) Versions() []ConfigDoc {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	out := make([]ConfigDoc, len(cs.audit))
	copy(out, cs.audit)
	return out
}

// DemoConfig performs two updates and returns final value and audit length.
func DemoConfig() (final string, version int64, auditLen int) {
	cfg := NewConfigService()
	cfg.Set("feature_x", "on")
	time.Sleep(10 * time.Millisecond)
	doc := cfg.Set("feature_x", "off")
	return doc.Value, doc.Version, len(cfg.Versions())
}
