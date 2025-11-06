package main

import (
	"testing"
	"time"
)

func TestConfigServiceVersioning(t *testing.T) {
	cs := NewConfigService()
	d1 := cs.Set("k", "v1")
	time.Sleep(5 * time.Millisecond)
	d2 := cs.Set("k", "v2")
	if d2.Version != d1.Version+1 {
		t.Fatalf("expected version increment")
	}
	final, ok := cs.Get("k")
	if !ok || final.Value != "v2" {
		t.Fatalf("expected final value v2")
	}
	if len(cs.Versions()) != 2 {
		t.Fatalf("expected audit length 2")
	}
}
