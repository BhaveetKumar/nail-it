package main

import "testing"

func TestFlagStoreSetAndGet(t *testing.T) {
	fs := NewFlagStore()
	initial, ok := fs.Get("new_ui")
	if !ok || !initial.On {
		t.Fatalf("expected new_ui to exist and be on")
	}
	fs.Set(Flag{Name: "beta_payment", On: false})
	f2, ok2 := fs.Get("beta_payment")
	if !ok2 || f2.On {
		t.Fatalf("expected beta_payment off")
	}
	if fs.version < 1 {
		t.Fatalf("expected version increment")
	}
}
