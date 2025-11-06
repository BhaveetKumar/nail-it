package main

import (
	"testing"
	"time"
)

func TestTokenBucketBasic(t *testing.T) {
	b := NewTokenBucket(5, 5)
	for i := 0; i < 5; i++ {
		if !b.Allow(1) {
			t.Fatalf("expected allowance at iteration %d", i)
		}
	}
	if b.Allow(1) {
		t.Fatalf("should be empty after 5 tokens consumed")
	}
	// wait for refill
	time.Sleep(1100 * time.Millisecond)
	if !b.Allow(1) {
		t.Fatalf("expected token after refill")
	}
}
