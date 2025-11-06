package main

import (
	"sync"
	"time"
)

// TokenBucket represents a simple thread-safe token bucket.
type TokenBucket struct {
	capacity     int
	refillRate   int // tokens per second
	available    int
	lastRefillTs time.Time
	mu           sync.Mutex
}

func NewTokenBucket(capacity, refillRate int) *TokenBucket {
	return &TokenBucket{capacity: capacity, refillRate: refillRate, available: capacity, lastRefillTs: time.Now()}
}

func (tb *TokenBucket) Allow(n int) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	elapsed := time.Since(tb.lastRefillTs).Seconds()
	if elapsed >= 1 {
		refillTokens := int(elapsed) * tb.refillRate
		if refillTokens > 0 {
			tb.available += refillTokens
			if tb.available > tb.capacity {
				tb.available = tb.capacity
			}
			tb.lastRefillTs = time.Now()
		}
	}
	if n <= tb.available {
		tb.available -= n
		return true
	}
	return false
}

// Remaining returns current available tokens (non-atomic snapshot).
func (tb *TokenBucket) Remaining() int {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	return tb.available
}

// DemoRun executes a short demonstration sequence (used in tests or optional).
func DemoRun(tb *TokenBucket, iterations int, sleep time.Duration) []bool {
	results := make([]bool, 0, iterations)
	for i := 0; i < iterations; i++ {
		results = append(results, tb.Allow(1))
		time.Sleep(sleep)
	}
	return results
}
