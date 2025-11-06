package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
)

var bucket *TokenBucket

func init() {
	cap := 10
	refill := 5
	if v := os.Getenv("RL_CAPACITY"); v != "" {
		if iv, err := strconv.Atoi(v); err == nil {
			cap = iv
		}
	}
	if v := os.Getenv("RL_REFILL"); v != "" {
		if iv, err := strconv.Atoi(v); err == nil {
			refill = iv
		}
	}
	bucket = NewTokenBucket(cap, refill)
}

func allowHandler(w http.ResponseWriter, r *http.Request) {
	tokens := 1
	if v := r.URL.Query().Get("tokens"); v != "" {
		if iv, err := strconv.Atoi(v); err == nil {
			tokens = iv
		}
	}
	allowed := bucket.Allow(tokens)
	resp := map[string]any{"allowed": allowed, "remaining": bucket.Remaining(), "tokens": tokens}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/ratelimit/allow", allowHandler)
	mux.HandleFunc("/health", healthHandler)
	srv := &http.Server{Addr: ":8080", Handler: mux, ReadTimeout: 2 * time.Second, WriteTimeout: 2 * time.Second}
	log.Println("rate limiter server listening on :8080")
	log.Fatal(srv.ListenAndServe())
}
