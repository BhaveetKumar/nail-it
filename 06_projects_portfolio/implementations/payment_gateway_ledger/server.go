package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
)

var gw = NewGateway()

func authorizeHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	req := PaymentRequest{
		UserID:    q.Get("user"),
		Currency:  q.Get("currency"),
		RequestID: q.Get("requestId"),
	}
	if amt := q.Get("amount"); amt != "" {
		if v, err := strconv.ParseInt(amt, 10, 64); err == nil {
			req.Amount = v
		}
	}
	authID, fresh := gw.Authorize(req)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"authId": authID, "fresh": fresh})
}

func captureHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	authID := q.Get("authId")
	req := PaymentRequest{UserID: q.Get("user"), Currency: q.Get("currency"), RequestID: q.Get("requestId")}
	if amt := q.Get("amount"); amt != "" {
		if v, err := strconv.ParseInt(amt, 10, 64); err == nil {
			req.Amount = v
		}
	}
	entry := gw.Capture(req, authID)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(entry)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/payments/authorize", authorizeHandler)
	mux.HandleFunc("/payments/capture", captureHandler)
	log.Println("payment gateway server on :8081")
	log.Fatal(http.ListenAndServe(":8081", mux))
}
