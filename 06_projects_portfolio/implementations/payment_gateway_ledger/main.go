package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

type PaymentRequest struct {
	UserID    string
	Amount    int64
	Currency  string
	RequestID string // client provided idempotency key
}

type LedgerEntry struct {
	ID       string
	UserID   string
	Amount   int64
	Currency string
}

type Gateway struct {
	seen     sync.Map // idempotency key -> result
	ledgerMu sync.Mutex
	ledger   []LedgerEntry
}

func NewGateway() *Gateway { return &Gateway{} }

func hashReq(pr PaymentRequest) string {
	h := sha256.Sum256([]byte(pr.UserID + pr.RequestID + fmt.Sprint(pr.Amount) + pr.Currency))
	return hex.EncodeToString(h[:])
}

func (g *Gateway) Authorize(pr PaymentRequest) (string, bool) {
	key := hashReq(pr)
	if val, ok := g.seen.Load(key); ok {
		return val.(string), false // duplicate
	}
	// Simulate auth success
	authID := "auth_" + key[:12]
	g.seen.Store(key, authID)
	return authID, true
}

func (g *Gateway) Capture(pr PaymentRequest, authID string) LedgerEntry {
	g.ledgerMu.Lock()
	defer g.ledgerMu.Unlock()
	entry := LedgerEntry{ID: "ledg_" + authID[5:], UserID: pr.UserID, Amount: pr.Amount, Currency: pr.Currency}
	g.ledger = append(g.ledger, entry)
	return entry
}

// DemoTransaction demonstrates an auth + idempotent replay + capture and returns ledger size.
func DemoTransaction() (firstAuth string, secondFresh bool, ledgerCount int) {
	gw := NewGateway()
	req := PaymentRequest{UserID: "u1", Amount: 5000, Currency: "INR", RequestID: "rk-123"}
	authID, _ := gw.Authorize(req)
	_, fresh2 := gw.Authorize(req) // replay
	gw.Capture(req, authID)
	return authID, fresh2, len(gw.ledger)
}
