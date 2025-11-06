package main

import "testing"

func TestGatewayIdempotentAuthorize(t *testing.T) {
	gw := NewGateway()
	req := PaymentRequest{UserID: "u", Amount: 100, Currency: "INR", RequestID: "X"}
	auth1, fresh1 := gw.Authorize(req)
	auth2, fresh2 := gw.Authorize(req)
	if !fresh1 {
		t.Fatalf("first should be fresh")
	}
	if fresh2 {
		t.Fatalf("second should be replay")
	}
	if auth1 != auth2 {
		t.Fatalf("auth ids differ; expected idempotent")
	}
	entry := gw.Capture(req, auth1)
	if entry.ID == "" {
		t.Fatalf("expected ledger entry ID")
	}
}
