package main

import "testing"

func TestDispatcherChannels(t *testing.T) {
	d := NewDispatcher()
	err := d.Dispatch(Notification{UserID: "u1", Channel: "email", Content: "Hi"})
	if err != nil {
		t.Fatalf("unexpected email error: %v", err)
	}
	err = d.Dispatch(Notification{UserID: "u2", Channel: "sms", Content: "Hi"})
	if err != nil {
		t.Fatalf("unexpected sms error: %v", err)
	}
	err = d.Dispatch(Notification{UserID: "u3", Channel: "push", Content: "Hi"})
	if err == nil {
		t.Fatalf("expected unknown channel error")
	}
}
