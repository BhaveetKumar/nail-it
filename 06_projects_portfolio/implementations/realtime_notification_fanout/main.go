package main

import (
	"fmt"
	"time"
)

type Notification struct {
	UserID  string
	Channel string
	Content string
}

type Channel interface{ Send(n Notification) error }

type EmailChannel struct{}

func (e EmailChannel) Send(n Notification) error {
	fmt.Println("EMAIL:", n.UserID, n.Content)
	return nil
}

type SMSChannel struct{}

func (s SMSChannel) Send(n Notification) error { fmt.Println("SMS:", n.UserID, n.Content); return nil }

type Dispatcher struct{ channels map[string]Channel }

func NewDispatcher() *Dispatcher {
	return &Dispatcher{channels: map[string]Channel{"email": EmailChannel{}, "sms": SMSChannel{}}}
}

func (d *Dispatcher) Dispatch(n Notification) error {
	ch, ok := d.channels[n.Channel]
	if !ok {
		return fmt.Errorf("unknown channel %s", n.Channel)
	}
	return ch.Send(n)
}

// DemoDispatch runs a short batch returning number of dispatched notifications.
func DemoDispatch() int {
	d := NewDispatcher()
	_ = d.Dispatch(Notification{UserID: "u1", Channel: "email", Content: "Welcome!"})
	_ = d.Dispatch(Notification{UserID: "u2", Channel: "sms", Content: "OTP 1234"})
	count := 2
	for i := 0; i < 5; i++ {
		_ = d.Dispatch(Notification{UserID: fmt.Sprintf("u%d", i), Channel: "email", Content: "Digest"})
		time.Sleep(50 * time.Millisecond)
		count++
	}
	return count
}
