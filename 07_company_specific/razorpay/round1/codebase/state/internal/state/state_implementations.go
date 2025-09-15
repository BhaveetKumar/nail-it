package state

import (
	"context"
	"fmt"
	"time"
)

// PaymentPendingState represents the pending state for payments
type PaymentPendingState struct {
	stateName string
	stateType string
}

// NewPaymentPendingState creates a new payment pending state
func NewPaymentPendingState() *PaymentPendingState {
	return &PaymentPendingState{
		stateName: "pending",
		stateType: "normal",
	}
}

// Enter enters the pending state
func (s *PaymentPendingState) Enter(ctx context.Context, entity *StateEntity) error {
	// Simulate entering pending state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Exit exits the pending state
func (s *PaymentPendingState) Exit(ctx context.Context, entity *StateEntity) error {
	// Simulate exiting pending state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Handle handles events in the pending state
func (s *PaymentPendingState) Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	switch event.Type {
	case "payment_processed":
		return &StateTransition{
			ToState: "completed",
			Event:   event,
			Data:    event.Data,
			Metadata: map[string]string{
				"transition_reason": "payment_processed",
			},
		}, nil
	case "payment_failed":
		return &StateTransition{
			ToState: "failed",
			Event:   event,
			Data:    event.Data,
			Metadata: map[string]string{
				"transition_reason": "payment_failed",
			},
		}, nil
	case "payment_cancelled":
		return &StateTransition{
			ToState: "cancelled",
			Event:   event,
			Data:    event.Data,
			Metadata: map[string]string{
				"transition_reason": "payment_cancelled",
			},
		}, nil
	default:
		return nil, fmt.Errorf("event %s not handled in pending state", event.Type)
	}
}

// GetStateName returns the state name
func (s *PaymentPendingState) GetStateName() string {
	return s.stateName
}

// GetStateType returns the state type
func (s *PaymentPendingState) GetStateType() string {
	return s.stateType
}

// GetDescription returns the state description
func (s *PaymentPendingState) GetDescription() string {
	return "Payment is pending processing"
}

// IsFinal returns if this is a final state
func (s *PaymentPendingState) IsFinal() bool {
	return false
}

// CanTransitionTo checks if transition to target state is allowed
func (s *PaymentPendingState) CanTransitionTo(targetState string) bool {
	allowedStates := []string{"completed", "failed", "cancelled"}
	for _, state := range allowedStates {
		if state == targetState {
			return true
		}
	}
	return false
}

// GetAllowedTransitions returns allowed transitions
func (s *PaymentPendingState) GetAllowedTransitions() []string {
	return []string{"completed", "failed", "cancelled"}
}

// Validate validates the entity in this state
func (s *PaymentPendingState) Validate(entity *StateEntity) error {
	if entity.Type != "payment" {
		return fmt.Errorf("entity type must be payment")
	}
	return nil
}

// PaymentCompletedState represents the completed state for payments
type PaymentCompletedState struct {
	stateName string
	stateType string
}

// NewPaymentCompletedState creates a new payment completed state
func NewPaymentCompletedState() *PaymentCompletedState {
	return &PaymentCompletedState{
		stateName: "completed",
		stateType: "final",
	}
}

// Enter enters the completed state
func (s *PaymentCompletedState) Enter(ctx context.Context, entity *StateEntity) error {
	// Simulate entering completed state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Exit exits the completed state
func (s *PaymentCompletedState) Exit(ctx context.Context, entity *StateEntity) error {
	// Simulate exiting completed state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Handle handles events in the completed state
func (s *PaymentCompletedState) Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	switch event.Type {
	case "refund_requested":
		return &StateTransition{
			ToState: "refunded",
			Event:   event,
			Data:    event.Data,
			Metadata: map[string]string{
				"transition_reason": "refund_requested",
			},
		}, nil
	default:
		return nil, fmt.Errorf("event %s not handled in completed state", event.Type)
	}
}

// GetStateName returns the state name
func (s *PaymentCompletedState) GetStateName() string {
	return s.stateName
}

// GetStateType returns the state type
func (s *PaymentCompletedState) GetStateType() string {
	return s.stateType
}

// GetDescription returns the state description
func (s *PaymentCompletedState) GetDescription() string {
	return "Payment has been completed successfully"
}

// IsFinal returns if this is a final state
func (s *PaymentCompletedState) IsFinal() bool {
	return true
}

// CanTransitionTo checks if transition to target state is allowed
func (s *PaymentCompletedState) CanTransitionTo(targetState string) bool {
	allowedStates := []string{"refunded"}
	for _, state := range allowedStates {
		if state == targetState {
			return true
		}
	}
	return false
}

// GetAllowedTransitions returns allowed transitions
func (s *PaymentCompletedState) GetAllowedTransitions() []string {
	return []string{"refunded"}
}

// Validate validates the entity in this state
func (s *PaymentCompletedState) Validate(entity *StateEntity) error {
	if entity.Type != "payment" {
		return fmt.Errorf("entity type must be payment")
	}
	return nil
}

// PaymentFailedState represents the failed state for payments
type PaymentFailedState struct {
	stateName string
	stateType string
}

// NewPaymentFailedState creates a new payment failed state
func NewPaymentFailedState() *PaymentFailedState {
	return &PaymentFailedState{
		stateName: "failed",
		stateType: "final",
	}
}

// Enter enters the failed state
func (s *PaymentFailedState) Enter(ctx context.Context, entity *StateEntity) error {
	// Simulate entering failed state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Exit exits the failed state
func (s *PaymentFailedState) Exit(ctx context.Context, entity *StateEntity) error {
	// Simulate exiting failed state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Handle handles events in the failed state
func (s *PaymentFailedState) Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	switch event.Type {
	case "retry_payment":
		return &StateTransition{
			ToState: "pending",
			Event:   event,
			Data:    event.Data,
			Metadata: map[string]string{
				"transition_reason": "retry_payment",
			},
		}, nil
	default:
		return nil, fmt.Errorf("event %s not handled in failed state", event.Type)
	}
}

// GetStateName returns the state name
func (s *PaymentFailedState) GetStateName() string {
	return s.stateName
}

// GetStateType returns the state type
func (s *PaymentFailedState) GetStateType() string {
	return s.stateType
}

// GetDescription returns the state description
func (s *PaymentFailedState) GetDescription() string {
	return "Payment has failed"
}

// IsFinal returns if this is a final state
func (s *PaymentFailedState) IsFinal() bool {
	return true
}

// CanTransitionTo checks if transition to target state is allowed
func (s *PaymentFailedState) CanTransitionTo(targetState string) bool {
	allowedStates := []string{"pending"}
	for _, state := range allowedStates {
		if state == targetState {
			return true
		}
	}
	return false
}

// GetAllowedTransitions returns allowed transitions
func (s *PaymentFailedState) GetAllowedTransitions() []string {
	return []string{"pending"}
}

// Validate validates the entity in this state
func (s *PaymentFailedState) Validate(entity *StateEntity) error {
	if entity.Type != "payment" {
		return fmt.Errorf("entity type must be payment")
	}
	return nil
}

// PaymentCancelledState represents the cancelled state for payments
type PaymentCancelledState struct {
	stateName string
	stateType string
}

// NewPaymentCancelledState creates a new payment cancelled state
func NewPaymentCancelledState() *PaymentCancelledState {
	return &PaymentCancelledState{
		stateName: "cancelled",
		stateType: "final",
	}
}

// Enter enters the cancelled state
func (s *PaymentCancelledState) Enter(ctx context.Context, entity *StateEntity) error {
	// Simulate entering cancelled state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Exit exits the cancelled state
func (s *PaymentCancelledState) Exit(ctx context.Context, entity *StateEntity) error {
	// Simulate exiting cancelled state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Handle handles events in the cancelled state
func (s *PaymentCancelledState) Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	// No transitions allowed from cancelled state
	return nil, fmt.Errorf("no transitions allowed from cancelled state")
}

// GetStateName returns the state name
func (s *PaymentCancelledState) GetStateName() string {
	return s.stateName
}

// GetStateType returns the state type
func (s *PaymentCancelledState) GetStateType() string {
	return s.stateType
}

// GetDescription returns the state description
func (s *PaymentCancelledState) GetDescription() string {
	return "Payment has been cancelled"
}

// IsFinal returns if this is a final state
func (s *PaymentCancelledState) IsFinal() bool {
	return true
}

// CanTransitionTo checks if transition to target state is allowed
func (s *PaymentCancelledState) CanTransitionTo(targetState string) bool {
	return false
}

// GetAllowedTransitions returns allowed transitions
func (s *PaymentCancelledState) GetAllowedTransitions() []string {
	return []string{}
}

// Validate validates the entity in this state
func (s *PaymentCancelledState) Validate(entity *StateEntity) error {
	if entity.Type != "payment" {
		return fmt.Errorf("entity type must be payment")
	}
	return nil
}

// PaymentRefundedState represents the refunded state for payments
type PaymentRefundedState struct {
	stateName string
	stateType string
}

// NewPaymentRefundedState creates a new payment refunded state
func NewPaymentRefundedState() *PaymentRefundedState {
	return &PaymentRefundedState{
		stateName: "refunded",
		stateType: "final",
	}
}

// Enter enters the refunded state
func (s *PaymentRefundedState) Enter(ctx context.Context, entity *StateEntity) error {
	// Simulate entering refunded state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Exit exits the refunded state
func (s *PaymentRefundedState) Exit(ctx context.Context, entity *StateEntity) error {
	// Simulate exiting refunded state
	time.Sleep(10 * time.Millisecond)
	return nil
}

// Handle handles events in the refunded state
func (s *PaymentRefundedState) Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	// No transitions allowed from refunded state
	return nil, fmt.Errorf("no transitions allowed from refunded state")
}

// GetStateName returns the state name
func (s *PaymentRefundedState) GetStateName() string {
	return s.stateName
}

// GetStateType returns the state type
func (s *PaymentRefundedState) GetStateType() string {
	return s.stateType
}

// GetDescription returns the state description
func (s *PaymentRefundedState) GetDescription() string {
	return "Payment has been refunded"
}

// IsFinal returns if this is a final state
func (s *PaymentRefundedState) IsFinal() bool {
	return true
}

// CanTransitionTo checks if transition to target state is allowed
func (s *PaymentRefundedState) CanTransitionTo(targetState string) bool {
	return false
}

// GetAllowedTransitions returns allowed transitions
func (s *PaymentRefundedState) GetAllowedTransitions() []string {
	return []string{}
}

// Validate validates the entity in this state
func (s *PaymentRefundedState) Validate(entity *StateEntity) error {
	if entity.Type != "payment" {
		return fmt.Errorf("entity type must be payment")
	}
	return nil
}
