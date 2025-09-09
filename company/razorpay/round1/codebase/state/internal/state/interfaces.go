package state

import (
	"context"
	"time"
)

// State defines the interface for all states
type State interface {
	Enter(ctx context.Context, entity *StateEntity) error
	Exit(ctx context.Context, entity *StateEntity) error
	Handle(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error)
	GetStateName() string
	GetStateType() string
	GetDescription() string
	IsFinal() bool
	CanTransitionTo(targetState string) bool
	GetAllowedTransitions() []string
	Validate(entity *StateEntity) error
}

// StateEntity represents an entity that can have states
type StateEntity struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	CurrentState string                `json:"current_state"`
	PreviousState string               `json:"previous_state"`
	StateHistory []*StateHistoryEntry  `json:"state_history"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]string      `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// StateEvent represents an event that can trigger state transitions
type StateEvent struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	EntityID    string                 `json:"entity_id"`
	EntityType  string                 `json:"entity_type"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]string      `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
}

// StateTransition represents a state transition
type StateTransition struct {
	ID            string                 `json:"id"`
	EntityID      string                 `json:"entity_id"`
	FromState     string                 `json:"from_state"`
	ToState       string                 `json:"to_state"`
	Event         *StateEvent            `json:"event"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]string      `json:"metadata"`
	TransitionedAt time.Time             `json:"transitioned_at"`
}

// StateHistoryEntry represents a state history entry
type StateHistoryEntry struct {
	ID            string                 `json:"id"`
	EntityID      string                 `json:"entity_id"`
	State         string                 `json:"state"`
	Event         *StateEvent            `json:"event"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]string      `json:"metadata"`
	EnteredAt     time.Time              `json:"entered_at"`
	ExitedAt      time.Time              `json:"exited_at"`
	Duration      time.Duration          `json:"duration"`
}

// StateMachine manages state transitions
type StateMachine interface {
	AddState(state State) error
	RemoveState(stateName string) error
	GetState(stateName string) (State, error)
	GetAllStates() map[string]State
	Transition(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error)
	CanTransition(entity *StateEntity, event *StateEvent) (bool, error)
	GetPossibleTransitions(entity *StateEntity) ([]string, error)
	GetStateMachineName() string
	GetInitialState() string
	GetFinalStates() []string
}

// StateManager manages state entities
type StateManager interface {
	CreateEntity(ctx context.Context, entity *StateEntity) error
	GetEntity(ctx context.Context, entityID string) (*StateEntity, error)
	UpdateEntity(ctx context.Context, entity *StateEntity) error
	DeleteEntity(ctx context.Context, entityID string) error
	GetEntitiesByState(ctx context.Context, state string) ([]*StateEntity, error)
	GetEntitiesByType(ctx context.Context, entityType string) ([]*StateEntity, error)
	GetEntityHistory(ctx context.Context, entityID string) ([]*StateHistoryEntry, error)
}

// StateEventBus manages state events
type StateEventBus interface {
	PublishEvent(ctx context.Context, event *StateEvent) error
	SubscribeToEvents(ctx context.Context, eventType string, handler StateEventHandler) error
	UnsubscribeFromEvents(ctx context.Context, eventType string, handlerID string) error
	GetEventTypes() []string
	GetSubscribers(eventType string) []string
}

// StateEventHandler handles state events
type StateEventHandler interface {
	HandleEvent(ctx context.Context, event *StateEvent) error
	GetHandlerID() string
	GetEventTypes() []string
	GetHandlerName() string
}

// StateValidator validates state transitions
type StateValidator interface {
	ValidateTransition(ctx context.Context, entity *StateEntity, event *StateEvent, targetState string) error
	ValidateEntity(ctx context.Context, entity *StateEntity) error
	ValidateEvent(ctx context.Context, event *StateEvent) error
	GetValidationRules() []ValidationRule
	AddValidationRule(rule ValidationRule) error
	RemoveValidationRule(ruleID string) error
}

// StateMetrics collects state transition metrics
type StateMetrics interface {
	RecordStateTransition(entityType string, fromState string, toState string, duration time.Duration, success bool)
	GetStateMetrics(entityType string) (*StateMetricsData, error)
	GetAllMetrics() (map[string]*StateMetricsData, error)
	ResetMetrics(entityType string) error
	ResetAllMetrics() error
}

// StateConfig holds configuration for state management
type StateConfig struct {
	MaxHistorySize    int           `json:"max_history_size"`
	EnableMetrics     bool          `json:"enable_metrics"`
	EnableValidation  bool          `json:"enable_validation"`
	EnableEventBus    bool          `json:"enable_event_bus"`
	EnablePersistence bool          `json:"enable_persistence"`
	DefaultTimeout    time.Duration `json:"default_timeout"`
	MaxRetries        int           `json:"max_retries"`
	RetryDelay        time.Duration `json:"retry_delay"`
	CircuitBreaker    CircuitBreakerConfig `json:"circuit_breaker"`
}

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	Enabled           bool          `json:"enabled"`
	FailureThreshold  int           `json:"failure_threshold"`
	RecoveryTimeout   time.Duration `json:"recovery_timeout"`
	HalfOpenMaxCalls  int           `json:"half_open_max_calls"`
}

// ValidationRule represents a validation rule
type ValidationRule struct {
	RuleID      string                 `json:"rule_id"`
	RuleType    string                 `json:"rule_type"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Enabled     bool                   `json:"enabled"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// StateMetricsData holds metrics for state transitions
type StateMetricsData struct {
	EntityType         string        `json:"entity_type"`
	TotalTransitions   int64         `json:"total_transitions"`
	SuccessfulTransitions int64      `json:"successful_transitions"`
	FailedTransitions  int64         `json:"failed_transitions"`
	AverageDuration    time.Duration `json:"average_duration"`
	MinDuration        time.Duration `json:"min_duration"`
	MaxDuration        time.Duration `json:"max_duration"`
	LastTransition     time.Time     `json:"last_transition"`
	SuccessRate        float64       `json:"success_rate"`
	Availability       float64       `json:"availability"`
}

// StateStatus represents state status
type StateStatus string

const (
	StateStatusActive   StateStatus = "active"
	StateStatusInactive StateStatus = "inactive"
	StateStatusPending  StateStatus = "pending"
	StateStatusFinal    StateStatus = "final"
	StateStatusError    StateStatus = "error"
)

// String returns the string representation of StateStatus
func (ss StateStatus) String() string {
	return string(ss)
}

// StateType represents state type
type StateType string

const (
	StateTypeInitial StateType = "initial"
	StateTypeNormal  StateType = "normal"
	StateTypeFinal   StateType = "final"
	StateTypeError   StateType = "error"
)

// String returns the string representation of StateType
func (st StateType) String() string {
	return string(st)
}
