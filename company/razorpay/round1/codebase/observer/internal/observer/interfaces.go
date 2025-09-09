package observer

import (
	"context"
	"time"
)

// Event represents a generic event
type Event interface {
	GetType() string
	GetID() string
	GetTimestamp() time.Time
	GetData() interface{}
	GetSource() string
}

// Observer defines the contract for event observers
type Observer interface {
	OnEvent(ctx context.Context, event Event) error
	GetObserverID() string
	GetEventTypes() []string
	IsAsync() bool
}

// Subject defines the contract for event subjects
type Subject interface {
	Attach(observer Observer) error
	Detach(observer Observer) error
	Notify(ctx context.Context, event Event) error
	GetObserverCount() int
	GetObservers() []Observer
}

// EventBus manages event distribution
type EventBus interface {
	Subscribe(eventType string, observer Observer) error
	Unsubscribe(eventType string, observerID string) error
	Publish(ctx context.Context, event Event) error
	GetSubscriberCount(eventType string) int
	GetEventTypes() []string
}

// EventStore persists events
type EventStore interface {
	Store(ctx context.Context, event Event) error
	GetEvents(ctx context.Context, eventType string, limit, offset int) ([]Event, error)
	GetEventByID(ctx context.Context, eventID string) (Event, error)
	GetEventsByTimeRange(ctx context.Context, start, end time.Time) ([]Event, error)
}

// EventHandler processes events
type EventHandler interface {
	Handle(ctx context.Context, event Event) error
	CanHandle(eventType string) bool
	GetHandlerName() string
}

// EventFilter filters events
type EventFilter interface {
	ShouldProcess(event Event) bool
	GetFilterName() string
}

// EventTransformer transforms events
type EventTransformer interface {
	Transform(event Event) (Event, error)
	GetTransformerName() string
}

// EventAggregator aggregates events
type EventAggregator interface {
	Aggregate(ctx context.Context, events []Event) (Event, error)
	GetAggregationWindow() time.Duration
	GetAggregatorName() string
}

// EventReplay replays events
type EventReplay interface {
	Replay(ctx context.Context, from, to time.Time) error
	ReplayFromEvent(ctx context.Context, fromEventID string) error
	GetReplayStatus() ReplayStatus
}

// ReplayStatus represents the status of event replay
type ReplayStatus struct {
	IsRunning      bool      `json:"is_running"`
	StartTime      time.Time `json:"start_time"`
	EndTime        time.Time `json:"end_time"`
	ProcessedCount int64     `json:"processed_count"`
	ErrorCount     int64     `json:"error_count"`
	LastError      string    `json:"last_error,omitempty"`
}

// EventMetrics tracks event metrics
type EventMetrics interface {
	IncrementEventCount(eventType string)
	IncrementErrorCount(eventType string)
	RecordProcessingTime(eventType string, duration time.Duration)
	GetMetrics() EventMetricsData
}

// EventMetricsData represents event metrics
type EventMetricsData struct {
	EventCounts     map[string]int64         `json:"event_counts"`
	ErrorCounts     map[string]int64         `json:"error_counts"`
	ProcessingTimes map[string]time.Duration `json:"processing_times"`
	TotalEvents     int64                    `json:"total_events"`
	TotalErrors     int64                    `json:"total_errors"`
	AverageTime     time.Duration            `json:"average_time"`
}

// EventRetry handles event retry logic
type EventRetry interface {
	ShouldRetry(event Event, attempt int, err error) bool
	GetRetryDelay(attempt int) time.Duration
	GetMaxRetries() int
}

// EventDeadLetter handles failed events
type EventDeadLetter interface {
	HandleFailedEvent(ctx context.Context, event Event, err error) error
	GetFailedEvents(ctx context.Context, limit, offset int) ([]FailedEvent, error)
	RetryFailedEvent(ctx context.Context, eventID string) error
}

// FailedEvent represents a failed event
type FailedEvent struct {
	Event     Event     `json:"event"`
	Error     string    `json:"error"`
	Attempts  int       `json:"attempts"`
	FailedAt  time.Time `json:"failed_at"`
	LastRetry time.Time `json:"last_retry,omitempty"`
}

// EventScheduler schedules events
type EventScheduler interface {
	Schedule(ctx context.Context, event Event, delay time.Duration) error
	ScheduleAt(ctx context.Context, event Event, at time.Time) error
	Cancel(ctx context.Context, eventID string) error
	GetScheduledEvents(ctx context.Context) ([]ScheduledEvent, error)
}

// ScheduledEvent represents a scheduled event
type ScheduledEvent struct {
	Event       Event     `json:"event"`
	ScheduledAt time.Time `json:"scheduled_at"`
	CreatedAt   time.Time `json:"created_at"`
}

// EventValidator validates events
type EventValidator interface {
	Validate(event Event) error
	GetValidationRules() []ValidationRule
}

// ValidationRule represents a validation rule
type ValidationRule struct {
	Field    string      `json:"field"`
	Required bool        `json:"required"`
	Type     string      `json:"type"`
	Min      interface{} `json:"min,omitempty"`
	Max      interface{} `json:"max,omitempty"`
	Pattern  string      `json:"pattern,omitempty"`
}

// EventSerializer serializes events
type EventSerializer interface {
	Serialize(event Event) ([]byte, error)
	Deserialize(data []byte) (Event, error)
	GetContentType() string
}

// EventCompressor compresses events
type EventCompressor interface {
	Compress(data []byte) ([]byte, error)
	Decompress(data []byte) ([]byte, error)
	GetCompressionType() string
}

// EventEncryption encrypts events
type EventEncryption interface {
	Encrypt(data []byte) ([]byte, error)
	Decrypt(data []byte) ([]byte, error)
	GetEncryptionType() string
}

// EventRateLimiter limits event processing rate
type EventRateLimiter interface {
	Allow(eventType string) bool
	GetRateLimit(eventType string) RateLimit
	SetRateLimit(eventType string, limit RateLimit) error
}

// RateLimit represents rate limiting configuration
type RateLimit struct {
	Requests int           `json:"requests"`
	Window   time.Duration `json:"window"`
	Burst    int           `json:"burst"`
}

// EventCircuitBreaker implements circuit breaker pattern
type EventCircuitBreaker interface {
	Execute(ctx context.Context, eventType string, fn func() error) error
	GetState(eventType string) CircuitBreakerState
	Reset(eventType string) error
}

// CircuitBreakerState represents circuit breaker state
type CircuitBreakerState struct {
	State        string        `json:"state"` // "closed", "open", "half-open"
	FailureCount int           `json:"failure_count"`
	LastFailure  time.Time     `json:"last_failure,omitempty"`
	NextAttempt  time.Time     `json:"next_attempt,omitempty"`
	SuccessCount int           `json:"success_count"`
	Timeout      time.Duration `json:"timeout"`
}

// EventBatchProcessor processes events in batches
type EventBatchProcessor interface {
	ProcessBatch(ctx context.Context, events []Event) error
	GetBatchSize() int
	GetBatchTimeout() time.Duration
	SetBatchSize(size int) error
	SetBatchTimeout(timeout time.Duration) error
}

// EventPartitioner partitions events
type EventPartitioner interface {
	GetPartition(event Event, partitionCount int) int
	GetPartitionKey(event Event) string
}

// EventOrdering ensures event ordering
type EventOrdering interface {
	GetOrderingKey(event Event) string
	ShouldOrder(eventType string) bool
	GetOrderingStrategy() string
}

// EventDeduplication removes duplicate events
type EventDeduplication interface {
	IsDuplicate(event Event) bool
	MarkAsProcessed(event Event) error
	GetDeduplicationWindow() time.Duration
}

// EventCorrelation correlates related events
type EventCorrelation interface {
	GetCorrelationID(event Event) string
	SetCorrelationID(event Event, correlationID string) error
	GetRelatedEvents(ctx context.Context, correlationID string) ([]Event, error)
}

// EventEnrichment enriches events with additional data
type EventEnrichment interface {
	Enrich(ctx context.Context, event Event) (Event, error)
	GetEnrichmentRules() []EnrichmentRule
}

// EnrichmentRule represents an enrichment rule
type EnrichmentRule struct {
	EventType string                 `json:"event_type"`
	Fields    []string               `json:"fields"`
	Source    string                 `json:"source"`
	Transform map[string]interface{} `json:"transform,omitempty"`
	Condition string                 `json:"condition,omitempty"`
}

// EventRouting routes events to appropriate handlers
type EventRouting interface {
	Route(ctx context.Context, event Event) ([]EventHandler, error)
	GetRoutingRules() []RoutingRule
	AddRoutingRule(rule RoutingRule) error
	RemoveRoutingRule(ruleID string) error
}

// RoutingRule represents a routing rule
type RoutingRule struct {
	ID         string   `json:"id"`
	EventTypes []string `json:"event_types"`
	Conditions []string `json:"conditions"`
	Handlers   []string `json:"handlers"`
	Priority   int      `json:"priority"`
	Enabled    bool     `json:"enabled"`
}
