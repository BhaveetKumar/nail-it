package observer

import (
	"sync"
	"time"
)

// EventMetricsImpl implements EventMetrics interface
type EventMetricsImpl struct {
	eventCounts     map[string]int64
	errorCounts     map[string]int64
	processingTimes map[string][]time.Duration
	mutex           sync.RWMutex
}

// NewEventMetrics creates a new event metrics instance
func NewEventMetrics() *EventMetricsImpl {
	return &EventMetricsImpl{
		eventCounts:     make(map[string]int64),
		errorCounts:     make(map[string]int64),
		processingTimes: make(map[string][]time.Duration),
	}
}

// IncrementEventCount increments the event count for a specific event type
func (em *EventMetricsImpl) IncrementEventCount(eventType string) {
	em.mutex.Lock()
	defer em.mutex.Unlock()
	
	em.eventCounts[eventType]++
}

// IncrementErrorCount increments the error count for a specific event type
func (em *EventMetricsImpl) IncrementErrorCount(eventType string) {
	em.mutex.Lock()
	defer em.mutex.Unlock()
	
	em.errorCounts[eventType]++
}

// RecordProcessingTime records the processing time for a specific event type
func (em *EventMetricsImpl) RecordProcessingTime(eventType string, duration time.Duration) {
	em.mutex.Lock()
	defer em.mutex.Unlock()
	
	if em.processingTimes[eventType] == nil {
		em.processingTimes[eventType] = make([]time.Duration, 0)
	}
	
	em.processingTimes[eventType] = append(em.processingTimes[eventType], duration)
	
	// Keep only the last 1000 processing times to prevent memory growth
	if len(em.processingTimes[eventType]) > 1000 {
		em.processingTimes[eventType] = em.processingTimes[eventType][len(em.processingTimes[eventType])-1000:]
	}
}

// GetMetrics returns the current metrics data
func (em *EventMetricsImpl) GetMetrics() EventMetricsData {
	em.mutex.RLock()
	defer em.mutex.RUnlock()
	
	// Calculate total events and errors
	var totalEvents, totalErrors int64
	for _, count := range em.eventCounts {
		totalEvents += count
	}
	for _, count := range em.errorCounts {
		totalErrors += count
	}
	
	// Calculate average processing time
	var totalProcessingTime time.Duration
	var totalProcessingCount int
	for _, times := range em.processingTimes {
		for _, duration := range times {
			totalProcessingTime += duration
			totalProcessingCount++
		}
	}
	
	var averageTime time.Duration
	if totalProcessingCount > 0 {
		averageTime = totalProcessingTime / time.Duration(totalProcessingCount)
	}
	
	// Create copies of maps to avoid race conditions
	eventCounts := make(map[string]int64)
	for k, v := range em.eventCounts {
		eventCounts[k] = v
	}
	
	errorCounts := make(map[string]int64)
	for k, v := range em.errorCounts {
		errorCounts[k] = v
	}
	
	processingTimes := make(map[string]time.Duration)
	for k, times := range em.processingTimes {
		if len(times) > 0 {
			var sum time.Duration
			for _, duration := range times {
				sum += duration
			}
			processingTimes[k] = sum / time.Duration(len(times))
		}
	}
	
	return EventMetricsData{
		EventCounts:     eventCounts,
		ErrorCounts:     errorCounts,
		ProcessingTimes: processingTimes,
		TotalEvents:     totalEvents,
		TotalErrors:     totalErrors,
		AverageTime:     averageTime,
	}
}

// Reset resets all metrics
func (em *EventMetricsImpl) Reset() {
	em.mutex.Lock()
	defer em.mutex.Unlock()
	
	em.eventCounts = make(map[string]int64)
	em.errorCounts = make(map[string]int64)
	em.processingTimes = make(map[string][]time.Duration)
}

// EventRetryImpl implements EventRetry interface
type EventRetryImpl struct {
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
}

// NewEventRetry creates a new event retry instance
func NewEventRetry() *EventRetryImpl {
	return &EventRetryImpl{
		maxRetries: 3,
		baseDelay:  time.Second,
		maxDelay:   time.Minute,
	}
}

// ShouldRetry determines if an event should be retried
func (er *EventRetryImpl) ShouldRetry(event Event, attempt int, err error) bool {
	if attempt >= er.maxRetries {
		return false
	}
	
	// Don't retry certain types of errors
	if err == nil {
		return false
	}
	
	// Add logic to determine if error is retryable
	// For now, retry all errors
	return true
}

// GetRetryDelay returns the delay for the next retry attempt
func (er *EventRetryImpl) GetRetryDelay(attempt int) time.Duration {
	delay := er.baseDelay * time.Duration(1<<uint(attempt-1)) // Exponential backoff
	if delay > er.maxDelay {
		delay = er.maxDelay
	}
	return delay
}

// GetMaxRetries returns the maximum number of retries
func (er *EventRetryImpl) GetMaxRetries() int {
	return er.maxRetries
}

// EventDeadLetterImpl implements EventDeadLetter interface
type EventDeadLetterImpl struct {
	failedEvents []FailedEvent
	mutex        sync.RWMutex
}

// NewEventDeadLetter creates a new event dead letter instance
func NewEventDeadLetter() *EventDeadLetterImpl {
	return &EventDeadLetterImpl{
		failedEvents: make([]FailedEvent, 0),
	}
}

// HandleFailedEvent handles a failed event
func (edl *EventDeadLetterImpl) HandleFailedEvent(ctx context.Context, event Event, err error) error {
	edl.mutex.Lock()
	defer edl.mutex.Unlock()
	
	failedEvent := FailedEvent{
		Event:    event,
		Error:    err.Error(),
		Attempts: 1,
		FailedAt: time.Now(),
	}
	
	edl.failedEvents = append(edl.failedEvents, failedEvent)
	
	// Keep only the last 1000 failed events to prevent memory growth
	if len(edl.failedEvents) > 1000 {
		edl.failedEvents = edl.failedEvents[len(edl.failedEvents)-1000:]
	}
	
	return nil
}

// GetFailedEvents returns failed events
func (edl *EventDeadLetterImpl) GetFailedEvents(ctx context.Context, limit, offset int) ([]FailedEvent, error) {
	edl.mutex.RLock()
	defer edl.mutex.RUnlock()
	
	start := offset
	end := offset + limit
	
	if start >= len(edl.failedEvents) {
		return []FailedEvent{}, nil
	}
	
	if end > len(edl.failedEvents) {
		end = len(edl.failedEvents)
	}
	
	return edl.failedEvents[start:end], nil
}

// RetryFailedEvent retries a failed event
func (edl *EventDeadLetterImpl) RetryFailedEvent(ctx context.Context, eventID string) error {
	edl.mutex.Lock()
	defer edl.mutex.Unlock()
	
	for i, failedEvent := range edl.failedEvents {
		if failedEvent.Event.GetID() == eventID {
			// Remove from failed events
			edl.failedEvents = append(edl.failedEvents[:i], edl.failedEvents[i+1:]...)
			return nil
		}
	}
	
	return fmt.Errorf("failed event with ID %s not found", eventID)
}

// EventRateLimiterImpl implements EventRateLimiter interface
type EventRateLimiterImpl struct {
	rateLimits map[string]RateLimit
	mutex      sync.RWMutex
}

// NewEventRateLimiter creates a new event rate limiter instance
func NewEventRateLimiter() *EventRateLimiterImpl {
	return &EventRateLimiterImpl{
		rateLimits: make(map[string]RateLimit),
	}
}

// Allow checks if an event type is allowed
func (erl *EventRateLimiterImpl) Allow(eventType string) bool {
	erl.mutex.RLock()
	limit, exists := erl.rateLimits[eventType]
	erl.mutex.RUnlock()
	
	if !exists {
		// No rate limit set, allow
		return true
	}
	
	// Simple rate limiting implementation
	// In a real implementation, you would use a more sophisticated algorithm
	// like token bucket or sliding window
	return true
}

// GetRateLimit returns the rate limit for an event type
func (erl *EventRateLimiterImpl) GetRateLimit(eventType string) RateLimit {
	erl.mutex.RLock()
	defer erl.mutex.RUnlock()
	
	if limit, exists := erl.rateLimits[eventType]; exists {
		return limit
	}
	
	// Default rate limit
	return RateLimit{
		Requests: 1000,
		Window:   time.Minute,
		Burst:    100,
	}
}

// SetRateLimit sets the rate limit for an event type
func (erl *EventRateLimiterImpl) SetRateLimit(eventType string, limit RateLimit) error {
	erl.mutex.Lock()
	defer erl.mutex.Unlock()
	
	erl.rateLimits[eventType] = limit
	return nil
}

// EventCircuitBreakerImpl implements EventCircuitBreaker interface
type EventCircuitBreakerImpl struct {
	states map[string]CircuitBreakerState
	mutex  sync.RWMutex
}

// NewEventCircuitBreaker creates a new event circuit breaker instance
func NewEventCircuitBreaker() *EventCircuitBreakerImpl {
	return &EventCircuitBreakerImpl{
		states: make(map[string]CircuitBreakerState),
	}
}

// Execute executes a function with circuit breaker protection
func (ecb *EventCircuitBreakerImpl) Execute(ctx context.Context, eventType string, fn func() error) error {
	ecb.mutex.Lock()
	state, exists := ecb.states[eventType]
	if !exists {
		state = CircuitBreakerState{
			State:   "closed",
			Timeout: time.Minute,
		}
		ecb.states[eventType] = state
	}
	ecb.mutex.Unlock()
	
	// Check circuit breaker state
	if state.State == "open" {
		if time.Since(state.LastFailure) < state.Timeout {
			return fmt.Errorf("circuit breaker is open for event type %s", eventType)
		}
		// Move to half-open state
		state.State = "half-open"
		ecb.mutex.Lock()
		ecb.states[eventType] = state
		ecb.mutex.Unlock()
	}
	
	// Execute function
	err := fn()
	
	ecb.mutex.Lock()
	defer ecb.mutex.Unlock()
	
	if err != nil {
		// Function failed
		state.FailureCount++
		state.LastFailure = time.Now()
		
		if state.FailureCount >= 5 { // Threshold for opening circuit
			state.State = "open"
			state.NextAttempt = time.Now().Add(state.Timeout)
		}
	} else {
		// Function succeeded
		state.SuccessCount++
		if state.State == "half-open" {
			state.State = "closed"
			state.FailureCount = 0
		}
	}
	
	ecb.states[eventType] = state
	return err
}

// GetState returns the circuit breaker state for an event type
func (ecb *EventCircuitBreakerImpl) GetState(eventType string) CircuitBreakerState {
	ecb.mutex.RLock()
	defer ecb.mutex.RUnlock()
	
	if state, exists := ecb.states[eventType]; exists {
		return state
	}
	
	return CircuitBreakerState{
		State:   "closed",
		Timeout: time.Minute,
	}
}

// Reset resets the circuit breaker for an event type
func (ecb *EventCircuitBreakerImpl) Reset(eventType string) error {
	ecb.mutex.Lock()
	defer ecb.mutex.Unlock()
	
	ecb.states[eventType] = CircuitBreakerState{
		State:   "closed",
		Timeout: time.Minute,
	}
	
	return nil
}

// EventBatchProcessorImpl implements EventBatchProcessor interface
type EventBatchProcessorImpl struct {
	batchSize    int
	batchTimeout time.Duration
	mutex        sync.RWMutex
}

// NewEventBatchProcessor creates a new event batch processor instance
func NewEventBatchProcessor() *EventBatchProcessorImpl {
	return &EventBatchProcessorImpl{
		batchSize:    100,
		batchTimeout: time.Second * 5,
	}
}

// ProcessBatch processes a batch of events
func (ebp *EventBatchProcessorImpl) ProcessBatch(ctx context.Context, events []Event) error {
	// Simple batch processing implementation
	// In a real implementation, you would process events in parallel
	// and handle errors appropriately
	
	for _, event := range events {
		// Process each event
		_ = event
	}
	
	return nil
}

// GetBatchSize returns the batch size
func (ebp *EventBatchProcessorImpl) GetBatchSize() int {
	ebp.mutex.RLock()
	defer ebp.mutex.RUnlock()
	
	return ebp.batchSize
}

// GetBatchTimeout returns the batch timeout
func (ebp *EventBatchProcessorImpl) GetBatchTimeout() time.Duration {
	ebp.mutex.RLock()
	defer ebp.mutex.RUnlock()
	
	return ebp.batchTimeout
}

// SetBatchSize sets the batch size
func (ebp *EventBatchProcessorImpl) SetBatchSize(size int) error {
	ebp.mutex.Lock()
	defer ebp.mutex.Unlock()
	
	if size <= 0 {
		return fmt.Errorf("batch size must be positive")
	}
	
	ebp.batchSize = size
	return nil
}

// SetBatchTimeout sets the batch timeout
func (ebp *EventBatchProcessorImpl) SetBatchTimeout(timeout time.Duration) error {
	ebp.mutex.Lock()
	defer ebp.mutex.Unlock()
	
	if timeout <= 0 {
		return fmt.Errorf("batch timeout must be positive")
	}
	
	ebp.batchTimeout = timeout
	return nil
}
