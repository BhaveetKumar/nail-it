package observer

import (
	"context"
	"fmt"
	"sync"
	"time"

	"observer-service/internal/logger"
)

// EventBusImpl implements EventBus interface
type EventBusImpl struct {
	subscribers map[string][]Observer
	mutex       sync.RWMutex
	logger      *logger.Logger
	metrics     EventMetrics
	retry       EventRetry
	deadLetter  EventDeadLetter
	rateLimiter EventRateLimiter
	circuitBreaker EventCircuitBreaker
}

// NewEventBus creates a new event bus
func NewEventBus() *EventBusImpl {
	return &EventBusImpl{
		subscribers: make(map[string][]Observer),
		logger:      logger.GetLogger(),
		metrics:     NewEventMetrics(),
		retry:       NewEventRetry(),
		deadLetter:  NewEventDeadLetter(),
		rateLimiter: NewEventRateLimiter(),
		circuitBreaker: NewEventCircuitBreaker(),
	}
}

// Subscribe subscribes an observer to an event type
func (eb *EventBusImpl) Subscribe(eventType string, observer Observer) error {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	// Check if observer is already subscribed
	for _, existingObserver := range eb.subscribers[eventType] {
		if existingObserver.GetObserverID() == observer.GetObserverID() {
			return fmt.Errorf("observer %s is already subscribed to event type %s", 
				observer.GetObserverID(), eventType)
		}
	}
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], observer)
	
	eb.logger.Info("Observer subscribed to event type", 
		"observer_id", observer.GetObserverID(), 
		"event_type", eventType)
	
	return nil
}

// Unsubscribe unsubscribes an observer from an event type
func (eb *EventBusImpl) Unsubscribe(eventType string, observerID string) error {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	observers, exists := eb.subscribers[eventType]
	if !exists {
		return fmt.Errorf("no subscribers found for event type %s", eventType)
	}
	
	for i, observer := range observers {
		if observer.GetObserverID() == observerID {
			eb.subscribers[eventType] = append(observers[:i], observers[i+1:]...)
			
			eb.logger.Info("Observer unsubscribed from event type", 
				"observer_id", observerID, 
				"event_type", eventType)
			
			return nil
		}
	}
	
	return fmt.Errorf("observer %s not found for event type %s", observerID, eventType)
}

// Publish publishes an event to all subscribers
func (eb *EventBusImpl) Publish(ctx context.Context, event Event) error {
	eb.mutex.RLock()
	observers, exists := eb.subscribers[event.GetType()]
	eb.mutex.RUnlock()
	
	if !exists {
		eb.logger.Debug("No subscribers found for event type", "event_type", event.GetType())
		return nil
	}
	
	eb.metrics.IncrementEventCount(event.GetType())
	
	// Check rate limit
	if !eb.rateLimiter.Allow(event.GetType()) {
		eb.logger.Warn("Rate limit exceeded for event type", "event_type", event.GetType())
		return fmt.Errorf("rate limit exceeded for event type %s", event.GetType())
	}
	
	// Publish to all subscribers
	for _, observer := range observers {
		go eb.publishToObserver(ctx, event, observer)
	}
	
	return nil
}

// publishToObserver publishes an event to a specific observer
func (eb *EventBusImpl) publishToObserver(ctx context.Context, event Event, observer Observer) {
	startTime := time.Now()
	
	// Use circuit breaker
	err := eb.circuitBreaker.Execute(ctx, event.GetType(), func() error {
		return eb.processEventWithRetry(ctx, event, observer)
	})
	
	processingTime := time.Since(startTime)
	eb.metrics.RecordProcessingTime(event.GetType(), processingTime)
	
	if err != nil {
		eb.metrics.IncrementErrorCount(event.GetType())
		eb.logger.Error("Failed to process event", 
			"event_id", event.GetID(),
			"event_type", event.GetType(),
			"observer_id", observer.GetObserverID(),
			"error", err)
		
		// Send to dead letter queue
		eb.deadLetter.HandleFailedEvent(ctx, event, err)
	}
}

// processEventWithRetry processes an event with retry logic
func (eb *EventBusImpl) processEventWithRetry(ctx context.Context, event Event, observer Observer) error {
	var lastErr error
	
	for attempt := 1; attempt <= eb.retry.GetMaxRetries(); attempt++ {
		err := observer.OnEvent(ctx, event)
		if err == nil {
			return nil
		}
		
		lastErr = err
		
		if !eb.retry.ShouldRetry(event, attempt, err) {
			break
		}
		
		if attempt < eb.retry.GetMaxRetries() {
			delay := eb.retry.GetRetryDelay(attempt)
			eb.logger.Debug("Retrying event processing", 
				"event_id", event.GetID(),
				"observer_id", observer.GetObserverID(),
				"attempt", attempt,
				"delay", delay)
			
			time.Sleep(delay)
		}
	}
	
	return lastErr
}

// GetSubscriberCount returns the number of subscribers for an event type
func (eb *EventBusImpl) GetSubscriberCount(eventType string) int {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	return len(eb.subscribers[eventType])
}

// GetEventTypes returns all event types with subscribers
func (eb *EventBusImpl) GetEventTypes() []string {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	eventTypes := make([]string, 0, len(eb.subscribers))
	for eventType := range eb.subscribers {
		eventTypes = append(eventTypes, eventType)
	}
	
	return eventTypes
}

// GetSubscribers returns all subscribers for an event type
func (eb *EventBusImpl) GetSubscribers(eventType string) []Observer {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	subscribers := make([]Observer, len(eb.subscribers[eventType]))
	copy(subscribers, eb.subscribers[eventType])
	
	return subscribers
}

// GetMetrics returns event bus metrics
func (eb *EventBusImpl) GetMetrics() EventMetricsData {
	return eb.metrics.GetMetrics()
}

// ClearSubscribers clears all subscribers (for testing)
func (eb *EventBusImpl) ClearSubscribers() {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	eb.subscribers = make(map[string][]Observer)
}

// EventBusWithFilters extends EventBus with filtering capabilities
type EventBusWithFilters struct {
	*EventBusImpl
	filters []EventFilter
}

// NewEventBusWithFilters creates a new event bus with filters
func NewEventBusWithFilters() *EventBusWithFilters {
	return &EventBusWithFilters{
		EventBusImpl: NewEventBus(),
		filters:      make([]EventFilter, 0),
	}
}

// AddFilter adds an event filter
func (eb *EventBusWithFilters) AddFilter(filter EventFilter) {
	eb.filters = append(eb.filters, filter)
}

// Publish publishes an event with filtering
func (eb *EventBusWithFilters) Publish(ctx context.Context, event Event) error {
	// Apply filters
	for _, filter := range eb.filters {
		if !filter.ShouldProcess(event) {
			eb.logger.Debug("Event filtered out", 
				"event_id", event.GetID(),
				"event_type", event.GetType(),
				"filter", filter.GetFilterName())
			return nil
		}
	}
	
	return eb.EventBusImpl.Publish(ctx, event)
}

// EventBusWithTransformation extends EventBus with transformation capabilities
type EventBusWithTransformation struct {
	*EventBusImpl
	transformers []EventTransformer
}

// NewEventBusWithTransformation creates a new event bus with transformers
func NewEventBusWithTransformation() *EventBusWithTransformation {
	return &EventBusWithTransformation{
		EventBusImpl: NewEventBus(),
		transformers: make([]EventTransformer, 0),
	}
}

// AddTransformer adds an event transformer
func (eb *EventBusWithTransformation) AddTransformer(transformer EventTransformer) {
	eb.transformers = append(eb.transformers, transformer)
}

// Publish publishes an event with transformation
func (eb *EventBusWithTransformation) Publish(ctx context.Context, event Event) error {
	// Apply transformations
	transformedEvent := event
	for _, transformer := range eb.transformers {
		var err error
		transformedEvent, err = transformer.Transform(transformedEvent)
		if err != nil {
			eb.logger.Error("Failed to transform event", 
				"event_id", event.GetID(),
				"transformer", transformer.GetTransformerName(),
				"error", err)
			return err
		}
	}
	
	return eb.EventBusImpl.Publish(ctx, transformedEvent)
}

// EventBusWithBatching extends EventBus with batching capabilities
type EventBusWithBatching struct {
	*EventBusImpl
	batchProcessor EventBatchProcessor
	eventBuffer    []Event
	bufferMutex    sync.Mutex
	batchTimer     *time.Timer
}

// NewEventBusWithBatching creates a new event bus with batching
func NewEventBusWithBatching() *EventBusWithBatching {
	eb := &EventBusWithBatching{
		EventBusImpl:  NewEventBus(),
		batchProcessor: NewEventBatchProcessor(),
		eventBuffer:    make([]Event, 0),
	}
	
	// Start batch processing timer
	eb.startBatchTimer()
	
	return eb
}

// Publish publishes an event with batching
func (eb *EventBusWithBatching) Publish(ctx context.Context, event Event) error {
	eb.bufferMutex.Lock()
	defer eb.bufferMutex.Unlock()
	
	eb.eventBuffer = append(eb.eventBuffer, event)
	
	// Process batch if buffer is full
	if len(eb.eventBuffer) >= eb.batchProcessor.GetBatchSize() {
		return eb.processBatch(ctx)
	}
	
	return nil
}

// processBatch processes the current batch
func (eb *EventBusWithBatching) processBatch(ctx context.Context) error {
	if len(eb.eventBuffer) == 0 {
		return nil
	}
	
	// Create a copy of the buffer
	batch := make([]Event, len(eb.eventBuffer))
	copy(batch, eb.eventBuffer)
	
	// Clear the buffer
	eb.eventBuffer = eb.eventBuffer[:0]
	
	// Process the batch
	return eb.batchProcessor.ProcessBatch(ctx, batch)
}

// startBatchTimer starts the batch processing timer
func (eb *EventBusWithBatching) startBatchTimer() {
	eb.batchTimer = time.AfterFunc(eb.batchProcessor.GetBatchTimeout(), func() {
		eb.bufferMutex.Lock()
		defer eb.bufferMutex.Unlock()
		
		if len(eb.eventBuffer) > 0 {
			eb.processBatch(context.Background())
		}
		
		// Restart timer
		eb.startBatchTimer()
	})
}

// Stop stops the batch processing timer
func (eb *EventBusWithBatching) Stop() {
	if eb.batchTimer != nil {
		eb.batchTimer.Stop()
	}
}

// EventBusWithOrdering extends EventBus with ordering capabilities
type EventBusWithOrdering struct {
	*EventBusImpl
	ordering EventOrdering
	partitions map[string][]Event
	partitionMutex sync.Mutex
}

// NewEventBusWithOrdering creates a new event bus with ordering
func NewEventBusWithOrdering() *EventBusWithOrdering {
	return &EventBusWithOrdering{
		EventBusImpl: NewEventBus(),
		ordering:     NewEventOrdering(),
		partitions:   make(map[string][]Event),
	}
}

// Publish publishes an event with ordering
func (eb *EventBusWithOrdering) Publish(ctx context.Context, event Event) error {
	if !eb.ordering.ShouldOrder(event.GetType()) {
		return eb.EventBusImpl.Publish(ctx, event)
	}
	
	eb.partitionMutex.Lock()
	defer eb.partitionMutex.Unlock()
	
	orderingKey := eb.ordering.GetOrderingKey(event)
	eb.partitions[orderingKey] = append(eb.partitions[orderingKey], event)
	
	// Process events in order for this partition
	go eb.processPartition(ctx, orderingKey)
	
	return nil
}

// processPartition processes events in a partition in order
func (eb *EventBusWithOrdering) processPartition(ctx context.Context, orderingKey string) {
	eb.partitionMutex.Lock()
	events := make([]Event, len(eb.partitions[orderingKey]))
	copy(events, eb.partitions[orderingKey])
	eb.partitions[orderingKey] = eb.partitions[orderingKey][:0]
	eb.partitionMutex.Unlock()
	
	// Process events in order
	for _, event := range events {
		eb.EventBusImpl.Publish(ctx, event)
	}
}
