package observer

import (
	"context"
	"fmt"
	"sync"
	"time"

	"observer-service/internal/logger"
)

// BaseObserver represents a base observer implementation
type BaseObserver struct {
	ID         string    `json:"id"`
	EventTypes []string  `json:"event_types"`
	Async      bool      `json:"async"`
	Enabled    bool      `json:"enabled"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// NewBaseObserver creates a new base observer
func NewBaseObserver(id string, eventTypes []string, async bool) *BaseObserver {
	return &BaseObserver{
		ID:         id,
		EventTypes: eventTypes,
		Async:      async,
		Enabled:    true,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
}

func (bo *BaseObserver) GetObserverID() string {
	return bo.ID
}

func (bo *BaseObserver) GetEventTypes() []string {
	return bo.EventTypes
}

func (bo *BaseObserver) IsAsync() bool {
	return bo.Async
}

func (bo *BaseObserver) IsEnabled() bool {
	return bo.Enabled
}

func (bo *BaseObserver) SetEnabled(enabled bool) {
	bo.Enabled = enabled
	bo.UpdatedAt = time.Now()
}

// PaymentObserver handles payment-related events
type PaymentObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *PaymentEvent) error
	logger  *logger.Logger
}

// NewPaymentObserver creates a new payment observer
func NewPaymentObserver(id string, handler func(ctx context.Context, event *PaymentEvent) error) *PaymentObserver {
	return &PaymentObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypePaymentCreated,
			EventTypePaymentUpdated,
			EventTypePaymentCompleted,
			EventTypePaymentFailed,
			EventTypePaymentRefunded,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (po *PaymentObserver) OnEvent(ctx context.Context, event Event) error {
	if !po.IsEnabled() {
		return nil
	}

	paymentEvent, ok := event.(*PaymentEvent)
	if !ok {
		return fmt.Errorf("expected PaymentEvent, got %T", event)
	}

	po.logger.Info("Processing payment event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"payment_id", paymentEvent.PaymentID,
		"user_id", paymentEvent.UserID)

	return po.handler(ctx, paymentEvent)
}

// UserObserver handles user-related events
type UserObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *UserEvent) error
	logger  *logger.Logger
}

// NewUserObserver creates a new user observer
func NewUserObserver(id string, handler func(ctx context.Context, event *UserEvent) error) *PaymentObserver {
	return &PaymentObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeUserCreated,
			EventTypeUserUpdated,
			EventTypeUserDeleted,
			EventTypeUserActivated,
			EventTypeUserDeactivated,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (uo *UserObserver) OnEvent(ctx context.Context, event Event) error {
	if !uo.IsEnabled() {
		return nil
	}

	userEvent, ok := event.(*UserEvent)
	if !ok {
		return fmt.Errorf("expected UserEvent, got %T", event)
	}

	uo.logger.Info("Processing user event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"user_id", userEvent.UserID,
		"email", userEvent.Email)

	return uo.handler(ctx, userEvent)
}

// OrderObserver handles order-related events
type OrderObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *OrderEvent) error
	logger  *logger.Logger
}

// NewOrderObserver creates a new order observer
func NewOrderObserver(id string, handler func(ctx context.Context, event *OrderEvent) error) *OrderObserver {
	return &OrderObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeOrderCreated,
			EventTypeOrderUpdated,
			EventTypeOrderCancelled,
			EventTypeOrderCompleted,
			EventTypeOrderShipped,
			EventTypeOrderDelivered,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (oo *OrderObserver) OnEvent(ctx context.Context, event Event) error {
	if !oo.IsEnabled() {
		return nil
	}

	orderEvent, ok := event.(*OrderEvent)
	if !ok {
		return fmt.Errorf("expected OrderEvent, got %T", event)
	}

	oo.logger.Info("Processing order event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"order_id", orderEvent.OrderID,
		"user_id", orderEvent.UserID)

	return oo.handler(ctx, orderEvent)
}

// ProductObserver handles product-related events
type ProductObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *ProductEvent) error
	logger  *logger.Logger
}

// NewProductObserver creates a new product observer
func NewProductObserver(id string, handler func(ctx context.Context, event *ProductEvent) error) *ProductObserver {
	return &ProductObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeProductCreated,
			EventTypeProductUpdated,
			EventTypeProductDeleted,
			EventTypeProductStockLow,
			EventTypeProductOutOfStock,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (po *ProductObserver) OnEvent(ctx context.Context, event Event) error {
	if !po.IsEnabled() {
		return nil
	}

	productEvent, ok := event.(*ProductEvent)
	if !ok {
		return fmt.Errorf("expected ProductEvent, got %T", event)
	}

	po.logger.Info("Processing product event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"product_id", productEvent.ProductID,
		"name", productEvent.Name)

	return po.handler(ctx, productEvent)
}

// NotificationObserver handles notification-related events
type NotificationObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *NotificationEvent) error
	logger  *logger.Logger
}

// NewNotificationObserver creates a new notification observer
func NewNotificationObserver(id string, handler func(ctx context.Context, event *NotificationEvent) error) *NotificationObserver {
	return &NotificationObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeNotificationSent,
			EventTypeNotificationFailed,
			EventTypeNotificationDelivered,
			EventTypeNotificationRead,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (no *NotificationObserver) OnEvent(ctx context.Context, event Event) error {
	if !no.IsEnabled() {
		return nil
	}

	notificationEvent, ok := event.(*NotificationEvent)
	if !ok {
		return fmt.Errorf("expected NotificationEvent, got %T", event)
	}

	no.logger.Info("Processing notification event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"notification_id", notificationEvent.NotificationID,
		"user_id", notificationEvent.UserID)

	return no.handler(ctx, notificationEvent)
}

// AuditObserver handles audit-related events
type AuditObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *AuditEvent) error
	logger  *logger.Logger
}

// NewAuditObserver creates a new audit observer
func NewAuditObserver(id string, handler func(ctx context.Context, event *AuditEvent) error) *AuditObserver {
	return &AuditObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeAuditLogCreated,
			EventTypeAuditLogUpdated,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (ao *AuditObserver) OnEvent(ctx context.Context, event Event) error {
	if !ao.IsEnabled() {
		return nil
	}

	auditEvent, ok := event.(*AuditEvent)
	if !ok {
		return fmt.Errorf("expected AuditEvent, got %T", event)
	}

	ao.logger.Info("Processing audit event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"entity_type", auditEvent.EntityType,
		"entity_id", auditEvent.EntityID)

	return ao.handler(ctx, auditEvent)
}

// SystemObserver handles system-related events
type SystemObserver struct {
	*BaseObserver
	handler func(ctx context.Context, event *SystemEvent) error
	logger  *logger.Logger
}

// NewSystemObserver creates a new system observer
func NewSystemObserver(id string, handler func(ctx context.Context, event *SystemEvent) error) *SystemObserver {
	return &SystemObserver{
		BaseObserver: NewBaseObserver(id, []string{
			EventTypeSystemStartup,
			EventTypeSystemShutdown,
			EventTypeSystemError,
			EventTypeSystemWarning,
			EventTypeSystemInfo,
		}, true),
		handler: handler,
		logger:  logger.GetLogger(),
	}
}

func (so *SystemObserver) OnEvent(ctx context.Context, event Event) error {
	if !so.IsEnabled() {
		return nil
	}

	systemEvent, ok := event.(*SystemEvent)
	if !ok {
		return fmt.Errorf("expected SystemEvent, got %T", event)
	}

	so.logger.Info("Processing system event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"component", systemEvent.Component,
		"level", systemEvent.Level)

	return so.handler(ctx, systemEvent)
}

// CompositeObserver combines multiple observers
type CompositeObserver struct {
	*BaseObserver
	observers []Observer
	logger    *logger.Logger
}

// NewCompositeObserver creates a new composite observer
func NewCompositeObserver(id string, observers []Observer) *CompositeObserver {
	eventTypes := make([]string, 0)
	for _, observer := range observers {
		eventTypes = append(eventTypes, observer.GetEventTypes()...)
	}

	return &CompositeObserver{
		BaseObserver: NewBaseObserver(id, eventTypes, true),
		observers:    observers,
		logger:       logger.GetLogger(),
	}
}

func (co *CompositeObserver) OnEvent(ctx context.Context, event Event) error {
	if !co.IsEnabled() {
		return nil
	}

	co.logger.Info("Processing composite event",
		"event_id", event.GetID(),
		"event_type", event.GetType(),
		"observer_count", len(co.observers))

	var wg sync.WaitGroup
	errChan := make(chan error, len(co.observers))

	for _, observer := range co.observers {
		wg.Add(1)
		go func(obs Observer) {
			defer wg.Done()

			if err := obs.OnEvent(ctx, event); err != nil {
				errChan <- err
			}
		}(observer)
	}

	wg.Wait()
	close(errChan)

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("composite observer failed with %d errors: %v", len(errors), errors)
	}

	return nil
}

// FilteredObserver wraps an observer with filtering
type FilteredObserver struct {
	*BaseObserver
	observer Observer
	filter   EventFilter
	logger   *logger.Logger
}

// NewFilteredObserver creates a new filtered observer
func NewFilteredObserver(id string, observer Observer, filter EventFilter) *FilteredObserver {
	return &FilteredObserver{
		BaseObserver: NewBaseObserver(id, observer.GetEventTypes(), observer.IsAsync()),
		observer:     observer,
		filter:       filter,
		logger:       logger.GetLogger(),
	}
}

func (fo *FilteredObserver) OnEvent(ctx context.Context, event Event) error {
	if !fo.IsEnabled() {
		return nil
	}

	if !fo.filter.ShouldProcess(event) {
		fo.logger.Debug("Event filtered out",
			"event_id", event.GetID(),
			"event_type", event.GetType(),
			"filter", fo.filter.GetFilterName())
		return nil
	}

	return fo.observer.OnEvent(ctx, event)
}

// RetryObserver wraps an observer with retry logic
type RetryObserver struct {
	*BaseObserver
	observer   Observer
	maxRetries int
	retryDelay time.Duration
	logger     *logger.Logger
}

// NewRetryObserver creates a new retry observer
func NewRetryObserver(id string, observer Observer, maxRetries int, retryDelay time.Duration) *RetryObserver {
	return &RetryObserver{
		BaseObserver: NewBaseObserver(id, observer.GetEventTypes(), observer.IsAsync()),
		observer:     observer,
		maxRetries:   maxRetries,
		retryDelay:   retryDelay,
		logger:       logger.GetLogger(),
	}
}

func (ro *RetryObserver) OnEvent(ctx context.Context, event Event) error {
	if !ro.IsEnabled() {
		return nil
	}

	var lastErr error

	for attempt := 1; attempt <= ro.maxRetries; attempt++ {
		err := ro.observer.OnEvent(ctx, event)
		if err == nil {
			return nil
		}

		lastErr = err

		if attempt < ro.maxRetries {
			ro.logger.Debug("Retrying observer",
				"event_id", event.GetID(),
				"observer_id", ro.observer.GetObserverID(),
				"attempt", attempt,
				"delay", ro.retryDelay)

			time.Sleep(ro.retryDelay)
		}
	}

	return lastErr
}

// MetricsObserver tracks observer metrics
type MetricsObserver struct {
	*BaseObserver
	observer Observer
	metrics  EventMetrics
	logger   *logger.Logger
}

// NewMetricsObserver creates a new metrics observer
func NewMetricsObserver(id string, observer Observer, metrics EventMetrics) *MetricsObserver {
	return &MetricsObserver{
		BaseObserver: NewBaseObserver(id, observer.GetEventTypes(), observer.IsAsync()),
		observer:     observer,
		metrics:      metrics,
		logger:       logger.GetLogger(),
	}
}

func (mo *MetricsObserver) OnEvent(ctx context.Context, event Event) error {
	if !mo.IsEnabled() {
		return nil
	}

	startTime := time.Now()

	err := mo.observer.OnEvent(ctx, event)

	processingTime := time.Since(startTime)
	mo.metrics.RecordProcessingTime(event.GetType(), processingTime)

	if err != nil {
		mo.metrics.IncrementErrorCount(event.GetType())
	}

	return err
}
