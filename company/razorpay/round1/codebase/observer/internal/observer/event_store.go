package observer

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"observer-service/internal/logger"
)

// EventStoreImpl implements EventStore interface
type EventStoreImpl struct {
	collection *mongo.Collection
	logger     *logger.Logger
}

// NewEventStore creates a new event store
func NewEventStore(collection *mongo.Collection) *EventStoreImpl {
	return &EventStoreImpl{
		collection: collection,
		logger:     logger.GetLogger(),
	}
}

// Store stores an event
func (es *EventStoreImpl) Store(ctx context.Context, event Event) error {
	eventDoc := bson.M{
		"id":        event.GetID(),
		"type":      event.GetType(),
		"timestamp": event.GetTimestamp(),
		"data":      event.GetData(),
		"source":    event.GetSource(),
		"version":   "1.0",
		"created_at": time.Now(),
	}
	
	// Add metadata if available
	if baseEvent, ok := event.(*BaseEvent); ok {
		eventDoc["metadata"] = baseEvent.GetMetadata()
	}
	
	_, err := es.collection.InsertOne(ctx, eventDoc)
	if err != nil {
		es.logger.Error("Failed to store event", 
			"event_id", event.GetID(),
			"event_type", event.GetType(),
			"error", err)
		return fmt.Errorf("failed to store event: %w", err)
	}
	
	es.logger.Debug("Event stored successfully", 
		"event_id", event.GetID(),
		"event_type", event.GetType())
	
	return nil
}

// GetEvents retrieves events by type with pagination
func (es *EventStoreImpl) GetEvents(ctx context.Context, eventType string, limit, offset int) ([]Event, error) {
	filter := bson.M{"type": eventType}
	opts := options.Find().
		SetSort(bson.D{{"timestamp", -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))
	
	cursor, err := es.collection.Find(ctx, filter, opts)
	if err != nil {
		es.logger.Error("Failed to get events", 
			"event_type", eventType,
			"error", err)
		return nil, fmt.Errorf("failed to get events: %w", err)
	}
	defer cursor.Close(ctx)
	
	var events []Event
	for cursor.Next(ctx) {
		var eventDoc bson.M
		if err := cursor.Decode(&eventDoc); err != nil {
			es.logger.Error("Failed to decode event", "error", err)
			continue
		}
		
		event, err := es.documentToEvent(eventDoc)
		if err != nil {
			es.logger.Error("Failed to convert document to event", "error", err)
			continue
		}
		
		events = append(events, event)
	}
	
	return events, nil
}

// GetEventByID retrieves an event by ID
func (es *EventStoreImpl) GetEventByID(ctx context.Context, eventID string) (Event, error) {
	filter := bson.M{"id": eventID}
	
	var eventDoc bson.M
	err := es.collection.FindOne(ctx, filter).Decode(&eventDoc)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			return nil, fmt.Errorf("event with ID %s not found", eventID)
		}
		es.logger.Error("Failed to get event by ID", 
			"event_id", eventID,
			"error", err)
		return nil, fmt.Errorf("failed to get event by ID: %w", err)
	}
	
	event, err := es.documentToEvent(eventDoc)
	if err != nil {
		es.logger.Error("Failed to convert document to event", "error", err)
		return nil, fmt.Errorf("failed to convert document to event: %w", err)
	}
	
	return event, nil
}

// GetEventsByTimeRange retrieves events within a time range
func (es *EventStoreImpl) GetEventsByTimeRange(ctx context.Context, start, end time.Time) ([]Event, error) {
	filter := bson.M{
		"timestamp": bson.M{
			"$gte": start,
			"$lte": end,
		},
	}
	
	opts := options.Find().SetSort(bson.D{{"timestamp", -1}})
	
	cursor, err := es.collection.Find(ctx, filter, opts)
	if err != nil {
		es.logger.Error("Failed to get events by time range", 
			"start", start,
			"end", end,
			"error", err)
		return nil, fmt.Errorf("failed to get events by time range: %w", err)
	}
	defer cursor.Close(ctx)
	
	var events []Event
	for cursor.Next(ctx) {
		var eventDoc bson.M
		if err := cursor.Decode(&eventDoc); err != nil {
			es.logger.Error("Failed to decode event", "error", err)
			continue
		}
		
		event, err := es.documentToEvent(eventDoc)
		if err != nil {
			es.logger.Error("Failed to convert document to event", "error", err)
			continue
		}
		
		events = append(events, event)
	}
	
	return events, nil
}

// documentToEvent converts a MongoDB document to an Event
func (es *EventStoreImpl) documentToEvent(doc bson.M) (Event, error) {
	eventType, ok := doc["type"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}
	
	eventID, ok := doc["id"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid event ID")
	}
	
	timestamp, ok := doc["timestamp"].(time.Time)
	if !ok {
		return nil, fmt.Errorf("invalid timestamp")
	}
	
	source, ok := doc["source"].(string)
	if !ok {
		source = "unknown"
	}
	
	baseEvent := &BaseEvent{
		ID:        eventID,
		Type:      eventType,
		Timestamp: timestamp,
		Data:      doc["data"],
		Source:    source,
		Version:   "1.0",
	}
	
	// Add metadata if available
	if metadata, ok := doc["metadata"].(bson.M); ok {
		baseEvent.Metadata = make(map[string]interface{})
		for k, v := range metadata {
			baseEvent.Metadata[k] = v
		}
	}
	
	// Create specific event type based on event type
	switch eventType {
	case EventTypePaymentCreated, EventTypePaymentUpdated, EventTypePaymentCompleted, EventTypePaymentFailed, EventTypePaymentRefunded:
		return es.createPaymentEvent(baseEvent, doc)
	case EventTypeUserCreated, EventTypeUserUpdated, EventTypeUserDeleted, EventTypeUserActivated, EventTypeUserDeactivated:
		return es.createUserEvent(baseEvent, doc)
	case EventTypeOrderCreated, EventTypeOrderUpdated, EventTypeOrderCancelled, EventTypeOrderCompleted, EventTypeOrderShipped, EventTypeOrderDelivered:
		return es.createOrderEvent(baseEvent, doc)
	case EventTypeProductCreated, EventTypeProductUpdated, EventTypeProductDeleted, EventTypeProductStockLow, EventTypeProductOutOfStock:
		return es.createProductEvent(baseEvent, doc)
	case EventTypeNotificationSent, EventTypeNotificationFailed, EventTypeNotificationDelivered, EventTypeNotificationRead:
		return es.createNotificationEvent(baseEvent, doc)
	case EventTypeAuditLogCreated, EventTypeAuditLogUpdated:
		return es.createAuditEvent(baseEvent, doc)
	case EventTypeSystemStartup, EventTypeSystemShutdown, EventTypeSystemError, EventTypeSystemWarning, EventTypeSystemInfo:
		return es.createSystemEvent(baseEvent, doc)
	default:
		return baseEvent, nil
	}
}

// createPaymentEvent creates a PaymentEvent from base event and document
func (es *EventStoreImpl) createPaymentEvent(baseEvent *BaseEvent, doc bson.M) (*PaymentEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid payment event data")
	}
	
	paymentID, _ := data["payment_id"].(string)
	userID, _ := data["user_id"].(string)
	amount, _ := data["amount"].(float64)
	currency, _ := data["currency"].(string)
	status, _ := data["status"].(string)
	gateway, _ := data["gateway"].(string)
	transactionID, _ := data["transaction_id"].(string)
	
	return &PaymentEvent{
		BaseEvent:     baseEvent,
		PaymentID:     paymentID,
		UserID:        userID,
		Amount:        amount,
		Currency:      currency,
		Status:        status,
		Gateway:       gateway,
		TransactionID: transactionID,
	}, nil
}

// createUserEvent creates a UserEvent from base event and document
func (es *EventStoreImpl) createUserEvent(baseEvent *BaseEvent, doc bson.M) (*UserEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid user event data")
	}
	
	userID, _ := data["user_id"].(string)
	email, _ := data["email"].(string)
	name, _ := data["name"].(string)
	status, _ := data["status"].(string)
	action, _ := data["action"].(string)
	
	return &UserEvent{
		BaseEvent: baseEvent,
		UserID:    userID,
		Email:     email,
		Name:      name,
		Status:    status,
		Action:    action,
	}, nil
}

// createOrderEvent creates an OrderEvent from base event and document
func (es *EventStoreImpl) createOrderEvent(baseEvent *BaseEvent, doc bson.M) (*OrderEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid order event data")
	}
	
	orderID, _ := data["order_id"].(string)
	userID, _ := data["user_id"].(string)
	paymentID, _ := data["payment_id"].(string)
	totalAmount, _ := data["total_amount"].(float64)
	currency, _ := data["currency"].(string)
	status, _ := data["status"].(string)
	
	var items []OrderItem
	if itemsData, ok := data["items"].(bson.A); ok {
		for _, itemData := range itemsData {
			if item, ok := itemData.(bson.M); ok {
				orderItem := OrderItem{
					ProductID: item["product_id"].(string),
					Quantity:  int(item["quantity"].(int32)),
					Price:     item["price"].(float64),
					Total:     item["total"].(float64),
				}
				items = append(items, orderItem)
			}
		}
	}
	
	return &OrderEvent{
		BaseEvent:   baseEvent,
		OrderID:     orderID,
		UserID:      userID,
		PaymentID:   paymentID,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      status,
		Items:       items,
	}, nil
}

// createProductEvent creates a ProductEvent from base event and document
func (es *EventStoreImpl) createProductEvent(baseEvent *BaseEvent, doc bson.M) (*ProductEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid product event data")
	}
	
	productID, _ := data["product_id"].(string)
	name, _ := data["name"].(string)
	category, _ := data["category"].(string)
	price, _ := data["price"].(float64)
	currency, _ := data["currency"].(string)
	stock, _ := data["stock"].(int32)
	status, _ := data["status"].(string)
	action, _ := data["action"].(string)
	
	return &ProductEvent{
		BaseEvent: baseEvent,
		ProductID: productID,
		Name:      name,
		Category:  category,
		Price:     price,
		Currency:  currency,
		Stock:     int(stock),
		Status:    status,
		Action:    action,
	}, nil
}

// createNotificationEvent creates a NotificationEvent from base event and document
func (es *EventStoreImpl) createNotificationEvent(baseEvent *BaseEvent, doc bson.M) (*NotificationEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid notification event data")
	}
	
	notificationID, _ := data["notification_id"].(string)
	userID, _ := data["user_id"].(string)
	channel, _ := data["channel"].(string)
	notificationType, _ := data["type"].(string)
	subject, _ := data["subject"].(string)
	message, _ := data["message"].(string)
	status, _ := data["status"].(string)
	
	var sentAt *time.Time
	if sentAtData, ok := data["sent_at"].(time.Time); ok {
		sentAt = &sentAtData
	}
	
	return &NotificationEvent{
		BaseEvent:      baseEvent,
		NotificationID: notificationID,
		UserID:         userID,
		Channel:        channel,
		Type:           notificationType,
		Subject:        subject,
		Message:        message,
		Status:         status,
		SentAt:         sentAt,
	}, nil
}

// createAuditEvent creates an AuditEvent from base event and document
func (es *EventStoreImpl) createAuditEvent(baseEvent *BaseEvent, doc bson.M) (*AuditEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid audit event data")
	}
	
	entityType, _ := data["entity_type"].(string)
	entityID, _ := data["entity_id"].(string)
	action, _ := data["action"].(string)
	userID, _ := data["user_id"].(string)
	ipAddress, _ := data["ip_address"].(string)
	userAgent, _ := data["user_agent"].(string)
	
	var changes map[string]interface{}
	if changesData, ok := data["changes"].(bson.M); ok {
		changes = make(map[string]interface{})
		for k, v := range changesData {
			changes[k] = v
		}
	}
	
	return &AuditEvent{
		BaseEvent:  baseEvent,
		EntityType: entityType,
		EntityID:   entityID,
		Action:     action,
		UserID:     userID,
		Changes:    changes,
		IPAddress:  ipAddress,
		UserAgent:  userAgent,
	}, nil
}

// createSystemEvent creates a SystemEvent from base event and document
func (es *EventStoreImpl) createSystemEvent(baseEvent *BaseEvent, doc bson.M) (*SystemEvent, error) {
	data, ok := doc["data"].(bson.M)
	if !ok {
		return nil, fmt.Errorf("invalid system event data")
	}
	
	component, _ := data["component"].(string)
	level, _ := data["level"].(string)
	message, _ := data["message"].(string)
	
	var details map[string]interface{}
	if detailsData, ok := data["details"].(bson.M); ok {
		details = make(map[string]interface{})
		for k, v := range detailsData {
			details[k] = v
		}
	}
	
	return &SystemEvent{
		BaseEvent: baseEvent,
		Component: component,
		Level:     level,
		Message:   message,
		Details:   details,
	}, nil
}

// CreateIndexes creates necessary indexes for the event store
func (es *EventStoreImpl) CreateIndexes(ctx context.Context) error {
	indexes := []mongo.IndexModel{
		{
			Keys: bson.D{{"id", 1}},
			Options: options.Index().SetUnique(true),
		},
		{
			Keys: bson.D{{"type", 1}},
		},
		{
			Keys: bson.D{{"timestamp", -1}},
		},
		{
			Keys: bson.D{{"source", 1}},
		},
		{
			Keys: bson.D{{"type", 1}, {"timestamp", -1}},
		},
		{
			Keys: bson.D{{"created_at", -1}},
		},
	}
	
	_, err := es.collection.Indexes().CreateMany(ctx, indexes)
	if err != nil {
		es.logger.Error("Failed to create indexes", "error", err)
		return fmt.Errorf("failed to create indexes: %w", err)
	}
	
	es.logger.Info("Event store indexes created successfully")
	return nil
}

// GetEventCount returns the total number of events
func (es *EventStoreImpl) GetEventCount(ctx context.Context) (int64, error) {
	count, err := es.collection.CountDocuments(ctx, bson.M{})
	if err != nil {
		es.logger.Error("Failed to get event count", "error", err)
		return 0, fmt.Errorf("failed to get event count: %w", err)
	}
	
	return count, nil
}

// GetEventCountByType returns the number of events by type
func (es *EventStoreImpl) GetEventCountByType(ctx context.Context, eventType string) (int64, error) {
	filter := bson.M{"type": eventType}
	count, err := es.collection.CountDocuments(ctx, filter)
	if err != nil {
		es.logger.Error("Failed to get event count by type", 
			"event_type", eventType,
			"error", err)
		return 0, fmt.Errorf("failed to get event count by type: %w", err)
	}
	
	return count, nil
}

// DeleteOldEvents deletes events older than the specified duration
func (es *EventStoreImpl) DeleteOldEvents(ctx context.Context, olderThan time.Duration) (int64, error) {
	cutoffTime := time.Now().Add(-olderThan)
	filter := bson.M{"timestamp": bson.M{"$lt": cutoffTime}}
	
	result, err := es.collection.DeleteMany(ctx, filter)
	if err != nil {
		es.logger.Error("Failed to delete old events", 
			"cutoff_time", cutoffTime,
			"error", err)
		return 0, fmt.Errorf("failed to delete old events: %w", err)
	}
	
	es.logger.Info("Deleted old events", 
		"count", result.DeletedCount,
		"cutoff_time", cutoffTime)
	
	return result.DeletedCount, nil
}
