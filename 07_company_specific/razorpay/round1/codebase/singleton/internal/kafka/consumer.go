package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"singleton-service/internal/config"
	"singleton-service/internal/logger"

	"github.com/segmentio/kafka-go"
)

// KafkaConsumer implements Singleton pattern for Kafka consumer
type KafkaConsumer struct {
	reader *kafka.Reader
	mutex  sync.RWMutex
}

var (
	kafkaConsumer *KafkaConsumer
	consumerOnce  sync.Once
)

// GetKafkaConsumer returns the singleton instance of KafkaConsumer
func GetKafkaConsumer() *KafkaConsumer {
	consumerOnce.Do(func() {
		kafkaConsumer = &KafkaConsumer{}
		kafkaConsumer.connect()
	})
	return kafkaConsumer
}

// connect establishes connection to Kafka
func (kc *KafkaConsumer) connect() {
	cfg := config.GetConfigManager()
	kafkaConfig := cfg.GetKafkaConfig()
	log := logger.GetLogger()

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:        kafkaConfig.Brokers,
		Topic:          kafkaConfig.Topic,
		GroupID:        kafkaConfig.GroupID,
		MinBytes:       10e3, // 10KB
		MaxBytes:       10e6, // 10MB
		CommitInterval: time.Second,
		StartOffset:    kafka.LastOffset,
	})

	kc.mutex.Lock()
	kc.reader = reader
	kc.mutex.Unlock()

	log.Info("Kafka consumer connected successfully")
}

// Start starts consuming messages from Kafka
func (kc *KafkaConsumer) Start(ctx context.Context) {
	kc.mutex.RLock()
	reader := kc.reader
	kc.mutex.RUnlock()

	if reader == nil {
		log := logger.GetLogger()
		log.Error("Kafka reader is not initialized")
		return
	}

	log := logger.GetLogger()
	log.Info("Starting Kafka consumer")

	for {
		select {
		case <-ctx.Done():
			log.Info("Kafka consumer stopping due to context cancellation")
			return
		default:
			// Read message
			message, err := reader.ReadMessage(ctx)
			if err != nil {
				log.Error("Failed to read message from Kafka", "error", err)
				time.Sleep(time.Second)
				continue
			}

			// Process message
			if err := kc.processMessage(ctx, message); err != nil {
				log.Error("Failed to process message", "error", err)
			}
		}
	}
}

// processMessage processes a Kafka message
func (kc *KafkaConsumer) processMessage(ctx context.Context, message kafka.Message) error {
	log := logger.GetLogger()

	// Parse message
	var event map[string]interface{}
	if err := json.Unmarshal(message.Value, &event); err != nil {
		return fmt.Errorf("failed to unmarshal message: %w", err)
	}

	// Log message details
	log.Info("Processing Kafka message",
		"topic", message.Topic,
		"partition", message.Partition,
		"offset", message.Offset,
		"key", string(message.Key),
		"event_type", event["type"],
	)

	// Process based on event type
	eventType, ok := event["type"].(string)
	if !ok {
		return fmt.Errorf("invalid event type")
	}

	switch eventType {
	case "user_created":
		return kc.handleUserCreated(ctx, event)
	case "user_updated":
		return kc.handleUserUpdated(ctx, event)
	case "payment_created":
		return kc.handlePaymentCreated(ctx, event)
	case "payment_updated":
		return kc.handlePaymentUpdated(ctx, event)
	case "health_check":
		return kc.handleHealthCheck(ctx, event)
	default:
		log.Warn("Unknown event type", "event_type", eventType)
		return nil
	}
}

// handleUserCreated handles user created events
func (kc *KafkaConsumer) handleUserCreated(ctx context.Context, event map[string]interface{}) error {
	log := logger.GetLogger()
	log.Info("Handling user created event", "event", event)

	// Here you would typically:
	// 1. Update cache
	// 2. Send notifications
	// 3. Update analytics
	// 4. Trigger other business logic

	return nil
}

// handleUserUpdated handles user updated events
func (kc *KafkaConsumer) handleUserUpdated(ctx context.Context, event map[string]interface{}) error {
	log := logger.GetLogger()
	log.Info("Handling user updated event", "event", event)

	// Here you would typically:
	// 1. Update cache
	// 2. Send notifications
	// 3. Update analytics
	// 4. Trigger other business logic

	return nil
}

// handlePaymentCreated handles payment created events
func (kc *KafkaConsumer) handlePaymentCreated(ctx context.Context, event map[string]interface{}) error {
	log := logger.GetLogger()
	log.Info("Handling payment created event", "event", event)

	// Here you would typically:
	// 1. Update cache
	// 2. Send notifications
	// 3. Update analytics
	// 4. Trigger other business logic

	return nil
}

// handlePaymentUpdated handles payment updated events
func (kc *KafkaConsumer) handlePaymentUpdated(ctx context.Context, event map[string]interface{}) error {
	log := logger.GetLogger()
	log.Info("Handling payment updated event", "event", event)

	// Here you would typically:
	// 1. Update cache
	// 2. Send notifications
	// 3. Update analytics
	// 4. Trigger other business logic

	return nil
}

// handleHealthCheck handles health check events
func (kc *KafkaConsumer) handleHealthCheck(ctx context.Context, event map[string]interface{}) error {
	log := logger.GetLogger()
	log.Debug("Handling health check event", "event", event)
	return nil
}

// Close closes the Kafka consumer
func (kc *KafkaConsumer) Close() error {
	kc.mutex.Lock()
	defer kc.mutex.Unlock()

	if kc.reader != nil {
		return kc.reader.Close()
	}
	return nil
}

// Health check for Kafka consumer
func (kc *KafkaConsumer) HealthCheck(ctx context.Context) error {
	kc.mutex.RLock()
	defer kc.mutex.RUnlock()

	if kc.reader == nil {
		return fmt.Errorf("Kafka reader is not initialized")
	}

	// Check if reader is still active
	// This is a simple check - in production you might want more sophisticated health checks
	return nil
}
