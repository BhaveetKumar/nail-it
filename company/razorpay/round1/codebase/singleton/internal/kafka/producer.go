package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"
	"singleton-service/internal/config"
	"singleton-service/internal/logger"
)

// KafkaProducer implements Singleton pattern for Kafka producer
type KafkaProducer struct {
	writer *kafka.Writer
	mutex  sync.RWMutex
}

var (
	kafkaProducer *KafkaProducer
	producerOnce  sync.Once
)

// GetKafkaProducer returns the singleton instance of KafkaProducer
func GetKafkaProducer() *KafkaProducer {
	producerOnce.Do(func() {
		kafkaProducer = &KafkaProducer{}
		kafkaProducer.connect()
	})
	return kafkaProducer
}

// connect establishes connection to Kafka
func (kp *KafkaProducer) connect() {
	cfg := config.GetConfigManager()
	kafkaConfig := cfg.GetKafkaConfig()
	log := logger.GetLogger()

	writer := &kafka.Writer{
		Addr:         kafka.Dialer{Timeout: 10 * time.Second, DualStack: true},
		Topic:        kafkaConfig.Topic,
		Balancer:     &kafka.LeastBytes{},
		BatchTimeout: 10 * time.Millisecond,
		BatchSize:    100,
		RequiredAcks: kafka.RequireOne,
		Async:        false,
	}

	kp.mutex.Lock()
	kp.writer = writer
	kp.mutex.Unlock()

	log.Info("Kafka producer connected successfully")
}

// PublishMessage publishes a message to Kafka
func (kp *KafkaProducer) PublishMessage(ctx context.Context, key string, message interface{}) error {
	kp.mutex.RLock()
	defer kp.mutex.RUnlock()

	if kp.writer == nil {
		return fmt.Errorf("Kafka writer is not initialized")
	}

	// Serialize message to JSON
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	// Create Kafka message
	kafkaMessage := kafka.Message{
		Key:   []byte(key),
		Value: messageBytes,
		Time:  time.Now(),
	}

	// Publish message
	if err := kp.writer.WriteMessages(ctx, kafkaMessage); err != nil {
		return fmt.Errorf("failed to write message to Kafka: %w", err)
	}

	return nil
}

// PublishMessages publishes multiple messages to Kafka
func (kp *KafkaProducer) PublishMessages(ctx context.Context, messages []kafka.Message) error {
	kp.mutex.RLock()
	defer kp.mutex.RUnlock()

	if kp.writer == nil {
		return fmt.Errorf("Kafka writer is not initialized")
	}

	if err := kp.writer.WriteMessages(ctx, messages...); err != nil {
		return fmt.Errorf("failed to write messages to Kafka: %w", err)
	}

	return nil
}

// PublishEvent publishes an event to Kafka
func (kp *KafkaProducer) PublishEvent(ctx context.Context, eventType string, eventData interface{}) error {
	event := map[string]interface{}{
		"type":      eventType,
		"data":      eventData,
		"timestamp": time.Now().Unix(),
		"source":    "singleton-service",
	}

	return kp.PublishMessage(ctx, eventType, event)
}

// PublishUserEvent publishes a user-related event
func (kp *KafkaProducer) PublishUserEvent(ctx context.Context, userID string, eventType string, eventData interface{}) error {
	event := map[string]interface{}{
		"type":      eventType,
		"user_id":   userID,
		"data":      eventData,
		"timestamp": time.Now().Unix(),
		"source":    "singleton-service",
	}

	return kp.PublishMessage(ctx, userID, event)
}

// PublishPaymentEvent publishes a payment-related event
func (kp *KafkaProducer) PublishPaymentEvent(ctx context.Context, paymentID string, eventType string, eventData interface{}) error {
	event := map[string]interface{}{
		"type":       eventType,
		"payment_id": paymentID,
		"data":       eventData,
		"timestamp":  time.Now().Unix(),
		"source":     "singleton-service",
	}

	return kp.PublishMessage(ctx, paymentID, event)
}

// Close closes the Kafka producer
func (kp *KafkaProducer) Close() error {
	kp.mutex.Lock()
	defer kp.mutex.Unlock()

	if kp.writer != nil {
		return kp.writer.Close()
	}
	return nil
}

// Health check for Kafka producer
func (kp *KafkaProducer) HealthCheck(ctx context.Context) error {
	kp.mutex.RLock()
	defer kp.mutex.RUnlock()

	if kp.writer == nil {
		return fmt.Errorf("Kafka writer is not initialized")
	}

	// Try to publish a test message
	testMessage := map[string]interface{}{
		"type":      "health_check",
		"timestamp": time.Now().Unix(),
		"source":    "singleton-service",
	}

	return kp.PublishMessage(ctx, "health_check", testMessage)
}
