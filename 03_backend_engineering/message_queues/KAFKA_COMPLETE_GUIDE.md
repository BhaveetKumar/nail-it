---
# Auto-generated front matter
Title: Kafka Complete Guide
LastUpdated: 2025-11-06T20:45:58.291434
Tags: []
Status: draft
---

# 🚀 Apache Kafka Complete Guide

> **Comprehensive guide to Apache Kafka for event streaming and real-time data processing**

## 📚 Table of Contents

1. [Introduction to Kafka](#-introduction-to-kafka)
2. [Kafka Architecture](#-kafka-architecture)
3. [Kafka Concepts](#-kafka-concepts)
4. [Kafka Setup & Configuration](#-kafka-setup--configuration)
5. [Go Kafka Client](#-go-kafka-client)
6. [Producers](#-producers)
7. [Consumers](#-consumers)
8. [Streams Processing](#-streams-processing)
9. [Schema Registry](#-schema-registry)
10. [Monitoring & Observability](#-monitoring--observability)
11. [Best Practices](#-best-practices)
12. [Real-world Examples](#-real-world-examples)

---

## 🌟 Introduction to Kafka

### What is Apache Kafka?

Apache Kafka is a distributed event streaming platform capable of handling trillions of events per day. It's designed to handle data streams from multiple sources and deliver them to multiple consumers in real-time.

### Key Features

- **High Throughput**: Can handle millions of messages per second
- **Low Latency**: Sub-millisecond latency for real-time processing
- **Durability**: Messages are persisted to disk and replicated
- **Scalability**: Horizontal scaling across multiple brokers
- **Fault Tolerance**: Built-in replication and failover
- **Real-time Processing**: Stream processing capabilities

### Use Cases

- **Event Streaming**: Real-time event processing
- **Microservices Communication**: Asynchronous messaging
- **Data Integration**: ETL pipelines and data lakes
- **Real-time Analytics**: Stream processing and analytics
- **Log Aggregation**: Centralized logging
- **Change Data Capture**: Database change streams

---

## 🏗️ Kafka Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Producer 1    │    │   Producer 2    │    │   Producer N    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Kafka Cluster        │
                    │  ┌─────────────────────┐  │
                    │  │    Broker 1         │  │
                    │  │  ┌───────────────┐  │  │
                    │  │  │   Topic A     │  │  │
                    │  │  │   Topic B     │  │  │
                    │  │  └───────────────┘  │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │    Broker 2         │  │
                    │  │  ┌───────────────┐  │  │
                    │  │  │   Topic A     │  │  │
                    │  │  │   Topic B     │  │  │
                    │  │  └───────────────┘  │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │    Broker 3         │  │
                    │  │  ┌───────────────┐  │  │
                    │  │  │   Topic A     │  │  │
                    │  │  │   Topic B     │  │  │
                    │  │  └───────────────┘  │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴───────┐    ┌─────────┴───────┐    ┌─────────┴───────┐
│   Consumer 1    │    │   Consumer 2    │    │   Consumer N    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Producer**: Applications that send data to Kafka
2. **Consumer**: Applications that read data from Kafka
3. **Broker**: Kafka server that stores and serves data
4. **Topic**: Category or feed name to which records are published
5. **Partition**: Ordered sequence of records within a topic
6. **Offset**: Unique identifier for each record in a partition
7. **Consumer Group**: Group of consumers that work together to consume data

---

## 📋 Kafka Concepts

### Topics and Partitions

```go
// Topic: A category or feed name
// Partition: Ordered sequence of records within a topic
// Offset: Unique identifier for each record in a partition

type Topic struct {
    Name       string
    Partitions []Partition
    ReplicationFactor int
}

type Partition struct {
    ID     int32
    Leader int32
    Replicas []int32
    ISR     []int32 // In-Sync Replicas
}

type Record struct {
    Key     []byte
    Value   []byte
    Offset  int64
    Partition int32
    Timestamp time.Time
}
```

### Consumer Groups

```go
// Consumer Group: Group of consumers that work together
type ConsumerGroup struct {
    GroupID string
    Consumers []Consumer
    Topics   []string
}

// Each partition is consumed by only one consumer in a group
// Multiple consumer groups can consume the same topic independently
```

### Message Ordering

```go
// Messages within a partition are ordered
// Messages across partitions are not guaranteed to be ordered
// Use the same key to ensure ordering for related messages
```

---

## ⚙️ Kafka Setup & Configuration

### Docker Compose Setup

```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
```

### Go Dependencies

```go
// go.mod
module kafka-example

go 1.21

require (
    github.com/Shopify/sarama v1.38.1
    github.com/IBM/sarama v1.41.2
    github.com/confluentinc/confluent-kafka-go v2.2.0+incompatible
    github.com/segmentio/kafka-go v0.4.42
)
```

---

## 🚀 Go Kafka Client

### Basic Configuration

```go
package main

import (
    "context"
    "log"
    "time"

    "github.com/IBM/sarama"
)

type KafkaConfig struct {
    Brokers []string
    GroupID string
    Topics  []string
}

func NewKafkaConfig() *KafkaConfig {
    return &KafkaConfig{
        Brokers: []string{"localhost:9092"},
        GroupID: "my-consumer-group",
        Topics:  []string{"user-events", "order-events", "payment-events"},
    }
}

func (c *KafkaConfig) GetProducerConfig() *sarama.Config {
    config := sarama.NewConfig()
    config.Producer.RequiredAcks = sarama.WaitForAll
    config.Producer.Retry.Max = 3
    config.Producer.Return.Successes = true
    config.Producer.Compression = sarama.SnappyCompression
    config.Producer.Flush.Frequency = 100 * time.Millisecond
    return config
}

func (c *KafkaConfig) GetConsumerConfig() *sarama.Config {
    config := sarama.NewConfig()
    config.Consumer.Group.Rebalance.Strategy = sarama.BalanceStrategyRoundRobin
    config.Consumer.Offsets.Initial = sarama.OffsetOldest
    config.Consumer.Return.Errors = true
    config.Consumer.Group.Session.Timeout = 10 * time.Second
    config.Consumer.Group.Heartbeat.Interval = 3 * time.Second
    return config
}
```

---

## 📤 Producers

### Basic Producer

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "time"

    "github.com/IBM/sarama"
)

type Producer struct {
    producer sarama.SyncProducer
    config   *KafkaConfig
}

func NewProducer(config *KafkaConfig) (*Producer, error) {
    producer, err := sarama.NewSyncProducer(config.Brokers, config.GetProducerConfig())
    if err != nil {
        return nil, err
    }
    
    return &Producer{
        producer: producer,
        config:   config,
    }, nil
}

func (p *Producer) SendMessage(topic string, key string, value interface{}) error {
    // Serialize value to JSON
    jsonValue, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    // Create message
    message := &sarama.ProducerMessage{
        Topic: topic,
        Key:   sarama.StringEncoder(key),
        Value: sarama.ByteEncoder(jsonValue),
        Headers: []sarama.RecordHeader{
            {
                Key:   []byte("content-type"),
                Value: []byte("application/json"),
            },
            {
                Key:   []byte("timestamp"),
                Value: []byte(time.Now().Format(time.RFC3339)),
            },
        },
    }
    
    // Send message
    partition, offset, err := p.producer.SendMessage(message)
    if err != nil {
        return err
    }
    
    log.Printf("Message sent to topic %s, partition %d, offset %d", topic, partition, offset)
    return nil
}

func (p *Producer) Close() error {
    return p.producer.Close()
}

// Usage example
func main() {
    config := NewKafkaConfig()
    producer, err := NewProducer(config)
    if err != nil {
        log.Fatalf("Failed to create producer: %v", err)
    }
    defer producer.Close()
    
    // Send user event
    userEvent := map[string]interface{}{
        "user_id":    123,
        "event_type": "user_registered",
        "timestamp":  time.Now().Unix(),
        "data": map[string]interface{}{
            "email": "user@example.com",
            "name":  "John Doe",
        },
    }
    
    err = producer.SendMessage("user-events", "123", userEvent)
    if err != nil {
        log.Fatalf("Failed to send message: %v", err)
    }
}
```

### Async Producer

```go
type AsyncProducer struct {
    producer sarama.AsyncProducer
    config   *KafkaConfig
}

func NewAsyncProducer(config *KafkaConfig) (*AsyncProducer, error) {
    config.Producer.Return.Successes = true
    config.Producer.Return.Errors = true
    
    producer, err := sarama.NewAsyncProducer(config.Brokers, config.GetProducerConfig())
    if err != nil {
        return nil, err
    }
    
    asyncProducer := &AsyncProducer{
        producer: producer,
        config:   config,
    }
    
    // Start goroutines to handle successes and errors
    go asyncProducer.handleSuccesses()
    go asyncProducer.handleErrors()
    
    return asyncProducer, nil
}

func (p *AsyncProducer) SendMessage(topic string, key string, value interface{}) error {
    jsonValue, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    message := &sarama.ProducerMessage{
        Topic: topic,
        Key:   sarama.StringEncoder(key),
        Value: sarama.ByteEncoder(jsonValue),
    }
    
    select {
    case p.producer.Input() <- message:
        return nil
    case <-time.After(5 * time.Second):
        return fmt.Errorf("timeout sending message")
    }
}

func (p *AsyncProducer) handleSuccesses() {
    for msg := range p.producer.Successes() {
        log.Printf("Message sent successfully: topic=%s, partition=%d, offset=%d",
            msg.Topic, msg.Partition, msg.Offset)
    }
}

func (p *AsyncProducer) handleErrors() {
    for err := range p.producer.Errors() {
        log.Printf("Failed to send message: %v", err)
    }
}

func (p *AsyncProducer) Close() error {
    return p.producer.Close()
}
```

### Batch Producer

```go
type BatchProducer struct {
    producer sarama.SyncProducer
    config   *KafkaConfig
    batch    []*sarama.ProducerMessage
    maxSize  int
    timeout  time.Duration
    lastFlush time.Time
}

func NewBatchProducer(config *KafkaConfig, maxSize int, timeout time.Duration) (*BatchProducer, error) {
    producer, err := sarama.NewSyncProducer(config.Brokers, config.GetProducerConfig())
    if err != nil {
        return nil, err
    }
    
    return &BatchProducer{
        producer: producer,
        config:   config,
        batch:    make([]*sarama.ProducerMessage, 0, maxSize),
        maxSize:  maxSize,
        timeout:  timeout,
        lastFlush: time.Now(),
    }, nil
}

func (p *BatchProducer) AddMessage(topic string, key string, value interface{}) error {
    jsonValue, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    message := &sarama.ProducerMessage{
        Topic: topic,
        Key:   sarama.StringEncoder(key),
        Value: sarama.ByteEncoder(jsonValue),
    }
    
    p.batch = append(p.batch, message)
    
    // Flush if batch is full or timeout reached
    if len(p.batch) >= p.maxSize || time.Since(p.lastFlush) > p.timeout {
        return p.Flush()
    }
    
    return nil
}

func (p *BatchProducer) Flush() error {
    if len(p.batch) == 0 {
        return nil
    }
    
    err := p.producer.SendMessages(p.batch)
    if err != nil {
        return err
    }
    
    log.Printf("Flushed %d messages", len(p.batch))
    p.batch = p.batch[:0]
    p.lastFlush = time.Now()
    
    return nil
}

func (p *BatchProducer) Close() error {
    // Flush remaining messages
    if err := p.Flush(); err != nil {
        return err
    }
    return p.producer.Close()
}
```

---

## 📥 Consumers

### Basic Consumer

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "sync"

    "github.com/IBM/sarama"
)

type Consumer struct {
    consumer sarama.ConsumerGroup
    config   *KafkaConfig
    handler  MessageHandler
}

type MessageHandler interface {
    HandleMessage(ctx context.Context, message *sarama.ConsumerMessage) error
}

type DefaultMessageHandler struct{}

func (h *DefaultMessageHandler) HandleMessage(ctx context.Context, message *sarama.ConsumerMessage) error {
    log.Printf("Received message: topic=%s, partition=%d, offset=%d, key=%s, value=%s",
        message.Topic, message.Partition, message.Offset, string(message.Key), string(message.Value))
    
    // Process message based on topic
    switch message.Topic {
    case "user-events":
        return h.handleUserEvent(message)
    case "order-events":
        return h.handleOrderEvent(message)
    case "payment-events":
        return h.handlePaymentEvent(message)
    default:
        log.Printf("Unknown topic: %s", message.Topic)
    }
    
    return nil
}

func (h *DefaultMessageHandler) handleUserEvent(message *sarama.ConsumerMessage) error {
    var userEvent map[string]interface{}
    if err := json.Unmarshal(message.Value, &userEvent); err != nil {
        return err
    }
    
    log.Printf("Processing user event: %+v", userEvent)
    // Process user event...
    
    return nil
}

func (h *DefaultMessageHandler) handleOrderEvent(message *sarama.ConsumerMessage) error {
    var orderEvent map[string]interface{}
    if err := json.Unmarshal(message.Value, &orderEvent); err != nil {
        return err
    }
    
    log.Printf("Processing order event: %+v", orderEvent)
    // Process order event...
    
    return nil
}

func (h *DefaultMessageHandler) handlePaymentEvent(message *sarama.ConsumerMessage) error {
    var paymentEvent map[string]interface{}
    if err := json.Unmarshal(message.Value, &paymentEvent); err != nil {
        return err
    }
    
    log.Printf("Processing payment event: %+v", paymentEvent)
    // Process payment event...
    
    return nil
}

func NewConsumer(config *KafkaConfig, handler MessageHandler) (*Consumer, error) {
    consumer, err := sarama.NewConsumerGroup(config.Brokers, config.GroupID, config.GetConsumerConfig())
    if err != nil {
        return nil, err
    }
    
    return &Consumer{
        consumer: consumer,
        config:   config,
        handler:  handler,
    }, nil
}

func (c *Consumer) Start(ctx context.Context) error {
    wg := &sync.WaitGroup{}
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for {
            if err := c.consumer.Consume(ctx, c.config.Topics, c); err != nil {
                log.Printf("Error from consumer: %v", err)
                return
            }
            
            if ctx.Err() != nil {
                return
            }
        }
    }()
    
    wg.Wait()
    return c.consumer.Close()
}

// Implement sarama.ConsumerGroupHandler
func (c *Consumer) Setup(sarama.ConsumerGroupSession) error   { return nil }
func (c *Consumer) Cleanup(sarama.ConsumerGroupSession) error { return nil }

func (c *Consumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            if err := c.handler.HandleMessage(session.Context(), message); err != nil {
                log.Printf("Error handling message: %v", err)
                // Handle error (retry, dead letter queue, etc.)
            }
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}

// Usage example
func main() {
    config := NewKafkaConfig()
    handler := &DefaultMessageHandler{}
    
    consumer, err := NewConsumer(config, handler)
    if err != nil {
        log.Fatalf("Failed to create consumer: %v", err)
    }
    defer consumer.Close()
    
    ctx := context.Background()
    if err := consumer.Start(ctx); err != nil {
        log.Fatalf("Consumer error: %v", err)
    }
}
```

### Consumer with Error Handling

```go
type ConsumerWithRetry struct {
    consumer sarama.ConsumerGroup
    config   *KafkaConfig
    handler  MessageHandler
    maxRetries int
    retryDelay time.Duration
}

func NewConsumerWithRetry(config *KafkaConfig, handler MessageHandler, maxRetries int, retryDelay time.Duration) (*ConsumerWithRetry, error) {
    consumer, err := sarama.NewConsumerGroup(config.Brokers, config.GroupID, config.GetConsumerConfig())
    if err != nil {
        return nil, err
    }
    
    return &ConsumerWithRetry{
        consumer:   consumer,
        config:     config,
        handler:    handler,
        maxRetries: maxRetries,
        retryDelay: retryDelay,
    }, nil
}

func (c *ConsumerWithRetry) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            if err := c.handleMessageWithRetry(session.Context(), message); err != nil {
                log.Printf("Failed to process message after retries: %v", err)
                // Send to dead letter queue or handle failure
                c.sendToDeadLetterQueue(message, err)
            }
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}

func (c *ConsumerWithRetry) handleMessageWithRetry(ctx context.Context, message *sarama.ConsumerMessage) error {
    var lastErr error
    
    for i := 0; i < c.maxRetries; i++ {
        if err := c.handler.HandleMessage(ctx, message); err != nil {
            lastErr = err
            log.Printf("Attempt %d failed: %v", i+1, err)
            
            if i < c.maxRetries-1 {
                time.Sleep(c.retryDelay * time.Duration(i+1))
            }
        } else {
            return nil
        }
    }
    
    return lastErr
}

func (c *ConsumerWithRetry) sendToDeadLetterQueue(message *sarama.ConsumerMessage, err error) {
    // Implementation for sending to dead letter queue
    log.Printf("Sending message to dead letter queue: %v", err)
}
```

---

## 🔄 Streams Processing

### Kafka Streams with Go

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "time"

    "github.com/IBM/sarama"
)

type StreamProcessor struct {
    consumer sarama.ConsumerGroup
    producer sarama.SyncProducer
    config   *KafkaConfig
}

func NewStreamProcessor(config *KafkaConfig) (*StreamProcessor, error) {
    consumer, err := sarama.NewConsumerGroup(config.Brokers, config.GroupID, config.GetConsumerConfig())
    if err != nil {
        return nil, err
    }
    
    producer, err := sarama.NewSyncProducer(config.Brokers, config.GetProducerConfig())
    if err != nil {
        return nil, err
    }
    
    return &StreamProcessor{
        consumer: consumer,
        producer: producer,
        config:   config,
    }, nil
}

func (sp *StreamProcessor) ProcessUserEvents(ctx context.Context) error {
    handler := &UserEventProcessor{producer: sp.producer}
    
    for {
        if err := sp.consumer.Consume(ctx, []string{"user-events"}, handler); err != nil {
            log.Printf("Error from consumer: %v", err)
            return err
        }
        
        if ctx.Err() != nil {
            return ctx.Err()
        }
    }
}

type UserEventProcessor struct {
    producer sarama.SyncProducer
}

func (p *UserEventProcessor) Setup(sarama.ConsumerGroupSession) error   { return nil }
func (p *UserEventProcessor) Cleanup(sarama.ConsumerGroupSession) error { return nil }

func (p *UserEventProcessor) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            if err := p.processUserEvent(session.Context(), message); err != nil {
                log.Printf("Error processing user event: %v", err)
            }
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}

func (p *UserEventProcessor) processUserEvent(ctx context.Context, message *sarama.ConsumerMessage) error {
    var userEvent map[string]interface{}
    if err := json.Unmarshal(message.Value, &userEvent); err != nil {
        return err
    }
    
    // Process user event
    eventType := userEvent["event_type"].(string)
    
    switch eventType {
    case "user_registered":
        return p.handleUserRegistration(userEvent)
    case "user_updated":
        return p.handleUserUpdate(userEvent)
    case "user_deleted":
        return p.handleUserDeletion(userEvent)
    default:
        log.Printf("Unknown event type: %s", eventType)
    }
    
    return nil
}

func (p *UserEventProcessor) handleUserRegistration(userEvent map[string]interface{}) error {
    // Send welcome email event
    welcomeEvent := map[string]interface{}{
        "user_id":    userEvent["user_id"],
        "event_type": "welcome_email_sent",
        "timestamp":  time.Now().Unix(),
        "data": map[string]interface{}{
            "email": userEvent["data"].(map[string]interface{})["email"],
        },
    }
    
    return p.sendEvent("email-events", welcomeEvent)
}

func (p *UserEventProcessor) handleUserUpdate(userEvent map[string]interface{}) error {
    // Send profile update notification
    notificationEvent := map[string]interface{}{
        "user_id":    userEvent["user_id"],
        "event_type": "profile_updated",
        "timestamp":  time.Now().Unix(),
        "data":       userEvent["data"],
    }
    
    return p.sendEvent("notification-events", notificationEvent)
}

func (p *UserEventProcessor) handleUserDeletion(userEvent map[string]interface{}) error {
    // Send cleanup event
    cleanupEvent := map[string]interface{}{
        "user_id":    userEvent["user_id"],
        "event_type": "user_cleanup",
        "timestamp":  time.Now().Unix(),
    }
    
    return p.sendEvent("cleanup-events", cleanupEvent)
}

func (p *UserEventProcessor) sendEvent(topic string, event map[string]interface{}) error {
    jsonValue, err := json.Marshal(event)
    if err != nil {
        return err
    }
    
    message := &sarama.ProducerMessage{
        Topic: topic,
        Value: sarama.ByteEncoder(jsonValue),
    }
    
    _, _, err = p.producer.SendMessage(message)
    return err
}
```

---

## 📊 Schema Registry

### Schema Registry Integration

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type SchemaRegistry struct {
    baseURL string
    client  *http.Client
}

func NewSchemaRegistry(baseURL string) *SchemaRegistry {
    return &SchemaRegistry{
        baseURL: baseURL,
        client: &http.Client{
            Timeout: 10 * time.Second,
        },
    }
}

type Schema struct {
    ID         int    `json:"id"`
    Version    int    `json:"version"`
    Subject    string `json:"subject"`
    Schema     string `json:"schema"`
    SchemaType string `json:"schemaType"`
}

type SchemaResponse struct {
    ID int `json:"id"`
}

func (sr *SchemaRegistry) RegisterSchema(subject string, schema string) (*SchemaResponse, error) {
    url := fmt.Sprintf("%s/subjects/%s/versions", sr.baseURL, subject)
    
    payload := map[string]interface{}{
        "schema":     schema,
        "schemaType": "AVRO",
    }
    
    jsonPayload, err := json.Marshal(payload)
    if err != nil {
        return nil, err
    }
    
    resp, err := sr.client.Post(url, "application/vnd.schemaregistry.v1+json", bytes.NewBuffer(jsonPayload))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("schema registry error: %s", string(body))
    }
    
    var schemaResp SchemaResponse
    if err := json.NewDecoder(resp.Body).Decode(&schemaResp); err != nil {
        return nil, err
    }
    
    return &schemaResp, nil
}

func (sr *SchemaRegistry) GetSchema(subject string, version string) (*Schema, error) {
    url := fmt.Sprintf("%s/subjects/%s/versions/%s", sr.baseURL, subject, version)
    
    resp, err := sr.client.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("schema registry error: %s", string(body))
    }
    
    var schema Schema
    if err := json.NewDecoder(resp.Body).Decode(&schema); err != nil {
        return nil, err
    }
    
    return &schema, nil
}

// Usage with Kafka
func (p *Producer) SendMessageWithSchema(topic string, key string, value interface{}, subject string) error {
    // Get schema from registry
    schemaRegistry := NewSchemaRegistry("http://localhost:8081")
    schema, err := schemaRegistry.GetSchema(subject, "latest")
    if err != nil {
        return err
    }
    
    // Serialize value with schema
    jsonValue, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    // Create message with schema ID
    message := &sarama.ProducerMessage{
        Topic: topic,
        Key:   sarama.StringEncoder(key),
        Value: sarama.ByteEncoder(jsonValue),
        Headers: []sarama.RecordHeader{
            {
                Key:   []byte("schema-id"),
                Value: []byte(fmt.Sprintf("%d", schema.ID)),
            },
        },
    }
    
    _, _, err = p.producer.SendMessage(message)
    return err
}
```

---

## 📈 Monitoring & Observability

### Metrics Collection

```go
package main

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // Producer metrics
    messagesProduced = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "kafka_messages_produced_total",
            Help: "Total number of messages produced",
        },
        []string{"topic", "status"},
    )
    
    messageProduceDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "kafka_message_produce_duration_seconds",
            Help: "Duration of message production",
        },
        []string{"topic"},
    )
    
    // Consumer metrics
    messagesConsumed = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "kafka_messages_consumed_total",
            Help: "Total number of messages consumed",
        },
        []string{"topic", "consumer_group", "status"},
    )
    
    messageConsumeDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "kafka_message_consume_duration_seconds",
            Help: "Duration of message consumption",
        },
        []string{"topic", "consumer_group"},
    )
    
    consumerLag = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "kafka_consumer_lag",
            Help: "Consumer lag in messages",
        },
        []string{"topic", "partition", "consumer_group"},
    )
)

// Instrumented Producer
type InstrumentedProducer struct {
    producer sarama.SyncProducer
    topic    string
}

func (p *InstrumentedProducer) SendMessage(key string, value interface{}) error {
    start := time.Now()
    
    // Send message
    err := p.producer.SendMessage(/* ... */)
    
    duration := time.Since(start).Seconds()
    messageProduceDuration.WithLabelValues(p.topic).Observe(duration)
    
    status := "success"
    if err != nil {
        status = "error"
    }
    messagesProduced.WithLabelValues(p.topic, status).Inc()
    
    return err
}

// Instrumented Consumer
type InstrumentedConsumer struct {
    consumer sarama.ConsumerGroup
    topic    string
    groupID  string
}

func (c *InstrumentedConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            start := time.Now()
            
            // Process message
            err := c.processMessage(message)
            
            duration := time.Since(start).Seconds()
            messageConsumeDuration.WithLabelValues(c.topic, c.groupID).Observe(duration)
            
            status := "success"
            if err != nil {
                status = "error"
            }
            messagesConsumed.WithLabelValues(c.topic, c.groupID, status).Inc()
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}
```

### Health Checks

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"

    "github.com/IBM/sarama"
)

type KafkaHealthChecker struct {
    config *KafkaConfig
}

func NewKafkaHealthChecker(config *KafkaConfig) *KafkaHealthChecker {
    return &KafkaHealthChecker{config: config}
}

func (h *KafkaHealthChecker) CheckHealth() error {
    // Create admin client
    admin, err := sarama.NewClusterAdmin(h.config.Brokers, h.config.GetProducerConfig())
    if err != nil {
        return fmt.Errorf("failed to create admin client: %v", err)
    }
    defer admin.Close()
    
    // Check cluster metadata
    metadata, err := admin.DescribeCluster()
    if err != nil {
        return fmt.Errorf("failed to describe cluster: %v", err)
    }
    
    // Check if cluster is healthy
    if len(metadata.Brokers) == 0 {
        return fmt.Errorf("no brokers available")
    }
    
    // Check topic metadata
    for _, topic := range h.config.Topics {
        if err := h.checkTopicHealth(admin, topic); err != nil {
            return fmt.Errorf("topic %s is unhealthy: %v", topic, err)
        }
    }
    
    return nil
}

func (h *KafkaHealthChecker) checkTopicHealth(admin sarama.ClusterAdmin, topic string) error {
    metadata, err := admin.DescribeTopics([]string{topic})
    if err != nil {
        return err
    }
    
    topicMetadata, exists := metadata[topic]
    if !exists {
        return fmt.Errorf("topic not found")
    }
    
    // Check if topic has partitions
    if len(topicMetadata.Partitions) == 0 {
        return fmt.Errorf("topic has no partitions")
    }
    
    // Check partition health
    for _, partition := range topicMetadata.Partitions {
        if partition.Leader == -1 {
            return fmt.Errorf("partition %d has no leader", partition.ID)
        }
        
        if len(partition.ISR) == 0 {
            return fmt.Errorf("partition %d has no in-sync replicas", partition.ID)
        }
    }
    
    return nil
}

// HTTP health check endpoint
func (h *KafkaHealthChecker) HealthCheckHandler(w http.ResponseWriter, r *http.Request) {
    if err := h.CheckHealth(); err != nil {
        w.WriteHeader(http.StatusServiceUnavailable)
        fmt.Fprintf(w, "Kafka is unhealthy: %v", err)
        return
    }
    
    w.WriteHeader(http.StatusOK)
    fmt.Fprint(w, "Kafka is healthy")
}
```

---

## 🏆 Best Practices

### 1. Topic Design

```go
// Good: Use descriptive topic names
const (
    UserEventsTopic    = "user-events"
    OrderEventsTopic   = "order-events"
    PaymentEventsTopic = "payment-events"
)

// Bad: Generic topic names
const (
    EventsTopic = "events"
    DataTopic   = "data"
)
```

### 2. Message Key Design

```go
// Good: Use meaningful keys for partitioning
func createUserEventKey(userID int64) string {
    return fmt.Sprintf("user-%d", userID)
}

func createOrderEventKey(orderID int64) string {
    return fmt.Sprintf("order-%d", orderID)
}

// Bad: Random or no keys
func createRandomKey() string {
    return fmt.Sprintf("random-%d", time.Now().UnixNano())
}
```

### 3. Error Handling

```go
// Good: Comprehensive error handling
func (c *Consumer) handleMessage(message *sarama.ConsumerMessage) error {
    // Validate message
    if message == nil || message.Value == nil {
        return fmt.Errorf("invalid message")
    }
    
    // Parse message
    var event map[string]interface{}
    if err := json.Unmarshal(message.Value, &event); err != nil {
        return fmt.Errorf("failed to parse message: %v", err)
    }
    
    // Process message
    if err := c.processEvent(event); err != nil {
        // Log error and decide on retry
        log.Printf("Failed to process event: %v", err)
        
        // Send to dead letter queue if retries exhausted
        if c.shouldSendToDLQ(event) {
            return c.sendToDeadLetterQueue(message, err)
        }
        
        return err
    }
    
    return nil
}
```

### 4. Performance Optimization

```go
// Good: Batch processing
func (c *Consumer) processBatch(messages []*sarama.ConsumerMessage) error {
    var events []map[string]interface{}
    
    // Parse all messages
    for _, message := range messages {
        var event map[string]interface{}
        if err := json.Unmarshal(message.Value, &event); err != nil {
            log.Printf("Failed to parse message: %v", err)
            continue
        }
        events = append(events, event)
    }
    
    // Process batch
    if err := c.processEventsBatch(events); err != nil {
        return err
    }
    
    return nil
}

// Good: Connection pooling
func (c *Consumer) createConnectionPool() error {
    config := c.config.GetConsumerConfig()
    config.Net.MaxOpenRequests = 1
    config.Consumer.Fetch.Min = 1024
    config.Consumer.Fetch.Max = 1024 * 1024
    
    return nil
}
```

---

## 🌟 Real-world Examples

### E-commerce Event Streaming

```go
// User Service Events
type UserService struct {
    producer *Producer
}

func (s *UserService) RegisterUser(user *User) error {
    // Create user in database
    if err := s.createUser(user); err != nil {
        return err
    }
    
    // Send user registered event
    event := map[string]interface{}{
        "user_id":    user.ID,
        "event_type": "user_registered",
        "timestamp":  time.Now().Unix(),
        "data": map[string]interface{}{
            "email": user.Email,
            "name":  user.Name,
        },
    }
    
    return s.producer.SendMessage(UserEventsTopic, fmt.Sprintf("user-%d", user.ID), event)
}

// Order Service Events
type OrderService struct {
    producer *Producer
    consumer *Consumer
}

func (s *OrderService) CreateOrder(order *Order) error {
    // Create order in database
    if err := s.createOrder(order); err != nil {
        return err
    }
    
    // Send order created event
    event := map[string]interface{}{
        "order_id":   order.ID,
        "user_id":    order.UserID,
        "event_type": "order_created",
        "timestamp":  time.Now().Unix(),
        "data": map[string]interface{}{
            "items":      order.Items,
            "total":      order.Total,
            "status":     order.Status,
        },
    }
    
    return s.producer.SendMessage(OrderEventsTopic, fmt.Sprintf("order-%d", order.ID), event)
}

// Payment Service Events
type PaymentService struct {
    producer *Producer
    consumer *Consumer
}

func (s *PaymentService) ProcessPayment(payment *Payment) error {
    // Process payment
    if err := s.processPayment(payment); err != nil {
        return err
    }
    
    // Send payment processed event
    event := map[string]interface{}{
        "payment_id": payment.ID,
        "order_id":   payment.OrderID,
        "event_type": "payment_processed",
        "timestamp":  time.Now().Unix(),
        "data": map[string]interface{}{
            "amount": payment.Amount,
            "method": payment.Method,
            "status": payment.Status,
        },
    }
    
    return s.producer.SendMessage(PaymentEventsTopic, fmt.Sprintf("payment-%d", payment.ID), event)
}
```

### Real-time Analytics

```go
// Analytics Service
type AnalyticsService struct {
    consumer *Consumer
    db       *Database
}

func (s *AnalyticsService) ProcessUserEvents() error {
    handler := &UserEventAnalyticsHandler{db: s.db}
    
    consumer, err := NewConsumer(s.config, handler)
    if err != nil {
        return err
    }
    
    return consumer.Start(context.Background())
}

type UserEventAnalyticsHandler struct {
    db *Database
}

func (h *UserEventAnalyticsHandler) HandleMessage(ctx context.Context, message *sarama.ConsumerMessage) error {
    var event map[string]interface{}
    if err := json.Unmarshal(message.Value, &event); err != nil {
        return err
    }
    
    // Update analytics
    switch event["event_type"] {
    case "user_registered":
        return h.updateUserRegistrationStats(event)
    case "user_login":
        return h.updateUserLoginStats(event)
    case "user_purchase":
        return h.updatePurchaseStats(event)
    }
    
    return nil
}

func (h *UserEventAnalyticsHandler) updateUserRegistrationStats(event map[string]interface{}) error {
    // Update daily registration count
    date := time.Unix(event["timestamp"].(int64), 0).Format("2006-01-02")
    
    query := `
        INSERT INTO daily_registrations (date, count) 
        VALUES (?, 1) 
        ON DUPLICATE KEY UPDATE count = count + 1
    `
    
    _, err := h.db.Exec(query, date)
    return err
}
```

---

## 🚀 Getting Started

### 1. Start Kafka

```bash
# Using Docker Compose
docker-compose up -d

# Or using Confluent Platform
confluent local start
```

### 2. Create Topics

```bash
# Create topic
kafka-topics --create --topic user-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# List topics
kafka-topics --list --bootstrap-server localhost:9092
```

### 3. Run Producer

```bash
go run producer/main.go
```

### 4. Run Consumer

```bash
go run consumer/main.go
```

---

**🎉 You now have a comprehensive understanding of Apache Kafka! Use this knowledge to build scalable event-driven systems and ace your Razorpay interviews! 🚀**
