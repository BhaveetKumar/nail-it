# RabbitMQ Patterns - Message Queue Implementation

## Overview

RabbitMQ is a message broker that implements the Advanced Message Queuing Protocol (AMQP). It provides reliable message delivery, routing, and queuing capabilities for distributed systems.

## Key Concepts

- **Producer**: Sends messages to exchanges
- **Consumer**: Receives messages from queues
- **Exchange**: Routes messages to queues based on rules
- **Queue**: Stores messages until consumed
- **Binding**: Links exchanges to queues
- **Routing Key**: Determines message routing

## RabbitMQ Architecture

```mermaid
graph TD
    A[Producer] --> B[Exchange]
    B --> C[Queue 1]
    B --> D[Queue 2]
    B --> E[Queue 3]
    
    C --> F[Consumer 1]
    D --> G[Consumer 2]
    E --> H[Consumer 3]
    
    I[Dead Letter Queue] --> J[Error Handler]
    C --> I
    D --> I
    E --> I
```

## Go Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/streadway/amqp"
)

// Message represents a message structure
type Message struct {
    ID        string                 `json:"id"`
    Type      string                 `json:"type"`
    Data      map[string]interface{} `json:"data"`
    Timestamp time.Time              `json:"timestamp"`
    RetryCount int                   `json:"retry_count"`
}

// RabbitMQClient represents a RabbitMQ client
type RabbitMQClient struct {
    conn    *amqp.Connection
    channel *amqp.Channel
    config  *RabbitMQConfig
}

// RabbitMQConfig represents RabbitMQ configuration
type RabbitMQConfig struct {
    URL      string
    Exchange string
    Queue    string
    RoutingKey string
}

// NewRabbitMQClient creates a new RabbitMQ client
func NewRabbitMQClient(config *RabbitMQConfig) (*RabbitMQClient, error) {
    conn, err := amqp.Dial(config.URL)
    if err != nil {
        return nil, fmt.Errorf("failed to connect to RabbitMQ: %v", err)
    }

    channel, err := conn.Channel()
    if err != nil {
        return nil, fmt.Errorf("failed to open channel: %v", err)
    }

    return &RabbitMQClient{
        conn:    conn,
        channel: channel,
        config:  config,
    }, nil
}

// DeclareExchange declares an exchange
func (client *RabbitMQClient) DeclareExchange(name, kind string) error {
    return client.channel.ExchangeDeclare(
        name,    // name
        kind,    // type
        true,    // durable
        false,   // auto-deleted
        false,   // internal
        false,   // no-wait
        nil,     // arguments
    )
}

// DeclareQueue declares a queue
func (client *RabbitMQClient) DeclareQueue(name string) (amqp.Queue, error) {
    return client.channel.QueueDeclare(
        name,  // name
        true,  // durable
        false, // delete when unused
        false, // exclusive
        false, // no-wait
        nil,   // arguments
    )
}

// BindQueue binds a queue to an exchange
func (client *RabbitMQClient) BindQueue(queueName, exchangeName, routingKey string) error {
    return client.channel.QueueBind(
        queueName,    // queue name
        routingKey,   // routing key
        exchangeName, // exchange
        false,        // no-wait
        nil,          // arguments
    )
}

// PublishMessage publishes a message
func (client *RabbitMQClient) PublishMessage(exchange, routingKey string, message Message) error {
    body, err := json.Marshal(message)
    if err != nil {
        return fmt.Errorf("failed to marshal message: %v", err)
    }

    return client.channel.Publish(
        exchange,   // exchange
        routingKey, // routing key
        false,      // mandatory
        false,      // immediate
        amqp.Publishing{
            ContentType:  "application/json",
            Body:         body,
            Timestamp:    message.Timestamp,
            MessageId:    message.ID,
            DeliveryMode: amqp.Persistent,
        },
    )
}

// ConsumeMessages consumes messages from a queue
func (client *RabbitMQClient) ConsumeMessages(queueName string, handler func(Message) error) error {
    msgs, err := client.channel.Consume(
        queueName, // queue
        "",        // consumer
        false,     // auto-ack
        false,     // exclusive
        false,     // no-local
        false,     // no-wait
        nil,       // args
    )
    if err != nil {
        return fmt.Errorf("failed to register consumer: %v", err)
    }

    go func() {
        for msg := range msgs {
            var message Message
            if err := json.Unmarshal(msg.Body, &message); err != nil {
                log.Printf("Failed to unmarshal message: %v", err)
                msg.Nack(false, false)
                continue
            }

            if err := handler(message); err != nil {
                log.Printf("Failed to process message: %v", err)
                msg.Nack(false, true) // Requeue on failure
            } else {
                msg.Ack(false)
            }
        }
    }()

    return nil
}

// Close closes the RabbitMQ client
func (client *RabbitMQClient) Close() error {
    if err := client.channel.Close(); err != nil {
        return err
    }
    return client.conn.Close()
}

// MessageHandler represents a message handler
type MessageHandler struct {
    client *RabbitMQClient
    queue  string
}

// NewMessageHandler creates a new message handler
func NewMessageHandler(client *RabbitMQClient, queue string) *MessageHandler {
    return &MessageHandler{
        client: client,
        queue:  queue,
    }
}

// HandleMessage handles a message
func (h *MessageHandler) HandleMessage(message Message) error {
    log.Printf("Processing message: %s", message.ID)
    
    // Simulate processing
    time.Sleep(100 * time.Millisecond)
    
    // Simulate error for retry testing
    if message.RetryCount > 0 && message.RetryCount%3 == 0 {
        return fmt.Errorf("simulated error for message %s", message.ID)
    }
    
    log.Printf("Successfully processed message: %s", message.ID)
    return nil
}

// RetryHandler handles message retries
type RetryHandler struct {
    client      *RabbitMQClient
    retryQueue  string
    maxRetries  int
    retryDelay  time.Duration
}

// NewRetryHandler creates a new retry handler
func NewRetryHandler(client *RabbitMQClient, retryQueue string, maxRetries int, retryDelay time.Duration) *RetryHandler {
    return &RetryHandler{
        client:     client,
        retryQueue: retryQueue,
        maxRetries: maxRetries,
        retryDelay: retryDelay,
    }
}

// HandleRetry handles message retries
func (h *RetryHandler) HandleRetry(message Message) error {
    if message.RetryCount >= h.maxRetries {
        log.Printf("Message %s exceeded max retries, sending to DLQ", message.ID)
        return h.sendToDLQ(message)
    }
    
    message.RetryCount++
    message.Timestamp = time.Now()
    
    // Delay before retry
    time.Sleep(h.retryDelay)
    
    return h.client.PublishMessage("", h.retryQueue, message)
}

// sendToDLQ sends message to dead letter queue
func (h *RetryHandler) sendToDLQ(message Message) error {
    dlqName := h.retryQueue + ".dlq"
    return h.client.PublishMessage("", dlqName, message)
}

// DeadLetterHandler handles dead letter messages
type DeadLetterHandler struct {
    client *RabbitMQClient
    dlq    string
}

// NewDeadLetterHandler creates a new dead letter handler
func NewDeadLetterHandler(client *RabbitMQClient, dlq string) *DeadLetterHandler {
    return &DeadLetterHandler{
        client: client,
        dlq:    dlq,
    }
}

// HandleDeadLetter handles dead letter messages
func (h *DeadLetterHandler) HandleDeadLetter(message Message) error {
    log.Printf("Processing dead letter message: %s", message.ID)
    
    // Log the message for analysis
    log.Printf("DLQ Message: %+v", message)
    
    // Could send to monitoring system, database, etc.
    return nil
}

// Example usage
func main() {
    // RabbitMQ configuration
    config := &RabbitMQConfig{
        URL:       "amqp://guest:guest@localhost:5672/",
        Exchange:  "test-exchange",
        Queue:     "test-queue",
        RoutingKey: "test.routing.key",
    }
    
    // Create RabbitMQ client
    client, err := NewRabbitMQClient(config)
    if err != nil {
        log.Fatalf("Failed to create RabbitMQ client: %v", err)
    }
    defer client.Close()
    
    // Declare exchange
    if err := client.DeclareExchange(config.Exchange, "topic"); err != nil {
        log.Fatalf("Failed to declare exchange: %v", err)
    }
    
    // Declare queue
    queue, err := client.DeclareQueue(config.Queue)
    if err != nil {
        log.Fatalf("Failed to declare queue: %v", err)
    }
    
    // Bind queue to exchange
    if err := client.BindQueue(queue.Name, config.Exchange, config.RoutingKey); err != nil {
        log.Fatalf("Failed to bind queue: %v", err)
    }
    
    // Create message handler
    handler := NewMessageHandler(client, queue.Name)
    
    // Create retry handler
    retryHandler := NewRetryHandler(client, queue.Name, 3, 1*time.Second)
    
    // Create dead letter handler
    dlqHandler := NewDeadLetterHandler(client, queue.Name+".dlq")
    
    // Start consuming messages
    if err := client.ConsumeMessages(queue.Name, func(message Message) error {
        if err := handler.HandleMessage(message); err != nil {
            return retryHandler.HandleRetry(message)
        }
        return nil
    }); err != nil {
        log.Fatalf("Failed to start consuming: %v", err)
    }
    
    // Publish some test messages
    for i := 0; i < 5; i++ {
        message := Message{
            ID:         fmt.Sprintf("msg-%d", i),
            Type:       "test",
            Data:       map[string]interface{}{"value": i},
            Timestamp:  time.Now(),
            RetryCount: 0,
        }
        
        if err := client.PublishMessage(config.Exchange, config.RoutingKey, message); err != nil {
            log.Printf("Failed to publish message: %v", err)
        } else {
            log.Printf("Published message: %s", message.ID)
        }
    }
    
    // Keep the program running
    select {}
}
```

## Message Patterns

### 1. Work Queue Pattern
- Single producer, multiple consumers
- Load balancing across consumers
- Message acknowledgment

### 2. Pub/Sub Pattern
- One-to-many message distribution
- Fanout exchange
- Broadcast messages to all consumers

### 3. Routing Pattern
- Route messages based on routing key
- Direct exchange
- Selective message delivery

### 4. Topic Pattern
- Route messages based on pattern matching
- Topic exchange
- Wildcard routing

### 5. RPC Pattern
- Request-reply messaging
- Correlation ID
- Response routing

## Benefits

1. **Reliability**: Guaranteed message delivery
2. **Scalability**: Easy to scale consumers
3. **Decoupling**: Loose coupling between services
4. **Flexibility**: Multiple routing patterns
5. **Monitoring**: Built-in monitoring capabilities

## Trade-offs

1. **Complexity**: Adds complexity to the system
2. **Latency**: Additional network hop
3. **Resource Usage**: Requires additional infrastructure
4. **Learning Curve**: Requires understanding of AMQP

## Use Cases

- **Microservices**: Inter-service communication
- **Event Processing**: Event-driven architectures
- **Task Queues**: Background job processing
- **Log Aggregation**: Centralized logging
- **Data Pipeline**: Data processing workflows

## Best Practices

1. **Message Design**: Design messages carefully
2. **Error Handling**: Implement proper error handling
3. **Monitoring**: Monitor queue health and performance
4. **Resource Management**: Manage connections and channels
5. **Testing**: Test message flows thoroughly

## Common Pitfalls

1. **Message Loss**: Not handling acknowledgments properly
2. **Memory Leaks**: Not closing connections
3. **Deadlocks**: Improper queue configuration
4. **Performance Issues**: Not optimizing message size
5. **Error Handling**: Not handling failures gracefully

## Interview Questions

1. **What is RabbitMQ?**
   - A message broker that implements AMQP for reliable message delivery

2. **What are the main components?**
   - Producer, Consumer, Exchange, Queue, Binding, Routing Key

3. **How do you ensure message reliability?**
   - Use acknowledgments, persistent messages, and dead letter queues

4. **What are the different exchange types?**
   - Direct, Fanout, Topic, Headers

## Time Complexity

- **Publish**: O(1) for single message
- **Consume**: O(1) for single message
- **Routing**: O(1) for simple routing

## Space Complexity

- **Queue Storage**: O(n) where n is number of messages
- **Connection**: O(1) per connection
- **Channel**: O(1) per channel

The optimal solution uses:
1. **Proper Configuration**: Configure exchanges and queues correctly
2. **Error Handling**: Implement robust error handling
3. **Monitoring**: Monitor queue health and performance
4. **Resource Management**: Manage connections and channels properly
