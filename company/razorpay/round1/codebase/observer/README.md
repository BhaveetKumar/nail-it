# Observer Pattern Implementation

This is a complete microservice implementation demonstrating the Observer design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Observer pattern for:

- **Event-Driven Architecture**: Decoupled event publishing and subscription
- **Real-time Notifications**: WebSocket-based event distribution
- **Event Persistence**: MongoDB-based event store
- **Event Processing**: Kafka integration for event streaming
- **Metrics and Monitoring**: Comprehensive event metrics tracking
- **Circuit Breaker**: Fault tolerance for event processing
- **Rate Limiting**: Event processing rate control

## Features

### Core Functionality

- **Event Publishing**: Publish events to multiple subscribers
- **Event Subscription**: Subscribe to specific event types
- **Event Storage**: Persistent event store with MongoDB
- **Event Replay**: Replay events from event store
- **Real-time Updates**: WebSocket-based real-time event distribution
- **Event Streaming**: Kafka integration for event streaming
- **Metrics Tracking**: Comprehensive event processing metrics
- **Fault Tolerance**: Circuit breaker and retry mechanisms

### Observer Implementations

#### 1. Event Bus

```go
// Event bus for managing event distribution
type EventBus interface {
    Subscribe(eventType string, observer Observer) error
    Unsubscribe(eventType string, observerID string) error
    Publish(ctx context.Context, event Event) error
    GetSubscriberCount(eventType string) int
    GetEventTypes() []string
}
```

#### 2. Event Store

```go
// Event store for persistent event storage
type EventStore interface {
    Store(ctx context.Context, event Event) error
    GetEvents(ctx context.Context, eventType string, limit, offset int) ([]Event, error)
    GetEventByID(ctx context.Context, eventID string) (Event, error)
    GetEventsByTimeRange(ctx context.Context, start, end time.Time) ([]Event, error)
}
```

#### 3. Event Observers

```go
// Observer interface for event handling
type Observer interface {
    OnEvent(ctx context.Context, event Event) error
    GetObserverID() string
    GetEventTypes() []string
    IsAsync() bool
}
```

#### 4. Event Metrics

```go
// Event metrics for monitoring
type EventMetrics interface {
    IncrementEventCount(eventType string)
    IncrementErrorCount(eventType string)
    RecordProcessingTime(eventType string, duration time.Duration)
    GetMetrics() EventMetricsData
}
```

## API Endpoints

### Events

- `POST /api/v1/events` - Publish event
- `GET /api/v1/events` - Get events (paginated)
- `GET /api/v1/events/:id` - Get event by ID
- `GET /api/v1/events/type/:type` - Get events by type
- `GET /api/v1/events/range` - Get events by time range

### Observers

- `POST /api/v1/observers` - Create observer
- `GET /api/v1/observers` - Get all observers
- `GET /api/v1/observers/:id` - Get observer by ID
- `PUT /api/v1/observers/:id` - Update observer
- `DELETE /api/v1/observers/:id` - Delete observer

### Subscriptions

- `POST /api/v1/subscriptions` - Subscribe observer to event type
- `DELETE /api/v1/subscriptions` - Unsubscribe observer from event type
- `GET /api/v1/subscriptions` - Get all subscriptions

### Event Store

- `GET /api/v1/event-store/events` - Get events from store
- `GET /api/v1/event-store/events/:id` - Get event from store
- `POST /api/v1/event-store/replay` - Replay events
- `DELETE /api/v1/event-store/events/old` - Delete old events

### Metrics

- `GET /api/v1/metrics` - Get event metrics
- `GET /api/v1/metrics/events` - Get event count metrics
- `GET /api/v1/metrics/errors` - Get error metrics
- `GET /api/v1/metrics/processing-times` - Get processing time metrics

### Circuit Breaker

- `GET /api/v1/circuit-breaker/status` - Get circuit breaker status
- `POST /api/v1/circuit-breaker/reset` - Reset circuit breaker

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## Event Types

### Payment Events

- `payment.created` - Payment created
- `payment.updated` - Payment updated
- `payment.completed` - Payment completed
- `payment.failed` - Payment failed
- `payment.refunded` - Payment refunded

### User Events

- `user.created` - User created
- `user.updated` - User updated
- `user.deleted` - User deleted
- `user.activated` - User activated
- `user.deactivated` - User deactivated

### Order Events

- `order.created` - Order created
- `order.updated` - Order updated
- `order.cancelled` - Order cancelled
- `order.completed` - Order completed
- `order.shipped` - Order shipped
- `order.delivered` - Order delivered

### Product Events

- `product.created` - Product created
- `product.updated` - Product updated
- `product.deleted` - Product deleted
- `product.stock_low` - Product stock low
- `product.out_of_stock` - Product out of stock

### Notification Events

- `notification.sent` - Notification sent
- `notification.failed` - Notification failed
- `notification.delivered` - Notification delivered
- `notification.read` - Notification read

### System Events

- `system.startup` - System startup
- `system.shutdown` - System shutdown
- `system.error` - System error
- `system.warning` - System warning
- `system.info` - System info

## WebSocket Events

### Event Distribution

- `event.published` - Event published
- `event.processed` - Event processed
- `event.failed` - Event processing failed

### Observer Events

- `observer.subscribed` - Observer subscribed
- `observer.unsubscribed` - Observer unsubscribed
- `observer.enabled` - Observer enabled
- `observer.disabled` - Observer disabled

## Kafka Events

### Event Streaming

- All event types are streamed to Kafka for external consumption
- Event replay capabilities from Kafka
- Event correlation and ordering

## Setup Instructions

### Prerequisites

- Go 1.21+
- MySQL 8.0+
- MongoDB 4.4+
- Redis 6.0+
- Kafka 2.8+

### Installation

1. **Clone and setup**:

```bash
cd observer
go mod tidy
```

2. **Start dependencies**:

```bash
# Start MySQL
docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -p 3306:3306 mysql:8.0

# Start MongoDB
docker run -d --name mongodb -p 27017:27017 mongo:4.4

# Start Redis
docker run -d --name redis -p 6379:6379 redis:6.0

# Start Kafka
docker-compose up -d kafka zookeeper
```

3. **Create database**:

```sql
CREATE DATABASE observer_db;
```

4. **Run the service**:

```bash
go run main.go
```

## Configuration

The service uses a YAML configuration file (`configs/config.yaml`) with the following sections:

- **Server**: HTTP server configuration
- **Database**: MySQL connection settings
- **Redis**: Redis connection settings
- **Kafka**: Kafka broker and topic configuration
- **MongoDB**: MongoDB connection settings
- **EventStore**: Event store configuration
- **Metrics**: Metrics collection configuration

## Testing

### Unit Tests

```bash
go test ./...
```

### Integration Tests

```bash
go test -tags=integration ./...
```

### Load Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Test event publishing
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"type":"payment.created","data":{"payment_id":"pay123","user_id":"user123","amount":100.50,"currency":"USD","status":"completed","gateway":"stripe"}}' http://localhost:8080/api/v1/events

# Test observer creation
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"id":"observer123","event_types":["payment.created","payment.updated"],"async":true}' http://localhost:8080/api/v1/observers
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Event Metrics

```bash
curl http://localhost:8080/api/v1/metrics
curl http://localhost:8080/api/v1/metrics/events
curl http://localhost:8080/api/v1/metrics/errors
```

### Circuit Breaker Status

```bash
curl http://localhost:8080/api/v1/circuit-breaker/status
```

### WebSocket Connection

```javascript
const ws = new WebSocket(
  "ws://localhost:8080/ws?user_id=user123&client_id=client456"
);
ws.onmessage = function (event) {
  console.log("Received:", JSON.parse(event.data));
};
```

### Kafka Events

```bash
# Consume events
kafka-console-consumer --bootstrap-server localhost:9092 --topic observer-events --from-beginning
```

## Performance Considerations

### Observer Benefits

- **Decoupling**: Publishers and subscribers are decoupled
- **Scalability**: Easy to add new observers
- **Flexibility**: Dynamic subscription management
- **Real-time**: Immediate event distribution
- **Persistence**: Event store for replay and audit

### Optimization Strategies

- **Async Processing**: Non-blocking event processing
- **Batch Processing**: Process events in batches
- **Circuit Breaker**: Fault tolerance for event processing
- **Rate Limiting**: Control event processing rate
- **Event Filtering**: Filter events before processing
- **Event Transformation**: Transform events as needed

## Error Handling

The service implements comprehensive error handling:

- **Event Processing Errors**: Retry logic with exponential backoff
- **Observer Errors**: Circuit breaker protection
- **Dead Letter Queue**: Failed event handling
- **Rate Limiting**: Event processing rate control
- **Validation Errors**: Event validation and error reporting
- **Graceful Shutdown**: Clean resource cleanup

## Security Considerations

- **Input Validation**: Event data validation
- **Rate Limiting**: Prevent event flooding
- **Circuit Breaker**: Prevent cascade failures
- **Event Filtering**: Filter sensitive events
- **Audit Logging**: Complete event audit trail

## Scalability

### Horizontal Scaling

- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **Event Partitioning**: Distribute events across instances
- **Observer Distribution**: Distribute observers across instances

### Vertical Scaling

- **Async Processing**: Non-blocking event processing
- **Memory Management**: Efficient event storage
- **CPU Optimization**: Concurrent event processing
- **Connection Pooling**: Database connection optimization

## Troubleshooting

### Common Issues

1. **Event Processing Failed**

   - Check observer implementation
   - Verify event data format
   - Check circuit breaker status
   - Monitor error metrics

2. **Observer Not Receiving Events**

   - Verify subscription status
   - Check event type matching
   - Verify observer is enabled
   - Check event filtering

3. **High Event Processing Latency**

   - Check processing time metrics
   - Verify observer performance
   - Check circuit breaker status
   - Monitor system resources

4. **Event Store Issues**
   - Check MongoDB connection
   - Verify event store indexes
   - Check event data format
   - Monitor storage usage

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View event logs
grep "Event" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
