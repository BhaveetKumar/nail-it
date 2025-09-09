# Event Sourcing Pattern Implementation

This microservice demonstrates the Event Sourcing design pattern in a real-world fintech application, specifically for managing domain events and aggregate state changes.

## Overview

Event Sourcing is a pattern that stores the state of an application as a sequence of events. Instead of storing the current state, we store all the events that led to that state. This implementation provides a flexible framework for managing domain events, aggregates, and snapshots.

## Features

- **Event Sourcing Pattern**: Implements the classic Event Sourcing pattern with event store, event bus, and snapshot store
- **Domain Events**: Support for user, order, payment, and custom domain events
- **Aggregates**: User, order, payment, and custom aggregate types
- **Event Store**: Persistent storage for all domain events
- **Event Bus**: Publish-subscribe mechanism for event distribution
- **Snapshot Store**: Efficient aggregate state reconstruction
- **Real-time Updates**: WebSocket integration for live event streaming
- **Message Queue Integration**: Kafka integration for event-driven processing
- **Caching**: Redis caching for improved performance
- **Monitoring**: Comprehensive metrics and health checks
- **Security**: JWT authentication and rate limiting
- **Auditing**: Complete audit trail for all events

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   WebSocket     │    │   Kafka         │
│                 │    │   Hub           │    │   Producer      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Event Sourcing Service                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Event Store   │  │   Event Bus     │  │   Snapshot      │ │
│  │                 │  │                 │  │   Store         │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   User          │  │   Order         │  │   Payment       │ │
│  │   Aggregate     │  │   Aggregate     │  │   Aggregate     │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MySQL         │    │   MongoDB       │    │   Redis         │
│   Database      │    │   Database      │    │   Cache         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Event Sourcing Pattern Structure

### Event Interface
```go
type Event interface {
    GetID() string
    GetType() string
    GetAggregateID() string
    GetAggregateType() string
    GetVersion() int
    GetData() map[string]interface{}
    GetMetadata() map[string]interface{}
    GetTimestamp() time.Time
    GetCorrelationID() string
    GetCausationID() string
    SetCorrelationID(correlationID string)
    SetCausationID(causationID string)
    IsProcessed() bool
    SetProcessed(processed bool)
    GetProcessedAt() time.Time
    SetProcessedAt(processedAt time.Time)
}
```

### Aggregate Interface
```go
type Aggregate interface {
    GetID() string
    GetType() string
    GetVersion() int
    GetEvents() []Event
    GetUncommittedEvents() []Event
    MarkEventsAsCommitted()
    ApplyEvent(event Event) error
    GetState() map[string]interface{}
    SetState(state map[string]interface{})
    GetCreatedAt() time.Time
    GetUpdatedAt() time.Time
    IsActive() bool
    SetActive(active bool)
    GetMetadata() map[string]interface{}
    SetMetadata(key string, value interface{})
}
```

### Event Store Interface
```go
type EventStore interface {
    SaveEvents(ctx context.Context, aggregateID string, events []Event, expectedVersion int) error
    GetEvents(ctx context.Context, aggregateID string, fromVersion int) ([]Event, error)
    GetEventsByType(ctx context.Context, eventType string, fromTimestamp time.Time) ([]Event, error)
    GetEventsByAggregateType(ctx context.Context, aggregateType string, fromTimestamp time.Time) ([]Event, error)
    GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error)
    GetEventCount(ctx context.Context, aggregateID string) (int64, error)
    GetAggregateCount(ctx context.Context) (int64, error)
    GetEventTypeCount(ctx context.Context) (int64, error)
    GetStoreStats(ctx context.Context) map[string]interface{}
    Cleanup(ctx context.Context, beforeTimestamp time.Time) error
}
```

### Event Bus Interface
```go
type EventBus interface {
    Publish(ctx context.Context, event Event) error
    Subscribe(ctx context.Context, eventType string, handler EventHandler) error
    Unsubscribe(ctx context.Context, eventType string, handler EventHandler) error
    GetSubscribers(ctx context.Context, eventType string) ([]EventHandler, error)
    GetEventTypes(ctx context.Context) ([]string, error)
    GetHandlerCount(ctx context.Context) (int64, error)
    GetBusStats(ctx context.Context) map[string]interface{}
    Cleanup(ctx context.Context) error
}
```

### Snapshot Store Interface
```go
type SnapshotStore interface {
    SaveSnapshot(ctx context.Context, aggregateID string, snapshot Snapshot) error
    GetSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
    GetLatestSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
    GetSnapshotsByType(ctx context.Context, aggregateType string) ([]Snapshot, error)
    GetSnapshotCount(ctx context.Context, aggregateID string) (int64, error)
    GetStoreStats(ctx context.Context) map[string]interface{}
    Cleanup(ctx context.Context, beforeTimestamp time.Time) error
}
```

### Concrete Events
- **User Created Event**: User account creation with email, name, and role
- **User Updated Event**: User profile updates with change tracking
- **User Deleted Event**: User account deletion with reason and audit trail
- **Order Created Event**: Order creation with items, amounts, and status
- **Order Status Changed Event**: Order status transitions with change tracking
- **Payment Processed Event**: Payment processing with transaction details

### Concrete Aggregates
- **User Aggregate**: Manages user state and user-related events
- **Order Aggregate**: Manages order state and order-related events
- **Payment Aggregate**: Manages payment state and payment-related events

## API Endpoints

### Aggregate Management
- `POST /api/v1/aggregates/` - Create a new aggregate
- `GET /api/v1/aggregates/:id` - Get aggregate details
- `PUT /api/v1/aggregates/:id` - Save aggregate changes

### Event Management
- `GET /api/v1/events/aggregate/:id` - Get events for an aggregate
- `GET /api/v1/events/type/:type` - Get events by type
- `GET /api/v1/events/aggregate-type/:type` - Get events by aggregate type
- `GET /api/v1/events/` - Get all events
- `POST /api/v1/events/publish` - Publish an event

### Snapshot Management
- `POST /api/v1/snapshots/` - Create a snapshot
- `GET /api/v1/snapshots/:aggregate_id` - Get snapshot for an aggregate
- `GET /api/v1/snapshots/:aggregate_id/latest` - Get latest snapshot for an aggregate

### Statistics and Information
- `GET /api/v1/stats` - Get service statistics
- `GET /api/v1/info` - Get service information
- `GET /health` - Health check endpoint

### WebSocket
- `GET /ws` - WebSocket endpoint for real-time event streaming

## Configuration

The service can be configured via `configs/config.yaml`:

```yaml
name: "Event Sourcing Service"
version: "1.0.0"
description: "Event Sourcing pattern implementation with microservice architecture"

max_events: 100000
max_aggregates: 10000
max_snapshots: 1000
snapshot_interval: 1h
cleanup_interval: 24h

validation_enabled: true
caching_enabled: true
monitoring_enabled: true
auditing_enabled: true

supported_event_types:
  - "user_created"
  - "user_updated"
  - "user_deleted"
  - "order_created"
  - "order_status_changed"
  - "payment_processed"
  - "custom"

supported_aggregate_types:
  - "user"
  - "order"
  - "payment"
  - "custom"

validation_rules:
  max_event_data_size: 1048576  # 1MB
  max_aggregate_events: 10000

metadata:
  environment: "production"
  region: "us-east-1"

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "event_sourcing_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "event_sourcing_db"
  redis:
    host: "localhost"
    port: 6379
    password: ""
    db: 0

cache:
  enabled: true
  type: "memory"
  ttl: 5m
  max_size: 1000
  cleanup_interval: 10m

message_queue:
  enabled: true
  brokers:
    - "localhost:9092"
  topics:
    - "event-sourcing-events"

websocket:
  enabled: true
  port: 8080
  read_buffer_size: 1024
  write_buffer_size: 1024
  handshake_timeout: 10s

security:
  enabled: true
  jwt_secret: "your-secret-key"
  token_expiry: 24h
  allowed_origins:
    - "*"
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: 1m

monitoring:
  enabled: true
  port: 9090
  path: "/metrics"
  collect_interval: 30s

logging:
  level: "info"
  format: "json"
  output: "stdout"
```

## Usage Examples

### Creating a User Aggregate
```bash
curl -X POST http://localhost:8080/api/v1/aggregates/ \
  -H "Content-Type: application/json" \
  -d '{
    "type": "user",
    "id": "user-123",
    "initial_state": {
      "email": "john@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "role": "user"
    }
  }'
```

### Publishing a User Created Event
```bash
curl -X POST http://localhost:8080/api/v1/events/publish \
  -H "Content-Type: application/json" \
  -d '{
    "type": "user_created",
    "aggregate_id": "user-123",
    "aggregate_type": "user",
    "version": 1,
    "data": {
      "user_id": "user-123",
      "email": "john@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "role": "user"
    },
    "metadata": {
      "source": "user-service",
      "correlation_id": "corr-123"
    }
  }'
```

### Getting Events for an Aggregate
```bash
curl http://localhost:8080/api/v1/events/aggregate/user-123
```

### Creating a Snapshot
```bash
curl -X POST http://localhost:8080/api/v1/snapshots/ \
  -H "Content-Type: application/json" \
  -d '{
    "aggregate_id": "user-123",
    "version": 10,
    "data": {
      "email": "john@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "role": "user",
      "status": "active"
    }
  }'
```

### Getting Service Statistics
```bash
curl http://localhost:8080/api/v1/stats
```

## Event Sourcing Pattern Benefits

1. **Complete Audit Trail**: Every change is recorded as an event
2. **Time Travel**: Can reconstruct state at any point in time
3. **Event Replay**: Can replay events to rebuild state
4. **Scalability**: Events can be processed asynchronously
5. **Flexibility**: New event handlers can be added without changing existing code
6. **Debugging**: Easy to trace the sequence of events that led to current state

## Real-World Use Cases

### Fintech Applications
- **User Management**: User registration, profile updates, and account changes
- **Order Processing**: Order creation, status changes, and fulfillment tracking
- **Payment Processing**: Payment initiation, processing, and completion
- **Transaction History**: Complete audit trail of all financial transactions
- **Compliance**: Regulatory reporting and audit requirements
- **Fraud Detection**: Event pattern analysis for fraud detection

### Other Applications
- **E-commerce**: Order management, inventory tracking, and customer behavior
- **Healthcare**: Patient record changes, treatment history, and medical events
- **Education**: Student enrollment, grade changes, and academic progress
- **Manufacturing**: Production events, quality control, and supply chain tracking

## Testing

The service includes comprehensive tests for all components:

```bash
# Run unit tests
go test ./...

# Run integration tests
go test -tags=integration ./...

# Run benchmarks
go test -bench=. ./...

# Run tests with coverage
go test -cover ./...
```

## Monitoring and Observability

The service provides comprehensive monitoring capabilities:

- **Health Checks**: Database, cache, and external service health monitoring
- **Metrics**: Event counts, aggregate counts, and performance metrics
- **Logging**: Structured logging with configurable levels and formats
- **Tracing**: Distributed tracing for event flow analysis
- **Alerting**: Configurable alerts for critical events and thresholds

## Security

The service implements multiple security layers:

- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Rate Limiting**: Request rate limiting to prevent abuse
- **Input Validation**: Comprehensive input validation and sanitization
- **Audit Logging**: Complete audit trail for all operations
- **Encryption**: Data encryption at rest and in transit

## Performance Optimization

The service is optimized for high performance:

- **Caching**: Redis caching for frequently accessed data
- **Connection Pooling**: Database connection pooling for efficient resource usage
- **Async Processing**: Asynchronous processing for long-running operations
- **Load Balancing**: Horizontal scaling support
- **Resource Management**: Efficient memory and CPU usage

## Deployment

The service can be deployed using various methods:

- **Docker**: Containerized deployment with Docker
- **Kubernetes**: Kubernetes deployment with Helm charts
- **Cloud**: AWS, GCP, or Azure deployment
- **On-premise**: Traditional server deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the development team or create an issue in the repository.
