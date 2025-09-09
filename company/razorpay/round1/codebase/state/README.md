# State Pattern Implementation

This is a complete microservice implementation demonstrating the State design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the State pattern for:
- **State Management**: Manage entity states and transitions
- **State Machine**: Define state machines with states and transitions
- **State Events**: Handle events that trigger state transitions
- **State History**: Track state transition history
- **State Validation**: Validate state transitions
- **State Metrics**: Collect state transition metrics
- **State Persistence**: Persist state information
- **State Notifications**: Real-time state change notifications

## Features

### Core Functionality
- **State Machine**: Define and manage state machines
- **State Transitions**: Handle state transitions based on events
- **State History**: Track complete state transition history
- **State Validation**: Validate state transitions
- **State Metrics**: Comprehensive metrics and monitoring
- **State Persistence**: Persist state information
- **Real-time Updates**: WebSocket-based real-time state updates
- **Event Streaming**: Kafka integration for state events
- **State Queries**: Query entities by state and type

### State Types

#### 1. Payment States
```go
// Payment state machine with states:
// pending -> completed, failed, cancelled
// completed -> refunded
// failed -> pending (retry)
// cancelled -> (final)
// refunded -> (final)

type PaymentState struct {
    PaymentID     string            `json:"payment_id"`
    UserID        string            `json:"user_id"`
    Amount        float64           `json:"amount"`
    Currency      string            `json:"currency"`
    PaymentMethod string            `json:"payment_method"`
    Gateway       string            `json:"gateway"`
    Description   string            `json:"description"`
    Status        string            `json:"status"`
    Metadata      map[string]string `json:"metadata"`
}
```

**Available States:**
- **Pending**: Payment is pending processing
- **Completed**: Payment has been completed successfully
- **Failed**: Payment has failed
- **Cancelled**: Payment has been cancelled
- **Refunded**: Payment has been refunded

#### 2. Order States
```go
// Order state machine with states:
// created -> confirmed, cancelled
// confirmed -> shipped, cancelled
// shipped -> delivered, returned
// delivered -> (final)
// cancelled -> (final)
// returned -> (final)

type OrderState struct {
    OrderID       string            `json:"order_id"`
    UserID        string            `json:"user_id"`
    Status        string            `json:"status"`
    Items         []OrderItem       `json:"items"`
    TotalAmount   float64           `json:"total_amount"`
    Currency      string            `json:"currency"`
    ShippingAddress string          `json:"shipping_address"`
    BillingAddress  string          `json:"billing_address"`
    Metadata      map[string]string `json:"metadata"`
}
```

#### 3. User States
```go
// User state machine with states:
// inactive -> active, suspended
// active -> suspended, deactivated
// suspended -> active, deactivated
// deactivated -> (final)

type UserState struct {
    UserID      string            `json:"user_id"`
    Status      string            `json:"status"`
    Email       string            `json:"email"`
    Name        string            `json:"name"`
    Phone       string            `json:"phone"`
    Address     string            `json:"address"`
    Metadata    map[string]string `json:"metadata"`
}
```

#### 4. Inventory States
```go
// Inventory state machine with states:
// available -> out_of_stock, discontinued
// out_of_stock -> available, discontinued
// discontinued -> (final)

type InventoryState struct {
    ProductID   string            `json:"product_id"`
    Status      string            `json:"status"`
    Name        string            `json:"name"`
    Description string            `json:"description"`
    Price       float64           `json:"price"`
    Stock       int               `json:"stock"`
    Category    string            `json:"category"`
    Metadata    map[string]string `json:"metadata"`
}
```

#### 5. Notification States
```go
// Notification state machine with states:
// pending -> sent, failed
// sent -> delivered, failed
// delivered -> (final)
// failed -> (final)

type NotificationState struct {
    NotificationID string            `json:"notification_id"`
    UserID         string            `json:"user_id"`
    Status         string            `json:"status"`
    Channel        string            `json:"channel"`
    Type           string            `json:"type"`
    Title          string            `json:"title"`
    Message        string            `json:"message"`
    Priority       string            `json:"priority"`
    Metadata       map[string]string `json:"metadata"`
}
```

#### 6. Refund States
```go
// Refund state machine with states:
// requested -> approved, rejected
// approved -> processed, failed
// processed -> (final)
// rejected -> (final)
// failed -> (final)

type RefundState struct {
    RefundID   string            `json:"refund_id"`
    PaymentID  string            `json:"payment_id"`
    UserID     string            `json:"user_id"`
    Status     string            `json:"status"`
    Amount     float64           `json:"amount"`
    Currency   string            `json:"currency"`
    Reason     string            `json:"reason"`
    Metadata   map[string]string `json:"metadata"`
}
```

#### 7. Audit States
```go
// Audit state machine with states:
// pending -> completed, failed
// completed -> (final)
// failed -> (final)

type AuditState struct {
    AuditID    string            `json:"audit_id"`
    EntityType string            `json:"entity_type"`
    EntityID   string            `json:"entity_id"`
    Status     string            `json:"status"`
    Action     string            `json:"action"`
    Changes    map[string]interface{} `json:"changes"`
    UserID     string            `json:"user_id"`
    IPAddress  string            `json:"ip_address"`
    UserAgent  string            `json:"user_agent"`
    Metadata   map[string]string `json:"metadata"`
}
```

#### 8. System States
```go
// System state machine with states:
// starting -> running, failed
// running -> stopping, failed
// stopping -> stopped, failed
// stopped -> starting, failed
// failed -> (final)

type SystemState struct {
    SystemID   string            `json:"system_id"`
    Status     string            `json:"status"`
    Component  string            `json:"component"`
    Version    string            `json:"version"`
    Health     string            `json:"health"`
    Metadata   map[string]string `json:"metadata"`
}
```

## API Endpoints

### State Machine Management
- `GET /api/v1/state-machines` - Get all state machines
- `GET /api/v1/state-machines/:name` - Get state machine by name
- `POST /api/v1/state-machines` - Create state machine
- `PUT /api/v1/state-machines/:name` - Update state machine
- `DELETE /api/v1/state-machines/:name` - Delete state machine

### State Management
- `GET /api/v1/states` - Get all states
- `GET /api/v1/states/:name` - Get state by name
- `POST /api/v1/states` - Create state
- `PUT /api/v1/states/:name` - Update state
- `DELETE /api/v1/states/:name` - Delete state

### Entity Management
- `POST /api/v1/entities` - Create entity
- `GET /api/v1/entities/:id` - Get entity by ID
- `PUT /api/v1/entities/:id` - Update entity
- `DELETE /api/v1/entities/:id` - Delete entity
- `GET /api/v1/entities` - Get all entities
- `GET /api/v1/entities/state/:state` - Get entities by state
- `GET /api/v1/entities/type/:type` - Get entities by type

### State Transitions
- `POST /api/v1/entities/:id/transition` - Transition entity state
- `GET /api/v1/entities/:id/transitions` - Get possible transitions
- `GET /api/v1/entities/:id/history` - Get entity state history

### State Events
- `POST /api/v1/events` - Create state event
- `GET /api/v1/events` - Get all events
- `GET /api/v1/events/:id` - Get event by ID
- `GET /api/v1/events/type/:type` - Get events by type

### State Metrics
- `GET /api/v1/state-metrics` - Get state metrics
- `GET /api/v1/state-metrics/:type` - Get metrics by entity type
- `GET /api/v1/state-metrics/transitions` - Get transition metrics
- `POST /api/v1/state-metrics/reset` - Reset state metrics

### State Validation
- `POST /api/v1/validate/transition` - Validate state transition
- `POST /api/v1/validate/entity` - Validate entity state
- `POST /api/v1/validate/event` - Validate state event

### WebSocket
- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check
- `GET /health` - Service health status

## WebSocket Events

### State Events
- `state.transitioned` - State transitioned
- `state.failed` - State transition failed
- `state.validated` - State validated
- `state.invalidated` - State invalidated

### Entity Events
- `entity.created` - Entity created
- `entity.updated` - Entity updated
- `entity.deleted` - Entity deleted
- `entity.state_changed` - Entity state changed

## Kafka Events

### State Events
- All state operations are streamed to Kafka for external consumption
- State transition and event events
- State metrics and validation events

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
cd state
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
CREATE DATABASE state_db;
```

4. **Run the service**:
```bash
go run main.go
```

## Configuration

The service uses a YAML configuration file (`configs/config.yaml`) with the following sections:

- **Server**: HTTP server configuration
- **Database**: MySQL connection settings
- **MongoDB**: MongoDB connection settings
- **Redis**: Redis connection settings
- **Kafka**: Kafka broker and topic configuration
- **State**: State management configuration

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

# Test state transition
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"entity_id":"entity123","entity_type":"payment","state":"pending","data":{"payment_id":"pay123","user_id":"user123","amount":100.50,"currency":"USD","payment_method":"card","gateway":"stripe","description":"Test payment"}}' http://localhost:8080/api/v1/entities

# Test state transition
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"type":"payment_processed","entity_id":"entity123","entity_type":"payment","data":{"transaction_id":"txn123"}}' http://localhost:8080/api/v1/entities/entity123/transition
```

## Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### State Metrics
```bash
curl http://localhost:8080/api/v1/state-metrics
curl http://localhost:8080/api/v1/state-metrics/payment
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
# Consume state events
kafka-console-consumer --bootstrap-server localhost:9092 --topic state-events --from-beginning
```

## Performance Considerations

### State Benefits
- **Encapsulation**: States encapsulate behavior
- **Transitions**: Clear state transition rules
- **History**: Complete state transition history
- **Validation**: State transition validation
- **Metrics**: State transition metrics
- **Persistence**: State information persistence

### Optimization Strategies
- **State Caching**: Cache state information
- **Batch Processing**: Process multiple state transitions
- **Async Processing**: Process state transitions asynchronously
- **Metrics Collection**: Monitor state performance
- **Health Checks**: Regular state health monitoring

## Error Handling

The service implements comprehensive error handling:
- **State Validation**: Validate state transitions
- **Error Recovery**: Recover from state errors
- **Error Logging**: Comprehensive error logging
- **Graceful Degradation**: Continue operation with reduced functionality
- **Audit Trail**: Complete audit trail of errors

## Security Considerations

- **Input Validation**: Validate all state inputs
- **Authentication**: Secure state access
- **Authorization**: Control state permissions
- **Rate Limiting**: Prevent state abuse
- **Audit Logging**: Log all state operations
- **Encryption**: Encrypt sensitive state data

## Scalability

### Horizontal Scaling
- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **State Distribution**: Distribute states across instances
- **Metrics Aggregation**: Aggregate metrics across instances

### Vertical Scaling
- **Memory Management**: Efficient state memory usage
- **CPU Optimization**: Concurrent state processing
- **Connection Pooling**: Database connection optimization
- **Caching**: State result caching

## Troubleshooting

### Common Issues

1. **State Transition Failed**
   - Check state validation rules
   - Verify state machine configuration
   - Check entity state
   - Monitor state metrics

2. **Invalid State Transition**
   - Check transition rules
   - Verify state machine states
   - Check entity type
   - Monitor validation errors

3. **High State Transition Latency**
   - Check state performance metrics
   - Verify external service performance
   - Check network connectivity
   - Monitor system resources

4. **State Machine Issues**
   - Check state machine configuration
   - Verify state definitions
   - Check transition rules
   - Monitor state machine health

### Logs
```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View state logs
grep "State" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
