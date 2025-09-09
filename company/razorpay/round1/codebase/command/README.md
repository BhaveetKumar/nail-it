# Command Pattern Implementation

This is a complete microservice implementation demonstrating the Command design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Command pattern for:

- **Command Encapsulation**: Encapsulate requests as objects
- **Command Invocation**: Execute commands through invokers
- **Command Undo**: Support for undo operations
- **Command Queuing**: Queue commands for later execution
- **Command Scheduling**: Schedule commands for future execution
- **Command Auditing**: Audit all command executions
- **Command Metrics**: Collect command execution metrics
- **Command Validation**: Validate commands before execution

## Features

### Core Functionality

- **Command Execution**: Execute commands synchronously and asynchronously
- **Command Batching**: Execute multiple commands in batch
- **Command History**: Track command execution history
- **Command Undo**: Undo executed commands
- **Command Queuing**: Queue commands for processing
- **Command Scheduling**: Schedule commands for future execution
- **Command Auditing**: Complete audit trail of command executions
- **Command Metrics**: Comprehensive metrics and monitoring
- **Circuit Breaker**: Fault tolerance for command execution
- **Real-time Updates**: WebSocket-based real-time command updates
- **Event Streaming**: Kafka integration for command events

### Command Types

#### 1. Payment Commands

```go
// Payment command for processing payments
type PaymentCommand struct {
    CommandID     string            `json:"command_id"`
    UserID        string            `json:"user_id"`
    Amount        float64           `json:"amount"`
    Currency      string            `json:"currency"`
    PaymentMethod string            `json:"payment_method"`
    Gateway       string            `json:"gateway"`
    Description   string            `json:"description"`
    Metadata      map[string]string `json:"metadata"`
}
```

#### 2. User Commands

```go
// User command for user operations
type UserCommand struct {
    CommandID string            `json:"command_id"`
    UserID    string            `json:"user_id"`
    Action    string            `json:"action"`
    Data      map[string]interface{} `json:"data"`
    Metadata  map[string]string `json:"metadata"`
}
```

#### 3. Order Commands

```go
// Order command for order operations
type OrderCommand struct {
    CommandID   string            `json:"command_id"`
    OrderID     string            `json:"order_id"`
    UserID      string            `json:"user_id"`
    Action      string            `json:"action"`
    Items       []OrderItem       `json:"items"`
    TotalAmount float64           `json:"total_amount"`
    Currency    string            `json:"currency"`
    Metadata    map[string]string `json:"metadata"`
}
```

#### 4. Notification Commands

```go
// Notification command for sending notifications
type NotificationCommand struct {
    CommandID      string            `json:"command_id"`
    UserID         string            `json:"user_id"`
    Channel        string            `json:"channel"`
    Type           string            `json:"type"`
    Title          string            `json:"title"`
    Message        string            `json:"message"`
    Priority       string            `json:"priority"`
    Metadata       map[string]string `json:"metadata"`
}
```

#### 5. Inventory Commands

```go
// Inventory command for inventory operations
type InventoryCommand struct {
    CommandID   string            `json:"command_id"`
    ProductID   string            `json:"product_id"`
    Action      string            `json:"action"`
    Quantity    int               `json:"quantity"`
    Reason      string            `json:"reason"`
    Metadata    map[string]string `json:"metadata"`
}
```

#### 6. Refund Commands

```go
// Refund command for processing refunds
type RefundCommand struct {
    CommandID     string            `json:"command_id"`
    PaymentID     string            `json:"payment_id"`
    UserID        string            `json:"user_id"`
    Amount        float64           `json:"amount"`
    Currency      string            `json:"currency"`
    Reason        string            `json:"reason"`
    Metadata      map[string]string `json:"metadata"`
}
```

#### 7. Audit Commands

```go
// Audit command for audit logging
type AuditCommand struct {
    CommandID   string            `json:"command_id"`
    EntityType  string            `json:"entity_type"`
    EntityID    string            `json:"entity_id"`
    Action      string            `json:"action"`
    Changes     map[string]interface{} `json:"changes"`
    UserID      string            `json:"user_id"`
    IPAddress   string            `json:"ip_address"`
    UserAgent   string            `json:"user_agent"`
    Metadata    map[string]string `json:"metadata"`
}
```

#### 8. System Commands

```go
// System command for system operations
type SystemCommand struct {
    CommandID   string            `json:"command_id"`
    Action      string            `json:"action"`
    Parameters  map[string]interface{} `json:"parameters"`
    Metadata    map[string]string `json:"metadata"`
}
```

## API Endpoints

### Command Execution

- `POST /api/v1/commands/execute` - Execute command synchronously
- `POST /api/v1/commands/execute-async` - Execute command asynchronously
- `POST /api/v1/commands/execute-batch` - Execute multiple commands in batch
- `POST /api/v1/commands/undo` - Undo command execution

### Command Management

- `GET /api/v1/commands` - Get all commands
- `GET /api/v1/commands/:id` - Get command by ID
- `GET /api/v1/commands/history` - Get command execution history
- `DELETE /api/v1/commands/history` - Clear command history

### Command Handlers

- `GET /api/v1/commands/handlers` - Get available command handlers
- `GET /api/v1/commands/handlers/:name` - Get command handler by name
- `POST /api/v1/commands/handlers/:name/register` - Register command handler
- `DELETE /api/v1/commands/handlers/:name` - Unregister command handler

### Command Queuing

- `POST /api/v1/commands/queue` - Enqueue command
- `GET /api/v1/commands/queue` - Get queued commands
- `DELETE /api/v1/commands/queue` - Clear command queue

### Command Scheduling

- `POST /api/v1/commands/schedule` - Schedule command execution
- `GET /api/v1/commands/scheduled` - Get scheduled commands
- `DELETE /api/v1/commands/scheduled/:id` - Cancel scheduled command

### Command Auditing

- `GET /api/v1/commands/audit` - Get audit logs
- `GET /api/v1/commands/audit/:id` - Get audit log by ID
- `GET /api/v1/commands/audit/type/:type` - Get audit logs by command type
- `GET /api/v1/commands/audit/range` - Get audit logs by time range

### Command Metrics

- `GET /api/v1/commands/metrics` - Get command metrics
- `GET /api/v1/commands/metrics/:type` - Get metrics by command type
- `POST /api/v1/commands/metrics/reset` - Reset command metrics

### Circuit Breaker

- `GET /api/v1/commands/circuit-breaker/status` - Get circuit breaker status
- `POST /api/v1/commands/circuit-breaker/reset` - Reset circuit breaker

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## WebSocket Events

### Command Events

- `command.executed` - Command executed
- `command.failed` - Command execution failed
- `command.queued` - Command queued
- `command.scheduled` - Command scheduled
- `command.cancelled` - Command cancelled
- `command.undone` - Command undone

### Handler Events

- `handler.registered` - Command handler registered
- `handler.unregistered` - Command handler unregistered
- `handler.available` - Command handler available
- `handler.unavailable` - Command handler unavailable

## Kafka Events

### Command Events

- All command operations are streamed to Kafka for external consumption
- Command execution and result events
- Command audit and metrics events

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
cd command
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
CREATE DATABASE command_db;
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
- **Command**: Command configuration and parameters

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

# Test command execution
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"command_id":"cmd123","user_id":"user123","amount":100.50,"currency":"USD","payment_method":"card","gateway":"stripe","description":"Test payment"}' http://localhost:8080/api/v1/commands/execute

# Test batch command execution
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '[{"command_id":"cmd123","user_id":"user123","action":"create","data":{"name":"John Doe","email":"john@example.com"}},{"command_id":"cmd124","user_id":"user123","action":"update","data":{"name":"Jane Doe"}}]' http://localhost:8080/api/v1/commands/execute-batch
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Command Metrics

```bash
curl http://localhost:8080/api/v1/commands/metrics
curl http://localhost:8080/api/v1/commands/metrics/payment
```

### Circuit Breaker Status

```bash
curl http://localhost:8080/api/v1/commands/circuit-breaker/status
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
# Consume command events
kafka-console-consumer --bootstrap-server localhost:9092 --topic command-events --from-beginning
```

## Performance Considerations

### Command Benefits

- **Encapsulation**: Commands encapsulate requests as objects
- **Undo Support**: Easy to implement undo functionality
- **Queuing**: Commands can be queued for later execution
- **Scheduling**: Commands can be scheduled for future execution
- **Auditing**: Complete audit trail of command executions
- **Metrics**: Comprehensive metrics and monitoring

### Optimization Strategies

- **Command Caching**: Cache command instances
- **Batch Processing**: Process multiple commands in batch
- **Async Execution**: Execute commands asynchronously
- **Circuit Breaker**: Prevent cascade failures
- **Metrics Collection**: Monitor command performance
- **Health Checks**: Regular command health monitoring

## Error Handling

The service implements comprehensive error handling:

- **Command Validation**: Validate commands before execution
- **Retry Logic**: Retry failed command executions
- **Circuit Breaker**: Prevent cascade failures
- **Error Logging**: Comprehensive error logging
- **Graceful Degradation**: Continue operation with reduced functionality
- **Audit Trail**: Complete audit trail of errors

## Security Considerations

- **Input Validation**: Validate all command inputs
- **Authentication**: Secure command access
- **Authorization**: Control command permissions
- **Rate Limiting**: Prevent command abuse
- **Audit Logging**: Log all command operations
- **Encryption**: Encrypt sensitive command data

## Scalability

### Horizontal Scaling

- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **Command Distribution**: Distribute commands across instances
- **Metrics Aggregation**: Aggregate metrics across instances

### Vertical Scaling

- **Memory Management**: Efficient command memory usage
- **CPU Optimization**: Concurrent command execution
- **Connection Pooling**: Database connection optimization
- **Caching**: Command result caching

## Troubleshooting

### Common Issues

1. **Command Execution Failed**

   - Check command validation
   - Verify command handler availability
   - Check circuit breaker status
   - Monitor command metrics

2. **Command Handler Not Found**

   - Check handler registration
   - Verify handler availability
   - Check handler configuration
   - Monitor handler health

3. **High Command Execution Latency**

   - Check command performance metrics
   - Verify external service performance
   - Check network connectivity
   - Monitor system resources

4. **Command Queue Issues**
   - Check queue size limits
   - Verify queue processing
   - Check queue configuration
   - Monitor queue metrics

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View command logs
grep "Command" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
