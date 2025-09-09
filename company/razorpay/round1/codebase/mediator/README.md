# Mediator Pattern Microservice

A complete microservice implementation demonstrating the Mediator design pattern with MongoDB, MySQL, REST API, WebSockets, and Kafka integration.

## Overview

The Mediator pattern defines an object that encapsulates how a set of objects interact. This pattern is considered a behavioral pattern as it can alter the program's running behavior. The mediator promotes loose coupling by keeping objects from referring to each other explicitly and allows their interaction to be varied independently.

## Features

- **Mediator Management**: Create, manage, and coordinate multiple mediators
- **Colleague Registration**: Register and unregister colleagues with mediators
- **Message Passing**: Send messages between colleagues through mediators
- **Broadcast Communication**: Broadcast messages to all colleagues
- **Event Handling**: Handle various types of events and commands
- **Workflow Management**: Manage complex workflows and processes
- **Service Coordination**: Coordinate services and resources
- **Task Management**: Manage tasks and jobs
- **Caching**: In-memory caching for improved performance
- **Real-time Updates**: WebSocket support for real-time communication
- **Message Queue**: Kafka integration for event-driven architecture
- **Database Integration**: MySQL and MongoDB support
- **Monitoring**: Comprehensive metrics and health checks
- **Security**: JWT authentication and rate limiting
- **Audit Logging**: Complete audit trail for all operations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │   WebSocket     │    │   Kafka         │
│   (Gin)         │    │   (Gorilla)     │    │   (Sarama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Mediator Service│
                    │   (Core Logic)  │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MySQL       │    │    MongoDB      │    │     Redis       │
│   (GORM)        │    │   (MongoDB)     │    │   (go-redis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Mediator Types

### 1. Message Mediator
Handles message passing between colleagues.

```go
mediator := NewConcreteMediator(config)
colleague1 := &MessageColleague{ID: "colleague1", Name: "Colleague 1"}
colleague2 := &MessageColleague{ID: "colleague2", Name: "Colleague 2"}

mediator.RegisterColleague(colleague1)
mediator.RegisterColleague(colleague2)

// Send message
mediator.SendMessage("colleague1", "colleague2", "Hello from colleague 1")
```

### 2. Event Mediator
Handles event distribution and processing.

```go
mediator := NewConcreteMediator(config)
eventHandler := &EventHandler{ID: "handler1", Name: "Event Handler"}

mediator.RegisterColleague(eventHandler)

// Broadcast event
mediator.BroadcastMessage("handler1", &Event{
    Type: "user_created",
    Data: map[string]interface{}{"user_id": "123"},
})
```

### 3. Command Mediator
Handles command execution and coordination.

```go
mediator := NewConcreteMediator(config)
commandHandler := &CommandHandler{ID: "handler1", Name: "Command Handler"}

mediator.RegisterColleague(commandHandler)

// Execute command
mediator.SendMessage("handler1", "handler1", &Command{
    Type: "create_user",
    Data: map[string]interface{}{"name": "John Doe"},
})
```

### 4. Query Mediator
Handles query processing and data retrieval.

```go
mediator := NewConcreteMediator(config)
queryHandler := &QueryHandler{ID: "handler1", Name: "Query Handler"}

mediator.RegisterColleague(queryHandler)

// Process query
mediator.SendMessage("handler1", "handler1", &Query{
    Type: "get_user",
    Data: map[string]interface{}{"user_id": "123"},
})
```

### 5. Notification Mediator
Handles notification delivery and management.

```go
mediator := NewConcreteMediator(config)
notificationHandler := &NotificationHandler{ID: "handler1", Name: "Notification Handler"}

mediator.RegisterColleague(notificationHandler)

// Send notification
mediator.SendMessage("handler1", "handler1", &Notification{
    Type: "email",
    Content: "Welcome to our service!",
    RecipientID: "user123",
})
```

### 6. Workflow Mediator
Handles workflow execution and coordination.

```go
mediator := NewConcreteMediator(config)
workflowHandler := &WorkflowHandler{ID: "handler1", Name: "Workflow Handler"}

mediator.RegisterColleague(workflowHandler)

// Execute workflow
mediator.SendMessage("handler1", "handler1", &Workflow{
    Name: "user_registration",
    Steps: []WorkflowStep{
        {Name: "validate_input", Type: "validation"},
        {Name: "create_user", Type: "creation"},
        {Name: "send_welcome_email", Type: "notification"},
    },
})
```

### 7. Service Mediator
Handles service coordination and management.

```go
mediator := NewConcreteMediator(config)
serviceHandler := &ServiceHandler{ID: "handler1", Name: "Service Handler"}

mediator.RegisterColleague(serviceHandler)

// Coordinate service
mediator.SendMessage("handler1", "handler1", &Service{
    Name: "user_service",
    Type: "microservice",
    Status: "running",
})
```

### 8. Resource Mediator
Handles resource allocation and management.

```go
mediator := NewConcreteMediator(config)
resourceHandler := &ResourceHandler{ID: "handler1", Name: "Resource Handler"}

mediator.RegisterColleague(resourceHandler)

// Manage resource
mediator.SendMessage("handler1", "handler1", &Resource{
    Name: "database_connection",
    Type: "connection",
    Capacity: 100,
    Used: 50,
})
```

### 9. Task Mediator
Handles task execution and coordination.

```go
mediator := NewConcreteMediator(config)
taskHandler := &TaskHandler{ID: "handler1", Name: "Task Handler"}

mediator.RegisterColleague(taskHandler)

// Execute task
mediator.SendMessage("handler1", "handler1", &Task{
    Name: "process_payment",
    Type: "computation",
    Priority: 1,
})
```

### 10. Job Mediator
Handles job scheduling and execution.

```go
mediator := NewConcreteMediator(config)
jobHandler := &JobHandler{ID: "handler1", Name: "Job Handler"}

mediator.RegisterColleague(jobHandler)

// Schedule job
mediator.SendMessage("handler1", "handler1", &Job{
    Name: "daily_report",
    Type: "scheduled",
    Schedule: "0 0 * * *",
})
```

## API Endpoints

### Mediator Management

#### Create Mediator
```http
POST /api/v1/mediators
Content-Type: application/json

{
    "name": "my-mediator"
}
```

#### Get Mediator
```http
GET /api/v1/mediators/{name}
```

#### List Mediators
```http
GET /api/v1/mediators
```

#### Remove Mediator
```http
DELETE /api/v1/mediators/{name}
```

### Colleague Management

#### Register Colleague
```http
POST /api/v1/mediators/{name}/colleagues
Content-Type: application/json

{
    "id": "colleague1",
    "name": "Colleague 1",
    "type": "service"
}
```

#### Unregister Colleague
```http
DELETE /api/v1/mediators/{name}/colleagues/{colleagueID}
```

### Message Operations

#### Send Message
```http
POST /api/v1/mediators/{name}/messages
Content-Type: application/json

{
    "sender_id": "colleague1",
    "recipient_id": "colleague2",
    "message": "Hello from colleague 1"
}
```

#### Broadcast Message
```http
POST /api/v1/mediators/{name}/broadcast
Content-Type: application/json

{
    "sender_id": "colleague1",
    "message": "Hello everyone!"
}
```

### Health Check
```http
GET /health
```

## WebSocket Events

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Events
- `mediator_created`: When a new mediator is created
- `mediator_updated`: When a mediator is updated
- `mediator_deleted`: When a mediator is deleted
- `colleague_registered`: When a colleague is registered
- `colleague_unregistered`: When a colleague is unregistered
- `message_sent`: When a message is sent
- `message_broadcasted`: When a message is broadcasted

## Kafka Topics

- `mediator-events`: Mediator lifecycle events
- `mediator-commands`: Mediator operation commands
- `mediator-responses`: Mediator operation responses

## Configuration

The service is configured via `configs/config.yaml`:

```yaml
name: "Mediator Service"
version: "1.0.0"
max_mediators: 100
max_colleagues: 1000
timeout: "30m"
retry_count: 3

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "mediator_db"
  
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "mediator_db"
  
  redis:
    host: "localhost"
    port: 6379
    password: ""
    db: 0

cache:
  enabled: true
  type: "memory"
  ttl: "5m"
  max_size: 1000

message_queue:
  enabled: true
  brokers: ["localhost:9092"]
  topics: ["mediator-events"]

websocket:
  enabled: true
  port: 8080
  read_buffer_size: 1024
  write_buffer_size: 1024

security:
  enabled: true
  jwt_secret: "your-secret-key"
  token_expiry: "24h"
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: "1m"

monitoring:
  enabled: true
  port: 9090
  path: "/metrics"
  collect_interval: "30s"

logging:
  level: "info"
  format: "json"
  output: "stdout"
```

## Usage Examples

### Basic Message Mediator
```go
// Create mediator
config := &MediatorConfig{
    MaxMediators: 100,
    MaxColleagues: 1000,
    Timeout: 30 * time.Minute,
}
mediator := NewConcreteMediator(config)

// Create colleagues
colleague1 := &BaseColleague{
    ID:   "colleague1",
    Name: "Colleague 1",
    Type: "service",
    Active: true,
}
colleague2 := &BaseColleague{
    ID:   "colleague2",
    Name: "Colleague 2",
    Type: "service",
    Active: true,
}

// Register colleagues
mediator.RegisterColleague(colleague1)
mediator.RegisterColleague(colleague2)

// Send message
mediator.SendMessage("colleague1", "colleague2", "Hello from colleague 1")

// Broadcast message
mediator.BroadcastMessage("colleague1", "Hello everyone!")
```

### Event-Driven Mediator
```go
// Create event mediator
mediator := NewConcreteMediator(config)

// Create event handlers
userHandler := &EventHandler{
    ID:   "user_handler",
    Name: "User Event Handler",
    Type: "event",
    Active: true,
}
emailHandler := &EventHandler{
    ID:   "email_handler",
    Name: "Email Event Handler",
    Type: "event",
    Active: true,
}

// Register handlers
mediator.RegisterColleague(userHandler)
mediator.RegisterColleague(emailHandler)

// Broadcast user created event
event := &BaseEvent{
    ID:   "event1",
    Type: "user_created",
    Data: map[string]interface{}{
        "user_id": "123",
        "name":    "John Doe",
        "email":   "john@example.com",
    },
    Timestamp: time.Now(),
    Source:    "user_service",
}

mediator.BroadcastMessage("user_handler", event)
```

### Command Mediator
```go
// Create command mediator
mediator := NewConcreteMediator(config)

// Create command handlers
userCommandHandler := &CommandHandler{
    ID:   "user_command_handler",
    Name: "User Command Handler",
    Type: "command",
    Active: true,
}

// Register handler
mediator.RegisterColleague(userCommandHandler)

// Execute create user command
command := &BaseCommand{
    ID:   "cmd1",
    Type: "create_user",
    Data: map[string]interface{}{
        "name":  "John Doe",
        "email": "john@example.com",
    },
    Timestamp: time.Now(),
    Source:    "api",
}

mediator.SendMessage("user_command_handler", "user_command_handler", command)
```

### Workflow Mediator
```go
// Create workflow mediator
mediator := NewConcreteMediator(config)

// Create workflow handler
workflowHandler := &WorkflowHandler{
    ID:   "workflow_handler",
    Name: "Workflow Handler",
    Type: "workflow",
    Active: true,
}

// Register handler
mediator.RegisterColleague(workflowHandler)

// Execute workflow
workflow := &BaseWorkflow{
    ID:   "workflow1",
    Name: "user_registration",
    Steps: []WorkflowStep{
        &BaseWorkflowStep{
            ID:   "step1",
            Name: "validate_input",
            Type: "validation",
            Status: "pending",
        },
        &BaseWorkflowStep{
            ID:   "step2",
            Name: "create_user",
            Type: "creation",
            Status: "pending",
        },
        &BaseWorkflowStep{
            ID:   "step3",
            Name: "send_welcome_email",
            Type: "notification",
            Status: "pending",
        },
    },
    CurrentStep: 0,
    Status:      "running",
    Data: map[string]interface{}{
        "name":  "John Doe",
        "email": "john@example.com",
    },
}

mediator.SendMessage("workflow_handler", "workflow_handler", workflow)
```

## Testing

### Unit Tests
```bash
go test ./internal/mediator/...
```

### Integration Tests
```bash
go test -tags=integration ./...
```

### Benchmark Tests
```bash
go test -bench=. ./internal/mediator/...
```

## Performance Considerations

1. **Memory Usage**: Mediators hold references to colleagues, so large numbers of colleagues should be managed carefully
2. **Concurrency**: Use thread-safe implementations for concurrent access
3. **Caching**: Enable caching for frequently accessed mediators and colleagues
4. **Message Processing**: Process messages asynchronously to avoid blocking
5. **Resource Management**: Implement proper resource cleanup and lifecycle management

## Security

- JWT authentication for API access
- Rate limiting to prevent abuse
- CORS configuration for web clients
- Input validation for all endpoints
- Audit logging for compliance

## Monitoring

- Health checks for all components
- Metrics collection and export
- Performance monitoring
- Error tracking and alerting
- Resource usage monitoring

## Deployment

### Docker
```bash
docker build -t mediator-service .
docker run -p 8080:8080 mediator-service
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Environment Variables
```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USERNAME=root
export MYSQL_PASSWORD=password
export MYSQL_DATABASE=mediator_db
export MONGODB_URI=mongodb://localhost:27017
export MONGODB_DATABASE=mediator_db
export REDIS_HOST=localhost
export REDIS_PORT=6379
export KAFKA_BROKERS=localhost:9092
export JWT_SECRET=your-secret-key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the examples

## Changelog

### v1.0.0
- Initial release
- Basic mediator types
- REST API
- WebSocket support
- Kafka integration
- Database support
- Caching
- Monitoring
- Security features
