# CQRS (Command Query Responsibility Segregation) Pattern Implementation

This microservice demonstrates the CQRS (Command Query Responsibility Segregation) pattern, which separates the read and write operations of a data store. This pattern is particularly useful in complex systems where read and write operations have different performance requirements, scalability needs, or data consistency requirements.

## Architecture Overview

The CQRS pattern separates the system into two distinct parts:

- **Command Side**: Handles write operations (create, update, delete)
- **Query Side**: Handles read operations (get, list, search)

### Key Components

1. **Command Bus**: Routes commands to appropriate handlers
2. **Query Bus**: Routes queries to appropriate handlers
3. **Event Bus**: Publishes and subscribes to domain events
4. **Read Model Store**: Manages read-optimized data models
5. **Write Model Store**: Manages write-optimized data models

## Features

- **Command Handling**: Process write operations through command handlers
- **Query Handling**: Process read operations through query handlers
- **Event Sourcing**: Track all changes as a sequence of events
- **Read Model Projection**: Maintain read-optimized views of data
- **Event Publishing**: Publish domain events for other services
- **Handler Registration**: Dynamic registration of command/query/event handlers
- **Validation**: Input validation for commands and queries
- **Caching**: Optional caching for read models
- **Monitoring**: Health checks and service statistics
- **Auditing**: Track all operations for compliance

## Technology Stack

- **Language**: Go 1.21+
- **Framework**: Gin (HTTP router)
- **Database**: MySQL (write model), MongoDB (read model)
- **Message Queue**: Apache Kafka
- **WebSocket**: Gorilla WebSocket
- **Logging**: Zap
- **Configuration**: Viper
- **Testing**: Testify

## Project Structure

```
cqrs/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── cqrs/
│   │   ├── interfaces.go
│   │   ├── models.go
│   │   ├── cqrs.go
│   │   └── service.go
│   ├── handlers/
│   │   ├── handlers.go
│   │   └── middleware.go
│   ├── kafka/
│   │   ├── producer.go
│   │   └── consumer.go
│   └── websocket/
│       ├── hub.go
│       └── client.go
├── configs/
│   └── config.yaml
├── go.mod
├── go.sum
└── README.md
```

## Getting Started

### Prerequisites

- Go 1.21 or higher
- MySQL 8.0 or higher
- MongoDB 4.4 or higher
- Apache Kafka 2.8 or higher

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd cqrs
```

2. Install dependencies:

```bash
go mod tidy
```

3. Set up the databases:

```bash
# MySQL
mysql -u root -p -e "CREATE DATABASE cqrs_db;"

# MongoDB
mongosh --eval "use cqrs_db"
```

4. Start Kafka:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic cqrs-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

5. Update configuration in `configs/config.yaml`:

```yaml
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "your_username"
    password: "your_password"
    database: "cqrs_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "cqrs_db"
kafka:
  brokers:
    - "localhost:9092"
  group_id: "cqrs-group"
  topic: "cqrs-events"
```

6. Run the service:

```bash
go run main.go
```

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Commands

- `POST /api/v1/cqrs/commands` - Send a command
- `GET /api/v1/cqrs/commands/:id` - Get command by ID
- `GET /api/v1/cqrs/commands` - List commands

### Queries

- `POST /api/v1/cqrs/queries` - Send a query
- `GET /api/v1/cqrs/queries/:id` - Get query by ID
- `GET /api/v1/cqrs/queries` - List queries

### Events

- `POST /api/v1/cqrs/events` - Publish an event
- `GET /api/v1/cqrs/events/:id` - Get event by ID
- `GET /api/v1/cqrs/events` - List events

### Read Models

- `POST /api/v1/cqrs/read-models` - Save a read model
- `GET /api/v1/cqrs/read-models/:id` - Get read model by ID
- `GET /api/v1/cqrs/read-models` - List read models
- `DELETE /api/v1/cqrs/read-models/:id` - Delete read model

### Handler Management

- `POST /api/v1/cqrs/handlers/commands` - Register command handler
- `POST /api/v1/cqrs/handlers/queries` - Register query handler
- `POST /api/v1/cqrs/handlers/events` - Register event handler
- `DELETE /api/v1/cqrs/handlers/commands/:type` - Unregister command handler
- `DELETE /api/v1/cqrs/handlers/queries/:type` - Unregister query handler
- `DELETE /api/v1/cqrs/handlers/events/:type` - Unregister event handler

### Service Management

- `GET /api/v1/cqrs/stats` - Get service statistics
- `POST /api/v1/cqrs/cleanup` - Perform cleanup
- `GET /api/v1/cqrs/health` - Get detailed health status

### WebSocket

- `GET /ws` - WebSocket connection for real-time updates

## Usage Examples

### Sending a Command

```bash
curl -X POST http://localhost:8080/api/v1/cqrs/commands \
  -H "Content-Type: application/json" \
  -d '{
    "type": "CreateUser",
    "data": {
      "name": "John Doe",
      "email": "john@example.com",
      "age": 30
    }
  }'
```

### Sending a Query

```bash
curl -X POST http://localhost:8080/api/v1/cqrs/queries \
  -H "Content-Type: application/json" \
  -d '{
    "type": "GetUser",
    "data": {
      "id": "user-123"
    }
  }'
```

### Publishing an Event

```bash
curl -X POST http://localhost:8080/api/v1/cqrs/events \
  -H "Content-Type: application/json" \
  -d '{
    "type": "UserCreated",
    "data": {
      "user_id": "user-123",
      "name": "John Doe",
      "email": "john@example.com",
      "created_at": "2023-01-01T00:00:00Z"
    }
  }'
```

### Saving a Read Model

```bash
curl -X POST http://localhost:8080/api/v1/cqrs/read-models \
  -H "Content-Type: application/json" \
  -d '{
    "type": "UserReadModel",
    "data": {
      "id": "user-123",
      "name": "John Doe",
      "email": "john@example.com",
      "age": 30,
      "created_at": "2023-01-01T00:00:00Z",
      "updated_at": "2023-01-01T00:00:00Z"
    }
  }'
```

## Configuration

The service can be configured through the `configs/config.yaml` file:

```yaml
server:
  port: ":8080"
  read_timeout: 30s
  write_timeout: 30s
  idle_timeout: 120s

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "cqrs_db"
    max_idle_conns: 10
    max_open_conns: 100
    conn_max_lifetime: 3600
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "cqrs_db"
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time: 30s

kafka:
  brokers:
    - "localhost:9092"
  group_id: "cqrs-group"
  topic: "cqrs-events"
  consumer_timeout: 10s
  producer_timeout: 10s

cqrs:
  name: "CQRS Service"
  version: "1.0.0"
  description: "CQRS service for handling commands, queries, and events"
  max_commands: 1000
  max_queries: 1000
  max_events: 10000
  max_read_models: 10000
  cleanup_interval: "1h"
  validation_enabled: true
  caching_enabled: true
  monitoring_enabled: true
  auditing_enabled: true
  supported_command_types:
    - "CreateUser"
    - "UpdateUser"
    - "DeleteUser"
    - "CreateOrder"
    - "UpdateOrder"
    - "CancelOrder"
    - "ProcessPayment"
    - "RefundPayment"
  supported_query_types:
    - "GetUser"
    - "ListUsers"
    - "GetOrder"
    - "ListOrders"
    - "GetPayment"
    - "ListPayments"
    - "GetUserOrders"
    - "GetOrderPayments"
  supported_event_types:
    - "UserCreated"
    - "UserUpdated"
    - "UserDeleted"
    - "OrderCreated"
    - "OrderUpdated"
    - "OrderCancelled"
    - "PaymentProcessed"
    - "PaymentRefunded"
  supported_read_model_types:
    - "UserReadModel"
    - "OrderReadModel"
    - "PaymentReadModel"
    - "UserOrderReadModel"
    - "OrderPaymentReadModel"
  validation_rules:
    user:
      name:
        required: true
        min_length: 2
        max_length: 100
      email:
        required: true
        format: "email"
      age:
        required: true
        min: 18
        max: 120
    order:
      user_id:
        required: true
        format: "uuid"
      amount:
        required: true
        min: 0.01
      currency:
        required: true
        enum: ["USD", "EUR", "GBP", "INR"]
    payment:
      order_id:
        required: true
        format: "uuid"
      amount:
        required: true
        min: 0.01
      currency:
        required: true
        enum: ["USD", "EUR", "GBP", "INR"]
      method:
        required: true
        enum: ["credit_card", "debit_card", "net_banking", "upi", "wallet"]
  metadata:
    environment: "development"
    region: "us-east-1"
    team: "backend"
    project: "cqrs-service"
```

## Testing

Run the tests:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestCQRSService ./internal/cqrs/

# Run benchmarks
go test -bench=. ./...
```

## Monitoring

The service provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Detailed Health**: `GET /api/v1/cqrs/health` - Detailed health status with component checks
- **Service Stats**: `GET /api/v1/cqrs/stats` - Service statistics and metrics

## WebSocket Support

The service supports WebSocket connections for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8080/ws");

ws.onopen = function (event) {
  console.log("Connected to WebSocket");
};

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Received:", data);
};

ws.onclose = function (event) {
  console.log("WebSocket connection closed");
};
```

## Error Handling

The service provides comprehensive error handling:

- **Validation Errors**: Input validation failures
- **Business Logic Errors**: Domain-specific errors
- **Infrastructure Errors**: Database, Kafka, or other infrastructure failures
- **Timeout Errors**: Request timeout handling
- **Rate Limiting**: Request rate limiting

## Security

Security features include:

- **Input Validation**: All inputs are validated
- **SQL Injection Prevention**: Parameterized queries
- **NoSQL Injection Prevention**: Proper MongoDB query construction
- **Rate Limiting**: Request rate limiting
- **CORS**: Cross-origin resource sharing configuration
- **Authentication**: JWT token validation (optional)
- **Authorization**: Role-based access control (optional)

## Performance

Performance optimizations:

- **Connection Pooling**: Database connection pooling
- **Caching**: Optional read model caching
- **Async Processing**: Asynchronous event processing
- **Batch Operations**: Batch database operations
- **Indexing**: Database indexing for queries
- **Compression**: Response compression

## Scalability

Scalability features:

- **Horizontal Scaling**: Stateless service design
- **Load Balancing**: Support for load balancers
- **Database Sharding**: Support for database sharding
- **Event Partitioning**: Kafka topic partitioning
- **Read Replicas**: Database read replicas
- **Caching**: Distributed caching support

## Deployment

### Docker

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .
COPY --from=builder /app/configs ./configs

CMD ["./main"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  cqrs-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=cqrs_db
      - MONGODB_URI=mongodb://mongodb:27017
      - MONGODB_DB=cqrs_db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - mysql
      - mongodb
      - kafka

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: cqrs_db
    ports:
      - "3306:3306"

  mongodb:
    image: mongo:4.4
    ports:
      - "27017:27017"

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the repository or contact the development team.

## Changelog

### v1.0.0

- Initial release
- Basic CQRS implementation
- Command, Query, and Event handling
- Read model management
- WebSocket support
- Kafka integration
- MySQL and MongoDB support
- Health checks and monitoring
- Comprehensive testing
- Docker support
