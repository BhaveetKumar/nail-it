---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.567024
Tags: []
Status: draft
---

# Saga Pattern Implementation

This microservice demonstrates the Saga pattern, which is used to manage distributed transactions across multiple services. The Saga pattern ensures data consistency in distributed systems by breaking down complex transactions into a series of local transactions, each with its own compensation action.

## Architecture Overview

The Saga pattern provides:

- **Saga**: A sequence of local transactions that can be executed or compensated
- **Saga Step**: Individual local transaction with its own compensation
- **Saga Manager**: Manages saga lifecycle and state
- **Saga Executor**: Executes sagas and handles compensation
- **Step Executor**: Executes individual saga steps

### Key Components

1. **Saga Interface**: Defines the contract for sagas
2. **Saga Step Interface**: Defines the contract for saga steps
3. **Saga Manager**: Manages saga registration and retrieval
4. **Saga Executor**: Executes sagas and handles compensation
5. **Step Executor**: Executes individual steps
6. **Saga Service**: High-level service for saga operations

## Features

- **Saga Management**: Create, destroy, and manage sagas
- **Step Management**: Add, remove, and manage saga steps
- **Execution**: Execute sagas with automatic compensation on failure
- **Compensation**: Compensate failed sagas to maintain consistency
- **Retry Logic**: Configurable retry logic for failed steps
- **Timeout Handling**: Configurable timeouts for steps
- **Validation**: Input validation for sagas and steps
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
saga/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── saga/
│   │   ├── interfaces.go
│   │   ├── models.go
│   │   ├── saga.go
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
cd saga
```

2. Install dependencies:

```bash
go mod tidy
```

3. Set up the databases:

```bash
# MySQL
mysql -u root -p -e "CREATE DATABASE saga_db;"

# MongoDB
mongosh --eval "use saga_db"
```

4. Start Kafka:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic saga-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

5. Update configuration in `configs/config.yaml`:

```yaml
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "your_username"
    password: "your_password"
    database: "saga_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "saga_db"
kafka:
  brokers:
    - "localhost:9092"
  group_id: "saga-group"
  topic: "saga-events"
```

6. Run the service:

```bash
go run main.go
```

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Saga Management

- `POST /api/v1/saga/sagas` - Create a saga
- `DELETE /api/v1/saga/sagas/:id` - Destroy a saga
- `GET /api/v1/saga/sagas/:id` - Get saga by ID
- `GET /api/v1/saga/sagas` - List all sagas
- `GET /api/v1/saga/sagas/:id/stats` - Get saga statistics
- `GET /api/v1/saga/sagas/stats` - Get all saga statistics
- `PUT /api/v1/saga/sagas/:id/active` - Set saga active status

### Saga Execution

- `POST /api/v1/saga/sagas/:id/execute` - Execute a saga
- `POST /api/v1/saga/sagas/:id/compensate` - Compensate a saga
- `GET /api/v1/saga/sagas/:id/status` - Get saga status

### Service Management

- `GET /api/v1/saga/stats` - Get service statistics
- `POST /api/v1/saga/cleanup` - Perform cleanup
- `GET /api/v1/saga/health` - Get detailed health status

### WebSocket

- `GET /ws` - WebSocket connection for real-time updates

## Usage Examples

### Creating a Saga

```bash
curl -X POST http://localhost:8080/api/v1/saga/sagas \
  -H "Content-Type: application/json" \
  -d '{
    "id": "payment-saga",
    "name": "Payment Processing Saga",
    "description": "Saga for processing payments with compensation",
    "version": "1.0.0",
    "steps": [
      {
        "id": "validate-payment",
        "name": "Validate Payment",
        "description": "Validate payment details",
        "order": 1,
        "action": "validate_payment",
        "compensation": "cancel_payment",
        "max_retries": 3,
        "timeout": "30s",
        "active": true
      },
      {
        "id": "process-payment",
        "name": "Process Payment",
        "description": "Process the payment",
        "order": 2,
        "action": "process_payment",
        "compensation": "refund_payment",
        "max_retries": 3,
        "timeout": "60s",
        "active": true
      },
      {
        "id": "send-notification",
        "name": "Send Notification",
        "description": "Send payment confirmation",
        "order": 3,
        "action": "send_notification",
        "compensation": "cancel_notification",
        "max_retries": 2,
        "timeout": "10s",
        "active": true
      }
    ],
    "active": true
  }'
```

### Executing a Saga

```bash
curl -X POST http://localhost:8080/api/v1/saga/sagas/payment-saga/execute
```

### Compensating a Saga

```bash
curl -X POST http://localhost:8080/api/v1/saga/sagas/payment-saga/compensate
```

### Getting Saga Status

```bash
curl -X GET http://localhost:8080/api/v1/saga/sagas/payment-saga/status
```

### Getting Saga Statistics

```bash
curl -X GET http://localhost:8080/api/v1/saga/sagas/payment-saga/stats
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
    database: "saga_db"
    max_idle_conns: 10
    max_open_conns: 100
    conn_max_lifetime: 3600
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "saga_db"
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time: 30s

kafka:
  brokers:
    - "localhost:9092"
  group_id: "saga-group"
  topic: "saga-events"
  consumer_timeout: 10s
  producer_timeout: 10s

saga:
  name: "Saga Service"
  version: "1.0.0"
  description: "Saga pattern service for managing distributed transactions"
  max_sagas: 1000
  max_steps: 10000
  cleanup_interval: "1h"
  validation_enabled: true
  caching_enabled: true
  monitoring_enabled: true
  auditing_enabled: true
  retry_enabled: true
  compensation_enabled: true
  supported_saga_types:
    - "PaymentSaga"
    - "OrderSaga"
    - "UserSaga"
    - "NotificationSaga"
    - "InventorySaga"
    - "ShippingSaga"
  supported_step_types:
    - "PaymentStep"
    - "OrderStep"
    - "UserStep"
    - "NotificationStep"
    - "InventoryStep"
    - "ShippingStep"
  validation_rules:
    saga:
      id:
        required: true
        min_length: 2
        max_length: 100
      name:
        required: true
        min_length: 2
        max_length: 100
      description:
        required: true
        min_length: 10
        max_length: 500
      version:
        required: true
        format: "semver"
    step:
      id:
        required: true
        min_length: 2
        max_length: 100
      name:
        required: true
        min_length: 2
        max_length: 100
      description:
        required: true
        min_length: 10
        max_length: 500
      action:
        required: true
        min_length: 2
        max_length: 100
      compensation:
        required: true
        min_length: 2
        max_length: 100
      max_retries:
        required: true
        min: 0
        max: 10
      timeout:
        required: true
        min: "1s"
        max: "1h"
  metadata:
    environment: "development"
    region: "us-east-1"
    team: "backend"
    project: "saga-service"
```

## Testing

Run the tests:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestSaga ./internal/saga/

# Run benchmarks
go test -bench=. ./...
```

## Monitoring

The service provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Detailed Health**: `GET /api/v1/saga/health` - Detailed health status with component checks
- **Service Stats**: `GET /api/v1/saga/stats` - Service statistics and metrics

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
- **Retry Exhaustion**: When retry attempts are exhausted
- **Compensation Errors**: When compensation fails

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
- **Caching**: Optional saga caching
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
  saga-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=saga_db
      - MONGODB_URI=mongodb://mongodb:27017
      - MONGODB_DB=saga_db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - mysql
      - mongodb
      - kafka

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: saga_db
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
- Basic Saga pattern implementation
- Saga and Step management
- Execution and compensation
- Retry logic and timeout handling
- WebSocket support
- Kafka integration
- MySQL and MongoDB support
- Health checks and monitoring
- Comprehensive testing
- Docker support
