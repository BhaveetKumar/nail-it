# Circuit Breaker Pattern Implementation

This microservice demonstrates the Circuit Breaker pattern, which is used to prevent cascading failures in distributed systems. The Circuit Breaker pattern monitors the success/failure rate of operations and opens the circuit when the failure rate exceeds a threshold, preventing further calls to the failing service.

## Architecture Overview

The Circuit Breaker pattern provides:

- **Circuit Breaker**: Monitors operations and controls the flow of requests
- **Circuit States**: Closed, Open, and Half-Open states
- **Circuit Manager**: Manages circuit breaker lifecycle and state
- **Circuit Executor**: Executes operations through circuit breakers
- **Monitoring**: Tracks circuit breaker statistics and health

### Key Components

1. **Circuit Breaker Interface**: Defines the contract for circuit breakers
2. **Circuit States**: Closed (normal), Open (failing), Half-Open (testing)
3. **Circuit Manager**: Manages circuit breaker registration and retrieval
4. **Circuit Executor**: Executes operations through circuit breakers
5. **Circuit Service**: High-level service for circuit breaker operations

## Features

- **Circuit Management**: Create, destroy, and manage circuit breakers
- **State Management**: Automatic state transitions based on failure rates
- **Threshold Configuration**: Configurable failure and success thresholds
- **Timeout Handling**: Configurable timeouts for operations
- **Slow Call Detection**: Detection and handling of slow operations
- **Async Execution**: Asynchronous operation execution
- **Statistics Tracking**: Comprehensive statistics and monitoring
- **Health Monitoring**: Health checks and service statistics
- **Validation**: Input validation for circuit breaker configurations

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
circuit_breaker/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── circuit_breaker/
│   │   ├── interfaces.go
│   │   ├── models.go
│   │   ├── circuit_breaker.go
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
cd circuit_breaker
```

2. Install dependencies:

```bash
go mod tidy
```

3. Set up the databases:

```bash
# MySQL
mysql -u root -p -e "CREATE DATABASE circuit_breaker_db;"

# MongoDB
mongosh --eval "use circuit_breaker_db"
```

4. Start Kafka:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic circuit-breaker-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

5. Update configuration in `configs/config.yaml`:

```yaml
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "your_username"
    password: "your_password"
    database: "circuit_breaker_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "circuit_breaker_db"
kafka:
  brokers:
    - "localhost:9092"
  group_id: "circuit-breaker-group"
  topic: "circuit-breaker-events"
```

6. Run the service:

```bash
go run main.go
```

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Circuit Breaker Management

- `POST /api/v1/circuit-breaker/circuit-breakers` - Create a circuit breaker
- `DELETE /api/v1/circuit-breaker/circuit-breakers/:id` - Destroy a circuit breaker
- `GET /api/v1/circuit-breaker/circuit-breakers/:id` - Get circuit breaker by ID
- `GET /api/v1/circuit-breaker/circuit-breakers` - List all circuit breakers
- `GET /api/v1/circuit-breaker/circuit-breakers/:id/stats` - Get circuit breaker statistics
- `GET /api/v1/circuit-breaker/circuit-breakers/stats` - Get all circuit breaker statistics
- `PUT /api/v1/circuit-breaker/circuit-breakers/:id/active` - Set circuit breaker active status

### Circuit Breaker Execution

- `POST /api/v1/circuit-breaker/circuit-breakers/:id/execute` - Execute operation through circuit breaker
- `POST /api/v1/circuit-breaker/circuit-breakers/:id/execute-async` - Execute operation asynchronously

### Service Management

- `GET /api/v1/circuit-breaker/stats` - Get service statistics
- `POST /api/v1/circuit-breaker/cleanup` - Perform cleanup
- `GET /api/v1/circuit-breaker/health` - Get detailed health status

### WebSocket

- `GET /ws` - WebSocket connection for real-time updates

## Usage Examples

### Creating a Circuit Breaker

```bash
curl -X POST http://localhost:8080/api/v1/circuit-breaker/circuit-breakers \
  -H "Content-Type: application/json" \
  -d '{
    "id": "payment-service-cb",
    "name": "Payment Service Circuit Breaker",
    "description": "Circuit breaker for payment service operations",
    "failure_threshold": 5,
    "success_threshold": 3,
    "timeout": "30s",
    "reset_timeout": "60s",
    "max_requests": 10,
    "request_volume_threshold": 10,
    "sleep_window": "60s",
    "error_threshold": 0.5,
    "slow_call_threshold": "5s",
    "slow_call_ratio_threshold": 0.5,
    "active": true
  }'
```

### Executing an Operation

```bash
curl -X POST http://localhost:8080/api/v1/circuit-breaker/circuit-breakers/payment-service-cb/execute \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "process_payment",
    "data": {
      "amount": 100.0,
      "currency": "USD",
      "card_number": "4111111111111111"
    }
  }'
```

### Getting Circuit Breaker Statistics

```bash
curl -X GET http://localhost:8080/api/v1/circuit-breaker/circuit-breakers/payment-service-cb/stats
```

### Getting All Circuit Breaker Statistics

```bash
curl -X GET http://localhost:8080/api/v1/circuit-breaker/circuit-breakers/stats
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
    database: "circuit_breaker_db"
    max_idle_conns: 10
    max_open_conns: 100
    conn_max_lifetime: 3600
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "circuit_breaker_db"
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time: 30s

kafka:
  brokers:
    - "localhost:9092"
  group_id: "circuit-breaker-group"
  topic: "circuit-breaker-events"
  consumer_timeout: 10s
  producer_timeout: 10s

circuit_breaker:
  name: "Circuit Breaker Service"
  version: "1.0.0"
  description: "Circuit Breaker pattern service for fault tolerance and resilience"
  max_circuit_breakers: 1000
  cleanup_interval: "1h"
  validation_enabled: true
  caching_enabled: true
  monitoring_enabled: true
  auditing_enabled: true
  default_failure_threshold: 5
  default_success_threshold: 3
  default_timeout: "30s"
  default_reset_timeout: "60s"
  supported_types:
    - "PaymentCircuitBreaker"
    - "OrderCircuitBreaker"
    - "UserCircuitBreaker"
    - "NotificationCircuitBreaker"
    - "InventoryCircuitBreaker"
    - "ShippingCircuitBreaker"
  validation_rules:
    circuit_breaker:
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
      failure_threshold:
        required: true
        min: 1
        max: 100
      success_threshold:
        required: true
        min: 1
        max: 100
      timeout:
        required: true
        min: "1s"
        max: "1h"
      reset_timeout:
        required: true
        min: "1s"
        max: "1h"
      max_requests:
        required: true
        min: 1
        max: 1000
      request_volume_threshold:
        required: true
        min: 1
        max: 1000
      sleep_window:
        required: true
        min: "1s"
        max: "1h"
      error_threshold:
        required: true
        min: 0.0
        max: 1.0
      slow_call_threshold:
        required: true
        min: "1s"
        max: "1h"
      slow_call_ratio_threshold:
        required: true
        min: 0.0
        max: 1.0
  metadata:
    environment: "development"
    region: "us-east-1"
    team: "backend"
    project: "circuit-breaker-service"
```

## Testing

Run the tests:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestCircuitBreaker ./internal/circuit_breaker/

# Run benchmarks
go test -bench=. ./...
```

## Monitoring

The service provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Detailed Health**: `GET /api/v1/circuit-breaker/health` - Detailed health status with component checks
- **Service Stats**: `GET /api/v1/circuit-breaker/stats` - Service statistics and metrics

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
- **Circuit Breaker Errors**: Circuit open, half-open, or inactive errors
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
- **Caching**: Optional circuit breaker caching
- **Async Processing**: Asynchronous operation execution
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
  circuit-breaker-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=circuit_breaker_db
      - MONGODB_URI=mongodb://mongodb:27017
      - MONGODB_DB=circuit_breaker_db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - mysql
      - mongodb
      - kafka

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: circuit_breaker_db
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
- Basic Circuit Breaker pattern implementation
- Circuit Breaker management
- State management and transitions
- Threshold configuration
- Async execution support
- WebSocket support
- Kafka integration
- MySQL and MongoDB support
- Health checks and monitoring
- Comprehensive testing
- Docker support
