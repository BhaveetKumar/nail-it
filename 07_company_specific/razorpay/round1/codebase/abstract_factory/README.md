---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.587386
Tags: []
Status: draft
---

# Abstract Factory Pattern Implementation

This microservice demonstrates the Abstract Factory pattern, which provides an interface for creating families of related objects without specifying their concrete classes. This pattern is particularly useful when you need to create objects that belong to the same family but have different implementations.

## Architecture Overview

The Abstract Factory pattern provides:

- **Abstract Factory Interface**: Defines methods for creating products
- **Concrete Factories**: Implement the abstract factory interface
- **Abstract Products**: Define interfaces for products
- **Concrete Products**: Implement the abstract product interfaces
- **Client Code**: Uses the abstract factory to create products

### Key Components

1. **Abstract Factory**: Interface for creating families of related objects
2. **Concrete Factories**: Specific implementations of the abstract factory
3. **Abstract Products**: Interfaces for products
4. **Concrete Products**: Specific implementations of products
5. **Factory Registry**: Manages factory registration and retrieval
6. **Product Registry**: Manages product registration and retrieval

## Features

- **Factory Management**: Create, destroy, and manage factories
- **Product Creation**: Create products using registered factories
- **Product Management**: Manage product lifecycle and statistics
- **Registry Management**: Register and unregister factories and products
- **Validation**: Input validation for factories and products
- **Caching**: Optional caching for products
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
abstract_factory/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── abstract_factory/
│   │   ├── interfaces.go
│   │   ├── models.go
│   │   ├── abstract_factory.go
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
cd abstract_factory
```

2. Install dependencies:

```bash
go mod tidy
```

3. Set up the databases:

```bash
# MySQL
mysql -u root -p -e "CREATE DATABASE abstract_factory_db;"

# MongoDB
mongosh --eval "use abstract_factory_db"
```

4. Start Kafka:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic abstract-factory-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

5. Update configuration in `configs/config.yaml`:

```yaml
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "your_username"
    password: "your_password"
    database: "abstract_factory_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "abstract_factory_db"
kafka:
  brokers:
    - "localhost:9092"
  group_id: "abstract-factory-group"
  topic: "abstract-factory-events"
```

6. Run the service:

```bash
go run main.go
```

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Factory Management

- `POST /api/v1/abstract-factory/factories` - Create a factory
- `DELETE /api/v1/abstract-factory/factories/:type` - Destroy a factory
- `GET /api/v1/abstract-factory/factories/:type` - Get factory by type
- `GET /api/v1/abstract-factory/factories` - List all factories
- `GET /api/v1/abstract-factory/factories/:type/stats` - Get factory statistics
- `GET /api/v1/abstract-factory/factories/stats` - Get all factory statistics
- `PUT /api/v1/abstract-factory/factories/:type/active` - Set factory active status

### Product Management

- `POST /api/v1/abstract-factory/products` - Create a product
- `DELETE /api/v1/abstract-factory/products/:id` - Destroy a product
- `GET /api/v1/abstract-factory/products/:id` - Get product by ID
- `GET /api/v1/abstract-factory/products` - List all products
- `GET /api/v1/abstract-factory/products/:id/stats` - Get product statistics
- `GET /api/v1/abstract-factory/products/stats` - Get all product statistics
- `PUT /api/v1/abstract-factory/products/:id/active` - Set product active status

### Service Management

- `GET /api/v1/abstract-factory/stats` - Get service statistics
- `POST /api/v1/abstract-factory/cleanup` - Perform cleanup
- `GET /api/v1/abstract-factory/health` - Get detailed health status

### WebSocket

- `GET /ws` - WebSocket connection for real-time updates

## Usage Examples

### Creating a Factory

```bash
curl -X POST http://localhost:8080/api/v1/abstract-factory/factories \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ConcreteFactory1",
    "name": "Payment Factory",
    "description": "Factory for creating payment-related products",
    "version": "1.0.0",
    "metadata": {
      "category": "payment",
      "region": "us-east-1"
    }
  }'
```

### Creating a Product

```bash
curl -X POST http://localhost:8080/api/v1/abstract-factory/products \
  -H "Content-Type: application/json" \
  -d '{
    "factory_type": "ConcreteFactory1",
    "product_type": "ProductA",
    "config": {
      "name": "Credit Card Payment",
      "description": "Credit card payment product",
      "price": 100.0
    }
  }'
```

### Getting Factory Statistics

```bash
curl -X GET http://localhost:8080/api/v1/abstract-factory/factories/ConcreteFactory1/stats
```

### Getting Product Statistics

```bash
curl -X GET http://localhost:8080/api/v1/abstract-factory/products/product-a-1/stats
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
    database: "abstract_factory_db"
    max_idle_conns: 10
    max_open_conns: 100
    conn_max_lifetime: 3600
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "abstract_factory_db"
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time: 30s

kafka:
  brokers:
    - "localhost:9092"
  group_id: "abstract-factory-group"
  topic: "abstract-factory-events"
  consumer_timeout: 10s
  producer_timeout: 10s

abstract_factory:
  name: "Abstract Factory Service"
  version: "1.0.0"
  description: "Abstract Factory pattern service for creating families of related objects"
  max_factories: 100
  max_products: 1000
  cleanup_interval: "1h"
  validation_enabled: true
  caching_enabled: true
  monitoring_enabled: true
  auditing_enabled: true
  supported_factory_types:
    - "ConcreteFactory1"
    - "ConcreteFactory2"
    - "PaymentFactory"
    - "OrderFactory"
    - "UserFactory"
    - "NotificationFactory"
  supported_product_types:
    - "ProductA"
    - "ProductB"
    - "ProductC"
    - "PaymentProduct"
    - "OrderProduct"
    - "UserProduct"
    - "NotificationProduct"
  validation_rules:
    factory:
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
    product:
      name:
        required: true
        min_length: 2
        max_length: 100
      description:
        required: true
        min_length: 10
        max_length: 500
      price:
        required: true
        min: 0.01
        max: 1000000
  metadata:
    environment: "development"
    region: "us-east-1"
    team: "backend"
    project: "abstract-factory-service"
```

## Testing

Run the tests:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestAbstractFactory ./internal/abstract_factory/

# Run benchmarks
go test -bench=. ./...
```

## Monitoring

The service provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Detailed Health**: `GET /api/v1/abstract-factory/health` - Detailed health status with component checks
- **Service Stats**: `GET /api/v1/abstract-factory/stats` - Service statistics and metrics

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
- **Caching**: Optional product caching
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
  abstract-factory-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=abstract_factory_db
      - MONGODB_URI=mongodb://mongodb:27017
      - MONGODB_DB=abstract_factory_db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - mysql
      - mongodb
      - kafka

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: abstract_factory_db
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
- Basic Abstract Factory implementation
- Factory and Product management
- Registry management
- WebSocket support
- Kafka integration
- MySQL and MongoDB support
- Health checks and monitoring
- Comprehensive testing
- Docker support
