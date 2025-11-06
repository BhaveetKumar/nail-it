---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.598486
Tags: []
Status: draft
---

# Builder Pattern Implementation

This microservice demonstrates the Builder pattern, which provides a way to construct complex objects step by step. This pattern is particularly useful when you need to create objects with many optional parameters or when the construction process is complex.

## Architecture Overview

The Builder pattern provides:

- **Builder Interface**: Defines methods for building parts of the product
- **Concrete Builder**: Implements the builder interface and constructs the product
- **Product**: The complex object being built
- **Director**: Optional class that orchestrates the building process
- **Client**: Uses the builder to create products

### Key Components

1. **Builder Interface**: Defines methods for building parts of the product
2. **Concrete Builder**: Implements the builder interface
3. **Product**: The complex object being built
4. **Director**: Orchestrates the building process
5. **Builder Registry**: Manages builder registration and retrieval
6. **Product Registry**: Manages product registration and retrieval

## Features

- **Builder Management**: Create, destroy, and manage builders
- **Product Creation**: Create products using builders
- **Director Management**: Manage directors for orchestrating builds
- **Step-by-Step Building**: Build products step by step
- **Validation**: Input validation for builders and products
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
builder/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── builder/
│   │   ├── interfaces.go
│   │   ├── models.go
│   │   ├── builder.go
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
cd builder
```

2. Install dependencies:

```bash
go mod tidy
```

3. Set up the databases:

```bash
# MySQL
mysql -u root -p -e "CREATE DATABASE builder_db;"

# MongoDB
mongosh --eval "use builder_db"
```

4. Start Kafka:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic builder-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

5. Update configuration in `configs/config.yaml`:

```yaml
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "your_username"
    password: "your_password"
    database: "builder_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "builder_db"
kafka:
  brokers:
    - "localhost:9092"
  group_id: "builder-group"
  topic: "builder-events"
```

6. Run the service:

```bash
go run main.go
```

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Builder Management

- `POST /api/v1/builder/builders` - Create a builder
- `DELETE /api/v1/builder/builders/:type` - Destroy a builder
- `GET /api/v1/builder/builders/:type` - Get builder by type
- `GET /api/v1/builder/builders` - List all builders
- `GET /api/v1/builder/builders/:type/stats` - Get builder statistics
- `GET /api/v1/builder/builders/stats` - Get all builder statistics
- `PUT /api/v1/builder/builders/:type/active` - Set builder active status

### Product Management

- `POST /api/v1/builder/products` - Create a product
- `DELETE /api/v1/builder/products/:id` - Destroy a product
- `GET /api/v1/builder/products/:id` - Get product by ID
- `GET /api/v1/builder/products` - List all products
- `GET /api/v1/builder/products/:id/stats` - Get product statistics
- `GET /api/v1/builder/products/stats` - Get all product statistics
- `PUT /api/v1/builder/products/:id/active` - Set product active status

### Director Management

- `POST /api/v1/builder/directors` - Create a director
- `DELETE /api/v1/builder/directors/:id` - Destroy a director
- `GET /api/v1/builder/directors/:id` - Get director by ID
- `GET /api/v1/builder/directors` - List all directors
- `GET /api/v1/builder/directors/:id/stats` - Get director statistics
- `GET /api/v1/builder/directors/stats` - Get all director statistics
- `PUT /api/v1/builder/directors/:id/active` - Set director active status

### Service Management

- `GET /api/v1/builder/stats` - Get service statistics
- `POST /api/v1/builder/cleanup` - Perform cleanup
- `GET /api/v1/builder/health` - Get detailed health status

### WebSocket

- `GET /ws` - WebSocket connection for real-time updates

## Usage Examples

### Creating a Builder

```bash
curl -X POST http://localhost:8080/api/v1/builder/builders \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ConcreteBuilder",
    "name": "Product Builder",
    "description": "Builder for creating products",
    "version": "1.0.0",
    "metadata": {
      "category": "product",
      "region": "us-east-1"
    }
  }'
```

### Creating a Product

```bash
curl -X POST http://localhost:8080/api/v1/builder/products \
  -H "Content-Type: application/json" \
  -d '{
    "builder_type": "ConcreteBuilder",
    "config": {
      "name": "Laptop",
      "description": "High-performance laptop",
      "price": 1500.0,
      "category": "Electronics",
      "tags": ["laptop", "computer", "electronics"],
      "metadata": {
        "brand": "TechCorp",
        "model": "TC-2023"
      },
      "active": true
    }
  }'
```

### Creating a Director

```bash
curl -X POST http://localhost:8080/api/v1/builder/directors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "product-director",
    "name": "Product Director",
    "description": "Director for building products",
    "version": "1.0.0",
    "metadata": {
      "category": "product",
      "region": "us-east-1"
    }
  }'
```

### Getting Builder Statistics

```bash
curl -X GET http://localhost:8080/api/v1/builder/builders/ConcreteBuilder/stats
```

### Getting Product Statistics

```bash
curl -X GET http://localhost:8080/api/v1/builder/products/product-123/stats
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
    database: "builder_db"
    max_idle_conns: 10
    max_open_conns: 100
    conn_max_lifetime: 3600
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "builder_db"
    max_pool_size: 100
    min_pool_size: 10
    max_idle_time: 30s

kafka:
  brokers:
    - "localhost:9092"
  group_id: "builder-group"
  topic: "builder-events"
  consumer_timeout: 10s
  producer_timeout: 10s

builder:
  name: "Builder Service"
  version: "1.0.0"
  description: "Builder pattern service for constructing complex objects step by step"
  max_builders: 100
  max_products: 1000
  max_directors: 50
  cleanup_interval: "1h"
  validation_enabled: true
  caching_enabled: true
  monitoring_enabled: true
  auditing_enabled: true
  supported_builder_types:
    - "ConcreteBuilder"
    - "PaymentBuilder"
    - "OrderBuilder"
    - "UserBuilder"
    - "NotificationBuilder"
    - "ReportBuilder"
  supported_product_types:
    - "ConcreteProduct"
    - "PaymentProduct"
    - "OrderProduct"
    - "UserProduct"
    - "NotificationProduct"
    - "ReportProduct"
  validation_rules:
    builder:
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
      category:
        required: true
        min_length: 2
        max_length: 50
    director:
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
  metadata:
    environment: "development"
    region: "us-east-1"
    team: "backend"
    project: "builder-service"
```

## Testing

Run the tests:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestBuilder ./internal/builder/

# Run benchmarks
go test -bench=. ./...
```

## Monitoring

The service provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Detailed Health**: `GET /api/v1/builder/health` - Detailed health status with component checks
- **Service Stats**: `GET /api/v1/builder/stats` - Service statistics and metrics

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
  builder-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=builder_db
      - MONGODB_URI=mongodb://mongodb:27017
      - MONGODB_DB=builder_db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - mysql
      - mongodb
      - kafka

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: builder_db
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
- Basic Builder pattern implementation
- Builder and Product management
- Director management
- Registry management
- WebSocket support
- Kafka integration
- MySQL and MongoDB support
- Health checks and monitoring
- Comprehensive testing
- Docker support
