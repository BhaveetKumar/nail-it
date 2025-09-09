# Visitor Pattern Implementation

This microservice demonstrates the Visitor design pattern in a real-world fintech application, specifically for document processing, data validation, and analytics operations.

## Overview

The Visitor pattern allows you to define new operations on existing object structures without modifying those structures. This implementation provides a flexible framework for processing various types of elements with different visitor types.

## Features

- **Visitor Pattern**: Implements the classic Visitor pattern with abstract visitors and concrete implementations
- **Element Types**: Support for document, data, service, and custom element types
- **Visitor Types**: Validation, processing, analytics, and custom visitor types
- **Element Collections**: Group and manage multiple elements together
- **Visit History**: Complete audit trail of all visit operations
- **Real-time Updates**: WebSocket integration for live progress updates
- **Message Queue Integration**: Kafka integration for event-driven processing
- **Caching**: Redis caching for improved performance
- **Monitoring**: Comprehensive metrics and health checks
- **Security**: JWT authentication and rate limiting
- **Auditing**: Complete audit trail for all operations

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
│                      Visitor Service                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Validation    │  │   Processing    │  │   Analytics     │ │
│  │   Visitor       │  │   Visitor       │  │   Visitor       │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Document      │  │   Data          │  │   Service       │ │
│  │   Element       │  │   Element       │  │   Element       │ │
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

## Visitor Pattern Structure

### Element Interface
```go
type Element interface {
    Accept(visitor Visitor) error
    GetID() string
    GetType() string
    GetName() string
    GetDescription() string
    GetMetadata() map[string]interface{}
    SetMetadata(key string, value interface{})
    GetCreatedAt() time.Time
    GetUpdatedAt() time.Time
    IsActive() bool
    SetActive(active bool)
}
```

### Visitor Interface
```go
type Visitor interface {
    VisitElement(element Element) error
    GetName() string
    GetType() string
    GetDescription() string
    GetMetadata() map[string]interface{}
    SetMetadata(key string, value interface{})
    GetCreatedAt() time.Time
    GetUpdatedAt() time.Time
    IsActive() bool
    SetActive(active bool)
}
```

### Concrete Elements
- **Document Element**: Handles document content, type, size, language, and encoding
- **Data Element**: Manages data values, types, formats, and constraints
- **Service Element**: Manages service endpoints, methods, headers, parameters, and timeouts

### Concrete Visitors
- **Validation Visitor**: Validates elements based on configurable rules
- **Processing Visitor**: Processes elements with batch operations and error handling
- **Analytics Visitor**: Analyzes elements and generates reports

## API Endpoints

### Visitor Management
- `POST /api/v1/visitors/` - Create a new visitor
- `GET /api/v1/visitors/:id` - Get visitor details
- `DELETE /api/v1/visitors/:id` - Remove a visitor
- `GET /api/v1/visitors/` - List all visitors

### Element Management
- `POST /api/v1/elements/` - Create a new element
- `GET /api/v1/elements/:id` - Get element details
- `DELETE /api/v1/elements/:id` - Remove an element
- `GET /api/v1/elements/` - List all elements

### Element Collection Management
- `POST /api/v1/collections/` - Create a new element collection
- `GET /api/v1/collections/:id` - Get collection details
- `DELETE /api/v1/collections/:id` - Remove a collection
- `GET /api/v1/collections/` - List all collections

### Visit Operations
- `POST /api/v1/visits/element` - Visit an element
- `POST /api/v1/visits/collection` - Visit an element collection
- `GET /api/v1/visits/history` - Get visit history
- `DELETE /api/v1/visits/history` - Clear visit history

### Statistics and Information
- `GET /api/v1/stats` - Get visitor statistics
- `GET /api/v1/info` - Get service information
- `GET /health` - Health check endpoint

### WebSocket
- `GET /ws` - WebSocket endpoint for real-time updates

## Configuration

The service can be configured via `configs/config.yaml`:

```yaml
name: "Visitor Service"
version: "1.0.0"
description: "Visitor pattern implementation with microservice architecture"

max_visitors: 1000
max_elements: 10000
max_element_collections: 1000
max_visit_history: 10000
visit_timeout: 30s
cleanup_interval: 1h

validation_enabled: true
caching_enabled: true
monitoring_enabled: true
auditing_enabled: true

supported_visitor_types:
  - "validation"
  - "processing"
  - "analytics"
  - "custom"

supported_element_types:
  - "document"
  - "data"
  - "service"
  - "custom"

default_visitor_type: "custom"
default_element_type: "custom"

validation_rules:
  max_name_length: 100
  max_description_length: 500

metadata:
  environment: "production"
  region: "us-east-1"

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "visitor_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "visitor_db"
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
    - "visitor-events"

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

### Creating a Validation Visitor
```bash
curl -X POST http://localhost:8080/api/v1/visitors/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "document-validator",
    "type": "validation",
    "description": "Validates document elements"
  }'
```

### Creating a Document Element
```bash
curl -X POST http://localhost:8080/api/v1/elements/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "invoice-doc",
    "type": "document",
    "description": "Invoice document"
  }'
```

### Performing a Visit
```bash
curl -X POST http://localhost:8080/api/v1/visits/element \
  -H "Content-Type: application/json" \
  -d '{
    "visitor_id": "visitor-id",
    "element_id": "element-id"
  }'
```

### Getting Visitor Statistics
```bash
curl http://localhost:8080/api/v1/stats
```

## Visitor Pattern Benefits

1. **Separation of Concerns**: Operations are separated from the object structure
2. **Extensibility**: New operations can be added without modifying existing classes
3. **Flexibility**: Different visitors can perform different operations on the same elements
4. **Maintainability**: Changes to operations don't affect the element structure
5. **Reusability**: Visitors can be reused across different element types

## Real-World Use Cases

### Fintech Applications
- **Document Processing**: KYC document validation, invoice processing, contract analysis
- **Data Validation**: Customer data validation, transaction data verification, compliance checks
- **Analytics**: Transaction analysis, risk assessment, fraud detection
- **Service Integration**: Third-party API validation, webhook processing, data synchronization

### Other Applications
- **E-commerce**: Product validation, order processing, inventory analysis
- **Healthcare**: Patient data validation, medical record analysis, treatment planning
- **Education**: Student data validation, grade analysis, course evaluation
- **Manufacturing**: Quality control, production analysis, supply chain validation

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
- **Metrics**: Visit counts, success rates, error rates, and custom metrics
- **Logging**: Structured logging with configurable levels and formats
- **Tracing**: Distributed tracing for visit flow analysis
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
