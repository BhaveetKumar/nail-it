# Unit of Work Pattern Implementation

This microservice demonstrates the Unit of Work design pattern in a real-world fintech application, specifically for managing database transactions and entity state changes.

## Overview

The Unit of Work pattern maintains a list of objects affected by a business transaction and coordinates writing out changes and resolving concurrency problems. This implementation provides a flexible framework for managing entity state changes and database transactions.

## Features

- **Unit of Work Pattern**: Implements the classic Unit of Work pattern with entity tracking and transaction management
- **Entity Management**: Support for user, order, product, payment, and custom entity types
- **Repository Pattern**: Repository-based data access with entity type-specific repositories
- **Transaction Management**: Coordinated commit and rollback operations
- **Entity State Tracking**: Tracks new, dirty, deleted, and clean entities
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
│                    Unit of Work Service                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   User          │  │   Order         │  │   Product       │ │
│  │   Repository    │  │   Repository    │  │   Repository    │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Payment       │  │   Custom        │  │   Entity        │ │
│  │   Repository    │  │   Repository    │  │   Tracking      │ │
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

## Unit of Work Pattern Structure

### Entity Interface

```go
type Entity interface {
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
    IsDirty() bool
    SetDirty(dirty bool)
    IsNew() bool
    SetNew(isNew bool)
    IsDeleted() bool
    SetDeleted(deleted bool)
}
```

### Repository Interface

```go
type Repository interface {
    Create(ctx context.Context, entity Entity) error
    Update(ctx context.Context, entity Entity) error
    Delete(ctx context.Context, entityID string) error
    GetByID(ctx context.Context, entityID string) (Entity, error)
    List(ctx context.Context, filters map[string]interface{}) ([]Entity, error)
    Count(ctx context.Context, filters map[string]interface{}) (int64, error)
    Exists(ctx context.Context, entityID string) (bool, error)
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

### Unit of Work Interface

```go
type UnitOfWork interface {
    RegisterNew(entity Entity) error
    RegisterDirty(entity Entity) error
    RegisterDeleted(entity Entity) error
    RegisterClean(entity Entity) error
    Commit() error
    Rollback() error
    GetRepository(entityType string) (Repository, error)
    RegisterRepository(entityType string, repository Repository) error
    GetEntities() map[string][]Entity
    GetNewEntities() []Entity
    GetDirtyEntities() []Entity
    GetDeletedEntities() []Entity
    GetCleanEntities() []Entity
    Clear() error
    IsActive() bool
    SetActive(active bool)
    GetID() string
    GetName() string
    GetDescription() string
    GetMetadata() map[string]interface{}
    SetMetadata(key string, value interface{})
    GetCreatedAt() time.Time
    GetUpdatedAt() time.Time
}
```

### Concrete Entities

- **User Entity**: Manages user information, email, name, role, and status
- **Order Entity**: Handles order details, items, amounts, and status
- **Product Entity**: Manages product information, SKU, price, category, and stock
- **Payment Entity**: Handles payment details, amounts, methods, and transaction status

### Concrete Repositories

- **User Repository**: User-specific data access operations
- **Order Repository**: Order-specific data access operations
- **Product Repository**: Product-specific data access operations
- **Payment Repository**: Payment-specific data access operations

## API Endpoints

### Entity Management

- `POST /api/v1/unit-of-work/entities/new` - Register a new entity
- `POST /api/v1/unit-of-work/entities/dirty` - Register a dirty entity
- `POST /api/v1/unit-of-work/entities/deleted` - Register a deleted entity
- `POST /api/v1/unit-of-work/entities/clean` - Register a clean entity
- `GET /api/v1/unit-of-work/entities` - Get all entities
- `GET /api/v1/unit-of-work/entities/new` - Get new entities
- `GET /api/v1/unit-of-work/entities/dirty` - Get dirty entities
- `GET /api/v1/unit-of-work/entities/deleted` - Get deleted entities
- `GET /api/v1/unit-of-work/entities/clean` - Get clean entities
- `DELETE /api/v1/unit-of-work/entities` - Clear all entities

### Transaction Management

- `POST /api/v1/unit-of-work/commit` - Commit the unit of work
- `POST /api/v1/unit-of-work/rollback` - Rollback the unit of work

### Repository Management

- `POST /api/v1/repositories/` - Register a new repository
- `GET /api/v1/repositories/:entity_type` - Get repository by entity type
- `GET /api/v1/repositories/` - List all repository types

### Statistics and Information

- `GET /api/v1/stats` - Get unit of work statistics
- `GET /api/v1/info` - Get service information
- `GET /health` - Health check endpoint

### WebSocket

- `GET /ws` - WebSocket endpoint for real-time updates

## Configuration

The service can be configured via `configs/config.yaml`:

```yaml
name: "Unit of Work Service"
version: "1.0.0"
description: "Unit of Work pattern implementation with microservice architecture"

max_entities: 10000
max_repositories: 100
transaction_timeout: 30s
cleanup_interval: 1h

validation_enabled: true
caching_enabled: true
monitoring_enabled: true
auditing_enabled: true

supported_entity_types:
  - "user"
  - "order"
  - "product"
  - "payment"
  - "custom"

default_entity_type: "custom"

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
    database: "unit_of_work_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "unit_of_work_db"
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
    - "unit-of-work-events"

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

### Registering a New User Entity

```bash
curl -X POST http://localhost:8080/api/v1/unit-of-work/entities/new \
  -H "Content-Type: application/json" \
  -d '{
    "type": "user",
    "name": "john-doe",
    "description": "John Doe user account"
  }'
```

### Registering a Dirty Order Entity

```bash
curl -X POST http://localhost:8080/api/v1/unit-of-work/entities/dirty \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "order-123"
  }'
```

### Committing the Unit of Work

```bash
curl -X POST http://localhost:8080/api/v1/unit-of-work/commit
```

### Getting Entity Statistics

```bash
curl http://localhost:8080/api/v1/stats
```

## Unit of Work Pattern Benefits

1. **Transaction Management**: Coordinates multiple database operations in a single transaction
2. **Entity State Tracking**: Tracks changes to entities without immediate database writes
3. **Performance**: Reduces database round trips by batching operations
4. **Consistency**: Ensures data consistency across multiple entities
5. **Rollback Support**: Provides easy rollback capabilities for failed operations

## Real-World Use Cases

### Fintech Applications

- **User Management**: User registration, profile updates, and account management
- **Order Processing**: Order creation, modification, and cancellation
- **Product Management**: Product catalog updates, pricing changes, and inventory management
- **Payment Processing**: Payment creation, status updates, and transaction management
- **Transaction Coordination**: Coordinating multiple related operations across different entities

### Other Applications

- **E-commerce**: Order management, inventory updates, and customer data changes
- **Healthcare**: Patient record updates, appointment scheduling, and medical history changes
- **Education**: Student enrollment, grade updates, and course management
- **Manufacturing**: Production planning, inventory management, and quality control

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
- **Metrics**: Entity counts, transaction success rates, and performance metrics
- **Logging**: Structured logging with configurable levels and formats
- **Tracing**: Distributed tracing for transaction flow analysis
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
