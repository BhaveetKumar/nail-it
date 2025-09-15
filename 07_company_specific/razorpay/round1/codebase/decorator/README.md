# Decorator Pattern Implementation

This microservice demonstrates the **Decorator Pattern** implementation in Go, providing a flexible and extensible architecture for adding cross-cutting concerns to components without modifying their core functionality.

## Overview

The Decorator Pattern allows behavior to be added to objects dynamically without altering their structure. In this implementation:

- **Component**: Core business logic components (Payment, Notification, User, etc.)
- **Decorator**: Cross-cutting concerns (Logging, Metrics, Caching, Security, etc.)
- **Client**: Uses decorated components transparently

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Decorator      │    │   Component     │
│                 │    │   Chain          │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Request     │◄┼────┼─┤ Logging      │◄┼────┼─┤ Payment     │ │
│ └─────────────┘ │    │ │ Decorator    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Metrics      │ │    │ │ Notification│ │
│                 │    │ │ Decorator    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Cache        │ │    │ │ User        │ │
│                 │    │ │ Decorator    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Security     │ │    │ │ Order       │ │
│                 │    │ │ Decorator    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Rate Limit   │ │    │ │ Inventory   │ │
│                 │    │ │ Decorator    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Circuit      │ │    │ │ Analytics   │ │
│                 │    │ │ Breaker      │ │    │ │ Component   │ │
│                 │    │ │ Decorator    │ │    │ └─────────────┘ │
│                 │    │ └──────────────┘ │    │                 │
│                 │    │                  │    │ ┌─────────────┐ │
│                 │    │ ┌──────────────┐ │    │ │ Audit       │ │
│                 │    │ │ Retry        │ │    │ │ Component   │ │
│                 │    │ │ Decorator    │ │    │ └─────────────┘ │
│                 │    │ └──────────────┘ │    │                 │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │                 │
│                 │    │ │ Monitoring   │ │    │                 │
│                 │    │ │ Decorator    │ │    │                 │
│                 │    │ └──────────────┘ │    │                 │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │                 │
│                 │    │ │ Validation   │ │    │                 │
│                 │    │ │ Decorator    │ │    │                 │
│                 │    │ └──────────────┘ │    │                 │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │                 │
│                 │    │ │ Encryption   │ │    │                 │
│                 │    │ │ Decorator    │ │    │                 │
│                 │    │ └──────────────┘ │    │                 │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │                 │
│                 │    │ │ Compression  │ │    │                 │
│                 │    │ │ Decorator    │ │    │                 │
│                 │    │ └──────────────┘ │    │                 │
│                 │    └──────────────────┘    │                 │
│                 │                             │                 │
└─────────────────┘                             └─────────────────┘
```

## Features

### Core Components

- **Payment Component**: Payment processing and transactions
- **Notification Component**: Email, SMS, push notifications
- **User Component**: User management and authentication
- **Order Component**: Order processing and management
- **Inventory Component**: Product inventory management
- **Analytics Component**: Analytics and tracking
- **Audit Component**: Audit logging and compliance

### Decorators

- **Logging Decorator**: Request/response logging
- **Metrics Decorator**: Performance metrics collection
- **Cache Decorator**: Response caching
- **Security Decorator**: Input validation and sanitization
- **Rate Limit Decorator**: Request rate limiting
- **Circuit Breaker Decorator**: Failure handling
- **Retry Decorator**: Automatic retry logic
- **Monitoring Decorator**: Health monitoring
- **Validation Decorator**: Data validation
- **Encryption Decorator**: Data encryption/decryption
- **Compression Decorator**: Data compression
- **Serialization Decorator**: Data serialization
- **Notification Decorator**: Event notifications
- **Analytics Decorator**: Event tracking
- **Audit Decorator**: Audit logging

## API Endpoints

### Component Execution

```bash
# Execute component with decorators
POST /api/v1/components/{component}/execute
Content-Type: application/json

{
  "data": {
    "amount": 100.50,
    "currency": "INR"
  },
  "decorators": ["logging", "metrics", "cache", "security"]
}
```

### Health Checks

```bash
# Check specific component health
GET /api/v1/components/{component}/health

# Check all components health
GET /health
```

### Metrics

```bash
# Get component metrics
GET /api/v1/components/{component}/metrics
```

### Decorator Management

```bash
# List all decorators
GET /api/v1/decorators/

# Get decorator chain for component
GET /api/v1/decorators/{component}/chain?decorators=logging,metrics,cache
```

### Component Management

```bash
# List all components
GET /api/v1/management/components

# Remove component
DELETE /api/v1/management/components/{component}

# Remove decorator
DELETE /api/v1/management/decorators/{decorator}
```

### WebSocket

```bash
# Connect for real-time updates
WS /ws
```

## Configuration

The service uses YAML configuration with the following key sections:

- **Server**: Port, timeouts, CORS settings
- **Database**: MySQL and MongoDB connection settings
- **Redis**: Caching configuration
- **Kafka**: Message queue settings
- **Components**: Core component configurations
- **Decorators**: Decorator-specific configurations
- **Logging**: Log level and output settings
- **Security**: Authentication and authorization settings
- **Monitoring**: Metrics and health check configuration

## Dependencies

```go
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/gorilla/websocket v1.5.0
    github.com/go-redis/redis/v8 v8.11.5
    github.com/Shopify/sarama v1.38.1
    go.mongodb.org/mongo-driver v1.12.1
    gorm.io/driver/mysql v1.5.2
    gorm.io/gorm v1.25.5
    go.uber.org/zap v1.25.0
    github.com/patrickmn/go-cache v2.1.0+incompatible
    github.com/prometheus/client_golang v1.17.0
    github.com/sirupsen/logrus v1.9.3
    github.com/opentracing/opentracing-go v1.2.0
    github.com/uber/jaeger-client-go v2.30.0+incompatible
)
```

## Running the Service

1. **Start dependencies**:

   ```bash
   # Start MySQL
   docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=decorator_db -p 3306:3306 mysql:8.0

   # Start MongoDB
   docker run -d --name mongodb -p 27017:27017 mongo:6.0

   # Start Redis
   docker run -d --name redis -p 6379:6379 redis:7.0

   # Start Kafka
   docker run -d --name kafka -p 9092:9092 apache/kafka:3.4.0
   ```

2. **Run the service**:

   ```bash
   go mod tidy
   go run main.go
   ```

3. **Test the service**:

   ```bash
   # Health check
   curl http://localhost:8080/health

   # Execute payment component with decorators
   curl -X POST http://localhost:8080/api/v1/components/payment/execute \
     -H "Content-Type: application/json" \
     -d '{"data":{"amount":100.50,"currency":"INR"},"decorators":["logging","metrics","cache"]}'
   ```

## Design Patterns Used

### Decorator Pattern

- **Component Interface**: Defines the contract for components
- **Decorator Interface**: Defines the contract for decorators
- **Concrete Components**: Implement core business logic
- **Concrete Decorators**: Add cross-cutting concerns
- **Decorator Manager**: Manages component and decorator registration

### Additional Patterns

- **Factory Pattern**: Component and decorator creation
- **Strategy Pattern**: Different decorator implementations
- **Observer Pattern**: WebSocket real-time updates
- **Chain of Responsibility**: Decorator chain execution
- **Template Method**: Base decorator implementation

## Benefits

1. **Flexibility**: Add/remove decorators at runtime
2. **Reusability**: Decorators can be applied to any component
3. **Separation of Concerns**: Business logic separated from cross-cutting concerns
4. **Composability**: Mix and match decorators as needed
5. **Testability**: Each decorator can be tested independently
6. **Maintainability**: Easy to add new decorators without modifying existing code
7. **Performance**: Decorators can be optimized independently

## Use Cases

- **Cross-cutting Concerns**: Logging, metrics, caching, security
- **Middleware**: Request/response processing
- **Aspect-Oriented Programming**: Adding aspects to components
- **Plugin Architecture**: Extensible component behavior
- **Monitoring**: Adding observability to components
- **Security**: Adding security layers to components

## Decorator Chain Examples

### Basic Chain

```
Component → Logging → Metrics → Cache
```

### Security Chain

```
Component → Security → Validation → Encryption → Logging
```

### Performance Chain

```
Component → Cache → Compression → Metrics → Monitoring
```

### Full Chain

```
Component → Security → Validation → RateLimit → CircuitBreaker →
Retry → Cache → Compression → Encryption → Logging → Metrics →
Monitoring → Notification → Analytics → Audit
```

## Monitoring

- **Metrics**: Component performance, decorator effectiveness
- **Health Checks**: Component and decorator health
- **Logging**: Structured logging with decorator context
- **Tracing**: Request tracing through decorator chain
- **WebSocket**: Real-time monitoring dashboards

## Security Features

- **Input Validation**: Validate all incoming requests
- **Data Sanitization**: Clean and sanitize input data
- **Encryption**: Encrypt sensitive data
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Audit Logging**: Track all security events
- **Permission Checking**: Verify user permissions

## Performance Features

- **Caching**: Response caching with TTL
- **Compression**: Data compression for efficiency
- **Circuit Breaking**: Prevent cascading failures
- **Retry Logic**: Automatic retry with backoff
- **Connection Pooling**: Efficient connection management
- **Async Processing**: Non-blocking operations

This implementation demonstrates how the Decorator Pattern can be used to create a flexible, extensible, and maintainable system for adding cross-cutting concerns to components without modifying their core functionality.
