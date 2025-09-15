# Chain of Responsibility Pattern Implementation

This microservice demonstrates the **Chain of Responsibility Pattern** implementation in Go, providing a way to pass requests along a chain of handlers, where each handler decides either to process the request or pass it to the next handler in the chain.

## Overview

The Chain of Responsibility Pattern allows you to pass requests along a chain of handlers. In this implementation:

- **Handler**: Interface for processing requests
- **Concrete Handlers**: Specific implementations (Authentication, Authorization, Validation, Rate Limiting, Logging)
- **Chain Manager**: Manages the chain of handlers
- **Client**: Sends requests through the chain

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Chain          │    │   Handlers      │
│                 │    │   Service        │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Auth        │◄┼────┼─┤ ProcessAuth  │◄┼────┼─┤ Auth        │ │
│ │ Request     │ │    │ │ Request      │ │    │ │ Handler     │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │        │        │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │        ▼        │
│ │ Authz       │◄┼────┼─┤ ProcessAuthz │◄┼────┼─┌─────────────┐ │
│ │ Request     │ │    │ │ Request      │ │    │ │ Authz       │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ Handler     │ │
│                 │    │                  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │        │        │
│ │ Validation  │◄┼────┼─┤ ProcessVal   │◄┼────┼─┌─────────────┐ │
│ │ Request     │ │    │ │ Request      │ │    │ │ Validation  │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ Handler     │ │
│                 │    │                  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │        │        │
│ │ Rate Limit  │◄┼────┼─┤ ProcessRate  │◄┼────┼─┌─────────────┐ │
│ │ Request     │ │    │ │ Limit        │ │    │ │ Rate Limit  │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ Handler     │ │
│                 │    │                  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │        │        │
│ │ Logging     │◄┼────┼─┤ ProcessLog   │◄┼────┼─┌─────────────┐ │
│ │ Request     │ │    │ │ Request      │ │    │ │ Logging     │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ Handler     │ │
│                 │    │                  │    │ └─────────────┘ │
│                 │    │ ┌──────────────┐ │    │                 │
│                 │    │ │ Chain        │ │    │                 │
│                 │    │ │ Manager      │ │    │                 │
│                 │    │ └──────────────┘ │    │                 │
│                 │    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

## Features

### Core Handlers
- **AuthenticationHandler**: Validates user authentication tokens
- **AuthorizationHandler**: Checks user permissions and roles
- **ValidationHandler**: Validates request data against rules
- **RateLimitHandler**: Implements rate limiting for requests
- **LoggingHandler**: Logs requests and responses

### Chain Management
- **Dynamic Handler Addition/Removal**: Add or remove handlers at runtime
- **Priority-based Ordering**: Handlers are ordered by priority
- **Chain Optimization**: Automatic chain structure optimization
- **Statistics Collection**: Comprehensive metrics for each handler
- **Chain Validation**: Validates chain structure and detects cycles

### Service Layer
- **High-Level Operations**: Simplified API for request processing
- **Request Routing**: Automatic routing to appropriate handlers
- **Response Aggregation**: Combines responses from multiple handlers
- **Error Handling**: Graceful error handling and recovery

## API Endpoints

### Authentication
```bash
# Process authentication request
POST /api/v1/auth/authenticate
Content-Type: application/json

{
  "user_id": "user_123",
  "token": "jwt_token_here"
}
```

### Authorization
```bash
# Process authorization request
POST /api/v1/auth/authorize
Content-Type: application/json

{
  "user_id": "user_123",
  "role": "admin",
  "action": "delete",
  "resource": "user_data"
}
```

### Validation
```bash
# Process validation request
POST /api/v1/validation/validate
Content-Type: application/json

{
  "user_id": "user_123",
  "data": {
    "email": "user@example.com",
    "password": "securepassword",
    "username": "testuser"
  }
}
```

### Rate Limiting
```bash
# Process rate limit request
POST /api/v1/rate-limit/check
Content-Type: application/json

{
  "user_id": "user_123",
  "action": "api_call"
}
```

### Logging
```bash
# Process logging request
POST /api/v1/logging/log
Content-Type: application/json

{
  "user_id": "user_123",
  "message": "User action performed",
  "level": "info"
}
```

### Chain Management
```bash
# Get chain statistics
GET /api/v1/chain/statistics

# Get all handlers
GET /api/v1/chain/handlers

# Get specific handler
GET /api/v1/chain/handlers/{name}

# Get handler statistics
GET /api/v1/chain/handlers/{name}/statistics

# Optimize chain
POST /api/v1/chain/optimize

# Validate chain
POST /api/v1/chain/validate
```

### Health Check
```bash
# Health check
GET /health
```

### WebSocket
```bash
# Connect for real-time updates
WS /ws
```

## Configuration

The service uses YAML configuration with the following key sections:

- **Server**: Port, timeouts, CORS settings
- **Database**: MySQL, MongoDB, Redis configurations
- **Kafka**: Message queue settings
- **Chain**: Chain-specific settings and handler configurations
- **Cache**: Caching configuration
- **Message Queue**: Queue settings
- **WebSocket**: Real-time communication settings
- **Security**: Authentication and authorization settings
- **Monitoring**: Metrics and health checks
- **Business Logic**: Feature flags and optimizations

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
   docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=chain_db -p 3306:3306 mysql:8.0
   
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
   
   # Process authentication request
   curl -X POST http://localhost:8080/api/v1/auth/authenticate \
     -H "Content-Type: application/json" \
     -d '{"user_id":"user_123","token":"jwt_token_here"}'
   
   # Get chain statistics
   curl http://localhost:8080/api/v1/chain/statistics
   ```

## Design Patterns Used

### Chain of Responsibility Pattern
- **Handler Interface**: Common interface for all handlers
- **Concrete Handlers**: Specific implementations for different request types
- **Chain Manager**: Manages the chain of handlers
- **Request Processing**: Passes requests through the chain

### Additional Patterns
- **Factory Pattern**: Handler creation and management
- **Service Layer Pattern**: High-level business operations
- **Observer Pattern**: WebSocket real-time updates
- **Strategy Pattern**: Different handler implementations
- **Template Method Pattern**: Common handler processing flow

## Benefits

1. **Decoupling**: Handlers are decoupled from each other
2. **Flexibility**: Easy to add, remove, or reorder handlers
3. **Single Responsibility**: Each handler has a single responsibility
4. **Open/Closed Principle**: Open for extension, closed for modification
5. **Dynamic Chain**: Chain can be modified at runtime
6. **Statistics**: Comprehensive metrics for each handler
7. **Validation**: Chain structure validation and cycle detection

## Use Cases

- **Request Processing**: Authentication, authorization, validation
- **Middleware**: Web server middleware chains
- **Event Processing**: Event handling pipelines
- **Data Processing**: Data transformation pipelines
- **Workflow Management**: Business process workflows
- **Error Handling**: Error handling chains
- **Logging**: Logging and audit trails

## Handler Flow Examples

### Authentication Flow
```
Request → AuthenticationHandler → AuthorizationHandler → ValidationHandler → RateLimitHandler → LoggingHandler → Response
```

### Authorization Flow
```
Request → AuthorizationHandler → ValidationHandler → RateLimitHandler → LoggingHandler → Response
```

### Validation Flow
```
Request → ValidationHandler → RateLimitHandler → LoggingHandler → Response
```

### Rate Limiting Flow
```
Request → RateLimitHandler → LoggingHandler → Response
```

### Logging Flow
```
Request → LoggingHandler → Response
```

## Performance Characteristics

### Request Processing
- **Chain Traversal**: O(n) where n is the number of handlers
- **Handler Processing**: O(1) for each handler
- **Cache Lookup**: O(1) for cached responses
- **Statistics Update**: O(1) for each handler

### Memory Usage
- **Handler Storage**: O(n) where n is the number of handlers
- **Request Storage**: O(1) per request
- **Cache Storage**: O(k) where k is the cache size
- **Statistics Storage**: O(n) where n is the number of handlers

### Scalability
- **Concurrent Requests**: Thread-safe handler processing
- **Handler Addition**: O(1) for adding new handlers
- **Chain Optimization**: O(n log n) for sorting handlers
- **Statistics Collection**: O(n) for collecting all handler statistics

## Monitoring

- **Metrics**: Request processing times, handler statistics, chain performance
- **Health Checks**: Chain health and handler status
- **Logging**: Structured logging with request context
- **Statistics**: Detailed handler and chain statistics
- **WebSocket**: Real-time monitoring dashboards

## Security Features

- **Input Validation**: Validate all request data
- **Authentication**: JWT token validation
- **Authorization**: Role-based access control
- **Rate Limiting**: Request rate limiting
- **Audit Logging**: Comprehensive audit trails
- **Data Encryption**: Encrypt sensitive data

## Performance Features

- **Multi-Level Caching**: Memory + Redis caching
- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking operations
- **Chain Optimization**: Automatic chain structure optimization
- **Memory Monitoring**: Real-time memory usage tracking
- **Statistics Aggregation**: Efficient statistics collection

## Best Practices

1. **Handler Design**: Keep handlers focused on single responsibilities
2. **Chain Ordering**: Order handlers by priority and dependencies
3. **Error Handling**: Implement proper error handling in each handler
4. **Statistics**: Monitor handler performance and chain statistics
5. **Validation**: Validate chain structure and detect cycles
6. **Optimization**: Regularly optimize chain structure
7. **Testing**: Test each handler independently and as part of the chain

This implementation demonstrates how the Chain of Responsibility Pattern can be used to create flexible, maintainable systems for processing requests through a series of handlers, making it ideal for middleware, request processing, and workflow management systems.
