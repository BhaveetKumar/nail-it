# Flyweight Pattern Implementation

This microservice demonstrates the **Flyweight Pattern** implementation in Go, providing an efficient way to handle large numbers of similar objects by sharing common intrinsic state and storing extrinsic state separately.

## Overview

The Flyweight Pattern is used to minimize memory usage by sharing as much data as possible with similar objects. In this implementation:

- **Flyweight**: Shared objects with intrinsic state (Product, User, Order, Notification, Configuration)
- **Factory**: Manages flyweight creation and sharing
- **Client**: Uses flyweights through a service layer
- **Intrinsic State**: Shared, immutable data (product details, user profiles, etc.)
- **Extrinsic State**: Context-specific data (prices, quantities, etc.)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Flyweight      │    │   Flyweight     │
│                 │    │   Service        │    │   Factory       │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Get Product │◄┼────┼─┤ GetProduct   │◄┼────┼─┤ GetFlyweight│ │
│ └─────────────┘ │    │ │              │ │    │ │             │ │
│                 │    │ │              │ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ │ GetUser      │ │    │                 │
│ │ Get User    │◄┼────┼─┤              │ │    │ ┌─────────────┐ │
│ └─────────────┘ │    │ │              │ │    │ │ Create      │ │
│                 │    │ │              │ │    │ │ Flyweight   │ │
│ ┌─────────────┐ │    │ │ GetOrder     │ │    │ └─────────────┘ │
│ │ Get Order   │◄┼────┼─┤              │ │    │                 │
│ └─────────────┘ │    │ │              │ │    │ ┌─────────────┐ │
│                 │    │ │ GetNotification│ │    │ │ Manage      │ │
│ ┌─────────────┐ │    │ │ Template     │ │    │ │ Cache       │ │
│ │ Get Config  │◄┼────┼─┤              │ │    │ └─────────────┘ │
│ └─────────────┘ │    │ │              │ │    │                 │
│                 │    │ │ GetConfiguration│ │    │ ┌─────────────┐ │
│                 │    │ └──────────────┘ │    │ │ Cleanup     │ │
│                 │    │                  │    │ │ Unused      │ │
│                 │    │                  │    │ └─────────────┘ │
│                 │    │                  │    └─────────────────┘
│                 │    │                  │
│                 │    │ ┌──────────────┐ │
│                 │    │ │ Cache        │ │
│                 │    │ │ Layer        │ │
│                 │    │ └──────────────┘ │
│                 │    │                  │
│                 │    │ ┌──────────────┐ │
│                 │    │ │ Database     │ │
│                 │    │ │ Layer        │ │
│                 │    │ └──────────────┘ │
│                 │    └──────────────────┘
└─────────────────┘
```

## Features

### Core Flyweight Types
- **Product Flyweight**: Shared product information (name, description, category, brand)
- **User Flyweight**: Shared user profiles and preferences
- **Order Flyweight**: Shared order templates and metadata
- **Notification Flyweight**: Shared notification templates
- **Configuration Flyweight**: Shared configuration settings

### Factory Management
- **Flyweight Creation**: Creates and manages flyweight instances
- **Cache Management**: In-memory and persistent caching
- **Memory Optimization**: Automatic cleanup of unused flyweights
- **Statistics**: Comprehensive metrics and monitoring
- **Type Safety**: Type-specific flyweight creation

### Service Layer
- **High-Level Operations**: Simplified API for flyweight operations
- **Database Integration**: Fallback to database when flyweights not found
- **Caching Strategy**: Multi-level caching for optimal performance
- **Error Handling**: Graceful error handling and recovery

## API Endpoints

### Product Operations
```bash
# Get product information
GET /api/v1/products/{id}
```

### User Operations
```bash
# Get user information
GET /api/v1/users/{id}
```

### Order Operations
```bash
# Get order information
GET /api/v1/orders/{id}
```

### Notification Template Operations
```bash
# Get notification template
GET /api/v1/notifications/templates/{id}
```

### Configuration Operations
```bash
# Get configuration
GET /api/v1/configurations/{key}
```

### Factory Management
```bash
# Get factory statistics
GET /api/v1/factory/stats

# Cleanup unused flyweights
POST /api/v1/factory/cleanup
```

### Health Check
```bash
# Health check with statistics
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
- **Flyweight**: Flyweight-specific settings
- **Cache**: Caching configuration
- **Message Queue**: Queue settings
- **WebSocket**: Real-time communication settings
- **Security**: Authentication and authorization
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
   docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=flyweight_db -p 3306:3306 mysql:8.0
   
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
   
   # Get product
   curl http://localhost:8080/api/v1/products/prod_123
   
   # Get factory stats
   curl http://localhost:8080/api/v1/factory/stats
   ```

## Design Patterns Used

### Flyweight Pattern
- **Flyweight Interface**: Defines common flyweight operations
- **Concrete Flyweights**: Product, User, Order, Notification, Configuration
- **Flyweight Factory**: Manages flyweight creation and sharing
- **Intrinsic State**: Shared, immutable data
- **Extrinsic State**: Context-specific data

### Additional Patterns
- **Factory Pattern**: Flyweight creation and management
- **Cache Pattern**: Multi-level caching strategy
- **Service Layer Pattern**: High-level business operations
- **Observer Pattern**: WebSocket real-time updates
- **Strategy Pattern**: Different flyweight types

## Benefits

1. **Memory Efficiency**: Significant memory savings by sharing common data
2. **Performance**: Faster object creation and access
3. **Scalability**: Handles large numbers of similar objects efficiently
4. **Maintainability**: Centralized management of shared state
5. **Flexibility**: Easy to add new flyweight types
6. **Caching**: Multi-level caching for optimal performance
7. **Monitoring**: Comprehensive metrics and statistics

## Use Cases

- **E-commerce Systems**: Product catalogs with shared attributes
- **User Management**: User profiles with common preferences
- **Configuration Management**: Shared configuration settings
- **Notification Systems**: Template-based notifications
- **Game Development**: Shared game objects and sprites
- **Document Processing**: Shared formatting and styling
- **UI Components**: Shared component templates

## Memory Optimization Examples

### Before Flyweight Pattern
```go
// Each product instance stores all data
type Product struct {
    ID          string
    Name        string
    Description string
    Category    string
    Brand       string
    BasePrice   float64
    Currency    string
    Attributes  map[string]interface{}
    // ... 1000+ products = 1000+ instances
}
```

### After Flyweight Pattern
```go
// Shared intrinsic state
type ProductFlyweight struct {
    Name        string
    Description string
    Category    string
    Brand       string
    Currency    string
    Attributes  map[string]interface{}
    // ... 1000+ products = 1 shared instance
}

// Extrinsic state stored separately
type ProductContext struct {
    ID        string
    Price     float64
    Quantity  int
    // ... context-specific data
}
```

## Performance Characteristics

### Memory Usage
- **Shared Flyweights**: ~90% memory reduction for similar objects
- **Cache Efficiency**: High hit rates for frequently accessed objects
- **Cleanup**: Automatic removal of unused flyweights

### Access Patterns
- **Cache Hit**: ~1ms response time
- **Memory Hit**: ~5ms response time
- **Database Fallback**: ~50ms response time

### Scalability
- **Concurrent Access**: Thread-safe flyweight operations
- **Memory Management**: Automatic cleanup and garbage collection
- **Cache Warming**: Pre-loading of frequently used flyweights

## Monitoring

- **Metrics**: Flyweight creation, cache hits/misses, memory usage
- **Health Checks**: Factory health and performance metrics
- **Logging**: Structured logging with flyweight context
- **Statistics**: Detailed factory and cache statistics
- **WebSocket**: Real-time monitoring dashboards

## Security Features

- **Input Validation**: Validate all flyweight data
- **Access Control**: Permission-based flyweight access
- **Audit Logging**: Track flyweight operations
- **Rate Limiting**: Prevent abuse of flyweight creation
- **Data Encryption**: Encrypt sensitive flyweight data

## Performance Features

- **Multi-Level Caching**: Memory + Redis caching
- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking operations
- **Memory Monitoring**: Real-time memory usage tracking
- **Cleanup Scheduling**: Automatic unused flyweight removal
- **Batch Operations**: Efficient bulk operations

## Best Practices

1. **Identify Intrinsic State**: Separate shared from context-specific data
2. **Use Appropriate Types**: Choose the right flyweight type for your use case
3. **Monitor Memory Usage**: Track flyweight creation and cleanup
4. **Implement Caching**: Use multi-level caching for optimal performance
5. **Handle Cleanup**: Regularly clean up unused flyweights
6. **Validate Data**: Ensure flyweight data integrity
7. **Monitor Performance**: Track cache hit rates and response times

This implementation demonstrates how the Flyweight Pattern can be used to create a memory-efficient, high-performance system for managing large numbers of similar objects, making it ideal for e-commerce, user management, and other systems with repetitive data patterns.
