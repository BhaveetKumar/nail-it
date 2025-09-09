# Repository Pattern Implementation

This is a complete microservice implementation demonstrating the Repository design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Repository pattern for:
- **Data Access Abstraction**: Clean separation between business logic and data access
- **Multiple Data Sources**: Support for MySQL, MongoDB, and Redis
- **Caching Layer**: Redis-based caching for performance optimization
- **Transaction Management**: Unit of Work pattern for transaction coordination
- **Specification Pattern**: Flexible query building and execution
- **Audit Logging**: Complete audit trail for all data operations

## Features

### Core Functionality
- **User Management**: Complete CRUD operations for users
- **Payment Processing**: Payment creation, tracking, and management
- **Order Management**: Order creation, status tracking, and fulfillment
- **Product Catalog**: Product management with inventory tracking
- **Real-time Notifications**: WebSocket-based updates
- **Event Streaming**: Kafka integration for event-driven architecture
- **Caching**: Redis-based caching for performance
- **Dual Database**: MySQL for transactional data, MongoDB for analytics

### Repository Implementations

#### 1. Basic Repository
```go
// Generic repository interface
type Repository[T Entity] interface {
    Create(ctx context.Context, entity T) error
    GetByID(ctx context.Context, id string) (T, error)
    Update(ctx context.Context, entity T) error
    Delete(ctx context.Context, id string) error
    GetAll(ctx context.Context, limit, offset int) ([]T, error)
    Count(ctx context.Context) (int64, error)
    Exists(ctx context.Context, id string) (bool, error)
    FindBy(ctx context.Context, field string, value interface{}) ([]T, error)
    FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error)
}
```

#### 2. Cached Repository
```go
// Repository with caching capabilities
type CachedRepository[T Entity] interface {
    Repository[T]
    GetFromCache(ctx context.Context, id string) (T, error)
    SetCache(ctx context.Context, id string, entity T, expiration time.Duration) error
    InvalidateCache(ctx context.Context, id string) error
    ClearCache(ctx context.Context) error
}
```

#### 3. Transactional Repository
```go
// Repository with transaction support
type TransactionalRepository[T Entity] interface {
    Repository[T]
    BeginTransaction(ctx context.Context) (Transaction, error)
    WithTransaction(ctx context.Context, fn func(Transaction) error) error
}
```

#### 4. Specification Repository
```go
// Repository with specification support
type SpecificationRepository[T Entity] interface {
    Repository[T]
    FindBySpecification(ctx context.Context, spec Specification) ([]T, error)
    CountBySpecification(ctx context.Context, spec Specification) (int64, error)
}
```

#### 5. Unit of Work
```go
// Unit of Work for transaction management
type UnitOfWork interface {
    GetUserRepository() Repository[*User]
    GetPaymentRepository() Repository[*Payment]
    GetOrderRepository() Repository[*Order]
    GetProductRepository() Repository[*Product]
    Begin() error
    Commit() error
    Rollback() error
    IsActive() bool
}
```

## API Endpoints

### Users
- `POST /api/v1/users` - Create user
- `GET /api/v1/users/:id` - Get user by ID
- `PUT /api/v1/users/:id` - Update user
- `DELETE /api/v1/users/:id` - Delete user
- `GET /api/v1/users` - Get all users (paginated)
- `GET /api/v1/users/search` - Search users

### Payments
- `POST /api/v1/payments` - Create payment
- `GET /api/v1/payments/:id` - Get payment by ID
- `PUT /api/v1/payments/:id` - Update payment
- `DELETE /api/v1/payments/:id` - Delete payment
- `GET /api/v1/payments` - Get all payments (paginated)
- `GET /api/v1/payments/user/:user_id` - Get payments by user

### Orders
- `POST /api/v1/orders` - Create order
- `GET /api/v1/orders/:id` - Get order by ID
- `PUT /api/v1/orders/:id` - Update order
- `DELETE /api/v1/orders/:id` - Delete order
- `GET /api/v1/orders` - Get all orders (paginated)
- `GET /api/v1/orders/user/:user_id` - Get orders by user

### Products
- `POST /api/v1/products` - Create product
- `GET /api/v1/products/:id` - Get product by ID
- `PUT /api/v1/products/:id` - Update product
- `DELETE /api/v1/products/:id` - Delete product
- `GET /api/v1/products` - Get all products (paginated)
- `GET /api/v1/products/category/:category` - Get products by category

### Repository Operations
- `GET /api/v1/repositories` - List available repository types
- `GET /api/v1/repositories/:type` - Get repository information
- `POST /api/v1/repositories/query` - Execute custom query
- `GET /api/v1/repositories/cache/stats` - Get cache statistics
- `POST /api/v1/repositories/cache/warm` - Warm cache

### WebSocket
- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check
- `GET /health` - Service health status

## Repository Types

### MySQL Repository
- **Purpose**: Primary transactional data storage
- **Features**: ACID compliance, complex queries, relationships
- **Use Cases**: User data, payment records, order management

### MongoDB Repository
- **Purpose**: Document-based storage for analytics
- **Features**: Flexible schema, aggregation, full-text search
- **Use Cases**: Product catalog, audit logs, analytics data

### Cached Repository
- **Purpose**: Performance optimization with Redis
- **Features**: TTL-based expiration, cache invalidation
- **Use Cases**: Frequently accessed data, session management

### Specification Repository
- **Purpose**: Flexible query building
- **Features**: Composite specifications, type-safe queries
- **Use Cases**: Complex search, filtering, reporting

## WebSocket Events

### User Events
- `user_created` - User created
- `user_updated` - User updated
- `user_deleted` - User deleted

### Payment Events
- `payment_created` - Payment created
- `payment_updated` - Payment status updated
- `payment_deleted` - Payment deleted

### Order Events
- `order_created` - Order created
- `order_updated` - Order status updated
- `order_deleted` - Order deleted

### Product Events
- `product_created` - Product created
- `product_updated` - Product updated
- `product_deleted` - Product deleted

## Kafka Events

### Event Types
- `user_created` - User creation event
- `user_updated` - User update event
- `user_deleted` - User deletion event
- `payment_created` - Payment creation event
- `payment_updated` - Payment update event
- `payment_deleted` - Payment deletion event
- `order_created` - Order creation event
- `order_updated` - Order update event
- `order_deleted` - Order deletion event
- `product_created` - Product creation event
- `product_updated` - Product update event
- `product_deleted` - Product deletion event

## Setup Instructions

### Prerequisites
- Go 1.21+
- MySQL 8.0+
- MongoDB 4.4+
- Redis 6.0+
- Kafka 2.8+

### Installation

1. **Clone and setup**:
```bash
cd repository
go mod tidy
```

2. **Start dependencies**:
```bash
# Start MySQL
docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -p 3306:3306 mysql:8.0

# Start MongoDB
docker run -d --name mongodb -p 27017:27017 mongo:4.4

# Start Redis
docker run -d --name redis -p 6379:6379 redis:6.0

# Start Kafka
docker-compose up -d kafka zookeeper
```

3. **Create database**:
```sql
CREATE DATABASE repository_db;
```

4. **Run the service**:
```bash
go run main.go
```

## Configuration

The service uses a YAML configuration file (`configs/config.yaml`) with the following sections:

- **Server**: HTTP server configuration
- **Database**: MySQL connection settings
- **Redis**: Redis connection settings
- **Kafka**: Kafka broker and topic configuration
- **MongoDB**: MongoDB connection settings
- **Cache**: Cache configuration (TTL, etc.)

## Testing

### Unit Tests
```bash
go test ./...
```

### Integration Tests
```bash
go test -tags=integration ./...
```

### Load Testing
```bash
# Install hey
go install github.com/rakyll/hey@latest

# Test user creation
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"email":"test@example.com","name":"Test User"}' http://localhost:8080/api/v1/users

# Test payment creation
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"user_id":"user123","amount":100.50,"currency":"USD","gateway":"stripe"}' http://localhost:8080/api/v1/payments
```

## Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### Repository Information
```bash
curl http://localhost:8080/api/v1/repositories
curl http://localhost:8080/api/v1/repositories/mysql
curl http://localhost:8080/api/v1/repositories/mongodb
curl http://localhost:8080/api/v1/repositories/cached
```

### Cache Statistics
```bash
curl http://localhost:8080/api/v1/repositories/cache/stats
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws?user_id=user123&client_id=client456');
ws.onmessage = function(event) {
    console.log('Received:', JSON.parse(event.data));
};
```

### Kafka Events
```bash
# Consume events
kafka-console-consumer --bootstrap-server localhost:9092 --topic repository-events --from-beginning
```

## Performance Considerations

### Repository Benefits
- **Data Access Abstraction**: Clean separation of concerns
- **Testability**: Easy to mock and test
- **Flexibility**: Easy to change data sources
- **Caching**: Performance optimization with Redis
- **Transactions**: ACID compliance with Unit of Work

### Optimization Strategies
- **Connection Pooling**: Database connection reuse
- **Caching**: Redis-based response caching
- **Async Processing**: Non-blocking WebSocket and Kafka operations
- **Batch Operations**: Efficient database operations
- **Indexing**: Proper database indexing

## Error Handling

The service implements comprehensive error handling:
- **Repository Errors**: Data access failures, validation errors
- **Database Errors**: Connection failures, query errors
- **Cache Errors**: Redis connection failures, cache misses
- **Transaction Errors**: Rollback handling, deadlock detection
- **Validation Errors**: Input validation, business rule enforcement
- **Graceful Shutdown**: Clean resource cleanup

## Security Considerations

- **Input Validation**: Request parameter validation
- **SQL Injection Prevention**: Parameterized queries
- **CORS Configuration**: Cross-origin request handling
- **Rate Limiting**: Request throttling (can be added)
- **Audit Logging**: Complete audit trail

## Scalability

### Horizontal Scaling
- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **Database Sharding**: User-based sharding strategy
- **Cache Distribution**: Redis cluster support

### Vertical Scaling
- **Connection Pool Tuning**: Database connection optimization
- **Memory Management**: Efficient resource utilization
- **CPU Optimization**: Concurrent request processing

## Troubleshooting

### Common Issues

1. **Repository Creation Failed**
   - Check database connections
   - Verify configuration parameters
   - Check repository factory initialization

2. **Cache Misses**
   - Verify Redis connection
   - Check cache TTL configuration
   - Monitor cache statistics

3. **Transaction Failures**
   - Check database transaction support
   - Verify Unit of Work implementation
   - Monitor transaction logs

4. **Specification Queries Failed**
   - Verify specification implementation
   - Check query syntax
   - Monitor query performance

### Logs
```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View repository logs
grep "Repository" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
