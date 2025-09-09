# Singleton Pattern Implementation

This is a complete microservice implementation demonstrating the Singleton design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Singleton pattern for:

- **Configuration Management**: Global application configuration
- **Database Connections**: MySQL and MongoDB connection pools
- **Redis Client**: Caching and session management
- **Kafka Producer/Consumer**: Event streaming
- **WebSocket Hub**: Real-time communication
- **Logger**: Centralized logging

## Features

### Core Functionality

- **User Management**: Create, read, update users
- **Payment Processing**: Create, read, update payments
- **Real-time Notifications**: WebSocket-based updates
- **Event Streaming**: Kafka integration for event-driven architecture
- **Caching**: Redis-based caching for performance
- **Dual Database**: MySQL for transactional data, MongoDB for analytics

### Singleton Implementations

#### 1. Configuration Manager

```go
// Thread-safe singleton for application configuration
config := config.GetConfigManager()
serverConfig := config.GetServerConfig()
```

#### 2. Database Managers

```go
// MySQL singleton
mysqlDB := database.GetMySQLManager()

// MongoDB singleton
mongoDB := database.GetMongoManager()
```

#### 3. Redis Manager

```go
// Redis singleton for caching
redisClient := redis.GetRedisManager()
```

#### 4. Kafka Components

```go
// Kafka producer singleton
kafkaProducer := kafka.GetKafkaProducer()

// Kafka consumer singleton
kafkaConsumer := kafka.GetKafkaConsumer()
```

#### 5. WebSocket Hub

```go
// WebSocket hub singleton
wsHub := websocket.GetWebSocketHub()
```

#### 6. Logger

```go
// Centralized logging singleton
logger := logger.GetLogger()
```

## API Endpoints

### Users

- `POST /api/v1/users` - Create user
- `GET /api/v1/users/:id` - Get user by ID
- `PUT /api/v1/users/:id` - Update user
- `GET /api/v1/users/:user_id/payments` - Get user payments

### Payments

- `POST /api/v1/payments` - Create payment
- `GET /api/v1/payments/:id` - Get payment by ID
- `PUT /api/v1/payments/:id/status` - Update payment status

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## WebSocket Events

### User Events

- `user_created` - User created
- `user_updated` - User updated

### Payment Events

- `payment_created` - Payment created
- `payment_updated` - Payment status updated

## Kafka Events

### Event Types

- `user_created` - User creation event
- `user_updated` - User update event
- `payment_created` - Payment creation event
- `payment_updated` - Payment update event
- `health_check` - Health check event

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
cd singleton
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
CREATE DATABASE singleton_db;
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
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"user_id":"user123","amount":100.50,"currency":"USD"}' http://localhost:8080/api/v1/payments
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### WebSocket Connection

```javascript
const ws = new WebSocket(
  "ws://localhost:8080/ws?user_id=user123&client_id=client456"
);
ws.onmessage = function (event) {
  console.log("Received:", JSON.parse(event.data));
};
```

### Kafka Events

```bash
# Consume events
kafka-console-consumer --bootstrap-server localhost:9092 --topic singleton-events --from-beginning
```

## Performance Considerations

### Singleton Benefits

- **Memory Efficiency**: Single instance per component
- **Resource Sharing**: Shared connection pools
- **Configuration Consistency**: Single source of truth
- **Thread Safety**: Mutex-protected access

### Optimization Strategies

- **Connection Pooling**: Database connection reuse
- **Caching**: Redis-based response caching
- **Async Processing**: Non-blocking WebSocket and Kafka operations
- **Batch Operations**: Efficient database operations

## Error Handling

The service implements comprehensive error handling:

- **Database Errors**: Connection failures, query errors
- **Network Errors**: Timeout handling, retry logic
- **Validation Errors**: Input validation, business rule enforcement
- **Graceful Shutdown**: Clean resource cleanup

## Security Considerations

- **Input Validation**: Request parameter validation
- **SQL Injection Prevention**: Parameterized queries
- **CORS Configuration**: Cross-origin request handling
- **Rate Limiting**: Request throttling (can be added)

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

1. **Database Connection Failed**

   - Check database service status
   - Verify connection parameters
   - Check network connectivity

2. **Redis Connection Failed**

   - Verify Redis service status
   - Check Redis configuration
   - Monitor Redis memory usage

3. **Kafka Connection Failed**

   - Check Kafka broker status
   - Verify topic configuration
   - Monitor Kafka logs

4. **WebSocket Connection Failed**
   - Check WebSocket endpoint
   - Verify client parameters
   - Monitor connection limits

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
