---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.562840
Tags: []
Status: draft
---

# Adapter Pattern Implementation

This is a complete microservice implementation demonstrating the Adapter design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Adapter pattern for:

- **Payment Gateway Integration**: Adapt different payment gateways (Stripe, Razorpay, PayPal, Bank Transfer)
- **Notification Service Integration**: Adapt different notification services (Email, SMS, Push, Webhook, Slack)
- **Database Integration**: Adapt different databases (MySQL, PostgreSQL, MongoDB)
- **Cache Integration**: Adapt different cache systems (Redis, Memcached, Memory)
- **Message Queue Integration**: Adapt different message queues (Kafka, RabbitMQ, SQS)
- **File Storage Integration**: Adapt different file storage systems (S3, GCS, Azure, Local)
- **Authentication Integration**: Adapt different authentication systems (JWT, OAuth, LDAP, Basic)

## Features

### Core Functionality

- **Adapter Management**: Register, unregister, and manage multiple adapters
- **Adapter Factory**: Factory pattern for creating adapters
- **Adapter Metrics**: Comprehensive metrics and monitoring
- **Adapter Health**: Health monitoring for all adapters
- **Adapter Fallback**: Automatic fallback to backup adapters
- **Real-time Updates**: WebSocket-based real-time adapter updates
- **Event Streaming**: Kafka integration for adapter events
- **Adapter Persistence**: Persist adapter configuration and metrics

### Adapter Types

#### 1. Payment Gateway Adapters

```go
// Payment gateway interface
type PaymentGateway interface {
    ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error)
    RefundPayment(ctx context.Context, request RefundRequest) (*RefundResponse, error)
    GetPaymentStatus(ctx context.Context, paymentID string) (*PaymentStatus, error)
    GetGatewayName() string
    IsAvailable() bool
}
```

**Available Adapters:**

- **Stripe**: Credit card payments (USD, EUR)
- **Razorpay**: Indian payment gateway (INR)
- **PayPal**: International payments (USD, EUR, GBP)
- **Bank Transfer**: Direct bank transfers (USD, EUR, GBP, INR)

#### 2. Notification Service Adapters

```go
// Notification service interface
type NotificationService interface {
    SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error)
    GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error)
    GetServiceName() string
    IsAvailable() bool
}
```

**Available Adapters:**

- **Email**: SMTP-based email notifications
- **SMS**: SMS gateway notifications
- **Push**: Mobile push notifications
- **Webhook**: HTTP webhook notifications
- **Slack**: Slack channel notifications

#### 3. Database Adapters

```go
// Database adapter interface
type DatabaseAdapter interface {
    Connect(ctx context.Context) error
    Disconnect(ctx context.Context) error
    Query(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error)
    Execute(ctx context.Context, query string, args ...interface{}) (int64, error)
    BeginTransaction(ctx context.Context) (Transaction, error)
    GetAdapterName() string
    IsConnected() bool
}
```

**Available Adapters:**

- **MySQL**: MySQL database adapter
- **PostgreSQL**: PostgreSQL database adapter
- **MongoDB**: MongoDB database adapter

#### 4. Cache Adapters

```go
// Cache adapter interface
type CacheAdapter interface {
    Get(ctx context.Context, key string) (interface{}, error)
    Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
    Clear(ctx context.Context) error
    GetAdapterName() string
    IsConnected() bool
}
```

**Available Adapters:**

- **Redis**: Redis cache adapter
- **Memcached**: Memcached cache adapter
- **Memory**: In-memory cache adapter

#### 5. Message Queue Adapters

```go
// Message queue adapter interface
type MessageQueueAdapter interface {
    Publish(ctx context.Context, topic string, message interface{}) error
    Subscribe(ctx context.Context, topic string, handler MessageHandler) error
    Unsubscribe(ctx context.Context, topic string) error
    GetAdapterName() string
    IsConnected() bool
}
```

**Available Adapters:**

- **Kafka**: Apache Kafka message queue adapter
- **RabbitMQ**: RabbitMQ message queue adapter
- **SQS**: AWS SQS message queue adapter

#### 6. File Storage Adapters

```go
// File storage adapter interface
type FileStorageAdapter interface {
    Upload(ctx context.Context, file File) (*UploadResponse, error)
    Download(ctx context.Context, fileID string) (*File, error)
    Delete(ctx context.Context, fileID string) error
    List(ctx context.Context, prefix string) ([]FileInfo, error)
    GetAdapterName() string
    IsAvailable() bool
}
```

**Available Adapters:**

- **S3**: AWS S3 file storage adapter
- **GCS**: Google Cloud Storage adapter
- **Azure**: Azure Blob Storage adapter
- **Local**: Local file system adapter

#### 7. Authentication Adapters

```go
// Authentication adapter interface
type AuthenticationAdapter interface {
    Authenticate(ctx context.Context, credentials Credentials) (*AuthResponse, error)
    ValidateToken(ctx context.Context, token string) (*TokenValidation, error)
    RefreshToken(ctx context.Context, refreshToken string) (*AuthResponse, error)
    GetAdapterName() string
    IsAvailable() bool
}
```

**Available Adapters:**

- **JWT**: JSON Web Token authentication adapter
- **OAuth**: OAuth 2.0 authentication adapter
- **LDAP**: LDAP authentication adapter
- **Basic**: Basic HTTP authentication adapter

## API Endpoints

### Adapter Management

- `GET /api/v1/adapters` - Get all adapters
- `GET /api/v1/adapters/:type` - Get adapters by type
- `GET /api/v1/adapters/:type/:name` - Get specific adapter
- `POST /api/v1/adapters/:type/:name/register` - Register adapter
- `DELETE /api/v1/adapters/:type/:name` - Unregister adapter

### Payment Gateway Adapters

- `POST /api/v1/payments/process` - Process payment using selected gateway
- `POST /api/v1/payments/refund` - Refund payment using selected gateway
- `GET /api/v1/payments/:id/status` - Get payment status
- `GET /api/v1/payments/gateways` - Get available payment gateways

### Notification Service Adapters

- `POST /api/v1/notifications/send` - Send notification using selected service
- `GET /api/v1/notifications/:id/status` - Get notification status
- `GET /api/v1/notifications/services` - Get available notification services

### Database Adapters

- `POST /api/v1/database/query` - Execute query using selected database
- `POST /api/v1/database/execute` - Execute command using selected database
- `POST /api/v1/database/transaction` - Begin transaction using selected database
- `GET /api/v1/database/adapters` - Get available database adapters

### Cache Adapters

- `GET /api/v1/cache/:key` - Get value from cache using selected adapter
- `POST /api/v1/cache/:key` - Set value in cache using selected adapter
- `DELETE /api/v1/cache/:key` - Delete value from cache using selected adapter
- `GET /api/v1/cache/adapters` - Get available cache adapters

### Message Queue Adapters

- `POST /api/v1/messages/publish` - Publish message using selected adapter
- `POST /api/v1/messages/subscribe` - Subscribe to topic using selected adapter
- `DELETE /api/v1/messages/subscribe` - Unsubscribe from topic using selected adapter
- `GET /api/v1/messages/adapters` - Get available message queue adapters

### File Storage Adapters

- `POST /api/v1/files/upload` - Upload file using selected adapter
- `GET /api/v1/files/:id/download` - Download file using selected adapter
- `DELETE /api/v1/files/:id` - Delete file using selected adapter
- `GET /api/v1/files` - List files using selected adapter
- `GET /api/v1/files/adapters` - Get available file storage adapters

### Authentication Adapters

- `POST /api/v1/auth/authenticate` - Authenticate using selected adapter
- `POST /api/v1/auth/validate` - Validate token using selected adapter
- `POST /api/v1/auth/refresh` - Refresh token using selected adapter
- `GET /api/v1/auth/adapters` - Get available authentication adapters

### Adapter Health

- `GET /api/v1/adapters/health` - Get health status of all adapters
- `GET /api/v1/adapters/:type/:name/health` - Get health status of specific adapter

### Adapter Metrics

- `GET /api/v1/adapters/metrics` - Get adapter metrics
- `GET /api/v1/adapters/:type/:name/metrics` - Get metrics for specific adapter
- `POST /api/v1/adapters/metrics/reset` - Reset adapter metrics

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## WebSocket Events

### Adapter Events

- `adapter.registered` - Adapter registered
- `adapter.unregistered` - Adapter unregistered
- `adapter.available` - Adapter available
- `adapter.unavailable` - Adapter unavailable
- `adapter.health_changed` - Adapter health changed

### Payment Events

- `payment.processed` - Payment processed
- `payment.refunded` - Payment refunded
- `payment.status_changed` - Payment status changed

### Notification Events

- `notification.sent` - Notification sent
- `notification.delivered` - Notification delivered
- `notification.failed` - Notification failed

## Kafka Events

### Adapter Events

- All adapter operations are streamed to Kafka for external consumption
- Adapter registration and health events
- Adapter metrics and performance events

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
cd adapter
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
CREATE DATABASE adapter_db;
```

4. **Run the service**:

```bash
go run main.go
```

## Configuration

The service uses a YAML configuration file (`configs/config.yaml`) with the following sections:

- **Server**: HTTP server configuration
- **Database**: MySQL connection settings
- **MongoDB**: MongoDB connection settings
- **Redis**: Redis connection settings
- **Kafka**: Kafka broker and topic configuration
- **Adapter**: Adapter configuration and parameters

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

# Test payment processing
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"payment_id":"pay123","user_id":"user123","amount":100.50,"currency":"USD","payment_method":"card","description":"Test payment"}' http://localhost:8080/api/v1/payments/process

# Test notification sending
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"notification_id":"notif123","user_id":"user123","channel":"email","type":"payment","title":"Payment Confirmation","message":"Your payment has been processed successfully","priority":"high"}' http://localhost:8080/api/v1/notifications/send
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Adapter Health

```bash
curl http://localhost:8080/api/v1/adapters/health
curl http://localhost:8080/api/v1/adapters/payment_gateway/stripe/health
```

### Adapter Metrics

```bash
curl http://localhost:8080/api/v1/adapters/metrics
curl http://localhost:8080/api/v1/adapters/payment_gateway/stripe/metrics
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
# Consume adapter events
kafka-console-consumer --bootstrap-server localhost:9092 --topic adapter-events --from-beginning
```

## Performance Considerations

### Adapter Benefits

- **Integration**: Easy integration with external services
- **Abstraction**: Unified interface for different services
- **Flexibility**: Easy to add new adapters
- **Fallback**: Automatic fallback to backup adapters
- **Metrics**: Comprehensive adapter performance metrics
- **Health Monitoring**: Real-time adapter health monitoring

### Optimization Strategies

- **Adapter Pooling**: Pool adapter connections
- **Caching**: Cache adapter responses
- **Circuit Breaker**: Prevent cascade failures
- **Metrics Collection**: Monitor adapter performance
- **Health Checks**: Regular adapter health monitoring
- **Load Balancing**: Distribute load across adapters

## Error Handling

The service implements comprehensive error handling:

- **Adapter Validation**: Validate adapter inputs
- **Fallback Mechanisms**: Automatic fallback to backup adapters
- **Circuit Breaker**: Prevent cascade failures
- **Error Logging**: Comprehensive error logging
- **Graceful Degradation**: Continue operation with reduced functionality
- **Retry Logic**: Retry failed adapter calls

## Security Considerations

- **Input Validation**: Validate all adapter inputs
- **Authentication**: Secure adapter access
- **Authorization**: Control adapter permissions
- **Rate Limiting**: Prevent adapter abuse
- **Audit Logging**: Log all adapter operations
- **Encryption**: Encrypt sensitive adapter data

## Scalability

### Horizontal Scaling

- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **Adapter Distribution**: Distribute adapters across instances
- **Metrics Aggregation**: Aggregate metrics across instances

### Vertical Scaling

- **Memory Management**: Efficient adapter memory usage
- **CPU Optimization**: Concurrent adapter processing
- **Connection Pooling**: Database connection optimization
- **Caching**: Adapter result caching

## Troubleshooting

### Common Issues

1. **Adapter Not Available**

   - Check adapter registration
   - Verify adapter configuration
   - Check adapter health status
   - Monitor adapter metrics

2. **Adapter Call Failed**

   - Check adapter implementation
   - Verify external service availability
   - Check network connectivity
   - Monitor adapter error logs

3. **High Adapter Latency**

   - Check adapter performance metrics
   - Verify external service performance
   - Check network connectivity
   - Monitor system resources

4. **Adapter Fallback Issues**
   - Check fallback configuration
   - Verify backup adapter availability
   - Check fallback logic
   - Monitor fallback metrics

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View adapter logs
grep "Adapter" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
