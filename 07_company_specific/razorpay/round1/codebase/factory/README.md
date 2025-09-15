# Factory Pattern Implementation

This is a complete microservice implementation demonstrating the Factory design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Factory pattern for:

- **Payment Gateway Factory**: Creates different payment gateway implementations
- **Notification Channel Factory**: Creates different notification channel implementations
- **Database Factory**: Creates different database connection implementations
- **Abstract Factory**: Creates families of related objects

## Features

### Core Functionality

- **Payment Processing**: Multiple payment gateway support (Stripe, PayPal, Razorpay, Bank Transfer, Digital Wallet)
- **Notification System**: Multiple notification channels (Email, SMS, Push, WhatsApp, Slack)
- **Database Support**: Multiple database types (MySQL, PostgreSQL, MongoDB, SQLite)
- **Real-time Notifications**: WebSocket-based updates
- **Event Streaming**: Kafka integration for event-driven architecture
- **Caching**: Redis-based caching for performance
- **Dual Database**: MySQL for transactional data, MongoDB for analytics

### Factory Implementations

#### 1. Payment Gateway Factory

```go
// Factory for creating payment gateways
factory := factory.GetPaymentGatewayFactory()

// Create different payment gateways
stripeGateway, _ := factory.CreateGateway("stripe")
paypalGateway, _ := factory.CreateGateway("paypal")
razorpayGateway, _ := factory.CreateGateway("razorpay")
```

#### 2. Notification Channel Factory

```go
// Factory for creating notification channels
factory := factory.GetNotificationChannelFactory()

// Create different notification channels
emailChannel, _ := factory.CreateChannel("email")
smsChannel, _ := factory.CreateChannel("sms")
pushChannel, _ := factory.CreateChannel("push")
```

#### 3. Database Factory

```go
// Factory for creating database connections
factory := factory.GetDatabaseFactory()

// Create different database connections
mysqlConn, _ := factory.CreateDatabase("mysql")
postgresConn, _ := factory.CreateDatabase("postgresql")
mongoConn, _ := factory.CreateDatabase("mongodb")
```

#### 4. Abstract Factory

```go
// Abstract factory for creating related objects
abstractFactory := factory.GetAbstractFactory()

// Create payment system components
paymentSystemFactory := abstractFactory.GetPaymentSystemFactory()
gateway := paymentSystemFactory.CreatePaymentGateway()
channel := paymentSystemFactory.CreateNotificationChannel()
database := paymentSystemFactory.CreateDatabaseConnection()
```

## API Endpoints

### Payment Gateways

- `GET /api/v1/payment-gateways` - List available payment gateways
- `POST /api/v1/payments` - Process payment with specified gateway
- `GET /api/v1/payments/:id` - Get payment status
- `POST /api/v1/payments/:id/refund` - Process refund

### Notification Channels

- `GET /api/v1/notification-channels` - List available notification channels
- `POST /api/v1/notifications` - Send notification via specified channel
- `POST /api/v1/notifications/multi-channel` - Send notification via multiple channels

### Database Operations

- `GET /api/v1/databases` - List available database types
- `POST /api/v1/databases/query` - Execute query on specified database
- `POST /api/v1/databases/query-all` - Execute query on all databases

### Factory Information

- `GET /api/v1/factories` - List all available factories
- `GET /api/v1/factories/:type` - Get factory information
- `GET /api/v1/system-info` - Get system information

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## Factory Types

### Payment Gateways

- **Stripe**: Credit card payments
- **PayPal**: PayPal payments
- **Razorpay**: Indian payment gateway
- **Bank Transfer**: Direct bank transfers
- **Digital Wallet**: Mobile wallet payments

### Notification Channels

- **Email**: SMTP-based email notifications
- **SMS**: SMS notifications via API
- **Push**: Mobile push notifications
- **WhatsApp**: WhatsApp Business API
- **Slack**: Slack webhook notifications

### Database Types

- **MySQL**: Relational database
- **PostgreSQL**: Advanced relational database
- **MongoDB**: Document database
- **SQLite**: Embedded database

## WebSocket Events

### Payment Events

- `payment_created` - Payment created
- `payment_processed` - Payment processed
- `payment_failed` - Payment failed
- `refund_processed` - Refund processed

### Notification Events

- `notification_sent` - Notification sent
- `notification_failed` - Notification failed

## Kafka Events

### Event Types

- `payment_created` - Payment creation event
- `payment_processed` - Payment processing event
- `payment_failed` - Payment failure event
- `refund_processed` - Refund processing event
- `notification_sent` - Notification sent event
- `notification_failed` - Notification failure event

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
cd factory
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
CREATE DATABASE factory_db;
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
- **Payment Gateways**: Stripe, PayPal, Razorpay, Bank Transfer, Digital Wallet settings
- **Notification Channels**: Email, SMS, Push, WhatsApp, Slack settings

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
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"user_id":"user123","amount":100.50,"currency":"USD","payment_method":"stripe"}' http://localhost:8080/api/v1/payments

# Test notification sending
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"recipient":"user@example.com","subject":"Test","message":"Test message","channel":"email"}' http://localhost:8080/api/v1/notifications
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Factory Information

```bash
curl http://localhost:8080/api/v1/factories
curl http://localhost:8080/api/v1/factories/payment-gateways
curl http://localhost:8080/api/v1/factories/notification-channels
curl http://localhost:8080/api/v1/factories/databases
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
kafka-console-consumer --bootstrap-server localhost:9092 --topic factory-events --from-beginning
```

## Performance Considerations

### Factory Benefits

- **Object Creation Abstraction**: Centralized object creation logic
- **Easy Extension**: Add new implementations without modifying existing code
- **Configuration-Based**: Runtime selection of implementations
- **Type Safety**: Compile-time type checking

### Optimization Strategies

- **Connection Pooling**: Database connection reuse
- **Caching**: Redis-based response caching
- **Async Processing**: Non-blocking WebSocket and Kafka operations
- **Batch Operations**: Efficient database operations

## Error Handling

The service implements comprehensive error handling:

- **Factory Errors**: Invalid factory types, creation failures
- **Database Errors**: Connection failures, query errors
- **Network Errors**: Timeout handling, retry logic
- **Validation Errors**: Input validation, business rule enforcement
- **Graceful Shutdown**: Clean resource cleanup

## Security Considerations

- **Input Validation**: Request parameter validation
- **SQL Injection Prevention**: Parameterized queries
- **CORS Configuration**: Cross-origin request handling
- **API Key Management**: Secure storage of API keys
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

1. **Factory Creation Failed**

   - Check factory type registration
   - Verify configuration parameters
   - Check factory implementation

2. **Payment Gateway Failed**

   - Verify API keys and credentials
   - Check gateway configuration
   - Monitor gateway logs

3. **Notification Channel Failed**

   - Verify channel configuration
   - Check API credentials
   - Monitor channel logs

4. **Database Connection Failed**
   - Check database service status
   - Verify connection parameters
   - Check network connectivity

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View factory logs
grep "Factory" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
