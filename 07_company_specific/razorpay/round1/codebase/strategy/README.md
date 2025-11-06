---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.603840
Tags: []
Status: draft
---

# Strategy Pattern Implementation

This is a complete microservice implementation demonstrating the Strategy design pattern in Go, with full integration of MongoDB, MySQL, Redis, REST API, WebSockets, and Kafka.

## Architecture Overview

The service implements the Strategy pattern for:

- **Payment Processing**: Multiple payment gateway strategies (Stripe, Razorpay, PayPal, Bank Transfer)
- **Notification Delivery**: Multiple notification channel strategies (Email, SMS, Push, Webhook, Slack)
- **Pricing Calculation**: Multiple pricing strategies (Standard, Discount, Dynamic, Tiered)
- **Authentication**: Multiple authentication strategies (JWT, OAuth, Basic, API Key)
- **Caching**: Multiple caching strategies (Redis, Memory, Database, Hybrid)
- **Logging**: Multiple logging strategies (File, Console, Database, Remote)
- **Data Processing**: Multiple data processing strategies (JSON, XML, CSV, Binary)

## Features

### Core Functionality

- **Strategy Management**: Register, unregister, and manage multiple strategies
- **Strategy Selection**: Intelligent strategy selection based on context and criteria
- **Strategy Factory**: Factory pattern for creating strategies
- **Strategy Metrics**: Comprehensive metrics and monitoring
- **Circuit Breaker**: Fault tolerance for strategy execution
- **Real-time Updates**: WebSocket-based real-time strategy updates
- **Event Streaming**: Kafka integration for strategy events
- **Persistence**: Strategy configuration and metrics storage

### Strategy Types

#### 1. Payment Strategies

```go
// Payment strategy interface
type PaymentStrategy interface {
    ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error)
    ValidatePayment(ctx context.Context, request PaymentRequest) error
    GetStrategyName() string
    GetSupportedCurrencies() []string
    GetProcessingTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **Stripe**: Credit card payments (USD, EUR)
- **Razorpay**: Indian payment gateway (INR)
- **PayPal**: International payments (USD, EUR, GBP)
- **Bank Transfer**: Direct bank transfers (USD, EUR, GBP, INR)

#### 2. Notification Strategies

```go
// Notification strategy interface
type NotificationStrategy interface {
    SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error)
    ValidateNotification(ctx context.Context, request NotificationRequest) error
    GetStrategyName() string
    GetSupportedChannels() []string
    GetDeliveryTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **Email**: SMTP-based email notifications
- **SMS**: SMS gateway notifications
- **Push**: Mobile push notifications
- **Webhook**: HTTP webhook notifications
- **Slack**: Slack channel notifications

#### 3. Pricing Strategies

```go
// Pricing strategy interface
type PricingStrategy interface {
    CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error)
    ValidatePricing(ctx context.Context, request PricingRequest) error
    GetStrategyName() string
    GetSupportedProducts() []string
    GetCalculationTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **Standard**: Basic pricing calculation
- **Discount**: Discount-based pricing
- **Dynamic**: Time-based dynamic pricing
- **Tiered**: Volume-based tiered pricing

#### 4. Authentication Strategies

```go
// Authentication strategy interface
type AuthenticationStrategy interface {
    Authenticate(ctx context.Context, request AuthRequest) (*AuthResponse, error)
    ValidateAuth(ctx context.Context, request AuthRequest) error
    GetStrategyName() string
    GetSupportedMethods() []string
    GetAuthTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **JWT**: JSON Web Token authentication
- **OAuth**: OAuth 2.0 authentication
- **Basic**: Basic HTTP authentication
- **API Key**: API key-based authentication

#### 5. Caching Strategies

```go
// Caching strategy interface
type CachingStrategy interface {
    Get(ctx context.Context, key string) (interface{}, error)
    Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
    Clear(ctx context.Context) error
    GetStrategyName() string
    GetSupportedTypes() []string
    GetAccessTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **Redis**: Redis-based caching
- **Memory**: In-memory caching
- **Database**: Database-based caching
- **Hybrid**: Redis + Memory hybrid caching

#### 6. Logging Strategies

```go
// Logging strategy interface
type LoggingStrategy interface {
    Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error
    GetStrategyName() string
    GetSupportedLevels() []LogLevel
    GetLogTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **File**: File-based logging
- **Console**: Console output logging
- **Database**: Database-based logging
- **Remote**: Remote service logging

#### 7. Data Processing Strategies

```go
// Data processing strategy interface
type DataProcessingStrategy interface {
    ProcessData(ctx context.Context, data interface{}) (interface{}, error)
    ValidateData(ctx context.Context, data interface{}) error
    GetStrategyName() string
    GetSupportedFormats() []string
    GetProcessingTime() time.Duration
    IsAvailable() bool
}
```

**Available Strategies:**

- **JSON**: JSON data processing
- **XML**: XML data processing
- **CSV**: CSV data processing
- **Binary**: Binary data processing

## API Endpoints

### Payment Strategies

- `POST /api/v1/payments/process` - Process payment using selected strategy
- `GET /api/v1/payments/strategies` - Get available payment strategies
- `POST /api/v1/payments/strategies/:name/register` - Register payment strategy
- `DELETE /api/v1/payments/strategies/:name` - Unregister payment strategy

### Notification Strategies

- `POST /api/v1/notifications/send` - Send notification using selected strategy
- `GET /api/v1/notifications/strategies` - Get available notification strategies
- `POST /api/v1/notifications/strategies/:name/register` - Register notification strategy
- `DELETE /api/v1/notifications/strategies/:name` - Unregister notification strategy

### Pricing Strategies

- `POST /api/v1/pricing/calculate` - Calculate price using selected strategy
- `GET /api/v1/pricing/strategies` - Get available pricing strategies
- `POST /api/v1/pricing/strategies/:name/register` - Register pricing strategy
- `DELETE /api/v1/pricing/strategies/:name` - Unregister pricing strategy

### Authentication Strategies

- `POST /api/v1/auth/authenticate` - Authenticate using selected strategy
- `GET /api/v1/auth/strategies` - Get available authentication strategies
- `POST /api/v1/auth/strategies/:name/register` - Register authentication strategy
- `DELETE /api/v1/auth/strategies/:name` - Unregister authentication strategy

### Caching Strategies

- `GET /api/v1/cache/:key` - Get value from cache using selected strategy
- `POST /api/v1/cache/:key` - Set value in cache using selected strategy
- `DELETE /api/v1/cache/:key` - Delete value from cache using selected strategy
- `GET /api/v1/cache/strategies` - Get available caching strategies

### Logging Strategies

- `POST /api/v1/logs` - Log message using selected strategy
- `GET /api/v1/logs/strategies` - Get available logging strategies
- `POST /api/v1/logs/strategies/:name/register` - Register logging strategy
- `DELETE /api/v1/logs/strategies/:name` - Unregister logging strategy

### Data Processing Strategies

- `POST /api/v1/data/process` - Process data using selected strategy
- `GET /api/v1/data/strategies` - Get available data processing strategies
- `POST /api/v1/data/strategies/:name/register` - Register data processing strategy
- `DELETE /api/v1/data/strategies/:name` - Unregister data processing strategy

### Strategy Management

- `GET /api/v1/strategies` - Get all available strategies
- `GET /api/v1/strategies/:type` - Get strategies by type
- `GET /api/v1/strategies/:type/:name` - Get specific strategy
- `POST /api/v1/strategies/:type/:name/register` - Register strategy
- `DELETE /api/v1/strategies/:type/:name` - Unregister strategy
- `PUT /api/v1/strategies/:type/:name/enable` - Enable strategy
- `PUT /api/v1/strategies/:type/:name/disable` - Disable strategy

### Strategy Selection

- `POST /api/v1/strategies/select` - Select best strategy based on criteria
- `GET /api/v1/strategies/selection/criteria` - Get selection criteria
- `POST /api/v1/strategies/selection/criteria` - Set selection criteria

### Strategy Metrics

- `GET /api/v1/strategies/metrics` - Get all strategy metrics
- `GET /api/v1/strategies/metrics/:name` - Get strategy metrics by name
- `GET /api/v1/strategies/metrics/top` - Get top performing strategies
- `GET /api/v1/strategies/metrics/worst` - Get worst performing strategies
- `GET /api/v1/strategies/metrics/health` - Get strategy health scores
- `POST /api/v1/strategies/metrics/reset` - Reset strategy metrics

### Circuit Breaker

- `GET /api/v1/strategies/circuit-breaker/status` - Get circuit breaker status
- `POST /api/v1/strategies/circuit-breaker/reset` - Reset circuit breaker

### WebSocket

- `GET /ws?user_id=:user_id&client_id=:client_id` - WebSocket connection

### Health Check

- `GET /health` - Service health status

## WebSocket Events

### Strategy Events

- `strategy.registered` - Strategy registered
- `strategy.unregistered` - Strategy unregistered
- `strategy.enabled` - Strategy enabled
- `strategy.disabled` - Strategy disabled
- `strategy.selected` - Strategy selected
- `strategy.executed` - Strategy executed
- `strategy.failed` - Strategy execution failed

### Metrics Events

- `metrics.updated` - Strategy metrics updated
- `metrics.threshold_exceeded` - Metrics threshold exceeded
- `metrics.health_changed` - Strategy health changed

## Kafka Events

### Strategy Events

- All strategy operations are streamed to Kafka for external consumption
- Strategy selection and execution events
- Strategy health and metrics events

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
cd strategy
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
CREATE DATABASE strategy_db;
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
- **Strategy**: Strategy configuration and parameters

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
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"payment_id":"pay123","user_id":"user123","amount":100.50,"currency":"USD","payment_method":"card","gateway":"stripe","description":"Test payment"}' http://localhost:8080/api/v1/payments/process

# Test notification sending
hey -n 1000 -c 10 -m POST -H "Content-Type: application/json" -d '{"notification_id":"notif123","user_id":"user123","channel":"email","type":"payment","title":"Payment Confirmation","message":"Your payment has been processed successfully","priority":"high"}' http://localhost:8080/api/v1/notifications/send
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Strategy Metrics

```bash
curl http://localhost:8080/api/v1/strategies/metrics
curl http://localhost:8080/api/v1/strategies/metrics/top
curl http://localhost:8080/api/v1/strategies/metrics/health
```

### Circuit Breaker Status

```bash
curl http://localhost:8080/api/v1/strategies/circuit-breaker/status
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
# Consume strategy events
kafka-console-consumer --bootstrap-server localhost:9092 --topic strategy-events --from-beginning
```

## Performance Considerations

### Strategy Benefits

- **Flexibility**: Easy to add new strategies
- **Maintainability**: Each strategy is isolated
- **Testability**: Strategies can be tested independently
- **Scalability**: Strategies can be scaled independently
- **Reusability**: Strategies can be reused across services

### Optimization Strategies

- **Strategy Caching**: Cache strategy instances
- **Lazy Loading**: Load strategies on demand
- **Connection Pooling**: Reuse connections for external services
- **Circuit Breaker**: Prevent cascade failures
- **Metrics Collection**: Monitor strategy performance
- **Health Checks**: Regular strategy health monitoring

## Error Handling

The service implements comprehensive error handling:

- **Strategy Validation**: Validate strategy inputs
- **Fallback Strategies**: Automatic fallback to backup strategies
- **Circuit Breaker**: Prevent cascade failures
- **Retry Logic**: Retry failed strategy executions
- **Error Logging**: Comprehensive error logging
- **Graceful Degradation**: Continue operation with reduced functionality

## Security Considerations

- **Input Validation**: Validate all strategy inputs
- **Authentication**: Secure strategy access
- **Authorization**: Control strategy permissions
- **Rate Limiting**: Prevent strategy abuse
- **Audit Logging**: Log all strategy operations
- **Encryption**: Encrypt sensitive strategy data

## Scalability

### Horizontal Scaling

- **Stateless Design**: No server-side session storage
- **Load Balancer Ready**: Multiple instance support
- **Strategy Distribution**: Distribute strategies across instances
- **Metrics Aggregation**: Aggregate metrics across instances

### Vertical Scaling

- **Memory Management**: Efficient strategy memory usage
- **CPU Optimization**: Concurrent strategy execution
- **Connection Pooling**: Database connection optimization
- **Caching**: Strategy result caching

## Troubleshooting

### Common Issues

1. **Strategy Not Found**

   - Check strategy registration
   - Verify strategy name
   - Check strategy availability
   - Monitor strategy health

2. **Strategy Execution Failed**

   - Check strategy implementation
   - Verify strategy configuration
   - Check external service availability
   - Monitor circuit breaker status

3. **High Strategy Execution Latency**

   - Check strategy performance metrics
   - Verify external service performance
   - Check network connectivity
   - Monitor system resources

4. **Strategy Selection Issues**
   - Check selection criteria
   - Verify strategy availability
   - Check strategy priorities
   - Monitor selection metrics

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
grep "ERROR" logs/app.log

# View strategy logs
grep "Strategy" logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
