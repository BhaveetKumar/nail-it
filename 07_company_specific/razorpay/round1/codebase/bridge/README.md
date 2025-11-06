---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.551920
Tags: []
Status: draft
---

# Bridge Pattern Implementation

This microservice demonstrates the **Bridge Pattern** implementation in Go, providing a flexible architecture for handling multiple payment gateways and notification channels.

## Overview

The Bridge Pattern decouples an abstraction from its implementation, allowing them to vary independently. In this implementation:

- **Abstraction**: Payment processing and notification sending interfaces
- **Implementation**: Specific payment gateways (Razorpay, Stripe, PayUMoney) and notification channels (Email, SMS, Push, WhatsApp)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Bridge Service │    │   Implementations│
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Payment     │◄┼────┼─┤ Payment      │◄┼────┼─┤ Razorpay    │ │
│ │ Request     │ │    │ │ Manager      │ │    │ │ Stripe      │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ PayUMoney   │ │
│                 │    │                  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │                 │
│ │ Notification│◄┼────┼─┤ Notification │◄┼────┼─┌─────────────┐ │
│ │ Request     │ │    │ │ Manager      │ │    │ │ Email       │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ │ SMS         │ │
└─────────────────┘    └──────────────────┘    │ │ Push        │ │
                                               │ │ WhatsApp    │ │
                                               │ └─────────────┘ │
                                               └─────────────────┘
```

## Features

### Payment Gateways

- **Razorpay**: Indian payment gateway
- **Stripe**: International payment gateway
- **PayUMoney**: Alternative Indian payment gateway

### Notification Channels

- **Email**: SendGrid integration
- **SMS**: Twilio integration
- **Push**: Firebase Cloud Messaging
- **WhatsApp**: WhatsApp Business API

### Core Services

- **Payment Manager**: Handles payment processing across gateways
- **Notification Manager**: Manages notification sending across channels
- **Bridge Service**: Combines payment and notification services
- **Metrics Service**: Collects and aggregates performance metrics

## API Endpoints

### Payment Processing

```bash
# Process payment with specific gateway
POST /api/v1/payments/{gateway}
Content-Type: application/json

{
  "id": "payment_123",
  "amount": 100.50,
  "currency": "INR",
  "customer_id": "cust_123",
  "merchant_id": "merchant_123",
  "description": "Test payment",
  "metadata": {
    "order_id": "order_123"
  }
}

# Refund payment
POST /api/v1/payments/{gateway}/refund
Content-Type: application/json

{
  "transaction_id": "txn_123",
  "amount": 100.50
}
```

### Notification Sending

```bash
# Send notification via specific channel
POST /api/v1/notifications/{channel}
Content-Type: application/json

{
  "id": "notif_123",
  "type": "payment_success",
  "recipient": "user@example.com",
  "subject": "Payment Successful",
  "message": "Your payment of ₹100.50 has been processed successfully",
  "metadata": {
    "payment_id": "payment_123"
  }
}
```

### Bridge Service

```bash
# Process payment with notification
POST /api/v1/bridge/payment-with-notification
Content-Type: application/json

{
  "gateway": "razorpay",
  "channel": "email",
  "payment": {
    "id": "payment_123",
    "amount": 100.50,
    "currency": "INR",
    "customer_id": "cust_123",
    "merchant_id": "merchant_123",
    "description": "Test payment"
  },
  "notification": {
    "id": "notif_123",
    "type": "payment_success",
    "recipient": "user@example.com",
    "subject": "Payment Successful",
    "message": "Your payment has been processed successfully"
  }
}
```

### Metrics

```bash
# Get payment metrics
GET /api/v1/metrics/payments

# Get notification metrics
GET /api/v1/metrics/notifications

# Health check
GET /health
```

### WebSocket

```bash
# Connect to WebSocket for real-time updates
WS /ws
```

## Configuration

The service uses YAML configuration with the following key sections:

- **Server**: Port, timeouts, CORS settings
- **Database**: MySQL and MongoDB connection settings
- **Redis**: Caching configuration
- **Kafka**: Message queue settings
- **Payment Gateways**: API keys and endpoints
- **Notification Channels**: Channel-specific configurations
- **Logging**: Log level and output settings
- **Security**: Rate limiting and API key settings

## Dependencies

```go
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/gorilla/websocket v1.5.0
    go.mongodb.org/mongo-driver v1.12.1
    gorm.io/driver/mysql v1.5.2
    gorm.io/gorm v1.25.5
    github.com/go-redis/redis/v8 v8.11.5
    github.com/Shopify/sarama v1.38.1
)
```

## Running the Service

1. **Start dependencies**:

   ```bash
   # Start MySQL
   docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=bridge_db -p 3306:3306 mysql:8.0

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

   # Process payment
   curl -X POST http://localhost:8080/api/v1/payments/razorpay \
     -H "Content-Type: application/json" \
     -d '{"id":"payment_123","amount":100.50,"currency":"INR","customer_id":"cust_123","merchant_id":"merchant_123","description":"Test payment"}'
   ```

## Design Patterns Used

### Bridge Pattern

- **Abstraction**: `PaymentGateway` and `NotificationChannel` interfaces
- **Implementation**: Specific gateway and channel implementations
- **Benefit**: Easy to add new gateways/channels without changing existing code

### Factory Pattern

- Gateway and channel creation
- Configuration-based instantiation

### Strategy Pattern

- Different payment processing strategies
- Different notification sending strategies

### Observer Pattern

- WebSocket real-time updates
- Metrics collection

## Benefits

1. **Flexibility**: Easy to add new payment gateways or notification channels
2. **Maintainability**: Clear separation of concerns
3. **Testability**: Each component can be tested independently
4. **Scalability**: Services can be scaled independently
5. **Extensibility**: New features can be added without modifying existing code

## Use Cases

- **E-commerce**: Multiple payment options and notification channels
- **Fintech**: Payment processing with various gateways
- **SaaS**: Multi-tenant notification systems
- **Microservices**: Service-to-service communication

## Monitoring

- **Metrics**: Payment success rates, notification delivery rates
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Service health and dependency status
- **WebSocket**: Real-time updates for monitoring dashboards

## Security

- **API Keys**: Secure storage of gateway credentials
- **Rate Limiting**: Protection against abuse
- **CORS**: Cross-origin request handling
- **Validation**: Input validation and sanitization

This implementation demonstrates how the Bridge Pattern can be used to create a flexible, maintainable, and scalable payment and notification system.
