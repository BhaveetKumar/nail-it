# Template Method Pattern Implementation

This microservice demonstrates the Template Method design pattern in a real-world fintech application, specifically for document processing workflows.

## Overview

The Template Method pattern defines the skeleton of an algorithm in a base class, allowing subclasses to override specific steps without changing the algorithm's structure. This implementation provides a flexible framework for processing various types of documents and workflows.

## Features

- **Template Method Pattern**: Implements the classic Template Method pattern with abstract base classes and concrete implementations
- **Document Processing**: Specialized templates for different document types (PDF, Word, Excel, etc.)
- **Workflow Management**: Sequential and parallel workflow execution
- **Data Validation**: Configurable validation rules and error handling
- **API Request Processing**: RESTful API request templates with retry logic
- **Database Operations**: CRUD operation templates with transaction support
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
│                    Template Method Service                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Document      │  │   Workflow      │  │   Data          │ │
│  │   Processing    │  │   Template      │  │   Validation    │ │
│  │   Template      │  │                 │  │   Template      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   API Request   │  │   Database      │  │   Custom        │ │
│  │   Template      │  │   Operation     │  │   Template      │ │
│  │                 │  │   Template      │  │                 │ │
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

## Template Method Pattern Structure

### Abstract Template Method
```go
type TemplateMethod interface {
    Execute() error
    GetName() string
    GetDescription() string
    GetStatus() string
    GetSteps() []Step
    GetCurrentStep() int
    GetStartTime() time.Time
    GetEndTime() time.Time
    GetDuration() time.Duration
    IsCompleted() bool
    IsFailed() bool
    IsRunning() bool
    GetError() error
    GetMetadata() map[string]interface{}
}
```

### Concrete Template Methods
- **Document Processing Template**: Handles PDF, Word, Excel document processing
- **Data Validation Template**: Validates data against configurable rules
- **Workflow Template**: Manages sequential and parallel workflow execution
- **API Request Template**: Processes RESTful API requests with retry logic
- **Database Operation Template**: Handles CRUD operations with transaction support

## API Endpoints

### Template Method Management
- `POST /api/v1/template-methods/` - Create a new template method
- `GET /api/v1/template-methods/:name` - Get template method details
- `DELETE /api/v1/template-methods/:name` - Remove a template method
- `GET /api/v1/template-methods/` - List all template methods

### Template Method Execution
- `POST /api/v1/template-methods/:name/execute` - Execute a template method
- `GET /api/v1/template-methods/:name/stats` - Get execution statistics
- `GET /api/v1/template-methods/stats` - Get overall statistics
- `GET /api/v1/template-methods/history` - Get execution history
- `DELETE /api/v1/template-methods/history` - Clear execution history

### Step Management
- `POST /api/v1/template-methods/:name/steps` - Add a step to template method
- `GET /api/v1/template-methods/:name/steps/:stepName` - Get step details
- `POST /api/v1/template-methods/:name/steps/:stepName/execute` - Execute a specific step

### WebSocket
- `GET /ws` - WebSocket endpoint for real-time updates

## Configuration

The service can be configured via `configs/config.yaml`:

```yaml
name: "Template Method Service"
version: "1.0.0"
description: "Template Method pattern implementation with microservice architecture"

max_template_methods: 1000
max_steps: 100
max_execution_time: 30m
max_retries: 3
retry_delay: 1s
retry_backoff: 2.0

validation_enabled: true
caching_enabled: true
monitoring_enabled: true
auditing_enabled: true
scheduling_enabled: true

supported_types:
  - "document_processing"
  - "data_validation"
  - "workflow"
  - "api_request"
  - "database_operation"

default_type: "workflow"

validation_rules:
  max_steps: 100
  max_execution_time: 30m
  max_retries: 3

metadata:
  environment: "production"
  region: "us-east-1"

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "template_method_db"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "template_method_db"
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
    - "template-method-events"

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

### Creating a Document Processing Template
```bash
curl -X POST http://localhost:8080/api/v1/template-methods/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "pdf-processing",
    "description": "Process PDF documents",
    "type": "document_processing"
  }'
```

### Adding Steps to a Template Method
```bash
curl -X POST http://localhost:8080/api/v1/template-methods/pdf-processing/steps \
  -H "Content-Type: application/json" \
  -d '{
    "step_name": "extract-text",
    "description": "Extract text from PDF",
    "type": "extraction",
    "dependencies": []
  }'
```

### Executing a Template Method
```bash
curl -X POST http://localhost:8080/api/v1/template-methods/pdf-processing/execute
```

### Getting Template Method Statistics
```bash
curl http://localhost:8080/api/v1/template-methods/pdf-processing/stats
```

## Template Method Pattern Benefits

1. **Code Reuse**: Common algorithm structure is defined once in the base class
2. **Flexibility**: Subclasses can override specific steps without changing the overall algorithm
3. **Consistency**: All implementations follow the same execution flow
4. **Maintainability**: Changes to the algorithm structure only need to be made in one place
5. **Extensibility**: New template methods can be easily added by extending the base class

## Real-World Use Cases

### Fintech Applications
- **Document Processing**: KYC document verification, invoice processing, contract analysis
- **Data Validation**: Customer data validation, transaction data verification, compliance checks
- **Workflow Management**: Loan approval workflows, payment processing pipelines, risk assessment
- **API Request Processing**: Third-party API integrations, webhook processing, data synchronization
- **Database Operations**: Batch data processing, data migration, backup and restore operations

### Other Applications
- **E-commerce**: Order processing, inventory management, customer service workflows
- **Healthcare**: Patient data processing, medical record validation, treatment workflows
- **Education**: Student enrollment, grade processing, course management
- **Manufacturing**: Production workflows, quality control, supply chain management

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
- **Metrics**: Request counts, response times, error rates, and custom metrics
- **Logging**: Structured logging with configurable levels and formats
- **Tracing**: Distributed tracing for request flow analysis
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
