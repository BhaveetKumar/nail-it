# Memento Pattern Microservice

A complete microservice implementation demonstrating the Memento design pattern with MongoDB, MySQL, REST API, WebSockets, and Kafka integration.

## Overview

The Memento pattern provides the ability to restore an object to its previous state (undo via rollback). This pattern is used to implement the undo functionality in applications. The memento pattern is implemented with three objects: the Originator, a Caretaker, and a Memento.

## Features

- **Memento Management**: Create, save, and restore mementos
- **Originator Support**: Support for various originator types (document, database, file, configuration)
- **Caretaker Management**: Manage multiple caretakers for different memento collections
- **State Restoration**: Restore objects to previous states
- **Version Control**: Track versions of mementos
- **Metadata Support**: Store additional metadata with mementos
- **Validation**: Validate mementos before saving
- **Compression**: Compress mementos to save space
- **Encryption**: Encrypt sensitive mementos
- **Caching**: In-memory caching for improved performance
- **Real-time Updates**: WebSocket support for real-time memento updates
- **Message Queue**: Kafka integration for event-driven architecture
- **Database Integration**: MySQL and MongoDB support
- **Monitoring**: Comprehensive metrics and health checks
- **Security**: JWT authentication and rate limiting
- **Audit Logging**: Complete audit trail for all operations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │   WebSocket     │    │   Kafka         │
│   (Gin)         │    │   (Gorilla)     │    │   (Sarama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Memento Service │
                    │   (Core Logic)  │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MySQL       │    │    MongoDB      │    │     Redis       │
│   (GORM)        │    │   (MongoDB)     │    │   (go-redis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Memento Types

### 1. Document Memento

Saves and restores document states.

```go
originator := NewDocumentOriginator("doc1", "My Document")
originator.SetContent("Hello, World!")

// Create memento
memento := originator.CreateMemento()

// Modify document
originator.SetContent("Hello, Universe!")

// Restore from memento
originator.RestoreMemento(memento)
```

### 2. Database Memento

Saves and restores database states.

```go
originator := NewDatabaseOriginator("db1", "User Database", "users", "public")
originator.AddRecord(map[string]interface{}{
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
})

// Create memento
memento := originator.CreateMemento()

// Modify database
originator.AddRecord(map[string]interface{}{
    "id": 2,
    "name": "Jane Doe",
    "email": "jane@example.com",
})

// Restore from memento
originator.RestoreMemento(memento)
```

### 3. File Memento

Saves and restores file states.

```go
originator := NewFileOriginator("file1", "config.txt", "/path/to/config.txt")
originator.SetContent([]byte("key1=value1\nkey2=value2"))

// Create memento
memento := originator.CreateMemento()

// Modify file
originator.SetContent([]byte("key1=value1\nkey2=value2\nkey3=value3"))

// Restore from memento
originator.RestoreMemento(memento)
```

### 4. Configuration Memento

Saves and restores configuration states.

```go
originator := NewConfigurationOriginator("config1", "App Configuration")
originator.SetConfig(map[string]interface{}{
    "database": map[string]interface{}{
        "host": "localhost",
        "port": 3306,
    },
    "cache": map[string]interface{}{
        "enabled": true,
        "ttl": 300,
    },
})

// Create memento
memento := originator.CreateMemento()

// Modify configuration
originator.SetConfig(map[string]interface{}{
    "database": map[string]interface{}{
        "host": "production-db",
        "port": 3306,
    },
    "cache": map[string]interface{}{
        "enabled": true,
        "ttl": 600,
    },
})

// Restore from memento
originator.RestoreMemento(memento)
```

## API Endpoints

### Caretaker Management

#### Create Caretaker

```http
POST /api/v1/mementos/caretakers
Content-Type: application/json

{
    "name": "my-caretaker"
}
```

#### Get Caretaker

```http
GET /api/v1/mementos/caretakers/{name}
```

#### List Caretakers

```http
GET /api/v1/mementos/caretakers
```

#### Remove Caretaker

```http
DELETE /api/v1/mementos/caretakers/{name}
```

### Memento Operations

#### Save Memento

```http
POST /api/v1/mementos/caretakers/{name}/mementos
Content-Type: application/json

{
    "id": "memento1",
    "originator_id": "originator1",
    "state": {
        "content": "Hello, World!",
        "metadata": {
            "author": "John Doe",
            "created": "2023-01-01T00:00:00Z"
        }
    },
    "type": "document",
    "description": "Document memento"
}
```

#### Get Memento

```http
GET /api/v1/mementos/caretakers/{name}/mementos/{id}
```

#### List Mementos

```http
GET /api/v1/mementos/caretakers/{name}/mementos?originator_id=originator1&type=document
```

#### Delete Memento

```http
DELETE /api/v1/mementos/caretakers/{name}/mementos/{id}
```

#### Get Caretaker Statistics

```http
GET /api/v1/mementos/caretakers/{name}/stats
```

#### Cleanup Caretaker

```http
POST /api/v1/mementos/caretakers/{name}/cleanup
```

### Health Check

```http
GET /health
```

## WebSocket Events

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8080/ws");
```

### Events

- `memento_created`: When a new memento is created
- `memento_restored`: When a memento is restored
- `memento_deleted`: When a memento is deleted
- `caretaker_created`: When a new caretaker is created
- `caretaker_removed`: When a caretaker is removed
- `cleanup_completed`: When cleanup is completed

## Kafka Topics

- `memento-events`: Memento lifecycle events
- `memento-commands`: Memento operation commands
- `memento-responses`: Memento operation responses

## Configuration

The service is configured via `configs/config.yaml`:

```yaml
name: "Memento Service"
version: "1.0.0"
max_mementos: 10000
max_memento_size: 1048576 # 1MB
max_memento_age: "24h"
cleanup_interval: "1h"
backup_interval: "6h"
replication_interval: "1h"
validation_interval: "30m"
compression_enabled: true
encryption_enabled: true
caching_enabled: true
indexing_enabled: true
monitoring_enabled: true
auditing_enabled: true
scheduling_enabled: true
backup_enabled: true
replication_enabled: true
validation_enabled: true

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "memento_db"

  mongodb:
    uri: "mongodb://localhost:27017"
    database: "memento_db"

  redis:
    host: "localhost"
    port: 6379
    password: ""
    db: 0

cache:
  enabled: true
  type: "memory"
  ttl: "5m"
  max_size: 1000

message_queue:
  enabled: true
  brokers: ["localhost:9092"]
  topics: ["memento-events"]

websocket:
  enabled: true
  port: 8080
  read_buffer_size: 1024
  write_buffer_size: 1024

security:
  enabled: true
  jwt_secret: "your-secret-key"
  token_expiry: "24h"
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: "1m"

monitoring:
  enabled: true
  port: 9090
  path: "/metrics"
  collect_interval: "30s"

logging:
  level: "info"
  format: "json"
  output: "stdout"
```

## Usage Examples

### Basic Document Memento

```go
// Create document originator
originator := NewDocumentOriginator("doc1", "My Document")
originator.SetContent("Hello, World!")

// Create caretaker
caretaker := NewConcreteCaretaker(config)

// Create memento
memento := originator.CreateMemento()
caretaker.SaveMemento(memento)

// Modify document
originator.SetContent("Hello, Universe!")

// Restore from memento
savedMemento, _ := caretaker.GetMemento(memento.GetID())
originator.RestoreMemento(savedMemento)
```

### Database Memento with Version Control

```go
// Create database originator
originator := NewDatabaseOriginator("db1", "User Database", "users", "public")

// Add initial records
originator.AddRecord(map[string]interface{}{
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
})

// Create memento (version 1)
memento1 := originator.CreateMemento()
caretaker.SaveMemento(memento1)

// Add more records
originator.AddRecord(map[string]interface{}{
    "id": 2,
    "name": "Jane Doe",
    "email": "jane@example.com",
})

// Create memento (version 2)
memento2 := originator.CreateMemento()
caretaker.SaveMemento(memento2)

// Restore to version 1
savedMemento1, _ := caretaker.GetMemento(memento1.GetID())
originator.RestoreMemento(savedMemento1)
```

### File Memento with Metadata

```go
// Create file originator
originator := NewFileOriginator("file1", "config.txt", "/path/to/config.txt")
originator.SetContent([]byte("key1=value1\nkey2=value2"))
originator.SetMetadata(map[string]interface{}{
    "author": "John Doe",
    "version": "1.0.0",
    "last_modified": time.Now(),
})

// Create memento
memento := originator.CreateMemento()
memento.SetDescription("Initial configuration")
memento.SetMetadata(map[string]interface{}{
    "backup_reason": "before deployment",
    "environment": "production",
})

caretaker.SaveMemento(memento)

// Modify file
originator.SetContent([]byte("key1=value1\nkey2=value2\nkey3=value3"))

// Restore from memento
savedMemento, _ := caretaker.GetMemento(memento.GetID())
originator.RestoreMemento(savedMemento)
```

### Configuration Memento with Validation

```go
// Create configuration originator
originator := NewConfigurationOriginator("config1", "App Configuration")
originator.SetConfig(map[string]interface{}{
    "database": map[string]interface{}{
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "password",
    },
    "cache": map[string]interface{}{
        "enabled": true,
        "ttl": 300,
        "max_size": 1000,
    },
    "logging": map[string]interface{}{
        "level": "info",
        "format": "json",
    },
})

// Create memento
memento := originator.CreateMemento()
caretaker.SaveMemento(memento)

// Modify configuration
originator.SetConfig(map[string]interface{}{
    "database": map[string]interface{}{
        "host": "production-db",
        "port": 3306,
        "username": "prod_user",
        "password": "prod_password",
    },
    "cache": map[string]interface{}{
        "enabled": true,
        "ttl": 600,
        "max_size": 2000,
    },
    "logging": map[string]interface{}{
        "level": "warn",
        "format": "json",
    },
})

// Restore from memento
savedMemento, _ := caretaker.GetMemento(memento.GetID())
originator.RestoreMemento(savedMemento)
```

### Memento with Compression and Encryption

```go
// Create memento with compression
memento := &BaseMemento{
    ID:           "memento1",
    OriginatorID: "originator1",
    State:        largeData,
    Timestamp:    time.Now(),
    Version:      1,
    Type:         "document",
    Description:  "Compressed and encrypted memento",
    Metadata:     make(map[string]interface{}),
    Valid:        true,
    Size:         calculateSize(largeData),
    Checksum:     calculateChecksum(largeData),
}

// Compress memento
compressor := NewMementoCompressor()
compressedData, _ := compressor.Compress(serialize(memento.GetState()))
memento.SetState(compressedData)

// Encrypt memento
encryptor := NewMementoEncryptor()
key, _ := encryptor.GenerateKey()
encryptedData, _ := encryptor.Encrypt(compressedData, key)
memento.SetState(encryptedData)

// Save memento
caretaker.SaveMemento(memento)
```

## Testing

### Unit Tests

```bash
go test ./internal/memento/...
```

### Integration Tests

```bash
go test -tags=integration ./...
```

### Benchmark Tests

```bash
go test -bench=. ./internal/memento/...
```

## Performance Considerations

1. **Memory Usage**: Mementos hold references to state data, so large states should be compressed
2. **Storage**: Use compression and encryption for sensitive or large mementos
3. **Caching**: Enable caching for frequently accessed mementos
4. **Cleanup**: Implement regular cleanup to remove old mementos
5. **Validation**: Validate mementos before saving to prevent corruption

## Security

- JWT authentication for API access
- Rate limiting to prevent abuse
- CORS configuration for web clients
- Input validation for all endpoints
- Audit logging for compliance
- Encryption for sensitive mementos

## Monitoring

- Health checks for all components
- Metrics collection and export
- Performance monitoring
- Error tracking and alerting
- Resource usage monitoring

## Deployment

### Docker

```bash
docker build -t memento-service .
docker run -p 8080:8080 memento-service
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

### Environment Variables

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USERNAME=root
export MYSQL_PASSWORD=password
export MYSQL_DATABASE=memento_db
export MONGODB_URI=mongodb://localhost:27017
export MONGODB_DATABASE=memento_db
export REDIS_HOST=localhost
export REDIS_PORT=6379
export KAFKA_BROKERS=localhost:9092
export JWT_SECRET=your-secret-key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:

- Create an issue in the repository
- Check the documentation
- Review the examples

## Changelog

### v1.0.0

- Initial release
- Basic memento types
- REST API
- WebSocket support
- Kafka integration
- Database support
- Caching
- Monitoring
- Security features
