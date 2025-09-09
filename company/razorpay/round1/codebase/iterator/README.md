# Iterator Pattern Microservice

A complete microservice implementation demonstrating the Iterator design pattern with MongoDB, MySQL, REST API, WebSockets, and Kafka integration.

## Overview

The Iterator pattern provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation. This microservice implements various iterator types and provides a comprehensive API for managing and processing collections.

## Features

- **Multiple Iterator Types**: Slice, Map, Channel, Database, File, Filtered, Sorted, and Transformed iterators
- **Collection Management**: Support for different collection types with iterator creation
- **Filtering & Sorting**: Built-in support for filtering and sorting operations
- **Transformation**: Data transformation capabilities for iterator items
- **Caching**: In-memory caching for improved performance
- **Real-time Updates**: WebSocket support for real-time iterator updates
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
                    │ Iterator Service│
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

## Iterator Types

### 1. Slice Iterator

Iterates over a slice of items.

```go
items := []interface{}{"item1", "item2", "item3"}
iterator := NewSliceIterator(items)
```

### 2. Map Iterator

Iterates over a map, returning key-value pairs.

```go
items := map[string]interface{}{
    "key1": "value1",
    "key2": "value2",
}
iterator := NewMapIterator(items)
```

### 3. Channel Iterator

Iterates over a channel, processing items as they arrive.

```go
channel := make(chan interface{})
iterator := NewChannelIterator(channel)
```

### 4. Database Iterator

Iterates over database query results.

```go
query := map[string]interface{}{"status": "active"}
results := []interface{}{/* query results */}
iterator := NewDatabaseIterator(query, results)
```

### 5. File Iterator

Iterates over file lines.

```go
lines := []string{"line1", "line2", "line3"}
iterator := NewFileIterator("/path/to/file", lines)
```

### 6. Filtered Iterator

Wraps another iterator with filtering logic.

```go
baseIterator := NewSliceIterator(items)
filter := &MyFilter{}
iterator := NewFilteredIterator(baseIterator, filter)
```

### 7. Sorted Iterator

Wraps another iterator with sorting logic.

```go
baseIterator := NewSliceIterator(items)
sorter := &MySorter{}
iterator := NewSortedIterator(baseIterator, sorter)
```

### 8. Transformed Iterator

Wraps another iterator with transformation logic.

```go
baseIterator := NewSliceIterator(items)
transformer := &MyTransformer{}
iterator := NewTransformedIterator(baseIterator, transformer)
```

## API Endpoints

### Iterator Management

#### Create Iterator

```http
POST /api/v1/iterators
Content-Type: application/json

{
    "name": "my-iterator",
    "type": "slice",
    "data": ["item1", "item2", "item3"]
}
```

#### Get Iterator

```http
GET /api/v1/iterators/{name}
```

#### List Iterators

```http
GET /api/v1/iterators
```

#### Remove Iterator

```http
DELETE /api/v1/iterators/{name}
```

#### Get Iterator Statistics

```http
GET /api/v1/iterators/{name}/stats
```

### Iterator Operations

#### Iterate Over Items

```http
POST /api/v1/iterators/{name}/iterate
```

#### Reset Iterator

```http
POST /api/v1/iterators/{name}/reset
```

#### Close Iterator

```http
POST /api/v1/iterators/{name}/close
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

- `iterator_created`: When a new iterator is created
- `iterator_updated`: When an iterator is updated
- `iterator_deleted`: When an iterator is deleted
- `iterator_stats`: Iterator statistics updates

## Kafka Topics

- `iterator-events`: Iterator lifecycle events
- `iterator-commands`: Iterator operation commands
- `iterator-responses`: Iterator operation responses

## Configuration

The service is configured via `configs/config.yaml`:

```yaml
name: "Iterator Service"
version: "1.0.0"
max_iterators: 1000
timeout: "30m"
retry_count: 3

database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "iterator_db"

  mongodb:
    uri: "mongodb://localhost:27017"
    database: "iterator_db"

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
  topics: ["iterator-events"]

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

### Basic Slice Iterator

```go
// Create a slice iterator
items := []interface{}{"apple", "banana", "cherry"}
iterator := NewSliceIterator(items)

// Iterate over items
for iterator.HasNext() {
    item := iterator.Next()
    fmt.Println(item)
}
```

### Filtered Iterator

```go
// Create a filter
type EvenNumberFilter struct{}

func (f *EvenNumberFilter) Filter(item interface{}) bool {
    if num, ok := item.(int); ok {
        return num%2 == 0
    }
    return false
}

// Create filtered iterator
numbers := []interface{}{1, 2, 3, 4, 5, 6}
baseIterator := NewSliceIterator(numbers)
filter := &EvenNumberFilter{}
filteredIterator := NewFilteredIterator(baseIterator, filter)

// Iterate over filtered items
for filteredIterator.HasNext() {
    item := filteredIterator.Next()
    fmt.Println(item) // Will print: 2, 4, 6
}
```

### Sorted Iterator

```go
// Create a sorter
type StringSorter struct{}

func (s *StringSorter) Sort(items []interface{}) []interface{} {
    // Sort items alphabetically
    sort.Slice(items, func(i, j int) bool {
        return items[i].(string) < items[j].(string)
    })
    return items
}

// Create sorted iterator
words := []interface{}{"zebra", "apple", "banana"}
baseIterator := NewSliceIterator(words)
sorter := &StringSorter{}
sortedIterator := NewSortedIterator(baseIterator, sorter)

// Iterate over sorted items
for sortedIterator.HasNext() {
    item := sortedIterator.Next()
    fmt.Println(item) // Will print: apple, banana, zebra
}
```

### Transformed Iterator

```go
// Create a transformer
type UpperCaseTransformer struct{}

func (t *UpperCaseTransformer) Transform(item interface{}) interface{} {
    if str, ok := item.(string); ok {
        return strings.ToUpper(str)
    }
    return item
}

// Create transformed iterator
words := []interface{}{"hello", "world"}
baseIterator := NewSliceIterator(words)
transformer := &UpperCaseTransformer{}
transformedIterator := NewTransformedIterator(baseIterator, transformer)

// Iterate over transformed items
for transformedIterator.HasNext() {
    item := transformedIterator.Next()
    fmt.Println(item) // Will print: HELLO, WORLD
}
```

## Testing

### Unit Tests

```bash
go test ./internal/iterator/...
```

### Integration Tests

```bash
go test -tags=integration ./...
```

### Benchmark Tests

```bash
go test -bench=. ./internal/iterator/...
```

## Performance Considerations

1. **Memory Usage**: Iterators hold references to data, so large datasets should be processed in batches
2. **Concurrency**: Use channel iterators for concurrent processing
3. **Caching**: Enable caching for frequently accessed iterators
4. **Filtering**: Apply filters early to reduce processing overhead
5. **Sorting**: Consider using external sorting for large datasets

## Security

- JWT authentication for API access
- Rate limiting to prevent abuse
- CORS configuration for web clients
- Input validation for all endpoints
- Audit logging for compliance

## Monitoring

- Health checks for all components
- Metrics collection and export
- Performance monitoring
- Error tracking and alerting
- Resource usage monitoring

## Deployment

### Docker

```bash
docker build -t iterator-service .
docker run -p 8080:8080 iterator-service
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
export MYSQL_DATABASE=iterator_db
export MONGODB_URI=mongodb://localhost:27017
export MONGODB_DATABASE=iterator_db
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
- Basic iterator types
- REST API
- WebSocket support
- Kafka integration
- Database support
- Caching
- Monitoring
- Security features
