# Logging Patterns and Best Practices

## Table of Contents
1. [Introduction](#introduction)
2. [Logging Levels](#logging-levels)
3. [Structured Logging](#structured-logging)
4. [Log Aggregation Patterns](#log-aggregation-patterns)
5. [Performance Considerations](#performance-considerations)
6. [Security and Compliance](#security-and-compliance)
7. [Golang Implementation](#golang-implementation)
8. [Microservices Logging](#microservices-logging)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

## Introduction

Logging is a critical component of observability that provides insights into application behavior, performance, and issues. Effective logging patterns are essential for debugging, monitoring, and maintaining distributed systems.

## Logging Levels

### Standard Levels
- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about program execution
- **WARN**: Warning messages for potential issues
- **ERROR**: Error events that might still allow the application to continue
- **FATAL**: Very severe error events that will presumably lead to application termination

### Level Selection Guidelines
```go
// Good: Appropriate level usage
logger.Debug("Processing user request", "userID", userID, "requestID", requestID)
logger.Info("User login successful", "userID", userID)
logger.Warn("Rate limit approaching", "userID", userID, "current", current, "limit", limit)
logger.Error("Database connection failed", "error", err, "retry", retryCount)
logger.Fatal("Cannot start application", "error", err)
```

## Structured Logging

### JSON Format
```go
type LogEntry struct {
    Timestamp string            `json:"timestamp"`
    Level     string            `json:"level"`
    Message   string            `json:"message"`
    Fields    map[string]interface{} `json:"fields"`
    TraceID   string            `json:"trace_id,omitempty"`
    SpanID    string            `json:"span_id,omitempty"`
}

// Example log entry
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "message": "Payment processed successfully",
    "fields": {
        "user_id": "12345",
        "amount": 100.50,
        "currency": "USD",
        "payment_method": "credit_card"
    },
    "trace_id": "abc123def456",
    "span_id": "span789"
}
```

### Key-Value Pairs
```go
// Using slog (Go 1.21+)
logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
logger.Info("Payment processed",
    "user_id", userID,
    "amount", amount,
    "currency", currency,
    "payment_method", paymentMethod,
    "trace_id", traceID)
```

## Log Aggregation Patterns

### Centralized Logging
```go
// Log shipping to centralized system
type LogShipper struct {
    client    *http.Client
    endpoint  string
    batchSize int
    logs      chan LogEntry
}

func (ls *LogShipper) ShipLogs() {
    batch := make([]LogEntry, 0, ls.batchSize)
    ticker := time.NewTicker(5 * time.Second)
    
    for {
        select {
        case log := <-ls.logs:
            batch = append(batch, log)
            if len(batch) >= ls.batchSize {
                ls.sendBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                ls.sendBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

### Log Sampling
```go
type SampledLogger struct {
    logger    Logger
    sampleRate float64
    rng       *rand.Rand
}

func (sl *SampledLogger) Info(msg string, fields ...interface{}) {
    if sl.rng.Float64() < sl.sampleRate {
        sl.logger.Info(msg, fields...)
    }
}
```

## Performance Considerations

### Asynchronous Logging
```go
type AsyncLogger struct {
    logs    chan LogEntry
    workers int
    logger  Logger
}

func NewAsyncLogger(workers int, bufferSize int) *AsyncLogger {
    al := &AsyncLogger{
        logs:    make(chan LogEntry, bufferSize),
        workers: workers,
    }
    
    // Start worker goroutines
    for i := 0; i < workers; i++ {
        go al.worker()
    }
    
    return al
}

func (al *AsyncLogger) worker() {
    for log := range al.logs {
        al.logger.Write(log)
    }
}
```

### Log Buffering
```go
type BufferedLogger struct {
    buffer    []LogEntry
    maxSize   int
    flushTime time.Duration
    mutex     sync.Mutex
}

func (bl *BufferedLogger) Log(entry LogEntry) {
    bl.mutex.Lock()
    defer bl.mutex.Unlock()
    
    bl.buffer = append(bl.buffer, entry)
    
    if len(bl.buffer) >= bl.maxSize {
        bl.flush()
    }
}
```

## Security and Compliance

### Sensitive Data Masking
```go
type SecureLogger struct {
    logger Logger
    masker DataMasker
}

type DataMasker struct {
    patterns []*regexp.Regexp
    masks    []string
}

func (dm *DataMasker) Mask(data string) string {
    result := data
    for i, pattern := range dm.patterns {
        result = pattern.ReplaceAllString(result, dm.masks[i])
    }
    return result
}

// Usage
masker := DataMasker{
    patterns: []*regexp.Regexp{
        regexp.MustCompile(`"password":\s*"[^"]*"`),
        regexp.MustCompile(`"ssn":\s*"[^"]*"`),
        regexp.MustCompile(`"credit_card":\s*"[^"]*"`),
    },
    masks: []string{
        `"password": "***"`,
        `"ssn": "***-**-****"`,
        `"credit_card": "****-****-****-****"`,
    },
}
```

### Audit Logging
```go
type AuditLogger struct {
    logger Logger
}

func (al *AuditLogger) LogAccess(userID, resource, action string, success bool) {
    al.logger.Info("Access attempt",
        "user_id", userID,
        "resource", resource,
        "action", action,
        "success", success,
        "timestamp", time.Now().UTC(),
        "type", "audit")
}
```

## Golang Implementation

### Using slog (Go 1.21+)
```go
package main

import (
    "log/slog"
    "os"
    "time"
)

func main() {
    // JSON handler for structured logging
    handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
        AddSource: true,
    })
    
    logger := slog.New(handler)
    
    // Set as default logger
    slog.SetDefault(logger)
    
    // Usage
    slog.Info("Application started", "port", 8080)
    slog.Error("Database connection failed", "error", "connection timeout")
}
```

### Custom Logger Implementation
```go
type Logger interface {
    Debug(msg string, fields ...Field)
    Info(msg string, fields ...Field)
    Warn(msg string, fields ...Field)
    Error(msg string, fields ...Field)
    Fatal(msg string, fields ...Field)
}

type Field struct {
    Key   string
    Value interface{}
}

type SimpleLogger struct {
    level  Level
    writer io.Writer
    mutex  sync.Mutex
}

func (l *SimpleLogger) Info(msg string, fields ...Field) {
    if l.level <= InfoLevel {
        l.log(InfoLevel, msg, fields...)
    }
}

func (l *SimpleLogger) log(level Level, msg string, fields ...Field) {
    l.mutex.Lock()
    defer l.mutex.Unlock()
    
    entry := LogEntry{
        Timestamp: time.Now().UTC().Format(time.RFC3339),
        Level:     level.String(),
        Message:   msg,
        Fields:    make(map[string]interface{}),
    }
    
    for _, field := range fields {
        entry.Fields[field.Key] = field.Value
    }
    
    json.NewEncoder(l.writer).Encode(entry)
}
```

## Microservices Logging

### Correlation IDs
```go
type ContextLogger struct {
    logger Logger
    ctx    context.Context
}

func NewContextLogger(ctx context.Context, logger Logger) *ContextLogger {
    return &ContextLogger{
        logger: logger,
        ctx:    ctx,
    }
}

func (cl *ContextLogger) Info(msg string, fields ...Field) {
    // Extract correlation ID from context
    if traceID := cl.ctx.Value("trace_id"); traceID != nil {
        fields = append(fields, Field{Key: "trace_id", Value: traceID})
    }
    
    cl.logger.Info(msg, fields...)
}
```

### Service Mesh Integration
```go
type ServiceMeshLogger struct {
    logger    Logger
    service   string
    version   string
    instance  string
}

func (sml *ServiceMeshLogger) LogRequest(method, path string, statusCode int, duration time.Duration) {
    sml.logger.Info("HTTP request",
        "method", method,
        "path", path,
        "status_code", statusCode,
        "duration_ms", duration.Milliseconds(),
        "service", sml.service,
        "version", sml.version,
        "instance", sml.instance)
}
```

## Best Practices

### 1. Use Structured Logging
```go
// Bad
logger.Info("User 12345 logged in from IP 192.168.1.1")

// Good
logger.Info("User login successful",
    "user_id", userID,
    "ip_address", ipAddress,
    "user_agent", userAgent,
    "login_method", "password")
```

### 2. Include Context
```go
// Always include relevant context
logger.Error("Payment processing failed",
    "payment_id", paymentID,
    "user_id", userID,
    "amount", amount,
    "error", err.Error(),
    "retry_count", retryCount)
```

### 3. Use Appropriate Log Levels
```go
// Debug: Detailed flow information
logger.Debug("Processing payment step 1", "payment_id", paymentID)

// Info: Important business events
logger.Info("Payment completed", "payment_id", paymentID, "amount", amount)

// Warn: Recoverable issues
logger.Warn("Rate limit exceeded", "user_id", userID, "retry_after", retryAfter)

// Error: Unrecoverable issues
logger.Error("Database connection lost", "error", err)
```

### 4. Avoid Logging Sensitive Data
```go
// Bad
logger.Info("User data", "password", password, "ssn", ssn)

// Good
logger.Info("User data", "user_id", userID, "email", email)
```

## Common Pitfalls

### 1. Logging Too Much
```go
// Bad: Logging every iteration
for i, item := range items {
    logger.Debug("Processing item", "index", i, "item", item)
    // process item
}

// Good: Log summary
logger.Debug("Processing items", "count", len(items))
for i, item := range items {
    // process item
}
logger.Debug("Items processed", "count", len(items))
```

### 2. Inconsistent Log Format
```go
// Bad: Inconsistent formatting
logger.Info("User login: " + userID)
logger.Info("Payment: %s amount: %f", paymentID, amount)

// Good: Consistent structured logging
logger.Info("User login", "user_id", userID)
logger.Info("Payment processed", "payment_id", paymentID, "amount", amount)
```

### 3. Blocking on Log Writes
```go
// Bad: Synchronous logging in hot path
func ProcessPayment(payment Payment) {
    logger.Info("Processing payment", "payment_id", payment.ID) // Blocks
    // process payment
}

// Good: Asynchronous logging
func ProcessPayment(payment Payment) {
    asyncLogger.Info("Processing payment", "payment_id", payment.ID) // Non-blocking
    // process payment
}
```

## Conclusion

Effective logging patterns are essential for building maintainable and observable systems. Key principles include:

1. **Structured Logging**: Use consistent, machine-readable formats
2. **Appropriate Levels**: Choose the right log level for each message
3. **Context**: Include relevant context in every log entry
4. **Performance**: Use asynchronous logging for high-throughput systems
5. **Security**: Mask sensitive data and implement audit logging
6. **Correlation**: Use trace IDs to track requests across services

These patterns will help you build robust logging systems that provide valuable insights into your application's behavior and performance.
