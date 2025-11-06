---
# Auto-generated front matter
Title: Tracing
LastUpdated: 2025-11-06T20:45:59.163620
Tags: []
Status: draft
---

# 🔍 Distributed Tracing: OpenTelemetry and Jaeger

> **Master distributed tracing for microservices observability and performance analysis**

## 📚 Concept

**Detailed Explanation:**
Distributed tracing is a critical observability technique that provides end-to-end visibility into request flows across distributed systems. It tracks individual requests as they traverse through multiple services, databases, and external APIs, creating a complete picture of the request's journey. This visibility is essential for understanding system behavior, identifying performance bottlenecks, debugging issues, and optimizing complex microservices architectures.

**Core Philosophy:**

- **End-to-End Visibility**: Track complete request journeys across all services
- **Context Propagation**: Maintain request context across service boundaries
- **Performance Analysis**: Measure latency and identify bottlenecks at each step
- **Error Tracking**: Trace error sources and propagation paths
- **Dependency Mapping**: Understand service interactions and dependencies
- **Root Cause Analysis**: Quickly identify the source of issues

**Why Distributed Tracing Matters:**

- **Microservices Complexity**: Essential for understanding complex service interactions
- **Performance Optimization**: Identify bottlenecks and optimize critical paths
- **Debugging**: Quickly locate and resolve issues in distributed systems
- **Service Dependencies**: Map and understand service relationships
- **SLA Monitoring**: Track request performance against service level agreements
- **Capacity Planning**: Understand resource usage patterns and scaling needs

### Key Features

**Detailed Feature Breakdown:**

**1. Request Flow Tracking:**

- **Trace Context**: Unique trace ID that follows requests across services
- **Span Hierarchy**: Parent-child relationships between operations
- **Service Boundaries**: Clear visibility into cross-service communication
- **Request Lifecycle**: Complete request journey from entry to exit
- **Async Operations**: Track asynchronous and background processing
- **External Dependencies**: Monitor calls to external services and APIs

**2. Latency Analysis:**

- **Operation Timing**: Precise timing for each operation and service call
- **Critical Path Analysis**: Identify the longest path through the system
- **Bottleneck Detection**: Find slow operations and services
- **Performance Baselines**: Establish performance expectations and thresholds
- **Trend Analysis**: Track performance changes over time
- **Comparative Analysis**: Compare performance across different requests

**3. Error Propagation:**

- **Error Tracking**: Capture and propagate error information across services
- **Error Context**: Rich context about where and why errors occurred
- **Error Classification**: Categorize errors by type and severity
- **Error Correlation**: Link related errors across different services
- **Error Impact Analysis**: Understand the impact of errors on user experience
- **Error Recovery**: Track error handling and recovery mechanisms

**4. Service Dependencies:**

- **Dependency Mapping**: Visual representation of service relationships
- **Service Health**: Monitor the health and availability of dependent services
- **Circuit Breaker Tracking**: Monitor circuit breaker states and failures
- **Load Balancer Visibility**: Track request routing and load distribution
- **Database Dependencies**: Monitor database connections and query performance
- **External Service Dependencies**: Track third-party service calls and responses

**5. Performance Optimization:**

- **Hot Path Analysis**: Identify the most frequently used code paths
- **Resource Utilization**: Track CPU, memory, and I/O usage per operation
- **Caching Effectiveness**: Monitor cache hit rates and effectiveness
- **Database Query Analysis**: Identify slow queries and optimization opportunities
- **Network Performance**: Track network latency and bandwidth usage
- **Concurrency Analysis**: Understand concurrent request patterns and bottlenecks

**6. Debugging and Root Cause Analysis:**

- **Request Correlation**: Link related operations and events
- **Context Preservation**: Maintain request context across all operations
- **Event Sequencing**: Understand the order of operations and events
- **State Tracking**: Monitor application state changes during request processing
- **Log Correlation**: Link traces with application logs for comprehensive debugging
- **Incident Analysis**: Quickly identify the root cause of production issues

**Discussion Questions & Answers:**

**Q1: How do you implement comprehensive distributed tracing in a microservices architecture?**

**Answer:** Comprehensive distributed tracing implementation:

**Instrumentation Strategy:**

- Instrument all services with OpenTelemetry SDKs for consistent tracing
- Implement automatic instrumentation for common frameworks and libraries
- Add custom instrumentation for business logic and critical operations
- Use correlation IDs to link traces across different systems and protocols
- Implement trace context propagation for HTTP, gRPC, and message queues

**Sampling and Performance:**

- Implement head-based sampling for high-volume services to control costs
- Use tail-based sampling for error cases to ensure all errors are traced
- Implement adaptive sampling based on service load and error rates
- Use trace sampling to balance observability with performance overhead
- Monitor tracing overhead and adjust sampling rates accordingly

**Storage and Retention:**

- Use distributed storage systems like Elasticsearch for trace storage
- Implement tiered storage with hot and cold data retention policies
- Use trace compression and aggregation to reduce storage costs
- Implement trace archiving for long-term compliance and analysis
- Monitor storage costs and optimize retention policies

**Analysis and Alerting:**

- Set up automated alerts for high latency and error rates
- Implement trace-based SLI/SLO monitoring for service reliability
- Use trace analysis to identify performance bottlenecks and optimization opportunities
- Implement automated anomaly detection for unusual trace patterns
- Create dashboards for trace-based performance monitoring

**Q2: What are the key considerations for implementing trace sampling in production systems?**

**Answer:** Trace sampling implementation considerations:

**Sampling Strategies:**

- **Head-based Sampling**: Make sampling decisions at the start of requests
- **Tail-based Sampling**: Sample based on request outcomes and performance
- **Adaptive Sampling**: Adjust sampling rates based on system load and error rates
- **Stratified Sampling**: Sample different types of requests at different rates
- **Cost-based Sampling**: Optimize sampling for cost while maintaining observability

**Performance Impact:**

- Monitor CPU and memory overhead of tracing instrumentation
- Use asynchronous trace export to avoid blocking request processing
- Implement trace batching to reduce network overhead
- Use local buffering to handle temporary network issues
- Monitor and optimize trace export performance

**Data Quality:**

- Ensure representative sampling across all services and request types
- Implement sampling consistency across related services
- Use deterministic sampling for reproducible results
- Implement sampling validation to ensure data quality
- Monitor sampling coverage and adjust rates as needed

**Cost Optimization:**

- Use different sampling rates for different environments and services
- Implement trace filtering to reduce noise and focus on important traces
- Use trace aggregation and summarization to reduce storage costs
- Implement trace compression and deduplication
- Monitor and optimize trace storage and processing costs

**Q3: How do you implement effective trace analysis and performance optimization?**

**Answer:** Trace analysis and performance optimization implementation:

**Performance Analysis:**

- Use trace analysis to identify the critical path through your system
- Implement automated bottleneck detection based on trace data
- Use trace correlation to understand the impact of slow services
- Implement performance regression detection using trace data
- Use trace analysis to optimize resource allocation and scaling

**Error Analysis:**

- Implement automated error pattern detection using trace data
- Use trace correlation to understand error propagation and impact
- Implement error rate monitoring and alerting based on trace data
- Use trace analysis to identify root causes of production issues
- Implement error impact analysis to prioritize fixes

**Capacity Planning:**

- Use trace data to understand resource usage patterns
- Implement capacity planning based on trace-based performance metrics
- Use trace analysis to identify scaling bottlenecks and opportunities
- Implement predictive scaling based on trace patterns
- Use trace data to optimize resource allocation and costs

**Service Optimization:**

- Use trace analysis to identify optimization opportunities
- Implement A/B testing based on trace performance data
- Use trace correlation to understand service dependencies and interactions
- Implement service health monitoring based on trace data
- Use trace analysis to optimize service communication patterns

## 🏗️ Distributed Tracing Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Distributed Tracing Stack               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  │      A      │  │      B      │  │      C      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              OpenTelemetry SDK                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Tracer    │  │   Span      │  │   Context   │ │ │
│  │  │   Provider  │  │   Processor │  │   Propagator│ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Trace Collection                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   OTLP      │  │   Jaeger    │  │   Zipkin    │ │ │
│  │  │   Exporter  │  │   Exporter  │  │   Exporter  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Jaeger    │  │   Zipkin    │  │   X-Ray     │     │
│  │  Collector  │  │  Collector  │  │  Collector  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Go Application with OpenTelemetry

```go
// main.go
package main

import (
    "context"
    "fmt"
    "net/http"
    "os"
    "time"

    "github.com/gin-gonic/gin"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.12.0"
    "go.opentelemetry.io/otel/trace"
    "go.uber.org/zap"
)

// Initialize OpenTelemetry
func initTracer() func() {
    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String("my-app"),
            semconv.ServiceVersionKey.String("1.0.0"),
            semconv.DeploymentEnvironmentKey.String(os.Getenv("ENVIRONMENT")),
        ),
    )
    if err != nil {
        panic(err)
    }

    // Create Jaeger exporter
    jaegerEndpoint := os.Getenv("JAEGER_ENDPOINT")
    if jaegerEndpoint == "" {
        jaegerEndpoint = "http://localhost:14268/api/traces"
    }

    jaegerExporter, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint(jaegerEndpoint),
    ))
    if err != nil {
        panic(err)
    }

    // Create OTLP exporter
    otlpEndpoint := os.Getenv("OTLP_ENDPOINT")
    if otlpEndpoint == "" {
        otlpEndpoint = "localhost:4317"
    }

    otlpExporter, err := otlptracegrpc.New(context.Background(),
        otlptracegrpc.WithEndpoint(otlpEndpoint),
        otlptracegrpc.WithInsecure(),
    )
    if err != nil {
        panic(err)
    }

    // Create trace provider
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(jaegerExporter),
        sdktrace.WithBatcher(otlpExporter),
        sdktrace.WithResource(res),
        sdktrace.WithSampler(sdktrace.TraceIDRatioBased(1.0)),
    )

    // Set global tracer provider
    otel.SetTracerProvider(tp)

    // Set global propagator
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))

    return func() {
        tp.Shutdown(context.Background())
    }
}

// Tracing middleware
func tracingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Extract trace context from headers
        ctx := otel.GetTextMapPropagator().Extract(
            c.Request.Context(),
            propagation.HeaderCarrier(c.Request.Header),
        )

        // Start span
        tracer := otel.Tracer("gin-server")
        ctx, span := tracer.Start(ctx, fmt.Sprintf("%s %s", c.Request.Method, c.FullPath()))
        defer span.End()

        // Add span attributes
        span.SetAttributes(
            attribute.String("http.method", c.Request.Method),
            attribute.String("http.url", c.Request.URL.String()),
            attribute.String("http.user_agent", c.Request.UserAgent()),
            attribute.String("http.remote_addr", c.ClientIP()),
        )

        // Add request ID to context
        requestID := c.GetHeader("X-Request-ID")
        if requestID == "" {
            requestID = span.SpanContext().TraceID().String()
        }
        span.SetAttributes(attribute.String("request.id", requestID))

        // Store context in request
        c.Request = c.Request.WithContext(ctx)

        // Process request
        c.Next()

        // Add response attributes
        span.SetAttributes(
            attribute.Int("http.status_code", c.Writer.Status()),
            attribute.Int("http.response_size", c.Writer.Size()),
        )

        // Add error if any
        if c.Writer.Status() >= 400 {
            span.SetAttributes(attribute.String("error", "true"))
        }
    }
}

// Business logic with tracing
type UserService struct {
    logger *zap.Logger
    tracer trace.Tracer
}

func NewUserService(logger *zap.Logger) *UserService {
    return &UserService{
        logger: logger,
        tracer: otel.Tracer("user-service"),
    }
}

func (s *UserService) GetUser(ctx context.Context, userID string) (*User, error) {
    // Start span
    ctx, span := s.tracer.Start(ctx, "UserService.GetUser")
    defer span.End()

    // Add span attributes
    span.SetAttributes(
        attribute.String("user.id", userID),
        attribute.String("operation", "get_user"),
    )

    // Simulate database call
    user, err := s.fetchUserFromDB(ctx, userID)
    if err != nil {
        span.SetAttributes(
            attribute.String("error", "true"),
            attribute.String("error.message", err.Error()),
        )
        s.logger.Error("Failed to get user",
            zap.String("user_id", userID),
            zap.Error(err),
        )
        return nil, err
    }

    // Add success attributes
    span.SetAttributes(
        attribute.String("user.email", user.Email),
        attribute.String("user.role", user.Role),
    )

    s.logger.Info("User retrieved successfully",
        zap.String("user_id", userID),
    )

    return user, nil
}

func (s *UserService) fetchUserFromDB(ctx context.Context, userID string) (*User, error) {
    // Start span
    ctx, span := s.tracer.Start(ctx, "UserService.fetchUserFromDB")
    defer span.End()

    // Add span attributes
    span.SetAttributes(
        attribute.String("db.operation", "select"),
        attribute.String("db.table", "users"),
        attribute.String("db.query", "SELECT * FROM users WHERE id = ?"),
    )

    // Simulate database operation
    time.Sleep(100 * time.Millisecond)

    if userID == "error" {
        span.SetAttributes(
            attribute.String("error", "true"),
            attribute.String("error.message", "database connection failed"),
        )
        return nil, fmt.Errorf("database connection failed")
    }

    // Add success attributes
    span.SetAttributes(
        attribute.String("db.rows_affected", "1"),
        attribute.String("db.duration_ms", "100"),
    )

    return &User{
        ID:    userID,
        Email: "user@example.com",
        Role:  "admin",
    }, nil
}

type User struct {
    ID    string `json:"id"`
    Email string `json:"email"`
    Role  string `json:"role"`
}

// HTTP handlers
func setupRoutes(logger *zap.Logger) *gin.Engine {
    r := gin.New()

    // Add middleware
    r.Use(tracingMiddleware())
    r.Use(gin.Recovery())

    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "timestamp": time.Now().UTC(),
        })
    })

    // User endpoints
    userService := NewUserService(logger)

    r.GET("/users/:id", func(c *gin.Context) {
        userID := c.Param("id")

        user, err := userService.GetUser(c.Request.Context(), userID)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "Failed to get user",
            })
            return
        }

        c.JSON(http.StatusOK, user)
    })

    return r
}

func main() {
    // Initialize tracing
    shutdown := initTracer()
    defer shutdown()

    // Setup logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Setup routes
    router := setupRoutes(logger)

    // Start server
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    logger.Info("Starting server",
        zap.String("port", port),
        zap.String("environment", os.Getenv("ENVIRONMENT")),
    )

    if err := router.Run(":" + port); err != nil {
        logger.Fatal("Failed to start server", zap.Error(err))
    }
}
```

### Jaeger Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686" # Jaeger UI
      - "14268:14268" # Jaeger collector HTTP
      - "14250:14250" # Jaeger collector gRPC
      - "4317:4317" # OTLP gRPC
      - "4318:4318" # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
      - ES_USERNAME=elastic
      - ES_PASSWORD=changeme
    depends_on:
      - elasticsearch

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=changeme
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  my-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces
      - OTLP_ENDPOINT=jaeger:4317
      - ENVIRONMENT=production
    depends_on:
      - jaeger

volumes:
  elasticsearch_data:
```

### OpenTelemetry Collector Configuration

```yaml
# otel-collector.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

  jaeger:
    protocols:
      grpc:
        endpoint: 0.0.0.0:14250
      thrift_http:
        endpoint: 0.0.0.0:14268

  zipkin:
    endpoint: 0.0.0.0:9411

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
    send_batch_max_size: 2048

  memory_limiter:
    limit_mib: 512
    spike_limit_mib: 128
    check_interval: 5s

  resource:
    attributes:
      - key: service.name
        value: "my-app"
        action: upsert
      - key: service.version
        value: "1.0.0"
        action: upsert
      - key: deployment.environment
        value: "production"
        action: upsert

  span:
    name:
      to_attributes:
        rules:
          - "^/api/v1/users/(?P<user_id>.*)"
        separator: ""

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  otlp:
    endpoint: jaeger:4317
    tls:
      insecure: true

  logging:
    loglevel: debug

  elasticsearch:
    endpoints:
      - http://elasticsearch:9200
    mapping:
      mode: ecs
    auth:
      authenticator: basicauth/client
    timeout: 30s

service:
  pipelines:
    traces:
      receivers: [otlp, jaeger, zipkin]
      processors: [memory_limiter, batch, resource, span]
      exporters: [jaeger, otlp, elasticsearch, logging]
```

### Kubernetes Tracing Configuration

```yaml
# jaeger-operator.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: production
  collector:
    maxReplicas: 5
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 100m
        memory: 128Mi
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      storage:
        storageClassName: fast-ssd
        size: 50Gi
      resources:
        limits:
          cpu: 1000m
          memory: 2Gi
        requests:
          cpu: 500m
          memory: 1Gi
  query:
    replicas: 2
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 100m
        memory: 128Mi
```

### OpenTelemetry Collector DaemonSet

```yaml
# otel-collector-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      serviceAccountName: otel-collector
      containers:
        - name: otel-collector
          image: otel/opentelemetry-collector-contrib:latest
          args:
            - --config=/etc/otel-collector-config.yaml
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          ports:
            - containerPort: 4317
              name: otlp-grpc
            - containerPort: 4318
              name: otlp-http
            - containerPort: 14268
              name: jaeger-thrift-http
            - containerPort: 14250
              name: jaeger-grpc
            - containerPort: 9411
              name: zipkin
          volumeMounts:
            - name: otel-collector-config
              mountPath: /etc/otel-collector-config.yaml
              subPath: otel-collector-config.yaml
          resources:
            limits:
              cpu: 500m
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 128Mi
      volumes:
        - name: otel-collector-config
          configMap:
            name: otel-collector-config
      tolerations:
        - operator: Exists
          effect: NoSchedule
```

### Trace Analysis Queries

```sql
-- Jaeger Query Language (JQL) examples
-- Find traces with high latency
SELECT * FROM traces WHERE duration > 1000ms

-- Find traces with errors
SELECT * FROM traces WHERE tags.error = true

-- Find traces by service
SELECT * FROM traces WHERE service.name = "my-app"

-- Find traces by operation
SELECT * FROM traces WHERE operation.name = "UserService.GetUser"

-- Find traces by user
SELECT * FROM traces WHERE tags.user.id = "12345"

-- Find traces with specific HTTP status
SELECT * FROM traces WHERE tags.http.status_code = 500

-- Find traces with database operations
SELECT * FROM traces WHERE tags.db.operation = "select"

-- Find traces with specific error message
SELECT * FROM traces WHERE tags.error.message = "database connection failed"
```

### Performance Analysis

```go
// Performance analysis with tracing
func (s *UserService) GetUserWithAnalysis(ctx context.Context, userID string) (*User, error) {
    // Start span
    ctx, span := s.tracer.Start(ctx, "UserService.GetUserWithAnalysis")
    defer span.End()

    // Add span attributes
    span.SetAttributes(
        attribute.String("user.id", userID),
        attribute.String("operation", "get_user_with_analysis"),
    )

    // Measure database call
    start := time.Now()
    user, err := s.fetchUserFromDB(ctx, userID)
    dbDuration := time.Since(start)

    // Add performance attributes
    span.SetAttributes(
        attribute.Int64("db.duration_ms", dbDuration.Milliseconds()),
        attribute.String("db.duration", dbDuration.String()),
    )

    if err != nil {
        span.SetAttributes(
            attribute.String("error", "true"),
            attribute.String("error.message", err.Error()),
        )
        return nil, err
    }

    // Measure business logic
    start = time.Now()
    err = s.processUserData(ctx, user)
    processDuration := time.Since(start)

    // Add performance attributes
    span.SetAttributes(
        attribute.Int64("process.duration_ms", processDuration.Milliseconds()),
        attribute.String("process.duration", processDuration.String()),
    )

    if err != nil {
        span.SetAttributes(
            attribute.String("error", "true"),
            attribute.String("error.message", err.Error()),
        )
        return nil, err
    }

    // Add success attributes
    span.SetAttributes(
        attribute.String("user.email", user.Email),
        attribute.String("user.role", user.Role),
        attribute.Int64("total.duration_ms", dbDuration.Milliseconds()+processDuration.Milliseconds()),
    )

    return user, nil
}

func (s *UserService) processUserData(ctx context.Context, user *User) error {
    // Start span
    ctx, span := s.tracer.Start(ctx, "UserService.processUserData")
    defer span.End()

    // Simulate processing
    time.Sleep(50 * time.Millisecond)

    // Add processing attributes
    span.SetAttributes(
        attribute.String("process.type", "data_validation"),
        attribute.String("process.status", "completed"),
    )

    return nil
}
```

## 🚀 Best Practices

### 1. Span Naming

```go
// Use consistent span naming
ctx, span := tracer.Start(ctx, "ServiceName.OperationName")
ctx, span := tracer.Start(ctx, "UserService.GetUser")
ctx, span := tracer.Start(ctx, "Database.Select")
```

### 2. Attribute Management

```go
// Use meaningful attributes
span.SetAttributes(
    attribute.String("user.id", userID),
    attribute.String("db.operation", "select"),
    attribute.String("http.status_code", "200"),
)
```

### 3. Error Handling

```go
// Add error information to spans
if err != nil {
    span.SetAttributes(
        attribute.String("error", "true"),
        attribute.String("error.message", err.Error()),
    )
}
```

## 🏢 Industry Insights

### Tracing Usage Patterns

- **Microservices**: Service-to-service communication
- **Performance Analysis**: Latency and bottleneck identification
- **Error Tracking**: Root cause analysis
- **Dependency Mapping**: Service interaction visualization

### Enterprise Tracing Strategy

- **Sampling**: Cost-effective tracing
- **Storage**: Long-term trace retention
- **Security**: Trace data protection
- **Compliance**: Audit trail requirements

## 🎯 Interview Questions

### Basic Level

1. **What is distributed tracing?**

   - Request flow tracking
   - Performance analysis
   - Error propagation
   - Service dependencies

2. **What is OpenTelemetry?**

   - Observability framework
   - Vendor-neutral
   - Multiple languages
   - Standardized APIs

3. **What is a span?**
   - Single operation
   - Start and end time
   - Attributes and events
   - Parent-child relationships

### Intermediate Level

4. **How do you implement distributed tracing?**

   ```go
   ctx, span := tracer.Start(ctx, "OperationName")
   defer span.End()
   span.SetAttributes(attribute.String("key", "value"))
   ```

5. **How do you handle trace context propagation?**

   - HTTP headers
   - gRPC metadata
   - Message queues
   - Database connections

6. **How do you implement trace sampling?**
   - Head-based sampling
   - Tail-based sampling
   - Adaptive sampling
   - Cost optimization

### Advanced Level

7. **How do you implement trace analysis?**

   - Performance profiling
   - Error analysis
   - Dependency mapping
   - SLA monitoring

8. **How do you handle trace storage?**

   - Elasticsearch
   - Cassandra
   - S3
   - Cost optimization

9. **How do you implement trace security?**
   - Data encryption
   - Access control
   - PII filtering
   - Compliance

---

**Next**: [Alerting](Alerting.md) - Alert management, notification channels, escalation policies
