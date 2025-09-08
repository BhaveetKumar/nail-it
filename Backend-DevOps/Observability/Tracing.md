# 🔍 Distributed Tracing: OpenTelemetry and Jaeger

> **Master distributed tracing for microservices observability and performance analysis**

## 📚 Concept

Distributed tracing is a method of tracking requests as they flow through multiple services in a distributed system. It provides visibility into the path of requests, latency at each step, and helps identify bottlenecks and failures across service boundaries.

### Key Features

- **Request Flow Tracking**: Follow requests across services
- **Latency Analysis**: Measure performance at each step
- **Error Propagation**: Track error sources and paths
- **Service Dependencies**: Map service interactions
- **Performance Optimization**: Identify bottlenecks
- **Debugging**: Root cause analysis

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

**Next**: [Alerting](./Alerting.md) - Alert management, notification channels, escalation policies
