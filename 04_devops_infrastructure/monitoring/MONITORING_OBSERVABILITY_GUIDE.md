---
# Auto-generated front matter
Title: Monitoring Observability Guide
LastUpdated: 2025-11-06T20:45:59.157390
Tags: []
Status: draft
---

# 📊 Monitoring & Observability Guide

> **Complete guide to monitoring, logging, and observability for backend systems**

## 📚 Table of Contents

1. [Monitoring Fundamentals](#-monitoring-fundamentals)
2. [Metrics Collection](#-metrics-collection)
3. [Logging Strategies](#-logging-strategies)
4. [Distributed Tracing](#-distributed-tracing)
5. [Alerting Systems](#-alerting-systems)
6. [Performance Monitoring](#-performance-monitoring)
7. [Health Checks](#-health-checks)
8. [Observability Tools](#-observability-tools)

---

## 🎯 Monitoring Fundamentals

### Three Pillars of Observability

```
┌─────────────────────────────────────────────────────────────┐
│                Three Pillars of Observability              │
├─────────────────────────────────────────────────────────────┤
│  Metrics                    Logs                    Traces  │
│  ┌─────────────────┐       ┌─────────────────┐     ┌───────┐ │
│  │ • Counters      │       │ • Structured    │     │ • Spans│ │
│  │ • Gauges        │       │ • Unstructured  │     │ • Tags │ │
│  │ • Histograms    │       │ • Events        │     │ • Bags │ │
│  │ • Summaries     │       │ • Context       │     │ • Links│ │
│  └─────────────────┘       └─────────────────┘     └───────┘ │
│         │                           │                   │     │
│         └───────────────────────────┼───────────────────┘     │
│                                     │                         │
│            ┌────────────────────────▼─────────────────────────┐ │
│            │           Observability Platform                 │ │
│            │  • Prometheus + Grafana                         │ │
│            │  • ELK Stack (Elasticsearch, Logstash, Kibana)  │ │
│            │  • Jaeger / Zipkin                              │ │
│            │  • Custom Dashboards                            │ │
│            └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Monitoring Strategy

1. **What to Monitor**: Key business and technical metrics
2. **How to Monitor**: Tools and techniques for data collection
3. **When to Alert**: Thresholds and alerting rules
4. **Who to Notify**: Escalation procedures and on-call rotation

---

## 📈 Metrics Collection

### 1. Application Metrics

```go
// Prometheus Metrics Collection
package main

import (
    "net/http"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

// Custom Metrics
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status_code"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
    
    activeConnections = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Number of active connections",
        },
    )
    
    businessMetrics = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "business_events_total",
            Help: "Total number of business events",
        },
        []string{"event_type", "status"},
    )
)

func init() {
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(httpRequestDuration)
    prometheus.MustRegister(activeConnections)
    prometheus.MustRegister(businessMetrics)
}

// Metrics Middleware
func MetricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap response writer to capture status code
        ww := &ResponseWriterWrapper{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(ww, r)
        
        duration := time.Since(start).Seconds()
        
        // Record metrics
        httpRequestsTotal.WithLabelValues(
            r.Method,
            r.URL.Path,
            fmt.Sprintf("%d", ww.statusCode),
        ).Inc()
        
        httpRequestDuration.WithLabelValues(
            r.Method,
            r.URL.Path,
        ).Observe(duration)
    })
}

// Business Metrics
func RecordUserRegistration(status string) {
    businessMetrics.WithLabelValues("user_registration", status).Inc()
}

func RecordOrderCreated(status string) {
    businessMetrics.WithLabelValues("order_created", status).Inc()
}

func RecordPaymentProcessed(status string) {
    businessMetrics.WithLabelValues("payment_processed", status).Inc()
}

// Custom Metrics Collector
type CustomCollector struct {
    userCount    prometheus.Gauge
    orderCount   prometheus.Gauge
    revenue      prometheus.Gauge
}

func NewCustomCollector() *CustomCollector {
    return &CustomCollector{
        userCount: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "users_total",
            Help: "Total number of users",
        }),
        orderCount: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "orders_total",
            Help: "Total number of orders",
        }),
        revenue: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "revenue_total",
            Help: "Total revenue",
        }),
    }
}

func (cc *CustomCollector) Describe(ch chan<- *prometheus.Desc) {
    cc.userCount.Describe(ch)
    cc.orderCount.Describe(ch)
    cc.revenue.Describe(ch)
}

func (cc *CustomCollector) Collect(ch chan<- prometheus.Metric) {
    // Update metrics from database
    userCount := getUserCount()
    orderCount := getOrderCount()
    revenue := getTotalRevenue()
    
    cc.userCount.Set(float64(userCount))
    cc.orderCount.Set(float64(orderCount))
    cc.revenue.Set(revenue)
    
    cc.userCount.Collect(ch)
    cc.orderCount.Collect(ch)
    cc.revenue.Collect(ch)
}

func main() {
    // Register custom collector
    prometheus.MustRegister(NewCustomCollector())
    
    // Setup HTTP server
    mux := http.NewServeMux()
    mux.Handle("/metrics", promhttp.Handler())
    mux.Handle("/", MetricsMiddleware(http.HandlerFunc(handler)))
    
    http.ListenAndServe(":8080", mux)
}
```

### 2. System Metrics

```go
// System Metrics Collection
package main

import (
    "runtime"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/shirou/gopsutil/v3/cpu"
    "github.com/shirou/gopsutil/v3/mem"
    "github.com/shirou/gopsutil/v3/disk"
)

type SystemMetrics struct {
    cpuUsage    prometheus.Gauge
    memoryUsage prometheus.Gauge
    diskUsage   prometheus.Gauge
    goroutines  prometheus.Gauge
    gcDuration  prometheus.Histogram
}

func NewSystemMetrics() *SystemMetrics {
    return &SystemMetrics{
        cpuUsage: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "system_cpu_usage_percent",
            Help: "CPU usage percentage",
        }),
        memoryUsage: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "system_memory_usage_percent",
            Help: "Memory usage percentage",
        }),
        diskUsage: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "system_disk_usage_percent",
            Help: "Disk usage percentage",
        }),
        goroutines: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "go_goroutines",
            Help: "Number of goroutines",
        }),
        gcDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name: "go_gc_duration_seconds",
            Help: "GC duration in seconds",
        }),
    }
}

func (sm *SystemMetrics) Collect() {
    // CPU usage
    cpuPercent, _ := cpu.Percent(time.Second, false)
    if len(cpuPercent) > 0 {
        sm.cpuUsage.Set(cpuPercent[0])
    }
    
    // Memory usage
    mem, _ := mem.VirtualMemory()
    sm.memoryUsage.Set(mem.UsedPercent)
    
    // Disk usage
    disk, _ := disk.Usage("/")
    sm.diskUsage.Set(disk.UsedPercent)
    
    // Go runtime metrics
    sm.goroutines.Set(float64(runtime.NumGoroutine()))
    
    // GC metrics
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    sm.gcDuration.Observe(float64(m.PauseTotalNs) / 1e9)
}

func (sm *SystemMetrics) Register() {
    prometheus.MustRegister(sm.cpuUsage)
    prometheus.MustRegister(sm.memoryUsage)
    prometheus.MustRegister(sm.diskUsage)
    prometheus.MustRegister(sm.goroutines)
    prometheus.MustRegister(sm.gcDuration)
}

// Start metrics collection
func StartMetricsCollection() {
    sm := NewSystemMetrics()
    sm.Register()
    
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        
        for range ticker.C {
            sm.Collect()
        }
    }()
}
```

---

## 📝 Logging Strategies

### 1. Structured Logging

```go
// Structured Logging with Logrus
package main

import (
    "context"
    "os"
    
    "github.com/sirupsen/logrus"
)

type Logger struct {
    *logrus.Logger
}

func NewLogger() *Logger {
    logger := logrus.New()
    
    // Set JSON formatter for structured logging
    logger.SetFormatter(&logrus.JSONFormatter{
        TimestampFormat: "2006-01-02T15:04:05.000Z07:00",
        FieldMap: logrus.FieldMap{
            logrus.FieldKeyTime:  "timestamp",
            logrus.FieldKeyLevel: "level",
            logrus.FieldKeyMsg:   "message",
        },
    })
    
    // Set log level from environment
    level := os.Getenv("LOG_LEVEL")
    if level == "" {
        level = "info"
    }
    
    logLevel, err := logrus.ParseLevel(level)
    if err != nil {
        logLevel = logrus.InfoLevel
    }
    
    logger.SetLevel(logLevel)
    logger.SetOutput(os.Stdout)
    
    return &Logger{Logger: logger}
}

// Context-aware logging
func (l *Logger) WithContext(ctx context.Context) *logrus.Entry {
    return l.Logger.WithContext(ctx)
}

func (l *Logger) WithFields(fields logrus.Fields) *logrus.Entry {
    return l.Logger.WithFields(fields)
}

// Business event logging
func (l *Logger) LogUserRegistration(ctx context.Context, userID int, email string) {
    l.WithContext(ctx).WithFields(logrus.Fields{
        "event":     "user_registration",
        "user_id":   userID,
        "email":     email,
        "timestamp": time.Now().Unix(),
    }).Info("User registered successfully")
}

func (l *Logger) LogOrderCreated(ctx context.Context, orderID int, userID int, amount float64) {
    l.WithContext(ctx).WithFields(logrus.Fields{
        "event":     "order_created",
        "order_id":  orderID,
        "user_id":   userID,
        "amount":    amount,
        "timestamp": time.Now().Unix(),
    }).Info("Order created successfully")
}

func (l *Logger) LogPaymentProcessed(ctx context.Context, paymentID int, orderID int, amount float64, status string) {
    l.WithContext(ctx).WithFields(logrus.Fields{
        "event":      "payment_processed",
        "payment_id": paymentID,
        "order_id":   orderID,
        "amount":     amount,
        "status":     status,
        "timestamp":  time.Now().Unix(),
    }).Info("Payment processed")
}

// Error logging with stack trace
func (l *Logger) LogError(ctx context.Context, err error, message string, fields logrus.Fields) {
    l.WithContext(ctx).WithFields(fields).WithError(err).Error(message)
}

// Performance logging
func (l *Logger) LogPerformance(ctx context.Context, operation string, duration time.Duration, fields logrus.Fields) {
    l.WithContext(ctx).WithFields(logrus.Fields{
        "operation": operation,
        "duration":  duration.Milliseconds(),
        "timestamp": time.Now().Unix(),
    }).WithFields(fields).Info("Performance metric")
}

// Usage in application
func (us *UserService) CreateUser(ctx context.Context, req CreateUserRequest) (*User, error) {
    logger := NewLogger()
    
    logger.WithContext(ctx).WithFields(logrus.Fields{
        "operation": "create_user",
        "email":     req.Email,
    }).Info("Starting user creation")
    
    start := time.Now()
    
    // Validate request
    if req.Name == "" || req.Email == "" {
        logger.LogError(ctx, ErrInvalidRequest, "Invalid user creation request", logrus.Fields{
            "name":  req.Name,
            "email": req.Email,
        })
        return nil, ErrInvalidRequest
    }
    
    // Check if user exists
    existingUser, err := us.repo.FindByEmail(ctx, req.Email)
    if err != nil && err != ErrUserNotFound {
        logger.LogError(ctx, err, "Failed to check existing user", logrus.Fields{
            "email": req.Email,
        })
        return nil, err
    }
    
    if existingUser != nil {
        logger.WithContext(ctx).WithFields(logrus.Fields{
            "email": req.Email,
        }).Warn("User already exists")
        return nil, ErrUserAlreadyExists
    }
    
    // Create user
    user := &User{
        Name:      req.Name,
        Email:     req.Email,
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    if err := us.repo.Save(ctx, user); err != nil {
        logger.LogError(ctx, err, "Failed to save user", logrus.Fields{
            "email": req.Email,
        })
        return nil, err
    }
    
    duration := time.Since(start)
    logger.LogPerformance(ctx, "create_user", duration, logrus.Fields{
        "user_id": user.ID,
        "email":   user.Email,
    })
    
    logger.LogUserRegistration(ctx, user.ID, user.Email)
    
    return user, nil
}
```

### 2. Log Aggregation

```go
// Log Aggregation with Fluentd
package main

import (
    "context"
    "encoding/json"
    "net/http"
    "time"
    
    "github.com/sirupsen/logrus"
)

type FluentdLogger struct {
    *logrus.Logger
    fluentdURL string
    client     *http.Client
}

func NewFluentdLogger(fluentdURL string) *FluentdLogger {
    logger := logrus.New()
    logger.SetFormatter(&logrus.JSONFormatter{})
    
    return &FluentdLogger{
        Logger:     logger,
        fluentdURL: fluentdURL,
        client:     &http.Client{Timeout: 5 * time.Second},
    }
}

func (fl *FluentdLogger) SendToFluentd(tag string, data map[string]interface{}) error {
    payload := map[string]interface{}{
        tag: data,
    }
    
    jsonData, err := json.Marshal(payload)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequest("POST", fl.fluentdURL, bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    
    req.Header.Set("Content-Type", "application/json")
    
    resp, err := fl.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("fluentd returned status %d", resp.StatusCode)
    }
    
    return nil
}

func (fl *FluentdLogger) LogWithFluentd(ctx context.Context, level logrus.Level, message string, fields logrus.Fields) {
    // Log locally
    entry := fl.WithContext(ctx).WithFields(fields)
    switch level {
    case logrus.DebugLevel:
        entry.Debug(message)
    case logrus.InfoLevel:
        entry.Info(message)
    case logrus.WarnLevel:
        entry.Warn(message)
    case logrus.ErrorLevel:
        entry.Error(message)
    case logrus.FatalLevel:
        entry.Fatal(message)
    }
    
    // Send to Fluentd
    data := map[string]interface{}{
        "timestamp": time.Now().Unix(),
        "level":     level.String(),
        "message":   message,
        "fields":    fields,
    }
    
    if err := fl.SendToFluentd("app.logs", data); err != nil {
        fl.WithError(err).Error("Failed to send log to Fluentd")
    }
}
```

---

## 🔍 Distributed Tracing

### 1. OpenTelemetry Integration

```go
// Distributed Tracing with OpenTelemetry
package main

import (
    "context"
    "net/http"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/trace"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
)

func initTracer() func() {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint("http://localhost:14268/api/traces")))
    if err != nil {
        panic(err)
    }
    
    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String("user-service"),
            semconv.ServiceVersionKey.String("1.0.0"),
        ),
    )
    if err != nil {
        panic(err)
    }
    
    // Create tracer provider
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exp),
        sdktrace.WithResource(res),
    )
    
    otel.SetTracerProvider(tp)
    
    return func() {
        tp.Shutdown(context.Background())
    }
}

// Tracing middleware
func TracingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tracer := otel.Tracer("user-service")
        
        ctx, span := tracer.Start(r.Context(), r.URL.Path)
        defer span.End()
        
        // Add attributes
        span.SetAttributes(
            attribute.String("http.method", r.Method),
            attribute.String("http.url", r.URL.String()),
            attribute.String("http.user_agent", r.UserAgent()),
        )
        
        // Wrap response writer
        ww := &ResponseWriterWrapper{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(ww, r.WithContext(ctx))
        
        // Add response attributes
        span.SetAttributes(
            attribute.Int("http.status_code", ww.statusCode),
        )
        
        // Set span status
        if ww.statusCode >= 400 {
            span.SetStatus(codes.Error, "HTTP error")
        }
    })
}

// Service with tracing
func (us *UserService) CreateUser(ctx context.Context, req CreateUserRequest) (*User, error) {
    tracer := otel.Tracer("user-service")
    ctx, span := tracer.Start(ctx, "UserService.CreateUser")
    defer span.End()
    
    // Add span attributes
    span.SetAttributes(
        attribute.String("user.email", req.Email),
        attribute.String("user.name", req.Name),
    )
    
    // Validate request
    if req.Name == "" || req.Email == "" {
        span.SetStatus(codes.Error, "Invalid request")
        span.RecordError(ErrInvalidRequest)
        return nil, ErrInvalidRequest
    }
    
    // Check existing user
    ctx, checkSpan := tracer.Start(ctx, "UserService.CheckExistingUser")
    existingUser, err := us.repo.FindByEmail(ctx, req.Email)
    checkSpan.End()
    
    if err != nil && err != ErrUserNotFound {
        span.SetStatus(codes.Error, "Database error")
        span.RecordError(err)
        return nil, err
    }
    
    if existingUser != nil {
        span.SetAttributes(attribute.Bool("user.exists", true))
        span.SetStatus(codes.Error, "User already exists")
        return nil, ErrUserAlreadyExists
    }
    
    // Create user
    ctx, createSpan := tracer.Start(ctx, "UserService.CreateUserInDB")
    user := &User{
        Name:      req.Name,
        Email:     req.Email,
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    if err := us.repo.Save(ctx, user); err != nil {
        createSpan.SetStatus(codes.Error, "Failed to save user")
        createSpan.RecordError(err)
        createSpan.End()
        span.SetStatus(codes.Error, "Database error")
        span.RecordError(err)
        return nil, err
    }
    createSpan.End()
    
    // Add success attributes
    span.SetAttributes(
        attribute.Int("user.id", user.ID),
        attribute.String("user.status", user.Status),
    )
    
    return user, nil
}

// Database tracing
func (ur *UserRepository) Save(ctx context.Context, user *User) error {
    tracer := otel.Tracer("user-service")
    ctx, span := tracer.Start(ctx, "UserRepository.Save")
    defer span.End()
    
    span.SetAttributes(
        attribute.Int("user.id", user.ID),
        attribute.String("user.email", user.Email),
    )
    
    // Execute database operation
    result := ur.db.WithContext(ctx).Create(user)
    if result.Error != nil {
        span.SetStatus(codes.Error, "Database operation failed")
        span.RecordError(result.Error)
        return result.Error
    }
    
    return nil
}
```

### 2. Custom Span Creation

```go
// Custom span creation
func (ps *PaymentService) ProcessPayment(ctx context.Context, req ProcessPaymentRequest) (*Payment, error) {
    tracer := otel.Tracer("payment-service")
    ctx, span := tracer.Start(ctx, "PaymentService.ProcessPayment")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("payment.order_id", req.OrderID),
        attribute.Float64("payment.amount", req.Amount),
        attribute.String("payment.method", req.Method),
    )
    
    // Validate payment
    if err := ps.validatePayment(req); err != nil {
        span.SetStatus(codes.Error, "Payment validation failed")
        span.RecordError(err)
        return nil, err
    }
    
    // Process with payment gateway
    ctx, gatewaySpan := tracer.Start(ctx, "PaymentGateway.Process")
    gatewaySpan.SetAttributes(
        attribute.String("gateway.name", "stripe"),
        attribute.String("payment.method", req.Method),
    )
    
    payment, err := ps.gateway.ProcessPayment(ctx, req)
    if err != nil {
        gatewaySpan.SetStatus(codes.Error, "Gateway processing failed")
        gatewaySpan.RecordError(err)
        gatewaySpan.End()
        span.SetStatus(codes.Error, "Payment processing failed")
        span.RecordError(err)
        return nil, err
    }
    
    gatewaySpan.SetAttributes(
        attribute.String("payment.gateway_id", payment.GatewayID),
        attribute.String("payment.status", payment.Status),
    )
    gatewaySpan.End()
    
    // Save payment
    ctx, saveSpan := tracer.Start(ctx, "PaymentRepository.Save")
    if err := ps.repo.Save(ctx, payment); err != nil {
        saveSpan.SetStatus(codes.Error, "Failed to save payment")
        saveSpan.RecordError(err)
        saveSpan.End()
        span.SetStatus(codes.Error, "Database error")
        span.RecordError(err)
        return nil, err
    }
    saveSpan.End()
    
    span.SetAttributes(
        attribute.Int("payment.id", payment.ID),
        attribute.String("payment.status", payment.Status),
    )
    
    return payment, nil
}
```

---

## 🚨 Alerting Systems

### 1. Alert Rules

```yaml
# Prometheus Alert Rules
groups:
- name: user-service
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }} seconds"
  
  - alert: HighMemoryUsage
    expr: system_memory_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}%"
  
  - alert: DatabaseConnectionFailure
    expr: up{job="database"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "Database is not responding"
  
  - alert: LowDiskSpace
    expr: system_disk_usage_percent > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Low disk space"
      description: "Disk usage is {{ $value }}%"
```

### 2. Alert Manager Configuration

```yaml
# Alert Manager Configuration
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@example.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'

- name: 'critical-alerts'
  email_configs:
  - to: 'oncall@example.com'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/...'
    channel: '#alerts'
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  email_configs:
  - to: 'team@example.com'
    subject: 'WARNING: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

---

## ⚡ Performance Monitoring

### 1. APM Integration

```go
// Application Performance Monitoring
package main

import (
    "context"
    "time"
    
    "github.com/DataDog/datadog-go/statsd"
)

type APMClient struct {
    client *statsd.Client
}

func NewAPMClient() (*APMClient, error) {
    client, err := statsd.New("127.0.0.1:8125")
    if err != nil {
        return nil, err
    }
    
    return &APMClient{client: client}, nil
}

func (apm *APMClient) RecordTiming(operation string, duration time.Duration, tags []string) {
    apm.client.Timing(operation, duration, tags, 1)
}

func (apm *APMClient) RecordCounter(metric string, value int64, tags []string) {
    apm.client.Count(metric, value, tags, 1)
}

func (apm *APMClient) RecordGauge(metric string, value float64, tags []string) {
    apm.client.Gauge(metric, value, tags, 1)
}

// Performance monitoring middleware
func APMMiddleware(apm *APMClient) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            start := time.Now()
            
            ww := &ResponseWriterWrapper{ResponseWriter: w, statusCode: 200}
            next.ServeHTTP(ww, r)
            
            duration := time.Since(start)
            
            tags := []string{
                "method:" + r.Method,
                "endpoint:" + r.URL.Path,
                "status:" + fmt.Sprintf("%d", ww.statusCode),
            }
            
            apm.RecordTiming("http.request.duration", duration, tags)
            apm.RecordCounter("http.request.count", 1, tags)
        })
    }
}

// Business metrics
func (us *UserService) CreateUserWithAPM(ctx context.Context, req CreateUserRequest) (*User, error) {
    apm, _ := NewAPMClient()
    
    start := time.Now()
    defer func() {
        duration := time.Since(start)
        apm.RecordTiming("user.create.duration", duration, []string{"operation:create_user"})
    }()
    
    user, err := us.CreateUser(ctx, req)
    if err != nil {
        apm.RecordCounter("user.create.error", 1, []string{"error:" + err.Error()})
        return nil, err
    }
    
    apm.RecordCounter("user.create.success", 1, []string{"status:success"})
    apm.RecordGauge("user.total", float64(user.ID), []string{})
    
    return user, nil
}
```

---

## 🏥 Health Checks

### 1. Health Check Endpoints

```go
// Health Check System
package main

import (
    "context"
    "database/sql"
    "net/http"
    "time"
)

type HealthChecker interface {
    Check(ctx context.Context) error
}

type DatabaseHealthChecker struct {
    db *sql.DB
}

func (dhc *DatabaseHealthChecker) Check(ctx context.Context) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    return dhc.db.PingContext(ctx)
}

type RedisHealthChecker struct {
    client *redis.Client
}

func (rhc *RedisHealthChecker) Check(ctx context.Context) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    return rhc.client.Ping(ctx).Err()
}

type HealthCheckService struct {
    checkers map[string]HealthChecker
}

func NewHealthCheckService() *HealthCheckService {
    return &HealthCheckService{
        checkers: make(map[string]HealthChecker),
    }
}

func (hcs *HealthCheckService) AddChecker(name string, checker HealthChecker) {
    hcs.checkers[name] = checker
}

func (hcs *HealthCheckService) CheckAll(ctx context.Context) map[string]string {
    results := make(map[string]string)
    
    for name, checker := range hcs.checkers {
        if err := checker.Check(ctx); err != nil {
            results[name] = "unhealthy: " + err.Error()
        } else {
            results[name] = "healthy"
        }
    }
    
    return results
}

func (hcs *HealthCheckService) IsHealthy(ctx context.Context) bool {
    for _, checker := range hcs.checkers {
        if err := checker.Check(ctx); err != nil {
            return false
        }
    }
    return true
}

// Health check endpoints
func (hcs *HealthCheckService) HealthHandler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    results := hcs.CheckAll(ctx)
    
    w.Header().Set("Content-Type", "application/json")
    
    if hcs.IsHealthy(ctx) {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status":    hcs.IsHealthy(ctx),
        "checks":    results,
        "timestamp": time.Now().Unix(),
    })
}

func (hcs *HealthCheckService) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    // Check critical dependencies
    criticalCheckers := []string{"database", "redis"}
    for _, name := range criticalCheckers {
        if checker, exists := hcs.checkers[name]; exists {
            if err := checker.Check(ctx); err != nil {
                w.WriteHeader(http.StatusServiceUnavailable)
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "status": "not ready",
                    "reason": name + " is not available",
                })
                return
            }
        }
    }
    
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": "ready",
    })
}

func (hcs *HealthCheckService) LivenessHandler(w http.ResponseWriter, r *http.Request) {
    // Simple liveness check - just return OK if the service is running
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": "alive",
    })
}
```

---

## 🎯 Best Practices Summary

### 1. Monitoring Strategy
- **Define SLIs/SLOs**: Set clear service level indicators and objectives
- **Monitor the Right Things**: Focus on business and technical metrics
- **Set Appropriate Thresholds**: Avoid alert fatigue
- **Test Your Alerts**: Ensure alerts work and are actionable

### 2. Logging Best Practices
- **Use Structured Logging**: JSON format for better parsing
- **Include Context**: Add request IDs, user IDs, etc.
- **Log at Appropriate Levels**: Debug, Info, Warn, Error
- **Avoid Logging Sensitive Data**: No passwords, tokens, PII

### 3. Tracing Guidelines
- **Trace Across Service Boundaries**: Follow requests through the system
- **Add Meaningful Attributes**: Include business context
- **Keep Traces Lightweight**: Don't impact performance
- **Use Sampling**: Reduce trace volume in production

### 4. Alerting Rules
- **Alert on Symptoms, Not Causes**: Focus on user impact
- **Use Appropriate Severity**: Critical, Warning, Info
- **Include Runbooks**: Provide context and resolution steps
- **Test Alerting**: Regular alert testing and tuning

---

**📊 Master these monitoring and observability patterns to build reliable, observable, and maintainable backend systems! 🚀**


##  Observability Tools

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-observability-tools -->

Placeholder content. Please replace with proper section.
