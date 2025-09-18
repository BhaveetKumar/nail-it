# Advanced Monitoring Comprehensive

Comprehensive guide to advanced monitoring and observability for senior backend engineers.

## 🎯 Advanced Observability Stack

### Distributed Tracing Implementation
```go
// Advanced Distributed Tracing with OpenTelemetry
package monitoring

import (
    "context"
    "fmt"
    "time"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    "go.opentelemetry.io/otel/trace"
)

type TracingService struct {
    tracer trace.Tracer
    exporter *jaeger.Exporter
    provider *trace.TracerProvider
}

type TraceContext struct {
    TraceID    string            `json:"trace_id"`
    SpanID     string            `json:"span_id"`
    Baggage    map[string]string `json:"baggage"`
    Attributes map[string]string `json:"attributes"`
}

func NewTracingService(serviceName string, jaegerEndpoint string) (*TracingService, error) {
    // Create Jaeger exporter
    exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerEndpoint)))
    if err != nil {
        return nil, fmt.Errorf("failed to create Jaeger exporter: %w", err)
    }
    
    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            attribute.String("service.name", serviceName),
            attribute.String("service.version", "1.0.0"),
        ),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create resource: %w", err)
    }
    
    // Create tracer provider
    provider := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithResource(res),
        trace.WithSampler(trace.TraceIDRatioBased(0.1)), // 10% sampling
    )
    
    // Set global tracer provider
    otel.SetTracerProvider(provider)
    
    // Set global propagator
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))
    
    return &TracingService{
        tracer:   provider.Tracer(serviceName),
        exporter: exporter,
        provider: provider,
    }, nil
}

func (ts *TracingService) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
    return ts.tracer.Start(ctx, name, opts...)
}

func (ts *TracingService) StartSpanWithAttributes(ctx context.Context, name string, attrs map[string]interface{}) (context.Context, trace.Span) {
    var spanAttrs []attribute.KeyValue
    for key, value := range attrs {
        spanAttrs = append(spanAttrs, attribute.String(key, fmt.Sprintf("%v", value)))
    }
    
    return ts.tracer.Start(ctx, name, trace.WithAttributes(spanAttrs...))
}

func (ts *TracingService) AddSpanEvent(span trace.Span, name string, attrs map[string]interface{}) {
    var spanAttrs []attribute.KeyValue
    for key, value := range attrs {
        spanAttrs = append(spanAttrs, attribute.String(key, fmt.Sprintf("%v", value)))
    }
    
    span.AddEvent(name, trace.WithAttributes(spanAttrs...))
}

func (ts *TracingService) SetSpanStatus(span trace.Span, err error) {
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
    } else {
        span.SetStatus(codes.Ok, "")
    }
}

func (ts *TracingService) ExtractTraceContext(ctx context.Context) *TraceContext {
    span := trace.SpanFromContext(ctx)
    spanContext := span.SpanContext()
    
    return &TraceContext{
        TraceID: spanContext.TraceID().String(),
        SpanID:  spanContext.SpanID().String(),
        Baggage: make(map[string]string),
        Attributes: make(map[string]string),
    }
}

func (ts *TracingService) InjectTraceContext(ctx context.Context, traceCtx *TraceContext) context.Context {
    // Create new span context
    traceID, _ := trace.TraceIDFromHex(traceCtx.TraceID)
    spanID, _ := trace.SpanIDFromHex(traceCtx.SpanID)
    
    spanContext := trace.NewSpanContext(trace.SpanContextConfig{
        TraceID: traceID,
        SpanID:  spanID,
    })
    
    // Create new span
    span := trace.SpanFromContext(ctx)
    newSpan := trace.SpanFromContext(trace.ContextWithSpan(ctx, span))
    
    return trace.ContextWithSpan(ctx, newSpan)
}

// HTTP Middleware for Tracing
func (ts *TracingService) HTTPMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract trace context from headers
        ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
        
        // Start span
        spanName := fmt.Sprintf("%s %s", r.Method, r.URL.Path)
        ctx, span := ts.StartSpanWithAttributes(ctx, spanName, map[string]interface{}{
            "http.method":     r.Method,
            "http.url":        r.URL.String(),
            "http.user_agent": r.UserAgent(),
            "http.remote_addr": r.RemoteAddr,
        })
        defer span.End()
        
        // Create response writer wrapper
        wrapper := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // Call next handler
        next.ServeHTTP(wrapper, r.WithContext(ctx))
        
        // Set span attributes
        span.SetAttributes(
            attribute.Int("http.status_code", wrapper.statusCode),
            attribute.Int("http.response_size", wrapper.size),
        )
        
        // Set span status
        if wrapper.statusCode >= 400 {
            span.SetStatus(codes.Error, fmt.Sprintf("HTTP %d", wrapper.statusCode))
        } else {
            span.SetStatus(codes.Ok, "")
        }
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
    size       int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
    size, err := rw.ResponseWriter.Write(b)
    rw.size += size
    return size, err
}

// Database Tracing
func (ts *TracingService) TraceDatabaseQuery(ctx context.Context, query string, args []interface{}) (context.Context, trace.Span) {
    return ts.StartSpanWithAttributes(ctx, "database.query", map[string]interface{}{
        "db.statement": query,
        "db.args_count": len(args),
    })
}

// External Service Tracing
func (ts *TracingService) TraceExternalService(ctx context.Context, serviceName string, operation string) (context.Context, trace.Span) {
    return ts.StartSpanWithAttributes(ctx, "external.service", map[string]interface{}{
        "service.name": serviceName,
        "operation.name": operation,
    })
}
```

### Advanced Metrics Collection
```go
// Advanced Metrics Collection with Prometheus
package monitoring

import (
    "context"
    "fmt"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

type MetricsCollector struct {
    // HTTP Metrics
    httpRequestsTotal     *prometheus.CounterVec
    httpRequestDuration   *prometheus.HistogramVec
    httpRequestSize       *prometheus.HistogramVec
    httpResponseSize      *prometheus.HistogramVec
    
    // Database Metrics
    dbConnectionsActive   prometheus.Gauge
    dbConnectionsIdle     prometheus.Gauge
    dbQueriesTotal        *prometheus.CounterVec
    dbQueryDuration       *prometheus.HistogramVec
    
    // Business Metrics
    userRegistrations     prometheus.Counter
    userLogins           prometheus.Counter
    ordersCreated        prometheus.Counter
    revenueGenerated     prometheus.Counter
    
    // System Metrics
    memoryUsage          prometheus.Gauge
    cpuUsage             prometheus.Gauge
    diskUsage            prometheus.Gauge
    goroutineCount       prometheus.Gauge
    
    // Custom Metrics
    customCounters       map[string]*prometheus.CounterVec
    customGauges         map[string]*prometheus.GaugeVec
    customHistograms     map[string]*prometheus.HistogramVec
}

func NewMetricsCollector() *MetricsCollector {
    return &MetricsCollector{
        // HTTP Metrics
        httpRequestsTotal: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_requests_total",
                Help: "Total number of HTTP requests",
            },
            []string{"method", "endpoint", "status_code"},
        ),
        httpRequestDuration: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "http_request_duration_seconds",
                Help:    "HTTP request duration in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"method", "endpoint"},
        ),
        httpRequestSize: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "http_request_size_bytes",
                Help:    "HTTP request size in bytes",
                Buckets: prometheus.ExponentialBuckets(100, 10, 8),
            },
            []string{"method", "endpoint"},
        ),
        httpResponseSize: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "http_response_size_bytes",
                Help:    "HTTP response size in bytes",
                Buckets: prometheus.ExponentialBuckets(100, 10, 8),
            },
            []string{"method", "endpoint"},
        ),
        
        // Database Metrics
        dbConnectionsActive: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "db_connections_active",
                Help: "Number of active database connections",
            },
        ),
        dbConnectionsIdle: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "db_connections_idle",
                Help: "Number of idle database connections",
            },
        ),
        dbQueriesTotal: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "db_queries_total",
                Help: "Total number of database queries",
            },
            []string{"operation", "table"},
        ),
        dbQueryDuration: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "db_query_duration_seconds",
                Help:    "Database query duration in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"operation", "table"},
        ),
        
        // Business Metrics
        userRegistrations: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "user_registrations_total",
                Help: "Total number of user registrations",
            },
        ),
        userLogins: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "user_logins_total",
                Help: "Total number of user logins",
            },
        ),
        ordersCreated: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "orders_created_total",
                Help: "Total number of orders created",
            },
        ),
        revenueGenerated: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "revenue_generated_total",
                Help: "Total revenue generated",
            },
        ),
        
        // System Metrics
        memoryUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "memory_usage_bytes",
                Help: "Memory usage in bytes",
            },
        ),
        cpuUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "cpu_usage_percent",
                Help: "CPU usage percentage",
            },
        ),
        diskUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "disk_usage_bytes",
                Help: "Disk usage in bytes",
            },
        ),
        goroutineCount: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "goroutines_total",
                Help: "Number of goroutines",
            },
        ),
        
        // Custom Metrics
        customCounters:   make(map[string]*prometheus.CounterVec),
        customGauges:     make(map[string]*prometheus.GaugeVec),
        customHistograms: make(map[string]*prometheus.HistogramVec),
    }
}

func (mc *MetricsCollector) RecordHTTPRequest(method, endpoint, statusCode string, duration time.Duration, requestSize, responseSize int64) {
    mc.httpRequestsTotal.WithLabelValues(method, endpoint, statusCode).Inc()
    mc.httpRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
    mc.httpRequestSize.WithLabelValues(method, endpoint).Observe(float64(requestSize))
    mc.httpResponseSize.WithLabelValues(method, endpoint).Observe(float64(responseSize))
}

func (mc *MetricsCollector) RecordDatabaseQuery(operation, table string, duration time.Duration) {
    mc.dbQueriesTotal.WithLabelValues(operation, table).Inc()
    mc.dbQueryDuration.WithLabelValues(operation, table).Observe(duration.Seconds())
}

func (mc *MetricsCollector) SetDatabaseConnections(active, idle int) {
    mc.dbConnectionsActive.Set(float64(active))
    mc.dbConnectionsIdle.Set(float64(idle))
}

func (mc *MetricsCollector) RecordUserRegistration() {
    mc.userRegistrations.Inc()
}

func (mc *MetricsCollector) RecordUserLogin() {
    mc.userLogins.Inc()
}

func (mc *MetricsCollector) RecordOrderCreated() {
    mc.ordersCreated.Inc()
}

func (mc *MetricsCollector) RecordRevenue(amount float64) {
    mc.revenueGenerated.Add(amount)
}

func (mc *MetricsCollector) UpdateSystemMetrics() {
    // Update memory usage
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    mc.memoryUsage.Set(float64(m.Alloc))
    
    // Update goroutine count
    mc.goroutineCount.Set(float64(runtime.NumGoroutine()))
    
    // Update CPU usage (simplified)
    // In practice, you'd use a more sophisticated CPU monitoring library
    mc.cpuUsage.Set(0.0) // Placeholder
    
    // Update disk usage (simplified)
    // In practice, you'd check actual disk usage
    mc.diskUsage.Set(0.0) // Placeholder
}

func (mc *MetricsCollector) CreateCustomCounter(name, help string, labels []string) *prometheus.CounterVec {
    if counter, exists := mc.customCounters[name]; exists {
        return counter
    }
    
    counter := promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: name,
            Help: help,
        },
        labels,
    )
    
    mc.customCounters[name] = counter
    return counter
}

func (mc *MetricsCollector) CreateCustomGauge(name, help string, labels []string) *prometheus.GaugeVec {
    if gauge, exists := mc.customGauges[name]; exists {
        return gauge
    }
    
    gauge := promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: name,
            Help: help,
        },
        labels,
    )
    
    mc.customGauges[name] = gauge
    return gauge
}

func (mc *MetricsCollector) CreateCustomHistogram(name, help string, labels []string) *prometheus.HistogramVec {
    if histogram, exists := mc.customHistograms[name]; exists {
        return histogram
    }
    
    histogram := promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    name,
            Help:    help,
            Buckets: prometheus.DefBuckets,
        },
        labels,
    )
    
    mc.customHistograms[name] = histogram
    return histogram
}
```

### Advanced Logging System
```go
// Advanced Logging System with Structured Logging
package monitoring

import (
    "context"
    "encoding/json"
    "fmt"
    "os"
    "time"
    
    "github.com/sirupsen/logrus"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

type LogLevel string

const (
    DebugLevel LogLevel = "debug"
    InfoLevel  LogLevel = "info"
    WarnLevel  LogLevel = "warn"
    ErrorLevel LogLevel = "error"
    FatalLevel LogLevel = "fatal"
)

type LogEntry struct {
    Timestamp   time.Time              `json:"timestamp"`
    Level       LogLevel               `json:"level"`
    Message     string                 `json:"message"`
    Fields      map[string]interface{} `json:"fields"`
    TraceID     string                 `json:"trace_id,omitempty"`
    SpanID      string                 `json:"span_id,omitempty"`
    Service     string                 `json:"service"`
    Environment string                 `json:"environment"`
    Version     string                 `json:"version"`
}

type AdvancedLogger struct {
    logger      *zap.Logger
    service     string
    environment string
    version     string
}

func NewAdvancedLogger(service, environment, version string) (*AdvancedLogger, error) {
    // Configure Zap logger
    config := zap.NewProductionConfig()
    config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
    config.EncoderConfig.TimeKey = "timestamp"
    config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
    config.EncoderConfig.MessageKey = "message"
    config.EncoderConfig.LevelKey = "level"
    config.EncoderConfig.CallerKey = "caller"
    config.EncoderConfig.StacktraceKey = "stacktrace"
    
    // Set output format
    config.Encoding = "json"
    
    // Create logger
    logger, err := config.Build()
    if err != nil {
        return nil, fmt.Errorf("failed to create logger: %w", err)
    }
    
    return &AdvancedLogger{
        logger:      logger,
        service:     service,
        environment: environment,
        version:     version,
    }, nil
}

func (al *AdvancedLogger) Debug(ctx context.Context, message string, fields ...map[string]interface{}) {
    al.log(ctx, DebugLevel, message, fields...)
}

func (al *AdvancedLogger) Info(ctx context.Context, message string, fields ...map[string]interface{}) {
    al.log(ctx, InfoLevel, message, fields...)
}

func (al *AdvancedLogger) Warn(ctx context.Context, message string, fields ...map[string]interface{}) {
    al.log(ctx, WarnLevel, message, fields...)
}

func (al *AdvancedLogger) Error(ctx context.Context, message string, fields ...map[string]interface{}) {
    al.log(ctx, ErrorLevel, message, fields...)
}

func (al *AdvancedLogger) Fatal(ctx context.Context, message string, fields ...map[string]interface{}) {
    al.log(ctx, FatalLevel, message, fields...)
}

func (al *AdvancedLogger) log(ctx context.Context, level LogLevel, message string, fields ...map[string]interface{}) {
    // Extract trace context
    traceID, spanID := al.extractTraceContext(ctx)
    
    // Merge fields
    mergedFields := make(map[string]interface{})
    for _, fieldMap := range fields {
        for k, v := range fieldMap {
            mergedFields[k] = v
        }
    }
    
    // Add service information
    mergedFields["service"] = al.service
    mergedFields["environment"] = al.environment
    mergedFields["version"] = al.version
    mergedFields["trace_id"] = traceID
    mergedFields["span_id"] = spanID
    
    // Convert to Zap fields
    zapFields := make([]zap.Field, 0, len(mergedFields))
    for k, v := range mergedFields {
        zapFields = append(zapFields, zap.Any(k, v))
    }
    
    // Log with appropriate level
    switch level {
    case DebugLevel:
        al.logger.Debug(message, zapFields...)
    case InfoLevel:
        al.logger.Info(message, zapFields...)
    case WarnLevel:
        al.logger.Warn(message, zapFields...)
    case ErrorLevel:
        al.logger.Error(message, zapFields...)
    case FatalLevel:
        al.logger.Fatal(message, zapFields...)
    }
}

func (al *AdvancedLogger) extractTraceContext(ctx context.Context) (string, string) {
    // Extract trace context from context
    // This is a simplified version - in practice, you'd use OpenTelemetry
    return "", ""
}

func (al *AdvancedLogger) WithFields(fields map[string]interface{}) *AdvancedLogger {
    return &AdvancedLogger{
        logger:      al.logger.With(zap.Any("fields", fields)),
        service:     al.service,
        environment: al.environment,
        version:     al.version,
    }
}

func (al *AdvancedLogger) WithTraceContext(ctx context.Context) *AdvancedLogger {
    traceID, spanID := al.extractTraceContext(ctx)
    return &AdvancedLogger{
        logger:      al.logger.With(zap.String("trace_id", traceID), zap.String("span_id", spanID)),
        service:     al.service,
        environment: al.environment,
        version:     al.version,
    }
}

// HTTP Middleware for Logging
func (al *AdvancedLogger) HTTPMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Create response writer wrapper
        wrapper := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // Call next handler
        next.ServeHTTP(wrapper, r.WithContext(ctx))
        
        // Calculate duration
        duration := time.Since(start)
        
        // Log request
        al.Info(r.Context(), "HTTP request completed", map[string]interface{}{
            "method":      r.Method,
            "url":         r.URL.String(),
            "status_code": wrapper.statusCode,
            "duration_ms": duration.Milliseconds(),
            "user_agent":  r.UserAgent(),
            "remote_addr": r.RemoteAddr,
        })
    })
}

// Database Query Logging
func (al *AdvancedLogger) LogDatabaseQuery(ctx context.Context, query string, args []interface{}, duration time.Duration, err error) {
    fields := map[string]interface{}{
        "query":        query,
        "args_count":   len(args),
        "duration_ms":  duration.Milliseconds(),
    }
    
    if err != nil {
        fields["error"] = err.Error()
        al.Error(ctx, "Database query failed", fields)
    } else {
        al.Debug(ctx, "Database query executed", fields)
    }
}

// External Service Logging
func (al *AdvancedLogger) LogExternalService(ctx context.Context, serviceName, operation string, duration time.Duration, err error) {
    fields := map[string]interface{}{
        "service_name": serviceName,
        "operation":    operation,
        "duration_ms":  duration.Milliseconds(),
    }
    
    if err != nil {
        fields["error"] = err.Error()
        al.Error(ctx, "External service call failed", fields)
    } else {
        al.Debug(ctx, "External service call completed", fields)
    }
}
```

## 🎯 Best Practices

### Observability Principles
1. **Three Pillars**: Metrics, Logs, and Traces
2. **Context Propagation**: Maintain context across services
3. **Sampling**: Use appropriate sampling strategies
4. **Correlation**: Correlate events across systems
5. **Alerting**: Set up meaningful alerts

### Monitoring Best Practices
1. **SLI/SLO/SLA**: Define and monitor service level objectives
2. **Golden Signals**: Monitor latency, traffic, errors, and saturation
3. **Dashboard Design**: Create effective monitoring dashboards
4. **Alert Fatigue**: Avoid alert fatigue with proper thresholds
5. **Incident Response**: Have clear incident response procedures

### Performance Considerations
1. **Sampling**: Use sampling to reduce overhead
2. **Batching**: Batch metrics and logs for efficiency
3. **Compression**: Use compression for log storage
4. **Retention**: Set appropriate retention policies
5. **Cost Optimization**: Optimize monitoring costs

---

**Last Updated**: December 2024  
**Category**: Advanced Monitoring Comprehensive  
**Complexity**: Expert Level
