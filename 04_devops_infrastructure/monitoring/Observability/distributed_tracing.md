# Distributed Tracing Deep Dive

## Table of Contents
1. [Introduction](#introduction/)
2. [OpenTelemetry Integration](#opentelemetry-integration/)
3. [Jaeger Implementation](#jaeger-implementation/)
4. [Golang Tracing](#golang-tracing/)
5. [Microservices Tracing](#microservices-tracing/)
6. [Performance Optimization](#performance-optimization/)
7. [Error Tracking](#error-tracking/)
8. [Best Practices](#best-practices/)
9. [Advanced Patterns](#advanced-patterns/)
10. [Troubleshooting](#troubleshooting/)

## Introduction

Distributed tracing is essential for understanding request flow across microservices. It helps identify performance bottlenecks, debug issues, and understand system behavior in complex distributed architectures.

## OpenTelemetry Integration

### Basic Setup
```go
package main

import (
    "context"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/trace"
    "go.opentelemetry.io/otel/trace"
)

func initTracer() func() {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint("http://localhost:14268/api/traces")))
    if err != nil {
        log.Fatal(err)
    }

    // Create trace provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("payment-service"),
            semconv.ServiceVersionKey.String("1.0.0"),
        )),
    )

    // Set global tracer provider
    otel.SetTracerProvider(tp)

    // Return cleanup function
    return func() {
        tp.Shutdown(context.Background())
    }
}
```

### Tracer Usage
```go
func ProcessPayment(ctx context.Context, payment Payment) error {
    tracer := otel.Tracer("payment-service")
    
    // Create span
    ctx, span := tracer.Start(ctx, "ProcessPayment")
    defer span.End()
    
    // Add attributes
    span.SetAttributes(
        attribute.String("payment.id", payment.ID),
        attribute.Float64("payment.amount", payment.Amount),
        attribute.String("payment.currency", payment.Currency),
    )
    
    // Process payment steps
    if err := validatePayment(ctx, payment); err != nil {
        span.RecordError(err)
        return err
    }
    
    if err := chargeCard(ctx, payment); err != nil {
        span.RecordError(err)
        return err
    }
    
    return nil
}
```

## Jaeger Implementation

### Configuration
```go
type JaegerConfig struct {
    ServiceName    string
    ServiceVersion string
    Endpoint       string
    SampleRate     float64
}

func NewJaegerTracer(config JaegerConfig) (trace.TracerProvider, error) {
    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String(config.ServiceName),
            semconv.ServiceVersionKey.String(config.ServiceVersion),
        ),
    )
    if err != nil {
        return nil, err
    }

    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint(config.Endpoint),
    ))
    if err != nil {
        return nil, err
    }

    // Create trace provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(res),
        trace.WithSampler(trace.TraceIDRatioBased(config.SampleRate)),
    )

    return tp, nil
}
```

### Custom Sampler
```go
type CustomSampler struct {
    baseSampler trace.Sampler
    rules       []SamplingRule
}

type SamplingRule struct {
    ServiceName string
    Operation   string
    SampleRate  float64
}

func (cs *CustomSampler) ShouldSample(p trace.SamplingParameters) trace.SamplingResult {
    // Check custom rules first
    for _, rule := range cs.rules {
        if rule.ServiceName == p.Name && rule.Operation == p.Name {
            if rand.Float64() < rule.SampleRate {
                return trace.SamplingResult{Decision: trace.RecordAndSample}
            }
        }
    }
    
    // Fall back to base sampler
    return cs.baseSampler.ShouldSample(p)
}
```

## Golang Tracing

### HTTP Middleware
```go
func TracingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tracer := otel.Tracer("http-server")
        
        // Extract trace context from headers
        ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
        
        // Create span
        ctx, span := tracer.Start(ctx, r.URL.Path)
        defer span.End()
        
        // Add attributes
        span.SetAttributes(
            attribute.String("http.method", r.Method),
            attribute.String("http.url", r.URL.String()),
            attribute.String("http.user_agent", r.UserAgent()),
        )
        
        // Wrap response writer to capture status code
        ww := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // Process request
        next.ServeHTTP(ww, r.WithContext(ctx))
        
        // Add response attributes
        span.SetAttributes(
            attribute.Int("http.status_code", ww.statusCode),
        )
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

### Database Tracing
```go
func (db *Database) QueryWithTracing(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    tracer := otel.Tracer("database")
    
    ctx, span := tracer.Start(ctx, "database.query")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("db.statement", query),
        attribute.String("db.system", "postgresql"),
    )
    
    start := time.Now()
    rows, err := db.QueryContext(ctx, query, args...)
    duration := time.Since(start)
    
    span.SetAttributes(
        attribute.Int64("db.duration_ms", duration.Milliseconds()),
    )
    
    if err != nil {
        span.RecordError(err)
    }
    
    return rows, err
}
```

### gRPC Tracing
```go
func UnaryServerInterceptor() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        tracer := otel.Tracer("grpc-server")
        
        ctx, span := tracer.Start(ctx, info.FullMethod)
        defer span.End()
        
        span.SetAttributes(
            attribute.String("rpc.system", "grpc"),
            attribute.String("rpc.service", info.FullMethod),
        )
        
        resp, err := handler(ctx, req)
        if err != nil {
            span.RecordError(err)
        }
        
        return resp, err
    }
}
```

## Microservices Tracing

### Context Propagation
```go
func PropagateTraceContext(ctx context.Context, headers http.Header) {
    // Inject trace context into headers
    otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
}

func ExtractTraceContext(ctx context.Context, headers http.Header) context.Context {
    // Extract trace context from headers
    return otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(headers))
}
```

### Service Mesh Integration
```go
type ServiceMeshTracer struct {
    tracer trace.Tracer
}

func (smt *ServiceMeshTracer) TraceServiceCall(ctx context.Context, service, method string, req interface{}) (interface{}, error) {
    ctx, span := smt.tracer.Start(ctx, fmt.Sprintf("%s.%s", service, method))
    defer span.End()
    
    span.SetAttributes(
        attribute.String("service.name", service),
        attribute.String("service.method", method),
    )
    
    // Make service call
    resp, err := smt.callService(ctx, service, method, req)
    if err != nil {
        span.RecordError(err)
    }
    
    return resp, err
}
```

### Message Queue Tracing
```go
func (mq *MessageQueue) PublishWithTracing(ctx context.Context, topic string, message interface{}) error {
    tracer := otel.Tracer("message-queue")
    
    ctx, span := tracer.Start(ctx, "mq.publish")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("mq.topic", topic),
        attribute.String("mq.operation", "publish"),
    )
    
    // Inject trace context into message headers
    headers := make(map[string]string)
    otel.GetTextMapPropagator().Inject(ctx, propagation.MapCarrier(headers))
    
    return mq.publish(topic, message, headers)
}

func (mq *MessageQueue) ConsumeWithTracing(ctx context.Context, topic string, handler MessageHandler) error {
    tracer := otel.Tracer("message-queue")
    
    return mq.consume(topic, func(ctx context.Context, message interface{}, headers map[string]string) error {
        // Extract trace context from message headers
        ctx = otel.GetTextMapPropagator().Extract(ctx, propagation.MapCarrier(headers))
        
        ctx, span := tracer.Start(ctx, "mq.consume")
        defer span.End()
        
        span.SetAttributes(
            attribute.String("mq.topic", topic),
            attribute.String("mq.operation", "consume"),
        )
        
        return handler(ctx, message)
    })
}
```

## Performance Optimization

### Sampling Strategies
```go
type AdaptiveSampler struct {
    baseRate    float64
    maxRate     float64
    minRate     float64
    currentRate float64
    mutex       sync.RWMutex
}

func (as *AdaptiveSampler) ShouldSample(p trace.SamplingParameters) trace.SamplingResult {
    as.mutex.RLock()
    rate := as.currentRate
    as.mutex.RUnlock()
    
    if rand.Float64() < rate {
        return trace.SamplingResult{Decision: trace.RecordAndSample}
    }
    
    return trace.SamplingResult{Decision: trace.Drop}
}

func (as *AdaptiveSampler) AdjustRate(load float64) {
    as.mutex.Lock()
    defer as.mutex.Unlock()
    
    // Adjust sampling rate based on system load
    if load > 0.8 {
        as.currentRate = math.Max(as.minRate, as.currentRate*0.9)
    } else if load < 0.3 {
        as.currentRate = math.Min(as.maxRate, as.currentRate*1.1)
    }
}
```

### Batch Processing
```go
type BatchProcessor struct {
    spans    chan trace.ReadOnlySpan
    batchSize int
    timeout   time.Duration
}

func (bp *BatchProcessor) ProcessSpans() {
    batch := make([]trace.ReadOnlySpan, 0, bp.batchSize)
    ticker := time.NewTicker(bp.timeout)
    
    for {
        select {
        case span := <-bp.spans:
            batch = append(batch, span)
            if len(batch) >= bp.batchSize {
                bp.exportBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                bp.exportBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

## Error Tracking

### Error Spans
```go
func (s *Service) ProcessRequest(ctx context.Context, req Request) error {
    tracer := otel.Tracer("service")
    
    ctx, span := tracer.Start(ctx, "ProcessRequest")
    defer span.End()
    
    // Add request attributes
    span.SetAttributes(
        attribute.String("request.id", req.ID),
        attribute.String("request.type", req.Type),
    )
    
    // Process request
    if err := s.validateRequest(ctx, req); err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    
    if err := s.processRequest(ctx, req); err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    
    span.SetStatus(codes.Ok, "Request processed successfully")
    return nil
}
```

### Error Aggregation
```go
type ErrorTracker struct {
    errors    map[string]int
    mutex     sync.RWMutex
    exporter  trace.SpanExporter
}

func (et *ErrorTracker) TrackError(span trace.ReadOnlySpan, err error) {
    et.mutex.Lock()
    defer et.mutex.Unlock()
    
    errorKey := fmt.Sprintf("%s:%s", span.Name(), err.Error())
    et.errors[errorKey]++
    
    // Export error span
    et.exporter.ExportSpans(context.Background(), []trace.ReadOnlySpan{span})
}
```

## Best Practices

### 1. Span Naming
```go
// Good: Descriptive span names
tracer.Start(ctx, "user.login")
tracer.Start(ctx, "payment.process")
tracer.Start(ctx, "database.query.users")

// Bad: Generic span names
tracer.Start(ctx, "process")
tracer.Start(ctx, "handle")
tracer.Start(ctx, "query")
```

### 2. Attribute Usage
```go
// Good: Meaningful attributes
span.SetAttributes(
    attribute.String("user.id", userID),
    attribute.String("payment.method", method),
    attribute.Float64("payment.amount", amount),
)

// Bad: Too many or irrelevant attributes
span.SetAttributes(
    attribute.String("internal.var1", "value1"),
    attribute.String("internal.var2", "value2"),
    attribute.String("internal.var3", "value3"),
)
```

### 3. Context Propagation
```go
// Good: Proper context propagation
func (s *Service) CallOtherService(ctx context.Context, req Request) error {
    // Context is properly passed through
    return s.otherService.Process(ctx, req)
}

// Bad: Context not propagated
func (s *Service) CallOtherService(ctx context.Context, req Request) error {
    // Context is lost
    return s.otherService.Process(context.Background(), req)
}
```

### 4. Resource Management
```go
// Good: Proper span lifecycle management
func (s *Service) ProcessRequest(ctx context.Context, req Request) error {
    ctx, span := s.tracer.Start(ctx, "ProcessRequest")
    defer span.End() // Always defer span.End()
    
    // Process request
    return s.doProcess(ctx, req)
}
```

## Advanced Patterns

### Custom Span Processors
```go
type CustomSpanProcessor struct {
    next trace.SpanProcessor
}

func (csp *CustomSpanProcessor) OnStart(parent context.Context, s trace.ReadWriteSpan) {
    // Add custom attributes to all spans
    s.SetAttributes(
        attribute.String("service.instance", os.Getenv("HOSTNAME")),
        attribute.String("service.version", "1.0.0"),
    )
    
    if csp.next != nil {
        csp.next.OnStart(parent, s)
    }
}

func (csp *CustomSpanProcessor) OnEnd(s trace.ReadOnlySpan) {
    // Custom processing on span end
    if csp.next != nil {
        csp.next.OnEnd(s)
    }
}

func (csp *CustomSpanProcessor) Shutdown(ctx context.Context) error {
    if csp.next != nil {
        return csp.next.Shutdown(ctx)
    }
    return nil
}

func (csp *CustomSpanProcessor) ForceFlush(ctx context.Context) error {
    if csp.next != nil {
        return csp.next.ForceFlush(ctx)
    }
    return nil
}
```

### Trace Correlation
```go
type TraceCorrelator struct {
    traces map[string][]trace.ReadOnlySpan
    mutex  sync.RWMutex
}

func (tc *TraceCorrelator) CorrelateTraces(traceID string) []trace.ReadOnlySpan {
    tc.mutex.RLock()
    defer tc.mutex.RUnlock()
    
    return tc.traces[traceID]
}

func (tc *TraceCorrelator) AddSpan(span trace.ReadOnlySpan) {
    tc.mutex.Lock()
    defer tc.mutex.Unlock()
    
    traceID := span.SpanContext().TraceID().String()
    tc.traces[traceID] = append(tc.traces[traceID], span)
}
```

## Troubleshooting

### Common Issues

#### 1. Missing Traces
```go
// Check if tracing is properly initialized
func checkTracingSetup() {
    tp := otel.GetTracerProvider()
    if tp == nil {
        log.Fatal("Tracer provider not initialized")
    }
    
    tracer := tp.Tracer("test")
    if tracer == nil {
        log.Fatal("Tracer not created")
    }
}
```

#### 2. Context Propagation Issues
```go
// Ensure context is properly propagated
func ensureContextPropagation(ctx context.Context, headers http.Header) context.Context {
    // Check if trace context exists in headers
    if traceID := headers.Get("traceparent"); traceID == "" {
        log.Warn("No trace context found in headers")
    }
    
    return otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(headers))
}
```

#### 3. Performance Issues
```go
// Monitor tracing overhead
func (s *Service) ProcessWithTracing(ctx context.Context, req Request) error {
    start := time.Now()
    
    ctx, span := s.tracer.Start(ctx, "ProcessRequest")
    defer span.End()
    
    err := s.process(ctx, req)
    
    duration := time.Since(start)
    if duration > 100*time.Millisecond {
        log.Warn("Tracing overhead detected", "duration", duration)
    }
    
    return err
}
```

## Conclusion

Distributed tracing is essential for:

1. **Request Flow Understanding**: Track requests across services
2. **Performance Analysis**: Identify bottlenecks and slow operations
3. **Error Debugging**: Understand error propagation and root causes
4. **System Observability**: Gain insights into system behavior
5. **Capacity Planning**: Understand resource usage patterns

Key principles:
- Use consistent span naming and attributes
- Properly propagate trace context
- Implement appropriate sampling strategies
- Monitor tracing overhead
- Correlate traces across services
- Use tracing for both technical and business insights
