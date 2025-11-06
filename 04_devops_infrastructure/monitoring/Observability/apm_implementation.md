---
# Auto-generated front matter
Title: Apm Implementation
LastUpdated: 2025-11-06T20:45:59.164878
Tags: []
Status: draft
---

# APM (Application Performance Monitoring) Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [APM Architecture](#apm-architecture)
3. [Data Collection](#data-collection)
4. [Performance Metrics](#performance-metrics)
5. [Error Tracking](#error-tracking)
6. [User Experience Monitoring](#user-experience-monitoring)
7. [Golang Implementation](#golang-implementation)
8. [Microservices APM](#microservices-apm)
9. [Alerting and Dashboards](#alerting-and-dashboards)
10. [Best Practices](#best-practices)

## Introduction

APM provides comprehensive monitoring of application performance, user experience, and business metrics. It combines metrics, traces, and logs to give a complete picture of system health and performance.

## APM Architecture

### Core Components
```go
type APMSystem struct {
    MetricsCollector  *MetricsCollector
    TraceCollector    *TraceCollector
    LogCollector      *LogCollector
    ErrorTracker      *ErrorTracker
    UserTracker       *UserTracker
    AlertManager      *AlertManager
    DashboardManager  *DashboardManager
}

type APMConfig struct {
    ServiceName    string
    ServiceVersion string
    Environment    string
    SampleRate     float64
    Endpoints      APMEndpoints
}

type APMEndpoints struct {
    Metrics string
    Traces  string
    Logs    string
    Errors  string
}
```

### Data Flow
```go
type APMDataFlow struct {
    collectors map[string]DataCollector
    processors map[string]DataProcessor
    exporters  map[string]DataExporter
}

func (adf *APMDataFlow) ProcessData(dataType string, data interface{}) error {
    // Collect data
    if collector, exists := adf.collectors[dataType]; exists {
        rawData, err := collector.Collect(data)
        if err != nil {
            return err
        }
        
        // Process data
        if processor, exists := adf.processors[dataType]; exists {
            processedData, err := processor.Process(rawData)
            if err != nil {
                return err
            }
            
            // Export data
            if exporter, exists := adf.exporters[dataType]; exists {
                return exporter.Export(processedData)
            }
        }
    }
    
    return fmt.Errorf("no handler for data type: %s", dataType)
}
```

## Data Collection

### Metrics Collection
```go
type MetricsCollector struct {
    counters   map[string]*prometheus.CounterVec
    gauges     map[string]*prometheus.GaugeVec
    histograms map[string]*prometheus.HistogramVec
    summaries  map[string]*prometheus.SummaryVec
    mutex      sync.RWMutex
}

func (mc *MetricsCollector) RecordCounter(name string, labels map[string]string, value float64) {
    mc.mutex.RLock()
    counter, exists := mc.counters[name]
    mc.mutex.RUnlock()
    
    if exists {
        labelValues := make([]string, 0, len(labels))
        for _, value := range labels {
            labelValues = append(labelValues, value)
        }
        counter.WithLabelValues(labelValues...).Add(value)
    }
}

func (mc *MetricsCollector) RecordGauge(name string, labels map[string]string, value float64) {
    mc.mutex.RLock()
    gauge, exists := mc.gauges[name]
    mc.mutex.RUnlock()
    
    if exists {
        labelValues := make([]string, 0, len(labels))
        for _, value := range labels {
            labelValues = append(labelValues, value)
        }
        gauge.WithLabelValues(labelValues...).Set(value)
    }
}
```

### Trace Collection
```go
type TraceCollector struct {
    tracer     trace.Tracer
    spans      chan trace.ReadOnlySpan
    batchSize  int
    timeout    time.Duration
}

func (tc *TraceCollector) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
    ctx, span := tc.tracer.Start(ctx, name, opts...)
    
    // Send span to processing channel
    select {
    case tc.spans <- span:
    default:
        // Channel full, drop span
        log.Warn("Trace span dropped due to full channel")
    }
    
    return ctx, span
}

func (tc *TraceCollector) ProcessSpans() {
    batch := make([]trace.ReadOnlySpan, 0, tc.batchSize)
    ticker := time.NewTicker(tc.timeout)
    
    for {
        select {
        case span := <-tc.spans:
            batch = append(batch, span)
            if len(batch) >= tc.batchSize {
                tc.exportBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                tc.exportBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

### Log Collection
```go
type LogCollector struct {
    logs      chan LogEntry
    batchSize int
    timeout   time.Duration
    exporter  LogExporter
}

type LogEntry struct {
    Timestamp time.Time
    Level     string
    Message   string
    Fields    map[string]interface{}
    TraceID   string
    SpanID    string
}

func (lc *LogCollector) CollectLog(entry LogEntry) {
    select {
    case lc.logs <- entry:
    default:
        // Channel full, drop log
        log.Warn("Log entry dropped due to full channel")
    }
}

func (lc *LogCollector) ProcessLogs() {
    batch := make([]LogEntry, 0, lc.batchSize)
    ticker := time.NewTicker(lc.timeout)
    
    for {
        select {
        case logEntry := <-lc.logs:
            batch = append(batch, logEntry)
            if len(batch) >= lc.batchSize {
                lc.exportBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                lc.exportBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

## Performance Metrics

### Application Performance
```go
type ApplicationMetrics struct {
    requestRate      *prometheus.CounterVec
    requestDuration  *prometheus.HistogramVec
    responseSize     *prometheus.HistogramVec
    errorRate        *prometheus.CounterVec
    activeRequests   prometheus.Gauge
    memoryUsage      prometheus.Gauge
    cpuUsage         prometheus.Gauge
    goroutines       prometheus.Gauge
}

func NewApplicationMetrics() *ApplicationMetrics {
    return &ApplicationMetrics{
        requestRate: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "app_requests_total",
                Help: "Total number of requests",
            },
            []string{"method", "endpoint", "status"},
        ),
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "app_request_duration_seconds",
                Help:    "Request duration in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"method", "endpoint"},
        ),
        responseSize: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "app_response_size_bytes",
                Help:    "Response size in bytes",
                Buckets: []float64{100, 1000, 10000, 100000, 1000000},
            },
            []string{"method", "endpoint"},
        ),
        errorRate: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "app_errors_total",
                Help: "Total number of errors",
            },
            []string{"error_type", "endpoint"},
        ),
        activeRequests: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "app_active_requests",
                Help: "Current number of active requests",
            },
        ),
        memoryUsage: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "app_memory_usage_bytes",
                Help: "Current memory usage in bytes",
            },
        ),
        cpuUsage: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "app_cpu_usage_percent",
                Help: "Current CPU usage percentage",
            },
        ),
        goroutines: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "app_goroutines",
                Help: "Current number of goroutines",
            },
        ),
    }
}
```

### Database Performance
```go
type DatabaseMetrics struct {
    queryDuration    *prometheus.HistogramVec
    queryErrors      *prometheus.CounterVec
    connectionPool   *prometheus.GaugeVec
    transactionCount *prometheus.CounterVec
    lockWaitTime     *prometheus.HistogramVec
}

func (dm *DatabaseMetrics) RecordQuery(query string, duration time.Duration, err error) {
    labels := []string{query}
    dm.queryDuration.WithLabelValues(labels...).Observe(duration.Seconds())
    
    if err != nil {
        dm.queryErrors.WithLabelValues(query, err.Error()).Inc()
    }
}

func (dm *DatabaseMetrics) RecordTransaction(operation string, duration time.Duration, err error) {
    labels := []string{operation}
    dm.transactionCount.WithLabelValues(labels...).Inc()
    
    if err != nil {
        dm.queryErrors.WithLabelValues(operation, err.Error()).Inc()
    }
}
```

### External Service Performance
```go
type ExternalServiceMetrics struct {
    callDuration     *prometheus.HistogramVec
    callErrors       *prometheus.CounterVec
    callRate         *prometheus.CounterVec
    timeoutRate      *prometheus.CounterVec
    circuitBreaker   *prometheus.GaugeVec
}

func (esm *ExternalServiceMetrics) RecordCall(service, method string, duration time.Duration, err error) {
    labels := []string{service, method}
    esm.callDuration.WithLabelValues(labels...).Observe(duration.Seconds())
    esm.callRate.WithLabelValues(labels...).Inc()
    
    if err != nil {
        esm.callErrors.WithLabelValues(labels...).Inc()
        
        if isTimeout(err) {
            esm.timeoutRate.WithLabelValues(labels...).Inc()
        }
    }
}
```

## Error Tracking

### Error Collection
```go
type ErrorTracker struct {
    errors     chan ErrorEvent
    batchSize  int
    timeout    time.Duration
    exporter   ErrorExporter
}

type ErrorEvent struct {
    Timestamp   time.Time
    Error       error
    StackTrace  string
    Context     map[string]interface{}
    UserID      string
    SessionID   string
    TraceID     string
    SpanID      string
    Service     string
    Version     string
    Environment string
}

func (et *ErrorTracker) TrackError(err error, ctx context.Context, additionalContext map[string]interface{}) {
    errorEvent := ErrorEvent{
        Timestamp:   time.Now(),
        Error:       err,
        StackTrace:  getStackTrace(),
        Context:     additionalContext,
        TraceID:     getTraceID(ctx),
        SpanID:      getSpanID(ctx),
        Service:     getServiceName(),
        Version:     getVersion(),
        Environment: getEnvironment(),
    }
    
    select {
    case et.errors <- errorEvent:
    default:
        // Channel full, drop error
        log.Warn("Error event dropped due to full channel")
    }
}

func (et *ErrorTracker) ProcessErrors() {
    batch := make([]ErrorEvent, 0, et.batchSize)
    ticker := time.NewTicker(et.timeout)
    
    for {
        select {
        case errorEvent := <-et.errors:
            batch = append(batch, errorEvent)
            if len(batch) >= et.batchSize {
                et.exportBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                et.exportBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

### Error Aggregation
```go
type ErrorAggregator struct {
    errors    map[string]*ErrorSummary
    mutex     sync.RWMutex
    exporter  ErrorExporter
}

type ErrorSummary struct {
    Count       int
    FirstSeen   time.Time
    LastSeen    time.Time
    StackTraces map[string]int
    Contexts    map[string]int
}

func (ea *ErrorAggregator) AggregateError(errorEvent ErrorEvent) {
    errorKey := generateErrorKey(errorEvent.Error)
    
    ea.mutex.Lock()
    defer ea.mutex.Unlock()
    
    summary, exists := ea.errors[errorKey]
    if !exists {
        summary = &ErrorSummary{
            FirstSeen:   errorEvent.Timestamp,
            StackTraces: make(map[string]int),
            Contexts:    make(map[string]int),
        }
        ea.errors[errorKey] = summary
    }
    
    summary.Count++
    summary.LastSeen = errorEvent.Timestamp
    summary.StackTraces[errorEvent.StackTrace]++
    
    // Aggregate context
    for key, value := range errorEvent.Context {
        contextKey := fmt.Sprintf("%s:%v", key, value)
        summary.Contexts[contextKey]++
    }
}
```

## User Experience Monitoring

### User Session Tracking
```go
type UserTracker struct {
    sessions  map[string]*UserSession
    mutex     sync.RWMutex
    exporter  UserExporter
}

type UserSession struct {
    SessionID    string
    UserID       string
    StartTime    time.Time
    EndTime      time.Time
    PageViews    []PageView
    Actions      []UserAction
    Errors       []ErrorEvent
    Performance  *PerformanceMetrics
}

type PageView struct {
    URL         string
    Timestamp   time.Time
    Duration    time.Duration
    LoadTime    time.Duration
    RenderTime  time.Duration
}

type UserAction struct {
    Action      string
    Element     string
    Timestamp   time.Time
    Duration    time.Duration
    Success     bool
}

func (ut *UserTracker) TrackPageView(sessionID, userID, url string, loadTime, renderTime time.Duration) {
    ut.mutex.Lock()
    defer ut.mutex.Unlock()
    
    session, exists := ut.sessions[sessionID]
    if !exists {
        session = &UserSession{
            SessionID:   sessionID,
            UserID:      userID,
            StartTime:   time.Now(),
            PageViews:   make([]PageView, 0),
            Actions:     make([]UserAction, 0),
            Errors:      make([]ErrorEvent, 0),
            Performance: &PerformanceMetrics{},
        }
        ut.sessions[sessionID] = session
    }
    
    pageView := PageView{
        URL:        url,
        Timestamp:  time.Now(),
        LoadTime:   loadTime,
        RenderTime: renderTime,
    }
    
    session.PageViews = append(session.PageViews, pageView)
}
```

### Performance Metrics
```go
type PerformanceMetrics struct {
    FirstContentfulPaint time.Duration
    LargestContentfulPaint time.Duration
    FirstInputDelay      time.Duration
    CumulativeLayoutShift float64
    FirstByteTime        time.Duration
    DOMContentLoaded     time.Duration
    LoadComplete         time.Duration
}

func (pm *PerformanceMetrics) UpdateFromWebVitals(vitals WebVitals) {
    pm.FirstContentfulPaint = vitals.FCP
    pm.LargestContentfulPaint = vitals.LCP
    pm.FirstInputDelay = vitals.FID
    pm.CumulativeLayoutShift = vitals.CLS
    pm.FirstByteTime = vitals.TTFB
    pm.DOMContentLoaded = vitals.DOMContentLoaded
    pm.LoadComplete = vitals.LoadComplete
}
```

## Golang Implementation

### APM Middleware
```go
func APMMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Extract user context
        userID := getUserID(r)
        sessionID := getSessionID(r)
        
        // Create trace context
        ctx := r.Context()
        ctx = context.WithValue(ctx, "user_id", userID)
        ctx = context.WithValue(ctx, "session_id", sessionID)
        
        // Wrap response writer
        ww := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // Process request
        next.ServeHTTP(ww, r.WithContext(ctx))
        
        // Record metrics
        duration := time.Since(start)
        recordRequestMetrics(r, ww.statusCode, duration)
        
        // Record user experience
        recordUserExperience(userID, sessionID, r.URL.Path, duration)
    })
}
```

### APM Client
```go
type APMClient struct {
    config     *APMConfig
    metrics    *ApplicationMetrics
    tracer     trace.Tracer
    errorTracker *ErrorTracker
    userTracker  *UserTracker
}

func NewAPMClient(config *APMConfig) *APMClient {
    return &APMClient{
        config:       config,
        metrics:      NewApplicationMetrics(),
        tracer:       otel.Tracer(config.ServiceName),
        errorTracker: NewErrorTracker(),
        userTracker:  NewUserTracker(),
    }
}

func (ac *APMClient) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
    return ac.tracer.Start(ctx, name, opts...)
}

func (ac *APMClient) TrackError(err error, ctx context.Context, additionalContext map[string]interface{}) {
    ac.errorTracker.TrackError(err, ctx, additionalContext)
}

func (ac *APMClient) TrackUserAction(userID, sessionID, action, element string, duration time.Duration, success bool) {
    ac.userTracker.TrackAction(userID, sessionID, action, element, duration, success)
}
```

## Microservices APM

### Service Discovery
```go
type ServiceDiscovery struct {
    services  map[string]*ServiceInfo
    mutex     sync.RWMutex
    updater   ServiceUpdater
}

type ServiceInfo struct {
    Name        string
    Version     string
    Environment string
    Endpoints   []string
    Health      string
    Metrics     map[string]float64
}

func (sd *ServiceDiscovery) UpdateService(service *ServiceInfo) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    sd.services[service.Name] = service
    sd.updater.UpdateService(service)
}
```

### Cross-Service Tracing
```go
type CrossServiceTracer struct {
    tracer    trace.Tracer
    propagator propagation.TextMapPropagator
}

func (cst *CrossServiceTracer) StartServiceCall(ctx context.Context, service, method string, req interface{}) (context.Context, trace.Span) {
    ctx, span := cst.tracer.Start(ctx, fmt.Sprintf("%s.%s", service, method))
    
    span.SetAttributes(
        attribute.String("service.name", service),
        attribute.String("service.method", method),
    )
    
    return ctx, span
}

func (cst *CrossServiceTracer) PropagateContext(ctx context.Context, headers http.Header) {
    cst.propagator.Inject(ctx, propagation.HeaderCarrier(headers))
}

func (cst *CrossServiceTracer) ExtractContext(ctx context.Context, headers http.Header) context.Context {
    return cst.propagator.Extract(ctx, propagation.HeaderCarrier(headers))
}
```

## Alerting and Dashboards

### Alert Rules
```go
type AlertRule struct {
    Name        string
    Expression  string
    Duration    time.Duration
    Severity    AlertSeverity
    Labels      map[string]string
    Annotations map[string]string
}

type AlertManager struct {
    rules   []AlertRule
    client  *http.Client
    baseURL string
}

func (am *AlertManager) EvaluateRules(metrics map[string]float64) []Alert {
    var alerts []Alert
    
    for _, rule := range am.rules {
        if am.evaluateRule(rule, metrics) {
            alert := Alert{
                Name:        rule.Name,
                Severity:    rule.Severity,
                Labels:      rule.Labels,
                Annotations: rule.Annotations,
                Timestamp:   time.Now(),
            }
            alerts = append(alerts, alert)
        }
    }
    
    return alerts
}
```

### Dashboard Configuration
```go
type DashboardConfig struct {
    Title       string
    Panels      []Panel
    RefreshRate time.Duration
    TimeRange   TimeRange
}

type Panel struct {
    Title  string
    Type   string
    Query  string
    Width  int
    Height int
}

func (dc *DashboardConfig) GenerateDashboard() *Dashboard {
    dashboard := &Dashboard{
        Title:       dc.Title,
        RefreshRate: dc.RefreshRate,
        TimeRange:   dc.TimeRange,
        Panels:      make([]Panel, len(dc.Panels)),
    }
    
    for i, panel := range dc.Panels {
        dashboard.Panels[i] = panel
    }
    
    return dashboard
}
```

## Best Practices

### 1. Data Collection
```go
// Good: Efficient data collection
func (ac *APMClient) CollectMetrics() {
    // Collect only necessary metrics
    ac.metrics.Update()
    
    // Use batching for efficiency
    ac.batchCollector.Collect()
}

// Bad: Collecting too much data
func (ac *APMClient) CollectAllData() {
    // Collecting everything is expensive
    ac.collectEverything()
}
```

### 2. Error Handling
```go
// Good: Proper error handling
func (ac *APMClient) TrackError(err error, ctx context.Context) {
    if err != nil {
        ac.errorTracker.TrackError(err, ctx, nil)
    }
}

// Bad: Ignoring errors
func (ac *APMClient) TrackError(err error, ctx context.Context) {
    // Error ignored
}
```

### 3. Performance Optimization
```go
// Good: Asynchronous processing
func (ac *APMClient) ProcessData(data interface{}) {
    go func() {
        ac.processor.Process(data)
    }()
}

// Bad: Synchronous processing
func (ac *APMClient) ProcessData(data interface{}) {
    ac.processor.Process(data) // Blocks
}
```

### 4. Resource Management
```go
// Good: Proper resource cleanup
func (ac *APMClient) Shutdown() error {
    var errs []error
    
    if err := ac.metrics.Shutdown(); err != nil {
        errs = append(errs, err)
    }
    
    if err := ac.tracer.Shutdown(); err != nil {
        errs = append(errs, err)
    }
    
    if len(errs) > 0 {
        return fmt.Errorf("shutdown errors: %v", errs)
    }
    
    return nil
}
```

## Conclusion

APM implementation provides:

1. **Comprehensive Monitoring**: Metrics, traces, logs, and errors
2. **User Experience**: Track user behavior and performance
3. **Performance Analysis**: Identify bottlenecks and optimize
4. **Error Tracking**: Debug issues and improve reliability
5. **Business Insights**: Connect technical metrics to business outcomes

Key principles:
- Collect relevant data efficiently
- Use appropriate sampling strategies
- Implement proper error handling
- Monitor both technical and business metrics
- Continuously optimize based on insights
- Ensure data privacy and security
