# 📊 **Observability and Monitoring Engineering Guide**

## **Table of Contents**
1. [🔍 Structured Logging Systems](#-structured-logging-systems)
2. [📈 Metrics Collection and Aggregation](#-metrics-collection-and-aggregation)
3. [🔗 Distributed Tracing](#-distributed-tracing)
4. [🚨 Alerting and Incident Management](#-alerting-and-incident-management)
5. [📊 Monitoring Dashboards](#-monitoring-dashboards)
6. [🔧 SRE Practices](#-sre-practices)
7. [🐛 Troubleshooting Methodologies](#-troubleshooting-methodologies)
8. [❓ Interview Questions](#-interview-questions)

---

## 🔍 **Structured Logging Systems**

### **Advanced Logging Framework**

```go
package logging

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "os"
    "runtime"
    "sync"
    "time"
    "github.com/google/uuid"
)

// Logger levels
type LogLevel int

const (
    DEBUG LogLevel = iota
    INFO
    WARN
    ERROR
    FATAL
)

var logLevelNames = map[LogLevel]string{
    DEBUG: "DEBUG",
    INFO:  "INFO",
    WARN:  "WARN",
    ERROR: "ERROR",
    FATAL: "FATAL",
}

// Structured logger
type StructuredLogger struct {
    level       LogLevel
    outputs     []io.Writer
    hooks       []LogHook
    fields      map[string]interface{}
    mu          sync.RWMutex
    formatter   LogFormatter
    sampler     *LogSampler
    asyncBuffer chan LogEntry
    bufferSize  int
}

type LogEntry struct {
    Level       LogLevel               `json:"level"`
    Message     string                 `json:"message"`
    Timestamp   time.Time              `json:"timestamp"`
    Fields      map[string]interface{} `json:"fields"`
    TraceID     string                 `json:"trace_id,omitempty"`
    SpanID      string                 `json:"span_id,omitempty"`
    ServiceName string                 `json:"service_name"`
    Hostname    string                 `json:"hostname"`
    Caller      string                 `json:"caller,omitempty"`
    StackTrace  string                 `json:"stack_trace,omitempty"`
}

type LogHook interface {
    Execute(entry LogEntry) error
    Levels() []LogLevel
}

type LogFormatter interface {
    Format(entry LogEntry) ([]byte, error)
}

type LogSampler struct {
    sampleRate float64
    counter    uint64
    mu         sync.Mutex
}

func NewStructuredLogger(level LogLevel, bufferSize int) *StructuredLogger {
    hostname, _ := os.Hostname()
    
    logger := &StructuredLogger{
        level:       level,
        outputs:     []io.Writer{os.Stdout},
        fields:      make(map[string]interface{}),
        formatter:   &JSONFormatter{},
        bufferSize:  bufferSize,
        asyncBuffer: make(chan LogEntry, bufferSize),
    }
    
    logger.fields["hostname"] = hostname
    logger.fields["pid"] = os.Getpid()
    
    // Start async logging goroutine
    go logger.processAsyncLogs()
    
    return logger
}

func (sl *StructuredLogger) processAsyncLogs() {
    for entry := range sl.asyncBuffer {
        sl.writeLogEntry(entry)
    }
}

func (sl *StructuredLogger) WithField(key string, value interface{}) *StructuredLogger {
    sl.mu.Lock()
    defer sl.mu.Unlock()
    
    newLogger := *sl
    newLogger.fields = make(map[string]interface{})
    
    // Copy existing fields
    for k, v := range sl.fields {
        newLogger.fields[k] = v
    }
    
    newLogger.fields[key] = value
    return &newLogger
}

func (sl *StructuredLogger) WithFields(fields map[string]interface{}) *StructuredLogger {
    sl.mu.Lock()
    defer sl.mu.Unlock()
    
    newLogger := *sl
    newLogger.fields = make(map[string]interface{})
    
    // Copy existing fields
    for k, v := range sl.fields {
        newLogger.fields[k] = v
    }
    
    // Add new fields
    for k, v := range fields {
        newLogger.fields[k] = v
    }
    
    return &newLogger
}

func (sl *StructuredLogger) WithContext(ctx context.Context) *StructuredLogger {
    logger := sl
    
    // Extract trace information from context
    if traceID := getTraceIDFromContext(ctx); traceID != "" {
        logger = logger.WithField("trace_id", traceID)
    }
    
    if spanID := getSpanIDFromContext(ctx); spanID != "" {
        logger = logger.WithField("span_id", spanID)
    }
    
    // Extract user information
    if userID := getUserIDFromContext(ctx); userID != "" {
        logger = logger.WithField("user_id", userID)
    }
    
    // Extract request ID
    if requestID := getRequestIDFromContext(ctx); requestID != "" {
        logger = logger.WithField("request_id", requestID)
    }
    
    return logger
}

func (sl *StructuredLogger) log(level LogLevel, message string, fields map[string]interface{}) {
    if level < sl.level {
        return
    }
    
    // Apply sampling if configured
    if sl.sampler != nil && !sl.sampler.ShouldSample() {
        return
    }
    
    sl.mu.RLock()
    defer sl.mu.RUnlock()
    
    // Merge fields
    mergedFields := make(map[string]interface{})
    for k, v := range sl.fields {
        mergedFields[k] = v
    }
    for k, v := range fields {
        mergedFields[k] = v
    }
    
    entry := LogEntry{
        Level:       level,
        Message:     message,
        Timestamp:   time.Now().UTC(),
        Fields:      mergedFields,
        ServiceName: getServiceName(),
        Hostname:    sl.fields["hostname"].(string),
    }
    
    // Add caller information for ERROR and FATAL levels
    if level >= ERROR {
        entry.Caller = getCaller(3)
    }
    
    // Add stack trace for FATAL level
    if level == FATAL {
        entry.StackTrace = getStackTrace()
    }
    
    // Execute hooks
    for _, hook := range sl.hooks {
        if contains(hook.Levels(), level) {
            go hook.Execute(entry) // Execute hooks asynchronously
        }
    }
    
    // Send to async buffer or write directly
    select {
    case sl.asyncBuffer <- entry:
        // Successfully queued for async processing
    default:
        // Buffer full, write directly
        sl.writeLogEntry(entry)
    }
}

func (sl *StructuredLogger) writeLogEntry(entry LogEntry) {
    formatted, err := sl.formatter.Format(entry)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to format log entry: %v\n", err)
        return
    }
    
    for _, output := range sl.outputs {
        output.Write(formatted)
        output.Write([]byte("\n"))
    }
}

func (sl *StructuredLogger) Debug(message string, fields ...map[string]interface{}) {
    mergedFields := mergeFields(fields...)
    sl.log(DEBUG, message, mergedFields)
}

func (sl *StructuredLogger) Info(message string, fields ...map[string]interface{}) {
    mergedFields := mergeFields(fields...)
    sl.log(INFO, message, mergedFields)
}

func (sl *StructuredLogger) Warn(message string, fields ...map[string]interface{}) {
    mergedFields := mergeFields(fields...)
    sl.log(WARN, message, mergedFields)
}

func (sl *StructuredLogger) Error(message string, fields ...map[string]interface{}) {
    mergedFields := mergeFields(fields...)
    sl.log(ERROR, message, mergedFields)
}

func (sl *StructuredLogger) Fatal(message string, fields ...map[string]interface{}) {
    mergedFields := mergeFields(fields...)
    sl.log(FATAL, message, mergedFields)
    os.Exit(1)
}

// JSON Formatter
type JSONFormatter struct{}

func (jf *JSONFormatter) Format(entry LogEntry) ([]byte, error) {
    return json.Marshal(entry)
}

// Log Sampling
func NewLogSampler(sampleRate float64) *LogSampler {
    return &LogSampler{
        sampleRate: sampleRate,
    }
}

func (ls *LogSampler) ShouldSample() bool {
    ls.mu.Lock()
    defer ls.mu.Unlock()
    
    ls.counter++
    return (float64(ls.counter) / 100.0) <= ls.sampleRate
}

// Slack notification hook
type SlackHook struct {
    WebhookURL string
    Channel    string
    Username   string
    levels     []LogLevel
}

func NewSlackHook(webhookURL, channel, username string) *SlackHook {
    return &SlackHook{
        WebhookURL: webhookURL,
        Channel:    channel,
        Username:   username,
        levels:     []LogLevel{ERROR, FATAL},
    }
}

func (sh *SlackHook) Execute(entry LogEntry) error {
    payload := map[string]interface{}{
        "channel":  sh.Channel,
        "username": sh.Username,
        "text":     fmt.Sprintf("🚨 %s: %s", logLevelNames[entry.Level], entry.Message),
        "attachments": []map[string]interface{}{
            {
                "color": getColorForLevel(entry.Level),
                "fields": []map[string]interface{}{
                    {"title": "Service", "value": entry.ServiceName, "short": true},
                    {"title": "Hostname", "value": entry.Hostname, "short": true},
                    {"title": "Timestamp", "value": entry.Timestamp.Format(time.RFC3339), "short": true},
                },
            },
        },
    }
    
    // Send HTTP request to Slack webhook
    return sendSlackMessage(sh.WebhookURL, payload)
}

func (sh *SlackHook) Levels() []LogLevel {
    return sh.levels
}

// Utility functions
func getCaller(skip int) string {
    _, file, line, ok := runtime.Caller(skip)
    if !ok {
        return "unknown"
    }
    return fmt.Sprintf("%s:%d", file, line)
}

func getStackTrace() string {
    buf := make([]byte, 4096)
    n := runtime.Stack(buf, false)
    return string(buf[:n])
}

func getServiceName() string {
    if name := os.Getenv("SERVICE_NAME"); name != "" {
        return name
    }
    return "unknown-service"
}

func mergeFields(fields ...map[string]interface{}) map[string]interface{} {
    result := make(map[string]interface{})
    for _, field := range fields {
        for k, v := range field {
            result[k] = v
        }
    }
    return result
}
```

---

## 📈 **Metrics Collection and Aggregation**

### **Advanced Metrics System**

```go
package metrics

import (
    "context"
    "encoding/json"
    "fmt"
    "math"
    "net/http"
    "sort"
    "sync"
    "sync/atomic"
    "time"
)

// Metric types
type MetricType int

const (
    CounterType MetricType = iota
    GaugeType
    HistogramType
    TimerType
    SummaryType
)

// Metric interface
type Metric interface {
    Name() string
    Type() MetricType
    Value() interface{}
    Labels() map[string]string
    Timestamp() time.Time
}

// Metrics registry
type MetricsRegistry struct {
    metrics map[string]Metric
    mu      sync.RWMutex
    tags    map[string]string
}

func NewMetricsRegistry() *MetricsRegistry {
    return &MetricsRegistry{
        metrics: make(map[string]Metric),
        tags:    make(map[string]string),
    }
}

// Counter metric
type Counter struct {
    name      string
    value     uint64
    labels    map[string]string
    timestamp time.Time
}

func NewCounter(name string, labels map[string]string) *Counter {
    return &Counter{
        name:      name,
        labels:    labels,
        timestamp: time.Now(),
    }
}

func (c *Counter) Name() string { return c.name }
func (c *Counter) Type() MetricType { return CounterType }
func (c *Counter) Labels() map[string]string { return c.labels }
func (c *Counter) Timestamp() time.Time { return c.timestamp }

func (c *Counter) Inc() {
    atomic.AddUint64(&c.value, 1)
    c.timestamp = time.Now()
}

func (c *Counter) Add(delta uint64) {
    atomic.AddUint64(&c.value, delta)
    c.timestamp = time.Now()
}

func (c *Counter) Value() interface{} {
    return atomic.LoadUint64(&c.value)
}

// Gauge metric
type Gauge struct {
    name      string
    value     int64
    labels    map[string]string
    timestamp time.Time
    mu        sync.RWMutex
}

func NewGauge(name string, labels map[string]string) *Gauge {
    return &Gauge{
        name:      name,
        labels:    labels,
        timestamp: time.Now(),
    }
}

func (g *Gauge) Name() string { return g.name }
func (g *Gauge) Type() MetricType { return GaugeType }
func (g *Gauge) Labels() map[string]string { return g.labels }
func (g *Gauge) Timestamp() time.Time { return g.timestamp }

func (g *Gauge) Set(value float64) {
    g.mu.Lock()
    defer g.mu.Unlock()
    g.value = int64(math.Float64bits(value))
    g.timestamp = time.Now()
}

func (g *Gauge) Inc() {
    g.Add(1)
}

func (g *Gauge) Dec() {
    g.Add(-1)
}

func (g *Gauge) Add(delta float64) {
    g.mu.Lock()
    defer g.mu.Unlock()
    current := math.Float64frombits(uint64(g.value))
    g.value = int64(math.Float64bits(current + delta))
    g.timestamp = time.Now()
}

func (g *Gauge) Value() interface{} {
    g.mu.RLock()
    defer g.mu.RUnlock()
    return math.Float64frombits(uint64(g.value))
}

// Histogram metric
type Histogram struct {
    name      string
    buckets   []float64
    counts    []uint64
    sum       uint64
    count     uint64
    labels    map[string]string
    timestamp time.Time
    mu        sync.RWMutex
}

func NewHistogram(name string, buckets []float64, labels map[string]string) *Histogram {
    sort.Float64s(buckets)
    return &Histogram{
        name:      name,
        buckets:   buckets,
        counts:    make([]uint64, len(buckets)+1), // +1 for +Inf bucket
        labels:    labels,
        timestamp: time.Now(),
    }
}

func (h *Histogram) Name() string { return h.name }
func (h *Histogram) Type() MetricType { return HistogramType }
func (h *Histogram) Labels() map[string]string { return h.labels }
func (h *Histogram) Timestamp() time.Time { return h.timestamp }

func (h *Histogram) Observe(value float64) {
    h.mu.Lock()
    defer h.mu.Unlock()
    
    // Update sum and count
    atomic.AddUint64(&h.sum, uint64(math.Float64bits(value)))
    atomic.AddUint64(&h.count, 1)
    
    // Find appropriate bucket
    for i, bucket := range h.buckets {
        if value <= bucket {
            atomic.AddUint64(&h.counts[i], 1)
        }
    }
    // Always increment the +Inf bucket
    atomic.AddUint64(&h.counts[len(h.buckets)], 1)
    
    h.timestamp = time.Now()
}

func (h *Histogram) Value() interface{} {
    h.mu.RLock()
    defer h.mu.RUnlock()
    
    bucketCounts := make(map[string]uint64)
    for i, bucket := range h.buckets {
        bucketCounts[fmt.Sprintf("%.2f", bucket)] = atomic.LoadUint64(&h.counts[i])
    }
    bucketCounts["+Inf"] = atomic.LoadUint64(&h.counts[len(h.buckets)])
    
    return map[string]interface{}{
        "buckets": bucketCounts,
        "sum":     math.Float64frombits(atomic.LoadUint64(&h.sum)),
        "count":   atomic.LoadUint64(&h.count),
    }
}

// Timer metric (specialized histogram for duration measurements)
type Timer struct {
    histogram *Histogram
    startTime time.Time
}

func NewTimer(name string, labels map[string]string) *Timer {
    buckets := []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0}
    return &Timer{
        histogram: NewHistogram(name, buckets, labels),
    }
}

func (t *Timer) Start() {
    t.startTime = time.Now()
}

func (t *Timer) Stop() {
    if !t.startTime.IsZero() {
        duration := time.Since(t.startTime).Seconds()
        t.histogram.Observe(duration)
        t.startTime = time.Time{}
    }
}

func (t *Timer) Time(fn func()) {
    start := time.Now()
    fn()
    duration := time.Since(start).Seconds()
    t.histogram.Observe(duration)
}

func (t *Timer) Name() string { return t.histogram.Name() }
func (t *Timer) Type() MetricType { return TimerType }
func (t *Timer) Labels() map[string]string { return t.histogram.Labels() }
func (t *Timer) Timestamp() time.Time { return t.histogram.Timestamp() }
func (t *Timer) Value() interface{} { return t.histogram.Value() }

// Metrics aggregator
type MetricsAggregator struct {
    registry     *MetricsRegistry
    exporters    []MetricExporter
    scrapeInterval time.Duration
    stopChan     chan struct{}
    wg           sync.WaitGroup
}

type MetricExporter interface {
    Export(metrics []Metric) error
    Name() string
}

func NewMetricsAggregator(registry *MetricsRegistry, scrapeInterval time.Duration) *MetricsAggregator {
    return &MetricsAggregator{
        registry:       registry,
        scrapeInterval: scrapeInterval,
        stopChan:       make(chan struct{}),
    }
}

func (ma *MetricsAggregator) AddExporter(exporter MetricExporter) {
    ma.exporters = append(ma.exporters, exporter)
}

func (ma *MetricsAggregator) Start() {
    ma.wg.Add(1)
    go ma.run()
}

func (ma *MetricsAggregator) Stop() {
    close(ma.stopChan)
    ma.wg.Wait()
}

func (ma *MetricsAggregator) run() {
    defer ma.wg.Done()
    
    ticker := time.NewTicker(ma.scrapeInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            metrics := ma.collectMetrics()
            ma.exportMetrics(metrics)
        case <-ma.stopChan:
            return
        }
    }
}

func (ma *MetricsAggregator) collectMetrics() []Metric {
    ma.registry.mu.RLock()
    defer ma.registry.mu.RUnlock()
    
    metrics := make([]Metric, 0, len(ma.registry.metrics))
    for _, metric := range ma.registry.metrics {
        metrics = append(metrics, metric)
    }
    
    return metrics
}

func (ma *MetricsAggregator) exportMetrics(metrics []Metric) {
    for _, exporter := range ma.exporters {
        go func(exp MetricExporter) {
            if err := exp.Export(metrics); err != nil {
                fmt.Printf("Failed to export metrics via %s: %v\n", exp.Name(), err)
            }
        }(exporter)
    }
}

// Prometheus exporter
type PrometheusExporter struct {
    endpoint string
    client   *http.Client
}

func NewPrometheusExporter(endpoint string) *PrometheusExporter {
    return &PrometheusExporter{
        endpoint: endpoint,
        client:   &http.Client{Timeout: 10 * time.Second},
    }
}

func (pe *PrometheusExporter) Name() string {
    return "prometheus"
}

func (pe *PrometheusExporter) Export(metrics []Metric) error {
    var prometheusFormat []string
    
    for _, metric := range metrics {
        line := pe.formatMetric(metric)
        if line != "" {
            prometheusFormat = append(prometheusFormat, line)
        }
    }
    
    // Send to Prometheus push gateway
    return pe.pushToPrometheus(prometheusFormat)
}

func (pe *PrometheusExporter) formatMetric(metric Metric) string {
    name := metric.Name()
    labels := metric.Labels()
    value := metric.Value()
    
    labelStr := ""
    if len(labels) > 0 {
        var labelPairs []string
        for k, v := range labels {
            labelPairs = append(labelPairs, fmt.Sprintf(`%s="%s"`, k, v))
        }
        labelStr = "{" + fmt.Sprintf("%s", labelPairs) + "}"
    }
    
    switch metric.Type() {
    case CounterType:
        return fmt.Sprintf("%s_total%s %v", name, labelStr, value)
    case GaugeType:
        return fmt.Sprintf("%s%s %v", name, labelStr, value)
    case HistogramType:
        // Handle histogram formatting
        return pe.formatHistogram(name, labelStr, value)
    default:
        return fmt.Sprintf("%s%s %v", name, labelStr, value)
    }
}

func (pe *PrometheusExporter) formatHistogram(name, labelStr string, value interface{}) string {
    histData := value.(map[string]interface{})
    buckets := histData["buckets"].(map[string]uint64)
    sum := histData["sum"].(float64)
    count := histData["count"].(uint64)
    
    var lines []string
    
    // Bucket lines
    for bucket, count := range buckets {
        lines = append(lines, fmt.Sprintf(`%s_bucket{le="%s"%s} %d`, 
            name, bucket, labelStr, count))
    }
    
    // Sum and count lines
    lines = append(lines, fmt.Sprintf("%s_sum%s %f", name, labelStr, sum))
    lines = append(lines, fmt.Sprintf("%s_count%s %d", name, labelStr, count))
    
    return fmt.Sprintf("%s", lines)
}

func (pe *PrometheusExporter) pushToPrometheus(metrics []string) error {
    // Implementation would send HTTP POST to Prometheus push gateway
    // This is a placeholder
    fmt.Printf("Pushing %d metrics to Prometheus\n", len(metrics))
    return nil
}

// Application metrics collector
type ApplicationMetricsCollector struct {
    registry    *MetricsRegistry
    httpMetrics *HTTPMetrics
    dbMetrics   *DatabaseMetrics
    grpcMetrics *GRPCMetrics
}

type HTTPMetrics struct {
    requestCount    *Counter
    requestDuration *Timer
    requestSize     *Histogram
    responseSize    *Histogram
}

func NewApplicationMetricsCollector(registry *MetricsRegistry) *ApplicationMetricsCollector {
    return &ApplicationMetricsCollector{
        registry: registry,
        httpMetrics: &HTTPMetrics{
            requestCount: NewCounter("http_requests_total", 
                map[string]string{"method": "", "status_code": ""}),
            requestDuration: NewTimer("http_request_duration_seconds", 
                map[string]string{"method": "", "handler": ""}),
            requestSize: NewHistogram("http_request_size_bytes", 
                []float64{100, 1000, 10000, 100000, 1000000}, 
                map[string]string{"method": ""}),
            responseSize: NewHistogram("http_response_size_bytes", 
                []float64{100, 1000, 10000, 100000, 1000000}, 
                map[string]string{"method": "", "status_code": ""}),
        },
    }
}

// HTTP middleware for metrics collection
func (amc *ApplicationMetricsCollector) HTTPMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap response writer to capture status code and size
        wrappedWriter := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // Record request size
        if r.ContentLength > 0 {
            amc.httpMetrics.requestSize.Observe(float64(r.ContentLength))
        }
        
        // Execute handler
        next.ServeHTTP(wrappedWriter, r)
        
        // Record metrics
        duration := time.Since(start).Seconds()
        amc.httpMetrics.requestDuration.histogram.Observe(duration)
        
        // Update counter with labels
        counter := NewCounter("http_requests_total", map[string]string{
            "method":      r.Method,
            "status_code": fmt.Sprintf("%d", wrappedWriter.statusCode),
        })
        counter.Inc()
        
        // Record response size
        amc.httpMetrics.responseSize.Observe(float64(wrappedWriter.bytesWritten))
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode   int
    bytesWritten int64
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
    n, err := rw.ResponseWriter.Write(b)
    rw.bytesWritten += int64(n)
    return n, err
}
```

---

## 🔗 **Distributed Tracing**

### **OpenTelemetry Integration**

```go
package tracing

import (
    "context"
    "fmt"
    "net/http"
    "time"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    "go.opentelemetry.io/otel/semconv/v1.4.0"
    oteltrace "go.opentelemetry.io/otel/trace"
)

// Distributed tracing manager
type TracingManager struct {
    tracer   oteltrace.Tracer
    provider *trace.TracerProvider
    config   TracingConfig
}

type TracingConfig struct {
    ServiceName     string
    ServiceVersion  string
    Environment     string
    JaegerEndpoint  string
    SamplingRatio   float64
    MaxTagLength    int
    MaxSpansPerTrace int
}

func NewTracingManager(config TracingConfig) (*TracingManager, error) {
    // Create Jaeger exporter
    exporter, err := jaeger.New(
        jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(config.JaegerEndpoint)),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create Jaeger exporter: %w", err)
    }
    
    // Create resource
    res := resource.NewWithAttributes(
        semconv.SchemaURL,
        semconv.ServiceNameKey.String(config.ServiceName),
        semconv.ServiceVersionKey.String(config.ServiceVersion),
        attribute.String("environment", config.Environment),
    )
    
    // Create trace provider
    provider := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithResource(res),
        trace.WithSampler(trace.TraceIDRatioBased(config.SamplingRatio)),
    )
    
    // Set global trace provider
    otel.SetTracerProvider(provider)
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))
    
    tracer := provider.Tracer(config.ServiceName)
    
    return &TracingManager{
        tracer:   tracer,
        provider: provider,
        config:   config,
    }, nil
}

func (tm *TracingManager) StartSpan(ctx context.Context, operationName string, opts ...oteltrace.SpanStartOption) (context.Context, oteltrace.Span) {
    return tm.tracer.Start(ctx, operationName, opts...)
}

func (tm *TracingManager) SpanFromContext(ctx context.Context) oteltrace.Span {
    return oteltrace.SpanFromContext(ctx)
}

// HTTP tracing middleware
func (tm *TracingManager) HTTPMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract trace context from headers
        ctx := otel.GetTextMapPropagator().Extract(r.Context(), 
            propagation.HeaderCarrier(r.Header))
        
        // Start new span
        spanName := fmt.Sprintf("%s %s", r.Method, r.URL.Path)
        ctx, span := tm.StartSpan(ctx, spanName,
            oteltrace.WithAttributes(
                semconv.HTTPMethodKey.String(r.Method),
                semconv.HTTPURLKey.String(r.URL.String()),
                semconv.HTTPSchemeKey.String(r.URL.Scheme),
                semconv.HTTPHostKey.String(r.Host),
                semconv.HTTPTargetKey.String(r.URL.Path),
                semconv.HTTPUserAgentKey.String(r.UserAgent()),
                semconv.HTTPRequestContentLengthKey.Int64(r.ContentLength),
            ),
            oteltrace.WithSpanKind(oteltrace.SpanKindServer),
        )
        defer span.End()
        
        // Wrap response writer to capture status code
        wrappedWriter := &tracingResponseWriter{
            ResponseWriter: w,
            span:          span,
        }
        
        // Inject trace context into response headers
        otel.GetTextMapPropagator().Inject(ctx, 
            propagation.HeaderCarrier(w.Header()))
        
        // Execute handler with traced context
        next.ServeHTTP(wrappedWriter, r.WithContext(ctx))
        
        // Set final span attributes
        span.SetAttributes(
            semconv.HTTPStatusCodeKey.Int(wrappedWriter.statusCode),
            semconv.HTTPResponseContentLengthKey.Int64(wrappedWriter.bytesWritten),
        )
        
        // Set span status based on HTTP status code
        if wrappedWriter.statusCode >= 400 {
            span.SetStatus(codes.Error, fmt.Sprintf("HTTP %d", wrappedWriter.statusCode))
        } else {
            span.SetStatus(codes.Ok, "")
        }
    })
}

type tracingResponseWriter struct {
    http.ResponseWriter
    span         oteltrace.Span
    statusCode   int
    bytesWritten int64
}

func (trw *tracingResponseWriter) WriteHeader(code int) {
    trw.statusCode = code
    trw.ResponseWriter.WriteHeader(code)
}

func (trw *tracingResponseWriter) Write(b []byte) (int, error) {
    if trw.statusCode == 0 {
        trw.statusCode = 200
    }
    n, err := trw.ResponseWriter.Write(b)
    trw.bytesWritten += int64(n)
    return n, err
}

// Database tracing wrapper
type TracedDB struct {
    db     interface{} // Could be *sql.DB, *gorm.DB, etc.
    tracer oteltrace.Tracer
}

func NewTracedDB(db interface{}, tracer oteltrace.Tracer) *TracedDB {
    return &TracedDB{
        db:     db,
        tracer: tracer,
    }
}

func (tdb *TracedDB) Query(ctx context.Context, query string, args ...interface{}) error {
    ctx, span := tdb.tracer.Start(ctx, "db.query",
        oteltrace.WithAttributes(
            attribute.String("db.statement", query),
            attribute.String("db.operation", "SELECT"),
            attribute.Int("db.args.count", len(args)),
        ),
        oteltrace.WithSpanKind(oteltrace.SpanKindClient),
    )
    defer span.End()
    
    start := time.Now()
    
    // Execute actual database query (implementation depends on DB driver)
    err := tdb.executeQuery(ctx, query, args...)
    
    duration := time.Since(start)
    span.SetAttributes(
        attribute.Int64("db.duration_ms", duration.Milliseconds()),
    )
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    
    span.SetStatus(codes.Ok, "")
    return nil
}

func (tdb *TracedDB) executeQuery(ctx context.Context, query string, args ...interface{}) error {
    // Placeholder for actual database execution
    // In practice, this would call the underlying database driver
    time.Sleep(10 * time.Millisecond) // Simulate query execution
    return nil
}

// gRPC tracing interceptor
func (tm *TracingManager) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        // Extract trace context from gRPC metadata
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            ctx = otel.GetTextMapPropagator().Extract(ctx, 
                &metadataCarrier{md: md})
        }
        
        // Start new span
        ctx, span := tm.StartSpan(ctx, info.FullMethod,
            oteltrace.WithAttributes(
                attribute.String("rpc.system", "grpc"),
                attribute.String("rpc.service", extractService(info.FullMethod)),
                attribute.String("rpc.method", extractMethod(info.FullMethod)),
            ),
            oteltrace.WithSpanKind(oteltrace.SpanKindServer),
        )
        defer span.End()
        
        // Execute handler
        resp, err := handler(ctx, req)
        
        if err != nil {
            span.RecordError(err)
            span.SetStatus(codes.Error, err.Error())
        } else {
            span.SetStatus(codes.Ok, "")
        }
        
        return resp, err
    }
}

// Custom span operations
type SpanManager struct {
    tracer oteltrace.Tracer
}

func NewSpanManager(tracer oteltrace.Tracer) *SpanManager {
    return &SpanManager{tracer: tracer}
}

func (sm *SpanManager) TraceFunction(ctx context.Context, functionName string, fn func(context.Context) error) error {
    ctx, span := sm.tracer.Start(ctx, functionName,
        oteltrace.WithAttributes(
            attribute.String("function.name", functionName),
        ),
    )
    defer span.End()
    
    start := time.Now()
    err := fn(ctx)
    duration := time.Since(start)
    
    span.SetAttributes(
        attribute.Int64("function.duration_ms", duration.Milliseconds()),
    )
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    
    span.SetStatus(codes.Ok, "")
    return nil
}

func (sm *SpanManager) AddSpanEvent(ctx context.Context, name string, attributes ...attribute.KeyValue) {
    span := oteltrace.SpanFromContext(ctx)
    if span.IsRecording() {
        span.AddEvent(name, oteltrace.WithAttributes(attributes...))
    }
}

func (sm *SpanManager) SetSpanAttributes(ctx context.Context, attributes ...attribute.KeyValue) {
    span := oteltrace.SpanFromContext(ctx)
    if span.IsRecording() {
        span.SetAttributes(attributes...)
    }
}

// Cross-service trace correlation
type TraceCorrelator struct {
    httpClient *http.Client
    tracer     oteltrace.Tracer
}

func NewTraceCorrelator(tracer oteltrace.Tracer) *TraceCorrelator {
    return &TraceCorrelator{
        httpClient: &http.Client{Timeout: 30 * time.Second},
        tracer:     tracer,
    }
}

func (tc *TraceCorrelator) MakeHTTPRequest(ctx context.Context, method, url string, body io.Reader) (*http.Response, error) {
    req, err := http.NewRequestWithContext(ctx, method, url, body)
    if err != nil {
        return nil, err
    }
    
    // Inject trace context into HTTP headers
    otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))
    
    // Start span for outbound HTTP request
    ctx, span := tc.tracer.Start(ctx, fmt.Sprintf("HTTP %s", method),
        oteltrace.WithAttributes(
            semconv.HTTPMethodKey.String(method),
            semconv.HTTPURLKey.String(url),
        ),
        oteltrace.WithSpanKind(oteltrace.SpanKindClient),
    )
    defer span.End()
    
    // Execute request
    start := time.Now()
    resp, err := tc.httpClient.Do(req)
    duration := time.Since(start)
    
    span.SetAttributes(
        attribute.Int64("http.duration_ms", duration.Milliseconds()),
    )
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return nil, err
    }
    
    span.SetAttributes(semconv.HTTPStatusCodeKey.Int(resp.StatusCode))
    
    if resp.StatusCode >= 400 {
        span.SetStatus(codes.Error, fmt.Sprintf("HTTP %d", resp.StatusCode))
    } else {
        span.SetStatus(codes.Ok, "")
    }
    
    return resp, nil
}

// Utility functions and types
type metadataCarrier struct {
    md metadata.MD
}

func (mc *metadataCarrier) Get(key string) string {
    values := mc.md.Get(key)
    if len(values) > 0 {
        return values[0]
    }
    return ""
}

func (mc *metadataCarrier) Set(key, value string) {
    mc.md.Set(key, value)
}

func (mc *metadataCarrier) Keys() []string {
    keys := make([]string, 0, len(mc.md))
    for k := range mc.md {
        keys = append(keys, k)
    }
    return keys
}

func extractService(fullMethod string) string {
    parts := strings.Split(fullMethod, "/")
    if len(parts) >= 2 {
        return parts[1]
    }
    return ""
}

func extractMethod(fullMethod string) string {
    parts := strings.Split(fullMethod, "/")
    if len(parts) >= 3 {
        return parts[2]
    }
    return ""
}
```

---

## 🚨 **Alerting and Incident Management**

### **Intelligent Alerting System**

```go
package alerting

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

// Alert severity levels
type AlertSeverity int

const (
    Info AlertSeverity = iota
    Warning
    Critical
    Fatal
)

var severityNames = map[AlertSeverity]string{
    Info:     "INFO",
    Warning:  "WARNING",
    Critical: "CRITICAL",
    Fatal:    "FATAL",
}

// Alert definition
type Alert struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Severity    AlertSeverity          `json:"severity"`
    Labels      map[string]string      `json:"labels"`
    Annotations map[string]string      `json:"annotations"`
    StartsAt    time.Time              `json:"starts_at"`
    EndsAt      time.Time              `json:"ends_at,omitempty"`
    Status      AlertStatus            `json:"status"`
    Value       float64                `json:"value"`
    Threshold   float64                `json:"threshold"`
    RuleID      string                 `json:"rule_id"`
}

type AlertStatus int

const (
    Firing AlertStatus = iota
    Pending
    Resolved
    Suppressed
)

// Alert rule
type AlertRule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Query       string                 `json:"query"`
    Condition   string                 `json:"condition"`
    Threshold   float64                `json:"threshold"`
    Duration    time.Duration          `json:"duration"`
    Severity    AlertSeverity          `json:"severity"`
    Labels      map[string]string      `json:"labels"`
    Annotations map[string]string      `json:"annotations"`
    Enabled     bool                   `json:"enabled"`
}

// Alert manager
type AlertManager struct {
    rules         map[string]*AlertRule
    activeAlerts  map[string]*Alert
    resolvedAlerts map[string]*Alert
    notifiers     []AlertNotifier
    evaluator     *RuleEvaluator
    mu            sync.RWMutex
    stopChan      chan struct{}
    wg            sync.WaitGroup
}

type AlertNotifier interface {
    Notify(alert Alert) error
    Name() string
    ShouldNotify(alert Alert) bool
}

func NewAlertManager() *AlertManager {
    return &AlertManager{
        rules:          make(map[string]*AlertRule),
        activeAlerts:   make(map[string]*Alert),
        resolvedAlerts: make(map[string]*Alert),
        evaluator:      NewRuleEvaluator(),
        stopChan:       make(chan struct{}),
    }
}

func (am *AlertManager) AddRule(rule *AlertRule) {
    am.mu.Lock()
    defer am.mu.Unlock()
    am.rules[rule.ID] = rule
}

func (am *AlertManager) AddNotifier(notifier AlertNotifier) {
    am.notifiers = append(am.notifiers, notifier)
}

func (am *AlertManager) Start() {
    am.wg.Add(1)
    go am.evaluateRules()
}

func (am *AlertManager) Stop() {
    close(am.stopChan)
    am.wg.Wait()
}

func (am *AlertManager) evaluateRules() {
    defer am.wg.Done()
    
    ticker := time.NewTicker(15 * time.Second) // Evaluate every 15 seconds
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            am.processRules()
        case <-am.stopChan:
            return
        }
    }
}

func (am *AlertManager) processRules() {
    am.mu.RLock()
    rules := make([]*AlertRule, 0, len(am.rules))
    for _, rule := range am.rules {
        if rule.Enabled {
            rules = append(rules, rule)
        }
    }
    am.mu.RUnlock()
    
    for _, rule := range rules {
        go am.evaluateRule(rule)
    }
}

func (am *AlertManager) evaluateRule(rule *AlertRule) {
    // Evaluate the rule query
    value, err := am.evaluator.Evaluate(rule.Query)
    if err != nil {
        fmt.Printf("Failed to evaluate rule %s: %v\n", rule.Name, err)
        return
    }
    
    // Check if condition is met
    conditionMet := am.checkCondition(value, rule.Condition, rule.Threshold)
    
    am.mu.Lock()
    defer am.mu.Unlock()
    
    alertID := fmt.Sprintf("%s_%s", rule.ID, rule.Name)
    existingAlert, exists := am.activeAlerts[alertID]
    
    if conditionMet {
        if !exists {
            // Create new alert
            alert := &Alert{
                ID:          alertID,
                Name:        rule.Name,
                Description: rule.Annotations["description"],
                Severity:    rule.Severity,
                Labels:      rule.Labels,
                Annotations: rule.Annotations,
                StartsAt:    time.Now(),
                Status:      Pending,
                Value:       value,
                Threshold:   rule.Threshold,
                RuleID:      rule.ID,
            }
            
            am.activeAlerts[alertID] = alert
            
            // Check if alert should fire immediately or after duration
            if rule.Duration == 0 {
                alert.Status = Firing
                am.sendNotifications(*alert)
            }
        } else if existingAlert.Status == Pending {
            // Check if pending duration has elapsed
            if time.Since(existingAlert.StartsAt) >= rule.Duration {
                existingAlert.Status = Firing
                am.sendNotifications(*existingAlert)
            }
        }
        
        // Update alert value
        if exists {
            existingAlert.Value = value
        }
    } else {
        // Condition not met, resolve alert if it exists
        if exists && existingAlert.Status == Firing {
            existingAlert.Status = Resolved
            existingAlert.EndsAt = time.Now()
            
            // Move to resolved alerts
            am.resolvedAlerts[alertID] = existingAlert
            delete(am.activeAlerts, alertID)
            
            am.sendNotifications(*existingAlert)
        }
    }
}

func (am *AlertManager) checkCondition(value float64, condition string, threshold float64) bool {
    switch condition {
    case ">":
        return value > threshold
    case ">=":
        return value >= threshold
    case "<":
        return value < threshold
    case "<=":
        return value <= threshold
    case "==":
        return value == threshold
    case "!=":
        return value != threshold
    default:
        return false
    }
}

func (am *AlertManager) sendNotifications(alert Alert) {
    for _, notifier := range am.notifiers {
        if notifier.ShouldNotify(alert) {
            go func(n AlertNotifier) {
                if err := n.Notify(alert); err != nil {
                    fmt.Printf("Failed to send notification via %s: %v\n", n.Name(), err)
                }
            }(notifier)
        }
    }
}

// Rule evaluator
type RuleEvaluator struct {
    metricsClient MetricsClient
}

type MetricsClient interface {
    Query(query string) (float64, error)
}

func NewRuleEvaluator() *RuleEvaluator {
    return &RuleEvaluator{
        metricsClient: &PrometheusClient{}, // Default to Prometheus
    }
}

func (re *RuleEvaluator) Evaluate(query string) (float64, error) {
    return re.metricsClient.Query(query)
}

// Prometheus client for rule evaluation
type PrometheusClient struct {
    endpoint string
    client   *http.Client
}

func (pc *PrometheusClient) Query(query string) (float64, error) {
    // Implementation would make HTTP request to Prometheus
    // This is a placeholder that simulates metric values
    
    // Simulate different metric scenarios
    switch {
    case contains(query, "cpu_usage"):
        return 85.5, nil // High CPU usage
    case contains(query, "memory_usage"):
        return 78.2, nil // High memory usage
    case contains(query, "error_rate"):
        return 0.05, nil // 5% error rate
    case contains(query, "response_time"):
        return 1200, nil // 1.2 second response time
    default:
        return 0, nil
    }
}

// Slack notifier
type SlackNotifier struct {
    WebhookURL   string
    Channel      string
    Username     string
    MinSeverity  AlertSeverity
}

func NewSlackNotifier(webhookURL, channel, username string, minSeverity AlertSeverity) *SlackNotifier {
    return &SlackNotifier{
        WebhookURL:  webhookURL,
        Channel:     channel,
        Username:    username,
        MinSeverity: minSeverity,
    }
}

func (sn *SlackNotifier) Name() string {
    return "slack"
}

func (sn *SlackNotifier) ShouldNotify(alert Alert) bool {
    return alert.Severity >= sn.MinSeverity
}

func (sn *SlackNotifier) Notify(alert Alert) error {
    color := sn.getColorForSeverity(alert.Severity)
    emoji := sn.getEmojiForStatus(alert.Status)
    
    payload := map[string]interface{}{
        "channel":  sn.Channel,
        "username": sn.Username,
        "text":     fmt.Sprintf("%s %s Alert: %s", emoji, severityNames[alert.Severity], alert.Name),
        "attachments": []map[string]interface{}{
            {
                "color": color,
                "fields": []map[string]interface{}{
                    {"title": "Description", "value": alert.Description, "short": false},
                    {"title": "Current Value", "value": fmt.Sprintf("%.2f", alert.Value), "short": true},
                    {"title": "Threshold", "value": fmt.Sprintf("%.2f", alert.Threshold), "short": true},
                    {"title": "Started At", "value": alert.StartsAt.Format(time.RFC3339), "short": true},
                },
            },
        },
    }
    
    return sn.sendSlackMessage(payload)
}

func (sn *SlackNotifier) getColorForSeverity(severity AlertSeverity) string {
    switch severity {
    case Info:
        return "good"
    case Warning:
        return "warning"
    case Critical:
        return "danger"
    case Fatal:
        return "#FF0000"
    default:
        return "good"
    }
}

func (sn *SlackNotifier) getEmojiForStatus(status AlertStatus) string {
    switch status {
    case Firing:
        return "🔥"
    case Resolved:
        return "✅"
    case Pending:
        return "⏳"
    case Suppressed:
        return "🔇"
    default:
        return "❓"
    }
}

func (sn *SlackNotifier) sendSlackMessage(payload map[string]interface{}) error {
    // Implementation would send HTTP POST to Slack webhook
    // This is a placeholder
    fmt.Printf("Sending Slack notification: %+v\n", payload)
    return nil
}

// Email notifier
type EmailNotifier struct {
    SMTPHost     string
    SMTPPort     int
    Username     string
    Password     string
    From         string
    Recipients   []string
    MinSeverity  AlertSeverity
}

func (en *EmailNotifier) Name() string {
    return "email"
}

func (en *EmailNotifier) ShouldNotify(alert Alert) bool {
    return alert.Severity >= en.MinSeverity
}

func (en *EmailNotifier) Notify(alert Alert) error {
    subject := fmt.Sprintf("[%s] %s - %s", severityNames[alert.Severity], alert.Name, alert.Status)
    body := en.buildEmailBody(alert)
    
    return en.sendEmail(subject, body)
}

func (en *EmailNotifier) buildEmailBody(alert Alert) string {
    return fmt.Sprintf(`
Alert: %s
Severity: %s
Status: %v
Description: %s

Current Value: %.2f
Threshold: %.2f
Started At: %s

Labels: %+v
Annotations: %+v
`, alert.Name, severityNames[alert.Severity], alert.Status, alert.Description,
        alert.Value, alert.Threshold, alert.StartsAt.Format(time.RFC3339),
        alert.Labels, alert.Annotations)
}

func (en *EmailNotifier) sendEmail(subject, body string) error {
    // Implementation would use SMTP to send email
    // This is a placeholder
    fmt.Printf("Sending email: %s\n%s\n", subject, body)
    return nil
}

// Incident management
type IncidentManager struct {
    incidents    map[string]*Incident
    escalations  []EscalationRule
    oncallManager *OnCallManager
    mu           sync.RWMutex
}

type Incident struct {
    ID          string                 `json:"id"`
    Title       string                 `json:"title"`
    Description string                 `json:"description"`
    Severity    AlertSeverity          `json:"severity"`
    Status      IncidentStatus         `json:"status"`
    AssignedTo  string                 `json:"assigned_to"`
    CreatedAt   time.Time              `json:"created_at"`
    UpdatedAt   time.Time              `json:"updated_at"`
    ResolvedAt  time.Time              `json:"resolved_at,omitempty"`
    Alerts      []string               `json:"alerts"`
    Timeline    []IncidentEvent        `json:"timeline"`
    PostMortem  *PostMortem            `json:"post_mortem,omitempty"`
}

type IncidentStatus int

const (
    Open IncidentStatus = iota
    Acknowledged
    InProgress
    Resolved
    Closed
)

type IncidentEvent struct {
    Timestamp time.Time `json:"timestamp"`
    Type      string    `json:"type"`
    Message   string    `json:"message"`
    User      string    `json:"user"`
}

type EscalationRule struct {
    Severity      AlertSeverity
    DelayMinutes  int
    NotifyGroups  []string
}

type PostMortem struct {
    Summary       string    `json:"summary"`
    Timeline      []string  `json:"timeline"`
    RootCause     string    `json:"root_cause"`
    ActionItems   []string  `json:"action_items"`
    CreatedBy     string    `json:"created_by"`
    CreatedAt     time.Time `json:"created_at"`
}

func NewIncidentManager() *IncidentManager {
    return &IncidentManager{
        incidents:     make(map[string]*Incident),
        oncallManager: NewOnCallManager(),
    }
}

func (im *IncidentManager) CreateIncident(alert Alert) *Incident {
    im.mu.Lock()
    defer im.mu.Unlock()
    
    incident := &Incident{
        ID:          generateIncidentID(),
        Title:       fmt.Sprintf("%s Alert: %s", severityNames[alert.Severity], alert.Name),
        Description: alert.Description,
        Severity:    alert.Severity,
        Status:      Open,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
        Alerts:      []string{alert.ID},
        Timeline:    []IncidentEvent{
            {
                Timestamp: time.Now(),
                Type:      "created",
                Message:   "Incident created from alert",
                User:      "system",
            },
        },
    }
    
    // Auto-assign based on on-call schedule
    assignee := im.oncallManager.GetOnCallEngineer(alert.Severity)
    if assignee != "" {
        incident.AssignedTo = assignee
        incident.Timeline = append(incident.Timeline, IncidentEvent{
            Timestamp: time.Now(),
            Type:      "assigned",
            Message:   fmt.Sprintf("Auto-assigned to %s", assignee),
            User:      "system",
        })
    }
    
    im.incidents[incident.ID] = incident
    
    // Start escalation timer
    go im.startEscalation(incident)
    
    return incident
}

func (im *IncidentManager) startEscalation(incident *Incident) {
    for _, rule := range im.escalations {
        if rule.Severity == incident.Severity {
            time.Sleep(time.Duration(rule.DelayMinutes) * time.Minute)
            
            im.mu.RLock()
            currentIncident := im.incidents[incident.ID]
            im.mu.RUnlock()
            
            // Only escalate if incident is still open/acknowledged
            if currentIncident.Status == Open || currentIncident.Status == Acknowledged {
                im.escalateIncident(currentIncident, rule)
            }
            break
        }
    }
}

func (im *IncidentManager) escalateIncident(incident *Incident, rule EscalationRule) {
    // Notify escalation groups
    for _, group := range rule.NotifyGroups {
        // Implementation would notify the group
        fmt.Printf("Escalating incident %s to group %s\n", incident.ID, group)
    }
    
    // Add timeline event
    im.mu.Lock()
    incident.Timeline = append(incident.Timeline, IncidentEvent{
        Timestamp: time.Now(),
        Type:      "escalated",
        Message:   fmt.Sprintf("Escalated to groups: %v", rule.NotifyGroups),
        User:      "system",
    })
    incident.UpdatedAt = time.Now()
    im.mu.Unlock()
}

// On-call management
type OnCallManager struct {
    schedules map[AlertSeverity][]OnCallSchedule
    mu        sync.RWMutex
}

type OnCallSchedule struct {
    Engineer  string
    StartTime time.Time
    EndTime   time.Time
}

func NewOnCallManager() *OnCallManager {
    return &OnCallManager{
        schedules: make(map[AlertSeverity][]OnCallSchedule),
    }
}

func (ocm *OnCallManager) GetOnCallEngineer(severity AlertSeverity) string {
    ocm.mu.RLock()
    defer ocm.mu.RUnlock()
    
    schedules, exists := ocm.schedules[severity]
    if !exists {
        return ""
    }
    
    now := time.Now()
    for _, schedule := range schedules {
        if now.After(schedule.StartTime) && now.Before(schedule.EndTime) {
            return schedule.Engineer
        }
    }
    
    return ""
}
```

---

## 📊 **Monitoring Dashboards**

### **Dynamic Dashboard System**

```go
package dashboard

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

// Dashboard configuration
type Dashboard struct {
    ID          string      `json:"id"`
    Title       string      `json:"title"`
    Description string      `json:"description"`
    Tags        []string    `json:"tags"`
    Panels      []Panel     `json:"panels"`
    Variables   []Variable  `json:"variables"`
    TimeRange   TimeRange   `json:"time_range"`
    RefreshRate int         `json:"refresh_rate"` // seconds
    CreatedAt   time.Time   `json:"created_at"`
    UpdatedAt   time.Time   `json:"updated_at"`
}

type Panel struct {
    ID          string                 `json:"id"`
    Title       string                 `json:"title"`
    Type        PanelType              `json:"type"`
    Position    Position               `json:"position"`
    Size        Size                   `json:"size"`
    Queries     []Query                `json:"queries"`
    Options     map[string]interface{} `json:"options"`
    Thresholds  []Threshold            `json:"thresholds"`
}

type PanelType string

const (
    GraphPanel      PanelType = "graph"
    SingleStatPanel PanelType = "singlestat"
    TablePanel      PanelType = "table"
    HeatmapPanel    PanelType = "heatmap"
    LogPanel        PanelType = "logs"
    AlertListPanel  PanelType = "alertlist"
)

type Position struct {
    X int `json:"x"`
    Y int `json:"y"`
}

type Size struct {
    Width  int `json:"width"`
    Height int `json:"height"`
}

type Query struct {
    Expression string            `json:"expression"`
    Legend     string            `json:"legend"`
    RefID      string            `json:"ref_id"`
    Datasource string            `json:"datasource"`
    Variables  map[string]string `json:"variables"`
}

type Variable struct {
    Name        string   `json:"name"`
    Type        string   `json:"type"`
    Query       string   `json:"query"`
    Options     []string `json:"options"`
    DefaultValue string   `json:"default_value"`
}

type TimeRange struct {
    From string `json:"from"`
    To   string `json:"to"`
}

type Threshold struct {
    Value float64 `json:"value"`
    Color string  `json:"color"`
    Op    string  `json:"op"` // gt, lt, eq
}

// Dashboard manager
type DashboardManager struct {
    dashboards   map[string]*Dashboard
    dataSource   DataSource
    alertManager AlertManagerInterface
    mu           sync.RWMutex
}

type DataSource interface {
    Query(query string, timeRange TimeRange) ([]DataPoint, error)
    GetMetricNames() ([]string, error)
}

type AlertManagerInterface interface {
    GetActiveAlerts() ([]Alert, error)
}

type DataPoint struct {
    Timestamp time.Time `json:"timestamp"`
    Value     float64   `json:"value"`
    Labels    map[string]string `json:"labels"`
}

func NewDashboardManager(dataSource DataSource, alertManager AlertManagerInterface) *DashboardManager {
    return &DashboardManager{
        dashboards:   make(map[string]*Dashboard),
        dataSource:   dataSource,
        alertManager: alertManager,
    }
}

func (dm *DashboardManager) CreateDashboard(dashboard *Dashboard) error {
    dm.mu.Lock()
    defer dm.mu.Unlock()
    
    dashboard.ID = generateDashboardID()
    dashboard.CreatedAt = time.Now()
    dashboard.UpdatedAt = time.Now()
    
    dm.dashboards[dashboard.ID] = dashboard
    return nil
}

func (dm *DashboardManager) GetDashboard(id string) (*Dashboard, error) {
    dm.mu.RLock()
    defer dm.mu.RUnlock()
    
    dashboard, exists := dm.dashboards[id]
    if !exists {
        return nil, fmt.Errorf("dashboard not found: %s", id)
    }
    
    return dashboard, nil
}

func (dm *DashboardManager) RenderDashboard(id string, variables map[string]string) (*RenderedDashboard, error) {
    dashboard, err := dm.GetDashboard(id)
    if err != nil {
        return nil, err
    }
    
    rendered := &RenderedDashboard{
        Dashboard: *dashboard,
        Panels:    make([]RenderedPanel, len(dashboard.Panels)),
    }
    
    // Render each panel
    for i, panel := range dashboard.Panels {
        renderedPanel, err := dm.renderPanel(panel, variables, dashboard.TimeRange)
        if err != nil {
            return nil, fmt.Errorf("failed to render panel %s: %w", panel.ID, err)
        }
        rendered.Panels[i] = *renderedPanel
    }
    
    return rendered, nil
}

func (dm *DashboardManager) renderPanel(panel Panel, variables map[string]string, timeRange TimeRange) (*RenderedPanel, error) {
    rendered := &RenderedPanel{
        Panel: panel,
        Data:  make([]SeriesData, len(panel.Queries)),
    }
    
    // Execute queries
    for i, query := range panel.Queries {
        // Substitute variables in query
        processedQuery := dm.substituteVariables(query.Expression, variables)
        
        // Query data source
        dataPoints, err := dm.dataSource.Query(processedQuery, timeRange)
        if err != nil {
            return nil, fmt.Errorf("query failed: %w", err)
        }
        
        rendered.Data[i] = SeriesData{
            Name:   query.Legend,
            Points: dataPoints,
        }
    }
    
    return rendered, nil
}

func (dm *DashboardManager) substituteVariables(query string, variables map[string]string) string {
    result := query
    for varName, varValue := range variables {
        placeholder := fmt.Sprintf("$%s", varName)
        result = fmt.Sprintf("%s", result) // In practice, use proper string replacement
    }
    return result
}

type RenderedDashboard struct {
    Dashboard Dashboard       `json:"dashboard"`
    Panels    []RenderedPanel `json:"panels"`
}

type RenderedPanel struct {
    Panel Panel        `json:"panel"`
    Data  []SeriesData `json:"data"`
}

type SeriesData struct {
    Name   string      `json:"name"`
    Points []DataPoint `json:"points"`
}

// Pre-built dashboard templates
func (dm *DashboardManager) CreateServiceDashboard(serviceName string) *Dashboard {
    return &Dashboard{
        Title:       fmt.Sprintf("%s Service Dashboard", serviceName),
        Description: fmt.Sprintf("Monitoring dashboard for %s service", serviceName),
        Tags:        []string{"service", serviceName},
        Panels: []Panel{
            {
                ID:    "requests_per_second",
                Title: "Requests per Second",
                Type:  GraphPanel,
                Position: Position{X: 0, Y: 0},
                Size:     Size{Width: 12, Height: 8},
                Queries: []Query{
                    {
                        Expression: fmt.Sprintf(`rate(http_requests_total{service="%s"}[5m])`, serviceName),
                        Legend:     "RPS",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
            },
            {
                ID:    "response_time",
                Title: "Response Time (95th percentile)",
                Type:  GraphPanel,
                Position: Position{X: 12, Y: 0},
                Size:     Size{Width: 12, Height: 8},
                Queries: []Query{
                    {
                        Expression: fmt.Sprintf(`histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{service="%s"}[5m]))`, serviceName),
                        Legend:     "95th percentile",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
                Thresholds: []Threshold{
                    {Value: 0.5, Color: "green", Op: "lt"},
                    {Value: 1.0, Color: "yellow", Op: "lt"},
                    {Value: 2.0, Color: "red", Op: "gte"},
                },
            },
            {
                ID:    "error_rate",
                Title: "Error Rate",
                Type:  SingleStatPanel,
                Position: Position{X: 0, Y: 8},
                Size:     Size{Width: 6, Height: 4},
                Queries: []Query{
                    {
                        Expression: fmt.Sprintf(`rate(http_requests_total{service="%s",status=~"5.."}[5m]) / rate(http_requests_total{service="%s"}[5m]) * 100`, serviceName, serviceName),
                        Legend:     "Error Rate %",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
                Thresholds: []Threshold{
                    {Value: 1.0, Color: "green", Op: "lt"},
                    {Value: 5.0, Color: "yellow", Op: "lt"},
                    {Value: 10.0, Color: "red", Op: "gte"},
                },
            },
            {
                ID:    "active_connections",
                Title: "Active Connections",
                Type:  SingleStatPanel,
                Position: Position{X: 6, Y: 8},
                Size:     Size{Width: 6, Height: 4},
                Queries: []Query{
                    {
                        Expression: fmt.Sprintf(`active_connections{service="%s"}`, serviceName),
                        Legend:     "Connections",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
            },
        },
        TimeRange: TimeRange{
            From: "now-1h",
            To:   "now",
        },
        RefreshRate: 30,
    }
}

func (dm *DashboardManager) CreateInfrastructureDashboard() *Dashboard {
    return &Dashboard{
        Title:       "Infrastructure Overview",
        Description: "System-wide infrastructure monitoring",
        Tags:        []string{"infrastructure", "system"},
        Panels: []Panel{
            {
                ID:    "cpu_usage",
                Title: "CPU Usage by Host",
                Type:  GraphPanel,
                Position: Position{X: 0, Y: 0},
                Size:     Size{Width: 12, Height: 8},
                Queries: []Query{
                    {
                        Expression: `100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`,
                        Legend:     "{{instance}}",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
                Thresholds: []Threshold{
                    {Value: 70, Color: "yellow", Op: "gt"},
                    {Value: 90, Color: "red", Op: "gt"},
                },
            },
            {
                ID:    "memory_usage",
                Title: "Memory Usage by Host",
                Type:  GraphPanel,
                Position: Position{X: 12, Y: 0},
                Size:     Size{Width: 12, Height: 8},
                Queries: []Query{
                    {
                        Expression: `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`,
                        Legend:     "{{instance}}",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
                Thresholds: []Threshold{
                    {Value: 80, Color: "yellow", Op: "gt"},
                    {Value: 95, Color: "red", Op: "gt"},
                },
            },
            {
                ID:    "disk_usage",
                Title: "Disk Usage",
                Type:  TablePanel,
                Position: Position{X: 0, Y: 8},
                Size:     Size{Width: 24, Height: 6},
                Queries: []Query{
                    {
                        Expression: `(1 - (node_filesystem_free_bytes / node_filesystem_size_bytes)) * 100`,
                        Legend:     "Usage %",
                        RefID:      "A",
                        Datasource: "prometheus",
                    },
                },
            },
        },
        Variables: []Variable{
            {
                Name:         "host",
                Type:         "query",
                Query:        "label_values(node_cpu_seconds_total, instance)",
                DefaultValue: "all",
            },
        },
        TimeRange: TimeRange{
            From: "now-6h",
            To:   "now",
        },
        RefreshRate: 60,
    }
}

// Dashboard HTTP API
type DashboardAPI struct {
    manager *DashboardManager
}

func NewDashboardAPI(manager *DashboardManager) *DashboardAPI {
    return &DashboardAPI{manager: manager}
}

func (api *DashboardAPI) SetupRoutes(mux *http.ServeMux) {
    mux.HandleFunc("/api/dashboards", api.handleDashboards)
    mux.HandleFunc("/api/dashboards/", api.handleDashboard)
    mux.HandleFunc("/api/dashboards/render/", api.handleRenderDashboard)
}

func (api *DashboardAPI) handleDashboards(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodPost:
        api.createDashboard(w, r)
    case http.MethodGet:
        api.listDashboards(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func (api *DashboardAPI) createDashboard(w http.ResponseWriter, r *http.Request) {
    var dashboard Dashboard
    if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    if err := api.manager.CreateDashboard(&dashboard); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(dashboard)
}

func (api *DashboardAPI) handleRenderDashboard(w http.ResponseWriter, r *http.Request) {
    // Extract dashboard ID from URL
    id := extractIDFromPath(r.URL.Path, "/api/dashboards/render/")
    
    // Parse variables from query parameters
    variables := make(map[string]string)
    for key, values := range r.URL.Query() {
        if len(values) > 0 {
            variables[key] = values[0]
        }
    }
    
    rendered, err := api.manager.RenderDashboard(id, variables)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(rendered)
}
```

---

## 🔧 **SRE Practices**

### **Site Reliability Engineering Framework**

```go
package sre

import (
    "context"
    "fmt"
    "math"
    "sync"
    "time"
)

// SLI (Service Level Indicator)
type SLI struct {
    Name        string    `json:"name"`
    Description string    `json:"description"`
    Query       string    `json:"query"`
    Unit        string    `json:"unit"`
    Value       float64   `json:"value"`
    Timestamp   time.Time `json:"timestamp"`
}

// SLO (Service Level Objective)
type SLO struct {
    ID              string        `json:"id"`
    Name            string        `json:"name"`
    Description     string        `json:"description"`
    SLI             SLI           `json:"sli"`
    Target          float64       `json:"target"`          // e.g., 99.9
    TimeWindow      time.Duration `json:"time_window"`     // e.g., 30 days
    AlertingWindow  time.Duration `json:"alerting_window"` // e.g., 1 hour
    ErrorBudget     float64       `json:"error_budget"`
    BurnRate        float64       `json:"burn_rate"`
    ComplianceRate  float64       `json:"compliance_rate"`
    Status          SLOStatus     `json:"status"`
    LastEvaluated   time.Time     `json:"last_evaluated"`
}

type SLOStatus int

const (
    SLOHealthy SLOStatus = iota
    SLOWarning
    SLOCritical
    SLOBreach
)

// Error Budget Manager
type ErrorBudgetManager struct {
    slos        map[string]*SLO
    budgets     map[string]*ErrorBudget
    policies    []ErrorBudgetPolicy
    mu          sync.RWMutex
    evaluator   SLOEvaluator
}

type ErrorBudget struct {
    SLOID               string        `json:"slo_id"`
    TotalBudget         float64       `json:"total_budget"`
    RemainingBudget     float64       `json:"remaining_budget"`
    ConsumedBudget      float64       `json:"consumed_budget"`
    BudgetUtilization   float64       `json:"budget_utilization"`
    TimeWindow          time.Duration `json:"time_window"`
    WindowStart         time.Time     `json:"window_start"`
    WindowEnd           time.Time     `json:"window_end"`
    BurnRateMultiplier  float64       `json:"burn_rate_multiplier"`
    EstimatedDepletion  time.Time     `json:"estimated_depletion"`
}

type ErrorBudgetPolicy struct {
    SLOID           string
    BudgetThreshold float64 // When to trigger policy (e.g., 50% consumed)
    Actions         []PolicyAction
}

type PolicyAction struct {
    Type        string                 `json:"type"`
    Parameters  map[string]interface{} `json:"parameters"`
    Description string                 `json:"description"`
}

type SLOEvaluator interface {
    EvaluateSLI(sli SLI, timeRange TimeRange) (float64, error)
    CalculateBurnRate(slo SLO, window time.Duration) (float64, error)
}

func NewErrorBudgetManager(evaluator SLOEvaluator) *ErrorBudgetManager {
    return &ErrorBudgetManager{
        slos:      make(map[string]*SLO),
        budgets:   make(map[string]*ErrorBudget),
        evaluator: evaluator,
    }
}

func (ebm *ErrorBudgetManager) AddSLO(slo *SLO) {
    ebm.mu.Lock()
    defer ebm.mu.Unlock()
    
    ebm.slos[slo.ID] = slo
    
    // Initialize error budget
    budget := &ErrorBudget{
        SLOID:       slo.ID,
        TotalBudget: 100.0 - slo.Target, // e.g., 99.9% target = 0.1% error budget
        TimeWindow:  slo.TimeWindow,
        WindowStart: time.Now().Add(-slo.TimeWindow),
        WindowEnd:   time.Now(),
    }
    
    ebm.budgets[slo.ID] = budget
}

func (ebm *ErrorBudgetManager) EvaluateErrorBudgets() error {
    ebm.mu.RLock()
    slos := make([]*SLO, 0, len(ebm.slos))
    for _, slo := range ebm.slos {
        slos = append(slos, slo)
    }
    ebm.mu.RUnlock()
    
    for _, slo := range slos {
        if err := ebm.evaluateSLO(slo); err != nil {
            return fmt.Errorf("failed to evaluate SLO %s: %w", slo.ID, err)
        }
    }
    
    return nil
}

func (ebm *ErrorBudgetManager) evaluateSLO(slo *SLO) error {
    // Evaluate current SLI value
    currentValue, err := ebm.evaluator.EvaluateSLI(slo.SLI, TimeRange{
        From: time.Now().Add(-slo.AlertingWindow),
        To:   time.Now(),
    })
    if err != nil {
        return err
    }
    
    // Calculate burn rate
    burnRate, err := ebm.evaluator.CalculateBurnRate(*slo, slo.AlertingWindow)
    if err != nil {
        return err
    }
    
    ebm.mu.Lock()
    defer ebm.mu.Unlock()
    
    // Update SLO
    slo.SLI.Value = currentValue
    slo.BurnRate = burnRate
    slo.ComplianceRate = currentValue
    slo.LastEvaluated = time.Now()
    
    // Update error budget
    budget := ebm.budgets[slo.ID]
    budget.ConsumedBudget = math.Max(0, slo.Target-currentValue)
    budget.RemainingBudget = budget.TotalBudget - budget.ConsumedBudget
    budget.BudgetUtilization = (budget.ConsumedBudget / budget.TotalBudget) * 100
    
    // Calculate estimated depletion time
    if burnRate > 0 {
        remainingTime := budget.RemainingBudget / burnRate
        budget.EstimatedDepletion = time.Now().Add(time.Duration(remainingTime) * time.Hour)
    }
    
    // Update SLO status
    ebm.updateSLOStatus(slo, budget)
    
    // Apply error budget policies
    ebm.applyPolicies(slo, budget)
    
    return nil
}

func (ebm *ErrorBudgetManager) updateSLOStatus(slo *SLO, budget *ErrorBudget) {
    switch {
    case budget.BudgetUtilization >= 100:
        slo.Status = SLOBreach
    case budget.BudgetUtilization >= 80:
        slo.Status = SLOCritical
    case budget.BudgetUtilization >= 50:
        slo.Status = SLOWarning
    default:
        slo.Status = SLOHealthy
    }
}

func (ebm *ErrorBudgetManager) applyPolicies(slo *SLO, budget *ErrorBudget) {
    for _, policy := range ebm.policies {
        if policy.SLOID == slo.ID && budget.BudgetUtilization >= policy.BudgetThreshold {
            ebm.executePolicyActions(policy.Actions, slo, budget)
        }
    }
}

func (ebm *ErrorBudgetManager) executePolicyActions(actions []PolicyAction, slo *SLO, budget *ErrorBudget) {
    for _, action := range actions {
        switch action.Type {
        case "alert":
            ebm.sendAlert(slo, budget, action)
        case "scale":
            ebm.triggerScaling(slo, action)
        case "circuit_breaker":
            ebm.activateCircuitBreaker(slo, action)
        case "rate_limit":
            ebm.adjustRateLimit(slo, action)
        }
    }
}

// Availability Calculator
type AvailabilityCalculator struct {
    metricsClient MetricsClient
}

func NewAvailabilityCalculator(client MetricsClient) *AvailabilityCalculator {
    return &AvailabilityCalculator{metricsClient: client}
}

func (ac *AvailabilityCalculator) CalculateAvailability(serviceName string, timeWindow time.Duration) (*AvailabilityReport, error) {
    endTime := time.Now()
    startTime := endTime.Add(-timeWindow)
    
    // Calculate total requests
    totalRequests, err := ac.metricsClient.Query(
        fmt.Sprintf(`sum(increase(http_requests_total{service="%s"}[%s]))`, 
            serviceName, formatDuration(timeWindow)))
    if err != nil {
        return nil, err
    }
    
    // Calculate successful requests
    successfulRequests, err := ac.metricsClient.Query(
        fmt.Sprintf(`sum(increase(http_requests_total{service="%s",status!~"5.."}[%s]))`, 
            serviceName, formatDuration(timeWindow)))
    if err != nil {
        return nil, err
    }
    
    // Calculate error requests
    errorRequests := totalRequests - successfulRequests
    
    // Calculate availability percentage
    availability := 0.0
    if totalRequests > 0 {
        availability = (successfulRequests / totalRequests) * 100
    }
    
    // Calculate downtime
    downtime := calculateDowntime(availability, timeWindow)
    
    return &AvailabilityReport{
        ServiceName:        serviceName,
        TimeWindow:         timeWindow,
        StartTime:          startTime,
        EndTime:            endTime,
        TotalRequests:      int64(totalRequests),
        SuccessfulRequests: int64(successfulRequests),
        ErrorRequests:      int64(errorRequests),
        AvailabilityPercent: availability,
        Downtime:          downtime,
        MTBF:              ac.calculateMTBF(serviceName, timeWindow),
        MTTR:              ac.calculateMTTR(serviceName, timeWindow),
    }, nil
}

type AvailabilityReport struct {
    ServiceName         string        `json:"service_name"`
    TimeWindow          time.Duration `json:"time_window"`
    StartTime           time.Time     `json:"start_time"`
    EndTime             time.Time     `json:"end_time"`
    TotalRequests       int64         `json:"total_requests"`
    SuccessfulRequests  int64         `json:"successful_requests"`
    ErrorRequests       int64         `json:"error_requests"`
    AvailabilityPercent float64       `json:"availability_percent"`
    Downtime            time.Duration `json:"downtime"`
    MTBF                time.Duration `json:"mtbf"` // Mean Time Between Failures
    MTTR                time.Duration `json:"mttr"` // Mean Time To Recovery
}

func (ac *AvailabilityCalculator) calculateMTBF(serviceName string, timeWindow time.Duration) time.Duration {
    // Implementation would analyze incident history
    // This is a simplified calculation
    return 168 * time.Hour // 1 week average
}

func (ac *AvailabilityCalculator) calculateMTTR(serviceName string, timeWindow time.Duration) time.Duration {
    // Implementation would analyze incident resolution times
    // This is a simplified calculation
    return 30 * time.Minute // 30 minutes average
}

// Chaos Engineering
type ChaosEngineer struct {
    experiments []ChaosExperiment
    scheduler   *ExperimentScheduler
    safety      *SafetyValidator
    results     map[string]*ExperimentResult
    mu          sync.RWMutex
}

type ChaosExperiment struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Type        ExperimentType         `json:"type"`
    Target      ExperimentTarget       `json:"target"`
    Parameters  map[string]interface{} `json:"parameters"`
    Duration    time.Duration          `json:"duration"`
    Schedule    string                 `json:"schedule"`
    SafetyRules []SafetyRule           `json:"safety_rules"`
    Enabled     bool                   `json:"enabled"`
}

type ExperimentType string

const (
    NetworkLatency ExperimentType = "network_latency"
    NetworkLoss    ExperimentType = "network_loss"
    CPUStress      ExperimentType = "cpu_stress"
    MemoryStress   ExperimentType = "memory_stress"
    PodKill        ExperimentType = "pod_kill"
    DiskFill       ExperimentType = "disk_fill"
)

type ExperimentTarget struct {
    Service   string            `json:"service"`
    Namespace string            `json:"namespace"`
    Labels    map[string]string `json:"labels"`
}

type SafetyRule struct {
    Type      string  `json:"type"`
    Threshold float64 `json:"threshold"`
    Query     string  `json:"query"`
}

type ExperimentResult struct {
    ExperimentID   string                 `json:"experiment_id"`
    StartTime      time.Time              `json:"start_time"`
    EndTime        time.Time              `json:"end_time"`
    Status         ExperimentStatus       `json:"status"`
    Metrics        map[string]float64     `json:"metrics"`
    Observations   []string               `json:"observations"`
    Impact         ImpactAssessment       `json:"impact"`
    Recommendations []string              `json:"recommendations"`
}

type ExperimentStatus string

const (
    ExperimentRunning   ExperimentStatus = "running"
    ExperimentCompleted ExperimentStatus = "completed"
    ExperimentFailed    ExperimentStatus = "failed"
    ExperimentAborted   ExperimentStatus = "aborted"
)

type ImpactAssessment struct {
    AvailabilityImpact float64 `json:"availability_impact"`
    LatencyImpact      float64 `json:"latency_impact"`
    ErrorRateImpact    float64 `json:"error_rate_impact"`
    UserImpact         string  `json:"user_impact"`
}

func NewChaosEngineer() *ChaosEngineer {
    return &ChaosEngineer{
        experiments: make([]ChaosExperiment, 0),
        results:     make(map[string]*ExperimentResult),
        safety:      NewSafetyValidator(),
    }
}

func (ce *ChaosEngineer) RunExperiment(experimentID string) error {
    ce.mu.RLock()
    var experiment *ChaosExperiment
    for _, exp := range ce.experiments {
        if exp.ID == experimentID {
            experiment = &exp
            break
        }
    }
    ce.mu.RUnlock()
    
    if experiment == nil {
        return fmt.Errorf("experiment not found: %s", experimentID)
    }
    
    // Validate safety rules
    if !ce.safety.ValidateExperiment(*experiment) {
        return fmt.Errorf("experiment failed safety validation")
    }
    
    // Create result tracking
    result := &ExperimentResult{
        ExperimentID: experimentID,
        StartTime:    time.Now(),
        Status:       ExperimentRunning,
        Metrics:      make(map[string]float64),
    }
    
    ce.mu.Lock()
    ce.results[experimentID] = result
    ce.mu.Unlock()
    
    // Execute experiment
    go ce.executeExperiment(*experiment, result)
    
    return nil
}

func (ce *ChaosEngineer) executeExperiment(experiment ChaosExperiment, result *ExperimentResult) {
    defer func() {
        result.EndTime = time.Now()
        result.Status = ExperimentCompleted
    }()
    
    // Collect baseline metrics
    baseline := ce.collectMetrics(experiment.Target)
    
    // Apply chaos
    if err := ce.applyChaos(experiment); err != nil {
        result.Status = ExperimentFailed
        return
    }
    
    // Monitor during experiment
    go ce.monitorExperiment(experiment, result)
    
    // Wait for experiment duration
    time.Sleep(experiment.Duration)
    
    // Stop chaos
    ce.stopChaos(experiment)
    
    // Collect post-experiment metrics
    postMetrics := ce.collectMetrics(experiment.Target)
    
    // Analyze impact
    result.Impact = ce.analyzeImpact(baseline, postMetrics)
    
    // Generate recommendations
    result.Recommendations = ce.generateRecommendations(experiment, result)
}

func (ce *ChaosEngineer) collectMetrics(target ExperimentTarget) map[string]float64 {
    // Implementation would collect actual metrics
    return map[string]float64{
        "availability": 99.5,
        "latency_p95":  200.0,
        "error_rate":   0.1,
    }
}

func (ce *ChaosEngineer) applyChaos(experiment ChaosExperiment) error {
    // Implementation would apply actual chaos based on experiment type
    fmt.Printf("Applying chaos experiment: %s\n", experiment.Name)
    return nil
}

func (ce *ChaosEngineer) analyzeImpact(baseline, current map[string]float64) ImpactAssessment {
    return ImpactAssessment{
        AvailabilityImpact: baseline["availability"] - current["availability"],
        LatencyImpact:      current["latency_p95"] - baseline["latency_p95"],
        ErrorRateImpact:    current["error_rate"] - baseline["error_rate"],
        UserImpact:         "Low - no customer-facing impact observed",
    }
}

// Safety validator
type SafetyValidator struct {
    rules []GlobalSafetyRule
}

type GlobalSafetyRule struct {
    Name        string
    Query       string
    Threshold   float64
    Operator    string
    Description string
}

func NewSafetyValidator() *SafetyValidator {
    return &SafetyValidator{
        rules: []GlobalSafetyRule{
            {
                Name:        "system_availability",
                Query:       "avg(up)",
                Threshold:   0.95,
                Operator:    ">=",
                Description: "System availability must be >= 95%",
            },
            {
                Name:        "error_rate",
                Query:       "rate(http_requests_total{status=~'5..'}[5m])",
                Threshold:   0.1,
                Operator:    "<=",
                Description: "Error rate must be <= 10%",
            },
        },
    }
}

func (sv *SafetyValidator) ValidateExperiment(experiment ChaosExperiment) bool {
    // Check global safety rules
    for _, rule := range sv.rules {
        if !sv.evaluateRule(rule) {
            fmt.Printf("Safety rule failed: %s\n", rule.Description)
            return false
        }
    }
    
    // Check experiment-specific safety rules
    for _, rule := range experiment.SafetyRules {
        if !sv.evaluateExperimentRule(rule) {
            return false
        }
    }
    
    return true
}

func (sv *SafetyValidator) evaluateRule(rule GlobalSafetyRule) bool {
    // Implementation would query metrics and evaluate rule
    // This is a placeholder
    return true
}

func (sv *SafetyValidator) evaluateExperimentRule(rule SafetyRule) bool {
    // Implementation would query metrics and evaluate rule
    // This is a placeholder
    return true
}
```

---

## 🐛 **Troubleshooting Methodologies**

### **Systematic Debugging Framework**

```go
package troubleshooting

import (
    "context"
    "fmt"
    "sort"
    "strings"
    "time"
)

// Incident investigation framework
type IncidentInvestigator struct {
    logAnalyzer     *LogAnalyzer
    metricsAnalyzer *MetricsAnalyzer
    traceAnalyzer   *TraceAnalyzer
    knowledgeBase   *TroubleshootingKB
    runbooks        map[string]*Runbook
}

type Investigation struct {
    ID           string                 `json:"id"`
    IncidentID   string                 `json:"incident_id"`
    StartTime    time.Time              `json:"start_time"`
    EndTime      time.Time              `json:"end_time"`
    Status       InvestigationStatus    `json:"status"`
    Findings     []Finding              `json:"findings"`
    Hypotheses   []Hypothesis           `json:"hypotheses"`
    Timeline     []InvestigationEvent   `json:"timeline"`
    RootCause    *RootCause             `json:"root_cause,omitempty"`
    ActionItems  []ActionItem           `json:"action_items"`
}

type InvestigationStatus int

const (
    InvestigationActive InvestigationStatus = iota
    InvestigationSuspended
    InvestigationCompleted
    InvestigationCancelled
)

type Finding struct {
    ID          string                 `json:"id"`
    Type        FindingType            `json:"type"`
    Description string                 `json:"description"`
    Evidence    []Evidence             `json:"evidence"`
    Confidence  float64                `json:"confidence"`
    Timestamp   time.Time              `json:"timestamp"`
    Source      string                 `json:"source"`
    Tags        []string               `json:"tags"`
}

type FindingType string

const (
    AnomalyFinding     FindingType = "anomaly"
    CorrelationFinding FindingType = "correlation"
    PatternFinding     FindingType = "pattern"
    ThresholdFinding   FindingType = "threshold"
    TimelineFinding    FindingType = "timeline"
)

type Evidence struct {
    Type        EvidenceType           `json:"type"`
    Description string                 `json:"description"`
    Data        map[string]interface{} `json:"data"`
    Source      string                 `json:"source"`
    Timestamp   time.Time              `json:"timestamp"`
}

type EvidenceType string

const (
    LogEvidence     EvidenceType = "log"
    MetricEvidence  EvidenceType = "metric"
    TraceEvidence   EvidenceType = "trace"
    EventEvidence   EvidenceType = "event"
    ConfigEvidence  EvidenceType = "config"
)

type Hypothesis struct {
    ID          string    `json:"id"`
    Description string    `json:"description"`
    Probability float64   `json:"probability"`
    Status      HypothesisStatus `json:"status"`
    Evidence    []string  `json:"evidence_ids"`
    TestPlan    []string  `json:"test_plan"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

type HypothesisStatus int

const (
    HypothesisUnTested HypothesisStatus = iota
    HypothesisTesting
    HypothesisConfirmed
    HypothesisRefuted
)

func NewIncidentInvestigator() *IncidentInvestigator {
    return &IncidentInvestigator{
        logAnalyzer:     NewLogAnalyzer(),
        metricsAnalyzer: NewMetricsAnalyzer(),
        traceAnalyzer:   NewTraceAnalyzer(),
        knowledgeBase:   NewTroubleshootingKB(),
        runbooks:        make(map[string]*Runbook),
    }
}

func (ii *IncidentInvestigator) StartInvestigation(incidentID string, timeRange TimeRange) (*Investigation, error) {
    investigation := &Investigation{
        ID:         generateInvestigationID(),
        IncidentID: incidentID,
        StartTime:  time.Now(),
        Status:     InvestigationActive,
        Findings:   []Finding{},
        Hypotheses: []Hypothesis{},
        Timeline:   []InvestigationEvent{},
    }
    
    // Gather initial evidence
    ctx := context.Background()
    
    // Analyze logs
    logFindings, err := ii.logAnalyzer.AnalyzeLogs(ctx, timeRange)
    if err != nil {
        return nil, fmt.Errorf("log analysis failed: %w", err)
    }
    investigation.Findings = append(investigation.Findings, logFindings...)
    
    // Analyze metrics
    metricFindings, err := ii.metricsAnalyzer.AnalyzeMetrics(ctx, timeRange)
    if err != nil {
        return nil, fmt.Errorf("metrics analysis failed: %w", err)
    }
    investigation.Findings = append(investigation.Findings, metricFindings...)
    
    // Analyze traces
    traceFindings, err := ii.traceAnalyzer.AnalyzeTraces(ctx, timeRange)
    if err != nil {
        return nil, fmt.Errorf("trace analysis failed: %w", err)
    }
    investigation.Findings = append(investigation.Findings, traceFindings...)
    
    // Generate initial hypotheses
    investigation.Hypotheses = ii.generateHypotheses(investigation.Findings)
    
    // Check knowledge base for similar incidents
    similarIncidents := ii.knowledgeBase.FindSimilarIncidents(investigation.Findings)
    
    // Add timeline event
    investigation.Timeline = append(investigation.Timeline, InvestigationEvent{
        Timestamp:   time.Now(),
        Type:        "investigation_started",
        Description: "Investigation started with automatic evidence gathering",
        Data: map[string]interface{}{
            "findings_count":         len(investigation.Findings),
            "hypotheses_count":       len(investigation.Hypotheses),
            "similar_incidents_count": len(similarIncidents),
        },
    })
    
    return investigation, nil
}

// Log analyzer for troubleshooting
type LogAnalyzer struct {
    patterns       []LogPattern
    anomalyDetector *LogAnomalyDetector
}

type LogPattern struct {
    Name        string   `json:"name"`
    Pattern     string   `json:"pattern"`
    Severity    string   `json:"severity"`
    Description string   `json:"description"`
    Tags        []string `json:"tags"`
}

func NewLogAnalyzer() *LogAnalyzer {
    return &LogAnalyzer{
        patterns: []LogPattern{
            {
                Name:        "out_of_memory",
                Pattern:     "OutOfMemoryError|java.lang.OutOfMemoryError|killed by OOM killer",
                Severity:    "critical",
                Description: "Application running out of memory",
                Tags:        []string{"memory", "resource", "crash"},
            },
            {
                Name:        "connection_timeout",
                Pattern:     "connection timeout|connection refused|connection reset",
                Severity:    "high",
                Description: "Network connectivity issues",
                Tags:        []string{"network", "timeout", "connectivity"},
            },
            {
                Name:        "database_error",
                Pattern:     "database connection failed|SQL error|deadlock detected",
                Severity:    "high",
                Description: "Database connectivity or query issues",
                Tags:        []string{"database", "sql", "connectivity"},
            },
        },
        anomalyDetector: NewLogAnomalyDetector(),
    }
}

func (la *LogAnalyzer) AnalyzeLogs(ctx context.Context, timeRange TimeRange) ([]Finding, error) {
    var findings []Finding
    
    // Pattern matching analysis
    patternFindings, err := la.analyzePatterns(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, patternFindings...)
    
    // Anomaly detection
    anomalyFindings, err := la.anomalyDetector.DetectAnomalies(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, anomalyFindings...)
    
    // Volume analysis
    volumeFindings, err := la.analyzeLogVolume(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, volumeFindings...)
    
    return findings, nil
}

func (la *LogAnalyzer) analyzePatterns(ctx context.Context, timeRange TimeRange) ([]Finding, error) {
    var findings []Finding
    
    for _, pattern := range la.patterns {
        // In practice, this would query log storage (e.g., Elasticsearch)
        matches := la.searchLogPattern(pattern.Pattern, timeRange)
        
        if len(matches) > 0 {
            finding := Finding{
                ID:          generateFindingID(),
                Type:        PatternFinding,
                Description: fmt.Sprintf("Found %d occurrences of pattern: %s", len(matches), pattern.Description),
                Confidence:  0.9,
                Timestamp:   time.Now(),
                Source:      "log_analyzer",
                Tags:        pattern.Tags,
                Evidence: []Evidence{
                    {
                        Type:        LogEvidence,
                        Description: fmt.Sprintf("Log pattern matches for: %s", pattern.Name),
                        Data: map[string]interface{}{
                            "pattern":     pattern.Pattern,
                            "match_count": len(matches),
                            "matches":     matches[:minInt(len(matches), 10)], // Limit to 10 examples
                        },
                        Source:    "log_storage",
                        Timestamp: time.Now(),
                    },
                },
            }
            findings = append(findings, finding)
        }
    }
    
    return findings, nil
}

func (la *LogAnalyzer) searchLogPattern(pattern string, timeRange TimeRange) []LogEntry {
    // This is a placeholder - in practice would query actual log storage
    return []LogEntry{
        {
            Timestamp: time.Now().Add(-10 * time.Minute),
            Level:     "ERROR",
            Message:   "OutOfMemoryError: Java heap space",
            Service:   "payment-service",
        },
    }
}

// Metrics analyzer for troubleshooting
type MetricsAnalyzer struct {
    thresholds      []MetricThreshold
    correlator      *MetricCorrelator
    anomalyDetector *MetricAnomalyDetector
}

type MetricThreshold struct {
    MetricName  string  `json:"metric_name"`
    Operator    string  `json:"operator"`
    Value       float64 `json:"value"`
    Severity    string  `json:"severity"`
    Description string  `json:"description"`
}

func NewMetricsAnalyzer() *MetricsAnalyzer {
    return &MetricsAnalyzer{
        thresholds: []MetricThreshold{
            {
                MetricName:  "cpu_usage_percent",
                Operator:    ">",
                Value:       90.0,
                Severity:    "high",
                Description: "High CPU usage detected",
            },
            {
                MetricName:  "memory_usage_percent",
                Operator:    ">",
                Value:       85.0,
                Severity:    "high",
                Description: "High memory usage detected",
            },
            {
                MetricName:  "error_rate_percent",
                Operator:    ">",
                Value:       5.0,
                Severity:    "critical",
                Description: "High error rate detected",
            },
        },
        correlator:      NewMetricCorrelator(),
        anomalyDetector: NewMetricAnomalyDetector(),
    }
}

func (ma *MetricsAnalyzer) AnalyzeMetrics(ctx context.Context, timeRange TimeRange) ([]Finding, error) {
    var findings []Finding
    
    // Threshold analysis
    thresholdFindings, err := ma.analyzeThresholds(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, thresholdFindings...)
    
    // Correlation analysis
    correlationFindings, err := ma.correlator.FindCorrelations(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, correlationFindings...)
    
    // Anomaly detection
    anomalyFindings, err := ma.anomalyDetector.DetectAnomalies(ctx, timeRange)
    if err != nil {
        return nil, err
    }
    findings = append(findings, anomalyFindings...)
    
    return findings, nil
}

func (ma *MetricsAnalyzer) analyzeThresholds(ctx context.Context, timeRange TimeRange) ([]Finding, error) {
    var findings []Finding
    
    for _, threshold := range ma.thresholds {
        // Query metric values (placeholder implementation)
        values := ma.queryMetric(threshold.MetricName, timeRange)
        
        violatedValues := []float64{}
        for _, value := range values {
            if ma.evaluateThreshold(value, threshold.Operator, threshold.Value) {
                violatedValues = append(violatedValues, value)
            }
        }
        
        if len(violatedValues) > 0 {
            finding := Finding{
                ID:          generateFindingID(),
                Type:        ThresholdFinding,
                Description: fmt.Sprintf("Threshold violation: %s", threshold.Description),
                Confidence:  0.95,
                Timestamp:   time.Now(),
                Source:      "metrics_analyzer",
                Tags:        []string{"threshold", "metrics", threshold.Severity},
                Evidence: []Evidence{
                    {
                        Type:        MetricEvidence,
                        Description: fmt.Sprintf("Threshold violations for %s", threshold.MetricName),
                        Data: map[string]interface{}{
                            "metric_name": threshold.MetricName,
                            "threshold":   threshold.Value,
                            "operator":    threshold.Operator,
                            "violations":  len(violatedValues),
                            "max_value":   maxFloat64(violatedValues),
                            "avg_value":   avgFloat64(violatedValues),
                        },
                        Source:    "metrics_storage",
                        Timestamp: time.Now(),
                    },
                },
            }
            findings = append(findings, finding)
        }
    }
    
    return findings, nil
}

func (ma *MetricsAnalyzer) queryMetric(metricName string, timeRange TimeRange) []float64 {
    // Placeholder implementation - would query actual metrics storage
    switch metricName {
    case "cpu_usage_percent":
        return []float64{92.5, 94.2, 91.8, 95.1}
    case "memory_usage_percent":
        return []float64{87.3, 88.9, 86.1, 89.2}
    case "error_rate_percent":
        return []float64{7.2, 8.1, 6.9, 9.3}
    default:
        return []float64{}
    }
}

func (ma *MetricsAnalyzer) evaluateThreshold(value float64, operator string, threshold float64) bool {
    switch operator {
    case ">":
        return value > threshold
    case ">=":
        return value >= threshold
    case "<":
        return value < threshold
    case "<=":
        return value <= threshold
    case "==":
        return value == threshold
    default:
        return false
    }
}

// Metric correlator for finding relationships
type MetricCorrelator struct {
    correlationRules []CorrelationRule
}

type CorrelationRule struct {
    Name        string   `json:"name"`
    Metrics     []string `json:"metrics"`
    Threshold   float64  `json:"threshold"`
    Description string   `json:"description"`
}

func NewMetricCorrelator() *MetricCorrelator {
    return &MetricCorrelator{
        correlationRules: []CorrelationRule{
            {
                Name:        "cpu_memory_correlation",
                Metrics:     []string{"cpu_usage_percent", "memory_usage_percent"},
                Threshold:   0.7,
                Description: "High correlation between CPU and memory usage",
            },
            {
                Name:        "error_latency_correlation",
                Metrics:     []string{"error_rate_percent", "response_time_ms"},
                Threshold:   0.8,
                Description: "High correlation between error rate and response time",
            },
        },
    }
}

func (mc *MetricCorrelator) FindCorrelations(ctx context.Context, timeRange TimeRange) ([]Finding, error) {
    var findings []Finding
    
    for _, rule := range mc.correlationRules {
        correlation := mc.calculateCorrelation(rule.Metrics, timeRange)
        
        if correlation >= rule.Threshold {
            finding := Finding{
                ID:          generateFindingID(),
                Type:        CorrelationFinding,
                Description: fmt.Sprintf("Strong correlation detected: %s (r=%.3f)", rule.Description, correlation),
                Confidence:  correlation,
                Timestamp:   time.Now(),
                Source:      "metric_correlator",
                Tags:        []string{"correlation", "metrics"},
                Evidence: []Evidence{
                    {
                        Type:        MetricEvidence,
                        Description: fmt.Sprintf("Correlation analysis for %s", rule.Name),
                        Data: map[string]interface{}{
                            "metrics":          rule.Metrics,
                            "correlation":      correlation,
                            "threshold":        rule.Threshold,
                            "analysis_method":  "pearson",
                        },
                        Source:    "metric_correlator",
                        Timestamp: time.Now(),
                    },
                },
            }
            findings = append(findings, finding)
        }
    }
    
    return findings, nil
}

func (mc *MetricCorrelator) calculateCorrelation(metrics []string, timeRange TimeRange) float64 {
    // Placeholder implementation - would calculate actual correlation
    // This simulates finding strong correlations
    return 0.85
}

// Troubleshooting knowledge base
type TroubleshootingKB struct {
    symptoms        map[string][]Symptom
    solutions       map[string][]Solution
    incidentHistory []HistoricalIncident
}

type Symptom struct {
    ID          string   `json:"id"`
    Description string   `json:"description"`
    Indicators  []string `json:"indicators"`
    Severity    string   `json:"severity"`
    Tags        []string `json:"tags"`
}

type Solution struct {
    ID          string   `json:"id"`
    Title       string   `json:"title"`
    Description string   `json:"description"`
    Steps       []string `json:"steps"`
    Confidence  float64  `json:"confidence"`
    Tags        []string `json:"tags"`
}

type HistoricalIncident struct {
    ID          string    `json:"id"`
    Title       string    `json:"title"`
    Symptoms    []string  `json:"symptoms"`
    RootCause   string    `json:"root_cause"`
    Resolution  string    `json:"resolution"`
    OccurredAt  time.Time `json:"occurred_at"`
    ResolvedAt  time.Time `json:"resolved_at"`
    Tags        []string  `json:"tags"`
}

func NewTroubleshootingKB() *TroubleshootingKB {
    kb := &TroubleshootingKB{
        symptoms:  make(map[string][]Symptom),
        solutions: make(map[string][]Solution),
    }
    
    // Load predefined symptoms and solutions
    kb.loadKnowledgeBase()
    
    return kb
}

func (kb *TroubleshootingKB) FindSimilarIncidents(findings []Finding) []HistoricalIncident {
    var similarIncidents []HistoricalIncident
    
    for _, incident := range kb.incidentHistory {
        similarity := kb.calculateSimilarity(findings, incident)
        if similarity > 0.7 {
            similarIncidents = append(similarIncidents, incident)
        }
    }
    
    // Sort by similarity (most similar first)
    sort.Slice(similarIncidents, func(i, j int) bool {
        return kb.calculateSimilarity(findings, similarIncidents[i]) > 
               kb.calculateSimilarity(findings, similarIncidents[j])
    })
    
    return similarIncidents
}

func (kb *TroubleshootingKB) calculateSimilarity(findings []Finding, incident HistoricalIncident) float64 {
    // Simplified similarity calculation based on tag overlap
    findingTags := make(map[string]bool)
    for _, finding := range findings {
        for _, tag := range finding.Tags {
            findingTags[tag] = true
        }
    }
    
    commonTags := 0
    for _, tag := range incident.Tags {
        if findingTags[tag] {
            commonTags++
        }
    }
    
    if len(incident.Tags) == 0 {
        return 0
    }
    
    return float64(commonTags) / float64(len(incident.Tags))
}

func (kb *TroubleshootingKB) loadKnowledgeBase() {
    // Load symptoms
    kb.symptoms["memory"] = []Symptom{
        {
            ID:          "high_memory_usage",
            Description: "Application consuming excessive memory",
            Indicators:  []string{"memory_usage_percent > 85", "out_of_memory_errors", "gc_pressure"},
            Severity:    "high",
            Tags:        []string{"memory", "resource", "performance"},
        },
    }
    
    // Load solutions
    kb.solutions["memory"] = []Solution{
        {
            ID:          "memory_optimization",
            Title:       "Memory Usage Optimization",
            Description: "Steps to reduce memory consumption and optimize garbage collection",
            Steps: []string{
                "Analyze heap dump to identify memory leaks",
                "Review object allocation patterns",
                "Optimize data structures and caching",
                "Tune garbage collection parameters",
                "Consider scaling up or out if needed",
            },
            Confidence: 0.9,
            Tags:       []string{"memory", "optimization", "tuning"},
        },
    }
    
    // Load historical incidents
    kb.incidentHistory = []HistoricalIncident{
        {
            ID:         "incident_001",
            Title:      "Payment Service Memory Leak",
            Symptoms:   []string{"high_memory_usage", "out_of_memory_errors"},
            RootCause:  "Memory leak in connection pool management",
            Resolution: "Fixed connection pool configuration and implemented proper cleanup",
            OccurredAt: time.Now().Add(-30 * 24 * time.Hour),
            ResolvedAt: time.Now().Add(-30*24*time.Hour + 2*time.Hour),
            Tags:       []string{"memory", "connection_pool", "payment_service"},
        },
    }
}

// Runbook system
type Runbook struct {
    ID          string      `json:"id"`
    Title       string      `json:"title"`
    Description string      `json:"description"`
    Triggers    []Trigger   `json:"triggers"`
    Steps       []Step      `json:"steps"`
    Tags        []string    `json:"tags"`
    Owner       string      `json:"owner"`
    LastUpdated time.Time   `json:"last_updated"`
}

type Trigger struct {
    Type        string                 `json:"type"`
    Condition   string                 `json:"condition"`
    Parameters  map[string]interface{} `json:"parameters"`
}

type Step struct {
    ID          string                 `json:"id"`
    Title       string                 `json:"title"`
    Description string                 `json:"description"`
    Type        StepType               `json:"type"`
    Command     string                 `json:"command,omitempty"`
    Parameters  map[string]interface{} `json:"parameters"`
    Expected    string                 `json:"expected"`
    OnFailure   string                 `json:"on_failure"`
}

type StepType string

const (
    ManualStep     StepType = "manual"
    CommandStep    StepType = "command"
    APICallStep    StepType = "api_call"
    CheckStep      StepType = "check"
    DecisionStep   StepType = "decision"
)
```

---

## ❓ **Interview Questions**

### **Advanced Observability Engineering Questions**

#### **1. Distributed Tracing Implementation**

**Q: Design a distributed tracing system that can handle 100,000+ requests per second with minimal performance impact.**

**A: High-performance distributed tracing architecture:**

```go
// High-performance tracing system design
type HighPerformanceTracer struct {
    samplingManager *AdaptiveSamplingManager
    batchProcessor  *TraceBatchProcessor
    exporters       []TraceExporter
    contextPool     sync.Pool
    spanPool        sync.Pool
    bufferPool      sync.Pool
}

// Adaptive sampling for high-throughput systems
type AdaptiveSamplingManager struct {
    strategies     map[string]SamplingStrategy
    rateLimiter    *TokenBucketRateLimiter
    loadBalancer   *SamplingLoadBalancer
    dynamicConfig  *DynamicSamplingConfig
}

type SamplingStrategy interface {
    ShouldSample(ctx context.Context, traceID string, spanName string, 
                 attributes map[string]interface{}) SamplingDecision
    UpdateConfig(config SamplingConfig)
}

// Probabilistic sampling with rate limiting
type ProbabilisticSampling struct {
    rate        float64
    maxPerSecond int
    rateLimiter *TokenBucketRateLimiter
}

func (ps *ProbabilisticSampling) ShouldSample(ctx context.Context, traceID string, 
    spanName string, attributes map[string]interface{}) SamplingDecision {
    
    // Check rate limit first
    if !ps.rateLimiter.Allow() {
        return SamplingDecision{Sample: false, Reason: "rate_limited"}
    }
    
    // Deterministic sampling based on trace ID
    hash := hashTraceID(traceID)
    threshold := uint64(ps.rate * float64(^uint64(0)))
    
    if hash < threshold {
        return SamplingDecision{
            Sample: true, 
            Rate: ps.rate,
            Attributes: map[string]interface{}{
                "sampling.probability": ps.rate,
                "sampling.method": "probabilistic",
            },
        }
    }
    
    return SamplingDecision{Sample: false, Reason: "probability"}
}

// Tail-based sampling for critical transactions
type TailBasedSampling struct {
    errorSampler     *ErrorBasedSampler
    latencySampler   *LatencyBasedSampler
    criticalSampler  *CriticalPathSampler
    bufferManager    *TraceBufferManager
}

func (tbs *TailBasedSampling) ShouldSample(trace CompleteTrace) SamplingDecision {
    // Always sample traces with errors
    if tbs.errorSampler.HasErrors(trace) {
        return SamplingDecision{Sample: true, Reason: "error_present"}
    }
    
    // Sample slow transactions
    if tbs.latencySampler.IsSlowTransaction(trace) {
        return SamplingDecision{Sample: true, Reason: "high_latency"}
    }
    
    // Sample critical business transactions
    if tbs.criticalSampler.IsCriticalPath(trace) {
        return SamplingDecision{Sample: true, Reason: "critical_path"}
    }
    
    return SamplingDecision{Sample: false, Reason: "tail_based_filter"}
}

// Batch processing for efficient export
type TraceBatchProcessor struct {
    batchSize     int
    flushInterval time.Duration
    buffer        []Span
    mu            sync.Mutex
    flushChan     chan struct{}
}

func (tbp *TraceBatchProcessor) ProcessSpan(span Span) {
    tbp.mu.Lock()
    defer tbp.mu.Unlock()
    
    tbp.buffer = append(tbp.buffer, span)
    
    if len(tbp.buffer) >= tbp.batchSize {
        go tbp.flush()
    }
}

func (tbp *TraceBatchProcessor) flush() {
    tbp.mu.Lock()
    batch := make([]Span, len(tbp.buffer))
    copy(batch, tbp.buffer)
    tbp.buffer = tbp.buffer[:0]
    tbp.mu.Unlock()
    
    // Export batch to all configured exporters
    for _, exporter := range tbp.exporters {
        go exporter.Export(batch)
    }
}
```

#### **2. SLO and Error Budget Management**

**Q: Design an error budget management system that automatically adjusts deployment policies based on budget consumption.**

**A: Intelligent error budget management:**

```go
// Intelligent error budget system
type IntelligentErrorBudgetManager struct {
    budgetCalculator *ErrorBudgetCalculator
    policyEngine     *DeploymentPolicyEngine
    riskAssessor     *RiskAssessment
    alertManager     *BudgetAlertManager
    predictor        *BudgetPredictor
}

type DeploymentPolicyEngine struct {
    policies        []DeploymentPolicy
    riskThresholds  RiskThresholds
    approvalChain   ApprovalChain
    rollbackManager *AutoRollbackManager
}

type DeploymentPolicy struct {
    ID              string                 `json:"id"`
    Name            string                 `json:"name"`
    Conditions      []PolicyCondition      `json:"conditions"`
    Actions         []PolicyAction         `json:"actions"`
    RiskLevel       RiskLevel              `json:"risk_level"`
    ApprovalRequired bool                  `json:"approval_required"`
    Constraints     DeploymentConstraints  `json:"constraints"`
}

type PolicyCondition struct {
    Type      string  `json:"type"`
    Metric    string  `json:"metric"`
    Operator  string  `json:"operator"`
    Threshold float64 `json:"threshold"`
    Window    time.Duration `json:"window"`
}

// Risk-based deployment decisions
func (dpе *DeploymentPolicyEngine) EvaluateDeployment(
    deployment DeploymentRequest, 
    errorBudget ErrorBudget) DeploymentDecision {
    
    risk := dpe.riskAssessor.AssessRisk(deployment, errorBudget)
    
    decision := DeploymentDecision{
        RequestID:   deployment.ID,
        Approved:    false,
        RiskLevel:   risk.Level,
        Reasoning:   []string{},
        Constraints: DeploymentConstraints{},
    }
    
    // Apply policies based on error budget status
    for _, policy := range dpe.policies {
        if dpe.evaluatePolicyConditions(policy.Conditions, errorBudget) {
            dpe.applyPolicy(policy, &decision, risk)
        }
    }
    
    return decision
}

func (dpe *DeploymentPolicyEngine) applyPolicy(
    policy DeploymentPolicy, 
    decision *DeploymentDecision, 
    risk RiskAssessment) {
    
    switch policy.Name {
    case "conservative_deployment":
        if risk.BudgetUtilization > 75 {
            decision.Constraints.MaxCanaryPercent = 5
            decision.Constraints.CanaryDuration = 60 * time.Minute
            decision.ApprovalRequired = true
            decision.Reasoning = append(decision.Reasoning, 
                "High error budget utilization requires conservative rollout")
        }
        
    case "freeze_deployment":
        if risk.BudgetUtilization > 90 {
            decision.Approved = false
            decision.Reasoning = append(decision.Reasoning,
                "Deployment frozen due to error budget depletion")
        }
        
    case "auto_rollback":
        decision.Constraints.EnableAutoRollback = true
        decision.Constraints.RollbackThreshold = 5.0 // 5% error rate
        decision.Constraints.RollbackWindow = 15 * time.Minute
    }
}

// Predictive error budget analysis
type BudgetPredictor struct {
    historicalData []BudgetDataPoint
    models         map[string]PredictionModel
    seasonality    SeasonalityAnalyzer
}

func (bp *BudgetPredictor) PredictBudgetDepletion(
    currentBudget ErrorBudget, 
    forecastPeriod time.Duration) BudgetPrediction {
    
    // Analyze current burn rate trends
    burnRateTrend := bp.analyzeBurnRateTrend(currentBudget)
    
    // Consider seasonal patterns
    seasonalAdjustment := bp.seasonality.GetSeasonalFactor(time.Now(), forecastPeriod)
    
    // Predict future consumption
    predictedConsumption := burnRateTrend.Rate * seasonalAdjustment * 
                           forecastPeriod.Hours()
    
    return BudgetPrediction{
        CurrentUtilization:    currentBudget.BudgetUtilization,
        PredictedUtilization: currentBudget.BudgetUtilization + predictedConsumption,
        DepletionRisk:        bp.calculateDepletionRisk(predictedConsumption),
        RecommendedActions:   bp.generateRecommendations(predictedConsumption),
        Confidence:          burnRateTrend.Confidence,
    }
}
```

#### **3. Chaos Engineering Implementation**

**Q: Design a chaos engineering platform that can safely inject failures across microservices with automatic safety controls.**

**A: Comprehensive chaos engineering platform:**

```go
// Advanced chaos engineering platform
type ChaosEngineeringPlatform struct {
    experimentManager   *ExperimentManager
    safetyController   *SafetyController
    blastRadiusManager *BlastRadiusManager
    observabilityHook  *ObservabilityHook
    scheduler          *ExperimentScheduler
}

type SafetyController struct {
    steadyStateChecker  *SteadyStateChecker
    circuitBreaker      *ChaosCircuitBreaker
    emergencyStop       *EmergencyStopController
    safetyMetrics      []SafetyMetric
    guardrails         []SafetyGuardrail
}

type SteadyStateChecker struct {
    metrics         []SteadyStateMetric
    healthCheckers  []HealthChecker
    baselineWindow  time.Duration
    toleranceLevel  float64
}

type SteadyStateMetric struct {
    Name            string    `json:"name"`
    Query           string    `json:"query"`
    ExpectedValue   float64   `json:"expected_value"`
    ToleranceBand   float64   `json:"tolerance_band"`
    CriticalLimit   float64   `json:"critical_limit"`
}

func (ssc *SteadyStateChecker) VerifySteadyState(
    experiment ChaosExperiment) (bool, []SteadyStateViolation) {
    
    var violations []SteadyStateViolation
    
    // Check each steady state metric
    for _, metric := range ssc.metrics {
        currentValue, err := ssc.queryMetric(metric.Query)
        if err != nil {
            violations = append(violations, SteadyStateViolation{
                Metric: metric.Name,
                Type:   "query_error",
                Reason: err.Error(),
            })
            continue
        }
        
        // Check if within tolerance
        deviation := math.Abs(currentValue - metric.ExpectedValue)
        toleranceThreshold := metric.ExpectedValue * metric.ToleranceBand
        
        if deviation > toleranceThreshold {
            severity := "warning"
            if deviation > metric.CriticalLimit {
                severity = "critical"
            }
            
            violations = append(violations, SteadyStateViolation{
                Metric:        metric.Name,
                Type:          "threshold_violation",
                CurrentValue:  currentValue,
                ExpectedValue: metric.ExpectedValue,
                Deviation:     deviation,
                Severity:      severity,
            })
        }
    }
    
    // System is in steady state if no critical violations
    steadyState := true
    for _, violation := range violations {
        if violation.Severity == "critical" {
            steadyState = false
            break
        }
    }
    
    return steadyState, violations
}

// Blast radius management
type BlastRadiusManager struct {
    scopeCalculator    *ScopeCalculator
    impactAssessor     *ImpactAssessor
    dependencyGraph    *ServiceDependencyGraph
    isolationManager   *IsolationManager
}

func (brm *BlastRadiusManager) CalculateBlastRadius(
    experiment ChaosExperiment) BlastRadiusAnalysis {
    
    // Identify directly affected services
    directlyAffected := brm.scopeCalculator.GetDirectlyAffectedServices(experiment)
    
    // Calculate downstream impact through dependency graph
    downstreamImpact := brm.dependencyGraph.CalculateDownstreamImpact(directlyAffected)
    
    // Assess user impact
    userImpact := brm.impactAssessor.AssessUserImpact(directlyAffected, downstreamImpact)
    
    // Calculate business impact
    businessImpact := brm.impactAssessor.AssessBusinessImpact(userImpact)
    
    return BlastRadiusAnalysis{
        DirectlyAffected:   directlyAffected,
        DownstreamImpact:   downstreamImpact,
        UserImpact:         userImpact,
        BusinessImpact:     businessImpact,
        RiskLevel:          brm.calculateRiskLevel(businessImpact),
        Recommendations:    brm.generateRecommendations(businessImpact),
    }
}

// Automated experiment orchestration
type ExperimentOrchestrator struct {
    preflightChecker   *PreflightChecker
    executionEngine    *ExecutionEngine
    monitoringSystem   *ExperimentMonitoring
    rollbackManager    *AutoRollbackManager
}

func (eo *ExperimentOrchestrator) ExecuteExperiment(
    experiment ChaosExperiment) (*ExperimentExecution, error) {
    
    execution := &ExperimentExecution{
        ExperimentID: experiment.ID,
        StartTime:    time.Now(),
        Status:       ExecutionRunning,
        Phases:       []ExecutionPhase{},
    }
    
    // Phase 1: Preflight checks
    preflightResult := eo.preflightChecker.RunPreflightChecks(experiment)
    execution.Phases = append(execution.Phases, ExecutionPhase{
        Name:      "preflight",
        Status:    PhaseCompleted,
        Result:    preflightResult,
        Duration:  time.Since(execution.StartTime),
    })
    
    if !preflightResult.Passed {
        execution.Status = ExecutionFailed
        return execution, fmt.Errorf("preflight checks failed: %v", 
                                    preflightResult.Failures)
    }
    
    // Phase 2: Steady state verification
    steadyStateStart := time.Now()
    steadyState, violations := eo.verifySteadyState(experiment)
    execution.Phases = append(execution.Phases, ExecutionPhase{
        Name:      "steady_state_verification", 
        Status:    PhaseCompleted,
        Result:    map[string]interface{}{
            "steady_state": steadyState,
            "violations":   violations,
        },
        Duration:  time.Since(steadyStateStart),
    })
    
    if !steadyState {
        execution.Status = ExecutionAborted
        return execution, fmt.Errorf("system not in steady state")
    }
    
    // Phase 3: Chaos injection
    chaosStart := time.Now()
    chaosResult := eo.executionEngine.InjectChaos(experiment)
    execution.Phases = append(execution.Phases, ExecutionPhase{
        Name:      "chaos_injection",
        Status:    PhaseCompleted,
        Result:    chaosResult,
        Duration:  time.Since(chaosStart),
    })
    
    // Phase 4: Monitoring and observation
    monitoringStart := time.Now()
    go eo.monitorExperiment(execution, experiment)
    
    // Wait for experiment duration
    time.Sleep(experiment.Duration)
    
    // Phase 5: Chaos removal
    removalStart := time.Now()
    removalResult := eo.executionEngine.RemoveChaos(experiment)
    execution.Phases = append(execution.Phases, ExecutionPhase{
        Name:      "chaos_removal",
        Status:    PhaseCompleted,
        Result:    removalResult,
        Duration:  time.Since(removalStart),
    })
    
    // Phase 6: Recovery verification
    recoveryStart := time.Now()
    recoveryVerified := eo.verifyRecovery(experiment)
    execution.Phases = append(execution.Phases, ExecutionPhase{
        Name:      "recovery_verification",
        Status:    PhaseCompleted,
        Result:    map[string]interface{}{"recovered": recoveryVerified},
        Duration:  time.Since(recoveryStart),
    })
    
    execution.Status = ExecutionCompleted
    execution.EndTime = time.Now()
    
    return execution, nil
}
```

This comprehensive Observability and Monitoring Guide provides advanced implementations for structured logging, metrics collection, distributed tracing, alerting systems, monitoring dashboards, SRE practices, chaos engineering, and troubleshooting methodologies. The guide includes production-ready Go code that demonstrates the observability expertise expected from senior backend engineers in technical interviews at companies like Razorpay, FAANG, and other high-scale environments.

The complete guide covers all aspects of modern observability engineering, from basic monitoring to advanced chaos engineering and error budget management, providing the depth and breadth needed for senior engineering roles.