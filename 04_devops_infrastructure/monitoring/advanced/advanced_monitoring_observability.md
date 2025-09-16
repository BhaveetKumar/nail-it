# 📊 Advanced Monitoring & Observability

## Table of Contents
1. [Observability Pillars](#observability-pillars/)
2. [Distributed Tracing](#distributed-tracing/)
3. [Metrics Collection](#metrics-collection/)
4. [Log Aggregation](#log-aggregation/)
5. [Alerting Systems](#alerting-systems/)
6. [Performance Monitoring](#performance-monitoring/)
7. [SLA/SLO Management](#slaslo-management/)
8. [Go Implementation Examples](#go-implementation-examples/)
9. [Interview Questions](#interview-questions/)

## Observability Pillars

### Three Pillars of Observability

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
)

// Metrics - Quantitative data about system behavior
type MetricsCollector struct {
    counters   map[string]int64
    gauges     map[string]float64
    histograms map[string][]float64
    mutex      sync.RWMutex
}

func NewMetricsCollector() *MetricsCollector {
    return &MetricsCollector{
        counters:   make(map[string]int64),
        gauges:     make(map[string]float64),
        histograms: make(map[string][]float64),
    }
}

func (mc *MetricsCollector) IncrementCounter(name string, labels map[string]string) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.counters[name]++
}

func (mc *MetricsCollector) SetGauge(name string, value float64, labels map[string]string) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.gauges[name] = value
}

func (mc *MetricsCollector) RecordHistogram(name string, value float64, labels map[string]string) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.histograms[name] = append(mc.histograms[name], value)
}

// Logs - Discrete events with timestamps
type LogEntry struct {
    Timestamp time.Time
    Level     LogLevel
    Message   string
    Fields    map[string]interface{}
    TraceID   string
    SpanID    string
}

type LogLevel int

const (
    DEBUG LogLevel = iota
    INFO
    WARN
    ERROR
    FATAL
)

type Logger struct {
    level  LogLevel
    fields map[string]interface{}
}

func NewLogger(level LogLevel) *Logger {
    return &Logger{
        level:  level,
        fields: make(map[string]interface{}),
    }
}

func (l *Logger) WithField(key string, value interface{}) *Logger {
    newLogger := &Logger{
        level:  l.level,
        fields: make(map[string]interface{}),
    }
    
    for k, v := range l.fields {
        newLogger.fields[k] = v
    }
    newLogger.fields[key] = value
    
    return newLogger
}

func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
    newLogger := &Logger{
        level:  l.level,
        fields: make(map[string]interface{}),
    }
    
    for k, v := range l.fields {
        newLogger.fields[k] = v
    }
    for k, v := range fields {
        newLogger.fields[k] = v
    }
    
    return newLogger
}

func (l *Logger) log(level LogLevel, msg string) {
    if level < l.level {
        return
    }
    
    entry := LogEntry{
        Timestamp: time.Now(),
        Level:     level,
        Message:   msg,
        Fields:    l.fields,
    }
    
    // In production, this would send to a log aggregation system
    log.Printf("[%s] %s %+v", entry.Timestamp.Format(time.RFC3339), msg, entry.Fields)
}

func (l *Logger) Debug(msg string) {
    l.log(DEBUG, msg)
}

func (l *Logger) Info(msg string) {
    l.log(INFO, msg)
}

func (l *Logger) Warn(msg string) {
    l.log(WARN, msg)
}

func (l *Logger) Error(msg string) {
    l.log(ERROR, msg)
}

func (l *Logger) Fatal(msg string) {
    l.log(FATAL, msg)
}

// Traces - Request flow through distributed systems
type Trace struct {
    TraceID   string
    Spans     []*Span
    StartTime time.Time
    EndTime   time.Time
}

type Span struct {
    SpanID     string
    TraceID    string
    ParentID   string
    Name       string
    StartTime  time.Time
    EndTime    time.Time
    Tags       map[string]string
    Logs       []LogEntry
    Children   []*Span
}

type Tracer struct {
    spans map[string]*Span
    mutex sync.RWMutex
}

func NewTracer() *Tracer {
    return &Tracer{
        spans: make(map[string]*Span),
    }
}

func (t *Tracer) StartSpan(name string, parentID string) *Span {
    spanID := generateID()
    traceID := generateID()
    
    if parentID != "" {
        if parent, exists := t.spans[parentID]; exists {
            traceID = parent.TraceID
        }
    }
    
    span := &Span{
        SpanID:    spanID,
        TraceID:   traceID,
        ParentID:  parentID,
        Name:      name,
        StartTime: time.Now(),
        Tags:      make(map[string]string),
        Logs:      make([]LogEntry, 0),
        Children:  make([]*Span, 0),
    }
    
    t.mutex.Lock()
    t.spans[spanID] = span
    t.mutex.Unlock()
    
    return span
}

func (t *Tracer) FinishSpan(spanID string) {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if span, exists := t.spans[spanID]; exists {
        span.EndTime = time.Now()
        
        // Add to parent's children
        if span.ParentID != "" {
            if parent, exists := t.spans[span.ParentID]; exists {
                parent.Children = append(parent.Children, span)
            }
        }
    }
}

func (t *Tracer) AddTag(spanID string, key, value string) {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if span, exists := t.spans[spanID]; exists {
        span.Tags[key] = value
    }
}

func (t *Tracer) AddLog(spanID string, level LogLevel, message string) {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if span, exists := t.spans[spanID]; exists {
        logEntry := LogEntry{
            Timestamp: time.Now(),
            Level:     level,
            Message:   message,
            Fields:    make(map[string]interface{}),
            TraceID:   span.TraceID,
            SpanID:    span.SpanID,
        }
        span.Logs = append(span.Logs, logEntry)
    }
}

func generateID() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}
```

## Distributed Tracing

### OpenTelemetry Implementation

```go
package main

import (
    "context"
    "fmt"
    "time"
)

type OpenTelemetryTracer struct {
    traces map[string]*Trace
    mutex  sync.RWMutex
}

func NewOpenTelemetryTracer() *OpenTelemetryTracer {
    return &OpenTelemetryTracer{
        traces: make(map[string]*Trace),
    }
}

func (ot *OpenTelemetryTracer) StartTrace(ctx context.Context, name string) (context.Context, *Span) {
    traceID := generateID()
    spanID := generateID()
    
    span := &Span{
        SpanID:    spanID,
        TraceID:   traceID,
        Name:      name,
        StartTime: time.Now(),
        Tags:      make(map[string]string),
        Logs:      make([]LogEntry, 0),
        Children:  make([]*Span, 0),
    }
    
    ot.mutex.Lock()
    if trace, exists := ot.traces[traceID]; exists {
        trace.Spans = append(trace.Spans, span)
    } else {
        ot.traces[traceID] = &Trace{
            TraceID:   traceID,
            Spans:     []*Span{span},
            StartTime: time.Now(),
        }
    }
    ot.mutex.Unlock()
    
    // Add trace context to the context
    ctx = context.WithValue(ctx, "traceID", traceID)
    ctx = context.WithValue(ctx, "spanID", spanID)
    
    return ctx, span
}

func (ot *OpenTelemetryTracer) StartSpan(ctx context.Context, name string) (context.Context, *Span) {
    traceID, _ := ctx.Value("traceID").(string)
    parentSpanID, _ := ctx.Value("spanID").(string)
    
    if traceID == "" {
        return ot.StartTrace(ctx, name)
    }
    
    spanID := generateID()
    span := &Span{
        SpanID:    spanID,
        TraceID:   traceID,
        ParentID:  parentSpanID,
        Name:      name,
        StartTime: time.Now(),
        Tags:      make(map[string]string),
        Logs:      make([]LogEntry, 0),
        Children:  make([]*Span, 0),
    }
    
    ot.mutex.Lock()
    if trace, exists := ot.traces[traceID]; exists {
        trace.Spans = append(trace.Spans, span)
    }
    ot.mutex.Unlock()
    
    // Update context with new span ID
    ctx = context.WithValue(ctx, "spanID", spanID)
    
    return ctx, span
}

func (ot *OpenTelemetryTracer) FinishSpan(ctx context.Context, span *Span) {
    span.EndTime = time.Now()
    
    // Add to parent's children
    if span.ParentID != "" {
        ot.mutex.Lock()
        if trace, exists := ot.traces[span.TraceID]; exists {
            for _, s := range trace.Spans {
                if s.SpanID == span.ParentID {
                    s.Children = append(s.Children, span)
                    break
                }
            }
        }
        ot.mutex.Unlock()
    }
}

// Trace Context Propagation
func (ot *OpenTelemetryTracer) InjectTraceContext(ctx context.Context, headers map[string]string) {
    if traceID, ok := ctx.Value("traceID").(string); ok {
        headers["X-Trace-ID"] = traceID
    }
    if spanID, ok := ctx.Value("spanID").(string); ok {
        headers["X-Span-ID"] = spanID
    }
}

func (ot *OpenTelemetryTracer) ExtractTraceContext(ctx context.Context, headers map[string]string) context.Context {
    if traceID, ok := headers["X-Trace-ID"]; ok {
        ctx = context.WithValue(ctx, "traceID", traceID)
    }
    if spanID, ok := headers["X-Span-ID"]; ok {
        ctx = context.WithValue(ctx, "spanID", spanID)
    }
    return ctx
}

// Trace Sampling
type SamplingStrategy interface {
    ShouldSample(traceID string) bool
}

type ProbabilisticSampler struct {
    probability float64
}

func NewProbabilisticSampler(probability float64) *ProbabilisticSampler {
    return &ProbabilisticSampler{probability: probability}
}

func (ps *ProbabilisticSampler) ShouldSample(traceID string) bool {
    // Simple hash-based sampling
    hash := hashString(traceID)
    return float64(hash%100) < ps.probability*100
}

type RateLimitingSampler struct {
    maxTracesPerSecond int
    currentCount       int
    lastReset          time.Time
    mutex              sync.Mutex
}

func NewRateLimitingSampler(maxTracesPerSecond int) *RateLimitingSampler {
    return &RateLimitingSampler{
        maxTracesPerSecond: maxTracesPerSecond,
        lastReset:          time.Now(),
    }
}

func (rls *RateLimitingSampler) ShouldSample(traceID string) bool {
    rls.mutex.Lock()
    defer rls.mutex.Unlock()
    
    now := time.Now()
    if now.Sub(rls.lastReset) >= time.Second {
        rls.currentCount = 0
        rls.lastReset = now
    }
    
    if rls.currentCount >= rls.maxTracesPerSecond {
        return false
    }
    
    rls.currentCount++
    return true
}

func hashString(s string) uint32 {
    hash := uint32(0)
    for _, c := range s {
        hash = hash*31 + uint32(c)
    }
    return hash
}
```

## Metrics Collection

### Prometheus-style Metrics

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type PrometheusMetrics struct {
    counters   map[string]*Counter
    gauges     map[string]*Gauge
    histograms map[string]*Histogram
    summaries  map[string]*Summary
    mutex      sync.RWMutex
}

type Counter struct {
    Name   string
    Value  float64
    Labels map[string]string
    mutex  sync.RWMutex
}

type Gauge struct {
    Name   string
    Value  float64
    Labels map[string]string
    mutex  sync.RWMutex
}

type Histogram struct {
    Name     string
    Buckets  []float64
    Counts   []uint64
    Sum      float64
    Labels   map[string]string
    mutex    sync.RWMutex
}

type Summary struct {
    Name      string
    Count     uint64
    Sum       float64
    Quantiles map[float64]float64
    Labels    map[string]string
    mutex     sync.RWMutex
}

func NewPrometheusMetrics() *PrometheusMetrics {
    return &PrometheusMetrics{
        counters:   make(map[string]*Counter),
        gauges:     make(map[string]*Gauge),
        histograms: make(map[string]*Histogram),
        summaries:  make(map[string]*Summary),
    }
}

func (pm *PrometheusMetrics) NewCounter(name string, labels map[string]string) *Counter {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    counter := &Counter{
        Name:   name,
        Value:  0,
        Labels: labels,
    }
    
    pm.counters[name] = counter
    return counter
}

func (pm *PrometheusMetrics) NewGauge(name string, labels map[string]string) *Gauge {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    gauge := &Gauge{
        Name:   name,
        Value:  0,
        Labels: labels,
    }
    
    pm.gauges[name] = gauge
    return gauge
}

func (pm *PrometheusMetrics) NewHistogram(name string, buckets []float64, labels map[string]string) *Histogram {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    histogram := &Histogram{
        Name:    name,
        Buckets: buckets,
        Counts:  make([]uint64, len(buckets)+1),
        Sum:     0,
        Labels:  labels,
    }
    
    pm.histograms[name] = histogram
    return histogram
}

func (pm *PrometheusMetrics) NewSummary(name string, quantiles []float64, labels map[string]string) *Summary {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    summary := &Summary{
        Name:      name,
        Count:     0,
        Sum:       0,
        Quantiles: make(map[float64]float64),
        Labels:    labels,
    }
    
    for _, q := range quantiles {
        summary.Quantiles[q] = 0
    }
    
    pm.summaries[name] = summary
    return summary
}

func (c *Counter) Inc() {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    c.Value++
}

func (c *Counter) Add(delta float64) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    c.Value += delta
}

func (g *Gauge) Set(value float64) {
    g.mutex.Lock()
    defer g.mutex.Unlock()
    g.Value = value
}

func (g *Gauge) Inc() {
    g.mutex.Lock()
    defer g.mutex.Unlock()
    g.Value++
}

func (g *Gauge) Dec() {
    g.mutex.Lock()
    defer g.mutex.Unlock()
    g.Value--
}

func (h *Histogram) Observe(value float64) {
    h.mutex.Lock()
    defer h.mutex.Unlock()
    
    h.Sum += value
    h.Counts[len(h.Counts)-1]++ // +Inf bucket
    
    for i, bucket := range h.Buckets {
        if value <= bucket {
            h.Counts[i]++
            break
        }
    }
}

func (s *Summary) Observe(value float64) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    s.Count++
    s.Sum += value
    
    // Update quantiles (simplified implementation)
    for q := range s.Quantiles {
        s.Quantiles[q] = value // In practice, use a more sophisticated algorithm
    }
}

// Metrics Exporter
func (pm *PrometheusMetrics) Export() string {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    var output string
    
    // Export counters
    for _, counter := range pm.counters {
        counter.mutex.RLock()
        output += fmt.Sprintf("# TYPE %s counter\n", counter.Name)
        output += fmt.Sprintf("%s %f\n", counter.Name, counter.Value)
        counter.mutex.RUnlock()
    }
    
    // Export gauges
    for _, gauge := range pm.gauges {
        gauge.mutex.RLock()
        output += fmt.Sprintf("# TYPE %s gauge\n", gauge.Name)
        output += fmt.Sprintf("%s %f\n", gauge.Name, gauge.Value)
        gauge.mutex.RUnlock()
    }
    
    // Export histograms
    for _, histogram := range pm.histograms {
        histogram.mutex.RLock()
        output += fmt.Sprintf("# TYPE %s histogram\n", histogram.Name)
        for i, bucket := range histogram.Buckets {
            output += fmt.Sprintf("%s_bucket{le=\"%f\"} %d\n", histogram.Name, bucket, histogram.Counts[i])
        }
        output += fmt.Sprintf("%s_bucket{le=\"+Inf\"} %d\n", histogram.Name, histogram.Counts[len(histogram.Counts)-1])
        output += fmt.Sprintf("%s_sum %f\n", histogram.Name, histogram.Sum)
        histogram.mutex.RUnlock()
    }
    
    return output
}
```

## Log Aggregation

### Centralized Logging System

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

type LogAggregator struct {
    logs    []LogEntry
    mutex   sync.RWMutex
    maxSize int
}

func NewLogAggregator(maxSize int) *LogAggregator {
    return &LogAggregator{
        logs:    make([]LogEntry, 0),
        maxSize: maxSize,
    }
}

func (la *LogAggregator) AddLog(entry LogEntry) {
    la.mutex.Lock()
    defer la.mutex.Unlock()
    
    la.logs = append(la.logs, entry)
    
    // Maintain max size
    if len(la.logs) > la.maxSize {
        la.logs = la.logs[1:]
    }
}

func (la *LogAggregator) SearchLogs(query string, level LogLevel, startTime, endTime time.Time) []LogEntry {
    la.mutex.RLock()
    defer la.mutex.RUnlock()
    
    var results []LogEntry
    
    for _, entry := range la.logs {
        // Filter by time range
        if !entry.Timestamp.After(startTime) || !entry.Timestamp.Before(endTime) {
            continue
        }
        
        // Filter by level
        if level != DEBUG && entry.Level < level {
            continue
        }
        
        // Filter by query
        if query != "" && !contains(entry.Message, query) {
            continue
        }
        
        results = append(results, entry)
    }
    
    return results
}

func (la *LogAggregator) GetLogsByTraceID(traceID string) []LogEntry {
    la.mutex.RLock()
    defer la.mutex.RUnlock()
    
    var results []LogEntry
    
    for _, entry := range la.logs {
        if entry.TraceID == traceID {
            results = append(results, entry)
        }
    }
    
    return results
}

func (la *LogAggregator) ExportLogs() ([]byte, error) {
    la.mutex.RLock()
    defer la.mutex.RUnlock()
    
    return json.Marshal(la.logs)
}

func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}

// Log Shipping
type LogShipper struct {
    aggregator *LogAggregator
    endpoint   string
    interval   time.Duration
    stopCh     chan bool
}

func NewLogShipper(aggregator *LogAggregator, endpoint string, interval time.Duration) *LogShipper {
    return &LogShipper{
        aggregator: aggregator,
        endpoint:   endpoint,
        interval:   interval,
        stopCh:     make(chan bool),
    }
}

func (ls *LogShipper) Start() {
    go ls.shipLogs()
}

func (ls *LogShipper) Stop() {
    close(ls.stopCh)
}

func (ls *LogShipper) shipLogs() {
    ticker := time.NewTicker(ls.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            ls.sendLogs()
        case <-ls.stopCh:
            return
        }
    }
}

func (ls *LogShipper) sendLogs() {
    logs, err := ls.aggregator.ExportLogs()
    if err != nil {
        fmt.Printf("Error exporting logs: %v\n", err)
        return
    }
    
    // In production, this would send to a log aggregation service
    fmt.Printf("Shipping %d bytes of logs to %s\n", len(logs), ls.endpoint)
}
```

## Alerting Systems

### Alert Manager

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Alert struct {
    ID          string
    Name        string
    Description string
    Severity    AlertSeverity
    Status      AlertStatus
    Labels      map[string]string
    Annotations map[string]string
    StartsAt    time.Time
    EndsAt      time.Time
    mutex       sync.RWMutex
}

type AlertSeverity int

const (
    INFO AlertSeverity = iota
    WARNING
    CRITICAL
)

type AlertStatus int

const (
    FIRING AlertStatus = iota
    RESOLVED
    SUPPRESSED
)

type AlertRule struct {
    Name        string
    Expression  string
    Duration    time.Duration
    Severity    AlertSeverity
    Labels      map[string]string
    Annotations map[string]string
}

type AlertManager struct {
    alerts    map[string]*Alert
    rules     []AlertRule
    notifiers []Notifier
    mutex     sync.RWMutex
}

type Notifier interface {
    SendAlert(alert *Alert) error
}

type EmailNotifier struct {
    SMTPHost string
    SMTPPort int
    Username string
    Password string
    From     string
    To       []string
}

func (en *EmailNotifier) SendAlert(alert *Alert) error {
    // Simulate email sending
    fmt.Printf("Sending email alert: %s - %s\n", alert.Name, alert.Description)
    return nil
}

type SlackNotifier struct {
    WebhookURL string
    Channel    string
}

func (sn *SlackNotifier) SendAlert(alert *Alert) error {
    // Simulate Slack notification
    fmt.Printf("Sending Slack alert: %s - %s\n", alert.Name, alert.Description)
    return nil
}

type PagerDutyNotifier struct {
    IntegrationKey string
}

func (pn *PagerDutyNotifier) SendAlert(alert *Alert) error {
    // Simulate PagerDuty notification
    fmt.Printf("Sending PagerDuty alert: %s - %s\n", alert.Name, alert.Description)
    return nil
}

func NewAlertManager() *AlertManager {
    return &AlertManager{
        alerts:    make(map[string]*Alert),
        rules:     make([]AlertRule, 0),
        notifiers: make([]Notifier, 0),
    }
}

func (am *AlertManager) AddRule(rule AlertRule) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    am.rules = append(am.rules, rule)
}

func (am *AlertManager) AddNotifier(notifier Notifier) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    am.notifiers = append(am.notifiers, notifier)
}

func (am *AlertManager) EvaluateRules(metrics map[string]float64) {
    am.mutex.RLock()
    rules := make([]AlertRule, len(am.rules))
    copy(rules, am.rules)
    am.mutex.RUnlock()
    
    for _, rule := range rules {
        if am.evaluateExpression(rule.Expression, metrics) {
            am.fireAlert(rule)
        } else {
            am.resolveAlert(rule.Name)
        }
    }
}

func (am *AlertManager) evaluateExpression(expression string, metrics map[string]float64) bool {
    // Simplified expression evaluation
    // In production, use a proper expression evaluator
    switch expression {
    case "cpu_usage > 80":
        return metrics["cpu_usage"] > 80
    case "memory_usage > 90":
        return metrics["memory_usage"] > 90
    case "error_rate > 5":
        return metrics["error_rate"] > 5
    default:
        return false
    }
}

func (am *AlertManager) fireAlert(rule AlertRule) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    
    alertID := fmt.Sprintf("%s_%d", rule.Name, time.Now().Unix())
    
    alert := &Alert{
        ID:          alertID,
        Name:        rule.Name,
        Description: rule.Annotations["description"],
        Severity:    rule.Severity,
        Status:      FIRING,
        Labels:      rule.Labels,
        Annotations: rule.Annotations,
        StartsAt:    time.Now(),
    }
    
    am.alerts[alertID] = alert
    
    // Send notifications
    for _, notifier := range am.notifiers {
        go notifier.SendAlert(alert)
    }
}

func (am *AlertManager) resolveAlert(ruleName string) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    
    for _, alert := range am.alerts {
        if alert.Name == ruleName && alert.Status == FIRING {
            alert.Status = RESOLVED
            alert.EndsAt = time.Now()
        }
    }
}

func (am *AlertManager) GetActiveAlerts() []*Alert {
    am.mutex.RLock()
    defer am.mutex.RUnlock()
    
    var activeAlerts []*Alert
    for _, alert := range am.alerts {
        if alert.Status == FIRING {
            activeAlerts = append(activeAlerts, alert)
        }
    }
    
    return activeAlerts
}
```

## Performance Monitoring

### APM (Application Performance Monitoring)

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type APM struct {
    transactions map[string]*Transaction
    mutex        sync.RWMutex
}

type Transaction struct {
    ID        string
    Name      string
    StartTime time.Time
    EndTime   time.Time
    Duration  time.Duration
    Status    TransactionStatus
    Spans     []*Span
    mutex     sync.RWMutex
}

type TransactionStatus int

const (
    SUCCESS TransactionStatus = iota
    FAILURE
    TIMEOUT
)

type APMCollector struct {
    apm *APM
}

func NewAPM() *APM {
    return &APM{
        transactions: make(map[string]*Transaction),
    }
}

func (apm *APM) StartTransaction(name string) *Transaction {
    transactionID := generateID()
    
    transaction := &Transaction{
        ID:        transactionID,
        Name:      name,
        StartTime: time.Now(),
        Spans:     make([]*Span, 0),
    }
    
    apm.mutex.Lock()
    apm.transactions[transactionID] = transaction
    apm.mutex.Unlock()
    
    return transaction
}

func (apm *APM) EndTransaction(transactionID string, status TransactionStatus) {
    apm.mutex.Lock()
    defer apm.mutex.Unlock()
    
    if transaction, exists := apm.transactions[transactionID]; exists {
        transaction.EndTime = time.Now()
        transaction.Duration = transaction.EndTime.Sub(transaction.StartTime)
        transaction.Status = status
    }
}

func (apm *APM) AddSpan(transactionID string, span *Span) {
    apm.mutex.Lock()
    defer apm.mutex.Unlock()
    
    if transaction, exists := apm.transactions[transactionID]; exists {
        transaction.Spans = append(transaction.Spans, span)
    }
}

func (apm *APM) GetTransactionMetrics() map[string]interface{} {
    apm.mutex.RLock()
    defer apm.mutex.RUnlock()
    
    totalTransactions := len(apm.transactions)
    successfulTransactions := 0
    failedTransactions := 0
    totalDuration := time.Duration(0)
    
    for _, transaction := range apm.transactions {
        if transaction.Status == SUCCESS {
            successfulTransactions++
        } else {
            failedTransactions++
        }
        totalDuration += transaction.Duration
    }
    
    avgDuration := time.Duration(0)
    if totalTransactions > 0 {
        avgDuration = totalDuration / time.Duration(totalTransactions)
    }
    
    return map[string]interface{}{
        "total_transactions":     totalTransactions,
        "successful_transactions": successfulTransactions,
        "failed_transactions":    failedTransactions,
        "success_rate":          float64(successfulTransactions) / float64(totalTransactions),
        "average_duration":      avgDuration,
    }
}

// Middleware for automatic transaction tracking
func APMMiddleware(apm *APM, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        transaction := apm.StartTransaction(r.URL.Path)
        defer apm.EndTransaction(transaction.ID, SUCCESS)
        
        // Add transaction ID to response headers
        w.Header().Set("X-Transaction-ID", transaction.ID)
        
        next.ServeHTTP(w, r)
    })
}
```

## SLA/SLO Management

### Service Level Management

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type SLA struct {
    Name        string
    Description string
    Uptime      float64 // 99.9%
    Latency     time.Duration
    Throughput  int
    ErrorRate   float64
}

type SLO struct {
    Name        string
    Description string
    Target      float64
    Window      time.Duration
    Measurements []Measurement
    mutex       sync.RWMutex
}

type Measurement struct {
    Timestamp time.Time
    Value     float64
    Labels    map[string]string
}

type SLOManager struct {
    slos map[string]*SLO
    mutex sync.RWMutex
}

func NewSLOManager() *SLOManager {
    return &SLOManager{
        slos: make(map[string]*SLO),
    }
}

func (sm *SLOManager) AddSLO(slo *SLO) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    sm.slos[slo.Name] = slo
}

func (sm *SLOManager) RecordMeasurement(sloName string, value float64, labels map[string]string) {
    sm.mutex.RLock()
    slo, exists := sm.slos[sloName]
    sm.mutex.RUnlock()
    
    if !exists {
        return
    }
    
    measurement := Measurement{
        Timestamp: time.Now(),
        Value:     value,
        Labels:    labels,
    }
    
    slo.mutex.Lock()
    slo.Measurements = append(slo.Measurements, measurement)
    
    // Keep only measurements within the window
    cutoff := time.Now().Add(-slo.Window)
    var validMeasurements []Measurement
    for _, m := range slo.Measurements {
        if m.Timestamp.After(cutoff) {
            validMeasurements = append(validMeasurements, m)
        }
    }
    slo.Measurements = validMeasurements
    slo.mutex.Unlock()
}

func (sm *SLOManager) GetSLOStatus(sloName string) (float64, bool) {
    sm.mutex.RLock()
    slo, exists := sm.slos[sloName]
    sm.mutex.RUnlock()
    
    if !exists {
        return 0, false
    }
    
    slo.mutex.RLock()
    defer slo.mutex.RUnlock()
    
    if len(slo.Measurements) == 0 {
        return 0, false
    }
    
    // Calculate current SLO value
    totalValue := 0.0
    for _, measurement := range slo.Measurements {
        totalValue += measurement.Value
    }
    
    currentValue := totalValue / float64(len(slo.Measurements))
    isMet := currentValue >= slo.Target
    
    return currentValue, isMet
}

func (sm *SLOManager) GetSLOReport(sloName string) map[string]interface{} {
    sm.mutex.RLock()
    slo, exists := sm.slos[sloName]
    sm.mutex.RUnlock()
    
    if !exists {
        return nil
    }
    
    slo.mutex.RLock()
    defer slo.mutex.RUnlock()
    
    if len(slo.Measurements) == 0 {
        return map[string]interface{}{
            "slo_name":    slo.Name,
            "target":      slo.Target,
            "current":     0.0,
            "is_met":      false,
            "measurements": 0,
        }
    }
    
    totalValue := 0.0
    for _, measurement := range slo.Measurements {
        totalValue += measurement.Value
    }
    
    currentValue := totalValue / float64(len(slo.Measurements))
    isMet := currentValue >= slo.Target
    
    return map[string]interface{}{
        "slo_name":     slo.Name,
        "target":       slo.Target,
        "current":      currentValue,
        "is_met":       isMet,
        "measurements": len(slo.Measurements),
        "window":       slo.Window,
    }
}
```

## Interview Questions

### Basic Concepts
1. **What are the three pillars of observability?**
2. **How do you implement distributed tracing?**
3. **What is the difference between metrics and logs?**
4. **How do you set up alerting for a distributed system?**
5. **What are SLAs and SLOs?**

### Advanced Topics
1. **How would you implement a custom metrics collector?**
2. **How do you handle log aggregation at scale?**
3. **What are the best practices for distributed tracing?**
4. **How do you implement SLO monitoring?**
5. **How do you optimize observability costs?**

### System Design
1. **Design a monitoring and alerting system.**
2. **How would you implement distributed tracing?**
3. **Design a log aggregation system.**
4. **How would you implement APM?**
5. **Design a SLO management system.**

## Conclusion

Advanced monitoring and observability are essential for maintaining reliable systems. Key areas to master:

- **Observability Pillars**: Metrics, logs, traces
- **Distributed Tracing**: Request flow, context propagation
- **Metrics Collection**: Counters, gauges, histograms
- **Log Aggregation**: Centralized logging, search, analysis
- **Alerting**: Rules, notifications, escalation
- **Performance Monitoring**: APM, profiling, optimization
- **SLA/SLO Management**: Service level management, reporting

Understanding these concepts helps in:
- Building observable systems
- Debugging distributed applications
- Maintaining service reliability
- Optimizing performance
- Preparing for technical interviews

This guide provides a comprehensive foundation for advanced monitoring concepts and their practical implementation in Go.
