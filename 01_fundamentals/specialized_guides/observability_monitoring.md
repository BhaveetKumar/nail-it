# Observability and Monitoring Systems

## Table of Contents
- [Introduction](#introduction)
- [Logging Systems](#logging-systems)
- [Metrics Collection](#metrics-collection)
- [Distributed Tracing](#distributed-tracing)
- [Alerting Systems](#alerting-systems)
- [Dashboard and Visualization](#dashboard-and-visualization)
- [Performance Monitoring](#performance-monitoring)
- [Error Tracking](#error-tracking)
- [Synthetic Monitoring](#synthetic-monitoring)
- [Observability Platforms](#observability-platforms)

## Introduction

Observability and monitoring systems are critical for understanding system behavior, detecting issues, and ensuring optimal performance. This guide covers the essential components, patterns, and technologies for building comprehensive observability solutions.

## Logging Systems

### Centralized Logging

```go
// Centralized Logging System
type LoggingSystem struct {
    collectors   []*LogCollector
    processors   []*LogProcessor
    storage      *LogStorage
    search       *LogSearch
    retention    *RetentionManager
    monitoring   *LogMonitoring
}

type LogCollector struct {
    ID            string
    Name          string
    Type          string
    Sources       []*LogSource
    Config        map[string]interface{}
    Status        string
}

type LogSource struct {
    ID            string
    Type          string
    Path          string
    Format        string
    Parser        *LogParser
    Filter        *LogFilter
    RateLimit     int
}

type LogProcessor struct {
    ID            string
    Name          string
    Function      func(*LogEntry) *LogEntry
    Filter        func(*LogEntry) bool
    BatchSize     int
    FlushInterval time.Duration
}

type LogEntry struct {
    ID            string
    Timestamp     time.Time
    Level         string
    Message       string
    Source        string
    Service       string
    Environment   string
    Fields        map[string]interface{}
    Tags          []string
    TraceID       string
    SpanID        string
}

func (ls *LoggingSystem) CollectLogs(source *LogSource) error {
    // Create collector for source
    collector := &LogCollector{
        ID:      generateCollectorID(),
        Name:    source.ID,
        Type:    source.Type,
        Sources: []*LogSource{source},
        Config:  make(map[string]interface{}),
        Status:  "active",
    }
    
    // Start collection
    go ls.startCollection(collector)
    
    // Add to collectors
    ls.collectors = append(ls.collectors, collector)
    
    return nil
}

func (ls *LoggingSystem) startCollection(collector *LogCollector) {
    for _, source := range collector.Sources {
        switch source.Type {
        case "file":
            ls.collectFromFile(source)
        case "syslog":
            ls.collectFromSyslog(source)
        case "journald":
            ls.collectFromJournald(source)
        case "api":
            ls.collectFromAPI(source)
        default:
            log.Printf("Unknown source type: %s", source.Type)
        }
    }
}

func (ls *LoggingSystem) collectFromFile(source *LogSource) {
    file, err := os.Open(source.Path)
    if err != nil {
        log.Printf("Error opening file %s: %v", source.Path, err)
        return
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        
        // Parse log entry
        entry, err := source.Parser.Parse(line)
        if err != nil {
            log.Printf("Error parsing log line: %v", err)
            continue
        }
        
        // Apply filter
        if source.Filter != nil && !source.Filter.Match(entry) {
            continue
        }
        
        // Process entry
        ls.processLogEntry(entry)
    }
}

func (ls *LoggingSystem) processLogEntry(entry *LogEntry) {
    // Process with all processors
    for _, processor := range ls.processors {
        if processor.Filter == nil || processor.Filter(entry) {
            processed := processor.Function(entry)
            if processed != nil {
                entry = processed
            }
        }
    }
    
    // Store entry
    if err := ls.storage.Store(entry); err != nil {
        log.Printf("Error storing log entry: %v", err)
    }
}

// Log Parser
type LogParser struct {
    Format        string
    Pattern       string
    Fields        []string
    TimeFormat    string
    Compiler      *regexp.Regexp
}

func (lp *LogParser) Parse(line string) (*LogEntry, error) {
    if lp.Compiler == nil {
        var err error
        lp.Compiler, err = regexp.Compile(lp.Pattern)
        if err != nil {
            return nil, err
        }
    }
    
    matches := lp.Compiler.FindStringSubmatch(line)
    if len(matches) < len(lp.Fields)+1 {
        return nil, fmt.Errorf("pattern does not match line")
    }
    
    entry := &LogEntry{
        ID:        generateLogID(),
        Timestamp: time.Now(),
        Fields:    make(map[string]interface{}),
        Tags:      make([]string, 0),
    }
    
    // Parse fields
    for i, field := range lp.Fields {
        if i+1 < len(matches) {
            entry.Fields[field] = matches[i+1]
        }
    }
    
    // Parse timestamp if available
    if timestamp, exists := entry.Fields["timestamp"]; exists {
        if t, err := time.Parse(lp.TimeFormat, timestamp.(string)); err == nil {
            entry.Timestamp = t
        }
    }
    
    // Set level
    if level, exists := entry.Fields["level"]; exists {
        entry.Level = level.(string)
    }
    
    // Set message
    if message, exists := entry.Fields["message"]; exists {
        entry.Message = message.(string)
    }
    
    return entry, nil
}

// Common Log Formats
func (ls *LoggingSystem) GetCommonParsers() map[string]*LogParser {
    return map[string]*LogParser{
        "nginx": {
            Format:     "nginx",
            Pattern:    `^(\S+) - (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)"`,
            Fields:     []string{"remote_addr", "remote_user", "time_local", "method", "request_uri", "protocol", "status", "body_bytes_sent", "http_referer", "http_user_agent"},
            TimeFormat: "02/Jan/2006:15:04:05 -0700",
        },
        "apache": {
            Format:     "apache",
            Pattern:    `^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)`,
            Fields:     []string{"remote_addr", "remote_user", "remote_user", "time_local", "method", "request_uri", "protocol", "status", "body_bytes_sent"},
            TimeFormat: "02/Jan/2006:15:04:05 -0700",
        },
        "json": {
            Format:     "json",
            Pattern:    `^(.+)$`,
            Fields:     []string{"message"},
            TimeFormat: time.RFC3339,
        },
    }
}
```

### Structured Logging

```go
// Structured Logging
type StructuredLogger struct {
    service       string
    environment   string
    version       string
    fields        map[string]interface{}
    processors    []*LogProcessor
    output        *LogOutput
}

type LogOutput struct {
    Type          string
    Config        map[string]interface{}
    Writer        io.Writer
}

func NewStructuredLogger(service, environment, version string) *StructuredLogger {
    return &StructuredLogger{
        service:     service,
        environment: environment,
        version:     version,
        fields:      make(map[string]interface{}),
        processors:  make([]*LogProcessor, 0),
        output:      NewLogOutput("json"),
    }
}

func (sl *StructuredLogger) Info(message string, fields ...map[string]interface{}) {
    sl.log("info", message, fields...)
}

func (sl *StructuredLogger) Error(message string, err error, fields ...map[string]interface{}) {
    logFields := make(map[string]interface{})
    if err != nil {
        logFields["error"] = err.Error()
        logFields["error_type"] = reflect.TypeOf(err).String()
    }
    
    // Merge additional fields
    for _, f := range fields {
        for k, v := range f {
            logFields[k] = v
        }
    }
    
    sl.log("error", message, logFields)
}

func (sl *StructuredLogger) Debug(message string, fields ...map[string]interface{}) {
    sl.log("debug", message, fields...)
}

func (sl *StructuredLogger) Warn(message string, fields ...map[string]interface{}) {
    sl.log("warn", message, fields...)
}

func (sl *StructuredLogger) log(level string, message string, fields ...map[string]interface{}) {
    // Create log entry
    entry := &LogEntry{
        ID:          generateLogID(),
        Timestamp:   time.Now(),
        Level:       level,
        Message:     message,
        Service:     sl.service,
        Environment: sl.environment,
        Fields:      make(map[string]interface{}),
        Tags:        make([]string, 0),
    }
    
    // Add default fields
    entry.Fields["service"] = sl.service
    entry.Fields["environment"] = sl.environment
    entry.Fields["version"] = sl.version
    
    // Add logger fields
    for k, v := range sl.fields {
        entry.Fields[k] = v
    }
    
    // Add message fields
    for _, f := range fields {
        for k, v := range f {
            entry.Fields[k] = v
        }
    }
    
    // Process entry
    for _, processor := range sl.processors {
        processed := processor.Function(entry)
        if processed != nil {
            entry = processed
        }
    }
    
    // Output entry
    sl.output.Write(entry)
}

func (sl *StructuredLogger) WithFields(fields map[string]interface{}) *StructuredLogger {
    newLogger := &StructuredLogger{
        service:     sl.service,
        environment: sl.environment,
        version:     sl.version,
        fields:      make(map[string]interface{}),
        processors:  sl.processors,
        output:      sl.output,
    }
    
    // Copy existing fields
    for k, v := range sl.fields {
        newLogger.fields[k] = v
    }
    
    // Add new fields
    for k, v := range fields {
        newLogger.fields[k] = v
    }
    
    return newLogger
}

func (sl *StructuredLogger) WithTrace(traceID, spanID string) *StructuredLogger {
    return sl.WithFields(map[string]interface{}{
        "trace_id": traceID,
        "span_id":  spanID,
    })
}
```

## Metrics Collection

### Metrics System

```go
// Metrics Collection System
type MetricsSystem struct {
    collectors   []*MetricsCollector
    processors   []*MetricsProcessor
    storage      *MetricsStorage
    exporters    []*MetricsExporter
    aggregation  *MetricsAggregation
    monitoring   *MetricsMonitoring
}

type MetricsCollector struct {
    ID            string
    Name          string
    Type          string
    Sources       []*MetricsSource
    Interval      time.Duration
    Status        string
}

type MetricsSource struct {
    ID            string
    Type          string
    Endpoint      string
    Query         string
    Parser        *MetricsParser
    Filter        *MetricsFilter
}

type Metric struct {
    Name          string
    Value         float64
    Timestamp     time.Time
    Labels        map[string]string
    Type          string
    Help          string
}

type MetricsProcessor struct {
    ID            string
    Name          string
    Function      func(*Metric) *Metric
    Filter        func(*Metric) bool
    BatchSize     int
    FlushInterval time.Duration
}

func (ms *MetricsSystem) CollectMetrics(source *MetricsSource) error {
    // Create collector for source
    collector := &MetricsCollector{
        ID:       generateCollectorID(),
        Name:     source.ID,
        Type:     source.Type,
        Sources:  []*MetricsSource{source},
        Interval: time.Minute,
        Status:   "active",
    }
    
    // Start collection
    go ms.startCollection(collector)
    
    // Add to collectors
    ms.collectors = append(ms.collectors, collector)
    
    return nil
}

func (ms *MetricsSystem) startCollection(collector *MetricsCollector) {
    ticker := time.NewTicker(collector.Interval)
    defer ticker.Stop()
    
    for range ticker.C {
        for _, source := range collector.Sources {
            metrics, err := ms.collectFromSource(source)
            if err != nil {
                log.Printf("Error collecting metrics from %s: %v", source.ID, err)
                continue
            }
            
            // Process metrics
            for _, metric := range metrics {
                ms.processMetric(metric)
            }
        }
    }
}

func (ms *MetricsSystem) collectFromSource(source *MetricsSource) ([]*Metric, error) {
    switch source.Type {
    case "prometheus":
        return ms.collectFromPrometheus(source)
    case "statsd":
        return ms.collectFromStatsd(source)
    case "jmx":
        return ms.collectFromJMX(source)
    case "snmp":
        return ms.collectFromSNMP(source)
    default:
        return nil, fmt.Errorf("unknown source type: %s", source.Type)
    }
}

func (ms *MetricsSystem) collectFromPrometheus(source *MetricsSource) ([]*Metric, error) {
    // Make HTTP request to Prometheus endpoint
    resp, err := http.Get(source.Endpoint)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    // Parse Prometheus format
    return ms.parsePrometheusFormat(resp.Body)
}

func (ms *MetricsSystem) parsePrometheusFormat(reader io.Reader) ([]*Metric, error) {
    var metrics []*Metric
    
    scanner := bufio.NewScanner(reader)
    for scanner.Scan() {
        line := scanner.Text()
        
        // Skip comments and empty lines
        if strings.HasPrefix(line, "#") || line == "" {
            continue
        }
        
        // Parse metric line
        metric, err := ms.parseMetricLine(line)
        if err != nil {
            log.Printf("Error parsing metric line: %v", err)
            continue
        }
        
        metrics = append(metrics, metric)
    }
    
    return metrics, nil
}

func (ms *MetricsSystem) parseMetricLine(line string) (*Metric, error) {
    // Parse Prometheus metric format
    // Example: http_requests_total{method="GET",status="200"} 1234
    
    parts := strings.Split(line, " ")
    if len(parts) != 2 {
        return nil, fmt.Errorf("invalid metric format")
    }
    
    nameAndLabels := parts[0]
    valueStr := parts[1]
    
    // Parse value
    value, err := strconv.ParseFloat(valueStr, 64)
    if err != nil {
        return nil, err
    }
    
    // Parse name and labels
    name, labels, err := ms.parseNameAndLabels(nameAndLabels)
    if err != nil {
        return nil, err
    }
    
    return &Metric{
        Name:      name,
        Value:     value,
        Timestamp: time.Now(),
        Labels:    labels,
        Type:      "gauge", // Default type
    }, nil
}

func (ms *MetricsSystem) parseNameAndLabels(nameAndLabels string) (string, map[string]string, error) {
    // Parse name{label1="value1",label2="value2"}
    
    if !strings.Contains(nameAndLabels, "{") {
        return nameAndLabels, make(map[string]string), nil
    }
    
    parts := strings.SplitN(nameAndLabels, "{", 2)
    name := parts[0]
    labelsStr := strings.TrimSuffix(parts[1], "}")
    
    labels := make(map[string]string)
    
    // Parse labels
    labelPairs := strings.Split(labelsStr, ",")
    for _, pair := range labelPairs {
        if pair == "" {
            continue
        }
        
        kv := strings.SplitN(pair, "=", 2)
        if len(kv) != 2 {
            continue
        }
        
        key := strings.TrimSpace(kv[0])
        value := strings.TrimSpace(kv[1])
        value = strings.Trim(value, "\"")
        
        labels[key] = value
    }
    
    return name, labels, nil
}

func (ms *MetricsSystem) processMetric(metric *Metric) {
    // Process with all processors
    for _, processor := range ms.processors {
        if processor.Filter == nil || processor.Filter(metric) {
            processed := processor.Function(metric)
            if processed != nil {
                metric = processed
            }
        }
    }
    
    // Store metric
    if err := ms.storage.Store(metric); err != nil {
        log.Printf("Error storing metric: %v", err)
    }
    
    // Export metric
    for _, exporter := range ms.exporters {
        if err := exporter.Export(metric); err != nil {
            log.Printf("Error exporting metric: %v", err)
        }
    }
}
```

### Custom Metrics

```go
// Custom Metrics
type CustomMetrics struct {
    counters     map[string]*Counter
    gauges       map[string]*Gauge
    histograms   map[string]*Histogram
    summaries    map[string]*Summary
    mu           sync.RWMutex
}

type Counter struct {
    Name          string
    Value         float64
    Labels        map[string]string
    LastUpdated   time.Time
}

type Gauge struct {
    Name          string
    Value         float64
    Labels        map[string]string
    LastUpdated   time.Time
}

type Histogram struct {
    Name          string
    Buckets       []float64
    Counts        []int64
    Sum           float64
    Labels        map[string]string
    LastUpdated   time.Time
}

type Summary struct {
    Name          string
    Count         int64
    Sum           float64
    Quantiles     map[float64]float64
    Labels        map[string]string
    LastUpdated   time.Time
}

func NewCustomMetrics() *CustomMetrics {
    return &CustomMetrics{
        counters:   make(map[string]*Counter),
        gauges:     make(map[string]*Gauge),
        histograms: make(map[string]*Histogram),
        summaries:  make(map[string]*Summary),
    }
}

func (cm *CustomMetrics) IncrementCounter(name string, labels map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    key := cm.getKey(name, labels)
    if counter, exists := cm.counters[key]; exists {
        counter.Value++
        counter.LastUpdated = time.Now()
    } else {
        cm.counters[key] = &Counter{
            Name:        name,
            Value:       1,
            Labels:      labels,
            LastUpdated: time.Now(),
        }
    }
}

func (cm *CustomMetrics) SetGauge(name string, value float64, labels map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    key := cm.getKey(name, labels)
    cm.gauges[key] = &Gauge{
        Name:        name,
        Value:       value,
        Labels:      labels,
        LastUpdated: time.Now(),
    }
}

func (cm *CustomMetrics) ObserveHistogram(name string, value float64, labels map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    key := cm.getKey(name, labels)
    if histogram, exists := cm.histograms[key]; exists {
        cm.updateHistogram(histogram, value)
    } else {
        histogram = &Histogram{
            Name:        name,
            Buckets:     []float64{0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000},
            Counts:      make([]int64, 12),
            Sum:         0,
            Labels:      labels,
            LastUpdated: time.Now(),
        }
        cm.updateHistogram(histogram, value)
        cm.histograms[key] = histogram
    }
}

func (cm *CustomMetrics) updateHistogram(histogram *Histogram, value float64) {
    histogram.Sum += value
    
    for i, bucket := range histogram.Buckets {
        if value <= bucket {
            histogram.Counts[i]++
        }
    }
    
    histogram.LastUpdated = time.Now()
}

func (cm *CustomMetrics) ObserveSummary(name string, value float64, labels map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    key := cm.getKey(name, labels)
    if summary, exists := cm.summaries[key]; exists {
        summary.Count++
        summary.Sum += value
        cm.updateQuantiles(summary, value)
    } else {
        summary = &Summary{
            Name:        name,
            Count:       1,
            Sum:         value,
            Quantiles:   make(map[float64]float64),
            Labels:      labels,
            LastUpdated: time.Now(),
        }
        cm.updateQuantiles(summary, value)
        cm.summaries[key] = summary
    }
}

func (cm *CustomMetrics) updateQuantiles(summary *Summary, value float64) {
    // Simple quantile calculation
    // In practice, this would use a more sophisticated algorithm
    if summary.Count == 1 {
        summary.Quantiles[0.5] = value
        summary.Quantiles[0.95] = value
        summary.Quantiles[0.99] = value
    } else {
        // Update quantiles based on new value
        // This is a simplified implementation
        for q := range summary.Quantiles {
            if value > summary.Quantiles[q] {
                summary.Quantiles[q] = value
            }
        }
    }
    
    summary.LastUpdated = time.Now()
}

func (cm *CustomMetrics) getKey(name string, labels map[string]string) string {
    var parts []string
    parts = append(parts, name)
    
    for k, v := range labels {
        parts = append(parts, fmt.Sprintf("%s=%s", k, v))
    }
    
    return strings.Join(parts, ",")
}

func (cm *CustomMetrics) GetMetrics() []*Metric {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    
    var metrics []*Metric
    
    // Add counters
    for _, counter := range cm.counters {
        metrics = append(metrics, &Metric{
            Name:      counter.Name,
            Value:     counter.Value,
            Timestamp: counter.LastUpdated,
            Labels:    counter.Labels,
            Type:      "counter",
        })
    }
    
    // Add gauges
    for _, gauge := range cm.gauges {
        metrics = append(metrics, &Metric{
            Name:      gauge.Name,
            Value:     gauge.Value,
            Timestamp: gauge.LastUpdated,
            Labels:    gauge.Labels,
            Type:      "gauge",
        })
    }
    
    // Add histograms
    for _, histogram := range cm.histograms {
        for i, bucket := range histogram.Buckets {
            metrics = append(metrics, &Metric{
                Name:      fmt.Sprintf("%s_bucket", histogram.Name),
                Value:     float64(histogram.Counts[i]),
                Timestamp: histogram.LastUpdated,
                Labels:    mergeLabels(histogram.Labels, map[string]string{"le": fmt.Sprintf("%.2f", bucket)}),
                Type:      "histogram",
            })
        }
        
        metrics = append(metrics, &Metric{
            Name:      fmt.Sprintf("%s_sum", histogram.Name),
            Value:     histogram.Sum,
            Timestamp: histogram.LastUpdated,
            Labels:    histogram.Labels,
            Type:      "histogram",
        })
    }
    
    // Add summaries
    for _, summary := range cm.summaries {
        for q, value := range summary.Quantiles {
            metrics = append(metrics, &Metric{
                Name:      fmt.Sprintf("%s", summary.Name),
                Value:     value,
                Timestamp: summary.LastUpdated,
                Labels:    mergeLabels(summary.Labels, map[string]string{"quantile": fmt.Sprintf("%.2f", q)}),
                Type:      "summary",
            })
        }
    }
    
    return metrics
}

func mergeLabels(labels1, labels2 map[string]string) map[string]string {
    result := make(map[string]string)
    
    for k, v := range labels1 {
        result[k] = v
    }
    
    for k, v := range labels2 {
        result[k] = v
    }
    
    return result
}
```

## Distributed Tracing

### Tracing System

```go
// Distributed Tracing System
type TracingSystem struct {
    tracer        *Tracer
    samplers      []*Sampler
    processors    []*TraceProcessor
    storage       *TraceStorage
    exporters     []*TraceExporter
    monitoring    *TraceMonitoring
}

type Tracer struct {
    serviceName   string
    version       string
    environment   string
    samplers      []*Sampler
    processors    []*TraceProcessor
    mu            sync.RWMutex
}

type Span struct {
    TraceID       string
    SpanID        string
    ParentSpanID  string
    OperationName string
    StartTime     time.Time
    EndTime       time.Time
    Duration      time.Duration
    Tags          map[string]interface{}
    Logs          []*SpanLog
    Status        string
    Error         string
}

type SpanLog struct {
    Timestamp     time.Time
    Fields        map[string]interface{}
}

type Sampler struct {
    ID            string
    Name          string
    Function      func(*Span) bool
    Rate          float64
    Enabled       bool
}

type TraceProcessor struct {
    ID            string
    Name          string
    Function      func(*Span) *Span
    Filter        func(*Span) bool
    BatchSize     int
    FlushInterval time.Duration
}

func NewTracingSystem(serviceName, version, environment string) *TracingSystem {
    tracer := &Tracer{
        serviceName: serviceName,
        version:     version,
        environment: environment,
        samplers:    make([]*Sampler, 0),
        processors:  make([]*TraceProcessor, 0),
    }
    
    return &TracingSystem{
        tracer:     tracer,
        samplers:   make([]*Sampler, 0),
        processors: make([]*TraceProcessor, 0),
        storage:    NewTraceStorage(),
        exporters:  make([]*TraceExporter, 0),
        monitoring: NewTraceMonitoring(),
    }
}

func (ts *TracingSystem) StartSpan(operationName string, parentSpan *Span) *Span {
    span := &Span{
        TraceID:       generateTraceID(),
        SpanID:        generateSpanID(),
        OperationName: operationName,
        StartTime:     time.Now(),
        Tags:          make(map[string]interface{}),
        Logs:          make([]*SpanLog, 0),
        Status:        "active",
    }
    
    if parentSpan != nil {
        span.TraceID = parentSpan.TraceID
        span.ParentSpanID = parentSpan.SpanID
    }
    
    // Add default tags
    span.Tags["service.name"] = ts.tracer.serviceName
    span.Tags["service.version"] = ts.tracer.version
    span.Tags["service.environment"] = ts.tracer.environment
    
    return span
}

func (ts *TracingSystem) FinishSpan(span *Span) {
    span.EndTime = time.Now()
    span.Duration = span.EndTime.Sub(span.StartTime)
    span.Status = "finished"
    
    // Check if span should be sampled
    if !ts.shouldSample(span) {
        return
    }
    
    // Process span
    ts.processSpan(span)
}

func (ts *TracingSystem) shouldSample(span *Span) bool {
    for _, sampler := range ts.samplers {
        if sampler.Enabled && sampler.Function(span) {
            return true
        }
    }
    return false
}

func (ts *TracingSystem) processSpan(span *Span) {
    // Process with all processors
    for _, processor := range ts.processors {
        if processor.Filter == nil || processor.Filter(span) {
            processed := processor.Function(span)
            if processed != nil {
                span = processed
            }
        }
    }
    
    // Store span
    if err := ts.storage.Store(span); err != nil {
        log.Printf("Error storing span: %v", err)
    }
    
    // Export span
    for _, exporter := range ts.exporters {
        if err := exporter.Export(span); err != nil {
            log.Printf("Error exporting span: %v", err)
        }
    }
}

func (ts *TracingSystem) AddTag(span *Span, key string, value interface{}) {
    span.Tags[key] = value
}

func (ts *TracingSystem) AddLog(span *Span, fields map[string]interface{}) {
    log := &SpanLog{
        Timestamp: time.Now(),
        Fields:    fields,
    }
    span.Logs = append(span.Logs, log)
}

func (ts *TracingSystem) SetError(span *Span, err error) {
    span.Status = "error"
    span.Error = err.Error()
    span.Tags["error"] = true
    span.Tags["error.message"] = err.Error()
    span.Tags["error.type"] = reflect.TypeOf(err).String()
}

// Probability Sampler
func (ts *TracingSystem) ProbabilitySampler(rate float64) *Sampler {
    return &Sampler{
        ID:       generateSamplerID(),
        Name:     "probability",
        Function: func(span *Span) bool { return rand.Float64() < rate },
        Rate:     rate,
        Enabled:  true,
    }
}

// Rate Limiting Sampler
func (ts *TracingSystem) RateLimitingSampler(maxTracesPerSecond float64) *Sampler {
    limiter := rate.NewLimiter(rate.Limit(maxTracesPerSecond), 1)
    
    return &Sampler{
        ID:       generateSamplerID(),
        Name:     "rate_limiting",
        Function: func(span *Span) bool { return limiter.Allow() },
        Rate:     maxTracesPerSecond,
        Enabled:  true,
    }
}
```

## Alerting Systems

### Alert Management

```go
// Alert Management System
type AlertSystem struct {
    rules         []*AlertRule
    evaluators    []*AlertEvaluator
    notifiers     []*AlertNotifier
    storage       *AlertStorage
    escalation    *AlertEscalation
    monitoring    *AlertMonitoring
}

type AlertRule struct {
    ID            string
    Name          string
    Description   string
    Query         string
    Condition     string
    Threshold     float64
    Duration      time.Duration
    Severity      string
    Labels        map[string]string
    Annotations   map[string]string
    Enabled       bool
    CreatedAt     time.Time
    UpdatedAt     time.Time
}

type AlertEvaluator struct {
    ID            string
    Name          string
    Function      func(*AlertRule, []*Metric) *Alert
    Interval      time.Duration
    Enabled       bool
}

type AlertNotifier struct {
    ID            string
    Name          string
    Type          string
    Config        map[string]interface{}
    Function      func(*Alert) error
    Enabled       bool
}

type Alert struct {
    ID            string
    RuleID        string
    Name          string
    Description   string
    Severity      string
    Status        string
    Labels        map[string]string
    Annotations   map[string]string
    Value         float64
    Threshold     float64
    FiredAt       time.Time
    ResolvedAt    time.Time
    Notifications []*AlertNotification
}

type AlertNotification struct {
    ID            string
    AlertID       string
    NotifierID    string
    Status        string
    SentAt        time.Time
    Error         string
}

func (as *AlertSystem) EvaluateRules() error {
    for _, rule := range as.rules {
        if !rule.Enabled {
            continue
        }
        
        // Get metrics for rule
        metrics, err := as.getMetricsForRule(rule)
        if err != nil {
            log.Printf("Error getting metrics for rule %s: %v", rule.ID, err)
            continue
        }
        
        // Evaluate rule
        for _, evaluator := range as.evaluators {
            if evaluator.Enabled {
                alert := evaluator.Function(rule, metrics)
                if alert != nil {
                    // Process alert
                    if err := as.processAlert(alert); err != nil {
                        log.Printf("Error processing alert: %v", err)
                    }
                }
            }
        }
    }
    
    return nil
}

func (as *AlertSystem) processAlert(alert *Alert) error {
    // Check if alert already exists
    existing, err := as.storage.GetAlert(alert.RuleID)
    if err == nil && existing != nil {
        // Update existing alert
        alert.ID = existing.ID
        alert.FiredAt = existing.FiredAt
        alert.Status = "firing"
    } else {
        // Create new alert
        alert.ID = generateAlertID()
        alert.FiredAt = time.Now()
        alert.Status = "firing"
    }
    
    // Store alert
    if err := as.storage.StoreAlert(alert); err != nil {
        return err
    }
    
    // Send notifications
    if err := as.sendNotifications(alert); err != nil {
        return err
    }
    
    // Check escalation
    if err := as.escalation.CheckEscalation(alert); err != nil {
        return err
    }
    
    return nil
}

func (as *AlertSystem) sendNotifications(alert *Alert) error {
    for _, notifier := range as.notifiers {
        if !notifier.Enabled {
            continue
        }
        
        notification := &AlertNotification{
            ID:        generateNotificationID(),
            AlertID:   alert.ID,
            NotifierID: notifier.ID,
            Status:    "pending",
            SentAt:    time.Now(),
        }
        
        // Send notification
        if err := notifier.Function(alert); err != nil {
            notification.Status = "failed"
            notification.Error = err.Error()
        } else {
            notification.Status = "sent"
        }
        
        // Store notification
        alert.Notifications = append(alert.Notifications, notification)
    }
    
    return nil
}

// Threshold Evaluator
func (as *AlertSystem) ThresholdEvaluator(rule *AlertRule, metrics []*Metric) *Alert {
    // Find metric matching rule query
    var targetMetric *Metric
    for _, metric := range metrics {
        if as.matchesQuery(metric, rule.Query) {
            targetMetric = metric
            break
        }
    }
    
    if targetMetric == nil {
        return nil
    }
    
    // Check threshold
    var triggered bool
    switch rule.Condition {
    case "greater_than":
        triggered = targetMetric.Value > rule.Threshold
    case "less_than":
        triggered = targetMetric.Value < rule.Threshold
    case "equals":
        triggered = targetMetric.Value == rule.Threshold
    case "not_equals":
        triggered = targetMetric.Value != rule.Threshold
    default:
        return nil
    }
    
    if !triggered {
        return nil
    }
    
    return &Alert{
        RuleID:      rule.ID,
        Name:        rule.Name,
        Description: rule.Description,
        Severity:    rule.Severity,
        Labels:      rule.Labels,
        Annotations: rule.Annotations,
        Value:       targetMetric.Value,
        Threshold:   rule.Threshold,
    }
}

// Email Notifier
func (as *AlertSystem) EmailNotifier(config map[string]interface{}) *AlertNotifier {
    return &AlertNotifier{
        ID:       generateNotifierID(),
        Name:     "email",
        Type:     "email",
        Config:   config,
        Function: func(alert *Alert) error {
            // Send email notification
            return as.sendEmail(alert, config)
        },
        Enabled:  true,
    }
}

// Slack Notifier
func (as *AlertSystem) SlackNotifier(config map[string]interface{}) *AlertNotifier {
    return &AlertNotifier{
        ID:       generateNotifierID(),
        Name:     "slack",
        Type:     "slack",
        Config:   config,
        Function: func(alert *Alert) error {
            // Send Slack notification
            return as.sendSlack(alert, config)
        },
        Enabled:  true,
    }
}
```

## Conclusion

Observability and monitoring systems are essential for maintaining system health and performance. Key areas to focus on include:

1. **Logging**: Centralized logging, structured logging, and log analysis
2. **Metrics**: Collection, processing, and visualization of system metrics
3. **Tracing**: Distributed tracing for understanding request flows
4. **Alerting**: Rule-based alerting and notification systems
5. **Dashboards**: Visualization and monitoring interfaces
6. **Performance**: Application and infrastructure performance monitoring
7. **Error Tracking**: Error detection, analysis, and resolution
8. **Synthetic Monitoring**: Proactive monitoring and testing

Mastering these areas will prepare you for building comprehensive observability solutions that provide deep insights into system behavior and performance.

## Additional Resources

- [OpenTelemetry](https://opentelemetry.io/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Jaeger](https://www.jaegertracing.io/)
- [ELK Stack](https://www.elastic.co/elastic-stack/)
- [Fluentd](https://www.fluentd.org/)
- [DataDog](https://www.datadoghq.com/)
- [New Relic](https://newrelic.com/)
