# Metrics and Monitoring Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [Types of Metrics](#types-of-metrics)
3. [Prometheus Integration](#prometheus-integration)
4. [Grafana Dashboards](#grafana-dashboards)
5. [Alerting Strategies](#alerting-strategies)
6. [Golang Implementation](#golang-implementation)
7. [Microservices Monitoring](#microservices-monitoring)
8. [Performance Metrics](#performance-metrics)
9. [Business Metrics](#business-metrics)
10. [Best Practices](#best-practices)

## Introduction

Metrics and monitoring are crucial for understanding system health, performance, and business outcomes. This guide covers comprehensive monitoring strategies for distributed systems, with a focus on Prometheus and Grafana.

## Types of Metrics

### The Four Golden Signals
1. **Latency**: Time taken to serve a request
2. **Traffic**: How much demand is placed on your system
3. **Errors**: Rate of requests that fail
4. **Saturation**: How "full" your service is

### Metric Categories

#### Counter Metrics
```go
// Total number of requests
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status_code"},
    )
)

// Usage
httpRequestsTotal.WithLabelValues("GET", "/api/users", "200").Inc()
```

#### Gauge Metrics
```go
// Current number of active connections
var (
    activeConnections = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Current number of active connections",
        },
    )
)

// Usage
activeConnections.Inc()  // New connection
activeConnections.Dec()  // Connection closed
```

#### Histogram Metrics
```go
// Request duration distribution
var (
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
)

// Usage
timer := prometheus.NewTimer(httpRequestDuration.WithLabelValues("GET", "/api/users"))
defer timer.ObserveDuration()
```

#### Summary Metrics
```go
// Custom quantiles for response time
var (
    responseTime = prometheus.NewSummaryVec(
        prometheus.SummaryOpts{
            Name:       "response_time_seconds",
            Help:       "Response time in seconds",
            Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
        },
        []string{"service", "endpoint"},
    )
)
```

## Prometheus Integration

### Basic Setup
```go
package main

import (
    "net/http"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
    // Register metrics
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(activeConnections)
    prometheus.MustRegister(httpRequestDuration)
    
    // Expose metrics endpoint
    http.Handle("/metrics", promhttp.Handler())
    
    // Start server
    http.ListenAndServe(":8080", nil)
}
```

### Custom Metrics
```go
// Business-specific metrics
type PaymentMetrics struct {
    paymentsTotal     *prometheus.CounterVec
    paymentAmount     *prometheus.HistogramVec
    paymentErrors     *prometheus.CounterVec
    activePayments    prometheus.Gauge
}

func NewPaymentMetrics() *PaymentMetrics {
    return &PaymentMetrics{
        paymentsTotal: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "payments_total",
                Help: "Total number of payments processed",
            },
            []string{"status", "payment_method", "currency"},
        ),
        paymentAmount: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "payment_amount",
                Help:    "Payment amount distribution",
                Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000},
            },
            []string{"currency", "payment_method"},
        ),
        paymentErrors: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "payment_errors_total",
                Help: "Total number of payment errors",
            },
            []string{"error_type", "payment_method"},
        ),
        activePayments: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "active_payments",
                Help: "Current number of active payments",
            },
        ),
    }
}

func (pm *PaymentMetrics) RecordPayment(status, method, currency string, amount float64) {
    pm.paymentsTotal.WithLabelValues(status, method, currency).Inc()
    pm.paymentAmount.WithLabelValues(currency, method).Observe(amount)
}

func (pm *PaymentMetrics) RecordError(errorType, method string) {
    pm.paymentErrors.WithLabelValues(errorType, method).Inc()
}
```

### Service Discovery
```go
// Kubernetes service discovery
type K8sServiceDiscovery struct {
    client    kubernetes.Interface
    namespace string
}

func (ksd *K8sServiceDiscovery) Run(ctx context.Context) {
    // Watch for service changes
    watcher, err := ksd.client.CoreV1().Services(ksd.namespace).Watch(ctx, metav1.ListOptions{})
    if err != nil {
        log.Fatal(err)
    }
    
    for event := range watcher.ResultChan() {
        switch event.Type {
        case watch.Added, watch.Modified, watch.Deleted:
            ksd.updateTargets(event.Object.(*v1.Service))
        }
    }
}
```

## Grafana Dashboards

### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Payment Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules
```yaml
groups:
- name: payment_service
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
      description: "95th percentile latency is {{ $value }}s"
```

## Alerting Strategies

### Alert Severity Levels
```go
type AlertSeverity string

const (
    SeverityInfo     AlertSeverity = "info"
    SeverityWarning  AlertSeverity = "warning"
    SeverityCritical AlertSeverity = "critical"
)

type Alert struct {
    Name        string
    Expression  string
    Duration    time.Duration
    Severity    AlertSeverity
    Labels      map[string]string
    Annotations map[string]string
}
```

### Alert Manager Integration
```go
type AlertManager struct {
    client  *http.Client
    baseURL string
}

func (am *AlertManager) SendAlert(alert Alert) error {
    payload := map[string]interface{}{
        "labels": alert.Labels,
        "annotations": alert.Annotations,
        "startsAt": time.Now().Format(time.RFC3339),
    }
    
    jsonData, _ := json.Marshal(payload)
    resp, err := am.client.Post(am.baseURL+"/api/v1/alerts", "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}
```

## Golang Implementation

### Middleware for Metrics
```go
func MetricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap ResponseWriter to capture status code
        ww := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(ww, r)
        
        duration := time.Since(start).Seconds()
        
        // Record metrics
        httpRequestsTotal.WithLabelValues(
            r.Method,
            r.URL.Path,
            strconv.Itoa(ww.statusCode),
        ).Inc()
        
        httpRequestDuration.WithLabelValues(
            r.Method,
            r.URL.Path,
        ).Observe(duration)
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

### Custom Collectors
```go
type DatabaseCollector struct {
    db *sql.DB
}

func (dc *DatabaseCollector) Describe(ch chan<- *prometheus.Desc) {
    ch <- prometheus.NewDesc("database_connections_active", "Active database connections", nil, nil)
    ch <- prometheus.NewDesc("database_connections_idle", "Idle database connections", nil, nil)
}

func (dc *DatabaseCollector) Collect(ch chan<- prometheus.Metric) {
    stats := dc.db.Stats()
    
    ch <- prometheus.MustNewConstMetric(
        prometheus.NewDesc("database_connections_active", "Active database connections", nil, nil),
        prometheus.GaugeValue,
        float64(stats.OpenConnections-stats.Idle),
    )
    
    ch <- prometheus.MustNewConstMetric(
        prometheus.NewDesc("database_connections_idle", "Idle database connections", nil, nil),
        prometheus.GaugeValue,
        float64(stats.Idle),
    )
}
```

## Microservices Monitoring

### Distributed Tracing
```go
import "go.opentelemetry.io/otel"

func TraceMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx := r.Context()
        
        // Extract trace context
        span := trace.SpanFromContext(ctx)
        if span.IsRecording() {
            span.SetAttributes(
                attribute.String("http.method", r.Method),
                attribute.String("http.url", r.URL.String()),
            )
        }
        
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

### Service Mesh Metrics
```go
type ServiceMeshMetrics struct {
    requestDuration *prometheus.HistogramVec
    requestSize     *prometheus.HistogramVec
    responseSize    *prometheus.HistogramVec
    tcpConnections  *prometheus.GaugeVec
}

func NewServiceMeshMetrics() *ServiceMeshMetrics {
    return &ServiceMeshMetrics{
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "istio_request_duration_seconds",
                Help: "Request duration in seconds",
            },
            []string{"source", "destination", "protocol"},
        ),
        requestSize: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "istio_request_bytes",
                Help: "Request size in bytes",
            },
            []string{"source", "destination"},
        ),
        responseSize: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "istio_response_bytes",
                Help: "Response size in bytes",
            },
            []string{"source", "destination"},
        ),
        tcpConnections: prometheus.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "istio_tcp_connections_opened_total",
                Help: "Total number of TCP connections opened",
            },
            []string{"source", "destination"},
        ),
    }
}
```

## Performance Metrics

### Application Performance
```go
type PerformanceMetrics struct {
    goroutines    prometheus.Gauge
    memoryUsage   prometheus.Gauge
    gcDuration    prometheus.Histogram
    cpuUsage      prometheus.Gauge
}

func NewPerformanceMetrics() *PerformanceMetrics {
    return &PerformanceMetrics{
        goroutines: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "go_goroutines",
                Help: "Number of goroutines",
            },
        ),
        memoryUsage: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "go_memstats_alloc_bytes",
                Help: "Current memory allocation in bytes",
            },
        ),
        gcDuration: prometheus.NewHistogram(
            prometheus.HistogramOpts{
                Name: "go_gc_duration_seconds",
                Help: "GC duration in seconds",
            },
        ),
        cpuUsage: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "process_cpu_seconds_total",
                Help: "Total CPU time in seconds",
            },
        ),
    }
}

func (pm *PerformanceMetrics) Update() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    pm.goroutines.Set(float64(runtime.NumGoroutine()))
    pm.memoryUsage.Set(float64(m.Alloc))
    pm.gcDuration.Observe(m.GCCPUFraction)
}
```

### Database Performance
```go
type DatabaseMetrics struct {
    queryDuration    *prometheus.HistogramVec
    queryErrors      *prometheus.CounterVec
    connectionPool   *prometheus.GaugeVec
    transactionCount *prometheus.CounterVec
}

func (dm *DatabaseMetrics) RecordQuery(query string, duration time.Duration, err error) {
    labels := []string{query}
    dm.queryDuration.WithLabelValues(labels...).Observe(duration.Seconds())
    
    if err != nil {
        dm.queryErrors.WithLabelValues(query, err.Error()).Inc()
    }
}
```

## Business Metrics

### Revenue Metrics
```go
type RevenueMetrics struct {
    dailyRevenue     *prometheus.GaugeVec
    monthlyRevenue   *prometheus.GaugeVec
    revenueGrowth    *prometheus.GaugeVec
    averageOrderValue prometheus.Gauge
}

func (rm *RevenueMetrics) RecordTransaction(amount float64, currency string) {
    rm.dailyRevenue.WithLabelValues(currency).Add(amount)
    rm.monthlyRevenue.WithLabelValues(currency).Add(amount)
}
```

### User Engagement
```go
type EngagementMetrics struct {
    activeUsers      *prometheus.GaugeVec
    sessionDuration  *prometheus.HistogramVec
    pageViews        *prometheus.CounterVec
    conversionRate   *prometheus.GaugeVec
}

func (em *EngagementMetrics) RecordSession(userID string, duration time.Duration) {
    em.sessionDuration.WithLabelValues("total").Observe(duration.Seconds())
    em.activeUsers.WithLabelValues("concurrent").Inc()
}
```

## Best Practices

### 1. Metric Naming
```go
// Good: Descriptive and consistent naming
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",        // _total suffix for counters
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds", // _seconds suffix for time
            Help: "HTTP request duration",
        },
        []string{"method", "endpoint"},
    )
)
```

### 2. Label Cardinality
```go
// Bad: High cardinality labels
var badMetric = prometheus.NewCounterVec(
    prometheus.CounterOpts{Name: "requests_total"},
    []string{"user_id", "request_id"}, // Too many unique combinations
)

// Good: Low cardinality labels
var goodMetric = prometheus.NewCounterVec(
    prometheus.CounterOpts{Name: "requests_total"},
    []string{"method", "endpoint", "status"}, // Limited unique combinations
)
```

### 3. Metric Collection
```go
// Collect metrics in background
func (s *Server) collectMetrics() {
    ticker := time.NewTicker(15 * time.Second)
    go func() {
        for range ticker.C {
            s.updateMetrics()
        }
    }()
}

func (s *Server) updateMetrics() {
    // Update custom metrics
    s.performanceMetrics.Update()
    s.businessMetrics.Update()
}
```

### 4. Error Handling
```go
func (m *Metrics) RecordError(operation string, err error) {
    if err != nil {
        m.errorsTotal.WithLabelValues(operation, err.Error()).Inc()
    }
}
```

## Conclusion

Effective metrics and monitoring are essential for:

1. **System Health**: Monitor the four golden signals
2. **Performance**: Track latency, throughput, and resource usage
3. **Business Value**: Measure revenue, user engagement, and growth
4. **Reliability**: Set up proper alerting and incident response
5. **Scalability**: Monitor capacity and plan for growth

Key principles:
- Use structured metrics with appropriate types
- Implement proper labeling without high cardinality
- Set up comprehensive dashboards and alerting
- Monitor both technical and business metrics
- Continuously improve based on monitoring insights
