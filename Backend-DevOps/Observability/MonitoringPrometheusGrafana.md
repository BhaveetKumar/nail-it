# 📊 Monitoring with Prometheus and Grafana: Metrics Collection and Visualization

> **Master Prometheus for metrics collection and Grafana for visualization and alerting**

## 📚 Concept

Prometheus is a monitoring and alerting toolkit that collects metrics from configured targets, stores them in a time-series database, and provides a query language for analysis. Grafana is a visualization and analytics platform that creates dashboards from various data sources including Prometheus.

### Key Features
- **Time-Series Database**: Efficient storage of metrics
- **Pull Model**: Prometheus scrapes metrics from targets
- **Query Language**: PromQL for data analysis
- **Alerting**: Rule-based alerting system
- **Visualization**: Grafana dashboards and graphs
- **Service Discovery**: Automatic target discovery

## 🏗️ Monitoring Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Monitoring Stack                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Application │  │  System     │  │  Database   │     │
│  │  Metrics    │  │  Metrics    │  │  Metrics    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Prometheus                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Scraper   │  │   Storage   │  │   Query     │ │ │
│  │  │   Engine    │  │   Engine    │  │   Engine    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Grafana   │  │  Alertmanager│  │   Pushgateway│     │
│  │ (Dashboards)│  │  (Alerts)   │  │  (Push)     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Go Application with Prometheus Metrics

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
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "go.uber.org/zap"
)

// Prometheus metrics
var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )

    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )

    httpRequestsInFlight = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "http_requests_in_flight",
            Help: "Number of HTTP requests currently being processed",
        },
    )

    databaseConnections = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "database_connections_active",
            Help: "Number of active database connections",
        },
        []string{"database", "state"},
    )

    businessMetrics = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "business_operations_total",
            Help: "Total number of business operations",
        },
        []string{"operation", "status"},
    )
)

// Prometheus middleware
func prometheusMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        
        // Increment in-flight requests
        httpRequestsInFlight.Inc()
        defer httpRequestsInFlight.Dec()
        
        // Process request
        c.Next()
        
        // Record metrics
        duration := time.Since(start).Seconds()
        status := fmt.Sprintf("%d", c.Writer.Status())
        
        httpRequestsTotal.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
            status,
        ).Inc()
        
        httpRequestDuration.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
        ).Observe(duration)
    }
}

// Business logic with metrics
type UserService struct {
    logger *zap.Logger
}

func NewUserService(logger *zap.Logger) *UserService {
    return &UserService{logger: logger}
}

func (s *UserService) GetUser(ctx context.Context, userID string) (*User, error) {
    start := time.Now()
    
    // Simulate database call
    user, err := s.fetchUserFromDB(ctx, userID)
    
    duration := time.Since(start).Seconds()
    
    if err != nil {
        businessMetrics.WithLabelValues("get_user", "error").Inc()
        s.logger.Error("Failed to get user",
            zap.String("user_id", userID),
            zap.Error(err),
            zap.Duration("duration", time.Since(start)),
        )
        return nil, err
    }
    
    businessMetrics.WithLabelValues("get_user", "success").Inc()
    s.logger.Info("User retrieved successfully",
        zap.String("user_id", userID),
        zap.Duration("duration", time.Since(start)),
    )
    
    return user, nil
}

func (s *UserService) fetchUserFromDB(ctx context.Context, userID string) (*User, error) {
    // Simulate database operation
    time.Sleep(100 * time.Millisecond)
    
    if userID == "error" {
        return nil, fmt.Errorf("database connection failed")
    }
    
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

// Custom metrics collector
type CustomCollector struct {
    userCountDesc *prometheus.Desc
}

func NewCustomCollector() *CustomCollector {
    return &CustomCollector{
        userCountDesc: prometheus.NewDesc(
            "custom_user_count",
            "Total number of users",
            []string{"role"},
            nil,
        ),
    }
}

func (c *CustomCollector) Describe(ch chan<- *prometheus.Desc) {
    ch <- c.userCountDesc
}

func (c *CustomCollector) Collect(ch chan<- prometheus.Metric) {
    // Simulate user count by role
    userCounts := map[string]float64{
        "admin": 10,
        "user":  100,
        "guest": 50,
    }
    
    for role, count := range userCounts {
        ch <- prometheus.MustNewConstMetric(
            c.userCountDesc,
            prometheus.CounterValue,
            count,
            role,
        )
    }
}

// HTTP handlers
func setupRoutes(logger *zap.Logger) *gin.Engine {
    r := gin.New()
    
    // Add middleware
    r.Use(prometheusMiddleware())
    r.Use(gin.Recovery())
    
    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "timestamp": time.Now().UTC(),
        })
    })
    
    // Metrics endpoint
    r.GET("/metrics", gin.WrapH(promhttp.Handler()))
    
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
    // Setup logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // Register custom collector
    prometheus.MustRegister(NewCustomCollector())
    
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

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    environment: 'prod'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'my-app'
    static_configs:
      - targets: ['my-app:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
    scrape_timeout: 5s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics

  - job_name: 'kubernetes-cadvisor'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics/cadvisor

  - job_name: 'kubernetes-service-endpoints'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scheme]
        action: replace
        target_label: __scheme__
        regex: (https?)
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_service_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
      - action: labelmap
        regex: __meta_kubernetes_service_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_service_name]
        action: replace
        target_label: kubernetes_name
```

### Alert Rules

```yaml
# rules/alerts.yml
groups:
- name: application.rules
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}%"

  - alert: DiskSpaceLow
    expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Disk space low"
      description: "Disk space is {{ $value | humanizePercentage }}"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "Service {{ $labels.instance }} is down"

  - alert: DatabaseConnectionsHigh
    expr: database_connections_active > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database connections"
      description: "Database connections are {{ $value }}"

  - alert: BusinessOperationFailure
    expr: rate(business_operations_total{status="error"}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Business operation failure rate high"
      description: "Failure rate is {{ $value }} failures per second"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'alerts@example.com'
  smtp_auth_password: 'password'

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
  - url: 'http://webhook:5001/'

- name: 'critical-alerts'
  email_configs:
  - to: 'critical@example.com'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#critical-alerts'
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  email_configs:
  - to: 'warnings@example.com'
    subject: 'WARNING: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#warnings'
    title: 'Warning Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Application Monitoring",
    "tags": ["application", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "HTTP Requests Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "xAxis": {
          "mode": "time"
        }
      },
      {
        "id": 2,
        "title": "HTTP Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          },
          {
            "expr": "rate(http_requests_total{status=~\"4..\"}[5m])",
            "legendFormat": "4xx errors"
          }
        ],
        "yAxes": [
          {
            "label": "Errors/sec",
            "min": 0
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "http_requests_in_flight",
            "legendFormat": "In Flight"
          }
        ],
        "valueName": "current"
      },
      {
        "id": 5,
        "title": "Business Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(business_operations_total[5m])",
            "legendFormat": "{{operation}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Operations/sec",
            "min": 0
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### Docker Compose for Monitoring Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning

  my-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=info
      - ENVIRONMENT=production
    depends_on:
      - prometheus

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

volumes:
  prometheus_data:
  alertmanager_data:
  grafana_data:
```

## 🚀 Best Practices

### 1. Metric Naming
```go
// Use consistent naming conventions
var httpRequestsTotal = prometheus.NewCounterVec(
    prometheus.CounterOpts{
        Name: "http_requests_total",  // metric_name_unit
        Help: "Total number of HTTP requests",
    },
    []string{"method", "endpoint", "status"},
)
```

### 2. Label Management
```go
// Use meaningful labels
httpRequestsTotal.WithLabelValues(
    c.Request.Method,    // method
    c.FullPath(),        // endpoint
    status,              // status
).Inc()
```

### 3. Alert Thresholds
```yaml
# Set appropriate thresholds
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
```

## 🏢 Industry Insights

### Monitoring Usage Patterns
- **Application Metrics**: Business and technical metrics
- **Infrastructure Metrics**: System and resource metrics
- **Custom Metrics**: Business-specific measurements
- **SLA Monitoring**: Service level agreements

### Enterprise Monitoring Strategy
- **Multi-Tenant**: Isolated monitoring per tenant
- **Scalability**: Horizontal scaling
- **Retention**: Long-term storage
- **Cost Optimization**: Efficient storage

## 🎯 Interview Questions

### Basic Level
1. **What is Prometheus?**
   - Time-series database
   - Metrics collection
   - Query language (PromQL)
   - Alerting system

2. **What is Grafana?**
   - Visualization platform
   - Dashboard creation
   - Multiple data sources
   - Alerting integration

3. **What are Prometheus metrics types?**
   - Counter: Monotonically increasing
   - Gauge: Can go up or down
   - Histogram: Distribution of values
   - Summary: Quantiles and counts

### Intermediate Level
4. **How do you implement Prometheus metrics?**
   ```go
   var httpRequestsTotal = prometheus.NewCounterVec(
       prometheus.CounterOpts{
           Name: "http_requests_total",
           Help: "Total number of HTTP requests",
       },
       []string{"method", "endpoint", "status"},
   )
   ```

5. **How do you create Grafana dashboards?**
   - Panel configuration
   - Query definition
   - Visualization options
   - Alert rules

6. **How do you handle Prometheus alerting?**
   - Alert rules
   - Alertmanager configuration
   - Notification channels
   - Alert grouping

### Advanced Level
7. **How do you implement Prometheus patterns?**
   - Service discovery
   - Federation
   - Remote storage
   - High availability

8. **How do you handle Prometheus scaling?**
   - Horizontal scaling
   - Sharding
   - Federation
   - Remote storage

9. **How do you implement Prometheus security?**
   - Authentication
   - Authorization
   - TLS encryption
   - Network policies

---

**Next**: [Tracing](./Tracing.md) - Distributed tracing, OpenTelemetry, Jaeger
