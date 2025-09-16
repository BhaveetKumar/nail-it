# 📊 Monitoring with Prometheus and Grafana: Metrics Collection and Visualization

> **Master Prometheus for metrics collection and Grafana for visualization and alerting**

## 📚 Concept

**Detailed Explanation:**
Prometheus and Grafana form a powerful monitoring and observability stack that provides comprehensive insights into application and infrastructure performance. Prometheus serves as the metrics collection and storage engine, while Grafana provides the visualization and alerting capabilities. Together, they enable organizations to monitor, analyze, and respond to system behavior in real-time.

**Core Philosophy:**

- **Pull-Based Monitoring**: Prometheus actively scrapes metrics from targets rather than receiving pushed data
- **Time-Series Focus**: Optimized for storing and querying time-series data
- **Multi-Dimensional Data**: Metrics are identified by metric name and key-value pairs (labels)
- **PromQL Power**: Powerful query language for data analysis and alerting
- **Visualization First**: Grafana provides intuitive dashboards and visualizations
- **Alert-Driven Operations**: Proactive alerting based on defined rules and thresholds

**Why Prometheus and Grafana Matter:**

- **Observability**: Comprehensive visibility into system behavior and performance
- **Proactive Monitoring**: Early detection of issues before they impact users
- **Performance Optimization**: Data-driven insights for system optimization
- **Capacity Planning**: Historical data for resource planning and scaling decisions
- **Incident Response**: Quick identification and resolution of production issues
- **Business Intelligence**: Metrics that connect technical performance to business outcomes
- **Compliance**: Audit trails and monitoring for regulatory requirements
- **Cost Optimization**: Resource usage monitoring for cost management

**Key Features:**

**1. Time-Series Database:**

- **Definition**: Specialized database optimized for time-ordered data points
- **Purpose**: Efficient storage and retrieval of metrics over time
- **Benefits**: Fast queries, compression, retention policies, high cardinality support
- **Use Cases**: Application metrics, system metrics, business metrics, custom measurements
- **Best Practices**: Use appropriate retention periods, implement data lifecycle management

**2. Pull Model:**

- **Definition**: Prometheus actively scrapes metrics from configured targets
- **Purpose**: Centralized control over data collection and reliability
- **Benefits**: No data loss during target downtime, centralized configuration, service discovery
- **Use Cases**: Microservices monitoring, infrastructure monitoring, application monitoring
- **Best Practices**: Configure appropriate scrape intervals, implement health checks

**3. Query Language (PromQL):**

- **Definition**: Powerful query language for time-series data analysis
- **Purpose**: Flexible data analysis, aggregation, and alerting
- **Benefits**: Complex queries, mathematical operations, time-based analysis
- **Use Cases**: Alerting rules, dashboard queries, ad-hoc analysis, capacity planning
- **Best Practices**: Use efficient queries, leverage functions, implement proper aggregation

**4. Alerting System:**

- **Definition**: Rule-based alerting with configurable thresholds and conditions
- **Purpose**: Proactive notification of issues and anomalies
- **Benefits**: Early warning, automated response, escalation policies
- **Use Cases**: Performance degradation, error rate spikes, resource exhaustion, business metrics
- **Best Practices**: Set appropriate thresholds, implement alert grouping, use multiple notification channels

**5. Visualization (Grafana):**

- **Definition**: Rich visualization platform with multiple data source support
- **Purpose**: Intuitive dashboards and graphs for data analysis
- **Benefits**: Custom dashboards, real-time updates, sharing capabilities, alerting integration
- **Use Cases**: Operations dashboards, executive reporting, troubleshooting, capacity planning
- **Best Practices**: Design clear dashboards, use appropriate visualizations, implement proper refresh rates

**6. Service Discovery:**

- **Definition**: Automatic discovery and configuration of monitoring targets
- **Purpose**: Dynamic monitoring in containerized and cloud environments
- **Benefits**: Reduced manual configuration, automatic scaling, environment adaptation
- **Use Cases**: Kubernetes monitoring, cloud infrastructure, microservices, dynamic environments
- **Best Practices**: Use appropriate discovery mechanisms, implement proper labeling, configure relabeling

**Advanced Monitoring Concepts:**

- **Metric Types**: Counter, Gauge, Histogram, Summary for different measurement types
- **Labeling Strategy**: Effective use of labels for multi-dimensional data
- **Federation**: Hierarchical monitoring for large-scale deployments
- **Remote Storage**: Long-term storage integration with external systems
- **High Availability**: Clustering and redundancy for production environments
- **Security**: Authentication, authorization, and encryption for monitoring data
- **Cost Management**: Efficient storage and query optimization
- **Custom Metrics**: Business-specific measurements and KPIs

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive monitoring strategy using Prometheus and Grafana for a large-scale microservices architecture?**

**Answer:** Comprehensive monitoring strategy design:

- **Service Discovery**: Implement Kubernetes service discovery for automatic target configuration
- **Multi-Level Metrics**: Collect application, infrastructure, and business metrics
- **Labeling Strategy**: Use consistent labeling for service identification and filtering
- **Dashboard Hierarchy**: Create dashboards for different audiences (operations, developers, executives)
- **Alerting Strategy**: Implement multi-level alerting with proper escalation and grouping
- **Retention Policies**: Configure appropriate data retention for different metric types
- **Federation**: Use Prometheus federation for multi-cluster monitoring
- **Custom Metrics**: Implement business-specific metrics for KPI monitoring
- **Performance Optimization**: Optimize queries and storage for large-scale deployments
- **Security**: Implement proper authentication and authorization for monitoring access
- **Documentation**: Maintain comprehensive documentation of monitoring setup and procedures
- **Testing**: Implement monitoring testing and validation procedures

**Q2: What are the key considerations when implementing Prometheus metrics in Go applications and how do you ensure optimal performance?**

**Answer:** Go application metrics implementation:

- **Metric Types**: Choose appropriate metric types (Counter, Gauge, Histogram, Summary) for different measurements
- **Labeling**: Use meaningful labels but avoid high cardinality to prevent performance issues
- **Middleware Integration**: Implement metrics collection in HTTP middleware for consistent coverage
- **Business Metrics**: Collect business-specific metrics alongside technical metrics
- **Performance Impact**: Minimize performance overhead of metrics collection
- **Memory Management**: Use efficient data structures and avoid memory leaks
- **Concurrency**: Ensure thread-safe metric collection in concurrent applications
- **Custom Collectors**: Implement custom collectors for complex business logic metrics
- **Testing**: Include metrics testing in application test suites
- **Documentation**: Document metric definitions and usage for team understanding
- **Monitoring**: Monitor the monitoring system itself for health and performance
- **Optimization**: Profile and optimize metrics collection for production workloads

**Q3: How do you implement effective alerting and incident response using Prometheus and Grafana?**

**Answer:** Effective alerting and incident response:

- **Alert Design**: Create clear, actionable alerts with appropriate thresholds and conditions
- **Alert Grouping**: Group related alerts to reduce noise and improve response efficiency
- **Escalation Policies**: Implement proper escalation procedures for different alert severities
- **Notification Channels**: Use multiple notification channels (email, Slack, PagerDuty) for redundancy
- **Runbook Integration**: Link alerts to runbooks and documentation for quick resolution
- **Alert Testing**: Regularly test alerting system to ensure reliability
- **Incident Management**: Integrate with incident management tools for tracking and resolution
- **Post-Incident Analysis**: Use monitoring data for post-incident analysis and improvement
- **Alert Fatigue Prevention**: Implement alert suppression and intelligent grouping
- **Business Impact**: Connect technical alerts to business impact for prioritization
- **Automation**: Implement automated responses for common issues where possible
- **Continuous Improvement**: Regularly review and improve alerting based on incident data

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
    cluster: "production"
    environment: "prod"

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "my-app"
    static_configs:
      - targets: ["my-app:8080"]
    metrics_path: "/metrics"
    scrape_interval: 5s
    scrape_timeout: 5s

  - job_name: "kubernetes-pods"
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
      - source_labels:
          [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
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

  - job_name: "kubernetes-nodes"
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

  - job_name: "kubernetes-cadvisor"
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

  - job_name: "kubernetes-service-endpoints"
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels:
          [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels:
          [__meta_kubernetes_service_annotation_prometheus_io_scheme]
        action: replace
        target_label: __scheme__
        regex: (https?)
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels:
          [__address__, __meta_kubernetes_service_annotation_prometheus_io_port]
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
  smtp_smarthost: "localhost:587"
  smtp_from: "alerts@example.com"
  smtp_auth_username: "alerts@example.com"
  smtp_auth_password: "password"

route:
  group_by: ["alertname"]
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: "web.hook"
  routes:
    - match:
        severity: critical
      receiver: "critical-alerts"
    - match:
        severity: warning
      receiver: "warning-alerts"

receivers:
  - name: "web.hook"
    webhook_configs:
      - url: "http://webhook:5001/"

  - name: "critical-alerts"
    email_configs:
      - to: "critical@example.com"
        subject: "CRITICAL: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#critical-alerts"
        title: "Critical Alert"
        text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"

  - name: "warning-alerts"
    email_configs:
      - to: "warnings@example.com"
        subject: "WARNING: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#warnings"
        title: "Warning Alert"
        text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
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
version: "3.8"

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
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"
      - "--storage.tsdb.retention.time=200h"
      - "--web.enable-lifecycle"

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - "--config.file=/etc/alertmanager/alertmanager.yml"
      - "--storage.path=/alertmanager"

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
      - "--path.procfs=/host/proc"
      - "--path.rootfs=/rootfs"
      - "--path.sysfs=/host/sys"
      - "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"

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

**Next**: [Tracing](Tracing.md/) - Distributed tracing, OpenTelemetry, Jaeger
