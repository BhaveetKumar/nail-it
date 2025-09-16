# Advanced Development Tools for Backend Engineers

## Table of Contents
- [Introduction](#introduction/)
- [Performance Profiling Tools](#performance-profiling-tools/)
- [Database Tools](#database-tools/)
- [Monitoring and Observability](#monitoring-and-observability/)
- [Testing Tools](#testing-tools/)
- [Security Tools](#security-tools/)
- [DevOps and Deployment](#devops-and-deployment/)
- [Code Quality and Analysis](#code-quality-and-analysis/)
- [API Development Tools](#api-development-tools/)
- [Cloud and Infrastructure](#cloud-and-infrastructure/)

## Introduction

Advanced development tools are essential for backend engineers to build, test, deploy, and maintain high-quality systems. This guide covers the most important tools across different categories that every senior backend engineer should know.

## Performance Profiling Tools

### Go Profiling Tools

#### pprof - Go's Built-in Profiler

```go
// CPU Profiling
import _ "net/http/pprof"
import "net/http"

func main() {
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // Your application code
}

// Memory Profiling
func profileMemory() {
    f, err := os.Create("mem.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    runtime.GC()
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal(err)
    }
}

// Goroutine Profiling
func profileGoroutines() {
    f, err := os.Create("goroutine.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.Lookup("goroutine").WriteTo(f, 0); err != nil {
        log.Fatal(err)
    }
}
```

#### Benchmarking and Performance Testing

```go
// Benchmarking
func BenchmarkDatabaseQuery(b *testing.B) {
    db := setupTestDB()
    defer db.Close()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := db.Query("SELECT * FROM users WHERE id = ?", i)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Memory Benchmarking
func BenchmarkMemoryAllocation(b *testing.B) {
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        data := make([]byte, 1024)
        _ = data
    }
}

// Custom Benchmark
func BenchmarkCustomFunction(b *testing.B) {
    b.Run("SmallData", func(b *testing.B) {
        data := generateData(100)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            processData(data)
        }
    })
    
    b.Run("LargeData", func(b *testing.B) {
        data := generateData(10000)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            processData(data)
        }
    })
}
```

### Node.js Profiling Tools

#### Clinic.js - Node.js Performance Profiler

```bash
# Install Clinic.js
npm install -g clinic

# CPU profiling
clinic doctor -- node app.js

# Memory profiling
clinic heapprofiler -- node app.js

# Event loop profiling
clinic bubbleprof -- node app.js

# Flame graph
clinic flame -- node app.js
```

#### Node.js Built-in Profiler

```javascript
// CPU Profiling
const v8 = require('v8');
const fs = require('fs');

// Start profiling
v8.setFlagsFromString('--prof');

// Your application code
setTimeout(() => {
    // Stop profiling
    v8.setFlagsFromString('--no-prof');
}, 10000);

// Memory Profiling
const { performance, PerformanceObserver } = require('perf_hooks');

const obs = new PerformanceObserver((list) => {
    console.log(list.getEntries());
});

obs.observe({ entryTypes: ['measure', 'mark'] });

// Mark start
performance.mark('start');

// Your code
setTimeout(() => {
    performance.mark('end');
    performance.measure('operation', 'start', 'end');
}, 1000);
```

## Database Tools

### Database Performance Analysis

#### PostgreSQL Tools

```sql
-- Query Analysis
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM users WHERE email = 'user@example.com';

-- Index Usage Analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Table Statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- Slow Query Log Analysis
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

#### Redis Performance Analysis

```bash
# Redis CLI Commands
redis-cli --latency-history -i 1
redis-cli --stat
redis-cli --bigkeys
redis-cli --memkeys

# Memory Analysis
redis-cli INFO memory
redis-cli MEMORY USAGE key_name
redis-cli MEMORY STATS

# Slow Log Analysis
redis-cli SLOWLOG GET 10
redis-cli SLOWLOG LEN
```

### Database Migration Tools

#### Go Migration Tool

```go
// Migration Tool
package main

import (
    "database/sql"
    "fmt"
    "io/ioutil"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
)

type Migration struct {
    Version int
    Name    string
    Up      string
    Down    string
}

type MigrationTool struct {
    db *sql.DB
}

func NewMigrationTool(db *sql.DB) *MigrationTool {
    return &MigrationTool{db: db}
}

func (mt *MigrationTool) CreateMigration(name string) error {
    version := mt.getNextVersion()
    
    upFile := fmt.Sprintf("%d_%s.up.sql", version, name)
    downFile := fmt.Sprintf("%d_%s.down.sql", version, name)
    
    // Create up migration
    upContent := fmt.Sprintf("-- Migration: %s\n-- Version: %d\n", name, version)
    if err := ioutil.WriteFile(upFile, []byte(upContent), 0644); err != nil {
        return err
    }
    
    // Create down migration
    downContent := fmt.Sprintf("-- Rollback: %s\n-- Version: %d\n", name, version)
    if err := ioutil.WriteFile(downFile, []byte(downContent), 0644); err != nil {
        return err
    }
    
    return nil
}

func (mt *MigrationTool) Migrate() error {
    if err := mt.createMigrationsTable(); err != nil {
        return err
    }
    
    migrations, err := mt.getPendingMigrations()
    if err != nil {
        return err
    }
    
    for _, migration := range migrations {
        if err := mt.runMigration(migration); err != nil {
            return err
        }
    }
    
    return nil
}

func (mt *MigrationTool) Rollback(steps int) error {
    migrations, err := mt.getAppliedMigrations()
    if err != nil {
        return err
    }
    
    for i := 0; i < steps && i < len(migrations); i++ {
        migration := migrations[len(migrations)-1-i]
        if err := mt.rollbackMigration(migration); err != nil {
            return err
        }
    }
    
    return nil
}
```

## Monitoring and Observability

### Prometheus and Grafana

#### Prometheus Metrics

```go
// Prometheus Metrics
package main

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // Counter metrics
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    // Gauge metrics
    activeConnections = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Number of active connections",
        },
    )
    
    // Histogram metrics
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
    
    // Summary metrics
    responseSize = promauto.NewSummaryVec(
        prometheus.SummaryOpts{
            Name: "http_response_size_bytes",
            Help: "HTTP response size in bytes",
        },
        []string{"method", "endpoint"},
    )
)

// Custom metrics
type CustomMetrics struct {
    requestsTotal    prometheus.Counter
    requestDuration  prometheus.Histogram
    activeUsers      prometheus.Gauge
}

func NewCustomMetrics() *CustomMetrics {
    return &CustomMetrics{
        requestsTotal: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "custom_requests_total",
            Help: "Total custom requests",
        }),
        requestDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name: "custom_request_duration_seconds",
            Help: "Custom request duration",
        }),
        activeUsers: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "custom_active_users",
            Help: "Number of active users",
        }),
    }
}
```

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Backend Service Dashboard",
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
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Distributed Tracing

#### OpenTelemetry Implementation

```go
// OpenTelemetry Tracing
package main

import (
    "context"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/trace"
    "go.opentelemetry.io/otel/trace"
)

func setupTracing() func() {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint("http://localhost:14268/api/traces")))
    if err != nil {
        panic(err)
    }
    
    // Create trace provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("backend-service"),
            semconv.ServiceVersionKey.String("1.0.0"),
        )),
    )
    
    otel.SetTracerProvider(tp)
    
    return func() {
        tp.Shutdown(context.Background())
    }
}

// Tracing middleware
func TracingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx, span := otel.Tracer("backend-service").Start(r.Context(), r.URL.Path)
        defer span.End()
        
        // Add attributes
        span.SetAttributes(
            attribute.String("http.method", r.Method),
            attribute.String("http.url", r.URL.String()),
            attribute.String("http.user_agent", r.UserAgent()),
        )
        
        // Process request
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Custom spans
func processOrder(ctx context.Context, orderID string) error {
    ctx, span := otel.Tracer("backend-service").Start(ctx, "processOrder")
    defer span.End()
    
    span.SetAttributes(attribute.String("order.id", orderID))
    
    // Process order logic
    if err := validateOrder(ctx, orderID); err != nil {
        span.RecordError(err)
        return err
    }
    
    if err := chargePayment(ctx, orderID); err != nil {
        span.RecordError(err)
        return err
    }
    
    return nil
}
```

## Testing Tools

### Load Testing

#### Artillery.js Load Testing

```yaml
# artillery-config.yml
config:
  target: 'http://localhost:3000'
  phases:
    - duration: 60
      arrivalRate: 10
    - duration: 120
      arrivalRate: 20
    - duration: 60
      arrivalRate: 10
  defaults:
    headers:
      Content-Type: 'application/json'

scenarios:
  - name: "API Load Test"
    weight: 100
    flow:
      - post:
          url: "/api/users"
          json:
            name: "{{ $randomString() }}"
            email: "{{ $randomEmail() }}"
      - get:
          url: "/api/users/{{ userId }}"
      - put:
          url: "/api/users/{{ userId }}"
          json:
            name: "Updated Name"
      - delete:
          url: "/api/users/{{ userId }}"
```

#### Go Load Testing

```go
// Load Testing Tool
package main

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type LoadTest struct {
    URL         string
    Concurrency int
    Duration    time.Duration
    Requests    int
}

func NewLoadTest(url string, concurrency int, duration time.Duration) *LoadTest {
    return &LoadTest{
        URL:         url,
        Concurrency: concurrency,
        Duration:    duration,
    }
}

func (lt *LoadTest) Run() *LoadTestResults {
    results := &LoadTestResults{
        StartTime: time.Now(),
        Requests:  make([]RequestResult, 0),
    }
    
    var wg sync.WaitGroup
    var mu sync.Mutex
    
    // Create worker pool
    for i := 0; i < lt.Concurrency; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            
            client := &http.Client{
                Timeout: time.Second * 30,
            }
            
            for time.Since(results.StartTime) < lt.Duration {
                start := time.Now()
                
                resp, err := client.Get(lt.URL)
                duration := time.Since(start)
                
                result := RequestResult{
                    Duration: duration,
                    Status:   resp.StatusCode,
                    Error:    err,
                }
                
                if resp != nil {
                    resp.Body.Close()
                }
                
                mu.Lock()
                results.Requests = append(results.Requests, result)
                mu.Unlock()
            }
        }()
    }
    
    wg.Wait()
    results.EndTime = time.Now()
    results.CalculateStats()
    
    return results
}

type LoadTestResults struct {
    StartTime    time.Time
    EndTime      time.Time
    Requests     []RequestResult
    TotalRequests int
    SuccessCount  int
    ErrorCount    int
    AvgDuration   time.Duration
    MinDuration   time.Duration
    MaxDuration   time.Duration
    RPS          float64
}

type RequestResult struct {
    Duration time.Duration
    Status   int
    Error    error
}

func (ltr *LoadTestResults) CalculateStats() {
    ltr.TotalRequests = len(ltr.Requests)
    
    var totalDuration time.Duration
    var successCount int
    var errorCount int
    
    for _, req := range ltr.Requests {
        totalDuration += req.Duration
        
        if req.Error != nil || req.Status >= 400 {
            errorCount++
        } else {
            successCount++
        }
    }
    
    ltr.SuccessCount = successCount
    ltr.ErrorCount = errorCount
    
    if ltr.TotalRequests > 0 {
        ltr.AvgDuration = totalDuration / time.Duration(ltr.TotalRequests)
        ltr.RPS = float64(ltr.TotalRequests) / ltr.EndTime.Sub(ltr.StartTime).Seconds()
    }
    
    // Find min/max duration
    if ltr.TotalRequests > 0 {
        ltr.MinDuration = ltr.Requests[0].Duration
        ltr.MaxDuration = ltr.Requests[0].Duration
        
        for _, req := range ltr.Requests {
            if req.Duration < ltr.MinDuration {
                ltr.MinDuration = req.Duration
            }
            if req.Duration > ltr.MaxDuration {
                ltr.MaxDuration = req.Duration
            }
        }
    }
}
```

### API Testing

#### Postman/Newman Testing

```json
{
  "info": {
    "name": "Backend API Tests",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "User Management",
      "item": [
        {
          "name": "Create User",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n  \"name\": \"John Doe\",\n  \"email\": \"john@example.com\"\n}"
            },
            "url": {
              "raw": "{{base_url}}/api/users",
              "host": ["{{base_url}}"],
              "path": ["api", "users"]
            }
          },
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test(\"Status code is 201\", function () {",
                  "    pm.response.to.have.status(201);",
                  "});",
                  "",
                  "pm.test(\"Response has user ID\", function () {",
                  "    var jsonData = pm.response.json();",
                  "    pm.expect(jsonData).to.have.property('id');",
                  "    pm.globals.set('userId', jsonData.id);",
                  "});"
                ]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## Security Tools

### Vulnerability Scanning

#### OWASP ZAP Integration

```go
// Security Scanner
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"
)

type SecurityScanner struct {
    zapURL    string
    targetURL string
    client    *http.Client
}

func NewSecurityScanner(zapURL, targetURL string) *SecurityScanner {
    return &SecurityScanner{
        zapURL:    zapURL,
        targetURL: targetURL,
        client:    &http.Client{Timeout: time.Second * 30},
    }
}

func (ss *SecurityScanner) RunScan() (*SecurityReport, error) {
    // Start spider scan
    if err := ss.startSpiderScan(); err != nil {
        return nil, err
    }
    
    // Wait for spider to complete
    if err := ss.waitForSpiderComplete(); err != nil {
        return nil, err
    }
    
    // Start active scan
    if err := ss.startActiveScan(); err != nil {
        return nil, err
    }
    
    // Wait for active scan to complete
    if err := ss.waitForActiveScanComplete(); err != nil {
        return nil, err
    }
    
    // Generate report
    return ss.generateReport()
}

func (ss *SecurityScanner) startSpiderScan() error {
    url := fmt.Sprintf("%s/JSON/spider/action/scan/?url=%s", ss.zapURL, ss.targetURL)
    resp, err := ss.client.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        return fmt.Errorf("failed to start spider scan: %d", resp.StatusCode)
    }
    
    return nil
}

func (ss *SecurityScanner) waitForSpiderComplete() error {
    for {
        url := fmt.Sprintf("%s/JSON/spider/view/status/", ss.zapURL)
        resp, err := ss.client.Get(url)
        if err != nil {
            return err
        }
        defer resp.Body.Close()
        
        // Parse response to check status
        // Implementation depends on ZAP API response format
        
        time.Sleep(time.Second * 5)
    }
}
```

### Dependency Scanning

#### Go Security Scanning

```bash
# Install security tools
go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run security scan
gosec ./...

# Run comprehensive linting
golangci-lint run

# Check for vulnerabilities
go list -json -m all | nancy sleuth
```

#### Node.js Security Scanning

```bash
# Install security tools
npm install -g audit-ci
npm install -g retire

# Run security audit
npm audit
npm audit --audit-level moderate

# Check for outdated packages
npm outdated

# Run retire.js
retire --path . --outputformat json
```

## DevOps and Deployment

### Docker Optimization

#### Multi-stage Dockerfile

```dockerfile
# Multi-stage Dockerfile for Go application
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .
COPY --from=builder /app/config ./config

EXPOSE 8080
CMD ["./main"]
```

#### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
    command: go run main.go

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Kubernetes Manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend-service
  template:
    metadata:
      labels:
        app: backend-service
    spec:
      containers:
      - name: backend
        image: backend-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend-service
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Code Quality and Analysis

### Static Analysis Tools

#### Go Static Analysis

```bash
# Install static analysis tools
go install honnef.co/go/tools/cmd/staticcheck@latest
go install github.com/gordonklaus/ineffassign@latest
go install github.com/client9/misspell/cmd/misspell@latest

# Run static analysis
staticcheck ./...
ineffassign ./...
misspell ./...

# Run all checks
golangci-lint run --enable-all
```

#### Code Coverage

```go
// Code Coverage Testing
package main

import (
    "testing"
    "os"
    "os/exec"
)

func TestMain(m *testing.M) {
    // Run tests with coverage
    cmd := exec.Command("go", "test", "-coverprofile=coverage.out", "./...")
    cmd.Run()
    
    // Generate HTML coverage report
    exec.Command("go", "tool", "cover", "-html=coverage.out", "-o=coverage.html").Run()
    
    os.Exit(m.Run())
}

func TestWithCoverage(t *testing.T) {
    // Your tests here
    result := calculateSum(2, 3)
    if result != 5 {
        t.Errorf("Expected 5, got %d", result)
    }
}
```

### Code Review Tools

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: go-fmt
        name: go-fmt
        entry: gofmt
        args: [-w]
        language: system
        files: \.go$
      
      - id: go-vet
        name: go-vet
        entry: go
        args: [vet, ./...]
        language: system
        files: \.go$
      
      - id: go-test
        name: go-test
        entry: go
        args: [test, ./...]
        language: system
        files: \.go$
      
      - id: golangci-lint
        name: golangci-lint
        entry: golangci-lint
        args: [run]
        language: system
        files: \.go$
```

## API Development Tools

### API Documentation

#### Swagger/OpenAPI

```go
// Swagger Documentation
package main

import (
    "github.com/swaggo/swag"
    "github.com/swaggo/gin-swagger"
    "github.com/swaggo/files"
)

// @title Backend API
// @version 1.0
// @description This is a sample backend API
// @host localhost:8080
// @BasePath /api/v1

// @Summary Get user by ID
// @Description Get user information by user ID
// @Tags users
// @Accept json
// @Produce json
// @Param id path int true "User ID"
// @Success 200 {object} User
// @Failure 400 {object} ErrorResponse
// @Failure 404 {object} ErrorResponse
// @Router /users/{id} [get]
func GetUser(c *gin.Context) {
    // Implementation
}

// @Summary Create user
// @Description Create a new user
// @Tags users
// @Accept json
// @Produce json
// @Param user body CreateUserRequest true "User data"
// @Success 201 {object} User
// @Failure 400 {object} ErrorResponse
// @Router /users [post]
func CreateUser(c *gin.Context) {
    // Implementation
}
```

### API Testing

#### Postman Collections

```json
{
  "info": {
    "name": "Backend API Tests",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8080"
    }
  ],
  "item": [
    {
      "name": "Authentication",
      "item": [
        {
          "name": "Login",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n  \"email\": \"user@example.com\",\n  \"password\": \"password123\"\n}"
            },
            "url": {
              "raw": "{{base_url}}/api/auth/login",
              "host": ["{{base_url}}"],
              "path": ["api", "auth", "login"]
            }
          },
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test(\"Status code is 200\", function () {",
                  "    pm.response.to.have.status(200);",
                  "});",
                  "",
                  "pm.test(\"Response has token\", function () {",
                  "    var jsonData = pm.response.json();",
                  "    pm.expect(jsonData).to.have.property('token');",
                  "    pm.globals.set('authToken', jsonData.token);",
                  "});"
                ]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## Cloud and Infrastructure

### AWS Tools

#### AWS CLI Configuration

```bash
# Configure AWS CLI
aws configure

# List S3 buckets
aws s3 ls

# Create S3 bucket
aws s3 mb s3://my-bucket

# Upload file to S3
aws s3 cp file.txt s3://my-bucket/

# List EC2 instances
aws ec2 describe-instances

# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier mydb \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --master-user-password password123 \
    --allocated-storage 20
```

#### Terraform Infrastructure

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  
  tags = {
    Name = "public-subnet"
  }
}

resource "aws_security_group" "web" {
  name_prefix = "web-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1d0"
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.public.id
  security_groups = [aws_security_group.web.id]
  
  tags = {
    Name = "web-server"
  }
}
```

## Conclusion

These advanced development tools are essential for backend engineers to build, test, deploy, and maintain high-quality systems. The key is to:

1. **Choose the right tools** for your specific needs and technology stack
2. **Integrate tools** into your development workflow
3. **Automate** as much as possible
4. **Monitor and measure** the effectiveness of your tools
5. **Stay updated** with new tools and best practices

Mastering these tools will significantly improve your productivity and the quality of your backend systems.

## Additional Resources

- [Go Tools](https://golang.org/cmd/go/#hdr-List_of_tools/)
- [Node.js Tools](https://nodejs.org/en/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/latest/reference/)
- [Terraform Documentation](https://www.terraform.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
