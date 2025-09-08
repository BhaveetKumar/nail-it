# 📝 Logging: Centralized Log Management and Analysis

> **Master logging strategies, log aggregation, and log analysis for production systems**

## 📚 Concept

Logging is the practice of recording events, errors, and activities in applications and infrastructure. Effective logging provides visibility into system behavior, helps with debugging, and enables monitoring and alerting.

### Key Features
- **Structured Logging**: JSON-formatted logs
- **Log Levels**: DEBUG, INFO, WARN, ERROR, FATAL
- **Log Aggregation**: Centralized log collection
- **Log Analysis**: Search and analysis tools
- **Log Retention**: Storage and archival policies
- **Real-time Processing**: Stream processing

## 🏗️ Logging Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Logging Pipeline                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Application │  │  System     │  │  Security   │     │
│  │    Logs     │  │   Logs      │  │   Logs      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Log Collectors                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Fluentd   │  │  Filebeat   │  │   Logstash  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Log Storage                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Elastic   │  │   Splunk    │  │   CloudWatch│ │ │
│  │  │   Search    │  │   Enterprise│  │   Logs      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Kibana    │  │   Grafana   │  │   CloudWatch│     │
│  │ (Visualization)│  │ (Dashboards)│  │ (Insights) │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Go Application Logging

```go
// main.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/sirupsen/logrus"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

// Structured logging with logrus
func setupLogrus() *logrus.Logger {
    logger := logrus.New()
    
    // Set log level
    level, err := logrus.ParseLevel(os.Getenv("LOG_LEVEL"))
    if err != nil {
        level = logrus.InfoLevel
    }
    logger.SetLevel(level)
    
    // Set JSON formatter
    logger.SetFormatter(&logrus.JSONFormatter{
        TimestampFormat: time.RFC3339,
        FieldMap: logrus.FieldMap{
            logrus.FieldKeyTime:  "timestamp",
            logrus.FieldKeyLevel: "level",
            logrus.FieldKeyMsg:   "message",
            logrus.FieldKeyFunc:  "function",
        },
    })
    
    // Add fields
    logger = logger.WithFields(logrus.Fields{
        "service": "my-app",
        "version": "1.0.0",
        "environment": os.Getenv("ENVIRONMENT"),
    })
    
    return logger
}

// Structured logging with zap
func setupZap() *zap.Logger {
    config := zap.NewProductionConfig()
    
    // Set log level
    level := os.Getenv("LOG_LEVEL")
    if level != "" {
        if l, err := zapcore.ParseLevel(level); err == nil {
            config.Level = zap.NewAtomicLevelAt(l)
        }
    }
    
    // Configure encoder
    config.EncoderConfig.TimeKey = "timestamp"
    config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
    config.EncoderConfig.LevelKey = "level"
    config.EncoderConfig.MessageKey = "message"
    config.EncoderConfig.CallerKey = "caller"
    
    // Add fields
    config.InitialFields = map[string]interface{}{
        "service": "my-app",
        "version": "1.0.0",
        "environment": os.Getenv("ENVIRONMENT"),
    }
    
    logger, err := config.Build()
    if err != nil {
        log.Fatal("Failed to create logger:", err)
    }
    
    return logger
}

// Logging middleware
func loggingMiddleware(logger *zap.Logger) gin.HandlerFunc {
    return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
        logger.Info("HTTP Request",
            zap.String("method", param.Method),
            zap.String("path", param.Path),
            zap.Int("status", param.StatusCode),
            zap.Duration("latency", param.Latency),
            zap.String("client_ip", param.ClientIP),
            zap.String("user_agent", param.Request.UserAgent()),
        )
        return ""
    })
}

// Request context with logger
func requestLoggerMiddleware(logger *zap.Logger) gin.HandlerFunc {
    return func(c *gin.Context) {
        // Create request-scoped logger
        requestLogger := logger.With(
            zap.String("request_id", c.GetHeader("X-Request-ID")),
            zap.String("user_id", c.GetHeader("X-User-ID")),
        )
        
        // Add logger to context
        ctx := context.WithValue(c.Request.Context(), "logger", requestLogger)
        c.Request = c.Request.WithContext(ctx)
        
        c.Next()
    }
}

// Business logic with logging
type UserService struct {
    logger *zap.Logger
}

func NewUserService(logger *zap.Logger) *UserService {
    return &UserService{logger: logger}
}

func (s *UserService) GetUser(ctx context.Context, userID string) (*User, error) {
    logger := s.logger.With(zap.String("user_id", userID))
    
    logger.Info("Getting user")
    
    // Simulate database call
    user, err := s.fetchUserFromDB(ctx, userID)
    if err != nil {
        logger.Error("Failed to get user",
            zap.Error(err),
            zap.String("operation", "fetchUserFromDB"),
        )
        return nil, err
    }
    
    logger.Info("User retrieved successfully",
        zap.String("user_email", user.Email),
        zap.String("user_role", user.Role),
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

// HTTP handlers
func setupRoutes(logger *zap.Logger) *gin.Engine {
    r := gin.New()
    
    // Add middleware
    r.Use(loggingMiddleware(logger))
    r.Use(requestLoggerMiddleware(logger))
    r.Use(gin.Recovery())
    
    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "timestamp": time.Now().UTC(),
        })
    })
    
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
    logger := setupZap()
    defer logger.Sync()
    
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

### Fluentd Configuration

```yaml
# fluentd.conf
<source>
  @type tail
  @id input_tail
  path /var/log/app/*.log
  pos_file /var/log/fluentd/app.log.pos
  tag app.logs
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%L%z
</source>

<source>
  @type tail
  @id input_system
  path /var/log/syslog
  pos_file /var/log/fluentd/syslog.pos
  tag system.logs
  format syslog
</source>

<filter app.logs>
  @type record_transformer
  <record>
    service_name "my-app"
    environment "#{ENV['ENVIRONMENT'] || 'development'}"
    hostname "#{Socket.gethostname}"
  </record>
</filter>

<filter system.logs>
  @type record_transformer
  <record>
    service_name "system"
    environment "#{ENV['ENVIRONMENT'] || 'development'}"
    hostname "#{Socket.gethostname}"
  </record>
</filter>

<match app.logs>
  @type elasticsearch
  @id output_elasticsearch
  @log_level info
  include_tag_key true
  host "#{ENV['ELASTICSEARCH_HOST'] || 'localhost'}"
  port "#{ENV['ELASTICSEARCH_PORT'] || '9200'}"
  path ""
  scheme https
  ssl_verify false
  user "#{ENV['ELASTICSEARCH_USER']}"
  password "#{ENV['ELASTICSEARCH_PASSWORD']}"
  index_name app-logs
  type_name _doc
  <buffer>
    @type file
    path /var/log/fluentd/buffers/app
    flush_mode interval
    retry_type exponential_backoff
    flush_thread_count 2
    flush_interval 5s
    retry_forever
    retry_max_interval 30
    chunk_limit_size 2M
    queue_limit_length 8
    overflow_action block
  </buffer>
</match>

<match system.logs>
  @type elasticsearch
  @id output_elasticsearch_system
  @log_level info
  include_tag_key true
  host "#{ENV['ELASTICSEARCH_HOST'] || 'localhost'}"
  port "#{ENV['ELASTICSEARCH_PORT'] || '9200'}"
  path ""
  scheme https
  ssl_verify false
  user "#{ENV['ELASTICSEARCH_USER']}"
  password "#{ENV['ELASTICSEARCH_PASSWORD']}"
  index_name system-logs
  type_name _doc
  <buffer>
    @type file
    path /var/log/fluentd/buffers/system
    flush_mode interval
    retry_type exponential_backoff
    flush_thread_count 2
    flush_interval 5s
    retry_forever
    retry_max_interval 30
    chunk_limit_size 2M
    queue_limit_length 8
    overflow_action block
  </buffer>
</match>
```

### Filebeat Configuration

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/app/*.log
  fields:
    service: my-app
    environment: production
  fields_under_root: true
  multiline.pattern: '^\d{4}-\d{2}-\d{2}'
  multiline.negate: true
  multiline.match: after
  processors:
    - add_host_metadata:
        when.not.contains.tags: forwarded
    - add_docker_metadata: ~
    - add_kubernetes_metadata: ~

- type: log
  enabled: true
  paths:
    - /var/log/nginx/*.log
  fields:
    service: nginx
    environment: production
  fields_under_root: true
  processors:
    - add_host_metadata:
        when.not.contains.tags: forwarded

output.elasticsearch:
  hosts: ["${ELASTICSEARCH_HOST:localhost:9200}"]
  username: "${ELASTICSEARCH_USERNAME:}"
  password: "${ELASTICSEARCH_PASSWORD:}"
  index: "filebeat-%{+yyyy.MM.dd}"
  template.name: "filebeat"
  template.pattern: "filebeat-*"

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
  - add_docker_metadata: ~
  - add_kubernetes_metadata: ~

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

### Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
  
  udp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [service] == "my-app" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      convert => { "level" => "string" }
      lowercase => [ "level" ]
    }
    
    if [level] == "error" or [level] == "fatal" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
  
  if [service] == "nginx" {
    grok {
      match => { "message" => "%{NGINXACCESS}" }
    }
    
    date {
      match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
    
    mutate {
      convert => { "response" => "integer" }
      convert => { "bytes" => "integer" }
    }
    
    if [response] >= 400 {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
  
  # Add common fields
  mutate {
    add_field => { "hostname" => "%{[host][name]}" }
    add_field => { "environment" => "%{[fields][environment]}" }
  }
}

output {
  elasticsearch {
    hosts => ["${ELASTICSEARCH_HOST:localhost:9200}"]
    user => "${ELASTICSEARCH_USERNAME:}"
    password => "${ELASTICSEARCH_PASSWORD:}"
    index => "logs-%{+YYYY.MM.dd}"
  }
  
  if "error" in [tags] {
    email {
      to => "alerts@example.com"
      subject => "Error Alert: %{service} - %{level}"
      body => "Error detected in %{service}: %{message}"
    }
  }
}
```

### Docker Logging Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    image: my-app:latest
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=info
      - ENVIRONMENT=production
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    volumes:
      - ./logs:/var/log/app

  fluentd:
    image: fluent/fluentd:v1.14-1
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - ./logs:/var/log/app
      - /var/log:/var/log
    environment:
      - ELASTICSEARCH_HOST=elasticsearch:9200
      - ELASTICSEARCH_USERNAME=elastic
      - ELASTICSEARCH_PASSWORD=changeme
    depends_on:
      - elasticsearch

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=changeme
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
      - ELASTICSEARCH_USERNAME=elastic
      - ELASTICSEARCH_PASSWORD=changeme
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Kubernetes Logging Configuration

```yaml
# fluentd-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
  labels:
    app: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.14-debian-elasticsearch7-1
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        - name: FLUENT_ELASTICSEARCH_SCHEME
          value: "https"
        - name: FLUENT_ELASTICSEARCH_USER
          value: "elastic"
        - name: FLUENT_ELASTICSEARCH_PASSWORD
          valueFrom:
            secretKeyRef:
              name: elasticsearch-secret
              key: password
        - name: FLUENT_ELASTICSEARCH_SSL_VERIFY
          value: "false"
        - name: FLUENT_ELASTICSEARCH_SSL_VERSION
          value: "TLSv1_2"
        resources:
          limits:
            memory: 512Mi
            cpu: 100m
          requests:
            memory: 256Mi
            cpu: 100m
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentdconf
          mountPath: /fluentd/etc
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentdconf
        configMap:
          name: fluentd-config
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      - operator: Exists
        effect: NoExecute
      - operator: Exists
        effect: NoSchedule
```

## 🚀 Best Practices

### 1. Structured Logging
```go
// Use structured logging with consistent fields
logger.Info("User login",
    zap.String("user_id", userID),
    zap.String("ip_address", ip),
    zap.String("user_agent", userAgent),
    zap.Duration("duration", duration),
)
```

### 2. Log Levels
```go
// Use appropriate log levels
logger.Debug("Detailed debug information")
logger.Info("General information")
logger.Warn("Warning conditions")
logger.Error("Error conditions")
logger.Fatal("Fatal conditions")
```

### 3. Log Rotation
```yaml
# Configure log rotation
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 🏢 Industry Insights

### Logging Usage Patterns
- **Application Logs**: Business logic and errors
- **System Logs**: OS and infrastructure events
- **Security Logs**: Authentication and authorization
- **Audit Logs**: Compliance and governance

### Enterprise Logging Strategy
- **Centralized Collection**: ELK stack, Splunk
- **Log Retention**: Compliance requirements
- **Real-time Processing**: Stream processing
- **Security**: Log encryption and access control

## 🎯 Interview Questions

### Basic Level
1. **What is structured logging?**
   - JSON-formatted logs
   - Consistent field structure
   - Machine-readable format
   - Easy parsing and analysis

2. **What are log levels?**
   - DEBUG: Detailed information
   - INFO: General information
   - WARN: Warning conditions
   - ERROR: Error conditions
   - FATAL: Fatal conditions

3. **What is log aggregation?**
   - Centralized log collection
   - Multiple sources
   - Unified storage
   - Search and analysis

### Intermediate Level
4. **How do you implement structured logging?**
   ```go
   logger.Info("User action",
       zap.String("user_id", userID),
       zap.String("action", action),
       zap.Duration("duration", duration),
   )
   ```

5. **How do you handle log rotation?**
   - Size-based rotation
   - Time-based rotation
   - Compression
   - Retention policies

6. **How do you implement log aggregation?**
   - Fluentd/Filebeat collection
   - Elasticsearch storage
   - Kibana visualization
   - Real-time processing

### Advanced Level
7. **How do you implement log analysis?**
   - Pattern recognition
   - Anomaly detection
   - Trend analysis
   - Machine learning

8. **How do you handle log security?**
   - Encryption in transit
   - Encryption at rest
   - Access control
   - Audit trails

9. **How do you implement log monitoring?**
   - Real-time alerts
   - Dashboard creation
   - SLA monitoring
   - Performance metrics

---

**Next**: [Monitoring with Prometheus and Grafana](./MonitoringPrometheusGrafana.md) - Metrics collection, visualization, alerting
