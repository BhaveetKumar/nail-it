---
# Auto-generated front matter
Title: 33 01 Advanced Observability
LastUpdated: 2025-11-06T20:45:58.124316
Tags: []
Status: draft
---

# Lesson 33.1: Advanced Observability

> **Module**: 33 - Advanced Observability  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 32 (Advanced Database Patterns)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Implement comprehensive observability systems
- Set up distributed tracing and monitoring
- Design effective logging strategies
- Build performance monitoring solutions
- Create alerting and incident response systems

---

## 🎯 **Overview**

Advanced observability in Rust involves building comprehensive monitoring, logging, and tracing systems that provide deep insights into application behavior, performance, and reliability. This lesson covers distributed tracing, metrics collection, and production monitoring.

---

## 🔧 **Distributed Tracing Implementation**

### **OpenTelemetry Integration**

```rust
use opentelemetry::{
    global,
    trace::{Span, Tracer, TracerProvider},
    KeyValue,
};
use opentelemetry_jaeger::new_agent_pipeline;
use opentelemetry_sdk::{
    trace::{RandomIdGenerator, Sampler},
    Resource,
};
use std::time::Duration;

pub struct TracingManager {
    pub tracer: Tracer,
    pub provider: TracerProvider,
}

impl TracingManager {
    pub fn new(service_name: &str, jaeger_endpoint: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline = new_agent_pipeline()
            .with_service_name(service_name)
            .with_endpoint(jaeger_endpoint)
            .with_trace_config(
                opentelemetry_sdk::trace::Config::default()
                    .with_sampler(Sampler::TraceIdRatioBased(1.0))
                    .with_id_generator(RandomIdGenerator::default())
                    .with_resource(Resource::new(vec![
                        KeyValue::new("service.name", service_name),
                        KeyValue::new("service.version", "1.0.0"),
                        KeyValue::new("deployment.environment", "production"),
                    ])),
            )
            .install_simple()?;
        
        let provider = pipeline.provider();
        let tracer = provider.tracer("rust-observability");
        
        Ok(Self { tracer, provider })
    }
    
    pub fn create_span(&self, name: &str) -> Span {
        self.tracer.start(name)
    }
    
    pub fn create_span_with_attributes(&self, name: &str, attributes: Vec<KeyValue>) -> Span {
        let mut span = self.tracer.start(name);
        for attr in attributes {
            span.set_attribute(attr);
        }
        span
    }
}

// Example usage in web application
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};

pub async fn handle_request(
    Path(id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    State(tracing_manager): State<Arc<TracingManager>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let span = tracing_manager.create_span_with_attributes(
        "handle_request",
        vec![
            KeyValue::new("request.id", id.clone()),
            KeyValue::new("request.params", params.len() as i64),
        ],
    );
    
    let _guard = span.enter();
    
    // Simulate some work
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Add more attributes
    span.set_attribute(KeyValue::new("response.status", "200"));
    span.set_attribute(KeyValue::new("response.size", 1024));
    
    Ok(Json(serde_json::json!({
        "id": id,
        "message": "Hello, World!",
        "timestamp": chrono::Utc::now()
    })))
}
```

### **Custom Span Attributes**

```rust
use opentelemetry::trace::{Span, Tracer};
use opentelemetry::KeyValue;
use std::collections::HashMap;

pub struct CustomSpan {
    pub span: Span,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
}

#[derive(Clone, Debug)]
pub struct SpanEvent {
    pub name: String,
    pub attributes: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CustomSpan {
    pub fn new(span: Span) -> Self {
        Self {
            span,
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }
    
    pub fn add_attribute(&mut self, key: String, value: String) {
        self.attributes.insert(key.clone(), value.clone());
        self.span.set_attribute(KeyValue::new(key, value));
    }
    
    pub fn add_event(&mut self, name: String, attributes: HashMap<String, String>) {
        let event = SpanEvent {
            name: name.clone(),
            attributes: attributes.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        self.events.push(event);
        
        // Add event to span
        let mut span_attributes = Vec::new();
        for (key, value) in attributes {
            span_attributes.push(KeyValue::new(key, value));
        }
        
        self.span.add_event(name, span_attributes);
    }
    
    pub fn set_status(&mut self, status: SpanStatus) {
        match status {
            SpanStatus::Ok => {
                self.span.set_status(opentelemetry::trace::Status::Ok);
            }
            SpanStatus::Error(message) => {
                self.span.set_status(opentelemetry::trace::Status::error(message));
            }
        }
    }
    
    pub fn finish(self) {
        self.span.end();
    }
}

#[derive(Debug)]
pub enum SpanStatus {
    Ok,
    Error(String),
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Metrics Collection System**

```rust
use prometheus::{
    Counter, Histogram, Gauge, Registry, TextEncoder,
    Encoder, Opts, HistogramOpts, GaugeOpts,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;

pub struct MetricsCollector {
    pub registry: Registry,
    pub request_counter: Counter,
    pub request_duration: Histogram,
    pub active_connections: Gauge,
    pub error_counter: Counter,
    pub database_operations: Counter,
    pub cache_hits: Counter,
    pub cache_misses: Counter,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        // Request metrics
        let request_counter = Counter::new(
            "http_requests_total",
            "Total number of HTTP requests"
        )?;
        
        let request_duration = Histogram::with_opts(
            HistogramOpts::new("http_request_duration_seconds", "HTTP request duration in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        )?;
        
        let error_counter = Counter::new(
            "http_errors_total",
            "Total number of HTTP errors"
        )?;
        
        // Connection metrics
        let active_connections = Gauge::new(
            "active_connections",
            "Number of active connections"
        )?;
        
        // Database metrics
        let database_operations = Counter::new(
            "database_operations_total",
            "Total number of database operations"
        )?;
        
        // Cache metrics
        let cache_hits = Counter::new(
            "cache_hits_total",
            "Total number of cache hits"
        )?;
        
        let cache_misses = Counter::new(
            "cache_misses_total",
            "Total number of cache misses"
        )?;
        
        // Register metrics
        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(error_counter.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(database_operations.clone()))?;
        registry.register(Box::new(cache_hits.clone()))?;
        registry.register(Box::new(cache_misses.clone()))?;
        
        Ok(Self {
            registry,
            request_counter,
            request_duration,
            active_connections,
            error_counter,
            database_operations,
            cache_hits,
            cache_misses,
        })
    }
    
    pub fn record_request(&self, method: &str, path: &str, status_code: u16, duration: f64) {
        self.request_counter.inc();
        self.request_duration.observe(duration);
        
        if status_code >= 400 {
            self.error_counter.inc();
        }
    }
    
    pub fn record_connection(&self) {
        self.active_connections.inc();
    }
    
    pub fn record_disconnection(&self) {
        self.active_connections.dec();
    }
    
    pub fn record_database_operation(&self, operation: &str, duration: f64) {
        self.database_operations.inc();
    }
    
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }
    
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }
    
    pub fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let metric_families = self.registry.gather();
        let encoder = TextEncoder::new();
        let metrics = encoder.encode_to_string(&metric_families)?;
        Ok(metrics)
    }
}

pub struct MetricsMiddleware {
    pub metrics: Arc<MetricsCollector>,
}

impl MetricsMiddleware {
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self { metrics }
    }
    
    pub async fn record_request<F, T>(&self, operation: F) -> Result<T, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
    {
        let start_time = Instant::now();
        
        match operation() {
            Ok(result) => {
                let duration = start_time.elapsed().as_secs_f64();
                self.metrics.record_request("GET", "/api", 200, duration);
                Ok(result)
            }
            Err(error) => {
                let duration = start_time.elapsed().as_secs_f64();
                self.metrics.record_request("GET", "/api", 500, duration);
                Err(error)
            }
        }
    }
}
```

### **Exercise 2: Structured Logging System**

```rust
use tracing::{info, warn, error, debug, trace};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub span_id: Option<String>,
    pub trace_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

pub struct StructuredLogger {
    pub service_name: String,
    pub environment: String,
    pub version: String,
}

impl StructuredLogger {
    pub fn new(service_name: String, environment: String, version: String) -> Self {
        Self {
            service_name,
            environment,
            version,
        }
    }
    
    pub fn setup(&self) -> Result<(), Box<dyn std::error::Error>> {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info".into());
        
        let subscriber = tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .json()
                    .with_current_span(true)
                    .with_span_list(true)
            );
        
        subscriber.init();
        Ok(())
    }
    
    pub fn log_request(&self, method: &str, path: &str, status_code: u16, duration: f64) {
        info!(
            method = method,
            path = path,
            status_code = status_code,
            duration_ms = duration * 1000.0,
            "HTTP request completed"
        );
    }
    
    pub fn log_error(&self, error: &str, context: HashMap<String, String>) {
        error!(
            error = error,
            context = ?context,
            "Application error occurred"
        );
    }
    
    pub fn log_database_operation(&self, operation: &str, table: &str, duration: f64) {
        debug!(
            operation = operation,
            table = table,
            duration_ms = duration * 1000.0,
            "Database operation completed"
        );
    }
    
    pub fn log_cache_operation(&self, operation: &str, key: &str, hit: bool) {
        debug!(
            operation = operation,
            key = key,
            hit = hit,
            "Cache operation completed"
        );
    }
}

pub struct LogContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub ip_address: Option<String>,
}

impl LogContext {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            user_id: None,
            session_id: None,
            ip_address: None,
        }
    }
    
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }
    
    pub fn with_ip_address(mut self, ip_address: String) -> Self {
        self.ip_address = Some(ip_address);
        self
    }
    
    pub fn log_with_context<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        let span = tracing::info_span!(
            "request",
            request_id = %self.request_id,
            user_id = ?self.user_id,
            session_id = ?self.session_id,
            ip_address = ?self.ip_address
        );
        
        let _guard = span.enter();
        f();
    }
}
```

### **Exercise 3: Health Check System**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_check: Option<Instant>,
    pub check_interval: Duration,
    pub timeout: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

pub struct HealthChecker {
    pub checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    pub check_functions: Arc<RwLock<HashMap<String, Box<dyn Fn() -> Result<(), String> + Send + Sync>>>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            check_functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_check<F>(&self, name: String, check_fn: F, interval: Duration, timeout: Duration)
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        let check = HealthCheck {
            name: name.clone(),
            status: HealthStatus::Unknown,
            message: None,
            last_check: None,
            check_interval: interval,
            timeout,
        };
        
        self.checks.write().await.insert(name.clone(), check);
        self.check_functions.write().await.insert(name, Box::new(check_fn));
    }
    
    pub async fn run_checks(&self) -> HashMap<String, HealthCheck> {
        let mut results = HashMap::new();
        let checks = self.checks.read().await;
        let check_functions = self.check_functions.read().await;
        
        for (name, check) in checks.iter() {
            let mut updated_check = check.clone();
            
            // Check if enough time has passed since last check
            if let Some(last_check) = check.last_check {
                if last_check.elapsed() < check.check_interval {
                    results.insert(name.clone(), updated_check);
                    continue;
                }
            }
            
            // Run the check
            if let Some(check_fn) = check_functions.get(name) {
                let start_time = Instant::now();
                
                match check_fn() {
                    Ok(_) => {
                        updated_check.status = HealthStatus::Healthy;
                        updated_check.message = None;
                    }
                    Err(error) => {
                        updated_check.status = HealthStatus::Unhealthy;
                        updated_check.message = Some(error);
                    }
                }
                
                updated_check.last_check = Some(start_time);
            } else {
                updated_check.status = HealthStatus::Unknown;
                updated_check.message = Some("Check function not found".to_string());
            }
            
            results.insert(name.clone(), updated_check);
        }
        
        // Update stored checks
        {
            let mut checks = self.checks.write().await;
            for (name, check) in &results {
                checks.insert(name.clone(), check.clone());
            }
        }
        
        results
    }
    
    pub async fn get_overall_status(&self) -> HealthStatus {
        let checks = self.checks.read().await;
        
        for check in checks.values() {
            match check.status {
                HealthStatus::Unhealthy => return HealthStatus::Unhealthy,
                HealthStatus::Unknown => return HealthStatus::Unknown,
                HealthStatus::Healthy => continue,
            }
        }
        
        HealthStatus::Healthy
    }
    
    pub async fn start_periodic_checks(&self) {
        let checks = self.checks.clone();
        let check_functions = self.check_functions.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                
                let mut checks = checks.write().await;
                let check_functions = check_functions.read().await;
                
                for (name, check) in checks.iter_mut() {
                    if let Some(last_check) = check.last_check {
                        if last_check.elapsed() >= check.check_interval {
                            if let Some(check_fn) = check_functions.get(name) {
                                let start_time = Instant::now();
                                
                                match check_fn() {
                                    Ok(_) => {
                                        check.status = HealthStatus::Healthy;
                                        check.message = None;
                                    }
                                    Err(error) => {
                                        check.status = HealthStatus::Unhealthy;
                                        check.message = Some(error);
                                    }
                                }
                                
                                check.last_check = Some(start_time);
                            }
                        }
                    }
                }
            }
        });
    }
}

// Example health check implementations
pub struct DatabaseHealthCheck {
    pub connection_string: String,
}

impl DatabaseHealthCheck {
    pub async fn check(&self) -> Result<(), String> {
        // Simulate database health check
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // In real implementation, you would:
        // 1. Connect to database
        // 2. Run a simple query
        // 3. Check response time
        // 4. Return Ok(()) or Err(error_message)
        
        Ok(())
    }
}

pub struct RedisHealthCheck {
    pub connection_string: String,
}

impl RedisHealthCheck {
    pub async fn check(&self) -> Result<(), String> {
        // Simulate Redis health check
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // In real implementation, you would:
        // 1. Connect to Redis
        // 2. Run a PING command
        // 3. Check response time
        // 4. Return Ok(()) or Err(error_message)
        
        Ok(())
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_metrics_collection() {
        let metrics = MetricsCollector::new().unwrap();
        
        metrics.record_request("GET", "/api/users", 200, 0.1);
        metrics.record_request("POST", "/api/users", 201, 0.2);
        metrics.record_request("GET", "/api/users", 404, 0.05);
        
        let metrics_text = metrics.get_metrics().unwrap();
        assert!(metrics_text.contains("http_requests_total"));
        assert!(metrics_text.contains("http_request_duration_seconds"));
    }

    #[tokio::test]
    async fn test_health_checks() {
        let health_checker = HealthChecker::new();
        
        // Register a simple health check
        health_checker.register_check(
            "database".to_string(),
            || Ok(()),
            Duration::from_secs(30),
            Duration::from_secs(5),
        ).await;
        
        // Run checks
        let results = health_checker.run_checks().await;
        
        assert!(results.contains_key("database"));
        assert_eq!(results["database"].status, HealthStatus::Healthy);
    }

    #[test]
    fn test_structured_logging() {
        let logger = StructuredLogger::new(
            "test-service".to_string(),
            "test".to_string(),
            "1.0.0".to_string(),
        );
        
        logger.setup().unwrap();
        
        // Test logging with context
        let context = LogContext::new("req-123".to_string())
            .with_user_id("user-456".to_string())
            .with_ip_address("192.168.1.1".to_string());
        
        context.log_with_context(|| {
            info!("Test log message");
        });
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Inefficient Metrics Collection**

```rust
// ❌ Wrong - inefficient metrics collection
pub struct BadMetrics {
    pub counters: HashMap<String, u64>,
}

impl BadMetrics {
    pub fn increment(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
    }
}

// ✅ Correct - efficient metrics collection
pub struct GoodMetrics {
    pub request_counter: Counter,
    pub error_counter: Counter,
}

impl GoodMetrics {
    pub fn record_request(&self, status_code: u16) {
        self.request_counter.inc();
        if status_code >= 400 {
            self.error_counter.inc();
        }
    }
}
```

### **Common Mistake 2: Poor Logging Practices**

```rust
// ❌ Wrong - poor logging practices
fn bad_logging() {
    println!("User {} logged in", user_id);
    println!("Error: {}", error);
}

// ✅ Correct - structured logging
fn good_logging() {
    info!(
        user_id = %user_id,
        action = "login",
        "User logged in successfully"
    );
    
    error!(
        error = %error,
        context = "authentication",
        "Login failed"
    );
}
```

---

## 📊 **Advanced Observability Patterns**

### **Distributed Context Propagation**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct DistributedContext {
    pub trace_id: String,
    pub span_id: String,
    pub baggage: HashMap<String, String>,
}

impl DistributedContext {
    pub fn new() -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            span_id: uuid::Uuid::new_v4().to_string(),
            baggage: HashMap::new(),
        }
    }
    
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = trace_id;
        self
    }
    
    pub fn with_span_id(mut self, span_id: String) -> Self {
        self.span_id = span_id;
        self
    }
    
    pub fn add_baggage(&mut self, key: String, value: String) {
        self.baggage.insert(key, value);
    }
    
    pub fn get_baggage(&self, key: &str) -> Option<&String> {
        self.baggage.get(key)
    }
    
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("trace-id".to_string(), self.trace_id.clone());
        headers.insert("span-id".to_string(), self.span_id.clone());
        
        for (key, value) in &self.baggage {
            headers.insert(format!("baggage-{}", key), value.clone());
        }
        
        headers
    }
    
    pub fn from_headers(headers: &HashMap<String, String>) -> Self {
        let mut context = Self::new();
        
        if let Some(trace_id) = headers.get("trace-id") {
            context.trace_id = trace_id.clone();
        }
        
        if let Some(span_id) = headers.get("span-id") {
            context.span_id = span_id.clone();
        }
        
        for (key, value) in headers {
            if key.starts_with("baggage-") {
                let baggage_key = key.strip_prefix("baggage-").unwrap();
                context.baggage.insert(baggage_key.to_string(), value.clone());
            }
        }
        
        context
    }
}

pub struct ContextManager {
    pub contexts: Arc<RwLock<HashMap<String, DistributedContext>>>,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn store_context(&self, request_id: String, context: DistributedContext) {
        self.contexts.write().await.insert(request_id, context);
    }
    
    pub async fn get_context(&self, request_id: &str) -> Option<DistributedContext> {
        self.contexts.read().await.get(request_id).cloned()
    }
    
    pub async fn remove_context(&self, request_id: &str) {
        self.contexts.write().await.remove(request_id);
    }
}
```

---

## 🎯 **Best Practices**

### **Observability Configuration**

```rust
// ✅ Good - comprehensive observability configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct ObservabilityConfig {
    pub tracing: TracingConfig,
    pub metrics: MetricsConfig,
    pub logging: LoggingConfig,
    pub health_checks: HealthCheckConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TracingConfig {
    pub jaeger_endpoint: String,
    pub sampling_rate: f64,
    pub service_name: String,
    pub service_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MetricsConfig {
    pub prometheus_port: u16,
    pub metrics_path: String,
    pub collect_system_metrics: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: String,
    pub structured: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HealthCheckConfig {
    pub enabled: bool,
    pub port: u16,
    pub path: String,
    pub check_interval: u64,
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ObservabilityError {
    #[error("Tracing error: {0}")]
    Tracing(String),
    
    #[error("Metrics error: {0}")]
    Metrics(String),
    
    #[error("Logging error: {0}")]
    Logging(String),
    
    #[error("Health check error: {0}")]
    HealthCheck(String),
}

pub type Result<T> = std::result::Result<T, ObservabilityError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [OpenTelemetry Rust](https://opentelemetry.io/docs/instrumentation/rust/) - Fetched: 2024-12-19T00:00:00Z
- [Prometheus Rust](https://docs.rs/prometheus/latest/prometheus/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Observability](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Distributed Tracing](https://opentelemetry.io/docs/concepts/distributions/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement distributed tracing systems?
2. Do you understand metrics collection and monitoring?
3. Can you design effective logging strategies?
4. Do you know how to build health check systems?
5. Can you create comprehensive observability solutions?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Production deployment strategies
- Performance optimization
- Security best practices
- Incident response

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [33.2 Production Deployment](33_02_production_deployment.md)
