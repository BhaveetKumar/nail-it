# Production Rust Quick Reference Guide

> **Level**: Expert  
> **Last Updated**: 2024-12-19T00:00:00Z  
> **Rust Version**: 1.75.0

---

## 🚀 **Quick Start Commands**

### **Production Build**
```bash
# Optimized release build
cargo build --release

# Build with specific target
cargo build --release --target x86_64-unknown-linux-gnu

# Build with LTO and optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build for production with panic=abort
cargo build --release --config 'profile.release.panic = "abort"'
```

### **Performance Optimization**
```bash
# Profile with perf
perf record --call-graph dwarf target/release/my_app
perf report

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph --bin my_app

# Benchmark
cargo bench

# Check for performance regressions
cargo bench --baseline main
```

---

## 📦 **Production Dependencies**

### **Core Production Crates**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `tokio` | Async runtime | 1.35 | Use `rt-multi-thread` feature |
| `axum` | Web framework | 0.7 | Production-ready HTTP |
| `sqlx` | Database toolkit | 0.7 | Async, compile-time checked |
| `redis` | Redis client | 0.24 | Connection pooling |
| `tracing` | Structured logging | 0.1 | Production logging |
| `opentelemetry` | Distributed tracing | 0.21 | Observability |
| `prometheus` | Metrics | 0.13 | Monitoring |
| `anyhow` | Error handling | 1.0 | Error context |
| `thiserror` | Custom errors | 1.0 | Error types |

### **Security & Authentication**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `argon2` | Password hashing | 0.5 | Secure password storage |
| `jsonwebtoken` | JWT tokens | 9.2 | Authentication |
| `ring` | Cryptography | 0.17 | Encryption, hashing |
| `rustls` | TLS implementation | 0.21 | Secure connections |
| `uuid` | UUID generation | 1.6 | Unique identifiers |

### **Monitoring & Observability**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `opentelemetry-jaeger` | Distributed tracing | 0.20 | Jaeger integration |
| `opentelemetry-prometheus` | Metrics export | 0.10 | Prometheus metrics |
| `tracing-subscriber` | Log formatting | 0.3 | Structured logging |
| `tower-http` | HTTP middleware | 0.5 | Request/response logging |

---

## 🔧 **Production Configuration**

### **Cargo.toml for Production**
```toml
[package]
name = "my-production-app"
version = "0.1.0"
edition = "2021"
authors = ["Your Team <team@company.com>"]
description = "Production Rust application"
license = "MIT OR Apache-2.0"
repository = "https://github.com/company/my-app"
homepage = "https://my-app.company.com"
documentation = "https://docs.rs/my-app"

[dependencies]
# Core dependencies
tokio = { version = "1.35", features = ["full"] }
axum = "0.7"
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls"] }
redis = { version = "0.24", features = ["tokio-comp"] }

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
opentelemetry = { version = "0.21", features = ["trace", "metrics"] }
opentelemetry-jaeger = "0.20"
prometheus = "0.13"

# Security
argon2 = "0.5"
jsonwebtoken = "9.2"
ring = "0.17"
rustls = "0.21"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
overflow-checks = true
strip = true

[profile.dev]
opt-level = 1
debug = true
overflow-checks = true
```

### **Environment Configuration**
```rust
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub observability: ObservabilityConfig,
    pub security: SecurityConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_connections: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub acquire_timeout: u64,
    pub idle_timeout: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ObservabilityConfig {
    pub jaeger_endpoint: String,
    pub prometheus_port: u16,
    pub log_level: String,
    pub log_format: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SecurityConfig {
    pub jwt_secret: String,
    pub password_salt: String,
    pub session_timeout: u64,
}

impl Config {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config {
            server: ServerConfig {
                host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("SERVER_PORT")?.parse()?,
                workers: env::var("SERVER_WORKERS")?.parse()?,
                max_connections: env::var("SERVER_MAX_CONNECTIONS")?.parse()?,
            },
            database: DatabaseConfig {
                url: env::var("DATABASE_URL")?,
                max_connections: env::var("DATABASE_MAX_CONNECTIONS")?.parse()?,
                min_connections: env::var("DATABASE_MIN_CONNECTIONS")?.parse()?,
                acquire_timeout: env::var("DATABASE_ACQUIRE_TIMEOUT")?.parse()?,
                idle_timeout: env::var("DATABASE_IDLE_TIMEOUT")?.parse()?,
            },
            redis: RedisConfig {
                url: env::var("REDIS_URL")?,
                max_connections: env::var("REDIS_MAX_CONNECTIONS")?.parse()?,
                connection_timeout: env::var("REDIS_CONNECTION_TIMEOUT")?.parse()?,
            },
            observability: ObservabilityConfig {
                jaeger_endpoint: env::var("JAEGER_ENDPOINT")?,
                prometheus_port: env::var("PROMETHEUS_PORT")?.parse()?,
                log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
                log_format: env::var("LOG_FORMAT").unwrap_or_else(|_| "json".to_string()),
            },
            security: SecurityConfig {
                jwt_secret: env::var("JWT_SECRET")?,
                password_salt: env::var("PASSWORD_SALT")?,
                session_timeout: env::var("SESSION_TIMEOUT")?.parse()?,
            },
        };
        
        Ok(config)
    }
}
```

---

## 🧪 **Testing Strategy**

### **Unit Tests**
```bash
# Run unit tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_database_connection

# Run with coverage
cargo test --coverage
```

### **Integration Tests**
```bash
# Run integration tests
cargo test --test integration

# Test with testcontainers
cargo test --features testcontainers

# Test with real database
DATABASE_URL=postgres://user:pass@localhost/test cargo test
```

### **Load Testing**
```bash
# Install load testing tools
cargo install cargo-loadtest

# Run load tests
cargo loadtest --duration 60s --concurrency 100

# Run stress tests
cargo loadtest --duration 300s --concurrency 1000
```

---

## 📊 **Performance Monitoring**

### **Metrics Collection**
```rust
use prometheus::{Counter, Histogram, Gauge, Registry, TextEncoder};

pub struct MetricsCollector {
    pub registry: Registry,
    pub request_counter: Counter,
    pub request_duration: Histogram,
    pub active_connections: Gauge,
    pub error_counter: Counter,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        let request_counter = Counter::new(
            "http_requests_total",
            "Total number of HTTP requests"
        )?;
        
        let request_duration = Histogram::with_opts(
            HistogramOpts::new("http_request_duration_seconds", "HTTP request duration")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        )?;
        
        let active_connections = Gauge::new(
            "active_connections",
            "Number of active connections"
        )?;
        
        let error_counter = Counter::new(
            "http_errors_total",
            "Total number of HTTP errors"
        )?;
        
        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(error_counter.clone()))?;
        
        Ok(Self {
            registry,
            request_counter,
            request_duration,
            active_connections,
            error_counter,
        })
    }
}
```

### **Health Checks**
```rust
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};

pub async fn health_check(State(health_checker): State<Arc<HealthChecker>>) -> Result<Json<serde_json::Value>, StatusCode> {
    let checks = health_checker.run_checks().await;
    let overall_status = health_checker.get_overall_status().await;
    
    let status_code = match overall_status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
        HealthStatus::Unknown => StatusCode::SERVICE_UNAVAILABLE,
    };
    
    Ok(Json(serde_json::json!({
        "status": overall_status,
        "checks": checks,
        "timestamp": chrono::Utc::now()
    })))
}

pub async fn readiness_check(State(health_checker): State<Arc<HealthChecker>>) -> Result<Json<serde_json::Value>, StatusCode> {
    let checks = health_checker.run_checks().await;
    
    // Check if all critical services are healthy
    let critical_services = ["database", "redis", "external_api"];
    let all_healthy = critical_services.iter().all(|service| {
        checks.get(*service)
            .map(|check| matches!(check.status, HealthStatus::Healthy))
            .unwrap_or(false)
    });
    
    let status_code = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    
    Ok(Json(serde_json::json!({
        "ready": all_healthy,
        "checks": checks,
        "timestamp": chrono::Utc::now()
    })))
}
```

---

## 🚀 **Deployment Strategies**

### **Docker Deployment**
```dockerfile
# Multi-stage build
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/my-app /usr/local/bin/my-app

EXPOSE 8080

CMD ["my-app"]
```

### **Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: my-app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: my-app-secrets
              key: redis-url
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
```

### **CI/CD Pipeline**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    - name: Run tests
      run: cargo test
    - name: Run clippy
      run: cargo clippy -- -D warnings
    - name: Run fmt check
      run: cargo fmt -- --check

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t my-app:${{ github.sha }} .
    - name: Push to registry
      run: docker push my-app:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: kubectl set image deployment/my-app my-app=my-app:${{ github.sha }}
```

---

## 🔍 **Troubleshooting**

### **Common Issues**
```bash
# Check for memory leaks
valgrind --leak-check=full target/release/my-app

# Profile memory usage
cargo install cargo-profdata
cargo profdata --bin my-app

# Check for undefined behavior
cargo install cargo-miri
cargo miri test

# Check for security vulnerabilities
cargo install cargo-audit
cargo audit
```

### **Debug Commands**
```bash
# Verbose logging
RUST_LOG=debug cargo run

# Backtrace on panic
RUST_BACKTRACE=1 cargo run

# Check for unused dependencies
cargo install cargo-machete
cargo machete

# Check for outdated dependencies
cargo install cargo-outdated
cargo outdated
```

---

## 📚 **Useful Resources**

### **Production Tools**
- [cargo-audit](https://github.com/RustSec/cargo-audit) - Security auditing
- [cargo-outdated](https://github.com/kbknapp/cargo-outdated) - Check outdated deps
- [cargo-machete](https://github.com/bnjbvr/cargo-machete) - Find unused deps
- [cargo-expand](https://github.com/dtolnay/cargo-expand) - Expand macros
- [flamegraph](https://github.com/flamegraph-rs/flamegraph) - Performance profiling

### **Monitoring Tools**
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Metrics visualization
- [Jaeger](https://www.jaegertracing.io/) - Distributed tracing
- [ELK Stack](https://www.elastic.co/elk-stack) - Log aggregation

### **Documentation**
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Security Advisory Database](https://rustsec.org/) - Fetched: 2024-12-19T00:00:00Z
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - Fetched: 2024-12-19T00:00:00Z

---

## 🎯 **Quick Reference**

### **Essential Commands**
```bash
# Development
cargo build
cargo test
cargo clippy
cargo fmt

# Production
cargo build --release
cargo test --release
cargo audit
cargo outdated

# Deployment
docker build -t my-app .
docker run -p 8080:8080 my-app
kubectl apply -f k8s/
```

### **Environment Variables**
```bash
# Required
DATABASE_URL=postgres://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key

# Optional
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=info
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

---

**Cheat Sheet Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
