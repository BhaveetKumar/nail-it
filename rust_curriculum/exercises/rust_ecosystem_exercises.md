# Rust Ecosystem Exercises

> **Modules**: 30-35 (Ecosystem Mastery, Advanced Web Development, Database Patterns)  
> **Difficulty**: Expert  
> **Estimated Time**: 6-8 hours  
> **Prerequisites**: All previous modules

---

## 🎯 **Exercise Overview**

These exercises will challenge your understanding of the Rust ecosystem, advanced web development, and production-ready applications. Complete these to demonstrate expert-level Rust proficiency in real-world scenarios.

---

## 🔴 **Expert Level Exercises**

### **Exercise 1: Crate Analysis and Selection Tool**

**Task**: Build a comprehensive tool that analyzes Rust crates and provides recommendations.

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio;

#[derive(Debug, Serialize, Deserialize)]
pub struct CrateAnalysis {
    pub name: String,
    pub version: String,
    pub downloads: u64,
    pub recent_downloads: u64,
    pub documentation: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub license: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub dependencies: Vec<DependencyInfo>,
    pub dev_dependencies: Vec<DependencyInfo>,
    pub quality_score: f32,
    pub security_audit: SecurityAudit,
    pub maintenance_status: MaintenanceStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityAudit {
    pub vulnerabilities: u32,
    pub last_audit: Option<chrono::DateTime<chrono::Utc>>,
    pub security_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MaintenanceStatus {
    Active,
    Community,
    Archived,
    Deprecated,
}

pub struct CrateAnalyzer {
    client: Client,
    base_url: String,
}

impl CrateAnalyzer {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://crates.io/api/v1/crates".to_string(),
        }
    }
    
    pub async fn analyze_crate(&self, crate_name: &str) -> Result<CrateAnalysis, Box<dyn std::error::Error>> {
        // Fetch crate information
        let crate_info = self.fetch_crate_info(crate_name).await?;
        
        // Fetch version information
        let version_info = self.fetch_version_info(crate_name, &crate_info.max_version).await?;
        
        // Fetch dependency information
        let dependencies = self.fetch_dependencies(crate_name, &crate_info.max_version).await?;
        
        // Calculate quality score
        let quality_score = self.calculate_quality_score(&crate_info, &version_info);
        
        // Perform security audit
        let security_audit = self.perform_security_audit(crate_name).await?;
        
        // Determine maintenance status
        let maintenance_status = self.determine_maintenance_status(&crate_info);
        
        Ok(CrateAnalysis {
            name: crate_info.name,
            version: crate_info.max_version,
            downloads: crate_info.downloads,
            recent_downloads: crate_info.recent_downloads,
            documentation: crate_info.documentation,
            repository: crate_info.repository,
            homepage: crate_info.homepage,
            license: version_info.license,
            keywords: version_info.keywords,
            categories: version_info.categories,
            dependencies,
            dev_dependencies: Vec::new(), // Implement separately
            quality_score,
            security_audit,
            maintenance_status,
        })
    }
    
    async fn fetch_crate_info(&self, crate_name: &str) -> Result<CrateInfo, Box<dyn std::error::Error>> {
        let url = format!("{}/{}", self.base_url, crate_name);
        let response = self.client.get(&url).send().await?;
        let crate_info: CrateInfo = response.json().await?;
        Ok(crate_info)
    }
    
    async fn fetch_version_info(&self, crate_name: &str, version: &str) -> Result<VersionInfo, Box<dyn std::error::Error>> {
        let url = format!("{}/{}/{}", self.base_url, crate_name, version);
        let response = self.client.get(&url).send().await?;
        let version_info: VersionInfo = response.json().await?;
        Ok(version_info)
    }
    
    async fn fetch_dependencies(&self, crate_name: &str, version: &str) -> Result<Vec<DependencyInfo>, Box<dyn std::error::Error>> {
        let url = format!("{}/{}/{}/dependencies", self.base_url, crate_name, version);
        let response = self.client.get(&url).send().await?;
        let deps_response: DependenciesResponse = response.json().await?;
        
        let dependencies = deps_response.dependencies
            .into_iter()
            .map(|dep| DependencyInfo {
                name: dep.crate_id,
                version: dep.req,
                features: dep.features,
            })
            .collect();
        
        Ok(dependencies)
    }
    
    fn calculate_quality_score(&self, crate_info: &CrateInfo, version_info: &VersionInfo) -> f32 {
        let mut score = 0.0;
        
        // Download count (30% weight)
        if crate_info.downloads > 1_000_000 {
            score += 0.3;
        } else if crate_info.downloads > 100_000 {
            score += 0.25;
        } else if crate_info.downloads > 10_000 {
            score += 0.2;
        } else if crate_info.downloads > 1_000 {
            score += 0.1;
        }
        
        // Documentation (20% weight)
        if crate_info.documentation.is_some() {
            score += 0.2;
        }
        
        // Repository (15% weight)
        if crate_info.repository.is_some() {
            score += 0.15;
        }
        
        // License (10% weight)
        if version_info.license.is_some() {
            score += 0.1;
        }
        
        // Keywords and categories (10% weight)
        if !version_info.keywords.is_empty() && !version_info.categories.is_empty() {
            score += 0.1;
        }
        
        // Recent activity (15% weight)
        if crate_info.recent_downloads > crate_info.downloads / 10 {
            score += 0.15;
        }
        
        score
    }
    
    async fn perform_security_audit(&self, crate_name: &str) -> Result<SecurityAudit, Box<dyn std::error::Error>> {
        // This would typically call out to cargo-audit or similar
        // For now, return a mock result
        Ok(SecurityAudit {
            vulnerabilities: 0,
            last_audit: Some(chrono::Utc::now()),
            security_score: 1.0,
        })
    }
    
    fn determine_maintenance_status(&self, crate_info: &CrateInfo) -> MaintenanceStatus {
        // This would analyze recent commits, issues, etc.
        // For now, return a mock result
        MaintenanceStatus::Active
    }
}

#[derive(Debug, Deserialize)]
struct CrateInfo {
    name: String,
    max_version: String,
    downloads: u64,
    recent_downloads: u64,
    documentation: Option<String>,
    repository: Option<String>,
    homepage: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VersionInfo {
    license: Option<String>,
    keywords: Vec<String>,
    categories: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DependenciesResponse {
    dependencies: Vec<Dependency>,
}

#[derive(Debug, Deserialize)]
struct Dependency {
    crate_id: String,
    req: String,
    features: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = CrateAnalyzer::new();
    let analysis = analyzer.analyze_crate("tokio").await?;
    println!("{:#?}", analysis);
    Ok(())
}
```

### **Exercise 2: Advanced Web Application with Real-time Features**

**Task**: Build a real-time collaborative editor with WebSocket support.

```rust
use axum::{
    extract::{ws::WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    pub version: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub collaborators: Vec<Collaborator>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Collaborator {
    pub id: String,
    pub name: String,
    pub cursor_position: Option<u64>,
    pub color: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EditorMessage {
    Join { document_id: String, user: Collaborator },
    Leave { document_id: String, user_id: String },
    Edit { document_id: String, operation: EditOperation },
    CursorMove { document_id: String, user_id: String, position: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EditOperation {
    Insert { position: u64, text: String },
    Delete { position: u64, length: u64 },
    Replace { position: u64, length: u64, text: String },
}

pub struct CollaborativeEditor {
    pub documents: Arc<RwLock<HashMap<String, Document>>>,
    pub message_broadcast: broadcast::Sender<EditorMessage>,
    pub user_sessions: Arc<RwLock<HashMap<String, UserSession>>>,
}

pub struct UserSession {
    pub user_id: String,
    pub document_id: String,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

impl CollaborativeEditor {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            message_broadcast: tx,
            user_sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn create_document(&self, title: String) -> Result<String, EditorError> {
        let document_id = uuid::Uuid::new_v4().to_string();
        let document = Document {
            id: document_id.clone(),
            title,
            content: String::new(),
            version: 0,
            last_modified: chrono::Utc::now(),
            collaborators: Vec::new(),
        };
        
        self.documents.write().await.insert(document_id.clone(), document);
        Ok(document_id)
    }
    
    pub async fn join_document(&self, document_id: &str, user: Collaborator) -> Result<(), EditorError> {
        let mut documents = self.documents.write().await;
        if let Some(document) = documents.get_mut(document_id) {
            document.collaborators.push(user.clone());
            document.last_modified = chrono::Utc::now();
        } else {
            return Err(EditorError::DocumentNotFound);
        }
        
        // Broadcast join message
        let message = EditorMessage::Join {
            document_id: document_id.to_string(),
            user,
        };
        let _ = self.message_broadcast.send(message);
        
        Ok(())
    }
    
    pub async fn apply_edit(&self, document_id: &str, operation: EditOperation) -> Result<(), EditorError> {
        let mut documents = self.documents.write().await;
        if let Some(document) = documents.get_mut(document_id) {
            match operation {
                EditOperation::Insert { position, text } => {
                    if position <= document.content.len() as u64 {
                        document.content.insert_str(position as usize, &text);
                        document.version += 1;
                        document.last_modified = chrono::Utc::now();
                    }
                }
                EditOperation::Delete { position, length } => {
                    if position + length <= document.content.len() as u64 {
                        let start = position as usize;
                        let end = (position + length) as usize;
                        document.content.replace_range(start..end, "");
                        document.version += 1;
                        document.last_modified = chrono::Utc::now();
                    }
                }
                EditOperation::Replace { position, length, text } => {
                    if position + length <= document.content.len() as u64 {
                        let start = position as usize;
                        let end = (position + length) as usize;
                        document.content.replace_range(start..end, &text);
                        document.version += 1;
                        document.last_modified = chrono::Utc::now();
                    }
                }
            }
        } else {
            return Err(EditorError::DocumentNotFound);
        }
        
        // Broadcast edit message
        let message = EditorMessage::Edit {
            document_id: document_id.to_string(),
            operation,
        };
        let _ = self.message_broadcast.send(message);
        
        Ok(())
    }
}

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(editor): State<Arc<CollaborativeEditor>>,
) -> Response {
    ws.on_upgrade(|socket| websocket_connection(socket, editor))
}

async fn websocket_connection(socket: WebSocket, editor: Arc<CollaborativeEditor>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = editor.message_broadcast.subscribe();
    
    // Send messages to client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(axum::extract::ws::Message::Text(
                serde_json::to_string(&msg).unwrap()
            )).await.is_err() {
                break;
            }
        }
    });
    
    // Receive messages from client
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(axum::extract::ws::Message::Text(text)) = msg {
                if let Ok(editor_message) = serde_json::from_str::<EditorMessage>(&text) {
                    match editor_message {
                        EditorMessage::Join { document_id, user } => {
                            let _ = editor.join_document(&document_id, user).await;
                        }
                        EditorMessage::Edit { document_id, operation } => {
                            let _ = editor.apply_edit(&document_id, operation).await;
                        }
                        _ => {}
                    }
                }
            }
        }
    });
    
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

#[derive(Debug)]
pub enum EditorError {
    DocumentNotFound,
    InvalidOperation,
    UserNotFound,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let editor = Arc::new(CollaborativeEditor::new());
    
    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .with_state(editor);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

### **Exercise 3: Production Monitoring and Observability**

**Task**: Implement comprehensive monitoring and observability for a Rust application.

```rust
use opentelemetry::{
    global,
    trace::{Span, Tracer},
    KeyValue,
};
use prometheus::{Counter, Histogram, Registry, TextEncoder};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

pub struct MetricsCollector {
    pub registry: Registry,
    pub request_counter: Counter,
    pub request_duration: Histogram,
    pub error_counter: Counter,
    pub active_connections: Counter,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        let request_counter = Counter::new(
            "http_requests_total",
            "Total number of HTTP requests"
        )?;
        
        let request_duration = Histogram::new(
            "http_request_duration_seconds",
            "HTTP request duration in seconds"
        )?;
        
        let error_counter = Counter::new(
            "http_errors_total",
            "Total number of HTTP errors"
        )?;
        
        let active_connections = Counter::new(
            "active_connections_total",
            "Total number of active connections"
        )?;
        
        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(error_counter.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        
        Ok(Self {
            registry,
            request_counter,
            request_duration,
            error_counter,
            active_connections,
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
        // Note: Prometheus counters can't be decremented
        // Use a gauge for active connections instead
    }
    
    pub fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let metric_families = self.registry.gather();
        let encoder = TextEncoder::new();
        let metrics = encoder.encode_to_string(&metric_families)?;
        Ok(metrics)
    }
}

pub struct HealthChecker {
    pub checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
}

#[derive(Clone, Debug)]
pub struct HealthCheck {
    pub name: String,
    pub check_fn: fn() -> HealthStatus,
    pub timeout: std::time::Duration,
}

#[derive(Clone, Debug)]
pub enum HealthStatus {
    Healthy,
    Unhealthy(String),
    Unknown,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn add_check(&self, check: HealthCheck) {
        self.checks.write().await.insert(check.name.clone(), check);
    }
    
    pub async fn run_checks(&self) -> HashMap<String, HealthStatus> {
        let checks = self.checks.read().await;
        let mut results = HashMap::new();
        
        for (name, check) in checks.iter() {
            let status = (check.check_fn)();
            results.insert(name.clone(), status);
        }
        
        results
    }
}

pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: String,
}

impl LoggingConfig {
    pub fn new() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            output: "stdout".to_string(),
        }
    }
    
    pub fn setup(&self) -> Result<(), Box<dyn std::error::Error>> {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into())
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .json()
            );
        
        subscriber.init();
        Ok(())
    }
}

pub struct ObservabilityManager {
    pub metrics: Arc<MetricsCollector>,
    pub health_checker: Arc<HealthChecker>,
    pub logging_config: LoggingConfig,
}

impl ObservabilityManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let metrics = Arc::new(MetricsCollector::new()?);
        let health_checker = Arc::new(HealthChecker::new());
        let logging_config = LoggingConfig::new();
        
        logging_config.setup()?;
        
        Ok(Self {
            metrics,
            health_checker,
            logging_config,
        })
    }
    
    pub async fn start_health_checks(&self) {
        let health_checker = self.health_checker.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                let results = health_checker.run_checks().await;
                
                for (name, status) in results {
                    match status {
                        HealthStatus::Healthy => info!("Health check '{}' is healthy", name),
                        HealthStatus::Unhealthy(reason) => {
                            error!("Health check '{}' is unhealthy: {}", name, reason)
                        }
                        HealthStatus::Unknown => warn!("Health check '{}' status is unknown", name),
                    }
                }
            }
        });
    }
    
    pub fn record_request(&self, method: &str, path: &str, status_code: u16, duration: f64) {
        self.metrics.record_request(method, path, status_code, duration);
    }
    
    pub fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.metrics.get_metrics()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observability = ObservabilityManager::new()?;
    
    // Add health checks
    observability.health_checker.add_check(HealthCheck {
        name: "database".to_string(),
        check_fn: || HealthStatus::Healthy,
        timeout: std::time::Duration::from_secs(5),
    }).await;
    
    // Start health checks
    observability.start_health_checks().await;
    
    // Record some metrics
    observability.record_request("GET", "/api/users", 200, 0.1);
    observability.record_request("POST", "/api/users", 201, 0.2);
    observability.record_request("GET", "/api/users", 404, 0.05);
    
    // Get metrics
    let metrics = observability.get_metrics()?;
    println!("Metrics:\n{}", metrics);
    
    Ok(())
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
    async fn test_crate_analyzer() {
        let analyzer = CrateAnalyzer::new();
        let analysis = analyzer.analyze_crate("serde").await.unwrap();
        
        assert!(!analysis.name.is_empty());
        assert!(analysis.quality_score >= 0.0 && analysis.quality_score <= 1.0);
    }

    #[tokio::test]
    async fn test_collaborative_editor() {
        let editor = CollaborativeEditor::new();
        let document_id = editor.create_document("Test Document".to_string()).await.unwrap();
        
        let user = Collaborator {
            id: "user1".to_string(),
            name: "Test User".to_string(),
            cursor_position: Some(0),
            color: "#ff0000".to_string(),
        };
        
        editor.join_document(&document_id, user).await.unwrap();
        
        let operation = EditOperation::Insert {
            position: 0,
            text: "Hello, World!".to_string(),
        };
        
        editor.apply_edit(&document_id, operation).await.unwrap();
        
        let documents = editor.documents.read().await;
        let document = documents.get(&document_id).unwrap();
        assert_eq!(document.content, "Hello, World!");
        assert_eq!(document.version, 1);
    }

    #[test]
    fn test_metrics_collector() {
        let metrics = MetricsCollector::new().unwrap();
        
        metrics.record_request("GET", "/api/users", 200, 0.1);
        metrics.record_request("POST", "/api/users", 201, 0.2);
        metrics.record_request("GET", "/api/users", 404, 0.05);
        
        let metrics_text = metrics.get_metrics().unwrap();
        assert!(metrics_text.contains("http_requests_total"));
        assert!(metrics_text.contains("http_request_duration_seconds"));
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Memory Leaks in Web Applications**

```rust
// ❌ Wrong - potential memory leak
pub struct BadWebServer {
    connections: Vec<Connection>, // Never cleaned up
}

// ✅ Correct - proper cleanup
pub struct GoodWebServer {
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    cleanup_task: JoinHandle<()>,
}

impl GoodWebServer {
    pub fn new() -> Self {
        let connections = Arc::new(RwLock::new(HashMap::new()));
        let cleanup_task = Self::start_cleanup_task(connections.clone());
        
        Self {
            connections,
            cleanup_task,
        }
    }
    
    fn start_cleanup_task(connections: Arc<RwLock<HashMap<String, Connection>>>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let mut conns = connections.write().await;
                conns.retain(|_, conn| !conn.is_expired());
            }
        })
    }
}
```

### **Common Mistake 2: Inefficient Metrics Collection**

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

---

## 📊 **Advanced Patterns**

### **Circuit Breaker Pattern**

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    pub failure_count: AtomicU32,
    pub failure_threshold: u32,
    pub timeout: Duration,
    pub last_failure_time: std::sync::Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            failure_threshold,
            timeout,
            last_failure_time: std::sync::Mutex::new(None),
        }
    }
    
    pub fn call<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if self.is_open() {
            return Err(/* Circuit breaker error */);
        }
        
        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }
    
    fn is_open(&self) -> bool {
        let count = self.failure_count.load(Ordering::Relaxed);
        if count >= self.failure_threshold {
            if let Ok(last_failure) = self.last_failure_time.lock() {
                if let Some(time) = *last_failure {
                    return Instant::now().duration_since(time) < self.timeout;
                }
            }
        }
        false
    }
    
    fn on_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
    }
    
    fn on_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= self.failure_threshold {
            if let Ok(mut last_failure) = self.last_failure_time.lock() {
                *last_failure = Some(Instant::now());
            }
        }
    }
}
```

---

## 🎯 **Best Practices**

### **Error Handling**

```rust
// ✅ Good - comprehensive error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApplicationError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Authentication error: {0}")]
    Authentication(String),
    
    #[error("Authorization error: {0}")]
    Authorization(String),
}

pub type Result<T> = std::result::Result<T, ApplicationError>;
```

### **Configuration Management**

```rust
// ✅ Good - structured configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MonitoringConfig {
    pub jaeger_endpoint: String,
    pub prometheus_port: u16,
    pub log_level: String,
}

impl Config {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let mut settings = config::Config::default();
        settings.merge(config::Environment::default())?;
        settings.try_into()
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Web Development](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [OpenTelemetry Rust](https://opentelemetry.io/docs/instrumentation/rust/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Web Development](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you analyze and select Rust crates effectively?
2. Do you understand advanced web development patterns?
3. Can you implement real-time features?
4. Do you know how to set up monitoring and observability?
5. Can you build production-ready applications?

---

**Exercise Set Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
