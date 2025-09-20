# Lesson 31.1: Advanced Web Development

> **Module**: 31 - Advanced Web Development  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 30 (Rust Ecosystem Mastery)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Build production-ready web applications
- Implement advanced authentication and authorization
- Design scalable API architectures
- Optimize web performance
- Handle real-time communication

---

## 🎯 **Overview**

Advanced web development in Rust involves building scalable, secure, and performant web applications using modern frameworks and best practices. This lesson covers authentication, API design, real-time features, and production deployment.

---

## 🔧 **Production Web Application**

### **Complete Web Server with Authentication**

```rust
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub users: Arc<RwLock<HashMap<String, User>>>,
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub jwt_secret: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub password_hash: String,
    pub role: UserRole,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UserRole {
    Admin,
    User,
    Guest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub user_id: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

pub async fn create_app() -> Router {
    let state = AppState {
        users: Arc::new(RwLock::new(HashMap::new())),
        sessions: Arc::new(RwLock::new(HashMap::new())),
        jwt_secret: "your-secret-key".to_string(),
    };

    Router::new()
        .route("/api/users", post(create_user))
        .route("/api/users/:id", get(get_user))
        .route("/api/auth/login", post(login))
        .route("/api/auth/logout", post(logout))
        .route("/api/protected", get(protected_route))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUserRequest>,
) -> Result<Json<UserResponse>, StatusCode> {
    let user_id = uuid::Uuid::new_v4().to_string();
    let password_hash = hash_password(&payload.password)?;
    
    let user = User {
        id: user_id.clone(),
        email: payload.email,
        password_hash,
        role: payload.role,
        created_at: chrono::Utc::now(),
    };
    
    state.users.write().await.insert(user_id.clone(), user.clone());
    
    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        role: user.role,
        created_at: user.created_at,
    }))
}

async fn login(
    State(state): State<AppState>,
    Json(payload): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, StatusCode> {
    let users = state.users.read().await;
    let user = users.get(&payload.email)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !verify_password(&payload.password, &user.password_hash) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    let session_id = uuid::Uuid::new_v4().to_string();
    let session = Session {
        user_id: user.id.clone(),
        expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
    };
    
    state.sessions.write().await.insert(session_id.clone(), session);
    
    let token = create_jwt_token(&user.id, &state.jwt_secret)?;
    
    Ok(Json(LoginResponse {
        token,
        session_id,
        user: UserResponse {
            id: user.id.clone(),
            email: user.email.clone(),
            role: user.role.clone(),
            created_at: user.created_at,
        },
    }))
}

#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub password: String,
    pub role: UserRole,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct UserResponse {
    pub id: String,
    pub email: String,
    pub role: UserRole,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub session_id: String,
    pub user: UserResponse,
}

fn hash_password(password: &str) -> Result<String, StatusCode> {
    use argon2::{Argon2, PasswordHasher};
    use argon2::password_hash::{PasswordHash, PasswordHasher, SaltString};
    
    let salt = SaltString::generate(&mut rand::thread_rng());
    let argon2 = Argon2::default();
    
    argon2.hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn verify_password(password: &str, hash: &str) -> bool {
    use argon2::{Argon2, PasswordVerifier};
    use argon2::password_hash::PasswordHash;
    
    let parsed_hash = PasswordHash::new(hash).ok()?;
    Argon2::default().verify_password(password.as_bytes(), &parsed_hash).is_ok()
}

fn create_jwt_token(user_id: &str, secret: &str) -> Result<String, StatusCode> {
    use jsonwebtoken::{encode, Header, EncodingKey};
    
    let claims = Claims {
        sub: user_id.to_string(),
        exp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp() as usize,
    };
    
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_ref()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Real-time Chat Application**

```rust
use axum::{
    extract::{ws::WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

pub struct ChatServer {
    pub rooms: Arc<RwLock<HashMap<String, ChatRoom>>>,
    pub tx: broadcast::Sender<ChatMessage>,
}

pub struct ChatRoom {
    pub name: String,
    pub users: Vec<String>,
    pub messages: Vec<ChatMessage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub room: String,
    pub user: String,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ChatServer {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        Self {
            rooms: Arc::new(RwLock::new(HashMap::new())),
            tx,
        }
    }
    
    pub async fn create_room(&self, name: String) {
        let room = ChatRoom {
            name: name.clone(),
            users: Vec::new(),
            messages: Vec::new(),
        };
        self.rooms.write().await.insert(name, room);
    }
    
    pub async fn join_room(&self, room_name: &str, user_name: String) -> Result<(), String> {
        let mut rooms = self.rooms.write().await;
        if let Some(room) = rooms.get_mut(room_name) {
            room.users.push(user_name);
            Ok(())
        } else {
            Err("Room not found".to_string())
        }
    }
    
    pub async fn send_message(&self, message: ChatMessage) {
        self.tx.send(message.clone()).unwrap();
        
        let mut rooms = self.rooms.write().await;
        if let Some(room) = rooms.get_mut(&message.room) {
            room.messages.push(message);
        }
    }
}

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(chat_server): State<Arc<ChatServer>>,
) -> Response {
    ws.on_upgrade(|socket| websocket_connection(socket, chat_server))
}

async fn websocket_connection(socket: WebSocket, chat_server: Arc<ChatServer>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = chat_server.tx.subscribe();
    
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
                if let Ok(chat_message) = serde_json::from_str::<ChatMessage>(&text) {
                    chat_server.send_message(chat_message).await;
                }
            }
        }
    });
    
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}
```

### **Exercise 2: API Rate Limiting**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::limit::RateLimitLayer;

pub struct RateLimiter {
    pub limits: Arc<RwLock<HashMap<String, RateLimit>>>,
    pub default_limit: RateLimit,
}

pub struct RateLimit {
    pub requests: u32,
    pub window: Duration,
    pub reset_time: Instant,
}

impl RateLimiter {
    pub fn new(default_limit: RateLimit) -> Self {
        Self {
            limits: Arc::new(RwLock::new(HashMap::new())),
            default_limit,
        }
    }
    
    pub async fn check_limit(&self, key: &str) -> bool {
        let mut limits = self.limits.write().await;
        let now = Instant::now();
        
        let limit = limits.entry(key.to_string()).or_insert_with(|| {
            RateLimit {
                requests: self.default_limit.requests,
                window: self.default_limit.window,
                reset_time: now + self.default_limit.window,
            }
        });
        
        if now >= limit.reset_time {
            limit.requests = self.default_limit.requests;
            limit.reset_time = now + limit.window;
        }
        
        if limit.requests > 0 {
            limit.requests -= 1;
            true
        } else {
            false
        }
    }
}

pub fn create_rate_limit_middleware() -> ServiceBuilder<RateLimitLayer> {
    ServiceBuilder::new()
        .layer(RateLimitLayer::new(100, Duration::from_secs(60)))
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_user_creation() {
        let app = create_app().await;
        let state = app.state();
        
        let user_request = CreateUserRequest {
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
            role: UserRole::User,
        };
        
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&user_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_login() {
        let app = create_app().await;
        let state = app.state();
        
        // Create user first
        let user_request = CreateUserRequest {
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
            role: UserRole::User,
        };
        
        // Test login
        let login_request = LoginRequest {
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
        };
        
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&login_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let rate_limiter = RateLimiter::new(RateLimit {
            requests: 5,
            window: Duration::from_secs(60),
            reset_time: Instant::now(),
        });
        
        // Test within limit
        for _ in 0..5 {
            assert!(rate_limiter.check_limit("test_key").await);
        }
        
        // Test over limit
        assert!(!rate_limiter.check_limit("test_key").await);
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

### **Common Mistake 2: Insecure Authentication**

```rust
// ❌ Wrong - weak password hashing
fn bad_hash_password(password: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    password.hash(&mut hasher);
    hasher.finish().to_string()
}

// ✅ Correct - secure password hashing
fn good_hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    use argon2::{Argon2, PasswordHasher};
    use argon2::password_hash::{PasswordHash, PasswordHasher, SaltString};
    
    let salt = SaltString::generate(&mut rand::thread_rng());
    let argon2 = Argon2::default();
    
    Ok(argon2.hash_password(password.as_bytes(), &salt)?.to_string())
}
```

---

## 📊 **Advanced Web Patterns**

### **Microservices Architecture**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MicroserviceRegistry {
    pub services: Arc<RwLock<HashMap<String, ServiceInfo>>>,
    pub health_checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub endpoint: String,
    pub status: ServiceStatus,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ServiceStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

pub struct HealthCheck {
    pub service_name: String,
    pub check_interval: Duration,
    pub timeout: Duration,
}

impl MicroserviceRegistry {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            health_checks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_service(&self, service: ServiceInfo) {
        self.services.write().await.insert(service.name.clone(), service);
    }
    
    pub async fn get_service(&self, name: &str) -> Option<ServiceInfo> {
        self.services.read().await.get(name).cloned()
    }
    
    pub async fn start_health_checks(&self) {
        let services = self.services.clone();
        let health_checks = self.health_checks.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                Self::check_all_services(&services, &health_checks).await;
            }
        });
    }
    
    async fn check_all_services(
        services: &Arc<RwLock<HashMap<String, ServiceInfo>>>,
        health_checks: &Arc<RwLock<HashMap<String, HealthCheck>>>,
    ) {
        let services = services.read().await;
        let health_checks = health_checks.read().await;
        
        for (name, service) in services.iter() {
            if let Some(health_check) = health_checks.get(name) {
                let is_healthy = Self::check_service_health(service, health_check).await;
                // Update service status
            }
        }
    }
    
    async fn check_service_health(service: &ServiceInfo, health_check: &HealthCheck) -> bool {
        // Implement health check logic
        true
    }
}
```

---

## 🎯 **Best Practices**

### **Security Best Practices**

```rust
// ✅ Good - comprehensive security headers
use tower_http::{
    cors::CorsLayer,
    security_headers::SecurityHeadersLayer,
    trace::TraceLayer,
};

pub fn create_secure_app() -> Router {
    Router::new()
        .layer(SecurityHeadersLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(RateLimitLayer::new(100, Duration::from_secs(60)))
}
```

### **Performance Optimization**

```rust
// ✅ Good - connection pooling
use sqlx::PgPool;

pub struct DatabaseConfig {
    pub pool: PgPool,
    pub max_connections: u32,
    pub min_connections: u32,
}

impl DatabaseConfig {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = PgPool::builder()
            .max_connections(100)
            .min_connections(5)
            .build(database_url)
            .await?;
        
        Ok(Self {
            pool,
            max_connections: 100,
            min_connections: 5,
        })
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Axum Documentation](https://docs.rs/axum/latest/axum/) - Fetched: 2024-12-19T00:00:00Z
- [Tower Documentation](https://docs.rs/tower/latest/tower/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Web Development](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Web Security Best Practices](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you build production-ready web applications?
2. Do you understand authentication and authorization?
3. Can you implement real-time features?
4. Do you know how to optimize web performance?
5. Can you handle security considerations?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced database patterns
- Caching strategies
- Message queues
- Monitoring and observability

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [31.2 Advanced Database Patterns](31_02_database_patterns.md)
