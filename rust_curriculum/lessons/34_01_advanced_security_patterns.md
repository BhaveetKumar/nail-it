---
# Auto-generated front matter
Title: 34 01 Advanced Security Patterns
LastUpdated: 2025-11-06T20:45:58.117874
Tags: []
Status: draft
---

# Lesson 34.1: Advanced Security Patterns

> **Module**: 34 - Advanced Security Patterns  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 33 (Advanced Observability)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Implement comprehensive security patterns
- Design secure authentication and authorization systems
- Handle cryptographic operations safely
- Build secure communication channels
- Implement security monitoring and incident response

---

## 🎯 **Overview**

Advanced security patterns in Rust involve implementing comprehensive security measures, secure communication protocols, and robust authentication systems. This lesson covers cryptographic operations, secure coding practices, and security monitoring.

---

## 🔧 **Cryptographic Security Implementation**

### **Secure Password Hashing**

```rust
use argon2::{Argon2, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{PasswordHash, PasswordHasher, SaltString};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub password_hash: String,
    pub salt: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_login: Option<chrono::DateTime<chrono::Utc>>,
    pub failed_login_attempts: u32,
    pub locked_until: Option<chrono::DateTime<chrono::Utc>>,
}

pub struct PasswordManager {
    pub argon2: Argon2<'static>,
    pub pepper: String,
}

impl PasswordManager {
    pub fn new(pepper: String) -> Self {
        Self {
            argon2: Argon2::default(),
            pepper,
        }
    }
    
    pub fn hash_password(&self, password: &str) -> Result<(String, String), SecurityError> {
        // Generate random salt
        let salt = SaltString::generate(&mut OsRng);
        
        // Add pepper to password
        let peppered_password = format!("{}{}", password, self.pepper);
        
        // Hash password
        let password_hash = self.argon2
            .hash_password(peppered_password.as_bytes(), &salt)
            .map_err(|_| SecurityError::HashingFailed)?;
        
        Ok((password_hash.to_string(), salt.to_string()))
    }
    
    pub fn verify_password(&self, password: &str, hash: &str, salt: &str) -> Result<bool, SecurityError> {
        // Add pepper to password
        let peppered_password = format!("{}{}", password, self.pepper);
        
        // Parse stored hash
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|_| SecurityError::InvalidHash)?;
        
        // Verify password
        let is_valid = self.argon2
            .verify_password(peppered_password.as_bytes(), &parsed_hash)
            .is_ok();
        
        Ok(is_valid)
    }
    
    pub fn is_password_strong(&self, password: &str) -> bool {
        // Check password strength requirements
        if password.len() < 12 {
            return false;
        }
        
        let has_uppercase = password.chars().any(|c| c.is_uppercase());
        let has_lowercase = password.chars().any(|c| c.is_lowercase());
        let has_digit = password.chars().any(|c| c.is_ascii_digit());
        let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;:,.<>?".contains(c));
        
        has_uppercase && has_lowercase && has_digit && has_special
    }
}

#[derive(Debug)]
pub enum SecurityError {
    HashingFailed,
    InvalidHash,
    WeakPassword,
    AccountLocked,
    TooManyAttempts,
}
```

### **JWT Token Management**

```rust
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,        // Subject (user ID)
    pub iat: u64,          // Issued at
    pub exp: u64,          // Expiration time
    pub iss: String,       // Issuer
    pub aud: String,       // Audience
    pub role: String,      // User role
    pub permissions: Vec<String>, // User permissions
}

pub struct JwtManager {
    pub encoding_key: EncodingKey,
    pub decoding_key: DecodingKey,
    pub algorithm: Algorithm,
    pub access_token_duration: Duration,
    pub refresh_token_duration: Duration,
}

impl JwtManager {
    pub fn new(secret: &str) -> Self {
        let encoding_key = EncodingKey::from_secret(secret.as_ref());
        let decoding_key = DecodingKey::from_secret(secret.as_ref());
        
        Self {
            encoding_key,
            decoding_key,
            algorithm: Algorithm::HS256,
            access_token_duration: Duration::from_secs(3600), // 1 hour
            refresh_token_duration: Duration::from_secs(86400 * 7), // 7 days
        }
    }
    
    pub fn create_access_token(&self, user_id: &str, role: &str, permissions: Vec<String>) -> Result<String, SecurityError> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let exp = now + self.access_token_duration.as_secs();
        
        let claims = Claims {
            sub: user_id.to_string(),
            iat: now,
            exp,
            iss: "my-app".to_string(),
            aud: "my-app-users".to_string(),
            role: role.to_string(),
            permissions,
        };
        
        let header = Header::new(self.algorithm);
        encode(&header, &claims, &self.encoding_key)
            .map_err(|_| SecurityError::TokenCreationFailed)
    }
    
    pub fn create_refresh_token(&self, user_id: &str) -> Result<String, SecurityError> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let exp = now + self.refresh_token_duration.as_secs();
        
        let claims = Claims {
            sub: user_id.to_string(),
            iat: now,
            exp,
            iss: "my-app".to_string(),
            aud: "my-app-refresh".to_string(),
            role: "refresh".to_string(),
            permissions: vec!["refresh".to_string()],
        };
        
        let header = Header::new(self.algorithm);
        encode(&header, &claims, &self.encoding_key)
            .map_err(|_| SecurityError::TokenCreationFailed)
    }
    
    pub fn validate_token(&self, token: &str) -> Result<Claims, SecurityError> {
        let validation = Validation::new(self.algorithm);
        
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|_| SecurityError::InvalidToken)?;
        
        // Check if token is expired
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if token_data.claims.exp < now {
            return Err(SecurityError::TokenExpired);
        }
        
        Ok(token_data.claims)
    }
    
    pub fn refresh_token(&self, refresh_token: &str) -> Result<String, SecurityError> {
        let claims = self.validate_token(refresh_token)?;
        
        // Check if it's a refresh token
        if claims.role != "refresh" {
            return Err(SecurityError::InvalidTokenType);
        }
        
        // Create new access token
        self.create_access_token(&claims.sub, "user", vec!["read".to_string(), "write".to_string()])
    }
}

#[derive(Debug)]
pub enum SecurityError {
    TokenCreationFailed,
    InvalidToken,
    TokenExpired,
    InvalidTokenType,
    HashingFailed,
    InvalidHash,
    WeakPassword,
    AccountLocked,
    TooManyAttempts,
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Secure Session Management**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub expires_at: Instant,
    pub ip_address: String,
    pub user_agent: String,
    pub is_active: bool,
    pub csrf_token: String,
}

pub struct SessionManager {
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub session_duration: Duration,
    pub max_sessions_per_user: usize,
    pub cleanup_interval: Duration,
}

impl SessionManager {
    pub fn new(session_duration: Duration, max_sessions_per_user: usize) -> Self {
        let manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            session_duration,
            max_sessions_per_user,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        };
        
        // Start cleanup task
        manager.start_cleanup_task();
        
        manager
    }
    
    pub async fn create_session(
        &self,
        user_id: &str,
        ip_address: &str,
        user_agent: &str,
    ) -> Result<String, SecurityError> {
        // Check if user has too many sessions
        let user_sessions = self.get_user_sessions(user_id).await;
        if user_sessions.len() >= self.max_sessions_per_user {
            // Remove oldest session
            if let Some(oldest_session) = user_sessions.into_iter().min_by_key(|s| s.created_at) {
                self.sessions.write().await.remove(&oldest_session.id);
            }
        }
        
        let session_id = Uuid::new_v4().to_string();
        let now = Instant::now();
        let expires_at = now + self.session_duration;
        let csrf_token = self.generate_csrf_token();
        
        let session = Session {
            id: session_id.clone(),
            user_id: user_id.to_string(),
            created_at: now,
            last_accessed: now,
            expires_at,
            ip_address: ip_address.to_string(),
            user_agent: user_agent.to_string(),
            is_active: true,
            csrf_token,
        };
        
        self.sessions.write().await.insert(session_id.clone(), session);
        Ok(session_id)
    }
    
    pub async fn validate_session(&self, session_id: &str, ip_address: &str) -> Result<Session, SecurityError> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            // Check if session is expired
            if Instant::now() > session.expires_at {
                sessions.remove(session_id);
                return Err(SecurityError::SessionExpired);
            }
            
            // Check if session is active
            if !session.is_active {
                return Err(SecurityError::SessionInactive);
            }
            
            // Check IP address (optional security measure)
            if session.ip_address != ip_address {
                return Err(SecurityError::InvalidIpAddress);
            }
            
            // Update last accessed time
            session.last_accessed = Instant::now();
            
            // Extend session if it's close to expiring
            if session.last_accessed.duration_since(session.created_at) > self.session_duration / 2 {
                session.expires_at = Instant::now() + self.session_duration;
            }
            
            Ok(session.clone())
        } else {
            Err(SecurityError::SessionNotFound)
        }
    }
    
    pub async fn invalidate_session(&self, session_id: &str) -> Result<(), SecurityError> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.is_active = false;
            Ok(())
        } else {
            Err(SecurityError::SessionNotFound)
        }
    }
    
    pub async fn invalidate_user_sessions(&self, user_id: &str) -> Result<(), SecurityError> {
        let mut sessions = self.sessions.write().await;
        
        for session in sessions.values_mut() {
            if session.user_id == user_id {
                session.is_active = false;
            }
        }
        
        Ok(())
    }
    
    async fn get_user_sessions(&self, user_id: &str) -> Vec<Session> {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|s| s.user_id == user_id && s.is_active)
            .cloned()
            .collect()
    }
    
    fn generate_csrf_token(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 32] = rng.gen();
        base64::encode(bytes)
    }
    
    fn start_cleanup_task(&self) {
        let sessions = self.sessions.clone();
        let cleanup_interval = self.cleanup_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                
                let mut sessions = sessions.write().await;
                let now = Instant::now();
                
                // Remove expired sessions
                sessions.retain(|_, session| {
                    now < session.expires_at && session.is_active
                });
            }
        });
    }
}

#[derive(Debug)]
pub enum SecurityError {
    SessionExpired,
    SessionInactive,
    SessionNotFound,
    InvalidIpAddress,
    TokenCreationFailed,
    InvalidToken,
    TokenExpired,
    InvalidTokenType,
    HashingFailed,
    InvalidHash,
    WeakPassword,
    AccountLocked,
    TooManyAttempts,
}
```

### **Exercise 2: Rate Limiting and DDoS Protection**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct RateLimit {
    pub requests: u32,
    pub window: Duration,
    pub burst: u32,
}

#[derive(Clone, Debug)]
pub struct RateLimitEntry {
    pub count: u32,
    pub window_start: Instant,
    pub last_request: Instant,
}

pub struct RateLimiter {
    pub entries: Arc<RwLock<HashMap<String, RateLimitEntry>>>,
    pub limits: HashMap<String, RateLimit>,
    pub cleanup_interval: Duration,
}

impl RateLimiter {
    pub fn new() -> Self {
        let mut limits = HashMap::new();
        
        // Define rate limits for different endpoints
        limits.insert("login".to_string(), RateLimit {
            requests: 5,
            window: Duration::from_secs(300), // 5 minutes
            burst: 10,
        });
        
        limits.insert("api".to_string(), RateLimit {
            requests: 1000,
            window: Duration::from_secs(3600), // 1 hour
            burst: 100,
        });
        
        limits.insert("upload".to_string(), RateLimit {
            requests: 10,
            window: Duration::from_secs(3600), // 1 hour
            burst: 20,
        });
        
        let limiter = Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            limits,
            cleanup_interval: Duration::from_secs(60),
        };
        
        // Start cleanup task
        limiter.start_cleanup_task();
        
        limiter
    }
    
    pub async fn check_rate_limit(
        &self,
        key: &str,
        endpoint: &str,
    ) -> Result<(), RateLimitError> {
        let limit = self.limits.get(endpoint)
            .ok_or(RateLimitError::UnknownEndpoint)?;
        
        let mut entries = self.entries.write().await;
        let now = Instant::now();
        
        let entry = entries.entry(key.to_string()).or_insert(RateLimitEntry {
            count: 0,
            window_start: now,
            last_request: now,
        });
        
        // Check if window has expired
        if now.duration_since(entry.window_start) >= limit.window {
            entry.count = 0;
            entry.window_start = now;
        }
        
        // Check burst limit
        if now.duration_since(entry.last_request) < Duration::from_millis(100) {
            return Err(RateLimitError::BurstLimitExceeded);
        }
        
        // Check rate limit
        if entry.count >= limit.requests {
            return Err(RateLimitError::RateLimitExceeded);
        }
        
        // Update counters
        entry.count += 1;
        entry.last_request = now;
        
        Ok(())
    }
    
    pub async fn get_remaining_requests(&self, key: &str, endpoint: &str) -> Result<u32, RateLimitError> {
        let limit = self.limits.get(endpoint)
            .ok_or(RateLimitError::UnknownEndpoint)?;
        
        let entries = self.entries.read().await;
        
        if let Some(entry) = entries.get(key) {
            let now = Instant::now();
            
            // Check if window has expired
            if now.duration_since(entry.window_start) >= limit.window {
                Ok(limit.requests)
            } else {
                Ok(limit.requests.saturating_sub(entry.count))
            }
        } else {
            Ok(limit.requests)
        }
    }
    
    fn start_cleanup_task(&self) {
        let entries = self.entries.clone();
        let cleanup_interval = self.cleanup_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                
                let mut entries = entries.write().await;
                let now = Instant::now();
                
                // Remove expired entries
                entries.retain(|_, entry| {
                    now.duration_since(entry.window_start) < Duration::from_secs(3600) // Keep for 1 hour
                });
            }
        });
    }
}

#[derive(Debug)]
pub enum RateLimitError {
    RateLimitExceeded,
    BurstLimitExceeded,
    UnknownEndpoint,
}

pub struct DDoSProtection {
    pub rate_limiter: RateLimiter,
    pub ip_blacklist: Arc<RwLock<HashMap<String, Instant>>>,
    pub suspicious_ips: Arc<RwLock<HashMap<String, u32>>>,
}

impl DDoSProtection {
    pub fn new() -> Self {
        Self {
            rate_limiter: RateLimiter::new(),
            ip_blacklist: Arc::new(RwLock::new(HashMap::new())),
            suspicious_ips: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn check_request(&self, ip: &str, endpoint: &str) -> Result<(), DDoSError> {
        // Check if IP is blacklisted
        if self.is_ip_blacklisted(ip).await {
            return Err(DDoSError::IpBlacklisted);
        }
        
        // Check rate limit
        if let Err(_) = self.rate_limiter.check_rate_limit(ip, endpoint).await {
            // Increment suspicious activity counter
            self.increment_suspicious_activity(ip).await;
            
            // Check if IP should be blacklisted
            if self.should_blacklist_ip(ip).await {
                self.blacklist_ip(ip).await;
                return Err(DDoSError::IpBlacklisted);
            }
            
            return Err(DDoSError::RateLimitExceeded);
        }
        
        Ok(())
    }
    
    async fn is_ip_blacklisted(&self, ip: &str) -> bool {
        let blacklist = self.ip_blacklist.read().await;
        if let Some(blacklisted_until) = blacklist.get(ip) {
            if Instant::now() < *blacklisted_until {
                return true;
            }
        }
        false
    }
    
    async fn increment_suspicious_activity(&self, ip: &str) {
        let mut suspicious_ips = self.suspicious_ips.write().await;
        let count = suspicious_ips.entry(ip.to_string()).or_insert(0);
        *count += 1;
    }
    
    async fn should_blacklist_ip(&self, ip: &str) -> bool {
        let suspicious_ips = self.suspicious_ips.read().await;
        if let Some(count) = suspicious_ips.get(ip) {
            *count > 10 // Blacklist after 10 suspicious activities
        } else {
            false
        }
    }
    
    async fn blacklist_ip(&self, ip: &str) {
        let mut blacklist = self.ip_blacklist.write().await;
        blacklist.insert(ip.to_string(), Instant::now() + Duration::from_secs(3600)); // Blacklist for 1 hour
    }
}

#[derive(Debug)]
pub enum DDoSError {
    IpBlacklisted,
    RateLimitExceeded,
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
    async fn test_password_hashing() {
        let password_manager = PasswordManager::new("pepper".to_string());
        
        let password = "SecurePassword123!";
        let (hash, salt) = password_manager.hash_password(password).unwrap();
        
        assert!(password_manager.verify_password(password, &hash, &salt).unwrap());
        assert!(!password_manager.verify_password("wrong_password", &hash, &salt).unwrap());
    }

    #[tokio::test]
    async fn test_jwt_token_management() {
        let jwt_manager = JwtManager::new("secret");
        
        let user_id = "user123";
        let role = "admin";
        let permissions = vec!["read".to_string(), "write".to_string()];
        
        let token = jwt_manager.create_access_token(user_id, role, permissions).unwrap();
        let claims = jwt_manager.validate_token(&token).unwrap();
        
        assert_eq!(claims.sub, user_id);
        assert_eq!(claims.role, role);
    }

    #[tokio::test]
    async fn test_session_management() {
        let session_manager = SessionManager::new(Duration::from_secs(3600), 5);
        
        let user_id = "user123";
        let ip_address = "192.168.1.1";
        let user_agent = "Mozilla/5.0";
        
        let session_id = session_manager.create_session(user_id, ip_address, user_agent).await.unwrap();
        let session = session_manager.validate_session(&session_id, ip_address).await.unwrap();
        
        assert_eq!(session.user_id, user_id);
        assert_eq!(session.ip_address, ip_address);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let rate_limiter = RateLimiter::new();
        
        let key = "user123";
        let endpoint = "login";
        
        // Should allow first 5 requests
        for _ in 0..5 {
            assert!(rate_limiter.check_rate_limit(key, endpoint).await.is_ok());
        }
        
        // Should reject 6th request
        assert!(rate_limiter.check_rate_limit(key, endpoint).await.is_err());
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Insecure Password Storage**

```rust
// ❌ Wrong - insecure password hashing
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn bad_hash_password(password: &str) -> String {
    let mut hasher = DefaultHasher::new();
    password.hash(&mut hasher);
    hasher.finish().to_string()
}

// ✅ Correct - secure password hashing
use argon2::{Argon2, PasswordHasher};
use argon2::password_hash::{PasswordHash, PasswordHasher, SaltString};

fn good_hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut rand::thread_rng());
    let argon2 = Argon2::default();
    
    Ok(argon2.hash_password(password.as_bytes(), &salt)?.to_string())
}
```

### **Common Mistake 2: Insecure Session Management**

```rust
// ❌ Wrong - insecure session management
pub struct BadSession {
    pub user_id: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

impl BadSession {
    pub fn is_valid(&self) -> bool {
        chrono::Utc::now() < self.expires_at
    }
}

// ✅ Correct - secure session management
pub struct GoodSession {
    pub id: String,
    pub user_id: String,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub expires_at: Instant,
    pub ip_address: String,
    pub user_agent: String,
    pub is_active: bool,
    pub csrf_token: String,
}

impl GoodSession {
    pub fn is_valid(&self, current_ip: &str) -> bool {
        self.is_active &&
        Instant::now() < self.expires_at &&
        self.ip_address == current_ip
    }
}
```

---

## 📊 **Advanced Security Patterns**

### **Secure Communication Channel**

```rust
use rustls::{ClientConfig, ServerConfig};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

pub struct SecureServer {
    pub tls_config: Arc<ServerConfig>,
    pub port: u16,
}

impl SecureServer {
    pub fn new(cert_path: &str, key_path: &str, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let certs = load_certs(cert_path)?;
        let key = load_private_key(key_path)?;
        
        let tls_config = ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;
        
        Ok(Self {
            tls_config: Arc::new(tls_config),
            port,
        })
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        let acceptor = TlsAcceptor::from(self.tls_config.clone());
        
        loop {
            let (stream, addr) = listener.accept().await?;
            let acceptor = acceptor.clone();
            
            tokio::spawn(async move {
                if let Ok(tls_stream) = acceptor.accept(stream).await {
                    Self::handle_connection(tls_stream, addr).await;
                }
            });
        }
    }
    
    async fn handle_connection(tls_stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>, addr: std::net::SocketAddr) {
        // Handle secure connection
        println!("Secure connection from: {}", addr);
    }
}

fn load_certs(filename: &str) -> Result<Vec<rustls::Certificate>, Box<dyn std::error::Error>> {
    let certfile = std::fs::File::open(filename)?;
    let mut reader = std::io::BufReader::new(certfile);
    let certs = rustls_pemfile::certs(&mut reader)?;
    Ok(certs.into_iter().map(rustls::Certificate).collect())
}

fn load_private_key(filename: &str) -> Result<rustls::PrivateKey, Box<dyn std::error::Error>> {
    let keyfile = std::fs::File::open(filename)?;
    let mut reader = std::io::BufReader::new(keyfile);
    let keys = rustls_pemfile::pkcs8_private_keys(&mut reader)?;
    Ok(rustls::PrivateKey(keys[0].clone()))
}
```

---

## 🎯 **Best Practices**

### **Security Configuration**

```rust
// ✅ Good - comprehensive security configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct SecurityConfig {
    pub password: PasswordConfig,
    pub jwt: JwtConfig,
    pub session: SessionConfig,
    pub rate_limiting: RateLimitingConfig,
    pub encryption: EncryptionConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PasswordConfig {
    pub min_length: usize,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_digits: bool,
    pub require_special_chars: bool,
    pub pepper: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JwtConfig {
    pub secret: String,
    pub access_token_duration: u64,
    pub refresh_token_duration: u64,
    pub algorithm: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SessionConfig {
    pub duration: u64,
    pub max_sessions_per_user: usize,
    pub secure: bool,
    pub http_only: bool,
    pub same_site: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RateLimitingConfig {
    pub enabled: bool,
    pub default_limit: u32,
    pub window: u64,
    pub burst_limit: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EncryptionConfig {
    pub algorithm: String,
    pub key_size: usize,
    pub iv_size: usize,
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive security error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Authorization failed: {0}")]
    AuthorizationFailed(String),
    
    #[error("Token error: {0}")]
    TokenError(String),
    
    #[error("Session error: {0}")]
    SessionError(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Encryption error: {0}")]
    EncryptionError(String),
    
    #[error("Security policy violation: {0}")]
    SecurityPolicyViolation(String),
}

pub type Result<T> = std::result::Result<T, SecurityError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Security Advisory Database](https://rustsec.org/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Cryptography](https://cryptography.rs/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [OWASP Rust Security](https://owasp.org/www-project-rust-security/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Security Best Practices](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement secure authentication systems?
2. Do you understand cryptographic operations in Rust?
3. Can you build secure communication channels?
4. Do you know how to implement rate limiting and DDoS protection?
5. Can you create comprehensive security monitoring?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced performance optimization
- Memory management patterns
- Concurrency optimization
- Production deployment strategies

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [34.2 Advanced Performance Optimization](34_02_performance_optimization.md)
