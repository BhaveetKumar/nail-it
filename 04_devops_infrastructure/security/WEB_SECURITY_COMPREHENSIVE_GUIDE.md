# 🔒 Web Security Comprehensive Guide

> **Complete guide to web security policies and practices with implementations in Node.js, Go, and Rust**

## 📚 Table of Contents

1. [CORS (Cross-Origin Resource Sharing)](#-cors-cross-origin-resource-sharing)
2. [CSRF (Cross-Site Request Forgery)](#-csrf-cross-site-request-forgery)
3. [CSP (Content Security Policy)](#-csp-content-security-policy)
4. [Authentication & Authorization](#-authentication--authorization)
5. [Input Validation & Sanitization](#-input-validation--sanitization)
6. [Rate Limiting](#-rate-limiting)
7. [Security Headers](#-security-headers)
8. [Session Management](#-session-management)
9. [HTTPS & TLS](#-https--tls)
10. [Security Best Practices](#-security-best-practices)

---

## 🌐 CORS (Cross-Origin Resource Sharing)

### What is CORS?

CORS is a security feature implemented by web browsers that blocks requests from one domain to another domain unless the server explicitly allows it.

### How CORS Works

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Browser       │    │   Server        │    │   API Server    │
│   (Origin A)    │    │   (Origin B)    │    │   (Origin C)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ 1. Request           │                      │
          │─────────────────────▶│                      │
          │                      │                      │
          │ 2. CORS Headers      │                      │
          │◀─────────────────────│                      │
          │                      │                      │
          │ 3. Actual Request    │                      │
          │─────────────────────▶│                      │
          │                      │                      │
          │ 4. Response          │                      │
          │◀─────────────────────│                      │
```

### Node.js Implementation

```javascript
const express = require('express');
const cors = require('cors');

const app = express();

// Basic CORS configuration
app.use(cors({
    origin: ['https://example.com', 'https://app.example.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
    maxAge: 86400 // 24 hours
}));

// Dynamic CORS based on request
app.use(cors((req, callback) => {
    const origin = req.header('Origin');
    const allowedOrigins = ['https://example.com', 'https://app.example.com'];
    
    if (allowedOrigins.includes(origin)) {
        callback(null, {
            origin: true,
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
        });
    } else {
        callback(new Error('Not allowed by CORS'));
    }
}));

// Manual CORS headers
app.use((req, res, next) => {
    const origin = req.headers.origin;
    const allowedOrigins = ['https://example.com', 'https://app.example.com'];
    
    if (allowedOrigins.includes(origin)) {
        res.header('Access-Control-Allow-Origin', origin);
        res.header('Access-Control-Allow-Credentials', 'true');
        res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
        res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
        res.header('Access-Control-Max-Age', '86400');
    }
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

### Go Implementation

```go
package main

import (
    "net/http"
    "strings"
    "time"
)

type CORSOptions struct {
    AllowedOrigins   []string
    AllowedMethods   []string
    AllowedHeaders   []string
    AllowCredentials bool
    MaxAge          time.Duration
}

func CORS(options CORSOptions) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            origin := r.Header.Get("Origin")
            
            // Check if origin is allowed
            if isOriginAllowed(origin, options.AllowedOrigins) {
                w.Header().Set("Access-Control-Allow-Origin", origin)
                w.Header().Set("Access-Control-Allow-Credentials", "true")
            }
            
            w.Header().Set("Access-Control-Allow-Methods", strings.Join(options.AllowedMethods, ", "))
            w.Header().Set("Access-Control-Allow-Headers", strings.Join(options.AllowedHeaders, ", "))
            w.Header().Set("Access-Control-Max-Age", options.MaxAge.String())
            
            // Handle preflight requests
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

func isOriginAllowed(origin string, allowedOrigins []string) bool {
    if origin == "" {
        return false
    }
    
    for _, allowed := range allowedOrigins {
        if origin == allowed {
            return true
        }
    }
    return false
}

// Usage
func main() {
    mux := http.NewServeMux()
    
    corsOptions := CORSOptions{
        AllowedOrigins:   []string{"https://example.com", "https://app.example.com"},
        AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowedHeaders:   []string{"Content-Type", "Authorization", "X-Requested-With"},
        AllowCredentials: true,
        MaxAge:          24 * time.Hour,
    }
    
    mux.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"message": "Hello from API"}`))
    })
    
    handler := CORS(corsOptions)(mux)
    http.ListenAndServe(":8080", handler)
}
```

### Rust Implementation

```rust
use actix_web::{web, App, HttpServer, HttpResponse, middleware::cors::Cors};
use serde_json::json;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(
                Cors::default()
                    .allowed_origin("https://example.com")
                    .allowed_origin("https://app.example.com")
                    .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
                    .allowed_headers(vec![
                        actix_web::http::header::AUTHORIZATION,
                        actix_web::http::header::ACCEPT,
                        actix_web::http::header::CONTENT_TYPE,
                    ])
                    .supports_credentials()
                    .max_age(86400)
            )
            .route("/api/data", web::get().to(api_handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn api_handler() -> HttpResponse {
    HttpResponse::Ok()
        .json(json!({
            "message": "Hello from Rust API"
        }))
}

// Custom CORS middleware
use actix_web::{dev::ServiceRequest, dev::ServiceResponse, Error, middleware::Service};
use futures::future::{ok, Ready};

pub struct CustomCors;

impl<S, B> Service<ServiceRequest> for CustomCors
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Ready<Result<Self::Response, Self::Error>>;

    fn poll_ready(&self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let origin = req.headers().get("origin");
        let allowed_origins = vec!["https://example.com", "https://app.example.com"];
        
        if let Some(origin) = origin {
            if let Ok(origin_str) = origin.to_str() {
                if allowed_origins.contains(&origin_str) {
                    // Add CORS headers
                    // Implementation details...
                }
            }
        }
        
        ok(req.into_response(ServiceResponse::new(req.into_parts().0, HttpResponse::Ok())))
    }
}
```

---

## 🛡️ CSRF (Cross-Site Request Forgery)

### What is CSRF?

CSRF is an attack that tricks a user into performing unwanted actions on a web application in which they're authenticated.

### How CSRF Works

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Attacker      │    │   User          │    │   Bank Website  │
│   (evil.com)    │    │   (Browser)     │    │   (bank.com)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ 1. User visits       │                      │
          │    evil.com          │                      │
          │─────────────────────▶│                      │
          │                      │                      │
          │ 2. Malicious form    │                      │
          │◀─────────────────────│                      │
          │                      │                      │
          │ 3. Form submission   │                      │
          │    to bank.com       │                      │
          │─────────────────────▶│─────────────────────▶│
          │                      │                      │
          │ 4. Unauthorized      │                      │
          │    transfer          │                      │
          │◀─────────────────────│◀─────────────────────│
```

### Node.js Implementation

```javascript
const express = require('express');
const csrf = require('csurf');
const cookieParser = require('cookie-parser');

const app = express();

// CSRF protection middleware
app.use(cookieParser());
app.use(csrf({ cookie: true }));

// Custom CSRF implementation
const crypto = require('crypto');

class CSRFProtection {
    constructor(secret) {
        this.secret = secret;
    }
    
    generateToken(sessionId) {
        const timestamp = Date.now().toString();
        const data = `${sessionId}:${timestamp}`;
        const token = crypto.createHmac('sha256', this.secret)
            .update(data)
            .digest('hex');
        return `${token}:${timestamp}`;
    }
    
    validateToken(token, sessionId) {
        const [tokenHash, timestamp] = token.split(':');
        const data = `${sessionId}:${timestamp}`;
        const expectedHash = crypto.createHmac('sha256', this.secret)
            .update(data)
            .digest('hex');
        
        // Check if token is not older than 1 hour
        const tokenAge = Date.now() - parseInt(timestamp);
        if (tokenAge > 3600000) { // 1 hour
            return false;
        }
        
        return tokenHash === expectedHash;
    }
}

const csrfProtection = new CSRFProtection('your-secret-key');

// Middleware to generate CSRF token
app.use((req, res, next) => {
    if (req.session) {
        req.csrfToken = csrfProtection.generateToken(req.session.id);
        res.locals.csrfToken = req.csrfToken;
    }
    next();
});

// Middleware to validate CSRF token
app.use('/api', (req, res, next) => {
    const token = req.headers['x-csrf-token'] || req.body._csrf;
    
    if (!token || !csrfProtection.validateToken(token, req.session.id)) {
        return res.status(403).json({ error: 'Invalid CSRF token' });
    }
    
    next();
});

// Example form with CSRF token
app.get('/form', (req, res) => {
    res.send(`
        <form action="/api/transfer" method="POST">
            <input type="hidden" name="_csrf" value="${req.csrfToken}">
            <input type="text" name="amount" placeholder="Amount">
            <input type="text" name="to" placeholder="To">
            <button type="submit">Transfer</button>
        </form>
    `);
});

app.listen(3000);
```

### Go Implementation

```go
package main

import (
    "crypto/hmac"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/hex"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type CSRFProtection struct {
    secret []byte
}

func NewCSRFProtection(secret string) *CSRFProtection {
    return &CSRFProtection{
        secret: []byte(secret),
    }
}

func (c *CSRFProtection) GenerateToken(sessionID string) string {
    timestamp := time.Now().Unix()
    data := fmt.Sprintf("%s:%d", sessionID, timestamp)
    
    h := hmac.New(sha256.New, c.secret)
    h.Write([]byte(data))
    token := hex.EncodeToString(h.Sum(nil))
    
    return fmt.Sprintf("%s:%d", token, timestamp)
}

func (c *CSRFProtection) ValidateToken(token, sessionID string) bool {
    parts := strings.Split(token, ":")
    if len(parts) != 2 {
        return false
    }
    
    tokenHash := parts[0]
    timestampStr := parts[1]
    
    timestamp, err := strconv.ParseInt(timestampStr, 10, 64)
    if err != nil {
        return false
    }
    
    // Check if token is not older than 1 hour
    if time.Now().Unix()-timestamp > 3600 {
        return false
    }
    
    data := fmt.Sprintf("%s:%d", sessionID, timestamp)
    h := hmac.New(sha256.New, c.secret)
    h.Write([]byte(data))
    expectedHash := hex.EncodeToString(h.Sum(nil))
    
    return hmac.Equal([]byte(tokenHash), []byte(expectedHash))
}

func CSRFMiddleware(csrf *CSRFProtection) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            if r.Method == "GET" {
                // Generate token for GET requests
                sessionID := getSessionID(r)
                token := csrf.GenerateToken(sessionID)
                w.Header().Set("X-CSRF-Token", token)
                next.ServeHTTP(w, r)
                return
            }
            
            // Validate token for POST, PUT, DELETE requests
            token := r.Header.Get("X-CSRF-Token")
            if token == "" {
                token = r.FormValue("_csrf")
            }
            
            sessionID := getSessionID(r)
            if !csrf.ValidateToken(token, sessionID) {
                http.Error(w, "Invalid CSRF token", http.StatusForbidden)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

func getSessionID(r *http.Request) string {
    // Get session ID from cookie or header
    cookie, err := r.Cookie("session_id")
    if err == nil {
        return cookie.Value
    }
    return r.Header.Get("X-Session-ID")
}

func main() {
    csrf := NewCSRFProtection("your-secret-key")
    
    mux := http.NewServeMux()
    mux.HandleFunc("/api/transfer", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Transfer successful"))
    })
    
    handler := CSRFMiddleware(csrf)(mux)
    http.ListenAndServe(":8080", handler)
}
```

### Rust Implementation

```rust
use actix_web::{web, App, HttpServer, HttpResponse, middleware, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub struct CSRFProtection {
    secret: Vec<u8>,
}

impl CSRFProtection {
    pub fn new(secret: &str) -> Self {
        Self {
            secret: secret.as_bytes().to_vec(),
        }
    }
    
    pub fn generate_token(&self, session_id: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let data = format!("{}:{}", session_id, timestamp);
        let mut mac = HmacSha256::new_from_slice(&self.secret).unwrap();
        mac.update(data.as_bytes());
        let token = hex::encode(mac.finalize().into_bytes());
        
        format!("{}:{}", token, timestamp)
    }
    
    pub fn validate_token(&self, token: &str, session_id: &str) -> bool {
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() != 2 {
            return false;
        }
        
        let token_hash = parts[0];
        let timestamp_str = parts[1];
        
        let timestamp: u64 = match timestamp_str.parse() {
            Ok(t) => t,
            Err(_) => return false,
        };
        
        // Check if token is not older than 1 hour
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        if current_time - timestamp > 3600 {
            return false;
        }
        
        let data = format!("{}:{}", session_id, timestamp);
        let mut mac = HmacSha256::new_from_slice(&self.secret).unwrap();
        mac.update(data.as_bytes());
        let expected_hash = hex::encode(mac.finalize().into_bytes());
        
        token_hash == expected_hash
    }
}

async fn csrf_middleware(
    req: actix_web::HttpRequest,
    next: actix_web::middleware::Next,
) -> Result<actix_web::HttpResponse, actix_web::Error> {
    let csrf = CSRFProtection::new("your-secret-key");
    
    if req.method() == "GET" {
        // Generate token for GET requests
        let session_id = get_session_id(&req);
        let token = csrf.generate_token(&session_id);
        let mut res = next.run(req).await?;
        res.headers_mut().insert(
            "X-CSRF-Token",
            token.parse().unwrap(),
        );
        Ok(res)
    } else {
        // Validate token for other methods
        let token = req.headers()
            .get("X-CSRF-Token")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        
        let session_id = get_session_id(&req);
        if !csrf.validate_token(token, &session_id) {
            return Ok(HttpResponse::Forbidden()
                .json(json!({"error": "Invalid CSRF token"})));
        }
        
        next.run(req).await
    }
}

fn get_session_id(req: &actix_web::HttpRequest) -> String {
    req.cookie("session_id")
        .map(|c| c.value().to_string())
        .unwrap_or_else(|| "default".to_string())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(csrf_middleware)
            .route("/api/transfer", web::post().to(transfer_handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn transfer_handler() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(json!({"message": "Transfer successful"})))
}
```

---

## 🛡️ CSP (Content Security Policy)

### What is CSP?

CSP is a security feature that helps prevent XSS attacks by controlling which resources can be loaded and executed.

### CSP Directives

```
Content-Security-Policy: 
    default-src 'self';
    script-src 'self' 'unsafe-inline' https://trusted-cdn.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    font-src 'self' https://fonts.gstatic.com;
    connect-src 'self' https://api.example.com;
    frame-src 'none';
    object-src 'none';
    base-uri 'self';
    form-action 'self';
    upgrade-insecure-requests;
```

### Node.js Implementation

```javascript
const express = require('express');
const helmet = require('helmet');

const app = express();

// Using helmet for CSP
app.use(helmet.contentSecurityPolicy({
    directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "https://trusted-cdn.com"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", "data:", "https:"],
        fontSrc: ["'self'", "https://fonts.gstatic.com"],
        connectSrc: ["'self'", "https://api.example.com"],
        frameSrc: ["'none'"],
        objectSrc: ["'none'"],
        baseUri: ["'self'"],
        formAction: ["'self'"],
        upgradeInsecureRequests: []
    }
}));

// Custom CSP implementation
app.use((req, res, next) => {
    const csp = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://trusted-cdn.com",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "font-src 'self' https://fonts.gstatic.com",
        "connect-src 'self' https://api.example.com",
        "frame-src 'none'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests"
    ].join('; ');
    
    res.setHeader('Content-Security-Policy', csp);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    
    next();
});

// Dynamic CSP based on route
app.use('/admin', (req, res, next) => {
    const adminCSP = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "connect-src 'self' https://admin-api.example.com"
    ].join('; ');
    
    res.setHeader('Content-Security-Policy', adminCSP);
    next();
});

app.listen(3000);
```

### Go Implementation

```go
package main

import (
    "net/http"
    "strings"
)

type CSPDirectives struct {
    DefaultSrc     []string
    ScriptSrc      []string
    StyleSrc       []string
    ImgSrc         []string
    FontSrc        []string
    ConnectSrc     []string
    FrameSrc       []string
    ObjectSrc      []string
    BaseUri        []string
    FormAction     []string
    UpgradeInsecure bool
}

func CSP(directives CSPDirectives) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            var csp []string
            
            if len(directives.DefaultSrc) > 0 {
                csp = append(csp, "default-src "+strings.Join(directives.DefaultSrc, " "))
            }
            
            if len(directives.ScriptSrc) > 0 {
                csp = append(csp, "script-src "+strings.Join(directives.ScriptSrc, " "))
            }
            
            if len(directives.StyleSrc) > 0 {
                csp = append(csp, "style-src "+strings.Join(directives.StyleSrc, " "))
            }
            
            if len(directives.ImgSrc) > 0 {
                csp = append(csp, "img-src "+strings.Join(directives.ImgSrc, " "))
            }
            
            if len(directives.FontSrc) > 0 {
                csp = append(csp, "font-src "+strings.Join(directives.FontSrc, " "))
            }
            
            if len(directives.ConnectSrc) > 0 {
                csp = append(csp, "connect-src "+strings.Join(directives.ConnectSrc, " "))
            }
            
            if len(directives.FrameSrc) > 0 {
                csp = append(csp, "frame-src "+strings.Join(directives.FrameSrc, " "))
            }
            
            if len(directives.ObjectSrc) > 0 {
                csp = append(csp, "object-src "+strings.Join(directives.ObjectSrc, " "))
            }
            
            if len(directives.BaseUri) > 0 {
                csp = append(csp, "base-uri "+strings.Join(directives.BaseUri, " "))
            }
            
            if len(directives.FormAction) > 0 {
                csp = append(csp, "form-action "+strings.Join(directives.FormAction, " "))
            }
            
            if directives.UpgradeInsecure {
                csp = append(csp, "upgrade-insecure-requests")
            }
            
            w.Header().Set("Content-Security-Policy", strings.Join(csp, "; "))
            w.Header().Set("X-Content-Type-Options", "nosniff")
            w.Header().Set("X-Frame-Options", "DENY")
            w.Header().Set("X-XSS-Protection", "1; mode=block")
            
            next.ServeHTTP(w, r)
        })
    }
}

func main() {
    mux := http.NewServeMux()
    
    cspDirectives := CSPDirectives{
        DefaultSrc:     []string{"'self'"},
        ScriptSrc:      []string{"'self'", "'unsafe-inline'", "https://trusted-cdn.com"},
        StyleSrc:       []string{"'self'", "'unsafe-inline'"},
        ImgSrc:         []string{"'self'", "data:", "https:"},
        FontSrc:        []string{"'self'", "https://fonts.gstatic.com"},
        ConnectSrc:     []string{"'self'", "https://api.example.com"},
        FrameSrc:       []string{"'none'"},
        ObjectSrc:      []string{"'none'"},
        BaseUri:        []string{"'self'"},
        FormAction:     []string{"'self'"},
        UpgradeInsecure: true,
    }
    
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Hello World"))
    })
    
    handler := CSP(cspDirectives)(mux)
    http.ListenAndServe(":8080", handler)
}
```

### Rust Implementation

```rust
use actix_web::{web, App, HttpServer, HttpResponse, middleware};
use serde_json::json;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(middleware::DefaultHeaders::new()
                .add(("Content-Security-Policy", 
                    "default-src 'self'; \
                     script-src 'self' 'unsafe-inline' https://trusted-cdn.com; \
                     style-src 'self' 'unsafe-inline'; \
                     img-src 'self' data: https:; \
                     font-src 'self' https://fonts.gstatic.com; \
                     connect-src 'self' https://api.example.com; \
                     frame-src 'none'; \
                     object-src 'none'; \
                     base-uri 'self'; \
                     form-action 'self'; \
                     upgrade-insecure-requests"))
                .add(("X-Content-Type-Options", "nosniff"))
                .add(("X-Frame-Options", "DENY"))
                .add(("X-XSS-Protection", "1; mode=block"))
            )
            .route("/", web::get().to(handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn handler() -> HttpResponse {
    HttpResponse::Ok().json(json!({"message": "Hello World"}))
}

// Custom CSP middleware
use actix_web::{dev::ServiceRequest, dev::ServiceResponse, Error, middleware::Service};
use futures::future::{ok, Ready};

pub struct CSPMiddleware {
    policy: String,
}

impl CSPMiddleware {
    pub fn new(policy: &str) -> Self {
        Self {
            policy: policy.to_string(),
        }
    }
}

impl<S, B> Service<ServiceRequest> for CSPMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Ready<Result<Self::Response, Self::Error>>;

    fn poll_ready(&self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let mut res = ServiceResponse::new(req.into_parts().0, HttpResponse::Ok());
        res.headers_mut().insert(
            "Content-Security-Policy",
            self.policy.parse().unwrap(),
        );
        ok(res)
    }
}
```

---

## 🔐 Authentication & Authorization

### JWT Implementation

#### Node.js JWT

```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

class AuthService {
    constructor(secret) {
        this.secret = secret;
    }
    
    async hashPassword(password) {
        return await bcrypt.hash(password, 10);
    }
    
    async verifyPassword(password, hash) {
        return await bcrypt.compare(password, hash);
    }
    
    generateToken(payload) {
        return jwt.sign(payload, this.secret, { expiresIn: '1h' });
    }
    
    verifyToken(token) {
        try {
            return jwt.verify(token, this.secret);
        } catch (error) {
            throw new Error('Invalid token');
        }
    }
}

// Middleware
const authMiddleware = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }
    
    try {
        const decoded = authService.verifyToken(token);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
};
```

#### Go JWT

```go
package main

import (
    "github.com/golang-jwt/jwt/v4"
    "golang.org/x/crypto/bcrypt"
    "time"
)

type Claims struct {
    UserID string `json:"user_id"`
    Email  string `json:"email"`
    jwt.RegisteredClaims
}

type AuthService struct {
    secret []byte
}

func NewAuthService(secret string) *AuthService {
    return &AuthService{
        secret: []byte(secret),
    }
}

func (a *AuthService) HashPassword(password string) (string, error) {
    hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    return string(hash), err
}

func (a *AuthService) VerifyPassword(password, hash string) bool {
    err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
    return err == nil
}

func (a *AuthService) GenerateToken(userID, email string) (string, error) {
    claims := Claims{
        UserID: userID,
        Email:  email,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(a.secret)
}

func (a *AuthService) VerifyToken(tokenString string) (*Claims, error) {
    claims := &Claims{}
    token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
        return a.secret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if !token.Valid {
        return nil, jwt.ErrTokenInvalid
    }
    
    return claims, nil
}
```

#### Rust JWT

```rust
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    user_id: String,
    email: String,
    exp: usize,
}

impl Claims {
    fn new(user_id: String, email: String) -> Self {
        Self {
            user_id,
            email,
            exp: (Utc::now() + Duration::hours(1)).timestamp() as usize,
        }
    }
}

pub struct AuthService {
    secret: String,
}

impl AuthService {
    pub fn new(secret: String) -> Self {
        Self { secret }
    }
    
    pub fn generate_token(&self, user_id: String, email: String) -> Result<String, jsonwebtoken::errors::Error> {
        let claims = Claims::new(user_id, email);
        encode(&Header::default(), &claims, &EncodingKey::from_secret(self.secret.as_ref()))
    }
    
    pub fn verify_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let validation = Validation::new(Algorithm::HS256);
        let token_data = decode::<Claims>(token, &DecodingKey::from_secret(self.secret.as_ref()), &validation)?;
        Ok(token_data.claims)
    }
}
```

---

## 🛡️ Security Headers

### Complete Security Headers Implementation

#### Node.js

```javascript
const helmet = require('helmet');

app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"],
        }
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    },
    noSniff: true,
    frameguard: { action: 'deny' },
    xssFilter: true,
    referrerPolicy: { policy: 'strict-origin-when-cross-origin' }
}));
```

#### Go

```go
func SecurityHeaders() func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
            w.Header().Set("X-Content-Type-Options", "nosniff")
            w.Header().Set("X-Frame-Options", "DENY")
            w.Header().Set("X-XSS-Protection", "1; mode=block")
            w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
            w.Header().Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
            
            next.ServeHTTP(w, r)
        })
    }
}
```

#### Rust

```rust
use actix_web::{middleware::DefaultHeaders, App, HttpServer};

App::new()
    .wrap(DefaultHeaders::new()
        .add(("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"))
        .add(("X-Content-Type-Options", "nosniff"))
        .add(("X-Frame-Options", "DENY"))
        .add(("X-XSS-Protection", "1; mode=block"))
        .add(("Referrer-Policy", "strict-origin-when-cross-origin"))
        .add(("Permissions-Policy", "geolocation=(), microphone=(), camera=()"))
    )
```

---

## 🚀 Rate Limiting

### Node.js Rate Limiting

```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP',
    standardHeaders: true,
    legacyHeaders: false,
});

app.use('/api/', limiter);
```

### Go Rate Limiting

```go
import "golang.org/x/time/rate"

type RateLimiter struct {
    limiters map[string]*rate.Limiter
    mu       sync.RWMutex
    rate     rate.Limit
    burst    int
}

func NewRateLimiter(r rate.Limit, b int) *RateLimiter {
    return &RateLimiter{
        limiters: make(map[string]*rate.Limiter),
        rate:     r,
        burst:    b,
    }
}

func (rl *RateLimiter) GetLimiter(ip string) *rate.Limiter {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    limiter, exists := rl.limiters[ip]
    if !exists {
        limiter = rate.NewLimiter(rl.rate, rl.burst)
        rl.limiters[ip] = limiter
    }
    
    return limiter
}
```

### Rust Rate Limiting

```rust
use actix_web::{web, App, HttpServer, middleware};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    limit: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(limit: usize, window: Duration) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            limit,
            window,
        }
    }
    
    fn is_allowed(&self, key: &str) -> bool {
        let mut requests = self.requests.lock().unwrap();
        let now = Instant::now();
        
        let entry = requests.entry(key.to_string()).or_insert_with(Vec::new);
        entry.retain(|&time| now.duration_since(time) < self.window);
        
        if entry.len() < self.limit {
            entry.push(now);
            true
        } else {
            false
        }
    }
}
```

---

## 📋 Security Best Practices

### 1. Input Validation
- Validate all inputs on both client and server
- Use whitelist validation when possible
- Sanitize data before processing
- Use parameterized queries to prevent SQL injection

### 2. Authentication
- Use strong password policies
- Implement multi-factor authentication
- Use secure session management
- Implement proper logout functionality

### 3. Authorization
- Implement principle of least privilege
- Use role-based access control (RBAC)
- Validate permissions on every request
- Implement proper session timeout

### 4. Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement proper key management
- Regular security audits

### 5. Monitoring & Logging
- Log all security events
- Monitor for suspicious activities
- Implement intrusion detection
- Regular security assessments

---

**🔒 Implement these security practices to protect your web applications from common vulnerabilities and attacks! 🛡️**


##  Input Validation  Sanitization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-input-validation--sanitization -->

Placeholder content. Please replace with proper section.


##  Session Management

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-session-management -->

Placeholder content. Please replace with proper section.


##  Https  Tls

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-https--tls -->

Placeholder content. Please replace with proper section.
