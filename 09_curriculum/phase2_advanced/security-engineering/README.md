# Security Engineering

## Table of Contents

1. [Overview](#overview)
2. [Cryptography](#cryptography)
3. [Authentication and Authorization](#authentication-and-authorization)
4. [Network Security](#network-security)
5. [Application Security](#application-security)
6. [Security Monitoring](#security-monitoring)
7. [Secure Coding Practices](#secure-coding-practices)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master cryptographic algorithms and implementations
- Implement secure authentication and authorization
- Design network security architectures
- Apply application security best practices
- Monitor and respond to security threats
- Write secure code and prevent vulnerabilities

### What is Security Engineering?

Security Engineering involves designing, implementing, and maintaining secure systems to protect against threats, vulnerabilities, and attacks.

## Cryptography

### 1. Symmetric Encryption

#### AES Implementation
```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "io"
)

func encryptAES(plaintext, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    // Create GCM mode
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    // Create nonce
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    // Encrypt
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return ciphertext, nil
}

func decryptAES(ciphertext, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    return gcm.Open(nil, nonce, ciphertext, nil)
}

func main() {
    key := []byte("32-byte-long-key-for-AES-256!")
    plaintext := []byte("Hello, World!")
    
    // Encrypt
    ciphertext, err := encryptAES(plaintext, key)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Encrypted: %s\n", base64.StdEncoding.EncodeToString(ciphertext))
    
    // Decrypt
    decrypted, err := decryptAES(ciphertext, key)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Decrypted: %s\n", string(decrypted))
}
```

### 2. Asymmetric Encryption

#### RSA Implementation
```go
package main

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/base64"
    "encoding/pem"
    "fmt"
)

func generateRSAKeyPair() (*rsa.PrivateKey, *rsa.PublicKey, error) {
    privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil {
        return nil, nil, err
    }
    
    return privateKey, &privateKey.PublicKey, nil
}

func encryptRSA(plaintext []byte, publicKey *rsa.PublicKey) ([]byte, error) {
    return rsa.EncryptOAEP(sha256.New(), rand.Reader, publicKey, plaintext, nil)
}

func decryptRSA(ciphertext []byte, privateKey *rsa.PrivateKey) ([]byte, error) {
    return rsa.DecryptOAEP(sha256.New(), rand.Reader, privateKey, ciphertext, nil)
}

func signRSA(message []byte, privateKey *rsa.PrivateKey) ([]byte, error) {
    hashed := sha256.Sum256(message)
    return rsa.SignPKCS1v15(rand.Reader, privateKey, crypto.SHA256, hashed[:])
}

func verifyRSA(message, signature []byte, publicKey *rsa.PublicKey) error {
    hashed := sha256.Sum256(message)
    return rsa.VerifyPKCS1v15(publicKey, crypto.SHA256, hashed[:], signature)
}

func main() {
    // Generate key pair
    privateKey, publicKey, err := generateRSAKeyPair()
    if err != nil {
        panic(err)
    }
    
    message := []byte("Hello, RSA!")
    
    // Encrypt
    ciphertext, err := encryptRSA(message, publicKey)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Encrypted: %s\n", base64.StdEncoding.EncodeToString(ciphertext))
    
    // Decrypt
    decrypted, err := decryptRSA(ciphertext, privateKey)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Decrypted: %s\n", string(decrypted))
    
    // Sign
    signature, err := signRSA(message, privateKey)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Signature: %s\n", base64.StdEncoding.EncodeToString(signature))
    
    // Verify
    err = verifyRSA(message, signature, publicKey)
    if err != nil {
        fmt.Printf("Verification failed: %v\n", err)
    } else {
        fmt.Println("Verification successful")
    }
}
```

## Authentication and Authorization

### 1. JWT Implementation

#### JWT Token Management
```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "time"
    
    "github.com/golang-jwt/jwt/v4"
)

type Claims struct {
    UserID   string `json:"user_id"`
    Username string `json:"username"`
    Role     string `json:"role"`
    jwt.RegisteredClaims
}

type JWTManager struct {
    secretKey []byte
}

func NewJWTManager() *JWTManager {
    secretKey := make([]byte, 32)
    rand.Read(secretKey)
    return &JWTManager{secretKey: secretKey}
}

func (jm *JWTManager) GenerateToken(userID, username, role string) (string, error) {
    claims := Claims{
        UserID:   userID,
        Username: username,
        Role:     role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(jm.secretKey)
}

func (jm *JWTManager) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return jm.secretKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}

func (jm *JWTManager) RefreshToken(tokenString string) (string, error) {
    claims, err := jm.ValidateToken(tokenString)
    if err != nil {
        return "", err
    }
    
    // Generate new token with extended expiration
    return jm.GenerateToken(claims.UserID, claims.Username, claims.Role)
}

func main() {
    jm := NewJWTManager()
    
    // Generate token
    token, err := jm.GenerateToken("123", "john_doe", "admin")
    if err != nil {
        panic(err)
    }
    fmt.Printf("Generated token: %s\n", token)
    
    // Validate token
    claims, err := jm.ValidateToken(token)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Token claims: %+v\n", claims)
    
    // Refresh token
    newToken, err := jm.RefreshToken(token)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Refreshed token: %s\n", newToken)
}
```

### 2. OAuth 2.0 Implementation

#### OAuth 2.0 Server
```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "net/http"
    "time"
)

type OAuthServer struct {
    clients    map[string]*Client
    authCodes  map[string]*AuthCode
    tokens     map[string]*AccessToken
}

type Client struct {
    ID          string
    Secret      string
    RedirectURI string
    Scopes      []string
}

type AuthCode struct {
    Code        string
    ClientID    string
    UserID      string
    RedirectURI string
    Scopes      []string
    ExpiresAt   time.Time
}

type AccessToken struct {
    Token     string
    ClientID  string
    UserID    string
    Scopes    []string
    ExpiresAt time.Time
}

func NewOAuthServer() *OAuthServer {
    return &OAuthServer{
        clients:   make(map[string]*Client),
        authCodes: make(map[string]*AuthCode),
        tokens:    make(map[string]*AccessToken),
    }
}

func (os *OAuthServer) RegisterClient(client *Client) {
    os.clients[client.ID] = client
}

func (os *OAuthServer) Authorize(w http.ResponseWriter, r *http.Request) {
    clientID := r.URL.Query().Get("client_id")
    redirectURI := r.URL.Query().Get("redirect_uri")
    responseType := r.URL.Query().Get("response_type")
    scope := r.URL.Query().Get("scope")
    
    if responseType != "code" {
        http.Error(w, "unsupported_response_type", http.StatusBadRequest)
        return
    }
    
    client, exists := os.clients[clientID]
    if !exists || client.RedirectURI != redirectURI {
        http.Error(w, "invalid_client", http.StatusBadRequest)
        return
    }
    
    // Generate authorization code
    code := generateRandomString(32)
    authCode := &AuthCode{
        Code:        code,
        ClientID:    clientID,
        UserID:      "user123", // In real implementation, get from session
        RedirectURI: redirectURI,
        Scopes:      []string{scope},
        ExpiresAt:   time.Now().Add(10 * time.Minute),
    }
    
    os.authCodes[code] = authCode
    
    // Redirect with authorization code
    redirectURL := fmt.Sprintf("%s?code=%s&state=%s", redirectURI, code, r.URL.Query().Get("state"))
    http.Redirect(w, r, redirectURL, http.StatusFound)
}

func (os *OAuthServer) Token(w http.ResponseWriter, r *http.Request) {
    grantType := r.FormValue("grant_type")
    
    if grantType != "authorization_code" {
        http.Error(w, "unsupported_grant_type", http.StatusBadRequest)
        return
    }
    
    code := r.FormValue("code")
    clientID := r.FormValue("client_id")
    clientSecret := r.FormValue("client_secret")
    redirectURI := r.FormValue("redirect_uri")
    
    // Validate client
    client, exists := os.clients[clientID]
    if !exists || client.Secret != clientSecret {
        http.Error(w, "invalid_client", http.StatusBadRequest)
        return
    }
    
    // Validate authorization code
    authCode, exists := os.authCodes[code]
    if !exists || authCode.ClientID != clientID || authCode.RedirectURI != redirectURI {
        http.Error(w, "invalid_grant", http.StatusBadRequest)
        return
    }
    
    if time.Now().After(authCode.ExpiresAt) {
        http.Error(w, "invalid_grant", http.StatusBadRequest)
        return
    }
    
    // Generate access token
    token := generateRandomString(32)
    accessToken := &AccessToken{
        Token:     token,
        ClientID:  clientID,
        UserID:    authCode.UserID,
        Scopes:    authCode.Scopes,
        ExpiresAt: time.Now().Add(1 * time.Hour),
    }
    
    os.tokens[token] = accessToken
    delete(os.authCodes, code)
    
    // Return token response
    w.Header().Set("Content-Type", "application/json")
    fmt.Fprintf(w, `{"access_token":"%s","token_type":"Bearer","expires_in":3600}`, token)
}

func generateRandomString(length int) string {
    bytes := make([]byte, length)
    rand.Read(bytes)
    return base64.URLEncoding.EncodeToString(bytes)
}

func main() {
    server := NewOAuthServer()
    
    // Register a client
    client := &Client{
        ID:          "test_client",
        Secret:      "test_secret",
        RedirectURI: "http://localhost:8080/callback",
        Scopes:      []string{"read", "write"},
    }
    server.RegisterClient(client)
    
    http.HandleFunc("/authorize", server.Authorize)
    http.HandleFunc("/token", server.Token)
    
    fmt.Println("OAuth server running on :8080")
    http.ListenAndServe(":8080", nil)
}
```

## Network Security

### 1. TLS Implementation

#### TLS Server
```go
package main

import (
    "crypto/tls"
    "fmt"
    "io"
    "net/http"
    "time"
)

func createTLSConfig() *tls.Config {
    return &tls.Config{
        MinVersion:               tls.VersionTLS12,
        CurvePreferences:         []tls.CurveID{tls.CurveP521, tls.CurveP384, tls.CurveP256},
        PreferServerCipherSuites: true,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
            tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        },
    }
}

func secureHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    w.Header().Set("X-Content-Type-Options", "nosniff")
    w.Header().Set("X-Frame-Options", "DENY")
    w.Header().Set("X-XSS-Protection", "1; mode=block")
    
    fmt.Fprintf(w, "Secure connection established!\n")
    fmt.Fprintf(w, "Protocol: %s\n", r.Proto)
    fmt.Fprintf(w, "TLS Version: %x\n", r.TLS.Version)
    fmt.Fprintf(w, "Cipher Suite: %x\n", r.TLS.CipherSuite)
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/", secureHandler)
    
    server := &http.Server{
        Addr:         ":8443",
        Handler:      mux,
        TLSConfig:    createTLSConfig(),
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    fmt.Println("TLS server running on :8443")
    fmt.Println("Test with: curl -k https://localhost:8443/")
    
    // In production, use real certificates
    err := server.ListenAndServeTLS("server.crt", "server.key")
    if err != nil {
        fmt.Printf("Server failed: %v\n", err)
    }
}
```

### 2. Rate Limiting

#### Rate Limiter Implementation
```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

type RateLimiter struct {
    requests map[string][]time.Time
    mutex    sync.RWMutex
    limit    int
    window   time.Duration
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        requests: make(map[string][]time.Time),
        limit:    limit,
        window:   window,
    }
}

func (rl *RateLimiter) Allow(clientID string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-rl.window)
    
    // Clean old requests
    if requests, exists := rl.requests[clientID]; exists {
        var validRequests []time.Time
        for _, reqTime := range requests {
            if reqTime.After(cutoff) {
                validRequests = append(validRequests, reqTime)
            }
        }
        rl.requests[clientID] = validRequests
    }
    
    // Check if under limit
    if len(rl.requests[clientID]) < rl.limit {
        rl.requests[clientID] = append(rl.requests[clientID], now)
        return true
    }
    
    return false
}

func (rl *RateLimiter) Middleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        clientID := r.RemoteAddr // In production, use proper client identification
        
        if !rl.Allow(clientID) {
            http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
            return
        }
        
        next(w, r)
    }
}

func main() {
    // Create rate limiter: 10 requests per minute
    limiter := NewRateLimiter(10, time.Minute)
    
    handler := func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Request processed for %s\n", r.RemoteAddr)
    }
    
    http.HandleFunc("/", limiter.Middleware(handler))
    
    fmt.Println("Rate-limited server running on :8080")
    http.ListenAndServe(":8080", nil)
}
```

## Application Security

### 1. Input Validation

#### Secure Input Handler
```go
package main

import (
    "fmt"
    "html/template"
    "net/http"
    "regexp"
    "strings"
)

type InputValidator struct {
    emailRegex    *regexp.Regexp
    usernameRegex *regexp.Regexp
}

func NewInputValidator() *InputValidator {
    return &InputValidator{
        emailRegex:    regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`),
        usernameRegex: regexp.MustCompile(`^[a-zA-Z0-9_-]{3,20}$`),
    }
}

func (iv *InputValidator) ValidateEmail(email string) error {
    if len(email) > 254 {
        return fmt.Errorf("email too long")
    }
    
    if !iv.emailRegex.MatchString(email) {
        return fmt.Errorf("invalid email format")
    }
    
    return nil
}

func (iv *InputValidator) ValidateUsername(username string) error {
    if len(username) < 3 || len(username) > 20 {
        return fmt.Errorf("username must be 3-20 characters")
    }
    
    if !iv.usernameRegex.MatchString(username) {
        return fmt.Errorf("username contains invalid characters")
    }
    
    return nil
}

func (iv *InputValidator) SanitizeHTML(input string) string {
    // Remove potentially dangerous HTML tags
    dangerousTags := []string{"<script", "</script", "<iframe", "</iframe", "<object", "</object"}
    sanitized := input
    
    for _, tag := range dangerousTags {
        sanitized = strings.ReplaceAll(sanitized, tag, "")
    }
    
    return sanitized
}

func (iv *InputValidator) ValidatePassword(password string) error {
    if len(password) < 8 {
        return fmt.Errorf("password must be at least 8 characters")
    }
    
    hasUpper := regexp.MustCompile(`[A-Z]`).MatchString(password)
    hasLower := regexp.MustCompile(`[a-z]`).MatchString(password)
    hasDigit := regexp.MustCompile(`[0-9]`).MatchString(password)
    hasSpecial := regexp.MustCompile(`[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]`).MatchString(password)
    
    if !hasUpper || !hasLower || !hasDigit || !hasSpecial {
        return fmt.Errorf("password must contain uppercase, lowercase, digit, and special character")
    }
    
    return nil
}

func main() {
    validator := NewInputValidator()
    
    // Test validation
    testCases := []struct {
        input string
        fn    func(string) error
    }{
        {"user@example.com", validator.ValidateEmail},
        {"john_doe123", validator.ValidateUsername},
        {"Password123!", validator.ValidatePassword},
    }
    
    for _, tc := range testCases {
        if err := tc.fn(tc.input); err != nil {
            fmt.Printf("Validation failed for '%s': %v\n", tc.input, err)
        } else {
            fmt.Printf("Validation passed for '%s'\n", tc.input)
        }
    }
}
```

### 2. SQL Injection Prevention

#### Parameterized Queries
```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    
    _ "github.com/lib/pq"
)

type UserService struct {
    db *sql.DB
}

func NewUserService(dsn string) (*UserService, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    return &UserService{db: db}, nil
}

// Secure method using parameterized queries
func (us *UserService) GetUserByID(id int) (*User, error) {
    query := "SELECT id, username, email FROM users WHERE id = $1"
    row := us.db.QueryRow(query, id)
    
    user := &User{}
    err := row.Scan(&user.ID, &user.Username, &user.Email)
    if err != nil {
        return nil, err
    }
    
    return user, nil
}

// Secure method for user search
func (us *UserService) SearchUsers(username string) ([]*User, error) {
    query := "SELECT id, username, email FROM users WHERE username ILIKE $1"
    rows, err := us.db.Query(query, "%"+username+"%")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []*User
    for rows.Next() {
        user := &User{}
        if err := rows.Scan(&user.ID, &user.Username, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, nil
}

// Secure method for user creation
func (us *UserService) CreateUser(username, email, password string) error {
    query := "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)"
    _, err := us.db.Exec(query, username, email, password)
    return err
}

type User struct {
    ID       int
    Username string
    Email    string
}

func main() {
    dsn := "user=postgres password=password dbname=test sslmode=disable"
    userService, err := NewUserService(dsn)
    if err != nil {
        log.Fatal(err)
    }
    
    // Example usage
    user, err := userService.GetUserByID(1)
    if err != nil {
        log.Printf("Error getting user: %v", err)
    } else {
        fmt.Printf("User: %+v\n", user)
    }
}
```

## Security Monitoring

### 1. Intrusion Detection

#### Simple IDS
```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "regexp"
    "sync"
    "time"
)

type SecurityEvent struct {
    Timestamp time.Time
    IP        string
    Event     string
    Severity  string
}

type IntrusionDetectionSystem struct {
    events     []SecurityEvent
    mutex      sync.RWMutex
    patterns   []*regexp.Regexp
    rateLimits map[string][]time.Time
}

func NewIDS() *IntrusionDetectionSystem {
    return &IntrusionDetectionSystem{
        events:   make([]SecurityEvent, 0),
        patterns: []*regexp.Regexp{
            regexp.MustCompile(`(?i)(union|select|insert|update|delete|drop)`),
            regexp.MustCompile(`(?i)(script|javascript|vbscript)`),
            regexp.MustCompile(`(?i)(<script|</script|onload|onerror)`),
            regexp.MustCompile(`(?i)(admin|administrator|root|sa)`),
        },
        rateLimits: make(map[string][]time.Time),
    }
}

func (ids *IntrusionDetectionSystem) CheckRequest(r *http.Request) []SecurityEvent {
    var events []SecurityEvent
    ip := r.RemoteAddr
    url := r.URL.String()
    
    // Check for SQL injection patterns
    for i, pattern := range ids.patterns {
        if pattern.MatchString(url) {
            event := SecurityEvent{
                Timestamp: time.Now(),
                IP:        ip,
                Event:     fmt.Sprintf("SQL injection attempt detected in URL: %s", url),
                Severity:  "HIGH",
            }
            events = append(events, event)
            ids.logEvent(event)
        }
    }
    
    // Check for XSS patterns
    if regexp.MustCompile(`(?i)(<script|</script|onload|onerror)`).MatchString(url) {
        event := SecurityEvent{
            Timestamp: time.Now(),
            IP:        ip,
            Event:     fmt.Sprintf("XSS attempt detected in URL: %s", url),
            Severity:  "HIGH",
        }
        events = append(events, event)
        ids.logEvent(event)
    }
    
    // Check rate limiting
    if ids.isRateLimited(ip) {
        event := SecurityEvent{
            Timestamp: time.Now(),
            IP:        ip,
            Event:     "Rate limit exceeded",
            Severity:  "MEDIUM",
        }
        events = append(events, event)
        ids.logEvent(event)
    }
    
    return events
}

func (ids *IntrusionDetectionSystem) isRateLimited(ip string) bool {
    now := time.Now()
    cutoff := now.Add(-1 * time.Minute)
    
    // Clean old requests
    var validRequests []time.Time
    for _, reqTime := range ids.rateLimits[ip] {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    ids.rateLimits[ip] = validRequests
    
    // Check if over limit (10 requests per minute)
    if len(ids.rateLimits[ip]) >= 10 {
        return true
    }
    
    // Add current request
    ids.rateLimits[ip] = append(ids.rateLimits[ip], now)
    return false
}

func (ids *IntrusionDetectionSystem) logEvent(event SecurityEvent) {
    ids.mutex.Lock()
    defer ids.mutex.Unlock()
    
    ids.events = append(ids.events, event)
    log.Printf("SECURITY EVENT: %s - %s - %s", event.Severity, event.IP, event.Event)
}

func (ids *IntrusionDetectionSystem) GetEvents() []SecurityEvent {
    ids.mutex.RLock()
    defer ids.mutex.RUnlock()
    
    return append([]SecurityEvent(nil), ids.events...)
}

func main() {
    ids := NewIDS()
    
    // Create secure handler
    handler := func(w http.ResponseWriter, r *http.Request) {
        // Check for security threats
        events := ids.CheckRequest(r)
        
        if len(events) > 0 {
            http.Error(w, "Request blocked due to security concerns", http.StatusForbidden)
            return
        }
        
        fmt.Fprintf(w, "Request processed successfully")
    }
    
    http.HandleFunc("/", handler)
    
    fmt.Println("Security monitoring enabled on :8080")
    http.ListenAndServe(":8080", nil)
}
```

## Follow-up Questions

### 1. Cryptography
**Q: What's the difference between symmetric and asymmetric encryption?**
A: Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses different keys (public/private key pair).

### 2. Authentication
**Q: What are the benefits of JWT over session-based authentication?**
A: JWT is stateless, scalable, and can be used across multiple services without server-side storage.

### 3. Application Security
**Q: How do you prevent SQL injection attacks?**
A: Use parameterized queries, input validation, least privilege database access, and regular security testing.

## Sources

### Books
- **Cryptography and Network Security** by William Stallings
- **Web Application Security** by Andrew Hoffman
- **The Web Application Hacker's Handbook** by Stuttard and Pinto

### Online Resources
- **OWASP Top 10** - Web application security risks
- **NIST Cybersecurity Framework** - Security guidelines
- **CIS Controls** - Critical security controls

## Projects

### 1. Security Framework
**Objective**: Build a comprehensive security framework
**Requirements**: Authentication, authorization, monitoring, encryption
**Deliverables**: Complete security framework

### 2. Vulnerability Scanner
**Objective**: Create a web application vulnerability scanner
**Requirements**: SQL injection, XSS, CSRF detection
**Deliverables**: Production-ready scanner

### 3. Security Monitoring System
**Objective**: Develop a real-time security monitoring system
**Requirements**: Intrusion detection, alerting, reporting
**Deliverables**: Complete monitoring platform

---

**Next**: [Phase 3: Expert](../../../README.md) | **Previous**: [Performance Engineering](../../../README.md) | **Up**: [Phase 2](README.md)

