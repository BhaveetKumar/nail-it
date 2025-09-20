# 🔐 Security Engineering Guide

> **Comprehensive security engineering for backend systems and interview preparation**

## 🎯 **Overview**

Security engineering is fundamental for backend systems, especially in fintech, healthcare, and enterprise environments. This guide covers authentication, authorization, encryption, secure coding practices, and compliance frameworks with practical implementations for senior engineering interviews.

## 📚 **Table of Contents**

1. [Authentication Systems](#authentication-systems)
2. [Authorization Patterns](#authorization-patterns)
3. [Encryption & Cryptography](#encryption--cryptography)
4. [Secure Coding Practices](#secure-coding-practices)
5. [OWASP Top 10](#owasp-top-10)
6. [Security Testing](#security-testing)
7. [Compliance Frameworks](#compliance-frameworks)
8. [API Security](#api-security)
9. [Infrastructure Security](#infrastructure-security)
10. [Security Monitoring](#security-monitoring)
11. [Interview Questions](#interview-questions)

---

## 🔑 **Authentication Systems**

### **JWT Implementation**

```go
package auth

import (
    "crypto/rand"
    "fmt"
    "time"
    "github.com/golang-jwt/jwt/v5"
    "golang.org/x/crypto/bcrypt"
)

// JWT Claims structure
type Claims struct {
    UserID   string   `json:"user_id"`
    Email    string   `json:"email"`
    Roles    []string `json:"roles"`
    jwt.RegisteredClaims
}

// JWT Manager
type JWTManager struct {
    secretKey     []byte
    tokenDuration time.Duration
}

func NewJWTManager(secretKey string, tokenDuration time.Duration) *JWTManager {
    return &JWTManager{
        secretKey:     []byte(secretKey),
        tokenDuration: tokenDuration,
    }
}

// Generate JWT token
func (manager *JWTManager) Generate(userID, email string, roles []string) (string, error) {
    claims := &Claims{
        UserID: userID,
        Email:  email,
        Roles:  roles,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(manager.tokenDuration)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
            Issuer:    "payment-service",
            Subject:   userID,
            ID:        generateJTI(),
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(manager.secretKey)
}

// Verify JWT token
func (manager *JWTManager) Verify(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(
        tokenString,
        &Claims{},
        func(token *jwt.Token) (interface{}, error) {
            if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
                return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
            }
            return manager.secretKey, nil
        },
    )

    if err != nil {
        return nil, fmt.Errorf("invalid token: %w", err)
    }

    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, fmt.Errorf("invalid token claims")
    }

    return claims, nil
}

// Generate unique JWT ID
func generateJTI() string {
    bytes := make([]byte, 16)
    rand.Read(bytes)
    return fmt.Sprintf("%x", bytes)
}

// User service with authentication
type UserService struct {
    jwtManager *JWTManager
    userRepo   UserRepository
}

func (s *UserService) Login(email, password string) (*LoginResponse, error) {
    user, err := s.userRepo.FindByEmail(email)
    if err != nil {
        return nil, fmt.Errorf("user not found: %w", err)
    }

    // Verify password
    if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
        return nil, fmt.Errorf("invalid credentials")
    }

    // Generate JWT
    token, err := s.jwtManager.Generate(user.ID, user.Email, user.Roles)
    if err != nil {
        return nil, fmt.Errorf("token generation failed: %w", err)
    }

    return &LoginResponse{
        Token:  token,
        User:   user,
        ExpiresAt: time.Now().Add(s.jwtManager.tokenDuration),
    }, nil
}

// Password hashing utilities
func HashPassword(password string) (string, error) {
    bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    return string(bytes), err
}

func CheckPasswordHash(password, hash string) bool {
    err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
    return err == nil
}
```

### **OAuth 2.0 Implementation**

```go
package oauth

import (
    "context"
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "net/http"
    "net/url"
    "time"
)

// OAuth2 Configuration
type OAuth2Config struct {
    ClientID     string
    ClientSecret string
    RedirectURL  string
    Scopes       []string
    AuthURL      string
    TokenURL     string
}

// OAuth2 Token
type OAuth2Token struct {
    AccessToken  string    `json:"access_token"`
    TokenType    string    `json:"token_type"`
    RefreshToken string    `json:"refresh_token"`
    ExpiresIn    int       `json:"expires_in"`
    ExpiresAt    time.Time `json:"expires_at"`
    Scope        string    `json:"scope"`
}

// OAuth2 Client
type OAuth2Client struct {
    config *OAuth2Config
    client *http.Client
}

func NewOAuth2Client(config *OAuth2Config) *OAuth2Client {
    return &OAuth2Client{
        config: config,
        client: &http.Client{Timeout: 30 * time.Second},
    }
}

// Generate authorization URL
func (c *OAuth2Client) GetAuthURL(state string) string {
    params := url.Values{
        "response_type": {"code"},
        "client_id":     {c.config.ClientID},
        "redirect_uri":  {c.config.RedirectURL},
        "scope":         {joinScopes(c.config.Scopes)},
        "state":         {state},
    }

    return c.config.AuthURL + "?" + params.Encode()
}

// Exchange authorization code for token
func (c *OAuth2Client) ExchangeCode(ctx context.Context, code string) (*OAuth2Token, error) {
    data := url.Values{
        "grant_type":   {"authorization_code"},
        "code":         {code},
        "redirect_uri": {c.config.RedirectURL},
        "client_id":    {c.config.ClientID},
        "client_secret": {c.config.ClientSecret},
    }

    resp, err := c.client.PostForm(c.config.TokenURL, data)
    if err != nil {
        return nil, fmt.Errorf("token exchange failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("token exchange returned status: %d", resp.StatusCode)
    }

    var token OAuth2Token
    if err := json.NewDecoder(resp.Body).Decode(&token); err != nil {
        return nil, fmt.Errorf("failed to decode token response: %w", err)
    }

    // Calculate expiration time
    token.ExpiresAt = time.Now().Add(time.Duration(token.ExpiresIn) * time.Second)

    return &token, nil
}

// Refresh access token
func (c *OAuth2Client) RefreshToken(ctx context.Context, refreshToken string) (*OAuth2Token, error) {
    data := url.Values{
        "grant_type":    {"refresh_token"},
        "refresh_token": {refreshToken},
        "client_id":     {c.config.ClientID},
        "client_secret": {c.config.ClientSecret},
    }

    resp, err := c.client.PostForm(c.config.TokenURL, data)
    if err != nil {
        return nil, fmt.Errorf("token refresh failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("token refresh returned status: %d", resp.StatusCode)
    }

    var token OAuth2Token
    if err := json.NewDecoder(resp.Body).Decode(&token); err != nil {
        return nil, fmt.Errorf("failed to decode refresh response: %w", err)
    }

    token.ExpiresAt = time.Now().Add(time.Duration(token.ExpiresIn) * time.Second)

    return &token, nil
}

// Generate secure state parameter
func GenerateState() (string, error) {
    bytes := make([]byte, 32)
    _, err := rand.Read(bytes)
    if err != nil {
        return "", err
    }
    return base64.URLEncoding.EncodeToString(bytes), nil
}

func joinScopes(scopes []string) string {
    result := ""
    for i, scope := range scopes {
        if i > 0 {
            result += " "
        }
        result += scope
    }
    return result
}
```

### **Multi-Factor Authentication (MFA)**

```go
package mfa

import (
    "crypto/rand"
    "crypto/hmac"
    "crypto/sha1"
    "encoding/base32"
    "encoding/binary"
    "fmt"
    "strings"
    "time"
)

// TOTP (Time-based One-Time Password) implementation
type TOTPManager struct {
    issuer string
    digits int
    period int64
}

func NewTOTPManager(issuer string) *TOTPManager {
    return &TOTPManager{
        issuer: issuer,
        digits: 6,
        period: 30,
    }
}

// Generate secret key for TOTP
func (t *TOTPManager) GenerateSecret() (string, error) {
    secret := make([]byte, 20)
    _, err := rand.Read(secret)
    if err != nil {
        return "", err
    }
    
    secretBase32 := base32.StdEncoding.EncodeToString(secret)
    return strings.TrimRight(secretBase32, "="), nil
}

// Generate TOTP URI for QR code
func (t *TOTPManager) GenerateURI(secret, accountName string) string {
    return fmt.Sprintf(
        "otpauth://totp/%s:%s?secret=%s&issuer=%s&digits=%d&period=%d",
        t.issuer,
        accountName,
        secret,
        t.issuer,
        t.digits,
        t.period,
    )
}

// Generate TOTP code
func (t *TOTPManager) GenerateCode(secret string, timestamp int64) (string, error) {
    if timestamp == 0 {
        timestamp = time.Now().Unix()
    }
    
    timeStep := timestamp / t.period
    
    secretBytes, err := base32.StdEncoding.DecodeString(secret + strings.Repeat("=", 8-len(secret)%8))
    if err != nil {
        return "", fmt.Errorf("invalid secret: %w", err)
    }
    
    // Convert time step to bytes
    timeBytes := make([]byte, 8)
    binary.BigEndian.PutUint64(timeBytes, uint64(timeStep))
    
    // HMAC-SHA1
    mac := hmac.New(sha1.New, secretBytes)
    mac.Write(timeBytes)
    hash := mac.Sum(nil)
    
    // Dynamic truncation
    offset := hash[len(hash)-1] & 0x0f
    code := binary.BigEndian.Uint32(hash[offset:offset+4]) & 0x7fffffff
    code = code % uint32(pow10(t.digits))
    
    return fmt.Sprintf("%0*d", t.digits, code), nil
}

// Verify TOTP code
func (t *TOTPManager) VerifyCode(secret, code string) bool {
    now := time.Now().Unix()
    
    // Check current time window and adjacent windows for clock skew
    for i := -1; i <= 1; i++ {
        timestamp := now + int64(i)*t.period
        expectedCode, err := t.GenerateCode(secret, timestamp)
        if err != nil {
            continue
        }
        
        if expectedCode == code {
            return true
        }
    }
    
    return false
}

// SMS-based MFA
type SMSMFAManager struct {
    smsService SMSService
    codeLength int
    expiration time.Duration
}

func NewSMSMFAManager(smsService SMSService) *SMSMFAManager {
    return &SMSMFAManager{
        smsService: smsService,
        codeLength: 6,
        expiration: 5 * time.Minute,
    }
}

func (s *SMSMFAManager) SendCode(phoneNumber string) (string, error) {
    code := s.generateSMSCode()
    
    message := fmt.Sprintf("Your verification code is: %s. This code expires in %d minutes.",
        code, int(s.expiration.Minutes()))
    
    if err := s.smsService.SendSMS(phoneNumber, message); err != nil {
        return "", fmt.Errorf("failed to send SMS: %w", err)
    }
    
    return code, nil
}

func (s *SMSMFAManager) generateSMSCode() string {
    const digits = "0123456789"
    code := make([]byte, s.codeLength)
    
    for i := 0; i < s.codeLength; i++ {
        randomBytes := make([]byte, 1)
        rand.Read(randomBytes)
        code[i] = digits[randomBytes[0]%10]
    }
    
    return string(code)
}

// MFA Service combining different methods
type MFAService struct {
    totpManager *TOTPManager
    smsManager  *SMSMFAManager
    userRepo    UserRepository
}

func NewMFAService(totpManager *TOTPManager, smsManager *SMSMFAManager, userRepo UserRepository) *MFAService {
    return &MFAService{
        totpManager: totpManager,
        smsManager:  smsManager,
        userRepo:    userRepo,
    }
}

func (m *MFAService) EnableTOTP(userID string) (*MFASetup, error) {
    user, err := m.userRepo.FindByID(userID)
    if err != nil {
        return nil, fmt.Errorf("user not found: %w", err)
    }
    
    secret, err := m.totpManager.GenerateSecret()
    if err != nil {
        return nil, fmt.Errorf("failed to generate secret: %w", err)
    }
    
    uri := m.totpManager.GenerateURI(secret, user.Email)
    
    // Store secret temporarily (user needs to verify before enabling)
    if err := m.userRepo.SetPendingTOTPSecret(userID, secret); err != nil {
        return nil, fmt.Errorf("failed to store secret: %w", err)
    }
    
    return &MFASetup{
        Secret: secret,
        URI:    uri,
        QRCode: generateQRCode(uri), // Generate QR code for easy setup
    }, nil
}

func (m *MFAService) VerifyAndEnableTOTP(userID, code string) error {
    pendingSecret, err := m.userRepo.GetPendingTOTPSecret(userID)
    if err != nil {
        return fmt.Errorf("no pending TOTP setup: %w", err)
    }
    
    if !m.totpManager.VerifyCode(pendingSecret, code) {
        return fmt.Errorf("invalid TOTP code")
    }
    
    // Enable TOTP for user
    if err := m.userRepo.EnableTOTP(userID, pendingSecret); err != nil {
        return fmt.Errorf("failed to enable TOTP: %w", err)
    }
    
    // Clean up pending secret
    m.userRepo.ClearPendingTOTPSecret(userID)
    
    return nil
}

func pow10(n int) int {
    result := 1
    for i := 0; i < n; i++ {
        result *= 10
    }
    return result
}

// Helper function to generate QR code (implementation would use a QR code library)
func generateQRCode(uri string) []byte {
    // Implementation would use a library like github.com/skip2/go-qrcode
    // to generate QR code image bytes
    return nil
}
```

---

## 🛡️ **Authorization Patterns**

### **Role-Based Access Control (RBAC)**

```go
package rbac

import (
    "fmt"
    "strings"
)

// Permission represents a specific permission
type Permission struct {
    ID          string `json:"id"`
    Name        string `json:"name"`
    Resource    string `json:"resource"`
    Action      string `json:"action"`
    Description string `json:"description"`
}

// Role represents a role with permissions
type Role struct {
    ID          string       `json:"id"`
    Name        string       `json:"name"`
    Permissions []Permission `json:"permissions"`
    Description string       `json:"description"`
}

// User with roles
type User struct {
    ID    string `json:"id"`
    Email string `json:"email"`
    Roles []Role `json:"roles"`
}

// RBAC Manager
type RBACManager struct {
    roles       map[string]Role
    permissions map[string]Permission
}

func NewRBACManager() *RBACManager {
    return &RBACManager{
        roles:       make(map[string]Role),
        permissions: make(map[string]Permission),
    }
}

// Define permissions
func (r *RBACManager) DefinePermission(id, name, resource, action, description string) {
    permission := Permission{
        ID:          id,
        Name:        name,
        Resource:    resource,
        Action:      action,
        Description: description,
    }
    r.permissions[id] = permission
}

// Define role
func (r *RBACManager) DefineRole(id, name, description string, permissionIDs []string) error {
    var permissions []Permission
    for _, permID := range permissionIDs {
        if perm, exists := r.permissions[permID]; exists {
            permissions = append(permissions, perm)
        } else {
            return fmt.Errorf("permission %s not found", permID)
        }
    }
    
    role := Role{
        ID:          id,
        Name:        name,
        Permissions: permissions,
        Description: description,
    }
    r.roles[id] = role
    return nil
}

// Check if user has permission
func (r *RBACManager) HasPermission(user User, resource, action string) bool {
    for _, role := range user.Roles {
        for _, permission := range role.Permissions {
            if permission.Resource == resource && permission.Action == action {
                return true
            }
            // Support wildcard permissions
            if permission.Resource == "*" || permission.Action == "*" {
                return true
            }
        }
    }
    return false
}

// HTTP Middleware for RBAC
func (r *RBACManager) RequirePermission(resource, action string) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
            // Extract user from context (set by authentication middleware)
            user, ok := req.Context().Value("user").(User)
            if !ok {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            if !r.HasPermission(user, resource, action) {
                http.Error(w, "Forbidden", http.StatusForbidden)
                return
            }
            
            next.ServeHTTP(w, req)
        })
    }
}

// Example usage setup
func SetupRBAC() *RBACManager {
    rbac := NewRBACManager()
    
    // Define permissions
    rbac.DefinePermission("user_read", "Read Users", "user", "read", "View user information")
    rbac.DefinePermission("user_write", "Write Users", "user", "write", "Create/update users")
    rbac.DefinePermission("payment_read", "Read Payments", "payment", "read", "View payments")
    rbac.DefinePermission("payment_write", "Write Payments", "payment", "write", "Process payments")
    rbac.DefinePermission("admin_all", "Admin All", "*", "*", "Full admin access")
    
    // Define roles
    rbac.DefineRole("viewer", "Viewer", "Read-only access", []string{"user_read", "payment_read"})
    rbac.DefineRole("operator", "Operator", "Operational access", []string{"user_read", "user_write", "payment_read"})
    rbac.DefineRole("admin", "Administrator", "Full access", []string{"admin_all"})
    
    return rbac
}
```

### **Attribute-Based Access Control (ABAC)**

```go
package abac

import (
    "context"
    "fmt"
    "time"
)

// Attribute represents an attribute used in access control
type Attribute struct {
    Name  string      `json:"name"`
    Value interface{} `json:"value"`
}

// Subject attributes (user making the request)
type Subject struct {
    ID         string            `json:"id"`
    Attributes map[string]interface{} `json:"attributes"`
}

// Resource being accessed
type Resource struct {
    ID         string            `json:"id"`
    Type       string            `json:"type"`
    Attributes map[string]interface{} `json:"attributes"`
}

// Action being performed
type Action struct {
    Name       string            `json:"name"`
    Attributes map[string]interface{} `json:"attributes"`
}

// Environment context
type Environment struct {
    Attributes map[string]interface{} `json:"attributes"`
}

// Access Request
type AccessRequest struct {
    Subject     Subject     `json:"subject"`
    Resource    Resource    `json:"resource"`
    Action      Action      `json:"action"`
    Environment Environment `json:"environment"`
}

// Policy Rule
type PolicyRule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Effect      string                 `json:"effect"` // "allow" or "deny"
    Condition   func(AccessRequest) bool `json:"-"`
    Description string                 `json:"description"`
}

// ABAC Policy Decision Point
type ABACManager struct {
    policies []PolicyRule
}

func NewABACManager() *ABACManager {
    return &ABACManager{
        policies: make([]PolicyRule, 0),
    }
}

// Add policy rule
func (a *ABACManager) AddPolicy(rule PolicyRule) {
    a.policies = append(a.policies, rule)
}

// Evaluate access request
func (a *ABACManager) Evaluate(request AccessRequest) bool {
    allowCount := 0
    denyCount := 0
    
    for _, policy := range a.policies {
        if policy.Condition(request) {
            switch policy.Effect {
            case "allow":
                allowCount++
            case "deny":
                denyCount++
                // Deny takes precedence
                return false
            }
        }
    }
    
    // Must have at least one allow and no denies
    return allowCount > 0 && denyCount == 0
}

// Example policy definitions
func SetupABACPolicies() *ABACManager {
    abac := NewABACManager()
    
    // Policy: Users can read their own data
    abac.AddPolicy(PolicyRule{
        ID:     "user_own_data_read",
        Name:   "Users can read their own data",
        Effect: "allow",
        Condition: func(req AccessRequest) bool {
            userID, ok := req.Subject.Attributes["user_id"].(string)
            if !ok {
                return false
            }
            
            resourceOwner, ok := req.Resource.Attributes["owner_id"].(string)
            if !ok {
                return false
            }
            
            return userID == resourceOwner && req.Action.Name == "read"
        },
        Description: "Allow users to read their own data",
    })
    
    // Policy: Admins can access everything
    abac.AddPolicy(PolicyRule{
        ID:     "admin_all_access",
        Name:   "Administrators have full access",
        Effect: "allow",
        Condition: func(req AccessRequest) bool {
            role, ok := req.Subject.Attributes["role"].(string)
            return ok && role == "admin"
        },
        Description: "Allow administrators full access",
    })
    
    // Policy: No access outside business hours except for admins
    abac.AddPolicy(PolicyRule{
        ID:     "business_hours_only",
        Name:   "Access only during business hours",
        Effect: "deny",
        Condition: func(req AccessRequest) bool {
            role, ok := req.Subject.Attributes["role"].(string)
            if ok && role == "admin" {
                return false // Don't apply this rule to admins
            }
            
            currentTime, ok := req.Environment.Attributes["current_time"].(time.Time)
            if !ok {
                return false
            }
            
            // Business hours: 9 AM to 6 PM, Monday to Friday
            hour := currentTime.Hour()
            weekday := currentTime.Weekday()
            
            isBusinessHours := hour >= 9 && hour < 18 && weekday >= time.Monday && weekday <= time.Friday
            return !isBusinessHours
        },
        Description: "Deny access outside business hours for non-admins",
    })
    
    // Policy: High-value transactions require manager approval
    abac.AddPolicy(PolicyRule{
        ID:     "high_value_transaction_approval",
        Name:   "High-value transactions require manager approval",
        Effect: "deny",
        Condition: func(req AccessRequest) bool {
            if req.Action.Name != "create_payment" {
                return false
            }
            
            amount, ok := req.Resource.Attributes["amount"].(float64)
            if !ok {
                return false
            }
            
            role, ok := req.Subject.Attributes["role"].(string)
            if !ok {
                return false
            }
            
            hasApproval, ok := req.Action.Attributes["manager_approved"].(bool)
            if !ok {
                hasApproval = false
            }
            
            // Deny high-value transactions without manager approval
            return amount > 10000 && role != "manager" && !hasApproval
        },
        Description: "Require manager approval for transactions over $10,000",
    })
    
    return abac
}

// HTTP Middleware for ABAC
func (a *ABACManager) RequireAccess() func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Build access request from HTTP request context
            subject := extractSubject(r)
            resource := extractResource(r)
            action := extractAction(r)
            environment := extractEnvironment(r)
            
            request := AccessRequest{
                Subject:     subject,
                Resource:    resource,
                Action:      action,
                Environment: environment,
            }
            
            if !a.Evaluate(request) {
                http.Error(w, "Access denied", http.StatusForbidden)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

// Helper functions to extract context from HTTP request
func extractSubject(r *http.Request) Subject {
    // Extract subject from JWT claims or session
    userID := r.Header.Get("X-User-ID")
    role := r.Header.Get("X-User-Role")
    
    return Subject{
        ID: userID,
        Attributes: map[string]interface{}{
            "user_id": userID,
            "role":    role,
        },
    }
}

func extractResource(r *http.Request) Resource {
    // Extract resource information from URL path and parameters
    resourceType := r.Header.Get("X-Resource-Type")
    resourceID := r.Header.Get("X-Resource-ID")
    
    return Resource{
        ID:   resourceID,
        Type: resourceType,
        Attributes: map[string]interface{}{
            "resource_id": resourceID,
            "owner_id":    r.Header.Get("X-Resource-Owner"),
        },
    }
}

func extractAction(r *http.Request) Action {
    actionName := strings.ToLower(r.Method)
    
    return Action{
        Name: actionName,
        Attributes: map[string]interface{}{
            "method": r.Method,
        },
    }
}

func extractEnvironment(r *http.Request) Environment {
    return Environment{
        Attributes: map[string]interface{}{
            "current_time": time.Now(),
            "ip_address":   r.RemoteAddr,
            "user_agent":   r.UserAgent(),
        },
    }
}
```

---

## 🔐 **Encryption & Cryptography**

### **AES Encryption Implementation**

```go
package encryption

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "fmt"
    "io"
)

// AES Encryption Manager
type AESManager struct {
    key []byte
}

func NewAESManager(password string) *AESManager {
    // Derive key from password using SHA-256
    hash := sha256.Sum256([]byte(password))
    return &AESManager{key: hash[:]}
}

// Encrypt data using AES-GCM
func (a *AESManager) Encrypt(plaintext []byte) (string, error) {
    block, err := aes.NewCipher(a.key)
    if err != nil {
        return "", fmt.Errorf("failed to create cipher: %w", err)
    }

    // GCM provides both confidentiality and authenticity
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", fmt.Errorf("failed to create GCM: %w", err)
    }

    // Generate random nonce
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", fmt.Errorf("failed to generate nonce: %w", err)
    }

    // Encrypt and authenticate
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

    // Encode to base64 for storage/transmission
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// Decrypt data using AES-GCM
func (a *AESManager) Decrypt(ciphertextB64 string) ([]byte, error) {
    // Decode from base64
    ciphertext, err := base64.StdEncoding.DecodeString(ciphertextB64)
    if err != nil {
        return nil, fmt.Errorf("failed to decode base64: %w", err)
    }

    block, err := aes.NewCipher(a.key)
    if err != nil {
        return nil, fmt.Errorf("failed to create cipher: %w", err)
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, fmt.Errorf("failed to create GCM: %w", err)
    }

    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }

    // Extract nonce and ciphertext
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

    // Decrypt and verify
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, fmt.Errorf("decryption failed: %w", err)
    }

    return plaintext, nil
}

// Field-level encryption for database storage
type FieldEncryption struct {
    aesManager *AESManager
}

func NewFieldEncryption(key string) *FieldEncryption {
    return &FieldEncryption{
        aesManager: NewAESManager(key),
    }
}

// Encrypt sensitive fields
func (f *FieldEncryption) EncryptField(value string) (string, error) {
    if value == "" {
        return "", nil
    }
    
    return f.aesManager.Encrypt([]byte(value))
}

// Decrypt sensitive fields
func (f *FieldEncryption) DecryptField(encryptedValue string) (string, error) {
    if encryptedValue == "" {
        return "", nil
    }
    
    decrypted, err := f.aesManager.Decrypt(encryptedValue)
    if err != nil {
        return "", err
    }
    
    return string(decrypted), nil
}

// User model with encrypted fields
type SecureUser struct {
    ID              string `json:"id"`
    Email           string `json:"email"`
    EncryptedSSN    string `json:"-"` // Hidden from JSON
    EncryptedPhone  string `json:"-"` // Hidden from JSON
    CreatedAt       time.Time `json:"created_at"`
}

// User service with field encryption
type SecureUserService struct {
    repo       UserRepository
    encryption *FieldEncryption
}

func NewSecureUserService(repo UserRepository, encryptionKey string) *SecureUserService {
    return &SecureUserService{
        repo:       repo,
        encryption: NewFieldEncryption(encryptionKey),
    }
}

func (s *SecureUserService) CreateUser(email, ssn, phone string) (*SecureUser, error) {
    encryptedSSN, err := s.encryption.EncryptField(ssn)
    if err != nil {
        return nil, fmt.Errorf("failed to encrypt SSN: %w", err)
    }
    
    encryptedPhone, err := s.encryption.EncryptField(phone)
    if err != nil {
        return nil, fmt.Errorf("failed to encrypt phone: %w", err)
    }
    
    user := &SecureUser{
        ID:             generateID(),
        Email:          email,
        EncryptedSSN:   encryptedSSN,
        EncryptedPhone: encryptedPhone,
        CreatedAt:      time.Now(),
    }
    
    return s.repo.Save(user)
}

func (s *SecureUserService) GetUserWithDecryptedData(userID string) (*UserWithSensitiveData, error) {
    user, err := s.repo.FindByID(userID)
    if err != nil {
        return nil, err
    }
    
    ssn, err := s.encryption.DecryptField(user.EncryptedSSN)
    if err != nil {
        return nil, fmt.Errorf("failed to decrypt SSN: %w", err)
    }
    
    phone, err := s.encryption.DecryptField(user.EncryptedPhone)
    if err != nil {
        return nil, fmt.Errorf("failed to decrypt phone: %w", err)
    }
    
    return &UserWithSensitiveData{
        ID:        user.ID,
        Email:     user.Email,
        SSN:       ssn,
        Phone:     phone,
        CreatedAt: user.CreatedAt,
    }, nil
}
```

### **RSA Key Management**

```go
package encryption

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/base64"
    "encoding/pem"
    "fmt"
)

// RSA Key Manager
type RSAManager struct {
    privateKey *rsa.PrivateKey
    publicKey  *rsa.PublicKey
}

func NewRSAManager(keySize int) (*RSAManager, error) {
    if keySize < 2048 {
        return nil, fmt.Errorf("key size must be at least 2048 bits")
    }
    
    privateKey, err := rsa.GenerateKey(rand.Reader, keySize)
    if err != nil {
        return nil, fmt.Errorf("failed to generate RSA key: %w", err)
    }
    
    return &RSAManager{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}

// Load RSA manager from PEM-encoded private key
func NewRSAManagerFromPEM(privateKeyPEM string) (*RSAManager, error) {
    block, _ := pem.Decode([]byte(privateKeyPEM))
    if block == nil {
        return nil, fmt.Errorf("failed to decode PEM block")
    }
    
    privateKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
    if err != nil {
        return nil, fmt.Errorf("failed to parse private key: %w", err)
    }
    
    return &RSAManager{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}

// Export private key as PEM
func (r *RSAManager) ExportPrivateKeyPEM() string {
    privateKeyBytes := x509.MarshalPKCS1PrivateKey(r.privateKey)
    privateKeyPEM := pem.EncodeToMemory(&pem.Block{
        Type:  "RSA PRIVATE KEY",
        Bytes: privateKeyBytes,
    })
    return string(privateKeyPEM)
}

// Export public key as PEM
func (r *RSAManager) ExportPublicKeyPEM() (string, error) {
    publicKeyBytes, err := x509.MarshalPKIXPublicKey(r.publicKey)
    if err != nil {
        return "", fmt.Errorf("failed to marshal public key: %w", err)
    }
    
    publicKeyPEM := pem.EncodeToMemory(&pem.Block{
        Type:  "PUBLIC KEY",
        Bytes: publicKeyBytes,
    })
    return string(publicKeyPEM), nil
}

// Encrypt data using RSA-OAEP
func (r *RSAManager) Encrypt(plaintext []byte) (string, error) {
    // RSA encryption has size limitations
    maxSize := r.publicKey.Size() - 2*sha256.Size - 2
    if len(plaintext) > maxSize {
        return "", fmt.Errorf("plaintext too large for RSA encryption (max %d bytes)", maxSize)
    }
    
    ciphertext, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, r.publicKey, plaintext, nil)
    if err != nil {
        return "", fmt.Errorf("RSA encryption failed: %w", err)
    }
    
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// Decrypt data using RSA-OAEP
func (r *RSAManager) Decrypt(ciphertextB64 string) ([]byte, error) {
    ciphertext, err := base64.StdEncoding.DecodeString(ciphertextB64)
    if err != nil {
        return nil, fmt.Errorf("failed to decode base64: %w", err)
    }
    
    plaintext, err := rsa.DecryptOAEP(sha256.New(), rand.Reader, r.privateKey, ciphertext, nil)
    if err != nil {
        return nil, fmt.Errorf("RSA decryption failed: %w", err)
    }
    
    return plaintext, nil
}

// Sign data using RSA-PSS
func (r *RSAManager) Sign(data []byte) (string, error) {
    hash := sha256.Sum256(data)
    
    signature, err := rsa.SignPSS(rand.Reader, r.privateKey, crypto.SHA256, hash[:], nil)
    if err != nil {
        return "", fmt.Errorf("RSA signing failed: %w", err)
    }
    
    return base64.StdEncoding.EncodeToString(signature), nil
}

// Verify signature using RSA-PSS
func (r *RSAManager) Verify(data []byte, signatureB64 string) error {
    signature, err := base64.StdEncoding.DecodeString(signatureB64)
    if err != nil {
        return fmt.Errorf("failed to decode signature: %w", err)
    }
    
    hash := sha256.Sum256(data)
    
    err = rsa.VerifyPSS(r.publicKey, crypto.SHA256, hash[:], signature, nil)
    if err != nil {
        return fmt.Errorf("signature verification failed: %w", err)
    }
    
    return nil
}

// Hybrid encryption: RSA + AES for large data
type HybridEncryption struct {
    rsaManager *RSAManager
}

func NewHybridEncryption(rsaManager *RSAManager) *HybridEncryption {
    return &HybridEncryption{rsaManager: rsaManager}
}

type HybridCiphertext struct {
    EncryptedKey  string `json:"encrypted_key"`
    EncryptedData string `json:"encrypted_data"`
}

func (h *HybridEncryption) Encrypt(plaintext []byte) (*HybridCiphertext, error) {
    // Generate random AES key
    aesKey := make([]byte, 32) // 256-bit key
    if _, err := rand.Read(aesKey); err != nil {
        return nil, fmt.Errorf("failed to generate AES key: %w", err)
    }
    
    // Encrypt data with AES
    aesManager := &AESManager{key: aesKey}
    encryptedData, err := aesManager.Encrypt(plaintext)
    if err != nil {
        return nil, fmt.Errorf("AES encryption failed: %w", err)
    }
    
    // Encrypt AES key with RSA
    encryptedKey, err := h.rsaManager.Encrypt(aesKey)
    if err != nil {
        return nil, fmt.Errorf("RSA encryption of key failed: %w", err)
    }
    
    return &HybridCiphertext{
        EncryptedKey:  encryptedKey,
        EncryptedData: encryptedData,
    }, nil
}

func (h *HybridEncryption) Decrypt(ciphertext *HybridCiphertext) ([]byte, error) {
    // Decrypt AES key with RSA
    aesKey, err := h.rsaManager.Decrypt(ciphertext.EncryptedKey)
    if err != nil {
        return nil, fmt.Errorf("RSA decryption of key failed: %w", err)
    }
    
    // Decrypt data with AES
    aesManager := &AESManager{key: aesKey}
    plaintext, err := aesManager.Decrypt(ciphertext.EncryptedData)
    if err != nil {
        return nil, fmt.Errorf("AES decryption failed: %w", err)
    }
    
    return plaintext, nil
}
```

---

## 🎯 **Interview Questions**

### **Security Architecture Questions**

**Q1: How would you design a secure authentication system for a fintech application?**

**Answer:**

**Multi-layered security approach:**

1. **Authentication Flow:**
```go
type SecureAuthFlow struct {
    // Step 1: Primary authentication
    primaryAuth   AuthenticationMethod // Password, biometric, etc.
    
    // Step 2: Multi-factor authentication
    mfaRequired   bool
    mfaMethod     MFAMethod // TOTP, SMS, hardware token
    
    // Step 3: Risk assessment
    riskAssessment RiskEngine
    
    // Step 4: Token generation
    tokenManager   JWTManager
    
    // Step 5: Session management
    sessionStore   SessionStore
}

func (s *SecureAuthFlow) Authenticate(request AuthRequest) (*AuthResponse, error) {
    // Risk-based authentication
    riskScore := s.riskAssessment.CalculateRisk(request)
    
    // Primary authentication
    if err := s.primaryAuth.Verify(request.Credentials); err != nil {
        return nil, fmt.Errorf("primary authentication failed: %w", err)
    }
    
    // Conditional MFA based on risk
    if s.requiresMFA(riskScore, request) {
        if err := s.performMFA(request); err != nil {
            return nil, fmt.Errorf("MFA failed: %w", err)
        }
    }
    
    // Generate secure tokens
    tokens, err := s.generateTokens(request.UserID)
    if err != nil {
        return nil, fmt.Errorf("token generation failed: %w", err)
    }
    
    // Create secure session
    session := s.createSecureSession(request)
    
    return &AuthResponse{
        AccessToken:  tokens.AccessToken,
        RefreshToken: tokens.RefreshToken,
        Session:      session,
        ExpiresAt:    tokens.ExpiresAt,
    }, nil
}
```

2. **Security Features:**
- Password complexity requirements
- Account lockout after failed attempts
- Device fingerprinting
- Behavioral analysis
- Geographic restriction options
- Secure token storage

**Q2: Explain how you would implement end-to-end encryption for sensitive data in transit and at rest.**

**Answer:**

**Comprehensive encryption strategy:**

1. **Data in Transit:**
```go
type SecureTransport struct {
    tlsConfig *tls.Config
    certManager CertificateManager
}

func (s *SecureTransport) SetupTLS() *tls.Config {
    return &tls.Config{
        MinVersion: tls.VersionTLS12,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
            tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
        },
        PreferServerCipherSuites: true,
        CurvePreferences: []tls.CurveID{
            tls.CurveP521,
            tls.CurveP384,
            tls.CurveP256,
        },
    }
}
```

2. **Data at Rest:**
```go
type DataEncryption struct {
    fieldEncryption *FieldEncryption // Individual field encryption
    volumeEncryption VolumeEncryption // Database/filesystem encryption
    keyManagement   KeyManagementService // Key rotation and management
}

// Layered encryption approach
func (d *DataEncryption) StoreSecureData(data *SensitiveData) error {
    // Layer 1: Field-level encryption
    encryptedFields, err := d.fieldEncryption.EncryptFields(data)
    if err != nil {
        return fmt.Errorf("field encryption failed: %w", err)
    }
    
    // Layer 2: Database encryption (transparent)
    // Handled by database encryption at rest
    
    // Layer 3: Volume encryption
    // Handled by OS/filesystem encryption
    
    return d.store(encryptedFields)
}
```

**Q3: How do you implement secure API authentication and authorization for microservices?**

**Answer:**

**Service-to-service security:**

```go
type ServiceSecurity struct {
    jwtManager    *JWTManager
    mtlsConfig    *tls.Config
    serviceRegistry ServiceRegistry
    rbacManager   *RBACManager
}

// Service authentication using mTLS + JWT
func (s *ServiceSecurity) AuthenticateService(request *http.Request) (*ServiceIdentity, error) {
    // Step 1: Verify mTLS certificate
    cert := request.TLS.PeerCertificates[0]
    serviceID, err := s.extractServiceID(cert)
    if err != nil {
        return nil, fmt.Errorf("invalid service certificate: %w", err)
    }
    
    // Step 2: Verify service registration
    if !s.serviceRegistry.IsRegistered(serviceID) {
        return nil, fmt.Errorf("service not registered: %s", serviceID)
    }
    
    // Step 3: Verify JWT token
    token := extractBearerToken(request)
    claims, err := s.jwtManager.Verify(token)
    if err != nil {
        return nil, fmt.Errorf("invalid JWT: %w", err)
    }
    
    return &ServiceIdentity{
        ServiceID: serviceID,
        Claims:    claims,
    }, nil
}

// API Gateway with security
func (s *ServiceSecurity) SecureGateway() http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Authenticate service
        identity, err := s.AuthenticateService(r)
        if err != nil {
            http.Error(w, "Authentication failed", http.StatusUnauthorized)
            return
        }
        
        // Authorize request
        if !s.rbacManager.HasServicePermission(identity, r) {
            http.Error(w, "Forbidden", http.StatusForbidden)
            return
        }
        
        // Forward request with identity context
        ctx := context.WithValue(r.Context(), "service_identity", identity)
        r = r.WithContext(ctx)
        
        // Continue to backend service
        s.proxyToService(w, r)
    })
}
```

This comprehensive Security Engineering Guide provides the foundation for implementing robust security measures in backend systems, covering all aspects from authentication and authorization to encryption and compliance. It demonstrates the advanced security knowledge expected from senior engineers in technical interviews.

## Secure Coding Practices

<!-- AUTO-GENERATED ANCHOR: originally referenced as #secure-coding-practices -->

Placeholder content. Please replace with proper section.


## Owasp Top 10

<!-- AUTO-GENERATED ANCHOR: originally referenced as #owasp-top-10 -->

Placeholder content. Please replace with proper section.


## Security Testing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #security-testing -->

Placeholder content. Please replace with proper section.


## Compliance Frameworks

<!-- AUTO-GENERATED ANCHOR: originally referenced as #compliance-frameworks -->

Placeholder content. Please replace with proper section.


## Api Security

<!-- AUTO-GENERATED ANCHOR: originally referenced as #api-security -->

Placeholder content. Please replace with proper section.


## Infrastructure Security

<!-- AUTO-GENERATED ANCHOR: originally referenced as #infrastructure-security -->

Placeholder content. Please replace with proper section.


## Security Monitoring

<!-- AUTO-GENERATED ANCHOR: originally referenced as #security-monitoring -->

Placeholder content. Please replace with proper section.
