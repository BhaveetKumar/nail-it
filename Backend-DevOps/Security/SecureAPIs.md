# 🔒 Secure APIs: Authentication, Authorization, and Rate Limiting

> **Master API security for comprehensive protection of REST and GraphQL APIs**

## 📚 Concept

API security involves protecting APIs from various threats including unauthorized access, data breaches, and abuse. It encompasses authentication, authorization, rate limiting, input validation, and monitoring to ensure secure and reliable API operations.

### Key Features
- **Authentication**: Verify user identity
- **Authorization**: Control access to resources
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize and validate requests
- **Encryption**: Protect data in transit and at rest
- **Monitoring**: Track and analyze API usage

## 🏗️ API Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Security Stack                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Client    │  │   API       │  │   Backend   │     │
│  │  Application│  │  Gateway    │  │  Services   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Security Layer                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Auth      │  │   Rate      │  │   Input     │ │ │
│  │  │   Service   │  │   Limiter   │  │   Validator │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Policy Engine                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Access    │  │   Resource  │  │   Audit     │ │ │
│  │  │   Control   │  │   Policies  │  │   Logging   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Identity  │  │   Database  │  │   Monitoring│     │
│  │   Provider  │  │   Security  │  │   System    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### JWT Authentication Service

```go
// auth.go
package main

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/x509"
    "encoding/pem"
    "fmt"
    "time"

    "github.com/golang-jwt/jwt/v4"
    "golang.org/x/crypto/bcrypt"
)

type User struct {
    ID           string    `json:"id"`
    Username     string    `json:"username"`
    Email        string    `json:"email"`
    PasswordHash string    `json:"password_hash"`
    Roles        []string  `json:"roles"`
    Permissions  []string  `json:"permissions"`
    IsActive     bool      `json:"is_active"`
    CreatedAt    time.Time `json:"created_at"`
    UpdatedAt    time.Time `json:"updated_at"`
}

type Claims struct {
    UserID      string   `json:"user_id"`
    Username    string   `json:"username"`
    Email       string   `json:"email"`
    Roles       []string `json:"roles"`
    Permissions []string `json:"permissions"`
    jwt.RegisteredClaims
}

type AuthService struct {
    users      map[string]*User
    jwtKey     *rsa.PrivateKey
    refreshTokens map[string]string
}

func NewAuthService() (*AuthService, error) {
    // Generate RSA key pair for JWT signing
    privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil {
        return nil, fmt.Errorf("failed to generate RSA key: %w", err)
    }

    return &AuthService{
        users:         make(map[string]*User),
        jwtKey:        privateKey,
        refreshTokens: make(map[string]string),
    }, nil
}

func (as *AuthService) RegisterUser(username, email, password string, roles []string) (*User, error) {
    // Hash password
    hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    if err != nil {
        return nil, fmt.Errorf("failed to hash password: %w", err)
    }

    // Generate user ID
    userID := generateID()

    user := &User{
        ID:           userID,
        Username:     username,
        Email:        email,
        PasswordHash: string(hashedPassword),
        Roles:        roles,
        Permissions:  as.getPermissionsForRoles(roles),
        IsActive:     true,
        CreatedAt:    time.Now(),
        UpdatedAt:    time.Now(),
    }

    as.users[userID] = user
    return user, nil
}

func (as *AuthService) AuthenticateUser(username, password string) (*User, error) {
    // Find user by username
    var user *User
    for _, u := range as.users {
        if u.Username == username {
            user = u
            break
        }
    }

    if user == nil {
        return nil, fmt.Errorf("user not found")
    }

    // Check if user is active
    if !user.IsActive {
        return nil, fmt.Errorf("user account is disabled")
    }

    // Verify password
    if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
        return nil, fmt.Errorf("invalid password")
    }

    return user, nil
}

func (as *AuthService) GenerateTokenPair(user *User) (string, string, error) {
    // Generate access token
    accessToken, err := as.generateAccessToken(user)
    if err != nil {
        return "", "", fmt.Errorf("failed to generate access token: %w", err)
    }

    // Generate refresh token
    refreshToken := generateID()
    as.refreshTokens[refreshToken] = user.ID

    return accessToken, refreshToken, nil
}

func (as *AuthService) generateAccessToken(user *User) (string, error) {
    claims := &Claims{
        UserID:      user.ID,
        Username:    user.Username,
        Email:       user.Email,
        Roles:       user.Roles,
        Permissions: user.Permissions,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(15 * time.Minute)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
            Issuer:    "my-app",
            Subject:   user.ID,
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
    return token.SignedString(as.jwtKey)
}

func (as *AuthService) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return &as.jwtKey.PublicKey, nil
    })

    if err != nil {
        return nil, fmt.Errorf("failed to parse token: %w", err)
    }

    if !token.Valid {
        return nil, fmt.Errorf("invalid token")
    }

    claims, ok := token.Claims.(*Claims)
    if !ok {
        return nil, fmt.Errorf("invalid token claims")
    }

    return claims, nil
}

func (as *AuthService) RefreshToken(refreshToken string) (string, error) {
    userID, exists := as.refreshTokens[refreshToken]
    if !exists {
        return "", fmt.Errorf("invalid refresh token")
    }

    user, exists := as.users[userID]
    if !exists {
        return "", fmt.Errorf("user not found")
    }

    // Generate new access token
    newAccessToken, err := as.generateAccessToken(user)
    if err != nil {
        return "", fmt.Errorf("failed to generate new access token: %w", err)
    }

    return newAccessToken, nil
}

func (as *AuthService) RevokeToken(refreshToken string) error {
    delete(as.refreshTokens, refreshToken)
    return nil
}

func (as *AuthService) getPermissionsForRoles(roles []string) []string {
    permissions := make([]string, 0)
    
    for _, role := range roles {
        switch role {
        case "admin":
            permissions = append(permissions, "read", "write", "delete", "admin")
        case "user":
            permissions = append(permissions, "read", "write")
        case "viewer":
            permissions = append(permissions, "read")
        }
    }
    
    return permissions
}
```

### Rate Limiting Service

```go
// rate_limiter.go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type RateLimiter struct {
    limits map[string]*RateLimit
    mutex  sync.RWMutex
}

type RateLimit struct {
    Requests   int           `json:"requests"`
    Window     time.Duration `json:"window"`
    Remaining  int           `json:"remaining"`
    ResetTime  time.Time     `json:"reset_time"`
    LastAccess time.Time     `json:"last_access"`
}

type RateLimitConfig struct {
    Requests int           `json:"requests"`
    Window   time.Duration `json:"window"`
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        limits: make(map[string]*RateLimit),
    }
}

func (rl *RateLimiter) CheckLimit(key string, config RateLimitConfig) (bool, *RateLimit, error) {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    limit, exists := rl.limits[key]

    if !exists {
        // Create new rate limit
        limit = &RateLimit{
            Requests:   config.Requests,
            Window:     config.Window,
            Remaining:  config.Requests - 1,
            ResetTime:  now.Add(config.Window),
            LastAccess: now,
        }
        rl.limits[key] = limit
        return true, limit, nil
    }

    // Check if window has expired
    if now.After(limit.ResetTime) {
        // Reset the limit
        limit.Remaining = config.Requests - 1
        limit.ResetTime = now.Add(config.Window)
        limit.LastAccess = now
        return true, limit, nil
    }

    // Check if limit is exceeded
    if limit.Remaining <= 0 {
        return false, limit, nil
    }

    // Decrement remaining requests
    limit.Remaining--
    limit.LastAccess = now

    return true, limit, nil
}

func (rl *RateLimiter) GetLimit(key string) (*RateLimit, bool) {
    rl.mutex.RLock()
    defer rl.mutex.RUnlock()

    limit, exists := rl.limits[key]
    return limit, exists
}

func (rl *RateLimiter) ResetLimit(key string) {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    delete(rl.limits, key)
}

func (rl *RateLimiter) CleanupExpired() {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    for key, limit := range rl.limits {
        if now.After(limit.ResetTime) {
            delete(rl.limits, key)
        }
    }
}

func (rl *RateLimiter) StartCleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    go func() {
        for range ticker.C {
            rl.CleanupExpired()
        }
    }()
}
```

### Input Validation Service

```go
// validation.go
package main

import (
    "fmt"
    "net/url"
    "regexp"
    "strings"
    "unicode"
)

type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
}

type ValidationResult struct {
    IsValid bool              `json:"is_valid"`
    Errors  []ValidationError `json:"errors"`
}

type Validator struct {
    rules map[string][]ValidationRule
}

type ValidationRule struct {
    Name    string                 `json:"name"`
    Message string                 `json:"message"`
    Check   func(interface{}) bool `json:"-"`
}

func NewValidator() *Validator {
    return &Validator{
        rules: make(map[string][]ValidationRule),
    }
}

func (v *Validator) AddRule(field string, rule ValidationRule) {
    v.rules[field] = append(v.rules[field], rule)
}

func (v *Validator) Validate(data map[string]interface{}) ValidationResult {
    result := ValidationResult{
        IsValid: true,
        Errors:  make([]ValidationError, 0),
    }

    for field, rules := range v.rules {
        value, exists := data[field]
        if !exists {
            continue
        }

        for _, rule := range rules {
            if !rule.Check(value) {
                result.IsValid = false
                result.Errors = append(result.Errors, ValidationError{
                    Field:   field,
                    Message: rule.Message,
                })
            }
        }
    }

    return result
}

// Common validation rules
func Required() ValidationRule {
    return ValidationRule{
        Name:    "required",
        Message: "Field is required",
        Check: func(value interface{}) bool {
            if value == nil {
                return false
            }
            if str, ok := value.(string); ok {
                return strings.TrimSpace(str) != ""
            }
            return true
        },
    }
}

func MinLength(min int) ValidationRule {
    return ValidationRule{
        Name:    "min_length",
        Message: fmt.Sprintf("Field must be at least %d characters long", min),
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                return len(str) >= min
            }
            return true
        },
    }
}

func MaxLength(max int) ValidationRule {
    return ValidationRule{
        Name:    "max_length",
        Message: fmt.Sprintf("Field must be at most %d characters long", max),
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                return len(str) <= max
            }
            return true
        },
    }
}

func Email() ValidationRule {
    emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
    return ValidationRule{
        Name:    "email",
        Message: "Field must be a valid email address",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                return emailRegex.MatchString(str)
            }
            return true
        },
    }
}

func URL() ValidationRule {
    return ValidationRule{
        Name:    "url",
        Message: "Field must be a valid URL",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                _, err := url.Parse(str)
                return err == nil
            }
            return true
        },
    }
}

func StrongPassword() ValidationRule {
    return ValidationRule{
        Name:    "strong_password",
        Message: "Password must contain at least 8 characters, including uppercase, lowercase, number, and special character",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                if len(str) < 8 {
                    return false
                }

                var hasUpper, hasLower, hasNumber, hasSpecial bool
                for _, char := range str {
                    switch {
                    case unicode.IsUpper(char):
                        hasUpper = true
                    case unicode.IsLower(char):
                        hasLower = true
                    case unicode.IsNumber(char):
                        hasNumber = true
                    case unicode.IsPunct(char) || unicode.IsSymbol(char):
                        hasSpecial = true
                    }
                }

                return hasUpper && hasLower && hasNumber && hasSpecial
            }
            return true
        },
    }
}

func Alphanumeric() ValidationRule {
    return ValidationRule{
        Name:    "alphanumeric",
        Message: "Field must contain only alphanumeric characters",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                for _, char := range str {
                    if !unicode.IsLetter(char) && !unicode.IsNumber(char) {
                        return false
                    }
                }
                return true
            }
            return true
        },
    }
}

func NoSQLInjection() ValidationRule {
    return ValidationRule{
        Name:    "no_sql_injection",
        Message: "Field contains potentially dangerous SQL injection characters",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                dangerousChars := []string{"'", "\"", ";", "--", "/*", "*/", "xp_", "sp_"}
                for _, char := range dangerousChars {
                    if strings.Contains(strings.ToLower(str), char) {
                        return false
                    }
                }
                return true
            }
            return true
        },
    }
}

func NoXSS() ValidationRule {
    return ValidationRule{
        Name:    "no_xss",
        Message: "Field contains potentially dangerous XSS characters",
        Check: func(value interface{}) bool {
            if str, ok := value.(string); ok {
                dangerousChars := []string{"<", ">", "script", "javascript:", "onload", "onerror"}
                for _, char := range dangerousChars {
                    if strings.Contains(strings.ToLower(str), char) {
                        return false
                    }
                }
                return true
            }
            return true
        },
    }
}
```

### API Security Middleware

```go
// middleware.go
package main

import (
    "context"
    "fmt"
    "net/http"
    "strings"
    "time"

    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
)

type SecurityMiddleware struct {
    authService    *AuthService
    rateLimiter    *RateLimiter
    validator      *Validator
    logger         *zap.Logger
}

func NewSecurityMiddleware(authService *AuthService, rateLimiter *RateLimiter, validator *Validator, logger *zap.Logger) *SecurityMiddleware {
    return &SecurityMiddleware{
        authService: authService,
        rateLimiter: rateLimiter,
        validator:   validator,
        logger:      logger,
    }
}

func (sm *SecurityMiddleware) AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Extract token from Authorization header
        authHeader := c.GetHeader("Authorization")
        if authHeader == "" {
            sm.logger.Warn("Missing authorization header",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Missing authorization header"})
            c.Abort()
            return
        }

        tokenString := strings.TrimPrefix(authHeader, "Bearer ")
        if tokenString == authHeader {
            sm.logger.Warn("Invalid authorization header format",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization header format"})
            c.Abort()
            return
        }

        // Validate token
        claims, err := sm.authService.ValidateToken(tokenString)
        if err != nil {
            sm.logger.Warn("Invalid token",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
                zap.Error(err),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
            c.Abort()
            return
        }

        // Store claims in context
        c.Set("claims", claims)
        c.Next()
    }
}

func (sm *SecurityMiddleware) RateLimitMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Get client identifier (IP address or user ID)
        clientID := c.ClientIP()
        
        // Check if user is authenticated
        if claims, exists := c.Get("claims"); exists {
            if userClaims, ok := claims.(*Claims); ok {
                clientID = userClaims.UserID
            }
        }

        // Check rate limit
        config := RateLimitConfig{
            Requests: 100,
            Window:   1 * time.Minute,
        }

        allowed, limit, err := sm.rateLimiter.CheckLimit(clientID, config)
        if err != nil {
            sm.logger.Error("Rate limit check failed",
                zap.String("client_id", clientID),
                zap.Error(err),
            )
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Rate limit check failed"})
            c.Abort()
            return
        }

        if !allowed {
            sm.logger.Warn("Rate limit exceeded",
                zap.String("client_id", clientID),
                zap.Int("limit", limit.Requests),
                zap.Int("remaining", limit.Remaining),
            )
            c.JSON(http.StatusTooManyRequests, gin.H{
                "error": "Rate limit exceeded",
                "limit": limit.Requests,
                "remaining": limit.Remaining,
                "reset_time": limit.ResetTime,
            })
            c.Abort()
            return
        }

        // Add rate limit headers
        c.Header("X-RateLimit-Limit", fmt.Sprintf("%d", limit.Requests))
        c.Header("X-RateLimit-Remaining", fmt.Sprintf("%d", limit.Remaining))
        c.Header("X-RateLimit-Reset", fmt.Sprintf("%d", limit.ResetTime.Unix()))

        c.Next()
    }
}

func (sm *SecurityMiddleware) ValidationMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Get request data
        var requestData map[string]interface{}
        if err := c.ShouldBindJSON(&requestData); err != nil {
            sm.logger.Warn("Invalid request data",
                zap.String("ip", c.ClientIP()),
                zap.Error(err),
            )
            c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request data"})
            c.Abort()
            return
        }

        // Validate request data
        result := sm.validator.Validate(requestData)
        if !result.IsValid {
            sm.logger.Warn("Validation failed",
                zap.String("ip", c.ClientIP()),
                zap.Any("errors", result.Errors),
            )
            c.JSON(http.StatusBadRequest, gin.H{
                "error": "Validation failed",
                "details": result.Errors,
            })
            c.Abort()
            return
        }

        // Store validated data in context
        c.Set("validated_data", requestData)
        c.Next()
    }
}

func (sm *SecurityMiddleware) CORSMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        origin := c.GetHeader("Origin")
        
        // Check if origin is allowed
        allowedOrigins := []string{
            "http://localhost:3000",
            "https://myapp.com",
            "https://www.myapp.com",
        }

        allowed := false
        for _, allowedOrigin := range allowedOrigins {
            if origin == allowedOrigin {
                allowed = true
                break
            }
        }

        if allowed {
            c.Header("Access-Control-Allow-Origin", origin)
        }

        c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")
        c.Header("Access-Control-Allow-Credentials", "true")
        c.Header("Access-Control-Max-Age", "86400")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(http.StatusNoContent)
            return
        }

        c.Next()
    }
}

func (sm *SecurityMiddleware) SecurityHeadersMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Add security headers
        c.Header("X-Content-Type-Options", "nosniff")
        c.Header("X-Frame-Options", "DENY")
        c.Header("X-XSS-Protection", "1; mode=block")
        c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
        c.Header("Content-Security-Policy", "default-src 'self'")

        c.Next()
    }
}

func (sm *SecurityMiddleware) LoggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        
        // Process request
        c.Next()
        
        // Log request
        duration := time.Since(start)
        
        sm.logger.Info("API request",
            zap.String("method", c.Request.Method),
            zap.String("path", c.FullPath()),
            zap.Int("status", c.Writer.Status()),
            zap.Duration("duration", duration),
            zap.String("ip", c.ClientIP()),
            zap.String("user_agent", c.GetHeader("User-Agent")),
            zap.String("user_id", sm.getUserID(c)),
        )
    }
}

func (sm *SecurityMiddleware) getUserID(c *gin.Context) string {
    if claims, exists := c.Get("claims"); exists {
        if userClaims, ok := claims.(*Claims); ok {
            return userClaims.UserID
        }
    }
    return "anonymous"
}
```

### Authorization Service

```go
// authorization.go
package main

import (
    "context"
    "fmt"
    "strings"
)

type AuthorizationService struct {
    policies map[string][]Policy
}

type Policy struct {
    ID          string   `json:"id"`
    Name        string   `json:"name"`
    Description string   `json:"description"`
    Rules       []Rule   `json:"rules"`
    Priority    int      `json:"priority"`
    IsActive    bool     `json:"is_active"`
}

type Rule struct {
    ID         string                 `json:"id"`
    Resource   string                 `json:"resource"`
    Action     string                 `json:"action"`
    Conditions []Condition            `json:"conditions"`
    Effect     string                 `json:"effect"`
    Metadata   map[string]interface{} `json:"metadata"`
}

type Condition struct {
    ID       string                 `json:"id"`
    Field    string                 `json:"field"`
    Operator string                 `json:"operator"`
    Value    interface{}            `json:"value"`
    Metadata map[string]interface{} `json:"metadata"`
}

func NewAuthorizationService() *AuthorizationService {
    return &AuthorizationService{
        policies: make(map[string][]Policy),
    }
}

func (as *AuthorizationService) AddPolicy(policy Policy) {
    as.policies[policy.Name] = append(as.policies[policy.Name], policy)
}

func (as *AuthorizationService) CheckPermission(ctx context.Context, userID, resource, action string, context map[string]interface{}) (bool, error) {
    // Get user claims from context
    claims, exists := ctx.Value("claims").(*Claims)
    if !exists {
        return false, fmt.Errorf("user claims not found in context")
    }

    // Check policies
    for _, policies := range as.policies {
        for _, policy := range policies {
            if !policy.IsActive {
                continue
            }

            if as.policyMatches(policy, resource, action, claims, context) {
                for _, rule := range policy.Rules {
                    if as.ruleMatches(rule, resource, action, claims, context) {
                        return rule.Effect == "allow", nil
                    }
                }
            }
        }
    }

    return false, nil
}

func (as *AuthorizationService) policyMatches(policy Policy, resource, action string, claims *Claims, context map[string]interface{}) bool {
    for _, rule := range policy.Rules {
        if rule.Resource == resource || rule.Resource == "*" {
            if rule.Action == action || rule.Action == "*" {
                return true
            }
        }
    }
    return false
}

func (as *AuthorizationService) ruleMatches(rule Rule, resource, action string, claims *Claims, context map[string]interface{}) bool {
    // Check resource and action
    if rule.Resource != resource && rule.Resource != "*" {
        return false
    }

    if rule.Action != action && rule.Action != "*" {
        return false
    }

    // Check conditions
    for _, condition := range rule.Conditions {
        if !as.conditionMatches(condition, claims, context) {
            return false
        }
    }

    return true
}

func (as *AuthorizationService) conditionMatches(condition Condition, claims *Claims, context map[string]interface{}) bool {
    var value interface{}

    // Get value from claims or context
    switch condition.Field {
    case "user_id":
        value = claims.UserID
    case "username":
        value = claims.Username
    case "email":
        value = claims.Email
    case "roles":
        value = claims.Roles
    case "permissions":
        value = claims.Permissions
    default:
        if contextValue, exists := context[condition.Field]; exists {
            value = contextValue
        } else {
            return false
        }
    }

    // Apply operator
    switch condition.Operator {
    case "equals":
        return value == condition.Value
    case "not_equals":
        return value != condition.Value
    case "contains":
        if str, ok := value.(string); ok {
            if target, ok := condition.Value.(string); ok {
                return strings.Contains(str, target)
            }
        }
        return false
    case "in":
        if list, ok := condition.Value.([]interface{}); ok {
            for _, item := range list {
                if value == item {
                    return true
                }
            }
        }
        return false
    case "not_in":
        if list, ok := condition.Value.([]interface{}); ok {
            for _, item := range list {
                if value == item {
                    return false
                }
            }
        }
        return true
    default:
        return false
    }
}

func (as *AuthorizationService) RequirePermission(resource, action string) gin.HandlerFunc {
    return func(c *gin.Context) {
        // Get user claims
        claims, exists := c.Get("claims")
        if !exists {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
            c.Abort()
            return
        }

        // Create context for authorization
        authContext := map[string]interface{}{
            "ip_address": c.ClientIP(),
            "user_agent": c.GetHeader("User-Agent"),
            "endpoint":   c.FullPath(),
            "method":     c.Request.Method,
        }

        // Check permission
        allowed, err := as.CheckPermission(
            context.WithValue(c.Request.Context(), "claims", claims),
            claims.(*Claims).UserID,
            resource,
            action,
            authContext,
        )
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Authorization check failed"})
            c.Abort()
            return
        }

        if !allowed {
            c.JSON(http.StatusForbidden, gin.H{"error": "Access denied"})
            c.Abort()
            return
        }

        c.Next()
    }
}
```

### API Routes with Security

```go
// routes.go
package main

import (
    "net/http"
    "time"

    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
)

func setupRoutes(authService *AuthService, rateLimiter *RateLimiter, validator *Validator, authzService *AuthorizationService, logger *zap.Logger) *gin.Engine {
    r := gin.New()
    
    // Create security middleware
    securityMiddleware := NewSecurityMiddleware(authService, rateLimiter, validator, logger)
    
    // Add global middleware
    r.Use(securityMiddleware.LoggingMiddleware())
    r.Use(securityMiddleware.SecurityHeadersMiddleware())
    r.Use(securityMiddleware.CORSMiddleware())
    r.Use(gin.Recovery())
    
    // Public routes
    public := r.Group("/api/v1")
    {
        // Health check
        public.GET("/health", func(c *gin.Context) {
            c.JSON(http.StatusOK, gin.H{
                "status": "healthy",
                "timestamp": time.Now().UTC(),
            })
        })
        
        // Authentication routes
        auth := public.Group("/auth")
        {
            auth.POST("/register", func(c *gin.Context) {
                var request struct {
                    Username string `json:"username" binding:"required"`
                    Email    string `json:"email" binding:"required"`
                    Password string `json:"password" binding:"required"`
                }
                
                if err := c.ShouldBindJSON(&request); err != nil {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
                    return
                }
                
                // Validate input
                validator.AddRule("username", Required())
                validator.AddRule("username", MinLength(3))
                validator.AddRule("username", MaxLength(20))
                validator.AddRule("username", Alphanumeric())
                
                validator.AddRule("email", Required())
                validator.AddRule("email", Email())
                
                validator.AddRule("password", Required())
                validator.AddRule("password", StrongPassword())
                
                result := validator.Validate(map[string]interface{}{
                    "username": request.Username,
                    "email":    request.Email,
                    "password": request.Password,
                })
                
                if !result.IsValid {
                    c.JSON(http.StatusBadRequest, gin.H{
                        "error": "Validation failed",
                        "details": result.Errors,
                    })
                    return
                }
                
                // Register user
                user, err := authService.RegisterUser(request.Username, request.Email, request.Password, []string{"user"})
                if err != nil {
                    c.JSON(http.StatusInternalServerError, gin.H{"error": "Registration failed"})
                    return
                }
                
                c.JSON(http.StatusCreated, gin.H{
                    "message": "User registered successfully",
                    "user_id": user.ID,
                })
            })
            
            auth.POST("/login", func(c *gin.Context) {
                var request struct {
                    Username string `json:"username" binding:"required"`
                    Password string `json:"password" binding:"required"`
                }
                
                if err := c.ShouldBindJSON(&request); err != nil {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
                    return
                }
                
                // Authenticate user
                user, err := authService.AuthenticateUser(request.Username, request.Password)
                if err != nil {
                    c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
                    return
                }
                
                // Generate tokens
                accessToken, refreshToken, err := authService.GenerateTokenPair(user)
                if err != nil {
                    c.JSON(http.StatusInternalServerError, gin.H{"error": "Token generation failed"})
                    return
                }
                
                c.JSON(http.StatusOK, gin.H{
                    "access_token":  accessToken,
                    "refresh_token": refreshToken,
                    "expires_in":    900, // 15 minutes
                })
            })
            
            auth.POST("/refresh", func(c *gin.Context) {
                var request struct {
                    RefreshToken string `json:"refresh_token" binding:"required"`
                }
                
                if err := c.ShouldBindJSON(&request); err != nil {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
                    return
                }
                
                // Refresh token
                newAccessToken, err := authService.RefreshToken(request.RefreshToken)
                if err != nil {
                    c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid refresh token"})
                    return
                }
                
                c.JSON(http.StatusOK, gin.H{
                    "access_token": newAccessToken,
                    "expires_in":   900, // 15 minutes
                })
            })
        }
    }
    
    // Protected routes
    protected := r.Group("/api/v1")
    protected.Use(securityMiddleware.AuthMiddleware())
    protected.Use(securityMiddleware.RateLimitMiddleware())
    {
        // User routes
        users := protected.Group("/users")
        {
            users.GET("/profile", authzService.RequirePermission("users", "read"), func(c *gin.Context) {
                claims := c.MustGet("claims").(*Claims)
                
                c.JSON(http.StatusOK, gin.H{
                    "user_id":   claims.UserID,
                    "username":  claims.Username,
                    "email":     claims.Email,
                    "roles":     claims.Roles,
                    "permissions": claims.Permissions,
                })
            })
            
            users.PUT("/profile", authzService.RequirePermission("users", "write"), func(c *gin.Context) {
                var request struct {
                    Email string `json:"email"`
                }
                
                if err := c.ShouldBindJSON(&request); err != nil {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
                    return
                }
                
                // Validate email
                validator.AddRule("email", Email())
                result := validator.Validate(map[string]interface{}{
                    "email": request.Email,
                })
                
                if !result.IsValid {
                    c.JSON(http.StatusBadRequest, gin.H{
                        "error": "Validation failed",
                        "details": result.Errors,
                    })
                    return
                }
                
                c.JSON(http.StatusOK, gin.H{"message": "Profile updated successfully"})
            })
        }
        
        // Admin routes
        admin := protected.Group("/admin")
        admin.Use(authzService.RequirePermission("admin", "admin"))
        {
            admin.GET("/users", func(c *gin.Context) {
                c.JSON(http.StatusOK, gin.H{"message": "Admin users endpoint"})
            })
            
            admin.POST("/users", func(c *gin.Context) {
                c.JSON(http.StatusOK, gin.H{"message": "Create user endpoint"})
            })
        }
    }
    
    return r
}
```

## 🚀 Best Practices

### 1. Token Security
```go
// Use short-lived access tokens
func (as *AuthService) generateAccessToken(user *User) (string, error) {
    claims := &Claims{
        // ... claims
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(15 * time.Minute)), // Short expiry
            // ... other claims
        },
    }
    // ... token generation
}
```

### 2. Rate Limiting
```go
// Implement different rate limits for different endpoints
func (sm *SecurityMiddleware) RateLimitMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Different limits based on endpoint
        var config RateLimitConfig
        switch c.FullPath() {
        case "/api/v1/auth/login":
            config = RateLimitConfig{Requests: 5, Window: 1 * time.Minute}
        case "/api/v1/auth/register":
            config = RateLimitConfig{Requests: 3, Window: 1 * time.Minute}
        default:
            config = RateLimitConfig{Requests: 100, Window: 1 * time.Minute}
        }
        // ... rate limit check
    }
}
```

### 3. Input Validation
```go
// Validate all inputs
func (v *Validator) Validate(data map[string]interface{}) ValidationResult {
    result := ValidationResult{IsValid: true, Errors: make([]ValidationError, 0)}
    
    for field, rules := range v.rules {
        value, exists := data[field]
        if !exists {
            continue
        }
        
        for _, rule := range rules {
            if !rule.Check(value) {
                result.IsValid = false
                result.Errors = append(result.Errors, ValidationError{
                    Field:   field,
                    Message: rule.Message,
                })
            }
        }
    }
    
    return result
}
```

## 🏢 Industry Insights

### API Security Usage Patterns
- **Authentication**: JWT, OAuth2, API keys
- **Authorization**: Role-based access control
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Input Validation**: Prevent injection attacks

### Enterprise API Security Strategy
- **API Gateway**: Centralized security management
- **Microservices**: Service-to-service authentication
- **Monitoring**: Real-time threat detection
- **Compliance**: Meet regulatory requirements

## 🎯 Interview Questions

### Basic Level
1. **What is API security?**
   - Authentication and authorization
   - Input validation
   - Rate limiting
   - Encryption

2. **What is JWT?**
   - JSON Web Token
   - Stateless authentication
   - Self-contained
   - Signed and verified

3. **What is rate limiting?**
   - Request throttling
   - Abuse prevention
   - Resource protection
   - SLA enforcement

### Intermediate Level
4. **How do you implement JWT authentication?**
   ```go
   func (as *AuthService) ValidateToken(tokenString string) (*Claims, error) {
       token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
           return &as.jwtKey.PublicKey, nil
       })
       // ... validation logic
   }
   ```

5. **How do you implement rate limiting?**
   - Token bucket algorithm
   - Sliding window
   - Fixed window
   - Distributed rate limiting

6. **How do you handle input validation?**
   - Schema validation
   - Sanitization
   - Type checking
   - Length limits

### Advanced Level
7. **How do you implement API security at scale?**
   - API gateway
   - Microservices security
   - Distributed rate limiting
   - Global policies

8. **How do you handle API security monitoring?**
   - Real-time monitoring
   - Threat detection
   - Incident response
   - Security analytics

9. **How do you implement API security compliance?**
   - Regulatory requirements
   - Audit trails
   - Data protection
   - Privacy controls

---

**Next**: [Advanced Topics](./AdvancedTopics/) - Multi-cloud, hybrid cloud, edge computing, serverless architecture
