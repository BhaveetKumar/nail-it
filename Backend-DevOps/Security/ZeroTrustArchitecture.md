# 🛡️ Zero Trust Architecture: Network Security and Identity Verification

> **Master Zero Trust security model for comprehensive network protection and identity verification**

## 📚 Concept

Zero Trust Architecture is a security model that assumes no implicit trust based on network location or user identity. It requires continuous verification of every user, device, and network connection, regardless of whether they are inside or outside the corporate network.

### Key Features

- **Never Trust, Always Verify**: Continuous authentication and authorization
- **Least Privilege Access**: Minimal necessary permissions
- **Micro-segmentation**: Network isolation and control
- **Identity-Centric**: User and device identity verification
- **Continuous Monitoring**: Real-time security assessment
- **Data Protection**: Encryption and access controls

## 🏗️ Zero Trust Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Zero Trust Security Model               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   User      │  │   Device    │  │   Network   │     │
│  │  Identity   │  │  Identity   │  │  Identity   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Identity Provider                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   MFA       │  │   SSO       │  │   Device    │ │ │
│  │  │   Auth      │  │   Provider  │  │   Management│ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Policy Engine                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Access    │  │   Network   │  │   Data      │ │ │
│  │  │   Control   │  │   Policies  │  │   Policies  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Network   │  │   Data      │  │   Security  │     │
│  │  Segmentation│  │  Protection │  │  Monitoring │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Identity and Access Management

```go
// identity.go
package main

import (
    "context"
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
    CreatedAt    time.Time `json:"created_at"`
    UpdatedAt    time.Time `json:"updated_at"`
    LastLogin    time.Time `json:"last_login"`
    IsActive     bool      `json:"is_active"`
    MFAEnabled   bool      `json:"mfa_enabled"`
    MFASecret    string    `json:"mfa_secret"`
}

type Device struct {
    ID           string    `json:"id"`
    UserID       string    `json:"user_id"`
    DeviceID     string    `json:"device_id"`
    DeviceName   string    `json:"device_name"`
    DeviceType   string    `json:"device_type"`
    OS           string    `json:"os"`
    Browser      string    `json:"browser"`
    IPAddress    string    `json:"ip_address"`
    UserAgent    string    `json:"user_agent"`
    IsTrusted    bool      `json:"is_trusted"`
    LastSeen     time.Time `json:"last_seen"`
    CreatedAt    time.Time `json:"created_at"`
}

type Session struct {
    ID           string    `json:"id"`
    UserID       string    `json:"user_id"`
    DeviceID     string    `json:"device_id"`
    Token        string    `json:"token"`
    RefreshToken string    `json:"refresh_token"`
    ExpiresAt    time.Time `json:"expires_at"`
    CreatedAt    time.Time `json:"created_at"`
    IsActive     bool      `json:"is_active"`
    IPAddress    string    `json:"ip_address"`
    UserAgent    string    `json:"user_agent"`
}

type IdentityProvider struct {
    users   map[string]*User
    devices map[string]*Device
    sessions map[string]*Session
    jwtKey  *rsa.PrivateKey
}

func NewIdentityProvider() (*IdentityProvider, error) {
    // Generate RSA key pair for JWT signing
    privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil {
        return nil, fmt.Errorf("failed to generate RSA key: %w", err)
    }

    return &IdentityProvider{
        users:    make(map[string]*User),
        devices:  make(map[string]*Device),
        sessions: make(map[string]*Session),
        jwtKey:   privateKey,
    }, nil
}

func (ip *IdentityProvider) CreateUser(username, email, password string, roles []string) (*User, error) {
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
        Permissions:  ip.getPermissionsForRoles(roles),
        CreatedAt:    time.Now(),
        UpdatedAt:    time.Now(),
        IsActive:     true,
        MFAEnabled:   false,
    }

    ip.users[userID] = user
    return user, nil
}

func (ip *IdentityProvider) AuthenticateUser(username, password string) (*User, error) {
    // Find user by username
    var user *User
    for _, u := range ip.users {
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

    // Update last login
    user.LastLogin = time.Now()
    user.UpdatedAt = time.Now()

    return user, nil
}

func (ip *IdentityProvider) RegisterDevice(userID, deviceID, deviceName, deviceType, os, browser, ipAddress, userAgent string) (*Device, error) {
    device := &Device{
        ID:         generateID(),
        UserID:     userID,
        DeviceID:   deviceID,
        DeviceName: deviceName,
        DeviceType: deviceType,
        OS:         os,
        Browser:    browser,
        IPAddress:  ipAddress,
        UserAgent:  userAgent,
        IsTrusted:  false,
        LastSeen:   time.Now(),
        CreatedAt:  time.Now(),
    }

    ip.devices[device.ID] = device
    return device, nil
}

func (ip *IdentityProvider) CreateSession(userID, deviceID, ipAddress, userAgent string) (*Session, error) {
    // Generate JWT token
    token, err := ip.generateJWT(userID, deviceID)
    if err != nil {
        return nil, fmt.Errorf("failed to generate JWT: %w", err)
    }

    // Generate refresh token
    refreshToken := generateID()

    session := &Session{
        ID:           generateID(),
        UserID:       userID,
        DeviceID:     deviceID,
        Token:        token,
        RefreshToken: refreshToken,
        ExpiresAt:    time.Now().Add(24 * time.Hour),
        CreatedAt:    time.Now(),
        IsActive:     true,
        IPAddress:    ipAddress,
        UserAgent:    userAgent,
    }

    ip.sessions[session.ID] = session
    return session, nil
}

func (ip *IdentityProvider) ValidateSession(tokenString string) (*Session, error) {
    // Parse and validate JWT token
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return &ip.jwtKey.PublicKey, nil
    })

    if err != nil {
        return nil, fmt.Errorf("failed to parse token: %w", err)
    }

    if !token.Valid {
        return nil, fmt.Errorf("invalid token")
    }

    claims, ok := token.Claims.(jwt.MapClaims)
    if !ok {
        return nil, fmt.Errorf("invalid token claims")
    }

    sessionID, ok := claims["session_id"].(string)
    if !ok {
        return nil, fmt.Errorf("invalid session ID in token")
    }

    session, exists := ip.sessions[sessionID]
    if !exists {
        return nil, fmt.Errorf("session not found")
    }

    if !session.IsActive {
        return nil, fmt.Errorf("session is inactive")
    }

    if time.Now().After(session.ExpiresAt) {
        return nil, fmt.Errorf("session expired")
    }

    return session, nil
}

func (ip *IdentityProvider) generateJWT(userID, deviceID string) (string, error) {
    claims := jwt.MapClaims{
        "user_id":    userID,
        "device_id":  deviceID,
        "session_id": generateID(),
        "exp":        time.Now().Add(24 * time.Hour).Unix(),
        "iat":        time.Now().Unix(),
    }

    token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
    return token.SignedString(ip.jwtKey)
}

func (ip *IdentityProvider) getPermissionsForRoles(roles []string) []string {
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

func generateID() string {
    b := make([]byte, 16)
    rand.Read(b)
    return fmt.Sprintf("%x", b)
}
```

### Policy Engine

```go
// policy.go
package main

import (
    "context"
    "fmt"
    "time"
)

type Policy struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Rules       []PolicyRule           `json:"rules"`
    Conditions  []PolicyCondition      `json:"conditions"`
    Actions     []PolicyAction         `json:"actions"`
    Priority    int                    `json:"priority"`
    IsActive    bool                   `json:"is_active"`
    CreatedAt   time.Time              `json:"created_at"`
    UpdatedAt   time.Time              `json:"updated_at"`
}

type PolicyRule struct {
    ID          string                 `json:"id"`
    Type        string                 `json:"type"`
    Conditions  []PolicyCondition      `json:"conditions"`
    Actions     []PolicyAction         `json:"actions"`
    Priority    int                    `json:"priority"`
}

type PolicyCondition struct {
    ID        string                 `json:"id"`
    Type      string                 `json:"type"`
    Field     string                 `json:"field"`
    Operator  string                 `json:"operator"`
    Value     interface{}            `json:"value"`
    Metadata  map[string]interface{} `json:"metadata"`
}

type PolicyAction struct {
    ID       string                 `json:"id"`
    Type     string                 `json:"type"`
    Name     string                 `json:"name"`
    Params   map[string]interface{} `json:"params"`
    Metadata map[string]interface{} `json:"metadata"`
}

type PolicyEngine struct {
    policies map[string]*Policy
    rules    map[string]*PolicyRule
}

func NewPolicyEngine() *PolicyEngine {
    return &PolicyEngine{
        policies: make(map[string]*Policy),
        rules:    make(map[string]*PolicyRule),
    }
}

func (pe *PolicyEngine) CreatePolicy(name, description string, rules []PolicyRule, conditions []PolicyCondition, actions []PolicyAction, priority int) (*Policy, error) {
    policy := &Policy{
        ID:          generateID(),
        Name:        name,
        Description: description,
        Rules:       rules,
        Conditions:  conditions,
        Actions:     actions,
        Priority:    priority,
        IsActive:    true,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }

    pe.policies[policy.ID] = policy
    return policy, nil
}

func (pe *PolicyEngine) EvaluatePolicy(ctx context.Context, policyID string, context map[string]interface{}) ([]PolicyAction, error) {
    policy, exists := pe.policies[policyID]
    if !exists {
        return nil, fmt.Errorf("policy not found")
    }

    if !policy.IsActive {
        return nil, fmt.Errorf("policy is inactive")
    }

    // Evaluate conditions
    if !pe.evaluateConditions(policy.Conditions, context) {
        return nil, fmt.Errorf("policy conditions not met")
    }

    // Evaluate rules
    actions := make([]PolicyAction, 0)
    for _, rule := range policy.Rules {
        if pe.evaluateRule(rule, context) {
            actions = append(actions, rule.Actions...)
        }
    }

    return actions, nil
}

func (pe *PolicyEngine) evaluateConditions(conditions []PolicyCondition, context map[string]interface{}) bool {
    for _, condition := range conditions {
        if !pe.evaluateCondition(condition, context) {
            return false
        }
    }
    return true
}

func (pe *PolicyEngine) evaluateCondition(condition PolicyCondition, context map[string]interface{}) bool {
    value, exists := context[condition.Field]
    if !exists {
        return false
    }

    switch condition.Operator {
    case "equals":
        return value == condition.Value
    case "not_equals":
        return value != condition.Value
    case "contains":
        if str, ok := value.(string); ok {
            if target, ok := condition.Value.(string); ok {
                return contains(str, target)
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
    case "greater_than":
        if num, ok := value.(float64); ok {
            if target, ok := condition.Value.(float64); ok {
                return num > target
            }
        }
        return false
    case "less_than":
        if num, ok := value.(float64); ok {
            if target, ok := condition.Value.(float64); ok {
                return num < target
            }
        }
        return false
    default:
        return false
    }
}

func (pe *PolicyEngine) evaluateRule(rule PolicyRule, context map[string]interface{}) bool {
    return pe.evaluateConditions(rule.Conditions, context)
}

func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}
```

### Network Segmentation

```go
// network.go
package main

import (
    "context"
    "fmt"
    "net"
    "time"
)

type NetworkSegment struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    CIDR        string    `json:"cidr"`
    Description string    `json:"description"`
    IsActive    bool      `json:"is_active"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

type NetworkRule struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Source      string    `json:"source"`
    Destination string    `json:"destination"`
    Protocol    string    `json:"protocol"`
    Port        int       `json:"port"`
    Action      string    `json:"action"`
    Priority    int       `json:"priority"`
    IsActive    bool      `json:"is_active"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

type NetworkController struct {
    segments map[string]*NetworkSegment
    rules    map[string]*NetworkRule
}

func NewNetworkController() *NetworkController {
    return &NetworkController{
        segments: make(map[string]*NetworkSegment),
        rules:    make(map[string]*NetworkRule),
    }
}

func (nc *NetworkController) CreateSegment(name, cidr, description string) (*NetworkSegment, error) {
    // Validate CIDR
    _, _, err := net.ParseCIDR(cidr)
    if err != nil {
        return nil, fmt.Errorf("invalid CIDR: %w", err)
    }

    segment := &NetworkSegment{
        ID:          generateID(),
        Name:        name,
        CIDR:        cidr,
        Description: description,
        IsActive:    true,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }

    nc.segments[segment.ID] = segment
    return segment, nil
}

func (nc *NetworkController) CreateRule(name, source, destination, protocol string, port int, action string, priority int) (*NetworkRule, error) {
    rule := &NetworkRule{
        ID:          generateID(),
        Name:        name,
        Source:      source,
        Destination: destination,
        Protocol:    protocol,
        Port:        port,
        Action:      action,
        Priority:    priority,
        IsActive:    true,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }

    nc.rules[rule.ID] = rule
    return rule, nil
}

func (nc *NetworkController) CheckAccess(ctx context.Context, sourceIP, destIP string, protocol string, port int) (bool, error) {
    // Find applicable rules
    applicableRules := make([]*NetworkRule, 0)

    for _, rule := range nc.rules {
        if !rule.IsActive {
            continue
        }

        if nc.ruleMatches(rule, sourceIP, destIP, protocol, port) {
            applicableRules = append(applicableRules, rule)
        }
    }

    if len(applicableRules) == 0 {
        return false, fmt.Errorf("no applicable rules found")
    }

    // Sort by priority (higher priority first)
    for i := 0; i < len(applicableRules); i++ {
        for j := i + 1; j < len(applicableRules); j++ {
            if applicableRules[i].Priority < applicableRules[j].Priority {
                applicableRules[i], applicableRules[j] = applicableRules[j], applicableRules[i]
            }
        }
    }

    // Apply first matching rule
    rule := applicableRules[0]
    return rule.Action == "allow", nil
}

func (nc *NetworkController) ruleMatches(rule *NetworkRule, sourceIP, destIP, protocol string, port int) bool {
    // Check source IP
    if !nc.ipMatches(rule.Source, sourceIP) {
        return false
    }

    // Check destination IP
    if !nc.ipMatches(rule.Destination, destIP) {
        return false
    }

    // Check protocol
    if rule.Protocol != "any" && rule.Protocol != protocol {
        return false
    }

    // Check port
    if rule.Port != 0 && rule.Port != port {
        return false
    }

    return true
}

func (nc *NetworkController) ipMatches(pattern, ip string) bool {
    if pattern == "any" {
        return true
    }

    // Check if pattern is a CIDR
    if _, network, err := net.ParseCIDR(pattern); err == nil {
        if ipAddr := net.ParseIP(ip); ipAddr != nil {
            return network.Contains(ipAddr)
        }
    }

    // Check if pattern is an exact IP
    return pattern == ip
}
```

### Security Monitoring

```go
// monitoring.go
package main

import (
    "context"
    "fmt"
    "time"
)

type SecurityEvent struct {
    ID          string                 `json:"id"`
    Type        string                 `json:"type"`
    Severity    string                 `json:"severity"`
    Source      string                 `json:"source"`
    Destination string                 `json:"destination"`
    UserID      string                 `json:"user_id"`
    DeviceID    string                 `json:"device_id"`
    Action      string                 `json:"action"`
    Result      string                 `json:"result"`
    Details     map[string]interface{} `json:"details"`
    Timestamp   time.Time              `json:"timestamp"`
}

type SecurityMonitor struct {
    events []SecurityEvent
    rules  []SecurityRule
}

type SecurityRule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Type        string                 `json:"type"`
    Conditions  []SecurityCondition    `json:"conditions"`
    Actions     []SecurityAction       `json:"actions"`
    IsActive    bool                   `json:"is_active"`
}

type SecurityCondition struct {
    ID       string                 `json:"id"`
    Field    string                 `json:"field"`
    Operator string                 `json:"operator"`
    Value    interface{}            `json:"value"`
}

type SecurityAction struct {
    ID       string                 `json:"id"`
    Type     string                 `json:"type"`
    Params   map[string]interface{} `json:"params"`
}

func NewSecurityMonitor() *SecurityMonitor {
    return &SecurityMonitor{
        events: make([]SecurityEvent, 0),
        rules:  make([]SecurityRule, 0),
    }
}

func (sm *SecurityMonitor) LogEvent(eventType, severity, source, destination, userID, deviceID, action, result string, details map[string]interface{}) {
    event := SecurityEvent{
        ID:          generateID(),
        Type:        eventType,
        Severity:    severity,
        Source:      source,
        Destination: destination,
        UserID:      userID,
        DeviceID:    deviceID,
        Action:      action,
        Result:      result,
        Details:     details,
        Timestamp:   time.Now(),
    }

    sm.events = append(sm.events, event)

    // Check against security rules
    sm.checkSecurityRules(event)
}

func (sm *SecurityMonitor) checkSecurityRules(event SecurityEvent) {
    for _, rule := range sm.rules {
        if !rule.IsActive {
            continue
        }

        if sm.ruleMatches(rule, event) {
            sm.executeActions(rule.Actions, event)
        }
    }
}

func (sm *SecurityMonitor) ruleMatches(rule SecurityRule, event SecurityEvent) bool {
    for _, condition := range rule.Conditions {
        if !sm.conditionMatches(condition, event) {
            return false
        }
    }
    return true
}

func (sm *SecurityMonitor) conditionMatches(condition SecurityCondition, event SecurityEvent) bool {
    var value interface{}

    switch condition.Field {
    case "type":
        value = event.Type
    case "severity":
        value = event.Severity
    case "source":
        value = event.Source
    case "destination":
        value = event.Destination
    case "user_id":
        value = event.UserID
    case "device_id":
        value = event.DeviceID
    case "action":
        value = event.Action
    case "result":
        value = event.Result
    default:
        if detailValue, exists := event.Details[condition.Field]; exists {
            value = detailValue
        } else {
            return false
        }
    }

    switch condition.Operator {
    case "equals":
        return value == condition.Value
    case "not_equals":
        return value != condition.Value
    case "contains":
        if str, ok := value.(string); ok {
            if target, ok := condition.Value.(string); ok {
                return contains(str, target)
            }
        }
        return false
    default:
        return false
    }
}

func (sm *SecurityMonitor) executeActions(actions []SecurityAction, event SecurityEvent) {
    for _, action := range actions {
        switch action.Type {
        case "alert":
            sm.sendAlert(action.Params, event)
        case "block":
            sm.blockAccess(action.Params, event)
        case "log":
            sm.logAction(action.Params, event)
        case "notify":
            sm.notify(action.Params, event)
        }
    }
}

func (sm *SecurityMonitor) sendAlert(params map[string]interface{}, event SecurityEvent) {
    // Send alert to security team
    fmt.Printf("SECURITY ALERT: %s - %s\n", event.Type, event.Details)
}

func (sm *SecurityMonitor) blockAccess(params map[string]interface{}, event SecurityEvent) {
    // Block access for user/device
    fmt.Printf("BLOCKING ACCESS: %s - %s\n", event.UserID, event.DeviceID)
}

func (sm *SecurityMonitor) logAction(params map[string]interface{}, event SecurityEvent) {
    // Log security action
    fmt.Printf("SECURITY ACTION: %s - %s\n", event.Type, event.Action)
}

func (sm *SecurityMonitor) notify(params map[string]interface{}, event SecurityEvent) {
    // Send notification
    fmt.Printf("NOTIFICATION: %s - %s\n", event.Type, event.Details)
}
```

### Zero Trust Middleware

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

type ZeroTrustMiddleware struct {
    identityProvider *IdentityProvider
    policyEngine     *PolicyEngine
    networkController *NetworkController
    securityMonitor  *SecurityMonitor
    logger           *zap.Logger
}

func NewZeroTrustMiddleware(ip *IdentityProvider, pe *PolicyEngine, nc *NetworkController, sm *SecurityMonitor, logger *zap.Logger) *ZeroTrustMiddleware {
    return &ZeroTrustMiddleware{
        identityProvider:  ip,
        policyEngine:      pe,
        networkController: nc,
        securityMonitor:   sm,
        logger:            logger,
    }
}

func (ztm *ZeroTrustMiddleware) AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Extract token from Authorization header
        authHeader := c.GetHeader("Authorization")
        if authHeader == "" {
            ztm.logger.Warn("Missing authorization header",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Missing authorization header"})
            c.Abort()
            return
        }

        tokenString := strings.TrimPrefix(authHeader, "Bearer ")
        if tokenString == authHeader {
            ztm.logger.Warn("Invalid authorization header format",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization header format"})
            c.Abort()
            return
        }

        // Validate session
        session, err := ztm.identityProvider.ValidateSession(tokenString)
        if err != nil {
            ztm.logger.Warn("Invalid session",
                zap.String("ip", c.ClientIP()),
                zap.String("user_agent", c.GetHeader("User-Agent")),
                zap.Error(err),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid session"})
            c.Abort()
            return
        }

        // Get user and device information
        user, exists := ztm.identityProvider.users[session.UserID]
        if !exists {
            ztm.logger.Warn("User not found",
                zap.String("user_id", session.UserID),
                zap.String("ip", c.ClientIP()),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "User not found"})
            c.Abort()
            return
        }

        device, exists := ztm.identityProvider.devices[session.DeviceID]
        if !exists {
            ztm.logger.Warn("Device not found",
                zap.String("device_id", session.DeviceID),
                zap.String("ip", c.ClientIP()),
            )
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Device not found"})
            c.Abort()
            return
        }

        // Check network access
        allowed, err := ztm.networkController.CheckAccess(
            context.Background(),
            c.ClientIP(),
            c.Request.Host,
            "tcp",
            80,
        )
        if err != nil || !allowed {
            ztm.logger.Warn("Network access denied",
                zap.String("user_id", session.UserID),
                zap.String("device_id", session.DeviceID),
                zap.String("ip", c.ClientIP()),
                zap.Error(err),
            )
            c.JSON(http.StatusForbidden, gin.H{"error": "Network access denied"})
            c.Abort()
            return
        }

        // Log security event
        ztm.securityMonitor.LogEvent(
            "authentication",
            "info",
            c.ClientIP(),
            c.Request.Host,
            session.UserID,
            session.DeviceID,
            "access_request",
            "allowed",
            map[string]interface{}{
                "endpoint": c.FullPath(),
                "method":   c.Request.Method,
                "user_agent": c.GetHeader("User-Agent"),
            },
        )

        // Store user and device in context
        c.Set("user", user)
        c.Set("device", device)
        c.Set("session", session)

        c.Next()
    }
}

func (ztm *ZeroTrustMiddleware) PolicyMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        user, exists := c.Get("user")
        if !exists {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "User not found in context"})
            c.Abort()
            return
        }

        device, exists := c.Get("device")
        if !exists {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Device not found in context"})
            c.Abort()
            return
        }

        // Create policy context
        policyContext := map[string]interface{}{
            "user_id":    user.(*User).ID,
            "username":   user.(*User).Username,
            "roles":      user.(*User).Roles,
            "permissions": user.(*User).Permissions,
            "device_id":  device.(*Device).DeviceID,
            "device_type": device.(*Device).DeviceType,
            "ip_address": c.ClientIP(),
            "endpoint":   c.FullPath(),
            "method":     c.Request.Method,
            "timestamp":  time.Now(),
        }

        // Evaluate policies
        actions, err := ztm.policyEngine.EvaluatePolicy(
            context.Background(),
            "default-policy",
            policyContext,
        )
        if err != nil {
            ztm.logger.Warn("Policy evaluation failed",
                zap.String("user_id", user.(*User).ID),
                zap.String("endpoint", c.FullPath()),
                zap.Error(err),
            )
            c.JSON(http.StatusForbidden, gin.H{"error": "Policy evaluation failed"})
            c.Abort()
            return
        }

        // Check if access is allowed
        allowed := false
        for _, action := range actions {
            if action.Type == "allow" {
                allowed = true
                break
            }
        }

        if !allowed {
            ztm.logger.Warn("Access denied by policy",
                zap.String("user_id", user.(*User).ID),
                zap.String("endpoint", c.FullPath()),
            )
            c.JSON(http.StatusForbidden, gin.H{"error": "Access denied by policy"})
            c.Abort()
            return
        }

        // Log policy decision
        ztm.securityMonitor.LogEvent(
            "policy_evaluation",
            "info",
            c.ClientIP(),
            c.Request.Host,
            user.(*User).ID,
            device.(*Device).DeviceID,
            "policy_check",
            "allowed",
            map[string]interface{}{
                "endpoint": c.FullPath(),
                "method":   c.Request.Method,
                "actions":  actions,
            },
        )

        c.Next()
    }
}
```

## 🚀 Best Practices

### 1. Identity Verification

```go
// Implement multi-factor authentication
func (ip *IdentityProvider) EnableMFA(userID string) error {
    user, exists := ip.users[userID]
    if !exists {
        return fmt.Errorf("user not found")
    }

    user.MFAEnabled = true
    user.MFASecret = generateMFASecret()
    return nil
}
```

### 2. Least Privilege Access

```go
// Implement role-based access control
func (ip *IdentityProvider) getPermissionsForRoles(roles []string) []string {
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

### 3. Continuous Monitoring

```go
// Monitor all access attempts
func (ztm *ZeroTrustMiddleware) LogAccess(userID, deviceID, endpoint, result string) {
    ztm.securityMonitor.LogEvent(
        "access_attempt",
        "info",
        ztm.getClientIP(),
        ztm.getServerHost(),
        userID,
        deviceID,
        "access",
        result,
        map[string]interface{}{
            "endpoint": endpoint,
            "timestamp": time.Now(),
        },
    )
}
```

## 🏢 Industry Insights

### Zero Trust Usage Patterns

- **Network Security**: Micro-segmentation and access control
- **Identity Management**: Multi-factor authentication and device trust
- **Data Protection**: Encryption and access controls
- **Security Monitoring**: Real-time threat detection

### Enterprise Zero Trust Strategy

- **Phased Implementation**: Gradual rollout across organization
- **Technology Integration**: Multiple security tools and platforms
- **User Experience**: Seamless authentication and access
- **Compliance**: Meet regulatory requirements

## 🎯 Interview Questions

### Basic Level

1. **What is Zero Trust Architecture?**

   - Never trust, always verify
   - Continuous authentication
   - Least privilege access
   - Micro-segmentation

2. **What are the key principles of Zero Trust?**

   - Verify explicitly
   - Use least privilege access
   - Assume breach
   - Continuous monitoring

3. **What is micro-segmentation?**
   - Network isolation
   - Granular access control
   - Security boundaries
   - Traffic inspection

### Intermediate Level

4. **How do you implement Zero Trust?**

   - Identity and access management
   - Network segmentation
   - Policy enforcement
   - Security monitoring

5. **How do you handle device trust?**

   - Device registration
   - Device compliance
   - Trust scoring
   - Access policies

6. **How do you implement continuous monitoring?**
   - Real-time monitoring
   - Threat detection
   - Incident response
   - Security analytics

### Advanced Level

7. **How do you implement Zero Trust at scale?**

   - Distributed architecture
   - Performance optimization
   - Scalable policies
   - Global deployment

8. **How do you handle Zero Trust compliance?**

   - Regulatory requirements
   - Audit trails
   - Data protection
   - Privacy controls

9. **How do you implement Zero Trust security?**
   - Threat modeling
   - Risk assessment
   - Security controls
   - Incident response

---

**Next**: [Secure APIs](./SecureAPIs.md) - API security, authentication, authorization, rate limiting
