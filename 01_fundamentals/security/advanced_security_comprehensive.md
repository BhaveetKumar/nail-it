# Advanced Security Comprehensive

Comprehensive guide to advanced security for senior backend engineers.

## 🎯 Zero Trust Architecture

### Zero Trust Implementation
```go
// Zero Trust Security Implementation
package security

import (
    "context"
    "crypto/tls"
    "crypto/x509"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
    "github.com/redis/go-redis/v9"
    "golang.org/x/crypto/bcrypt"
)

type ZeroTrustSecurity struct {
    jwtSecret     []byte
    redisClient   *redis.Client
    certPool      *x509.CertPool
    policyEngine  *PolicyEngine
    auditLogger   *AuditLogger
}

type PolicyEngine struct {
    policies map[string]*SecurityPolicy
    mutex    sync.RWMutex
}

type SecurityPolicy struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Rules       []PolicyRule           `json:"rules"`
    Conditions  map[string]interface{} `json:"conditions"`
    Actions     []PolicyAction         `json:"actions"`
    Priority    int                    `json:"priority"`
    Enabled     bool                   `json:"enabled"`
    CreatedAt   time.Time              `json:"created_at"`
    UpdatedAt   time.Time              `json:"updated_at"`
}

type PolicyRule struct {
    ID          string                 `json:"id"`
    Type        string                 `json:"type"` // "allow", "deny", "require_mfa"
    Resource    string                 `json:"resource"`
    Action      string                 `json:"action"`
    Conditions  map[string]interface{} `json:"conditions"`
    Priority    int                    `json:"priority"`
}

type PolicyAction struct {
    Type    string                 `json:"type"` // "log", "alert", "block", "require_mfa"
    Config  map[string]interface{} `json:"config"`
}

type AuditLogger struct {
    client *http.Client
    endpoint string
}

func NewZeroTrustSecurity(jwtSecret []byte, redisClient *redis.Client) *ZeroTrustSecurity {
    return &ZeroTrustSecurity{
        jwtSecret:    jwtSecret,
        redisClient:  redisClient,
        certPool:     x509.NewCertPool(),
        policyEngine: NewPolicyEngine(),
        auditLogger:  NewAuditLogger(),
    }
}

func (zts *ZeroTrustSecurity) Authenticate(ctx context.Context, token string) (*User, error) {
    // Parse and validate JWT token
    claims, err := zts.validateJWT(token)
    if err != nil {
        zts.auditLogger.Log(ctx, "authentication_failed", map[string]interface{}{
            "error": err.Error(),
            "token": token[:10] + "...",
        })
        return nil, err
    }
    
    // Check if token is blacklisted
    if zts.isTokenBlacklisted(ctx, token) {
        zts.auditLogger.Log(ctx, "blacklisted_token_used", map[string]interface{}{
            "user_id": claims.UserID,
            "token":   token[:10] + "...",
        })
        return nil, fmt.Errorf("token is blacklisted")
    }
    
    // Get user from database
    user, err := zts.getUser(ctx, claims.UserID)
    if err != nil {
        return nil, err
    }
    
    // Check if user is active
    if !user.IsActive {
        zts.auditLogger.Log(ctx, "inactive_user_attempt", map[string]interface{}{
            "user_id": user.ID,
        })
        return nil, fmt.Errorf("user is inactive")
    }
    
    // Update last seen
    zts.updateLastSeen(ctx, user.ID)
    
    zts.auditLogger.Log(ctx, "authentication_success", map[string]interface{}{
        "user_id": user.ID,
        "email":   user.Email,
    })
    
    return user, nil
}

func (zts *ZeroTrustSecurity) Authorize(ctx context.Context, user *User, resource string, action string) (bool, error) {
    // Get applicable policies
    policies := zts.policyEngine.GetApplicablePolicies(user, resource, action)
    
    // Evaluate policies in priority order
    for _, policy := range policies {
        if !policy.Enabled {
            continue
        }
        
        // Check if policy conditions are met
        if !zts.evaluateConditions(ctx, policy.Conditions, user) {
            continue
        }
        
        // Evaluate policy rules
        for _, rule := range policy.Rules {
            if zts.evaluateRule(ctx, rule, user, resource, action) {
                // Execute policy actions
                for _, action := range policy.Actions {
                    zts.executeAction(ctx, action, user, resource)
                }
                
                // Return rule decision
                if rule.Type == "allow" {
                    zts.auditLogger.Log(ctx, "authorization_granted", map[string]interface{}{
                        "user_id": user.ID,
                        "resource": resource,
                        "action": action,
                        "policy_id": policy.ID,
                        "rule_id": rule.ID,
                    })
                    return true, nil
                } else if rule.Type == "deny" {
                    zts.auditLogger.Log(ctx, "authorization_denied", map[string]interface{}{
                        "user_id": user.ID,
                        "resource": resource,
                        "action": action,
                        "policy_id": policy.ID,
                        "rule_id": rule.ID,
                    })
                    return false, fmt.Errorf("access denied by policy")
                }
            }
        }
    }
    
    // Default deny
    zts.auditLogger.Log(ctx, "authorization_denied_default", map[string]interface{}{
        "user_id": user.ID,
        "resource": resource,
        "action": action,
    })
    return false, fmt.Errorf("access denied")
}

func (zts *ZeroTrustSecurity) validateJWT(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return zts.jwtSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        // Check token expiration
        if claims.ExpiresAt != nil && time.Now().After(claims.ExpiresAt.Time) {
            return nil, fmt.Errorf("token expired")
        }
        
        // Check token issued at
        if claims.IssuedAt != nil && time.Now().Before(claims.IssuedAt.Time) {
            return nil, fmt.Errorf("token not yet valid")
        }
        
        return claims, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}

func (zts *ZeroTrustSecurity) isTokenBlacklisted(ctx context.Context, token string) bool {
    tokenHash := zts.hashToken(token)
    result := zts.redisClient.Get(ctx, "blacklist:"+tokenHash)
    return result.Err() == nil
}

func (zts *ZeroTrustSecurity) hashToken(token string) string {
    // Use SHA-256 to hash the token
    h := sha256.New()
    h.Write([]byte(token))
    return fmt.Sprintf("%x", h.Sum(nil))
}

func (zts *ZeroTrustSecurity) evaluateConditions(ctx context.Context, conditions map[string]interface{}, user *User) bool {
    for key, value := range conditions {
        switch key {
        case "time_range":
            if !zts.evaluateTimeRange(ctx, value) {
                return false
            }
        case "location":
            if !zts.evaluateLocation(ctx, value, user) {
                return false
            }
        case "device":
            if !zts.evaluateDevice(ctx, value, user) {
                return false
            }
        case "risk_score":
            if !zts.evaluateRiskScore(ctx, value, user) {
                return false
            }
        }
    }
    return true
}

func (zts *ZeroTrustSecurity) evaluateRule(ctx context.Context, rule *PolicyRule, user *User, resource string, action string) bool {
    // Check resource match
    if !zts.matchResource(rule.Resource, resource) {
        return false
    }
    
    // Check action match
    if !zts.matchAction(rule.Action, action) {
        return false
    }
    
    // Check rule conditions
    if !zts.evaluateConditions(ctx, rule.Conditions, user) {
        return false
    }
    
    return true
}

func (zts *ZeroTrustSecurity) matchResource(ruleResource string, actualResource string) bool {
    // Support wildcard matching
    if strings.Contains(ruleResource, "*") {
        pattern := strings.ReplaceAll(ruleResource, "*", ".*")
        matched, _ := regexp.MatchString("^"+pattern+"$", actualResource)
        return matched
    }
    return ruleResource == actualResource
}

func (zts *ZeroTrustSecurity) matchAction(ruleAction string, actualAction string) bool {
    if ruleAction == "*" {
        return true
    }
    return ruleAction == actualAction
}

func (zts *ZeroTrustSecurity) executeAction(ctx context.Context, action *PolicyAction, user *User, resource string) {
    switch action.Type {
    case "log":
        zts.auditLogger.Log(ctx, "policy_action_log", map[string]interface{}{
            "user_id": user.ID,
            "resource": resource,
            "action_config": action.Config,
        })
    case "alert":
        zts.sendAlert(ctx, action.Config, user, resource)
    case "block":
        zts.blockUser(ctx, user.ID, action.Config)
    case "require_mfa":
        zts.requireMFA(ctx, user.ID, action.Config)
    }
}

// Policy Engine Implementation
func NewPolicyEngine() *PolicyEngine {
    return &PolicyEngine{
        policies: make(map[string]*SecurityPolicy),
    }
}

func (pe *PolicyEngine) AddPolicy(policy *SecurityPolicy) {
    pe.mutex.Lock()
    defer pe.mutex.Unlock()
    
    policy.CreatedAt = time.Now()
    policy.UpdatedAt = time.Now()
    pe.policies[policy.ID] = policy
}

func (pe *PolicyEngine) GetApplicablePolicies(user *User, resource string, action string) []*SecurityPolicy {
    pe.mutex.RLock()
    defer pe.mutex.RUnlock()
    
    var applicable []*SecurityPolicy
    
    for _, policy := range pe.policies {
        if !policy.Enabled {
            continue
        }
        
        // Check if policy applies to this resource and action
        if pe.policyApplies(policy, resource, action) {
            applicable = append(applicable, policy)
        }
    }
    
    // Sort by priority (higher priority first)
    sort.Slice(applicable, func(i, j int) bool {
        return applicable[i].Priority > applicable[j].Priority
    })
    
    return applicable
}

func (pe *PolicyEngine) policyApplies(policy *SecurityPolicy, resource string, action string) bool {
    for _, rule := range policy.Rules {
        if pe.matchResource(rule.Resource, resource) && pe.matchAction(rule.Action, action) {
            return true
        }
    }
    return false
}

func (pe *PolicyEngine) matchResource(ruleResource string, actualResource string) bool {
    if strings.Contains(ruleResource, "*") {
        pattern := strings.ReplaceAll(ruleResource, "*", ".*")
        matched, _ := regexp.MatchString("^"+pattern+"$", actualResource)
        return matched
    }
    return ruleResource == actualResource
}

func (pe *PolicyEngine) matchAction(ruleAction string, actualAction string) bool {
    if ruleAction == "*" {
        return true
    }
    return ruleAction == actualAction
}

// Audit Logger Implementation
func NewAuditLogger() *AuditLogger {
    return &AuditLogger{
        client: &http.Client{
            Timeout: 10 * time.Second,
            Transport: &http.Transport{
                TLSClientConfig: &tls.Config{
                    InsecureSkipVerify: false,
                },
            },
        },
        endpoint: "https://audit.example.com/api/events",
    }
}

func (al *AuditLogger) Log(ctx context.Context, eventType string, data map[string]interface{}) {
    event := map[string]interface{}{
        "event_type": eventType,
        "timestamp":  time.Now().UTC().Format(time.RFC3339),
        "data":       data,
    }
    
    jsonData, err := json.Marshal(event)
    if err != nil {
        log.Printf("Failed to marshal audit event: %v", err)
        return
    }
    
    req, err := http.NewRequestWithContext(ctx, "POST", al.endpoint, bytes.NewBuffer(jsonData))
    if err != nil {
        log.Printf("Failed to create audit request: %v", err)
        return
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+os.Getenv("AUDIT_API_KEY"))
    
    resp, err := al.client.Do(req)
    if err != nil {
        log.Printf("Failed to send audit event: %v", err)
        return
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        log.Printf("Audit API returned status %d", resp.StatusCode)
    }
}
```

### Advanced Encryption
```go
// Advanced Encryption Implementation
package encryption

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/base64"
    "encoding/pem"
    "fmt"
    "io"
)

type AdvancedEncryption struct {
    aesKey    []byte
    rsaKey    *rsa.PrivateKey
    publicKey *rsa.PublicKey
}

func NewAdvancedEncryption(aesKey []byte, rsaKeyPath string) (*AdvancedEncryption, error) {
    // Load RSA private key
    rsaKey, err := loadRSAPrivateKey(rsaKeyPath)
    if err != nil {
        return nil, err
    }
    
    return &AdvancedEncryption{
        aesKey:    aesKey,
        rsaKey:    rsaKey,
        publicKey: &rsaKey.PublicKey,
    }, nil
}

func (ae *AdvancedEncryption) EncryptAES(plaintext []byte) ([]byte, error) {
    block, err := aes.NewCipher(ae.aesKey)
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

func (ae *AdvancedEncryption) DecryptAES(ciphertext []byte) ([]byte, error) {
    block, err := aes.NewCipher(ae.aesKey)
    if err != nil {
        return nil, err
    }
    
    // Create GCM mode
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    // Extract nonce
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    
    // Decrypt
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

func (ae *AdvancedEncryption) EncryptRSA(plaintext []byte) ([]byte, error) {
    // Use OAEP padding
    ciphertext, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, ae.publicKey, plaintext, nil)
    if err != nil {
        return nil, err
    }
    
    return ciphertext, nil
}

func (ae *AdvancedEncryption) DecryptRSA(ciphertext []byte) ([]byte, error) {
    // Use OAEP padding
    plaintext, err := rsa.DecryptOAEP(sha256.New(), rand.Reader, ae.rsaKey, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

func (ae *AdvancedEncryption) HybridEncrypt(plaintext []byte) ([]byte, error) {
    // Generate random AES key
    aesKey := make([]byte, 32) // 256 bits
    if _, err := io.ReadFull(rand.Reader, aesKey); err != nil {
        return nil, err
    }
    
    // Encrypt data with AES
    block, err := aes.NewCipher(aesKey)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    
    // Encrypt AES key with RSA
    encryptedKey, err := ae.EncryptRSA(aesKey)
    if err != nil {
        return nil, err
    }
    
    // Combine encrypted key and ciphertext
    result := make([]byte, len(encryptedKey)+len(ciphertext))
    copy(result, encryptedKey)
    copy(result[len(encryptedKey):], ciphertext)
    
    return result, nil
}

func (ae *AdvancedEncryption) HybridDecrypt(encryptedData []byte) ([]byte, error) {
    // Extract encrypted key and ciphertext
    keySize := ae.publicKey.Size()
    if len(encryptedData) < keySize {
        return nil, fmt.Errorf("encrypted data too short")
    }
    
    encryptedKey := encryptedData[:keySize]
    ciphertext := encryptedData[keySize:]
    
    // Decrypt AES key
    aesKey, err := ae.DecryptRSA(encryptedKey)
    if err != nil {
        return nil, err
    }
    
    // Decrypt data with AES
    block, err := aes.NewCipher(aesKey)
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
    
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

func loadRSAPrivateKey(keyPath string) (*rsa.PrivateKey, error) {
    keyData, err := os.ReadFile(keyPath)
    if err != nil {
        return nil, err
    }
    
    block, _ := pem.Decode(keyData)
    if block == nil {
        return nil, fmt.Errorf("failed to decode PEM block")
    }
    
    key, err := x509.ParsePKCS1PrivateKey(block.Bytes)
    if err != nil {
        return nil, err
    }
    
    return key, nil
}
```

## 🚀 Security Monitoring and Incident Response

### Security Monitoring System
```go
// Security Monitoring and Alerting System
package security

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
    
    "github.com/redis/go-redis/v9"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

type SecurityMonitor struct {
    redisClient     *redis.Client
    alertManager    *AlertManager
    metrics         *SecurityMetrics
    threatDetector  *ThreatDetector
    incidentManager *IncidentManager
}

type SecurityMetrics struct {
    failedLogins     prometheus.Counter
    blockedRequests  prometheus.Counter
    securityAlerts   prometheus.Counter
    activeThreats    prometheus.Gauge
    responseTime     prometheus.Histogram
}

type ThreatDetector struct {
    rules []ThreatRule
    mutex sync.RWMutex
}

type ThreatRule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Pattern     string                 `json:"pattern"`
    Severity    string                 `json:"severity"`
    Conditions  map[string]interface{} `json:"conditions"`
    Actions     []string               `json:"actions"`
    Enabled     bool                   `json:"enabled"`
}

type IncidentManager struct {
    incidents map[string]*SecurityIncident
    mutex     sync.RWMutex
}

type SecurityIncident struct {
    ID          string                 `json:"id"`
    Type        string                 `json:"type"`
    Severity    string                 `json:"severity"`
    Status      string                 `json:"status"`
    Description string                 `json:"description"`
    Evidence    map[string]interface{} `json:"evidence"`
    CreatedAt   time.Time              `json:"created_at"`
    UpdatedAt   time.Time              `json:"updated_at"`
    AssignedTo  string                 `json:"assigned_to"`
}

func NewSecurityMonitor(redisClient *redis.Client) *SecurityMonitor {
    return &SecurityMonitor{
        redisClient:     redisClient,
        alertManager:    NewAlertManager(),
        metrics:         NewSecurityMetrics(),
        threatDetector:  NewThreatDetector(),
        incidentManager: NewIncidentManager(),
    }
}

func (sm *SecurityMonitor) MonitorRequest(ctx context.Context, req *SecurityRequest) {
    start := time.Now()
    
    // Check for threats
    threats := sm.threatDetector.DetectThreats(ctx, req)
    
    // Update metrics
    sm.metrics.responseTime.Observe(time.Since(start).Seconds())
    
    if len(threats) > 0 {
        sm.metrics.activeThreats.Set(float64(len(threats)))
        sm.metrics.securityAlerts.Inc()
        
        // Create incident
        incident := sm.incidentManager.CreateIncident(ctx, threats[0])
        
        // Send alerts
        sm.alertManager.SendAlert(ctx, incident)
        
        // Block request if high severity
        if threats[0].Severity == "high" {
            sm.metrics.blockedRequests.Inc()
            req.Blocked = true
        }
    }
}

func (sm *SecurityMonitor) MonitorLogin(ctx context.Context, login *LoginAttempt) {
    // Check for brute force attacks
    if sm.isBruteForceAttempt(ctx, login) {
        sm.metrics.failedLogins.Inc()
        
        // Create incident
        incident := &SecurityIncident{
            ID:          generateIncidentID(),
            Type:        "brute_force",
            Severity:    "high",
            Status:      "open",
            Description: "Brute force attack detected",
            Evidence: map[string]interface{}{
                "ip_address": login.IPAddress,
                "username":   login.Username,
                "attempts":   sm.getFailedAttempts(ctx, login.IPAddress),
            },
            CreatedAt: time.Now(),
        }
        
        sm.incidentManager.CreateIncident(ctx, incident)
        sm.alertManager.SendAlert(ctx, incident)
    }
}

func (sm *SecurityMonitor) isBruteForceAttempt(ctx context.Context, login *LoginAttempt) bool {
    key := fmt.Sprintf("failed_logins:%s", login.IPAddress)
    
    // Increment counter
    count, err := sm.redisClient.Incr(ctx, key).Result()
    if err != nil {
        log.Printf("Failed to increment failed login counter: %v", err)
        return false
    }
    
    // Set expiration
    sm.redisClient.Expire(ctx, key, 15*time.Minute)
    
    // Check threshold
    return count > 5
}

func (sm *SecurityMonitor) getFailedAttempts(ctx context.Context, ipAddress string) int {
    key := fmt.Sprintf("failed_logins:%s", ipAddress)
    count, err := sm.redisClient.Get(ctx, key).Int()
    if err != nil {
        return 0
    }
    return count
}

// Threat Detector Implementation
func NewThreatDetector() *ThreatDetector {
    return &ThreatDetector{
        rules: []ThreatRule{},
    }
}

func (td *ThreatDetector) AddRule(rule *ThreatRule) {
    td.mutex.Lock()
    defer td.mutex.Unlock()
    
    td.rules = append(td.rules, *rule)
}

func (td *ThreatDetector) DetectThreats(ctx context.Context, req *SecurityRequest) []*Threat {
    td.mutex.RLock()
    defer td.mutex.RUnlock()
    
    var threats []*Threat
    
    for _, rule := range td.rules {
        if !rule.Enabled {
            continue
        }
        
        if td.ruleMatches(rule, req) {
            threat := &Threat{
                ID:       generateThreatID(),
                RuleID:   rule.ID,
                Severity: rule.Severity,
                Pattern:  rule.Pattern,
                Evidence: map[string]interface{}{
                    "request_id": req.ID,
                    "ip_address": req.IPAddress,
                    "user_agent": req.UserAgent,
                    "url":        req.URL,
                },
                DetectedAt: time.Now(),
            }
            
            threats = append(threats, threat)
        }
    }
    
    return threats
}

func (td *ThreatDetector) ruleMatches(rule ThreatRule, req *SecurityRequest) bool {
    // Check pattern matching
    if rule.Pattern != "" {
        if !strings.Contains(req.URL, rule.Pattern) {
            return false
        }
    }
    
    // Check conditions
    for key, value := range rule.Conditions {
        switch key {
        case "ip_whitelist":
            if td.isIPWhitelisted(req.IPAddress, value) {
                return false
            }
        case "user_agent_blacklist":
            if td.isUserAgentBlacklisted(req.UserAgent, value) {
                return true
            }
        case "rate_limit":
            if td.isRateLimited(req.IPAddress, value) {
                return true
            }
        }
    }
    
    return true
}

// Incident Manager Implementation
func NewIncidentManager() *IncidentManager {
    return &IncidentManager{
        incidents: make(map[string]*SecurityIncident),
    }
}

func (im *IncidentManager) CreateIncident(ctx context.Context, threat *Threat) *SecurityIncident {
    im.mutex.Lock()
    defer im.mutex.Unlock()
    
    incident := &SecurityIncident{
        ID:          generateIncidentID(),
        Type:        "security_threat",
        Severity:    threat.Severity,
        Status:      "open",
        Description: fmt.Sprintf("Security threat detected: %s", threat.Pattern),
        Evidence:    threat.Evidence,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }
    
    im.incidents[incident.ID] = incident
    return incident
}

func (im *IncidentManager) UpdateIncident(ctx context.Context, incidentID string, updates map[string]interface{}) error {
    im.mutex.Lock()
    defer im.mutex.Unlock()
    
    incident, exists := im.incidents[incidentID]
    if !exists {
        return fmt.Errorf("incident not found")
    }
    
    // Update fields
    for key, value := range updates {
        switch key {
        case "status":
            incident.Status = value.(string)
        case "assigned_to":
            incident.AssignedTo = value.(string)
        case "description":
            incident.Description = value.(string)
        }
    }
    
    incident.UpdatedAt = time.Now()
    return nil
}

func (im *IncidentManager) GetIncident(ctx context.Context, incidentID string) (*SecurityIncident, error) {
    im.mutex.RLock()
    defer im.mutex.RUnlock()
    
    incident, exists := im.incidents[incidentID]
    if !exists {
        return nil, fmt.Errorf("incident not found")
    }
    
    return incident, nil
}

func (im *IncidentManager) ListIncidents(ctx context.Context, status string) []*SecurityIncident {
    im.mutex.RLock()
    defer im.mutex.RUnlock()
    
    var incidents []*SecurityIncident
    
    for _, incident := range im.incidents {
        if status == "" || incident.Status == status {
            incidents = append(incidents, incident)
        }
    }
    
    return incidents
}

// Security Metrics Implementation
func NewSecurityMetrics() *SecurityMetrics {
    return &SecurityMetrics{
        failedLogins: promauto.NewCounter(prometheus.CounterOpts{
            Name: "security_failed_logins_total",
            Help: "Total number of failed login attempts",
        }),
        blockedRequests: promauto.NewCounter(prometheus.CounterOpts{
            Name: "security_blocked_requests_total",
            Help: "Total number of blocked requests",
        }),
        securityAlerts: promauto.NewCounter(prometheus.CounterOpts{
            Name: "security_alerts_total",
            Help: "Total number of security alerts",
        }),
        activeThreats: promauto.NewGauge(prometheus.GaugeOpts{
            Name: "security_active_threats",
            Help: "Number of active security threats",
        }),
        responseTime: promauto.NewHistogram(prometheus.HistogramOpts{
            Name: "security_response_time_seconds",
            Help: "Security monitoring response time",
        }),
    }
}
```

## 🎯 Best Practices

### Security Principles
1. **Defense in Depth**: Implement multiple layers of security
2. **Least Privilege**: Grant minimum necessary permissions
3. **Zero Trust**: Never trust, always verify
4. **Continuous Monitoring**: Monitor all security events
5. **Incident Response**: Have a clear incident response plan

### Implementation Guidelines
1. **Authentication**: Use strong authentication methods
2. **Authorization**: Implement proper access controls
3. **Encryption**: Encrypt data at rest and in transit
4. **Monitoring**: Set up comprehensive security monitoring
5. **Testing**: Regular security testing and audits

### Compliance and Standards
1. **OWASP**: Follow OWASP security guidelines
2. **NIST**: Implement NIST cybersecurity framework
3. **ISO 27001**: Follow ISO 27001 standards
4. **SOC 2**: Implement SOC 2 controls
5. **GDPR**: Ensure GDPR compliance

---

**Last Updated**: December 2024  
**Category**: Advanced Security Comprehensive  
**Complexity**: Expert Level
