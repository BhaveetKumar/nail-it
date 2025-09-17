# Advanced Security Patterns

Advanced security patterns and practices for backend systems.

## 🎯 Security Architecture

### Zero Trust Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Identity      │    │   Device        │    │   Network       │
│   Verification  │────│   Trust         │────│   Segmentation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Data          │
                       │   Protection    │
                       └─────────────────┘
```

### Security Layers
- **Identity & Access Management (IAM)**
- **Device Trust & Compliance**
- **Network Segmentation**
- **Data Protection & Encryption**
- **Application Security**
- **Monitoring & Incident Response**

## 🔐 Authentication & Authorization

### Multi-Factor Authentication
```go
type MFAProvider interface {
    GenerateSecret(userID string) (string, error)
    GenerateQRCode(secret, userID string) (string, error)
    ValidateToken(secret, token string) (bool, error)
    GenerateBackupCodes(userID string) ([]string, error)
}

type TOTPProvider struct {
    issuer string
    period int
}

func (t *TOTPProvider) GenerateSecret(userID string) (string, error) {
    key, err := totp.Generate(totp.GenerateOpts{
        Issuer:      t.issuer,
        AccountName: userID,
        Period:      uint(t.period),
    })
    if err != nil {
        return "", err
    }
    
    return key.Secret(), nil
}

func (t *TOTPProvider) ValidateToken(secret, token string) (bool, error) {
    return totp.Validate(token, secret), nil
}

type MFAService struct {
    providers map[string]MFAProvider
    repository UserRepository
}

func (m *MFAService) EnableMFA(userID string, providerType string) (*MFAConfig, error) {
    provider, exists := m.providers[providerType]
    if !exists {
        return nil, errors.New("provider not found")
    }
    
    secret, err := provider.GenerateSecret(userID)
    if err != nil {
        return nil, err
    }
    
    qrCode, err := provider.GenerateQRCode(secret, userID)
    if err != nil {
        return nil, err
    }
    
    backupCodes, err := provider.GenerateBackupCodes(userID)
    if err != nil {
        return nil, err
    }
    
    config := &MFAConfig{
        UserID:      userID,
        Provider:    providerType,
        Secret:      secret,
        QRCode:      qrCode,
        BackupCodes: backupCodes,
        Enabled:     false,
    }
    
    if err := m.repository.SaveMFAConfig(config); err != nil {
        return nil, err
    }
    
    return config, nil
}
```

### OAuth 2.0 & JWT Implementation
```go
type OAuth2Service struct {
    clientStore    ClientStore
    tokenStore     TokenStore
    userStore      UserStore
    cryptoService  CryptoService
}

type Client struct {
    ID           string
    Secret       string
    RedirectURIs []string
    Scopes       []string
    GrantTypes   []string
}

type Token struct {
    AccessToken  string
    RefreshToken string
    TokenType    string
    ExpiresIn    int
    Scope        string
}

func (o *OAuth2Service) AuthorizeCode(clientID, redirectURI, scope string) (string, error) {
    // Validate client
    client, err := o.clientStore.GetByID(clientID)
    if err != nil {
        return "", err
    }
    
    // Validate redirect URI
    if !o.validateRedirectURI(client, redirectURI) {
        return "", errors.New("invalid redirect URI")
    }
    
    // Generate authorization code
    code := o.generateAuthorizationCode(clientID, redirectURI, scope)
    
    // Store code
    if err := o.tokenStore.StoreAuthorizationCode(code, clientID, redirectURI, scope); err != nil {
        return "", err
    }
    
    return code, nil
}

func (o *OAuth2Service) ExchangeToken(code, clientID, clientSecret, redirectURI string) (*Token, error) {
    // Validate client credentials
    client, err := o.clientStore.GetByID(clientID)
    if err != nil {
        return nil, err
    }
    
    if client.Secret != clientSecret {
        return nil, errors.New("invalid client secret")
    }
    
    // Validate authorization code
    authCode, err := o.tokenStore.GetAuthorizationCode(code)
    if err != nil {
        return nil, err
    }
    
    if authCode.ClientID != clientID || authCode.RedirectURI != redirectURI {
        return nil, errors.New("invalid authorization code")
    }
    
    // Generate tokens
    accessToken := o.generateAccessToken(authCode.UserID, authCode.Scope)
    refreshToken := o.generateRefreshToken(authCode.UserID)
    
    token := &Token{
        AccessToken:  accessToken,
        RefreshToken: refreshToken,
        TokenType:    "Bearer",
        ExpiresIn:    3600,
        Scope:        authCode.Scope,
    }
    
    // Store tokens
    if err := o.tokenStore.StoreTokens(token, authCode.UserID); err != nil {
        return nil, err
    }
    
    return token, nil
}
```

### Role-Based Access Control (RBAC)
```go
type RBACService struct {
    roleStore      RoleStore
    permissionStore PermissionStore
    userStore      UserStore
}

type Role struct {
    ID          string
    Name        string
    Permissions []string
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

type Permission struct {
    ID          string
    Resource    string
    Action      string
    Description string
}

type UserRole struct {
    UserID string
    RoleID string
    AssignedAt time.Time
}

func (r *RBACService) CheckPermission(userID, resource, action string) (bool, error) {
    // Get user roles
    roles, err := r.roleStore.GetUserRoles(userID)
    if err != nil {
        return false, err
    }
    
    // Check each role for permission
    for _, role := range roles {
        if r.hasPermission(role, resource, action) {
            return true, nil
        }
    }
    
    return false, nil
}

func (r *RBACService) hasPermission(role Role, resource, action string) bool {
    for _, permission := range role.Permissions {
        if permission == fmt.Sprintf("%s:%s", resource, action) {
            return true
        }
    }
    return false
}

func (r *RBACService) AssignRole(userID, roleID string) error {
    // Validate role exists
    role, err := r.roleStore.GetByID(roleID)
    if err != nil {
        return err
    }
    
    // Check if user already has role
    hasRole, err := r.roleStore.UserHasRole(userID, roleID)
    if err != nil {
        return err
    }
    
    if hasRole {
        return errors.New("user already has this role")
    }
    
    // Assign role
    userRole := &UserRole{
        UserID:     userID,
        RoleID:     roleID,
        AssignedAt: time.Now(),
    }
    
    return r.roleStore.AssignRole(userRole)
}
```

## 🔒 Data Encryption

### Encryption Service
```go
type EncryptionService struct {
    aesKey    []byte
    rsaKey    *rsa.PrivateKey
    keyStore  KeyStore
}

type EncryptedData struct {
    Data      string `json:"data"`
    KeyID     string `json:"key_id"`
    Algorithm string `json:"algorithm"`
    IV        string `json:"iv"`
}

func (e *EncryptionService) Encrypt(data []byte) (*EncryptedData, error) {
    // Generate random IV
    iv := make([]byte, 12)
    if _, err := rand.Read(iv); err != nil {
        return nil, err
    }
    
    // Encrypt data
    cipher, err := aes.NewCipher(e.aesKey)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(cipher)
    if err != nil {
        return nil, err
    }
    
    encrypted := gcm.Seal(nil, iv, data, nil)
    
    return &EncryptedData{
        Data:      base64.StdEncoding.EncodeToString(encrypted),
        KeyID:     "default",
        Algorithm: "AES-256-GCM",
        IV:        base64.StdEncoding.EncodeToString(iv),
    }, nil
}

func (e *EncryptionService) Decrypt(encryptedData *EncryptedData) ([]byte, error) {
    // Decode IV
    iv, err := base64.StdEncoding.DecodeString(encryptedData.IV)
    if err != nil {
        return nil, err
    }
    
    // Decode encrypted data
    encrypted, err := base64.StdEncoding.DecodeString(encryptedData.Data)
    if err != nil {
        return nil, err
    }
    
    // Decrypt data
    cipher, err := aes.NewCipher(e.aesKey)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(cipher)
    if err != nil {
        return nil, err
    }
    
    decrypted, err := gcm.Open(nil, iv, encrypted, nil)
    if err != nil {
        return nil, err
    }
    
    return decrypted, nil
}
```

### Key Management
```go
type KeyManagementService struct {
    keyStore    KeyStore
    cryptoService CryptoService
    rotationPolicy RotationPolicy
}

type EncryptionKey struct {
    ID        string
    Algorithm string
    Key       []byte
    CreatedAt time.Time
    ExpiresAt time.Time
    Status    string
}

type RotationPolicy struct {
    RotationInterval time.Duration
    MaxKeyAge        time.Duration
    BackupKeys       int
}

func (kms *KeyManagementService) GenerateKey(algorithm string) (*EncryptionKey, error) {
    var key []byte
    var err error
    
    switch algorithm {
    case "AES-256":
        key = make([]byte, 32)
        _, err = rand.Read(key)
    case "AES-128":
        key = make([]byte, 16)
        _, err = rand.Read(key)
    default:
        return nil, errors.New("unsupported algorithm")
    }
    
    if err != nil {
        return nil, err
    }
    
    encryptionKey := &EncryptionKey{
        ID:        generateKeyID(),
        Algorithm: algorithm,
        Key:       key,
        CreatedAt: time.Now(),
        ExpiresAt: time.Now().Add(kms.rotationPolicy.MaxKeyAge),
        Status:    "active",
    }
    
    // Store key
    if err := kms.keyStore.Store(encryptionKey); err != nil {
        return nil, err
    }
    
    return encryptionKey, nil
}

func (kms *KeyManagementService) RotateKeys() error {
    // Get keys that need rotation
    keys, err := kms.keyStore.GetKeysForRotation()
    if err != nil {
        return err
    }
    
    for _, key := range keys {
        // Generate new key
        newKey, err := kms.GenerateKey(key.Algorithm)
        if err != nil {
            return err
        }
        
        // Update key status
        key.Status = "rotated"
        key.ExpiresAt = time.Now()
        kms.keyStore.Update(key)
        
        // Migrate data to new key
        if err := kms.migrateData(key, newKey); err != nil {
            return err
        }
    }
    
    return nil
}
```

## 🛡️ Security Monitoring

### Security Event Monitoring
```go
type SecurityMonitor struct {
    eventStore    EventStore
    alertService  AlertService
    rules         []SecurityRule
    metrics       MetricsService
}

type SecurityEvent struct {
    ID          string
    Type        string
    Severity    string
    UserID      string
    IPAddress   string
    UserAgent   string
    Timestamp   time.Time
    Details     map[string]interface{}
}

type SecurityRule struct {
    ID          string
    Name        string
    Condition   func(SecurityEvent) bool
    Action      SecurityAction
    Severity    string
}

type SecurityAction string

const (
    ActionBlock    SecurityAction = "block"
    ActionAlert    SecurityAction = "alert"
    ActionLog      SecurityAction = "log"
    ActionMFA      SecurityAction = "mfa"
)

func (sm *SecurityMonitor) ProcessEvent(event SecurityEvent) error {
    // Store event
    if err := sm.eventStore.Store(event); err != nil {
        return err
    }
    
    // Check rules
    for _, rule := range sm.rules {
        if rule.Condition(event) {
            if err := sm.executeAction(rule.Action, event); err != nil {
                return err
            }
        }
    }
    
    // Update metrics
    sm.metrics.IncrementCounter("security_events_total", map[string]string{
        "type":     event.Type,
        "severity": event.Severity,
    })
    
    return nil
}

func (sm *SecurityMonitor) executeAction(action SecurityAction, event SecurityEvent) error {
    switch action {
    case ActionBlock:
        return sm.blockUser(event.UserID)
    case ActionAlert:
        return sm.alertService.SendAlert("Security Event", event)
    case ActionLog:
        return sm.logEvent(event)
    case ActionMFA:
        return sm.requireMFA(event.UserID)
    default:
        return nil
    }
}
```

### Intrusion Detection
```go
type IntrusionDetection struct {
    monitor      SecurityMonitor
    mlModel      MLModel
    threshold    float64
    timeWindow   time.Duration
}

func (id *IntrusionDetection) DetectIntrusion(events []SecurityEvent) (bool, float64, error) {
    // Extract features from events
    features := id.extractFeatures(events)
    
    // Use ML model to detect anomalies
    score, err := id.mlModel.Predict(features)
    if err != nil {
        return false, 0, err
    }
    
    // Check if score exceeds threshold
    isIntrusion := score > id.threshold
    
    if isIntrusion {
        // Create security event
        event := SecurityEvent{
            ID:        generateEventID(),
            Type:      "intrusion_detected",
            Severity:  "high",
            Timestamp: time.Now(),
            Details: map[string]interface{}{
                "score":      score,
                "threshold":  id.threshold,
                "event_count": len(events),
            },
        }
        
        // Process event
        if err := id.monitor.ProcessEvent(event); err != nil {
            return false, 0, err
        }
    }
    
    return isIntrusion, score, nil
}

func (id *IntrusionDetection) extractFeatures(events []SecurityEvent) map[string]interface{} {
    features := make(map[string]interface{})
    
    // Count events by type
    eventCounts := make(map[string]int)
    for _, event := range events {
        eventCounts[event.Type]++
    }
    
    features["event_counts"] = eventCounts
    features["total_events"] = len(events)
    features["unique_ips"] = id.getUniqueIPs(events)
    features["time_span"] = id.getTimeSpan(events)
    
    return features
}
```

## 🔍 Vulnerability Management

### Vulnerability Scanner
```go
type VulnerabilityScanner struct {
    scanners    []Scanner
    repository  VulnerabilityRepository
    notifier    NotificationService
}

type Scanner interface {
    Scan(target string) ([]Vulnerability, error)
    GetName() string
}

type Vulnerability struct {
    ID          string
    Title       string
    Description string
    Severity    string
    CVE         string
    CVSS        float64
    Target      string
    DetectedAt  time.Time
    Status      string
}

type OWASPScanner struct {
    rules []OWASPRule
}

func (o *OWASPScanner) Scan(target string) ([]Vulnerability, error) {
    var vulnerabilities []Vulnerability
    
    for _, rule := range o.rules {
        if rule.Matches(target) {
            vuln := Vulnerability{
                ID:          generateVulnID(),
                Title:       rule.Title,
                Description: rule.Description,
                Severity:    rule.Severity,
                CVE:         rule.CVE,
                CVSS:        rule.CVSS,
                Target:      target,
                DetectedAt:  time.Now(),
                Status:      "open",
            }
            vulnerabilities = append(vulnerabilities, vuln)
        }
    }
    
    return vulnerabilities, nil
}

func (vs *VulnerabilityScanner) ScanTarget(target string) error {
    var allVulns []Vulnerability
    
    // Run all scanners
    for _, scanner := range vs.scanners {
        vulns, err := scanner.Scan(target)
        if err != nil {
            return err
        }
        allVulns = append(allVulns, vulns...)
    }
    
    // Store vulnerabilities
    for _, vuln := range allVulns {
        if err := vs.repository.Store(vuln); err != nil {
            return err
        }
    }
    
    // Send notifications for high severity vulnerabilities
    for _, vuln := range allVulns {
        if vuln.Severity == "critical" || vuln.Severity == "high" {
            vs.notifier.SendVulnerabilityAlert(vuln)
        }
    }
    
    return nil
}
```

## 🎯 Best Practices

### Security Design
1. **Defense in Depth**: Multiple security layers
2. **Least Privilege**: Minimal required permissions
3. **Zero Trust**: Verify everything
4. **Encryption**: Encrypt data at rest and in transit
5. **Monitoring**: Continuous security monitoring

### Implementation
1. **Secure Coding**: Follow secure coding practices
2. **Input Validation**: Validate all inputs
3. **Output Encoding**: Encode outputs properly
4. **Error Handling**: Don't expose sensitive information
5. **Logging**: Comprehensive security logging

### Compliance
1. **PCI DSS**: Payment card industry compliance
2. **GDPR**: Data protection compliance
3. **SOX**: Financial reporting compliance
4. **HIPAA**: Healthcare data compliance
5. **ISO 27001**: Information security management

---

**Last Updated**: December 2024  
**Category**: Advanced Security Patterns  
**Complexity**: Senior Level
