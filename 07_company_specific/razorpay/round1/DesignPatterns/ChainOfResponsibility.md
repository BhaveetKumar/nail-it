# Chain of Responsibility Pattern

## Pattern Name & Intent

**Chain of Responsibility** is a behavioral design pattern that lets you pass requests along a chain of handlers. Each handler decides either to process the request or to pass it to the next handler in the chain.

**Key Intent:**

- Decouple senders and receivers of requests
- Allow multiple objects to handle a request without specifying the receiver explicitly
- Chain handlers dynamically at runtime
- Process requests through a series of processing steps
- Enable flexible request processing pipelines
- Avoid coupling request sender to specific handlers

## When to Use

**Use Chain of Responsibility when:**

1. **Multiple Handlers**: More than one object can handle a request
2. **Dynamic Handling**: The set of handlers changes at runtime
3. **Sequential Processing**: Requests need to be processed through multiple steps
4. **Unknown Handlers**: You don't know which handler will process the request
5. **Middleware Pattern**: Implementing middleware or interceptor chains
6. **Validation Pipelines**: Multiple validation rules need to be applied
7. **Processing Pipelines**: Data needs to flow through multiple processors

**Don't use when:**

- Only one handler exists for each request type
- The chain is static and simple
- Performance is critical (chain traversal adds overhead)
- You need guaranteed handling of all requests
- The order of handlers is not important

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Pipeline

```go
// Payment processing request
type PaymentRequest struct {
    ID            string
    Amount        decimal.Decimal
    Currency      string
    PaymentMethod string
    CustomerID    string
    MerchantID    string
    Metadata      map[string]interface{}

    // Processing state
    ValidationResults map[string]bool
    RiskScore        float64
    ApprovalStatus   string
    ProcessingErrors []error
}

// Handler interface for payment processing
type PaymentHandler interface {
    SetNext(handler PaymentHandler) PaymentHandler
    Handle(ctx context.Context, request *PaymentRequest) error
}

// Base handler with chain management
type BasePaymentHandler struct {
    next PaymentHandler
}

func (b *BasePaymentHandler) SetNext(handler PaymentHandler) PaymentHandler {
    b.next = handler
    return handler
}

func (b *BasePaymentHandler) HandleNext(ctx context.Context, request *PaymentRequest) error {
    if b.next != nil {
        return b.next.Handle(ctx, request)
    }
    return nil
}

// Validation handler
type PaymentValidationHandler struct {
    BasePaymentHandler
    validator PaymentValidator
    logger    *zap.Logger
}

func NewPaymentValidationHandler(validator PaymentValidator, logger *zap.Logger) *PaymentValidationHandler {
    return &PaymentValidationHandler{
        validator: validator,
        logger:    logger,
    }
}

func (p *PaymentValidationHandler) Handle(ctx context.Context, request *PaymentRequest) error {
    p.logger.Info("Validating payment", zap.String("payment_id", request.ID))

    // Initialize validation results
    if request.ValidationResults == nil {
        request.ValidationResults = make(map[string]bool)
    }

    // Amount validation
    if err := p.validator.ValidateAmount(request.Amount); err != nil {
        request.ValidationResults["amount"] = false
        request.ProcessingErrors = append(request.ProcessingErrors, err)
        return fmt.Errorf("amount validation failed: %w", err)
    }
    request.ValidationResults["amount"] = true

    // Currency validation
    if err := p.validator.ValidateCurrency(request.Currency); err != nil {
        request.ValidationResults["currency"] = false
        request.ProcessingErrors = append(request.ProcessingErrors, err)
        return fmt.Errorf("currency validation failed: %w", err)
    }
    request.ValidationResults["currency"] = true

    // Payment method validation
    if err := p.validator.ValidatePaymentMethod(request.PaymentMethod); err != nil {
        request.ValidationResults["payment_method"] = false
        request.ProcessingErrors = append(request.ProcessingErrors, err)
        return fmt.Errorf("payment method validation failed: %w", err)
    }
    request.ValidationResults["payment_method"] = true

    p.logger.Info("Payment validation completed", zap.String("payment_id", request.ID))

    // Continue to next handler
    return p.HandleNext(ctx, request)
}

// Fraud detection handler
type FraudDetectionHandler struct {
    BasePaymentHandler
    fraudEngine FraudDetectionEngine
    logger      *zap.Logger
}

func NewFraudDetectionHandler(fraudEngine FraudDetectionEngine, logger *zap.Logger) *FraudDetectionHandler {
    return &FraudDetectionHandler{
        fraudEngine: fraudEngine,
        logger:      logger,
    }
}

func (f *FraudDetectionHandler) Handle(ctx context.Context, request *PaymentRequest) error {
    f.logger.Info("Performing fraud detection", zap.String("payment_id", request.ID))

    // Calculate risk score
    riskScore, err := f.fraudEngine.CalculateRiskScore(ctx, &FraudCheckRequest{
        Amount:        request.Amount,
        CustomerID:    request.CustomerID,
        PaymentMethod: request.PaymentMethod,
        Metadata:      request.Metadata,
    })
    if err != nil {
        f.logger.Error("Fraud detection failed", zap.Error(err))
        request.ProcessingErrors = append(request.ProcessingErrors, err)
        // Continue processing even if fraud detection fails
        request.RiskScore = 0.5 // Default medium risk
    } else {
        request.RiskScore = riskScore
    }

    // Check if transaction should be blocked
    if request.RiskScore > 0.8 {
        f.logger.Warn("High risk transaction detected",
            zap.String("payment_id", request.ID),
            zap.Float64("risk_score", request.RiskScore))
        return fmt.Errorf("transaction blocked due to high fraud risk: %.2f", request.RiskScore)
    }

    f.logger.Info("Fraud detection completed",
        zap.String("payment_id", request.ID),
        zap.Float64("risk_score", request.RiskScore))

    // Continue to next handler
    return f.HandleNext(ctx, request)
}

// Approval handler
type ApprovalHandler struct {
    BasePaymentHandler
    approvalService ApprovalService
    logger          *zap.Logger
}

func NewApprovalHandler(approvalService ApprovalService, logger *zap.Logger) *ApprovalHandler {
    return &ApprovalHandler{
        approvalService: approvalService,
        logger:          logger,
    }
}

func (a *ApprovalHandler) Handle(ctx context.Context, request *PaymentRequest) error {
    a.logger.Info("Processing approval", zap.String("payment_id", request.ID))

    // Determine if approval is needed
    needsApproval := a.needsApproval(request)

    if needsApproval {
        a.logger.Info("Payment requires approval", zap.String("payment_id", request.ID))

        approvalResult, err := a.approvalService.RequestApproval(ctx, &ApprovalRequest{
            PaymentID:     request.ID,
            Amount:        request.Amount,
            RiskScore:     request.RiskScore,
            CustomerID:    request.CustomerID,
            MerchantID:    request.MerchantID,
        })
        if err != nil {
            request.ProcessingErrors = append(request.ProcessingErrors, err)
            return fmt.Errorf("approval request failed: %w", err)
        }

        request.ApprovalStatus = approvalResult.Status

        if approvalResult.Status != "APPROVED" {
            return fmt.Errorf("payment not approved: %s", approvalResult.Reason)
        }
    } else {
        request.ApprovalStatus = "AUTO_APPROVED"
    }

    a.logger.Info("Approval completed",
        zap.String("payment_id", request.ID),
        zap.String("status", request.ApprovalStatus))

    // Continue to next handler
    return a.HandleNext(ctx, request)
}

func (a *ApprovalHandler) needsApproval(request *PaymentRequest) bool {
    // High amount transactions need approval
    if request.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        return true
    }

    // High risk transactions need approval
    if request.RiskScore > 0.6 {
        return true
    }

    // Certain payment methods need approval
    riskMethods := []string{"CRYPTO", "WIRE_TRANSFER"}
    for _, method := range riskMethods {
        if request.PaymentMethod == method {
            return true
        }
    }

    return false
}

// Payment execution handler
type PaymentExecutionHandler struct {
    BasePaymentHandler
    gateway PaymentGateway
    logger  *zap.Logger
}

func NewPaymentExecutionHandler(gateway PaymentGateway, logger *zap.Logger) *PaymentExecutionHandler {
    return &PaymentExecutionHandler{
        gateway: gateway,
        logger:  logger,
    }
}

func (p *PaymentExecutionHandler) Handle(ctx context.Context, request *PaymentRequest) error {
    p.logger.Info("Executing payment", zap.String("payment_id", request.ID))

    // Process payment through gateway
    gatewayRequest := &GatewayPaymentRequest{
        PaymentID:     request.ID,
        Amount:        request.Amount,
        Currency:      request.Currency,
        PaymentMethod: request.PaymentMethod,
        CustomerID:    request.CustomerID,
        MerchantID:    request.MerchantID,
    }

    result, err := p.gateway.ProcessPayment(ctx, gatewayRequest)
    if err != nil {
        p.logger.Error("Payment execution failed",
            zap.String("payment_id", request.ID),
            zap.Error(err))
        request.ProcessingErrors = append(request.ProcessingErrors, err)
        return fmt.Errorf("payment execution failed: %w", err)
    }

    // Store result in request metadata
    if request.Metadata == nil {
        request.Metadata = make(map[string]interface{})
    }
    request.Metadata["gateway_transaction_id"] = result.TransactionID
    request.Metadata["gateway_status"] = result.Status

    p.logger.Info("Payment execution completed",
        zap.String("payment_id", request.ID),
        zap.String("transaction_id", result.TransactionID))

    // Continue to next handler (e.g., notification handler)
    return p.HandleNext(ctx, request)
}

// Notification handler
type NotificationHandler struct {
    BasePaymentHandler
    notificationService NotificationService
    logger              *zap.Logger
}

func NewNotificationHandler(notificationService NotificationService, logger *zap.Logger) *NotificationHandler {
    return &NotificationHandler{
        notificationService: notificationService,
        logger:              logger,
    }
}

func (n *NotificationHandler) Handle(ctx context.Context, request *PaymentRequest) error {
    n.logger.Info("Sending notifications", zap.String("payment_id", request.ID))

    // Send notification for successful payment
    notification := &NotificationRequest{
        Type:       "PAYMENT_PROCESSED",
        PaymentID:  request.ID,
        CustomerID: request.CustomerID,
        MerchantID: request.MerchantID,
        Amount:     request.Amount,
        Currency:   request.Currency,
        Metadata:   request.Metadata,
    }

    if err := n.notificationService.SendNotification(ctx, notification); err != nil {
        n.logger.Warn("Failed to send notification", zap.Error(err))
        // Don't fail the payment for notification errors
        request.ProcessingErrors = append(request.ProcessingErrors, err)
    }

    n.logger.Info("Notifications sent", zap.String("payment_id", request.ID))

    // This is typically the last handler, so no next handler
    return n.HandleNext(ctx, request)
}

// Payment processing chain builder
type PaymentProcessingChain struct {
    firstHandler PaymentHandler
    logger       *zap.Logger
}

func NewPaymentProcessingChain(logger *zap.Logger) *PaymentProcessingChain {
    return &PaymentProcessingChain{
        logger: logger,
    }
}

func (p *PaymentProcessingChain) BuildChain(
    validator PaymentValidator,
    fraudEngine FraudDetectionEngine,
    approvalService ApprovalService,
    gateway PaymentGateway,
    notificationService NotificationService,
) *PaymentProcessingChain {

    // Create handlers
    validationHandler := NewPaymentValidationHandler(validator, p.logger)
    fraudHandler := NewFraudDetectionHandler(fraudEngine, p.logger)
    approvalHandler := NewApprovalHandler(approvalService, p.logger)
    executionHandler := NewPaymentExecutionHandler(gateway, p.logger)
    notificationHandler := NewNotificationHandler(notificationService, p.logger)

    // Chain handlers together
    validationHandler.SetNext(fraudHandler).
        SetNext(approvalHandler).
        SetNext(executionHandler).
        SetNext(notificationHandler)

    p.firstHandler = validationHandler
    return p
}

func (p *PaymentProcessingChain) ProcessPayment(ctx context.Context, request *PaymentRequest) error {
    if p.firstHandler == nil {
        return fmt.Errorf("payment processing chain not configured")
    }

    p.logger.Info("Starting payment processing chain", zap.String("payment_id", request.ID))

    start := time.Now()
    err := p.firstHandler.Handle(ctx, request)
    duration := time.Since(start)

    if err != nil {
        p.logger.Error("Payment processing failed",
            zap.String("payment_id", request.ID),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        p.logger.Info("Payment processing completed successfully",
            zap.String("payment_id", request.ID),
            zap.Duration("duration", duration))
    }

    return err
}
```

### 2. Request Authentication & Authorization Chain

```go
// Authentication request
type AuthRequest struct {
    Token      string
    RequestURI string
    Method     string
    ClientIP   string
    UserAgent  string
    Headers    map[string]string

    // Processing state
    User        *User
    Permissions []Permission
    RateLimit   *RateLimitInfo
    AuditTrail  []AuditEvent
}

// Authentication handler interface
type AuthHandler interface {
    SetNext(handler AuthHandler) AuthHandler
    Handle(ctx context.Context, request *AuthRequest) error
}

// Base authentication handler
type BaseAuthHandler struct {
    next AuthHandler
}

func (b *BaseAuthHandler) SetNext(handler AuthHandler) AuthHandler {
    b.next = handler
    return handler
}

func (b *BaseAuthHandler) HandleNext(ctx context.Context, request *AuthRequest) error {
    if b.next != nil {
        return b.next.Handle(ctx, request)
    }
    return nil
}

// Token validation handler
type TokenValidationHandler struct {
    BaseAuthHandler
    tokenService TokenService
    logger       *zap.Logger
}

func NewTokenValidationHandler(tokenService TokenService, logger *zap.Logger) *TokenValidationHandler {
    return &TokenValidationHandler{
        tokenService: tokenService,
        logger:       logger,
    }
}

func (t *TokenValidationHandler) Handle(ctx context.Context, request *AuthRequest) error {
    t.logger.Debug("Validating token", zap.String("token_prefix", request.Token[:10]+"..."))

    if request.Token == "" {
        return fmt.Errorf("missing authentication token")
    }

    // Validate token format
    if !t.tokenService.IsValidFormat(request.Token) {
        return fmt.Errorf("invalid token format")
    }

    // Verify token signature and expiration
    claims, err := t.tokenService.ValidateToken(request.Token)
    if err != nil {
        return fmt.Errorf("token validation failed: %w", err)
    }

    // Extract user information from token
    user, err := t.tokenService.GetUserFromClaims(claims)
    if err != nil {
        return fmt.Errorf("failed to extract user from token: %w", err)
    }

    request.User = user

    t.logger.Debug("Token validation successful", zap.String("user_id", user.ID))

    // Add audit event
    request.AuditTrail = append(request.AuditTrail, AuditEvent{
        Event:     "TOKEN_VALIDATED",
        UserID:    user.ID,
        Timestamp: time.Now(),
        Details:   map[string]interface{}{"token_type": claims.Type},
    })

    return t.HandleNext(ctx, request)
}

// Rate limiting handler
type RateLimitingHandler struct {
    BaseAuthHandler
    rateLimiter RateLimiter
    logger      *zap.Logger
}

func NewRateLimitingHandler(rateLimiter RateLimiter, logger *zap.Logger) *RateLimitingHandler {
    return &RateLimitingHandler{
        rateLimiter: rateLimiter,
        logger:      logger,
    }
}

func (r *RateLimitingHandler) Handle(ctx context.Context, request *AuthRequest) error {
    r.logger.Debug("Checking rate limits", zap.String("client_ip", request.ClientIP))

    // Check rate limit by user
    if request.User != nil {
        userLimit, err := r.rateLimiter.CheckUserLimit(request.User.ID)
        if err != nil {
            return fmt.Errorf("rate limit check failed: %w", err)
        }

        if !userLimit.Allowed {
            r.logger.Warn("User rate limit exceeded",
                zap.String("user_id", request.User.ID),
                zap.Time("retry_after", userLimit.RetryAfter))
            return fmt.Errorf("user rate limit exceeded, retry after %v", userLimit.RetryAfter)
        }

        request.RateLimit = &RateLimitInfo{
            UserLimitRemaining: userLimit.Remaining,
            UserLimitResetTime: userLimit.ResetTime,
        }
    }

    // Check rate limit by IP
    ipLimit, err := r.rateLimiter.CheckIPLimit(request.ClientIP)
    if err != nil {
        return fmt.Errorf("IP rate limit check failed: %w", err)
    }

    if !ipLimit.Allowed {
        r.logger.Warn("IP rate limit exceeded",
            zap.String("client_ip", request.ClientIP),
            zap.Time("retry_after", ipLimit.RetryAfter))
        return fmt.Errorf("IP rate limit exceeded, retry after %v", ipLimit.RetryAfter)
    }

    if request.RateLimit == nil {
        request.RateLimit = &RateLimitInfo{}
    }
    request.RateLimit.IPLimitRemaining = ipLimit.Remaining
    request.RateLimit.IPLimitResetTime = ipLimit.ResetTime

    r.logger.Debug("Rate limit check passed", zap.String("client_ip", request.ClientIP))

    // Add audit event
    request.AuditTrail = append(request.AuditTrail, AuditEvent{
        Event:     "RATE_LIMIT_CHECKED",
        UserID:    getStringOrDefault(request.User, "ID", ""),
        Timestamp: time.Now(),
        Details: map[string]interface{}{
            "user_remaining": request.RateLimit.UserLimitRemaining,
            "ip_remaining":   request.RateLimit.IPLimitRemaining,
        },
    })

    return r.HandleNext(ctx, request)
}

// Permission check handler
type PermissionHandler struct {
    BaseAuthHandler
    permissionService PermissionService
    logger           *zap.Logger
}

func NewPermissionHandler(permissionService PermissionService, logger *zap.Logger) *PermissionHandler {
    return &PermissionHandler{
        permissionService: permissionService,
        logger:           logger,
    }
}

func (p *PermissionHandler) Handle(ctx context.Context, request *AuthRequest) error {
    if request.User == nil {
        return fmt.Errorf("user not authenticated")
    }

    p.logger.Debug("Checking permissions",
        zap.String("user_id", request.User.ID),
        zap.String("method", request.Method),
        zap.String("uri", request.RequestURI))

    // Get user permissions
    permissions, err := p.permissionService.GetUserPermissions(request.User.ID)
    if err != nil {
        return fmt.Errorf("failed to get user permissions: %w", err)
    }

    request.Permissions = permissions

    // Check if user has permission for this endpoint
    resource := p.extractResource(request.RequestURI)
    action := p.mapMethodToAction(request.Method)

    hasPermission := p.hasPermission(permissions, resource, action)
    if !hasPermission {
        p.logger.Warn("Permission denied",
            zap.String("user_id", request.User.ID),
            zap.String("resource", resource),
            zap.String("action", action))
        return fmt.Errorf("insufficient permissions for %s %s", request.Method, request.RequestURI)
    }

    p.logger.Debug("Permission check passed",
        zap.String("user_id", request.User.ID),
        zap.String("resource", resource),
        zap.String("action", action))

    // Add audit event
    request.AuditTrail = append(request.AuditTrail, AuditEvent{
        Event:     "PERMISSION_CHECKED",
        UserID:    request.User.ID,
        Timestamp: time.Now(),
        Details: map[string]interface{}{
            "resource": resource,
            "action":   action,
            "granted":  true,
        },
    })

    return p.HandleNext(ctx, request)
}

func (p *PermissionHandler) extractResource(uri string) string {
    // Extract resource from URI (e.g., /api/v1/payments/123 -> payments)
    parts := strings.Split(strings.Trim(uri, "/"), "/")
    if len(parts) >= 3 {
        return parts[2]
    }
    return "unknown"
}

func (p *PermissionHandler) mapMethodToAction(method string) string {
    actionMap := map[string]string{
        "GET":    "read",
        "POST":   "create",
        "PUT":    "update",
        "PATCH":  "update",
        "DELETE": "delete",
    }

    if action, exists := actionMap[method]; exists {
        return action
    }
    return "unknown"
}

func (p *PermissionHandler) hasPermission(permissions []Permission, resource, action string) bool {
    for _, perm := range permissions {
        if perm.Resource == resource && perm.Action == action {
            return true
        }
        // Check for wildcard permissions
        if perm.Resource == "*" || perm.Action == "*" {
            return true
        }
    }
    return false
}

// Audit logging handler
type AuditLoggingHandler struct {
    BaseAuthHandler
    auditService AuditService
    logger       *zap.Logger
}

func NewAuditLoggingHandler(auditService AuditService, logger *zap.Logger) *AuditLoggingHandler {
    return &AuditLoggingHandler{
        auditService: auditService,
        logger:       logger,
    }
}

func (a *AuditLoggingHandler) Handle(ctx context.Context, request *AuthRequest) error {
    a.logger.Debug("Logging audit trail")

    // Create audit log entry
    auditEntry := &AuditLogEntry{
        UserID:     getStringOrDefault(request.User, "ID", ""),
        Action:     fmt.Sprintf("%s %s", request.Method, request.RequestURI),
        ClientIP:   request.ClientIP,
        UserAgent:  request.UserAgent,
        Timestamp:  time.Now(),
        Events:     request.AuditTrail,
        Success:    true, // Will be updated if next handlers fail
    }

    // Log the audit entry
    if err := a.auditService.LogEntry(auditEntry); err != nil {
        a.logger.Warn("Failed to log audit entry", zap.Error(err))
        // Don't fail the request for audit logging errors
    }

    a.logger.Debug("Audit logging completed")

    return a.HandleNext(ctx, request)
}
```

### 3. Data Validation Chain

```go
// Data validation request
type ValidationRequest struct {
    Data       interface{}
    Schema     string
    Context    map[string]interface{}

    // Processing state
    Errors     []ValidationError
    Warnings   []ValidationWarning
    Sanitized  interface{}
    Metadata   map[string]interface{}
}

// Validation handler interface
type ValidationHandler interface {
    SetNext(handler ValidationHandler) ValidationHandler
    Handle(ctx context.Context, request *ValidationRequest) error
}

// Base validation handler
type BaseValidationHandler struct {
    next ValidationHandler
}

func (b *BaseValidationHandler) SetNext(handler ValidationHandler) ValidationHandler {
    b.next = handler
    return handler
}

func (b *BaseValidationHandler) HandleNext(ctx context.Context, request *ValidationRequest) error {
    if b.next != nil {
        return b.next.Handle(ctx, request)
    }
    return nil
}

// Schema validation handler
type SchemaValidationHandler struct {
    BaseValidationHandler
    schemaRegistry SchemaRegistry
    logger         *zap.Logger
}

func NewSchemaValidationHandler(schemaRegistry SchemaRegistry, logger *zap.Logger) *SchemaValidationHandler {
    return &SchemaValidationHandler{
        schemaRegistry: schemaRegistry,
        logger:         logger,
    }
}

func (s *SchemaValidationHandler) Handle(ctx context.Context, request *ValidationRequest) error {
    s.logger.Debug("Performing schema validation", zap.String("schema", request.Schema))

    // Get schema definition
    schema, err := s.schemaRegistry.GetSchema(request.Schema)
    if err != nil {
        return fmt.Errorf("schema not found: %w", err)
    }

    // Validate data against schema
    validationResult, err := schema.Validate(request.Data)
    if err != nil {
        return fmt.Errorf("schema validation failed: %w", err)
    }

    // Collect validation errors
    for _, error := range validationResult.Errors {
        request.Errors = append(request.Errors, ValidationError{
            Field:   error.Field,
            Message: error.Message,
            Code:    "SCHEMA_VIOLATION",
            Value:   error.Value,
        })
    }

    // Collect warnings
    for _, warning := range validationResult.Warnings {
        request.Warnings = append(request.Warnings, ValidationWarning{
            Field:   warning.Field,
            Message: warning.Message,
            Code:    "SCHEMA_WARNING",
        })
    }

    // Fail if there are critical errors
    if len(request.Errors) > 0 {
        return fmt.Errorf("schema validation failed with %d errors", len(request.Errors))
    }

    s.logger.Debug("Schema validation passed",
        zap.String("schema", request.Schema),
        zap.Int("warnings", len(request.Warnings)))

    return s.HandleNext(ctx, request)
}

// Business rule validation handler
type BusinessRuleValidationHandler struct {
    BaseValidationHandler
    ruleEngine BusinessRuleEngine
    logger     *zap.Logger
}

func NewBusinessRuleValidationHandler(ruleEngine BusinessRuleEngine, logger *zap.Logger) *BusinessRuleValidationHandler {
    return &BusinessRuleValidationHandler{
        ruleEngine: ruleEngine,
        logger:     logger,
    }
}

func (b *BusinessRuleValidationHandler) Handle(ctx context.Context, request *ValidationRequest) error {
    b.logger.Debug("Performing business rule validation")

    // Execute business rules
    ruleResults, err := b.ruleEngine.ExecuteRules(request.Data, request.Context)
    if err != nil {
        return fmt.Errorf("business rule execution failed: %w", err)
    }

    // Process rule results
    for _, result := range ruleResults {
        if result.Severity == "ERROR" {
            request.Errors = append(request.Errors, ValidationError{
                Field:   result.Field,
                Message: result.Message,
                Code:    result.RuleCode,
                Value:   result.Value,
            })
        } else if result.Severity == "WARNING" {
            request.Warnings = append(request.Warnings, ValidationWarning{
                Field:   result.Field,
                Message: result.Message,
                Code:    result.RuleCode,
            })
        }
    }

    // Check if any critical business rules failed
    criticalErrors := 0
    for _, error := range request.Errors {
        if error.Code == "CRITICAL_BUSINESS_RULE" {
            criticalErrors++
        }
    }

    if criticalErrors > 0 {
        return fmt.Errorf("business rule validation failed with %d critical errors", criticalErrors)
    }

    b.logger.Debug("Business rule validation completed",
        zap.Int("errors", len(request.Errors)),
        zap.Int("warnings", len(request.Warnings)))

    return b.HandleNext(ctx, request)
}

// Data sanitization handler
type DataSanitizationHandler struct {
    BaseValidationHandler
    sanitizer DataSanitizer
    logger    *zap.Logger
}

func NewDataSanitizationHandler(sanitizer DataSanitizer, logger *zap.Logger) *DataSanitizationHandler {
    return &DataSanitizationHandler{
        sanitizer: sanitizer,
        logger:    logger,
    }
}

func (d *DataSanitizationHandler) Handle(ctx context.Context, request *ValidationRequest) error {
    d.logger.Debug("Performing data sanitization")

    // Sanitize the data
    sanitized, err := d.sanitizer.Sanitize(request.Data)
    if err != nil {
        return fmt.Errorf("data sanitization failed: %w", err)
    }

    request.Sanitized = sanitized

    // Initialize metadata if needed
    if request.Metadata == nil {
        request.Metadata = make(map[string]interface{})
    }
    request.Metadata["sanitized"] = true
    request.Metadata["sanitized_at"] = time.Now()

    d.logger.Debug("Data sanitization completed")

    return d.HandleNext(ctx, request)
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "strings"
    "time"
    "github.com/shopspring/decimal"
    "go.uber.org/zap"
)

// Example: HTTP Request Processing Chain
// This demonstrates a typical middleware chain for processing HTTP requests

// Request represents an HTTP request
type Request struct {
    ID        string
    Method    string
    URL       string
    Headers   map[string]string
    Body      []byte
    ClientIP  string
    UserAgent string

    // Processing state
    User         *User
    RequestTime  time.Time
    ProcessingLog []LogEntry
    Metadata     map[string]interface{}
}

// Response represents an HTTP response
type Response struct {
    StatusCode int
    Headers    map[string]string
    Body       []byte
    Error      error
}

// Handler interface for processing requests
type Handler interface {
    SetNext(handler Handler) Handler
    Handle(ctx context.Context, request *Request) (*Response, error)
}

// Base handler implementation
type BaseHandler struct {
    next Handler
}

func (b *BaseHandler) SetNext(handler Handler) Handler {
    b.next = handler
    return handler
}

func (b *BaseHandler) HandleNext(ctx context.Context, request *Request) (*Response, error) {
    if b.next != nil {
        return b.next.Handle(ctx, request)
    }

    // Default response if no more handlers
    return &Response{
        StatusCode: 200,
        Headers:    map[string]string{"Content-Type": "application/json"},
        Body:       []byte(`{"message": "Request processed successfully"}`),
    }, nil
}

// Logging handler - logs all requests
type LoggingHandler struct {
    BaseHandler
    logger *zap.Logger
}

func NewLoggingHandler(logger *zap.Logger) *LoggingHandler {
    return &LoggingHandler{
        logger: logger,
    }
}

func (l *LoggingHandler) Handle(ctx context.Context, request *Request) (*Response, error) {
    start := time.Now()

    l.logger.Info("Request received",
        zap.String("request_id", request.ID),
        zap.String("method", request.Method),
        zap.String("url", request.URL),
        zap.String("client_ip", request.ClientIP))

    // Add log entry to request
    request.ProcessingLog = append(request.ProcessingLog, LogEntry{
        Handler:   "LoggingHandler",
        Message:   "Request logged",
        Timestamp: start,
    })

    // Process request through next handlers
    response, err := l.HandleNext(ctx, request)

    duration := time.Since(start)

    if err != nil {
        l.logger.Error("Request failed",
            zap.String("request_id", request.ID),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        l.logger.Info("Request completed",
            zap.String("request_id", request.ID),
            zap.Int("status_code", response.StatusCode),
            zap.Duration("duration", duration))
    }

    return response, err
}

// Authentication handler
type AuthenticationHandler struct {
    BaseHandler
    authenticator Authenticator
    logger        *zap.Logger
}

type Authenticator interface {
    Authenticate(token string) (*User, error)
}

type User struct {
    ID       string
    Username string
    Role     string
    Email    string
}

func NewAuthenticationHandler(authenticator Authenticator, logger *zap.Logger) *AuthenticationHandler {
    return &AuthenticationHandler{
        authenticator: authenticator,
        logger:        logger,
    }
}

func (a *AuthenticationHandler) Handle(ctx context.Context, request *Request) (*Response, error) {
    a.logger.Debug("Processing authentication", zap.String("request_id", request.ID))

    // Check for authorization header
    authHeader, exists := request.Headers["Authorization"]
    if !exists {
        a.logger.Warn("Missing authorization header", zap.String("request_id", request.ID))
        return &Response{
            StatusCode: 401,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       []byte(`{"error": "Missing authorization header"}`),
        }, nil
    }

    // Extract token from header
    token := strings.TrimPrefix(authHeader, "Bearer ")
    if token == authHeader {
        a.logger.Warn("Invalid authorization header format", zap.String("request_id", request.ID))
        return &Response{
            StatusCode: 401,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       []byte(`{"error": "Invalid authorization header format"}`),
        }, nil
    }

    // Authenticate user
    user, err := a.authenticator.Authenticate(token)
    if err != nil {
        a.logger.Warn("Authentication failed",
            zap.String("request_id", request.ID),
            zap.Error(err))
        return &Response{
            StatusCode: 401,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       []byte(`{"error": "Authentication failed"}`),
        }, nil
    }

    // Set user in request
    request.User = user

    // Add log entry
    request.ProcessingLog = append(request.ProcessingLog, LogEntry{
        Handler:   "AuthenticationHandler",
        Message:   fmt.Sprintf("User authenticated: %s", user.Username),
        Timestamp: time.Now(),
    })

    a.logger.Debug("Authentication successful",
        zap.String("request_id", request.ID),
        zap.String("user_id", user.ID))

    return a.HandleNext(ctx, request)
}

// Rate limiting handler
type RateLimitHandler struct {
    BaseHandler
    limiter RateLimiter
    logger  *zap.Logger
}

type RateLimiter interface {
    IsAllowed(clientIP string, userID string) (bool, error)
    GetRemainingRequests(clientIP string, userID string) (int, error)
}

func NewRateLimitHandler(limiter RateLimiter, logger *zap.Logger) *RateLimitHandler {
    return &RateLimitHandler{
        limiter: limiter,
        logger:  logger,
    }
}

func (r *RateLimitHandler) Handle(ctx context.Context, request *Request) (*Response, error) {
    r.logger.Debug("Checking rate limits", zap.String("request_id", request.ID))

    var userID string
    if request.User != nil {
        userID = request.User.ID
    }

    // Check if request is allowed
    allowed, err := r.limiter.IsAllowed(request.ClientIP, userID)
    if err != nil {
        r.logger.Error("Rate limit check failed",
            zap.String("request_id", request.ID),
            zap.Error(err))
        return &Response{
            StatusCode: 500,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       []byte(`{"error": "Rate limit check failed"}`),
        }, nil
    }

    if !allowed {
        r.logger.Warn("Rate limit exceeded",
            zap.String("request_id", request.ID),
            zap.String("client_ip", request.ClientIP),
            zap.String("user_id", userID))

        return &Response{
            StatusCode: 429,
            Headers: map[string]string{
                "Content-Type":   "application/json",
                "Retry-After":    "60",
            },
            Body: []byte(`{"error": "Rate limit exceeded"}`),
        }, nil
    }

    // Get remaining requests for response header
    remaining, err := r.limiter.GetRemainingRequests(request.ClientIP, userID)
    if err != nil {
        r.logger.Warn("Failed to get remaining requests", zap.Error(err))
        remaining = -1
    }

    // Add rate limit info to metadata
    if request.Metadata == nil {
        request.Metadata = make(map[string]interface{})
    }
    request.Metadata["rate_limit_remaining"] = remaining

    // Add log entry
    request.ProcessingLog = append(request.ProcessingLog, LogEntry{
        Handler:   "RateLimitHandler",
        Message:   fmt.Sprintf("Rate limit check passed, remaining: %d", remaining),
        Timestamp: time.Now(),
    })

    r.logger.Debug("Rate limit check passed",
        zap.String("request_id", request.ID),
        zap.Int("remaining", remaining))

    return r.HandleNext(ctx, request)
}

// Request validation handler
type ValidationHandler struct {
    BaseHandler
    validator RequestValidator
    logger    *zap.Logger
}

type RequestValidator interface {
    ValidateRequest(request *Request) []ValidationError
}

type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
    Code    string `json:"code"`
}

func NewValidationHandler(validator RequestValidator, logger *zap.Logger) *ValidationHandler {
    return &ValidationHandler{
        validator: validator,
        logger:    logger,
    }
}

func (v *ValidationHandler) Handle(ctx context.Context, request *Request) (*Response, error) {
    v.logger.Debug("Validating request", zap.String("request_id", request.ID))

    // Validate the request
    errors := v.validator.ValidateRequest(request)

    if len(errors) > 0 {
        v.logger.Warn("Request validation failed",
            zap.String("request_id", request.ID),
            zap.Int("error_count", len(errors)))

        // Return validation errors
        errorResponse := map[string]interface{}{
            "error":   "Validation failed",
            "details": errors,
        }

        responseBody, _ := json.Marshal(errorResponse)

        return &Response{
            StatusCode: 400,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       responseBody,
        }, nil
    }

    // Add log entry
    request.ProcessingLog = append(request.ProcessingLog, LogEntry{
        Handler:   "ValidationHandler",
        Message:   "Request validation passed",
        Timestamp: time.Now(),
    })

    v.logger.Debug("Request validation passed", zap.String("request_id", request.ID))

    return v.HandleNext(ctx, request)
}

// Business logic handler (final handler in chain)
type BusinessLogicHandler struct {
    BaseHandler
    processor BusinessProcessor
    logger    *zap.Logger
}

type BusinessProcessor interface {
    ProcessRequest(ctx context.Context, request *Request) (*Response, error)
}

func NewBusinessLogicHandler(processor BusinessProcessor, logger *zap.Logger) *BusinessLogicHandler {
    return &BusinessLogicHandler{
        processor: processor,
        logger:    logger,
    }
}

func (b *BusinessLogicHandler) Handle(ctx context.Context, request *Request) (*Response, error) {
    b.logger.Debug("Processing business logic", zap.String("request_id", request.ID))

    // Add log entry
    request.ProcessingLog = append(request.ProcessingLog, LogEntry{
        Handler:   "BusinessLogicHandler",
        Message:   "Processing business logic",
        Timestamp: time.Now(),
    })

    // Process the actual business logic
    response, err := b.processor.ProcessRequest(ctx, request)
    if err != nil {
        b.logger.Error("Business logic processing failed",
            zap.String("request_id", request.ID),
            zap.Error(err))

        return &Response{
            StatusCode: 500,
            Headers:    map[string]string{"Content-Type": "application/json"},
            Body:       []byte(`{"error": "Internal server error"}`),
        }, err
    }

    // Add processing log to response headers for debugging
    if response.Headers == nil {
        response.Headers = make(map[string]string)
    }
    response.Headers["X-Processing-Steps"] = fmt.Sprintf("%d", len(request.ProcessingLog))

    b.logger.Debug("Business logic processing completed", zap.String("request_id", request.ID))

    return response, nil
}

// Supporting types
type LogEntry struct {
    Handler   string
    Message   string
    Timestamp time.Time
}

// Mock implementations for demonstration
type MockAuthenticator struct{}

func (m *MockAuthenticator) Authenticate(token string) (*User, error) {
    if token == "valid_token" {
        return &User{
            ID:       "user123",
            Username: "john.doe",
            Role:     "user",
            Email:    "john.doe@example.com",
        }, nil
    }
    return nil, fmt.Errorf("invalid token")
}

type MockRateLimiter struct{}

func (m *MockRateLimiter) IsAllowed(clientIP string, userID string) (bool, error) {
    // Simulate rate limiting logic
    return true, nil // Always allow for demo
}

func (m *MockRateLimiter) GetRemainingRequests(clientIP string, userID string) (int, error) {
    return 95, nil // Mock remaining requests
}

type MockRequestValidator struct{}

func (m *MockRequestValidator) ValidateRequest(request *Request) []ValidationError {
    var errors []ValidationError

    // Example validation: check for required headers
    if _, exists := request.Headers["Content-Type"]; !exists && request.Method == "POST" {
        errors = append(errors, ValidationError{
            Field:   "Content-Type",
            Message: "Content-Type header is required for POST requests",
            Code:    "MISSING_HEADER",
        })
    }

    // Example validation: check URL format
    if !strings.HasPrefix(request.URL, "/api/") {
        errors = append(errors, ValidationError{
            Field:   "URL",
            Message: "URL must start with /api/",
            Code:    "INVALID_URL_FORMAT",
        })
    }

    return errors
}

type MockBusinessProcessor struct{}

func (m *MockBusinessProcessor) ProcessRequest(ctx context.Context, request *Request) (*Response, error) {
    // Simulate business logic processing
    time.Sleep(10 * time.Millisecond)

    responseData := map[string]interface{}{
        "message":    "Request processed successfully",
        "request_id": request.ID,
        "user":       request.User,
        "timestamp":  time.Now(),
    }

    responseBody, _ := json.Marshal(responseData)

    return &Response{
        StatusCode: 200,
        Headers:    map[string]string{"Content-Type": "application/json"},
        Body:       responseBody,
    }, nil
}

// Chain builder
type RequestProcessingChain struct {
    firstHandler Handler
    logger       *zap.Logger
}

func NewRequestProcessingChain(logger *zap.Logger) *RequestProcessingChain {
    return &RequestProcessingChain{
        logger: logger,
    }
}

func (rpc *RequestProcessingChain) BuildChain() *RequestProcessingChain {
    // Create handlers
    loggingHandler := NewLoggingHandler(rpc.logger)
    authHandler := NewAuthenticationHandler(&MockAuthenticator{}, rpc.logger)
    rateLimitHandler := NewRateLimitHandler(&MockRateLimiter{}, rpc.logger)
    validationHandler := NewValidationHandler(&MockRequestValidator{}, rpc.logger)
    businessHandler := NewBusinessLogicHandler(&MockBusinessProcessor{}, rpc.logger)

    // Chain handlers together
    loggingHandler.SetNext(authHandler).
        SetNext(rateLimitHandler).
        SetNext(validationHandler).
        SetNext(businessHandler)

    rpc.firstHandler = loggingHandler
    return rpc
}

func (rpc *RequestProcessingChain) ProcessRequest(ctx context.Context, request *Request) (*Response, error) {
    if rpc.firstHandler == nil {
        return nil, fmt.Errorf("processing chain not configured")
    }

    request.RequestTime = time.Now()
    return rpc.firstHandler.Handle(ctx, request)
}

// Example usage
func main() {
    fmt.Println("=== Chain of Responsibility Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Build processing chain
    chain := NewRequestProcessingChain(logger).BuildChain()

    // Create test requests
    requests := []*Request{
        {
            ID:        "req-001",
            Method:    "GET",
            URL:       "/api/users/123",
            Headers:   map[string]string{"Authorization": "Bearer valid_token"},
            ClientIP:  "192.168.1.100",
            UserAgent: "test-client/1.0",
            Metadata:  make(map[string]interface{}),
        },
        {
            ID:        "req-002",
            Method:    "POST",
            URL:       "/api/users",
            Headers:   map[string]string{"Authorization": "Bearer valid_token", "Content-Type": "application/json"},
            Body:      []byte(`{"name": "John Doe", "email": "john@example.com"}`),
            ClientIP:  "192.168.1.101",
            UserAgent: "test-client/1.0",
            Metadata:  make(map[string]interface{}),
        },
        {
            ID:        "req-003",
            Method:    "GET",
            URL:       "/invalid/path",
            Headers:   map[string]string{"Authorization": "Bearer valid_token"},
            ClientIP:  "192.168.1.102",
            UserAgent: "test-client/1.0",
            Metadata:  make(map[string]interface{}),
        },
        {
            ID:        "req-004",
            Method:    "POST",
            URL:       "/api/orders",
            Headers:   map[string]string{"Authorization": "Bearer invalid_token"},
            ClientIP:  "192.168.1.103",
            UserAgent: "test-client/1.0",
            Metadata:  make(map[string]interface{}),
        },
    }

    // Process each request through the chain
    for i, request := range requests {
        fmt.Printf("=== Processing Request %d ===\n", i+1)
        fmt.Printf("Request: %s %s\n", request.Method, request.URL)

        ctx := context.Background()
        response, err := chain.ProcessRequest(ctx, request)

        if err != nil {
            fmt.Printf("Error: %v\n", err)
        } else {
            fmt.Printf("Response: %d\n", response.StatusCode)
            if len(response.Body) > 0 && len(response.Body) < 200 {
                fmt.Printf("Body: %s\n", string(response.Body))
            }
        }

        // Display processing log
        fmt.Printf("Processing Steps:\n")
        for j, entry := range request.ProcessingLog {
            fmt.Printf("  %d. %s: %s\n", j+1, entry.Handler, entry.Message)
        }

        fmt.Println()
    }

    // Demonstrate different chain configurations
    fmt.Println("=== Different Chain Configuration ===")

    // Create a simpler chain (without authentication)
    simpleChain := NewRequestProcessingChain(logger)

    loggingHandler := NewLoggingHandler(logger)
    validationHandler := NewValidationHandler(&MockRequestValidator{}, logger)
    businessHandler := NewBusinessLogicHandler(&MockBusinessProcessor{}, logger)

    loggingHandler.SetNext(validationHandler).SetNext(businessHandler)
    simpleChain.firstHandler = loggingHandler

    // Test with the simpler chain
    testRequest := &Request{
        ID:        "simple-req-001",
        Method:    "GET",
        URL:       "/api/health",
        Headers:   map[string]string{},
        ClientIP:  "192.168.1.200",
        UserAgent: "health-check/1.0",
        Metadata:  make(map[string]interface{}),
    }

    fmt.Printf("Processing request with simpler chain: %s %s\n", testRequest.Method, testRequest.URL)

    ctx := context.Background()
    response, err := simpleChain.ProcessRequest(ctx, testRequest)

    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Response: %d\n", response.StatusCode)
    }

    fmt.Printf("Processing Steps:\n")
    for j, entry := range testRequest.ProcessingLog {
        fmt.Printf("  %d. %s: %s\n", j+1, entry.Handler, entry.Message)
    }

    fmt.Println("\n=== Chain of Responsibility Pattern Demo Complete ===")
}

// Helper function for JSON marshaling (simplified for demo)
import "encoding/json"

var json = struct {
    Marshal func(v interface{}) ([]byte, error)
}{
    Marshal: func(v interface{}) ([]byte, error) {
        // Simplified JSON marshaling for demo
        return []byte(fmt.Sprintf("%+v", v)), nil
    },
}
```

## Variants & Trade-offs

### Variants

1. **Pure Chain of Responsibility**

```go
// Each handler either handles completely or passes to next
type PureHandler interface {
    Handle(request Request) (bool, error) // bool indicates if handled
}

type ConcreteHandler struct {
    next PureHandler
}

func (c *ConcreteHandler) Handle(request Request) (bool, error) {
    if c.canHandle(request) {
        // Handle the request completely
        return true, c.processRequest(request)
    }

    // Pass to next handler
    if c.next != nil {
        return c.next.Handle(request)
    }

    return false, fmt.Errorf("no handler found for request")
}
```

2. **Pipeline Chain**

```go
// Each handler processes and passes modified request to next
type PipelineHandler interface {
    Process(request Request) (Request, error)
}

type PipelineChain struct {
    handlers []PipelineHandler
}

func (p *PipelineChain) Execute(request Request) (Request, error) {
    current := request

    for _, handler := range p.handlers {
        processed, err := handler.Process(current)
        if err != nil {
            return current, err
        }
        current = processed
    }

    return current, nil
}
```

3. **Conditional Chain**

```go
type ConditionalHandler struct {
    condition func(Request) bool
    handler   Handler
    next      Handler
}

func (c *ConditionalHandler) Handle(request Request) error {
    if c.condition(request) {
        if err := c.handler.Handle(request); err != nil {
            return err
        }
    }

    if c.next != nil {
        return c.next.Handle(request)
    }

    return nil
}
```

### Trade-offs

**Pros:**

- **Flexibility**: Dynamic chain configuration at runtime
- **Decoupling**: Loose coupling between senders and receivers
- **Single Responsibility**: Each handler has one responsibility
- **Open-Closed**: Easy to add new handlers without modifying existing code
- **Reusability**: Handlers can be reused in different chains

**Cons:**

- **Performance**: Chain traversal adds overhead
- **Debugging**: Complex chains can be hard to debug
- **No Guarantee**: No guarantee that request will be handled
- **Runtime Errors**: Broken chains only discovered at runtime
- **Complexity**: Can become complex with many handlers

**When to Choose Chain of Responsibility vs Alternatives:**

| Scenario               | Pattern                 | Reason                      |
| ---------------------- | ----------------------- | --------------------------- |
| Sequential processing  | Chain of Responsibility | Multiple processing steps   |
| Conditional processing | Strategy                | Different algorithms        |
| Event handling         | Observer                | Event notification          |
| Request decoration     | Decorator               | Add behavior to requests    |
| State-based processing | State                   | Behavior changes with state |

## Integration Tips

### 1. Factory Pattern Integration

```go
type HandlerFactory interface {
    CreateHandler(handlerType string, config HandlerConfig) (Handler, error)
}

type HandlerConfig struct {
    Name       string
    Parameters map[string]interface{}
    Next       string
}

type ChainFactory struct {
    handlerFactory HandlerFactory
    configs        []HandlerConfig
}

func (cf *ChainFactory) BuildChain() (Handler, error) {
    handlers := make(map[string]Handler)

    // Create all handlers first
    for _, config := range cf.configs {
        handler, err := cf.handlerFactory.CreateHandler(config.Name, config)
        if err != nil {
            return nil, err
        }
        handlers[config.Name] = handler
    }

    // Link handlers together
    for _, config := range cf.configs {
        if config.Next != "" {
            handlers[config.Name].SetNext(handlers[config.Next])
        }
    }

    // Return first handler
    if len(cf.configs) > 0 {
        return handlers[cf.configs[0].Name], nil
    }

    return nil, fmt.Errorf("no handlers configured")
}
```

### 2. Builder Pattern Integration

```go
type ChainBuilder struct {
    handlers []Handler
    logger   *zap.Logger
}

func NewChainBuilder(logger *zap.Logger) *ChainBuilder {
    return &ChainBuilder{
        handlers: make([]Handler, 0),
        logger:   logger,
    }
}

func (cb *ChainBuilder) AddLogging() *ChainBuilder {
    cb.handlers = append(cb.handlers, NewLoggingHandler(cb.logger))
    return cb
}

func (cb *ChainBuilder) AddAuthentication(auth Authenticator) *ChainBuilder {
    cb.handlers = append(cb.handlers, NewAuthenticationHandler(auth, cb.logger))
    return cb
}

func (cb *ChainBuilder) AddRateLimit(limiter RateLimiter) *ChainBuilder {
    cb.handlers = append(cb.handlers, NewRateLimitHandler(limiter, cb.logger))
    return cb
}

func (cb *ChainBuilder) AddValidation(validator RequestValidator) *ChainBuilder {
    cb.handlers = append(cb.handlers, NewValidationHandler(validator, cb.logger))
    return cb
}

func (cb *ChainBuilder) AddBusinessLogic(processor BusinessProcessor) *ChainBuilder {
    cb.handlers = append(cb.handlers, NewBusinessLogicHandler(processor, cb.logger))
    return cb
}

func (cb *ChainBuilder) Build() Handler {
    if len(cb.handlers) == 0 {
        return nil
    }

    // Link handlers together
    for i := 0; i < len(cb.handlers)-1; i++ {
        cb.handlers[i].SetNext(cb.handlers[i+1])
    }

    return cb.handlers[0]
}
```

### 3. Command Pattern Integration

```go
type ChainCommand struct {
    chain   Handler
    request Request
}

func (cc *ChainCommand) Execute() error {
    return cc.chain.Handle(context.Background(), &cc.request)
}

func (cc *ChainCommand) Undo() error {
    // Implement undo logic if needed
    return fmt.Errorf("undo not supported for chain commands")
}

type CommandChainInvoker struct {
    commands []Command
}

func (cci *CommandChainInvoker) AddCommand(cmd Command) {
    cci.commands = append(cci.commands, cmd)
}

func (cci *CommandChainInvoker) ExecuteAll() []error {
    var errors []error

    for _, cmd := range cci.commands {
        if err := cmd.Execute(); err != nil {
            errors = append(errors, err)
        }
    }

    return errors
}
```

### 4. Observer Pattern Integration

```go
type ChainObserver interface {
    OnHandlerStarted(handler Handler, request Request)
    OnHandlerCompleted(handler Handler, request Request, err error)
    OnChainCompleted(request Request, finalErr error)
}

type ObservableHandler struct {
    BaseHandler
    observers []ChainObserver
    name      string
}

func (oh *ObservableHandler) Handle(request Request) error {
    // Notify observers that handler started
    for _, observer := range oh.observers {
        observer.OnHandlerStarted(oh, request)
    }

    // Process the request
    err := oh.handleRequest(request)

    // Notify observers that handler completed
    for _, observer := range oh.observers {
        observer.OnHandlerCompleted(oh, request, err)
    }

    if err != nil {
        return err
    }

    return oh.HandleNext(request)
}

func (oh *ObservableHandler) AddObserver(observer ChainObserver) {
    oh.observers = append(oh.observers, observer)
}
```

## Common Interview Questions

### 1. **How does Chain of Responsibility differ from Decorator pattern?**

**Answer:**
Both patterns involve chaining objects, but they serve different purposes:

**Chain of Responsibility:**

```go
// Purpose: Find a handler that can process the request
type Handler interface {
    SetNext(Handler) Handler
    Handle(Request) error
}

type ValidationHandler struct {
    next Handler
}

func (v *ValidationHandler) Handle(request Request) error {
    if request.Type == "VALIDATION" {
        // This handler processes validation requests
        return v.validate(request)
    }

    // Pass to next handler if this one can't handle it
    if v.next != nil {
        return v.next.Handle(request)
    }

    return fmt.Errorf("no handler found")
}

// Usage: Request travels until a handler can process it
handler := validationHandler.SetNext(authHandler).SetNext(businessHandler)
err := handler.Handle(request) // Stops at first suitable handler
```

**Decorator:**

```go
// Purpose: Add behavior to an object
type Service interface {
    Process(Request) Response
}

type LoggingDecorator struct {
    wrapped Service
    logger  Logger
}

func (l *LoggingDecorator) Process(request Request) Response {
    l.logger.Log("Processing request")

    // Always calls the wrapped service
    response := l.wrapped.Process(request)

    l.logger.Log("Request processed")
    return response
}

// Usage: All decorators process the request
service := NewLoggingDecorator(
    NewValidationDecorator(
        NewBusinessService()
    )
)
response := service.Process(request) // All layers process
```

**Key Differences:**

| Aspect                   | Chain of Responsibility | Decorator              |
| ------------------------ | ----------------------- | ---------------------- |
| **Purpose**              | Find suitable handler   | Add behavior           |
| **Processing**           | One handler processes   | All decorators process |
| **Control Flow**         | Stops at handler        | Flows through all      |
| **Request Modification** | Usually not modified    | Often modified         |
| **Use Case**             | Alternative handlers    | Layered functionality  |

### 2. **How do you handle errors in a chain of responsibility?**

**Answer:**
Error handling in chains requires careful consideration of whether to stop or continue processing:

**Fail-Fast Strategy:**

```go
type FailFastHandler struct {
    BaseHandler
    processor Processor
}

func (f *FailFastHandler) Handle(request Request) error {
    err := f.processor.Process(request)
    if err != nil {
        // Stop chain execution on first error
        return fmt.Errorf("handler failed: %w", err)
    }

    return f.HandleNext(request)
}
```

**Error Aggregation Strategy:**

```go
type ErrorAggregatingHandler struct {
    BaseHandler
    processor Processor
}

func (e *ErrorAggregatingHandler) Handle(request Request) error {
    err := e.processor.Process(request)
    if err != nil {
        // Collect error but continue processing
        if request.Errors == nil {
            request.Errors = make([]error, 0)
        }
        request.Errors = append(request.Errors, err)
    }

    // Continue to next handler regardless of error
    return e.HandleNext(request)
}
```

**Conditional Error Handling:**

```go
type ConditionalErrorHandler struct {
    BaseHandler
    processor      Processor
    isCriticalError func(error) bool
}

func (c *ConditionalErrorHandler) Handle(request Request) error {
    err := c.processor.Process(request)
    if err != nil {
        if c.isCriticalError(err) {
            // Stop for critical errors
            return fmt.Errorf("critical error: %w", err)
        }

        // Log non-critical errors but continue
        log.Warn("Non-critical error", zap.Error(err))
    }

    return c.HandleNext(request)
}
```

**Error Recovery Strategy:**

```go
type RecoveryHandler struct {
    BaseHandler
    processor       Processor
    fallbackProcessor Processor
}

func (r *RecoveryHandler) Handle(request Request) error {
    err := r.processor.Process(request)
    if err != nil {
        // Try fallback processor
        if r.fallbackProcessor != nil {
            log.Warn("Primary processor failed, trying fallback", zap.Error(err))

            if fallbackErr := r.fallbackProcessor.Process(request); fallbackErr != nil {
                return fmt.Errorf("both primary and fallback failed: primary=%w, fallback=%w", err, fallbackErr)
            }
        } else {
            return err
        }
    }

    return r.HandleNext(request)
}
```

### 3. **How do you test chain of responsibility implementations?**

**Answer:**
Testing chains requires both unit testing of individual handlers and integration testing of the complete chain:

**Unit Testing Individual Handlers:**

```go
func TestValidationHandler(t *testing.T) {
    validator := &MockValidator{}
    handler := NewValidationHandler(validator)

    // Test valid request
    validator.On("Validate", mock.Anything).Return(nil)
    request := &Request{Type: "test"}

    err := handler.Handle(request)
    assert.NoError(t, err)
    validator.AssertExpectations(t)

    // Test invalid request
    validator.On("Validate", mock.Anything).Return(fmt.Errorf("validation failed"))

    err = handler.Handle(request)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "validation failed")
}
```

**Integration Testing Complete Chain:**

```go
func TestPaymentProcessingChain(t *testing.T) {
    // Create real or mock services
    validator := &MockValidator{}
    fraudEngine := &MockFraudEngine{}
    gateway := &MockGateway{}

    // Build the chain
    chain := NewPaymentProcessingChain().
        AddValidation(validator).
        AddFraudDetection(fraudEngine).
        AddGatewayProcessing(gateway).
        Build()

    // Test successful processing
    validator.On("Validate", mock.Anything).Return(nil)
    fraudEngine.On("CheckFraud", mock.Anything).Return(0.1, nil) // Low risk
    gateway.On("Process", mock.Anything).Return(&GatewayResult{Status: "SUCCESS"}, nil)

    request := &PaymentRequest{Amount: decimal.NewFromFloat(100)}
    err := chain.Handle(request)

    assert.NoError(t, err)
    assert.Equal(t, "SUCCESS", request.Status)

    // Verify all services were called
    validator.AssertExpectations(t)
    fraudEngine.AssertExpectations(t)
    gateway.AssertExpectations(t)
}
```

**Testing Chain Order:**

```go
func TestChainOrder(t *testing.T) {
    var executionOrder []string

    handler1 := &OrderTestingHandler{Name: "Handler1", Order: &executionOrder}
    handler2 := &OrderTestingHandler{Name: "Handler2", Order: &executionOrder}
    handler3 := &OrderTestingHandler{Name: "Handler3", Order: &executionOrder}

    handler1.SetNext(handler2).SetNext(handler3)

    request := &Request{}
    handler1.Handle(request)

    expected := []string{"Handler1", "Handler2", "Handler3"}
    assert.Equal(t, expected, executionOrder)
}

type OrderTestingHandler struct {
    BaseHandler
    Name  string
    Order *[]string
}

func (o *OrderTestingHandler) Handle(request Request) error {
    *o.Order = append(*o.Order, o.Name)
    return o.HandleNext(request)
}
```

**Testing Error Scenarios:**

```go
func TestChainErrorHandling(t *testing.T) {
    handler1 := &MockHandler{}
    handler2 := &MockHandler{}
    handler3 := &MockHandler{}

    handler1.SetNext(handler2).SetNext(handler3)

    // Test error in middle of chain
    handler1.On("Process", mock.Anything).Return(nil)
    handler2.On("Process", mock.Anything).Return(fmt.Errorf("handler2 failed"))
    // handler3 should not be called

    request := &Request{}
    err := handler1.Handle(request)

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "handler2 failed")

    handler1.AssertExpectations(t)
    handler2.AssertExpectations(t)
    handler3.AssertNotCalled(t, "Process")
}
```

**Performance Testing:**

```go
func BenchmarkChainProcessing(b *testing.B) {
    chain := buildTestChain()
    request := &Request{Type: "benchmark"}

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        chain.Handle(request)
    }
}

func BenchmarkChainLengths(b *testing.B) {
    lengths := []int{1, 5, 10, 20}

    for _, length := range lengths {
        b.Run(fmt.Sprintf("Chain-%d", length), func(b *testing.B) {
            chain := buildChainWithLength(length)
            request := &Request{}

            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                chain.Handle(request)
            }
        })
    }
}
```

### 4. **How do you implement dynamic chain reconfiguration?**

**Answer:**
Dynamic reconfiguration allows chains to be modified at runtime:

**Dynamic Chain Manager:**

```go
type DynamicChain struct {
    handlers []Handler
    mu       sync.RWMutex
    logger   *zap.Logger
}

func NewDynamicChain(logger *zap.Logger) *DynamicChain {
    return &DynamicChain{
        handlers: make([]Handler, 0),
        logger:   logger,
    }
}

func (dc *DynamicChain) AddHandler(position int, handler Handler) error {
    dc.mu.Lock()
    defer dc.mu.Unlock()

    if position < 0 || position > len(dc.handlers) {
        return fmt.Errorf("invalid position: %d", position)
    }

    // Insert handler at specified position
    dc.handlers = append(dc.handlers[:position], append([]Handler{handler}, dc.handlers[position:]...)...)

    // Rebuild chain links
    dc.rebuildChain()

    dc.logger.Info("Handler added to chain",
        zap.String("handler", fmt.Sprintf("%T", handler)),
        zap.Int("position", position))

    return nil
}

func (dc *DynamicChain) RemoveHandler(position int) error {
    dc.mu.Lock()
    defer dc.mu.Unlock()

    if position < 0 || position >= len(dc.handlers) {
        return fmt.Errorf("invalid position: %d", position)
    }

    removedHandler := dc.handlers[position]
    dc.handlers = append(dc.handlers[:position], dc.handlers[position+1:]...)

    // Rebuild chain links
    dc.rebuildChain()

    dc.logger.Info("Handler removed from chain",
        zap.String("handler", fmt.Sprintf("%T", removedHandler)),
        zap.Int("position", position))

    return nil
}

func (dc *DynamicChain) rebuildChain() {
    for i := 0; i < len(dc.handlers)-1; i++ {
        dc.handlers[i].SetNext(dc.handlers[i+1])
    }

    // Clear next for last handler
    if len(dc.handlers) > 0 {
        dc.handlers[len(dc.handlers)-1].SetNext(nil)
    }
}

func (dc *DynamicChain) Handle(request Request) error {
    dc.mu.RLock()
    defer dc.mu.RUnlock()

    if len(dc.handlers) == 0 {
        return fmt.Errorf("no handlers in chain")
    }

    return dc.handlers[0].Handle(request)
}
```

**Configuration-Driven Chains:**

```go
type ChainConfig struct {
    Handlers []HandlerConfig `yaml:"handlers"`
}

type HandlerConfig struct {
    Type       string                 `yaml:"type"`
    Enabled    bool                   `yaml:"enabled"`
    Parameters map[string]interface{} `yaml:"parameters"`
}

type ConfigurableChain struct {
    config       *ChainConfig
    factory      HandlerFactory
    currentChain Handler
    mu           sync.RWMutex
    logger       *zap.Logger
}

func (cc *ConfigurableChain) LoadConfig(configPath string) error {
    cc.mu.Lock()
    defer cc.mu.Unlock()

    config, err := loadChainConfig(configPath)
    if err != nil {
        return err
    }

    cc.config = config
    return cc.rebuildFromConfig()
}

func (cc *ConfigurableChain) rebuildFromConfig() error {
    var handlers []Handler

    for _, handlerConfig := range cc.config.Handlers {
        if !handlerConfig.Enabled {
            continue
        }

        handler, err := cc.factory.CreateHandler(handlerConfig.Type, handlerConfig.Parameters)
        if err != nil {
            return fmt.Errorf("failed to create handler %s: %w", handlerConfig.Type, err)
        }

        handlers = append(handlers, handler)
    }

    // Link handlers
    for i := 0; i < len(handlers)-1; i++ {
        handlers[i].SetNext(handlers[i+1])
    }

    if len(handlers) > 0 {
        cc.currentChain = handlers[0]
    } else {
        cc.currentChain = nil
    }

    cc.logger.Info("Chain rebuilt from config", zap.Int("handlers", len(handlers)))
    return nil
}

func (cc *ConfigurableChain) Handle(request Request) error {
    cc.mu.RLock()
    defer cc.mu.RUnlock()

    if cc.currentChain == nil {
        return fmt.Errorf("no chain configured")
    }

    return cc.currentChain.Handle(request)
}
```

**Hot-Swappable Handlers:**

```go
type HotSwappableHandler struct {
    current atomic.Value // stores Handler
    logger  *zap.Logger
}

func NewHotSwappableHandler(initial Handler, logger *zap.Logger) *HotSwappableHandler {
    h := &HotSwappableHandler{logger: logger}
    h.current.Store(initial)
    return h
}

func (h *HotSwappableHandler) SwapHandler(newHandler Handler) {
    oldHandler := h.current.Load()
    h.current.Store(newHandler)

    h.logger.Info("Handler swapped",
        zap.String("old", fmt.Sprintf("%T", oldHandler)),
        zap.String("new", fmt.Sprintf("%T", newHandler)))
}

func (h *HotSwappableHandler) Handle(request Request) error {
    handler := h.current.Load().(Handler)
    return handler.Handle(request)
}

func (h *HotSwappableHandler) SetNext(next Handler) Handler {
    // This is more complex as we need to update the next for the current handler
    current := h.current.Load().(Handler)
    return current.SetNext(next)
}
```

### 5. **When should you avoid using Chain of Responsibility?**

**Answer:**
Chain of Responsibility should be avoided in certain scenarios:

**Simple, Static Processing:**

```go
// DON'T use chain for simple, unchanging logic
func ProcessPayment(payment *Payment) error {
    // Simple, direct processing is better
    if err := validatePayment(payment); err != nil {
        return err
    }

    if err := processWithGateway(payment); err != nil {
        return err
    }

    return sendConfirmation(payment)
}

// Instead of unnecessary chain:
// validationHandler.SetNext(gatewayHandler).SetNext(confirmationHandler)
```

**Performance-Critical Paths:**

```go
// DON'T use chain in high-frequency, performance-critical code
func ProcessHighFrequencyTrade(trade *Trade) error {
    // Direct calls are faster than chain traversal
    if !isValidTrade(trade) {
        return errors.New("invalid trade")
    }

    return executeTrade(trade)

    // Avoid: chain.Handle(trade) // Adds unnecessary overhead
}
```

**Known, Fixed Processing Order:**

```go
// DON'T use chain when processing order never changes
type UserRegistration struct {
    Email    string
    Password string
}

func RegisterUser(user *UserRegistration) error {
    // Fixed, known steps - direct calls are clearer
    if err := validateEmail(user.Email); err != nil {
        return err
    }

    if err := hashPassword(&user.Password); err != nil {
        return err
    }

    if err := saveToDatabase(user); err != nil {
        return err
    }

    return sendWelcomeEmail(user.Email)
}
```

**Single Responsibility Objects:**

```go
// DON'T use chain when each object should handle one specific type
type EmailProcessor struct{}

func (e *EmailProcessor) Process(request *EmailRequest) error {
    // This processor only handles email requests
    return e.sendEmail(request)
}

// No need for chain - direct usage is better
processor := &EmailProcessor{}
err := processor.Process(emailRequest)
```

**Better Alternatives:**

| Scenario                     | Alternative           | Reason                        |
| ---------------------------- | --------------------- | ----------------------------- |
| Simple sequential processing | Direct function calls | Less overhead                 |
| Different algorithms         | Strategy Pattern      | Clearer algorithm selection   |
| Event handling               | Observer Pattern      | Better for event notification |
| Conditional behavior         | State Pattern         | Behavior changes with state   |
| Data transformation          | Pipeline Pattern      | Better for data flow          |
| Fixed middleware             | Decorator Pattern     | Simpler for fixed layers      |

**Decision Framework:**

```go
type ChainDecision struct {
    ProcessingSteps      int
    StepVariability      string // "static", "dynamic", "configurable"
    PerformanceRequirements string // "low", "medium", "high"
    ErrorHandlingNeeds   string // "simple", "complex"
    RuntimeConfiguration bool
}

func (cd *ChainDecision) ShouldUseChain() (bool, string) {
    if cd.ProcessingSteps <= 2 && cd.StepVariability == "static" {
        return false, "Too simple for chain pattern"
    }

    if cd.PerformanceRequirements == "high" && cd.ProcessingSteps > 5 {
        return false, "Chain overhead too high for performance requirements"
    }

    if cd.StepVariability == "static" && !cd.RuntimeConfiguration {
        return false, "Static processing doesn't benefit from chain flexibility"
    }

    if cd.StepVariability == "dynamic" || cd.RuntimeConfiguration {
        return true, "Dynamic nature benefits from chain flexibility"
    }

    if cd.ProcessingSteps > 3 && cd.ErrorHandlingNeeds == "complex" {
        return true, "Complex processing benefits from chain organization"
    }

    return false, "No clear benefits identified"
}
```
