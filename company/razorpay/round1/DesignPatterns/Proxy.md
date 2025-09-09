# Proxy Pattern

## Pattern Name & Intent

**Proxy** is a structural design pattern that provides a placeholder or surrogate for another object to control access to it. The proxy acts as an intermediary that can add additional functionality like lazy loading, access control, caching, or logging without changing the original object's interface.

**Key Intent:**
- Control access to another object (the real subject)
- Add additional behavior without modifying the original object
- Implement lazy initialization and loading
- Provide a surrogate or placeholder for expensive objects
- Add cross-cutting concerns like security, caching, and logging
- Enable remote object access or virtual object representation

## When to Use

**Use Proxy when:**

1. **Lazy Initialization**: Delay expensive object creation until needed
2. **Access Control**: Control access to sensitive or restricted objects
3. **Caching**: Cache results of expensive operations
4. **Remote Objects**: Provide local representation of remote objects
5. **Virtual Objects**: Handle large objects that consume significant memory
6. **Smart References**: Add reference counting, locking, or validation
7. **Logging/Monitoring**: Add logging without modifying the original object

**Don't use when:**
- Direct object access is simpler and sufficient
- The proxy adds unnecessary complexity
- Performance overhead is unacceptable
- The interface is unstable and changes frequently

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Gateway Proxy
```go
// Payment gateway interface
type PaymentGateway interface {
    ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error)
    RefundPayment(ctx context.Context, transactionID string, amount decimal.Decimal) (*RefundResponse, error)
    GetTransactionStatus(ctx context.Context, transactionID string) (*TransactionStatus, error)
}

// Real payment gateway implementation
type StripePaymentGateway struct {
    apiKey     string
    httpClient *http.Client
    baseURL    string
}

func NewStripePaymentGateway(apiKey string) *StripePaymentGateway {
    return &StripePaymentGateway{
        apiKey:     apiKey,
        httpClient: &http.Client{Timeout: 30 * time.Second},
        baseURL:    "https://api.stripe.com/v1",
    }
}

func (s *StripePaymentGateway) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    // Actual Stripe API call
    url := s.baseURL + "/charges"
    
    payload := map[string]interface{}{
        "amount":   request.Amount.Mul(decimal.NewFromInt(100)).IntPart(), // Convert to cents
        "currency": request.Currency,
        "source":   request.PaymentMethod,
    }
    
    jsonPayload, _ := json.Marshal(payload)
    
    req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonPayload))
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Authorization", "Bearer "+s.apiKey)
    req.Header.Set("Content-Type", "application/json")
    
    resp, err := s.httpClient.Do(req)
    if err != nil {
        return nil, fmt.Errorf("stripe api call failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("stripe api returned status %d", resp.StatusCode)
    }
    
    var stripeResponse struct {
        ID     string `json:"id"`
        Status string `json:"status"`
    }
    
    if err := json.NewDecoder(resp.Body).Decode(&stripeResponse); err != nil {
        return nil, err
    }
    
    return &PaymentResponse{
        TransactionID: stripeResponse.ID,
        Status:        stripeResponse.Status,
        Amount:        request.Amount,
        Currency:      request.Currency,
        ProcessedAt:   time.Now(),
    }, nil
}

func (s *StripePaymentGateway) RefundPayment(ctx context.Context, transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    // Implement refund logic
    return &RefundResponse{
        RefundID:    "rf_" + transactionID,
        Amount:      amount,
        Status:      "succeeded",
        ProcessedAt: time.Now(),
    }, nil
}

func (s *StripePaymentGateway) GetTransactionStatus(ctx context.Context, transactionID string) (*TransactionStatus, error) {
    // Implement status check logic
    return &TransactionStatus{
        TransactionID: transactionID,
        Status:        "succeeded",
        LastUpdated:   time.Now(),
    }, nil
}

// Caching proxy for payment gateway
type CachingPaymentGatewayProxy struct {
    realGateway PaymentGateway
    cache       Cache
    cacheTTL    time.Duration
    logger      *zap.Logger
}

func NewCachingPaymentGatewayProxy(gateway PaymentGateway, cache Cache, ttl time.Duration, logger *zap.Logger) *CachingPaymentGatewayProxy {
    return &CachingPaymentGatewayProxy{
        realGateway: gateway,
        cache:       cache,
        cacheTTL:    ttl,
        logger:      logger,
    }
}

func (p *CachingPaymentGatewayProxy) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    // Payments should not be cached - always go to real gateway
    p.logger.Info("Processing payment", zap.String("amount", request.Amount.String()))
    return p.realGateway.ProcessPayment(ctx, request)
}

func (p *CachingPaymentGatewayProxy) RefundPayment(ctx context.Context, transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    // Refunds should not be cached - always go to real gateway
    p.logger.Info("Processing refund", zap.String("transaction_id", transactionID))
    
    // Invalidate transaction status cache
    statusCacheKey := fmt.Sprintf("transaction_status:%s", transactionID)
    p.cache.Delete(statusCacheKey)
    
    return p.realGateway.RefundPayment(ctx, transactionID, amount)
}

func (p *CachingPaymentGatewayProxy) GetTransactionStatus(ctx context.Context, transactionID string) (*TransactionStatus, error) {
    cacheKey := fmt.Sprintf("transaction_status:%s", transactionID)
    
    // Try to get from cache first
    if cached, err := p.cache.Get(cacheKey); err == nil {
        p.logger.Debug("Transaction status cache hit", zap.String("transaction_id", transactionID))
        return cached.(*TransactionStatus), nil
    }
    
    // Not in cache, fetch from real gateway
    p.logger.Debug("Transaction status cache miss", zap.String("transaction_id", transactionID))
    status, err := p.realGateway.GetTransactionStatus(ctx, transactionID)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    if err := p.cache.Set(cacheKey, status, p.cacheTTL); err != nil {
        p.logger.Warn("Failed to cache transaction status", zap.Error(err))
    }
    
    return status, nil
}

// Rate limiting proxy
type RateLimitingPaymentGatewayProxy struct {
    realGateway PaymentGateway
    limiter     RateLimiter
    logger      *zap.Logger
}

func NewRateLimitingPaymentGatewayProxy(gateway PaymentGateway, limiter RateLimiter, logger *zap.Logger) *RateLimitingPaymentGatewayProxy {
    return &RateLimitingPaymentGatewayProxy{
        realGateway: gateway,
        limiter:     limiter,
        logger:      logger,
    }
}

func (p *RateLimitingPaymentGatewayProxy) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    if !p.limiter.Allow() {
        p.logger.Warn("Payment request rate limited")
        return nil, fmt.Errorf("rate limit exceeded for payment processing")
    }
    
    return p.realGateway.ProcessPayment(ctx, request)
}

func (p *RateLimitingPaymentGatewayProxy) RefundPayment(ctx context.Context, transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    if !p.limiter.Allow() {
        p.logger.Warn("Refund request rate limited")
        return nil, fmt.Errorf("rate limit exceeded for refund processing")
    }
    
    return p.realGateway.RefundPayment(ctx, transactionID, amount)
}

func (p *RateLimitingPaymentGatewayProxy) GetTransactionStatus(ctx context.Context, transactionID string) (*TransactionStatus, error) {
    if !p.limiter.Allow() {
        p.logger.Warn("Status check request rate limited")
        return nil, fmt.Errorf("rate limit exceeded for status check")
    }
    
    return p.realGateway.GetTransactionStatus(ctx, transactionID)
}

// Security proxy with access control
type SecurityPaymentGatewayProxy struct {
    realGateway PaymentGateway
    authService AuthenticationService
    authzService AuthorizationService
    logger      *zap.Logger
}

func NewSecurityPaymentGatewayProxy(
    gateway PaymentGateway,
    authService AuthenticationService,
    authzService AuthorizationService,
    logger *zap.Logger,
) *SecurityPaymentGatewayProxy {
    return &SecurityPaymentGatewayProxy{
        realGateway:  gateway,
        authService:  authService,
        authzService: authzService,
        logger:       logger,
    }
}

func (p *SecurityPaymentGatewayProxy) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    // Authenticate the request
    user, err := p.authService.AuthenticateFromContext(ctx)
    if err != nil {
        p.logger.Warn("Payment authentication failed", zap.Error(err))
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    // Authorize the payment
    if !p.authzService.CanProcessPayment(user, request) {
        p.logger.Warn("Payment authorization failed", 
            zap.String("user_id", user.ID),
            zap.String("amount", request.Amount.String()))
        return nil, fmt.Errorf("insufficient permissions to process payment")
    }
    
    p.logger.Info("Payment authorized", 
        zap.String("user_id", user.ID),
        zap.String("amount", request.Amount.String()))
    
    return p.realGateway.ProcessPayment(ctx, request)
}

func (p *SecurityPaymentGatewayProxy) RefundPayment(ctx context.Context, transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    user, err := p.authService.AuthenticateFromContext(ctx)
    if err != nil {
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    if !p.authzService.CanProcessRefund(user, transactionID, amount) {
        return nil, fmt.Errorf("insufficient permissions to process refund")
    }
    
    return p.realGateway.RefundPayment(ctx, transactionID, amount)
}

func (p *SecurityPaymentGatewayProxy) GetTransactionStatus(ctx context.Context, transactionID string) (*TransactionStatus, error) {
    user, err := p.authService.AuthenticateFromContext(ctx)
    if err != nil {
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    if !p.authzService.CanViewTransaction(user, transactionID) {
        return nil, fmt.Errorf("insufficient permissions to view transaction")
    }
    
    return p.realGateway.GetTransactionStatus(ctx, transactionID)
}
```

### 2. Database Connection Proxy
```go
// Database interface
type Database interface {
    Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
    Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
    BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error)
    Close() error
}

// Real database implementation
type PostgreSQLDatabase struct {
    db *sql.DB
}

func NewPostgreSQLDatabase(connectionString string) (*PostgreSQLDatabase, error) {
    db, err := sql.Open("postgres", connectionString)
    if err != nil {
        return nil, err
    }
    
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return &PostgreSQLDatabase{db: db}, nil
}

func (p *PostgreSQLDatabase) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    return p.db.QueryContext(ctx, query, args...)
}

func (p *PostgreSQLDatabase) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    return p.db.ExecContext(ctx, query, args...)
}

func (p *PostgreSQLDatabase) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
    return p.db.BeginTx(ctx, opts)
}

func (p *PostgreSQLDatabase) Close() error {
    return p.db.Close()
}

// Lazy loading database proxy
type LazyDatabaseProxy struct {
    connectionString string
    realDB          Database
    initOnce        sync.Once
    initError       error
    logger          *zap.Logger
}

func NewLazyDatabaseProxy(connectionString string, logger *zap.Logger) *LazyDatabaseProxy {
    return &LazyDatabaseProxy{
        connectionString: connectionString,
        logger:          logger,
    }
}

func (l *LazyDatabaseProxy) initialize() {
    l.initOnce.Do(func() {
        l.logger.Info("Initializing database connection")
        db, err := NewPostgreSQLDatabase(l.connectionString)
        if err != nil {
            l.initError = err
            l.logger.Error("Failed to initialize database", zap.Error(err))
            return
        }
        l.realDB = db
        l.logger.Info("Database connection initialized successfully")
    })
}

func (l *LazyDatabaseProxy) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    l.initialize()
    if l.initError != nil {
        return nil, l.initError
    }
    return l.realDB.Query(ctx, query, args...)
}

func (l *LazyDatabaseProxy) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    l.initialize()
    if l.initError != nil {
        return nil, l.initError
    }
    return l.realDB.Exec(ctx, query, args...)
}

func (l *LazyDatabaseProxy) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
    l.initialize()
    if l.initError != nil {
        return nil, l.initError
    }
    return l.realDB.BeginTx(ctx, opts)
}

func (l *LazyDatabaseProxy) Close() error {
    if l.realDB != nil {
        return l.realDB.Close()
    }
    return nil
}

// Monitoring database proxy
type MonitoringDatabaseProxy struct {
    realDB  Database
    metrics DatabaseMetrics
    logger  *zap.Logger
}

type DatabaseMetrics interface {
    RecordQueryDuration(query string, duration time.Duration)
    RecordQueryError(query string, err error)
    IncrementConnectionCount()
    DecrementConnectionCount()
}

func NewMonitoringDatabaseProxy(db Database, metrics DatabaseMetrics, logger *zap.Logger) *MonitoringDatabaseProxy {
    return &MonitoringDatabaseProxy{
        realDB:  db,
        metrics: metrics,
        logger:  logger,
    }
}

func (m *MonitoringDatabaseProxy) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    start := time.Now()
    rows, err := m.realDB.Query(ctx, query, args...)
    duration := time.Since(start)
    
    m.metrics.RecordQueryDuration(query, duration)
    
    if err != nil {
        m.metrics.RecordQueryError(query, err)
        m.logger.Error("Database query failed", 
            zap.String("query", query),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        m.logger.Debug("Database query executed", 
            zap.String("query", query),
            zap.Duration("duration", duration))
    }
    
    return rows, err
}

func (m *MonitoringDatabaseProxy) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    start := time.Now()
    result, err := m.realDB.Exec(ctx, query, args...)
    duration := time.Since(start)
    
    m.metrics.RecordQueryDuration(query, duration)
    
    if err != nil {
        m.metrics.RecordQueryError(query, err)
        m.logger.Error("Database exec failed", 
            zap.String("query", query),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        m.logger.Debug("Database exec completed", 
            zap.String("query", query),
            zap.Duration("duration", duration))
    }
    
    return result, err
}

func (m *MonitoringDatabaseProxy) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
    m.metrics.IncrementConnectionCount()
    tx, err := m.realDB.BeginTx(ctx, opts)
    if err != nil {
        m.metrics.DecrementConnectionCount()
    }
    return tx, err
}

func (m *MonitoringDatabaseProxy) Close() error {
    m.metrics.DecrementConnectionCount()
    return m.realDB.Close()
}
```

### 3. Account Service Proxy
```go
// Account service interface
type AccountService interface {
    GetAccount(ctx context.Context, accountID string) (*Account, error)
    UpdateBalance(ctx context.Context, accountID string, amount decimal.Decimal) error
    FreezeAccount(ctx context.Context, accountID string) error
    UnfreezeAccount(ctx context.Context, accountID string) error
}

// Real account service
type BankAccountService struct {
    db     Database
    logger *zap.Logger
}

func NewBankAccountService(db Database, logger *zap.Logger) *BankAccountService {
    return &BankAccountService{
        db:     db,
        logger: logger,
    }
}

func (b *BankAccountService) GetAccount(ctx context.Context, accountID string) (*Account, error) {
    query := "SELECT id, balance, status, created_at, updated_at FROM accounts WHERE id = $1"
    row := b.db.QueryRow(ctx, query, accountID)
    
    var account Account
    err := row.Scan(&account.ID, &account.Balance, &account.Status, &account.CreatedAt, &account.UpdatedAt)
    if err != nil {
        return nil, fmt.Errorf("failed to get account: %w", err)
    }
    
    return &account, nil
}

func (b *BankAccountService) UpdateBalance(ctx context.Context, accountID string, amount decimal.Decimal) error {
    query := "UPDATE accounts SET balance = balance + $1, updated_at = NOW() WHERE id = $2"
    _, err := b.db.Exec(ctx, query, amount, accountID)
    if err != nil {
        return fmt.Errorf("failed to update balance: %w", err)
    }
    return nil
}

func (b *BankAccountService) FreezeAccount(ctx context.Context, accountID string) error {
    query := "UPDATE accounts SET status = 'FROZEN', updated_at = NOW() WHERE id = $1"
    _, err := b.db.Exec(ctx, query, accountID)
    return err
}

func (b *BankAccountService) UnfreezeAccount(ctx context.Context, accountID string) error {
    query := "UPDATE accounts SET status = 'ACTIVE', updated_at = NOW() WHERE id = $1"
    _, err := b.db.Exec(ctx, query, accountID)
    return err
}

// Caching account service proxy
type CachingAccountServiceProxy struct {
    realService AccountService
    cache       Cache
    cacheTTL    time.Duration
    logger      *zap.Logger
}

func NewCachingAccountServiceProxy(service AccountService, cache Cache, ttl time.Duration, logger *zap.Logger) *CachingAccountServiceProxy {
    return &CachingAccountServiceProxy{
        realService: service,
        cache:       cache,
        cacheTTL:    ttl,
        logger:      logger,
    }
}

func (c *CachingAccountServiceProxy) GetAccount(ctx context.Context, accountID string) (*Account, error) {
    cacheKey := fmt.Sprintf("account:%s", accountID)
    
    // Try cache first
    if cached, err := c.cache.Get(cacheKey); err == nil {
        c.logger.Debug("Account cache hit", zap.String("account_id", accountID))
        return cached.(*Account), nil
    }
    
    // Cache miss - get from real service
    c.logger.Debug("Account cache miss", zap.String("account_id", accountID))
    account, err := c.realService.GetAccount(ctx, accountID)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    if err := c.cache.Set(cacheKey, account, c.cacheTTL); err != nil {
        c.logger.Warn("Failed to cache account", zap.Error(err))
    }
    
    return account, nil
}

func (c *CachingAccountServiceProxy) UpdateBalance(ctx context.Context, accountID string, amount decimal.Decimal) error {
    // Update through real service
    err := c.realService.UpdateBalance(ctx, accountID, amount)
    if err != nil {
        return err
    }
    
    // Invalidate cache
    cacheKey := fmt.Sprintf("account:%s", accountID)
    c.cache.Delete(cacheKey)
    c.logger.Debug("Invalidated account cache", zap.String("account_id", accountID))
    
    return nil
}

func (c *CachingAccountServiceProxy) FreezeAccount(ctx context.Context, accountID string) error {
    err := c.realService.FreezeAccount(ctx, accountID)
    if err != nil {
        return err
    }
    
    // Invalidate cache
    cacheKey := fmt.Sprintf("account:%s", accountID)
    c.cache.Delete(cacheKey)
    
    return nil
}

func (c *CachingAccountServiceProxy) UnfreezeAccount(ctx context.Context, accountID string) error {
    err := c.realService.UnfreezeAccount(ctx, accountID)
    if err != nil {
        return err
    }
    
    // Invalidate cache
    cacheKey := fmt.Sprintf("account:%s", accountID)
    c.cache.Delete(cacheKey)
    
    return nil
}

// Validation proxy
type ValidatingAccountServiceProxy struct {
    realService AccountService
    validator   AccountValidator
    logger      *zap.Logger
}

type AccountValidator interface {
    ValidateAccountID(accountID string) error
    ValidateBalanceUpdate(accountID string, amount decimal.Decimal) error
}

func NewValidatingAccountServiceProxy(service AccountService, validator AccountValidator, logger *zap.Logger) *ValidatingAccountServiceProxy {
    return &ValidatingAccountServiceProxy{
        realService: service,
        validator:   validator,
        logger:      logger,
    }
}

func (v *ValidatingAccountServiceProxy) GetAccount(ctx context.Context, accountID string) (*Account, error) {
    if err := v.validator.ValidateAccountID(accountID); err != nil {
        v.logger.Warn("Invalid account ID", zap.String("account_id", accountID), zap.Error(err))
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    return v.realService.GetAccount(ctx, accountID)
}

func (v *ValidatingAccountServiceProxy) UpdateBalance(ctx context.Context, accountID string, amount decimal.Decimal) error {
    if err := v.validator.ValidateAccountID(accountID); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    if err := v.validator.ValidateBalanceUpdate(accountID, amount); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    return v.realService.UpdateBalance(ctx, accountID, amount)
}

func (v *ValidatingAccountServiceProxy) FreezeAccount(ctx context.Context, accountID string) error {
    if err := v.validator.ValidateAccountID(accountID); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    return v.realService.FreezeAccount(ctx, accountID)
}

func (v *ValidatingAccountServiceProxy) UnfreezeAccount(ctx context.Context, accountID string) error {
    if err := v.validator.ValidateAccountID(accountID); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    return v.realService.UnfreezeAccount(ctx, accountID)
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
    "github.com/shopspring/decimal"
    "go.uber.org/zap"
)

// Subject interface that both real subject and proxy implement
type ImageService interface {
    LoadImage(filename string) (*Image, error)
    GetImageMetadata(filename string) (*ImageMetadata, error)
    ResizeImage(filename string, width, height int) (*Image, error)
}

// Real subject - expensive to create and use
type RealImageService struct {
    logger *zap.Logger
}

func NewRealImageService(logger *zap.Logger) *RealImageService {
    return &RealImageService{
        logger: logger,
    }
}

func (r *RealImageService) LoadImage(filename string) (*Image, error) {
    r.logger.Info("Loading image from disk", zap.String("filename", filename))
    
    // Simulate expensive I/O operation
    time.Sleep(100 * time.Millisecond)
    
    return &Image{
        Filename: filename,
        Data:     []byte(fmt.Sprintf("image_data_for_%s", filename)),
        Width:    1920,
        Height:   1080,
        Size:     1024 * 1024, // 1MB
        LoadedAt: time.Now(),
    }, nil
}

func (r *RealImageService) GetImageMetadata(filename string) (*ImageMetadata, error) {
    r.logger.Info("Reading image metadata", zap.String("filename", filename))
    
    // Simulate metadata reading
    time.Sleep(20 * time.Millisecond)
    
    return &ImageMetadata{
        Filename:    filename,
        Width:       1920,
        Height:      1080,
        Size:        1024 * 1024,
        Format:      "JPEG",
        CreatedAt:   time.Now().Add(-24 * time.Hour),
        ModifiedAt:  time.Now().Add(-12 * time.Hour),
    }, nil
}

func (r *RealImageService) ResizeImage(filename string, width, height int) (*Image, error) {
    r.logger.Info("Resizing image", 
        zap.String("filename", filename),
        zap.Int("width", width),
        zap.Int("height", height))
    
    // Simulate expensive resize operation
    time.Sleep(200 * time.Millisecond)
    
    return &Image{
        Filename: fmt.Sprintf("%s_resized_%dx%d", filename, width, height),
        Data:     []byte(fmt.Sprintf("resized_data_%dx%d", width, height)),
        Width:    width,
        Height:   height,
        Size:     int64(width * height * 3), // Rough calculation
        LoadedAt: time.Now(),
    }, nil
}

// Image and metadata structures
type Image struct {
    Filename string
    Data     []byte
    Width    int
    Height   int
    Size     int64
    LoadedAt time.Time
}

type ImageMetadata struct {
    Filename   string
    Width      int
    Height     int
    Size       int64
    Format     string
    CreatedAt  time.Time
    ModifiedAt time.Time
}

// Virtual proxy - provides lazy loading
type VirtualImageProxy struct {
    filename     string
    realService  ImageService
    cachedImage  *Image
    loadOnce     sync.Once
    loadError    error
    logger       *zap.Logger
}

func NewVirtualImageProxy(filename string, realService ImageService, logger *zap.Logger) *VirtualImageProxy {
    return &VirtualImageProxy{
        filename:    filename,
        realService: realService,
        logger:      logger,
    }
}

func (v *VirtualImageProxy) LoadImage(filename string) (*Image, error) {
    if filename != v.filename {
        // Different filename, delegate to real service
        return v.realService.LoadImage(filename)
    }
    
    // Lazy loading for our specific filename
    v.loadOnce.Do(func() {
        v.logger.Info("Virtual proxy: lazy loading image", zap.String("filename", v.filename))
        v.cachedImage, v.loadError = v.realService.LoadImage(v.filename)
    })
    
    return v.cachedImage, v.loadError
}

func (v *VirtualImageProxy) GetImageMetadata(filename string) (*ImageMetadata, error) {
    // Metadata operations are usually lightweight, so we delegate
    return v.realService.GetImageMetadata(filename)
}

func (v *VirtualImageProxy) ResizeImage(filename string, width, height int) (*Image, error) {
    // Resize operations always delegate to real service
    return v.realService.ResizeImage(filename, width, height)
}

// Caching proxy
type CachingImageProxy struct {
    realService ImageService
    imageCache  map[string]*Image
    metaCache   map[string]*ImageMetadata
    resizeCache map[string]*Image
    cacheTTL    time.Duration
    mu          sync.RWMutex
    logger      *zap.Logger
}

func NewCachingImageProxy(realService ImageService, cacheTTL time.Duration, logger *zap.Logger) *CachingImageProxy {
    proxy := &CachingImageProxy{
        realService: realService,
        imageCache:  make(map[string]*Image),
        metaCache:   make(map[string]*ImageMetadata),
        resizeCache: make(map[string]*Image),
        cacheTTL:    cacheTTL,
        logger:      logger,
    }
    
    // Start cache cleanup goroutine
    go proxy.cleanupCache()
    
    return proxy
}

func (c *CachingImageProxy) LoadImage(filename string) (*Image, error) {
    c.mu.RLock()
    if cachedImage, exists := c.imageCache[filename]; exists {
        c.mu.RUnlock()
        c.logger.Debug("Image cache hit", zap.String("filename", filename))
        return cachedImage, nil
    }
    c.mu.RUnlock()
    
    c.logger.Debug("Image cache miss", zap.String("filename", filename))
    image, err := c.realService.LoadImage(filename)
    if err != nil {
        return nil, err
    }
    
    c.mu.Lock()
    c.imageCache[filename] = image
    c.mu.Unlock()
    
    return image, nil
}

func (c *CachingImageProxy) GetImageMetadata(filename string) (*ImageMetadata, error) {
    c.mu.RLock()
    if cachedMeta, exists := c.metaCache[filename]; exists {
        c.mu.RUnlock()
        c.logger.Debug("Metadata cache hit", zap.String("filename", filename))
        return cachedMeta, nil
    }
    c.mu.RUnlock()
    
    c.logger.Debug("Metadata cache miss", zap.String("filename", filename))
    metadata, err := c.realService.GetImageMetadata(filename)
    if err != nil {
        return nil, err
    }
    
    c.mu.Lock()
    c.metaCache[filename] = metadata
    c.mu.Unlock()
    
    return metadata, nil
}

func (c *CachingImageProxy) ResizeImage(filename string, width, height int) (*Image, error) {
    cacheKey := fmt.Sprintf("%s_%dx%d", filename, width, height)
    
    c.mu.RLock()
    if cachedResize, exists := c.resizeCache[cacheKey]; exists {
        c.mu.RUnlock()
        c.logger.Debug("Resize cache hit", zap.String("cache_key", cacheKey))
        return cachedResize, nil
    }
    c.mu.RUnlock()
    
    c.logger.Debug("Resize cache miss", zap.String("cache_key", cacheKey))
    resizedImage, err := c.realService.ResizeImage(filename, width, height)
    if err != nil {
        return nil, err
    }
    
    c.mu.Lock()
    c.resizeCache[cacheKey] = resizedImage
    c.mu.Unlock()
    
    return resizedImage, nil
}

func (c *CachingImageProxy) cleanupCache() {
    ticker := time.NewTicker(c.cacheTTL)
    defer ticker.Stop()
    
    for range ticker.C {
        c.logger.Debug("Cleaning up image cache")
        c.mu.Lock()
        
        // In a real implementation, you'd track timestamps and remove expired entries
        // For simplicity, we'll just clear old caches periodically
        if len(c.imageCache) > 100 {
            c.imageCache = make(map[string]*Image)
        }
        if len(c.metaCache) > 200 {
            c.metaCache = make(map[string]*ImageMetadata)
        }
        if len(c.resizeCache) > 50 {
            c.resizeCache = make(map[string]*Image)
        }
        
        c.mu.Unlock()
    }
}

// Access control proxy
type AccessControlImageProxy struct {
    realService   ImageService
    authenticator Authenticator
    authorizer    Authorizer
    logger        *zap.Logger
}

type Authenticator interface {
    Authenticate(ctx context.Context) (*User, error)
}

type Authorizer interface {
    CanAccessImage(user *User, filename string) bool
    CanResizeImage(user *User, filename string) bool
}

type User struct {
    ID       string
    Username string
    Role     string
    Permissions []string
}

func NewAccessControlImageProxy(realService ImageService, auth Authenticator, authz Authorizer, logger *zap.Logger) *AccessControlImageProxy {
    return &AccessControlImageProxy{
        realService:   realService,
        authenticator: auth,
        authorizer:    authz,
        logger:        logger,
    }
}

func (a *AccessControlImageProxy) LoadImage(filename string) (*Image, error) {
    ctx := context.Background()
    user, err := a.authenticator.Authenticate(ctx)
    if err != nil {
        a.logger.Warn("Authentication failed for image access", zap.String("filename", filename))
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    if !a.authorizer.CanAccessImage(user, filename) {
        a.logger.Warn("Access denied for image", 
            zap.String("user", user.Username),
            zap.String("filename", filename))
        return nil, fmt.Errorf("access denied to image: %s", filename)
    }
    
    a.logger.Info("Image access granted", 
        zap.String("user", user.Username),
        zap.String("filename", filename))
    
    return a.realService.LoadImage(filename)
}

func (a *AccessControlImageProxy) GetImageMetadata(filename string) (*ImageMetadata, error) {
    ctx := context.Background()
    user, err := a.authenticator.Authenticate(ctx)
    if err != nil {
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    if !a.authorizer.CanAccessImage(user, filename) {
        return nil, fmt.Errorf("access denied to image metadata: %s", filename)
    }
    
    return a.realService.GetImageMetadata(filename)
}

func (a *AccessControlImageProxy) ResizeImage(filename string, width, height int) (*Image, error) {
    ctx := context.Background()
    user, err := a.authenticator.Authenticate(ctx)
    if err != nil {
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    if !a.authorizer.CanResizeImage(user, filename) {
        a.logger.Warn("Resize access denied", 
            zap.String("user", user.Username),
            zap.String("filename", filename))
        return nil, fmt.Errorf("access denied to resize image: %s", filename)
    }
    
    return a.realService.ResizeImage(filename, width, height)
}

// Logging proxy
type LoggingImageProxy struct {
    realService ImageService
    logger      *zap.Logger
}

func NewLoggingImageProxy(realService ImageService, logger *zap.Logger) *LoggingImageProxy {
    return &LoggingImageProxy{
        realService: realService,
        logger:      logger,
    }
}

func (l *LoggingImageProxy) LoadImage(filename string) (*Image, error) {
    start := time.Now()
    l.logger.Info("Loading image started", zap.String("filename", filename))
    
    image, err := l.realService.LoadImage(filename)
    duration := time.Since(start)
    
    if err != nil {
        l.logger.Error("Loading image failed", 
            zap.String("filename", filename),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        l.logger.Info("Loading image completed", 
            zap.String("filename", filename),
            zap.Int64("size", image.Size),
            zap.Duration("duration", duration))
    }
    
    return image, err
}

func (l *LoggingImageProxy) GetImageMetadata(filename string) (*ImageMetadata, error) {
    start := time.Now()
    metadata, err := l.realService.GetImageMetadata(filename)
    duration := time.Since(start)
    
    if err != nil {
        l.logger.Error("Getting metadata failed", 
            zap.String("filename", filename),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        l.logger.Info("Getting metadata completed", 
            zap.String("filename", filename),
            zap.String("format", metadata.Format),
            zap.Duration("duration", duration))
    }
    
    return metadata, err
}

func (l *LoggingImageProxy) ResizeImage(filename string, width, height int) (*Image, error) {
    start := time.Now()
    l.logger.Info("Resizing image started", 
        zap.String("filename", filename),
        zap.Int("width", width),
        zap.Int("height", height))
    
    image, err := l.realService.ResizeImage(filename, width, height)
    duration := time.Since(start)
    
    if err != nil {
        l.logger.Error("Resizing image failed", 
            zap.String("filename", filename),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        l.logger.Info("Resizing image completed", 
            zap.String("filename", filename),
            zap.Int64("size", image.Size),
            zap.Duration("duration", duration))
    }
    
    return image, err
}

// Mock implementations for demonstration
type MockAuthenticator struct{}

func (m *MockAuthenticator) Authenticate(ctx context.Context) (*User, error) {
    return &User{
        ID:       "user123",
        Username: "john.doe",
        Role:     "user",
        Permissions: []string{"read_images", "resize_images"},
    }, nil
}

type MockAuthorizer struct{}

func (m *MockAuthorizer) CanAccessImage(user *User, filename string) bool {
    return contains(user.Permissions, "read_images")
}

func (m *MockAuthorizer) CanResizeImage(user *User, filename string) bool {
    return contains(user.Permissions, "resize_images")
}

func contains(slice []string, item string) bool {
    for _, s := range slice {
        if s == item {
            return true
        }
    }
    return false
}

// Example usage demonstrating different proxy types
func main() {
    fmt.Println("=== Proxy Pattern Demo ===\n")
    
    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()
    
    // Create real image service
    realService := NewRealImageService(logger)
    
    // Demonstrate virtual proxy (lazy loading)
    fmt.Println("=== Virtual Proxy (Lazy Loading) ===")
    virtualProxy := NewVirtualImageProxy("large_image.jpg", realService, logger)
    
    // The image won't be loaded until first access
    fmt.Println("Virtual proxy created, but image not loaded yet")
    
    // First access triggers loading
    image1, err := virtualProxy.LoadImage("large_image.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Loaded image: %s (%d bytes)\n", image1.Filename, image1.Size)
    }
    
    // Second access uses cached image
    image2, err := virtualProxy.LoadImage("large_image.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Loaded image from cache: %s (%d bytes)\n", image2.Filename, image2.Size)
        fmt.Printf("Same instance: %t\n", image1 == image2)
    }
    
    // Demonstrate caching proxy
    fmt.Println("\n=== Caching Proxy ===")
    cachingProxy := NewCachingImageProxy(realService, 5*time.Minute, logger)
    
    // First load - cache miss
    start := time.Now()
    image3, err := cachingProxy.LoadImage("photo.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("First load: %s (%d bytes) - took %v\n", image3.Filename, image3.Size, time.Since(start))
    }
    
    // Second load - cache hit
    start = time.Now()
    image4, err := cachingProxy.LoadImage("photo.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Second load: %s (%d bytes) - took %v\n", image4.Filename, image4.Size, time.Since(start))
    }
    
    // Demonstrate metadata caching
    start = time.Now()
    metadata1, err := cachingProxy.GetImageMetadata("photo.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("First metadata load: %s - took %v\n", metadata1.Format, time.Since(start))
    }
    
    start = time.Now()
    metadata2, err := cachingProxy.GetImageMetadata("photo.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Second metadata load: %s - took %v\n", metadata2.Format, time.Since(start))
    }
    
    // Demonstrate access control proxy
    fmt.Println("\n=== Access Control Proxy ===")
    authenticator := &MockAuthenticator{}
    authorizer := &MockAuthorizer{}
    accessProxy := NewAccessControlImageProxy(realService, authenticator, authorizer, logger)
    
    // Authorized access
    image5, err := accessProxy.LoadImage("secure_image.jpg")
    if err != nil {
        fmt.Printf("Access denied: %v\n", err)
    } else {
        fmt.Printf("Authorized access: %s (%d bytes)\n", image5.Filename, image5.Size)
    }
    
    // Authorized resize
    resized, err := accessProxy.ResizeImage("secure_image.jpg", 800, 600)
    if err != nil {
        fmt.Printf("Resize denied: %v\n", err)
    } else {
        fmt.Printf("Authorized resize: %s (%dx%d)\n", resized.Filename, resized.Width, resized.Height)
    }
    
    // Demonstrate logging proxy
    fmt.Println("\n=== Logging Proxy ===")
    loggingProxy := NewLoggingImageProxy(realService, logger)
    
    image6, err := loggingProxy.LoadImage("logged_image.jpg")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Logged operation completed: %s\n", image6.Filename)
    }
    
    // Demonstrate proxy chaining
    fmt.Println("\n=== Chained Proxies ===")
    
    // Chain: Real Service -> Caching -> Access Control -> Logging
    var service ImageService = realService
    service = NewCachingImageProxy(service, 5*time.Minute, logger)
    service = NewAccessControlImageProxy(service, authenticator, authorizer, logger)
    service = NewLoggingImageProxy(service, logger)
    
    fmt.Println("Processing request through chained proxies:")
    image7, err := service.LoadImage("chained_image.jpg")
    if err != nil {
        fmt.Printf("Chained request failed: %v\n", err)
    } else {
        fmt.Printf("Chained request succeeded: %s (%d bytes)\n", image7.Filename, image7.Size)
    }
    
    // Second request should hit cache
    fmt.Println("Second request through chained proxies:")
    image8, err := service.LoadImage("chained_image.jpg")
    if err != nil {
        fmt.Printf("Chained request failed: %v\n", err)
    } else {
        fmt.Printf("Chained request succeeded: %s (%d bytes)\n", image8.Filename, image8.Size)
    }
    
    fmt.Println("\n=== Proxy Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Remote Proxy**
```go
type RemoteImageProxy struct {
    serverURL  string
    httpClient *http.Client
    logger     *zap.Logger
}

func (r *RemoteImageProxy) LoadImage(filename string) (*Image, error) {
    url := fmt.Sprintf("%s/images/%s", r.serverURL, filename)
    
    resp, err := r.httpClient.Get(url)
    if err != nil {
        return nil, fmt.Errorf("remote call failed: %w", err)
    }
    defer resp.Body.Close()
    
    var image Image
    if err := json.NewDecoder(resp.Body).Decode(&image); err != nil {
        return nil, err
    }
    
    return &image, nil
}
```

2. **Smart Reference Proxy**
```go
type SmartReferenceProxy struct {
    realObject  *ExpensiveResource
    refCount    int
    mu          sync.Mutex
    lastAccess  time.Time
}

func (s *SmartReferenceProxy) UseResource() {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    s.refCount++
    s.lastAccess = time.Now()
    
    if s.realObject == nil {
        s.realObject = NewExpensiveResource()
    }
    
    s.realObject.DoWork()
}

func (s *SmartReferenceProxy) ReleaseResource() {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    s.refCount--
    if s.refCount <= 0 {
        s.realObject.Close()
        s.realObject = nil
    }
}
```

3. **Copy-on-Write Proxy**
```go
type CopyOnWriteProxy struct {
    original *SharedData
    copied   *SharedData
    isOwner  bool
    mu       sync.RWMutex
}

func (c *CopyOnWriteProxy) Read() interface{} {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    if c.copied != nil {
        return c.copied.Data
    }
    return c.original.Data
}

func (c *CopyOnWriteProxy) Write(data interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if !c.isOwner {
        // Copy on first write
        c.copied = c.original.Copy()
        c.isOwner = true
    }
    
    c.copied.Data = data
}
```

### Trade-offs

**Pros:**
- **Controlled Access**: Can control when and how objects are accessed
- **Lazy Loading**: Delays expensive operations until needed
- **Caching**: Can cache expensive operations and results
- **Security**: Can add authentication and authorization
- **Monitoring**: Can add logging and metrics without changing original code
- **Remote Access**: Can provide local interface to remote objects

**Cons:**
- **Additional Complexity**: Adds another layer of indirection
- **Performance Overhead**: Method calls through proxy add latency
- **Memory Overhead**: Proxy objects consume additional memory
- **Debugging Difficulty**: Stack traces become more complex
- **Interface Coupling**: Proxy must implement same interface as real object

**When to Choose Proxy vs Alternatives:**

| Scenario | Pattern | Reason |
|----------|---------|--------|
| Lazy loading | Proxy | Control object creation timing |
| Cross-cutting concerns | Decorator | Add behavior dynamically |
| Interface compatibility | Adapter | Interface conversion |
| Simplified interface | Facade | Hide complexity |
| Object pooling | Object Pool | Reuse expensive objects |

## Integration Tips

### 1. Factory Pattern Integration
```go
type ProxyFactory interface {
    CreateProxy(realObject interface{}, config ProxyConfig) interface{}
}

type ProxyConfig struct {
    EnableCaching     bool
    EnableLogging     bool
    EnableAccessControl bool
    CacheTTL         time.Duration
}

type ImageProxyFactory struct {
    logger        *zap.Logger
    authenticator Authenticator
    authorizer    Authorizer
}

func (ipf *ImageProxyFactory) CreateProxy(realService interface{}, config ProxyConfig) interface{} {
    service := realService.(ImageService)
    
    if config.EnableCaching {
        service = NewCachingImageProxy(service, config.CacheTTL, ipf.logger)
    }
    
    if config.EnableAccessControl {
        service = NewAccessControlImageProxy(service, ipf.authenticator, ipf.authorizer, ipf.logger)
    }
    
    if config.EnableLogging {
        service = NewLoggingImageProxy(service, ipf.logger)
    }
    
    return service
}
```

### 2. Decorator Pattern Integration
```go
type ProxyDecorator struct {
    proxy Proxy
    decorators []Decorator
}

func (pd *ProxyDecorator) AddDecorator(decorator Decorator) {
    pd.decorators = append(pd.decorators, decorator)
}

func (pd *ProxyDecorator) Invoke(method string, args ...interface{}) (interface{}, error) {
    // Apply decorators before proxy
    for _, decorator := range pd.decorators {
        if err := decorator.Before(method, args...); err != nil {
            return nil, err
        }
    }
    
    result, err := pd.proxy.Invoke(method, args...)
    
    // Apply decorators after proxy
    for i := len(pd.decorators) - 1; i >= 0; i-- {
        pd.decorators[i].After(method, result, err)
    }
    
    return result, err
}
```

### 3. Strategy Pattern Integration
```go
type CachingStrategy interface {
    ShouldCache(method string, args []interface{}) bool
    GenerateKey(method string, args []interface{}) string
    GetTTL(method string) time.Duration
}

type DefaultCachingStrategy struct{}

func (d *DefaultCachingStrategy) ShouldCache(method string, args []interface{}) bool {
    // Don't cache write operations
    writeMethods := []string{"Update", "Delete", "Insert", "Create"}
    for _, writeMethod := range writeMethods {
        if strings.Contains(method, writeMethod) {
            return false
        }
    }
    return true
}

type StrategicCachingProxy struct {
    realService ImageService
    cache       Cache
    strategy    CachingStrategy
}

func (scp *StrategicCachingProxy) LoadImage(filename string) (*Image, error) {
    if !scp.strategy.ShouldCache("LoadImage", []interface{}{filename}) {
        return scp.realService.LoadImage(filename)
    }
    
    key := scp.strategy.GenerateKey("LoadImage", []interface{}{filename})
    
    if cached, err := scp.cache.Get(key); err == nil {
        return cached.(*Image), nil
    }
    
    result, err := scp.realService.LoadImage(filename)
    if err == nil {
        ttl := scp.strategy.GetTTL("LoadImage")
        scp.cache.Set(key, result, ttl)
    }
    
    return result, err
}
```

### 4. Observer Pattern Integration
```go
type ProxyEventListener interface {
    OnMethodCalled(proxy interface{}, method string, args []interface{})
    OnMethodCompleted(proxy interface{}, method string, result interface{}, err error, duration time.Duration)
}

type ObservableProxy struct {
    realService ImageService
    listeners   []ProxyEventListener
}

func (op *ObservableProxy) LoadImage(filename string) (*Image, error) {
    // Notify listeners
    for _, listener := range op.listeners {
        listener.OnMethodCalled(op, "LoadImage", []interface{}{filename})
    }
    
    start := time.Now()
    result, err := op.realService.LoadImage(filename)
    duration := time.Since(start)
    
    // Notify completion
    for _, listener := range op.listeners {
        listener.OnMethodCompleted(op, "LoadImage", result, err, duration)
    }
    
    return result, err
}

func (op *ObservableProxy) AddListener(listener ProxyEventListener) {
    op.listeners = append(op.listeners, listener)
}
```

## Common Interview Questions

### 1. **How does Proxy pattern differ from Decorator pattern?**

**Answer:**
Both patterns involve wrapping objects, but they have different purposes and usage:

**Proxy Pattern:**
```go
// Proxy controls access to the real object
type AccountServiceProxy struct {
    realService AccountService
    accessControl AccessController
}

// Purpose: Control access, lazy loading, caching
func (p *AccountServiceProxy) GetAccount(id string) (*Account, error) {
    // Control access
    if !p.accessControl.CanAccess(id) {
        return nil, errors.New("access denied")
    }
    
    // Delegate to real service
    return p.realService.GetAccount(id)
}

// Client uses proxy as if it were the real service
proxy := NewAccountServiceProxy(realService, accessController)
account, err := proxy.GetAccount("123")
```

**Decorator Pattern:**
```go
// Decorator adds new functionality
type LoggingAccountServiceDecorator struct {
    wrapped AccountService
    logger  Logger
}

// Purpose: Add behavior without changing original
func (d *LoggingAccountServiceDecorator) GetAccount(id string) (*Account, error) {
    d.logger.Log("GetAccount called with ID: " + id)
    
    result, err := d.wrapped.GetAccount(id)
    
    if err != nil {
        d.logger.Log("GetAccount failed: " + err.Error())
    } else {
        d.logger.Log("GetAccount succeeded")
    }
    
    return result, err
}

// Multiple decorators can be stacked
service := NewLoggingAccountServiceDecorator(
    NewValidatingAccountServiceDecorator(
        NewCachingAccountServiceDecorator(realService)
    )
)
```

**Key Differences:**

| Aspect | Proxy | Decorator |
|--------|-------|-----------|
| **Purpose** | Control access to object | Add functionality to object |
| **Relationship** | Proxy "is-a" subject | Decorator "has-a" component |
| **Object Creation** | May control object creation | Wraps existing objects |
| **Interface** | Same interface as real object | Usually same interface |
| **Stacking** | Usually single proxy | Multiple decorators can be stacked |
| **Use Cases** | Lazy loading, access control, caching | Logging, validation, encryption |

### 2. **How do you implement thread-safe proxy objects?**

**Answer:**
Thread safety in proxies requires careful synchronization, especially for caching and lazy loading:

**Thread-Safe Lazy Loading:**
```go
type ThreadSafeLazyProxy struct {
    factory    func() (Service, error)
    service    Service
    initError  error
    initOnce   sync.Once
    mu         sync.RWMutex
}

func (p *ThreadSafeLazyProxy) GetService() (Service, error) {
    // Double-checked locking pattern
    p.mu.RLock()
    if p.service != nil {
        service := p.service
        p.mu.RUnlock()
        return service, nil
    }
    if p.initError != nil {
        err := p.initError
        p.mu.RUnlock()
        return nil, err
    }
    p.mu.RUnlock()
    
    // Initialize only once
    p.initOnce.Do(func() {
        p.mu.Lock()
        defer p.mu.Unlock()
        p.service, p.initError = p.factory()
    })
    
    p.mu.RLock()
    defer p.mu.RUnlock()
    return p.service, p.initError
}
```

**Thread-Safe Caching Proxy:**
```go
type ThreadSafeCachingProxy struct {
    realService Service
    cache       sync.Map // Built-in concurrent map
    cacheTTL    time.Duration
}

type cacheEntry struct {
    value     interface{}
    timestamp time.Time
}

func (p *ThreadSafeCachingProxy) Get(key string) (interface{}, error) {
    // Check cache
    if entry, ok := p.cache.Load(key); ok {
        cached := entry.(*cacheEntry)
        if time.Since(cached.timestamp) < p.cacheTTL {
            return cached.value, nil
        }
        // Expired - remove from cache
        p.cache.Delete(key)
    }
    
    // Get from real service
    value, err := p.realService.Get(key)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    p.cache.Store(key, &cacheEntry{
        value:     value,
        timestamp: time.Now(),
    })
    
    return value, nil
}
```

**Reference Counting Proxy:**
```go
type RefCountingProxy struct {
    realObject interface{}
    refCount   int64
    mu         sync.Mutex
    closed     bool
}

func (r *RefCountingProxy) AddRef() {
    atomic.AddInt64(&r.refCount, 1)
}

func (r *RefCountingProxy) Release() {
    newCount := atomic.AddInt64(&r.refCount, -1)
    if newCount == 0 {
        r.mu.Lock()
        defer r.mu.Unlock()
        if !r.closed {
            if closer, ok := r.realObject.(io.Closer); ok {
                closer.Close()
            }
            r.closed = true
        }
    }
}

func (r *RefCountingProxy) GetObject() (interface{}, error) {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    if r.closed {
        return nil, errors.New("object has been closed")
    }
    
    return r.realObject, nil
}
```

**Channel-Based Serialized Access:**
```go
type SerializedProxy struct {
    requests chan request
    done     chan struct{}
}

type request struct {
    method   string
    args     []interface{}
    response chan response
}

type response struct {
    result interface{}
    error  error
}

func (s *SerializedProxy) Start(realService Service) {
    go func() {
        for {
            select {
            case req := <-s.requests:
                s.handleRequest(realService, req)
            case <-s.done:
                return
            }
        }
    }()
}

func (s *SerializedProxy) Call(method string, args ...interface{}) (interface{}, error) {
    respChan := make(chan response, 1)
    
    s.requests <- request{
        method:   method,
        args:     args,
        response: respChan,
    }
    
    resp := <-respChan
    return resp.result, resp.error
}
```

### 3. **How do you handle error scenarios in proxy implementations?**

**Answer:**
Error handling in proxies should be robust and provide useful information to clients:

**Fallback Proxy:**
```go
type FallbackProxy struct {
    primaryService   Service
    fallbackService  Service
    circuitBreaker   *CircuitBreaker
    logger          *zap.Logger
}

func (f *FallbackProxy) Get(key string) (interface{}, error) {
    // Try primary service first
    if f.circuitBreaker.CanAttempt() {
        result, err := f.primaryService.Get(key)
        if err == nil {
            f.circuitBreaker.RecordSuccess()
            return result, nil
        }
        
        f.circuitBreaker.RecordFailure()
        f.logger.Warn("Primary service failed, trying fallback", zap.Error(err))
    }
    
    // Try fallback service
    result, err := f.fallbackService.Get(key)
    if err != nil {
        return nil, fmt.Errorf("both primary and fallback services failed: %w", err)
    }
    
    f.logger.Info("Fallback service succeeded")
    return result, nil
}
```

**Retry Proxy:**
```go
type RetryProxy struct {
    realService Service
    maxRetries  int
    backoff     BackoffStrategy
    logger      *zap.Logger
}

func (r *RetryProxy) Get(key string) (interface{}, error) {
    var lastErr error
    
    for attempt := 0; attempt <= r.maxRetries; attempt++ {
        if attempt > 0 {
            delay := r.backoff.NextDelay(attempt)
            r.logger.Info("Retrying request", 
                zap.Int("attempt", attempt),
                zap.Duration("delay", delay))
            time.Sleep(delay)
        }
        
        result, err := r.realService.Get(key)
        if err == nil {
            if attempt > 0 {
                r.logger.Info("Request succeeded after retry", zap.Int("attempts", attempt+1))
            }
            return result, nil
        }
        
        lastErr = err
        
        // Don't retry for certain error types
        if !r.isRetryableError(err) {
            break
        }
    }
    
    return nil, fmt.Errorf("request failed after %d attempts: %w", r.maxRetries+1, lastErr)
}

func (r *RetryProxy) isRetryableError(err error) bool {
    // Check for network errors, timeouts, etc.
    if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
        return true
    }
    
    // Check for specific error types
    if strings.Contains(err.Error(), "connection refused") {
        return true
    }
    
    return false
}
```

**Error Aggregation Proxy:**
```go
type ErrorAggregatingProxy struct {
    services []Service
    strategy AggregationStrategy
    logger   *zap.Logger
}

type AggregationStrategy int

const (
    FailFast AggregationStrategy = iota
    CollectAll
    Majority
)

func (e *ErrorAggregatingProxy) Get(key string) (interface{}, error) {
    switch e.strategy {
    case FailFast:
        return e.getFailFast(key)
    case CollectAll:
        return e.getCollectAll(key)
    case Majority:
        return e.getMajority(key)
    default:
        return e.getFailFast(key)
    }
}

func (e *ErrorAggregatingProxy) getFailFast(key string) (interface{}, error) {
    for i, service := range e.services {
        result, err := service.Get(key)
        if err == nil {
            return result, nil
        }
        e.logger.Warn("Service failed", zap.Int("service_index", i), zap.Error(err))
    }
    return nil, errors.New("all services failed")
}

func (e *ErrorAggregatingProxy) getCollectAll(key string) (interface{}, error) {
    var results []interface{}
    var errors []error
    
    for _, service := range e.services {
        result, err := service.Get(key)
        if err != nil {
            errors = append(errors, err)
        } else {
            results = append(results, result)
        }
    }
    
    if len(results) == 0 {
        return nil, fmt.Errorf("all services failed: %v", errors)
    }
    
    // Return first successful result
    return results[0], nil
}
```

**Context-Aware Error Handling:**
```go
type ContextAwareProxy struct {
    realService Service
    timeout     time.Duration
    logger      *zap.Logger
}

func (c *ContextAwareProxy) Get(ctx context.Context, key string) (interface{}, error) {
    // Create timeout context
    timeoutCtx, cancel := context.WithTimeout(ctx, c.timeout)
    defer cancel()
    
    // Use channel to handle concurrent execution and timeout
    resultChan := make(chan result, 1)
    
    go func() {
        value, err := c.realService.Get(key)
        resultChan <- result{value: value, err: err}
    }()
    
    select {
    case res := <-resultChan:
        return res.value, res.err
    case <-timeoutCtx.Done():
        if timeoutCtx.Err() == context.DeadlineExceeded {
            c.logger.Warn("Request timed out", 
                zap.String("key", key),
                zap.Duration("timeout", c.timeout))
            return nil, fmt.Errorf("request timed out after %v", c.timeout)
        }
        return nil, timeoutCtx.Err()
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

type result struct {
    value interface{}
    err   error
}
```

### 4. **How do you test proxy implementations effectively?**

**Answer:**
Testing proxies requires verifying both the proxy behavior and the interaction with the real object:

**Mock-based Testing:**
```go
type MockService struct {
    mock.Mock
}

func (m *MockService) Get(key string) (interface{}, error) {
    args := m.Called(key)
    return args.Get(0), args.Error(1)
}

func TestCachingProxy(t *testing.T) {
    mockService := &MockService{}
    cache := NewMockCache()
    proxy := NewCachingProxy(mockService, cache, 5*time.Minute)
    
    expectedValue := "test_value"
    mockService.On("Get", "test_key").Return(expectedValue, nil).Once()
    
    // First call should hit the real service
    result1, err := proxy.Get("test_key")
    assert.NoError(t, err)
    assert.Equal(t, expectedValue, result1)
    
    // Second call should hit the cache
    result2, err := proxy.Get("test_key")
    assert.NoError(t, err)
    assert.Equal(t, expectedValue, result2)
    
    // Verify mock was called only once
    mockService.AssertExpectations(t)
}
```

**Behavior Testing:**
```go
func TestLazyLoadingProxy(t *testing.T) {
    factory := &MockFactory{}
    proxy := NewLazyProxy(factory.CreateService)
    
    // Service should not be created initially
    factory.AssertNotCalled(t, "CreateService")
    
    // First access should trigger creation
    factory.On("CreateService").Return(&MockService{}, nil).Once()
    
    service, err := proxy.GetService()
    assert.NoError(t, err)
    assert.NotNil(t, service)
    
    // Subsequent access should reuse the same service
    service2, err := proxy.GetService()
    assert.NoError(t, err)
    assert.Same(t, service, service2)
    
    factory.AssertExpectations(t)
}
```

**Error Scenario Testing:**
```go
func TestProxyErrorHandling(t *testing.T) {
    mockService := &MockService{}
    proxy := NewRetryProxy(mockService, 3, &ExponentialBackoff{})
    
    // Test retry on failure
    expectedError := errors.New("service unavailable")
    mockService.On("Get", "failing_key").Return(nil, expectedError).Times(4) // Initial + 3 retries
    
    result, err := proxy.Get("failing_key")
    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "failed after 4 attempts")
    
    mockService.AssertExpectations(t)
}
```

**Integration Testing:**
```go
func TestProxyIntegration(t *testing.T) {
    // Use real service for integration test
    realService := NewRealService()
    proxy := NewLoggingProxy(realService, logger)
    
    // Test actual functionality
    result, err := proxy.Get("integration_test_key")
    assert.NoError(t, err)
    assert.NotNil(t, result)
    
    // Verify logging occurred (check log output or use test logger)
    // This requires a test-friendly logger implementation
}
```

**Performance Testing:**
```go
func BenchmarkProxyVsDirect(b *testing.B) {
    service := NewRealService()
    proxy := NewCachingProxy(service, cache, 5*time.Minute)
    
    b.Run("DirectAccess", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            service.Get("benchmark_key")
        }
    })
    
    b.Run("ProxyAccess", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            proxy.Get("benchmark_key")
        }
    })
    
    b.Run("ProxyCacheHit", func(b *testing.B) {
        // Prime the cache
        proxy.Get("cached_key")
        
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            proxy.Get("cached_key")
        }
    })
}
```

### 5. **When should you avoid using Proxy pattern?**

**Answer:**
Proxy pattern should be avoided in certain scenarios where its costs outweigh benefits:

**Simple Operations:**
```go
// DON'T use proxy for simple, lightweight operations
type SimpleCalculator struct{}

func (s *SimpleCalculator) Add(a, b int) int {
    return a + b // Simple operation doesn't need proxy
}

// Proxy overhead is unnecessary
type CalculatorProxy struct {
    calc *SimpleCalculator
}

func (p *CalculatorProxy) Add(a, b int) int {
    // Unnecessary indirection for simple operation
    return p.calc.Add(a, b)
}
```

**Performance-Critical Code:**
```go
// DON'T use proxy in hot paths where every nanosecond counts
func ProcessHighFrequencyData(data []DataPoint) {
    for _, point := range data {
        // Direct call is faster than proxy indirection
        result := calculator.Process(point.Value)
        
        // proxy.Process(point.Value) // Adds unnecessary overhead
        
        processResult(result)
    }
}
```

**Stable, Simple Interfaces:**
```go
// DON'T use proxy when interface is simple and unlikely to change
type FileReader interface {
    Read(filename string) ([]byte, error)
}

type SimpleFileReader struct{}

func (s *SimpleFileReader) Read(filename string) ([]byte, error) {
    return ioutil.ReadFile(filename)
}

// Proxy adds no value here
// reader := &SimpleFileReader{} // Direct usage is better
```

**Short-Lived Objects:**
```go
// DON'T use proxy for short-lived objects
func ProcessRequest(request *Request) *Response {
    processor := NewRequestProcessor() // Short-lived
    defer processor.Close()
    
    // No need for proxy since object lifetime is so short
    return processor.Process(request)
}
```

**Better Alternatives:**

| Scenario | Alternative | Reason |
|----------|-------------|--------|
| Simple operations | Direct calls | No overhead needed |
| Cross-cutting concerns | Middleware/Interceptors | More explicit |
| Object lifecycle | Factory/Builder | Better creation control |
| Interface adaptation | Adapter | Cleaner interface conversion |
| Behavior enhancement | Decorator | More flexible |
| Configuration-based behavior | Strategy | Runtime behavior selection |

**Decision Framework:**
```go
type ProxyDecision struct {
    OperationComplexity  string // "simple", "medium", "complex"
    PerformanceRequirements string // "low", "medium", "high"
    AccessControlNeeds   bool
    CachingNeeds        bool
    LazyLoadingNeeds    bool
    RemoteAccessNeeds   bool
    ObjectLifetime      string // "short", "medium", "long"
}

func (pd *ProxyDecision) ShouldUseProxy() (bool, string) {
    if pd.OperationComplexity == "simple" && pd.PerformanceRequirements == "high" {
        return false, "Proxy overhead too high for simple, performance-critical operations"
    }
    
    if pd.ObjectLifetime == "short" && !pd.AccessControlNeeds && !pd.CachingNeeds {
        return false, "No benefits for short-lived objects without special needs"
    }
    
    if pd.AccessControlNeeds || pd.CachingNeeds || pd.LazyLoadingNeeds || pd.RemoteAccessNeeds {
        return true, "Proxy provides valuable functionality"
    }
    
    return false, "No clear benefits identified"
}
```
