---
# Auto-generated front matter
Title: Singleton Pattern
LastUpdated: 2025-11-06T20:45:58.544356
Tags: []
Status: draft
---

# Singleton Pattern

Comprehensive guide to the Singleton pattern for Razorpay interviews.

## 🎯 Singleton Pattern Overview

The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

### Key Benefits
- **Single Instance**: Guarantees only one instance exists
- **Global Access**: Provides global access point
- **Lazy Initialization**: Can be initialized when first needed
- **Memory Efficiency**: Saves memory by avoiding multiple instances

### Use Cases
- Database connections
- Logger instances
- Configuration managers
- Cache managers
- Service locators

## 🚀 Implementation Examples

### Basic Singleton Pattern
```go
// Basic Singleton Implementation
type Singleton struct {
    data string
}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{
            data: "Initial data",
        }
    })
    return instance
}

func (s *Singleton) GetData() string {
    return s.data
}

func (s *Singleton) SetData(data string) {
    s.data = data
}
```

### Thread-Safe Singleton with Mutex
```go
// Thread-Safe Singleton with Mutex
type ThreadSafeSingleton struct {
    data  string
    mutex sync.RWMutex
}

var (
    threadSafeInstance *ThreadSafeSingleton
    threadSafeOnce     sync.Once
)

func GetThreadSafeInstance() *ThreadSafeSingleton {
    threadSafeOnce.Do(func() {
        threadSafeInstance = &ThreadSafeSingleton{
            data: "Thread-safe initial data",
        }
    })
    return threadSafeInstance
}

func (ts *ThreadSafeSingleton) GetData() string {
    ts.mutex.RLock()
    defer ts.mutex.RUnlock()
    return ts.data
}

func (ts *ThreadSafeSingleton) SetData(data string) {
    ts.mutex.Lock()
    defer ts.mutex.Unlock()
    ts.data = data
}
```

### Singleton with Initialization Parameters
```go
// Singleton with Initialization Parameters
type ConfigurableSingleton struct {
    config map[string]interface{}
    mutex  sync.RWMutex
}

var (
    configurableInstance *ConfigurableSingleton
    configurableOnce     sync.Once
)

func GetConfigurableInstance(config map[string]interface{}) *ConfigurableSingleton {
    configurableOnce.Do(func() {
        configurableInstance = &ConfigurableSingleton{
            config: config,
        }
    })
    return configurableInstance
}

func (cs *ConfigurableSingleton) GetConfig(key string) (interface{}, bool) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    value, exists := cs.config[key]
    return value, exists
}

func (cs *ConfigurableSingleton) SetConfig(key string, value interface{}) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.config[key] = value
}
```

## 🔧 Advanced Singleton Patterns

### Singleton with Interface
```go
// Singleton with Interface
type DatabaseConnection interface {
    Connect() error
    Disconnect() error
    Query(sql string) (interface{}, error)
    Execute(sql string) error
}

type MySQLConnection struct {
    host     string
    port     int
    username string
    password string
    database string
    conn     *sql.DB
    mutex    sync.RWMutex
}

var (
    dbInstance DatabaseConnection
    dbOnce     sync.Once
)

func GetDatabaseInstance() DatabaseConnection {
    dbOnce.Do(func() {
        dbInstance = &MySQLConnection{
            host:     "localhost",
            port:     3306,
            username: "root",
            password: "password",
            database: "razorpay",
        }
    })
    return dbInstance
}

func (mysql *MySQLConnection) Connect() error {
    mysql.mutex.Lock()
    defer mysql.mutex.Unlock()
    
    if mysql.conn != nil {
        return nil
    }
    
    dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s", 
        mysql.username, mysql.password, mysql.host, mysql.port, mysql.database)
    
    conn, err := sql.Open("mysql", dsn)
    if err != nil {
        return err
    }
    
    mysql.conn = conn
    return nil
}

func (mysql *MySQLConnection) Disconnect() error {
    mysql.mutex.Lock()
    defer mysql.mutex.Unlock()
    
    if mysql.conn == nil {
        return nil
    }
    
    err := mysql.conn.Close()
    mysql.conn = nil
    return err
}

func (mysql *MySQLConnection) Query(sql string) (interface{}, error) {
    mysql.mutex.RLock()
    defer mysql.mutex.RUnlock()
    
    if mysql.conn == nil {
        return nil, errors.New("database not connected")
    }
    
    return mysql.conn.Query(sql)
}

func (mysql *MySQLConnection) Execute(sql string) error {
    mysql.mutex.RLock()
    defer mysql.mutex.RUnlock()
    
    if mysql.conn == nil {
        return errors.New("database not connected")
    }
    
    _, err := mysql.conn.Exec(sql)
    return err
}
```

### Singleton Registry
```go
// Singleton Registry
type SingletonRegistry struct {
    instances map[string]interface{}
    mutex     sync.RWMutex
}

var (
    registry *SingletonRegistry
    regOnce  sync.Once
)

func GetRegistry() *SingletonRegistry {
    regOnce.Do(func() {
        registry = &SingletonRegistry{
            instances: make(map[string]interface{}),
        }
    })
    return registry
}

func (sr *SingletonRegistry) Register(name string, instance interface{}) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    sr.instances[name] = instance
}

func (sr *SingletonRegistry) Get(name string) (interface{}, bool) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    instance, exists := sr.instances[name]
    return instance, exists
}

func (sr *SingletonRegistry) Unregister(name string) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    delete(sr.instances, name)
}
```

## 🎯 Razorpay-Specific Examples

### Payment Gateway Singleton
```go
// Payment Gateway Singleton for Razorpay
type PaymentGateway interface {
    ProcessPayment(amount int64, currency string, method string) (*PaymentResponse, error)
    RefundPayment(paymentID string, amount int64) (*RefundResponse, error)
    GetPaymentStatus(paymentID string) (*PaymentStatus, error)
}

type RazorpayGateway struct {
    apiKey    string
    apiSecret string
    baseURL   string
    client    *http.Client
    mutex     sync.RWMutex
}

type PaymentResponse struct {
    PaymentID string `json:"payment_id"`
    Status    string `json:"status"`
    Amount    int64  `json:"amount"`
    Currency  string `json:"currency"`
    Method    string `json:"method"`
}

type RefundResponse struct {
    RefundID  string `json:"refund_id"`
    PaymentID string `json:"payment_id"`
    Amount    int64  `json:"amount"`
    Status    string `json:"status"`
}

type PaymentStatus struct {
    PaymentID string `json:"payment_id"`
    Status    string `json:"status"`
    Amount    int64  `json:"amount"`
    Currency  string `json:"currency"`
}

var (
    gatewayInstance PaymentGateway
    gatewayOnce     sync.Once
)

func GetPaymentGateway() PaymentGateway {
    gatewayOnce.Do(func() {
        gatewayInstance = &RazorpayGateway{
            apiKey:    os.Getenv("RAZORPAY_API_KEY"),
            apiSecret: os.Getenv("RAZORPAY_API_SECRET"),
            baseURL:   "https://api.razorpay.com/v1",
            client:    &http.Client{Timeout: 30 * time.Second},
        }
    })
    return gatewayInstance
}

func (rg *RazorpayGateway) ProcessPayment(amount int64, currency string, method string) (*PaymentResponse, error) {
    rg.mutex.RLock()
    defer rg.mutex.RUnlock()
    
    // Implement payment processing logic
    // This is a simplified example
    paymentID := generatePaymentID()
    
    return &PaymentResponse{
        PaymentID: paymentID,
        Status:    "created",
        Amount:    amount,
        Currency:  currency,
        Method:    method,
    }, nil
}

func (rg *RazorpayGateway) RefundPayment(paymentID string, amount int64) (*RefundResponse, error) {
    rg.mutex.RLock()
    defer rg.mutex.RUnlock()
    
    // Implement refund logic
    refundID := generateRefundID()
    
    return &RefundResponse{
        RefundID:  refundID,
        PaymentID: paymentID,
        Amount:    amount,
        Status:    "processed",
    }, nil
}

func (rg *RazorpayGateway) GetPaymentStatus(paymentID string) (*PaymentStatus, error) {
    rg.mutex.RLock()
    defer rg.mutex.RUnlock()
    
    // Implement status check logic
    return &PaymentStatus{
        PaymentID: paymentID,
        Status:    "captured",
        Amount:    1000,
        Currency:  "INR",
    }, nil
}

func generatePaymentID() string {
    return "pay_" + generateRandomString(14)
}

func generateRefundID() string {
    return "rfnd_" + generateRandomString(14)
}

func generateRandomString(length int) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b := make([]byte, length)
    for i := range b {
        b[i] = charset[rand.Intn(len(charset))]
    }
    return string(b)
}
```

### Logger Singleton
```go
// Logger Singleton for Razorpay
type Logger interface {
    Info(message string, fields ...map[string]interface{})
    Error(message string, err error, fields ...map[string]interface{})
    Debug(message string, fields ...map[string]interface{})
    Warn(message string, fields ...map[string]interface{})
}

type RazorpayLogger struct {
    level    string
    output   io.Writer
    formatter func(level, message string, fields map[string]interface{}) string
    mutex    sync.RWMutex
}

var (
    loggerInstance Logger
    loggerOnce     sync.Once
)

func GetLogger() Logger {
    loggerOnce.Do(func() {
        loggerInstance = &RazorpayLogger{
            level:  os.Getenv("LOG_LEVEL"),
            output: os.Stdout,
            formatter: func(level, message string, fields map[string]interface{}) string {
                timestamp := time.Now().Format(time.RFC3339)
                fieldsStr := ""
                if len(fields) > 0 {
                    fieldsStr = fmt.Sprintf(" %+v", fields)
                }
                return fmt.Sprintf("[%s] %s: %s%s", timestamp, level, message, fieldsStr)
            },
        }
    })
    return loggerInstance
}

func (rl *RazorpayLogger) Info(message string, fields ...map[string]interface{}) {
    rl.log("INFO", message, fields...)
}

func (rl *RazorpayLogger) Error(message string, err error, fields ...map[string]interface{}) {
    if len(fields) == 0 {
        fields = []map[string]interface{}{}
    }
    fields[0]["error"] = err.Error()
    rl.log("ERROR", message, fields...)
}

func (rl *RazorpayLogger) Debug(message string, fields ...map[string]interface{}) {
    rl.log("DEBUG", message, fields...)
}

func (rl *RazorpayLogger) Warn(message string, fields ...map[string]interface{}) {
    rl.log("WARN", message, fields...)
}

func (rl *RazorpayLogger) log(level, message string, fields ...map[string]interface{}) {
    rl.mutex.RLock()
    defer rl.mutex.RUnlock()
    
    var fieldsMap map[string]interface{}
    if len(fields) > 0 {
        fieldsMap = fields[0]
    } else {
        fieldsMap = make(map[string]interface{})
    }
    
    logMessage := rl.formatter(level, message, fieldsMap)
    fmt.Fprintln(rl.output, logMessage)
}
```

## 🎯 Best Practices

### Design Principles
1. **Single Responsibility**: Each singleton should have one clear purpose
2. **Thread Safety**: Ensure thread-safe access to singleton instances
3. **Lazy Initialization**: Initialize only when needed
4. **Global Access**: Provide controlled global access

### Implementation Guidelines
1. **Use sync.Once**: For thread-safe lazy initialization
2. **Avoid Global Variables**: Use functions to access singletons
3. **Handle Errors**: Properly handle initialization errors
4. **Memory Management**: Be careful with memory leaks

### Common Pitfalls
1. **Thread Safety**: Not using proper synchronization
2. **Memory Leaks**: Not cleaning up resources
3. **Testing Issues**: Hard to test due to global state
4. **Hidden Dependencies**: Can create hidden dependencies

### When to Use
1. **Database Connections**: Single connection pool
2. **Loggers**: Single logger instance
3. **Configuration**: Single configuration object
4. **Cache Managers**: Single cache instance
5. **Service Locators**: Single service registry

### When NOT to Use
1. **When you need multiple instances**
2. **When testing is important**
3. **When you need dependency injection**
4. **When you need flexibility**

---

**Last Updated**: December 2024  
**Category**: Singleton Pattern  
**Complexity**: Intermediate Level
