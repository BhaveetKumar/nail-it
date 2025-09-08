# Singleton Pattern

## Pattern Name & Intent

**Singleton** - Ensure a class has only one instance and provide a global point of access to it.

The Singleton pattern ensures that a class has only one instance throughout the application lifecycle and provides a global access point to that instance. This is particularly useful for resources that should be shared across the application, such as configuration managers, database connections, or logging systems.

## When to Use

### Appropriate Scenarios
- **Configuration Management**: Global application configuration
- **Logging Systems**: Centralized logging across the application
- **Database Connections**: Single connection pool manager
- **Cache Managers**: Global cache instance
- **Service Locators**: Centralized service discovery

### When NOT to Use
- **Stateless Services**: When multiple instances are beneficial
- **High Concurrency**: When singleton becomes a bottleneck
- **Testing**: Makes unit testing more difficult
- **Dependency Injection**: Conflicts with DI frameworks

## Real-World Use Cases (Fintech/Payments)

### Payment Gateway Configuration
```go
// Global payment gateway configuration
type PaymentGatewayConfig struct {
    APIKey     string
    BaseURL    string
    Timeout    time.Duration
    RetryCount int
}

// Singleton for payment gateway configuration
type PaymentConfigManager struct {
    config *PaymentGatewayConfig
    mutex  sync.RWMutex
}

var (
    instance *PaymentConfigManager
    once     sync.Once
)

func GetPaymentConfigManager() *PaymentConfigManager {
    once.Do(func() {
        instance = &PaymentConfigManager{
            config: &PaymentGatewayConfig{
                APIKey:     os.Getenv("PAYMENT_API_KEY"),
                BaseURL:    os.Getenv("PAYMENT_BASE_URL"),
                Timeout:    30 * time.Second,
                RetryCount: 3,
            },
        }
    })
    return instance
}
```

### Audit Logger
```go
// Centralized audit logging for financial transactions
type AuditLogger struct {
    logger *log.Logger
    file   *os.File
    mutex  sync.Mutex
}

var auditLogger *AuditLogger
var auditOnce sync.Once

func GetAuditLogger() *AuditLogger {
    auditOnce.Do(func() {
        file, err := os.OpenFile("audit.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
        if err != nil {
            panic(err)
        }
        
        auditLogger = &AuditLogger{
            logger: log.New(file, "AUDIT: ", log.LstdFlags|log.Lshortfile),
            file:   file,
        }
    })
    return auditLogger
}

func (al *AuditLogger) LogTransaction(transactionID, userID, action string, amount float64) {
    al.mutex.Lock()
    defer al.mutex.Unlock()
    
    al.logger.Printf("Transaction: %s, User: %s, Action: %s, Amount: %.2f", 
        transactionID, userID, action, amount)
}
```

## Go Implementation

### Thread-Safe Singleton with sync.Once
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type DatabaseConnection struct {
    connectionString string
    connectedAt      time.Time
    isConnected      bool
}

type DatabaseManager struct {
    connection *DatabaseConnection
    mutex      sync.RWMutex
}

var (
    dbManager *DatabaseManager
    once      sync.Once
)

// GetDatabaseManager returns the singleton instance
func GetDatabaseManager() *DatabaseManager {
    once.Do(func() {
        dbManager = &DatabaseManager{
            connection: &DatabaseConnection{
                connectionString: "postgresql://localhost:5432/payments",
                connectedAt:      time.Now(),
                isConnected:      true,
            },
        }
    })
    return dbManager
}

func (dm *DatabaseManager) GetConnection() *DatabaseConnection {
    dm.mutex.RLock()
    defer dm.mutex.RUnlock()
    return dm.connection
}

func (dm *DatabaseManager) IsConnected() bool {
    dm.mutex.RLock()
    defer dm.mutex.RUnlock()
    return dm.connection.isConnected
}

func (dm *DatabaseManager) Reconnect() error {
    dm.mutex.Lock()
    defer dm.mutex.Unlock()
    
    // Simulate reconnection
    dm.connection.isConnected = true
    dm.connection.connectedAt = time.Now()
    return nil
}

// Usage example
func main() {
    // Get singleton instances
    db1 := GetDatabaseManager()
    db2 := GetDatabaseManager()
    
    // Both references point to the same instance
    fmt.Printf("Same instance: %t\n", db1 == db2)
    fmt.Printf("Connected: %t\n", db1.IsConnected())
}
```

### Configuration Manager Singleton
```go
type AppConfig struct {
    DatabaseURL    string
    RedisURL       string
    PaymentAPIKey  string
    LogLevel       string
    MaxConnections int
}

type ConfigManager struct {
    config *AppConfig
    mutex  sync.RWMutex
}

var (
    configManager *ConfigManager
    configOnce    sync.Once
)

func GetConfigManager() *ConfigManager {
    configOnce.Do(func() {
        configManager = &ConfigManager{
            config: &AppConfig{
                DatabaseURL:    getEnv("DATABASE_URL", "postgresql://localhost:5432/app"),
                RedisURL:       getEnv("REDIS_URL", "redis://localhost:6379"),
                PaymentAPIKey:  getEnv("PAYMENT_API_KEY", ""),
                LogLevel:       getEnv("LOG_LEVEL", "info"),
                MaxConnections: getEnvAsInt("MAX_CONNECTIONS", 100),
            },
        }
    })
    return configManager
}

func (cm *ConfigManager) GetDatabaseURL() string {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    return cm.config.DatabaseURL
}

func (cm *ConfigManager) GetPaymentAPIKey() string {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    return cm.config.PaymentAPIKey
}

func (cm *ConfigManager) UpdateConfig(newConfig *AppConfig) {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.config = newConfig
}

func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
    if value := os.Getenv(key); value != "" {
        if intValue, err := strconv.Atoi(value); err == nil {
            return intValue
        }
    }
    return defaultValue
}
```

## Variants & Trade-offs

### Variants

#### 1. Eager Initialization
```go
type EagerSingleton struct {
    data string
}

var eagerInstance = &EagerSingleton{data: "eager"}

func GetEagerSingleton() *EagerSingleton {
    return eagerInstance
}
```

**Pros**: Simple, thread-safe by default
**Cons**: Always initialized, even if not used

#### 2. Lazy Initialization with Mutex
```go
type LazySingleton struct {
    data string
}

var (
    lazyInstance *LazySingleton
    lazyMutex    sync.Mutex
)

func GetLazySingleton() *LazySingleton {
    if lazyInstance == nil {
        lazyMutex.Lock()
        defer lazyMutex.Unlock()
        if lazyInstance == nil {
            lazyInstance = &LazySingleton{data: "lazy"}
        }
    }
    return lazyInstance
}
```

**Pros**: Only initialized when needed
**Cons**: Double-checked locking complexity

#### 3. sync.Once (Recommended)
```go
type OnceSingleton struct {
    data string
}

var (
    onceInstance *OnceSingleton
    once         sync.Once
)

func GetOnceSingleton() *OnceSingleton {
    once.Do(func() {
        onceInstance = &OnceSingleton{data: "once"}
    })
    return onceInstance
}
```

**Pros**: Thread-safe, simple, efficient
**Cons**: None significant

### Trade-offs

| Aspect | Pros | Cons |
|--------|------|------|
| **Memory** | Single instance saves memory | Global state can grow large |
| **Performance** | Fast access, no object creation | Can become bottleneck |
| **Testing** | Consistent state across tests | Hard to mock, global state issues |
| **Concurrency** | Shared resource access | Requires synchronization |
| **Flexibility** | Simple global access | Hard to extend or modify |

## Testable Example

```go
package main

import (
    "testing"
    "time"
)

// Testable singleton with dependency injection
type CacheManager struct {
    cache map[string]interface{}
    mutex sync.RWMutex
}

type CacheManagerInterface interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{})
    Clear()
}

var (
    cacheManager CacheManagerInterface
    cacheOnce    sync.Once
)

func GetCacheManager() CacheManagerInterface {
    cacheOnce.Do(func() {
        cacheManager = &CacheManager{
            cache: make(map[string]interface{}),
        }
    })
    return cacheManager
}

// For testing, allow injection of mock
func SetCacheManager(cm CacheManagerInterface) {
    cacheManager = cm
}

func (cm *CacheManager) Get(key string) (interface{}, bool) {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    value, exists := cm.cache[key]
    return value, exists
}

func (cm *CacheManager) Set(key string, value interface{}) {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.cache[key] = value
}

func (cm *CacheManager) Clear() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.cache = make(map[string]interface{})
}

// Tests
func TestSingleton(t *testing.T) {
    // Test that we get the same instance
    cm1 := GetCacheManager()
    cm2 := GetCacheManager()
    
    if cm1 != cm2 {
        t.Error("Expected same instance")
    }
}

func TestCacheOperations(t *testing.T) {
    cm := GetCacheManager()
    
    // Test cache operations
    cm.Set("key1", "value1")
    
    value, exists := cm.Get("key1")
    if !exists {
        t.Error("Expected key to exist")
    }
    
    if value != "value1" {
        t.Errorf("Expected 'value1', got %v", value)
    }
}

func TestSingletonWithMock(t *testing.T) {
    // Create mock cache manager
    mockCache := &MockCacheManager{
        cache: make(map[string]interface{}),
    }
    
    // Inject mock
    SetCacheManager(mockCache)
    
    // Test with mock
    cm := GetCacheManager()
    cm.Set("test", "value")
    
    value, exists := cm.Get("test")
    if !exists || value != "value" {
        t.Error("Mock cache not working")
    }
}

type MockCacheManager struct {
    cache map[string]interface{}
}

func (m *MockCacheManager) Get(key string) (interface{}, bool) {
    value, exists := m.cache[key]
    return value, exists
}

func (m *MockCacheManager) Set(key string, value interface{}) {
    m.cache[key] = value
}

func (m *MockCacheManager) Clear() {
    m.cache = make(map[string]interface{})
}
```

## Integration Tips

### 1. With Dependency Injection
```go
// Instead of global singleton, use DI container
type Container struct {
    configManager *ConfigManager
    dbManager     *DatabaseManager
}

func NewContainer() *Container {
    return &Container{
        configManager: &ConfigManager{},
        dbManager:     &DatabaseManager{},
    }
}

func (c *Container) GetConfigManager() *ConfigManager {
    return c.configManager
}
```

### 2. With Context
```go
// Pass singleton through context
type contextKey string

const configManagerKey contextKey = "configManager"

func WithConfigManager(ctx context.Context, cm *ConfigManager) context.Context {
    return context.WithValue(ctx, configManagerKey, cm)
}

func GetConfigManagerFromContext(ctx context.Context) *ConfigManager {
    if cm, ok := ctx.Value(configManagerKey).(*ConfigManager); ok {
        return cm
    }
    return nil
}
```

### 3. With Interfaces
```go
// Use interfaces for better testability
type LoggerInterface interface {
    Log(level string, message string)
    Error(message string)
    Info(message string)
}

type SingletonLogger struct {
    logger *log.Logger
}

var (
    loggerInstance LoggerInterface
    loggerOnce     sync.Once
)

func GetLogger() LoggerInterface {
    loggerOnce.Do(func() {
        loggerInstance = &SingletonLogger{
            logger: log.New(os.Stdout, "", log.LstdFlags),
        }
    })
    return loggerInstance
}
```

## Common Interview Questions

### 1. How do you implement a thread-safe singleton in Go?
**Answer**: Use `sync.Once` for thread-safe lazy initialization. It ensures the initialization function runs only once, even in concurrent environments. Example: `once.Do(func() { instance = &Singleton{} })`

### 2. What are the drawbacks of the Singleton pattern?
**Answer**: Makes testing difficult due to global state, can become a bottleneck in concurrent systems, violates single responsibility principle, and makes code tightly coupled to the singleton instance.

### 3. How would you make a Singleton testable?
**Answer**: Use dependency injection, implement interfaces, or provide a way to reset the singleton instance in tests. Consider using a factory function that can return different implementations.

### 4. When should you avoid using Singleton?
**Answer**: Avoid when you need multiple instances, when testing is important, when you need dependency injection, or when the singleton might become a performance bottleneck.

### 5. How does Go's sync.Once work internally?
**Answer**: `sync.Once` uses atomic operations and a mutex to ensure the function is called exactly once. It's more efficient than double-checked locking and is the recommended approach for singletons in Go.
