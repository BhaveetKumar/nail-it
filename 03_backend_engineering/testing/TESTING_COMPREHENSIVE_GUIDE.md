---
# Auto-generated front matter
Title: Testing Comprehensive Guide
LastUpdated: 2025-11-06T20:45:58.296080
Tags: []
Status: draft
---

# 🧪 Testing Comprehensive Guide

> **Complete guide to testing strategies, patterns, and best practices for backend systems**

## 📚 Table of Contents

1. [Testing Fundamentals](#-testing-fundamentals)
2. [Unit Testing](#-unit-testing)
3. [Integration Testing](#-integration-testing)
4. [End-to-End Testing](#-end-to-end-testing)
5. [Performance Testing](#-performance-testing)
6. [Security Testing](#-security-testing)
7. [Test Automation](#-test-automation)
8. [Testing Patterns](#-testing-patterns)

---

## 🎯 Testing Fundamentals

### Testing Pyramid

```
┌─────────────────────────────────────────────────────────────┐
│                    Testing Pyramid                         │
├─────────────────────────────────────────────────────────────┤
│  E2E Tests (Few)                                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • User Journey Tests                                  │ │
│  │  • Full System Tests                                   │ │
│  │  • Cross-Browser Tests                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Integration Tests (Some)                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • API Tests                                           │ │
│  │  • Database Tests                                      │ │
│  │  • Service Tests                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Unit Tests (Many)                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • Function Tests                                      │ │
│  │  • Method Tests                                        │ │
│  │  • Component Tests                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Testing Principles

1. **Fast**: Tests should run quickly
2. **Independent**: Tests should not depend on each other
3. **Repeatable**: Tests should produce consistent results
4. **Self-Validating**: Tests should have clear pass/fail criteria
5. **Timely**: Tests should be written close to the code

---

## 🔬 Unit Testing

### 1. Go Unit Testing

```go
// User Service Unit Tests
package main

import (
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

func TestUserService_CreateUser(t *testing.T) {
    // Arrange
    mockRepo := &MockUserRepository{}
    userService := NewUserService(mockRepo)
    
    req := CreateUserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    expectedUser := &User{
        ID:        1,
        Name:      "John Doe",
        Email:     "john@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    mockRepo.On("Save", mock.AnythingOfType("*User")).Return(nil)
    mockRepo.On("FindByEmail", "john@example.com").Return(nil, ErrUserNotFound)
    
    // Act
    user, err := userService.CreateUser(context.Background(), req)
    
    // Assert
    assert.NoError(t, err)
    assert.Equal(t, expectedUser.Name, user.Name)
    assert.Equal(t, expectedUser.Email, user.Email)
    assert.Equal(t, expectedUser.Status, user.Status)
    mockRepo.AssertExpectations(t)
}

func TestUserService_CreateUser_EmailExists(t *testing.T) {
    // Arrange
    mockRepo := &MockUserRepository{}
    userService := NewUserService(mockRepo)
    
    req := CreateUserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    existingUser := &User{
        ID:    1,
        Email: "john@example.com",
    }
    
    mockRepo.On("FindByEmail", "john@example.com").Return(existingUser, nil)
    
    // Act
    user, err := userService.CreateUser(context.Background(), req)
    
    // Assert
    assert.Error(t, err)
    assert.Nil(t, user)
    assert.Contains(t, err.Error(), "email already exists")
    mockRepo.AssertExpectations(t)
}

func TestUserService_GetUser(t *testing.T) {
    // Arrange
    mockRepo := &MockUserRepository{}
    userService := NewUserService(mockRepo)
    
    expectedUser := &User{
        ID:    1,
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    mockRepo.On("FindByID", 1).Return(expectedUser, nil)
    
    // Act
    user, err := userService.GetUser(context.Background(), 1)
    
    // Assert
    assert.NoError(t, err)
    assert.Equal(t, expectedUser, user)
    mockRepo.AssertExpectations(t)
}

func TestUserService_GetUser_NotFound(t *testing.T) {
    // Arrange
    mockRepo := &MockUserRepository{}
    userService := NewUserService(mockRepo)
    
    mockRepo.On("FindByID", 999).Return(nil, ErrUserNotFound)
    
    // Act
    user, err := userService.GetUser(context.Background(), 999)
    
    // Assert
    assert.Error(t, err)
    assert.Nil(t, user)
    assert.Equal(t, ErrUserNotFound, err)
    mockRepo.AssertExpectations(t)
}

// Mock Repository
type MockUserRepository struct {
    mock.Mock
}

func (m *MockUserRepository) Save(ctx context.Context, user *User) error {
    args := m.Called(ctx, user)
    return args.Error(0)
}

func (m *MockUserRepository) FindByID(ctx context.Context, id int) (*User, error) {
    args := m.Called(ctx, id)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepository) FindByEmail(ctx context.Context, email string) (*User, error) {
    args := m.Called(ctx, email)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepository) Delete(ctx context.Context, id int) error {
    args := m.Called(ctx, id)
    return args.Error(0)
}
```

### 2. Test Fixtures and Data Builders

```go
// Test Data Builders
package main

import (
    "time"
)

type UserBuilder struct {
    user *User
}

func NewUserBuilder() *UserBuilder {
    return &UserBuilder{
        user: &User{
            ID:        1,
            Name:      "John Doe",
            Email:     "john@example.com",
            Status:    "active",
            CreatedAt: time.Now(),
            UpdatedAt: time.Now(),
        },
    }
}

func (ub *UserBuilder) WithID(id int) *UserBuilder {
    ub.user.ID = id
    return ub
}

func (ub *UserBuilder) WithName(name string) *UserBuilder {
    ub.user.Name = name
    return ub
}

func (ub *UserBuilder) WithEmail(email string) *UserBuilder {
    ub.user.Email = email
    return ub
}

func (ub *UserBuilder) WithStatus(status string) *UserBuilder {
    ub.user.Status = status
    return ub
}

func (ub *UserBuilder) Build() *User {
    return ub.user
}

// Test Fixtures
func TestUserService_WithFixtures(t *testing.T) {
    // Arrange
    mockRepo := &MockUserRepository{}
    userService := NewUserService(mockRepo)
    
    user := NewUserBuilder().
        WithName("Jane Doe").
        WithEmail("jane@example.com").
        WithStatus("active").
        Build()
    
    mockRepo.On("Save", mock.AnythingOfType("*User")).Return(nil)
    mockRepo.On("FindByEmail", "jane@example.com").Return(nil, ErrUserNotFound)
    
    // Act
    result, err := userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  user.Name,
        Email: user.Email,
    })
    
    // Assert
    assert.NoError(t, err)
    assert.Equal(t, user.Name, result.Name)
    assert.Equal(t, user.Email, result.Email)
    mockRepo.AssertExpectations(t)
}
```

### 3. Table-Driven Tests

```go
// Table-Driven Tests
func TestUserService_ValidateEmail(t *testing.T) {
    tests := []struct {
        name     string
        email    string
        expected bool
    }{
        {
            name:     "valid email",
            email:    "user@example.com",
            expected: true,
        },
        {
            name:     "invalid email - no @",
            email:    "userexample.com",
            expected: false,
        },
        {
            name:     "invalid email - no domain",
            email:    "user@",
            expected: false,
        },
        {
            name:     "invalid email - no local part",
            email:    "@example.com",
            expected: false,
        },
        {
            name:     "empty email",
            email:    "",
            expected: false,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := validateEmail(tt.email)
            assert.Equal(t, tt.expected, result)
        })
    }
}

func TestUserService_CalculateAge(t *testing.T) {
    tests := []struct {
        name     string
        birthDate time.Time
        expected int
    }{
        {
            name:      "25 years old",
            birthDate: time.Now().AddDate(-25, 0, 0),
            expected:  25,
        },
        {
            name:      "30 years old",
            birthDate: time.Now().AddDate(-30, 0, 0),
            expected:  30,
        },
        {
            name:      "born today",
            birthDate: time.Now(),
            expected:  0,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := calculateAge(tt.birthDate)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

---

## 🔗 Integration Testing

### 1. Database Integration Tests

```go
// Database Integration Tests
package main

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
    "gorm.io/driver/sqlite"
    "gorm.io/gorm"
)

type UserIntegrationTestSuite struct {
    suite.Suite
    db          *gorm.DB
    userService *UserService
    userRepo    *UserRepository
}

func (suite *UserIntegrationTestSuite) SetupSuite() {
    // Setup test database
    db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
    assert.NoError(suite.T(), err)
    
    // Auto migrate
    err = db.AutoMigrate(&User{})
    assert.NoError(suite.T(), err)
    
    suite.db = db
    suite.userRepo = NewUserRepository(db)
    suite.userService = NewUserService(suite.userRepo)
}

func (suite *UserIntegrationTestSuite) TearDownSuite() {
    sqlDB, _ := suite.db.DB()
    sqlDB.Close()
}

func (suite *UserIntegrationTestSuite) SetupTest() {
    // Clean database before each test
    suite.db.Exec("DELETE FROM users")
}

func (suite *UserIntegrationTestSuite) TestCreateUser() {
    // Arrange
    req := CreateUserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    // Act
    user, err := suite.userService.CreateUser(context.Background(), req)
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.NotNil(suite.T(), user)
    assert.Equal(suite.T(), req.Name, user.Name)
    assert.Equal(suite.T(), req.Email, user.Email)
    assert.Equal(suite.T(), "active", user.Status)
    
    // Verify in database
    var dbUser User
    err = suite.db.First(&dbUser, user.ID).Error
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), user.ID, dbUser.ID)
}

func (suite *UserIntegrationTestSuite) TestGetUser() {
    // Arrange
    user := &User{
        Name:      "Jane Doe",
        Email:     "jane@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    err := suite.db.Create(user).Error
    assert.NoError(suite.T(), err)
    
    // Act
    result, err := suite.userService.GetUser(context.Background(), user.ID)
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), user.ID, result.ID)
    assert.Equal(suite.T(), user.Name, result.Name)
    assert.Equal(suite.T(), user.Email, result.Email)
}

func (suite *UserIntegrationTestSuite) TestUpdateUser() {
    // Arrange
    user := &User{
        Name:      "Original Name",
        Email:     "original@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    err := suite.db.Create(user).Error
    assert.NoError(suite.T(), err)
    
    updateReq := UpdateUserRequest{
        Name: "Updated Name",
    }
    
    // Act
    result, err := suite.userService.UpdateUser(context.Background(), user.ID, updateReq)
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "Updated Name", result.Name)
    assert.Equal(suite.T(), user.Email, result.Email)
    
    // Verify in database
    var dbUser User
    err = suite.db.First(&dbUser, user.ID).Error
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "Updated Name", dbUser.Name)
}

func (suite *UserIntegrationTestSuite) TestDeleteUser() {
    // Arrange
    user := &User{
        Name:      "To Delete",
        Email:     "delete@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    err := suite.db.Create(user).Error
    assert.NoError(suite.T(), err)
    
    // Act
    err = suite.userService.DeleteUser(context.Background(), user.ID)
    
    // Assert
    assert.NoError(suite.T(), err)
    
    // Verify deletion
    var dbUser User
    err = suite.db.First(&dbUser, user.ID).Error
    assert.Error(suite.T(), err)
    assert.Equal(suite.T(), gorm.ErrRecordNotFound, err)
}

func TestUserIntegrationTestSuite(t *testing.T) {
    suite.Run(t, new(UserIntegrationTestSuite))
}
```

### 2. API Integration Tests

```go
// API Integration Tests
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
)

type APIIntegrationTestSuite struct {
    suite.Suite
    server *httptest.Server
    client *http.Client
}

func (suite *APIIntegrationTestSuite) SetupSuite() {
    // Setup test server
    userService := NewUserService(NewUserRepository())
    mux := http.NewServeMux()
    mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "GET":
            userService.GetUsers(w, r)
        case "POST":
            userService.CreateUser(w, r)
        }
    })
    mux.HandleFunc("/users/", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "GET":
            userService.GetUser(w, r)
        case "PUT":
            userService.UpdateUser(w, r)
        case "DELETE":
            userService.DeleteUser(w, r)
        }
    })
    
    suite.server = httptest.NewServer(mux)
    suite.client = &http.Client{}
}

func (suite *APIIntegrationTestSuite) TearDownSuite() {
    suite.server.Close()
}

func (suite *APIIntegrationTestSuite) TestCreateUserAPI() {
    // Arrange
    reqBody := CreateUserRequest{
        Name:  "API Test User",
        Email: "apitest@example.com",
    }
    
    jsonBody, _ := json.Marshal(reqBody)
    
    // Act
    resp, err := suite.client.Post(suite.server.URL+"/users", "application/json", bytes.NewBuffer(jsonBody))
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusCreated, resp.StatusCode)
    
    var user User
    err = json.NewDecoder(resp.Body).Decode(&user)
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), reqBody.Name, user.Name)
    assert.Equal(suite.T(), reqBody.Email, user.Email)
}

func (suite *APIIntegrationTestSuite) TestGetUsersAPI() {
    // Act
    resp, err := suite.client.Get(suite.server.URL + "/users")
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
    
    var response map[string]interface{}
    err = json.NewDecoder(resp.Body).Decode(&response)
    assert.NoError(suite.T(), err)
    assert.Contains(suite.T(), response, "data")
    assert.Contains(suite.T(), response, "pagination")
}

func (suite *APIIntegrationTestSuite) TestGetUserAPI() {
    // Arrange - Create user first
    reqBody := CreateUserRequest{
        Name:  "Get Test User",
        Email: "gettest@example.com",
    }
    
    jsonBody, _ := json.Marshal(reqBody)
    createResp, err := suite.client.Post(suite.server.URL+"/users", "application/json", bytes.NewBuffer(jsonBody))
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusCreated, createResp.StatusCode)
    
    var createdUser User
    err = json.NewDecoder(createResp.Body).Decode(&createdUser)
    assert.NoError(suite.T(), err)
    
    // Act
    resp, err := suite.client.Get(suite.server.URL + "/users/" + string(rune(createdUser.ID)))
    
    // Assert
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
    
    var user User
    err = json.NewDecoder(resp.Body).Decode(&user)
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), createdUser.ID, user.ID)
    assert.Equal(suite.T(), createdUser.Name, user.Name)
}

func TestAPIIntegrationTestSuite(t *testing.T) {
    suite.Run(t, new(APIIntegrationTestSuite))
}
```

---

## 🌐 End-to-End Testing

### 1. E2E Test Framework

```go
// End-to-End Test Framework
package main

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
)

type E2ETestSuite struct {
    suite.Suite
    testApp *TestApplication
}

type TestApplication struct {
    userService    *UserService
    orderService   *OrderService
    paymentService *PaymentService
    db            *gorm.DB
}

func (suite *E2ETestSuite) SetupSuite() {
    // Setup test application
    suite.testApp = &TestApplication{
        userService:    NewUserService(NewUserRepository()),
        orderService:   NewOrderService(NewOrderRepository()),
        paymentService: NewPaymentService(NewPaymentRepository()),
    }
}

func (suite *E2ETestSuite) TearDownSuite() {
    // Cleanup test application
    if suite.testApp.db != nil {
        sqlDB, _ := suite.testApp.db.DB()
        sqlDB.Close()
    }
}

func (suite *E2ETestSuite) TestCompleteUserJourney() {
    // Test complete user journey: Register -> Login -> Create Order -> Make Payment
    
    // Step 1: Register user
    user, err := suite.testApp.userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  "E2E Test User",
        Email: "e2etest@example.com",
    })
    assert.NoError(suite.T(), err)
    assert.NotNil(suite.T(), user)
    
    // Step 2: Create order
    order, err := suite.testApp.orderService.CreateOrder(context.Background(), CreateOrderRequest{
        UserID: user.ID,
        Items: []OrderItem{
            {ProductID: 1, Quantity: 2, Price: 10.0},
            {ProductID: 2, Quantity: 1, Price: 15.0},
        },
    })
    assert.NoError(suite.T(), err)
    assert.NotNil(suite.T(), order)
    
    // Step 3: Process payment
    payment, err := suite.testApp.paymentService.ProcessPayment(context.Background(), ProcessPaymentRequest{
        OrderID: order.ID,
        Amount:  order.TotalAmount,
        Method:  "credit_card",
    })
    assert.NoError(suite.T(), err)
    assert.NotNil(suite.T(), payment)
    assert.Equal(suite.T(), "completed", payment.Status)
    
    // Step 4: Verify order status updated
    updatedOrder, err := suite.testApp.orderService.GetOrder(context.Background(), order.ID)
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "paid", updatedOrder.Status)
}

func (suite *E2ETestSuite) TestUserRegistrationFlow() {
    // Test user registration with validation
    
    // Test valid registration
    user, err := suite.testApp.userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  "Valid User",
        Email: "valid@example.com",
    })
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "Valid User", user.Name)
    assert.Equal(suite.T(), "valid@example.com", user.Email)
    
    // Test duplicate email
    _, err = suite.testApp.userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  "Duplicate User",
        Email: "valid@example.com",
    })
    assert.Error(suite.T(), err)
    assert.Contains(suite.T(), err.Error(), "email already exists")
    
    // Test invalid email
    _, err = suite.testApp.userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  "Invalid User",
        Email: "invalid-email",
    })
    assert.Error(suite.T(), err)
    assert.Contains(suite.T(), err.Error(), "invalid email")
}

func (suite *E2ETestSuite) TestOrderProcessingFlow() {
    // Test complete order processing flow
    
    // Create user
    user, err := suite.testApp.userService.CreateUser(context.Background(), CreateUserRequest{
        Name:  "Order Test User",
        Email: "ordertest@example.com",
    })
    assert.NoError(suite.T(), err)
    
    // Create order
    order, err := suite.testApp.orderService.CreateOrder(context.Background(), CreateOrderRequest{
        UserID: user.ID,
        Items: []OrderItem{
            {ProductID: 1, Quantity: 2, Price: 10.0},
        },
    })
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "pending", order.Status)
    
    // Process payment
    payment, err := suite.testApp.paymentService.ProcessPayment(context.Background(), ProcessPaymentRequest{
        OrderID: order.ID,
        Amount:  order.TotalAmount,
        Method:  "credit_card",
    })
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "completed", payment.Status)
    
    // Verify order status
    updatedOrder, err := suite.testApp.orderService.GetOrder(context.Background(), order.ID)
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), "paid", updatedOrder.Status)
}

func TestE2ETestSuite(t *testing.T) {
    suite.Run(t, new(E2ETestSuite))
}
```

---

## ⚡ Performance Testing

### 1. Load Testing

```go
// Load Testing Framework
package main

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
)

type LoadTestConfig struct {
    Concurrency int
    Duration    time.Duration
    RampUp      time.Duration
    TargetRPS   int
}

type LoadTestResult struct {
    TotalRequests    int
    SuccessfulRequests int
    FailedRequests   int
    AverageLatency   time.Duration
    P95Latency       time.Duration
    P99Latency       time.Duration
    RPS              float64
    ErrorRate        float64
}

func TestUserAPI_LoadTest(t *testing.T) {
    // Setup test server
    server := setupTestServer()
    defer server.Close()
    
    config := LoadTestConfig{
        Concurrency: 10,
        Duration:    30 * time.Second,
        RampUp:      5 * time.Second,
        TargetRPS:   100,
    }
    
    result := runLoadTest(server.URL, config)
    
    // Assertions
    assert.Greater(t, result.TotalRequests, 0)
    assert.Less(t, result.ErrorRate, 0.01) // Less than 1% error rate
    assert.Less(t, result.AverageLatency, 100*time.Millisecond)
    assert.Greater(t, result.RPS, float64(config.TargetRPS)*0.8) // At least 80% of target RPS
}

func runLoadTest(baseURL string, config LoadTestConfig) LoadTestResult {
    var wg sync.WaitGroup
    var mutex sync.Mutex
    
    var totalRequests int
    var successfulRequests int
    var failedRequests int
    var latencies []time.Duration
    
    startTime := time.Now()
    
    // Start workers
    for i := 0; i < config.Concurrency; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            
            client := &http.Client{Timeout: 30 * time.Second}
            
            for time.Since(startTime) < config.Duration {
                // Ramp up
                if time.Since(startTime) < config.RampUp {
                    time.Sleep(time.Duration(workerID) * 100 * time.Millisecond)
                }
                
                // Rate limiting
                if config.TargetRPS > 0 {
                    time.Sleep(time.Second / time.Duration(config.TargetRPS/config.Concurrency))
                }
                
                // Make request
                start := time.Now()
                resp, err := client.Get(baseURL + "/users")
                latency := time.Since(start)
                
                mutex.Lock()
                totalRequests++
                if err != nil || resp.StatusCode >= 400 {
                    failedRequests++
                } else {
                    successfulRequests++
                }
                latencies = append(latencies, latency)
                mutex.Unlock()
                
                if resp != nil {
                    resp.Body.Close()
                }
            }
        }(i)
    }
    
    wg.Wait()
    
    // Calculate statistics
    result := LoadTestResult{
        TotalRequests:     totalRequests,
        SuccessfulRequests: successfulRequests,
        FailedRequests:    failedRequests,
        RPS:              float64(totalRequests) / config.Duration.Seconds(),
        ErrorRate:        float64(failedRequests) / float64(totalRequests),
    }
    
    if len(latencies) > 0 {
        // Sort latencies
        sort.Slice(latencies, func(i, j int) bool {
            return latencies[i] < latencies[j]
        })
        
        result.AverageLatency = latencies[len(latencies)/2]
        result.P95Latency = latencies[int(float64(len(latencies))*0.95)]
        result.P99Latency = latencies[int(float64(len(latencies))*0.99)]
    }
    
    return result
}
```

### 2. Stress Testing

```go
// Stress Testing
func TestUserAPI_StressTest(t *testing.T) {
    server := setupTestServer()
    defer server.Close()
    
    // Gradually increase load until system breaks
    concurrencyLevels := []int{1, 5, 10, 20, 50, 100}
    
    for _, concurrency := range concurrencyLevels {
        t.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(t *testing.T) {
            config := LoadTestConfig{
                Concurrency: concurrency,
                Duration:    10 * time.Second,
                RampUp:      2 * time.Second,
            }
            
            result := runLoadTest(server.URL, config)
            
            t.Logf("Concurrency: %d, RPS: %.2f, Error Rate: %.2f%%, Avg Latency: %v",
                concurrency, result.RPS, result.ErrorRate*100, result.AverageLatency)
            
            // System should handle up to 50 concurrent users
            if concurrency <= 50 {
                assert.Less(t, result.ErrorRate, 0.05) // Less than 5% error rate
            }
        })
    }
}
```

---

## 🔒 Security Testing

### 1. Authentication Testing

```go
// Security Testing
func TestAuthentication_Security(t *testing.T) {
    server := setupTestServer()
    defer server.Close()
    
    tests := []struct {
        name           string
        token          string
        expectedStatus int
    }{
        {
            name:           "valid token",
            token:          "valid-jwt-token",
            expectedStatus: http.StatusOK,
        },
        {
            name:           "invalid token",
            token:          "invalid-token",
            expectedStatus: http.StatusUnauthorized,
        },
        {
            name:           "expired token",
            token:          "expired-jwt-token",
            expectedStatus: http.StatusUnauthorized,
        },
        {
            name:           "no token",
            token:          "",
            expectedStatus: http.StatusUnauthorized,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            req, _ := http.NewRequest("GET", server.URL+"/users", nil)
            if tt.token != "" {
                req.Header.Set("Authorization", "Bearer "+tt.token)
            }
            
            resp, err := http.DefaultClient.Do(req)
            assert.NoError(t, err)
            assert.Equal(t, tt.expectedStatus, resp.StatusCode)
        })
    }
}
```

### 2. Input Validation Testing

```go
// Input Validation Testing
func TestInputValidation_Security(t *testing.T) {
    server := setupTestServer()
    defer server.Close()
    
    tests := []struct {
        name           string
        payload        string
        expectedStatus int
    }{
        {
            name:           "valid payload",
            payload:        `{"name": "John Doe", "email": "john@example.com"}`,
            expectedStatus: http.StatusCreated,
        },
        {
            name:           "SQL injection attempt",
            payload:        `{"name": "'; DROP TABLE users; --", "email": "john@example.com"}`,
            expectedStatus: http.StatusBadRequest,
        },
        {
            name:           "XSS attempt",
            payload:        `{"name": "<script>alert('xss')</script>", "email": "john@example.com"}`,
            expectedStatus: http.StatusBadRequest,
        },
        {
            name:           "oversized payload",
            payload:        `{"name": "` + string(make([]byte, 10000)) + `", "email": "john@example.com"}`,
            expectedStatus: http.StatusBadRequest,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            resp, err := http.Post(server.URL+"/users", "application/json", strings.NewReader(tt.payload))
            assert.NoError(t, err)
            assert.Equal(t, tt.expectedStatus, resp.StatusCode)
        })
    }
}
```

---

## 🤖 Test Automation

### 1. CI/CD Integration

```yaml
# GitHub Actions Workflow
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Go
      uses: actions/setup-go@v2
      with:
        go-version: 1.19
    
    - name: Install dependencies
      run: go mod download
    
    - name: Run unit tests
      run: go test -v -race -coverprofile=coverage.out ./...
    
    - name: Run integration tests
      run: go test -v -tags=integration ./...
    
    - name: Run load tests
      run: go test -v -tags=loadtest ./...
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.out
```

### 2. Test Data Management

```go
// Test Data Management
package main

import (
    "context"
    "testing"
    "time"
)

type TestDataManager struct {
    db *gorm.DB
}

func NewTestDataManager(db *gorm.DB) *TestDataManager {
    return &TestDataManager{db: db}
}

func (tdm *TestDataManager) CreateTestUser() *User {
    user := &User{
        Name:      "Test User",
        Email:     "test@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    tdm.db.Create(user)
    return user
}

func (tdm *TestDataManager) CreateTestOrder(userID int) *Order {
    order := &Order{
        UserID:      userID,
        TotalAmount: 100.0,
        Status:      "pending",
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }
    
    tdm.db.Create(order)
    return order
}

func (tdm *TestDataManager) Cleanup() {
    tdm.db.Exec("DELETE FROM orders")
    tdm.db.Exec("DELETE FROM users")
}

// Test with data management
func TestWithDataManagement(t *testing.T) {
    db := setupTestDB()
    tdm := NewTestDataManager(db)
    defer tdm.Cleanup()
    
    // Create test data
    user := tdm.CreateTestUser()
    order := tdm.CreateTestOrder(user.ID)
    
    // Run test
    assert.NotNil(t, user)
    assert.NotNil(t, order)
    assert.Equal(t, user.ID, order.UserID)
}
```

---

## 🎯 Testing Patterns

### 1. Test Doubles

```go
// Test Doubles
type TestDouble interface {
    Setup()
    Teardown()
}

type MockDatabase struct {
    users map[int]*User
    nextID int
}

func (m *MockDatabase) Setup() {
    m.users = make(map[int]*User)
    m.nextID = 1
}

func (m *MockDatabase) Teardown() {
    m.users = nil
}

func (m *MockDatabase) Save(user *User) error {
    user.ID = m.nextID
    m.users[m.nextID] = user
    m.nextID++
    return nil
}

func (m *MockDatabase) FindByID(id int) (*User, error) {
    user, exists := m.users[id]
    if !exists {
        return nil, ErrUserNotFound
    }
    return user, nil
}
```

### 2. Test Factories

```go
// Test Factories
type TestFactory struct{}

func (tf *TestFactory) CreateUser(overrides ...func(*User)) *User {
    user := &User{
        Name:      "Default User",
        Email:     "default@example.com",
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    for _, override := range overrides {
        override(user)
    }
    
    return user
}

func (tf *TestFactory) CreateOrder(overrides ...func(*Order)) *Order {
    order := &Order{
        UserID:      1,
        TotalAmount: 100.0,
        Status:      "pending",
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }
    
    for _, override := range overrides {
        override(order)
    }
    
    return order
}

// Usage
func TestWithFactory(t *testing.T) {
    factory := &TestFactory{}
    
    user := factory.CreateUser(func(u *User) {
        u.Name = "Custom User"
        u.Email = "custom@example.com"
    })
    
    order := factory.CreateOrder(func(o *Order) {
        o.UserID = user.ID
        o.TotalAmount = 200.0
    })
    
    assert.Equal(t, "Custom User", user.Name)
    assert.Equal(t, user.ID, order.UserID)
}
```

---

## 🎯 Best Practices Summary

### 1. Test Organization
- **Arrange-Act-Assert**: Structure tests clearly
- **Test Doubles**: Use mocks and stubs appropriately
- **Test Data**: Use builders and factories for test data
- **Test Isolation**: Ensure tests don't depend on each other

### 2. Test Coverage
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user journeys
- **Performance Tests**: Test under load and stress

### 3. Test Maintenance
- **Keep Tests Simple**: Avoid complex test logic
- **Update Tests**: Keep tests in sync with code changes
- **Remove Dead Tests**: Remove obsolete tests
- **Monitor Test Performance**: Keep test suite fast

### 4. Test Automation
- **CI/CD Integration**: Run tests automatically
- **Test Data Management**: Manage test data effectively
- **Test Reporting**: Generate meaningful test reports
- **Test Monitoring**: Monitor test health and performance

---

**🧪 Master these testing patterns to build reliable, maintainable, and high-quality backend systems! 🚀**
