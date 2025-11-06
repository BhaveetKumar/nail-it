---
# Auto-generated front matter
Title: Advanced Testing Guide
LastUpdated: 2025-11-06T20:45:58.278271
Tags: []
Status: draft
---

# 🧪 Advanced Testing Strategy Guide

> **Comprehensive testing approaches for modern backend engineering interviews**

## 🎯 **Overview**

Testing is a critical skill for senior backend engineers. This guide covers testing strategies, frameworks, and best practices that are essential for technical interviews and real-world development.

## 📚 **Table of Contents**

1. [Testing Pyramid & Strategy](#testing-pyramid--strategy)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [End-to-End Testing](#end-to-end-testing)
5. [API Testing](#api-testing)
6. [Database Testing](#database-testing)
7. [Performance Testing](#performance-testing)
8. [Security Testing](#security-testing)
9. [Test Automation & CI/CD](#test-automation--cicd)
10. [Testing Patterns & Best Practices](#testing-patterns--best-practices)
11. [Interview Questions](#interview-questions)

---

## 🏗️ **Testing Pyramid & Strategy**

### **The Testing Pyramid**

```
       /\
      /  \     E2E Tests (Few)
     /____\    
    /      \   Integration Tests (Some)
   /________\  
  /          \ Unit Tests (Many)
 /____________\
```

### **Testing Strategy Framework**

```go
// Testing levels and their purposes
type TestingLevel struct {
    Level       string
    Purpose     string
    Speed       string
    Coverage    string
    Maintenance string
}

var TestingLevels = []TestingLevel{
    {
        Level:       "Unit",
        Purpose:     "Test individual functions/methods",
        Speed:       "Fast (milliseconds)",
        Coverage:    "High code coverage",
        Maintenance: "Low",
    },
    {
        Level:       "Integration",
        Purpose:     "Test component interactions",
        Speed:       "Medium (seconds)",
        Coverage:    "Module interactions",
        Maintenance: "Medium",
    },
    {
        Level:       "End-to-End",
        Purpose:     "Test complete user workflows",
        Speed:       "Slow (minutes)",
        Coverage:    "Business scenarios",
        Maintenance: "High",
    },
}
```

### **Test Classification Matrix**

```go
// Test classification by scope and environment
type TestClassification struct {
    Name        string
    Scope       string
    Environment string
    Examples    []string
}

var TestTypes = []TestClassification{
    {
        Name:        "Unit Tests",
        Scope:       "Single function/method",
        Environment: "In-memory, mocked dependencies",
        Examples:    []string{"Business logic", "Calculations", "Validations"},
    },
    {
        Name:        "Component Tests",
        Scope:       "Single service/module",
        Environment: "Real dependencies within boundary",
        Examples:    []string{"API endpoints", "Database layer", "Message handlers"},
    },
    {
        Name:        "Contract Tests",
        Scope:       "Service boundaries",
        Environment: "Mocked external services",
        Examples:    []string{"API contracts", "Message schemas", "Database contracts"},
    },
    {
        Name:        "Integration Tests",
        Scope:       "Multiple services",
        Environment: "Real external dependencies",
        Examples:    []string{"Database integration", "External API calls", "Message queues"},
    },
    {
        Name:        "System Tests",
        Scope:       "Entire application",
        Environment: "Production-like environment",
        Examples:    []string{"Full workflows", "Performance tests", "Security tests"},
    },
}
```

---

## 🔬 **Unit Testing**

### **Go Unit Testing with Testify**

```go
package payment

import (
    "context"
    "errors"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/suite"
)

// Payment domain model
type Payment struct {
    ID          string
    UserID      string
    Amount      float64
    Currency    string
    Status      PaymentStatus
    CreatedAt   time.Time
    ProcessedAt *time.Time
}

type PaymentStatus string

const (
    StatusPending   PaymentStatus = "pending"
    StatusCompleted PaymentStatus = "completed"
    StatusFailed    PaymentStatus = "failed"
)

// Dependencies (interfaces for mocking)
type PaymentRepository interface {
    Save(ctx context.Context, payment *Payment) error
    FindByID(ctx context.Context, id string) (*Payment, error)
    FindByUserID(ctx context.Context, userID string) ([]*Payment, error)
}

type PaymentGateway interface {
    ProcessPayment(ctx context.Context, payment *Payment) error
}

type NotificationService interface {
    SendPaymentConfirmation(ctx context.Context, payment *Payment) error
}

// Service under test
type PaymentService struct {
    repo         PaymentRepository
    gateway      PaymentGateway
    notification NotificationService
}

func NewPaymentService(repo PaymentRepository, gateway PaymentGateway, notification NotificationService) *PaymentService {
    return &PaymentService{
        repo:         repo,
        gateway:      gateway,
        notification: notification,
    }
}

func (s *PaymentService) ProcessPayment(ctx context.Context, userID string, amount float64, currency string) (*Payment, error) {
    // Input validation
    if amount <= 0 {
        return nil, errors.New("amount must be positive")
    }
    
    if currency == "" {
        return nil, errors.New("currency is required")
    }

    // Create payment
    payment := &Payment{
        ID:        generateID(),
        UserID:    userID,
        Amount:    amount,
        Currency:  currency,
        Status:    StatusPending,
        CreatedAt: time.Now(),
    }

    // Save to repository
    if err := s.repo.Save(ctx, payment); err != nil {
        return nil, err
    }

    // Process through gateway
    if err := s.gateway.ProcessPayment(ctx, payment); err != nil {
        payment.Status = StatusFailed
        s.repo.Save(ctx, payment)
        return payment, err
    }

    // Update status
    payment.Status = StatusCompleted
    now := time.Now()
    payment.ProcessedAt = &now

    if err := s.repo.Save(ctx, payment); err != nil {
        return nil, err
    }

    // Send notification (async, don't block on errors)
    go func() {
        if err := s.notification.SendPaymentConfirmation(context.Background(), payment); err != nil {
            // Log error but don't fail the payment
            log.Printf("Failed to send notification: %v", err)
        }
    }()

    return payment, nil
}

func generateID() string {
    return fmt.Sprintf("pay_%d", time.Now().UnixNano())
}

// Mock implementations
type MockPaymentRepository struct {
    mock.Mock
}

func (m *MockPaymentRepository) Save(ctx context.Context, payment *Payment) error {
    args := m.Called(ctx, payment)
    return args.Error(0)
}

func (m *MockPaymentRepository) FindByID(ctx context.Context, id string) (*Payment, error) {
    args := m.Called(ctx, id)
    return args.Get(0).(*Payment), args.Error(1)
}

func (m *MockPaymentRepository) FindByUserID(ctx context.Context, userID string) ([]*Payment, error) {
    args := m.Called(ctx, userID)
    return args.Get(0).([]*Payment), args.Error(1)
}

type MockPaymentGateway struct {
    mock.Mock
}

func (m *MockPaymentGateway) ProcessPayment(ctx context.Context, payment *Payment) error {
    args := m.Called(ctx, payment)
    return args.Error(0)
}

type MockNotificationService struct {
    mock.Mock
}

func (m *MockNotificationService) SendPaymentConfirmation(ctx context.Context, payment *Payment) error {
    args := m.Called(ctx, payment)
    return args.Error(0)
}

// Test Suite using testify/suite
type PaymentServiceTestSuite struct {
    suite.Suite
    service      *PaymentService
    mockRepo     *MockPaymentRepository
    mockGateway  *MockPaymentGateway
    mockNotifier *MockNotificationService
}

func (suite *PaymentServiceTestSuite) SetupTest() {
    suite.mockRepo = new(MockPaymentRepository)
    suite.mockGateway = new(MockPaymentGateway)
    suite.mockNotifier = new(MockNotificationService)
    suite.service = NewPaymentService(suite.mockRepo, suite.mockGateway, suite.mockNotifier)
}

func (suite *PaymentServiceTestSuite) TearDownTest() {
    suite.mockRepo.AssertExpectations(suite.T())
    suite.mockGateway.AssertExpectations(suite.T())
    suite.mockNotifier.AssertExpectations(suite.T())
}

// Test successful payment processing
func (suite *PaymentServiceTestSuite) TestProcessPayment_Success() {
    // Arrange
    ctx := context.Background()
    userID := "user_123"
    amount := 100.0
    currency := "USD"

    suite.mockRepo.On("Save", ctx, mock.AnythingOfType("*payment.Payment")).Return(nil).Twice()
    suite.mockGateway.On("ProcessPayment", ctx, mock.AnythingOfType("*payment.Payment")).Return(nil)

    // Act
    result, err := suite.service.ProcessPayment(ctx, userID, amount, currency)

    // Assert
    require.NoError(suite.T(), err)
    assert.NotNil(suite.T(), result)
    assert.Equal(suite.T(), userID, result.UserID)
    assert.Equal(suite.T(), amount, result.Amount)
    assert.Equal(suite.T(), currency, result.Currency)
    assert.Equal(suite.T(), StatusCompleted, result.Status)
    assert.NotNil(suite.T(), result.ProcessedAt)
}

// Test input validation
func (suite *PaymentServiceTestSuite) TestProcessPayment_InvalidAmount() {
    // Arrange
    ctx := context.Background()
    userID := "user_123"
    amount := -100.0
    currency := "USD"

    // Act
    result, err := suite.service.ProcessPayment(ctx, userID, amount, currency)

    // Assert
    assert.Error(suite.T(), err)
    assert.Nil(suite.T(), result)
    assert.Contains(suite.T(), err.Error(), "amount must be positive")
}

// Test gateway failure
func (suite *PaymentServiceTestSuite) TestProcessPayment_GatewayFailure() {
    // Arrange
    ctx := context.Background()
    userID := "user_123"
    amount := 100.0
    currency := "USD"
    gatewayError := errors.New("gateway timeout")

    suite.mockRepo.On("Save", ctx, mock.AnythingOfType("*payment.Payment")).Return(nil).Twice()
    suite.mockGateway.On("ProcessPayment", ctx, mock.AnythingOfType("*payment.Payment")).Return(gatewayError)

    // Act
    result, err := suite.service.ProcessPayment(ctx, userID, amount, currency)

    // Assert
    assert.Error(suite.T(), err)
    assert.Equal(suite.T(), gatewayError, err)
    assert.NotNil(suite.T(), result)
    assert.Equal(suite.T(), StatusFailed, result.Status)
}

// Test repository failure
func (suite *PaymentServiceTestSuite) TestProcessPayment_RepositoryFailure() {
    // Arrange
    ctx := context.Background()
    userID := "user_123"
    amount := 100.0
    currency := "USD"
    repoError := errors.New("database connection failed")

    suite.mockRepo.On("Save", ctx, mock.AnythingOfType("*payment.Payment")).Return(repoError)

    // Act
    result, err := suite.service.ProcessPayment(ctx, userID, amount, currency)

    // Assert
    assert.Error(suite.T(), err)
    assert.Equal(suite.T(), repoError, err)
    assert.Nil(suite.T(), result)
}

// Table-driven tests
func (suite *PaymentServiceTestSuite) TestProcessPayment_TableDriven() {
    testCases := []struct {
        name           string
        userID         string
        amount         float64
        currency       string
        repoError      error
        gatewayError   error
        expectedStatus PaymentStatus
        expectedError  string
    }{
        {
            name:           "Valid payment",
            userID:         "user_123",
            amount:         100.0,
            currency:       "USD",
            expectedStatus: StatusCompleted,
        },
        {
            name:          "Invalid amount",
            userID:        "user_123",
            amount:        -100.0,
            currency:      "USD",
            expectedError: "amount must be positive",
        },
        {
            name:          "Empty currency",
            userID:        "user_123",
            amount:        100.0,
            currency:      "",
            expectedError: "currency is required",
        },
        {
            name:           "Gateway failure",
            userID:         "user_123",
            amount:         100.0,
            currency:       "USD",
            gatewayError:   errors.New("gateway error"),
            expectedStatus: StatusFailed,
            expectedError:  "gateway error",
        },
    }

    for _, tc := range testCases {
        suite.Run(tc.name, func() {
            // Setup mocks based on test case
            if tc.expectedError == "" || tc.gatewayError != nil {
                suite.mockRepo.On("Save", mock.Anything, mock.Anything).Return(tc.repoError).Maybe()
                suite.mockGateway.On("ProcessPayment", mock.Anything, mock.Anything).Return(tc.gatewayError).Maybe()
            }

            // Act
            result, err := suite.service.ProcessPayment(context.Background(), tc.userID, tc.amount, tc.currency)

            // Assert
            if tc.expectedError != "" {
                assert.Error(suite.T(), err)
                assert.Contains(suite.T(), err.Error(), tc.expectedError)
            } else {
                assert.NoError(suite.T(), err)
                assert.Equal(suite.T(), tc.expectedStatus, result.Status)
            }

            // Clear mock expectations for next test
            suite.mockRepo.ExpectedCalls = nil
            suite.mockGateway.ExpectedCalls = nil
        })
    }
}

// Benchmark test
func BenchmarkPaymentService_ProcessPayment(b *testing.B) {
    mockRepo := new(MockPaymentRepository)
    mockGateway := new(MockPaymentGateway)
    mockNotifier := new(MockNotificationService)
    service := NewPaymentService(mockRepo, mockGateway, mockNotifier)

    mockRepo.On("Save", mock.Anything, mock.Anything).Return(nil)
    mockGateway.On("ProcessPayment", mock.Anything, mock.Anything).Return(nil)

    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        _, err := service.ProcessPayment(context.Background(), "user_123", 100.0, "USD")
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Run the test suite
func TestPaymentServiceSuite(t *testing.T) {
    suite.Run(t, new(PaymentServiceTestSuite))
}
```

### **Advanced Unit Testing Patterns**

```go
// Property-based testing with gopter
import "github.com/leanovate/gopter"

func TestPaymentValidation_PropertyBased(t *testing.T) {
    properties := gopter.NewProperties(nil)

    properties.Property("Amount validation", func(amount float64) bool {
        payment := &Payment{Amount: amount}
        err := validatePayment(payment)
        
        if amount <= 0 {
            return err != nil
        }
        return err == nil
    })

    properties.TestingRun(t)
}

// Test doubles pattern
type PaymentServiceTestDouble struct {
    ProcessPaymentFunc func(ctx context.Context, userID string, amount float64, currency string) (*Payment, error)
    CallCount          int
}

func (td *PaymentServiceTestDouble) ProcessPayment(ctx context.Context, userID string, amount float64, currency string) (*Payment, error) {
    td.CallCount++
    if td.ProcessPaymentFunc != nil {
        return td.ProcessPaymentFunc(ctx, userID, amount, currency)
    }
    return nil, nil
}

// Dependency injection for testing
type PaymentServiceDependencies struct {
    Repository   PaymentRepository
    Gateway      PaymentGateway
    Notification NotificationService
    Clock        Clock // For time testing
    IDGenerator  IDGenerator // For deterministic IDs
}

type Clock interface {
    Now() time.Time
}

type RealClock struct{}

func (c RealClock) Now() time.Time {
    return time.Now()
}

type MockClock struct {
    CurrentTime time.Time
}

func (c *MockClock) Now() time.Time {
    return c.CurrentTime
}

// Test with time control
func TestPaymentService_WithTimeControl(t *testing.T) {
    // Arrange
    fixedTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
    mockClock := &MockClock{CurrentTime: fixedTime}
    
    service := NewPaymentServiceWithDependencies(PaymentServiceDependencies{
        Repository: &MockPaymentRepository{},
        Clock:      mockClock,
    })

    // Act
    payment, _ := service.ProcessPayment(context.Background(), "user_123", 100.0, "USD")

    // Assert
    assert.Equal(t, fixedTime, payment.CreatedAt)
}
```

---

## 🔗 **Integration Testing**

### **Database Integration Testing**

```go
package integration

import (
    "context"
    "database/sql"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/wait"
)

// Database integration test setup
type DatabaseTestSuite struct {
    suite.Suite
    db        *sql.DB
    container testcontainers.Container
    repo      *PaymentRepository
}

func (suite *DatabaseTestSuite) SetupSuite() {
    ctx := context.Background()
    
    // Start PostgreSQL container
    req := testcontainers.ContainerRequest{
        Image:        "postgres:13",
        ExposedPorts: []string{"5432/tcp"},
        Env: map[string]string{
            "POSTGRES_DB":       "testdb",
            "POSTGRES_USER":     "testuser",
            "POSTGRES_PASSWORD": "testpass",
        },
        WaitingFor: wait.ForListeningPort("5432/tcp"),
    }

    container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: req,
        Started:          true,
    })
    require.NoError(suite.T(), err)
    suite.container = container

    // Get connection details
    host, err := container.Host(ctx)
    require.NoError(suite.T(), err)
    
    port, err := container.MappedPort(ctx, "5432")
    require.NoError(suite.T(), err)

    // Connect to database
    dsn := fmt.Sprintf("postgres://testuser:testpass@%s:%s/testdb?sslmode=disable", host, port.Port())
    db, err := sql.Open("postgres", dsn)
    require.NoError(suite.T(), err)
    suite.db = db

    // Run migrations
    err = suite.runMigrations()
    require.NoError(suite.T(), err)

    // Initialize repository
    suite.repo = NewPaymentRepository(db)
}

func (suite *DatabaseTestSuite) TearDownSuite() {
    if suite.db != nil {
        suite.db.Close()
    }
    if suite.container != nil {
        suite.container.Terminate(context.Background())
    }
}

func (suite *DatabaseTestSuite) SetupTest() {
    // Clean database before each test
    _, err := suite.db.Exec("TRUNCATE TABLE payments")
    require.NoError(suite.T(), err)
}

func (suite *DatabaseTestSuite) runMigrations() error {
    schema := `
    CREATE TABLE IF NOT EXISTS payments (
        id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        currency VARCHAR(3) NOT NULL,
        status VARCHAR(50) NOT NULL,
        created_at TIMESTAMP NOT NULL,
        processed_at TIMESTAMP
    );
    
    CREATE INDEX idx_payments_user_id ON payments(user_id);
    CREATE INDEX idx_payments_status ON payments(status);
    `
    
    _, err := suite.db.Exec(schema)
    return err
}

func (suite *DatabaseTestSuite) TestSaveAndFindPayment() {
    // Arrange
    ctx := context.Background()
    payment := &Payment{
        ID:        "pay_123",
        UserID:    "user_123",
        Amount:    100.50,
        Currency:  "USD",
        Status:    StatusPending,
        CreatedAt: time.Now(),
    }

    // Act - Save
    err := suite.repo.Save(ctx, payment)
    require.NoError(suite.T(), err)

    // Act - Find
    found, err := suite.repo.FindByID(ctx, payment.ID)
    require.NoError(suite.T(), err)

    // Assert
    assert.Equal(suite.T(), payment.ID, found.ID)
    assert.Equal(suite.T(), payment.UserID, found.UserID)
    assert.Equal(suite.T(), payment.Amount, found.Amount)
    assert.Equal(suite.T(), payment.Currency, found.Currency)
    assert.Equal(suite.T(), payment.Status, found.Status)
}

func (suite *DatabaseTestSuite) TestFindByUserID() {
    // Arrange
    ctx := context.Background()
    userID := "user_123"
    payments := []*Payment{
        {ID: "pay_1", UserID: userID, Amount: 100, Currency: "USD", Status: StatusCompleted, CreatedAt: time.Now()},
        {ID: "pay_2", UserID: userID, Amount: 200, Currency: "USD", Status: StatusPending, CreatedAt: time.Now()},
        {ID: "pay_3", UserID: "user_456", Amount: 300, Currency: "USD", Status: StatusCompleted, CreatedAt: time.Now()},
    }

    for _, payment := range payments {
        err := suite.repo.Save(ctx, payment)
        require.NoError(suite.T(), err)
    }

    // Act
    userPayments, err := suite.repo.FindByUserID(ctx, userID)
    require.NoError(suite.T(), err)

    // Assert
    assert.Len(suite.T(), userPayments, 2)
    for _, payment := range userPayments {
        assert.Equal(suite.T(), userID, payment.UserID)
    }
}

func (suite *DatabaseTestSuite) TestTransactionRollback() {
    // Arrange
    ctx := context.Background()
    payment := &Payment{
        ID:       "pay_123",
        UserID:   "user_123",
        Amount:   100.50,
        Currency: "USD",
        Status:   StatusPending,
        CreatedAt: time.Now(),
    }

    // Act - Start transaction and rollback
    tx, err := suite.db.BeginTx(ctx, nil)
    require.NoError(suite.T(), err)

    repoWithTx := NewPaymentRepositoryWithTx(tx)
    err = repoWithTx.Save(ctx, payment)
    require.NoError(suite.T(), err)

    // Rollback transaction
    tx.Rollback()

    // Assert - Payment should not exist
    found, err := suite.repo.FindByID(ctx, payment.ID)
    assert.Error(suite.T(), err)
    assert.Nil(suite.T(), found)
}

func TestDatabaseIntegration(t *testing.T) {
    suite.Run(t, new(DatabaseTestSuite))
}
```

### **API Integration Testing**

```go
package integration

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    
    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// API integration test setup
type APITestSuite struct {
    suite.Suite
    router  *gin.Engine
    server  *httptest.Server
    service *PaymentService
}

func (suite *APITestSuite) SetupTest() {
    // Setup test dependencies
    mockRepo := new(MockPaymentRepository)
    mockGateway := new(MockPaymentGateway)
    mockNotifier := new(MockNotificationService)
    
    suite.service = NewPaymentService(mockRepo, mockGateway, mockNotifier)
    
    // Setup router
    gin.SetMode(gin.TestMode)
    suite.router = gin.New()
    
    // Setup routes
    api := suite.router.Group("/api/v1")
    {
        api.POST("/payments", suite.createPayment)
        api.GET("/payments/:id", suite.getPayment)
        api.GET("/users/:userId/payments", suite.getUserPayments)
    }
    
    suite.server = httptest.NewServer(suite.router)
}

func (suite *APITestSuite) TearDownTest() {
    if suite.server != nil {
        suite.server.Close()
    }
}

func (suite *APITestSuite) createPayment(c *gin.Context) {
    var req CreatePaymentRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    payment, err := suite.service.ProcessPayment(c.Request.Context(), req.UserID, req.Amount, req.Currency)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }

    c.JSON(http.StatusCreated, payment)
}

func (suite *APITestSuite) getPayment(c *gin.Context) {
    // Implementation
}

func (suite *APITestSuite) getUserPayments(c *gin.Context) {
    // Implementation
}

type CreatePaymentRequest struct {
    UserID   string  `json:"user_id" binding:"required"`
    Amount   float64 `json:"amount" binding:"required,gt=0"`
    Currency string  `json:"currency" binding:"required,len=3"`
}

func (suite *APITestSuite) TestCreatePayment_Success() {
    // Arrange
    requestBody := CreatePaymentRequest{
        UserID:   "user_123",
        Amount:   100.50,
        Currency: "USD",
    }
    
    jsonBody, err := json.Marshal(requestBody)
    require.NoError(suite.T(), err)

    // Act
    resp, err := http.Post(suite.server.URL+"/api/v1/payments", "application/json", bytes.NewBuffer(jsonBody))
    require.NoError(suite.T(), err)
    defer resp.Body.Close()

    // Assert
    assert.Equal(suite.T(), http.StatusCreated, resp.StatusCode)
    
    var payment Payment
    err = json.NewDecoder(resp.Body).Decode(&payment)
    require.NoError(suite.T(), err)
    
    assert.Equal(suite.T(), requestBody.UserID, payment.UserID)
    assert.Equal(suite.T(), requestBody.Amount, payment.Amount)
    assert.Equal(suite.T(), requestBody.Currency, payment.Currency)
}

func (suite *APITestSuite) TestCreatePayment_ValidationError() {
    // Arrange
    requestBody := CreatePaymentRequest{
        UserID:   "",
        Amount:   -100,
        Currency: "INVALID",
    }
    
    jsonBody, err := json.Marshal(requestBody)
    require.NoError(suite.T(), err)

    // Act
    resp, err := http.Post(suite.server.URL+"/api/v1/payments", "application/json", bytes.NewBuffer(jsonBody))
    require.NoError(suite.T(), err)
    defer resp.Body.Close()

    // Assert
    assert.Equal(suite.T(), http.StatusBadRequest, resp.StatusCode)
}

func TestAPIIntegration(t *testing.T) {
    suite.Run(t, new(APITestSuite))
}
```

---

## 🌐 **End-to-End Testing**

### **E2E Testing with Playwright (Go)**

```go
package e2e

import (
    "context"
    "testing"
    
    "github.com/playwright-community/playwright-go"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

type E2ETestSuite struct {
    suite.Suite
    playwright *playwright.Playwright
    browser    playwright.Browser
    context    playwright.BrowserContext
    page       playwright.Page
}

func (suite *E2ETestSuite) SetupSuite() {
    // Start Playwright
    pw, err := playwright.Run()
    require.NoError(suite.T(), err)
    suite.playwright = pw

    // Launch browser
    browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{
        Headless: playwright.Bool(true),
    })
    require.NoError(suite.T(), err)
    suite.browser = browser
}

func (suite *E2ETestSuite) SetupTest() {
    // Create new context for each test
    context, err := suite.browser.NewContext()
    require.NoError(suite.T(), err)
    suite.context = context

    // Create new page
    page, err := context.NewPage()
    require.NoError(suite.T(), err)
    suite.page = page
}

func (suite *E2ETestSuite) TearDownTest() {
    if suite.context != nil {
        suite.context.Close()
    }
}

func (suite *E2ETestSuite) TearDownSuite() {
    if suite.browser != nil {
        suite.browser.Close()
    }
    if suite.playwright != nil {
        suite.playwright.Stop()
    }
}

func (suite *E2ETestSuite) TestPaymentFlow_Success() {
    // Navigate to payment page
    _, err := suite.page.Goto("http://localhost:3000/payment")
    require.NoError(suite.T(), err)

    // Fill payment form
    err = suite.page.Fill("#amount", "100.50")
    require.NoError(suite.T(), err)
    
    err = suite.page.SelectOption("#currency", "USD")
    require.NoError(suite.T(), err)
    
    err = suite.page.Fill("#user-id", "user_123")
    require.NoError(suite.T(), err)

    // Submit payment
    err = suite.page.Click("#submit-payment")
    require.NoError(suite.T(), err)

    // Wait for success message
    err = suite.page.WaitForSelector("#payment-success", playwright.PageWaitForSelectorOptions{
        Timeout: playwright.Float(5000),
    })
    require.NoError(suite.T(), err)

    // Verify success message
    text, err := suite.page.TextContent("#payment-success")
    require.NoError(suite.T(), err)
    assert.Contains(suite.T(), text, "Payment successful")

    // Verify payment ID is displayed
    paymentID, err := suite.page.GetAttribute("#payment-id", "data-payment-id")
    require.NoError(suite.T(), err)
    assert.NotEmpty(suite.T(), paymentID)
}

func (suite *E2ETestSuite) TestPaymentFlow_ValidationError() {
    // Navigate to payment page
    _, err := suite.page.Goto("http://localhost:3000/payment")
    require.NoError(suite.T(), err)

    // Submit without filling form
    err = suite.page.Click("#submit-payment")
    require.NoError(suite.T(), err)

    // Wait for error message
    err = suite.page.WaitForSelector("#validation-error")
    require.NoError(suite.T(), err)

    // Verify error message
    text, err := suite.page.TextContent("#validation-error")
    require.NoError(suite.T(), err)
    assert.Contains(suite.T(), text, "Amount is required")
}

func (suite *E2ETestSuite) TestPaymentHistory() {
    // Login first
    suite.login("user_123")

    // Navigate to payment history
    _, err := suite.page.Goto("http://localhost:3000/payments")
    require.NoError(suite.T(), err)

    // Wait for payments to load
    err = suite.page.WaitForSelector("#payment-list")
    require.NoError(suite.T(), err)

    // Count payment items
    count, err := suite.page.Locator(".payment-item").Count()
    require.NoError(suite.T(), err)
    assert.Greater(suite.T(), count, 0)

    // Check first payment details
    firstPayment := suite.page.Locator(".payment-item").First()
    amount, err := firstPayment.Locator(".amount").TextContent()
    require.NoError(suite.T(), err)
    assert.NotEmpty(suite.T(), amount)
}

func (suite *E2ETestSuite) login(userID string) {
    _, err := suite.page.Goto("http://localhost:3000/login")
    require.NoError(suite.T(), err)
    
    err = suite.page.Fill("#user-id", userID)
    require.NoError(suite.T(), err)
    
    err = suite.page.Click("#login-button")
    require.NoError(suite.T(), err)
    
    err = suite.page.WaitForSelector("#dashboard")
    require.NoError(suite.T(), err)
}

func TestE2E(t *testing.T) {
    suite.Run(t, new(E2ETestSuite))
}
```

### **Contract Testing with Pact**

```go
package contract

import (
    "fmt"
    "net/http"
    "testing"
    
    "github.com/pact-foundation/pact-go/dsl"
    "github.com/pact-foundation/pact-go/types"
    "github.com/stretchr/testify/assert"
)

// Consumer test (Payment Service consuming User Service)
func TestUserServiceContract(t *testing.T) {
    pact := dsl.Pact{
        Consumer: "payment-service",
        Provider: "user-service",
        Host:     "localhost",
    }
    defer pact.Teardown()

    // Define expected interaction
    pact.
        AddInteraction().
        Given("User exists").
        UponReceiving("A request for user details").
        WithRequest(dsl.Request{
            Method: http.MethodGet,
            Path:   dsl.String("/api/users/user_123"),
            Headers: dsl.MapMatcher{
                "Content-Type": dsl.String("application/json"),
            },
        }).
        WillRespondWith(dsl.Response{
            Status: http.StatusOK,
            Headers: dsl.MapMatcher{
                "Content-Type": dsl.String("application/json"),
            },
            Body: dsl.Match(map[string]interface{}{
                "id":    dsl.String("user_123"),
                "name":  dsl.String("John Doe"),
                "email": dsl.String("john@example.com"),
            }),
        })

    // Start mock server
    err := pact.Start()
    assert.NoError(t, err)

    // Test consumer code
    userService := NewUserServiceClient(fmt.Sprintf("http://localhost:%d", pact.Server.Port))
    user, err := userService.GetUser("user_123")
    
    assert.NoError(t, err)
    assert.Equal(t, "user_123", user.ID)
    assert.Equal(t, "John Doe", user.Name)
    assert.Equal(t, "john@example.com", user.Email)

    // Verify interactions
    err = pact.Verify()
    assert.NoError(t, err)

    // Publish contract
    err = pact.Publish(types.PublishRequest{
        PactURLs:        []string{pact.PactFile()},
        PactBroker:      "http://localhost:9292",
        ConsumerVersion: "1.0.0",
        Tags:            []string{"main"},
    })
    assert.NoError(t, err)
}
```

---

## 🚀 **Performance Testing**

### **Load Testing with Vegeta**

```go
package performance

import (
    "fmt"
    "net/http"
    "testing"
    "time"
    
    vegeta "github.com/tsenart/vegeta/v12/lib"
    "github.com/stretchr/testify/assert"
)

func TestPaymentAPI_LoadTest(t *testing.T) {
    // Define target
    target := vegeta.Target{
        Method: "POST",
        URL:    "http://localhost:8080/api/v1/payments",
        Header: http.Header{
            "Content-Type": []string{"application/json"},
        },
        Body: []byte(`{
            "user_id": "user_123",
            "amount": 100.50,
            "currency": "USD"
        }`),
    }

    // Create targeter
    targeter := vegeta.NewStaticTargeter(target)
    
    // Configure attack
    rate := vegeta.Rate{Freq: 100, Per: time.Second} // 100 RPS
    duration := 30 * time.Second
    attacker := vegeta.NewAttacker()

    // Execute attack
    var results vegeta.Metrics
    for res := range attacker.Attack(targeter, rate, duration, "Load Test") {
        results.Add(res)
    }
    results.Close()

    // Assert performance requirements
    assert.Less(t, results.Latencies.Mean, 100*time.Millisecond, "Mean latency should be < 100ms")
    assert.Less(t, results.Latencies.P95, 200*time.Millisecond, "95th percentile should be < 200ms")
    assert.Greater(t, results.Success, 0.99, "Success rate should be > 99%")
    
    // Print results
    fmt.Printf("Load Test Results:\n")
    fmt.Printf("Success Rate: %.2f%%\n", results.Success*100)
    fmt.Printf("Mean Latency: %v\n", results.Latencies.Mean)
    fmt.Printf("95th Percentile: %v\n", results.Latencies.P95)
    fmt.Printf("Max Latency: %v\n", results.Latencies.Max)
    fmt.Printf("Requests/sec: %.2f\n", results.Rate)
}

// Stress testing
func TestPaymentAPI_StressTest(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping stress test in short mode")
    }

    // Gradually increase load
    rates := []vegeta.Rate{
        {Freq: 10, Per: time.Second},
        {Freq: 50, Per: time.Second},
        {Freq: 100, Per: time.Second},
        {Freq: 200, Per: time.Second},
        {Freq: 500, Per: time.Second},
    }

    target := vegeta.Target{
        Method: "POST",
        URL:    "http://localhost:8080/api/v1/payments",
        Header: http.Header{"Content-Type": []string{"application/json"}},
        Body:   []byte(`{"user_id": "user_123", "amount": 100.50, "currency": "USD"}`),
    }

    attacker := vegeta.NewAttacker()
    targeter := vegeta.NewStaticTargeter(target)

    for _, rate := range rates {
        t.Run(fmt.Sprintf("Rate_%d_per_sec", rate.Freq), func(t *testing.T) {
            var results vegeta.Metrics
            duration := 10 * time.Second
            
            for res := range attacker.Attack(targeter, rate, duration, "Stress Test") {
                results.Add(res)
            }
            results.Close()

            // Log results for analysis
            t.Logf("Rate: %d/sec, Success: %.2f%%, Mean: %v, P95: %v",
                rate.Freq, results.Success*100, results.Latencies.Mean, results.Latencies.P95)

            // Assert basic health
            assert.Greater(t, results.Success, 0.95, "Success rate should be > 95%")
        })
    }
}

// Memory performance test
func TestPaymentService_MemoryUsage(t *testing.T) {
    service := NewPaymentService(/* dependencies */)
    
    // Baseline memory
    var m1 runtime.MemStats
    runtime.ReadMemStats(&m1)
    
    // Process many payments
    for i := 0; i < 10000; i++ {
        payment, err := service.ProcessPayment(
            context.Background(),
            fmt.Sprintf("user_%d", i),
            100.0,
            "USD",
        )
        assert.NoError(t, err)
        assert.NotNil(t, payment)
    }
    
    // Force garbage collection
    runtime.GC()
    
    // Check memory usage
    var m2 runtime.MemStats
    runtime.ReadMemStats(&m2)
    
    memoryIncrease := m2.Alloc - m1.Alloc
    t.Logf("Memory increase: %d bytes", memoryIncrease)
    
    // Assert reasonable memory usage (adjust threshold as needed)
    assert.Less(t, memoryIncrease, uint64(50*1024*1024), "Memory increase should be < 50MB")
}
```

---

## 🔒 **Security Testing**

### **Security Testing Framework**

```go
package security

import (
    "crypto/rand"
    "fmt"
    "net/http"
    "net/http/httptest"
    "strings"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

type SecurityTestSuite struct {
    suite.Suite
    server *httptest.Server
    router *gin.Engine
}

func (suite *SecurityTestSuite) SetupTest() {
    suite.router = setupSecureRouter()
    suite.server = httptest.NewServer(suite.router)
}

func (suite *SecurityTestSuite) TearDownTest() {
    suite.server.Close()
}

// SQL Injection Tests
func (suite *SecurityTestSuite) TestSQLInjection() {
    maliciousInputs := []string{
        "'; DROP TABLE payments; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "' OR 1=1#",
    }

    for _, input := range maliciousInputs {
        suite.Run(fmt.Sprintf("SQLInjection_%s", input), func() {
            url := fmt.Sprintf("%s/api/v1/payments?user_id=%s", suite.server.URL, input)
            resp, err := http.Get(url)
            require.NoError(suite.T(), err)
            defer resp.Body.Close()

            // Should not return internal server error or expose SQL errors
            assert.NotEqual(suite.T(), http.StatusInternalServerError, resp.StatusCode)
            
            // Check response doesn't contain SQL error messages
            body := suite.readBody(resp)
            assert.NotContains(suite.T(), strings.ToLower(body), "sql")
            assert.NotContains(suite.T(), strings.ToLower(body), "database")
            assert.NotContains(suite.T(), strings.ToLower(body), "syntax error")
        })
    }
}

// XSS Tests
func (suite *SecurityTestSuite) TestXSSProtection() {
    xssPayloads := []string{
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "';alert('xss');//",
        "<svg onload=alert('xss')>",
    }

    for _, payload := range xssPayloads {
        suite.Run(fmt.Sprintf("XSS_%s", payload), func() {
            requestBody := fmt.Sprintf(`{"user_id": "%s", "amount": 100, "currency": "USD"}`, payload)
            resp, err := http.Post(
                suite.server.URL+"/api/v1/payments",
                "application/json",
                strings.NewReader(requestBody),
            )
            require.NoError(suite.T(), err)
            defer resp.Body.Close()

            body := suite.readBody(resp)
            
            // Response should not contain unescaped script tags
            assert.NotContains(suite.T(), body, "<script>")
            assert.NotContains(suite.T(), body, "javascript:")
            assert.NotContains(suite.T(), body, "onerror=")
        })
    }
}

// Authentication Tests
func (suite *SecurityTestSuite) TestAuthenticationRequired() {
    protectedEndpoints := []struct {
        method string
        path   string
    }{
        {"GET", "/api/v1/payments"},
        {"POST", "/api/v1/payments"},
        {"GET", "/api/v1/users/user_123/payments"},
        {"DELETE", "/api/v1/payments/pay_123"},
    }

    for _, endpoint := range protectedEndpoints {
        suite.Run(fmt.Sprintf("Auth_%s_%s", endpoint.method, endpoint.path), func() {
            req, err := http.NewRequest(endpoint.method, suite.server.URL+endpoint.path, nil)
            require.NoError(suite.T(), err)

            client := &http.Client{}
            resp, err := client.Do(req)
            require.NoError(suite.T(), err)
            defer resp.Body.Close()

            // Should require authentication
            assert.Equal(suite.T(), http.StatusUnauthorized, resp.StatusCode)
        })
    }
}

// Authorization Tests
func (suite *SecurityTestSuite) TestAuthorization() {
    // Test with valid but insufficient permissions
    token := suite.generateToken("user_123", []string{"read"}) // No write permission
    
    req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/payments", strings.NewReader(`{
        "user_id": "user_123",
        "amount": 100,
        "currency": "USD"
    }`))
    require.NoError(suite.T(), err)
    
    req.Header.Set("Authorization", "Bearer "+token)
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(req)
    require.NoError(suite.T(), err)
    defer resp.Body.Close()

    assert.Equal(suite.T(), http.StatusForbidden, resp.StatusCode)
}

// Rate Limiting Tests
func (suite *SecurityTestSuite) TestRateLimiting() {
    token := suite.generateToken("user_123", []string{"read", "write"})
    
    // Make rapid requests
    successCount := 0
    rateLimitedCount := 0
    
    for i := 0; i < 20; i++ {
        req, _ := http.NewRequest("GET", suite.server.URL+"/api/v1/payments", nil)
        req.Header.Set("Authorization", "Bearer "+token)
        
        client := &http.Client{}
        resp, err := client.Do(req)
        require.NoError(suite.T(), err)
        resp.Body.Close()

        if resp.StatusCode == http.StatusOK {
            successCount++
        } else if resp.StatusCode == http.StatusTooManyRequests {
            rateLimitedCount++
        }
    }

    // Should have rate limiting in effect
    assert.Greater(suite.T(), rateLimitedCount, 0, "Rate limiting should be active")
    assert.Less(suite.T(), successCount, 20, "Not all requests should succeed")
}

// Input Validation Tests
func (suite *SecurityTestSuite) TestInputValidation() {
    testCases := []struct {
        name     string
        payload  string
        expected int
    }{
        {
            name:     "Negative amount",
            payload:  `{"user_id": "user_123", "amount": -100, "currency": "USD"}`,
            expected: http.StatusBadRequest,
        },
        {
            name:     "Invalid currency",
            payload:  `{"user_id": "user_123", "amount": 100, "currency": "INVALID"}`,
            expected: http.StatusBadRequest,
        },
        {
            name:     "Missing user_id",
            payload:  `{"amount": 100, "currency": "USD"}`,
            expected: http.StatusBadRequest,
        },
        {
            name:     "Extremely large amount",
            payload:  `{"user_id": "user_123", "amount": 999999999999999, "currency": "USD"}`,
            expected: http.StatusBadRequest,
        },
    }

    token := suite.generateToken("user_123", []string{"write"})
    
    for _, tc := range testCases {
        suite.Run(tc.name, func() {
            req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/payments", strings.NewReader(tc.payload))
            require.NoError(suite.T(), err)
            
            req.Header.Set("Authorization", "Bearer "+token)
            req.Header.Set("Content-Type", "application/json")

            client := &http.Client{}
            resp, err := client.Do(req)
            require.NoError(suite.T(), err)
            defer resp.Body.Close()

            assert.Equal(suite.T(), tc.expected, resp.StatusCode)
        })
    }
}

// CSRF Protection Tests
func (suite *SecurityTestSuite) TestCSRFProtection() {
    // Attempt to make a state-changing request without CSRF token
    req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/payments", strings.NewReader(`{
        "user_id": "user_123",
        "amount": 100,
        "currency": "USD"
    }`))
    require.NoError(suite.T(), err)
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Origin", "http://malicious-site.com")

    client := &http.Client{}
    resp, err := client.Do(req)
    require.NoError(suite.T(), err)
    defer resp.Body.Close()

    // Should be blocked due to CSRF protection
    assert.NotEqual(suite.T(), http.StatusOK, resp.StatusCode)
}

// Helper methods
func (suite *SecurityTestSuite) generateToken(userID string, permissions []string) string {
    // Generate a valid JWT token for testing
    // Implementation depends on your JWT library
    return "valid.jwt.token"
}

func (suite *SecurityTestSuite) readBody(resp *http.Response) string {
    body, _ := io.ReadAll(resp.Body)
    return string(body)
}

// Fuzz testing for critical functions
func FuzzPaymentValidation(f *testing.F) {
    // Seed corpus
    f.Add("user_123", 100.0, "USD")
    f.Add("user_456", 0.01, "EUR")
    f.Add("", -100.0, "GBP")

    f.Fuzz(func(t *testing.T, userID string, amount float64, currency string) {
        // This should never panic
        defer func() {
            if r := recover(); r != nil {
                t.Errorf("Payment validation panicked with: %v", r)
            }
        }()

        payment := &Payment{
            UserID:   userID,
            Amount:   amount,
            Currency: currency,
        }

        // Validation should handle any input gracefully
        err := validatePayment(payment)
        
        // Just ensure it doesn't crash
        _ = err
    })
}

func TestSecuritySuite(t *testing.T) {
    suite.Run(t, new(SecurityTestSuite))
}
```

---

## 🎯 **Interview Questions**

### **Testing Strategy Questions**

**Q1: Explain the testing pyramid and how you would implement it for a payment service.**

**Answer:**

The testing pyramid represents the ideal distribution of different types of tests:

**Unit Tests (Base - 70%):**
- Test individual functions and methods
- Fast execution (milliseconds)
- High code coverage
- Mock external dependencies

```go
func TestPaymentValidation(t *testing.T) {
    tests := []struct {
        payment *Payment
        wantErr bool
    }{
        {&Payment{Amount: 100, Currency: "USD"}, false},
        {&Payment{Amount: -100, Currency: "USD"}, true},
        {&Payment{Amount: 100, Currency: ""}, true},
    }
    
    for _, tt := range tests {
        err := validatePayment(tt.payment)
        assert.Equal(t, tt.wantErr, err != nil)
    }
}
```

**Integration Tests (Middle - 20%):**
- Test component interactions
- Database, external APIs, message queues
- Medium execution time

```go
func TestPaymentRepository_Integration(t *testing.T) {
    db := setupTestDB()
    repo := NewPaymentRepository(db)
    
    payment := &Payment{ID: "pay_123", Amount: 100}
    err := repo.Save(context.Background(), payment)
    assert.NoError(t, err)
    
    found, err := repo.FindByID(context.Background(), "pay_123")
    assert.NoError(t, err)
    assert.Equal(t, payment.Amount, found.Amount)
}
```

**E2E Tests (Top - 10%):**
- Test complete user workflows
- Real environment with all dependencies
- Slow execution but high confidence

**Q2: How do you test asynchronous code and event-driven systems?**

**Answer:**

**Async Testing Strategies:**

1. **Channel-based synchronization:**
```go
func TestAsyncPaymentProcessing(t *testing.T) {
    done := make(chan bool)
    var result *Payment
    
    service.ProcessPaymentAsync("pay_123", func(p *Payment, err error) {
        result = p
        done <- true
    })
    
    select {
    case <-done:
        assert.NotNil(t, result)
    case <-time.After(5 * time.Second):
        t.Fatal("Timeout waiting for async operation")
    }
}
```

2. **Event-driven testing:**
```go
func TestEventHandler(t *testing.T) {
    eventBus := NewTestEventBus()
    handler := NewPaymentEventHandler(eventBus)
    
    // Publish event
    event := PaymentCreatedEvent{PaymentID: "pay_123"}
    eventBus.Publish(event)
    
    // Wait for processing
    time.Sleep(100 * time.Millisecond)
    
    // Verify side effects
    assert.True(t, handler.WasProcessed("pay_123"))
}
```

**Q3: How do you test error scenarios and edge cases?**

**Answer:**

**Error Testing Approaches:**

1. **Dependency failures:**
```go
func TestPaymentService_DatabaseFailure(t *testing.T) {
    mockRepo := &MockRepository{}
    mockRepo.On("Save", mock.Anything).Return(errors.New("db connection failed"))
    
    service := NewPaymentService(mockRepo)
    _, err := service.ProcessPayment(ctx, "user_123", 100, "USD")
    
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "db connection failed")
}
```

2. **Timeout scenarios:**
```go
func TestPaymentService_Timeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
    defer cancel()
    
    service := NewPaymentService(slowRepository)
    _, err := service.ProcessPayment(ctx, "user_123", 100, "USD")
    
    assert.Equal(t, context.DeadlineExceeded, err)
}
```

3. **Resource exhaustion:**
```go
func TestPaymentService_ResourceLimits(t *testing.T) {
    service := NewPaymentService(limitedRepository)
    
    // Exhaust connection pool
    var wg sync.WaitGroup
    errors := make([]error, 100)
    
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(idx int) {
            defer wg.Done()
            _, errors[idx] = service.ProcessPayment(ctx, fmt.Sprintf("user_%d", idx), 100, "USD")
        }(i)
    }
    
    wg.Wait()
    
    // Some requests should fail due to resource limits
    errorCount := 0
    for _, err := range errors {
        if err != nil {
            errorCount++
        }
    }
    assert.Greater(t, errorCount, 0)
}
```

---

## 🚀 **Best Practices Summary**

### **Testing Principles**

1. **FIRST Principles:**
   - **Fast**: Tests should run quickly
   - **Independent**: Tests shouldn't depend on each other
   - **Repeatable**: Same results every time
   - **Self-Validating**: Clear pass/fail
   - **Timely**: Written before or with production code

2. **AAA Pattern:**
   - **Arrange**: Set up test data and conditions
   - **Act**: Execute the code under test
   - **Assert**: Verify the expected outcome

3. **Test Naming:**
   - Use descriptive names: `TestProcessPayment_WithInvalidAmount_ReturnsError`
   - Include expected behavior and conditions

### **Common Anti-patterns to Avoid**

1. **Flaky Tests**: Tests that randomly fail
2. **Overly Complex Tests**: Tests that are hard to understand
3. **Testing Implementation Details**: Focus on behavior, not internals
4. **Ignoring Test Maintenance**: Keep tests updated with code changes

### **CI/CD Integration**

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-go@v2
        with:
          go-version: 1.21
      
      - name: Unit Tests
        run: go test -short ./...
      
      - name: Integration Tests
        run: go test -tags=integration ./...
      
      - name: Coverage
        run: go test -coverprofile=coverage.out ./...
      
      - name: Upload Coverage
        uses: codecov/codecov-action@v1
```

This comprehensive testing guide provides the foundation for building robust, reliable software systems and demonstrates the testing expertise expected in senior engineering interviews.

## Api Testing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #api-testing -->

Placeholder content. Please replace with proper section.


## Database Testing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #database-testing -->

Placeholder content. Please replace with proper section.


## Test Automation  Cicd

<!-- AUTO-GENERATED ANCHOR: originally referenced as #test-automation--cicd -->

Placeholder content. Please replace with proper section.


## Testing Patterns  Best Practices

<!-- AUTO-GENERATED ANCHOR: originally referenced as #testing-patterns--best-practices -->

Placeholder content. Please replace with proper section.
