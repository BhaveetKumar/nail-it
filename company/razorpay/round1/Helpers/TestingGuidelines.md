# Go Testing Guidelines

This guide provides comprehensive testing strategies and best practices for Go applications, with a focus on backend services and fintech applications.

## Testing Types

### 1. Unit Tests

Test individual functions, methods, or components in isolation.

```go
// Example: Testing a payment validator
func TestPaymentValidator_ValidateAmount(t *testing.T) {
    validator := NewPaymentValidator()

    tests := []struct {
        name    string
        amount  float64
        wantErr bool
    }{
        {"valid amount", 100.50, false},
        {"zero amount", 0, true},
        {"negative amount", -50, true},
        {"very large amount", 1000000, false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := validator.ValidateAmount(tt.amount)
            if (err != nil) != tt.wantErr {
                t.Errorf("ValidateAmount() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 2. Integration Tests

Test the interaction between multiple components.

```go
// Example: Testing database operations
func TestPaymentRepository_Integration(t *testing.T) {
    // Setup test database
    db := setupTestDB(t)
    defer cleanupTestDB(t, db)

    repo := NewPaymentRepository(db)

    payment := &Payment{
        ID:     "test-payment-123",
        Amount: 100.50,
        Status: "pending",
    }

    // Test create
    err := repo.Create(context.Background(), payment)
    if err != nil {
        t.Fatalf("Create() error = %v", err)
    }

    // Test retrieve
    retrieved, err := repo.GetByID(context.Background(), payment.ID)
    if err != nil {
        t.Fatalf("GetByID() error = %v", err)
    }

    if retrieved.Amount != payment.Amount {
        t.Errorf("Expected amount %v, got %v", payment.Amount, retrieved.Amount)
    }
}
```

### 3. End-to-End Tests

Test complete user workflows.

```go
// Example: Testing payment flow
func TestPaymentFlow_E2E(t *testing.T) {
    // Setup test environment
    server := setupTestServer(t)
    defer server.Close()

    // Create payment
    paymentReq := PaymentRequest{
        Amount:   100.50,
        Currency: "USD",
        UserID:   "user-123",
    }

    resp, err := http.Post(server.URL+"/api/payments", "application/json",
        strings.NewReader(toJSON(paymentReq)))
    if err != nil {
        t.Fatalf("Failed to create payment: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusCreated {
        t.Errorf("Expected status 201, got %d", resp.StatusCode)
    }

    // Verify payment was created
    var payment Payment
    if err := json.NewDecoder(resp.Body).Decode(&payment); err != nil {
        t.Fatalf("Failed to decode response: %v", err)
    }

    if payment.Status != "pending" {
        t.Errorf("Expected status 'pending', got %s", payment.Status)
    }
}
```

## Testing Patterns

### 1. Table-Driven Tests

Use table-driven tests for multiple test cases.

```go
func TestPaymentProcessor_ProcessPayment(t *testing.T) {
    processor := NewPaymentProcessor()

    tests := []struct {
        name        string
        payment     *Payment
        gateway     PaymentGateway
        wantStatus  string
        wantErr     bool
    }{
        {
            name: "successful payment",
            payment: &Payment{
                ID:     "pay-123",
                Amount: 100.50,
                Status: "pending",
            },
            gateway:    &MockPaymentGateway{shouldSucceed: true},
            wantStatus: "completed",
            wantErr:    false,
        },
        {
            name: "failed payment",
            payment: &Payment{
                ID:     "pay-456",
                Amount: 100.50,
                Status: "pending",
            },
            gateway:    &MockPaymentGateway{shouldSucceed: false},
            wantStatus: "failed",
            wantErr:    false,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := processor.ProcessPayment(tt.payment, tt.gateway)
            if (err != nil) != tt.wantErr {
                t.Errorf("ProcessPayment() error = %v, wantErr %v", err, tt.wantErr)
            }

            if tt.payment.Status != tt.wantStatus {
                t.Errorf("Expected status %s, got %s", tt.wantStatus, tt.payment.Status)
            }
        })
    }
}
```

### 2. Test Helpers

Create helper functions for common test setup.

```go
// Test helper for database setup
func setupTestDB(t *testing.T) *sql.DB {
    db, err := sql.Open("postgres", "postgres://test:test@localhost/testdb?sslmode=disable")
    if err != nil {
        t.Fatalf("Failed to connect to test database: %v", err)
    }

    // Run migrations
    if err := runMigrations(db); err != nil {
        t.Fatalf("Failed to run migrations: %v", err)
    }

    return db
}

// Test helper for cleanup
func cleanupTestDB(t *testing.T, db *sql.DB) {
    if err := db.Close(); err != nil {
        t.Errorf("Failed to close database: %v", err)
    }
}

// Test helper for creating test data
func createTestPayment(t *testing.T, db *sql.DB) *Payment {
    payment := &Payment{
        ID:        generateTestID(),
        Amount:    100.50,
        Currency:  "USD",
        Status:    "pending",
        CreatedAt: time.Now(),
    }

    repo := NewPaymentRepository(db)
    if err := repo.Create(context.Background(), payment); err != nil {
        t.Fatalf("Failed to create test payment: %v", err)
    }

    return payment
}
```

### 3. Mock Objects

Create mock implementations for external dependencies.

```go
// Mock payment gateway
type MockPaymentGateway struct {
    shouldSucceed bool
    shouldTimeout bool
    calls         []PaymentRequest
}

func (m *MockPaymentGateway) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    m.calls = append(m.calls, req)

    if m.shouldTimeout {
        time.Sleep(2 * time.Second)
        return nil, errors.New("timeout")
    }

    if m.shouldSucceed {
        return &PaymentResponse{
            TransactionID: "txn-" + generateID(),
            Status:        "success",
            Amount:        req.Amount,
        }, nil
    }

    return nil, errors.New("payment failed")
}

func (m *MockPaymentGateway) GetCalls() []PaymentRequest {
    return m.calls
}

func (m *MockPaymentGateway) Reset() {
    m.calls = nil
}

// Mock notification service
type MockNotificationService struct {
    sentNotifications []Notification
    shouldFail        bool
}

func (m *MockNotificationService) SendNotification(notification Notification) error {
    if m.shouldFail {
        return errors.New("notification service unavailable")
    }

    m.sentNotifications = append(m.sentNotifications, notification)
    return nil
}

func (m *MockNotificationService) GetSentNotifications() []Notification {
    return m.sentNotifications
}
```

## Testing Utilities

### 1. Test Data Builders

Use builders for creating test data.

```go
type PaymentBuilder struct {
    payment *Payment
}

func NewPaymentBuilder() *PaymentBuilder {
    return &PaymentBuilder{
        payment: &Payment{
            ID:        generateID(),
            Amount:    100.50,
            Currency:  "USD",
            Status:    "pending",
            CreatedAt: time.Now(),
        },
    }
}

func (pb *PaymentBuilder) WithID(id string) *PaymentBuilder {
    pb.payment.ID = id
    return pb
}

func (pb *PaymentBuilder) WithAmount(amount float64) *PaymentBuilder {
    pb.payment.Amount = amount
    return pb
}

func (pb *PaymentBuilder) WithStatus(status string) *PaymentBuilder {
    pb.payment.Status = status
    return pb
}

func (pb *PaymentBuilder) Build() *Payment {
    return pb.payment
}

// Usage in tests
func TestPaymentService(t *testing.T) {
    payment := NewPaymentBuilder().
        WithAmount(200.75).
        WithStatus("completed").
        Build()

    // Test with the built payment
}
```

### 2. Test Fixtures

Use fixtures for complex test data.

```go
// fixtures/payments.json
{
  "valid_payment": {
    "id": "pay-123",
    "amount": 100.50,
    "currency": "USD",
    "status": "pending",
    "user_id": "user-456"
  },
  "large_payment": {
    "id": "pay-789",
    "amount": 10000.00,
    "currency": "USD",
    "status": "pending",
    "user_id": "user-789"
  }
}

// Load fixtures in tests
func loadPaymentFixture(t *testing.T, name string) *Payment {
    data, err := os.ReadFile(fmt.Sprintf("test/fixtures/payments.json"))
    if err != nil {
        t.Fatalf("Failed to load fixture: %v", err)
    }

    var fixtures map[string]*Payment
    if err := json.Unmarshal(data, &fixtures); err != nil {
        t.Fatalf("Failed to unmarshal fixture: %v", err)
    }

    fixture, exists := fixtures[name]
    if !exists {
        t.Fatalf("Fixture %s not found", name)
    }

    return fixture
}
```

### 3. Test Utilities

Create utility functions for common test operations.

```go
// Test utilities
func assertPaymentEqual(t *testing.T, expected, actual *Payment) {
    if expected.ID != actual.ID {
        t.Errorf("Expected ID %s, got %s", expected.ID, actual.ID)
    }
    if expected.Amount != actual.Amount {
        t.Errorf("Expected Amount %v, got %v", expected.Amount, actual.Amount)
    }
    if expected.Status != actual.Status {
        t.Errorf("Expected Status %s, got %s", expected.Status, actual.Status)
    }
}

func assertErrorContains(t *testing.T, err error, expected string) {
    if err == nil {
        t.Errorf("Expected error containing '%s', got nil", expected)
        return
    }

    if !strings.Contains(err.Error(), expected) {
        t.Errorf("Expected error to contain '%s', got '%s'", expected, err.Error())
    }
}

func assertHTTPStatus(t *testing.T, resp *http.Response, expected int) {
    if resp.StatusCode != expected {
        t.Errorf("Expected status %d, got %d", expected, resp.StatusCode)
    }
}
```

## Testing Best Practices

### 1. Test Organization

```go
// Organize tests by functionality
func TestPaymentService_CreatePayment(t *testing.T) {
    // Test create payment functionality
}

func TestPaymentService_ProcessPayment(t *testing.T) {
    // Test process payment functionality
}

func TestPaymentService_GetPayment(t *testing.T) {
    // Test get payment functionality
}

// Use subtests for related test cases
func TestPaymentValidator(t *testing.T) {
    t.Run("ValidAmount", func(t *testing.T) {
        // Test valid amount
    })

    t.Run("InvalidAmount", func(t *testing.T) {
        // Test invalid amount
    })

    t.Run("ZeroAmount", func(t *testing.T) {
        // Test zero amount
    })
}
```

### 2. Test Naming

```go
// Good test names
func TestPaymentService_CreatePayment_Success(t *testing.T) {}
func TestPaymentService_CreatePayment_InvalidAmount(t *testing.T) {}
func TestPaymentService_CreatePayment_DatabaseError(t *testing.T) {}

// Bad test names
func TestPaymentService(t *testing.T) {}
func TestCreate(t *testing.T) {}
func Test1(t *testing.T) {}
```

### 3. Test Isolation

```go
func TestPaymentService_Isolated(t *testing.T) {
    // Each test should be independent
    t.Run("Test1", func(t *testing.T) {
        // Setup
        service := NewPaymentService()

        // Test
        result := service.ProcessPayment(payment)

        // Assert
        assert.NoError(t, result)
    })

    t.Run("Test2", func(t *testing.T) {
        // Setup (independent of Test1)
        service := NewPaymentService()

        // Test
        result := service.ProcessPayment(payment)

        // Assert
        assert.NoError(t, result)
    })
}
```

## Performance Testing

### 1. Benchmark Tests

```go
func BenchmarkPaymentProcessor_ProcessPayment(b *testing.B) {
    processor := NewPaymentProcessor()
    payment := &Payment{
        ID:     "bench-payment",
        Amount: 100.50,
        Status: "pending",
    }

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        err := processor.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkPaymentRepository_Create(b *testing.B) {
    db := setupTestDB(b)
    defer cleanupTestDB(b, db)

    repo := NewPaymentRepository(db)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("bench-payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }

        err := repo.Create(context.Background(), payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### 2. Load Testing

```go
func TestPaymentService_Load(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping load test in short mode")
    }

    service := NewPaymentService()

    // Test concurrent payments
    numGoroutines := 100
    numPaymentsPerGoroutine := 10

    var wg sync.WaitGroup
    errors := make(chan error, numGoroutines*numPaymentsPerGoroutine)

    for i := 0; i < numGoroutines; i++ {
        wg.Add(1)
        go func(goroutineID int) {
            defer wg.Done()

            for j := 0; j < numPaymentsPerGoroutine; j++ {
                payment := &Payment{
                    ID:     fmt.Sprintf("load-payment-%d-%d", goroutineID, j),
                    Amount: 100.50,
                    Status: "pending",
                }

                if err := service.ProcessPayment(payment); err != nil {
                    errors <- err
                }
            }
        }(i)
    }

    wg.Wait()
    close(errors)

    // Check for errors
    var errorCount int
    for err := range errors {
        t.Errorf("Payment processing error: %v", err)
        errorCount++
    }

    if errorCount > 0 {
        t.Errorf("Expected 0 errors, got %d", errorCount)
    }
}
```

## Test Coverage

### 1. Coverage Analysis

```bash
# Run tests with coverage
go test -cover ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Coverage for specific packages
go test -cover ./internal/services/...
```

### 2. Coverage Goals

```go
// Example: Test with coverage requirements
func TestPaymentService_WithCoverage(t *testing.T) {
    // Ensure all code paths are tested
    service := NewPaymentService()

    // Test success path
    payment := &Payment{Amount: 100.50, Status: "pending"}
    err := service.ProcessPayment(payment)
    assert.NoError(t, err)

    // Test error path
    invalidPayment := &Payment{Amount: -100, Status: "pending"}
    err = service.ProcessPayment(invalidPayment)
    assert.Error(t, err)

    // Test edge cases
    zeroPayment := &Payment{Amount: 0, Status: "pending"}
    err = service.ProcessPayment(zeroPayment)
    assert.Error(t, err)
}
```

## Testing Tools

### 1. Testify

```go
import (
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/suite"
)

func TestPaymentService_WithTestify(t *testing.T) {
    service := NewPaymentService()

    payment := &Payment{Amount: 100.50, Status: "pending"}
    err := service.ProcessPayment(payment)

    assert.NoError(t, err)
    assert.Equal(t, "completed", payment.Status)
    assert.Greater(t, payment.ProcessedAt, time.Now().Add(-time.Minute))
}

// Test suite example
type PaymentServiceTestSuite struct {
    suite.Suite
    service *PaymentService
    db      *sql.DB
}

func (suite *PaymentServiceTestSuite) SetupTest() {
    suite.db = setupTestDB(suite.T())
    suite.service = NewPaymentService(suite.db)
}

func (suite *PaymentServiceTestSuite) TearDownTest() {
    cleanupTestDB(suite.T(), suite.db)
}

func (suite *PaymentServiceTestSuite) TestProcessPayment() {
    payment := &Payment{Amount: 100.50, Status: "pending"}
    err := suite.service.ProcessPayment(payment)

    suite.NoError(err)
    suite.Equal("completed", payment.Status)
}

func TestPaymentServiceTestSuite(t *testing.T) {
    suite.Run(t, new(PaymentServiceTestSuite))
}
```

### 2. GoMock

```go
// Generate mocks
//go:generate mockgen -source=payment_gateway.go -destination=mocks/payment_gateway_mock.go

// Use generated mocks
func TestPaymentService_WithMock(t *testing.T) {
    ctrl := gomock.NewController(t)
    defer ctrl.Finish()

    mockGateway := mocks.NewMockPaymentGateway(ctrl)
    service := NewPaymentService(mockGateway)

    // Set up mock expectations
    mockGateway.EXPECT().
        ProcessPayment(gomock.Any()).
        Return(&PaymentResponse{Status: "success"}, nil).
        Times(1)

    // Test
    payment := &Payment{Amount: 100.50, Status: "pending"}
    err := service.ProcessPayment(payment)

    assert.NoError(t, err)
}
```

## Continuous Integration

### 1. GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

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

    steps:
      - uses: actions/checkout@v3

      - name: Set up Go
        uses: actions/setup-go@v3
        with:
          go-version: 1.21

      - name: Install dependencies
        run: go mod download

      - name: Run tests
        run: go test -v -race -coverprofile=coverage.out ./...
        env:
          DATABASE_URL: postgres://postgres:postgres@localhost/testdb?sslmode=disable

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.out
```

### 2. Test Scripts

```bash
#!/bin/bash
# scripts/test.sh

set -e

echo "Running unit tests..."
go test -v -race ./...

echo "Running integration tests..."
go test -v -race -tags=integration ./...

echo "Running benchmarks..."
go test -bench=. -benchmem ./...

echo "Checking test coverage..."
go test -coverprofile=coverage.out ./...
go tool cover -func=coverage.out
```

This comprehensive testing guide provides the foundation for writing effective tests in Go applications, with particular emphasis on backend services and fintech applications. Follow these guidelines to ensure your code is well-tested, reliable, and maintainable.
