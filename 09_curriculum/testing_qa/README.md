# Testing & Quality Assurance

## Table of Contents

1. [Overview](#overview)
2. [Testing Strategy](#testing-strategy)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [End-to-End Testing](#end-to-end-testing)
6. [Performance Testing](#performance-testing)
7. [Security Testing](#security-testing)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)

## Overview

### Learning Objectives

- Implement comprehensive testing strategies
- Ensure code quality and reliability
- Automate testing processes
- Maintain high software quality standards

### What is Testing & QA?

Testing and Quality Assurance involve systematic processes to ensure the Master Engineer Curriculum meets quality standards, functions correctly, and provides a reliable learning experience.

## Testing Strategy

### 1. Testing Pyramid

#### Testing Levels
```go
// testing/strategy.go
package testing

import (
    "testing"
    "github.com/stretchr/testify/assert"
)

// Unit Tests (70%)
func TestUserService_CreateUser(t *testing.T) {
    // Test individual functions and methods
    service := NewUserService()
    user, err := service.CreateUser("test@example.com", "Test User")
    
    assert.NoError(t, err)
    assert.Equal(t, "test@example.com", user.Email)
    assert.Equal(t, "Test User", user.Name)
}

// Integration Tests (20%)
func TestUserAPI_CreateUser(t *testing.T) {
    // Test API endpoints with database
    app := setupTestApp()
    defer app.Close()
    
    response := app.Post("/api/users", map[string]string{
        "email": "test@example.com",
        "name":  "Test User",
    })
    
    assert.Equal(t, 201, response.StatusCode)
}

// End-to-End Tests (10%)
func TestUserRegistrationFlow(t *testing.T) {
    // Test complete user journey
    driver := setupSeleniumDriver()
    defer driver.Quit()
    
    driver.Get("http://localhost:3000/register")
    driver.FindElement(ByID("email")).SendKeys("test@example.com")
    driver.FindElement(ByID("name")).SendKeys("Test User")
    driver.FindElement(ByID("submit")).Click()
    
    assert.Contains(t, driver.PageSource(), "Registration successful")
}
```

### 2. Test Automation

#### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Run unit tests
      run: |
        go test -v -race -coverprofile=coverage.out ./...
        go tool cover -html=coverage.out -o coverage.html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
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
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run integration tests
      run: |
        go test -v -tags=integration ./tests/integration/...

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run E2E tests
      run: |
        npm run test:e2e
```

## Unit Testing

### 1. Go Unit Tests

#### Service Testing
```go
// services/user_service_test.go
package services

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

func TestUserService_CreateUser(t *testing.T) {
    tests := []struct {
        name        string
        email       string
        name        string
        expectedErr error
    }{
        {
            name:        "valid user",
            email:       "test@example.com",
            name:        "Test User",
            expectedErr: nil,
        },
        {
            name:        "invalid email",
            email:       "invalid-email",
            name:        "Test User",
            expectedErr: ErrInvalidEmail,
        },
        {
            name:        "empty name",
            email:       "test@example.com",
            name:        "",
            expectedErr: ErrEmptyName,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            mockRepo := &MockUserRepository{}
            service := NewUserService(mockRepo)
            
            if tt.expectedErr == nil {
                mockRepo.On("Create", mock.AnythingOfType("*User")).Return(nil)
            }
            
            user, err := service.CreateUser(tt.email, tt.name)
            
            if tt.expectedErr != nil {
                assert.Error(t, err)
                assert.Nil(t, user)
            } else {
                assert.NoError(t, err)
                assert.NotNil(t, user)
                assert.Equal(t, tt.email, user.Email)
                assert.Equal(t, tt.name, user.Name)
            }
            
            mockRepo.AssertExpectations(t)
        })
    }
}

// Mock implementation
type MockUserRepository struct {
    mock.Mock
}

func (m *MockUserRepository) Create(user *User) error {
    args := m.Called(user)
    return args.Error(0)
}

func (m *MockUserRepository) GetByID(id string) (*User, error) {
    args := m.Called(id)
    return args.Get(0).(*User), args.Error(1)
}
```

### 2. Node.js Unit Tests

#### API Testing
```javascript
// tests/unit/user.test.js
const request = require('supertest');
const app = require('../../src/app');
const UserService = require('../../src/services/UserService');

// Mock the UserService
jest.mock('../../src/services/UserService');

describe('User API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User'
      };

      const mockUser = {
        id: '123',
        ...userData,
        createdAt: new Date()
      };

      UserService.createUser.mockResolvedValue(mockUser);

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201);

      expect(response.body).toEqual(mockUser);
      expect(UserService.createUser).toHaveBeenCalledWith(userData);
    });

    it('should return 400 for invalid email', async () => {
      const userData = {
        email: 'invalid-email',
        name: 'Test User'
      };

      UserService.createUser.mockRejectedValue(new Error('Invalid email'));

      await request(app)
        .post('/api/users')
        .send(userData)
        .expect(400);
    });
  });

  describe('GET /api/users/:id', () => {
    it('should return user by id', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User'
      };

      UserService.getUserById.mockResolvedValue(mockUser);

      const response = await request(app)
        .get('/api/users/123')
        .expect(200);

      expect(response.body).toEqual(mockUser);
      expect(UserService.getUserById).toHaveBeenCalledWith('123');
    });

    it('should return 404 for non-existent user', async () => {
      UserService.getUserById.mockRejectedValue(new Error('User not found'));

      await request(app)
        .get('/api/users/999')
        .expect(404);
    });
  });
});
```

## Integration Testing

### 1. Database Integration

#### Database Testing
```go
// tests/integration/database_test.go
package integration

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
)

type DatabaseTestSuite struct {
    suite.Suite
    db *gorm.DB
}

func (suite *DatabaseTestSuite) SetupSuite() {
    dsn := "host=localhost user=test password=test dbname=testdb port=5432 sslmode=disable"
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    assert.NoError(suite.T(), err)
    
    suite.db = db
    
    // Run migrations
    err = db.AutoMigrate(&User{}, &Lesson{}, &Progress{})
    assert.NoError(suite.T(), err)
}

func (suite *DatabaseTestSuite) TearDownSuite() {
    // Clean up test database
    suite.db.Exec("DROP SCHEMA public CASCADE")
    suite.db.Exec("CREATE SCHEMA public")
}

func (suite *DatabaseTestSuite) SetupTest() {
    // Clean up data before each test
    suite.db.Exec("TRUNCATE users, lessons, progress RESTART IDENTITY CASCADE")
}

func (suite *DatabaseTestSuite) TestUserCRUD() {
    // Create user
    user := &User{
        Email: "test@example.com",
        Name:  "Test User",
    }
    
    result := suite.db.Create(user)
    assert.NoError(suite.T(), result.Error)
    assert.NotZero(suite.T(), user.ID)
    
    // Read user
    var foundUser User
    result = suite.db.First(&foundUser, user.ID)
    assert.NoError(suite.T(), result.Error)
    assert.Equal(suite.T(), user.Email, foundUser.Email)
    
    // Update user
    foundUser.Name = "Updated Name"
    result = suite.db.Save(&foundUser)
    assert.NoError(suite.T(), result.Error)
    
    // Delete user
    result = suite.db.Delete(&foundUser)
    assert.NoError(suite.T(), result.Error)
}

func TestDatabaseSuite(t *testing.T) {
    suite.Run(t, new(DatabaseTestSuite))
}
```

### 2. API Integration

#### API Testing
```go
// tests/integration/api_test.go
package integration

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
)

type APITestSuite struct {
    suite.Suite
    server *httptest.Server
}

func (suite *APITestSuite) SetupSuite() {
    app := setupTestApp()
    suite.server = httptest.NewServer(app)
}

func (suite *APITestSuite) TearDownSuite() {
    suite.server.Close()
}

func (suite *APITestSuite) TestUserEndpoints() {
    // Test POST /api/users
    userData := map[string]string{
        "email": "test@example.com",
        "name":  "Test User",
    }
    
    jsonData, _ := json.Marshal(userData)
    resp, err := http.Post(suite.server.URL+"/api/users", "application/json", bytes.NewBuffer(jsonData))
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusCreated, resp.StatusCode)
    
    // Test GET /api/users
    resp, err = http.Get(suite.server.URL + "/api/users")
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
}

func TestAPISuite(t *testing.T) {
    suite.Run(t, new(APITestSuite))
}
```

## End-to-End Testing

### 1. Selenium Testing

#### E2E Test Suite
```python
# tests/e2e/test_user_journey.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestUserJourney:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:3000")
        yield
        self.driver.quit()

    def test_user_registration_and_login(self):
        # Navigate to registration page
        self.driver.find_element(By.LINK_TEXT, "Sign Up").click()
        
        # Fill registration form
        self.driver.find_element(By.ID, "email").send_keys("test@example.com")
        self.driver.find_element(By.ID, "name").send_keys("Test User")
        self.driver.find_element(By.ID, "password").send_keys("password123")
        self.driver.find_element(By.ID, "confirm-password").send_keys("password123")
        
        # Submit form
        self.driver.find_element(By.ID, "submit").click()
        
        # Wait for success message
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
        )
        
        # Verify user is logged in
        assert "Welcome, Test User" in self.driver.page_source

    def test_lesson_progress_tracking(self):
        # Login first
        self.login_user("test@example.com", "password123")
        
        # Navigate to a lesson
        self.driver.find_element(By.LINK_TEXT, "Phase 0: Fundamentals").click()
        self.driver.find_element(By.LINK_TEXT, "Mathematics").click()
        self.driver.find_element(By.LINK_TEXT, "Linear Algebra").click()
        
        # Start lesson
        self.driver.find_element(By.ID, "start-lesson").click()
        
        # Complete lesson
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "complete-lesson"))
        )
        self.driver.find_element(By.ID, "complete-lesson").click()
        
        # Verify progress updated
        progress = self.driver.find_element(By.CLASS_NAME, "progress-bar")
        assert progress.get_attribute("aria-valuenow") == "100"

    def login_user(self, email, password):
        self.driver.find_element(By.LINK_TEXT, "Sign In").click()
        self.driver.find_element(By.ID, "email").send_keys(email)
        self.driver.find_element(By.ID, "password").send_keys(password)
        self.driver.find_element(By.ID, "submit").click()
```

## Performance Testing

### 1. Load Testing

#### Load Test Configuration
```yaml
# tests/performance/load-test.yml
version: '3.8'
services:
  k6:
    image: loadimpact/k6:latest
    volumes:
      - ./load-test.js:/scripts/load-test.js
    command: run /scripts/load-test.js
    environment:
      - BASE_URL=http://api:8080

---
# tests/performance/load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.1'],    // Error rate under 10%
  },
};

export default function() {
  // Test user registration
  let userData = {
    email: `user${__VU}@example.com`,
    name: `User ${__VU}`,
  };
  
  let response = http.post(`${__ENV.BASE_URL}/api/users`, JSON.stringify(userData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(response, {
    'status is 201': (r) => r.status === 201,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  // Test lesson retrieval
  response = http.get(`${__ENV.BASE_URL}/api/lessons`);
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  sleep(1);
}
```

## Security Testing

### 1. Security Test Suite

#### Security Tests
```go
// tests/security/security_test.go
package security

import (
    "net/http"
    "net/http/httptest"
    "testing"
    "github.com/stretchr/testify/assert"
)

func TestSQLInjection(t *testing.T) {
    app := setupTestApp()
    
    // Test SQL injection in user search
    maliciousInput := "'; DROP TABLE users; --"
    
    req := httptest.NewRequest("GET", "/api/users?search="+maliciousInput, nil)
    w := httptest.NewRecorder()
    
    app.ServeHTTP(w, req)
    
    // Should not return 500 error (SQL injection successful)
    assert.NotEqual(t, http.StatusInternalServerError, w.Code)
}

func TestXSSProtection(t *testing.T) {
    app := setupTestApp()
    
    // Test XSS in user input
    xssPayload := "<script>alert('XSS')</script>"
    
    req := httptest.NewRequest("POST", "/api/users", strings.NewReader(`{
        "name": "`+xssPayload+`",
        "email": "test@example.com"
    }`))
    req.Header.Set("Content-Type", "application/json")
    
    w := httptest.NewRecorder()
    app.ServeHTTP(w, req)
    
    // Check that script tags are escaped
    assert.NotContains(t, w.Body.String(), "<script>")
}

func TestAuthenticationRequired(t *testing.T) {
    app := setupTestApp()
    
    // Test protected endpoint without authentication
    req := httptest.NewRequest("GET", "/api/admin/users", nil)
    w := httptest.NewRecorder()
    
    app.ServeHTTP(w, req)
    
    assert.Equal(t, http.StatusUnauthorized, w.Code)
}
```

## Follow-up Questions

### 1. Testing Strategy
**Q: What's the most effective testing strategy for the curriculum?**
A: Use a testing pyramid with 70% unit tests, 20% integration tests, and 10% E2E tests, combined with automated CI/CD pipelines.

### 2. Test Coverage
**Q: What test coverage percentage should I aim for?**
A: Aim for 80-90% code coverage for critical business logic, with focus on quality over quantity.

### 3. Performance Testing
**Q: How do you ensure the curriculum performs well under load?**
A: Use load testing tools like k6 or JMeter to test under realistic load conditions and set performance benchmarks.

## Sources

### Testing Tools
- **Jest**: [JavaScript Testing Framework](https://jestjs.io/)
- **Go Testing**: [Go Testing Package](https://golang.org/pkg/testing/)
- **Selenium**: [Web Browser Automation](https://www.selenium.dev/)

### Performance Testing
- **k6**: [Load Testing Tool](https://k6.io/)
- **JMeter**: [Performance Testing Tool](https://jmeter.apache.org/)
- **Artillery**: [Load Testing Platform](https://artillery.io/)

### Security Testing
- **OWASP ZAP**: [Security Testing Tool](https://owasp.org/www-project-zap/)
- **Snyk**: [Vulnerability Scanner](https://snyk.io/)
- **Trivy**: [Container Security Scanner](https://trivy.dev/)

---

**Next**: [Deployment DevOps](../../README.md) | **Previous**: [Mobile App](../../README.md) | **Up**: [Testing QA](README.md/)
