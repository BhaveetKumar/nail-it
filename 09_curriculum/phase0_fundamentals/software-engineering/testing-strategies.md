# Testing Strategies

## Table of Contents

1. [Overview](#overview/)
2. [Unit Testing](#unit-testing/)
3. [Integration Testing](#integration-testing/)
4. [Test-Driven Development (TDD)](#test-driven-development-tdd/)
5. [Behavior-Driven Development (BDD)](#behavior-driven-development-bdd/)
6. [Implementations](#implementations/)
7. [Follow-up Questions](#follow-up-questions/)
8. [Sources](#sources/)
9. [Projects](#projects/)

## Overview

### Learning Objectives

- Master unit testing principles and practices
- Understand integration testing strategies
- Learn Test-Driven Development (TDD) and Behavior-Driven Development (BDD)
- Apply testing best practices and patterns
- Implement comprehensive testing frameworks

### What are Testing Strategies?

Testing strategies are systematic approaches to verify software functionality, ensure quality, and prevent regressions through various levels of testing.

## Unit Testing

### 1. Unit Test Framework

#### Go Testing Implementation
```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

// Calculator service
type Calculator struct{}

func (c *Calculator) Add(a, b int) int {
    return a + b
}

func (c *Calculator) Subtract(a, b int) int {
    return a - b
}

func (c *Calculator) Multiply(a, b int) int {
    return a * b
}

func (c *Calculator) Divide(a, b int) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return float64(a) / float64(b), nil
}

// Unit tests
func TestCalculator_Add(t *testing.T) {
    calc := &Calculator{}
    
    tests := []struct {
        name     string
        a        int
        b        int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -2, -3, -5},
        {"mixed numbers", -2, 3, 1},
        {"zero", 0, 5, 5},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := calc.Add(tt.a, tt.b)
            assert.Equal(t, tt.expected, result)
        })
    }
}

func TestCalculator_Divide(t *testing.T) {
    calc := &Calculator{}
    
    t.Run("valid division", func(t *testing.T) {
        result, err := calc.Divide(10, 2)
        assert.NoError(t, err)
        assert.Equal(t, 5.0, result)
    })
    
    t.Run("division by zero", func(t *testing.T) {
        result, err := calc.Divide(10, 0)
        assert.Error(t, err)
        assert.Equal(t, 0.0, result)
        assert.Contains(t, err.Error(), "division by zero")
    })
}

// Mock implementation
type MockCalculator struct {
    mock.Mock
}

func (m *MockCalculator) Add(a, b int) int {
    args := m.Called(a, b)
    return args.Int(0)
}

func TestServiceWithMock(t *testing.T) {
    mockCalc := new(MockCalculator)
    mockCalc.On("Add", 2, 3).Return(5)
    
    service := &Service{calc: mockCalc}
    result := service.Process(2, 3)
    
    assert.Equal(t, 10, result) // 5 * 2
    mockCalc.AssertExpectations(t)
}
```

#### Node.js Testing Implementation
```javascript
const assert = require('assert');
const { describe, it, beforeEach, afterEach } = require('mocha');
const sinon = require('sinon');

class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
    
    multiply(a, b) {
        return a * b;
    }
    
    divide(a, b) {
        if (b === 0) {
            throw new Error('Division by zero');
        }
        return a / b;
    }
}

describe('Calculator', () => {
    let calculator;
    
    beforeEach(() => {
        calculator = new Calculator();
    });
    
    describe('add', () => {
        it('should add positive numbers', () => {
            const result = calculator.add(2, 3);
            assert.strictEqual(result, 5);
        });
        
        it('should add negative numbers', () => {
            const result = calculator.add(-2, -3);
            assert.strictEqual(result, -5);
        });
        
        it('should add mixed numbers', () => {
            const result = calculator.add(-2, 3);
            assert.strictEqual(result, 1);
        });
    });
    
    describe('divide', () => {
        it('should divide valid numbers', () => {
            const result = calculator.divide(10, 2);
            assert.strictEqual(result, 5);
        });
        
        it('should throw error for division by zero', () => {
            assert.throws(() => {
                calculator.divide(10, 0);
            }, Error, 'Division by zero');
        });
    });
});

// Mock testing
describe('Service with Mock', () => {
    let service;
    let mockCalculator;
    
    beforeEach(() => {
        mockCalculator = sinon.createStubInstance(Calculator);
        service = new Service(mockCalculator);
    });
    
    it('should process with mocked calculator', () => {
        mockCalculator.add.returns(5);
        
        const result = service.process(2, 3);
        
        assert.strictEqual(result, 10);
        sinon.assert.calledWith(mockCalculator.add, 2, 3);
    });
});
```

## Integration Testing

### 1. Database Integration Tests

#### Go Database Testing
```go
package main

import (
    "database/sql"
    "testing"
    "github.com/DATA-DOG/go-sqlmock"
    "github.com/stretchr/testify/assert"
)

func TestUserRepository_CreateUser(t *testing.T) {
    db, mock, err := sqlmock.New()
    assert.NoError(t, err)
    defer db.Close()
    
    repo := &UserRepository{db: db}
    
    mock.ExpectExec("INSERT INTO users").
        WithArgs("john@example.com", "John Doe").
        WillReturnResult(sqlmock.NewResult(1, 1))
    
    user := &User{
        Email: "john@example.com",
        Name:  "John Doe",
    }
    
    err = repo.CreateUser(user)
    assert.NoError(t, err)
    assert.NoError(t, mock.ExpectationsWereMet())
}

func TestUserRepository_GetUser(t *testing.T) {
    db, mock, err := sqlmock.New()
    assert.NoError(t, err)
    defer db.Close()
    
    repo := &UserRepository{db: db}
    
    rows := sqlmock.NewRows([]string{"id", "email", "name"}).
        AddRow(1, "john@example.com", "John Doe")
    
    mock.ExpectQuery("SELECT id, email, name FROM users WHERE id = ?").
        WithArgs(1).
        WillReturnRows(rows)
    
    user, err := repo.GetUser(1)
    assert.NoError(t, err)
    assert.Equal(t, "john@example.com", user.Email)
    assert.NoError(t, mock.ExpectationsWereMet())
}
```

### 2. API Integration Tests

#### HTTP API Testing
```go
package main

import (
    "net/http"
    "net/http/httptest"
    "testing"
    "github.com/stretchr/testify/assert"
)

func TestAPI_GetUser(t *testing.T) {
    router := setupRouter()
    
    req, _ := http.NewRequest("GET", "/users/1", nil)
    w := httptest.NewRecorder()
    
    router.ServeHTTP(w, req)
    
    assert.Equal(t, 200, w.Code)
    assert.Contains(t, w.Body.String(), "john@example.com")
}

func TestAPI_CreateUser(t *testing.T) {
    router := setupRouter()
    
    jsonStr := `{"email":"jane@example.com","name":"Jane Doe"}`
    req, _ := http.NewRequest("POST", "/users", strings.NewReader(jsonStr))
    req.Header.Set("Content-Type", "application/json")
    
    w := httptest.NewRecorder()
    router.ServeHTTP(w, req)
    
    assert.Equal(t, 201, w.Code)
    assert.Contains(t, w.Body.String(), "jane@example.com")
}
```

## Test-Driven Development (TDD)

### 1. TDD Cycle Implementation

#### TDD Example
```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
)

// Step 1: Write failing test
func TestStringCalculator_Add(t *testing.T) {
    calc := &StringCalculator{}
    
    t.Run("empty string returns zero", func(t *testing.T) {
        result := calc.Add("")
        assert.Equal(t, 0, result)
    })
    
    t.Run("single number returns that number", func(t *testing.T) {
        result := calc.Add("5")
        assert.Equal(t, 5, result)
    })
    
    t.Run("two numbers returns sum", func(t *testing.T) {
        result := calc.Add("1,2")
        assert.Equal(t, 3, result)
    })
}

// Step 2: Write minimal code to pass
type StringCalculator struct{}

func (sc *StringCalculator) Add(numbers string) int {
    if numbers == "" {
        return 0
    }
    
    if !strings.Contains(numbers, ",") {
        num, _ := strconv.Atoi(numbers)
        return num
    }
    
    parts := strings.Split(numbers, ",")
    sum := 0
    for _, part := range parts {
        num, _ := strconv.Atoi(part)
        sum += num
    }
    return sum
}

// Step 3: Refactor and add more tests
func TestStringCalculator_Add_Advanced(t *testing.T) {
    calc := &StringCalculator{}
    
    t.Run("multiple numbers", func(t *testing.T) {
        result := calc.Add("1,2,3,4")
        assert.Equal(t, 10, result)
    })
    
    t.Run("newline delimiter", func(t *testing.T) {
        result := calc.Add("1\n2,3")
        assert.Equal(t, 6, result)
    })
}
```

## Behavior-Driven Development (BDD)

### 1. BDD Framework

#### Go BDD Implementation
```go
package main

import (
    "testing"
    "github.com/DATA-DOG/godog"
)

func TestFeatures(t *testing.T) {
    suite := godog.TestSuite{
        ScenarioInitializer: InitializeScenario,
        Options: &godog.Options{
            Format: "pretty",
            Paths:  []string{"features"},
        },
    }
    
    if suite.Run() != 0 {
        t.Fatal("non-zero status returned, failed to run feature tests")
    }
}

func InitializeScenario(ctx *godog.ScenarioContext) {
    var calc *StringCalculator
    
    ctx.Before(func(ctx context.Context, sc *godog.Scenario) (context.Context, error) {
        calc = &StringCalculator{}
        return ctx, nil
    })
    
    ctx.Step(`^I have a calculator$`, func() error {
        return nil
    })
    
    ctx.Step(`^I add "([^"]*)"$`, func(numbers string) error {
        calc.Add(numbers)
        return nil
    })
    
    ctx.Step(`^the result should be (\d+)$`, func(expected int) error {
        if calc.Result() != expected {
            return fmt.Errorf("expected %d, got %d", expected, calc.Result())
        }
        return nil
    })
}
```

## Follow-up Questions

### 1. Unit Testing
**Q: What's the difference between unit tests and integration tests?**
A: Unit tests test individual components in isolation, while integration tests verify that multiple components work together correctly.

### 2. TDD
**Q: What are the benefits of Test-Driven Development?**
A: TDD improves code quality, reduces bugs, provides better documentation, and encourages better design through testability.

### 3. BDD
**Q: How does BDD differ from TDD?**
A: BDD focuses on behavior and user stories, using natural language, while TDD focuses on technical implementation and unit tests.

## Sources

### Books
- **Test Driven Development** by Kent Beck
- **Growing Object-Oriented Software** by Steve Freeman
- **The Art of Unit Testing** by Roy Osherove

### Online Resources
- **Jest Documentation**: JavaScript testing framework
- **Go Testing**: Official Go testing documentation
- **Cucumber**: BDD framework documentation

## Projects

### 1. Testing Framework
**Objective**: Build a custom testing framework
**Requirements**: Test discovery, assertion library, mocking, reporting
**Deliverables**: Working testing framework with documentation

### 2. Test Automation
**Objective**: Create test automation system
**Requirements**: CI/CD integration, parallel execution, reporting
**Deliverables**: Test automation platform

### 3. BDD Tool
**Objective**: Implement BDD testing tool
**Requirements**: Gherkin parser, step definitions, reporting
**Deliverables**: BDD testing tool with GUI

---

**Next**: [Phase 1](../../../README.md) | **Previous**: [Version Control](version-control-git.md/) | **Up**: [Phase 0](README.md/)
