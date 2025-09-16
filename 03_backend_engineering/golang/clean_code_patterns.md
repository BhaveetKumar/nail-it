# Clean Code Patterns - Robert C. Martin Principles

## Table of Contents
1. [Introduction](#introduction/)
2. [Meaningful Names](#meaningful-names/)
3. [Functions](#functions/)
4. [Comments](#comments/)
5. [Formatting](#formatting/)
6. [Error Handling](#error-handling/)
7. [Classes and Data Structures](#classes-and-data-structures/)
8. [System Design](#system-design/)
9. [Concurrency](#concurrency/)
10. [Golang Best Practices](#golang-best-practices/)

## Introduction

This guide is based on Robert C. Martin's "Clean Code" principles, adapted for Golang. It focuses on writing code that is readable, maintainable, and professional.

### Core Principles
- **Readability**: Code should be self-documenting
- **Simplicity**: Keep functions and classes small and focused
- **Maintainability**: Code should be easy to modify and extend
- **Testability**: Code should be easy to test
- **Professionalism**: Write code as if your job depends on it

## Meaningful Names

### Use Intention-Revealing Names
```go
// Bad: Unclear purpose
func calc(a, b int) int {
    return a + b
}

// Good: Clear purpose
func calculateTotalPrice(unitPrice, quantity int) int {
    return unitPrice * quantity
}

// Bad: Vague variable names
func processData(d []string) {
    for i := 0; i < len(d); i++ {
        // process d[i]
    }
}

// Good: Descriptive variable names
func processUserEmails(userEmails []string) {
    for index := 0; index < len(userEmails); index++ {
        // process userEmails[index]
    }
}
```

### Avoid Disinformation
```go
// Bad: Misleading names
type UserList struct {
    // This is actually a map, not a list
    users map[string]User
}

// Good: Accurate names
type UserRegistry struct {
    users map[string]User
}

// Bad: Similar names that are confusing
func getAccount()
func getAccountData()
func getAccountInfo()

// Good: Distinct names
func getAccount()
func getAccountDetails()
func getAccountSummary()
```

### Make Meaningful Distinctions
```go
// Bad: No meaningful distinction
func copyUserData(user1, user2 User) {}
func copyUserInfo(user1, user2 User) {}

// Good: Clear distinction
func copyUserData(source, destination User) {}
func copyUserProfile(source, destination User) {}

// Bad: Redundant information
type UserData struct {
    userID   string
    userName string
    userEmail string
}

// Good: Remove redundancy
type User struct {
    ID    string
    Name  string
    Email string
}
```

### Use Pronounceable Names
```go
// Bad: Unpronounceable abbreviations
type DtRcrd struct {
    genymdhms time.Time
    modymdhms time.Time
}

// Good: Pronounceable names
type Customer struct {
    generationTimestamp time.Time
    modificationTimestamp time.Time
}
```

### Use Searchable Names
```go
// Bad: Magic numbers
func calculateDiscount(price float64) float64 {
    return price * 0.15
}

// Good: Named constants
const (
    STANDARD_DISCOUNT_RATE = 0.15
)

func calculateDiscount(price float64) float64 {
    return price * STANDARD_DISCOUNT_RATE
}
```

## Functions

### Small Functions
```go
// Bad: Large function doing multiple things
func processOrder(order Order) error {
    // Validate order
    if order.ID == "" {
        return errors.New("order ID is required")
    }
    if order.CustomerID == "" {
        return errors.New("customer ID is required")
    }
    if len(order.Items) == 0 {
        return errors.New("order must have at least one item")
    }
    
    // Calculate total
    total := 0.0
    for _, item := range order.Items {
        total += item.Price * float64(item.Quantity)
    }
    
    // Apply discount
    if order.CustomerType == "PREMIUM" {
        total *= 0.9
    }
    
    // Save to database
    if err := saveOrder(order); err != nil {
        return err
    }
    
    // Send notification
    if err := sendOrderConfirmation(order); err != nil {
        return err
    }
    
    return nil
}

// Good: Small, focused functions
func processOrder(order Order) error {
    if err := validateOrder(order); err != nil {
        return err
    }
    
    total := calculateOrderTotal(order)
    order.Total = total
    
    if err := saveOrder(order); err != nil {
        return err
    }
    
    if err := sendOrderConfirmation(order); err != nil {
        return err
    }
    
    return nil
}

func validateOrder(order Order) error {
    if order.ID == "" {
        return errors.New("order ID is required")
    }
    if order.CustomerID == "" {
        return errors.New("customer ID is required")
    }
    if len(order.Items) == 0 {
        return errors.New("order must have at least one item")
    }
    return nil
}

func calculateOrderTotal(order Order) float64 {
    total := 0.0
    for _, item := range order.Items {
        total += item.Price * float64(item.Quantity)
    }
    
    if order.CustomerType == "PREMIUM" {
        total *= 0.9
    }
    
    return total
}
```

### Do One Thing
```go
// Bad: Function doing multiple things
func processUserRegistration(userData map[string]interface{}) error {
    // Validate data
    if userData["email"] == nil {
        return errors.New("email is required")
    }
    
    // Create user object
    user := User{
        Email: userData["email"].(string),
        Name:  userData["name"].(string),
    }
    
    // Save to database
    if err := saveUser(user); err != nil {
        return err
    }
    
    // Send welcome email
    if err := sendWelcomeEmail(user.Email); err != nil {
        return err
    }
    
    // Log registration
    log.Printf("User %s registered successfully", user.Email)
    
    return nil
}

// Good: Each function does one thing
func processUserRegistration(userData map[string]interface{}) error {
    user, err := createUserFromData(userData)
    if err != nil {
        return err
    }
    
    if err := saveUser(user); err != nil {
        return err
    }
    
    if err := sendWelcomeEmail(user.Email); err != nil {
        return err
    }
    
    logUserRegistration(user.Email)
    return nil
}

func createUserFromData(userData map[string]interface{}) (User, error) {
    if userData["email"] == nil {
        return User{}, errors.New("email is required")
    }
    
    return User{
        Email: userData["email"].(string),
        Name:  userData["name"].(string),
    }, nil
}

func logUserRegistration(email string) {
    log.Printf("User %s registered successfully", email)
}
```

### Use Descriptive Names
```go
// Bad: Unclear function names
func calc(a, b int) int
func proc(data []string) error
func get(d string) (string, error)

// Good: Descriptive function names
func calculateTotalPrice(unitPrice, quantity int) int
func processUserEmails(emails []string) error
func getUserByID(userID string) (User, error)
```

### Function Arguments
```go
// Bad: Too many arguments
func createUser(name, email, phone, address, city, state, country, zipCode string) error

// Good: Use struct for multiple arguments
type UserData struct {
    Name    string
    Email   string
    Phone   string
    Address Address
}

type Address struct {
    Street  string
    City    string
    State   string
    Country string
    ZipCode string
}

func createUser(userData UserData) error

// Bad: Boolean arguments
func sendEmail(to string, isHTML bool) error

// Good: Use separate functions
func sendTextEmail(to string) error
func sendHTMLEmail(to string) error
```

### Command Query Separation
```go
// Bad: Function that both queries and commands
func setUserActive(userID string) bool {
    user, err := getUserByID(userID)
    if err != nil {
        return false
    }
    
    user.IsActive = true
    saveUser(user)
    return true
}

// Good: Separate command and query
func activateUser(userID string) error {
    user, err := getUserByID(userID)
    if err != nil {
        return err
    }
    
    user.IsActive = true
    return saveUser(user)
}

func isUserActive(userID string) (bool, error) {
    user, err := getUserByID(userID)
    if err != nil {
        return false, err
    }
    
    return user.IsActive, nil
}
```

## Comments

### Don't Comment Bad Code
```go
// Bad: Commenting bad code
// Check if user is valid
if user != nil && user.ID != "" && user.Email != "" {
    // Process user
    processUser(user)
}

// Good: Write self-documenting code
if isValidUser(user) {
    processUser(user)
}

func isValidUser(user *User) bool {
    return user != nil && user.ID != "" && user.Email != ""
}
```

### Good Comments
```go
// Legal comment
// Copyright (C) 2024 Company Name. All rights reserved.

// Informative comment
// Returns the user's full name in the format "First Last"
func (u *User) GetFullName() string {
    return fmt.Sprintf("%s %s", u.FirstName, u.LastName)
}

// Warning comment
// WARNING: This function modifies the original slice
func reverseSlice(s []int) {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}

// TODO comment
// TODO: Implement caching for better performance
func getUserByID(userID string) (User, error) {
    // Implementation
}
```

### Bad Comments
```go
// Bad: Obvious comments
// Increment i by 1
i++

// Bad: Commented out code
// if user.IsActive {
//     processUser(user)
// }

// Bad: Misleading comments
// This function calculates the total price
func calculateDiscount(price float64) float64 {
    return price * 0.1
}
```

## Formatting

### Vertical Formatting
```go
// Bad: No vertical spacing
package main
import "fmt"
func main() {
    fmt.Println("Hello, World!")
}

// Good: Proper vertical spacing
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

### Horizontal Formatting
```go
// Bad: Long lines
func calculateTotalPrice(unitPrice, quantity int, discountRate float64, taxRate float64) float64 {
    return (float64(unitPrice) * float64(quantity) * (1 - discountRate)) * (1 + taxRate)
}

// Good: Break long lines
func calculateTotalPrice(
    unitPrice, quantity int,
    discountRate, taxRate float64,
) float64 {
    subtotal := float64(unitPrice) * float64(quantity)
    discountedSubtotal := subtotal * (1 - discountRate)
    return discountedSubtotal * (1 + taxRate)
}
```

### Indentation
```go
// Bad: Inconsistent indentation
func processUsers(users []User) {
for _, user := range users {
if user.IsActive {
    if user.HasPermission("admin") {
    processAdminUser(user)
    } else {
    processRegularUser(user)
    }
}
}
}

// Good: Consistent indentation
func processUsers(users []User) {
    for _, user := range users {
        if user.IsActive {
            if user.HasPermission("admin") {
                processAdminUser(user)
            } else {
                processRegularUser(user)
            }
        }
    }
}
```

## Error Handling

### Use Errors, Not Exceptions
```go
// Bad: Panic for error handling
func divide(a, b int) int {
    if b == 0 {
        panic("division by zero")
    }
    return a / b
}

// Good: Return error
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}
```

### Don't Ignore Errors
```go
// Bad: Ignoring errors
func processFile(filename string) {
    file, _ := os.Open(filename)
    defer file.Close()
    // Process file
}

// Good: Handle errors
func processFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return fmt.Errorf("failed to open file %s: %w", filename, err)
    }
    defer file.Close()
    
    // Process file
    return nil
}
```

### Use Wrapped Errors
```go
// Bad: Losing error context
func processUser(userID string) error {
    user, err := getUserByID(userID)
    if err != nil {
        return err
    }
    
    if err := validateUser(user); err != nil {
        return err
    }
    
    return saveUser(user)
}

// Good: Wrapped errors with context
func processUser(userID string) error {
    user, err := getUserByID(userID)
    if err != nil {
        return fmt.Errorf("failed to get user %s: %w", userID, err)
    }
    
    if err := validateUser(user); err != nil {
        return fmt.Errorf("user validation failed for %s: %w", userID, err)
    }
    
    if err := saveUser(user); err != nil {
        return fmt.Errorf("failed to save user %s: %w", userID, err)
    }
    
    return nil
}
```

### Custom Error Types
```go
// Custom error type
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error for field %s: %s", e.Field, e.Message)
}

// Usage
func validateUser(user User) error {
    if user.Email == "" {
        return &ValidationError{
            Field:   "email",
            Message: "email is required",
        }
    }
    
    if !isValidEmail(user.Email) {
        return &ValidationError{
            Field:   "email",
            Message: "invalid email format",
        }
    }
    
    return nil
}
```

## Classes and Data Structures

### Data Abstraction
```go
// Bad: Exposing implementation details
type Point struct {
    X float64
    Y float64
}

func (p *Point) GetX() float64 {
    return p.X
}

func (p *Point) SetX(x float64) {
    p.X = x
}

// Good: Abstract interface
type Point interface {
    GetX() float64
    GetY() float64
    DistanceTo(other Point) float64
}

type CartesianPoint struct {
    x, y float64
}

func (p *CartesianPoint) GetX() float64 {
    return p.x
}

func (p *CartesianPoint) GetY() float64 {
    return p.y
}

func (p *CartesianPoint) DistanceTo(other Point) float64 {
    dx := p.x - other.GetX()
    dy := p.y - other.GetY()
    return math.Sqrt(dx*dx + dy*dy)
}
```

### Law of Demeter
```go
// Bad: Violating Law of Demeter
func processOrder(order Order) {
    if order.Customer.Address.Country == "US" {
        // Process US order
    }
}

// Good: Following Law of Demeter
func processOrder(order Order) {
    if order.IsUSOrder() {
        // Process US order
    }
}

func (o *Order) IsUSOrder() bool {
    return o.Customer.GetCountry() == "US"
}

func (c *Customer) GetCountry() string {
    return c.Address.Country
}
```

### Single Responsibility Principle
```go
// Bad: Multiple responsibilities
type User struct {
    ID       string
    Name     string
    Email    string
    Password string
}

func (u *User) Save() error {
    // Database logic
}

func (u *User) SendEmail() error {
    // Email logic
}

func (u *User) Validate() error {
    // Validation logic
}

// Good: Single responsibility
type User struct {
    ID    string
    Name  string
    Email string
}

type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) Save(user User) error {
    // Database logic
}

type EmailService struct {
    smtpClient *smtp.Client
}

func (s *EmailService) SendEmail(to, subject, body string) error {
    // Email logic
}

type UserValidator struct{}

func (v *UserValidator) Validate(user User) error {
    // Validation logic
}
```

## System Design

### Dependency Inversion
```go
// Bad: High-level module depending on low-level module
type OrderService struct {
    db *sql.DB
}

func (s *OrderService) CreateOrder(order Order) error {
    // Direct database dependency
    _, err := s.db.Exec("INSERT INTO orders...", order.ID, order.CustomerID)
    return err
}

// Good: Both depend on abstraction
type OrderRepository interface {
    Save(order Order) error
    GetByID(id string) (Order, error)
}

type OrderService struct {
    repository OrderRepository
}

func (s *OrderService) CreateOrder(order Order) error {
    return s.repository.Save(order)
}

type SQLOrderRepository struct {
    db *sql.DB
}

func (r *SQLOrderRepository) Save(order Order) error {
    _, err := r.db.Exec("INSERT INTO orders...", order.ID, order.CustomerID)
    return err
}
```

### Interface Segregation
```go
// Bad: Fat interface
type UserService interface {
    CreateUser(user User) error
    GetUser(id string) (User, error)
    UpdateUser(user User) error
    DeleteUser(id string) error
    SendEmail(userID string, message string) error
    GenerateReport(userID string) (Report, error)
}

// Good: Segregated interfaces
type UserRepository interface {
    Create(user User) error
    GetByID(id string) (User, error)
    Update(user User) error
    Delete(id string) error
}

type EmailService interface {
    SendEmail(to, subject, body string) error
}

type ReportService interface {
    GenerateUserReport(userID string) (Report, error)
}
```

## Concurrency

### Thread Safety
```go
// Bad: Not thread-safe
type Counter struct {
    value int
}

func (c *Counter) Increment() {
    c.value++
}

func (c *Counter) GetValue() int {
    return c.value
}

// Good: Thread-safe
type Counter struct {
    value int
    mutex sync.RWMutex
}

func (c *Counter) Increment() {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    c.value++
}

func (c *Counter) GetValue() int {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    return c.value
}
```

### Avoid Shared Mutable State
```go
// Bad: Shared mutable state
var globalCounter int

func incrementCounter() {
    globalCounter++
}

// Good: Immutable data
type Counter struct {
    value int
}

func (c Counter) Increment() Counter {
    return Counter{value: c.value + 1}
}

func (c Counter) GetValue() int {
    return c.value
}
```

### Use Channels for Communication
```go
// Bad: Shared memory communication
type Worker struct {
    id       int
    shared   *SharedData
    mutex    *sync.Mutex
}

func (w *Worker) Process() {
    w.mutex.Lock()
    w.shared.value++
    w.mutex.Unlock()
}

// Good: Channel communication
type Worker struct {
    id      int
    input   <-chan Work
    output  chan<- Result
}

func (w *Worker) Process() {
    for work := range w.input {
        result := w.doWork(work)
        w.output <- result
    }
}
```

## Golang Best Practices

### Package Organization
```go
// Good: Clear package structure
package user

import (
    "context"
    "errors"
    "fmt"
)

// User represents a user in the system
type User struct {
    ID    string
    Name  string
    Email string
}

// Repository defines the interface for user data access
type Repository interface {
    Create(ctx context.Context, user User) error
    GetByID(ctx context.Context, id string) (User, error)
    Update(ctx context.Context, user User) error
    Delete(ctx context.Context, id string) error
}

// Service provides business logic for users
type Service struct {
    repo Repository
}

// NewService creates a new user service
func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

// CreateUser creates a new user
func (s *Service) CreateUser(ctx context.Context, user User) error {
    if err := s.validateUser(user); err != nil {
        return fmt.Errorf("user validation failed: %w", err)
    }
    
    return s.repo.Create(ctx, user)
}

func (s *Service) validateUser(user User) error {
    if user.Name == "" {
        return errors.New("name is required")
    }
    if user.Email == "" {
        return errors.New("email is required")
    }
    return nil
}
```

### Context Usage
```go
// Good: Proper context usage
func (s *UserService) GetUser(ctx context.Context, userID string) (User, error) {
    // Check if context is cancelled
    select {
    case <-ctx.Done():
        return User{}, ctx.Err()
    default:
    }
    
    // Pass context to repository
    return s.repository.GetByID(ctx, userID)
}

// Good: Context with timeout
func (s *UserService) GetUserWithTimeout(userID string, timeout time.Duration) (User, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    
    return s.GetUser(ctx, userID)
}
```

### Error Handling
```go
// Good: Consistent error handling
func (s *UserService) CreateUser(user User) error {
    if err := s.validateUser(user); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    if err := s.repository.Create(user); err != nil {
        return fmt.Errorf("failed to create user: %w", err)
    }
    
    if err := s.sendWelcomeEmail(user.Email); err != nil {
        // Log error but don't fail the operation
        log.Printf("failed to send welcome email to %s: %v", user.Email, err)
    }
    
    return nil
}
```

### Testing
```go
// Good: Testable code
func (s *UserService) CreateUser(user User) error {
    if err := s.validateUser(user); err != nil {
        return err
    }
    
    if err := s.repository.Create(user); err != nil {
        return err
    }
    
    return s.sendWelcomeEmail(user.Email)
}

// Test
func TestUserService_CreateUser(t *testing.T) {
    mockRepo := &MockUserRepository{}
    mockEmailService := &MockEmailService{}
    
    service := &UserService{
        repository: mockRepo,
        emailService: mockEmailService,
    }
    
    user := User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    err := service.CreateUser(user)
    
    assert.NoError(t, err)
    assert.True(t, mockRepo.CreateCalled)
    assert.True(t, mockEmailService.SendEmailCalled)
}
```

## Conclusion

Clean code principles are essential for writing maintainable, readable, and professional code. Key takeaways:

1. **Meaningful Names**: Use descriptive, intention-revealing names
2. **Small Functions**: Keep functions small and focused on one thing
3. **Good Comments**: Write self-documenting code, comment only when necessary
4. **Proper Formatting**: Use consistent formatting and spacing
5. **Error Handling**: Handle errors properly, don't ignore them
6. **Single Responsibility**: Each class/struct should have one reason to change
7. **Dependency Inversion**: Depend on abstractions, not concretions
8. **Thread Safety**: Write thread-safe code when dealing with concurrency
9. **Testing**: Write testable code and comprehensive tests
10. **Golang Best Practices**: Follow Go idioms and conventions

By following these principles, you can write code that is not only functional but also maintainable and professional.
