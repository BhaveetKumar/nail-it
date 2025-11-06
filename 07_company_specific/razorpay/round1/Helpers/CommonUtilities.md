---
# Auto-generated front matter
Title: Commonutilities
LastUpdated: 2025-11-06T20:45:58.616528
Tags: []
Status: draft
---

# Common Utilities for Go Backend Services

This guide provides reusable utility functions and patterns commonly used in Go backend services, with a focus on fintech and payment applications.

## Error Handling Utilities

### 1. Custom Error Types

```go
package errors

import (
    "fmt"
    "net/http"
)

// Custom error types for different scenarios
type PaymentError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Details string `json:"details,omitempty"`
}

func (e *PaymentError) Error() string {
    return fmt.Sprintf("payment error [%s]: %s", e.Code, e.Message)
}

// Predefined error types
var (
    ErrPaymentNotFound     = &PaymentError{Code: "PAYMENT_NOT_FOUND", Message: "Payment not found"}
    ErrInvalidAmount       = &PaymentError{Code: "INVALID_AMOUNT", Message: "Invalid payment amount"}
    ErrInsufficientFunds   = &PaymentError{Code: "INSUFFICIENT_FUNDS", Message: "Insufficient funds"}
    ErrPaymentExpired      = &PaymentError{Code: "PAYMENT_EXPIRED", Message: "Payment has expired"}
    ErrDuplicatePayment    = &PaymentError{Code: "DUPLICATE_PAYMENT", Message: "Duplicate payment detected"}
    ErrPaymentGatewayError = &PaymentError{Code: "GATEWAY_ERROR", Message: "Payment gateway error"}
)

// Error with details
func NewPaymentError(code, message, details string) *PaymentError {
    return &PaymentError{
        Code:    code,
        Message: message,
        Details: details,
    }
}

// HTTP status code mapping
func (e *PaymentError) HTTPStatus() int {
    switch e.Code {
    case "PAYMENT_NOT_FOUND":
        return http.StatusNotFound
    case "INVALID_AMOUNT", "INSUFFICIENT_FUNDS":
        return http.StatusBadRequest
    case "PAYMENT_EXPIRED":
        return http.StatusGone
    case "DUPLICATE_PAYMENT":
        return http.StatusConflict
    case "GATEWAY_ERROR":
        return http.StatusBadGateway
    default:
        return http.StatusInternalServerError
    }
}
```

### 2. Error Wrapping and Context

```go
package errors

import (
    "fmt"
    "runtime"
    "strings"
)

// Error with stack trace
type StackError struct {
    Err   error
    Stack []string
}

func (e *StackError) Error() string {
    return fmt.Sprintf("%v\nStack trace:\n%s", e.Err, strings.Join(e.Stack, "\n"))
}

// Wrap error with stack trace
func WrapWithStack(err error) error {
    if err == nil {
        return nil
    }

    var stack []string
    for i := 0; i < 10; i++ {
        _, file, line, ok := runtime.Caller(i + 2)
        if !ok {
            break
        }
        stack = append(stack, fmt.Sprintf("  %s:%d", file, line))
    }

    return &StackError{
        Err:   err,
        Stack: stack,
    }
}

// Error with context
type ContextError struct {
    Err     error
    Context map[string]interface{}
}

func (e *ContextError) Error() string {
    return fmt.Sprintf("%v (context: %+v)", e.Err, e.Context)
}

// Wrap error with context
func WrapWithContext(err error, context map[string]interface{}) error {
    if err == nil {
        return nil
    }

    return &ContextError{
        Err:     err,
        Context: context,
    }
}
```

## Validation Utilities

### 1. Payment Validation

```go
package validation

import (
    "errors"
    "regexp"
    "strings"
    "unicode"
)

// Payment amount validation
func ValidateAmount(amount float64) error {
    if amount <= 0 {
        return errors.New("amount must be positive")
    }

    if amount > 1000000 {
        return errors.New("amount exceeds maximum limit")
    }

    // Check for reasonable precision (2 decimal places)
    if amount != float64(int(amount*100))/100 {
        return errors.New("amount precision exceeds 2 decimal places")
    }

    return nil
}

// Currency validation
func ValidateCurrency(currency string) error {
    validCurrencies := map[string]bool{
        "USD": true, "EUR": true, "GBP": true, "JPY": true,
        "INR": true, "CAD": true, "AUD": true, "CHF": true,
    }

    if !validCurrencies[strings.ToUpper(currency)] {
        return errors.New("invalid currency code")
    }

    return nil
}

// Payment ID validation
func ValidatePaymentID(paymentID string) error {
    if len(paymentID) == 0 {
        return errors.New("payment ID cannot be empty")
    }

    if len(paymentID) > 50 {
        return errors.New("payment ID too long")
    }

    // Check for valid characters (alphanumeric and hyphens)
    validIDRegex := regexp.MustCompile(`^[a-zA-Z0-9\-_]+$`)
    if !validIDRegex.MatchString(paymentID) {
        return errors.New("payment ID contains invalid characters")
    }

    return nil
}

// Email validation
func ValidateEmail(email string) error {
    if len(email) == 0 {
        return errors.New("email cannot be empty")
    }

    emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
    if !emailRegex.MatchString(email) {
        return errors.New("invalid email format")
    }

    return nil
}

// Phone number validation
func ValidatePhoneNumber(phone string) error {
    if len(phone) == 0 {
        return errors.New("phone number cannot be empty")
    }

    // Remove all non-digit characters
    digits := ""
    for _, r := range phone {
        if unicode.IsDigit(r) {
            digits += string(r)
        }
    }

    if len(digits) < 10 || len(digits) > 15 {
        return errors.New("phone number must be between 10 and 15 digits")
    }

    return nil
}
```

### 2. Generic Validation

```go
package validation

import (
    "reflect"
    "strings"
)

// Required field validation
func ValidateRequired(value interface{}, fieldName string) error {
    if value == nil {
        return fmt.Errorf("%s is required", fieldName)
    }

    switch v := value.(type) {
    case string:
        if strings.TrimSpace(v) == "" {
            return fmt.Errorf("%s cannot be empty", fieldName)
        }
    case []string:
        if len(v) == 0 {
            return fmt.Errorf("%s cannot be empty", fieldName)
        }
    case map[string]interface{}:
        if len(v) == 0 {
            return fmt.Errorf("%s cannot be empty", fieldName)
        }
    }

    return nil
}

// Length validation
func ValidateLength(value string, min, max int, fieldName string) error {
    length := len(strings.TrimSpace(value))
    if length < min {
        return fmt.Errorf("%s must be at least %d characters", fieldName, min)
    }
    if length > max {
        return fmt.Errorf("%s must be at most %d characters", fieldName, max)
    }
    return nil
}

// Range validation
func ValidateRange(value float64, min, max float64, fieldName string) error {
    if value < min {
        return fmt.Errorf("%s must be at least %.2f", fieldName, min)
    }
    if value > max {
        return fmt.Errorf("%s must be at most %.2f", fieldName, max)
    }
    return nil
}
```

## String Utilities

### 1. String Manipulation

```go
package strings

import (
    "crypto/rand"
    "math/big"
    "strings"
    "unicode"
)

// Generate random string
func RandomString(length int) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := make([]byte, length)

    for i := range result {
        num, _ := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
        result[i] = charset[num.Int64()]
    }

    return string(result)
}

// Generate UUID-like string
func GenerateID() string {
    return RandomString(32)
}

// Truncate string with ellipsis
func TruncateString(s string, maxLength int) string {
    if len(s) <= maxLength {
        return s
    }

    if maxLength <= 3 {
        return s[:maxLength]
    }

    return s[:maxLength-3] + "..."
}

// Capitalize first letter
func Capitalize(s string) string {
    if len(s) == 0 {
        return s
    }

    runes := []rune(s)
    runes[0] = unicode.ToUpper(runes[0])
    return string(runes)
}

// Convert to title case
func ToTitleCase(s string) string {
    words := strings.Fields(s)
    for i, word := range words {
        words[i] = Capitalize(strings.ToLower(word))
    }
    return strings.Join(words, " ")
}

// Mask sensitive data
func MaskSensitive(s string, visibleChars int) string {
    if len(s) <= visibleChars {
        return strings.Repeat("*", len(s))
    }

    return s[:visibleChars] + strings.Repeat("*", len(s)-visibleChars)
}

// Mask email
func MaskEmail(email string) string {
    parts := strings.Split(email, "@")
    if len(parts) != 2 {
        return strings.Repeat("*", len(email))
    }

    username := parts[0]
    domain := parts[1]

    if len(username) <= 2 {
        return strings.Repeat("*", len(username)) + "@" + domain
    }

    return username[:2] + strings.Repeat("*", len(username)-2) + "@" + domain
}
```

### 2. String Formatting

```go
package strings

import (
    "fmt"
    "strconv"
    "strings"
)

// Format currency
func FormatCurrency(amount float64, currency string) string {
    switch currency {
    case "USD", "CAD", "AUD":
        return fmt.Sprintf("$%.2f", amount)
    case "EUR":
        return fmt.Sprintf("€%.2f", amount)
    case "GBP":
        return fmt.Sprintf("£%.2f", amount)
    case "JPY":
        return fmt.Sprintf("¥%.0f", amount)
    case "INR":
        return fmt.Sprintf("₹%.2f", amount)
    default:
        return fmt.Sprintf("%.2f %s", amount, currency)
    }
}

// Format phone number
func FormatPhoneNumber(phone string) string {
    // Remove all non-digit characters
    digits := ""
    for _, r := range phone {
        if r >= '0' && r <= '9' {
            digits += string(r)
        }
    }

    if len(digits) == 10 {
        return fmt.Sprintf("(%s) %s-%s", digits[:3], digits[3:6], digits[6:])
    }

    if len(digits) == 11 && digits[0] == '1' {
        return fmt.Sprintf("+1 (%s) %s-%s", digits[1:4], digits[4:7], digits[7:])
    }

    return phone // Return original if can't format
}

// Format file size
func FormatFileSize(bytes int64) string {
    const unit = 1024
    if bytes < unit {
        return fmt.Sprintf("%d B", bytes)
    }

    div, exp := int64(unit), 0
    for n := bytes / unit; n >= unit; n /= unit {
        div *= unit
        exp++
    }

    units := []string{"KB", "MB", "GB", "TB", "PB"}
    return fmt.Sprintf("%.1f %s", float64(bytes)/float64(div), units[exp])
}
```

## Time Utilities

### 1. Time Manipulation

```go
package time

import (
    "time"
)

// Parse common date formats
func ParseDate(dateStr string) (time.Time, error) {
    formats := []string{
        "2006-01-02",
        "01/02/2006",
        "2006-01-02 15:04:05",
        "2006-01-02T15:04:05Z",
        "2006-01-02T15:04:05.000Z",
    }

    for _, format := range formats {
        if t, err := time.Parse(format, dateStr); err == nil {
            return t, nil
        }
    }

    return time.Time{}, fmt.Errorf("unable to parse date: %s", dateStr)
}

// Get start of day
func StartOfDay(t time.Time) time.Time {
    return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

// Get end of day
func EndOfDay(t time.Time) time.Time {
    return time.Date(t.Year(), t.Month(), t.Day(), 23, 59, 59, 999999999, t.Location())
}

// Get start of month
func StartOfMonth(t time.Time) time.Time {
    return time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, t.Location())
}

// Get end of month
func EndOfMonth(t time.Time) time.Time {
    return StartOfMonth(t).AddDate(0, 1, 0).Add(-time.Nanosecond)
}

// Check if time is in business hours
func IsBusinessHours(t time.Time) bool {
    hour := t.Hour()
    weekday := t.Weekday()

    // Monday to Friday, 9 AM to 5 PM
    return weekday >= time.Monday && weekday <= time.Friday && hour >= 9 && hour < 17
}

// Get next business day
func NextBusinessDay(t time.Time) time.Time {
    next := t.AddDate(0, 0, 1)
    for !IsBusinessHours(next) {
        next = next.AddDate(0, 0, 1)
    }
    return next
}
```

### 2. Time Formatting

```go
package time

import (
    "fmt"
    "time"
)

// Format duration in human-readable format
func FormatDuration(d time.Duration) string {
    if d < time.Minute {
        return fmt.Sprintf("%.0f seconds", d.Seconds())
    }

    if d < time.Hour {
        minutes := int(d.Minutes())
        seconds := int(d.Seconds()) % 60
        if seconds == 0 {
            return fmt.Sprintf("%d minutes", minutes)
        }
        return fmt.Sprintf("%d minutes %d seconds", minutes, seconds)
    }

    if d < 24*time.Hour {
        hours := int(d.Hours())
        minutes := int(d.Minutes()) % 60
        if minutes == 0 {
            return fmt.Sprintf("%d hours", hours)
        }
        return fmt.Sprintf("%d hours %d minutes", hours, minutes)
    }

    days := int(d.Hours() / 24)
    hours := int(d.Hours()) % 24
    if hours == 0 {
        return fmt.Sprintf("%d days", days)
    }
    return fmt.Sprintf("%d days %d hours", days, hours)
}

// Format relative time
func FormatRelativeTime(t time.Time) string {
    now := time.Now()
    diff := now.Sub(t)

    if diff < time.Minute {
        return "just now"
    }

    if diff < time.Hour {
        minutes := int(diff.Minutes())
        if minutes == 1 {
            return "1 minute ago"
        }
        return fmt.Sprintf("%d minutes ago", minutes)
    }

    if diff < 24*time.Hour {
        hours := int(diff.Hours())
        if hours == 1 {
            return "1 hour ago"
        }
        return fmt.Sprintf("%d hours ago", hours)
    }

    if diff < 7*24*time.Hour {
        days := int(diff.Hours() / 24)
        if days == 1 {
            return "1 day ago"
        }
        return fmt.Sprintf("%d days ago", days)
    }

    return t.Format("Jan 2, 2006")
}
```

## HTTP Utilities

### 1. HTTP Client

```go
package http

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

// HTTP client with timeout and retry
type Client struct {
    client  *http.Client
    timeout time.Duration
    retries int
}

func NewClient(timeout time.Duration, retries int) *Client {
    return &Client{
        client: &http.Client{
            Timeout: timeout,
        },
        timeout: timeout,
        retries: retries,
    }
}

// GET request
func (c *Client) Get(ctx context.Context, url string, headers map[string]string) (*http.Response, error) {
    return c.doRequest(ctx, "GET", url, nil, headers)
}

// POST request
func (c *Client) Post(ctx context.Context, url string, body interface{}, headers map[string]string) (*http.Response, error) {
    jsonBody, err := json.Marshal(body)
    if err != nil {
        return nil, err
    }

    return c.doRequest(ctx, "POST", url, bytes.NewBuffer(jsonBody), headers)
}

// PUT request
func (c *Client) Put(ctx context.Context, url string, body interface{}, headers map[string]string) (*http.Response, error) {
    jsonBody, err := json.Marshal(body)
    if err != nil {
        return nil, err
    }

    return c.doRequest(ctx, "PUT", url, bytes.NewBuffer(jsonBody), headers)
}

// DELETE request
func (c *Client) Delete(ctx context.Context, url string, headers map[string]string) (*http.Response, error) {
    return c.doRequest(ctx, "DELETE", url, nil, headers)
}

// Do request with retry logic
func (c *Client) doRequest(ctx context.Context, method, url string, body io.Reader, headers map[string]string) (*http.Response, error) {
    var lastErr error

    for i := 0; i <= c.retries; i++ {
        req, err := http.NewRequestWithContext(ctx, method, url, body)
        if err != nil {
            return nil, err
        }

        // Set headers
        for key, value := range headers {
            req.Header.Set(key, value)
        }

        // Set content type for JSON
        if body != nil {
            req.Header.Set("Content-Type", "application/json")
        }

        resp, err := c.client.Do(req)
        if err != nil {
            lastErr = err
            if i < c.retries {
                time.Sleep(time.Duration(i+1) * time.Second)
                continue
            }
            return nil, err
        }

        // Check for retryable status codes
        if resp.StatusCode >= 500 && i < c.retries {
            resp.Body.Close()
            time.Sleep(time.Duration(i+1) * time.Second)
            continue
        }

        return resp, nil
    }

    return nil, lastErr
}
```

### 2. HTTP Response Utilities

```go
package http

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

// Parse JSON response
func ParseJSONResponse(resp *http.Response, v interface{}) error {
    defer resp.Body.Close()

    if resp.StatusCode >= 400 {
        return fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
    }

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return err
    }

    return json.Unmarshal(body, v)
}

// Parse error response
func ParseErrorResponse(resp *http.Response) error {
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
    }

    var errorResp struct {
        Error   string `json:"error"`
        Message string `json:"message"`
        Code    string `json:"code"`
    }

    if err := json.Unmarshal(body, &errorResp); err != nil {
        return fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
    }

    if errorResp.Message != "" {
        return fmt.Errorf("API error: %s", errorResp.Message)
    }

    return fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
}
```

## Database Utilities

### 1. Database Connection

```go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"

    _ "github.com/lib/pq"
)

// Database configuration
type Config struct {
    Host     string
    Port     int
    Name     string
    User     string
    Password string
    SSLMode  string
    MaxConns int
    MaxIdle  int
}

// Database connection manager
type Manager struct {
    db *sql.DB
}

func NewManager(config Config) (*Manager, error) {
    dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
        config.Host, config.Port, config.User, config.Password, config.Name, config.SSLMode)

    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }

    // Configure connection pool
    db.SetMaxOpenConns(config.MaxConns)
    db.SetMaxIdleConns(config.MaxIdle)
    db.SetConnMaxLifetime(time.Hour)

    // Test connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        return nil, err
    }

    return &Manager{db: db}, nil
}

func (m *Manager) DB() *sql.DB {
    return m.db
}

func (m *Manager) Close() error {
    return m.db.Close()
}
```

### 2. Database Transaction Utilities

```go
package database

import (
    "context"
    "database/sql"
    "fmt"
)

// Transaction manager
type TransactionManager struct {
    db *sql.DB
}

func NewTransactionManager(db *sql.DB) *TransactionManager {
    return &TransactionManager{db: db}
}

// Execute function within transaction
func (tm *TransactionManager) WithTransaction(ctx context.Context, fn func(*sql.Tx) error) error {
    tx, err := tm.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    defer func() {
        if p := recover(); p != nil {
            tx.Rollback()
            panic(p)
        } else if err != nil {
            tx.Rollback()
        } else {
            err = tx.Commit()
        }
    }()

    err = fn(tx)
    return err
}

// Execute multiple operations in transaction
func (tm *TransactionManager) ExecuteInTransaction(ctx context.Context, operations ...func(*sql.Tx) error) error {
    return tm.WithTransaction(ctx, func(tx *sql.Tx) error {
        for _, op := range operations {
            if err := op(tx); err != nil {
                return err
            }
        }
        return nil
    })
}
```

## Logging Utilities

### 1. Structured Logging

```go
package logging

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "os"
    "time"
)

// Log levels
type Level int

const (
    DEBUG Level = iota
    INFO
    WARN
    ERROR
    FATAL
)

// Logger interface
type Logger interface {
    Debug(msg string, fields ...Field)
    Info(msg string, fields ...Field)
    Warn(msg string, fields ...Field)
    Error(msg string, fields ...Field)
    Fatal(msg string, fields ...Field)
}

// Log field
type Field struct {
    Key   string
    Value interface{}
}

// Structured logger
type StructuredLogger struct {
    level  Level
    logger *log.Logger
}

func NewStructuredLogger(level Level) *StructuredLogger {
    return &StructuredLogger{
        level:  level,
        logger: log.New(os.Stdout, "", 0),
    }
}

func (l *StructuredLogger) Debug(msg string, fields ...Field) {
    l.log(DEBUG, msg, fields...)
}

func (l *StructuredLogger) Info(msg string, fields ...Field) {
    l.log(INFO, msg, fields...)
}

func (l *StructuredLogger) Warn(msg string, fields ...Field) {
    l.log(WARN, msg, fields...)
}

func (l *StructuredLogger) Error(msg string, fields ...Field) {
    l.log(ERROR, msg, fields...)
}

func (l *StructuredLogger) Fatal(msg string, fields ...Field) {
    l.log(FATAL, msg, fields...)
    os.Exit(1)
}

func (l *StructuredLogger) log(level Level, msg string, fields ...Field) {
    if level < l.level {
        return
    }

    entry := map[string]interface{}{
        "timestamp": time.Now().UTC().Format(time.RFC3339),
        "level":     l.levelString(level),
        "message":   msg,
    }

    for _, field := range fields {
        entry[field.Key] = field.Value
    }

    jsonData, _ := json.Marshal(entry)
    l.logger.Println(string(jsonData))
}

func (l *StructuredLogger) levelString(level Level) string {
    switch level {
    case DEBUG:
        return "DEBUG"
    case INFO:
        return "INFO"
    case WARN:
        return "WARN"
    case ERROR:
        return "ERROR"
    case FATAL:
        return "FATAL"
    default:
        return "UNKNOWN"
    }
}

// Helper functions for creating fields
func String(key, value string) Field {
    return Field{Key: key, Value: value}
}

func Int(key string, value int) Field {
    return Field{Key: key, Value: value}
}

func Float64(key string, value float64) Field {
    return Field{Key: key, Value: value}
}

func Bool(key string, value bool) Field {
    return Field{Key: key, Value: value}
}

func Error(err error) Field {
    return Field{Key: "error", Value: err.Error()}
}
```

## Configuration Utilities

### 1. Environment Configuration

```go
package config

import (
    "os"
    "strconv"
    "time"
)

// Environment variable utilities
func GetString(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func GetInt(key string, defaultValue int) int {
    if value := os.Getenv(key); value != "" {
        if intValue, err := strconv.Atoi(value); err == nil {
            return intValue
        }
    }
    return defaultValue
}

func GetFloat64(key string, defaultValue float64) float64 {
    if value := os.Getenv(key); value != "" {
        if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
            return floatValue
        }
    }
    return defaultValue
}

func GetBool(key string, defaultValue bool) bool {
    if value := os.Getenv(key); value != "" {
        if boolValue, err := strconv.ParseBool(value); err == nil {
            return boolValue
        }
    }
    return defaultValue
}

func GetDuration(key string, defaultValue time.Duration) time.Duration {
    if value := os.Getenv(key); value != "" {
        if duration, err := time.ParseDuration(value); err == nil {
            return duration
        }
    }
    return defaultValue
}
```

This comprehensive utilities guide provides reusable functions and patterns commonly used in Go backend services, with particular emphasis on fintech and payment applications. Use these utilities to improve code reusability, maintainability, and consistency across your applications.
