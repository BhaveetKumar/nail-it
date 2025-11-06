---
# Auto-generated front matter
Title: Api Design Guide
LastUpdated: 2025-11-06T20:45:58.299767
Tags: []
Status: draft
---

# 🌐 API Design Guide

> **Essential guide to designing and building RESTful and GraphQL APIs**

## 📚 Table of Contents

1. [API Design Principles](#api-design-principles)
2. [RESTful API Design](#restful-api-design)
3. [GraphQL API Design](#graphql-api-design)
4. [API Security](#api-security)
5. [API Testing](#api-testing)

---

## 🎯 API Design Principles

### Core Principles
- **Resource-Based**: URLs represent resources, not actions
- **HTTP Methods**: Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- **Stateless**: Each request contains all necessary information
- **Cacheable**: Responses should be cacheable when appropriate
- **Uniform Interface**: Consistent interface across all resources

### Best Practices
- **Use Nouns, Not Verbs**: `/users` not `/getUsers`
- **Use Plural Nouns**: `/users` not `/user`
- **Use Hierarchical URLs**: `/users/123/orders`
- **Use Query Parameters**: `/users?status=active&limit=10`
- **Use HTTP Status Codes**: Proper status codes for responses

---

## 🔗 RESTful API Design

### 1. Resource Endpoints

```go
// User Resource API
type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    Status    string    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

// GET /users
func (us *UserService) GetUsers(w http.ResponseWriter, r *http.Request) {
    // Parse query parameters
    status := r.URL.Query().Get("status")
    limitStr := r.URL.Query().Get("limit")
    offsetStr := r.URL.Query().Get("offset")
    
    // Filter and paginate users
    users := us.filterUsers(status, limitStr, offsetStr)
    
    response := map[string]interface{}{
        "data": users,
        "pagination": map[string]interface{}{
            "total":  len(users),
            "limit":  parseLimit(limitStr),
            "offset": parseOffset(offsetStr),
        },
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// GET /users/{id}
func (us *UserService) GetUser(w http.ResponseWriter, r *http.Request) {
    id := extractIDFromPath(r.URL.Path)
    user, exists := us.users[id]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// POST /users
func (us *UserService) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    
    // Validate request
    if req.Name == "" || req.Email == "" {
        http.Error(w, "Name and email are required", http.StatusBadRequest)
        return
    }
    
    // Create user
    user := &User{
        ID:        us.nextID,
        Name:      req.Name,
        Email:     req.Email,
        Status:    "active",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    us.users[us.nextID] = user
    us.nextID++
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}

// PUT /users/{id}
func (us *UserService) UpdateUser(w http.ResponseWriter, r *http.Request) {
    id := extractIDFromPath(r.URL.Path)
    user, exists := us.users[id]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    var req UpdateUserRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    
    // Update user
    if req.Name != "" {
        user.Name = req.Name
    }
    if req.Email != "" {
        user.Email = req.Email
    }
    if req.Status != "" {
        user.Status = req.Status
    }
    user.UpdatedAt = time.Now()
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// DELETE /users/{id}
func (us *UserService) DeleteUser(w http.ResponseWriter, r *http.Request) {
    id := extractIDFromPath(r.URL.Path)
    _, exists := us.users[id]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    delete(us.users, id)
    w.WriteHeader(http.StatusNoContent)
}

type CreateUserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

type UpdateUserRequest struct {
    Name   string `json:"name,omitempty"`
    Email  string `json:"email,omitempty"`
    Status string `json:"status,omitempty"`
}
```

### 2. Error Handling

```go
// API Error Handling
type APIError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Details string `json:"details,omitempty"`
}

type ErrorResponse struct {
    Error APIError `json:"error"`
}

func (us *UserService) handleError(w http.ResponseWriter, err error, statusCode int) {
    apiError := APIError{
        Code:    getErrorCode(statusCode),
        Message: err.Error(),
    }
    
    response := ErrorResponse{Error: apiError}
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    json.NewEncoder(w).Encode(response)
}

func getErrorCode(statusCode int) string {
    switch statusCode {
    case http.StatusBadRequest:
        return "INVALID_REQUEST"
    case http.StatusUnauthorized:
        return "UNAUTHORIZED"
    case http.StatusForbidden:
        return "FORBIDDEN"
    case http.StatusNotFound:
        return "NOT_FOUND"
    case http.StatusConflict:
        return "CONFLICT"
    case http.StatusInternalServerError:
        return "INTERNAL_ERROR"
    default:
        return "UNKNOWN_ERROR"
    }
}
```

### 3. Middleware

```go
// API Middleware
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        ww := &ResponseWriterWrapper{ResponseWriter: w, statusCode: 200}
        next.ServeHTTP(ww, r)
        
        duration := time.Since(start)
        log.Printf("%s %s %d %v", r.Method, r.URL.Path, ww.statusCode, duration)
    })
}

func CORSMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Authorization header required", http.StatusUnauthorized)
            return
        }
        
        if !isValidToken(token) {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

type ResponseWriterWrapper struct {
    http.ResponseWriter
    statusCode int
}

func (rw *ResponseWriterWrapper) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

---

## 🔍 GraphQL API Design

### 1. Schema Design

```go
// GraphQL Schema
const schema = `
type User {
    id: Int!
    name: String!
    email: String!
    status: String!
    created_at: String!
    updated_at: String!
    orders: [Order!]
}

type Order {
    id: Int!
    user_id: Int!
    amount: Float!
    status: String!
    created_at: String!
}

type Query {
    user(id: Int!): User
    users(status: String, limit: Int, offset: Int): [User!]!
}

type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: Int!, input: UpdateUserInput!): User!
    deleteUser(id: Int!): Boolean!
}

input CreateUserInput {
    name: String!
    email: String!
}

input UpdateUserInput {
    name: String
    email: String
    status: String
}
`
```

### 2. Resolvers

```go
// GraphQL Resolvers
type UserResolver struct {
    userService *UserService
}

func (ur *UserResolver) GetUser(ctx context.Context, args struct {
    ID int `json:"id"`
}) (*User, error) {
    return ur.userService.GetUserByID(args.ID)
}

func (ur *UserResolver) GetUsers(ctx context.Context, args struct {
    Status *string `json:"status"`
    Limit  *int    `json:"limit"`
    Offset *int    `json:"offset"`
}) ([]*User, error) {
    return ur.userService.GetUsers(args.Status, args.Limit, args.Offset)
}

func (ur *UserResolver) CreateUser(ctx context.Context, args struct {
    Input CreateUserInput `json:"input"`
}) (*User, error) {
    return ur.userService.CreateUser(args.Input)
}

func (ur *UserResolver) UpdateUser(ctx context.Context, args struct {
    ID    int               `json:"id"`
    Input UpdateUserInput   `json:"input"`
}) (*User, error) {
    return ur.userService.UpdateUser(args.ID, args.Input)
}

func (ur *UserResolver) DeleteUser(ctx context.Context, args struct {
    ID int `json:"id"`
}) (bool, error) {
    return ur.userService.DeleteUser(args.ID)
}
```

---

## 🔒 API Security

### 1. Authentication

```go
// JWT Authentication
type AuthService struct {
    secretKey []byte
}

func (as *AuthService) Login(email, password string) (string, error) {
    // Validate credentials
    user, err := as.validateUser(email, password)
    if err != nil {
        return "", err
    }
    
    // Create JWT token
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
        "user_id": user.ID,
        "email":   user.Email,
        "exp":     time.Now().Add(time.Hour * 24).Unix(),
    })
    
    return token.SignedString(as.secretKey)
}

func (as *AuthService) ValidateToken(tokenString string) (*User, error) {
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        return as.secretKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
        return &User{
            ID:    int(claims["user_id"].(float64)),
            Email: claims["email"].(string),
        }, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}
```

### 2. Rate Limiting

```go
// Rate Limiting
type RateLimiter struct {
    requests map[string][]time.Time
    mutex    sync.RWMutex
    limit    int
    window   time.Duration
}

func (rl *RateLimiter) Allow(clientID string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-rl.window)
    
    // Clean old requests
    requests := rl.requests[clientID]
    var validRequests []time.Time
    for _, reqTime := range requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    
    // Check if under limit
    if len(validRequests) >= rl.limit {
        return false
    }
    
    // Add new request
    validRequests = append(validRequests, now)
    rl.requests[clientID] = validRequests
    
    return true
}

func RateLimitMiddleware(rateLimiter *RateLimiter) Middleware {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            clientID := getClientIP(r)
            
            if !rateLimiter.Allow(clientID) {
                http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}
```

---

## 🧪 API Testing

### 1. Unit Tests

```go
// API Unit Tests
func TestGetUsers(t *testing.T) {
    userService := NewUserService()
    server := httptest.NewServer(http.HandlerFunc(userService.GetUsers))
    defer server.Close()
    
    resp, err := http.Get(server.URL)
    if err != nil {
        t.Fatal(err)
    }
    
    if resp.StatusCode != http.StatusOK {
        t.Errorf("Expected status 200, got %d", resp.StatusCode)
    }
    
    var response map[string]interface{}
    if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
        t.Fatal(err)
    }
    
    if _, exists := response["data"]; !exists {
        t.Error("Expected 'data' field in response")
    }
}

func TestCreateUser(t *testing.T) {
    userService := NewUserService()
    server := httptest.NewServer(http.HandlerFunc(userService.CreateUser))
    defer server.Close()
    
    reqBody := CreateUserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    jsonBody, _ := json.Marshal(reqBody)
    resp, err := http.Post(server.URL, "application/json", bytes.NewBuffer(jsonBody))
    if err != nil {
        t.Fatal(err)
    }
    
    if resp.StatusCode != http.StatusCreated {
        t.Errorf("Expected status 201, got %d", resp.StatusCode)
    }
}
```

### 2. Integration Tests

```go
// API Integration Tests
func TestUserAPIIntegration(t *testing.T) {
    userService := NewUserService()
    mux := http.NewServeMux()
    mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "GET":
            userService.GetUsers(w, r)
        case "POST":
            userService.CreateUser(w, r)
        }
    })
    
    server := httptest.NewServer(mux)
    defer server.Close()
    
    // Test create user
    reqBody := CreateUserRequest{
        Name:  "Jane Doe",
        Email: "jane@example.com",
    }
    
    jsonBody, _ := json.Marshal(reqBody)
    resp, err := http.Post(server.URL+"/users", "application/json", bytes.NewBuffer(jsonBody))
    if err != nil {
        t.Fatal(err)
    }
    
    if resp.StatusCode != http.StatusCreated {
        t.Errorf("Expected status 201, got %d", resp.StatusCode)
    }
    
    // Test get users
    resp, err = http.Get(server.URL + "/users")
    if err != nil {
        t.Fatal(err)
    }
    
    if resp.StatusCode != http.StatusOK {
        t.Errorf("Expected status 200, got %d", resp.StatusCode)
    }
}
```

---

## 🎯 Best Practices

### 1. API Design
- **RESTful**: Follow REST principles
- **Consistent**: Use consistent naming and structure
- **Versioned**: Implement API versioning
- **Documented**: Comprehensive API documentation

### 2. Security
- **Authentication**: Implement proper authentication
- **Authorization**: Use role-based access control
- **Rate Limiting**: Implement rate limiting
- **Input Validation**: Validate all inputs

### 3. Performance
- **Caching**: Implement appropriate caching
- **Pagination**: Use pagination for large datasets
- **Compression**: Use response compression
- **Monitoring**: Monitor API performance

### 4. Testing
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test API endpoints
- **Load Tests**: Test under load
- **Security Tests**: Test security measures

---

**🌐 Master these API design patterns to build robust, scalable, and secure APIs! 🚀**
