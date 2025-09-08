# 🌐 HTTP Basics: The Foundation of Web Communication

> **Master HTTP/HTTPS protocols, methods, headers, and status codes for backend engineering**

## 📚 Concept

HTTP (HyperText Transfer Protocol) is the foundation of web communication. Understanding HTTP is crucial for backend engineers as it defines how clients and servers exchange data.

### HTTP vs HTTPS

- **HTTP**: Unencrypted, port 80, vulnerable to man-in-the-middle attacks
- **HTTPS**: Encrypted with TLS/SSL, port 443, secure communication

### HTTP Request Structure

```
GET /api/users/123 HTTP/1.1
Host: api.example.com
User-Agent: Mozilla/5.0
Accept: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
Content-Length: 45

{"name": "John Doe", "email": "john@example.com"}
```

### HTTP Response Structure

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 123
Cache-Control: max-age=3600
Set-Cookie: sessionId=abc123; HttpOnly; Secure

{"id": 123, "name": "John Doe", "email": "john@example.com"}
```

## 🛠️ Hands-on Example

### Building a Simple HTTP Server in Go

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "time"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type UserService struct {
    users map[int]User
}

func NewUserService() *UserService {
    return &UserService{
        users: map[int]User{
            1: {ID: 1, Name: "John Doe", Email: "john@example.com"},
            2: {ID: 2, Name: "Jane Smith", Email: "jane@example.com"},
        },
    }
}

func (us *UserService) GetUser(w http.ResponseWriter, r *http.Request) {
    // Extract user ID from URL path
    idStr := r.URL.Path[len("/api/users/"):]
    id, err := strconv.Atoi(idStr)
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    user, exists := us.users[id]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    // Set response headers
    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("Cache-Control", "max-age=300")

    // Add CORS headers
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
    w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

    // Encode and send response
    json.NewEncoder(w).Encode(user)
}

func (us *UserService) CreateUser(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Generate new ID
    user.ID = len(us.users) + 1
    us.users[user.ID] = user

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}

func (us *UserService) Middleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        // Log request
        log.Printf("%s %s %s", r.Method, r.URL.Path, r.RemoteAddr)

        // Add security headers
        w.Header().Set("X-Content-Type-Options", "nosniff")
        w.Header().Set("X-Frame-Options", "DENY")
        w.Header().Set("X-XSS-Protection", "1; mode=block")

        next(w, r)

        // Log response time
        log.Printf("Request completed in %v", time.Since(start))
    }
}

func main() {
    userService := NewUserService()

    // Define routes
    http.HandleFunc("/api/users/", userService.Middleware(userService.GetUser))
    http.HandleFunc("/api/users", userService.Middleware(userService.CreateUser))

    // Health check endpoint
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### Testing HTTP Endpoints

```bash
# Test GET request
curl -X GET http://localhost:8080/api/users/1 \
  -H "Accept: application/json" \
  -H "User-Agent: MyApp/1.0"

# Test POST request
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Johnson", "email": "alice@example.com"}'

# Test with authentication
curl -X GET http://localhost:8080/api/users/1 \
  -H "Authorization: Bearer your-jwt-token"

# Test error handling
curl -X GET http://localhost:8080/api/users/999
```

## 🏗️ HTTP Methods Deep Dive

### GET - Retrieve Data

```go
func (us *UserService) GetUsers(w http.ResponseWriter, r *http.Request) {
    // Query parameters
    page := r.URL.Query().Get("page")
    limit := r.URL.Query().Get("limit")

    // Pagination logic
    users := make([]User, 0)
    for _, user := range us.users {
        users = append(users, user)
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "users": users,
        "page": page,
        "limit": limit,
        "total": len(users),
    })
}
```

### POST - Create Resource

```go
func (us *UserService) CreateUser(w http.ResponseWriter, r *http.Request) {
    // Validate content type
    if r.Header.Get("Content-Type") != "application/json" {
        http.Error(w, "Content-Type must be application/json", http.StatusUnsupportedMediaType)
        return
    }

    // Limit request body size
    r.Body = http.MaxBytesReader(w, r.Body, 1048576) // 1MB limit

    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Validate required fields
    if user.Name == "" || user.Email == "" {
        http.Error(w, "Name and email are required", http.StatusBadRequest)
        return
    }

    // Business logic
    user.ID = generateID()
    us.users[user.ID] = user

    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("Location", fmt.Sprintf("/api/users/%d", user.ID))
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}
```

### PUT - Update Resource

```go
func (us *UserService) UpdateUser(w http.ResponseWriter, r *http.Request) {
    idStr := r.URL.Path[len("/api/users/"):]
    id, err := strconv.Atoi(idStr)
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    user.ID = id
    us.users[id] = user

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}
```

### DELETE - Remove Resource

```go
func (us *UserService) DeleteUser(w http.ResponseWriter, r *http.Request) {
    idStr := r.URL.Path[len("/api/users/"):]
    id, err := strconv.Atoi(idStr)
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    if _, exists := us.users[id]; !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    delete(us.users, id)
    w.WriteHeader(http.StatusNoContent)
}
```

## 📊 HTTP Status Codes

### 2xx Success

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for processing
- **204 No Content**: Request successful, no content returned

### 3xx Redirection

- **301 Moved Permanently**: Resource moved permanently
- **302 Found**: Resource temporarily moved
- **304 Not Modified**: Resource not modified (caching)

### 4xx Client Error

- **400 Bad Request**: Invalid request syntax
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Access denied
- **404 Not Found**: Resource not found
- **405 Method Not Allowed**: HTTP method not allowed
- **429 Too Many Requests**: Rate limit exceeded

### 5xx Server Error

- **500 Internal Server Error**: Server error
- **502 Bad Gateway**: Invalid response from upstream
- **503 Service Unavailable**: Service temporarily unavailable
- **504 Gateway Timeout**: Upstream timeout

## 🔒 Security Headers

```go
func SecurityHeaders(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Prevent MIME type sniffing
        w.Header().Set("X-Content-Type-Options", "nosniff")

        // Prevent clickjacking
        w.Header().Set("X-Frame-Options", "DENY")

        // XSS protection
        w.Header().Set("X-XSS-Protection", "1; mode=block")

        // Content Security Policy
        w.Header().Set("Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'")

        // HTTPS enforcement
        w.Header().Set("Strict-Transport-Security",
            "max-age=31536000; includeSubDomains")

        next.ServeHTTP(w, r)
    })
}
```

## 🚀 Best Practices

### 1. Request Validation

```go
func ValidateRequest(r *http.Request) error {
    // Check content length
    if r.ContentLength > 10*1024*1024 { // 10MB limit
        return errors.New("request too large")
    }

    // Validate content type for POST/PUT
    if r.Method == "POST" || r.Method == "PUT" {
        contentType := r.Header.Get("Content-Type")
        if !strings.HasPrefix(contentType, "application/json") {
            return errors.New("invalid content type")
        }
    }

    return nil
}
```

### 2. Response Caching

```go
func CacheControl(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // Set cache headers based on resource type
        if strings.Contains(r.URL.Path, "/static/") {
            w.Header().Set("Cache-Control", "public, max-age=31536000") // 1 year
        } else if strings.Contains(r.URL.Path, "/api/") {
            w.Header().Set("Cache-Control", "private, max-age=300") // 5 minutes
        }

        next(w, r)
    }
}
```

### 3. Rate Limiting

```go
type RateLimiter struct {
    requests map[string][]time.Time
    mutex    sync.RWMutex
    limit    int
    window   time.Duration
}

func (rl *RateLimiter) Allow(clientIP string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    cutoff := now.Add(-rl.window)

    // Clean old requests
    if requests, exists := rl.requests[clientIP]; exists {
        var validRequests []time.Time
        for _, req := range requests {
            if req.After(cutoff) {
                validRequests = append(validRequests, req)
            }
        }
        rl.requests[clientIP] = validRequests
    }

    // Check limit
    if len(rl.requests[clientIP]) >= rl.limit {
        return false
    }

    // Add current request
    rl.requests[clientIP] = append(rl.requests[clientIP], now)
    return true
}
```

## 🏢 Industry Insights

### Meta's HTTP Practices

- **GraphQL over REST**: More efficient data fetching
- **Custom headers**: `X-Facebook-Request-Id` for tracing
- **Compression**: Gzip/Brotli for all responses
- **HTTP/2**: Multiplexing for better performance

### Google's HTTP Practices

- **Protocol Buffers**: Binary serialization for internal APIs
- **gRPC**: High-performance RPC framework
- **Custom status codes**: Extended error information
- **Request batching**: Multiple operations in single request

### Amazon's HTTP Practices

- **AWS Signature**: Custom authentication for AWS APIs
- **Retry logic**: Exponential backoff with jitter
- **Request signing**: HMAC-SHA256 for security
- **Regional endpoints**: Geographic distribution

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between HTTP and HTTPS?**

   - HTTPS adds TLS/SSL encryption on top of HTTP
   - Uses port 443 instead of 80
   - Provides data integrity and authentication

2. **Explain HTTP methods and when to use each?**

   - GET: Retrieve data (idempotent, cacheable)
   - POST: Create resource (not idempotent)
   - PUT: Update/replace resource (idempotent)
   - DELETE: Remove resource (idempotent)
   - PATCH: Partial update

3. **What are HTTP status codes and their categories?**
   - 2xx: Success (200, 201, 204)
   - 3xx: Redirection (301, 302, 304)
   - 4xx: Client error (400, 401, 403, 404)
   - 5xx: Server error (500, 502, 503, 504)

### Intermediate Level

4. **How do you implement proper error handling in HTTP APIs?**

   ```go
   type APIError struct {
       Code    int    `json:"code"`
       Message string `json:"message"`
       Details string `json:"details,omitempty"`
   }

   func HandleError(w http.ResponseWriter, err error, statusCode int) {
       w.Header().Set("Content-Type", "application/json")
       w.WriteHeader(statusCode)

       apiError := APIError{
           Code:    statusCode,
           Message: err.Error(),
       }

       json.NewEncoder(w).Encode(apiError)
   }
   ```

5. **Explain HTTP caching strategies?**

   - **Browser caching**: Cache-Control, Expires headers
   - **CDN caching**: Edge servers for static content
   - **Application caching**: Redis/Memcached for dynamic data
   - **Database caching**: Query result caching

6. **How do you handle CORS in web applications?**
   ```go
   func CORSHandler(next http.Handler) http.Handler {
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
   ```

### Advanced Level

7. **Design a rate limiting system for HTTP APIs?**

   - **Token bucket**: Allow bursts up to bucket size
   - **Sliding window**: Track requests in time window
   - **Fixed window**: Reset counter at interval
   - **Distributed rate limiting**: Redis-based for multiple servers

8. **How do you implement HTTP/2 server push?**

   ```go
   // HTTP/2 push for critical resources
   func PushResource(w http.ResponseWriter, resource string) {
       if pusher, ok := w.(http.Pusher); ok {
           pusher.Push(resource, nil)
       }
   }
   ```

9. **Explain HTTP/3 and QUIC protocol benefits?**
   - **Multiplexing**: No head-of-line blocking
   - **Connection migration**: Seamless network changes
   - **Built-in encryption**: TLS 1.3 integration
   - **Faster handshake**: 0-RTT connection establishment

---

**Next**: [REST vs GraphQL](./RESTvsGraphQL.md) - API design patterns and trade-offs
