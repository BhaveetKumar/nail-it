# Full-Stack Integration

## Overview

This module covers full-stack integration concepts including API design, state management, authentication, real-time communication, and deployment strategies. These concepts are essential for building complete web applications.

## Table of Contents

1. [API Design](#api-design/)
2. [State Management](#state-management/)
3. [Authentication & Authorization](#authentication--authorization/)
4. [Real-Time Communication](#real-time-communication/)
5. [Deployment Strategies](#deployment-strategies/)
6. [Applications](#applications/)
7. [Complexity Analysis](#complexity-analysis/)
8. [Follow-up Questions](#follow-up-questions/)

## API Design

### Theory

API design is crucial for full-stack applications as it defines the contract between frontend and backend. Good API design follows REST principles, provides clear documentation, and handles errors gracefully.

### API Gateway Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "time"
)

type APIEndpoint struct {
    Path        string
    Method      string
    Handler     http.HandlerFunc
    Middleware  []Middleware
    RateLimit   int
    Timeout     time.Duration
}

type APIGateway struct {
    Endpoints   map[string]*APIEndpoint
    Middleware  []Middleware
    RateLimiter map[string]int
    mutex       sync.RWMutex
}

func NewAPIGateway() *APIGateway {
    return &APIGateway{
        Endpoints:   make(map[string]*APIEndpoint),
        Middleware:  make([]Middleware, 0),
        RateLimiter: make(map[string]int),
    }
}

func (gw *APIGateway) AddEndpoint(path, method string, handler http.HandlerFunc) *APIEndpoint {
    endpoint := &APIEndpoint{
        Path:       path,
        Method:     method,
        Handler:    handler,
        Middleware: make([]Middleware, 0),
        RateLimit:  100, // Default rate limit
        Timeout:    30 * time.Second,
    }
    
    key := method + ":" + path
    gw.Endpoints[key] = endpoint
    
    fmt.Printf("Added endpoint: %s %s\n", method, path)
    return endpoint
}

func (gw *APIGateway) AddMiddleware(middleware Middleware) {
    gw.Middleware = append(gw.Middleware, middleware)
    fmt.Printf("Added global middleware\n")
}

func (gw *APIGateway) SetRateLimit(path, method string, limit int) {
    key := method + ":" + path
    if endpoint, exists := gw.Endpoints[key]; exists {
        endpoint.RateLimit = limit
        fmt.Printf("Set rate limit for %s %s: %d requests/minute\n", method, path, limit)
    }
}

func (gw *APIGateway) SetTimeout(path, method string, timeout time.Duration) {
    key := method + ":" + path
    if endpoint, exists := gw.Endpoints[key]; exists {
        endpoint.Timeout = timeout
        fmt.Printf("Set timeout for %s %s: %v\n", method, path, timeout)
    }
}

func (gw *APIGateway) HandleRequest(w http.ResponseWriter, r *http.Request) {
    key := r.Method + ":" + r.URL.Path
    
    // Find endpoint
    endpoint, exists := gw.Endpoints[key]
    if !exists {
        SendError(w, http.StatusNotFound, "Endpoint not found")
        return
    }
    
    // Check rate limit
    if !gw.checkRateLimit(r.RemoteAddr, endpoint) {
        SendError(w, http.StatusTooManyRequests, "Rate limit exceeded")
        return
    }
    
    // Apply middleware
    handler := endpoint.Handler
    for i := len(endpoint.Middleware) - 1; i >= 0; i-- {
        handler = endpoint.Middleware[i](../../../08_interview_prep/practice/handler)
    }
    
    for i := len(gw.Middleware) - 1; i >= 0; i-- {
        handler = gw.Middleware[i](../../../08_interview_prep/practice/handler)
    }
    
    // Handle request with timeout
    done := make(chan bool, 1)
    go func() {
        handler(w, r)
        done <- true
    }()
    
    select {
    case <-done:
        // Request completed
    case <-time.After(endpoint.Timeout):
        SendError(w, http.StatusRequestTimeout, "Request timeout")
    }
}

func (gw *APIGateway) checkRateLimit(clientIP string, endpoint *APIEndpoint) bool {
    gw.mutex.Lock()
    defer gw.mutex.Unlock()
    
    key := clientIP + ":" + endpoint.Path
    current := gw.RateLimiter[key]
    
    if current >= endpoint.RateLimit {
        return false
    }
    
    gw.RateLimiter[key] = current + 1
    return true
}

func (gw *APIGateway) Start(port string) {
    mux := http.NewServeMux()
    mux.HandleFunc("/", gw.HandleRequest)
    
    server := &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("API Gateway starting on port %s\n", port)
    go server.ListenAndServe()
}

func main() {
    gateway := NewAPIGateway()
    
    fmt.Println("API Gateway Demo:")
    
    // Add global middleware
    gateway.AddMiddleware(Logger())
    gateway.AddMiddleware(CORS())
    gateway.AddMiddleware(JSONParser())
    
    // Add endpoints
    gateway.AddEndpoint("/", "GET", func(w http.ResponseWriter, r *http.Request) {
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "API Gateway is running",
            "version": "1.0.0",
        })
    })
    
    gateway.AddEndpoint("/users", "GET", func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        }
        SendJSON(w, http.StatusOK, users)
    })
    
    gateway.AddEndpoint("/users", "POST", func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            SendError(w, http.StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 3
        SendJSON(w, http.StatusCreated, user)
    })
    
    // Set rate limits
    gateway.SetRateLimit("/users", "GET", 50)
    gateway.SetRateLimit("/users", "POST", 10)
    
    // Set timeouts
    gateway.SetTimeout("/users", "GET", 5*time.Second)
    gateway.SetTimeout("/users", "POST", 10*time.Second)
    
    // Start gateway
    gateway.Start("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## State Management

### Theory

State management in full-stack applications involves synchronizing state between frontend and backend, handling offline scenarios, and managing complex application state.

### State Synchronization Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type StateManager struct {
    State      map[string]interface{}
    Subscribers map[string][]func(interface{})
    mutex      sync.RWMutex
}

func NewStateManager() *StateManager {
    return &StateManager{
        State:      make(map[string]interface{}),
        Subscribers: make(map[string][]func(interface{})),
    }
}

func (sm *StateManager) SetState(key string, value interface{}) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    sm.State[key] = value
    
    // Notify subscribers
    if subscribers, exists := sm.Subscribers[key]; exists {
        for _, subscriber := range subscribers {
            go subscriber(value)
        }
    }
    
    fmt.Printf("State updated: %s = %v\n", key, value)
}

func (sm *StateManager) GetState(key string) interface{} {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    return sm.State[key]
}

func (sm *StateManager) Subscribe(key string, callback func(interface{})) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    sm.Subscribers[key] = append(sm.Subscribers[key], callback)
    fmt.Printf("Subscribed to state changes for key: %s\n", key)
}

func (sm *StateManager) Unsubscribe(key string, callback func(interface{})) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    if subscribers, exists := sm.Subscribers[key]; exists {
        for i, sub := range subscribers {
            if &sub == &callback {
                sm.Subscribers[key] = append(subscribers[:i], subscribers[i+1:]...)
                break
            }
        }
    }
}

func (sm *StateManager) GetFullState() map[string]interface{} {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    state := make(map[string]interface{})
    for k, v := range sm.State {
        state[k] = v
    }
    
    return state
}

func (sm *StateManager) SyncWithBackend(backendState map[string]interface{}) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    for key, value := range backendState {
        if sm.State[key] != value {
            sm.State[key] = value
            
            // Notify subscribers
            if subscribers, exists := sm.Subscribers[key]; exists {
                for _, subscriber := range subscribers {
                    go subscriber(value)
                }
            }
        }
    }
    
    fmt.Println("State synchronized with backend")
}

func main() {
    sm := NewStateManager()
    
    fmt.Println("State Management Demo:")
    
    // Subscribe to state changes
    sm.Subscribe("user", func(value interface{}) {
        fmt.Printf("User state changed: %v\n", value)
    })
    
    sm.Subscribe("theme", func(value interface{}) {
        fmt.Printf("Theme state changed: %v\n", value)
    })
    
    // Set initial state
    sm.SetState("user", map[string]interface{}{
        "id":    1,
        "name":  "Alice",
        "email": "alice@example.com",
    })
    
    sm.SetState("theme", "dark")
    
    // Update state
    sm.SetState("user", map[string]interface{}{
        "id":    1,
        "name":  "Alice Smith",
        "email": "alice.smith@example.com",
    })
    
    sm.SetState("theme", "light")
    
    // Simulate backend sync
    backendState := map[string]interface{}{
        "user": map[string]interface{}{
            "id":    1,
            "name":  "Alice Smith",
            "email": "alice.smith@example.com",
        },
        "theme": "dark",
        "language": "en",
    }
    
    sm.SyncWithBackend(backendState)
    
    // Get full state
    state := sm.GetFullState()
    fmt.Printf("Full state: %v\n", state)
}
```

## Authentication & Authorization

### Theory

Authentication verifies user identity, while authorization determines what actions a user can perform. In full-stack applications, this involves JWT tokens, session management, and role-based access control.

### Authentication System Implementation

#### Golang Implementation

```go
package main

import (
    "crypto/rand"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Email    string `json:"email"`
    Role     string `json:"role"`
    Password string `json:"-"`
}

type AuthToken struct {
    Token     string    `json:"token"`
    ExpiresAt time.Time `json:"expires_at"`
    UserID    int       `json:"user_id"`
}

type AuthService struct {
    Users  map[string]*User
    Tokens map[string]*AuthToken
    mutex  sync.RWMutex
}

func NewAuthService() *AuthService {
    return &AuthService{
        Users:  make(map[string]*User),
        Tokens: make(map[string]*AuthToken),
    }
}

func (as *AuthService) Register(username, email, password, role string) (*User, error) {
    as.mutex.Lock()
    defer as.mutex.Unlock()
    
    if _, exists := as.Users[username]; exists {
        return nil, fmt.Errorf("username already exists")
    }
    
    user := &User{
        ID:       len(as.Users) + 1,
        Username: username,
        Email:    email,
        Role:     role,
        Password: password, // In real app, hash this
    }
    
    as.Users[username] = user
    fmt.Printf("User registered: %s\n", username)
    return user, nil
}

func (as *AuthService) Login(username, password string) (*AuthToken, error) {
    as.mutex.RLock()
    user, exists := as.Users[username]
    as.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    
    if user.Password != password {
        return nil, fmt.Errorf("invalid password")
    }
    
    // Generate token
    token := as.generateToken()
    authToken := &AuthToken{
        Token:     token,
        ExpiresAt: time.Now().Add(24 * time.Hour),
        UserID:    user.ID,
    }
    
    as.mutex.Lock()
    as.Tokens[token] = authToken
    as.mutex.Unlock()
    
    fmt.Printf("User logged in: %s\n", username)
    return authToken, nil
}

func (as *AuthService) generateToken() string {
    b := make([]byte, 32)
    rand.Read(b)
    return fmt.Sprintf("%x", b)
}

func (as *AuthService) ValidateToken(token string) (*User, error) {
    as.mutex.RLock()
    authToken, exists := as.Tokens[token]
    as.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("invalid token")
    }
    
    if time.Now().After(authToken.ExpiresAt) {
        return nil, fmt.Errorf("token expired")
    }
    
    // Find user by ID
    as.mutex.RLock()
    defer as.mutex.RUnlock()
    
    for _, user := range as.Users {
        if user.ID == authToken.UserID {
            return user, nil
        }
    }
    
    return nil, fmt.Errorf("user not found")
}

func (as *AuthService) Logout(token string) error {
    as.mutex.Lock()
    defer as.mutex.Unlock()
    
    if _, exists := as.Tokens[token]; exists {
        delete(as.Tokens, token)
        fmt.Println("User logged out")
        return nil
    }
    
    return fmt.Errorf("token not found")
}

func (as *AuthService) Authorize(user *User, resource, action string) bool {
    // Simple role-based authorization
    switch user.Role {
    case "admin":
        return true
    case "user":
        return resource == "profile" && action == "read"
    case "guest":
        return resource == "public" && action == "read"
    default:
        return false
    }
}

func (as *AuthService) Middleware() Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            token := r.Header.Get("Authorization")
            if token == "" {
                SendError(w, http.StatusUnauthorized, "No token provided")
                return
            }
            
            // Remove "Bearer " prefix
            if len(token) > 7 && token[:7] == "Bearer " {
                token = token[7:]
            }
            
            user, err := as.ValidateToken(token)
            if err != nil {
                SendError(w, http.StatusUnauthorized, "Invalid token")
                return
            }
            
            // Add user to request context
            r.Header.Set("X-User-ID", fmt.Sprintf("%d", user.ID))
            r.Header.Set("X-User-Role", user.Role)
            
            next(w, r)
        }
    }
}

func main() {
    authService := NewAuthService()
    
    fmt.Println("Authentication & Authorization Demo:")
    
    // Register users
    authService.Register("alice", "alice@example.com", "password123", "admin")
    authService.Register("bob", "bob@example.com", "password456", "user")
    authService.Register("guest", "guest@example.com", "password789", "guest")
    
    // Login
    token, err := authService.Login("alice", "password123")
    if err != nil {
        fmt.Printf("Login failed: %v\n", err)
        return
    }
    
    fmt.Printf("Login successful, token: %s\n", token.Token)
    
    // Validate token
    user, err := authService.ValidateToken(token.Token)
    if err != nil {
        fmt.Printf("Token validation failed: %v\n", err)
        return
    }
    
    fmt.Printf("Token valid for user: %s\n", user.Username)
    
    // Test authorization
    if authService.Authorize(user, "profile", "read") {
        fmt.Println("User authorized to read profile")
    }
    
    if authService.Authorize(user, "admin", "write") {
        fmt.Println("User authorized to write admin")
    }
    
    // Logout
    authService.Logout(token.Token)
    
    // Test expired token
    _, err = authService.ValidateToken(token.Token)
    if err != nil {
        fmt.Printf("Token validation after logout: %v\n", err)
    }
}
```

## Real-Time Communication

### Theory

Real-time communication enables instant data exchange between frontend and backend. Common technologies include WebSockets, Server-Sent Events, and WebRTC.

### WebSocket Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
    
    "github.com/gorilla/websocket"
)

type WebSocketServer struct {
    Clients    map[*websocket.Conn]bool
    Broadcast  chan []byte
    Register   chan *websocket.Conn
    Unregister chan *websocket.Conn
    mutex      sync.RWMutex
}

func NewWebSocketServer() *WebSocketServer {
    return &WebSocketServer{
        Clients:    make(map[*websocket.Conn]bool),
        Broadcast:  make(chan []byte),
        Register:   make(chan *websocket.Conn),
        Unregister: make(chan *websocket.Conn),
    }
}

func (ws *WebSocketServer) Run() {
    for {
        select {
        case conn := <-ws.Register:
            ws.mutex.Lock()
            ws.Clients[conn] = true
            ws.mutex.Unlock()
            fmt.Println("Client connected")
            
        case conn := <-ws.Unregister:
            ws.mutex.Lock()
            if _, ok := ws.Clients[conn]; ok {
                delete(ws.Clients, conn)
                conn.Close()
            }
            ws.mutex.Unlock()
            fmt.Println("Client disconnected")
            
        case message := <-ws.Broadcast:
            ws.mutex.RLock()
            for conn := range ws.Clients {
                err := conn.WriteMessage(websocket.TextMessage, message)
                if err != nil {
                    conn.Close()
                    delete(ws.Clients, conn)
                }
            }
            ws.mutex.RUnlock()
        }
    }
}

func (ws *WebSocketServer) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    upgrader := websocket.Upgrader{
        CheckOrigin: func(r *http.Request) bool {
            return true
        },
    }
    
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        fmt.Printf("WebSocket upgrade failed: %v\n", err)
        return
    }
    
    ws.Register <- conn
    
    go func() {
        defer func() {
            ws.Unregister <- conn
        }()
        
        for {
            _, message, err := conn.ReadMessage()
            if err != nil {
                break
            }
            
            // Echo message back to all clients
            ws.Broadcast <- message
        }
    }()
}

func (ws *WebSocketServer) SendToAll(message map[string]interface{}) {
    data, err := json.Marshal(message)
    if err != nil {
        fmt.Printf("Error marshaling message: %v\n", err)
        return
    }
    
    ws.Broadcast <- data
}

func (ws *WebSocketServer) GetClientCount() int {
    ws.mutex.RLock()
    defer ws.mutex.RUnlock()
    return len(ws.Clients)
}

func main() {
    wsServer := NewWebSocketServer()
    
    fmt.Println("WebSocket Server Demo:")
    
    // Start WebSocket server
    go wsServer.Run()
    
    // HTTP server
    http.HandleFunc("/ws", wsServer.HandleWebSocket)
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Demo</title>
</head>
<body>
    <h1>WebSocket Demo</h1>
    <div id="messages"></div>
    <input type="text" id="messageInput" placeholder="Enter message">
    <button onclick="sendMessage()">Send</button>
    
    <script>
        const ws = new WebSocket('ws://localhost:8080/ws');
        const messages = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        
        ws.onmessage = function(event) {
            const message = document.createElement('div');
            message.textContent = event.data;
            messages.appendChild(message);
        };
        
        function sendMessage() {
            const message = messageInput.value;
            if (message) {
                ws.send(message);
                messageInput.value = '';
            }
        }
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>`)
    })
    
    // Start HTTP server
    go http.ListenAndServe(":8080", nil)
    
    // Send periodic messages
    ticker := time.NewTicker(5 * time.Second)
    go func() {
        for range ticker.C {
            if wsServer.GetClientCount() > 0 {
                message := map[string]interface{}{
                    "type":    "notification",
                    "message": "Server time: " + time.Now().Format(time.RFC3339),
                    "clients": wsServer.GetClientCount(),
                }
                wsServer.SendToAll(message)
            }
        }
    }()
    
    fmt.Println("WebSocket server starting on :8080")
    fmt.Println("Open http://localhost:8080 in your browser")
    
    // Keep the program running
    time.Sleep(1 * time.Minute)
}
```

## Follow-up Questions

### 1. API Design
**Q: What are the key principles of good API design?**
A: Good API design follows REST principles, uses consistent naming conventions, provides clear documentation, handles errors gracefully, implements proper authentication and authorization, and maintains backward compatibility.

### 2. State Management
**Q: How do you handle state synchronization between frontend and backend?**
A: Use optimistic updates for better UX, implement conflict resolution strategies, handle offline scenarios with local storage, and use real-time updates for critical data changes.

### 3. Authentication & Authorization
**Q: What are the differences between JWT and session-based authentication?**
A: JWT tokens are stateless and contain user information, making them suitable for distributed systems. Session-based authentication stores session data on the server, providing better security but requiring server-side storage.

## Complexity Analysis

| Operation | API Gateway | State Management | Authentication | WebSocket |
|-----------|-------------|------------------|----------------|-----------|
| Request Processing | O(1) | O(1) | O(1) | O(1) |
| State Update | N/A | O(n) | N/A | N/A |
| Token Validation | N/A | N/A | O(1) | N/A |
| Message Broadcasting | N/A | N/A | N/A | O(n) |

## Applications

1. **API Design**: Microservices, mobile backends, third-party integrations
2. **State Management**: Complex applications, real-time apps, collaborative tools
3. **Authentication**: User management, security systems, access control
4. **Real-Time Communication**: Chat applications, live updates, gaming

---

**Next**: [API Design](../../../README.md) | **Previous**: [Web Development](README.md/) | **Up**: [Phase 1](README.md/)
