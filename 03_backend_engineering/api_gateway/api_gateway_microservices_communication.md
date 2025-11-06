---
# Auto-generated front matter
Title: Api Gateway Microservices Communication
LastUpdated: 2025-11-06T20:45:58.294594
Tags: []
Status: draft
---

# 🌐 API Gateway & Microservices Communication

## Table of Contents
1. [API Gateway Fundamentals](#api-gateway-fundamentals)
2. [Request Routing](#request-routing)
3. [Authentication & Authorization](#authentication--authorization)
4. [Rate Limiting & Throttling](#rate-limiting--throttling)
5. [Circuit Breaker Pattern](#circuit-breaker-pattern)
6. [Service Discovery](#service-discovery)
7. [Load Balancing](#load-balancing)
8. [Go Implementation Examples](#go-implementation-examples)
9. [Interview Questions](#interview-questions)

## API Gateway Fundamentals

### Core API Gateway Implementation

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type APIGateway struct {
    routes        map[string]*Route
    middlewares   []Middleware
    services      map[string]*Service
    mutex         sync.RWMutex
    config        *GatewayConfig
}

type Route struct {
    Path        string
    Method      string
    ServiceName string
    Middlewares []Middleware
    Timeout     time.Duration
}

type Service struct {
    Name     string
    Endpoints []string
    Health   bool
    mutex    sync.RWMutex
}

type GatewayConfig struct {
    Port            string
    ReadTimeout     time.Duration
    WriteTimeout    time.Duration
    IdleTimeout     time.Duration
    MaxHeaderBytes  int
}

type Middleware interface {
    Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter
}

func NewAPIGateway(config *GatewayConfig) *APIGateway {
    return &APIGateway{
        routes:      make(map[string]*Route),
        middlewares: make([]Middleware, 0),
        services:    make(map[string]*Service),
        config:      config,
    }
}

func (gw *APIGateway) AddRoute(path, method, serviceName string, timeout time.Duration) {
    gw.mutex.Lock()
    defer gw.mutex.Unlock()
    
    routeKey := fmt.Sprintf("%s %s", method, path)
    gw.routes[routeKey] = &Route{
        Path:        path,
        Method:      method,
        ServiceName: serviceName,
        Timeout:     timeout,
    }
}

func (gw *APIGateway) RegisterService(name string, endpoints []string) {
    gw.mutex.Lock()
    defer gw.mutex.Unlock()
    
    gw.services[name] = &Service{
        Name:      name,
        Endpoints: endpoints,
        Health:    true,
    }
}

func (gw *APIGateway) AddMiddleware(middleware Middleware) {
    gw.mutex.Lock()
    defer gw.mutex.Unlock()
    gw.middlewares = append(gw.middlewares, middleware)
}

func (gw *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    ctx := context.Background()
    
    // Find matching route
    routeKey := fmt.Sprintf("%s %s", r.Method, r.URL.Path)
    gw.mutex.RLock()
    route, exists := gw.routes[routeKey]
    gw.mutex.RUnlock()
    
    if !exists {
        http.NotFound(w, r)
        return
    }
    
    // Get service
    gw.mutex.RLock()
    service, serviceExists := gw.services[route.ServiceName]
    gw.mutex.RUnlock()
    
    if !serviceExists || !service.IsHealthy() {
        http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
        return
    }
    
    // Create request context
    reqCtx := &RequestContext{
        Route:   route,
        Service: service,
        StartTime: time.Now(),
    }
    ctx = context.WithValue(ctx, "requestContext", reqCtx)
    
    // Process through middlewares
    handler := gw.buildHandler(route)
    for i := len(gw.middlewares) - 1; i >= 0; i-- {
        middleware := gw.middlewares[i]
        handler = func(next http.Handler) http.Handler {
            return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                middleware.Process(ctx, r, next)
            })
        }(handler)
    }
    
    handler.ServeHTTP(w, r)
}

func (gw *APIGateway) buildHandler(route *Route) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Create timeout context
        ctx, cancel := context.WithTimeout(r.Context(), route.Timeout)
        defer cancel()
        
        // Select healthy endpoint
        endpoint := route.Service.SelectHealthyEndpoint()
        if endpoint == "" {
            http.Error(w, "No healthy endpoints available", http.StatusServiceUnavailable)
            return
        }
        
        // Forward request
        gw.forwardRequest(ctx, w, r, endpoint)
    })
}

func (gw *APIGateway) forwardRequest(ctx context.Context, w http.ResponseWriter, r *http.Request, endpoint string) {
    // Create new request with modified URL
    newReq := r.Clone(ctx)
    newReq.URL.Scheme = "http"
    newReq.URL.Host = endpoint
    newReq.URL.Path = r.URL.Path
    
    // Add gateway headers
    newReq.Header.Set("X-Gateway", "api-gateway")
    newReq.Header.Set("X-Forwarded-For", r.RemoteAddr)
    
    // Make request
    client := &http.Client{
        Timeout: 30 * time.Second,
    }
    
    resp, err := client.Do(newReq)
    if err != nil {
        http.Error(w, "Service error", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()
    
    // Copy response headers
    for key, values := range resp.Header {
        for _, value := range values {
            w.Header().Add(key, value)
        }
    }
    
    // Copy response status
    w.WriteHeader(resp.StatusCode)
    
    // Copy response body
    io.Copy(w, resp.Body)
}

type RequestContext struct {
    Route     *Route
    Service   *Service
    StartTime time.Time
}

func (s *Service) IsHealthy() bool {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Health
}

func (s *Service) SelectHealthyEndpoint() string {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    if !s.Health || len(s.Endpoints) == 0 {
        return ""
    }
    
    // Simple round-robin selection
    // In production, use a proper load balancer
    return s.Endpoints[0]
}

func (s *Service) SetHealth(health bool) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Health = health
}
```

## Request Routing

### Advanced Routing Patterns

```go
package main

import (
    "regexp"
    "strings"
)

type Router struct {
    routes []*RoutePattern
    mutex  sync.RWMutex
}

type RoutePattern struct {
    Pattern     *regexp.Regexp
    Method      string
    ServiceName string
    Variables   []string
    Priority    int
}

func NewRouter() *Router {
    return &Router{
        routes: make([]*RoutePattern, 0),
    }
}

func (r *Router) AddRoute(pattern, method, serviceName string, priority int) error {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    // Convert pattern to regex
    regexPattern := r.convertToRegex(pattern)
    compiled, err := regexp.Compile(regexPattern)
    if err != nil {
        return err
    }
    
    // Extract variable names
    variables := r.extractVariables(pattern)
    
    route := &RoutePattern{
        Pattern:     compiled,
        Method:      method,
        ServiceName: serviceName,
        Variables:   variables,
        Priority:    priority,
    }
    
    r.routes = append(r.routes, route)
    
    // Sort by priority (higher priority first)
    r.sortRoutes()
    
    return nil
}

func (r *Router) FindRoute(path, method string) (*RoutePattern, map[string]string, bool) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    for _, route := range r.routes {
        if route.Method != method {
            continue
        }
        
        if route.Pattern.MatchString(path) {
            matches := route.Pattern.FindStringSubmatch(path)
            params := make(map[string]string)
            
            for i, name := range route.Variables {
                if i+1 < len(matches) {
                    params[name] = matches[i+1]
                }
            }
            
            return route, params, true
        }
    }
    
    return nil, nil, false
}

func (r *Router) convertToRegex(pattern string) string {
    // Convert /users/{id} to /users/([^/]+)
    regex := strings.ReplaceAll(pattern, "{", "(")
    regex = strings.ReplaceAll(regex, "}", ")")
    regex = strings.ReplaceAll(regex, ")", "[^/]+)")
    
    // Add start and end anchors
    return "^" + regex + "$"
}

func (r *Router) extractVariables(pattern string) []string {
    var variables []string
    
    // Find all {variable} patterns
    re := regexp.MustCompile(`\{([^}]+)\}`)
    matches := re.FindAllStringSubmatch(pattern, -1)
    
    for _, match := range matches {
        if len(match) > 1 {
            variables = append(variables, match[1])
        }
    }
    
    return variables
}

func (r *Router) sortRoutes() {
    // Sort by priority (higher priority first)
    for i := 0; i < len(r.routes)-1; i++ {
        for j := i + 1; j < len(r.routes); j++ {
            if r.routes[i].Priority < r.routes[j].Priority {
                r.routes[i], r.routes[j] = r.routes[j], r.routes[i]
            }
        }
    }
}

// Path-based routing
type PathRouter struct {
    staticRoutes  map[string]*Route
    dynamicRoutes []*RoutePattern
    mutex         sync.RWMutex
}

func NewPathRouter() *PathRouter {
    return &PathRouter{
        staticRoutes:  make(map[string]*Route),
        dynamicRoutes: make([]*RoutePattern, 0),
    }
}

func (pr *PathRouter) AddStaticRoute(path, method, serviceName string) {
    pr.mutex.Lock()
    defer pr.mutex.Unlock()
    
    key := fmt.Sprintf("%s %s", method, path)
    pr.staticRoutes[key] = &Route{
        Path:        path,
        Method:      method,
        ServiceName: serviceName,
    }
}

func (pr *PathRouter) AddDynamicRoute(pattern, method, serviceName string) error {
    pr.mutex.Lock()
    defer pr.mutex.Unlock()
    
    regexPattern := pr.convertToRegex(pattern)
    compiled, err := regexp.Compile(regexPattern)
    if err != nil {
        return err
    }
    
    variables := pr.extractVariables(pattern)
    
    route := &RoutePattern{
        Pattern:     compiled,
        Method:      method,
        ServiceName: serviceName,
        Variables:   variables,
    }
    
    pr.dynamicRoutes = append(pr.dynamicRoutes, route)
    return nil
}

func (pr *PathRouter) FindRoute(path, method string) (*Route, map[string]string, bool) {
    pr.mutex.RLock()
    defer pr.mutex.RUnlock()
    
    // Check static routes first
    key := fmt.Sprintf("%s %s", method, path)
    if route, exists := pr.staticRoutes[key]; exists {
        return route, nil, true
    }
    
    // Check dynamic routes
    for _, route := range pr.dynamicRoutes {
        if route.Method != method {
            continue
        }
        
        if route.Pattern.MatchString(path) {
            matches := route.Pattern.FindStringSubmatch(path)
            params := make(map[string]string)
            
            for i, name := range route.Variables {
                if i+1 < len(matches) {
                    params[name] = matches[i+1]
                }
            }
            
            return &Route{
                Path:        path,
                Method:      method,
                ServiceName: route.ServiceName,
            }, params, true
        }
    }
    
    return nil, nil, false
}
```

## Authentication & Authorization

### JWT-based Authentication

```go
package main

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "strings"
    "time"
)

type AuthMiddleware struct {
    secretKey []byte
    issuer    string
}

func NewAuthMiddleware(secretKey string, issuer string) *AuthMiddleware {
    return &AuthMiddleware{
        secretKey: []byte(secretKey),
        issuer:    issuer,
    }
}

func (am *AuthMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    // Extract token from Authorization header
    authHeader := req.Header.Get("Authorization")
    if authHeader == "" {
        return am.unauthorizedResponse("Missing authorization header")
    }
    
    if !strings.HasPrefix(authHeader, "Bearer ") {
        return am.unauthorizedResponse("Invalid authorization header format")
    }
    
    token := strings.TrimPrefix(authHeader, "Bearer ")
    
    // Validate token
    claims, err := am.validateToken(token)
    if err != nil {
        return am.unauthorizedResponse("Invalid token")
    }
    
    // Add claims to context
    ctx = context.WithValue(ctx, "userClaims", claims)
    
    // Continue to next handler
    return next.ServeHTTP(w, r)
}

func (am *AuthMiddleware) validateToken(token string) (*Claims, error) {
    parts := strings.Split(token, ".")
    if len(parts) != 3 {
        return nil, fmt.Errorf("invalid token format")
    }
    
    // Verify signature
    message := parts[0] + "." + parts[1]
    expectedSignature := am.createSignature(message)
    if parts[2] != expectedSignature {
        return nil, fmt.Errorf("invalid signature")
    }
    
    // Decode claims
    claimsJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
    if err != nil {
        return nil, err
    }
    
    var claims Claims
    if err := json.Unmarshal(claimsJSON, &claims); err != nil {
        return nil, err
    }
    
    // Check expiration
    if time.Now().Unix() > claims.ExpiresAt {
        return nil, fmt.Errorf("token expired")
    }
    
    // Check issuer
    if claims.Issuer != am.issuer {
        return nil, fmt.Errorf("invalid issuer")
    }
    
    return &claims, nil
}

func (am *AuthMiddleware) createSignature(message string) string {
    h := hmac.New(sha256.New, am.secretKey)
    h.Write([]byte(message))
    return base64.RawURLEncoding.EncodeToString(h.Sum(nil))
}

func (am *AuthMiddleware) unauthorizedResponse(message string) http.ResponseWriter {
    w := &ResponseWriter{}
    w.WriteHeader(http.StatusUnauthorized)
    w.Write([]byte(fmt.Sprintf(`{"error": "%s"}`, message)))
    return w
}

type Claims struct {
    UserID    string   `json:"user_id"`
    Username  string   `json:"username"`
    Roles     []string `json:"roles"`
    ExpiresAt int64    `json:"exp"`
    IssuedAt  int64    `json:"iat"`
    Issuer    string   `json:"iss"`
}

// Role-based Authorization
type RBACMiddleware struct {
    requiredRoles map[string][]string // path -> required roles
}

func NewRBACMiddleware() *RBACMiddleware {
    return &RBACMiddleware{
        requiredRoles: make(map[string][]string),
    }
}

func (rbac *RBACMiddleware) AddRouteRoles(path string, roles []string) {
    rbac.requiredRoles[path] = roles
}

func (rbac *RBACMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    // Get user claims from context
    claims, ok := ctx.Value("userClaims").(*Claims)
    if !ok {
        return rbac.forbiddenResponse("No user claims found")
    }
    
    // Check if user has required roles
    requiredRoles, exists := rbac.requiredRoles[req.URL.Path]
    if !exists {
        // No role requirements for this path
        return next.ServeHTTP(w, r)
    }
    
    userRoles := claims.Roles
    hasRequiredRole := false
    
    for _, requiredRole := range requiredRoles {
        for _, userRole := range userRoles {
            if userRole == requiredRole {
                hasRequiredRole = true
                break
            }
        }
        if hasRequiredRole {
            break
        }
    }
    
    if !hasRequiredRole {
        return rbac.forbiddenResponse("Insufficient permissions")
    }
    
    return next.ServeHTTP(w, r)
}

func (rbac *RBACMiddleware) forbiddenResponse(message string) http.ResponseWriter {
    w := &ResponseWriter{}
    w.WriteHeader(http.StatusForbidden)
    w.Write([]byte(fmt.Sprintf(`{"error": "%s"}`, message)))
    return w
}
```

## Rate Limiting & Throttling

### Advanced Rate Limiting

```go
package main

import (
    "sync"
    "time"
)

type RateLimiter struct {
    limiters map[string]*Limiter
    mutex    sync.RWMutex
}

type Limiter struct {
    limit     int
    window    time.Duration
    requests  []time.Time
    mutex     sync.Mutex
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        limiters: make(map[string]*Limiter),
    }
}

func (rl *RateLimiter) AddLimiter(key string, limit int, window time.Duration) {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    rl.limiters[key] = &Limiter{
        limit:    limit,
        window:   window,
        requests: make([]time.Time, 0),
    }
}

func (rl *RateLimiter) IsAllowed(key string) bool {
    rl.mutex.RLock()
    limiter, exists := rl.limiters[key]
    rl.mutex.RUnlock()
    
    if !exists {
        return true // No limit configured
    }
    
    return limiter.IsAllowed()
}

func (l *Limiter) IsAllowed() bool {
    l.mutex.Lock()
    defer l.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-l.window)
    
    // Remove old requests
    var validRequests []time.Time
    for _, reqTime := range l.requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    l.requests = validRequests
    
    // Check if under limit
    if len(l.requests) >= l.limit {
        return false
    }
    
    // Add current request
    l.requests = append(l.requests, now)
    return true
}

// Token Bucket Rate Limiter
type TokenBucket struct {
    capacity     int
    tokens       int
    refillRate   int
    lastRefill   time.Time
    mutex        sync.Mutex
}

func NewTokenBucket(capacity, refillRate int) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        tokens:     capacity,
        refillRate: refillRate,
        lastRefill: time.Now(),
    }
}

func (tb *TokenBucket) TryConsume(tokens int) bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()
    
    // Refill tokens
    now := time.Now()
    elapsed := now.Sub(tb.lastRefill)
    tokensToAdd := int(elapsed.Seconds()) * tb.refillRate
    
    if tokensToAdd > 0 {
        tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
        tb.lastRefill = now
    }
    
    // Check if enough tokens
    if tb.tokens >= tokens {
        tb.tokens -= tokens
        return true
    }
    
    return false
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Sliding Window Rate Limiter
type SlidingWindowLimiter struct {
    limit     int
    window    time.Duration
    requests  []time.Time
    mutex     sync.Mutex
}

func NewSlidingWindowLimiter(limit int, window time.Duration) *SlidingWindowLimiter {
    return &SlidingWindowLimiter{
        limit:    limit,
        window:   window,
        requests: make([]time.Time, 0),
    }
}

func (swl *SlidingWindowLimiter) IsAllowed() bool {
    swl.mutex.Lock()
    defer swl.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-swl.window)
    
    // Remove old requests
    var validRequests []time.Time
    for _, reqTime := range swl.requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    swl.requests = validRequests
    
    // Check if under limit
    if len(swl.requests) >= swl.limit {
        return false
    }
    
    // Add current request
    swl.requests = append(swl.requests, now)
    return true
}

// Rate Limiting Middleware
type RateLimitMiddleware struct {
    rateLimiter *RateLimiter
}

func NewRateLimitMiddleware(rateLimiter *RateLimiter) *RateLimitMiddleware {
    return &RateLimitMiddleware{
        rateLimiter: rateLimiter,
    }
}

func (rlm *RateLimitMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    // Get client identifier
    clientID := rlm.getClientID(req)
    
    // Check rate limit
    if !rlm.rateLimiter.IsAllowed(clientID) {
        return rlm.rateLimitExceededResponse()
    }
    
    return next.ServeHTTP(w, r)
}

func (rlm *RateLimitMiddleware) getClientID(req *http.Request) string {
    // Try to get client IP
    clientIP := req.Header.Get("X-Forwarded-For")
    if clientIP == "" {
        clientIP = req.Header.Get("X-Real-IP")
    }
    if clientIP == "" {
        clientIP = req.RemoteAddr
    }
    
    return clientIP
}

func (rlm *RateLimitMiddleware) rateLimitExceededResponse() http.ResponseWriter {
    w := &ResponseWriter{}
    w.WriteHeader(http.StatusTooManyRequests)
    w.Write([]byte(`{"error": "Rate limit exceeded"}`))
    return w
}
```

## Circuit Breaker Pattern

### Circuit Breaker Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type CircuitBreaker struct {
    name            string
    maxFailures     int
    timeout         time.Duration
    resetTimeout    time.Duration
    state           CircuitState
    failureCount    int
    lastFailureTime time.Time
    mutex           sync.RWMutex
}

type CircuitState int

const (
    CLOSED CircuitState = iota
    OPEN
    HALF_OPEN
)

func NewCircuitBreaker(name string, maxFailures int, timeout, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        name:         name,
        maxFailures:  maxFailures,
        timeout:      timeout,
        resetTimeout: resetTimeout,
        state:        CLOSED,
    }
}

func (cb *CircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    if !cb.canExecute() {
        return nil, fmt.Errorf("circuit breaker %s is open", cb.name)
    }
    
    result, err := fn()
    
    if err != nil {
        cb.onFailure()
        return nil, err
    }
    
    cb.onSuccess()
    return result, nil
}

func (cb *CircuitBreaker) canExecute() bool {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    
    switch cb.state {
    case CLOSED:
        return true
    case OPEN:
        if time.Since(cb.lastFailureTime) > cb.resetTimeout {
            cb.state = HALF_OPEN
            return true
        }
        return false
    case HALF_OPEN:
        return true
    default:
        return false
    }
}

func (cb *CircuitBreaker) onSuccess() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    cb.failureCount = 0
    cb.state = CLOSED
}

func (cb *CircuitBreaker) onFailure() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    cb.failureCount++
    cb.lastFailureTime = time.Now()
    
    if cb.failureCount >= cb.maxFailures {
        cb.state = OPEN
    }
}

func (cb *CircuitBreaker) GetState() CircuitState {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    return cb.state
}

// Circuit Breaker Middleware
type CircuitBreakerMiddleware struct {
    circuitBreakers map[string]*CircuitBreaker
    mutex           sync.RWMutex
}

func NewCircuitBreakerMiddleware() *CircuitBreakerMiddleware {
    return &CircuitBreakerMiddleware{
        circuitBreakers: make(map[string]*CircuitBreaker),
    }
}

func (cbm *CircuitBreakerMiddleware) AddCircuitBreaker(serviceName string, cb *CircuitBreaker) {
    cbm.mutex.Lock()
    defer cbm.mutex.Unlock()
    cbm.circuitBreakers[serviceName] = cb
}

func (cbm *CircuitBreakerMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    // Get service name from context or request
    serviceName := cbm.getServiceName(req)
    
    cbm.mutex.RLock()
    cb, exists := cbm.circuitBreakers[serviceName]
    cbm.mutex.RUnlock()
    
    if !exists {
        return next.ServeHTTP(w, r)
    }
    
    // Execute with circuit breaker
    result, err := cb.Execute(func() (interface{}, error) {
        // Forward request
        return cbm.forwardRequest(ctx, req)
    })
    
    if err != nil {
        return cbm.errorResponse(err)
    }
    
    return result.(http.ResponseWriter)
}

func (cbm *CircuitBreakerMiddleware) getServiceName(req *http.Request) string {
    // Extract service name from request context or headers
    if serviceName := req.Header.Get("X-Service-Name"); serviceName != "" {
        return serviceName
    }
    
    // Default to path-based service name
    return req.URL.Path
}

func (cbm *CircuitBreakerMiddleware) forwardRequest(ctx context.Context, req *http.Request) (interface{}, error) {
    // Implementation would forward the request
    // This is a simplified version
    return nil, nil
}

func (cbm *CircuitBreakerMiddleware) errorResponse(err error) http.ResponseWriter {
    w := &ResponseWriter{}
    w.WriteHeader(http.StatusServiceUnavailable)
    w.Write([]byte(fmt.Sprintf(`{"error": "%s"}`, err.Error())))
    return w
}
```

## Service Discovery

### Service Registry Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type ServiceRegistry struct {
    services map[string]*ServiceInfo
    mutex    sync.RWMutex
}

type ServiceInfo struct {
    Name      string
    Endpoints []*Endpoint
    Health    bool
    mutex     sync.RWMutex
}

type Endpoint struct {
    Address string
    Port    int
    Health  bool
    mutex   sync.RWMutex
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string]*ServiceInfo),
    }
}

func (sr *ServiceRegistry) Register(serviceName, address string, port int) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    endpoint := &Endpoint{
        Address: address,
        Port:    port,
        Health:  true,
    }
    
    if service, exists := sr.services[serviceName]; exists {
        service.mutex.Lock()
        service.Endpoints = append(service.Endpoints, endpoint)
        service.mutex.Unlock()
    } else {
        sr.services[serviceName] = &ServiceInfo{
            Name:      serviceName,
            Endpoints: []*Endpoint{endpoint},
            Health:    true,
        }
    }
}

func (sr *ServiceRegistry) Deregister(serviceName, address string, port int) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    if service, exists := sr.services[serviceName]; exists {
        service.mutex.Lock()
        var newEndpoints []*Endpoint
        for _, endpoint := range service.Endpoints {
            if endpoint.Address != address || endpoint.Port != port {
                newEndpoints = append(newEndpoints, endpoint)
            }
        }
        service.Endpoints = newEndpoints
        service.mutex.Unlock()
    }
}

func (sr *ServiceRegistry) GetService(serviceName string) (*ServiceInfo, bool) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    service, exists := sr.services[serviceName]
    return service, exists
}

func (sr *ServiceRegistry) GetHealthyEndpoints(serviceName string) []*Endpoint {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    service, exists := sr.services[serviceName]
    if !exists {
        return nil
    }
    
    service.mutex.RLock()
    defer service.mutex.RUnlock()
    
    var healthyEndpoints []*Endpoint
    for _, endpoint := range service.Endpoints {
        endpoint.mutex.RLock()
        if endpoint.Health {
            healthyEndpoints = append(healthyEndpoints, endpoint)
        }
        endpoint.mutex.RUnlock()
    }
    
    return healthyEndpoints
}

func (sr *ServiceRegistry) SetEndpointHealth(serviceName, address string, port int, health bool) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    if service, exists := sr.services[serviceName]; exists {
        service.mutex.RLock()
        for _, endpoint := range service.Endpoints {
            if endpoint.Address == address && endpoint.Port == port {
                endpoint.mutex.Lock()
                endpoint.Health = health
                endpoint.mutex.Unlock()
                break
            }
        }
        service.mutex.RUnlock()
    }
}

// Health Checker
type HealthChecker struct {
    registry    *ServiceRegistry
    interval    time.Duration
    timeout     time.Duration
    stopCh      chan bool
}

func NewHealthChecker(registry *ServiceRegistry, interval, timeout time.Duration) *HealthChecker {
    return &HealthChecker{
        registry: registry,
        interval: interval,
        timeout:  timeout,
        stopCh:   make(chan bool),
    }
}

func (hc *HealthChecker) Start() {
    go hc.checkHealth()
}

func (hc *HealthChecker) Stop() {
    close(hc.stopCh)
}

func (hc *HealthChecker) checkHealth() {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            hc.performHealthChecks()
        case <-hc.stopCh:
            return
        }
    }
}

func (hc *HealthChecker) performHealthChecks() {
    hc.registry.mutex.RLock()
    services := make(map[string]*ServiceInfo)
    for name, service := range hc.registry.services {
        services[name] = service
    }
    hc.registry.mutex.RUnlock()
    
    for serviceName, service := range services {
        service.mutex.RLock()
        endpoints := make([]*Endpoint, len(service.Endpoints))
        copy(endpoints, service.Endpoints)
        service.mutex.RUnlock()
        
        for _, endpoint := range endpoints {
            go hc.checkEndpointHealth(serviceName, endpoint)
        }
    }
}

func (hc *HealthChecker) checkEndpointHealth(serviceName string, endpoint *Endpoint) {
    // Create health check URL
    healthURL := fmt.Sprintf("http://%s:%d/health", endpoint.Address, endpoint.Port)
    
    // Create timeout context
    ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
    defer cancel()
    
    // Make health check request
    req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
    if err != nil {
        hc.registry.SetEndpointHealth(serviceName, endpoint.Address, endpoint.Port, false)
        return
    }
    
    client := &http.Client{Timeout: hc.timeout}
    resp, err := client.Do(req)
    if err != nil {
        hc.registry.SetEndpointHealth(serviceName, endpoint.Address, endpoint.Port, false)
        return
    }
    defer resp.Body.Close()
    
    // Check response status
    health := resp.StatusCode == http.StatusOK
    hc.registry.SetEndpointHealth(serviceName, endpoint.Address, endpoint.Port, health)
}
```

## Interview Questions

### Basic Concepts
1. **What is an API Gateway and why is it important?**
2. **How do you implement request routing in an API Gateway?**
3. **What are the different types of load balancing algorithms?**
4. **How do you handle authentication and authorization in microservices?**
5. **What is the circuit breaker pattern?**

### Advanced Topics
1. **How would you implement rate limiting in an API Gateway?**
2. **How do you handle service discovery in a distributed system?**
3. **What are the challenges of implementing an API Gateway?**
4. **How do you ensure high availability of an API Gateway?**
5. **How do you handle versioning in an API Gateway?**

### System Design
1. **Design an API Gateway for a microservices architecture.**
2. **How would you implement service mesh communication?**
3. **Design a rate limiting system.**
4. **How would you implement circuit breakers?**
5. **Design a service discovery system.**

## Conclusion

API Gateway and microservices communication are essential for building scalable distributed systems. Key areas to master:

- **API Gateway**: Request routing, load balancing, authentication
- **Service Discovery**: Service registration, health checking, load balancing
- **Circuit Breaker**: Fault tolerance, resilience patterns
- **Rate Limiting**: Traffic control, throttling strategies
- **Authentication**: JWT, OAuth, RBAC
- **Monitoring**: Metrics, logging, tracing

Understanding these concepts helps in:
- Building scalable microservices
- Implementing fault-tolerant systems
- Managing service communication
- Ensuring security and performance
- Preparing for technical interviews

This guide provides a comprehensive foundation for API Gateway concepts and their practical implementation in Go.


## Load Balancing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #load-balancing -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
