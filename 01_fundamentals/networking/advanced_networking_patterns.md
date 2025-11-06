---
# Auto-generated front matter
Title: Advanced Networking Patterns
LastUpdated: 2025-11-06T20:45:58.666744
Tags: []
Status: draft
---

# Advanced Networking Patterns

Advanced networking patterns and practices for distributed systems.

## 🎯 Network Architecture

### Microservices Communication
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Service A     │    │   Service B     │    │   Service C     │
│   (HTTP/gRPC)   │────│   (gRPC/HTTP)   │────│   (HTTP/gRPC)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Load Balancer)│
                    └─────────────────┘
```

### Service Mesh Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Service A     │    │   Service B     │    │   Service C     │
│   + Sidecar     │    │   + Sidecar     │    │   + Sidecar     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Control Plane │
                    │   (Istio/Linkerd)│
                    └─────────────────┘
```

## 🔧 Load Balancing

### Load Balancer Implementation
```go
type LoadBalancer struct {
    servers    []Server
    strategy   LoadBalancingStrategy
    healthCheck HealthChecker
    mutex      sync.RWMutex
}

type Server struct {
    Address string
    Weight  int
    Healthy bool
}

type LoadBalancingStrategy interface {
    SelectServer(servers []Server) *Server
}

type RoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinStrategy) SelectServer(servers []Server) *Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    healthyServers := rr.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    server := healthyServers[rr.current]
    rr.current = (rr.current + 1) % len(healthyServers)
    return &server
}

type WeightedRoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (wrr *WeightedRoundRobinStrategy) SelectServer(servers []Server) *Server {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()
    
    healthyServers := wrr.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    // Calculate total weight
    totalWeight := 0
    for _, server := range healthyServers {
        totalWeight += server.Weight
    }
    
    // Select server based on weight
    target := wrr.current % totalWeight
    current := 0
    
    for _, server := range healthyServers {
        current += server.Weight
        if current > target {
            wrr.current++
            return &server
        }
    }
    
    return &healthyServers[0]
}

type LeastConnectionsStrategy struct{}

func (lc *LeastConnectionsStrategy) SelectServer(servers []Server) *Server {
    healthyServers := lc.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    minConnections := healthyServers[0].Connections
    selectedServer := &healthyServers[0]
    
    for i := 1; i < len(healthyServers); i++ {
        if healthyServers[i].Connections < minConnections {
            minConnections = healthyServers[i].Connections
            selectedServer = &healthyServers[i]
        }
    }
    
    return selectedServer
}

func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    return lb.strategy.SelectServer(lb.servers)
}

func (lb *LoadBalancer) UpdateServerHealth(address string, healthy bool) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i := range lb.servers {
        if lb.servers[i].Address == address {
            lb.servers[i].Healthy = healthy
            break
        }
    }
}
```

### Health Checking
```go
type HealthChecker struct {
    interval    time.Duration
    timeout     time.Duration
    httpClient  *http.Client
}

type HealthCheckResult struct {
    Server   string
    Healthy  bool
    Latency  time.Duration
    Error    error
    Timestamp time.Time
}

func (hc *HealthChecker) CheckHealth(server Server) HealthCheckResult {
    start := time.Now()
    
    // Create health check request
    req, err := http.NewRequest("GET", server.Address+"/health", nil)
    if err != nil {
        return HealthCheckResult{
            Server:   server.Address,
            Healthy:  false,
            Latency:  time.Since(start),
            Error:    err,
            Timestamp: time.Now(),
        }
    }
    
    // Set timeout
    ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
    defer cancel()
    
    req = req.WithContext(ctx)
    
    // Make request
    resp, err := hc.httpClient.Do(req)
    if err != nil {
        return HealthCheckResult{
            Server:   server.Address,
            Healthy:  false,
            Latency:  time.Since(start),
            Error:    err,
            Timestamp: time.Now(),
        }
    }
    defer resp.Body.Close()
    
    // Check response status
    healthy := resp.StatusCode >= 200 && resp.StatusCode < 300
    
    return HealthCheckResult{
        Server:   server.Address,
        Healthy:  healthy,
        Latency:  time.Since(start),
        Error:    nil,
        Timestamp: time.Now(),
    }
}

func (hc *HealthChecker) StartHealthChecks(servers []Server, callback func(HealthCheckResult)) {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for range ticker.C {
        for _, server := range servers {
            go func(s Server) {
                result := hc.CheckHealth(s)
                callback(result)
            }(server)
        }
    }
}
```

## 🌐 API Gateway

### API Gateway Implementation
```go
type APIGateway struct {
    routes      map[string]Route
    middleware  []Middleware
    loadBalancer LoadBalancer
    rateLimiter RateLimiter
    authService AuthService
    logger      Logger
}

type Route struct {
    Path        string
    Method      string
    Service     string
    Middleware  []Middleware
    RateLimit   RateLimit
    Auth        AuthConfig
}

type Middleware interface {
    Process(req *http.Request, next http.Handler) http.ResponseWriter
}

type RateLimitingMiddleware struct {
    rateLimiter RateLimiter
}

func (rlm *RateLimitingMiddleware) Process(req *http.Request, next http.Handler) http.ResponseWriter {
    // Get client IP
    clientIP := rlm.getClientIP(req)
    
    // Check rate limit
    if !rlm.rateLimiter.Allow(clientIP) {
        return rlm.rateLimiter.GetResponse()
    }
    
    // Continue to next middleware
    return next.ServeHTTP(req)
}

type AuthenticationMiddleware struct {
    authService AuthService
}

func (am *AuthenticationMiddleware) Process(req *http.Request, next http.Handler) http.ResponseWriter {
    // Extract token
    token := am.extractToken(req)
    if token == "" {
        return am.unauthorizedResponse()
    }
    
    // Validate token
    user, err := am.authService.ValidateToken(token)
    if err != nil {
        return am.unauthorizedResponse()
    }
    
    // Add user to context
    ctx := context.WithValue(req.Context(), "user", user)
    req = req.WithContext(ctx)
    
    // Continue to next middleware
    return next.ServeHTTP(req)
}

func (ag *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Find route
    route := ag.findRoute(r.URL.Path, r.Method)
    if route == nil {
        http.NotFound(w, r)
        return
    }
    
    // Apply middleware
    handler := ag.applyMiddleware(route, ag.proxyHandler)
    
    // Execute handler
    handler.ServeHTTP(w, r)
}

func (ag *APIGateway) proxyHandler(w http.ResponseWriter, r *http.Request) {
    // Get target service
    targetService := ag.getTargetService(r)
    if targetService == nil {
        http.NotFound(w, r)
        return
    }
    
    // Create proxy request
    proxyReq := ag.createProxyRequest(r, targetService)
    
    // Make request to target service
    resp, err := ag.httpClient.Do(proxyReq)
    if err != nil {
        http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
        return
    }
    defer resp.Body.Close()
    
    // Copy response headers
    for key, values := range resp.Header {
        for _, value := range values {
            w.Header().Add(key, value)
        }
    }
    
    // Set status code
    w.WriteHeader(resp.StatusCode)
    
    // Copy response body
    io.Copy(w, resp.Body)
}
```

## 🔄 Circuit Breaker

### Circuit Breaker Implementation
```go
type CircuitBreaker struct {
    name          string
    maxRequests   uint32
    interval      time.Duration
    timeout       time.Duration
    readyToTrip   func(counts Counts) bool
    onStateChange func(name string, from State, to State)
    
    mutex      sync.Mutex
    state      State
    generation uint64
    counts     Counts
    expiry     time.Time
}

type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

type Counts struct {
    Requests             uint32
    TotalSuccesses       uint32
    TotalFailures        uint32
    ConsecutiveSuccesses uint32
    ConsecutiveFailures  uint32
}

func (cb *CircuitBreaker) Execute(req func() (interface{}, error)) (interface{}, error) {
    generation, err := cb.beforeRequest()
    if err != nil {
        return nil, err
    }
    
    defer func() {
        e := recover()
        if e != nil {
            cb.afterRequest(generation, false)
            panic(e)
        }
    }()
    
    result, err := req()
    cb.afterRequest(generation, err == nil)
    return result, err
}

func (cb *CircuitBreaker) beforeRequest() (uint64, error) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    
    if state == StateOpen {
        return generation, ErrOpenState
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
        return generation, ErrTooManyRequests
    }
    
    cb.counts.onRequest()
    return generation, nil
}

func (cb *CircuitBreaker) afterRequest(before uint64, success bool) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    if generation != before {
        return
    }
    
    if success {
        cb.onSuccess(state, now)
    } else {
        cb.onFailure(state, now)
    }
}

func (cb *CircuitBreaker) currentState(now time.Time) (State, uint64) {
    switch cb.state {
    case StateClosed:
        if !cb.expiry.IsZero() && cb.expiry.Before(now) {
            cb.toNewGeneration(now)
        }
    case StateOpen:
        if cb.expiry.Before(now) {
            cb.setState(StateHalfOpen, now)
        }
    }
    return cb.state, cb.generation
}

func (cb *CircuitBreaker) onSuccess(state State, now time.Time) {
    switch state {
    case StateClosed:
        cb.counts.onSuccess()
    case StateHalfOpen:
        cb.counts.onSuccess()
        if cb.counts.ConsecutiveSuccesses >= cb.maxRequests {
            cb.setState(StateClosed, now)
        }
    }
}

func (cb *CircuitBreaker) onFailure(state State, now time.Time) {
    switch state {
    case StateClosed:
        cb.counts.onFailure()
        if cb.readyToTrip(cb.counts) {
            cb.setState(StateOpen, now)
        }
    case StateHalfOpen:
        cb.setState(StateOpen, now)
    }
}
```

## 📡 Service Discovery

### Service Discovery Implementation
```go
type ServiceRegistry struct {
    services  map[string][]ServiceInstance
    mutex     sync.RWMutex
    notifier  ServiceNotifier
}

type ServiceInstance struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Metadata map[string]string
    Health   HealthStatus
    LastSeen time.Time
}

type HealthStatus string

const (
    HealthUp   HealthStatus = "up"
    HealthDown HealthStatus = "down"
)

type ServiceNotifier interface {
    NotifyServiceAdded(service ServiceInstance)
    NotifyServiceRemoved(service ServiceInstance)
    NotifyServiceUpdated(service ServiceInstance)
}

func (sr *ServiceRegistry) Register(instance ServiceInstance) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    // Update last seen
    instance.LastSeen = time.Now()
    
    // Add or update service
    if sr.services[instance.Name] == nil {
        sr.services[instance.Name] = []ServiceInstance{}
    }
    
    // Check if instance already exists
    for i, existing := range sr.services[instance.Name] {
        if existing.ID == instance.ID {
            sr.services[instance.Name][i] = instance
            sr.notifier.NotifyServiceUpdated(instance)
            return nil
        }
    }
    
    // Add new instance
    sr.services[instance.Name] = append(sr.services[instance.Name], instance)
    sr.notifier.NotifyServiceAdded(instance)
    
    return nil
}

func (sr *ServiceRegistry) Deregister(serviceName, instanceID string) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instances := sr.services[serviceName]
    for i, instance := range instances {
        if instance.ID == instanceID {
            // Remove instance
            sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
            sr.notifier.NotifyServiceRemoved(instance)
            return nil
        }
    }
    
    return errors.New("instance not found")
}

func (sr *ServiceRegistry) GetInstances(serviceName string) ([]ServiceInstance, error) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    instances := sr.services[serviceName]
    if instances == nil {
        return nil, errors.New("service not found")
    }
    
    // Filter healthy instances
    healthyInstances := []ServiceInstance{}
    for _, instance := range instances {
        if instance.Health == HealthUp {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    return healthyInstances, nil
}

func (sr *ServiceRegistry) Watch(serviceName string) <-chan []ServiceInstance {
    ch := make(chan []ServiceInstance, 1)
    
    go func() {
        defer close(ch)
        
        // Send initial instances
        instances, err := sr.GetInstances(serviceName)
        if err == nil {
            ch <- instances
        }
        
        // Watch for changes
        // This would typically use a notification mechanism
        // like etcd watches or consul watches
    }()
    
    return ch
}
```

## 🔐 Network Security

### TLS Configuration
```go
type TLSConfig struct {
    CertFile    string
    KeyFile     string
    CAFile      string
    MinVersion  uint16
    MaxVersion  uint16
    CipherSuites []uint16
}

func (tc *TLSConfig) GetTLSConfig() (*tls.Config, error) {
    config := &tls.Config{
        MinVersion:         tc.MinVersion,
        MaxVersion:         tc.MaxVersion,
        CipherSuites:       tc.CipherSuites,
        InsecureSkipVerify: false,
    }
    
    // Load certificate
    cert, err := tls.LoadX509KeyPair(tc.CertFile, tc.KeyFile)
    if err != nil {
        return nil, err
    }
    config.Certificates = []tls.Certificate{cert}
    
    // Load CA certificate
    if tc.CAFile != "" {
        caCert, err := ioutil.ReadFile(tc.CAFile)
        if err != nil {
            return nil, err
        }
        
        caCertPool := x509.NewCertPool()
        if !caCertPool.AppendCertsFromPEM(caCert) {
            return nil, errors.New("failed to parse CA certificate")
        }
        config.RootCAs = caCertPool
    }
    
    return config, nil
}

func (tc *TLSConfig) GetServerTLSConfig() (*tls.Config, error) {
    config, err := tc.GetTLSConfig()
    if err != nil {
        return nil, err
    }
    
    // Server-specific configuration
    config.ClientAuth = tls.RequireAndVerifyClientCert
    
    return config, nil
}

func (tc *TLSConfig) GetClientTLSConfig() (*tls.Config, error) {
    config, err := tc.GetTLSConfig()
    if err != nil {
        return nil, err
    }
    
    // Client-specific configuration
    config.ServerName = "localhost"
    
    return config, nil
}
```

### Network Policies
```go
type NetworkPolicy struct {
    Name      string
    Namespace string
    Rules     []PolicyRule
}

type PolicyRule struct {
    Ingress []IngressRule
    Egress  []EgressRule
}

type IngressRule struct {
    From   []NetworkPolicyPeer
    Ports  []NetworkPolicyPort
}

type EgressRule struct {
    To    []NetworkPolicyPeer
    Ports []NetworkPolicyPort
}

type NetworkPolicyPeer struct {
    PodSelector       *LabelSelector
    NamespaceSelector *LabelSelector
    IPBlock          *IPBlock
}

type NetworkPolicyPort struct {
    Protocol *string
    Port     *int32
}

type IPBlock struct {
    CIDR   string
    Except []string
}

type LabelSelector struct {
    MatchLabels map[string]string
    MatchExpressions []LabelSelectorRequirement
}

type LabelSelectorRequirement struct {
    Key      string
    Operator string
    Values   []string
}

func (np *NetworkPolicy) Validate() error {
    // Validate policy rules
    for _, rule := range np.Rules {
        if err := np.validateIngressRules(rule.Ingress); err != nil {
            return err
        }
        if err := np.validateEgressRules(rule.Egress); err != nil {
            return err
        }
    }
    
    return nil
}

func (np *NetworkPolicy) validateIngressRules(rules []IngressRule) error {
    for _, rule := range rules {
        for _, peer := range rule.From {
            if err := np.validatePeer(peer); err != nil {
                return err
            }
        }
        for _, port := range rule.Ports {
            if err := np.validatePort(port); err != nil {
                return err
            }
        }
    }
    return nil
}
```

## 🎯 Best Practices

### Network Design
1. **Load Balancing**: Implement proper load balancing
2. **Health Checks**: Regular health checks
3. **Circuit Breakers**: Implement circuit breakers
4. **Service Discovery**: Use service discovery
5. **Monitoring**: Comprehensive network monitoring

### Security
1. **TLS**: Use TLS for all communications
2. **Authentication**: Implement proper authentication
3. **Authorization**: Use proper authorization
4. **Network Policies**: Implement network policies
5. **Audit Logging**: Log all network operations

### Performance
1. **Connection Pooling**: Use connection pooling
2. **Caching**: Implement appropriate caching
3. **Compression**: Use compression where appropriate
4. **CDN**: Use CDN for static content
5. **Monitoring**: Monitor network performance

---

**Last Updated**: December 2024  
**Category**: Advanced Networking Patterns  
**Complexity**: Senior Level
