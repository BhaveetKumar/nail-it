# 🚀 Scaling Microservices: Service Mesh, Load Balancing, Circuit Breakers

> **Master microservices scaling patterns for distributed systems**

## 📚 Concept

Microservices scaling involves managing distributed services, handling failures, and ensuring high availability. Key patterns include service mesh, load balancing, and circuit breakers.

### Scaling Strategies

- **Horizontal Scaling**: Add more instances
- **Vertical Scaling**: Increase resources per instance
- **Load Balancing**: Distribute traffic across instances
- **Service Mesh**: Manage service-to-service communication

## 🛠️ Hands-on Example

### Service Mesh Implementation (Istio)

```yaml
# istio-config.yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: api-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - api.example.com
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: api-routes
spec:
  hosts:
    - api.example.com
  gateways:
    - api-gateway
  http:
    - match:
        - uri:
            prefix: /users
      route:
        - destination:
            host: user-service
            port:
              number: 8080
    - match:
        - uri:
            prefix: /posts
      route:
        - destination:
            host: post-service
            port:
              number: 8080
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: user-service
spec:
  host: user-service
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
    circuitBreaker:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

### Load Balancer Implementation (Go)

```go
package main

import (
    "context"
    "fmt"
    "log"
    "math/rand"
    "net/http"
    "sync"
    "time"
)

type Backend struct {
    URL    string
    Weight int
    Health bool
    mutex  sync.RWMutex
}

type LoadBalancer struct {
    backends []*Backend
    strategy string
    mutex    sync.RWMutex
}

func NewLoadBalancer(strategy string) *LoadBalancer {
    return &LoadBalancer{
        backends: make([]*Backend, 0),
        strategy: strategy,
    }
}

func (lb *LoadBalancer) AddBackend(url string, weight int) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()

    backend := &Backend{
        URL:    url,
        Weight: weight,
        Health: true,
    }

    lb.backends = append(lb.backends, backend)
}

func (lb *LoadBalancer) GetBackend() (*Backend, error) {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()

    if len(lb.backends) == 0 {
        return nil, fmt.Errorf("no backends available")
    }

    // Filter healthy backends
    healthyBackends := make([]*Backend, 0)
    for _, backend := range lb.backends {
        backend.mutex.RLock()
        if backend.Health {
            healthyBackends = append(healthyBackends, backend)
        }
        backend.mutex.RUnlock()
    }

    if len(healthyBackends) == 0 {
        return nil, fmt.Errorf("no healthy backends available")
    }

    switch lb.strategy {
    case "round_robin":
        return lb.roundRobin(healthyBackends)
    case "random":
        return lb.random(healthyBackends)
    case "weighted":
        return lb.weighted(healthyBackends)
    case "least_connections":
        return lb.leastConnections(healthyBackends)
    default:
        return lb.roundRobin(healthyBackends)
    }
}

func (lb *LoadBalancer) roundRobin(backends []*Backend) (*Backend, error) {
    // Simple round-robin implementation
    index := rand.Intn(len(backends))
    return backends[index], nil
}

func (lb *LoadBalancer) random(backends []*Backend) (*Backend, error) {
    index := rand.Intn(len(backends))
    return backends[index], nil
}

func (lb *LoadBalancer) weighted(backends []*Backend) (*Backend, error) {
    totalWeight := 0
    for _, backend := range backends {
        totalWeight += backend.Weight
    }

    if totalWeight == 0 {
        return lb.random(backends)
    }

    random := rand.Intn(totalWeight)
    current := 0

    for _, backend := range backends {
        current += backend.Weight
        if random < current {
            return backend, nil
        }
    }

    return backends[0], nil
}

func (lb *LoadBalancer) leastConnections(backends []*Backend) (*Backend, error) {
    // Simplified - in real implementation, track connection counts
    return lb.roundRobin(backends)
}

func (lb *LoadBalancer) HealthCheck() {
    for {
        lb.mutex.RLock()
        backends := make([]*Backend, len(lb.backends))
        copy(backends, lb.backends)
        lb.mutex.RUnlock()

        for _, backend := range backends {
            go func(b *Backend) {
                client := &http.Client{Timeout: 5 * time.Second}
                resp, err := client.Get(b.URL + "/health")

                b.mutex.Lock()
                if err != nil || resp.StatusCode != http.StatusOK {
                    b.Health = false
                } else {
                    b.Health = true
                }
                b.mutex.Unlock()

                if resp != nil {
                    resp.Body.Close()
                }
            }(backend)
        }

        time.Sleep(30 * time.Second)
    }
}

// HTTP handler
func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    backend, err := lb.GetBackend()
    if err != nil {
        http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
        return
    }

    // Forward request to backend
    client := &http.Client{Timeout: 30 * time.Second}
    req, err := http.NewRequest(r.Method, backend.URL+r.URL.Path, r.Body)
    if err != nil {
        http.Error(w, "Internal server error", http.StatusInternalServerError)
        return
    }

    // Copy headers
    for key, values := range r.Header {
        for _, value := range values {
            req.Header.Add(key, value)
        }
    }

    resp, err := client.Do(req)
    if err != nil {
        // Mark backend as unhealthy
        backend.mutex.Lock()
        backend.Health = false
        backend.mutex.Unlock()

        http.Error(w, "Backend error", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    // Copy response headers
    for key, values := range resp.Header {
        for _, value := range values {
            w.Header().Add(key, value)
        }
    }

    w.WriteHeader(resp.StatusCode)

    // Copy response body
    if _, err := w.Write([]byte{}); err != nil {
        log.Printf("Error writing response: %v", err)
    }
}
```

### Circuit Breaker Implementation

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "time"
)

type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    name          string
    maxRequests   uint32
    interval      time.Duration
    timeout       time.Duration
    readyToTrip   func(counts Counts) bool
    onStateChange func(name string, from, to CircuitState)

    mutex      sync.Mutex
    state      CircuitState
    generation uint64
    counts     Counts
    expiry     time.Time
}

type Counts struct {
    Requests             uint32
    TotalSuccesses       uint32
    TotalFailures        uint32
    ConsecutiveSuccesses uint32
    ConsecutiveFailures  uint32
}

func NewCircuitBreaker(name string, maxRequests uint32, interval, timeout time.Duration) *CircuitBreaker {
    cb := &CircuitBreaker{
        name:        name,
        maxRequests: maxRequests,
        interval:    interval,
        timeout:     timeout,
        readyToTrip: defaultReadyToTrip,
        onStateChange: func(name string, from, to CircuitState) {
            log.Printf("Circuit breaker %s: %s -> %s", name, from, to)
        },
    }

    cb.toNewGeneration(time.Now())
    return cb
}

func defaultReadyToTrip(counts Counts) bool {
    return counts.ConsecutiveFailures > 5
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
        return generation, errors.New("circuit breaker is open")
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
        return generation, errors.New("circuit breaker is half-open and max requests reached")
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

func (cb *CircuitBreaker) currentState(now time.Time) (CircuitState, uint64) {
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

func (cb *CircuitBreaker) onSuccess(state CircuitState, now time.Time) {
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

func (cb *CircuitBreaker) onFailure(state CircuitState, now time.Time) {
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

func (cb *CircuitBreaker) setState(state CircuitState, now time.Time) {
    if cb.state == state {
        return
    }

    prev := cb.state
    cb.state = state

    cb.toNewGeneration(now)

    if cb.onStateChange != nil {
        cb.onStateChange(cb.name, prev, state)
    }
}

func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
    cb.generation++
    cb.counts = Counts{}

    var zero time.Time
    switch cb.state {
    case StateClosed:
        if cb.interval == 0 {
            cb.expiry = zero
        } else {
            cb.expiry = now.Add(cb.interval)
        }
    case StateOpen:
        cb.expiry = now.Add(cb.timeout)
    default: // StateHalfOpen
        cb.expiry = zero
    }
}

func (c *Counts) onRequest() {
    c.Requests++
}

func (c *Counts) onSuccess() {
    c.TotalSuccesses++
    c.ConsecutiveSuccesses++
    c.ConsecutiveFailures = 0
}

func (c *Counts) onFailure() {
    c.TotalFailures++
    c.ConsecutiveFailures++
    c.ConsecutiveSuccesses = 0
}

// Usage example
func main() {
    cb := NewCircuitBreaker("user-service", 3, time.Minute, time.Minute*2)

    // Simulate service call
    result, err := cb.Execute(func() (interface{}, error) {
        // Simulate API call
        time.Sleep(100 * time.Millisecond)

        // Simulate random failures
        if rand.Float32() < 0.3 {
            return nil, errors.New("service error")
        }

        return "success", nil
    })

    if err != nil {
        log.Printf("Circuit breaker error: %v", err)
    } else {
        log.Printf("Result: %v", result)
    }
}
```

### Service Discovery

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

type Service struct {
    Name     string            `json:"name"`
    Address  string            `json:"address"`
    Port     int               `json:"port"`
    Health   bool              `json:"health"`
    Metadata map[string]string `json:"metadata"`
    LastSeen time.Time         `json:"last_seen"`
}

type ServiceRegistry struct {
    services map[string][]*Service
    mutex    sync.RWMutex
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string][]*Service),
    }
}

func (sr *ServiceRegistry) Register(service *Service) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()

    service.LastSeen = time.Now()

    if sr.services[service.Name] == nil {
        sr.services[service.Name] = make([]*Service, 0)
    }

    // Check if service already exists
    for i, existing := range sr.services[service.Name] {
        if existing.Address == service.Address && existing.Port == service.Port {
            sr.services[service.Name][i] = service
            return
        }
    }

    sr.services[service.Name] = append(sr.services[service.Name], service)
}

func (sr *ServiceRegistry) Deregister(name, address string, port int) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()

    services := sr.services[name]
    for i, service := range services {
        if service.Address == address && service.Port == port {
            sr.services[name] = append(services[:i], services[i+1:]...)
            break
        }
    }
}

func (sr *ServiceRegistry) GetServices(name string) []*Service {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()

    services := sr.services[name]
    healthy := make([]*Service, 0)

    for _, service := range services {
        if service.Health && time.Since(service.LastSeen) < time.Minute*5 {
            healthy = append(healthy, service)
        }
    }

    return healthy
}

func (sr *ServiceRegistry) HealthCheck() {
    for {
        sr.mutex.Lock()
        for name, services := range sr.services {
            for _, service := range services {
                go func(s *Service) {
                    client := &http.Client{Timeout: 5 * time.Second}
                    resp, err := client.Get(fmt.Sprintf("http://%s:%d/health", s.Address, s.Port))

                    s.mutex.Lock()
                    if err != nil || resp.StatusCode != http.StatusOK {
                        s.Health = false
                    } else {
                        s.Health = true
                        s.LastSeen = time.Now()
                    }
                    s.mutex.Unlock()

                    if resp != nil {
                        resp.Body.Close()
                    }
                }(service)
            }
        }
        sr.mutex.Unlock()

        time.Sleep(30 * time.Second)
    }
}

// HTTP handlers
func (sr *ServiceRegistry) RegisterHandler(w http.ResponseWriter, r *http.Request) {
    var service Service
    if err := json.NewDecoder(r.Body).Decode(&service); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    sr.Register(&service)
    w.WriteHeader(http.StatusOK)
}

func (sr *ServiceRegistry) DiscoverHandler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("name")
    if name == "" {
        http.Error(w, "Service name required", http.StatusBadRequest)
        return
    }

    services := sr.GetServices(name)
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "services": services,
        "count":    len(services),
    })
}
```

## 🚀 Best Practices

### 1. Service Mesh Configuration

```yaml
# Traffic management
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
    - user-service
  http:
    - match:
        - headers:
            version:
              exact: v2
      route:
        - destination:
            host: user-service
            subset: v2
    - route:
        - destination:
            host: user-service
            subset: v1
          weight: 90
        - destination:
            host: user-service
            subset: v2
          weight: 10
```

### 2. Circuit Breaker Configuration

```go
// Configure circuit breaker based on service characteristics
func NewServiceCircuitBreaker(serviceName string) *CircuitBreaker {
    switch serviceName {
    case "user-service":
        return NewCircuitBreaker(serviceName, 5, time.Minute, time.Minute*2)
    case "payment-service":
        return NewCircuitBreaker(serviceName, 3, time.Minute*2, time.Minute*5)
    default:
        return NewCircuitBreaker(serviceName, 10, time.Minute, time.Minute)
    }
}
```

### 3. Load Balancing Strategies

```go
// Choose strategy based on service type
func (lb *LoadBalancer) SelectStrategy(serviceType string) {
    switch serviceType {
    case "stateless":
        lb.strategy = "round_robin"
    case "cpu_intensive":
        lb.strategy = "least_connections"
    case "weighted":
        lb.strategy = "weighted"
    default:
        lb.strategy = "random"
    }
}
```

## 🏢 Industry Insights

### Netflix's Microservices

- **Eureka**: Service discovery
- **Hystrix**: Circuit breaker
- **Ribbon**: Load balancing
- **Zuul**: API gateway

### Uber's Microservices

- **Ringpop**: Consistent hashing
- **TChannel**: RPC framework
- **Hyperbahn**: Service mesh
- **Jaeger**: Distributed tracing

### Amazon's Microservices

- **AWS ECS**: Container orchestration
- **Application Load Balancer**: Load balancing
- **Service Discovery**: AWS Cloud Map
- **X-Ray**: Distributed tracing

## 🎯 Interview Questions

### Basic Level

1. **What is a service mesh?**

   - Infrastructure layer for service-to-service communication
   - Handles load balancing, service discovery, security
   - Examples: Istio, Linkerd, Consul Connect

2. **What are the benefits of load balancing?**

   - Distribute traffic across multiple instances
   - Improve availability and performance
   - Handle failures gracefully
   - Scale horizontally

3. **What is a circuit breaker?**
   - Prevents cascading failures
   - Opens when failure threshold is reached
   - Allows recovery attempts
   - Improves system resilience

### Intermediate Level

4. **How do you implement service discovery?**

   ```go
   type ServiceRegistry struct {
       services map[string][]*Service
       mutex    sync.RWMutex
   }

   func (sr *ServiceRegistry) Register(service *Service) {
       sr.mutex.Lock()
       defer sr.mutex.Unlock()
       sr.services[service.Name] = append(sr.services[service.Name], service)
   }
   ```

5. **How do you handle service failures?**

   - Circuit breaker pattern
   - Retry with exponential backoff
   - Fallback mechanisms
   - Health checks and monitoring

6. **What are different load balancing algorithms?**
   - Round Robin: Distribute requests evenly
   - Weighted: Based on server capacity
   - Least Connections: Based on active connections
   - Random: Random selection

### Advanced Level

7. **How do you implement distributed tracing?**

   - Trace ID propagation
   - Span creation and correlation
   - Sampling strategies
   - Integration with monitoring systems

8. **How do you handle service versioning?**

   - API versioning strategies
   - Backward compatibility
   - Gradual rollout
   - Feature flags

9. **How do you implement service mesh security?**
   - mTLS for service-to-service communication
   - Policy enforcement
   - Identity management
   - Traffic encryption

---

**Next**: [Cloud Fundamentals](./CloudFundamentals.md) - Cloud computing basics, virtualization vs containers
