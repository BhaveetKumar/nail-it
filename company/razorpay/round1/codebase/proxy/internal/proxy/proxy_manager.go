package proxy

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ProxyManager manages all proxy services
type ProxyManager struct {
	services map[string]*ServiceProxy
	config   ProxyConfig
	logger   Logger
	metrics  Metrics
	mu       sync.RWMutex
	stats    *ProxyStats
}

// NewProxyManager creates a new proxy manager
func NewProxyManager(config ProxyConfig, logger Logger, metrics Metrics) *ProxyManager {
	return &ProxyManager{
		services: make(map[string]*ServiceProxy),
		config:   config,
		logger:   logger,
		metrics:  metrics,
		stats:    &ProxyStats{},
	}
}

// RegisterService registers a service with its proxy
func (pm *ProxyManager) RegisterService(service Service, proxy *ServiceProxy) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.services[service.GetName()] = proxy
	pm.logger.Info("Service registered", "service", service.GetName())
}

// ProcessRequest processes a request through the appropriate service proxy
func (pm *ProxyManager) ProcessRequest(ctx context.Context, serviceName string, request interface{}) (interface{}, error) {
	pm.mu.RLock()
	proxy, exists := pm.services[serviceName]
	pm.mu.RUnlock()

	if !exists {
		pm.metrics.IncrementCounter("proxy_service_not_found", map[string]string{"service": serviceName})
		return nil, fmt.Errorf("service %s not found", serviceName)
	}

	// Update stats
	pm.mu.Lock()
	pm.stats.TotalRequests++
	pm.mu.Unlock()

	// Process through proxy
	return proxy.Process(ctx, request)
}

// GetServiceHealth returns the health status of a service
func (pm *ProxyManager) GetServiceHealth(ctx context.Context, serviceName string) (*HealthCheck, error) {
	pm.mu.RLock()
	proxy, exists := pm.services[serviceName]
	pm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("service %s not found", serviceName)
	}

	start := time.Now()
	healthy := proxy.IsHealthy(ctx)
	latency := time.Since(start)

	status := "healthy"
	if !healthy {
		status = "unhealthy"
	}

	return &HealthCheck{
		Service:   serviceName,
		Healthy:   healthy,
		Status:    status,
		Message:   fmt.Sprintf("Service %s is %s", serviceName, status),
		Timestamp: time.Now(),
		Latency:   latency,
	}, nil
}

// GetAllServicesHealth returns the health status of all services
func (pm *ProxyManager) GetAllServicesHealth(ctx context.Context) ([]*HealthCheck, error) {
	pm.mu.RLock()
	services := make([]*ServiceProxy, 0, len(pm.services))
	for _, service := range pm.services {
		services = append(services, service)
	}
	pm.mu.RUnlock()

	healthChecks := make([]*HealthCheck, 0, len(services))

	for _, service := range services {
		health, err := pm.GetServiceHealth(ctx, service.GetName())
		if err != nil {
			pm.logger.Error("Failed to get service health", "service", service.GetName(), "error", err)
			continue
		}
		healthChecks = append(healthChecks, health)
	}

	return healthChecks, nil
}

// GetStats returns proxy statistics
func (pm *ProxyManager) GetStats() *ProxyStats {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Create a copy to avoid race conditions
	stats := *pm.stats
	return &stats
}

// ResetStats resets proxy statistics
func (pm *ProxyManager) ResetStats() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.stats = &ProxyStats{
		LastReset: time.Now(),
	}
}

// GetServiceMetrics returns metrics for a specific service
func (pm *ProxyManager) GetServiceMetrics(serviceName string) (*ServiceMetrics, error) {
	pm.mu.RLock()
	proxy, exists := pm.services[serviceName]
	pm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("service %s not found", serviceName)
	}

	// This would typically come from the monitoring system
	// For now, return mock data
	return &ServiceMetrics{
		ServiceName:        serviceName,
		TotalRequests:      1000,
		SuccessfulRequests: 950,
		FailedRequests:     50,
		AverageLatency:     150.5,
		MaxLatency:         500.0,
		MinLatency:         50.0,
		SuccessRate:        95.0,
		LastRequest:        time.Now(),
		LastError:          time.Now().Add(-time.Hour),
	}, nil
}

// LoadBalancerProxy implements load balancing functionality
type LoadBalancerProxy struct {
	services    []Service
	algorithm   string
	healthCheck bool
	interval    time.Duration
	logger      Logger
	metrics     Metrics
	mu          sync.RWMutex
}

// NewLoadBalancerProxy creates a new load balancer proxy
func NewLoadBalancerProxy(
	services []Service,
	algorithm string,
	healthCheck bool,
	interval time.Duration,
	logger Logger,
	metrics Metrics,
) *LoadBalancerProxy {
	lb := &LoadBalancerProxy{
		services:    services,
		algorithm:   algorithm,
		healthCheck: healthCheck,
		interval:    interval,
		logger:      logger,
		metrics:     metrics,
	}

	if healthCheck {
		go lb.startHealthCheck()
	}

	return lb
}

// SelectService selects a service using the configured algorithm
func (lb *LoadBalancerProxy) SelectService(services []Service) Service {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if len(services) == 0 {
		return nil
	}

	switch lb.algorithm {
	case "round_robin":
		return lb.roundRobin(services)
	case "random":
		return lb.random(services)
	case "least_connections":
		return lb.leastConnections(services)
	default:
		return lb.roundRobin(services)
	}
}

// roundRobin selects the next service in round-robin fashion
func (lb *LoadBalancerProxy) roundRobin(services []Service) Service {
	// Simple round-robin implementation
	// In a real implementation, you'd maintain state
	return services[0]
}

// random selects a random service
func (lb *LoadBalancerProxy) random(services []Service) Service {
	// Simple random selection
	// In a real implementation, you'd use proper random selection
	return services[0]
}

// leastConnections selects the service with the least connections
func (lb *LoadBalancerProxy) leastConnections(services []Service) Service {
	// Simple least connections implementation
	// In a real implementation, you'd track connection counts
	return services[0]
}

// startHealthCheck starts the health check routine
func (lb *LoadBalancerProxy) startHealthCheck() {
	ticker := time.NewTicker(lb.interval)
	defer ticker.Stop()

	for range ticker.C {
		lb.performHealthCheck()
	}
}

// performHealthCheck performs health checks on all services
func (lb *LoadBalancerProxy) performHealthCheck() {
	ctx := context.Background()

	lb.mu.RLock()
	services := make([]Service, len(lb.services))
	copy(services, lb.services)
	lb.mu.RUnlock()

	for _, service := range services {
		healthy := service.IsHealthy(ctx)
		lb.metrics.RecordGauge("service_health", float64(boolToInt(healthy)), map[string]string{"service": service.GetName()})

		if !healthy {
			lb.logger.Warn("Service unhealthy", "service", service.GetName())
		}
	}
}

// boolToInt converts a boolean to an integer
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
