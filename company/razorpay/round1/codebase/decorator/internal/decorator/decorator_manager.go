package decorator

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DecoratorManager manages decorators and components
type DecoratorManager struct {
	components map[string]Component
	decorators map[string]Decorator
	config     DecoratorConfig
	logger     Logger
	metrics    Metrics
	mu         sync.RWMutex
}

// NewDecoratorManager creates a new decorator manager
func NewDecoratorManager(config DecoratorConfig, logger Logger, metrics Metrics) *DecoratorManager {
	return &DecoratorManager{
		components: make(map[string]Component),
		decorators: make(map[string]Decorator),
		config:     config,
		logger:     logger,
		metrics:    metrics,
	}
}

// RegisterComponent registers a component
func (dm *DecoratorManager) RegisterComponent(component Component) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	dm.components[component.GetName()] = component
	dm.logger.Info("Component registered", "component", component.GetName())
}

// RegisterDecorator registers a decorator
func (dm *DecoratorManager) RegisterDecorator(decorator Decorator) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	dm.decorators[decorator.GetName()] = decorator
	dm.logger.Info("Decorator registered", "decorator", decorator.GetName())
}

// CreateDecoratedComponent creates a component with decorators applied
func (dm *DecoratorManager) CreateDecoratedComponent(componentName string, decoratorNames []string) (Component, error) {
	dm.mu.RLock()
	component, exists := dm.components[componentName]
	dm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("component %s not found", componentName)
	}
	
	// Start with the base component
	current := component
	
	// Apply decorators in order
	for _, decoratorName := range decoratorNames {
		dm.mu.RLock()
		decorator, exists := dm.decorators[decoratorName]
		dm.mu.RUnlock()
		
		if !exists {
			return nil, fmt.Errorf("decorator %s not found", decoratorName)
		}
		
		// Create a new instance of the decorator
		newDecorator := dm.cloneDecorator(decorator)
		newDecorator.SetComponent(current)
		current = newDecorator
	}
	
	return current, nil
}

// cloneDecorator creates a new instance of a decorator
func (dm *DecoratorManager) cloneDecorator(decorator Decorator) Decorator {
	// This is a simplified implementation
	// In a real implementation, you'd use reflection or a factory pattern
	switch decorator.GetName() {
	case "logging":
		return NewLoggingDecorator(dm.logger, dm.config.Logging)
	case "metrics":
		return NewMetricsDecorator(dm.metrics, dm.config.Metrics)
	case "cache":
		// This would need a cache instance
		return NewCacheDecorator(nil, dm.config.Cache)
	case "security":
		// This would need a security instance
		return NewSecurityDecorator(nil, dm.config.Security)
	case "rate_limit":
		// This would need a rate limiter instance
		return NewRateLimitDecorator(nil, dm.config.RateLimit)
	case "circuit_breaker":
		// This would need a circuit breaker instance
		return NewCircuitBreakerDecorator(nil, dm.config.CircuitBreaker)
	case "retry":
		// This would need a retry instance
		return NewRetryDecorator(nil, dm.config.Retry)
	case "monitoring":
		// This would need a monitoring instance
		return NewMonitoringDecorator(nil, dm.config.Monitoring)
	case "validation":
		// This would need a validation instance
		return NewValidationDecorator(nil, dm.config.Validation)
	case "encryption":
		// This would need an encryption instance
		return NewEncryptionDecorator(nil, dm.config.Encryption)
	case "compression":
		// This would need a compression instance
		return NewCompressionDecorator(nil, dm.config.Compression)
	default:
		return decorator
	}
}

// ExecuteComponent executes a component with decorators
func (dm *DecoratorManager) ExecuteComponent(ctx context.Context, componentName string, decoratorNames []string, request interface{}) (interface{}, error) {
	component, err := dm.CreateDecoratedComponent(componentName, decoratorNames)
	if err != nil {
		return nil, err
	}
	
	return component.Execute(ctx, request)
}

// GetComponentHealth returns the health status of a component
func (dm *DecoratorManager) GetComponentHealth(ctx context.Context, componentName string) (*HealthCheck, error) {
	dm.mu.RLock()
	component, exists := dm.components[componentName]
	dm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("component %s not found", componentName)
	}
	
	start := time.Now()
	
	// Try to execute a simple request
	_, err := component.Execute(ctx, map[string]interface{}{"health_check": true})
	
	latency := time.Since(start)
	healthy := err == nil
	
	status := "healthy"
	if !healthy {
		status = "unhealthy"
	}
	
	return &HealthCheck{
		Component: componentName,
		Healthy:   healthy,
		Status:    status,
		Message:   fmt.Sprintf("Component %s is %s", componentName, status),
		Timestamp: time.Now(),
		Latency:   latency,
	}, nil
}

// GetAllComponentsHealth returns the health status of all components
func (dm *DecoratorManager) GetAllComponentsHealth(ctx context.Context) ([]*HealthCheck, error) {
	dm.mu.RLock()
	components := make([]Component, 0, len(dm.components))
	for _, component := range dm.components {
		components = append(components, component)
	}
	dm.mu.RUnlock()
	
	healthChecks := make([]*HealthCheck, 0, len(components))
	
	for _, component := range components {
		health, err := dm.GetComponentHealth(ctx, component.GetName())
		if err != nil {
			dm.logger.Error("Failed to get component health", "component", component.GetName(), "error", err)
			continue
		}
		healthChecks = append(healthChecks, health)
	}
	
	return healthChecks, nil
}

// GetComponentMetrics returns metrics for a specific component
func (dm *DecoratorManager) GetComponentMetrics(componentName string) (*ComponentMetrics, error) {
	dm.mu.RLock()
	_, exists := dm.components[componentName]
	dm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("component %s not found", componentName)
	}
	
	// This would typically come from the metrics system
	// For now, return mock data
	return &ComponentMetrics{
		ComponentName:        componentName,
		TotalRequests:        1000,
		SuccessfulRequests:   950,
		FailedRequests:       50,
		AverageLatency:       150.5,
		MaxLatency:           500.0,
		MinLatency:           50.0,
		SuccessRate:          95.0,
		LastRequest:          time.Now(),
		LastError:            time.Now().Add(-time.Hour),
		CacheHits:            200,
		CacheMisses:          800,
		RateLimitHits:        10,
		CircuitBreakerTrips:  5,
	}, nil
}

// GetDecoratorChain returns the decorator chain for a component
func (dm *DecoratorManager) GetDecoratorChain(componentName string, decoratorNames []string) ([]string, error) {
	dm.mu.RLock()
	_, exists := dm.components[componentName]
	dm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("component %s not found", componentName)
	}
	
	// Validate decorators exist
	for _, decoratorName := range decoratorNames {
		dm.mu.RLock()
		_, exists := dm.decorators[decoratorName]
		dm.mu.RUnlock()
		
		if !exists {
			return nil, fmt.Errorf("decorator %s not found", decoratorName)
		}
	}
	
	// Return the chain
	chain := make([]string, 0, len(decoratorNames)+1)
	chain = append(chain, componentName)
	chain = append(chain, decoratorNames...)
	
	return chain, nil
}

// ListComponents returns a list of all registered components
func (dm *DecoratorManager) ListComponents() []string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	components := make([]string, 0, len(dm.components))
	for name := range dm.components {
		components = append(components, name)
	}
	
	return components
}

// ListDecorators returns a list of all registered decorators
func (dm *DecoratorManager) ListDecorators() []string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	decorators := make([]string, 0, len(dm.decorators))
	for name := range dm.decorators {
		decorators = append(decorators, name)
	}
	
	return decorators
}

// RemoveComponent removes a component
func (dm *DecoratorManager) RemoveComponent(componentName string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	if _, exists := dm.components[componentName]; !exists {
		return fmt.Errorf("component %s not found", componentName)
	}
	
	delete(dm.components, componentName)
	dm.logger.Info("Component removed", "component", componentName)
	
	return nil
}

// RemoveDecorator removes a decorator
func (dm *DecoratorManager) RemoveDecorator(decoratorName string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	if _, exists := dm.decorators[decoratorName]; !exists {
		return fmt.Errorf("decorator %s not found", decoratorName)
	}
	
	delete(dm.decorators, decoratorName)
	dm.logger.Info("Decorator removed", "decorator", decoratorName)
	
	return nil
}
