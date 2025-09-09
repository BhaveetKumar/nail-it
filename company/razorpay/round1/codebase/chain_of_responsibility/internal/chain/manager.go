package chain

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ChainManagerImpl implements the ChainManager interface
type ChainManagerImpl struct {
	handlers map[string]Handler
	chain    []Handler
	cache    Cache
	logger   Logger
	metrics  Metrics
	config   ChainConfig
	mu       sync.RWMutex
	stats    ChainStatistics
}

// NewChainManager creates a new chain manager
func NewChainManager(cache Cache, logger Logger, metrics Metrics, config ChainConfig) *ChainManagerImpl {
	manager := &ChainManagerImpl{
		handlers: make(map[string]Handler),
		chain:    make([]Handler, 0),
		cache:    cache,
		logger:   logger,
		metrics:  metrics,
		config:   config,
		stats:    ChainStatistics{},
	}

	// Initialize handlers from config
	manager.initializeHandlers()

	return manager
}

// AddHandler adds a handler to the chain
func (cm *ChainManagerImpl) AddHandler(handler Handler) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if handler == nil {
		return fmt.Errorf("handler cannot be nil")
	}

	if len(cm.handlers) >= cm.config.MaxHandlers {
		return fmt.Errorf("maximum number of handlers reached: %d", cm.config.MaxHandlers)
	}

	// Check if handler already exists
	if _, exists := cm.handlers[handler.GetName()]; exists {
		return fmt.Errorf("handler with name '%s' already exists", handler.GetName())
	}

	// Add handler
	cm.handlers[handler.GetName()] = handler
	cm.chain = append(cm.chain, handler)

	// Sort chain by priority
	cm.sortChain()

	// Update statistics
	cm.stats.TotalHandlers++
	if handler.IsEnabled() {
		cm.stats.EnabledHandlers++
	} else {
		cm.stats.DisabledHandlers++
	}

	cm.logger.Info("Handler added to chain", "name", handler.GetName(), "priority", handler.GetPriority())
	cm.metrics.IncrementCounter("handler_added", map[string]string{"name": handler.GetName()})

	return nil
}

// RemoveHandler removes a handler from the chain
func (cm *ChainManagerImpl) RemoveHandler(name string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	handler, exists := cm.handlers[name]
	if !exists {
		return fmt.Errorf("handler with name '%s' not found", name)
	}

	// Remove from handlers map
	delete(cm.handlers, name)

	// Remove from chain
	for i, h := range cm.chain {
		if h.GetName() == name {
			cm.chain = append(cm.chain[:i], cm.chain[i+1:]...)
			break
		}
	}

	// Update statistics
	cm.stats.TotalHandlers--
	if handler.IsEnabled() {
		cm.stats.EnabledHandlers--
	} else {
		cm.stats.DisabledHandlers--
	}

	cm.logger.Info("Handler removed from chain", "name", name)
	cm.metrics.IncrementCounter("handler_removed", map[string]string{"name": name})

	return nil
}

// GetHandler returns a handler by name
func (cm *ChainManagerImpl) GetHandler(name string) (Handler, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	handler, exists := cm.handlers[name]
	if !exists {
		return nil, fmt.Errorf("handler with name '%s' not found", name)
	}

	return handler, nil
}

// GetAllHandlers returns all handlers
func (cm *ChainManagerImpl) GetAllHandlers() []Handler {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	handlers := make([]Handler, 0, len(cm.handlers))
	for _, handler := range cm.handlers {
		handlers = append(handlers, handler)
	}

	return handlers
}

// ProcessRequest processes a request through the chain
func (cm *ChainManagerImpl) ProcessRequest(ctx context.Context, request *Request) (*Response, error) {
	start := time.Now()

	if request == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Set request context
	request.Context = ctx

	// Check cache first
	cacheKey := fmt.Sprintf("request_%s", request.ID)
	if cached, found := cm.cache.Get(cacheKey); found {
		cm.metrics.IncrementCounter("request_cache_hit", map[string]string{"type": request.Type})
		return cached.(*Response), nil
	}

	// Find the first handler that can handle the request
	var firstHandler Handler
	cm.mu.RLock()
	for _, handler := range cm.chain {
		if handler.IsEnabled() && handler.CanHandle(request) {
			firstHandler = handler
			break
		}
	}
	cm.mu.RUnlock()

	if firstHandler == nil {
		response := &Response{
			ID:          generateID(),
			RequestID:   request.ID,
			Status:      "no_handler",
			ProcessedAt: time.Now(),
			Duration:    time.Since(start),
			Error:       "no handler found for request type",
		}

		cm.metrics.IncrementCounter("request_no_handler", map[string]string{"type": request.Type})
		return response, nil
	}

	// Process request through the chain
	response, err := cm.processThroughChain(ctx, request, firstHandler, start)

	// Cache successful responses
	if err == nil && response.Status == "processed" {
		cm.cache.Set(cacheKey, response, cm.config.Cache.TTL)
	}

	// Update chain statistics
	cm.updateChainStatistics(response, time.Since(start))

	cm.logger.Debug("Request processed through chain", 
		"request_id", request.ID, 
		"type", request.Type, 
		"duration", time.Since(start),
		"status", response.Status)

	return response, err
}

// processThroughChain processes the request through the chain
func (cm *ChainManagerImpl) processThroughChain(ctx context.Context, request *Request, firstHandler Handler, startTime time.Time) (*Response, error) {
	currentHandler := firstHandler
	var lastResponse *Response

	for currentHandler != nil {
		// Check timeout
		if time.Since(startTime) > cm.config.Timeout {
			return &Response{
				ID:          generateID(),
				RequestID:   request.ID,
				Status:      "timeout",
				HandlerName: currentHandler.GetName(),
				ProcessedAt: time.Now(),
				Duration:    time.Since(startTime),
				Error:       "request timeout",
			}, fmt.Errorf("request timeout")
		}

		// Process with current handler
		response, err := currentHandler.Handle(ctx, request)
		if err != nil {
			cm.logger.Error("Handler processing failed", 
				"handler", currentHandler.GetName(), 
				"request_id", request.ID, 
				"error", err)
			
			return &Response{
				ID:          generateID(),
				RequestID:   request.ID,
				Status:      "error",
				HandlerName: currentHandler.GetName(),
				ProcessedAt: time.Now(),
				Duration:    time.Since(startTime),
				Error:       err.Error(),
			}, err
		}

		lastResponse = response

		// If handler processed the request and didn't pass to next, stop
		if response.Status == "processed" && response.NextHandler == "" {
			break
		}

		// If handler passed to next, find the next handler
		if response.NextHandler != "" {
			nextHandler, err := cm.GetHandler(response.NextHandler)
			if err != nil {
				cm.logger.Error("Next handler not found", 
					"handler", response.NextHandler, 
					"request_id", request.ID, 
					"error", err)
				break
			}
			currentHandler = nextHandler
		} else {
			// Find next handler in chain
			currentHandler = cm.findNextHandler(currentHandler)
		}
	}

	return lastResponse, nil
}

// findNextHandler finds the next handler in the chain
func (cm *ChainManagerImpl) findNextHandler(currentHandler Handler) Handler {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	currentIndex := -1
	for i, handler := range cm.chain {
		if handler.GetName() == currentHandler.GetName() {
			currentIndex = i
			break
		}
	}

	if currentIndex == -1 || currentIndex >= len(cm.chain)-1 {
		return nil
	}

	// Find next enabled handler
	for i := currentIndex + 1; i < len(cm.chain); i++ {
		if cm.chain[i].IsEnabled() {
			return cm.chain[i]
		}
	}

	return nil
}

// GetChainStatistics returns chain statistics
func (cm *ChainManagerImpl) GetChainStatistics() ChainStatistics {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Update handler statistics
	cm.stats.HandlerStats = make(map[string]HandlerStatistics)
	for _, handler := range cm.handlers {
		cm.stats.HandlerStats[handler.GetName()] = handler.GetStatistics()
	}

	// Calculate success rate
	if cm.stats.TotalRequests > 0 {
		cm.stats.SuccessfulRequests = 0
		cm.stats.FailedRequests = 0
		for _, stats := range cm.stats.HandlerStats {
			cm.stats.SuccessfulRequests += stats.SuccessfulRequests
			cm.stats.FailedRequests += stats.FailedRequests
		}
	}

	return cm.stats
}

// OptimizeChain optimizes the chain structure
func (cm *ChainManagerImpl) OptimizeChain() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Sort chain by priority
	cm.sortChain()

	// Remove disabled handlers from chain
	enabledHandlers := make([]Handler, 0)
	for _, handler := range cm.chain {
		if handler.IsEnabled() {
			enabledHandlers = append(enabledHandlers, handler)
		}
	}
	cm.chain = enabledHandlers

	// Update statistics
	cm.stats.EnabledHandlers = len(enabledHandlers)
	cm.stats.DisabledHandlers = cm.stats.TotalHandlers - cm.stats.EnabledHandlers

	cm.logger.Info("Chain optimized", 
		"total_handlers", cm.stats.TotalHandlers,
		"enabled_handlers", cm.stats.EnabledHandlers,
		"disabled_handlers", cm.stats.DisabledHandlers)

	cm.metrics.IncrementCounter("chain_optimized", map[string]string{})

	return nil
}

// ValidateChain validates the chain structure
func (cm *ChainManagerImpl) ValidateChain() error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Check for duplicate handler names
	names := make(map[string]bool)
	for _, handler := range cm.handlers {
		if names[handler.GetName()] {
			return fmt.Errorf("duplicate handler name: %s", handler.GetName())
		}
		names[handler.GetName()] = true
	}

	// Check for circular dependencies
	visited := make(map[string]bool)
	recursionStack := make(map[string]bool)

	for _, handler := range cm.handlers {
		if !visited[handler.GetName()] {
			if cm.hasCycle(handler, visited, recursionStack) {
				return fmt.Errorf("circular dependency detected in chain")
			}
		}
	}

	cm.logger.Info("Chain validation passed", "total_handlers", len(cm.handlers))
	cm.metrics.IncrementCounter("chain_validated", map[string]string{})

	return nil
}

// hasCycle checks for cycles in the handler chain
func (cm *ChainManagerImpl) hasCycle(handler Handler, visited, recursionStack map[string]bool) bool {
	visited[handler.GetName()] = true
	recursionStack[handler.GetName()] = true

	// Check if handler has a next handler
	if nextHandler := cm.findNextHandler(handler); nextHandler != nil {
		if !visited[nextHandler.GetName()] {
			if cm.hasCycle(nextHandler, visited, recursionStack) {
				return true
			}
		} else if recursionStack[nextHandler.GetName()] {
			return true
		}
	}

	recursionStack[handler.GetName()] = false
	return false
}

// sortChain sorts the chain by priority
func (cm *ChainManagerImpl) sortChain() {
	sort.Slice(cm.chain, func(i, j int) bool {
		return cm.chain[i].GetPriority() < cm.chain[j].GetPriority()
	})
}

// updateChainStatistics updates chain statistics
func (cm *ChainManagerImpl) updateChainStatistics(response *Response, duration time.Duration) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.stats.TotalRequests++
	cm.stats.LastRequest = time.Now()

	if response.Status == "processed" {
		cm.stats.SuccessfulRequests++
	} else {
		cm.stats.FailedRequests++
		cm.stats.LastError = time.Now()
	}

	// Update average latency
	if cm.stats.TotalRequests == 1 {
		cm.stats.AverageLatency = float64(duration.Nanoseconds()) / 1000000 // Convert to milliseconds
	} else {
		cm.stats.AverageLatency = (cm.stats.AverageLatency*float64(cm.stats.TotalRequests-1) + float64(duration.Nanoseconds())/1000000) / float64(cm.stats.TotalRequests)
	}

	// Update min/max latency
	if cm.stats.TotalRequests == 1 {
		cm.stats.MaxLatency = float64(duration.Nanoseconds()) / 1000000
		cm.stats.MinLatency = float64(duration.Nanoseconds()) / 1000000
	} else {
		latencyMs := float64(duration.Nanoseconds()) / 1000000
		if latencyMs > cm.stats.MaxLatency {
			cm.stats.MaxLatency = latencyMs
		}
		if latencyMs < cm.stats.MinLatency {
			cm.stats.MinLatency = latencyMs
		}
	}
}

// initializeHandlers initializes handlers from configuration
func (cm *ChainManagerImpl) initializeHandlers() {
	for _, handlerConfig := range cm.config.Handlers {
		var handler Handler

		switch handlerConfig.Type {
		case "authentication":
			handler = NewAuthenticationHandler(
				handlerConfig.Name,
				handlerConfig.Priority,
				cm.config.Security.JWTSecret,
				cm.config.Security.TokenExpiry,
			)
		case "authorization":
			handler = NewAuthorizationHandler(
				handlerConfig.Name,
				handlerConfig.Priority,
			)
		case "validation":
			handler = NewValidationHandler(
				handlerConfig.Name,
				handlerConfig.Priority,
			)
		case "rate_limit":
			handler = NewRateLimitHandler(
				handlerConfig.Name,
				handlerConfig.Priority,
				cm.config.Security.RateLimitRequests,
				20, // burst size
			)
		case "logging":
			handler = NewLoggingHandler(
				handlerConfig.Name,
				handlerConfig.Priority,
				cm.config.Logging.Level,
				cm.config.Logging.Format,
			)
		default:
			cm.logger.Warn("Unknown handler type", "type", handlerConfig.Type, "name", handlerConfig.Name)
			continue
		}

		// Set enabled status
		handler.SetEnabled(handlerConfig.Enabled)

		// Add to chain
		cm.handlers[handler.GetName()] = handler
		cm.chain = append(cm.chain, handler)

		cm.logger.Info("Handler initialized", 
			"name", handler.GetName(), 
			"type", handlerConfig.Type, 
			"priority", handler.GetPriority(),
			"enabled", handler.IsEnabled())
	}

	// Sort chain by priority
	cm.sortChain()

	// Update statistics
	cm.stats.TotalHandlers = len(cm.handlers)
	cm.stats.EnabledHandlers = 0
	cm.stats.DisabledHandlers = 0
	for _, handler := range cm.handlers {
		if handler.IsEnabled() {
			cm.stats.EnabledHandlers++
		} else {
			cm.stats.DisabledHandlers++
		}
	}
}
