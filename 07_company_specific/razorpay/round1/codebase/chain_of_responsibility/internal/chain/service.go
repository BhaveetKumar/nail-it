package chain

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ChainService provides high-level operations using chain of responsibility pattern
type ChainService struct {
	chainManager ChainManager
	cache        Cache
	database     Database
	logger       Logger
	metrics      Metrics
	config       ChainConfig
	mu           sync.RWMutex
}

// NewChainService creates a new chain service
func NewChainService(
	chainManager ChainManager,
	cache Cache,
	database Database,
	logger Logger,
	metrics Metrics,
	config ChainConfig,
) *ChainService {
	return &ChainService{
		chainManager: chainManager,
		cache:        cache,
		database:     database,
		logger:       logger,
		metrics:      metrics,
		config:       config,
	}
}

// ProcessAuthenticationRequest processes an authentication request
func (cs *ChainService) ProcessAuthenticationRequest(ctx context.Context, request AuthenticationRequest) (*AuthenticationResponse, error) {
	start := time.Now()

	// Create chain request
	chainRequest := &Request{
		ID:        generateID(),
		Type:      "authentication",
		Priority:  1,
		Data: map[string]interface{}{
			"user_id": request.UserID,
			"token":   request.Token,
		},
		Metadata: map[string]interface{}{
			"source": "authentication_service",
		},
		UserID:    request.UserID,
		Timestamp: time.Now(),
		Context:   ctx,
	}

	// Process through chain
	response, err := cs.chainManager.ProcessRequest(ctx, chainRequest)
	if err != nil {
		cs.logger.Error("Authentication request processing failed", "user_id", request.UserID, "error", err)
		return nil, fmt.Errorf("authentication processing failed: %w", err)
	}

	// Convert to authentication response
	authResponse := &AuthenticationResponse{
		ID:          response.ID,
		RequestID:   response.RequestID,
		UserID:      request.UserID,
		Authenticated: response.Data["authenticated"].(bool),
		Roles:       response.Data["roles"].([]string),
		Status:      response.Status,
		ProcessedAt: response.ProcessedAt,
		Duration:    response.Duration,
		Error:       response.Error,
	}

	duration := time.Since(start)
	cs.metrics.RecordTiming("authentication_request_duration", duration, map[string]string{"status": response.Status})
	cs.metrics.IncrementCounter("authentication_requests", map[string]string{"status": response.Status})

	cs.logger.Info("Authentication request processed", 
		"user_id", request.UserID, 
		"authenticated", authResponse.Authenticated,
		"duration", duration)

	return authResponse, nil
}

// ProcessAuthorizationRequest processes an authorization request
func (cs *ChainService) ProcessAuthorizationRequest(ctx context.Context, request AuthorizationRequest) (*AuthorizationResponse, error) {
	start := time.Now()

	// Create chain request
	chainRequest := &Request{
		ID:        generateID(),
		Type:      "authorization",
		Priority:  2,
		Data: map[string]interface{}{
			"role":     request.Role,
			"action":   request.Action,
			"resource": request.Resource,
		},
		Metadata: map[string]interface{}{
			"source": "authorization_service",
		},
		UserID:    request.UserID,
		Timestamp: time.Now(),
		Context:   ctx,
	}

	// Process through chain
	response, err := cs.chainManager.ProcessRequest(ctx, chainRequest)
	if err != nil {
		cs.logger.Error("Authorization request processing failed", "user_id", request.UserID, "error", err)
		return nil, fmt.Errorf("authorization processing failed: %w", err)
	}

	// Convert to authorization response
	authzResponse := &AuthorizationResponse{
		ID:          response.ID,
		RequestID:   response.RequestID,
		UserID:      request.UserID,
		Authorized:  response.Data["authorized"].(bool),
		Role:        response.Data["role"].(string),
		Action:      response.Data["action"].(string),
		Resource:    response.Data["resource"].(string),
		Status:      response.Status,
		ProcessedAt: response.ProcessedAt,
		Duration:    response.Duration,
		Error:       response.Error,
	}

	duration := time.Since(start)
	cs.metrics.RecordTiming("authorization_request_duration", duration, map[string]string{"status": response.Status})
	cs.metrics.IncrementCounter("authorization_requests", map[string]string{"status": response.Status})

	cs.logger.Info("Authorization request processed", 
		"user_id", request.UserID, 
		"authorized", authzResponse.Authorized,
		"duration", duration)

	return authzResponse, nil
}

// ProcessValidationRequest processes a validation request
func (cs *ChainService) ProcessValidationRequest(ctx context.Context, request ValidationRequest) (*ValidationResponse, error) {
	start := time.Now()

	// Create chain request
	chainRequest := &Request{
		ID:        generateID(),
		Type:      "validation",
		Priority:  3,
		Data:      request.Data,
		Metadata: map[string]interface{}{
			"source": "validation_service",
		},
		UserID:    request.UserID,
		Timestamp: time.Now(),
		Context:   ctx,
	}

	// Process through chain
	response, err := cs.chainManager.ProcessRequest(ctx, chainRequest)
	if err != nil {
		cs.logger.Error("Validation request processing failed", "user_id", request.UserID, "error", err)
		return nil, fmt.Errorf("validation processing failed: %w", err)
	}

	// Convert to validation response
	validationResponse := &ValidationResponse{
		ID:          response.ID,
		RequestID:   response.RequestID,
		UserID:      request.UserID,
		Valid:       response.Data["valid"].(bool),
		Errors:      response.Data["errors"].(map[string]string),
		Status:      response.Status,
		ProcessedAt: response.ProcessedAt,
		Duration:    response.Duration,
		Error:       response.Error,
	}

	duration := time.Since(start)
	cs.metrics.RecordTiming("validation_request_duration", duration, map[string]string{"status": response.Status})
	cs.metrics.IncrementCounter("validation_requests", map[string]string{"status": response.Status})

	cs.logger.Info("Validation request processed", 
		"user_id", request.UserID, 
		"valid", validationResponse.Valid,
		"duration", duration)

	return validationResponse, nil
}

// ProcessRateLimitRequest processes a rate limit request
func (cs *ChainService) ProcessRateLimitRequest(ctx context.Context, request RateLimitRequest) (*RateLimitResponse, error) {
	start := time.Now()

	// Create chain request
	chainRequest := &Request{
		ID:        generateID(),
		Type:      "rate_limit",
		Priority:  4,
		Data: map[string]interface{}{
			"user_id": request.UserID,
			"action":  request.Action,
		},
		Metadata: map[string]interface{}{
			"source": "rate_limit_service",
		},
		UserID:    request.UserID,
		Timestamp: time.Now(),
		Context:   ctx,
	}

	// Process through chain
	response, err := cs.chainManager.ProcessRequest(ctx, chainRequest)
	if err != nil {
		cs.logger.Error("Rate limit request processing failed", "user_id", request.UserID, "error", err)
		return nil, fmt.Errorf("rate limit processing failed: %w", err)
	}

	// Convert to rate limit response
	rateLimitResponse := &RateLimitResponse{
		ID:          response.ID,
		RequestID:   response.RequestID,
		UserID:      request.UserID,
		Allowed:     response.Data["allowed"].(bool),
		RequestsPerMinute: response.Data["requests_per_minute"].(int),
		BurstSize:   response.Data["burst_size"].(int),
		Status:      response.Status,
		ProcessedAt: response.ProcessedAt,
		Duration:    response.Duration,
		Error:       response.Error,
	}

	duration := time.Since(start)
	cs.metrics.RecordTiming("rate_limit_request_duration", duration, map[string]string{"status": response.Status})
	cs.metrics.IncrementCounter("rate_limit_requests", map[string]string{"status": response.Status})

	cs.logger.Info("Rate limit request processed", 
		"user_id", request.UserID, 
		"allowed", rateLimitResponse.Allowed,
		"duration", duration)

	return rateLimitResponse, nil
}

// ProcessLoggingRequest processes a logging request
func (cs *ChainService) ProcessLoggingRequest(ctx context.Context, request LoggingRequest) (*LoggingResponse, error) {
	start := time.Now()

	// Create chain request
	chainRequest := &Request{
		ID:        generateID(),
		Type:      "logging",
		Priority:  5,
		Data: map[string]interface{}{
			"message": request.Message,
			"level":   request.Level,
		},
		Metadata: map[string]interface{}{
			"source": "logging_service",
		},
		UserID:    request.UserID,
		Timestamp: time.Now(),
		Context:   ctx,
	}

	// Process through chain
	response, err := cs.chainManager.ProcessRequest(ctx, chainRequest)
	if err != nil {
		cs.logger.Error("Logging request processing failed", "user_id", request.UserID, "error", err)
		return nil, fmt.Errorf("logging processing failed: %w", err)
	}

	// Convert to logging response
	loggingResponse := &LoggingResponse{
		ID:          response.ID,
		RequestID:   response.RequestID,
		UserID:      request.UserID,
		Logged:      response.Data["logged"].(bool),
		LogLevel:    response.Data["log_level"].(string),
		LogFormat:   response.Data["log_format"].(string),
		Status:      response.Status,
		ProcessedAt: response.ProcessedAt,
		Duration:    response.Duration,
		Error:       response.Error,
	}

	duration := time.Since(start)
	cs.metrics.RecordTiming("logging_request_duration", duration, map[string]string{"status": response.Status})
	cs.metrics.IncrementCounter("logging_requests", map[string]string{"status": response.Status})

	cs.logger.Info("Logging request processed", 
		"user_id", request.UserID, 
		"logged", loggingResponse.Logged,
		"duration", duration)

	return loggingResponse, nil
}

// GetChainStatistics returns chain statistics
func (cs *ChainService) GetChainStatistics() ChainStatistics {
	return cs.chainManager.GetChainStatistics()
}

// GetHandlerStatistics returns statistics for a specific handler
func (cs *ChainService) GetHandlerStatistics(handlerName string) (HandlerStatistics, error) {
	handler, err := cs.chainManager.GetHandler(handlerName)
	if err != nil {
		return HandlerStatistics{}, fmt.Errorf("handler not found: %w", err)
	}

	return handler.GetStatistics(), nil
}

// AddHandler adds a handler to the chain
func (cs *ChainService) AddHandler(handler Handler) error {
	return cs.chainManager.AddHandler(handler)
}

// RemoveHandler removes a handler from the chain
func (cs *ChainService) RemoveHandler(handlerName string) error {
	return cs.chainManager.RemoveHandler(handlerName)
}

// GetHandler returns a handler by name
func (cs *ChainService) GetHandler(handlerName string) (Handler, error) {
	return cs.chainManager.GetHandler(handlerName)
}

// GetAllHandlers returns all handlers
func (cs *ChainService) GetAllHandlers() []Handler {
	return cs.chainManager.GetAllHandlers()
}

// OptimizeChain optimizes the chain structure
func (cs *ChainService) OptimizeChain() error {
	return cs.chainManager.OptimizeChain()
}

// ValidateChain validates the chain structure
func (cs *ChainService) ValidateChain() error {
	return cs.chainManager.ValidateChain()
}

// Request/Response models

type AuthenticationRequest struct {
	UserID string `json:"user_id"`
	Token  string `json:"token"`
}

type AuthenticationResponse struct {
	ID            string    `json:"id"`
	RequestID     string    `json:"request_id"`
	UserID        string    `json:"user_id"`
	Authenticated bool      `json:"authenticated"`
	Roles         []string  `json:"roles"`
	Status        string    `json:"status"`
	ProcessedAt   time.Time `json:"processed_at"`
	Duration      time.Duration `json:"duration"`
	Error         string    `json:"error,omitempty"`
}

type AuthorizationRequest struct {
	UserID   string `json:"user_id"`
	Role     string `json:"role"`
	Action   string `json:"action"`
	Resource string `json:"resource"`
}

type AuthorizationResponse struct {
	ID          string    `json:"id"`
	RequestID   string    `json:"request_id"`
	UserID      string    `json:"user_id"`
	Authorized  bool      `json:"authorized"`
	Role        string    `json:"role"`
	Action      string    `json:"action"`
	Resource    string    `json:"resource"`
	Status      string    `json:"status"`
	ProcessedAt time.Time `json:"processed_at"`
	Duration    time.Duration `json:"duration"`
	Error       string    `json:"error,omitempty"`
}

type ValidationRequest struct {
	UserID string                 `json:"user_id"`
	Data   map[string]interface{} `json:"data"`
}

type ValidationResponse struct {
	ID          string                 `json:"id"`
	RequestID   string                 `json:"request_id"`
	UserID      string                 `json:"user_id"`
	Valid       bool                   `json:"valid"`
	Errors      map[string]string      `json:"errors"`
	Status      string                 `json:"status"`
	ProcessedAt time.Time              `json:"processed_at"`
	Duration    time.Duration          `json:"duration"`
	Error       string                 `json:"error,omitempty"`
}

type RateLimitRequest struct {
	UserID string `json:"user_id"`
	Action string `json:"action"`
}

type RateLimitResponse struct {
	ID                string    `json:"id"`
	RequestID         string    `json:"request_id"`
	UserID            string    `json:"user_id"`
	Allowed           bool      `json:"allowed"`
	RequestsPerMinute int       `json:"requests_per_minute"`
	BurstSize         int       `json:"burst_size"`
	Status            string    `json:"status"`
	ProcessedAt       time.Time `json:"processed_at"`
	Duration          time.Duration `json:"duration"`
	Error             string    `json:"error,omitempty"`
}

type LoggingRequest struct {
	UserID  string `json:"user_id"`
	Message string `json:"message"`
	Level   string `json:"level"`
}

type LoggingResponse struct {
	ID          string    `json:"id"`
	RequestID   string    `json:"request_id"`
	UserID      string    `json:"user_id"`
	Logged      bool      `json:"logged"`
	LogLevel    string    `json:"log_level"`
	LogFormat   string    `json:"log_format"`
	Status      string    `json:"status"`
	ProcessedAt time.Time `json:"processed_at"`
	Duration    time.Duration `json:"duration"`
	Error       string    `json:"error,omitempty"`
}
