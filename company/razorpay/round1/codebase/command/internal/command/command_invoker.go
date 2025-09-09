package command

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CommandInvokerImpl implements CommandInvoker interface
type CommandInvokerImpl struct {
	handlers      map[string]CommandHandler
	auditor       CommandAuditor
	validator     CommandValidator
	metrics       CommandMetrics
	config        *CommandConfig
	history       []*CommandExecution
	historyMutex  sync.RWMutex
	circuitBreaker *CircuitBreaker
}

// NewCommandInvoker creates a new command invoker
func NewCommandInvoker(
	auditor CommandAuditor,
	validator CommandValidator,
	metrics CommandMetrics,
	config *CommandConfig,
) *CommandInvokerImpl {
	return &CommandInvokerImpl{
		handlers:      make(map[string]CommandHandler),
		auditor:       auditor,
		validator:     validator,
		metrics:       metrics,
		config:        config,
		history:       make([]*CommandExecution, 0),
		circuitBreaker: NewCircuitBreaker(
			config.CircuitBreaker.FailureThreshold,
			config.CircuitBreaker.RecoveryTimeout,
			config.CircuitBreaker.HalfOpenMaxCalls,
		),
	}
}

// RegisterHandler registers a command handler
func (ci *CommandInvokerImpl) RegisterHandler(handler CommandHandler) error {
	if handler == nil {
		return fmt.Errorf("handler cannot be nil")
	}
	
	handlerName := handler.GetHandlerName()
	if handlerName == "" {
		return fmt.Errorf("handler name cannot be empty")
	}
	
	ci.handlers[handlerName] = handler
	return nil
}

// Execute executes a command synchronously
func (ci *CommandInvokerImpl) Execute(ctx context.Context, command Command) (*CommandResult, error) {
	start := time.Now()
	
	// Validate command
	if err := command.Validate(); err != nil {
		return nil, fmt.Errorf("command validation failed: %w", err)
	}
	
	// Check circuit breaker
	if !ci.circuitBreaker.CanExecute() {
		return nil, fmt.Errorf("circuit breaker is open")
	}
	
	// Create execution record
	execution := &CommandExecution{
		CommandID:   command.GetCommandID(),
		CommandType: command.GetCommandType(),
		Description: command.GetDescription(),
		Status:      string(CommandStatusExecuting),
		StartTime:   start,
		Metadata:    make(map[string]interface{}),
	}
	
	// Add to history
	ci.addToHistory(execution)
	
	// Execute command with retry
	var result *CommandResult
	var err error
	
	for attempt := 0; attempt <= ci.config.MaxRetries; attempt++ {
		// Find handler
		handler, err := ci.findHandler(command.GetCommandType())
		if err != nil {
			execution.Status = string(CommandStatusFailed)
			execution.Error = err.Error()
			execution.EndTime = time.Now()
			execution.Duration = execution.EndTime.Sub(execution.StartTime)
			ci.updateHistory(execution)
			return nil, err
		}
		
		// Execute command
		result, err = handler.Handle(ctx, command)
		if err == nil {
			// Success
			ci.circuitBreaker.RecordSuccess()
			execution.Status = string(CommandStatusCompleted)
			execution.Result = result
			execution.EndTime = time.Now()
			execution.Duration = execution.EndTime.Sub(execution.StartTime)
			execution.RetryCount = attempt
			ci.updateHistory(execution)
			
			// Record metrics
			if ci.metrics != nil {
				ci.metrics.RecordCommandExecution(
					command.GetCommandType(),
					execution.Duration,
					true,
				)
			}
			
			// Audit
			if ci.auditor != nil {
				ci.auditor.Audit(ctx, command, result)
			}
			
			return result, nil
		}
		
		// Failure
		if attempt < ci.config.MaxRetries {
			// Wait before retry
			time.Sleep(ci.config.RetryDelay)
			execution.RetryCount = attempt + 1
			execution.Status = string(CommandStatusRetrying)
		}
	}
	
	// All retries failed
	ci.circuitBreaker.RecordFailure()
	execution.Status = string(CommandStatusFailed)
	execution.Error = err.Error()
	execution.EndTime = time.Now()
	execution.Duration = execution.EndTime.Sub(execution.StartTime)
	ci.updateHistory(execution)
	
	// Record metrics
	if ci.metrics != nil {
		ci.metrics.RecordCommandExecution(
			command.GetCommandType(),
			execution.Duration,
			false,
		)
	}
	
	// Audit
	if ci.auditor != nil {
		ci.auditor.Audit(ctx, command, &CommandResult{
			CommandID: command.GetCommandID(),
			Success:   false,
			Error:     err.Error(),
			ExecutedAt: time.Now(),
			Duration:  execution.Duration,
		})
	}
	
	return nil, fmt.Errorf("command execution failed after %d retries: %w", ci.config.MaxRetries, err)
}

// ExecuteAsync executes a command asynchronously
func (ci *CommandInvokerImpl) ExecuteAsync(ctx context.Context, command Command) (<-chan *CommandResult, error) {
	resultChan := make(chan *CommandResult, 1)
	
	go func() {
		defer close(resultChan)
		
		result, err := ci.Execute(ctx, command)
		if err != nil {
			resultChan <- &CommandResult{
				CommandID: command.GetCommandID(),
				Success:   false,
				Error:     err.Error(),
				ExecutedAt: time.Now(),
			}
		} else {
			resultChan <- result
		}
	}()
	
	return resultChan, nil
}

// ExecuteBatch executes multiple commands in batch
func (ci *CommandInvokerImpl) ExecuteBatch(ctx context.Context, commands []Command) ([]*CommandResult, error) {
	if len(commands) == 0 {
		return []*CommandResult{}, nil
	}
	
	results := make([]*CommandResult, len(commands))
	
	// Execute commands concurrently
	var wg sync.WaitGroup
	for i, command := range commands {
		wg.Add(1)
		go func(index int, cmd Command) {
			defer wg.Done()
			
			result, err := ci.Execute(ctx, cmd)
			if err != nil {
				results[index] = &CommandResult{
					CommandID: cmd.GetCommandID(),
					Success:   false,
					Error:     err.Error(),
					ExecutedAt: time.Now(),
				}
			} else {
				results[index] = result
			}
		}(i, command)
	}
	
	wg.Wait()
	return results, nil
}

// GetExecutionHistory returns command execution history
func (ci *CommandInvokerImpl) GetExecutionHistory() []*CommandExecution {
	ci.historyMutex.RLock()
	defer ci.historyMutex.RUnlock()
	
	// Return a copy to avoid race conditions
	history := make([]*CommandExecution, len(ci.history))
	copy(history, ci.history)
	
	return history
}

// ClearHistory clears command execution history
func (ci *CommandInvokerImpl) ClearHistory() {
	ci.historyMutex.Lock()
	defer ci.historyMutex.Unlock()
	
	ci.history = make([]*CommandExecution, 0)
}

// findHandler finds a handler for the given command type
func (ci *CommandInvokerImpl) findHandler(commandType string) (CommandHandler, error) {
	for _, handler := range ci.handlers {
		if handler.CanHandle(commandType) {
			return handler, nil
		}
	}
	
	return nil, fmt.Errorf("no handler found for command type: %s", commandType)
}

// addToHistory adds execution to history
func (ci *CommandInvokerImpl) addToHistory(execution *CommandExecution) {
	ci.historyMutex.Lock()
	defer ci.historyMutex.Unlock()
	
	ci.history = append(ci.history, execution)
	
	// Limit history size
	if len(ci.history) > ci.config.MaxHistorySize {
		ci.history = ci.history[1:]
	}
}

// updateHistory updates execution in history
func (ci *CommandInvokerImpl) updateHistory(execution *CommandExecution) {
	ci.historyMutex.Lock()
	defer ci.historyMutex.Unlock()
	
	// Find and update execution
	for i, hist := range ci.history {
		if hist.CommandID == execution.CommandID {
			ci.history[i] = execution
			break
		}
	}
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	failureThreshold int
	recoveryTimeout  time.Duration
	halfOpenMaxCalls int
	
	state         string
	failureCount  int
	successCount  int
	lastFailure   time.Time
	nextRetry     time.Time
	mu            sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(failureThreshold int, recoveryTimeout time.Duration, halfOpenMaxCalls int) *CircuitBreaker {
	return &CircuitBreaker{
		failureThreshold: failureThreshold,
		recoveryTimeout:  recoveryTimeout,
		halfOpenMaxCalls: halfOpenMaxCalls,
		state:           "closed",
	}
}

// CanExecute checks if the circuit breaker allows execution
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	switch cb.state {
	case "closed":
		return true
	case "open":
		return time.Now().After(cb.nextRetry)
	case "half-open":
		return cb.successCount < cb.halfOpenMaxCalls
	default:
		return false
	}
}

// RecordSuccess records a successful execution
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	cb.successCount++
	
	if cb.state == "half-open" {
		if cb.successCount >= cb.halfOpenMaxCalls {
			cb.state = "closed"
			cb.failureCount = 0
			cb.successCount = 0
		}
	}
}

// RecordFailure records a failed execution
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	cb.failureCount++
	cb.lastFailure = time.Now()
	
	if cb.state == "closed" {
		if cb.failureCount >= cb.failureThreshold {
			cb.state = "open"
			cb.nextRetry = time.Now().Add(cb.recoveryTimeout)
		}
	} else if cb.state == "half-open" {
		cb.state = "open"
		cb.nextRetry = time.Now().Add(cb.recoveryTimeout)
	}
}

// GetState returns the current state
func (cb *CircuitBreaker) GetState() string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// GetFailureCount returns the failure count
func (cb *CircuitBreaker) GetFailureCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.failureCount
}

// GetSuccessCount returns the success count
func (cb *CircuitBreaker) GetSuccessCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.successCount
}

// GetLastFailureTime returns the last failure time
func (cb *CircuitBreaker) GetLastFailureTime() time.Time {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.lastFailure
}

// GetNextRetryTime returns the next retry time
func (cb *CircuitBreaker) GetNextRetryTime() time.Time {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.nextRetry
}

// Reset resets the circuit breaker
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	cb.state = "closed"
	cb.failureCount = 0
	cb.successCount = 0
	cb.lastFailure = time.Time{}
	cb.nextRetry = time.Time{}
}
