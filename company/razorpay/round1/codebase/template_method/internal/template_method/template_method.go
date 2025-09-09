package template_method

import (
	"sync"
	"time"
)

// ConcreteTemplateMethodManager implements the TemplateMethodManager interface
type ConcreteTemplateMethodManager struct {
	templates map[string]TemplateMethod
	mutex     sync.RWMutex
	config    *TemplateMethodConfig
}

// NewConcreteTemplateMethodManager creates a new concrete template method manager
func NewConcreteTemplateMethodManager(config *TemplateMethodConfig) *ConcreteTemplateMethodManager {
	return &ConcreteTemplateMethodManager{
		templates: make(map[string]TemplateMethod),
		config:    config,
	}
}

// CreateTemplateMethod creates a new template method
func (ctmm *ConcreteTemplateMethodManager) CreateTemplateMethod(name string, template TemplateMethod) error {
	ctmm.mutex.Lock()
	defer ctmm.mutex.Unlock()

	if len(ctmm.templates) >= ctmm.config.GetMaxTemplateMethods() {
		return ErrMaxTemplateMethodsReached
	}

	ctmm.templates[name] = template
	return nil
}

// GetTemplateMethod retrieves a template method by name
func (ctmm *ConcreteTemplateMethodManager) GetTemplateMethod(name string) (TemplateMethod, error) {
	ctmm.mutex.RLock()
	defer ctmm.mutex.RUnlock()

	template, exists := ctmm.templates[name]
	if !exists {
		return nil, ErrTemplateMethodNotFound
	}

	return template, nil
}

// RemoveTemplateMethod removes a template method
func (ctmm *ConcreteTemplateMethodManager) RemoveTemplateMethod(name string) error {
	ctmm.mutex.Lock()
	defer ctmm.mutex.Unlock()

	_, exists := ctmm.templates[name]
	if !exists {
		return ErrTemplateMethodNotFound
	}

	delete(ctmm.templates, name)
	return nil
}

// ListTemplateMethods returns all template method names
func (ctmm *ConcreteTemplateMethodManager) ListTemplateMethods() []string {
	ctmm.mutex.RLock()
	defer ctmm.mutex.RUnlock()

	names := make([]string, 0, len(ctmm.templates))
	for name := range ctmm.templates {
		names = append(names, name)
	}

	return names
}

// GetTemplateMethodCount returns the number of template methods
func (ctmm *ConcreteTemplateMethodManager) GetTemplateMethodCount() int {
	ctmm.mutex.RLock()
	defer ctmm.mutex.RUnlock()

	return len(ctmm.templates)
}

// GetTemplateMethodStats returns template method statistics
func (ctmm *ConcreteTemplateMethodManager) GetTemplateMethodStats() map[string]interface{} {
	ctmm.mutex.RLock()
	defer ctmm.mutex.RUnlock()

	stats := map[string]interface{}{
		"total_templates": len(ctmm.templates),
		"templates":       make(map[string]interface{}),
	}

	for name, template := range ctmm.templates {
		stats["templates"].(map[string]interface{})[name] = map[string]interface{}{
			"name":         template.GetName(),
			"description":  template.GetDescription(),
			"status":       template.GetStatus(),
			"steps":        len(template.GetSteps()),
			"current_step": template.GetCurrentStep(),
			"start_time":   template.GetStartTime(),
			"end_time":     template.GetEndTime(),
			"duration":     template.GetDuration(),
			"completed":    template.IsCompleted(),
			"failed":       template.IsFailed(),
			"running":      template.IsRunning(),
		}
	}

	return stats
}

// Cleanup performs cleanup operations
func (ctmm *ConcreteTemplateMethodManager) Cleanup() error {
	ctmm.mutex.Lock()
	defer ctmm.mutex.Unlock()

	// Remove completed or failed templates older than max age
	now := time.Now()
	for name, template := range ctmm.templates {
		if (template.IsCompleted() || template.IsFailed()) &&
			now.Sub(template.GetEndTime()) > ctmm.config.GetMaxExecutionTime() {
			delete(ctmm.templates, name)
		}
	}

	return nil
}

// ConcreteTemplateMethodExecutor implements the TemplateMethodExecutor interface
type ConcreteTemplateMethodExecutor struct {
	manager          *ConcreteTemplateMethodManager
	config           *TemplateMethodConfig
	executionHistory []ExecutionRecord
	mutex            sync.RWMutex
}

// NewConcreteTemplateMethodExecutor creates a new concrete template method executor
func NewConcreteTemplateMethodExecutor(manager *ConcreteTemplateMethodManager, config *TemplateMethodConfig) *ConcreteTemplateMethodExecutor {
	return &ConcreteTemplateMethodExecutor{
		manager:          manager,
		config:           config,
		executionHistory: make([]ExecutionRecord, 0),
	}
}

// ExecuteTemplateMethod executes a template method
func (ctme *ConcreteTemplateMethodExecutor) ExecuteTemplateMethod(template TemplateMethod) error {
	// Create execution record
	record := &BaseExecutionRecord{
		ID:                 generateID(),
		TemplateMethodName: template.GetName(),
		StartTime:          time.Now(),
		EndTime:            time.Time{},
		Status:             "running",
		Error:              nil,
		Data:               template.GetData(),
		Result:             nil,
		Metadata:           make(map[string]interface{}),
	}

	ctme.mutex.Lock()
	ctme.executionHistory = append(ctme.executionHistory, record)
	ctme.mutex.Unlock()

	// Execute template method
	err := template.Execute()

	// Update execution record
	record.SetEndTime(time.Now())
	if err != nil {
		record.SetStatus("failed")
		record.SetError(err)
	} else {
		record.SetStatus("completed")
		record.SetResult(template.GetResult())
	}

	return err
}

// ExecuteStep executes a single step
func (ctme *ConcreteTemplateMethodExecutor) ExecuteStep(step Step) error {
	// Validate step
	if err := step.Validate(); err != nil {
		return err
	}

	// Execute step
	return step.Execute()
}

// ExecuteSteps executes multiple steps
func (ctme *ConcreteTemplateMethodExecutor) ExecuteSteps(steps []Step) error {
	for _, step := range steps {
		if err := ctme.ExecuteStep(step); err != nil {
			return err
		}
	}
	return nil
}

// ValidateTemplateMethod validates a template method
func (ctme *ConcreteTemplateMethodExecutor) ValidateTemplateMethod(template TemplateMethod) error {
	if template == nil {
		return ErrInvalidTemplateMethod
	}

	if template.GetName() == "" {
		return ErrEmptyTemplateMethodName
	}

	if len(template.GetSteps()) == 0 {
		return ErrNoSteps
	}

	// Validate each step
	for _, step := range template.GetSteps() {
		if err := ctme.ValidateStep(step); err != nil {
			return err
		}
	}

	return nil
}

// ValidateStep validates a single step
func (ctme *ConcreteTemplateMethodExecutor) ValidateStep(step Step) error {
	if step == nil {
		return ErrInvalidStep
	}

	if step.GetName() == "" {
		return ErrEmptyStepName
	}

	return step.Validate()
}

// GetExecutionStats returns execution statistics
func (ctme *ConcreteTemplateMethodExecutor) GetExecutionStats() map[string]interface{} {
	ctme.mutex.RLock()
	defer ctme.mutex.RUnlock()

	stats := map[string]interface{}{
		"total_executions":      len(ctme.executionHistory),
		"successful_executions": 0,
		"failed_executions":     0,
		"running_executions":    0,
		"average_duration":      0.0,
		"max_duration":          0.0,
		"min_duration":          0.0,
	}

	var totalDuration time.Duration
	var maxDuration time.Duration
	var minDuration time.Duration

	for _, record := range ctme.executionHistory {
		switch record.GetStatus() {
		case "completed":
			stats["successful_executions"] = stats["successful_executions"].(int) + 1
		case "failed":
			stats["failed_executions"] = stats["failed_executions"].(int) + 1
		case "running":
			stats["running_executions"] = stats["running_executions"].(int) + 1
		}

		duration := record.GetDuration()
		totalDuration += duration

		if duration > maxDuration {
			maxDuration = duration
		}

		if minDuration == 0 || duration < minDuration {
			minDuration = duration
		}
	}

	if len(ctme.executionHistory) > 0 {
		stats["average_duration"] = float64(totalDuration) / float64(len(ctme.executionHistory))
		stats["max_duration"] = float64(maxDuration)
		stats["min_duration"] = float64(minDuration)
	}

	return stats
}

// GetExecutionHistory returns execution history
func (ctme *ConcreteTemplateMethodExecutor) GetExecutionHistory() []ExecutionRecord {
	ctme.mutex.RLock()
	defer ctme.mutex.RUnlock()

	return ctme.executionHistory
}

// ClearExecutionHistory clears execution history
func (ctme *ConcreteTemplateMethodExecutor) ClearExecutionHistory() error {
	ctme.mutex.Lock()
	defer ctme.mutex.Unlock()

	ctme.executionHistory = make([]ExecutionRecord, 0)
	return nil
}

// BaseExecutionRecord provides common functionality for execution records
type BaseExecutionRecord struct {
	ID                 string                 `json:"id"`
	TemplateMethodName string                 `json:"template_method_name"`
	StartTime          time.Time              `json:"start_time"`
	EndTime            time.Time              `json:"end_time"`
	Status             string                 `json:"status"`
	Error              error                  `json:"error"`
	Data               interface{}            `json:"data"`
	Result             interface{}            `json:"result"`
	Metadata           map[string]interface{} `json:"metadata"`
}

// GetID returns the execution record ID
func (ber *BaseExecutionRecord) GetID() string {
	return ber.ID
}

// GetTemplateMethodName returns the template method name
func (ber *BaseExecutionRecord) GetTemplateMethodName() string {
	return ber.TemplateMethodName
}

// GetStartTime returns the start time
func (ber *BaseExecutionRecord) GetStartTime() time.Time {
	return ber.StartTime
}

// GetEndTime returns the end time
func (ber *BaseExecutionRecord) GetEndTime() time.Time {
	return ber.EndTime
}

// SetEndTime sets the end time
func (ber *BaseExecutionRecord) SetEndTime(endTime time.Time) {
	ber.EndTime = endTime
}

// GetDuration returns the duration
func (ber *BaseExecutionRecord) GetDuration() time.Duration {
	if ber.EndTime.IsZero() {
		return time.Since(ber.StartTime)
	}
	return ber.EndTime.Sub(ber.StartTime)
}

// GetStatus returns the status
func (ber *BaseExecutionRecord) GetStatus() string {
	return ber.Status
}

// SetStatus sets the status
func (ber *BaseExecutionRecord) SetStatus(status string) {
	ber.Status = status
}

// GetError returns the error
func (ber *BaseExecutionRecord) GetError() error {
	return ber.Error
}

// SetError sets the error
func (ber *BaseExecutionRecord) SetError(err error) {
	ber.Error = err
}

// GetData returns the data
func (ber *BaseExecutionRecord) GetData() interface{} {
	return ber.Data
}

// GetResult returns the result
func (ber *BaseExecutionRecord) GetResult() interface{} {
	return ber.Result
}

// SetResult sets the result
func (ber *BaseExecutionRecord) SetResult(result interface{}) {
	ber.Result = result
}

// GetMetadata returns the metadata
func (ber *BaseExecutionRecord) GetMetadata() map[string]interface{} {
	return ber.Metadata
}

// SetMetadata sets the metadata
func (ber *BaseExecutionRecord) SetMetadata(metadata map[string]interface{}) {
	ber.Metadata = metadata
}

// IsCompleted returns whether the execution is completed
func (ber *BaseExecutionRecord) IsCompleted() bool {
	return ber.Status == "completed"
}

// IsFailed returns whether the execution failed
func (ber *BaseExecutionRecord) IsFailed() bool {
	return ber.Status == "failed"
}

// IsRunning returns whether the execution is running
func (ber *BaseExecutionRecord) IsRunning() bool {
	return ber.Status == "running"
}

// TemplateMethodService provides template method operations
type TemplateMethodService struct {
	manager  *ConcreteTemplateMethodManager
	executor *ConcreteTemplateMethodExecutor
	config   *TemplateMethodConfig
}

// NewTemplateMethodService creates a new template method service
func NewTemplateMethodService(config *TemplateMethodConfig) *TemplateMethodService {
	manager := NewConcreteTemplateMethodManager(config)
	executor := NewConcreteTemplateMethodExecutor(manager, config)

	return &TemplateMethodService{
		manager:  manager,
		executor: executor,
		config:   config,
	}
}

// CreateTemplateMethod creates a template method
func (tms *TemplateMethodService) CreateTemplateMethod(name string, template TemplateMethod) error {
	return tms.manager.CreateTemplateMethod(name, template)
}

// GetTemplateMethod retrieves a template method
func (tms *TemplateMethodService) GetTemplateMethod(name string) (TemplateMethod, error) {
	return tms.manager.GetTemplateMethod(name)
}

// RemoveTemplateMethod removes a template method
func (tms *TemplateMethodService) RemoveTemplateMethod(name string) error {
	return tms.manager.RemoveTemplateMethod(name)
}

// ListTemplateMethods returns all template method names
func (tms *TemplateMethodService) ListTemplateMethods() []string {
	return tms.manager.ListTemplateMethods()
}

// GetTemplateMethodCount returns the number of template methods
func (tms *TemplateMethodService) GetTemplateMethodCount() int {
	return tms.manager.GetTemplateMethodCount()
}

// GetTemplateMethodStats returns template method statistics
func (tms *TemplateMethodService) GetTemplateMethodStats() map[string]interface{} {
	return tms.manager.GetTemplateMethodStats()
}

// ExecuteTemplateMethod executes a template method
func (tms *TemplateMethodService) ExecuteTemplateMethod(template TemplateMethod) error {
	return tms.executor.ExecuteTemplateMethod(template)
}

// ExecuteStep executes a single step
func (tms *TemplateMethodService) ExecuteStep(step Step) error {
	return tms.executor.ExecuteStep(step)
}

// ExecuteSteps executes multiple steps
func (tms *TemplateMethodService) ExecuteSteps(steps []Step) error {
	return tms.executor.ExecuteSteps(steps)
}

// ValidateTemplateMethod validates a template method
func (tms *TemplateMethodService) ValidateTemplateMethod(template TemplateMethod) error {
	return tms.executor.ValidateTemplateMethod(template)
}

// ValidateStep validates a single step
func (tms *TemplateMethodService) ValidateStep(step Step) error {
	return tms.executor.ValidateStep(step)
}

// GetExecutionStats returns execution statistics
func (tms *TemplateMethodService) GetExecutionStats() map[string]interface{} {
	return tms.executor.GetExecutionStats()
}

// GetExecutionHistory returns execution history
func (tms *TemplateMethodService) GetExecutionHistory() []ExecutionRecord {
	return tms.executor.GetExecutionHistory()
}

// ClearExecutionHistory clears execution history
func (tms *TemplateMethodService) ClearExecutionHistory() error {
	return tms.executor.ClearExecutionHistory()
}

// Cleanup performs cleanup operations
func (tms *TemplateMethodService) Cleanup() error {
	return tms.manager.Cleanup()
}

// TemplateMethodCache provides caching for template methods
type TemplateMethodCache struct {
	cache map[string]TemplateMethod
	mutex sync.RWMutex
	ttl   time.Duration
}

// NewTemplateMethodCache creates a new template method cache
func NewTemplateMethodCache(ttl time.Duration) *TemplateMethodCache {
	return &TemplateMethodCache{
		cache: make(map[string]TemplateMethod),
		ttl:   ttl,
	}
}

// Get gets a template method from the cache
func (tmc *TemplateMethodCache) Get(key string) (TemplateMethod, bool) {
	tmc.mutex.RLock()
	defer tmc.mutex.RUnlock()

	template, exists := tmc.cache[key]
	return template, exists
}

// Set sets a template method in the cache
func (tmc *TemplateMethodCache) Set(key string, template TemplateMethod, ttl time.Duration) error {
	tmc.mutex.Lock()
	defer tmc.mutex.Unlock()

	tmc.cache[key] = template
	return nil
}

// Delete deletes a template method from the cache
func (tmc *TemplateMethodCache) Delete(key string) error {
	tmc.mutex.Lock()
	defer tmc.mutex.Unlock()

	delete(tmc.cache, key)
	return nil
}

// Clear clears the cache
func (tmc *TemplateMethodCache) Clear() error {
	tmc.mutex.Lock()
	defer tmc.mutex.Unlock()

	tmc.cache = make(map[string]TemplateMethod)
	return nil
}

// Size returns the cache size
func (tmc *TemplateMethodCache) Size() int {
	tmc.mutex.RLock()
	defer tmc.mutex.RUnlock()

	return len(tmc.cache)
}

// Keys returns all cache keys
func (tmc *TemplateMethodCache) Keys() []string {
	tmc.mutex.RLock()
	defer tmc.mutex.RUnlock()

	keys := make([]string, 0, len(tmc.cache))
	for key := range tmc.cache {
		keys = append(keys, key)
	}

	return keys
}

// GetStats returns cache statistics
func (tmc *TemplateMethodCache) GetStats() map[string]interface{} {
	tmc.mutex.RLock()
	defer tmc.mutex.RUnlock()

	stats := map[string]interface{}{
		"size": len(tmc.cache),
		"ttl":  tmc.ttl,
	}

	return stats
}

// GetHitRate returns the cache hit rate
func (tmc *TemplateMethodCache) GetHitRate() float64 {
	// Mock implementation
	return 0.95
}

// GetMissRate returns the cache miss rate
func (tmc *TemplateMethodCache) GetMissRate() float64 {
	// Mock implementation
	return 0.05
}

// GetEvictionCount returns the eviction count
func (tmc *TemplateMethodCache) GetEvictionCount() int64 {
	// Mock implementation
	return 0
}

// GetExpirationCount returns the expiration count
func (tmc *TemplateMethodCache) GetExpirationCount() int64 {
	// Mock implementation
	return 0
}

// TemplateMethodValidator validates template methods
type TemplateMethodValidator struct {
	config *TemplateMethodConfig
}

// NewTemplateMethodValidator creates a new template method validator
func NewTemplateMethodValidator(config *TemplateMethodConfig) *TemplateMethodValidator {
	return &TemplateMethodValidator{
		config: config,
	}
}

// ValidateTemplateMethod validates a template method
func (tmv *TemplateMethodValidator) ValidateTemplateMethod(template TemplateMethod) error {
	if template == nil {
		return ErrInvalidTemplateMethod
	}

	if template.GetName() == "" {
		return ErrEmptyTemplateMethodName
	}

	if len(template.GetSteps()) == 0 {
		return ErrNoSteps
	}

	if len(template.GetSteps()) > tmv.config.GetMaxSteps() {
		return ErrTooManySteps
	}

	return nil
}

// ValidateStep validates a single step
func (tmv *TemplateMethodValidator) ValidateStep(step Step) error {
	if step == nil {
		return ErrInvalidStep
	}

	if step.GetName() == "" {
		return ErrEmptyStepName
	}

	return step.Validate()
}

// ValidateSteps validates multiple steps
func (tmv *TemplateMethodValidator) ValidateSteps(steps []Step) error {
	for _, step := range steps {
		if err := tmv.ValidateStep(step); err != nil {
			return err
		}
	}
	return nil
}

// GetValidationRules returns validation rules
func (tmv *TemplateMethodValidator) GetValidationRules() map[string]interface{} {
	return map[string]interface{}{
		"max_steps":          tmv.config.GetMaxSteps(),
		"max_execution_time": tmv.config.GetMaxExecutionTime(),
		"max_retries":        tmv.config.GetMaxRetries(),
	}
}

// SetValidationRules sets validation rules
func (tmv *TemplateMethodValidator) SetValidationRules(rules map[string]interface{}) {
	// Mock implementation
}

// GetValidationErrors returns validation errors
func (tmv *TemplateMethodValidator) GetValidationErrors() []string {
	// Mock implementation
	return []string{}
}

// ClearValidationErrors clears validation errors
func (tmv *TemplateMethodValidator) ClearValidationErrors() {
	// Mock implementation
}
