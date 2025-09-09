package template_method

import (
	"context"
	"sync"
	"time"
)

// TemplateMethodServiceProvider provides template method services
type TemplateMethodServiceProvider struct {
	service   *TemplateMethodService
	manager   *ConcreteTemplateMethodManager
	executor  *ConcreteTemplateMethodExecutor
	cache     *TemplateMethodCache
	validator *TemplateMethodValidator
	monitor   *TemplateMethodMonitor
	auditor   *TemplateMethodAuditor
	mutex     sync.RWMutex
}

// NewTemplateMethodServiceProvider creates a new template method service provider
func NewTemplateMethodServiceProvider(config *TemplateMethodConfig) *TemplateMethodServiceProvider {
	return &TemplateMethodServiceProvider{
		service:   NewTemplateMethodService(config),
		manager:   NewConcreteTemplateMethodManager(config),
		executor:  NewConcreteTemplateMethodExecutor(NewConcreteTemplateMethodManager(config), config),
		cache:     NewTemplateMethodCache(config.GetCache().GetTTL()),
		validator: NewTemplateMethodValidator(config),
		monitor:   NewTemplateMethodMonitor(config),
		auditor:   NewTemplateMethodAuditor(config),
	}
}

// GetService returns the template method service
func (tmsp *TemplateMethodServiceProvider) GetService() *TemplateMethodService {
	return tmsp.service
}

// GetManager returns the template method manager
func (tmsp *TemplateMethodServiceProvider) GetManager() *ConcreteTemplateMethodManager {
	return tmsp.manager
}

// GetExecutor returns the template method executor
func (tmsp *TemplateMethodServiceProvider) GetExecutor() *ConcreteTemplateMethodExecutor {
	return tmsp.executor
}

// GetCache returns the template method cache
func (tmsp *TemplateMethodServiceProvider) GetCache() *TemplateMethodCache {
	return tmsp.cache
}

// GetValidator returns the template method validator
func (tmsp *TemplateMethodServiceProvider) GetValidator() *TemplateMethodValidator {
	return tmsp.validator
}

// GetMonitor returns the template method monitor
func (tmsp *TemplateMethodServiceProvider) GetMonitor() *TemplateMethodMonitor {
	return tmsp.monitor
}

// GetAuditor returns the template method auditor
func (tmsp *TemplateMethodServiceProvider) GetAuditor() *TemplateMethodAuditor {
	return tmsp.auditor
}

// TemplateMethodHandler handles template method operations
type TemplateMethodHandler struct {
	provider *TemplateMethodServiceProvider
	config   *TemplateMethodConfig
}

// NewTemplateMethodHandler creates a new template method handler
func NewTemplateMethodHandler(config *TemplateMethodConfig) *TemplateMethodHandler {
	return &TemplateMethodHandler{
		provider: NewTemplateMethodServiceProvider(config),
		config:   config,
	}
}

// HandleCreateTemplateMethod handles template method creation
func (tmh *TemplateMethodHandler) HandleCreateTemplateMethod(ctx context.Context, name string, template TemplateMethod) error {
	// Validate template method
	if err := tmh.provider.GetValidator().ValidateTemplateMethod(template); err != nil {
		return err
	}

	// Create template method
	return tmh.provider.GetService().CreateTemplateMethod(name, template)
}

// HandleGetTemplateMethod handles template method retrieval
func (tmh *TemplateMethodHandler) HandleGetTemplateMethod(ctx context.Context, name string) (TemplateMethod, error) {
	// Check cache first
	if cached, exists := tmh.provider.GetCache().Get(name); exists {
		return cached, nil
	}

	// Get from service
	template, err := tmh.provider.GetService().GetTemplateMethod(name)
	if err != nil {
		return nil, err
	}

	// Cache the template method
	tmh.provider.GetCache().Set(name, template, tmh.config.GetCache().GetTTL())

	return template, nil
}

// HandleRemoveTemplateMethod handles template method removal
func (tmh *TemplateMethodHandler) HandleRemoveTemplateMethod(ctx context.Context, name string) error {
	// Remove from cache
	tmh.provider.GetCache().Delete(name)

	// Remove from service
	return tmh.provider.GetService().RemoveTemplateMethod(name)
}

// HandleListTemplateMethods handles template method listing
func (tmh *TemplateMethodHandler) HandleListTemplateMethods(ctx context.Context) []string {
	return tmh.provider.GetService().ListTemplateMethods()
}

// HandleExecuteTemplateMethod handles template method execution
func (tmh *TemplateMethodHandler) HandleExecuteTemplateMethod(ctx context.Context, name string) error {
	// Get template method
	template, err := tmh.provider.GetService().GetTemplateMethod(name)
	if err != nil {
		return err
	}

	// Execute template method
	return tmh.provider.GetService().ExecuteTemplateMethod(template)
}

// HandleExecuteStep handles step execution
func (tmh *TemplateMethodHandler) HandleExecuteStep(ctx context.Context, templateName string, stepName string) error {
	// Get template method
	template, err := tmh.provider.GetService().GetTemplateMethod(templateName)
	if err != nil {
		return err
	}

	// Find step
	var step Step
	for _, s := range template.GetSteps() {
		if s.GetName() == stepName {
			step = s
			break
		}
	}

	if step == nil {
		return ErrStepNotFound
	}

	// Execute step
	return tmh.provider.GetService().ExecuteStep(step)
}

// HandleGetTemplateMethodStats handles template method statistics retrieval
func (tmh *TemplateMethodHandler) HandleGetTemplateMethodStats(ctx context.Context, name string) (map[string]interface{}, error) {
	// Get template method
	template, err := tmh.provider.GetService().GetTemplateMethod(name)
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
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
		"error":        template.GetError(),
		"metadata":     template.GetMetadata(),
	}

	return stats, nil
}

// HandleGetExecutionStats handles execution statistics retrieval
func (tmh *TemplateMethodHandler) HandleGetExecutionStats(ctx context.Context) map[string]interface{} {
	return tmh.provider.GetService().GetExecutionStats()
}

// HandleGetExecutionHistory handles execution history retrieval
func (tmh *TemplateMethodHandler) HandleGetExecutionHistory(ctx context.Context) []ExecutionRecord {
	return tmh.provider.GetService().GetExecutionHistory()
}

// HandleClearExecutionHistory handles execution history clearing
func (tmh *TemplateMethodHandler) HandleClearExecutionHistory(ctx context.Context) error {
	return tmh.provider.GetService().ClearExecutionHistory()
}

// HandleCleanup handles cleanup operations
func (tmh *TemplateMethodHandler) HandleCleanup(ctx context.Context) error {
	return tmh.provider.GetService().Cleanup()
}

// TemplateMethodProcessor processes template method operations
type TemplateMethodProcessor struct {
	handler *TemplateMethodHandler
	config  *TemplateMethodConfig
}

// NewTemplateMethodProcessor creates a new template method processor
func NewTemplateMethodProcessor(config *TemplateMethodConfig) *TemplateMethodProcessor {
	return &TemplateMethodProcessor{
		handler: NewTemplateMethodHandler(config),
		config:  config,
	}
}

// ProcessTemplateMethod processes a template method
func (tmp *TemplateMethodProcessor) ProcessTemplateMethod(ctx context.Context, name string, processor func(TemplateMethod) error) error {
	// Get template method
	template, err := tmp.handler.HandleGetTemplateMethod(ctx, name)
	if err != nil {
		return err
	}

	// Validate template method
	if err := tmp.handler.provider.GetValidator().ValidateTemplateMethod(template); err != nil {
		return err
	}

	// Process template method
	return processor(template)
}

// ProcessStep processes a step
func (tmp *TemplateMethodProcessor) ProcessStep(ctx context.Context, templateName string, stepName string, processor func(Step) error) error {
	// Get template method
	template, err := tmp.handler.HandleGetTemplateMethod(ctx, templateName)
	if err != nil {
		return err
	}

	// Find step
	var step Step
	for _, s := range template.GetSteps() {
		if s.GetName() == stepName {
			step = s
			break
		}
	}

	if step == nil {
		return ErrStepNotFound
	}

	// Validate step
	if err := tmp.handler.provider.GetValidator().ValidateStep(step); err != nil {
		return err
	}

	// Process step
	return processor(step)
}

// ProcessSteps processes multiple steps
func (tmp *TemplateMethodProcessor) ProcessSteps(ctx context.Context, templateName string, processor func([]Step) error) error {
	// Get template method
	template, err := tmp.handler.HandleGetTemplateMethod(ctx, templateName)
	if err != nil {
		return err
	}

	// Validate steps
	if err := tmp.handler.provider.GetValidator().ValidateSteps(template.GetSteps()); err != nil {
		return err
	}

	// Process steps
	return processor(template.GetSteps())
}

// TemplateMethodScheduler schedules template method operations
type TemplateMethodScheduler struct {
	processor *TemplateMethodProcessor
	config    *TemplateMethodConfig
	ticker    *time.Ticker
	stop      chan bool
	mutex     sync.RWMutex
}

// NewTemplateMethodScheduler creates a new template method scheduler
func NewTemplateMethodScheduler(config *TemplateMethodConfig) *TemplateMethodScheduler {
	return &TemplateMethodScheduler{
		processor: NewTemplateMethodProcessor(config),
		config:    config,
		stop:      make(chan bool),
	}
}

// Start starts the scheduler
func (tms *TemplateMethodScheduler) Start() {
	tms.mutex.Lock()
	defer tms.mutex.Unlock()

	tms.ticker = time.NewTicker(tms.config.GetMonitoring().GetCollectInterval())

	go func() {
		for {
			select {
			case <-tms.ticker.C:
				tms.processScheduledTasks()
			case <-tms.stop:
				return
			}
		}
	}()
}

// Stop stops the scheduler
func (tms *TemplateMethodScheduler) Stop() {
	tms.mutex.Lock()
	defer tms.mutex.Unlock()

	if tms.ticker != nil {
		tms.ticker.Stop()
	}

	close(tms.stop)
}

// processScheduledTasks processes scheduled tasks
func (tms *TemplateMethodScheduler) processScheduledTasks() {
	// Process template method cleanup
	ctx := context.Background()

	// Get all template methods
	templates := tms.processor.handler.HandleListTemplateMethods(ctx)

	// Process each template method
	for _, name := range templates {
		tms.processor.handler.HandleCleanup(ctx)
	}
}

// TemplateMethodMonitor monitors template method operations
type TemplateMethodMonitor struct {
	processor *TemplateMethodProcessor
	config    *TemplateMethodConfig
	metrics   map[string]interface{}
	mutex     sync.RWMutex
}

// NewTemplateMethodMonitor creates a new template method monitor
func NewTemplateMethodMonitor(config *TemplateMethodConfig) *TemplateMethodMonitor {
	return &TemplateMethodMonitor{
		processor: NewTemplateMethodProcessor(config),
		config:    config,
		metrics:   make(map[string]interface{}),
	}
}

// RecordTemplateMethodExecution records template method execution
func (tmm *TemplateMethodMonitor) RecordTemplateMethodExecution(template TemplateMethod) error {
	tmm.mutex.Lock()
	defer tmm.mutex.Unlock()

	// Update metrics
	tmm.metrics["total_executions"] = tmm.metrics["total_executions"].(int) + 1
	tmm.metrics["last_execution"] = time.Now()

	return nil
}

// RecordStepExecution records step execution
func (tmm *TemplateMethodMonitor) RecordStepExecution(step Step) error {
	tmm.mutex.Lock()
	defer tmm.mutex.Unlock()

	// Update metrics
	tmm.metrics["total_step_executions"] = tmm.metrics["total_step_executions"].(int) + 1
	tmm.metrics["last_step_execution"] = time.Now()

	return nil
}

// GetTemplateMethodMetrics returns template method metrics
func (tmm *TemplateMethodMonitor) GetTemplateMethodMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetStepMetrics returns step metrics
func (tmm *TemplateMethodMonitor) GetStepMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetExecutionMetrics returns execution metrics
func (tmm *TemplateMethodMonitor) GetExecutionMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetPerformanceMetrics returns performance metrics
func (tmm *TemplateMethodMonitor) GetPerformanceMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetErrorMetrics returns error metrics
func (tmm *TemplateMethodMonitor) GetErrorMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetResourceMetrics returns resource metrics
func (tmm *TemplateMethodMonitor) GetResourceMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetHealthMetrics returns health metrics
func (tmm *TemplateMethodMonitor) GetHealthMetrics() map[string]interface{} {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return tmm.metrics
}

// GetAlertMetrics returns alert metrics
func (tmm *TemplateMethodMonitor) GetAlertMetrics() []Alert {
	tmm.mutex.RLock()
	defer tmm.mutex.RUnlock()

	return []Alert{}
}

// ResetMetrics resets the metrics
func (tmm *TemplateMethodMonitor) ResetMetrics() error {
	tmm.mutex.Lock()
	defer tmm.mutex.Unlock()

	tmm.metrics = make(map[string]interface{})
	return nil
}

// TemplateMethodAuditor audits template method operations
type TemplateMethodAuditor struct {
	processor *TemplateMethodProcessor
	config    *TemplateMethodConfig
	logs      []AuditLog
	mutex     sync.RWMutex
}

// NewTemplateMethodAuditor creates a new template method auditor
func NewTemplateMethodAuditor(config *TemplateMethodConfig) *TemplateMethodAuditor {
	return &TemplateMethodAuditor{
		processor: NewTemplateMethodProcessor(config),
		config:    config,
		logs:      make([]AuditLog, 0),
	}
}

// AuditTemplateMethodExecution audits template method execution
func (tma *TemplateMethodAuditor) AuditTemplateMethodExecution(template TemplateMethod, userID string) error {
	tma.mutex.Lock()
	defer tma.mutex.Unlock()

	log := &BaseAuditLog{
		ID:           generateID(),
		UserID:       userID,
		Action:       "execute_template_method",
		Resource:     template.GetName(),
		Details:      map[string]interface{}{"template_method": template.GetName()},
		Timestamp:    time.Now(),
		IP:           "",
		UserAgent:    "",
		SessionID:    "",
		RequestID:    "",
		ResponseCode: 0,
		ResponseTime: 0,
		Successful:   true,
		Error:        "",
		Metadata:     make(map[string]interface{}),
	}

	tma.logs = append(tma.logs, log)
	return nil
}

// AuditStepExecution audits step execution
func (tma *TemplateMethodAuditor) AuditStepExecution(step Step, userID string) error {
	tma.mutex.Lock()
	defer tma.mutex.Unlock()

	log := &BaseAuditLog{
		ID:           generateID(),
		UserID:       userID,
		Action:       "execute_step",
		Resource:     step.GetName(),
		Details:      map[string]interface{}{"step": step.GetName()},
		Timestamp:    time.Now(),
		IP:           "",
		UserAgent:    "",
		SessionID:    "",
		RequestID:    "",
		ResponseCode: 0,
		ResponseTime: 0,
		Successful:   true,
		Error:        "",
		Metadata:     make(map[string]interface{}),
	}

	tma.logs = append(tma.logs, log)
	return nil
}

// GetAuditLogs returns audit logs
func (tma *TemplateMethodAuditor) GetAuditLogs() ([]AuditLog, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	return tma.logs, nil
}

// GetAuditLogsByUser returns audit logs by user
func (tma *TemplateMethodAuditor) GetAuditLogsByUser(userID string) ([]AuditLog, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	var userLogs []AuditLog
	for _, log := range tma.logs {
		if log.GetUserID() == userID {
			userLogs = append(userLogs, log)
		}
	}

	return userLogs, nil
}

// GetAuditLogsByTemplateMethod returns audit logs by template method
func (tma *TemplateMethodAuditor) GetAuditLogsByTemplateMethod(templateMethodName string) ([]AuditLog, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	var templateLogs []AuditLog
	for _, log := range tma.logs {
		if log.GetResource() == templateMethodName {
			templateLogs = append(templateLogs, log)
		}
	}

	return templateLogs, nil
}

// GetAuditLogsByDateRange returns audit logs by date range
func (tma *TemplateMethodAuditor) GetAuditLogsByDateRange(start, end time.Time) ([]AuditLog, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	var dateLogs []AuditLog
	for _, log := range tma.logs {
		if log.GetTimestamp().After(start) && log.GetTimestamp().Before(end) {
			dateLogs = append(dateLogs, log)
		}
	}

	return dateLogs, nil
}

// GetAuditLogsByAction returns audit logs by action
func (tma *TemplateMethodAuditor) GetAuditLogsByAction(action string) ([]AuditLog, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	var actionLogs []AuditLog
	for _, log := range tma.logs {
		if log.GetAction() == action {
			actionLogs = append(actionLogs, log)
		}
	}

	return actionLogs, nil
}

// GetAuditStats returns audit statistics
func (tma *TemplateMethodAuditor) GetAuditStats() map[string]interface{} {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	stats := map[string]interface{}{
		"total_logs": len(tma.logs),
		"actions":    make(map[string]int),
		"users":      make(map[string]int),
		"resources":  make(map[string]int),
	}

	for _, log := range tma.logs {
		// Count actions
		action := log.GetAction()
		stats["actions"].(map[string]int)[action]++

		// Count users
		userID := log.GetUserID()
		stats["users"].(map[string]int)[userID]++

		// Count resources
		resource := log.GetResource()
		stats["resources"].(map[string]int)[resource]++
	}

	return stats
}

// ClearAuditLogs clears audit logs
func (tma *TemplateMethodAuditor) ClearAuditLogs() error {
	tma.mutex.Lock()
	defer tma.mutex.Unlock()

	tma.logs = make([]AuditLog, 0)
	return nil
}

// ExportAuditLogs exports audit logs
func (tma *TemplateMethodAuditor) ExportAuditLogs(format string) ([]byte, error) {
	tma.mutex.RLock()
	defer tma.mutex.RUnlock()

	// Mock implementation
	return []byte("audit logs"), nil
}

// BaseAuditLog provides common functionality for audit logs
type BaseAuditLog struct {
	ID           string                 `json:"id"`
	UserID       string                 `json:"user_id"`
	Action       string                 `json:"action"`
	Resource     string                 `json:"resource"`
	Details      map[string]interface{} `json:"details"`
	IP           string                 `json:"ip"`
	UserAgent    string                 `json:"user_agent"`
	Timestamp    time.Time              `json:"timestamp"`
	SessionID    string                 `json:"session_id"`
	RequestID    string                 `json:"request_id"`
	ResponseCode int                    `json:"response_code"`
	ResponseTime time.Duration          `json:"response_time"`
	Successful   bool                   `json:"successful"`
	Error        string                 `json:"error"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// GetID returns the audit log ID
func (bal *BaseAuditLog) GetID() string {
	return bal.ID
}

// GetUserID returns the user ID
func (bal *BaseAuditLog) GetUserID() string {
	return bal.UserID
}

// GetAction returns the action
func (bal *BaseAuditLog) GetAction() string {
	return bal.Action
}

// GetResource returns the resource
func (bal *BaseAuditLog) GetResource() string {
	return bal.Resource
}

// GetDetails returns the details
func (bal *BaseAuditLog) GetDetails() map[string]interface{} {
	return bal.Details
}

// GetIP returns the IP
func (bal *BaseAuditLog) GetIP() string {
	return bal.IP
}

// GetUserAgent returns the user agent
func (bal *BaseAuditLog) GetUserAgent() string {
	return bal.UserAgent
}

// GetTimestamp returns the timestamp
func (bal *BaseAuditLog) GetTimestamp() time.Time {
	return bal.Timestamp
}

// GetSessionID returns the session ID
func (bal *BaseAuditLog) GetSessionID() string {
	return bal.SessionID
}

// GetRequestID returns the request ID
func (bal *BaseAuditLog) GetRequestID() string {
	return bal.RequestID
}

// GetResponseCode returns the response code
func (bal *BaseAuditLog) GetResponseCode() int {
	return bal.ResponseCode
}

// SetResponseCode sets the response code
func (bal *BaseAuditLog) SetResponseCode(code int) {
	bal.ResponseCode = code
}

// GetResponseTime returns the response time
func (bal *BaseAuditLog) GetResponseTime() time.Duration {
	return bal.ResponseTime
}

// SetResponseTime sets the response time
func (bal *BaseAuditLog) SetResponseTime(responseTime time.Duration) {
	bal.ResponseTime = responseTime
}

// IsSuccessful returns whether the operation was successful
func (bal *BaseAuditLog) IsSuccessful() bool {
	return bal.Successful
}

// SetSuccessful sets the successful flag
func (bal *BaseAuditLog) SetSuccessful(successful bool) {
	bal.Successful = successful
}

// GetError returns the error
func (bal *BaseAuditLog) GetError() string {
	return bal.Error
}

// SetError sets the error
func (bal *BaseAuditLog) SetError(error string) {
	bal.Error = error
}

// GetMetadata returns the metadata
func (bal *BaseAuditLog) GetMetadata() map[string]interface{} {
	return bal.Metadata
}

// SetMetadata sets the metadata
func (bal *BaseAuditLog) SetMetadata(metadata map[string]interface{}) {
	bal.Metadata = metadata
}
