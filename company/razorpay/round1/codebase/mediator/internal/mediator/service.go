package mediator

import (
	"context"
	"sync"
	"time"
)

// MediatorServiceProvider provides mediator services
type MediatorServiceProvider struct {
	service    *MediatorService
	manager    *MediatorManager
	cache      *MediatorCache
	validator  *MediatorValidator
	pool       *MediatorPool
	metrics    *MediatorMetrics
	mutex      sync.RWMutex
}

// NewMediatorServiceProvider creates a new mediator service provider
func NewMediatorServiceProvider(config *MediatorConfig) *MediatorServiceProvider {
	return &MediatorServiceProvider{
		service:    NewMediatorService(config),
		manager:    NewMediatorManager(config),
		cache:      NewMediatorCache(config.Cache.TTL),
		validator:  NewMediatorValidator(config),
		pool:       NewMediatorPool(config),
		metrics:    &MediatorMetrics{},
	}
}

// GetService returns the mediator service
func (msp *MediatorServiceProvider) GetService() *MediatorService {
	return msp.service
}

// GetManager returns the mediator manager
func (msp *MediatorServiceProvider) GetManager() *MediatorManager {
	return msp.manager
}

// GetCache returns the mediator cache
func (msp *MediatorServiceProvider) GetCache() *MediatorCache {
	return msp.cache
}

// GetValidator returns the mediator validator
func (msp *MediatorServiceProvider) GetValidator() *MediatorValidator {
	return msp.validator
}

// GetPool returns the mediator pool
func (msp *MediatorServiceProvider) GetPool() *MediatorPool {
	return msp.pool
}

// GetMetrics returns the mediator metrics
func (msp *MediatorServiceProvider) GetMetrics() *MediatorMetrics {
	return msp.metrics
}

// MediatorHandler handles mediator operations
type MediatorHandler struct {
	provider *MediatorServiceProvider
	config   *MediatorConfig
}

// NewMediatorHandler creates a new mediator handler
func NewMediatorHandler(config *MediatorConfig) *MediatorHandler {
	return &MediatorHandler{
		provider: NewMediatorServiceProvider(config),
		config:   config,
	}
}

// HandleCreateMediator handles mediator creation
func (mh *MediatorHandler) HandleCreateMediator(ctx context.Context, name string) (Mediator, error) {
	// Check cache first
	if cached, exists := mh.provider.GetCache().Get(name); exists {
		if mediator, ok := cached.(Mediator); ok {
			return mediator, nil
		}
	}
	
	// Create new mediator
	mediator, err := mh.provider.GetManager().CreateMediator(name)
	if err != nil {
		return nil, err
	}
	
	// Cache the mediator
	mh.provider.GetCache().Set(name, mediator)
	
	return mediator, nil
}

// HandleGetMediator handles mediator retrieval
func (mh *MediatorHandler) HandleGetMediator(ctx context.Context, name string) (Mediator, error) {
	// Check cache first
	if cached, exists := mh.provider.GetCache().Get(name); exists {
		if mediator, ok := cached.(Mediator); ok {
			return mediator, nil
		}
	}
	
	// Get from manager
	mediator, err := mh.provider.GetManager().GetMediator(name)
	if err != nil {
		return nil, err
	}
	
	// Cache the mediator
	mh.provider.GetCache().Set(name, mediator)
	
	return mediator, nil
}

// HandleRemoveMediator handles mediator removal
func (mh *MediatorHandler) HandleRemoveMediator(ctx context.Context, name string) error {
	// Remove from cache
	mh.provider.GetCache().Delete(name)
	
	// Remove from manager
	return mh.provider.GetManager().RemoveMediator(name)
}

// HandleListMediators handles mediator listing
func (mh *MediatorHandler) HandleListMediators(ctx context.Context) []string {
	return mh.provider.GetManager().ListMediators()
}

// HandleRegisterColleague handles colleague registration
func (mh *MediatorHandler) HandleRegisterColleague(ctx context.Context, mediatorName string, colleague Colleague) error {
	// Validate colleague
	if err := mh.provider.GetValidator().ValidateColleague(colleague); err != nil {
		return err
	}
	
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return err
	}
	
	// Register colleague
	return mediator.RegisterColleague(colleague)
}

// HandleUnregisterColleague handles colleague unregistration
func (mh *MediatorHandler) HandleUnregisterColleague(ctx context.Context, mediatorName string, colleagueID string) error {
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return err
	}
	
	// Unregister colleague
	return mediator.UnregisterColleague(colleagueID)
}

// HandleSendMessage handles message sending
func (mh *MediatorHandler) HandleSendMessage(ctx context.Context, mediatorName string, senderID string, recipientID string, message interface{}) error {
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return err
	}
	
	// Send message
	return mediator.SendMessage(senderID, recipientID, message)
}

// HandleBroadcastMessage handles message broadcasting
func (mh *MediatorHandler) HandleBroadcastMessage(ctx context.Context, mediatorName string, senderID string, message interface{}) error {
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return err
	}
	
	// Broadcast message
	return mediator.BroadcastMessage(senderID, message)
}

// HandleGetColleagues handles colleague retrieval
func (mh *MediatorHandler) HandleGetColleagues(ctx context.Context, mediatorName string) ([]Colleague, error) {
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return nil, err
	}
	
	// Get colleagues
	return mediator.GetColleagues(), nil
}

// HandleGetColleague handles specific colleague retrieval
func (mh *MediatorHandler) HandleGetColleague(ctx context.Context, mediatorName string, colleagueID string) (Colleague, error) {
	// Get mediator
	mediator, err := mh.provider.GetManager().GetMediator(mediatorName)
	if err != nil {
		return nil, err
	}
	
	// Get colleague
	return mediator.GetColleague(colleagueID)
}

// MediatorProcessor processes mediator operations
type MediatorProcessor struct {
	handler *MediatorHandler
	config  *MediatorConfig
}

// NewMediatorProcessor creates a new mediator processor
func NewMediatorProcessor(config *MediatorConfig) *MediatorProcessor {
	return &MediatorProcessor{
		handler: NewMediatorHandler(config),
		config:  config,
	}
}

// ProcessMediator processes a mediator
func (mp *MediatorProcessor) ProcessMediator(ctx context.Context, name string, processor func(Mediator) error) error {
	mediator, err := mp.handler.HandleGetMediator(ctx, name)
	if err != nil {
		return err
	}
	
	// Validate mediator
	if err := mp.handler.provider.GetValidator().ValidateMediator(mediator); err != nil {
		return err
	}
	
	// Process mediator
	return processor(mediator)
}

// ProcessColleague processes a colleague
func (mp *MediatorProcessor) ProcessColleague(ctx context.Context, mediatorName string, colleagueID string, processor func(Colleague) error) error {
	colleague, err := mp.handler.HandleGetColleague(ctx, mediatorName, colleagueID)
	if err != nil {
		return err
	}
	
	// Validate colleague
	if err := mp.handler.provider.GetValidator().ValidateColleague(colleague); err != nil {
		return err
	}
	
	// Process colleague
	return processor(colleague)
}

// ProcessMessage processes a message
func (mp *MediatorProcessor) ProcessMessage(ctx context.Context, mediatorName string, senderID string, recipientID string, message interface{}, processor func(interface{}) error) error {
	// Send message
	if err := mp.handler.HandleSendMessage(ctx, mediatorName, senderID, recipientID, message); err != nil {
		return err
	}
	
	// Process message
	return processor(message)
}

// ProcessBroadcast processes a broadcast
func (mp *MediatorProcessor) ProcessBroadcast(ctx context.Context, mediatorName string, senderID string, message interface{}, processor func(interface{}) error) error {
	// Broadcast message
	if err := mp.handler.HandleBroadcastMessage(ctx, mediatorName, senderID, message); err != nil {
		return err
	}
	
	// Process broadcast
	return processor(message)
}

// MediatorScheduler schedules mediator operations
type MediatorScheduler struct {
	processor *MediatorProcessor
	config    *MediatorConfig
	ticker    *time.Ticker
	stop      chan bool
	mutex     sync.RWMutex
}

// NewMediatorScheduler creates a new mediator scheduler
func NewMediatorScheduler(config *MediatorConfig) *MediatorScheduler {
	return &MediatorScheduler{
		processor: NewMediatorProcessor(config),
		config:    config,
		stop:      make(chan bool),
	}
}

// Start starts the scheduler
func (ms *MediatorScheduler) Start() {
	ms.mutex.Lock()
	defer ms.mutex.Unlock()
	
	ms.ticker = time.NewTicker(ms.config.Monitoring.CollectInterval)
	
	go func() {
		for {
			select {
			case <-ms.ticker.C:
				ms.processScheduledTasks()
			case <-ms.stop:
				return
			}
		}
	}()
}

// Stop stops the scheduler
func (ms *MediatorScheduler) Stop() {
	ms.mutex.Lock()
	defer ms.mutex.Unlock()
	
	if ms.ticker != nil {
		ms.ticker.Stop()
	}
	
	close(ms.stop)
}

// processScheduledTasks processes scheduled tasks
func (ms *MediatorScheduler) processScheduledTasks() {
	// Process mediator cleanup
	ctx := context.Background()
	
	// Get all mediators
	mediators := ms.processor.handler.HandleListMediators(ctx)
	
	// Process each mediator
	for _, name := range mediators {
		mediator, err := ms.processor.handler.HandleGetMediator(ctx, name)
		if err != nil {
			continue
		}
		
		// Check if mediator needs cleanup
		colleagues := mediator.GetColleagues()
		for _, colleague := range colleagues {
			if !colleague.IsActive() && time.Since(colleague.GetLastActivity()) > ms.config.Timeout {
				mediator.UnregisterColleague(colleague.GetID())
			}
		}
	}
}

// MediatorMonitor monitors mediator operations
type MediatorMonitor struct {
	processor *MediatorProcessor
	config    *MediatorConfig
	metrics   *MediatorMetrics
	mutex     sync.RWMutex
}

// NewMediatorMonitor creates a new mediator monitor
func NewMediatorMonitor(config *MediatorConfig) *MediatorMonitor {
	return &MediatorMonitor{
		processor: NewMediatorProcessor(config),
		config:    config,
		metrics:   &MediatorMetrics{},
	}
}

// MonitorMediator monitors a mediator
func (mm *MediatorMonitor) MonitorMediator(ctx context.Context, name string) error {
	mediator, err := mm.processor.handler.HandleGetMediator(ctx, name)
	if err != nil {
		return err
	}
	
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	// Update metrics
	mm.metrics.UpdateMetrics(mediator)
	
	return nil
}

// GetMetrics returns current metrics
func (mm *MediatorMonitor) GetMetrics() *MediatorMetrics {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	
	return mm.metrics
}

// ResetMetrics resets the metrics
func (mm *MediatorMonitor) ResetMetrics() {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	mm.metrics = &MediatorMetrics{}
}

// MediatorAuditor audits mediator operations
type MediatorAuditor struct {
	processor *MediatorProcessor
	config    *MediatorConfig
	logs      []AuditLog
	mutex     sync.RWMutex
}

// NewMediatorAuditor creates a new mediator auditor
func NewMediatorAuditor(config *MediatorConfig) *MediatorAuditor {
	return &MediatorAuditor{
		processor: NewMediatorProcessor(config),
		config:    config,
		logs:      make([]AuditLog, 0),
	}
}

// AuditOperation audits an operation
func (ma *MediatorAuditor) AuditOperation(ctx context.Context, userID string, action string, resource string, details map[string]interface{}) {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	
	log := AuditLog{
		ID:        generateID(),
		UserID:    userID,
		Action:    action,
		Resource:  resource,
		Details:   details,
		Timestamp: time.Now(),
	}
	
	ma.logs = append(ma.logs, log)
}

// GetAuditLogs returns audit logs
func (ma *MediatorAuditor) GetAuditLogs() []AuditLog {
	ma.mutex.RLock()
	defer ma.mutex.RUnlock()
	
	return ma.logs
}

// ClearAuditLogs clears audit logs
func (ma *MediatorAuditor) ClearAuditLogs() {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	
	ma.logs = make([]AuditLog, 0)
}

// generateID generates a unique ID
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

// randomString generates a random string
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}
