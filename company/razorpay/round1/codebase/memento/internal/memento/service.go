package memento

import (
	"context"
	"sync"
	"time"
)

// MementoServiceProvider provides memento services
type MementoServiceProvider struct {
	service    *MementoService
	manager    *MementoManager
	cache      *MementoCache
	validator  *MementoValidator
	pool       *MementoPool
	metrics    *MementoMetrics
	mutex      sync.RWMutex
}

// NewMementoServiceProvider creates a new memento service provider
func NewMementoServiceProvider(config *MementoConfig) *MementoServiceProvider {
	return &MementoServiceProvider{
		service:    NewMementoService(config),
		manager:    NewMementoManager(config),
		cache:      NewMementoCache(config.GetCache().GetTTL()),
		validator:  NewMementoValidator(config),
		pool:       NewMementoPool(config),
		metrics:    &MementoMetrics{},
	}
}

// GetService returns the memento service
func (msp *MementoServiceProvider) GetService() *MementoService {
	return msp.service
}

// GetManager returns the memento manager
func (msp *MementoServiceProvider) GetManager() *MementoManager {
	return msp.manager
}

// GetCache returns the memento cache
func (msp *MementoServiceProvider) GetCache() *MementoCache {
	return msp.cache
}

// GetValidator returns the memento validator
func (msp *MementoServiceProvider) GetValidator() *MementoValidator {
	return msp.validator
}

// GetPool returns the memento pool
func (msp *MementoServiceProvider) GetPool() *MementoPool {
	return msp.pool
}

// GetMetrics returns the memento metrics
func (msp *MementoServiceProvider) GetMetrics() *MementoMetrics {
	return msp.metrics
}

// MementoHandler handles memento operations
type MementoHandler struct {
	provider *MementoServiceProvider
	config   *MementoConfig
}

// NewMementoHandler creates a new memento handler
func NewMementoHandler(config *MementoConfig) *MementoHandler {
	return &MementoHandler{
		provider: NewMementoServiceProvider(config),
		config:   config,
	}
}

// HandleCreateCaretaker handles caretaker creation
func (mh *MementoHandler) HandleCreateCaretaker(ctx context.Context, name string) (Caretaker, error) {
	// Check cache first
	if cached, exists := mh.provider.GetCache().Get(name); exists {
		if caretaker, ok := cached.(Caretaker); ok {
			return caretaker, nil
		}
	}
	
	// Create new caretaker
	caretaker, err := mh.provider.GetManager().CreateCaretaker(name)
	if err != nil {
		return nil, err
	}
	
	// Cache the caretaker
	mh.provider.GetCache().Set(name, caretaker, mh.config.GetCache().GetTTL())
	
	return caretaker, nil
}

// HandleGetCaretaker handles caretaker retrieval
func (mh *MementoHandler) HandleGetCaretaker(ctx context.Context, name string) (Caretaker, error) {
	// Check cache first
	if cached, exists := mh.provider.GetCache().Get(name); exists {
		if caretaker, ok := cached.(Caretaker); ok {
			return caretaker, nil
		}
	}
	
	// Get from manager
	caretaker, err := mh.provider.GetManager().GetCaretaker(name)
	if err != nil {
		return nil, err
	}
	
	// Cache the caretaker
	mh.provider.GetCache().Set(name, caretaker, mh.config.GetCache().GetTTL())
	
	return caretaker, nil
}

// HandleRemoveCaretaker handles caretaker removal
func (mh *MementoHandler) HandleRemoveCaretaker(ctx context.Context, name string) error {
	// Remove from cache
	mh.provider.GetCache().Delete(name)
	
	// Remove from manager
	return mh.provider.GetManager().RemoveCaretaker(name)
}

// HandleListCaretakers handles caretaker listing
func (mh *MementoHandler) HandleListCaretakers(ctx context.Context) []string {
	return mh.provider.GetManager().ListCaretakers()
}

// HandleSaveMemento handles memento saving
func (mh *MementoHandler) HandleSaveMemento(ctx context.Context, caretakerName string, memento Memento) error {
	// Validate memento
	if err := mh.provider.GetValidator().Validate(memento); err != nil {
		return err
	}
	
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Save memento
	return caretaker.SaveMemento(memento)
}

// HandleGetMemento handles memento retrieval
func (mh *MementoHandler) HandleGetMemento(ctx context.Context, caretakerName string, mementoID string) (Memento, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return nil, err
	}
	
	// Get memento
	return caretaker.GetMemento(mementoID)
}

// HandleGetMementosByOriginator handles memento retrieval by originator
func (mh *MementoHandler) HandleGetMementosByOriginator(ctx context.Context, caretakerName string, originatorID string) ([]Memento, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return nil, err
	}
	
	// Get mementos
	return caretaker.GetMementosByOriginator(originatorID)
}

// HandleGetMementosByType handles memento retrieval by type
func (mh *MementoHandler) HandleGetMementosByType(ctx context.Context, caretakerName string, mementoType string) ([]Memento, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return nil, err
	}
	
	// Get mementos
	return caretaker.GetMementosByType(mementoType)
}

// HandleGetMementosByDateRange handles memento retrieval by date range
func (mh *MementoHandler) HandleGetMementosByDateRange(ctx context.Context, caretakerName string, start, end time.Time) ([]Memento, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return nil, err
	}
	
	// Get mementos
	return caretaker.GetMementosByDateRange(start, end)
}

// HandleDeleteMemento handles memento deletion
func (mh *MementoHandler) HandleDeleteMemento(ctx context.Context, caretakerName string, mementoID string) error {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Delete memento
	return caretaker.DeleteMemento(mementoID)
}

// HandleDeleteMementosByOriginator handles memento deletion by originator
func (mh *MementoHandler) HandleDeleteMementosByOriginator(ctx context.Context, caretakerName string, originatorID string) error {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Delete mementos
	return caretaker.DeleteMementosByOriginator(originatorID)
}

// HandleDeleteMementosByType handles memento deletion by type
func (mh *MementoHandler) HandleDeleteMementosByType(ctx context.Context, caretakerName string, mementoType string) error {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Delete mementos
	return caretaker.DeleteMementosByType(mementoType)
}

// HandleDeleteMementosByDateRange handles memento deletion by date range
func (mh *MementoHandler) HandleDeleteMementosByDateRange(ctx context.Context, caretakerName string, start, end time.Time) error {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Delete mementos
	return caretaker.DeleteMementosByDateRange(start, end)
}

// HandleGetMementoCount handles memento count retrieval
func (mh *MementoHandler) HandleGetMementoCount(ctx context.Context, caretakerName string) (int, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return 0, err
	}
	
	// Get count
	return caretaker.GetMementoCount(), nil
}

// HandleGetMementoSize handles memento size retrieval
func (mh *MementoHandler) HandleGetMementoSize(ctx context.Context, caretakerName string) (int64, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return 0, err
	}
	
	// Get size
	return caretaker.GetMementoSize(), nil
}

// HandleGetCaretakerStats handles caretaker statistics retrieval
func (mh *MementoHandler) HandleGetCaretakerStats(ctx context.Context, caretakerName string) (map[string]interface{}, error) {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return nil, err
	}
	
	// Get stats
	return caretaker.GetStats(), nil
}

// HandleCleanup handles cleanup operations
func (mh *MementoHandler) HandleCleanup(ctx context.Context, caretakerName string) error {
	// Get caretaker
	caretaker, err := mh.provider.GetManager().GetCaretaker(caretakerName)
	if err != nil {
		return err
	}
	
	// Cleanup
	return caretaker.Cleanup()
}

// MementoProcessor processes memento operations
type MementoProcessor struct {
	handler *MementoHandler
	config  *MementoConfig
}

// NewMementoProcessor creates a new memento processor
func NewMementoProcessor(config *MementoConfig) *MementoProcessor {
	return &MementoProcessor{
		handler: NewMementoHandler(config),
		config:  config,
	}
}

// ProcessMemento processes a memento
func (mp *MementoProcessor) ProcessMemento(ctx context.Context, caretakerName string, memento Memento, processor func(Memento) error) error {
	// Validate memento
	if err := mp.handler.provider.GetValidator().Validate(memento); err != nil {
		return err
	}
	
	// Process memento
	return processor(memento)
}

// ProcessMementos processes multiple mementos
func (mp *MementoProcessor) ProcessMementos(ctx context.Context, caretakerName string, mementos []Memento, processor func([]Memento) error) error {
	// Validate mementos
	for _, memento := range mementos {
		if err := mp.handler.provider.GetValidator().Validate(memento); err != nil {
			return err
		}
	}
	
	// Process mementos
	return processor(mementos)
}

// ProcessCaretaker processes a caretaker
func (mp *MementoProcessor) ProcessCaretaker(ctx context.Context, caretakerName string, processor func(Caretaker) error) error {
	// Get caretaker
	caretaker, err := mp.handler.HandleGetCaretaker(ctx, caretakerName)
	if err != nil {
		return err
	}
	
	// Process caretaker
	return processor(caretaker)
}

// ProcessOriginator processes an originator
func (mp *MementoProcessor) ProcessOriginator(ctx context.Context, originator Originator, processor func(Originator) error) error {
	// Validate originator
	if originator == nil {
		return ErrInvalidOriginator
	}
	
	// Process originator
	return processor(originator)
}

// MementoScheduler schedules memento operations
type MementoScheduler struct {
	processor *MementoProcessor
	config    *MementoConfig
	ticker    *time.Ticker
	stop      chan bool
	mutex     sync.RWMutex
}

// NewMementoScheduler creates a new memento scheduler
func NewMementoScheduler(config *MementoConfig) *MementoScheduler {
	return &MementoScheduler{
		processor: NewMementoProcessor(config),
		config:    config,
		stop:      make(chan bool),
	}
}

// Start starts the scheduler
func (ms *MementoScheduler) Start() {
	ms.mutex.Lock()
	defer ms.mutex.Unlock()
	
	ms.ticker = time.NewTicker(ms.config.GetCleanupInterval())
	
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
func (ms *MementoScheduler) Stop() {
	ms.mutex.Lock()
	defer ms.mutex.Unlock()
	
	if ms.ticker != nil {
		ms.ticker.Stop()
	}
	
	close(ms.stop)
}

// processScheduledTasks processes scheduled tasks
func (ms *MementoScheduler) processScheduledTasks() {
	// Process memento cleanup
	ctx := context.Background()
	
	// Get all caretakers
	caretakers := ms.processor.handler.HandleListCaretakers(ctx)
	
	// Process each caretaker
	for _, name := range caretakers {
		ms.processor.handler.HandleCleanup(ctx, name)
	}
}

// MementoMonitor monitors memento operations
type MementoMonitor struct {
	processor *MementoProcessor
	config    *MementoConfig
	metrics   *MementoMetrics
	mutex     sync.RWMutex
}

// NewMementoMonitor creates a new memento monitor
func NewMementoMonitor(config *MementoConfig) *MementoMonitor {
	return &MementoMonitor{
		processor: NewMementoProcessor(config),
		config:    config,
		metrics:   &MementoMetrics{},
	}
}

// MonitorMemento monitors a memento
func (mm *MementoMonitor) MonitorMemento(ctx context.Context, memento Memento) error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	// Update metrics
	mm.metrics.UpdateMetrics(memento)
	
	return nil
}

// GetMetrics returns current metrics
func (mm *MementoMonitor) GetMetrics() *MementoMetrics {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	
	return mm.metrics
}

// ResetMetrics resets the metrics
func (mm *MementoMonitor) ResetMetrics() {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	mm.metrics = &MementoMetrics{}
}

// MementoAuditor audits memento operations
type MementoAuditor struct {
	processor *MementoProcessor
	config    *MementoConfig
	logs      []AuditLog
	mutex     sync.RWMutex
}

// NewMementoAuditor creates a new memento auditor
func NewMementoAuditor(config *MementoConfig) *MementoAuditor {
	return &MementoAuditor{
		processor: NewMementoProcessor(config),
		config:    config,
		logs:      make([]AuditLog, 0),
	}
}

// AuditOperation audits an operation
func (ma *MementoAuditor) AuditOperation(ctx context.Context, userID string, action string, resource string, details map[string]interface{}) {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	
	log := &BaseAuditLog{
		ID:           generateID(),
		UserID:       userID,
		Action:       action,
		Resource:     resource,
		Details:      details,
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
	
	ma.logs = append(ma.logs, log)
}

// GetAuditLogs returns audit logs
func (ma *MementoAuditor) GetAuditLogs() []AuditLog {
	ma.mutex.RLock()
	defer ma.mutex.RUnlock()
	
	return ma.logs
}

// ClearAuditLogs clears audit logs
func (ma *MementoAuditor) ClearAuditLogs() {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	
	ma.logs = make([]AuditLog, 0)
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
