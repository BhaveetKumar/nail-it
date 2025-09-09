package iterator

import (
	"context"
	"sync"
	"time"
)

// IteratorServiceProvider provides iterator services
type IteratorServiceProvider struct {
	service    *IteratorService
	collection *IteratorCollection
	cache      *IteratorCache
	validator  *IteratorValidator
	pool       *IteratorPool
	metrics    *IteratorMetrics
	mutex      sync.RWMutex
}

// NewIteratorServiceProvider creates a new iterator service provider
func NewIteratorServiceProvider(config *IteratorConfig) *IteratorServiceProvider {
	return &IteratorServiceProvider{
		service:    NewIteratorService(config),
		collection: NewIteratorCollection(),
		cache:      NewIteratorCache(config.Cache.TTL),
		validator:  NewIteratorValidator(config),
		pool:       NewIteratorPool(config),
		metrics:    &IteratorMetrics{},
	}
}

// GetService returns the iterator service
func (isp *IteratorServiceProvider) GetService() *IteratorService {
	return isp.service
}

// GetCollection returns the iterator collection
func (isp *IteratorServiceProvider) GetCollection() *IteratorCollection {
	return isp.collection
}

// GetCache returns the iterator cache
func (isp *IteratorServiceProvider) GetCache() *IteratorCache {
	return isp.cache
}

// GetValidator returns the iterator validator
func (isp *IteratorServiceProvider) GetValidator() *IteratorValidator {
	return isp.validator
}

// GetPool returns the iterator pool
func (isp *IteratorServiceProvider) GetPool() *IteratorPool {
	return isp.pool
}

// GetMetrics returns the iterator metrics
func (isp *IteratorServiceProvider) GetMetrics() *IteratorMetrics {
	return isp.metrics
}

// IteratorHandler handles iterator operations
type IteratorHandler struct {
	provider *IteratorServiceProvider
	config   *IteratorConfig
}

// NewIteratorHandler creates a new iterator handler
func NewIteratorHandler(config *IteratorConfig) *IteratorHandler {
	return &IteratorHandler{
		provider: NewIteratorServiceProvider(config),
		config:   config,
	}
}

// HandleCreateIterator handles iterator creation
func (ih *IteratorHandler) HandleCreateIterator(ctx context.Context, name string, iteratorType string, data interface{}) error {
	switch iteratorType {
	case "slice":
		if items, ok := data.([]interface{}); ok {
			return ih.provider.GetService().CreateSliceIterator(name, items)
		}
		return ErrInvalidData
	case "map":
		if items, ok := data.(map[string]interface{}); ok {
			return ih.provider.GetService().CreateMapIterator(name, items)
		}
		return ErrInvalidData
	case "channel":
		if channel, ok := data.(<-chan interface{}); ok {
			return ih.provider.GetService().CreateChannelIterator(name, channel)
		}
		return ErrInvalidData
	case "database":
		if query, ok := data.(map[string]interface{}); ok {
			if results, exists := query["results"]; exists {
				if resultsSlice, ok := results.([]interface{}); ok {
					return ih.provider.GetService().CreateDatabaseIterator(name, query, resultsSlice)
				}
			}
		}
		return ErrInvalidData
	case "file":
		if fileData, ok := data.(map[string]interface{}); ok {
			if filePath, exists := fileData["path"]; exists {
				if lines, exists := fileData["lines"]; exists {
					if linesSlice, ok := lines.([]string); ok {
						return ih.provider.GetService().CreateFileIterator(name, filePath.(string), linesSlice)
					}
				}
			}
		}
		return ErrInvalidData
	default:
		return ErrUnsupportedIteratorType
	}
}

// HandleGetIterator handles iterator retrieval
func (ih *IteratorHandler) HandleGetIterator(ctx context.Context, name string) (Iterator, error) {
	// Check cache first
	if cached, exists := ih.provider.GetCache().Get(name); exists {
		if iterator, ok := cached.(Iterator); ok {
			return iterator, nil
		}
	}
	
	// Get from service
	iterator, err := ih.provider.GetService().GetIterator(name)
	if err != nil {
		return nil, err
	}
	
	// Cache the iterator
	ih.provider.GetCache().Set(name, iterator)
	
	return iterator, nil
}

// HandleRemoveIterator handles iterator removal
func (ih *IteratorHandler) HandleRemoveIterator(ctx context.Context, name string) error {
	// Remove from cache
	ih.provider.GetCache().Delete(name)
	
	// Remove from service
	return ih.provider.GetService().RemoveIterator(name)
}

// HandleListIterators handles iterator listing
func (ih *IteratorHandler) HandleListIterators(ctx context.Context) []string {
	return ih.provider.GetService().ListIterators()
}

// HandleGetIteratorStats handles iterator statistics
func (ih *IteratorHandler) HandleGetIteratorStats(ctx context.Context, name string) (*IteratorStatistics, error) {
	return ih.provider.GetService().GetIteratorStats(name)
}

// HandleCloseAllIterators handles closing all iterators
func (ih *IteratorHandler) HandleCloseAllIterators(ctx context.Context) {
	ih.provider.GetService().CloseAll()
	ih.provider.GetCache().Clear()
}

// IteratorProcessor processes iterator operations
type IteratorProcessor struct {
	handler *IteratorHandler
	config  *IteratorConfig
}

// NewIteratorProcessor creates a new iterator processor
func NewIteratorProcessor(config *IteratorConfig) *IteratorProcessor {
	return &IteratorProcessor{
		handler: NewIteratorHandler(config),
		config:  config,
	}
}

// ProcessIterator processes an iterator
func (ip *IteratorProcessor) ProcessIterator(ctx context.Context, name string, processor func(interface{}) error) error {
	iterator, err := ip.handler.HandleGetIterator(ctx, name)
	if err != nil {
		return err
	}
	
	// Validate iterator
	if err := ip.handler.provider.GetValidator().ValidateIterator(iterator); err != nil {
		return err
	}
	
	// Process items
	for iterator.HasNext() {
		item := iterator.Next()
		if err := processor(item); err != nil {
			return err
		}
	}
	
	return nil
}

// ProcessIteratorWithFilter processes an iterator with filtering
func (ip *IteratorProcessor) ProcessIteratorWithFilter(ctx context.Context, name string, filter Filter, processor func(interface{}) error) error {
	iterator, err := ip.handler.HandleGetIterator(ctx, name)
	if err != nil {
		return err
	}
	
	// Validate iterator
	if err := ip.handler.provider.GetValidator().ValidateIterator(iterator); err != nil {
		return err
	}
	
	// Process items with filter
	for iterator.HasNext() {
		item := iterator.Next()
		if filter.Filter(item) {
			if err := processor(item); err != nil {
				return err
			}
		}
	}
	
	return nil
}

// ProcessIteratorWithTransform processes an iterator with transformation
func (ip *IteratorProcessor) ProcessIteratorWithTransform(ctx context.Context, name string, transformer Transformer, processor func(interface{}) error) error {
	iterator, err := ip.handler.HandleGetIterator(ctx, name)
	if err != nil {
		return err
	}
	
	// Validate iterator
	if err := ip.handler.provider.GetValidator().ValidateIterator(iterator); err != nil {
		return err
	}
	
	// Process items with transformation
	for iterator.HasNext() {
		item := iterator.Next()
		transformed := transformer.Transform(item)
		if err := processor(transformed); err != nil {
			return err
		}
	}
	
	return nil
}

// IteratorScheduler schedules iterator operations
type IteratorScheduler struct {
	processor *IteratorProcessor
	config    *IteratorConfig
	ticker    *time.Ticker
	stop      chan bool
	mutex     sync.RWMutex
}

// NewIteratorScheduler creates a new iterator scheduler
func NewIteratorScheduler(config *IteratorConfig) *IteratorScheduler {
	return &IteratorScheduler{
		processor: NewIteratorProcessor(config),
		config:    config,
		stop:      make(chan bool),
	}
}

// Start starts the scheduler
func (is *IteratorScheduler) Start() {
	is.mutex.Lock()
	defer is.mutex.Unlock()
	
	is.ticker = time.NewTicker(is.config.Monitoring.CollectInterval)
	
	go func() {
		for {
			select {
			case <-is.ticker.C:
				is.processScheduledTasks()
			case <-is.stop:
				return
			}
		}
	}()
}

// Stop stops the scheduler
func (is *IteratorScheduler) Stop() {
	is.mutex.Lock()
	defer is.mutex.Unlock()
	
	if is.ticker != nil {
		is.ticker.Stop()
	}
	
	close(is.stop)
}

// processScheduledTasks processes scheduled tasks
func (is *IteratorScheduler) processScheduledTasks() {
	// Process iterator cleanup
	ctx := context.Background()
	
	// Get all iterators
	iterators := is.processor.handler.HandleListIterators(ctx)
	
	// Process each iterator
	for _, name := range iterators {
		stats, err := is.processor.handler.HandleGetIteratorStats(ctx, name)
		if err != nil {
			continue
		}
		
		// Check if iterator needs cleanup
		if time.Since(stats.LastAccess) > is.config.Timeout {
			is.processor.handler.HandleRemoveIterator(ctx, name)
		}
	}
}

// IteratorMonitor monitors iterator operations
type IteratorMonitor struct {
	processor *IteratorProcessor
	config    *IteratorConfig
	metrics   *IteratorMetrics
	mutex     sync.RWMutex
}

// NewIteratorMonitor creates a new iterator monitor
func NewIteratorMonitor(config *IteratorConfig) *IteratorMonitor {
	return &IteratorMonitor{
		processor: NewIteratorProcessor(config),
		config:    config,
		metrics:   &IteratorMetrics{},
	}
}

// MonitorIterator monitors an iterator
func (im *IteratorMonitor) MonitorIterator(ctx context.Context, name string) error {
	iterator, err := im.processor.handler.HandleGetIterator(ctx, name)
	if err != nil {
		return err
	}
	
	im.mutex.Lock()
	defer im.mutex.Unlock()
	
	// Update metrics
	im.metrics.UpdateMetrics(iterator)
	
	return nil
}

// GetMetrics returns current metrics
func (im *IteratorMonitor) GetMetrics() *IteratorMetrics {
	im.mutex.RLock()
	defer im.mutex.RUnlock()
	
	return im.metrics
}

// ResetMetrics resets the metrics
func (im *IteratorMonitor) ResetMetrics() {
	im.mutex.Lock()
	defer im.mutex.Unlock()
	
	im.metrics = &IteratorMetrics{}
}

// IteratorAuditor audits iterator operations
type IteratorAuditor struct {
	processor *IteratorProcessor
	config    *IteratorConfig
	logs      []AuditLog
	mutex     sync.RWMutex
}

// NewIteratorAuditor creates a new iterator auditor
func NewIteratorAuditor(config *IteratorConfig) *IteratorAuditor {
	return &IteratorAuditor{
		processor: NewIteratorProcessor(config),
		config:    config,
		logs:      make([]AuditLog, 0),
	}
}

// AuditOperation audits an operation
func (ia *IteratorAuditor) AuditOperation(ctx context.Context, userID string, action string, resource string, details map[string]interface{}) {
	ia.mutex.Lock()
	defer ia.mutex.Unlock()
	
	log := AuditLog{
		ID:        generateID(),
		UserID:    userID,
		Action:    action,
		Resource:  resource,
		Details:   details,
		Timestamp: time.Now(),
	}
	
	ia.logs = append(ia.logs, log)
}

// GetAuditLogs returns audit logs
func (ia *IteratorAuditor) GetAuditLogs() []AuditLog {
	ia.mutex.RLock()
	defer ia.mutex.RUnlock()
	
	return ia.logs
}

// ClearAuditLogs clears audit logs
func (ia *IteratorAuditor) ClearAuditLogs() {
	ia.mutex.Lock()
	defer ia.mutex.Unlock()
	
	ia.logs = make([]AuditLog, 0)
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
