package mediator

import (
	"context"
	"sync"
	"time"
)

// ConcreteMediator implements the Mediator interface
type ConcreteMediator struct {
	colleagues map[string]Colleague
	mutex      sync.RWMutex
	config     *MediatorConfig
}

// NewConcreteMediator creates a new concrete mediator
func NewConcreteMediator(config *MediatorConfig) *ConcreteMediator {
	return &ConcreteMediator{
		colleagues: make(map[string]Colleague),
		config:     config,
	}
}

// RegisterColleague registers a colleague with the mediator
func (cm *ConcreteMediator) RegisterColleague(colleague Colleague) error {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	if len(cm.colleagues) >= cm.config.MaxColleagues {
		return ErrMaxColleaguesReached
	}
	
	colleague.SetMediator(cm)
	cm.colleagues[colleague.GetID()] = colleague
	return nil
}

// UnregisterColleague unregisters a colleague from the mediator
func (cm *ConcreteMediator) UnregisterColleague(colleagueID string) error {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	colleague, exists := cm.colleagues[colleagueID]
	if !exists {
		return ErrColleagueNotFound
	}
	
	colleague.SetActive(false)
	delete(cm.colleagues, colleagueID)
	return nil
}

// SendMessage sends a message from one colleague to another
func (cm *ConcreteMediator) SendMessage(senderID string, recipientID string, message interface{}) error {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	
	sender, exists := cm.colleagues[senderID]
	if !exists {
		return ErrSenderNotFound
	}
	
	recipient, exists := cm.colleagues[recipientID]
	if !exists {
		return ErrRecipientNotFound
	}
	
	if !sender.IsActive() || !recipient.IsActive() {
		return ErrColleagueNotActive
	}
	
	return recipient.ReceiveMessage(senderID, message)
}

// BroadcastMessage broadcasts a message from one colleague to all others
func (cm *ConcreteMediator) BroadcastMessage(senderID string, message interface{}) error {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	
	sender, exists := cm.colleagues[senderID]
	if !exists {
		return ErrSenderNotFound
	}
	
	if !sender.IsActive() {
		return ErrColleagueNotActive
	}
	
	for id, colleague := range cm.colleagues {
		if id != senderID && colleague.IsActive() {
			if err := colleague.ReceiveMessage(senderID, message); err != nil {
				// Log error but continue with other colleagues
				continue
			}
		}
	}
	
	return nil
}

// GetColleagues returns all registered colleagues
func (cm *ConcreteMediator) GetColleagues() []Colleague {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	
	colleagues := make([]Colleague, 0, len(cm.colleagues))
	for _, colleague := range cm.colleagues {
		colleagues = append(colleagues, colleague)
	}
	
	return colleagues
}

// GetColleague returns a specific colleague by ID
func (cm *ConcreteMediator) GetColleague(colleagueID string) (Colleague, error) {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	
	colleague, exists := cm.colleagues[colleagueID]
	if !exists {
		return nil, ErrColleagueNotFound
	}
	
	return colleague, nil
}

// MediatorService provides mediator operations
type MediatorService struct {
	mediator *ConcreteMediator
	config   *MediatorConfig
}

// NewMediatorService creates a new mediator service
func NewMediatorService(config *MediatorConfig) *MediatorService {
	return &MediatorService{
		mediator: NewConcreteMediator(config),
		config:   config,
	}
}

// RegisterColleague registers a colleague
func (ms *MediatorService) RegisterColleague(colleague Colleague) error {
	return ms.mediator.RegisterColleague(colleague)
}

// UnregisterColleague unregisters a colleague
func (ms *MediatorService) UnregisterColleague(colleagueID string) error {
	return ms.mediator.UnregisterColleague(colleagueID)
}

// SendMessage sends a message
func (ms *MediatorService) SendMessage(senderID string, recipientID string, message interface{}) error {
	return ms.mediator.SendMessage(senderID, recipientID, message)
}

// BroadcastMessage broadcasts a message
func (ms *MediatorService) BroadcastMessage(senderID string, message interface{}) error {
	return ms.mediator.BroadcastMessage(senderID, message)
}

// GetColleagues returns all colleagues
func (ms *MediatorService) GetColleagues() []Colleague {
	return ms.mediator.GetColleagues()
}

// GetColleague returns a specific colleague
func (ms *MediatorService) GetColleague(colleagueID string) (Colleague, error) {
	return ms.mediator.GetColleague(colleagueID)
}

// MediatorManager manages multiple mediators
type MediatorManager struct {
	mediators map[string]Mediator
	mutex     sync.RWMutex
	config    *MediatorConfig
}

// NewMediatorManager creates a new mediator manager
func NewMediatorManager(config *MediatorConfig) *MediatorManager {
	return &MediatorManager{
		mediators: make(map[string]Mediator),
		config:    config,
	}
}

// CreateMediator creates a new mediator
func (mm *MediatorManager) CreateMediator(name string) (Mediator, error) {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	if len(mm.mediators) >= mm.config.MaxMediators {
		return nil, ErrMaxMediatorsReached
	}
	
	mediator := NewConcreteMediator(mm.config)
	mm.mediators[name] = mediator
	return mediator, nil
}

// GetMediator retrieves a mediator by name
func (mm *MediatorManager) GetMediator(name string) (Mediator, error) {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	
	mediator, exists := mm.mediators[name]
	if !exists {
		return nil, ErrMediatorNotFound
	}
	
	return mediator, nil
}

// RemoveMediator removes a mediator
func (mm *MediatorManager) RemoveMediator(name string) error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	mediator, exists := mm.mediators[name]
	if !exists {
		return ErrMediatorNotFound
	}
	
	// Unregister all colleagues from the mediator
	colleagues := mediator.GetColleagues()
	for _, colleague := range colleagues {
		mediator.UnregisterColleague(colleague.GetID())
	}
	
	delete(mm.mediators, name)
	return nil
}

// ListMediators returns all mediator names
func (mm *MediatorManager) ListMediators() []string {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	
	names := make([]string, 0, len(mm.mediators))
	for name := range mm.mediators {
		names = append(names, name)
	}
	
	return names
}

// MediatorCache provides caching for mediators
type MediatorCache struct {
	cache map[string]interface{}
	mutex sync.RWMutex
	ttl   time.Duration
}

// NewMediatorCache creates a new mediator cache
func NewMediatorCache(ttl time.Duration) *MediatorCache {
	return &MediatorCache{
		cache: make(map[string]interface{}),
		ttl:   ttl,
	}
}

// Set sets a value in the cache
func (mc *MediatorCache) Set(key string, value interface{}) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	mc.cache[key] = value
}

// Get gets a value from the cache
func (mc *MediatorCache) Get(key string) (interface{}, bool) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	value, exists := mc.cache[key]
	return value, exists
}

// Delete deletes a value from the cache
func (mc *MediatorCache) Delete(key string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	delete(mc.cache, key)
}

// Clear clears the cache
func (mc *MediatorCache) Clear() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	mc.cache = make(map[string]interface{})
}

// Size returns the cache size
func (mc *MediatorCache) Size() int {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	return len(mc.cache)
}

// MediatorMetrics provides metrics for mediators
type MediatorMetrics struct {
	TotalMediators    int64
	ActiveMediators   int64
	TotalColleagues   int64
	ActiveColleagues  int64
	TotalMessages     int64
	SuccessfulMessages int64
	FailedMessages    int64
	AverageLatency    float64
	MaxLatency        float64
	MinLatency        float64
	LastUpdate        time.Time
}

// GetMetrics returns current metrics
func (mm *MediatorMetrics) GetMetrics() *MediatorMetrics {
	return mm
}

// UpdateMetrics updates the metrics
func (mm *MediatorMetrics) UpdateMetrics(mediator Mediator) {
	mm.TotalMediators++
	mm.ActiveMediators++
	
	colleagues := mediator.GetColleagues()
	mm.TotalColleagues += int64(len(colleagues))
	
	for _, colleague := range colleagues {
		if colleague.IsActive() {
			mm.ActiveColleagues++
		}
	}
	
	mm.LastUpdate = time.Now()
}

// MediatorValidator validates mediators
type MediatorValidator struct {
	config *MediatorConfig
}

// NewMediatorValidator creates a new mediator validator
func NewMediatorValidator(config *MediatorConfig) *MediatorValidator {
	return &MediatorValidator{
		config: config,
	}
}

// ValidateMediator validates a mediator
func (mv *MediatorValidator) ValidateMediator(mediator Mediator) error {
	if mediator == nil {
		return ErrInvalidMediator
	}
	
	colleagues := mediator.GetColleagues()
	if len(colleagues) > mv.config.MaxColleagues {
		return ErrTooManyColleagues
	}
	
	return nil
}

// ValidateColleague validates a colleague
func (mv *MediatorValidator) ValidateColleague(colleague Colleague) error {
	if colleague == nil {
		return ErrInvalidColleague
	}
	
	if colleague.GetID() == "" {
		return ErrEmptyColleagueID
	}
	
	if colleague.GetName() == "" {
		return ErrEmptyColleagueName
	}
	
	return nil
}

// MediatorContext provides context for mediator operations
type MediatorContext struct {
	context.Context
	MediatorName string
	StartTime    time.Time
	Timeout      time.Duration
}

// NewMediatorContext creates a new mediator context
func NewMediatorContext(ctx context.Context, mediatorName string, timeout time.Duration) *MediatorContext {
	return &MediatorContext{
		Context:      ctx,
		MediatorName: mediatorName,
		StartTime:    time.Now(),
		Timeout:      timeout,
	}
}

// IsExpired checks if the context is expired
func (mc *MediatorContext) IsExpired() bool {
	return time.Since(mc.StartTime) > mc.Timeout
}

// GetElapsedTime returns the elapsed time
func (mc *MediatorContext) GetElapsedTime() time.Duration {
	return time.Since(mc.StartTime)
}

// MediatorPool manages a pool of mediators
type MediatorPool struct {
	pool   chan Mediator
	mutex  sync.RWMutex
	config *MediatorConfig
}

// NewMediatorPool creates a new mediator pool
func NewMediatorPool(config *MediatorConfig) *MediatorPool {
	return &MediatorPool{
		pool:   make(chan Mediator, config.MaxMediators),
		config: config,
	}
}

// Get gets a mediator from the pool
func (mp *MediatorPool) Get() (Mediator, error) {
	select {
	case mediator := <-mp.pool:
		return mediator, nil
	default:
		return nil, ErrPoolEmpty
	}
}

// Put puts a mediator back into the pool
func (mp *MediatorPool) Put(mediator Mediator) error {
	select {
	case mp.pool <- mediator:
		return nil
	default:
		return ErrPoolFull
	}
}

// Size returns the pool size
func (mp *MediatorPool) Size() int {
	return len(mp.pool)
}

// Close closes the pool
func (mp *MediatorPool) Close() {
	close(mp.pool)
}
