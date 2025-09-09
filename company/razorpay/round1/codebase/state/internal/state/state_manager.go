package state

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// StateManagerImpl implements StateManager interface
type StateManagerImpl struct {
	entities map[string]*StateEntity
	mu       sync.RWMutex
}

// NewStateManager creates a new state manager
func NewStateManager() *StateManagerImpl {
	return &StateManagerImpl{
		entities: make(map[string]*StateEntity),
	}
}

// CreateEntity creates a new state entity
func (sm *StateManagerImpl) CreateEntity(ctx context.Context, entity *StateEntity) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if entity == nil {
		return fmt.Errorf("entity cannot be nil")
	}
	
	if entity.ID == "" {
		return fmt.Errorf("entity ID cannot be empty")
	}
	
	if _, exists := sm.entities[entity.ID]; exists {
		return fmt.Errorf("entity with ID %s already exists", entity.ID)
	}
	
	entity.CreatedAt = time.Now()
	entity.UpdatedAt = time.Now()
	entity.StateHistory = make([]*StateHistoryEntry, 0)
	
	sm.entities[entity.ID] = entity
	return nil
}

// GetEntity retrieves a state entity by ID
func (sm *StateManagerImpl) GetEntity(ctx context.Context, entityID string) (*StateEntity, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if entityID == "" {
		return nil, fmt.Errorf("entity ID cannot be empty")
	}
	
	entity, exists := sm.entities[entityID]
	if !exists {
		return nil, fmt.Errorf("entity with ID %s not found", entityID)
	}
	
	// Return a copy to avoid race conditions
	return &StateEntity{
		ID:            entity.ID,
		Type:          entity.Type,
		CurrentState:  entity.CurrentState,
		PreviousState: entity.PreviousState,
		StateHistory:  entity.StateHistory,
		Data:          entity.Data,
		Metadata:      entity.Metadata,
		CreatedAt:     entity.CreatedAt,
		UpdatedAt:     entity.UpdatedAt,
	}, nil
}

// UpdateEntity updates a state entity
func (sm *StateManagerImpl) UpdateEntity(ctx context.Context, entity *StateEntity) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if entity == nil {
		return fmt.Errorf("entity cannot be nil")
	}
	
	if entity.ID == "" {
		return fmt.Errorf("entity ID cannot be empty")
	}
	
	if _, exists := sm.entities[entity.ID]; !exists {
		return fmt.Errorf("entity with ID %s not found", entity.ID)
	}
	
	entity.UpdatedAt = time.Now()
	sm.entities[entity.ID] = entity
	return nil
}

// DeleteEntity deletes a state entity
func (sm *StateManagerImpl) DeleteEntity(ctx context.Context, entityID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if entityID == "" {
		return fmt.Errorf("entity ID cannot be empty")
	}
	
	if _, exists := sm.entities[entityID]; !exists {
		return fmt.Errorf("entity with ID %s not found", entityID)
	}
	
	delete(sm.entities, entityID)
	return nil
}

// GetEntitiesByState retrieves entities by state
func (sm *StateManagerImpl) GetEntitiesByState(ctx context.Context, state string) ([]*StateEntity, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if state == "" {
		return nil, fmt.Errorf("state cannot be empty")
	}
	
	var entities []*StateEntity
	for _, entity := range sm.entities {
		if entity.CurrentState == state {
			// Return a copy to avoid race conditions
			entities = append(entities, &StateEntity{
				ID:            entity.ID,
				Type:          entity.Type,
				CurrentState:  entity.CurrentState,
				PreviousState: entity.PreviousState,
				StateHistory:  entity.StateHistory,
				Data:          entity.Data,
				Metadata:      entity.Metadata,
				CreatedAt:     entity.CreatedAt,
				UpdatedAt:     entity.UpdatedAt,
			})
		}
	}
	
	return entities, nil
}

// GetEntitiesByType retrieves entities by type
func (sm *StateManagerImpl) GetEntitiesByType(ctx context.Context, entityType string) ([]*StateEntity, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if entityType == "" {
		return nil, fmt.Errorf("entity type cannot be empty")
	}
	
	var entities []*StateEntity
	for _, entity := range sm.entities {
		if entity.Type == entityType {
			// Return a copy to avoid race conditions
			entities = append(entities, &StateEntity{
				ID:            entity.ID,
				Type:          entity.Type,
				CurrentState:  entity.CurrentState,
				PreviousState: entity.PreviousState,
				StateHistory:  entity.StateHistory,
				Data:          entity.Data,
				Metadata:      entity.Metadata,
				CreatedAt:     entity.CreatedAt,
				UpdatedAt:     entity.UpdatedAt,
			})
		}
	}
	
	return entities, nil
}

// GetEntityHistory retrieves entity state history
func (sm *StateManagerImpl) GetEntityHistory(ctx context.Context, entityID string) ([]*StateHistoryEntry, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if entityID == "" {
		return nil, fmt.Errorf("entity ID cannot be empty")
	}
	
	entity, exists := sm.entities[entityID]
	if !exists {
		return nil, fmt.Errorf("entity with ID %s not found", entityID)
	}
	
	// Return a copy to avoid race conditions
	history := make([]*StateHistoryEntry, len(entity.StateHistory))
	for i, entry := range entity.StateHistory {
		history[i] = &StateHistoryEntry{
			ID:        entry.ID,
			EntityID:  entry.EntityID,
			State:     entry.State,
			Event:     entry.Event,
			Data:      entry.Data,
			Metadata:  entry.Metadata,
			EnteredAt: entry.EnteredAt,
			ExitedAt:  entry.ExitedAt,
			Duration:  entry.Duration,
		}
	}
	
	return history, nil
}

// GetAllEntities retrieves all entities
func (sm *StateManagerImpl) GetAllEntities(ctx context.Context) ([]*StateEntity, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	var entities []*StateEntity
	for _, entity := range sm.entities {
		// Return a copy to avoid race conditions
		entities = append(entities, &StateEntity{
			ID:            entity.ID,
			Type:          entity.Type,
			CurrentState:  entity.CurrentState,
			PreviousState: entity.PreviousState,
			StateHistory:  entity.StateHistory,
			Data:          entity.Data,
			Metadata:      entity.Metadata,
			CreatedAt:     entity.CreatedAt,
			UpdatedAt:     entity.UpdatedAt,
		})
	}
	
	return entities, nil
}

// GetEntityCount returns the total number of entities
func (sm *StateManagerImpl) GetEntityCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	return len(sm.entities)
}

// GetEntityCountByState returns the number of entities in each state
func (sm *StateManagerImpl) GetEntityCountByState() map[string]int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	counts := make(map[string]int)
	for _, entity := range sm.entities {
		counts[entity.CurrentState]++
	}
	
	return counts
}

// GetEntityCountByType returns the number of entities of each type
func (sm *StateManagerImpl) GetEntityCountByType() map[string]int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	counts := make(map[string]int)
	for _, entity := range sm.entities {
		counts[entity.Type]++
	}
	
	return counts
}

// ClearEntities clears all entities
func (sm *StateManagerImpl) ClearEntities() {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	sm.entities = make(map[string]*StateEntity)
}

// GetEntityStats returns statistics about entities
func (sm *StateManagerImpl) GetEntityStats() map[string]interface{} {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	stats := map[string]interface{}{
		"total_entities": len(sm.entities),
		"by_state":       make(map[string]int),
		"by_type":        make(map[string]int),
	}
	
	byState := make(map[string]int)
	byType := make(map[string]int)
	
	for _, entity := range sm.entities {
		byState[entity.CurrentState]++
		byType[entity.Type]++
	}
	
	stats["by_state"] = byState
	stats["by_type"] = byType
	
	return stats
}
