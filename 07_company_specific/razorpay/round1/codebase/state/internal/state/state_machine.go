package state

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// StateMachineImpl implements StateMachine interface
type StateMachineImpl struct {
	name         string
	states       map[string]State
	initialState string
	finalStates  []string
	mu           sync.RWMutex
}

// NewStateMachine creates a new state machine
func NewStateMachine(name string, initialState string, finalStates []string) *StateMachineImpl {
	return &StateMachineImpl{
		name:         name,
		states:       make(map[string]State),
		initialState: initialState,
		finalStates:  finalStates,
	}
}

// AddState adds a state to the state machine
func (sm *StateMachineImpl) AddState(state State) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	stateName := state.GetStateName()
	if stateName == "" {
		return fmt.Errorf("state name cannot be empty")
	}

	sm.states[stateName] = state
	return nil
}

// RemoveState removes a state from the state machine
func (sm *StateMachineImpl) RemoveState(stateName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if stateName == "" {
		return fmt.Errorf("state name cannot be empty")
	}

	if _, exists := sm.states[stateName]; !exists {
		return fmt.Errorf("state not found: %s", stateName)
	}

	delete(sm.states, stateName)
	return nil
}

// GetState returns a state by name
func (sm *StateMachineImpl) GetState(stateName string) (State, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	state, exists := sm.states[stateName]
	if !exists {
		return nil, fmt.Errorf("state not found: %s", stateName)
	}

	return state, nil
}

// GetAllStates returns all states
func (sm *StateMachineImpl) GetAllStates() map[string]State {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Return a copy to avoid race conditions
	states := make(map[string]State)
	for name, state := range sm.states {
		states[name] = state
	}

	return states
}

// Transition performs a state transition
func (sm *StateMachineImpl) Transition(ctx context.Context, entity *StateEntity, event *StateEvent) (*StateTransition, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Get current state
	currentState, exists := sm.states[entity.CurrentState]
	if !exists {
		return nil, fmt.Errorf("current state not found: %s", entity.CurrentState)
	}

	// Handle event in current state
	transition, err := currentState.Handle(ctx, entity, event)
	if err != nil {
		return nil, fmt.Errorf("failed to handle event in state %s: %w", entity.CurrentState, err)
	}

	// Validate transition
	if transition == nil {
		return nil, fmt.Errorf("no transition returned from state %s", entity.CurrentState)
	}

	// Check if target state exists
	targetState, exists := sm.states[transition.ToState]
	if !exists {
		return nil, fmt.Errorf("target state not found: %s", transition.ToState)
	}

	// Check if transition is allowed
	if !currentState.CanTransitionTo(transition.ToState) {
		return nil, fmt.Errorf("transition from %s to %s is not allowed", entity.CurrentState, transition.ToState)
	}

	// Exit current state
	if err := currentState.Exit(ctx, entity); err != nil {
		return nil, fmt.Errorf("failed to exit state %s: %w", entity.CurrentState, err)
	}

	// Update entity state
	entity.PreviousState = entity.CurrentState
	entity.CurrentState = transition.ToState
	entity.UpdatedAt = time.Now()

	// Add to state history
	historyEntry := &StateHistoryEntry{
		ID:        fmt.Sprintf("history_%s_%d", entity.ID, time.Now().UnixNano()),
		EntityID:  entity.ID,
		State:     entity.CurrentState,
		Event:     event,
		Data:      transition.Data,
		Metadata:  transition.Metadata,
		EnteredAt: time.Now(),
	}
	entity.StateHistory = append(entity.StateHistory, historyEntry)

	// Enter new state
	if err := targetState.Enter(ctx, entity); err != nil {
		// Rollback state change
		entity.CurrentState = entity.PreviousState
		entity.PreviousState = ""
		return nil, fmt.Errorf("failed to enter state %s: %w", transition.ToState, err)
	}

	// Update transition
	transition.ID = fmt.Sprintf("transition_%s_%d", entity.ID, time.Now().UnixNano())
	transition.EntityID = entity.ID
	transition.FromState = entity.PreviousState
	transition.TransitionedAt = time.Now()

	return transition, nil
}

// CanTransition checks if a transition is possible
func (sm *StateMachineImpl) CanTransition(entity *StateEntity, event *StateEvent) (bool, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Get current state
	currentState, exists := sm.states[entity.CurrentState]
	if !exists {
		return false, fmt.Errorf("current state not found: %s", entity.CurrentState)
	}

	// Check if state can handle the event
	transition, err := currentState.Handle(ctx, entity, event)
	if err != nil {
		return false, err
	}

	if transition == nil {
		return false, nil
	}

	// Check if target state exists
	_, exists = sm.states[transition.ToState]
	if !exists {
		return false, fmt.Errorf("target state not found: %s", transition.ToState)
	}

	// Check if transition is allowed
	return currentState.CanTransitionTo(transition.ToState), nil
}

// GetPossibleTransitions returns possible transitions for an entity
func (sm *StateMachineImpl) GetPossibleTransitions(entity *StateEntity) ([]string, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Get current state
	currentState, exists := sm.states[entity.CurrentState]
	if !exists {
		return nil, fmt.Errorf("current state not found: %s", entity.CurrentState)
	}

	return currentState.GetAllowedTransitions(), nil
}

// GetStateMachineName returns the state machine name
func (sm *StateMachineImpl) GetStateMachineName() string {
	return sm.name
}

// GetInitialState returns the initial state
func (sm *StateMachineImpl) GetInitialState() string {
	return sm.initialState
}

// GetFinalStates returns the final states
func (sm *StateMachineImpl) GetFinalStates() []string {
	return sm.finalStates
}

// IsFinalState checks if a state is final
func (sm *StateMachineImpl) IsFinalState(stateName string) bool {
	for _, finalState := range sm.finalStates {
		if finalState == stateName {
			return true
		}
	}
	return false
}

// GetStateInfo returns information about a state
func (sm *StateMachineImpl) GetStateInfo(stateName string) (*StateInfo, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	state, exists := sm.states[stateName]
	if !exists {
		return nil, fmt.Errorf("state not found: %s", stateName)
	}

	return &StateInfo{
		StateName:          state.GetStateName(),
		StateType:          state.GetStateType(),
		Description:        state.GetDescription(),
		IsFinal:            state.IsFinal(),
		AllowedTransitions: state.GetAllowedTransitions(),
		Metadata:           make(map[string]string),
		CreatedAt:          time.Now(),
		UpdatedAt:          time.Now(),
	}, nil
}

// GetStateMachineInfo returns information about the state machine
func (sm *StateMachineImpl) GetStateMachineInfo() map[string]interface{} {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	states := make([]string, 0, len(sm.states))
	for stateName := range sm.states {
		states = append(states, stateName)
	}

	return map[string]interface{}{
		"name":          sm.name,
		"initial_state": sm.initialState,
		"final_states":  sm.finalStates,
		"states":        states,
		"state_count":   len(sm.states),
	}
}
