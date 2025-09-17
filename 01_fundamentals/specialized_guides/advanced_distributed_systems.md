# Advanced Distributed Systems

Comprehensive guide to advanced distributed systems for senior backend engineers.

## 🎯 Distributed Systems Fundamentals

### Consensus Algorithms
```go
// Raft Consensus Algorithm Implementation
type RaftNode struct {
    ID          string
    State       NodeState
    CurrentTerm int
    VotedFor    string
    Log         []LogEntry
    CommitIndex int
    LastApplied int
    NextIndex   map[string]int
    MatchIndex  map[string]int
    mutex       sync.RWMutex
}

type NodeState int

const (
    Follower NodeState = iota
    Candidate
    Leader
)

type LogEntry struct {
    Term    int
    Index   int
    Command interface{}
}

type RaftRequestVote struct {
    Term         int
    CandidateID  string
    LastLogIndex int
    LastLogTerm  int
}

type RaftAppendEntries struct {
    Term         int
    LeaderID     string
    PrevLogIndex int
    PrevLogTerm  int
    Entries      []LogEntry
    LeaderCommit int
}

func (rn *RaftNode) RequestVote(args *RaftRequestVote, reply *RaftRequestVoteReply) error {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    reply.Term = rn.CurrentTerm
    reply.VoteGranted = false
    
    // Check if term is current
    if args.Term < rn.CurrentTerm {
        return nil
    }
    
    // Update term if necessary
    if args.Term > rn.CurrentTerm {
        rn.CurrentTerm = args.Term
        rn.State = Follower
        rn.VotedFor = ""
    }
    
    // Check if already voted for someone else
    if rn.VotedFor != "" && rn.VotedFor != args.CandidateID {
        return nil
    }
    
    // Check if candidate's log is at least as up-to-date
    lastLogIndex, lastLogTerm := rn.getLastLogInfo()
    if args.LastLogTerm < lastLogTerm || 
       (args.LastLogTerm == lastLogTerm && args.LastLogIndex < lastLogIndex) {
        return nil
    }
    
    // Grant vote
    rn.VotedFor = args.CandidateID
    reply.VoteGranted = true
    rn.resetElectionTimeout()
    
    return nil
}

func (rn *RaftNode) AppendEntries(args *RaftAppendEntries, reply *RaftAppendEntriesReply) error {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    reply.Term = rn.CurrentTerm
    reply.Success = false
    
    // Check if term is current
    if args.Term < rn.CurrentTerm {
        return nil
    }
    
    // Update term and become follower
    if args.Term > rn.CurrentTerm {
        rn.CurrentTerm = args.Term
        rn.State = Follower
        rn.VotedFor = ""
    }
    
    // Reset election timeout
    rn.resetElectionTimeout()
    
    // Check if log contains entry at prevLogIndex with matching term
    if args.PrevLogIndex >= 0 {
        if args.PrevLogIndex >= len(rn.Log) || 
           rn.Log[args.PrevLogIndex].Term != args.PrevLogTerm {
            return nil
        }
    }
    
    // Append new entries
    for i, entry := range args.Entries {
        logIndex := args.PrevLogIndex + 1 + i
        if logIndex < len(rn.Log) {
            if rn.Log[logIndex].Term != entry.Term {
                // Remove conflicting entries
                rn.Log = rn.Log[:logIndex]
            }
        }
        if logIndex >= len(rn.Log) {
            rn.Log = append(rn.Log, entry)
        }
    }
    
    // Update commit index
    if args.LeaderCommit > rn.CommitIndex {
        rn.CommitIndex = min(args.LeaderCommit, len(rn.Log)-1)
    }
    
    reply.Success = true
    return nil
}
```

### Distributed Hash Tables (DHT)
```go
// Chord DHT Implementation
type ChordNode struct {
    ID       string
    Predecessor *ChordNode
    Successor   *ChordNode
    FingerTable []*ChordNode
    Data       map[string]interface{}
    mutex      sync.RWMutex
}

type ChordKey struct {
    ID string
    Hash uint32
}

func (cn *ChordNode) FindSuccessor(key ChordKey) *ChordNode {
    if cn.isInRange(key.Hash, cn.ID, cn.Successor.ID) {
        return cn.Successor
    }
    
    // Find closest preceding node
    closest := cn.closestPrecedingNode(key.Hash)
    if closest == cn {
        return cn.Successor
    }
    
    return closest.FindSuccessor(key)
}

func (cn *ChordNode) closestPrecedingNode(hash uint32) *ChordNode {
    cn.mutex.RLock()
    defer cn.mutex.RUnlock()
    
    for i := len(cn.FingerTable) - 1; i >= 0; i-- {
        if cn.FingerTable[i] != nil && 
           cn.isInRange(cn.FingerTable[i].ID, cn.ID, hash) {
            return cn.FingerTable[i]
        }
    }
    
    return cn
}

func (cn *ChordNode) isInRange(hash, start, end uint32) bool {
    if start < end {
        return hash > start && hash <= end
    }
    return hash > start || hash <= end
}

func (cn *ChordNode) Store(key string, value interface{}) error {
    keyHash := cn.hash(key)
    keyObj := ChordKey{ID: key, Hash: keyHash}
    
    successor := cn.FindSuccessor(keyObj)
    return successor.storeLocally(key, value)
}

func (cn *ChordNode) Retrieve(key string) (interface{}, error) {
    keyHash := cn.hash(key)
    keyObj := ChordKey{ID: key, Hash: keyHash}
    
    successor := cn.FindSuccessor(keyObj)
    return successor.retrieveLocally(key)
}

func (cn *ChordNode) storeLocally(key string, value interface{}) error {
    cn.mutex.Lock()
    defer cn.mutex.Unlock()
    
    cn.Data[key] = value
    return nil
}

func (cn *ChordNode) retrieveLocally(key string) (interface{}, error) {
    cn.mutex.RLock()
    defer cn.mutex.RUnlock()
    
    value, exists := cn.Data[key]
    if !exists {
        return nil, errors.New("key not found")
    }
    
    return value, nil
}

func (cn *ChordNode) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}
```

## 🚀 Advanced Distributed Patterns

### Event Sourcing and CQRS
```go
// Event Sourcing Implementation
type EventStore struct {
    events    map[string][]Event
    snapshots map[string]*Snapshot
    mutex     sync.RWMutex
}

type Event struct {
    ID          string
    StreamID    string
    Type        string
    Data        map[string]interface{}
    Metadata    map[string]interface{}
    Version     int
    Timestamp   time.Time
}

type Snapshot struct {
    StreamID    string
    Version     int
    Data        map[string]interface{}
    Timestamp   time.Time
}

type AggregateRoot struct {
    ID      string
    Version int
    Events  []Event
    mutex   sync.RWMutex
}

func (es *EventStore) AppendEvents(streamID string, events []Event, expectedVersion int) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    currentEvents := es.events[streamID]
    if len(currentEvents) != expectedVersion {
        return ErrConcurrencyConflict
    }
    
    for i, event := range events {
        event.Version = expectedVersion + i + 1
        event.Timestamp = time.Now()
        currentEvents = append(currentEvents, event)
    }
    
    es.events[streamID] = currentEvents
    return nil
}

func (es *EventStore) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    events := es.events[streamID]
    if fromVersion >= len(events) {
        return []Event{}, nil
    }
    
    return events[fromVersion:], nil
}

// CQRS Implementation
type CommandHandler interface {
    Handle(command Command) error
}

type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

type CreateUserCommand struct {
    ID       string
    Name     string
    Email    string
    Password string
}

type CreateUserHandler struct {
    eventStore EventStore
    readModel  UserReadModel
}

func (h *CreateUserHandler) Handle(command CreateUserCommand) error {
    // Create user aggregate
    user := &User{
        ID:       command.ID,
        Name:     command.Name,
        Email:    command.Email,
        Password: command.Password,
    }
    
    // Generate events
    events := []Event{
        {
            ID:   generateEventID(),
            Type: "UserCreated",
            Data: map[string]interface{}{
                "id":       user.ID,
                "name":     user.Name,
                "email":    user.Email,
            },
        },
    }
    
    // Store events
    if err := h.eventStore.AppendEvents(user.ID, events, 0); err != nil {
        return err
    }
    
    // Update read model
    return h.readModel.Save(user)
}

type GetUserQuery struct {
    ID string
}

type GetUserHandler struct {
    readModel UserReadModel
}

func (h *GetUserHandler) Handle(query GetUserQuery) (*UserView, error) {
    return h.readModel.GetUser(query.ID)
}
```

### Saga Pattern
```go
// Saga Pattern Implementation
type Saga struct {
    ID        string
    Steps     []SagaStep
    Status    SagaStatus
    Events    []Event
    mutex     sync.RWMutex
}

type SagaStep struct {
    ID           string
    Command      Command
    Compensation Command
    Status       StepStatus
}

type SagaStatus int

const (
    SagaPending SagaStatus = iota
    SagaRunning
    SagaCompleted
    SagaFailed
    SagaCompensated
)

type StepStatus int

const (
    StepPending StepStatus = iota
    StepRunning
    StepCompleted
    StepFailed
    StepCompensated
)

func (s *Saga) Execute() error {
    s.mutex.Lock()
    s.Status = SagaRunning
    s.mutex.Unlock()
    
    for i, step := range s.Steps {
        if err := s.executeStep(step); err != nil {
            // Compensate previous steps
            return s.compensate(i)
        }
    }
    
    s.mutex.Lock()
    s.Status = SagaCompleted
    s.mutex.Unlock()
    
    return nil
}

func (s *Saga) executeStep(step SagaStep) error {
    s.mutex.Lock()
    step.Status = StepRunning
    s.mutex.Unlock()
    
    // Execute command
    if err := s.commandBus.Execute(step.Command); err != nil {
        s.mutex.Lock()
        step.Status = StepFailed
        s.mutex.Unlock()
        return err
    }
    
    s.mutex.Lock()
    step.Status = StepCompleted
    s.mutex.Unlock()
    
    return nil
}

func (s *Saga) compensate(failedStepIndex int) error {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Compensate in reverse order
    for i := failedStepIndex - 1; i >= 0; i-- {
        step := s.Steps[i]
        if step.Status == StepCompleted {
            if err := s.commandBus.Execute(step.Compensation); err != nil {
                return err
            }
            step.Status = StepCompensated
        }
    }
    
    s.Status = SagaFailed
    return nil
}
```

## 🔧 Distributed Data Management

### Distributed Caching
```go
// Distributed Cache Implementation
type DistributedCache struct {
    nodes      []*CacheNode
    hashRing   *ConsistentHash
    replicator *Replicator
    mutex      sync.RWMutex
}

type CacheNode struct {
    ID       string
    Address  string
    Cache    *cache.Cache
    Health   HealthStatus
    mutex    sync.RWMutex
}

type HealthStatus int

const (
    Healthy HealthStatus = iota
    Unhealthy
    Unknown
)

func (dc *DistributedCache) Get(key string) (interface{}, error) {
    // Get primary node
    primaryNode := dc.hashRing.GetNode(key)
    
    // Try primary node first
    if value, err := dc.getFromNode(primaryNode, key); err == nil {
        return value, nil
    }
    
    // Try replica nodes
    replicaNodes := dc.hashRing.GetNodes(key, 3)
    for _, node := range replicaNodes {
        if node.ID != primaryNode.ID {
            if value, err := dc.getFromNode(node, key); err == nil {
                return value, nil
            }
        }
    }
    
    return nil, ErrKeyNotFound
}

func (dc *DistributedCache) Set(key string, value interface{}, ttl time.Duration) error {
    // Get primary node
    primaryNode := dc.hashRing.GetNode(key)
    
    // Set on primary node
    if err := dc.setOnNode(primaryNode, key, value, ttl); err != nil {
        return err
    }
    
    // Replicate to replica nodes
    replicaNodes := dc.hashRing.GetNodes(key, 3)
    for _, node := range replicaNodes {
        if node.ID != primaryNode.ID {
            go dc.setOnNode(node, key, value, ttl)
        }
    }
    
    return nil
}

func (dc *DistributedCache) getFromNode(node *CacheNode, key string) (interface{}, error) {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    if node.Health != Healthy {
        return nil, ErrNodeUnhealthy
    }
    
    return node.Cache.Get(key)
}

func (dc *DistributedCache) setOnNode(node *CacheNode, key string, value interface{}, ttl time.Duration) error {
    node.mutex.Lock()
    defer node.mutex.Unlock()
    
    if node.Health != Healthy {
        return ErrNodeUnhealthy
    }
    
    return node.Cache.Set(key, value, ttl)
}
```

### Distributed Locking
```go
// Distributed Lock Implementation
type DistributedLock struct {
    key        string
    value      string
    ttl        time.Duration
    redis      *redis.Client
    mutex      sync.Mutex
    isLocked   bool
    renewalTicker *time.Ticker
    stopChan   chan struct{}
}

func (dl *DistributedLock) Acquire() (bool, error) {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    
    if dl.isLocked {
        return true, nil
    }
    
    // Try to acquire lock using SET with NX and EX
    result := dl.redis.SetNX(dl.key, dl.value, dl.ttl)
    if result.Err() != nil {
        return false, result.Err()
    }
    
    if result.Val() {
        dl.isLocked = true
        dl.startRenewal()
        return true, nil
    }
    
    return false, nil
}

func (dl *DistributedLock) Release() error {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    
    if !dl.isLocked {
        return nil
    }
    
    // Stop renewal
    if dl.renewalTicker != nil {
        dl.renewalTicker.Stop()
    }
    close(dl.stopChan)
    
    // Release lock using Lua script
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
    `
    
    result := dl.redis.Eval(script, []string{dl.key}, dl.value)
    if result.Err() != nil {
        return result.Err()
    }
    
    dl.isLocked = false
    return nil
}

func (dl *DistributedLock) startRenewal() {
    dl.renewalTicker = time.NewTicker(dl.ttl / 2)
    dl.stopChan = make(chan struct{})
    
    go func() {
        for {
            select {
            case <-dl.renewalTicker.C:
                if !dl.renew() {
                    dl.Release()
                    return
                }
            case <-dl.stopChan:
                return
            }
        }
    }()
}

func (dl *DistributedLock) renew() bool {
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("EXPIRE", KEYS[1], ARGV[2])
        else
            return 0
        end
    `
    
    result := dl.redis.Eval(script, []string{dl.key}, dl.value, int(dl.ttl.Seconds()))
    return result.Err() == nil && result.Val().(int64) == 1
}
```

## 🎯 Best Practices

### Design Principles
1. **Fault Tolerance**: Design for failure and recovery
2. **Consistency**: Choose appropriate consistency models
3. **Scalability**: Design for horizontal scaling
4. **Performance**: Optimize for latency and throughput
5. **Security**: Implement comprehensive security measures

### Common Patterns
1. **Event Sourcing**: Store events instead of state
2. **CQRS**: Separate read and write models
3. **Saga Pattern**: Manage distributed transactions
4. **Circuit Breaker**: Prevent cascading failures
5. **Bulkhead**: Isolate failures

### Monitoring and Observability
1. **Metrics**: Track system performance and health
2. **Logging**: Implement structured logging
3. **Tracing**: Use distributed tracing
4. **Alerting**: Set up proactive alerts
5. **Dashboards**: Create monitoring dashboards

---

**Last Updated**: December 2024  
**Category**: Advanced Distributed Systems  
**Complexity**: Expert Level
