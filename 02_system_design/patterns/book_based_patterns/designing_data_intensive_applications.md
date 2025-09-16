# Designing Data-Intensive Applications - Martin Kleppmann Patterns

## Table of Contents
1. [Introduction](#introduction/)
2. [Reliability Patterns](#reliability-patterns/)
3. [Scalability Patterns](#scalability-patterns/)
4. [Maintainability Patterns](#maintainability-patterns/)
5. [Data Models and Query Languages](#data-models-and-query-languages/)
6. [Storage and Retrieval](#storage-and-retrieval/)
7. [Encoding and Evolution](#encoding-and-evolution/)
8. [Distributed Systems Challenges](#distributed-systems-challenges/)
9. [Consensus and Consistency](#consensus-and-consistency/)
10. [Golang Implementation](#golang-implementation/)

## Introduction

This guide is based on Martin Kleppmann's "Designing Data-Intensive Applications" - a comprehensive resource for building reliable, scalable, and maintainable data systems. The book covers fundamental principles and practical patterns for modern data-intensive applications.

### Core Principles
- **Reliability**: System continues working correctly even when things go wrong
- **Scalability**: System can handle increased load gracefully
- **Maintainability**: System can be easily modified and extended over time

## Reliability Patterns

### Fault Tolerance
```go
// Circuit Breaker Pattern
type CircuitBreaker struct {
    State       CircuitState
    FailureCount int
    Threshold   int
    Timeout     time.Duration
    LastFailure time.Time
    mutex       sync.RWMutex
}

type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    if cb.State == StateOpen {
        if time.Since(cb.LastFailure) > cb.Timeout {
            cb.State = StateHalfOpen
        } else {
            return errors.New("circuit breaker is open")
        }
    }
    
    err := fn()
    if err != nil {
        cb.FailureCount++
        cb.LastFailure = time.Now()
        
        if cb.FailureCount >= cb.Threshold {
            cb.State = StateOpen
        }
        return err
    }
    
    cb.FailureCount = 0
    cb.State = StateClosed
    return nil
}

// Retry with Exponential Backoff
type RetryConfig struct {
    MaxRetries int
    BaseDelay  time.Duration
    MaxDelay   time.Duration
    Multiplier float64
}

func (rc *RetryConfig) Execute(fn func() error) error {
    var lastErr error
    
    for i := 0; i <= rc.MaxRetries; i++ {
        err := fn()
        if err == nil {
            return nil
        }
        
        lastErr = err
        
        if i < rc.MaxRetries {
            delay := time.Duration(float64(rc.BaseDelay) * math.Pow(rc.Multiplier, float64(i)))
            if delay > rc.MaxDelay {
                delay = rc.MaxDelay
            }
            time.Sleep(delay)
        }
    }
    
    return lastErr
}
```

### Redundancy and Replication
```go
// Master-Slave Replication
type MasterSlaveReplication struct {
    Master *sql.DB
    Slaves []*sql.DB
    LoadBalancer *LoadBalancer
}

func (msr *MasterSlaveReplication) Write(query string, args ...interface{}) (*sql.Result, error) {
    // All writes go to master
    return msr.Master.Exec(query, args...)
}

func (msr *MasterSlaveReplication) Read(query string, args ...interface{}) (*sql.Rows, error) {
    // Reads can go to any slave
    slave := msr.LoadBalancer.GetSlave()
    return slave.Query(query, args...)
}

// Multi-Master Replication
type MultiMasterReplication struct {
    Masters []*sql.DB
    ConflictResolver *ConflictResolver
}

func (mmr *MultiMasterReplication) Write(query string, args ...interface{}) (*sql.Result, error) {
    // Write to all masters
    var lastErr error
    for _, master := range mmr.Masters {
        if _, err := master.Exec(query, args...); err != nil {
            lastErr = err
        }
    }
    return nil, lastErr
}
```

## Scalability Patterns

### Load Balancing
```go
// Round Robin Load Balancer
type RoundRobinBalancer struct {
    servers []*Server
    current int
    mutex   sync.Mutex
}

func (rrb *RoundRobinBalancer) GetServer() *Server {
    rrb.mutex.Lock()
    defer rrb.mutex.Unlock()
    
    server := rrb.servers[rrb.current]
    rrb.current = (rrb.current + 1) % len(rrb.servers)
    return server
}

// Weighted Round Robin
type WeightedRoundRobinBalancer struct {
    servers []*WeightedServer
    current int
    mutex   sync.Mutex
}

type WeightedServer struct {
    Server *Server
    Weight int
    Current int
}

func (wrrb *WeightedRoundRobinBalancer) GetServer() *Server {
    wrrb.mutex.Lock()
    defer wrrb.mutex.Unlock()
    
    for {
        server := wrrb.servers[wrrb.current]
        if server.Current < server.Weight {
            server.Current++
            return server.Server
        }
        server.Current = 0
        wrrb.current = (wrrb.current + 1) % len(wrrb.servers)
    }
}

// Least Connections Load Balancer
type LeastConnectionsBalancer struct {
    servers []*Server
    mutex   sync.RWMutex
}

func (lcb *LeastConnectionsBalancer) GetServer() *Server {
    lcb.mutex.RLock()
    defer lcb.mutex.RUnlock()
    
    var minConnections int = int(^uint(0) >> 1)
    var selectedServer *Server
    
    for _, server := range lcb.servers {
        if server.ActiveConnections < minConnections {
            minConnections = server.ActiveConnections
            selectedServer = server
        }
    }
    
    return selectedServer
}
```

### Caching Strategies
```go
// Write-Through Cache
type WriteThroughCache struct {
    cache *redis.Client
    db    *sql.DB
    mutex sync.RWMutex
}

func (wtc *WriteThroughCache) Set(key string, value interface{}) error {
    wtc.mutex.Lock()
    defer wtc.mutex.Unlock()
    
    // Write to database first
    if err := wtc.writeToDB(key, value); err != nil {
        return err
    }
    
    // Then write to cache
    return wtc.cache.Set(key, value, 0).Err()
}

func (wtc *WriteThroughCache) Get(key string) (interface{}, error) {
    wtc.mutex.RLock()
    defer wtc.mutex.RUnlock()
    
    // Try cache first
    if value, err := wtc.cache.Get(key).Result(); err == nil {
        return value, nil
    }
    
    // Cache miss - read from database
    value, err := wtc.readFromDB(key)
    if err != nil {
        return nil, err
    }
    
    // Write to cache for next time
    wtc.cache.Set(key, value, 0)
    
    return value, nil
}

// Write-Behind Cache
type WriteBehindCache struct {
    cache     *redis.Client
    db        *sql.DB
    writeQueue chan WriteOperation
    mutex     sync.RWMutex
}

type WriteOperation struct {
    Key   string
    Value interface{}
    Time  time.Time
}

func (wbc *WriteBehindCache) Set(key string, value interface{}) error {
    wbc.mutex.Lock()
    defer wbc.mutex.Unlock()
    
    // Write to cache immediately
    if err := wbc.cache.Set(key, value, 0).Err(); err != nil {
        return err
    }
    
    // Queue for database write
    select {
    case wbc.writeQueue <- WriteOperation{Key: key, Value: value, Time: time.Now()}:
        return nil
    default:
        return errors.New("write queue full")
    }
}
```

## Maintainability Patterns

### Modular Architecture
```go
// Plugin Architecture
type Plugin interface {
    Name() string
    Version() string
    Initialize(config map[string]interface{}) error
    Process(data interface{}) (interface{}, error)
    Cleanup() error
}

type PluginManager struct {
    plugins map[string]Plugin
    mutex   sync.RWMutex
}

func (pm *PluginManager) Register(plugin Plugin) error {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if _, exists := pm.plugins[plugin.Name()]; exists {
        return fmt.Errorf("plugin %s already registered", plugin.Name())
    }
    
    pm.plugins[plugin.Name()] = plugin
    return nil
}

func (pm *PluginManager) GetPlugin(name string) (Plugin, error) {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    plugin, exists := pm.plugins[name]
    if !exists {
        return nil, fmt.Errorf("plugin %s not found", name)
    }
    
    return plugin, nil
}

// Event-Driven Architecture
type EventBus struct {
    subscribers map[string][]EventHandler
    mutex       sync.RWMutex
}

type EventHandler interface {
    Handle(event Event) error
}

type Event struct {
    Type      string
    Data      interface{}
    Timestamp time.Time
    ID        string
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
}

func (eb *EventBus) Publish(event Event) error {
    eb.mutex.RLock()
    defer eb.mutex.RUnlock()
    
    handlers, exists := eb.subscribers[event.Type]
    if !exists {
        return nil
    }
    
    for _, handler := range handlers {
        go func(h EventHandler) {
            if err := h.Handle(event); err != nil {
                log.Printf("Error handling event %s: %v", event.Type, err)
            }
        }(handler)
    }
    
    return nil
}
```

### Configuration Management
```go
// Configuration Manager
type ConfigManager struct {
    config map[string]interface{}
    mutex  sync.RWMutex
    watchers []ConfigWatcher
}

type ConfigWatcher interface {
    OnConfigChange(key string, oldValue, newValue interface{})
}

func (cm *ConfigManager) Get(key string) (interface{}, error) {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    
    value, exists := cm.config[key]
    if !exists {
        return nil, fmt.Errorf("config key %s not found", key)
    }
    
    return value, nil
}

func (cm *ConfigManager) Set(key string, value interface{}) error {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    
    oldValue, exists := cm.config[key]
    cm.config[key] = value
    
    // Notify watchers
    for _, watcher := range cm.watchers {
        watcher.OnConfigChange(key, oldValue, value)
    }
    
    return nil
}

func (cm *ConfigManager) Watch(watcher ConfigWatcher) {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    
    cm.watchers = append(cm.watchers, watcher)
}
```

## Data Models and Query Languages

### Relational vs Document Models
```go
// Relational Model
type User struct {
    ID       int64  `db:"id" json:"id"`
    Email    string `db:"email" json:"email"`
    Name     string `db:"name" json:"name"`
    CreatedAt time.Time `db:"created_at" json:"created_at"`
}

type Order struct {
    ID     int64 `db:"id" json:"id"`
    UserID int64 `db:"user_id" json:"user_id"`
    Total  float64 `db:"total" json:"total"`
    Items  []OrderItem `json:"items"`
}

type OrderItem struct {
    ID        int64   `db:"id" json:"id"`
    OrderID   int64   `db:"order_id" json:"order_id"`
    ProductID int64   `db:"product_id" json:"product_id"`
    Quantity  int     `db:"quantity" json:"quantity"`
    Price     float64 `db:"price" json:"price"`
}

// Document Model
type UserDocument struct {
    ID       string    `json:"id"`
    Email    string    `json:"email"`
    Name     string    `json:"name"`
    CreatedAt time.Time `json:"created_at"`
    Orders   []OrderDocument `json:"orders"`
}

type OrderDocument struct {
    ID    string `json:"id"`
    Total float64 `json:"total"`
    Items []OrderItemDocument `json:"items"`
}

type OrderItemDocument struct {
    ProductID int64   `json:"product_id"`
    Quantity  int     `json:"quantity"`
    Price     float64 `json:"price"`
}
```

### Graph Data Model
```go
// Graph Database Model
type GraphNode struct {
    ID    string
    Label string
    Properties map[string]interface{}
}

type GraphEdge struct {
    ID     string
    From   string
    To     string
    Label  string
    Properties map[string]interface{}
}

type GraphDatabase struct {
    nodes map[string]*GraphNode
    edges map[string]*GraphEdge
    mutex sync.RWMutex
}

func (gd *GraphDatabase) AddNode(node *GraphNode) {
    gd.mutex.Lock()
    defer gd.mutex.Unlock()
    
    gd.nodes[node.ID] = node
}

func (gd *GraphDatabase) AddEdge(edge *GraphEdge) {
    gd.mutex.Lock()
    defer gd.mutex.Unlock()
    
    gd.edges[edge.ID] = edge
}

func (gd *GraphDatabase) FindNeighbors(nodeID string) []*GraphNode {
    gd.mutex.RLock()
    defer gd.mutex.RUnlock()
    
    var neighbors []*GraphNode
    
    for _, edge := range gd.edges {
        if edge.From == nodeID {
            if neighbor, exists := gd.nodes[edge.To]; exists {
                neighbors = append(neighbors, neighbor)
            }
        } else if edge.To == nodeID {
            if neighbor, exists := gd.nodes[edge.From]; exists {
                neighbors = append(neighbors, neighbor)
            }
        }
    }
    
    return neighbors
}
```

## Storage and Retrieval

### B-Tree Indexes
```go
// B-Tree Node
type BTreeNode struct {
    Keys     []int
    Values   []interface{}
    Children []*BTreeNode
    IsLeaf   bool
    mutex    sync.RWMutex
}

// B-Tree Implementation
type BTree struct {
    Root   *BTreeNode
    Degree int
    mutex  sync.RWMutex
}

func (bt *BTree) Insert(key int, value interface{}) {
    bt.mutex.Lock()
    defer bt.mutex.Unlock()
    
    if bt.Root == nil {
        bt.Root = &BTreeNode{
            Keys:   []int{key},
            Values: []interface{}{value},
            IsLeaf: true,
        }
        return
    }
    
    if len(bt.Root.Keys) == 2*bt.Degree-1 {
        oldRoot := bt.Root
        bt.Root = &BTreeNode{
            Children: []*BTreeNode{oldRoot},
            IsLeaf:   false,
        }
        bt.splitChild(bt.Root, 0)
    }
    
    bt.insertNonFull(bt.Root, key, value)
}

func (bt *BTree) Search(key int) (interface{}, bool) {
    bt.mutex.RLock()
    defer bt.mutex.RUnlock()
    
    return bt.searchNode(bt.Root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) (interface{}, bool) {
    if node == nil {
        return nil, false
    }
    
    i := 0
    for i < len(node.Keys) && key > node.Keys[i] {
        i++
    }
    
    if i < len(node.Keys) && key == node.Keys[i] {
        return node.Values[i], true
    }
    
    if node.IsLeaf {
        return nil, false
    }
    
    return bt.searchNode(node.Children[i], key)
}
```

### LSM-Tree (Log-Structured Merge-Tree)
```go
// LSM-Tree Implementation
type LSMTree struct {
    memTable   *MemTable
    sstables   []*SSTable
    config     *LSMConfig
    mutex      sync.RWMutex
}

type MemTable struct {
    data map[string]interface{}
    size int
    mutex sync.RWMutex
}

type SSTable struct {
    filename string
    level    int
    size     int64
}

type LSMConfig struct {
    MemTableSize    int
    MaxLevels       int
    LevelSizeFactor int
}

func (lsm *LSMTree) Put(key string, value interface{}) error {
    lsm.mutex.Lock()
    defer lsm.mutex.Unlock()
    
    // Add to memtable
    lsm.memTable.Put(key, value)
    
    // Check if memtable is full
    if lsm.memTable.Size() >= lsm.config.MemTableSize {
        return lsm.flushMemTable()
    }
    
    return nil
}

func (lsm *LSMTree) Get(key string) (interface{}, bool) {
    lsm.mutex.RLock()
    defer lsm.mutex.RUnlock()
    
    // Check memtable first
    if value, exists := lsm.memTable.Get(key); exists {
        return value, true
    }
    
    // Check SSTables from newest to oldest
    for i := len(lsm.sstables) - 1; i >= 0; i-- {
        if value, exists := lsm.sstables[i].Get(key); exists {
            return value, true
        }
    }
    
    return nil, false
}

func (lsm *LSMTree) flushMemTable() error {
    // Create new SSTable from memtable
    sstable := lsm.createSSTable(lsm.memTable)
    lsm.sstables = append(lsm.sstables, sstable)
    
    // Clear memtable
    lsm.memTable = &MemTable{
        data: make(map[string]interface{}),
    }
    
    // Trigger compaction if needed
    return lsm.compactIfNeeded()
}
```

## Encoding and Evolution

### Schema Evolution
```go
// Schema Version Manager
type SchemaManager struct {
    versions map[int]Schema
    current  int
    mutex    sync.RWMutex
}

type Schema struct {
    Version int
    Fields  map[string]FieldDefinition
    Rules   []EvolutionRule
}

type FieldDefinition struct {
    Name     string
    Type     string
    Required bool
    Default  interface{}
}

type EvolutionRule struct {
    FromVersion int
    ToVersion   int
    Transform   func(interface{}) interface{}
}

func (sm *SchemaManager) RegisterSchema(version int, schema Schema) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    sm.versions[version] = schema
}

func (sm *SchemaManager) Migrate(data interface{}, fromVersion, toVersion int) (interface{}, error) {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    if fromVersion == toVersion {
        return data, nil
    }
    
    // Find migration path
    path := sm.findMigrationPath(fromVersion, toVersion)
    if path == nil {
        return nil, fmt.Errorf("no migration path from version %d to %d", fromVersion, toVersion)
    }
    
    // Apply transformations
    result := data
    for _, version := range path {
        if rule, exists := sm.findRule(fromVersion, version); exists {
            result = rule.Transform(result)
        }
        fromVersion = version
    }
    
    return result, nil
}
```

### Backward and Forward Compatibility
```go
// Compatible Data Structure
type CompatibleUser struct {
    ID       int64  `json:"id"`
    Email    string `json:"email"`
    Name     string `json:"name"`
    Age      *int   `json:"age,omitempty"`      // Optional field
    Phone    string `json:"phone,omitempty"`    // Optional field
    Metadata map[string]interface{} `json:"metadata,omitempty"` // Extension point
}

// Version-aware deserializer
func (cu *CompatibleUser) UnmarshalJSON(data []byte) error {
    var raw map[string]interface{}
    if err := json.Unmarshal(data, &raw); err != nil {
        return err
    }
    
    // Required fields
    if id, exists := raw["id"]; exists {
        if idFloat, ok := id.(float64); ok {
            cu.ID = int64(idFloat)
        }
    }
    
    if email, exists := raw["email"]; exists {
        if emailStr, ok := email.(string); ok {
            cu.Email = emailStr
        }
    }
    
    if name, exists := raw["name"]; exists {
        if nameStr, ok := name.(string); ok {
            cu.Name = nameStr
        }
    }
    
    // Optional fields
    if age, exists := raw["age"]; exists {
        if ageFloat, ok := age.(float64); ok {
            ageInt := int(ageFloat)
            cu.Age = &ageInt
        }
    }
    
    if phone, exists := raw["phone"]; exists {
        if phoneStr, ok := phone.(string); ok {
            cu.Phone = phoneStr
        }
    }
    
    // Extension point
    cu.Metadata = make(map[string]interface{})
    for key, value := range raw {
        if !isKnownField(key) {
            cu.Metadata[key] = value
        }
    }
    
    return nil
}

func isKnownField(field string) bool {
    knownFields := map[string]bool{
        "id": true, "email": true, "name": true, "age": true, "phone": true,
    }
    return knownFields[field]
}
```

## Distributed Systems Challenges

### Network Partitions
```go
// Network Partition Handler
type PartitionHandler struct {
    nodes       map[string]*Node
    partitions  map[string][]string
    mutex       sync.RWMutex
}

type Node struct {
    ID       string
    Address  string
    Status   NodeStatus
    LastSeen time.Time
}

type NodeStatus int

const (
    StatusHealthy NodeStatus = iota
    StatusUnhealthy
    StatusPartitioned
)

func (ph *PartitionHandler) HandlePartition(partitionID string, nodes []string) {
    ph.mutex.Lock()
    defer ph.mutex.Unlock()
    
    ph.partitions[partitionID] = nodes
    
    // Update node status
    for _, nodeID := range nodes {
        if node, exists := ph.nodes[nodeID]; exists {
            node.Status = StatusPartitioned
        }
    }
}

func (ph *PartitionHandler) GetAvailableNodes() []*Node {
    ph.mutex.RLock()
    defer ph.mutex.RUnlock()
    
    var available []*Node
    for _, node := range ph.nodes {
        if node.Status == StatusHealthy {
            available = append(available, node)
        }
    }
    
    return available
}
```

### Clock Synchronization
```go
// Logical Clock
type LogicalClock struct {
    counter int64
    mutex   sync.Mutex
}

func (lc *LogicalClock) Tick() int64 {
    lc.mutex.Lock()
    defer lc.mutex.Unlock()
    
    lc.counter++
    return lc.counter
}

func (lc *LogicalClock) Update(otherClock int64) {
    lc.mutex.Lock()
    defer lc.mutex.Unlock()
    
    if otherClock > lc.counter {
        lc.counter = otherClock
    }
    lc.counter++
}

// Vector Clock
type VectorClock struct {
    clocks map[string]int64
    nodeID string
    mutex  sync.Mutex
}

func (vc *VectorClock) Tick() {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    
    vc.clocks[vc.nodeID]++
}

func (vc *VectorClock) Update(otherClock map[string]int64) {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    
    for nodeID, clock := range otherClock {
        if clock > vc.clocks[nodeID] {
            vc.clocks[nodeID] = clock
        }
    }
    vc.clocks[vc.nodeID]++
}

func (vc *VectorClock) HappensBefore(other *VectorClock) bool {
    vc.mutex.Lock()
    other.mutex.Lock()
    defer vc.mutex.Unlock()
    defer other.mutex.Unlock()
    
    for nodeID, clock := range vc.clocks {
        if clock > other.clocks[nodeID] {
            return false
        }
    }
    
    return true
}
```

## Consensus and Consistency

### Raft Consensus Algorithm
```go
// Raft Node
type RaftNode struct {
    ID          string
    State       RaftState
    CurrentTerm int
    VotedFor    string
    Log         []LogEntry
    CommitIndex int
    LastApplied int
    mutex       sync.RWMutex
}

type RaftState int

const (
    StateFollower RaftState = iota
    StateCandidate
    StateLeader
)

type LogEntry struct {
    Term    int
    Index   int
    Command interface{}
}

// Raft Consensus Implementation
type RaftConsensus struct {
    nodes    map[string]*RaftNode
    leader   string
    mutex    sync.RWMutex
}

func (rc *RaftConsensus) RequestVote(candidateID string, term int, lastLogIndex int, lastLogTerm int) bool {
    rc.mutex.Lock()
    defer rc.mutex.Unlock()
    
    node, exists := rc.nodes[candidateID]
    if !exists {
        return false
    }
    
    // Check if term is current
    if term < node.CurrentTerm {
        return false
    }
    
    // Check if already voted for someone else
    if node.VotedFor != "" && node.VotedFor != candidateID {
        return false
    }
    
    // Check if candidate's log is up to date
    if lastLogTerm < node.Log[len(node.Log)-1].Term {
        return false
    }
    
    if lastLogTerm == node.Log[len(node.Log)-1].Term && lastLogIndex < len(node.Log)-1 {
        return false
    }
    
    // Grant vote
    node.VotedFor = candidateID
    node.CurrentTerm = term
    return true
}

func (rc *RaftConsensus) AppendEntries(leaderID string, term int, prevLogIndex int, prevLogTerm int, entries []LogEntry, leaderCommit int) bool {
    rc.mutex.Lock()
    defer rc.mutex.Unlock()
    
    node, exists := rc.nodes[leaderID]
    if !exists {
        return false
    }
    
    // Check if term is current
    if term < node.CurrentTerm {
        return false
    }
    
    // Check if log entry at prevLogIndex matches
    if prevLogIndex >= 0 && (prevLogIndex >= len(node.Log) || node.Log[prevLogIndex].Term != prevLogTerm) {
        return false
    }
    
    // Append new entries
    for i, entry := range entries {
        if prevLogIndex+1+i < len(node.Log) {
            if node.Log[prevLogIndex+1+i].Term != entry.Term {
                // Remove conflicting entries
                node.Log = node.Log[:prevLogIndex+1+i]
            }
        }
        node.Log = append(node.Log, entry)
    }
    
    // Update commit index
    if leaderCommit > node.CommitIndex {
        node.CommitIndex = min(leaderCommit, len(node.Log)-1)
    }
    
    return true
}
```

## Golang Implementation

### Event Sourcing
```go
// Event Store
type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

type Event struct {
    ID        string
    StreamID  string
    Type      string
    Data      []byte
    Version   int
    Timestamp time.Time
}

func (es *EventStore) Append(streamID string, events []Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    // Check for concurrency conflicts
    if len(es.events) > 0 {
        lastEvent := es.events[len(es.events)-1]
        if lastEvent.StreamID == streamID && lastEvent.Version >= events[0].Version {
            return errors.New("concurrency conflict")
        }
    }
    
    es.events = append(es.events, events...)
    return nil
}

func (es *EventStore) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    var result []Event
    for _, event := range es.events {
        if event.StreamID == streamID && event.Version >= fromVersion {
            result = append(result, event)
        }
    }
    
    return result, nil
}

// Aggregate Root
type UserAggregate struct {
    ID      string
    Email   string
    Name    string
    Version int
    Events  []Event
    mutex   sync.Mutex
}

func (ua *UserAggregate) CreateUser(email, name string) error {
    ua.mutex.Lock()
    defer ua.mutex.Unlock()
    
    event := Event{
        ID:        generateID(),
        StreamID:  ua.ID,
        Type:      "UserCreated",
        Data:      []byte(fmt.Sprintf(`{"email":"%s","name":"%s"}`, email, name)),
        Version:   ua.Version + 1,
        Timestamp: time.Now(),
    }
    
    ua.Events = append(ua.Events, event)
    ua.Email = email
    ua.Name = name
    ua.Version++
    
    return nil
}

func (ua *UserAggregate) UpdateEmail(newEmail string) error {
    ua.mutex.Lock()
    defer ua.mutex.Unlock()
    
    event := Event{
        ID:        generateID(),
        StreamID:  ua.ID,
        Type:      "UserEmailUpdated",
        Data:      []byte(fmt.Sprintf(`{"email":"%s"}`, newEmail)),
        Version:   ua.Version + 1,
        Timestamp: time.Now(),
    }
    
    ua.Events = append(ua.Events, event)
    ua.Email = newEmail
    ua.Version++
    
    return nil
}
```

## Conclusion

Designing data-intensive applications requires careful consideration of reliability, scalability, and maintainability. Key patterns from Martin Kleppmann's work:

1. **Fault Tolerance**: Implement circuit breakers, retries, and redundancy
2. **Scalability**: Use load balancing, caching, and partitioning
3. **Maintainability**: Design modular, configurable systems
4. **Data Models**: Choose appropriate models for your use case
5. **Storage**: Select the right storage engine for your workload
6. **Evolution**: Plan for schema and data evolution
7. **Distributed Systems**: Handle network partitions and clock synchronization
8. **Consensus**: Use appropriate consensus algorithms for consistency

By following these patterns and principles, you can build robust, scalable, and maintainable data-intensive applications.
