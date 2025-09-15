# 🏗️ Google System Design Advanced Patterns

> **Advanced system design patterns and architectures for Google-level interviews**

## 📋 Table of Contents

1. [🔗 Distributed Consensus & Consistency](#-distributed-consensus--consistency)
2. [🌐 Event-Driven Architecture](#-event-driven-architecture)
3. [📊 CQRS & Event Sourcing](#-cqrs--event-sourcing)
4. [⚡ Microservices Patterns](#-microservices-patterns)
5. [🎯 Advanced Caching Strategies](#-advanced-caching-strategies)
6. [🔍 Monitoring & Observability](#-monitoring--observability)
7. [🔐 Security & Authentication](#-security--authentication)
8. [📈 Scalability Patterns](#-scalability-patterns)

---

## 🔗 Distributed Consensus & Consistency

### **1. Raft Consensus Algorithm**

**Problem**: How do distributed systems achieve consensus when nodes can fail?

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type RaftNode struct {
    ID          int
    State       NodeState
    CurrentTerm int
    VotedFor    int
    Log         []LogEntry
    CommitIndex int
    LastApplied int
    NextIndex   map[int]int
    MatchIndex  map[int]int
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
    Command interface{}
}

type RaftCluster struct {
    nodes map[int]*RaftNode
    mutex sync.RWMutex
}

func NewRaftCluster() *RaftCluster {
    return &RaftCluster{
        nodes: make(map[int]*RaftNode),
    }
}

func (rc *RaftCluster) AddNode(id int) *RaftNode {
    rc.mutex.Lock()
    defer rc.mutex.Unlock()
    
    node := &RaftNode{
        ID:        id,
        State:     Follower,
        Log:       make([]LogEntry, 0),
        NextIndex: make(map[int]int),
        MatchIndex: make(map[int]int),
    }
    
    rc.nodes[id] = node
    return node
}

func (rc *RaftCluster) StartElection(nodeID int) {
    rc.mutex.RLock()
    node := rc.nodes[nodeID]
    rc.mutex.RUnlock()
    
    if node == nil {
        return
    }
    
    node.mutex.Lock()
    node.State = Candidate
    node.CurrentTerm++
    node.VotedFor = nodeID
    votes := 1
    node.mutex.Unlock()
    
    fmt.Printf("Node %d starting election for term %d\n", nodeID, node.CurrentTerm)
    
    for id, otherNode := range rc.nodes {
        if id != nodeID {
            go rc.requestVote(nodeID, id, otherNode, &votes)
        }
    }
    
    time.Sleep(100 * time.Millisecond)
    node.mutex.RLock()
    if votes > len(rc.nodes)/2 && node.State == Candidate {
        node.mutex.RUnlock()
        rc.becomeLeader(nodeID)
    } else {
        node.mutex.RUnlock()
    }
}

func (rc *RaftCluster) requestVote(candidateID, voterID int, voter *RaftNode, votes *int) {
    voter.mutex.Lock()
    defer voter.mutex.Unlock()
    
    if voter.VotedFor == -1 || voter.VotedFor == candidateID {
        if voter.CurrentTerm <= rc.nodes[candidateID].CurrentTerm {
            voter.VotedFor = candidateID
            voter.CurrentTerm = rc.nodes[candidateID].CurrentTerm
            *votes++
            fmt.Printf("Node %d voted for node %d\n", voterID, candidateID)
        }
    }
}

func (rc *RaftCluster) becomeLeader(nodeID int) {
    rc.mutex.RLock()
    node := rc.nodes[nodeID]
    rc.mutex.RUnlock()
    
    node.mutex.Lock()
    node.State = Leader
    node.mutex.Unlock()
    
    fmt.Printf("Node %d became leader for term %d\n", nodeID, node.CurrentTerm)
    
    go rc.sendHeartbeats(nodeID)
}

func (rc *RaftCluster) sendHeartbeats(leaderID int) {
    ticker := time.NewTicker(50 * time.Millisecond)
    defer ticker.Stop()
    
    for range ticker.C {
        rc.mutex.RLock()
        leader := rc.nodes[leaderID]
        rc.mutex.RUnlock()
        
        if leader == nil || leader.State != Leader {
            break
        }
        
        for id, follower := range rc.nodes {
            if id != leaderID {
                go rc.sendHeartbeat(leaderID, id, follower)
            }
        }
    }
}

func (rc *RaftCluster) sendHeartbeat(leaderID, followerID int, follower *RaftNode) {
    follower.mutex.Lock()
    defer follower.mutex.Unlock()
    
    if follower.CurrentTerm < rc.nodes[leaderID].CurrentTerm {
        follower.CurrentTerm = rc.nodes[leaderID].CurrentTerm
        follower.State = Follower
        follower.VotedFor = -1
    }
}

func main() {
    cluster := NewRaftCluster()
    
    for i := 1; i <= 5; i++ {
        cluster.AddNode(i)
    }
    
    cluster.StartElection(1)
    time.Sleep(200 * time.Millisecond)
    
    fmt.Println("Raft consensus demonstration completed")
}
```

### **2. CAP Theorem Implementation**

**Problem**: Understanding trade-offs between Consistency, Availability, and Partition Tolerance.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CAPSystem struct {
    nodes       map[string]*Node
    consistency bool
    availability bool
    partitionTolerance bool
    mutex       sync.RWMutex
}

type Node struct {
    ID       string
    Data     map[string]string
    Online   bool
    mutex    sync.RWMutex
}

func NewCAPSystem(consistency, availability, partitionTolerance bool) *CAPSystem {
    return &CAPSystem{
        nodes:              make(map[string]*Node),
        consistency:        consistency,
        availability:       availability,
        partitionTolerance: partitionTolerance,
    }
}

func (cs *CAPSystem) AddNode(id string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    cs.nodes[id] = &Node{
        ID:   id,
        Data: make(map[string]string),
        Online: true,
    }
}

func (cs *CAPSystem) Write(key, value string) error {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    if cs.consistency {
        return cs.writeWithConsistency(key, value)
    } else {
        return cs.writeWithAvailability(key, value)
    }
}

func (cs *CAPSystem) writeWithConsistency(key, value string) error {
    // Write to all nodes for consistency
    for _, node := range cs.nodes {
        if !node.Online {
            return fmt.Errorf("node %s is offline, cannot maintain consistency", node.ID)
        }
        
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
    }
    
    return nil
}

func (cs *CAPSystem) writeWithAvailability(key, value string) error {
    // Write to available nodes only
    written := false
    for _, node := range cs.nodes {
        if node.Online {
            node.mutex.Lock()
            node.Data[key] = value
            node.mutex.Unlock()
            written = true
        }
    }
    
    if !written {
        return fmt.Errorf("no nodes available for write")
    }
    
    return nil
}

func (cs *CAPSystem) Read(key string) (string, error) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    if cs.consistency {
        return cs.readWithConsistency(key)
    } else {
        return cs.readWithAvailability(key)
    }
}

func (cs *CAPSystem) readWithConsistency(key string) (string, error) {
    // Read from all nodes and ensure consistency
    var values []string
    for _, node := range cs.nodes {
        if node.Online {
            node.mutex.RLock()
            if value, exists := node.Data[key]; exists {
                values = append(values, value)
            }
            node.mutex.RUnlock()
        }
    }
    
    if len(values) == 0 {
        return "", fmt.Errorf("key not found")
    }
    
    // Check if all values are the same
    for i := 1; i < len(values); i++ {
        if values[i] != values[0] {
            return "", fmt.Errorf("inconsistent data across nodes")
        }
    }
    
    return values[0], nil
}

func (cs *CAPSystem) readWithAvailability(key string) (string, error) {
    // Read from any available node
    for _, node := range cs.nodes {
        if node.Online {
            node.mutex.RLock()
            if value, exists := node.Data[key]; exists {
                node.mutex.RUnlock()
                return value, nil
            }
            node.mutex.RUnlock()
        }
    }
    
    return "", fmt.Errorf("key not found")
}

func (cs *CAPSystem) SimulatePartition(nodeID string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    if node, exists := cs.nodes[nodeID]; exists {
        node.Online = false
        fmt.Printf("Node %s went offline (partition)\n", nodeID)
    }
}

func main() {
    // CP System (Consistency + Partition Tolerance)
    cpSystem := NewCAPSystem(true, false, true)
    cpSystem.AddNode("node1")
    cpSystem.AddNode("node2")
    cpSystem.AddNode("node3")
    
    fmt.Println("CP System (Consistency + Partition Tolerance):")
    cpSystem.Write("key1", "value1")
    if value, err := cpSystem.Read("key1"); err == nil {
        fmt.Printf("Read: %s\n", value)
    }
    
    cpSystem.SimulatePartition("node1")
    if err := cpSystem.Write("key2", "value2"); err != nil {
        fmt.Printf("Write failed: %v\n", err)
    }
    
    // AP System (Availability + Partition Tolerance)
    apSystem := NewCAPSystem(false, true, true)
    apSystem.AddNode("node1")
    apSystem.AddNode("node2")
    apSystem.AddNode("node3")
    
    fmt.Println("\nAP System (Availability + Partition Tolerance):")
    apSystem.Write("key1", "value1")
    if value, err := apSystem.Read("key1"); err == nil {
        fmt.Printf("Read: %s\n", value)
    }
    
    apSystem.SimulatePartition("node1")
    if err := apSystem.Write("key2", "value2"); err == nil {
        fmt.Printf("Write failed: %v\n", err)
    } else {
        fmt.Println("Write succeeded despite partition")
    }
}
```

---

## 🌐 Event-Driven Architecture

### **3. Event Sourcing Pattern**

**Problem**: How to maintain a complete audit trail of all changes?

```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type Event struct {
    ID        string      `json:"id"`
    Type      string      `json:"type"`
    Data      interface{} `json:"data"`
    Timestamp time.Time   `json:"timestamp"`
    Version   int         `json:"version"`
}

type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

func NewEventStore() *EventStore {
    return &EventStore{
        events: make([]Event, 0),
    }
}

func (es *EventStore) AppendEvent(event Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    event.Version = len(es.events) + 1
    es.events = append(es.events, event)
    
    return nil
}

func (es *EventStore) GetEvents(aggregateID string) []Event {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    var result []Event
    for _, event := range es.events {
        if event.ID == aggregateID {
            result = append(result, event)
        }
    }
    
    return result
}

type BankAccount struct {
    ID      string
    Balance int
    Version int
    mutex   sync.RWMutex
}

func NewBankAccount(id string) *BankAccount {
    return &BankAccount{
        ID:      id,
        Balance: 0,
        Version: 0,
    }
}

func (ba *BankAccount) ApplyEvent(event Event) {
    ba.mutex.Lock()
    defer ba.mutex.Unlock()
    
    switch event.Type {
    case "AccountOpened":
        data := event.Data.(map[string]interface{})
        ba.Balance = int(data["initialBalance"].(float64))
    case "MoneyDeposited":
        data := event.Data.(map[string]interface{})
        ba.Balance += int(data["amount"].(float64))
    case "MoneyWithdrawn":
        data := event.Data.(map[string]interface{})
        ba.Balance -= int(data["amount"].(float64))
    }
    
    ba.Version = event.Version
}

func (ba *BankAccount) Deposit(amount int, eventStore *EventStore) error {
    if amount <= 0 {
        return fmt.Errorf("deposit amount must be positive")
    }
    
    event := Event{
        ID:        ba.ID,
        Type:      "MoneyDeposited",
        Data:      map[string]interface{}{"amount": amount},
        Timestamp: time.Now(),
    }
    
    if err := eventStore.AppendEvent(event); err != nil {
        return err
    }
    
    ba.ApplyEvent(event)
    return nil
}

func (ba *BankAccount) Withdraw(amount int, eventStore *EventStore) error {
    if amount <= 0 {
        return fmt.Errorf("withdrawal amount must be positive")
    }
    
    ba.mutex.RLock()
    if ba.Balance < amount {
        ba.mutex.RUnlock()
        return fmt.Errorf("insufficient funds")
    }
    ba.mutex.RUnlock()
    
    event := Event{
        ID:        ba.ID,
        Type:      "MoneyWithdrawn",
        Data:      map[string]interface{}{"amount": amount},
        Timestamp: time.Now(),
    }
    
    if err := eventStore.AppendEvent(event); err != nil {
        return err
    }
    
    ba.ApplyEvent(event)
    return nil
}

func (ba *BankAccount) ReplayEvents(events []Event) {
    for _, event := range events {
        ba.ApplyEvent(event)
    }
}

func main() {
    eventStore := NewEventStore()
    
    account := NewBankAccount("account-1")
    
    openEvent := Event{
        ID:        account.ID,
        Type:      "AccountOpened",
        Data:      map[string]interface{}{"initialBalance": 1000},
        Timestamp: time.Now(),
    }
    eventStore.AppendEvent(openEvent)
    account.ApplyEvent(openEvent)
    
    account.Deposit(500, eventStore)
    account.Withdraw(200, eventStore)
    account.Deposit(100, eventStore)
    
    fmt.Printf("Final balance: %d\n", account.Balance)
    
    newAccount := NewBankAccount("account-1")
    events := eventStore.GetEvents("account-1")
    newAccount.ReplayEvents(events)
    
    fmt.Printf("Rebuilt balance: %d\n", newAccount.Balance)
    
    fmt.Println("Event History:")
    for _, event := range events {
        data, _ := json.Marshal(event.Data)
        fmt.Printf("Version %d: %s - %s\n", event.Version, event.Type, string(data))
    }
}
```

### **4. CQRS (Command Query Responsibility Segregation)**

**Problem**: How to optimize read and write operations separately?

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Command interface {
    Execute() error
}

type Query interface {
    Execute() (interface{}, error)
}

type CommandBus struct {
    handlers map[string]CommandHandler
    mutex    sync.RWMutex
}

type CommandHandler interface {
    Handle(command Command) error
}

type QueryBus struct {
    handlers map[string]QueryHandler
    mutex    sync.RWMutex
}

type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

func NewCommandBus() *CommandBus {
    return &CommandBus{
        handlers: make(map[string]CommandHandler),
    }
}

func NewQueryBus() *QueryBus {
    return &QueryBus{
        handlers: make(map[string]QueryHandler),
    }
}

func (cb *CommandBus) RegisterCommandHandler(commandType string, handler CommandHandler) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    cb.handlers[commandType] = handler
}

func (qb *QueryBus) RegisterQueryHandler(queryType string, handler QueryHandler) {
    qb.mutex.Lock()
    defer qb.mutex.Unlock()
    qb.handlers[queryType] = handler
}

func (cb *CommandBus) ExecuteCommand(command Command) error {
    cb.mutex.RLock()
    handler, exists := cb.handlers[cb.getCommandType(command)]
    cb.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("no handler for command type")
    }
    
    return handler.Handle(command)
}

func (qb *QueryBus) ExecuteQuery(query Query) (interface{}, error) {
    qb.mutex.RLock()
    handler, exists := qb.handlers[qb.getQueryType(query)]
    qb.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("no handler for query type")
    }
    
    return handler.Handle(query)
}

func (cb *CommandBus) getCommandType(command Command) string {
    switch command.(type) {
    case *CreateUserCommand:
        return "CreateUser"
    case *UpdateUserCommand:
        return "UpdateUser"
    default:
        return "Unknown"
    }
}

func (qb *QueryBus) getQueryType(query Query) string {
    switch query.(type) {
    case *GetUserQuery:
        return "GetUser"
    case *GetUsersQuery:
        return "GetUsers"
    default:
        return "Unknown"
    }
}

type User struct {
    ID        string
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

type CreateUserCommand struct {
    Name  string
    Email string
}

type UpdateUserCommand struct {
    ID    string
    Name  string
    Email string
}

type GetUserQuery struct {
    ID string
}

type GetUsersQuery struct{}

type UserCommandHandler struct {
    writeDB map[string]*User
    mutex   sync.RWMutex
}

func NewUserCommandHandler() *UserCommandHandler {
    return &UserCommandHandler{
        writeDB: make(map[string]*User),
    }
}

func (h *UserCommandHandler) Handle(command Command) error {
    switch cmd := command.(type) {
    case *CreateUserCommand:
        return h.handleCreateUser(cmd)
    case *UpdateUserCommand:
        return h.handleUpdateUser(cmd)
    default:
        return fmt.Errorf("unknown command type")
    }
}

func (h *UserCommandHandler) handleCreateUser(cmd *CreateUserCommand) error {
    h.mutex.Lock()
    defer h.mutex.Unlock()
    
    user := &User{
        ID:        fmt.Sprintf("user-%d", time.Now().UnixNano()),
        Name:      cmd.Name,
        Email:     cmd.Email,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    h.writeDB[user.ID] = user
    fmt.Printf("Created user: %s\n", user.ID)
    
    return nil
}

func (h *UserCommandHandler) handleUpdateUser(cmd *UpdateUserCommand) error {
    h.mutex.Lock()
    defer h.mutex.Unlock()
    
    user, exists := h.writeDB[cmd.ID]
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    user.Name = cmd.Name
    user.Email = cmd.Email
    user.UpdatedAt = time.Now()
    
    fmt.Printf("Updated user: %s\n", user.ID)
    return nil
}

type UserQueryHandler struct {
    readDB map[string]*User
    mutex  sync.RWMutex
}

func NewUserQueryHandler() *UserQueryHandler {
    return &UserQueryHandler{
        readDB: make(map[string]*User),
    }
}

func (h *UserQueryHandler) Handle(query Query) (interface{}, error) {
    switch q := query.(type) {
    case *GetUserQuery:
        return h.handleGetUser(q)
    case *GetUsersQuery:
        return h.handleGetUsers(q)
    default:
        return nil, fmt.Errorf("unknown query type")
    }
}

func (h *UserQueryHandler) handleGetUser(query *GetUserQuery) (*User, error) {
    h.mutex.RLock()
    defer h.mutex.RUnlock()
    
    user, exists := h.readDB[query.ID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    
    return user, nil
}

func (h *UserQueryHandler) handleGetUsers(query *GetUsersQuery) ([]*User, error) {
    h.mutex.RLock()
    defer h.mutex.RUnlock()
    
    users := make([]*User, 0, len(h.readDB))
    for _, user := range h.readDB {
        users = append(users, user)
    }
    
    return users, nil
}

func (h *UserQueryHandler) SyncReadDB(writeDB map[string]*User) {
    h.mutex.Lock()
    defer h.mutex.Unlock()
    
    h.readDB = make(map[string]*User)
    for id, user := range writeDB {
        h.readDB[id] = user
    }
}

func main() {
    commandBus := NewCommandBus()
    queryBus := NewQueryBus()
    
    commandHandler := NewUserCommandHandler()
    queryHandler := NewUserQueryHandler()
    
    commandBus.RegisterCommandHandler("CreateUser", commandHandler)
    commandBus.RegisterCommandHandler("UpdateUser", commandHandler)
    queryBus.RegisterQueryHandler("GetUser", queryHandler)
    queryBus.RegisterQueryHandler("GetUsers", queryHandler)
    
    createCmd := &CreateUserCommand{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    if err := commandBus.ExecuteCommand(createCmd); err != nil {
        fmt.Printf("Error creating user: %v\n", err)
    }
    
    queryHandler.SyncReadDB(commandHandler.writeDB)
    
    getUsersQuery := &GetUsersQuery{}
    users, err := queryBus.ExecuteQuery(getUsersQuery)
    if err != nil {
        fmt.Printf("Error getting users: %v\n", err)
    } else {
        fmt.Printf("Found %d users\n", len(users.([]*User)))
    }
    
    fmt.Println("CQRS demonstration completed")
}
```

---

## 📊 Advanced Caching Strategies

### **5. Multi-Level Cache System**

**Problem**: How to implement a hierarchical caching system?

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheLevel int

const (
    L1 CacheLevel = iota
    L2
    L3
)

type CacheItem struct {
    Key       string
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int
    LastAccess time.Time
}

type Cache struct {
    level       CacheLevel
    capacity    int
    items       map[string]*CacheItem
    mutex       sync.RWMutex
    nextLevel   *Cache
    prevLevel   *Cache
}

func NewCache(level CacheLevel, capacity int) *Cache {
    return &Cache{
        level:    level,
        capacity: capacity,
        items:    make(map[string]*CacheItem),
    }
}

func (c *Cache) SetNextLevel(next *Cache) {
    c.nextLevel = next
    if next != nil {
        next.prevLevel = c
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mutex.RLock()
    item, exists := c.items[key]
    c.mutex.RUnlock()
    
    if exists && !item.ExpiresAt.IsZero() && time.Now().After(item.ExpiresAt) {
        c.Delete(key)
        return nil, false
    }
    
    if exists {
        c.mutex.Lock()
        item.AccessCount++
        item.LastAccess = time.Now()
        c.mutex.Unlock()
        return item.Value, true
    }
    
    // Try next level
    if c.nextLevel != nil {
        if value, found := c.nextLevel.Get(key); found {
            c.Set(key, value, time.Hour)
            return value, true
        }
    }
    
    return nil, false
}

func (c *Cache) Set(key string, value interface{}, ttl time.Duration) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    var expiresAt time.Time
    if ttl > 0 {
        expiresAt = time.Now().Add(ttl)
    }
    
    item := &CacheItem{
        Key:        key,
        Value:      value,
        ExpiresAt:  expiresAt,
        AccessCount: 1,
        LastAccess: time.Now(),
    }
    
    c.items[key] = item
    
    // Evict if over capacity
    if len(c.items) > c.capacity {
        c.evict()
    }
}

func (c *Cache) Delete(key string) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    delete(c.items, key)
}

func (c *Cache) evict() {
    // LRU eviction
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range c.items {
        if oldestKey == "" || item.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(c.items, oldestKey)
    }
}

func (c *Cache) GetStats() map[string]interface{} {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    return map[string]interface{}{
        "level":        c.level,
        "capacity":     c.capacity,
        "size":         len(c.items),
        "utilization":  float64(len(c.items)) / float64(c.capacity),
    }
}

type MultiLevelCache struct {
    l1 *Cache
    l2 *Cache
    l3 *Cache
}

func NewMultiLevelCache() *MultiLevelCache {
    l1 := NewCache(L1, 100)  // Fast, small
    l2 := NewCache(L2, 1000) // Medium speed, medium size
    l3 := NewCache(L3, 10000) // Slow, large
    
    l1.SetNextLevel(l2)
    l2.SetNextLevel(l3)
    
    return &MultiLevelCache{
        l1: l1,
        l2: l2,
        l3: l3,
    }
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, bool) {
    return mlc.l1.Get(key)
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) {
    mlc.l1.Set(key, value, ttl)
}

func (mlc *MultiLevelCache) GetStats() map[string]interface{} {
    return map[string]interface{}{
        "L1": mlc.l1.GetStats(),
        "L2": mlc.l2.GetStats(),
        "L3": mlc.l3.GetStats(),
    }
}

func main() {
    cache := NewMultiLevelCache()
    
    // Set some values
    cache.Set("key1", "value1", time.Minute)
    cache.Set("key2", "value2", time.Minute)
    cache.Set("key3", "value3", time.Minute)
    
    // Get values
    if value, found := cache.Get("key1"); found {
        fmt.Printf("Found key1: %v\n", value)
    }
    
    if value, found := cache.Get("key2"); found {
        fmt.Printf("Found key2: %v\n", value)
    }
    
    // Get stats
    stats := cache.GetStats()
    fmt.Printf("Cache stats: %+v\n", stats)
}
```

---

## 📚 Additional Resources

### **Books**
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [Building Microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/) - Sam Newman
- [Patterns of Enterprise Application Architecture](https://martinfowler.com/books/eaa.html) - Martin Fowler

### **Online Resources**
- [Google's System Design Guide](https://www.google.com/about/careers/students/guide-to-technical-development.html)
- [High Scalability](http://highscalability.com/) - Real-world system design examples
- [System Design Primer](https://github.com/donnemartin/system-design-primer)

### **Video Resources**
- [Google Tech Talks](https://www.youtube.com/user/GoogleTechTalks)
- [System Design Interview](https://www.youtube.com/c/ExponentTV) - Exponent
- [Microservices Patterns](https://www.youtube.com/c/MicroservicesPatterns)

---

*This guide covers advanced system design patterns essential for Google-level interviews, with practical Go implementations and real-world examples.*
