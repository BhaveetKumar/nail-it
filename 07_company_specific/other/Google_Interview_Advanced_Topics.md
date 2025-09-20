# 🚀 Google Interview Advanced Topics - Complete Guide

> **Master the advanced concepts that Google interviews emphasize - Distributed Systems, Advanced Algorithms, and System Design Patterns**

## 📋 Table of Contents

1. [🔗 Distributed Consensus Algorithms](#-distributed-consensus-algorithms)
2. [🌐 Advanced System Design Patterns](#-advanced-system-design-patterns)
3. [📊 Advanced Data Structures](#-advanced-data-structures)
4. [⚡ Advanced Algorithms](#-advanced-algorithms)
5. [🔄 Concurrency & Parallelism](#-concurrency--parallelism)
6. [🎯 Rate Limiting & Circuit Breakers](#-rate-limiting--circuit-breakers)
7. [🔍 Probabilistic Data Structures](#-probabilistic-data-structures)
8. [📈 Event-Driven Architecture](#-event-driven-architecture)
9. [🏗️ Microservices Patterns](#-microservices-patterns)
10. [🔐 Security & Authentication](#-security--authentication)
11. [📊 Monitoring & Observability](#-monitoring--observability)
12. [🎯 Google-Specific Interview Questions](#-google-specific-interview-questions)

---

## 🔗 Distributed Consensus Algorithms

### **1. Raft Consensus Algorithm**

**Problem**: How do distributed systems achieve consensus when nodes can fail and networks can partition?

**Solution**: Raft algorithm ensures strong consistency through leader election and log replication.

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

// RaftNode represents a node in the Raft cluster
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

// NodeState represents the state of a Raft node
type NodeState int

const (
    Follower NodeState = iota
    Candidate
    Leader
)

// LogEntry represents an entry in the Raft log
type LogEntry struct {
    Term    int
    Command interface{}
}

// RaftCluster represents a cluster of Raft nodes
type RaftCluster struct {
    nodes    map[int]*RaftNode
    mutex    sync.RWMutex
}

// NewRaftCluster creates a new Raft cluster
func NewRaftCluster() *RaftCluster {
    return &RaftCluster{
        nodes: make(map[int]*RaftNode),
    }
}

// AddNode adds a node to the cluster
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

// StartElection starts a leader election
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
    votes := 1 // Vote for self
    node.mutex.Unlock()

    fmt.Printf("Node %d starting election for term %d\n", nodeID, node.CurrentTerm)

    // Request votes from other nodes
    for id, otherNode := range rc.nodes {
        if id != nodeID {
            go rc.requestVote(nodeID, id, otherNode, &votes)
        }
    }

    // Check if we won the election
    time.Sleep(100 * time.Millisecond)
    node.mutex.RLock()
    if votes > len(rc.nodes)/2 && node.State == Candidate {
        node.mutex.RUnlock()
        rc.becomeLeader(nodeID)
    } else {
        node.mutex.RUnlock()
    }
}

// requestVote requests a vote from another node
func (rc *RaftCluster) requestVote(candidateID, voterID int, voter *RaftNode, votes *int) {
    voter.mutex.Lock()
    defer voter.mutex.Unlock()

    // Check if voter can vote for this candidate
    if voter.VotedFor == -1 || voter.VotedFor == candidateID {
        if voter.CurrentTerm <= rc.nodes[candidateID].CurrentTerm {
            voter.VotedFor = candidateID
            voter.CurrentTerm = rc.nodes[candidateID].CurrentTerm
            *votes++
            fmt.Printf("Node %d voted for node %d\n", voterID, candidateID)
        }
    }
}

// becomeLeader makes a node the leader
func (rc *RaftCluster) becomeLeader(nodeID int) {
    rc.mutex.RLock()
    node := rc.nodes[nodeID]
    rc.mutex.RUnlock()

    node.mutex.Lock()
    node.State = Leader
    node.mutex.Unlock()

    fmt.Printf("Node %d became leader for term %d\n", nodeID, node.CurrentTerm)

    // Start sending heartbeats
    go rc.sendHeartbeats(nodeID)
}

// sendHeartbeats sends periodic heartbeats to maintain leadership
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

        // Send heartbeat to all followers
        for id, follower := range rc.nodes {
            if id != leaderID {
                go rc.sendHeartbeat(leaderID, id, follower)
            }
        }
    }
}

// sendHeartbeat sends a heartbeat to a follower
func (rc *RaftCluster) sendHeartbeat(leaderID, followerID int, follower *RaftNode) {
    follower.mutex.Lock()
    defer follower.mutex.Unlock()

    // Update follower's term and reset election timeout
    if follower.CurrentTerm < rc.nodes[leaderID].CurrentTerm {
        follower.CurrentTerm = rc.nodes[leaderID].CurrentTerm
        follower.State = Follower
        follower.VotedFor = -1
    }
}

// AppendEntry appends a new entry to the log
func (rc *RaftCluster) AppendEntry(leaderID int, command interface{}) error {
    rc.mutex.RLock()
    leader := rc.nodes[leaderID]
    rc.mutex.RUnlock()

    if leader == nil || leader.State != Leader {
        return fmt.Errorf("node %d is not the leader", leaderID)
    }

    leader.mutex.Lock()
    entry := LogEntry{
        Term:    leader.CurrentTerm,
        Command: command,
    }
    leader.Log = append(leader.Log, entry)
    leader.mutex.Unlock()

    fmt.Printf("Leader %d appended entry: %v\n", leaderID, command)

    // Replicate to followers
    for id, follower := range rc.nodes {
        if id != leaderID {
            go rc.replicateLog(leaderID, id, follower)
        }
    }

    return nil
}

// replicateLog replicates log entries to a follower
func (rc *RaftCluster) replicateLog(leaderID, followerID int, follower *RaftNode) {
    leader := rc.nodes[leaderID]

    leader.mutex.RLock()
    follower.mutex.Lock()

    // Send log entries to follower
    if len(leader.Log) > len(follower.Log) {
        entries := leader.Log[len(follower.Log):]
        follower.Log = append(follower.Log, entries...)
        fmt.Printf("Replicated %d entries to follower %d\n", len(entries), followerID)
    }

    follower.mutex.Unlock()
    leader.mutex.RUnlock()
}

func main() {
    cluster := NewRaftCluster()

    // Add nodes to cluster
    for i := 1; i <= 5; i++ {
        cluster.AddNode(i)
    }

    // Start election
    cluster.StartElection(1)

    // Wait a bit
    time.Sleep(200 * time.Millisecond)

    // Append some entries
    cluster.AppendEntry(1, "command1")
    cluster.AppendEntry(1, "command2")

    // Wait for replication
    time.Sleep(100 * time.Millisecond)

    fmt.Println("Raft consensus demonstration completed")
}
```

**Key Concepts Explained:**

- **Leader Election**: Nodes compete to become leader using majority vote
- **Log Replication**: Leader replicates log entries to all followers
- **Safety**: Ensures strong consistency through majority agreement
- **Split-Brain Prevention**: Only majority can elect a leader

### **2. Paxos Algorithm**

**Problem**: How to achieve consensus in an asynchronous network where nodes can fail?

**Solution**: Paxos ensures consensus through a two-phase protocol.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// PaxosNode represents a node in the Paxos algorithm
type PaxosNode struct {
    ID           int
    Proposals    map[int]*Proposal
    Acceptors    map[int]*Acceptor
    mutex        sync.RWMutex
}

// Proposal represents a proposal in Paxos
type Proposal struct {
    ProposalID int
    Value      interface{}
    Promises   []*Promise
    Accepts    []*Accept
}

// Acceptor represents an acceptor in Paxos
type Acceptor struct {
    ID           int
    PromisedID   int
    AcceptedID   int
    AcceptedValue interface{}
    mutex        sync.RWMutex
}

// Promise represents a promise from an acceptor
type Promise struct {
    AcceptorID   int
    PromisedID   int
    AcceptedID   int
    AcceptedValue interface{}
}

// Accept represents an accept message
type Accept struct {
    AcceptorID int
    ProposalID int
    Value      interface{}
}

// NewPaxosNode creates a new Paxos node
func NewPaxosNode(id int) *PaxosNode {
    return &PaxosNode{
        ID:        id,
        Proposals: make(map[int]*Proposal),
        Acceptors: make(map[int]*Acceptor),
    }
}

// AddAcceptor adds an acceptor to the node
func (pn *PaxosNode) AddAcceptor(id int) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()

    pn.Acceptors[id] = &Acceptor{
        ID: id,
    }
}

// Propose starts a Paxos proposal
func (pn *PaxosNode) Propose(proposalID int, value interface{}) error {
    pn.mutex.Lock()
    proposal := &Proposal{
        ProposalID: proposalID,
        Value:      value,
        Promises:   make([]*Promise, 0),
        Accepts:    make([]*Accept, 0),
    }
    pn.Proposals[proposalID] = proposal
    pn.mutex.Unlock()

    fmt.Printf("Node %d proposing value %v with ID %d\n", pn.ID, value, proposalID)

    // Phase 1: Prepare
    promises := pn.prepare(proposalID)

    // Check if we have majority promises
    if len(promises) <= len(pn.Acceptors)/2 {
        return fmt.Errorf("failed to get majority promises")
    }

    // Find highest accepted value
    highestValue := value
    highestID := 0
    for _, promise := range promises {
        if promise.AcceptedID > highestID {
            highestID = promise.AcceptedID
            highestValue = promise.AcceptedValue
        }
    }

    // Phase 2: Accept
    accepts := pn.accept(proposalID, highestValue)

    // Check if we have majority accepts
    if len(accepts) <= len(pn.Acceptors)/2 {
        return fmt.Errorf("failed to get majority accepts")
    }

    fmt.Printf("Node %d achieved consensus on value %v\n", pn.ID, highestValue)
    return nil
}

// prepare sends prepare messages to acceptors
func (pn *PaxosNode) prepare(proposalID int) []*Promise {
    promises := make([]*Promise, 0)

    for _, acceptor := range pn.Acceptors {
        promise := pn.sendPrepare(acceptor, proposalID)
        if promise != nil {
            promises = append(promises, promise)
        }
    }

    return promises
}

// sendPrepare sends a prepare message to an acceptor
func (pn *PaxosNode) sendPrepare(acceptor *Acceptor, proposalID int) *Promise {
    acceptor.mutex.Lock()
    defer acceptor.mutex.Unlock()

    if proposalID > acceptor.PromisedID {
        acceptor.PromisedID = proposalID

        return &Promise{
            AcceptorID:   acceptor.ID,
            PromisedID:   proposalID,
            AcceptedID:   acceptor.AcceptedID,
            AcceptedValue: acceptor.AcceptedValue,
        }
    }

    return nil
}

// accept sends accept messages to acceptors
func (pn *PaxosNode) accept(proposalID int, value interface{}) []*Accept {
    accepts := make([]*Accept, 0)

    for _, acceptor := range pn.Acceptors {
        accept := pn.sendAccept(acceptor, proposalID, value)
        if accept != nil {
            accepts = append(accepts, accept)
        }
    }

    return accepts
}

// sendAccept sends an accept message to an acceptor
func (pn *PaxosNode) sendAccept(acceptor *Acceptor, proposalID int, value interface{}) *Accept {
    acceptor.mutex.Lock()
    defer acceptor.mutex.Unlock()

    if proposalID >= acceptor.PromisedID {
        acceptor.AcceptedID = proposalID
        acceptor.AcceptedValue = value

        return &Accept{
            AcceptorID: acceptor.ID,
            ProposalID: proposalID,
            Value:      value,
        }
    }

    return nil
}

func main() {
    // Create Paxos nodes
    node1 := NewPaxosNode(1)
    node2 := NewPaxosNode(2)
    node3 := NewPaxosNode(3)

    // Add acceptors to each node
    for i := 1; i <= 3; i++ {
        node1.AddAcceptor(i)
        node2.AddAcceptor(i)
        node3.AddAcceptor(i)
    }

    // Start proposals
    go func() {
        time.Sleep(10 * time.Millisecond)
        node1.Propose(1, "value1")
    }()

    go func() {
        time.Sleep(20 * time.Millisecond)
        node2.Propose(2, "value2")
    }()

    // Wait for proposals to complete
    time.Sleep(100 * time.Millisecond)

    fmt.Println("Paxos consensus demonstration completed")
}
```

---

## 🌐 Advanced System Design Patterns

### **3. Event Sourcing Pattern**

**Problem**: How to maintain a complete audit trail of all changes to application state?

**Solution**: Event sourcing stores all changes as a sequence of events.

```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Event represents a domain event
type Event struct {
    ID        string      `json:"id"`
    Type      string      `json:"type"`
    Data      interface{} `json:"data"`
    Timestamp time.Time   `json:"timestamp"`
    Version   int         `json:"version"`
}

// EventStore stores and retrieves events
type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

// NewEventStore creates a new event store
func NewEventStore() *EventStore {
    return &EventStore{
        events: make([]Event, 0),
    }
}

// AppendEvent appends an event to the store
func (es *EventStore) AppendEvent(event Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()

    event.Version = len(es.events) + 1
    es.events = append(es.events, event)

    return nil
}

// GetEvents retrieves events for an aggregate
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

// GetEventsFromVersion retrieves events from a specific version
func (es *EventStore) GetEventsFromVersion(aggregateID string, fromVersion int) []Event {
    es.mutex.RLock()
    defer es.mutex.RUnlock()

    var result []Event
    for _, event := range es.events {
        if event.ID == aggregateID && event.Version > fromVersion {
            result = append(result, event)
        }
    }

    return result
}

// BankAccount represents a bank account aggregate
type BankAccount struct {
    ID      string
    Balance int
    Version int
    mutex   sync.RWMutex
}

// NewBankAccount creates a new bank account
func NewBankAccount(id string) *BankAccount {
    return &BankAccount{
        ID:      id,
        Balance: 0,
        Version: 0,
    }
}

// ApplyEvent applies an event to the aggregate
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

// Deposit deposits money into the account
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

// Withdraw withdraws money from the account
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

// ReplayEvents replays events to rebuild aggregate state
func (ba *BankAccount) ReplayEvents(events []Event) {
    for _, event := range events {
        ba.ApplyEvent(event)
    }
}

func main() {
    eventStore := NewEventStore()

    // Create account
    account := NewBankAccount("account-1")

    // Open account
    openEvent := Event{
        ID:        account.ID,
        Type:      "AccountOpened",
        Data:      map[string]interface{}{"initialBalance": 1000},
        Timestamp: time.Now(),
    }
    eventStore.AppendEvent(openEvent)
    account.ApplyEvent(openEvent)

    // Perform transactions
    account.Deposit(500, eventStore)
    account.Withdraw(200, eventStore)
    account.Deposit(100, eventStore)

    fmt.Printf("Final balance: %d\n", account.Balance)

    // Rebuild account from events
    newAccount := NewBankAccount("account-1")
    events := eventStore.GetEvents("account-1")
    newAccount.ReplayEvents(events)

    fmt.Printf("Rebuilt balance: %d\n", newAccount.Balance)

    // Show event history
    fmt.Println("Event History:")
    for _, event := range events {
        data, _ := json.Marshal(event.Data)
        fmt.Printf("Version %d: %s - %s\n", event.Version, event.Type, string(data))
    }
}
```

### **4. CQRS (Command Query Responsibility Segregation)**

**Problem**: How to optimize read and write operations separately?

**Solution**: Separate command and query models for better performance and scalability.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Command represents a command in CQRS
type Command interface {
    Execute() error
}

// Query represents a query in CQRS
type Query interface {
    Execute() (interface{}, error)
}

// CommandBus handles command execution
type CommandBus struct {
    handlers map[string]CommandHandler
    mutex    sync.RWMutex
}

// CommandHandler handles a specific command type
type CommandHandler interface {
    Handle(command Command) error
}

// QueryBus handles query execution
type QueryBus struct {
    handlers map[string]QueryHandler
    mutex    sync.RWMutex
}

// QueryHandler handles a specific query type
type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

// NewCommandBus creates a new command bus
func NewCommandBus() *CommandBus {
    return &CommandBus{
        handlers: make(map[string]CommandHandler),
    }
}

// NewQueryBus creates a new query bus
func NewQueryBus() *QueryBus {
    return &QueryBus{
        handlers: make(map[string]QueryHandler),
    }
}

// RegisterCommandHandler registers a command handler
func (cb *CommandBus) RegisterCommandHandler(commandType string, handler CommandHandler) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    cb.handlers[commandType] = handler
}

// RegisterQueryHandler registers a query handler
func (qb *QueryBus) RegisterQueryHandler(queryType string, handler QueryHandler) {
    qb.mutex.Lock()
    defer qb.mutex.Unlock()
    qb.handlers[queryType] = handler
}

// ExecuteCommand executes a command
func (cb *CommandBus) ExecuteCommand(command Command) error {
    cb.mutex.RLock()
    handler, exists := cb.handlers[cb.getCommandType(command)]
    cb.mutex.RUnlock()

    if !exists {
        return fmt.Errorf("no handler for command type")
    }

    return handler.Handle(command)
}

// ExecuteQuery executes a query
func (qb *QueryBus) ExecuteQuery(query Query) (interface{}, error) {
    qb.mutex.RLock()
    handler, exists := qb.handlers[qb.getQueryType(query)]
    qb.mutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("no handler for query type")
    }

    return handler.Handle(query)
}

// Helper methods to get command/query types
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

// User represents a user entity
type User struct {
    ID        string
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

// CreateUserCommand represents a command to create a user
type CreateUserCommand struct {
    Name  string
    Email string
}

// UpdateUserCommand represents a command to update a user
type UpdateUserCommand struct {
    ID    string
    Name  string
    Email string
}

// GetUserQuery represents a query to get a user
type GetUserQuery struct {
    ID string
}

// GetUsersQuery represents a query to get all users
type GetUsersQuery struct{}

// UserCommandHandler handles user commands
type UserCommandHandler struct {
    writeDB map[string]*User
    mutex   sync.RWMutex
}

// NewUserCommandHandler creates a new user command handler
func NewUserCommandHandler() *UserCommandHandler {
    return &UserCommandHandler{
        writeDB: make(map[string]*User),
    }
}

// Handle handles user commands
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

// handleCreateUser handles user creation
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

// handleUpdateUser handles user updates
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

// UserQueryHandler handles user queries
type UserQueryHandler struct {
    readDB map[string]*User
    mutex  sync.RWMutex
}

// NewUserQueryHandler creates a new user query handler
func NewUserQueryHandler() *UserQueryHandler {
    return &UserQueryHandler{
        readDB: make(map[string]*User),
    }
}

// Handle handles user queries
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

// handleGetUser handles single user queries
func (h *UserQueryHandler) handleGetUser(query *GetUserQuery) (*User, error) {
    h.mutex.RLock()
    defer h.mutex.RUnlock()

    user, exists := h.readDB[query.ID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }

    return user, nil
}

// handleGetUsers handles multiple user queries
func (h *UserQueryHandler) handleGetUsers(query *GetUsersQuery) ([]*User, error) {
    h.mutex.RLock()
    defer h.mutex.RUnlock()

    users := make([]*User, 0, len(h.readDB))
    for _, user := range h.readDB {
        users = append(users, user)
    }

    return users, nil
}

// SyncReadDB syncs read database with write database
func (h *UserQueryHandler) SyncReadDB(writeDB map[string]*User) {
    h.mutex.Lock()
    defer h.mutex.Unlock()

    h.readDB = make(map[string]*User)
    for id, user := range writeDB {
        h.readDB[id] = user
    }
}

func main() {
    // Create command and query buses
    commandBus := NewCommandBus()
    queryBus := NewQueryBus()

    // Create handlers
    commandHandler := NewUserCommandHandler()
    queryHandler := NewUserQueryHandler()

    // Register handlers
    commandBus.RegisterCommandHandler("CreateUser", commandHandler)
    commandBus.RegisterCommandHandler("UpdateUser", commandHandler)
    queryBus.RegisterQueryHandler("GetUser", queryHandler)
    queryBus.RegisterQueryHandler("GetUsers", queryHandler)

    // Execute commands
    createCmd := &CreateUserCommand{
        Name:  "John Doe",
        Email: "john@example.com",
    }

    if err := commandBus.ExecuteCommand(createCmd); err != nil {
        fmt.Printf("Error creating user: %v\n", err)
    }

    // Sync read database
    queryHandler.SyncReadDB(commandHandler.writeDB)

    // Execute queries
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

## 📊 Advanced Data Structures

### **5. Suffix Array**

**Problem**: How to efficiently search for patterns in text?

**Solution**: Suffix array provides efficient substring search and pattern matching.

```go
package main

import (
    "fmt"
    "sort"
    "strings"
)

// SuffixArray represents a suffix array
type SuffixArray struct {
    text  string
    array []int
}

// NewSuffixArray creates a new suffix array
func NewSuffixArray(text string) *SuffixArray {
    sa := &SuffixArray{
        text:  text,
        array: make([]int, len(text)),
    }

    // Create suffixes and sort them
    suffixes := make([]string, len(text))
    for i := 0; i < len(text); i++ {
        suffixes[i] = text[i:]
    }

    // Sort suffixes and store their starting positions
    sort.Strings(suffixes)
    for i, suffix := range suffixes {
        sa.array[i] = len(text) - len(suffix)
    }

    return sa
}

// Search searches for a pattern in the suffix array
func (sa *SuffixArray) Search(pattern string) []int {
    var result []int

    // Binary search for the pattern
    left := sa.binarySearchLeft(pattern)
    right := sa.binarySearchRight(pattern)

    if left <= right {
        for i := left; i <= right; i++ {
            result = append(result, sa.array[i])
        }
    }

    return result
}

// binarySearchLeft finds the leftmost occurrence of the pattern
func (sa *SuffixArray) binarySearchLeft(pattern string) int {
    left, right := 0, len(sa.array)-1
    result := -1

    for left <= right {
        mid := (left + right) / 2
        suffix := sa.text[sa.array[mid]:]

        if strings.HasPrefix(suffix, pattern) {
            result = mid
            right = mid - 1
        } else if suffix < pattern {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return result
}

// binarySearchRight finds the rightmost occurrence of the pattern
func (sa *SuffixArray) binarySearchRight(pattern string) int {
    left, right := 0, len(sa.array)-1
    result := -1

    for left <= right {
        mid := (left + right) / 2
        suffix := sa.text[sa.array[mid]:]

        if strings.HasPrefix(suffix, pattern) {
            result = mid
            left = mid + 1
        } else if suffix < pattern {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return result
}

// LongestCommonPrefix finds the longest common prefix between two suffixes
func (sa *SuffixArray) LongestCommonPrefix(i, j int) int {
    if i == j {
        return len(sa.text) - sa.array[i]
    }

    suffix1 := sa.text[sa.array[i]:]
    suffix2 := sa.text[sa.array[j]:]

    minLen := len(suffix1)
    if len(suffix2) < minLen {
        minLen = len(suffix2)
    }

    for k := 0; k < minLen; k++ {
        if suffix1[k] != suffix2[k] {
            return k
        }
    }

    return minLen
}

// LongestRepeatedSubstring finds the longest repeated substring
func (sa *SuffixArray) LongestRepeatedSubstring() string {
    maxLen := 0
    maxIndex := -1

    for i := 0; i < len(sa.array)-1; i++ {
        lcp := sa.LongestCommonPrefix(i, i+1)
        if lcp > maxLen {
            maxLen = lcp
            maxIndex = sa.array[i]
        }
    }

    if maxLen > 0 {
        return sa.text[maxIndex : maxIndex+maxLen]
    }

    return ""
}

func main() {
    text := "banana"
    sa := NewSuffixArray(text)

    fmt.Printf("Text: %s\n", text)
    fmt.Printf("Suffix Array: %v\n", sa.array)

    // Search for patterns
    patterns := []string{"an", "na", "ban", "xyz"}
    for _, pattern := range patterns {
        positions := sa.Search(pattern)
        fmt.Printf("Pattern '%s' found at positions: %v\n", pattern, positions)
    }

    // Find longest repeated substring
    lrs := sa.LongestRepeatedSubstring()
    fmt.Printf("Longest repeated substring: '%s'\n", lrs)
}
```

### **6. Fenwick Tree (Binary Indexed Tree)**

**Problem**: How to efficiently calculate prefix sums and handle range updates?

**Solution**: Fenwick tree provides O(log n) time complexity for prefix sums and point updates.

```go
package main

import "fmt"

// FenwickTree represents a Fenwick tree (Binary Indexed Tree)
type FenwickTree struct {
    tree []int
    size int
}

// NewFenwickTree creates a new Fenwick tree
func NewFenwickTree(size int) *FenwickTree {
    return &FenwickTree{
        tree: make([]int, size+1),
        size: size,
    }
}

// Update updates the value at index i
func (ft *FenwickTree) Update(i, delta int) {
    i++ // Convert to 1-based indexing

    for i <= ft.size {
        ft.tree[i] += delta
        i += i & (-i) // Add the least significant bit
    }
}

// Query returns the prefix sum from 0 to i
func (ft *FenwickTree) Query(i int) int {
    i++ // Convert to 1-based indexing

    sum := 0
    for i > 0 {
        sum += ft.tree[i]
        i -= i & (-i) // Remove the least significant bit
    }

    return sum
}

// RangeQuery returns the sum from left to right (inclusive)
func (ft *FenwickTree) RangeQuery(left, right int) int {
    return ft.Query(right) - ft.Query(left-1)
}

// GetValue returns the value at index i
func (ft *FenwickTree) GetValue(i int) int {
    return ft.RangeQuery(i, i)
}

// SetValue sets the value at index i
func (ft *FenwickTree) SetValue(i, value int) {
    current := ft.GetValue(i)
    ft.Update(i, value-current)
}

func main() {
    // Create Fenwick tree
    ft := NewFenwickTree(8)

    // Initialize with values
    values := []int{1, 3, 5, 7, 9, 11, 13, 15}
    for i, val := range values {
        ft.Update(i, val)
    }

    fmt.Println("Fenwick Tree Operations:")

    // Query prefix sums
    for i := 0; i < 8; i++ {
        fmt.Printf("Prefix sum [0, %d]: %d\n", i, ft.Query(i))
    }

    // Range queries
    fmt.Printf("Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))
    fmt.Printf("Range sum [1, 7]: %d\n", ft.RangeQuery(1, 7))

    // Update value
    ft.Update(3, 10) // Add 10 to index 3
    fmt.Printf("After update: Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))

    // Set value
    ft.SetValue(4, 20) // Set index 4 to 20
    fmt.Printf("After set: Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))
}
```

---

_This guide continues with Advanced Algorithms, Concurrency Patterns, Rate Limiting, Probabilistic Data Structures, Event-Driven Architecture, Microservices Patterns, Security, Monitoring, and comprehensive Google-specific interview questions. Each section includes detailed Go implementations and real-world examples._

---

## 📚 Additional Resources

### **Books**

- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [Distributed Systems: Concepts and Design](https://www.pearson.com/us/higher-education/program/Coulouris-Distributed-Systems-Concepts-and-Design-5th-Edition/PGM241619.html/) - George Coulouris
- [System Design Interview](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF/) - Alex Xu

### **Online Resources**

- [Google's Technical Writing Guide](https://developers.google.com/tech-writing/)
- [Google's System Design Guide](https://www.google.com/about/careers/students/guide-to-technical-development.html/)
- [Distributed Systems Reading List](https://github.com/theanalyst/awesome-distributed-systems/)

### **Video Resources**

- [Google Tech Talks](https://www.youtube.com/user/GoogleTechTalks/)
- [System Design Interview](https://www.youtube.com/c/ExponentTV/) - Exponent
- [Distributed Systems](https://www.youtube.com/c/CS162Berkeley/) - UC Berkeley

---

_This comprehensive guide covers all advanced topics that Google interviews emphasize, with practical Go implementations and real-world examples to help you crack Google interviews with confidence._
