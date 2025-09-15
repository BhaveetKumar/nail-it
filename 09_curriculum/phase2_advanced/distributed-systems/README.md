# Distributed Systems

## Table of Contents

1. [Overview](#overview)
2. [Consensus Algorithms](#consensus-algorithms)
3. [Distributed Storage](#distributed-storage)
4. [Event Sourcing](#event-sourcing)
5. [Service Mesh](#service-mesh)
6. [Distributed Tracing](#distributed-tracing)
7. [Fault Tolerance](#fault-tolerance)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master consensus algorithms (Raft, PBFT, PoW)
- Design distributed storage systems
- Implement event sourcing patterns
- Build service mesh architectures
- Implement distributed tracing
- Design fault-tolerant systems

### What are Distributed Systems?

Distributed Systems are collections of independent computers that appear to users as a single coherent system, designed to handle large-scale data processing and provide high availability.

## Consensus Algorithms

### 1. Raft Consensus

#### Raft Implementation
```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "sync"
    "time"
)

type RaftNode struct {
    ID          int
    State       string // Follower, Candidate, Leader
    CurrentTerm int
    VotedFor    int
    Log         []LogEntry
    CommitIndex int
    LastApplied int
    NextIndex   map[int]int
    MatchIndex  map[int]int
    Peers       []int
    mutex       sync.RWMutex
}

type LogEntry struct {
    Term    int
    Command string
}

type RaftCluster struct {
    nodes map[int]*RaftNode
    mutex sync.RWMutex
}

func NewRaftNode(id int, peers []int) *RaftNode {
    return &RaftNode{
        ID:          id,
        State:       "Follower",
        CurrentTerm: 0,
        VotedFor:    -1,
        Log:         make([]LogEntry, 0),
        CommitIndex: -1,
        LastApplied: -1,
        NextIndex:   make(map[int]int),
        MatchIndex:  make(map[int]int),
        Peers:       peers,
    }
}

func (rn *RaftNode) StartElection() {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    rn.State = "Candidate"
    rn.CurrentTerm++
    rn.VotedFor = rn.ID
    
    fmt.Printf("Node %d starting election for term %d\n", rn.ID, rn.CurrentTerm)
    
    // Request votes from peers
    votes := 1 // Vote for self
    for _, peerID := range rn.Peers {
        if rn.requestVote(peerID) {
            votes++
        }
    }
    
    // Check if we won the election
    if votes > len(rn.Peers)/2 {
        rn.becomeLeader()
    }
}

func (rn *RaftNode) requestVote(peerID int) bool {
    // Simulate vote request
    fmt.Printf("Node %d requesting vote from node %d\n", rn.ID, peerID)
    
    // In real implementation, this would be an RPC call
    // For simulation, randomly grant vote
    return rand.Float32() < 0.7
}

func (rn *RaftNode) becomeLeader() {
    rn.State = "Leader"
    fmt.Printf("Node %d became leader for term %d\n", rn.ID, rn.CurrentTerm)
    
    // Initialize nextIndex and matchIndex
    for _, peerID := range rn.Peers {
        rn.NextIndex[peerID] = len(rn.Log)
        rn.MatchIndex[peerID] = -1
    }
    
    // Start sending heartbeats
    go rn.sendHeartbeats()
}

func (rn *RaftNode) sendHeartbeats() {
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()
    
    for range ticker.C {
        rn.mutex.RLock()
        if rn.State != "Leader" {
            rn.mutex.RUnlock()
            return
        }
        rn.mutex.RUnlock()
        
        rn.sendAppendEntries()
    }
}

func (rn *RaftNode) sendAppendEntries() {
    for _, peerID := range rn.Peers {
        go rn.appendEntries(peerID)
    }
}

func (rn *RaftNode) appendEntries(peerID int) {
    rn.mutex.RLock()
    prevLogIndex := rn.NextIndex[peerID] - 1
    prevLogTerm := 0
    if prevLogIndex >= 0 {
        prevLogTerm = rn.Log[prevLogIndex].Term
    }
    
    entries := rn.Log[rn.NextIndex[peerID]:]
    
    // Simulate append entries RPC
    fmt.Printf("Node %d sending append entries to node %d\n", rn.ID, peerID)
    
    // In real implementation, this would be an RPC call
    // For simulation, randomly succeed
    if rand.Float32() < 0.8 {
        rn.mutex.RUnlock()
        rn.mutex.Lock()
        rn.NextIndex[peerID] = len(rn.Log)
        rn.MatchIndex[peerID] = len(rn.Log) - 1
        rn.mutex.Unlock()
    } else {
        rn.mutex.RUnlock()
    }
}

func (rn *RaftNode) AppendCommand(command string) {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    if rn.State != "Leader" {
        fmt.Printf("Node %d is not leader, cannot append command\n", rn.ID)
        return
    }
    
    entry := LogEntry{
        Term:    rn.CurrentTerm,
        Command: command,
    }
    
    rn.Log = append(rn.Log, entry)
    fmt.Printf("Node %d appended command: %s\n", rn.ID, command)
}

func main() {
    // Create a 3-node Raft cluster
    cluster := &RaftCluster{
        nodes: make(map[int]*RaftNode),
    }
    
    // Create nodes
    for i := 1; i <= 3; i++ {
        peers := []int{}
        for j := 1; j <= 3; j++ {
            if i != j {
                peers = append(peers, j)
            }
        }
        cluster.nodes[i] = NewRaftNode(i, peers)
    }
    
    // Start election
    cluster.nodes[1].StartElection()
    
    // Simulate some commands
    time.Sleep(1 * time.Second)
    cluster.nodes[1].AppendCommand("SET key1 value1")
    cluster.nodes[1].AppendCommand("SET key2 value2")
    
    time.Sleep(2 * time.Second)
}
```

### 2. PBFT (Practical Byzantine Fault Tolerance)

#### PBFT Implementation
```go
package main

import (
    "crypto/sha256"
    "fmt"
    "sync"
    "time"
)

type PBFTNode struct {
    ID           int
    View         int
    Sequence     int
    State        string
    Prepared     map[string]bool
    Committed    map[string]bool
    Messages     []Message
    mutex        sync.RWMutex
}

type Message struct {
    Type      string
    From      int
    To        int
    View      int
    Sequence  int
    Digest    string
    Content   string
    Timestamp time.Time
}

type PBFTCluster struct {
    nodes map[int]*PBFTNode
    f     int // Maximum number of faulty nodes
    mutex sync.RWMutex
}

func NewPBFTNode(id int) *PBFTNode {
    return &PBFTNode{
        ID:        id,
        View:      0,
        Sequence:  0,
        State:     "Normal",
        Prepared:  make(map[string]bool),
        Committed: make(map[string]bool),
        Messages:  make([]Message, 0),
    }
}

func (pn *PBFTNode) PrePrepare(request string) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    pn.Sequence++
    digest := pn.computeDigest(request)
    
    message := Message{
        Type:      "PRE-PREPARE",
        From:      pn.ID,
        View:      pn.View,
        Sequence:  pn.Sequence,
        Digest:    digest,
        Content:   request,
        Timestamp: time.Now(),
    }
    
    pn.Messages = append(pn.Messages, message)
    fmt.Printf("Node %d sent PRE-PREPARE for sequence %d\n", pn.ID, pn.Sequence)
    
    // Broadcast to all nodes
    pn.broadcast(message)
}

func (pn *PBFTNode) Prepare(message Message) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    // Verify PRE-PREPARE message
    if !pn.verifyPrePrepare(message) {
        return
    }
    
    prepareMessage := Message{
        Type:      "PREPARE",
        From:      pn.ID,
        View:      message.View,
        Sequence:  message.Sequence,
        Digest:    message.Digest,
        Content:   message.Content,
        Timestamp: time.Now(),
    }
    
    pn.Messages = append(pn.Messages, prepareMessage)
    fmt.Printf("Node %d sent PREPARE for sequence %d\n", pn.ID, message.Sequence)
    
    pn.broadcast(prepareMessage)
}

func (pn *PBFTNode) Commit(message Message) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    // Check if we have enough PREPARE messages
    prepareCount := pn.countMessages("PREPARE", message.Sequence, message.Digest)
    if prepareCount < 2*pn.f { // 2f + 1 total nodes, need 2f + 1 prepares
        return
    }
    
    commitMessage := Message{
        Type:      "COMMIT",
        From:      pn.ID,
        View:      message.View,
        Sequence:  message.Sequence,
        Digest:    message.Digest,
        Content:   message.Content,
        Timestamp: time.Now(),
    }
    
    pn.Messages = append(pn.Messages, commitMessage)
    fmt.Printf("Node %d sent COMMIT for sequence %d\n", pn.ID, message.Sequence)
    
    pn.broadcast(commitMessage)
}

func (pn *PBFTNode) Reply(message Message) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    // Check if we have enough COMMIT messages
    commitCount := pn.countMessages("COMMIT", message.Sequence, message.Digest)
    if commitCount < 2*pn.f {
        return
    }
    
    // Execute the request
    fmt.Printf("Node %d executing request: %s\n", pn.ID, message.Content)
    pn.Committed[message.Digest] = true
}

func (pn *PBFTNode) verifyPrePrepare(message Message) bool {
    // Verify view number
    if message.View != pn.View {
        return false
    }
    
    // Verify sequence number
    if message.Sequence <= pn.Sequence {
        return false
    }
    
    // Verify digest
    expectedDigest := pn.computeDigest(message.Content)
    return message.Digest == expectedDigest
}

func (pn *PBFTNode) countMessages(msgType string, sequence int, digest string) int {
    count := 0
    for _, msg := range pn.Messages {
        if msg.Type == msgType && msg.Sequence == sequence && msg.Digest == digest {
            count++
        }
    }
    return count
}

func (pn *PBFTNode) computeDigest(content string) string {
    hash := sha256.Sum256([]byte(content))
    return fmt.Sprintf("%x", hash)
}

func (pn *PBFTNode) broadcast(message Message) {
    // In real implementation, this would send to all nodes
    fmt.Printf("Broadcasting %s from node %d\n", message.Type, pn.ID)
}

func main() {
    // Create PBFT cluster with 4 nodes (tolerates 1 Byzantine fault)
    cluster := &PBFTCluster{
        nodes: make(map[int]*PBFTNode),
        f:     1,
    }
    
    for i := 1; i <= 4; i++ {
        node := NewPBFTNode(i)
        node.f = 1
        cluster.nodes[i] = node
    }
    
    // Simulate PBFT protocol
    fmt.Println("Starting PBFT protocol...")
    
    // Primary node (node 1) sends PRE-PREPARE
    cluster.nodes[1].PrePrepare("SET key value")
    
    time.Sleep(100 * time.Millisecond)
}
```

## Distributed Storage

### 1. Consistent Hashing

#### Consistent Hashing Implementation
```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "strconv"
)

type ConsistentHash struct {
    ring     map[uint32]string
    sortedKeys []uint32
    replicas  int
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        ring:      make(map[uint32]string),
        sortedKeys: make([]uint32, 0),
        replicas:  replicas,
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    for i := 0; i < ch.replicas; i++ {
        key := ch.hashKey(node + ":" + strconv.Itoa(i))
        ch.ring[key] = node
        ch.sortedKeys = append(ch.sortedKeys, key)
    }
    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
    fmt.Printf("Added node %s to consistent hash ring\n", node)
}

func (ch *ConsistentHash) RemoveNode(node string) {
    for i := 0; i < ch.replicas; i++ {
        key := ch.hashKey(node + ":" + strconv.Itoa(i))
        delete(ch.ring, key)
        
        // Remove from sorted keys
        for j, k := range ch.sortedKeys {
            if k == key {
                ch.sortedKeys = append(ch.sortedKeys[:j], ch.sortedKeys[j+1:]...)
                break
            }
        }
    }
    fmt.Printf("Removed node %s from consistent hash ring\n", node)
}

func (ch *ConsistentHash) GetNode(key string) string {
    if len(ch.sortedKeys) == 0 {
        return ""
    }
    
    hash := ch.hashKey(key)
    
    // Find the first node with hash >= key hash
    for _, nodeHash := range ch.sortedKeys {
        if nodeHash >= hash {
            return ch.ring[nodeHash]
        }
    }
    
    // Wrap around to first node
    return ch.ring[ch.sortedKeys[0]]
}

func (ch *ConsistentHash) hashKey(key string) uint32 {
    hash := md5.Sum([]byte(key))
    return uint32(hash[0])<<24 | uint32(hash[1])<<16 | uint32(hash[2])<<8 | uint32(hash[3])
}

func (ch *ConsistentHash) PrintRing() {
    fmt.Println("Consistent Hash Ring:")
    for _, key := range ch.sortedKeys {
        fmt.Printf("Hash: %d -> Node: %s\n", key, ch.ring[key])
    }
}

func main() {
    ch := NewConsistentHash(3)
    
    // Add nodes
    ch.AddNode("node1")
    ch.AddNode("node2")
    ch.AddNode("node3")
    
    ch.PrintRing()
    
    // Test key distribution
    keys := []string{"key1", "key2", "key3", "key4", "key5"}
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key %s -> Node %s\n", key, node)
    }
    
    // Remove a node and see redistribution
    fmt.Println("\nRemoving node2...")
    ch.RemoveNode("node2")
    
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key %s -> Node %s\n", key, node)
    }
}
```

### 2. Distributed Key-Value Store

#### Distributed KV Store
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type DistributedKVStore struct {
    nodes    map[string]*KVNode
    hashRing *ConsistentHash
    mutex    sync.RWMutex
}

type KVNode struct {
    ID    string
    Data  map[string]string
    mutex sync.RWMutex
}

func NewDistributedKVStore() *DistributedKVStore {
    return &DistributedKVStore{
        nodes:    make(map[string]*KVNode),
        hashRing: NewConsistentHash(3),
    }
}

func (dkv *DistributedKVStore) AddNode(nodeID string) {
    dkv.mutex.Lock()
    defer dkv.mutex.Unlock()
    
    node := &KVNode{
        ID:   nodeID,
        Data: make(map[string]string),
    }
    
    dkv.nodes[nodeID] = node
    dkv.hashRing.AddNode(nodeID)
}

func (dkv *DistributedKVStore) RemoveNode(nodeID string) {
    dkv.mutex.Lock()
    defer dkv.mutex.Unlock()
    
    delete(dkv.nodes, nodeID)
    dkv.hashRing.RemoveNode(nodeID)
}

func (dkv *DistributedKVStore) Set(key, value string) error {
    dkv.mutex.RLock()
    nodeID := dkv.hashRing.GetNode(key)
    dkv.mutex.RUnlock()
    
    if nodeID == "" {
        return fmt.Errorf("no nodes available")
    }
    
    node := dkv.nodes[nodeID]
    node.mutex.Lock()
    node.Data[key] = value
    node.mutex.Unlock()
    
    fmt.Printf("Set %s=%s on node %s\n", key, value, nodeID)
    return nil
}

func (dkv *DistributedKVStore) Get(key string) (string, error) {
    dkv.mutex.RLock()
    nodeID := dkv.hashRing.GetNode(key)
    dkv.mutex.RUnlock()
    
    if nodeID == "" {
        return "", fmt.Errorf("no nodes available")
    }
    
    node := dkv.nodes[nodeID]
    node.mutex.RLock()
    value, exists := node.Data[key]
    node.mutex.RUnlock()
    
    if !exists {
        return "", fmt.Errorf("key not found")
    }
    
    fmt.Printf("Get %s=%s from node %s\n", key, value, nodeID)
    return value, nil
}

func (dkv *DistributedKVStore) Delete(key string) error {
    dkv.mutex.RLock()
    nodeID := dkv.hashRing.GetNode(key)
    dkv.mutex.RUnlock()
    
    if nodeID == "" {
        return fmt.Errorf("no nodes available")
    }
    
    node := dkv.nodes[nodeID]
    node.mutex.Lock()
    delete(node.Data, key)
    node.mutex.Unlock()
    
    fmt.Printf("Deleted %s from node %s\n", key, nodeID)
    return nil
}

func main() {
    dkv := NewDistributedKVStore()
    
    // Add nodes
    dkv.AddNode("node1")
    dkv.AddNode("node2")
    dkv.AddNode("node3")
    
    // Set some values
    dkv.Set("user:1", "Alice")
    dkv.Set("user:2", "Bob")
    dkv.Set("user:3", "Charlie")
    
    // Get values
    value, _ := dkv.Get("user:1")
    fmt.Printf("Retrieved: %s\n", value)
    
    // Simulate node failure
    fmt.Println("\nSimulating node2 failure...")
    dkv.RemoveNode("node2")
    
    // Try to get value (should be redistributed)
    value, _ = dkv.Get("user:2")
    fmt.Printf("Retrieved after node failure: %s\n", value)
}
```

## Event Sourcing

### 1. Event Store Implementation

#### Event Store
```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type Event struct {
    ID        string    `json:"id"`
    Type      string    `json:"type"`
    AggregateID string  `json:"aggregate_id"`
    Version   int       `json:"version"`
    Data      map[string]interface{} `json:"data"`
    Timestamp time.Time `json:"timestamp"`
}

type EventStore struct {
    events map[string][]Event
    mutex  sync.RWMutex
}

func NewEventStore() *EventStore {
    return &EventStore{
        events: make(map[string][]Event),
    }
}

func (es *EventStore) AppendEvents(aggregateID string, events []Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    // Check for version conflicts
    currentVersion := len(es.events[aggregateID])
    for _, event := range events {
        if event.Version != currentVersion+1 {
            return fmt.Errorf("version conflict: expected %d, got %d", 
                currentVersion+1, event.Version)
        }
        currentVersion++
    }
    
    // Append events
    es.events[aggregateID] = append(es.events[aggregateID], events...)
    
    fmt.Printf("Appended %d events for aggregate %s\n", len(events), aggregateID)
    return nil
}

func (es *EventStore) GetEvents(aggregateID string, fromVersion int) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    events, exists := es.events[aggregateID]
    if !exists {
        return nil, fmt.Errorf("aggregate %s not found", aggregateID)
    }
    
    if fromVersion >= len(events) {
        return []Event{}, nil
    }
    
    return events[fromVersion:], nil
}

func (es *EventStore) GetSnapshot(aggregateID string) (map[string]interface{}, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    events, exists := es.events[aggregateID]
    if !exists {
        return nil, fmt.Errorf("aggregate %s not found", aggregateID)
    }
    
    // Replay all events to build current state
    snapshot := make(map[string]interface{})
    for _, event := range events {
        es.applyEvent(snapshot, event)
    }
    
    return snapshot, nil
}

func (es *EventStore) applyEvent(snapshot map[string]interface{}, event Event) {
    switch event.Type {
    case "UserCreated":
        snapshot["id"] = event.Data["id"]
        snapshot["name"] = event.Data["name"]
        snapshot["email"] = event.Data["email"]
        snapshot["status"] = "active"
    case "UserUpdated":
        if name, ok := event.Data["name"].(string); ok {
            snapshot["name"] = name
        }
        if email, ok := event.Data["email"].(string); ok {
            snapshot["email"] = email
        }
    case "UserDeactivated":
        snapshot["status"] = "inactive"
    }
}

type UserAggregate struct {
    ID      string
    Name    string
    Email   string
    Status  string
    Version int
    eventStore *EventStore
}

func NewUserAggregate(id string, eventStore *EventStore) *UserAggregate {
    return &UserAggregate{
        ID:         id,
        eventStore: eventStore,
    }
}

func (ua *UserAggregate) Load() error {
    events, err := ua.eventStore.GetEvents(ua.ID, 0)
    if err != nil {
        return err
    }
    
    for _, event := range events {
        ua.applyEvent(event)
    }
    
    return nil
}

func (ua *UserAggregate) Create(name, email string) error {
    event := Event{
        ID:          fmt.Sprintf("event-%d", time.Now().UnixNano()),
        Type:        "UserCreated",
        AggregateID: ua.ID,
        Version:     ua.Version + 1,
        Data: map[string]interface{}{
            "id":    ua.ID,
            "name":  name,
            "email": email,
        },
        Timestamp: time.Now(),
    }
    
    return ua.eventStore.AppendEvents(ua.ID, []Event{event})
}

func (ua *UserAggregate) Update(name, email string) error {
    event := Event{
        ID:          fmt.Sprintf("event-%d", time.Now().UnixNano()),
        Type:        "UserUpdated",
        AggregateID: ua.ID,
        Version:     ua.Version + 1,
        Data: map[string]interface{}{
            "name":  name,
            "email": email,
        },
        Timestamp: time.Now(),
    }
    
    return ua.eventStore.AppendEvents(ua.ID, []Event{event})
}

func (ua *UserAggregate) applyEvent(event Event) {
    switch event.Type {
    case "UserCreated":
        ua.Name = event.Data["name"].(string)
        ua.Email = event.Data["email"].(string)
        ua.Status = "active"
    case "UserUpdated":
        if name, ok := event.Data["name"].(string); ok {
            ua.Name = name
        }
        if email, ok := event.Data["email"].(string); ok {
            ua.Email = email
        }
    case "UserDeactivated":
        ua.Status = "inactive"
    }
    ua.Version = event.Version
}

func main() {
    eventStore := NewEventStore()
    
    // Create user aggregate
    user := NewUserAggregate("user-1", eventStore)
    
    // Create user
    user.Create("Alice", "alice@example.com")
    user.Load()
    fmt.Printf("User created: %+v\n", user)
    
    // Update user
    user.Update("Alice Smith", "alice.smith@example.com")
    user.Load()
    fmt.Printf("User updated: %+v\n", user)
    
    // Get snapshot
    snapshot, _ := eventStore.GetSnapshot("user-1")
    fmt.Printf("Current snapshot: %+v\n", snapshot)
}
```

## Service Mesh

### 1. Service Mesh Implementation

#### Service Mesh
```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

type ServiceMesh struct {
    services map[string]*Service
    proxies  map[string]*Proxy
    mutex    sync.RWMutex
}

type Service struct {
    Name     string
    Endpoints []string
    Health   bool
    Load     int
}

type Proxy struct {
    ServiceName string
    Port        int
    Rules       []RoutingRule
    CircuitBreaker *CircuitBreaker
}

type RoutingRule struct {
    Path        string
    Destination string
    Weight      int
}

type CircuitBreaker struct {
    FailureThreshold int
    SuccessThreshold int
    Timeout         time.Duration
    State          string // Closed, Open, HalfOpen
    Failures       int
    LastFailure    time.Time
    mutex          sync.RWMutex
}

func NewServiceMesh() *ServiceMesh {
    return &ServiceMesh{
        services: make(map[string]*Service),
        proxies:  make(map[string]*Proxy),
    }
}

func (sm *ServiceMesh) RegisterService(name string, endpoints []string) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    service := &Service{
        Name:      name,
        Endpoints: endpoints,
        Health:    true,
        Load:      0,
    }
    
    sm.services[name] = service
    
    // Create proxy for service
    proxy := &Proxy{
        ServiceName: name,
        Port:        8080 + len(sm.proxies),
        Rules:       make([]RoutingRule, 0),
        CircuitBreaker: &CircuitBreaker{
            FailureThreshold: 5,
            SuccessThreshold: 3,
            Timeout:         30 * time.Second,
            State:          "Closed",
        },
    }
    
    sm.proxies[name] = proxy
    
    fmt.Printf("Registered service %s with %d endpoints\n", name, len(endpoints))
}

func (sm *ServiceMesh) AddRoutingRule(serviceName, path, destination string, weight int) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    proxy, exists := sm.proxies[serviceName]
    if !exists {
        return
    }
    
    rule := RoutingRule{
        Path:        path,
        Destination: destination,
        Weight:      weight,
    }
    
    proxy.Rules = append(proxy.Rules, rule)
    fmt.Printf("Added routing rule for %s: %s -> %s (weight: %d)\n", 
        serviceName, path, destination, weight)
}

func (sm *ServiceMesh) RouteRequest(serviceName, path string) (string, error) {
    sm.mutex.RLock()
    proxy, exists := sm.proxies[serviceName]
    sm.mutex.RUnlock()
    
    if !exists {
        return "", fmt.Errorf("service %s not found", serviceName)
    }
    
    // Check circuit breaker
    if !proxy.CircuitBreaker.CanExecute() {
        return "", fmt.Errorf("circuit breaker open for service %s", serviceName)
    }
    
    // Find matching routing rule
    for _, rule := range proxy.Rules {
        if rule.Path == path {
            // Simulate request routing
            fmt.Printf("Routing request %s to %s\n", path, rule.Destination)
            
            // Simulate success/failure
            if time.Now().UnixNano()%10 < 8 { // 80% success rate
                proxy.CircuitBreaker.RecordSuccess()
                return rule.Destination, nil
            } else {
                proxy.CircuitBreaker.RecordFailure()
                return "", fmt.Errorf("request failed")
            }
        }
    }
    
    return "", fmt.Errorf("no routing rule found for path %s", path)
}

func (cb *CircuitBreaker) CanExecute() bool {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    
    switch cb.State {
    case "Closed":
        return true
    case "Open":
        if time.Since(cb.LastFailure) > cb.Timeout {
            cb.State = "HalfOpen"
            return true
        }
        return false
    case "HalfOpen":
        return true
    default:
        return false
    }
}

func (cb *CircuitBreaker) RecordSuccess() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    if cb.State == "HalfOpen" {
        cb.Failures = 0
        cb.State = "Closed"
    }
}

func (cb *CircuitBreaker) RecordFailure() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    cb.Failures++
    cb.LastFailure = time.Now()
    
    if cb.Failures >= cb.FailureThreshold {
        cb.State = "Open"
    }
}

func main() {
    mesh := NewServiceMesh()
    
    // Register services
    mesh.RegisterService("user-service", []string{"192.168.1.1:8080", "192.168.1.2:8080"})
    mesh.RegisterService("order-service", []string{"192.168.1.3:8080", "192.168.1.4:8080"})
    
    // Add routing rules
    mesh.AddRoutingRule("user-service", "/users", "user-service-v1", 100)
    mesh.AddRoutingRule("order-service", "/orders", "order-service-v1", 100)
    
    // Simulate requests
    for i := 0; i < 10; i++ {
        destination, err := mesh.RouteRequest("user-service", "/users")
        if err != nil {
            fmt.Printf("Request failed: %v\n", err)
        } else {
            fmt.Printf("Request routed to: %s\n", destination)
        }
        time.Sleep(100 * time.Millisecond)
    }
}
```

## Follow-up Questions

### 1. Consensus Algorithms
**Q: What's the difference between Raft and PBFT?**
A: Raft is designed for crash failures and is simpler, while PBFT handles Byzantine failures but is more complex and requires more nodes.

### 2. Distributed Storage
**Q: How does consistent hashing help with distributed systems?**
A: Consistent hashing provides even distribution of data across nodes and minimizes data movement when nodes are added or removed.

### 3. Event Sourcing
**Q: What are the benefits of event sourcing?**
A: Event sourcing provides complete audit trail, enables time travel, supports replay for debugging, and allows for flexible querying of historical data.

## Sources

### Books
- **Designing Data-Intensive Applications** by Martin Kleppmann
- **Distributed Systems** by Andrew Tanenbaum
- **Microservices Patterns** by Chris Richardson

### Online Resources
- **Raft Consensus Algorithm** - Official Raft paper
- **Consistent Hashing** - Distributed systems concepts
- **Event Sourcing** - Martin Fowler's articles

## Projects

### 1. Distributed Key-Value Store
**Objective**: Build a distributed key-value store
**Requirements**: Consistent hashing, replication, fault tolerance
**Deliverables**: Complete distributed storage system

### 2. Event Sourcing System
**Objective**: Implement event sourcing for a domain
**Requirements**: Event store, aggregates, snapshots
**Deliverables**: Event-sourced application with replay capabilities

### 3. Service Mesh Platform
**Objective**: Create a service mesh for microservices
**Requirements**: Service discovery, load balancing, circuit breakers
**Deliverables**: Complete service mesh implementation

---

**Next**: [Machine Learning](./machine-learning/README.md) | **Previous**: [Phase 1](../phase1_intermediate/README.md) | **Up**: [Phase 2](../README.md)

