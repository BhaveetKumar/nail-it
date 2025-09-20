# 🌐 Distributed Systems Engineering Guide

> **Advanced distributed systems patterns and implementations for senior backend engineers**

## 🎯 **Overview**

Distributed systems are the backbone of modern scalable applications. This guide covers essential distributed systems concepts including CAP theorem, consensus algorithms, distributed transactions, consistency patterns, service mesh architecture, and fault tolerance with practical Go implementations.

## 📚 **Table of Contents**

1. [CAP Theorem & Consistency Models](#cap-theorem--consistency-models)
2. [Consensus Algorithms](#consensus-algorithms)
3. [Distributed Transactions](#distributed-transactions)
4. [Data Consistency Patterns](#data-consistency-patterns)
5. [Service Mesh Architecture](#service-mesh-architecture)
6. [Fault Tolerance & Resilience](#fault-tolerance--resilience)
7. [Load Balancing Strategies](#load-balancing-strategies)
8. [Distributed Caching](#distributed-caching)
9. [Event-Driven Architecture](#event-driven-architecture)
10. [Monitoring & Observability](#monitoring--observability)
11. [Interview Questions](#interview-questions)

---

## ⚖️ **CAP Theorem & Consistency Models**

### **CAP Theorem Implementation**

```go
package cap

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "time"
)

// CAP Theorem: Consistency, Availability, Partition Tolerance
// You can only guarantee two out of three in a distributed system

// Consistency Models
type ConsistencyLevel int

const (
    StrongConsistency ConsistencyLevel = iota
    EventualConsistency
    WeakConsistency
    BoundedStaleness
    MonotonicRead
    MonotonicWrite
    ReadYourWrites
    WritesFollowReads
)

// Distributed Node Interface
type DistributedNode interface {
    GetID() string
    IsHealthy() bool
    Write(ctx context.Context, key string, value interface{}) error
    Read(ctx context.Context, key string) (interface{}, error)
    Replicate(ctx context.Context, key string, value interface{}, nodes []DistributedNode) error
}

// CP System (Consistency + Partition Tolerance)
// Sacrifices Availability during network partitions
type CPSystem struct {
    nodes           []DistributedNode
    quorumSize      int
    mu              sync.RWMutex
    partitionedNodes map[string]bool
}

func NewCPSystem(nodes []DistributedNode) *CPSystem {
    return &CPSystem{
        nodes:           nodes,
        quorumSize:      len(nodes)/2 + 1, // Majority quorum
        partitionedNodes: make(map[string]bool),
    }
}

// Write with strong consistency (requires quorum)
func (cp *CPSystem) Write(ctx context.Context, key string, value interface{}) error {
    availableNodes := cp.getAvailableNodes()
    
    if len(availableNodes) < cp.quorumSize {
        return errors.New("insufficient nodes for quorum - system unavailable")
    }
    
    // Write to quorum of nodes
    successCount := 0
    var lastError error
    
    for _, node := range availableNodes[:cp.quorumSize] {
        if err := node.Write(ctx, key, value); err != nil {
            lastError = err
            continue
        }
        successCount++
    }
    
    if successCount >= cp.quorumSize {
        return nil
    }
    
    return fmt.Errorf("failed to achieve write quorum: %w", lastError)
}

// Read with strong consistency (requires quorum)
func (cp *CPSystem) Read(ctx context.Context, key string) (interface{}, error) {
    availableNodes := cp.getAvailableNodes()
    
    if len(availableNodes) < cp.quorumSize {
        return nil, errors.New("insufficient nodes for quorum - system unavailable")
    }
    
    // Read from quorum and ensure consistency
    values := make(map[interface{}]int)
    var lastError error
    
    for _, node := range availableNodes[:cp.quorumSize] {
        value, err := node.Read(ctx, key)
        if err != nil {
            lastError = err
            continue
        }
        values[value]++
    }
    
    // Return most common value (should be same in strongly consistent system)
    for value, count := range values {
        if count >= cp.quorumSize/2+1 {
            return value, nil
        }
    }
    
    return nil, fmt.Errorf("failed to achieve read consistency: %w", lastError)
}

func (cp *CPSystem) getAvailableNodes() []DistributedNode {
    cp.mu.RLock()
    defer cp.mu.RUnlock()
    
    var available []DistributedNode
    for _, node := range cp.nodes {
        if !cp.partitionedNodes[node.GetID()] && node.IsHealthy() {
            available = append(available, node)
        }
    }
    return available
}

// AP System (Availability + Partition Tolerance)  
// Sacrifices Consistency for availability during partitions
type APSystem struct {
    nodes       []DistributedNode
    mu          sync.RWMutex
    vectorClock map[string]int64
}

func NewAPSystem(nodes []DistributedNode) *APSystem {
    return &APSystem{
        nodes:       nodes,
        vectorClock: make(map[string]int64),
    }
}

// Write to any available node (always available)
func (ap *APSystem) Write(ctx context.Context, key string, value interface{}) error {
    availableNodes := ap.getAvailableNodes()
    
    if len(availableNodes) == 0 {
        return errors.New("no nodes available")
    }
    
    // Write to first available node
    node := availableNodes[0]
    if err := node.Write(ctx, key, value); err != nil {
        return err
    }
    
    // Increment vector clock
    ap.mu.Lock()
    ap.vectorClock[node.GetID()]++
    ap.mu.Unlock()
    
    // Asynchronously replicate to other nodes (best effort)
    go ap.asyncReplicate(ctx, key, value, availableNodes[1:])
    
    return nil
}

// Read from any available node (may return stale data)
func (ap *APSystem) Read(ctx context.Context, key string) (interface{}, error) {
    availableNodes := ap.getAvailableNodes()
    
    if len(availableNodes) == 0 {
        return nil, errors.New("no nodes available")
    }
    
    // Read from first available node
    return availableNodes[0].Read(ctx, key)
}

func (ap *APSystem) asyncReplicate(ctx context.Context, key string, value interface{}, nodes []DistributedNode) {
    for _, node := range nodes {
        go func(n DistributedNode) {
            // Best effort replication with timeout
            replicaCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
            defer cancel()
            
            n.Write(replicaCtx, key, value)
        }(node)
    }
}

func (ap *APSystem) getAvailableNodes() []DistributedNode {
    var available []DistributedNode
    for _, node := range ap.nodes {
        if node.IsHealthy() {
            available = append(available, node)
        }
    }
    return available
}

// Eventually Consistent System with Conflict Resolution
type EventuallyConsistentSystem struct {
    nodes               []DistributedNode
    conflictResolver    ConflictResolver
    gossipInterval      time.Duration
    mu                  sync.RWMutex
    pendingUpdates      map[string][]Update
}

type Update struct {
    Key       string      `json:"key"`
    Value     interface{} `json:"value"`
    Timestamp time.Time   `json:"timestamp"`
    NodeID    string      `json:"node_id"`
    Version   int64       `json:"version"`
}

type ConflictResolver interface {
    Resolve(updates []Update) Update
}

// Last Write Wins resolver
type LWWResolver struct{}

func (r *LWWResolver) Resolve(updates []Update) Update {
    if len(updates) == 0 {
        return Update{}
    }
    
    latest := updates[0]
    for _, update := range updates[1:] {
        if update.Timestamp.After(latest.Timestamp) {
            latest = update
        }
    }
    return latest
}

func NewEventuallyConsistentSystem(nodes []DistributedNode, resolver ConflictResolver) *EventuallyConsistentSystem {
    ecs := &EventuallyConsistentSystem{
        nodes:            nodes,
        conflictResolver: resolver,
        gossipInterval:   1 * time.Second,
        pendingUpdates:   make(map[string][]Update),
    }
    
    // Start gossip protocol
    go ecs.startGossipProtocol()
    
    return ecs
}

func (ecs *EventuallyConsistentSystem) Write(ctx context.Context, key string, value interface{}) error {
    // Write to local node immediately
    update := Update{
        Key:       key,
        Value:     value,
        Timestamp: time.Now(),
        NodeID:    "local", // Would be actual node ID
        Version:   time.Now().UnixNano(),
    }
    
    ecs.mu.Lock()
    ecs.pendingUpdates[key] = append(ecs.pendingUpdates[key], update)
    ecs.mu.Unlock()
    
    // Write to first available node
    availableNodes := ecs.getAvailableNodes()
    if len(availableNodes) > 0 {
        return availableNodes[0].Write(ctx, key, value)
    }
    
    return errors.New("no nodes available")
}

func (ecs *EventuallyConsistentSystem) Read(ctx context.Context, key string) (interface{}, error) {
    // Read from any available node
    availableNodes := ecs.getAvailableNodes()
    if len(availableNodes) == 0 {
        return nil, errors.New("no nodes available")
    }
    
    return availableNodes[0].Read(ctx, key)
}

// Gossip protocol for eventual consistency
func (ecs *EventuallyConsistentSystem) startGossipProtocol() {
    ticker := time.NewTicker(ecs.gossipInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        ecs.performGossip()
    }
}

func (ecs *EventuallyConsistentSystem) performGossip() {
    ecs.mu.Lock()
    defer ecs.mu.Unlock()
    
    // Resolve conflicts for each key
    for key, updates := range ecs.pendingUpdates {
        if len(updates) > 1 {
            resolved := ecs.conflictResolver.Resolve(updates)
            ecs.pendingUpdates[key] = []Update{resolved}
            
            // Propagate resolved value to all nodes
            go ecs.propagateUpdate(key, resolved)
        }
    }
}

func (ecs *EventuallyConsistentSystem) propagateUpdate(key string, update Update) {
    availableNodes := ecs.getAvailableNodes()
    
    for _, node := range availableNodes {
        go func(n DistributedNode) {
            ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
            defer cancel()
            
            n.Write(ctx, key, update.Value)
        }(node)
    }
}

func (ecs *EventuallyConsistentSystem) getAvailableNodes() []DistributedNode {
    var available []DistributedNode
    for _, node := range ecs.nodes {
        if node.IsHealthy() {
            available = append(available, node)
        }
    }
    return available
}

// Vector Clock for causal consistency
type VectorClock struct {
    clocks map[string]int64
    mu     sync.RWMutex
}

func NewVectorClock(nodeIDs []string) *VectorClock {
    clocks := make(map[string]int64)
    for _, id := range nodeIDs {
        clocks[id] = 0
    }
    
    return &VectorClock{
        clocks: clocks,
    }
}

func (vc *VectorClock) Increment(nodeID string) {
    vc.mu.Lock()
    defer vc.mu.Unlock()
    vc.clocks[nodeID]++
}

func (vc *VectorClock) Update(other *VectorClock) {
    vc.mu.Lock()
    defer vc.mu.Unlock()
    
    other.mu.RLock()
    defer other.mu.RUnlock()
    
    for nodeID, clock := range other.clocks {
        if vc.clocks[nodeID] < clock {
            vc.clocks[nodeID] = clock
        }
    }
}

func (vc *VectorClock) Compare(other *VectorClock) string {
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    
    other.mu.RLock()
    defer other.mu.RUnlock()
    
    less := false
    greater := false
    
    for nodeID := range vc.clocks {
        if vc.clocks[nodeID] < other.clocks[nodeID] {
            less = true
        } else if vc.clocks[nodeID] > other.clocks[nodeID] {
            greater = true
        }
    }
    
    if less && !greater {
        return "before"
    } else if greater && !less {
        return "after"
    } else if !less && !greater {
        return "equal"
    } else {
        return "concurrent"
    }
}

func (vc *VectorClock) Copy() *VectorClock {
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    
    newClocks := make(map[string]int64)
    for k, v := range vc.clocks {
        newClocks[k] = v
    }
    
    return &VectorClock{clocks: newClocks}
}
```

---

## 🤝 **Consensus Algorithms**

### **Raft Consensus Implementation**

```go
package raft

import (
    "context"
    "fmt"
    "log"
    "math/rand"
    "sync"
    "sync/atomic"
    "time"
)

// Raft Node States
type NodeState int

const (
    Follower NodeState = iota
    Candidate
    Leader
)

func (s NodeState) String() string {
    switch s {
    case Follower:
        return "Follower"
    case Candidate:
        return "Candidate" 
    case Leader:
        return "Leader"
    default:
        return "Unknown"
    }
}

// Log Entry
type LogEntry struct {
    Term    int64       `json:"term"`
    Index   int64       `json:"index"`
    Command interface{} `json:"command"`
}

// Raft Node
type RaftNode struct {
    // Persistent state
    currentTerm int64
    votedFor    string
    log         []LogEntry
    
    // Volatile state
    commitIndex int64
    lastApplied int64
    
    // Leader state
    nextIndex  map[string]int64
    matchIndex map[string]int64
    
    // Node configuration
    id       string
    peers    []string
    state    NodeState
    
    // Timers and channels
    electionTimer  *time.Timer
    heartbeatTimer *time.Timer
    voteCh         chan bool
    appendCh       chan bool
    
    // Synchronization
    mu sync.RWMutex
    
    // Communication
    transport Transport
    
    // State machine
    stateMachine StateMachine
    
    // Metrics
    leadershipChanges int64
    electionTimeouts  int64
    logEntries        int64
}

// Transport interface for network communication
type Transport interface {
    SendRequestVote(ctx context.Context, target string, req *RequestVoteArgs) (*RequestVoteReply, error)
    SendAppendEntries(ctx context.Context, target string, req *AppendEntriesArgs) (*AppendEntriesReply, error)
}

// State machine interface
type StateMachine interface {
    Apply(command interface{}) interface{}
}

// RequestVote RPC structures
type RequestVoteArgs struct {
    Term         int64  `json:"term"`
    CandidateID  string `json:"candidate_id"`
    LastLogIndex int64  `json:"last_log_index"`
    LastLogTerm  int64  `json:"last_log_term"`
}

type RequestVoteReply struct {
    Term        int64 `json:"term"`
    VoteGranted bool  `json:"vote_granted"`
}

// AppendEntries RPC structures
type AppendEntriesArgs struct {
    Term         int64      `json:"term"`
    LeaderID     string     `json:"leader_id"`
    PrevLogIndex int64      `json:"prev_log_index"`
    PrevLogTerm  int64      `json:"prev_log_term"`
    Entries      []LogEntry `json:"entries"`
    LeaderCommit int64      `json:"leader_commit"`
}

type AppendEntriesReply struct {
    Term          int64 `json:"term"`
    Success       bool  `json:"success"`
    ConflictIndex int64 `json:"conflict_index"`
    ConflictTerm  int64 `json:"conflict_term"`
}

func NewRaftNode(id string, peers []string, transport Transport, stateMachine StateMachine) *RaftNode {
    node := &RaftNode{
        id:           id,
        peers:        peers,
        state:        Follower,
        transport:    transport,
        stateMachine: stateMachine,
        voteCh:       make(chan bool, len(peers)),
        appendCh:     make(chan bool, 1),
        nextIndex:    make(map[string]int64),
        matchIndex:   make(map[string]int64),
        log:          make([]LogEntry, 1), // Start with dummy entry at index 0
    }
    
    // Initialize log with dummy entry
    node.log[0] = LogEntry{Term: 0, Index: 0}
    
    return node
}

// Start the Raft node
func (rn *RaftNode) Start() {
    go rn.run()
}

func (rn *RaftNode) run() {
    rn.becomeFollower(0)
    
    for {
        switch rn.getState() {
        case Follower:
            rn.runFollower()
        case Candidate:
            rn.runCandidate()
        case Leader:
            rn.runLeader()
        }
    }
}

// Follower behavior
func (rn *RaftNode) runFollower() {
    rn.resetElectionTimer()
    
    for rn.getState() == Follower {
        select {
        case <-rn.electionTimer.C:
            rn.becomeCandidate()
        case <-rn.appendCh:
            rn.resetElectionTimer()
        }
    }
}

// Candidate behavior
func (rn *RaftNode) runCandidate() {
    rn.mu.Lock()
    rn.currentTerm++
    rn.votedFor = rn.id
    atomic.AddInt64(&rn.electionTimeouts, 1)
    rn.mu.Unlock()
    
    rn.resetElectionTimer()
    votes := 1 // Vote for self
    
    // Send RequestVote RPCs to all peers
    for _, peer := range rn.peers {
        go func(p string) {
            if rn.sendRequestVote(p) {
                rn.voteCh <- true
            } else {
                rn.voteCh <- false
            }
        }(peer)
    }
    
    // Count votes
    for rn.getState() == Candidate {
        select {
        case vote := <-rn.voteCh:
            if vote {
                votes++
                if votes > len(rn.peers)/2 {
                    rn.becomeLeader()
                    return
                }
            }
        case <-rn.electionTimer.C:
            // Election timeout, start new election
            return
        case <-rn.appendCh:
            // Received valid AppendEntries, become follower
            rn.becomeFollower(rn.getCurrentTerm())
            return
        }
    }
}

// Leader behavior
func (rn *RaftNode) runLeader() {
    rn.initLeaderState()
    
    // Send initial heartbeats
    rn.sendHeartbeats()
    rn.resetHeartbeatTimer()
    
    for rn.getState() == Leader {
        select {
        case <-rn.heartbeatTimer.C:
            rn.sendHeartbeats()
            rn.resetHeartbeatTimer()
        case <-rn.appendCh:
            // Step down if we receive a higher term
            rn.becomeFollower(rn.getCurrentTerm())
            return
        }
    }
}

// RequestVote RPC handler
func (rn *RaftNode) RequestVote(args *RequestVoteArgs) *RequestVoteReply {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    reply := &RequestVoteReply{
        Term:        rn.currentTerm,
        VoteGranted: false,
    }
    
    // Reply false if term < currentTerm
    if args.Term < rn.currentTerm {
        return reply
    }
    
    // If RPC request contains term T > currentTerm, set currentTerm = T and convert to follower
    if args.Term > rn.currentTerm {
        rn.currentTerm = args.Term
        rn.votedFor = ""
        if rn.state != Follower {
            rn.state = Follower
        }
    }
    
    reply.Term = rn.currentTerm
    
    // Grant vote if we haven't voted for anyone else and candidate's log is up-to-date
    if (rn.votedFor == "" || rn.votedFor == args.CandidateID) && rn.isLogUpToDate(args.LastLogIndex, args.LastLogTerm) {
        rn.votedFor = args.CandidateID
        reply.VoteGranted = true
        
        // Reset election timer when granting a vote
        rn.resetElectionTimer()
    }
    
    return reply
}

// AppendEntries RPC handler
func (rn *RaftNode) AppendEntries(args *AppendEntriesArgs) *AppendEntriesReply {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    reply := &AppendEntriesReply{
        Term:    rn.currentTerm,
        Success: false,
    }
    
    // Reply false if term < currentTerm
    if args.Term < rn.currentTerm {
        return reply
    }
    
    // Update term and convert to follower if necessary
    if args.Term > rn.currentTerm {
        rn.currentTerm = args.Term
        rn.votedFor = ""
    }
    
    if rn.state != Follower {
        rn.state = Follower
    }
    
    reply.Term = rn.currentTerm
    
    // Signal that we received a valid AppendEntries
    select {
    case rn.appendCh <- true:
    default:
    }
    
    // Reply false if log doesn't contain an entry at prevLogIndex whose term matches prevLogTerm
    if args.PrevLogIndex >= int64(len(rn.log)) {
        reply.ConflictIndex = int64(len(rn.log))
        reply.ConflictTerm = -1
        return reply
    }
    
    if args.PrevLogIndex > 0 && rn.log[args.PrevLogIndex].Term != args.PrevLogTerm {
        reply.ConflictTerm = rn.log[args.PrevLogIndex].Term
        reply.ConflictIndex = rn.findFirstIndex(reply.ConflictTerm)
        return reply
    }
    
    // If an existing entry conflicts with a new one, delete the existing entry and all that follow
    for i, entry := range args.Entries {
        index := args.PrevLogIndex + 1 + int64(i)
        if index < int64(len(rn.log)) {
            if rn.log[index].Term != entry.Term {
                rn.log = rn.log[:index]
                break
            }
        }
    }
    
    // Append any new entries not already in the log
    for i, entry := range args.Entries {
        index := args.PrevLogIndex + 1 + int64(i)
        if index >= int64(len(rn.log)) {
            rn.log = append(rn.log, entry)
            atomic.AddInt64(&rn.logEntries, 1)
        }
    }
    
    // If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry)
    if args.LeaderCommit > rn.commitIndex {
        rn.commitIndex = min(args.LeaderCommit, int64(len(rn.log)-1))
        rn.applyEntries()
    }
    
    reply.Success = true
    return reply
}

// Client command submission (only for leaders)
func (rn *RaftNode) SubmitCommand(command interface{}) error {
    if rn.getState() != Leader {
        return errors.New("not the leader")
    }
    
    rn.mu.Lock()
    entry := LogEntry{
        Term:    rn.currentTerm,
        Index:   int64(len(rn.log)),
        Command: command,
    }
    rn.log = append(rn.log, entry)
    atomic.AddInt64(&rn.logEntries, 1)
    rn.mu.Unlock()
    
    // Replicate to followers immediately
    go rn.sendHeartbeats()
    
    return nil
}

// Helper methods
func (rn *RaftNode) sendRequestVote(peer string) bool {
    rn.mu.RLock()
    args := &RequestVoteArgs{
        Term:         rn.currentTerm,
        CandidateID:  rn.id,
        LastLogIndex: int64(len(rn.log) - 1),
        LastLogTerm:  rn.log[len(rn.log)-1].Term,
    }
    rn.mu.RUnlock()
    
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()
    
    reply, err := rn.transport.SendRequestVote(ctx, peer, args)
    if err != nil {
        return false
    }
    
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    if reply.Term > rn.currentTerm {
        rn.currentTerm = reply.Term
        rn.votedFor = ""
        if rn.state != Follower {
            rn.state = Follower
        }
    }
    
    return reply.VoteGranted
}

func (rn *RaftNode) sendHeartbeats() {
    for _, peer := range rn.peers {
        go rn.sendAppendEntries(peer)
    }
}

func (rn *RaftNode) sendAppendEntries(peer string) {
    rn.mu.RLock()
    if rn.state != Leader {
        rn.mu.RUnlock()
        return
    }
    
    prevLogIndex := rn.nextIndex[peer] - 1
    prevLogTerm := int64(0)
    if prevLogIndex > 0 {
        prevLogTerm = rn.log[prevLogIndex].Term
    }
    
    entries := make([]LogEntry, 0)
    if rn.nextIndex[peer] < int64(len(rn.log)) {
        entries = rn.log[rn.nextIndex[peer]:]
    }
    
    args := &AppendEntriesArgs{
        Term:         rn.currentTerm,
        LeaderID:     rn.id,
        PrevLogIndex: prevLogIndex,
        PrevLogTerm:  prevLogTerm,
        Entries:      entries,
        LeaderCommit: rn.commitIndex,
    }
    rn.mu.RUnlock()
    
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()
    
    reply, err := rn.transport.SendAppendEntries(ctx, peer, args)
    if err != nil {
        return
    }
    
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    if reply.Term > rn.currentTerm {
        rn.currentTerm = reply.Term
        rn.votedFor = ""
        rn.state = Follower
        return
    }
    
    if rn.state != Leader || args.Term != rn.currentTerm {
        return
    }
    
    if reply.Success {
        rn.nextIndex[peer] = args.PrevLogIndex + int64(len(args.Entries)) + 1
        rn.matchIndex[peer] = rn.nextIndex[peer] - 1
        rn.updateCommitIndex()
    } else {
        // Decrement nextIndex and retry
        if reply.ConflictTerm != -1 {
            // Find last entry with ConflictTerm
            lastIndex := rn.findLastIndex(reply.ConflictTerm)
            if lastIndex != -1 {
                rn.nextIndex[peer] = lastIndex + 1
            } else {
                rn.nextIndex[peer] = reply.ConflictIndex
            }
        } else {
            rn.nextIndex[peer] = reply.ConflictIndex
        }
        
        if rn.nextIndex[peer] < 1 {
            rn.nextIndex[peer] = 1
        }
    }
}

func (rn *RaftNode) updateCommitIndex() {
    // Find highest N such that majority of matchIndex[i] >= N and log[N].term == currentTerm
    for n := int64(len(rn.log) - 1); n > rn.commitIndex; n-- {
        if rn.log[n].Term == rn.currentTerm {
            count := 1 // Count self
            for peer := range rn.matchIndex {
                if rn.matchIndex[peer] >= n {
                    count++
                }
            }
            
            if count > len(rn.peers)/2 {
                rn.commitIndex = n
                rn.applyEntries()
                break
            }
        }
    }
}

func (rn *RaftNode) applyEntries() {
    for rn.lastApplied < rn.commitIndex {
        rn.lastApplied++
        entry := rn.log[rn.lastApplied]
        rn.stateMachine.Apply(entry.Command)
    }
}

func (rn *RaftNode) isLogUpToDate(lastLogIndex, lastLogTerm int64) bool {
    ourLastIndex := int64(len(rn.log) - 1)
    ourLastTerm := rn.log[ourLastIndex].Term
    
    return lastLogTerm > ourLastTerm || (lastLogTerm == ourLastTerm && lastLogIndex >= ourLastIndex)
}

func (rn *RaftNode) findFirstIndex(term int64) int64 {
    for i := int64(1); i < int64(len(rn.log)); i++ {
        if rn.log[i].Term == term {
            return i
        }
    }
    return -1
}

func (rn *RaftNode) findLastIndex(term int64) int64 {
    for i := int64(len(rn.log) - 1); i >= 1; i-- {
        if rn.log[i].Term == term {
            return i
        }
    }
    return -1
}

func (rn *RaftNode) becomeFollower(term int64) {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    if rn.state == Leader {
        atomic.AddInt64(&rn.leadershipChanges, 1)
    }
    
    rn.state = Follower
    rn.currentTerm = term
    rn.votedFor = ""
    
    if rn.heartbeatTimer != nil {
        rn.heartbeatTimer.Stop()
    }
}

func (rn *RaftNode) becomeCandidate() {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    rn.state = Candidate
    
    if rn.heartbeatTimer != nil {
        rn.heartbeatTimer.Stop()
    }
}

func (rn *RaftNode) becomeLeader() {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    if rn.state == Candidate {
        atomic.AddInt64(&rn.leadershipChanges, 1)
    }
    
    rn.state = Leader
    
    if rn.electionTimer != nil {
        rn.electionTimer.Stop()
    }
}

func (rn *RaftNode) initLeaderState() {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    lastLogIndex := int64(len(rn.log))
    for _, peer := range rn.peers {
        rn.nextIndex[peer] = lastLogIndex
        rn.matchIndex[peer] = 0
    }
}

func (rn *RaftNode) resetElectionTimer() {
    timeout := time.Duration(150+rand.Intn(150)) * time.Millisecond
    if rn.electionTimer != nil {
        rn.electionTimer.Stop()
    }
    rn.electionTimer = time.NewTimer(timeout)
}

func (rn *RaftNode) resetHeartbeatTimer() {
    timeout := 50 * time.Millisecond
    if rn.heartbeatTimer != nil {
        rn.heartbeatTimer.Stop()
    }
    rn.heartbeatTimer = time.NewTimer(timeout)
}

func (rn *RaftNode) getState() NodeState {
    rn.mu.RLock()
    defer rn.mu.RUnlock()
    return rn.state
}

func (rn *RaftNode) getCurrentTerm() int64 {
    rn.mu.RLock()
    defer rn.mu.RUnlock()
    return rn.currentTerm
}

func (rn *RaftNode) GetMetrics() map[string]int64 {
    return map[string]int64{
        "leadership_changes": atomic.LoadInt64(&rn.leadershipChanges),
        "election_timeouts":  atomic.LoadInt64(&rn.electionTimeouts),
        "log_entries":        atomic.LoadInt64(&rn.logEntries),
        "current_term":       rn.getCurrentTerm(),
        "commit_index":       rn.commitIndex,
    }
}

func min(a, b int64) int64 {
    if a < b {
        return a
    }
    return b
}
```

---

## 💳 **Distributed Transactions**

### **Two-Phase Commit Implementation**

```go
package twophase

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "time"
)

// Transaction States
type TransactionState int

const (
    TransactionPending TransactionState = iota
    TransactionPrepared
    TransactionCommitted
    TransactionAborted
)

func (ts TransactionState) String() string {
    switch ts {
    case TransactionPending:
        return "Pending"
    case TransactionPrepared:
        return "Prepared"
    case TransactionCommitted:
        return "Committed"
    case TransactionAborted:
        return "Aborted"
    default:
        return "Unknown"
    }
}

// Transaction Participant Interface
type Participant interface {
    GetID() string
    Prepare(ctx context.Context, txnID string, operations []Operation) error
    Commit(ctx context.Context, txnID string) error
    Abort(ctx context.Context, txnID string) error
    IsHealthy() bool
}

// Operation represents a single database operation
type Operation struct {
    Type  string      `json:"type"`  // INSERT, UPDATE, DELETE
    Table string      `json:"table"`
    Key   string      `json:"key"`
    Value interface{} `json:"value"`
}

// Transaction Coordinator (implements 2PC protocol)
type TransactionCoordinator struct {
    id           string
    participants map[string]Participant
    transactions map[string]*Transaction
    mu           sync.RWMutex
    logger       Logger
}

type Transaction struct {
    ID           string                      `json:"id"`
    State        TransactionState           `json:"state"`
    Operations   []Operation                `json:"operations"`
    Participants map[string]Participant     `json:"-"`
    Prepared     map[string]bool            `json:"prepared"`
    StartTime    time.Time                  `json:"start_time"`
    Timeout      time.Duration              `json:"timeout"`
    mu           sync.RWMutex
}

type Logger interface {
    Info(msg string, fields ...interface{})
    Error(msg string, fields ...interface{})
    Warn(msg string, fields ...interface{})
}

func NewTransactionCoordinator(id string, logger Logger) *TransactionCoordinator {
    return &TransactionCoordinator{
        id:           id,
        participants: make(map[string]Participant),
        transactions: make(map[string]*Transaction),
        logger:       logger,
    }
}

func (tc *TransactionCoordinator) RegisterParticipant(participant Participant) {
    tc.mu.Lock()
    defer tc.mu.Unlock()
    tc.participants[participant.GetID()] = participant
}

// Begin a new distributed transaction
func (tc *TransactionCoordinator) BeginTransaction(txnID string, operations []Operation, timeout time.Duration) error {
    tc.mu.Lock()
    defer tc.mu.Unlock()
    
    if _, exists := tc.transactions[txnID]; exists {
        return fmt.Errorf("transaction %s already exists", txnID)
    }
    
    // Determine which participants are involved
    participantMap := make(map[string]Participant)
    for _, op := range operations {
        // Simple routing logic - in practice, this would be more sophisticated
        for participantID, participant := range tc.participants {
            if tc.isParticipantInvolved(participant, op) {
                participantMap[participantID] = participant
            }
        }
    }
    
    txn := &Transaction{
        ID:           txnID,
        State:        TransactionPending,
        Operations:   operations,
        Participants: participantMap,
        Prepared:     make(map[string]bool),
        StartTime:    time.Now(),
        Timeout:      timeout,
    }
    
    tc.transactions[txnID] = txn
    tc.logger.Info("Transaction started", "txn_id", txnID, "participants", len(participantMap))
    
    return nil
}

// Execute distributed transaction using 2PC
func (tc *TransactionCoordinator) ExecuteTransaction(ctx context.Context, txnID string) error {
    tc.mu.RLock()
    txn, exists := tc.transactions[txnID]
    tc.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("transaction %s not found", txnID)
    }
    
    // Check if transaction has timed out
    if time.Since(txn.StartTime) > txn.Timeout {
        tc.abortTransaction(ctx, txn)
        return fmt.Errorf("transaction %s timed out", txnID)
    }
    
    // Phase 1: Prepare
    if err := tc.preparePhase(ctx, txn); err != nil {
        tc.logger.Error("Prepare phase failed", "txn_id", txnID, "error", err)
        tc.abortTransaction(ctx, txn)
        return fmt.Errorf("prepare phase failed: %w", err)
    }
    
    // Phase 2: Commit
    if err := tc.commitPhase(ctx, txn); err != nil {
        tc.logger.Error("Commit phase failed", "txn_id", txnID, "error", err)
        // At this point, we must keep trying to commit since all participants are prepared
        go tc.recoverCommit(ctx, txn)
        return fmt.Errorf("commit phase failed: %w", err)
    }
    
    tc.logger.Info("Transaction committed successfully", "txn_id", txnID)
    return nil
}

// Phase 1: Send prepare requests to all participants
func (tc *TransactionCoordinator) preparePhase(ctx context.Context, txn *Transaction) error {
    txn.mu.Lock()
    txn.State = TransactionPending
    txn.mu.Unlock()
    
    // Send prepare to all participants concurrently
    var wg sync.WaitGroup
    errorCh := make(chan error, len(txn.Participants))
    
    for participantID, participant := range txn.Participants {
        wg.Add(1)
        go func(id string, p Participant) {
            defer wg.Done()
            
            // Filter operations for this participant
            participantOps := tc.getParticipantOperations(p, txn.Operations)
            
            prepareCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
            defer cancel()
            
            if err := p.Prepare(prepareCtx, txn.ID, participantOps); err != nil {
                errorCh <- fmt.Errorf("participant %s prepare failed: %w", id, err)
                return
            }
            
            txn.mu.Lock()
            txn.Prepared[id] = true
            txn.mu.Unlock()
            
            tc.logger.Info("Participant prepared", "txn_id", txn.ID, "participant", id)
        }(participantID, participant)
    }
    
    wg.Wait()
    close(errorCh)
    
    // Check for any prepare errors
    for err := range errorCh {
        return err
    }
    
    txn.mu.Lock()
    txn.State = TransactionPrepared
    txn.mu.Unlock()
    
    tc.logger.Info("All participants prepared", "txn_id", txn.ID)
    return nil
}

// Phase 2: Send commit requests to all participants
func (tc *TransactionCoordinator) commitPhase(ctx context.Context, txn *Transaction) error {
    var wg sync.WaitGroup
    errorCh := make(chan error, len(txn.Participants))
    
    for participantID, participant := range txn.Participants {
        wg.Add(1)
        go func(id string, p Participant) {
            defer wg.Done()
            
            commitCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
            defer cancel()
            
            if err := p.Commit(commitCtx, txn.ID); err != nil {
                errorCh <- fmt.Errorf("participant %s commit failed: %w", id, err)
                return
            }
            
            tc.logger.Info("Participant committed", "txn_id", txn.ID, "participant", id)
        }(participantID, participant)
    }
    
    wg.Wait()
    close(errorCh)
    
    // Check for commit errors
    var commitErrors []error
    for err := range errorCh {
        commitErrors = append(commitErrors, err)
    }
    
    if len(commitErrors) > 0 {
        return fmt.Errorf("commit failures: %v", commitErrors)
    }
    
    txn.mu.Lock()
    txn.State = TransactionCommitted
    txn.mu.Unlock()
    
    // Clean up transaction
    tc.mu.Lock()
    delete(tc.transactions, txn.ID)
    tc.mu.Unlock()
    
    return nil
}

// Abort transaction
func (tc *TransactionCoordinator) abortTransaction(ctx context.Context, txn *Transaction) {
    var wg sync.WaitGroup
    
    for participantID, participant := range txn.Participants {
        wg.Add(1)
        go func(id string, p Participant) {
            defer wg.Done()
            
            abortCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
            defer cancel()
            
            if err := p.Abort(abortCtx, txn.ID); err != nil {
                tc.logger.Error("Participant abort failed", "txn_id", txn.ID, "participant", id, "error", err)
            } else {
                tc.logger.Info("Participant aborted", "txn_id", txn.ID, "participant", id)
            }
        }(participantID, participant)
    }
    
    wg.Wait()
    
    txn.mu.Lock()
    txn.State = TransactionAborted
    txn.mu.Unlock()
    
    // Clean up transaction
    tc.mu.Lock()
    delete(tc.transactions, txn.ID)
    tc.mu.Unlock()
    
    tc.logger.Info("Transaction aborted", "txn_id", txn.ID)
}

// Recovery mechanism for commit phase failures
func (tc *TransactionCoordinator) recoverCommit(ctx context.Context, txn *Transaction) {
    maxRetries := 10
    backoff := time.Second
    
    for i := 0; i < maxRetries; i++ {
        if err := tc.commitPhase(ctx, txn); err == nil {
            tc.logger.Info("Transaction recovery successful", "txn_id", txn.ID, "attempt", i+1)
            return
        }
        
        tc.logger.Warn("Transaction recovery attempt failed", "txn_id", txn.ID, "attempt", i+1)
        time.Sleep(backoff)
        backoff *= 2 // Exponential backoff
    }
    
    tc.logger.Error("Transaction recovery failed after max retries", "txn_id", txn.ID)
}

// Helper methods
func (tc *TransactionCoordinator) isParticipantInvolved(participant Participant, operation Operation) bool {
    // Simple routing logic - in practice, this would consider data sharding, etc.
    return true // For now, assume all participants are involved
}

func (tc *TransactionCoordinator) getParticipantOperations(participant Participant, operations []Operation) []Operation {
    // Filter operations relevant to this participant
    // In practice, this would consider data distribution, sharding keys, etc.
    return operations
}

// Get transaction status
func (tc *TransactionCoordinator) GetTransactionStatus(txnID string) (TransactionState, error) {
    tc.mu.RLock()
    defer tc.mu.RUnlock()
    
    txn, exists := tc.transactions[txnID]
    if !exists {
        return TransactionAborted, fmt.Errorf("transaction %s not found", txnID)
    }
    
    txn.mu.RLock()
    defer txn.mu.RUnlock()
    
    return txn.State, nil
}

// Saga Pattern Implementation (alternative to 2PC)
type SagaCoordinator struct {
    id           string
    transactions map[string]*SagaTransaction
    mu           sync.RWMutex
    logger       Logger
}

type SagaTransaction struct {
    ID               string                 `json:"id"`
    Steps            []SagaStep            `json:"steps"`
    CurrentStep      int                   `json:"current_step"`
    State            TransactionState      `json:"state"`
    CompletedSteps   []int                 `json:"completed_steps"`
    CompensatedSteps []int                 `json:"compensated_steps"`
    StartTime        time.Time             `json:"start_time"`
}

type SagaStep struct {
    ID           string                 `json:"id"`
    Service      string                 `json:"service"`
    Action       func(ctx context.Context) error `json:"-"`
    Compensation func(ctx context.Context) error `json:"-"`
    Timeout      time.Duration          `json:"timeout"`
}

func NewSagaCoordinator(id string, logger Logger) *SagaCoordinator {
    return &SagaCoordinator{
        id:           id,
        transactions: make(map[string]*SagaTransaction),
        logger:       logger,
    }
}

func (sc *SagaCoordinator) BeginSaga(txnID string, steps []SagaStep) error {
    sc.mu.Lock()
    defer sc.mu.Unlock()
    
    if _, exists := sc.transactions[txnID]; exists {
        return fmt.Errorf("saga %s already exists", txnID)
    }
    
    saga := &SagaTransaction{
        ID:        txnID,
        Steps:     steps,
        State:     TransactionPending,
        StartTime: time.Now(),
    }
    
    sc.transactions[txnID] = saga
    sc.logger.Info("Saga started", "saga_id", txnID, "steps", len(steps))
    
    return nil
}

func (sc *SagaCoordinator) ExecuteSaga(ctx context.Context, txnID string) error {
    sc.mu.RLock()
    saga, exists := sc.transactions[txnID]
    sc.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("saga %s not found", txnID)
    }
    
    // Execute steps sequentially
    for i, step := range saga.Steps {
        saga.CurrentStep = i
        
        stepCtx, cancel := context.WithTimeout(ctx, step.Timeout)
        
        sc.logger.Info("Executing saga step", "saga_id", txnID, "step", step.ID)
        
        if err := step.Action(stepCtx); err != nil {
            cancel()
            sc.logger.Error("Saga step failed", "saga_id", txnID, "step", step.ID, "error", err)
            
            // Start compensation
            return sc.compensateSaga(ctx, saga, i-1)
        }
        
        cancel()
        saga.CompletedSteps = append(saga.CompletedSteps, i)
        sc.logger.Info("Saga step completed", "saga_id", txnID, "step", step.ID)
    }
    
    saga.State = TransactionCommitted
    sc.logger.Info("Saga completed successfully", "saga_id", txnID)
    
    // Clean up
    sc.mu.Lock()
    delete(sc.transactions, txnID)
    sc.mu.Unlock()
    
    return nil
}

func (sc *SagaCoordinator) compensateSaga(ctx context.Context, saga *SagaTransaction, fromStep int) error {
    sc.logger.Info("Starting saga compensation", "saga_id", saga.ID, "from_step", fromStep)
    
    // Compensate in reverse order
    for i := fromStep; i >= 0; i-- {
        step := saga.Steps[i]
        
        compensateCtx, cancel := context.WithTimeout(ctx, step.Timeout)
        
        sc.logger.Info("Compensating saga step", "saga_id", saga.ID, "step", step.ID)
        
        if err := step.Compensation(compensateCtx); err != nil {
            cancel()
            sc.logger.Error("Saga compensation failed", "saga_id", saga.ID, "step", step.ID, "error", err)
            // Continue with other compensations even if one fails
        } else {
            saga.CompensatedSteps = append(saga.CompensatedSteps, i)
            sc.logger.Info("Saga step compensated", "saga_id", saga.ID, "step", step.ID)
        }
        
        cancel()
    }
    
    saga.State = TransactionAborted
    sc.logger.Info("Saga compensation completed", "saga_id", saga.ID)
    
    // Clean up
    sc.mu.Lock()
    delete(sc.transactions, saga.ID)
    sc.mu.Unlock()
    
    return errors.New("saga failed and was compensated")
}
```

---

## 🔄 **Data Consistency Patterns**

### **Event Sourcing Implementation**

```go
package eventsourcing

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Event represents a domain event
type Event struct {
    ID           string                 `json:"id"`
    AggregateID  string                 `json:"aggregate_id"`
    EventType    string                 `json:"event_type"`
    Version      int64                  `json:"version"`
    Data         map[string]interface{} `json:"data"`
    Metadata     map[string]interface{} `json:"metadata"`
    Timestamp    time.Time              `json:"timestamp"`
}

// Event Store interface
type EventStore interface {
    SaveEvents(ctx context.Context, aggregateID string, events []Event, expectedVersion int64) error
    GetEvents(ctx context.Context, aggregateID string, fromVersion int64) ([]Event, error)
    GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error)
    CreateSnapshot(ctx context.Context, aggregateID string, version int64, data interface{}) error
    GetSnapshot(ctx context.Context, aggregateID string) (*Snapshot, error)
}

type Snapshot struct {
    AggregateID string      `json:"aggregate_id"`
    Version     int64       `json:"version"`
    Data        interface{} `json:"data"`
    Timestamp   time.Time   `json:"timestamp"`
}

// In-memory event store implementation
type InMemoryEventStore struct {
    events    map[string][]Event // aggregateID -> events
    snapshots map[string]*Snapshot
    mu        sync.RWMutex
}

func NewInMemoryEventStore() *InMemoryEventStore {
    return &InMemoryEventStore{
        events:    make(map[string][]Event),
        snapshots: make(map[string]*Snapshot),
    }
}

func (es *InMemoryEventStore) SaveEvents(ctx context.Context, aggregateID string, events []Event, expectedVersion int64) error {
    es.mu.Lock()
    defer es.mu.Unlock()
    
    existingEvents := es.events[aggregateID]
    currentVersion := int64(len(existingEvents))
    
    // Optimistic concurrency control
    if expectedVersion != -1 && currentVersion != expectedVersion {
        return fmt.Errorf("concurrency conflict: expected version %d, actual version %d", expectedVersion, currentVersion)
    }
    
    // Assign versions to new events
    for i := range events {
        events[i].Version = currentVersion + int64(i) + 1
        events[i].Timestamp = time.Now()
    }
    
    es.events[aggregateID] = append(existingEvents, events...)
    return nil
}

func (es *InMemoryEventStore) GetEvents(ctx context.Context, aggregateID string, fromVersion int64) ([]Event, error) {
    es.mu.RLock()
    defer es.mu.RUnlock()
    
    allEvents := es.events[aggregateID]
    var result []Event
    
    for _, event := range allEvents {
        if event.Version >= fromVersion {
            result = append(result, event)
        }
    }
    
    return result, nil
}

func (es *InMemoryEventStore) GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error) {
    es.mu.RLock()
    defer es.mu.RUnlock()
    
    var result []Event
    for _, events := range es.events {
        for _, event := range events {
            if event.Timestamp.After(fromTimestamp) || event.Timestamp.Equal(fromTimestamp) {
                result = append(result, event)
            }
        }
    }
    
    return result, nil
}

func (es *InMemoryEventStore) CreateSnapshot(ctx context.Context, aggregateID string, version int64, data interface{}) error {
    es.mu.Lock()
    defer es.mu.Unlock()
    
    es.snapshots[aggregateID] = &Snapshot{
        AggregateID: aggregateID,
        Version:     version,
        Data:        data,
        Timestamp:   time.Now(),
    }
    
    return nil
}

func (es *InMemoryEventStore) GetSnapshot(ctx context.Context, aggregateID string) (*Snapshot, error) {
    es.mu.RLock()
    defer es.mu.RUnlock()
    
    snapshot, exists := es.snapshots[aggregateID]
    if !exists {
        return nil, fmt.Errorf("snapshot not found for aggregate %s", aggregateID)
    }
    
    return snapshot, nil
}

// Aggregate Root interface
type AggregateRoot interface {
    GetID() string
    GetVersion() int64
    GetUncommittedEvents() []Event
    MarkEventsAsCommitted()
    LoadFromHistory(events []Event)
}

// Base aggregate implementation
type BaseAggregate struct {
    ID                string  `json:"id"`
    Version           int64   `json:"version"`
    UncommittedEvents []Event `json:"-"`
}

func (ba *BaseAggregate) GetID() string {
    return ba.ID
}

func (ba *BaseAggregate) GetVersion() int64 {
    return ba.Version
}

func (ba *BaseAggregate) GetUncommittedEvents() []Event {
    return ba.UncommittedEvents
}

func (ba *BaseAggregate) MarkEventsAsCommitted() {
    ba.UncommittedEvents = nil
}

func (ba *BaseAggregate) RaiseEvent(eventType string, data map[string]interface{}) {
    event := Event{
        ID:          fmt.Sprintf("%s-%d", ba.ID, ba.Version+1),
        AggregateID: ba.ID,
        EventType:   eventType,
        Data:        data,
        Metadata:    make(map[string]interface{}),
        Timestamp:   time.Now(),
    }
    
    ba.UncommittedEvents = append(ba.UncommittedEvents, event)
    ba.Version++
}

// Repository pattern for aggregates
type Repository struct {
    eventStore EventStore
}

func NewRepository(eventStore EventStore) *Repository {
    return &Repository{eventStore: eventStore}
}

func (r *Repository) Save(ctx context.Context, aggregate AggregateRoot) error {
    events := aggregate.GetUncommittedEvents()
    if len(events) == 0 {
        return nil
    }
    
    expectedVersion := aggregate.GetVersion() - int64(len(events))
    
    if err := r.eventStore.SaveEvents(ctx, aggregate.GetID(), events, expectedVersion); err != nil {
        return err
    }
    
    aggregate.MarkEventsAsCommitted()
    return nil
}

func (r *Repository) GetByID(ctx context.Context, aggregateID string, aggregateFactory func() AggregateRoot) (AggregateRoot, error) {
    // Try to load from snapshot first
    var fromVersion int64 = 0
    aggregate := aggregateFactory()
    
    if snapshot, err := r.eventStore.GetSnapshot(ctx, aggregateID); err == nil {
        // Load from snapshot
        if err := r.loadFromSnapshot(aggregate, snapshot); err != nil {
            return nil, fmt.Errorf("failed to load from snapshot: %w", err)
        }
        fromVersion = snapshot.Version + 1
    }
    
    // Load events after snapshot
    events, err := r.eventStore.GetEvents(ctx, aggregateID, fromVersion)
    if err != nil {
        return nil, err
    }
    
    if len(events) > 0 {
        aggregate.LoadFromHistory(events)
    }
    
    return aggregate, nil
}

func (r *Repository) loadFromSnapshot(aggregate AggregateRoot, snapshot *Snapshot) error {
    // This would typically involve deserializing the snapshot data
    // into the aggregate's state
    return nil
}

// CQRS (Command Query Responsibility Segregation) Implementation
type CommandHandler interface {
    Handle(ctx context.Context, command interface{}) error
}

type QueryHandler interface {
    Handle(ctx context.Context, query interface{}) (interface{}, error)
}

type CommandBus struct {
    handlers map[string]CommandHandler
    mu       sync.RWMutex
}

func NewCommandBus() *CommandBus {
    return &CommandBus{
        handlers: make(map[string]CommandHandler),
    }
}

func (cb *CommandBus) RegisterHandler(commandType string, handler CommandHandler) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.handlers[commandType] = handler
}

func (cb *CommandBus) Dispatch(ctx context.Context, command interface{}) error {
    commandType := fmt.Sprintf("%T", command)
    
    cb.mu.RLock()
    handler, exists := cb.handlers[commandType]
    cb.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("no handler registered for command type: %s", commandType)
    }
    
    return handler.Handle(ctx, command)
}

type QueryBus struct {
    handlers map[string]QueryHandler
    mu       sync.RWMutex
}

func NewQueryBus() *QueryBus {
    return &QueryBus{
        handlers: make(map[string]QueryHandler),
    }
}

func (qb *QueryBus) RegisterHandler(queryType string, handler QueryHandler) {
    qb.mu.Lock()
    defer qb.mu.Unlock()
    qb.handlers[queryType] = handler
}

func (qb *QueryBus) Dispatch(ctx context.Context, query interface{}) (interface{}, error) {
    queryType := fmt.Sprintf("%T", query)
    
    qb.mu.RLock()
    handler, exists := qb.handlers[queryType]
    qb.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("no handler registered for query type: %s", queryType)
    }
    
    return handler.Handle(ctx, query)
}

// Event Projection for read models
type EventProjection interface {
    ProjectEvent(ctx context.Context, event Event) error
    GetProjectionName() string
}

type ProjectionManager struct {
    eventStore  EventStore
    projections []EventProjection
    mu          sync.RWMutex
    stopChan    chan struct{}
}

func NewProjectionManager(eventStore EventStore) *ProjectionManager {
    return &ProjectionManager{
        eventStore: eventStore,
        stopChan:   make(chan struct{}),
    }
}

func (pm *ProjectionManager) RegisterProjection(projection EventProjection) {
    pm.mu.Lock()
    defer pm.mu.Unlock()
    pm.projections = append(pm.projections, projection)
}

func (pm *ProjectionManager) Start(ctx context.Context) {
    go pm.processEvents(ctx)
}

func (pm *ProjectionManager) processEvents(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    lastProcessedTime := time.Now().Add(-24 * time.Hour) // Start from 24 hours ago
    
    for {
        select {
        case <-ticker.C:
            events, err := pm.eventStore.GetAllEvents(ctx, lastProcessedTime)
            if err != nil {
                continue
            }
            
            for _, event := range events {
                pm.processEvent(ctx, event)
                if event.Timestamp.After(lastProcessedTime) {
                    lastProcessedTime = event.Timestamp
                }
            }
            
        case <-pm.stopChan:
            return
        case <-ctx.Done():
            return
        }
    }
}

func (pm *ProjectionManager) processEvent(ctx context.Context, event Event) {
    pm.mu.RLock()
    projections := make([]EventProjection, len(pm.projections))
    copy(projections, pm.projections)
    pm.mu.RUnlock()
    
    for _, projection := range projections {
        if err := projection.ProjectEvent(ctx, event); err != nil {
            // Log error but continue with other projections
            fmt.Printf("Projection %s failed to process event %s: %v\n", 
                projection.GetProjectionName(), event.ID, err)
        }
    }
}

func (pm *ProjectionManager) Stop() {
    close(pm.stopChan)
}
```

---

## 🛡️ **Fault Tolerance & Resilience**

### **Circuit Breaker Pattern**

```go
package circuitbreaker

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// Circuit Breaker States
type State int

const (
    StateClosed State = iota
    StateHalfOpen
    StateOpen
)

func (s State) String() string {
    switch s {
    case StateClosed:
        return "Closed"
    case StateHalfOpen:
        return "HalfOpen"
    case StateOpen:
        return "Open"
    default:
        return "Unknown"
    }
}

// Circuit Breaker Configuration
type Config struct {
    MaxRequests      uint32        `json:"max_requests"`      // Max requests allowed in half-open state
    Interval         time.Duration `json:"interval"`          // Interval to clear counters
    Timeout          time.Duration `json:"timeout"`           // Timeout duration in open state
    ReadyToTrip      func(counts Counts) bool `json:"-"`     // Function to determine when to trip
    OnStateChange    func(name string, from State, to State) `json:"-"` // State change callback
    IsSuccessful     func(err error) bool `json:"-"`         // Function to determine if result is successful
}

// Request Counts
type Counts struct {
    Requests        uint32 `json:"requests"`
    TotalSuccesses  uint32 `json:"total_successes"`
    TotalFailures   uint32 `json:"total_failures"`
    ConsecutiveSuccesses uint32 `json:"consecutive_successes"`
    ConsecutiveFailures  uint32 `json:"consecutive_failures"`
}

func (c Counts) SuccessRate() float64 {
    if c.Requests == 0 {
        return 0
    }
    return float64(c.TotalSuccesses) / float64(c.Requests)
}

func (c Counts) FailureRate() float64 {
    if c.Requests == 0 {
        return 0
    }
    return float64(c.TotalFailures) / float64(c.Requests)
}

// Circuit Breaker Implementation
type CircuitBreaker struct {
    name         string
    config       Config
    state        State
    generation   uint64
    counts       Counts
    expiry       time.Time
    mu           sync.RWMutex
}

func NewCircuitBreaker(name string, config Config) *CircuitBreaker {
    cb := &CircuitBreaker{
        name:   name,
        config: config,
        state:  StateClosed,
        expiry: time.Now().Add(config.Interval),
    }
    
    // Set default functions if not provided
    if cb.config.ReadyToTrip == nil {
        cb.config.ReadyToTrip = func(counts Counts) bool {
            return counts.Requests >= 5 && counts.FailureRate() >= 0.6
        }
    }
    
    if cb.config.IsSuccessful == nil {
        cb.config.IsSuccessful = func(err error) bool {
            return err == nil
        }
    }
    
    return cb
}

// Execute function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    generation, err := cb.beforeRequest()
    if err != nil {
        return nil, err
    }
    
    defer func() {
        if r := recover(); r != nil {
            cb.afterRequest(generation, false)
            panic(r)
        }
    }()
    
    result, err := fn()
    cb.afterRequest(generation, cb.config.IsSuccessful(err))
    return result, err
}

// Execute with context
func (cb *CircuitBreaker) ExecuteContext(ctx context.Context, fn func(ctx context.Context) (interface{}, error)) (interface{}, error) {
    generation, err := cb.beforeRequest()
    if err != nil {
        return nil, err
    }
    
    defer func() {
        if r := recover(); r != nil {
            cb.afterRequest(generation, false)
            panic(r)
        }
    }()
    
    result, err := fn(ctx)
    cb.afterRequest(generation, cb.config.IsSuccessful(err))
    return result, err
}

func (cb *CircuitBreaker) beforeRequest() (uint64, error) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    
    if state == StateOpen {
        return generation, errors.New("circuit breaker is open")
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.config.MaxRequests {
        return generation, errors.New("too many requests in half-open state")
    }
    
    cb.counts.Requests++
    return generation, nil
}

func (cb *CircuitBreaker) afterRequest(before uint64, success bool) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    
    if generation != before {
        return // Ignore stale request
    }
    
    if success {
        cb.onSuccess(state)
    } else {
        cb.onFailure(state)
    }
}

func (cb *CircuitBreaker) onSuccess(state State) {
    cb.counts.TotalSuccesses++
    cb.counts.ConsecutiveSuccesses++
    cb.counts.ConsecutiveFailures = 0
    
    if state == StateHalfOpen && cb.counts.ConsecutiveSuccesses >= cb.config.MaxRequests {
        cb.setState(StateClosed)
    }
}

func (cb *CircuitBreaker) onFailure(state State) {
    cb.counts.TotalFailures++
    cb.counts.ConsecutiveFailures++
    cb.counts.ConsecutiveSuccesses = 0
    
    if cb.config.ReadyToTrip(cb.counts) {
        cb.setState(StateOpen)
    }
}

func (cb *CircuitBreaker) currentState(now time.Time) (State, uint64) {
    switch cb.state {
    case StateClosed:
        if !cb.expiry.IsZero() && cb.expiry.Before(now) {
            cb.toNewGeneration(now)
        }
    case StateOpen:
        if cb.expiry.Before(now) {
            cb.setState(StateHalfOpen)
        }
    }
    
    return cb.state, cb.generation
}

func (cb *CircuitBreaker) setState(state State) {
    if cb.state == state {
        return
    }
    
    prev := cb.state
    cb.state = state
    
    now := time.Now()
    
    switch state {
    case StateClosed:
        cb.toNewGeneration(now)
    case StateOpen:
        cb.generation++
        cb.expiry = now.Add(cb.config.Timeout)
    case StateHalfOpen:
        cb.generation++
    }
    
    if cb.config.OnStateChange != nil {
        cb.config.OnStateChange(cb.name, prev, state)
    }
}

func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
    cb.generation++
    cb.counts = Counts{}
    
    var zero time.Time
    switch cb.state {
    case StateClosed:
        if cb.config.Interval == 0 {
            cb.expiry = zero
        } else {
            cb.expiry = now.Add(cb.config.Interval)
        }
    case StateHalfOpen:
        cb.expiry = zero
    }
}

// Get current circuit breaker state
func (cb *CircuitBreaker) State() State {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    state, _ := cb.currentState(time.Now())
    return state
}

// Get current counts
func (cb *CircuitBreaker) Counts() Counts {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    return cb.counts
}

// Retry Pattern with Exponential Backoff
type RetryConfig struct {
    MaxAttempts    int           `json:"max_attempts"`
    InitialDelay   time.Duration `json:"initial_delay"`
    MaxDelay       time.Duration `json:"max_delay"`
    BackoffFactor  float64       `json:"backoff_factor"`
    Jitter         bool          `json:"jitter"`
    RetryableErrors []error      `json:"-"`
}

type RetryableFunc func() (interface{}, error)
type RetryableFuncWithContext func(ctx context.Context) (interface{}, error)

func RetryWithExponentialBackoff(config RetryConfig, fn RetryableFunc) (interface{}, error) {
    var lastErr error
    delay := config.InitialDelay
    
    for attempt := 0; attempt < config.MaxAttempts; attempt++ {
        if attempt > 0 {
            // Apply jitter if enabled
            actualDelay := delay
            if config.Jitter {
                jitterRange := float64(delay) * 0.1 // 10% jitter
                jitter := time.Duration(float64(jitterRange) * (2*rand.Float64() - 1))
                actualDelay = delay + jitter
            }
            
            time.Sleep(actualDelay)
            
            // Calculate next delay
            delay = time.Duration(float64(delay) * config.BackoffFactor)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
        
        result, err := fn()
        if err == nil {
            return result, nil
        }
        
        lastErr = err
        
        // Check if error is retryable
        if !isRetryableError(err, config.RetryableErrors) {
            break
        }
    }
    
    return nil, fmt.Errorf("retry failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

func RetryWithExponentialBackoffContext(ctx context.Context, config RetryConfig, fn RetryableFuncWithContext) (interface{}, error) {
    var lastErr error
    delay := config.InitialDelay
    
    for attempt := 0; attempt < config.MaxAttempts; attempt++ {
        select {
        case <-ctx.Done():
            return nil, ctx.Err()
        default:
        }
        
        if attempt > 0 {
            // Apply jitter if enabled
            actualDelay := delay
            if config.Jitter {
                jitterRange := float64(delay) * 0.1 // 10% jitter
                jitter := time.Duration(float64(jitterRange) * (2*rand.Float64() - 1))
                actualDelay = delay + jitter
            }
            
            timer := time.NewTimer(actualDelay)
            select {
            case <-ctx.Done():
                timer.Stop()
                return nil, ctx.Err()
            case <-timer.C:
            }
            
            // Calculate next delay
            delay = time.Duration(float64(delay) * config.BackoffFactor)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
        
        result, err := fn(ctx)
        if err == nil {
            return result, nil
        }
        
        lastErr = err
        
        // Check if error is retryable
        if !isRetryableError(err, config.RetryableErrors) {
            break
        }
    }
    
    return nil, fmt.Errorf("retry failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

func isRetryableError(err error, retryableErrors []error) bool {
    if len(retryableErrors) == 0 {
        return true // If no specific errors defined, retry all errors
    }
    
    for _, retryableErr := range retryableErrors {
        if errors.Is(err, retryableErr) {
            return true
        }
    }
    
    return false
}

// Bulkhead Pattern Implementation
type Bulkhead struct {
    name        string
    semaphore   chan struct{}
    metrics     BulkheadMetrics
    mu          sync.RWMutex
}

type BulkheadMetrics struct {
    TotalRequests    uint64 `json:"total_requests"`
    ActiveRequests   int64  `json:"active_requests"`
    RejectedRequests uint64 `json:"rejected_requests"`
    CompletedRequests uint64 `json:"completed_requests"`
}

func NewBulkhead(name string, maxConcurrent int) *Bulkhead {
    return &Bulkhead{
        name:      name,
        semaphore: make(chan struct{}, maxConcurrent),
    }
}

func (b *Bulkhead) Execute(fn func() (interface{}, error)) (interface{}, error) {
    atomic.AddUint64(&b.metrics.TotalRequests, 1)
    
    select {
    case b.semaphore <- struct{}{}:
        atomic.AddInt64(&b.metrics.ActiveRequests, 1)
        defer func() {
            <-b.semaphore
            atomic.AddInt64(&b.metrics.ActiveRequests, -1)
            atomic.AddUint64(&b.metrics.CompletedRequests, 1)
        }()
        
        return fn()
    default:
        atomic.AddUint64(&b.metrics.RejectedRequests, 1)
        return nil, errors.New("bulkhead capacity exceeded")
    }
}

func (b *Bulkhead) ExecuteContext(ctx context.Context, fn func(ctx context.Context) (interface{}, error)) (interface{}, error) {
    atomic.AddUint64(&b.metrics.TotalRequests, 1)
    
    select {
    case b.semaphore <- struct{}{}:
        atomic.AddInt64(&b.metrics.ActiveRequests, 1)
        defer func() {
            <-b.semaphore
            atomic.AddInt64(&b.metrics.ActiveRequests, -1)
            atomic.AddUint64(&b.metrics.CompletedRequests, 1)
        }()
        
        return fn(ctx)
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
        atomic.AddUint64(&b.metrics.RejectedRequests, 1)
        return nil, errors.New("bulkhead capacity exceeded")
    }
}

func (b *Bulkhead) GetMetrics() BulkheadMetrics {
    return BulkheadMetrics{
        TotalRequests:     atomic.LoadUint64(&b.metrics.TotalRequests),
        ActiveRequests:    atomic.LoadInt64(&b.metrics.ActiveRequests),
        RejectedRequests:  atomic.LoadUint64(&b.metrics.RejectedRequests),
        CompletedRequests: atomic.LoadUint64(&b.metrics.CompletedRequests),
    }
}

// Timeout Pattern
func WithTimeout(timeout time.Duration, fn func() (interface{}, error)) (interface{}, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    
    return WithTimeoutContext(ctx, fn)
}

func WithTimeoutContext(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    resultChan := make(chan struct {
        result interface{}
        err    error
    }, 1)
    
    go func() {
        result, err := fn()
        resultChan <- struct {
            result interface{}
            err    error
        }{result, err}
    }()
    
    select {
    case res := <-resultChan:
        return res.result, res.err
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}
```

---

## ❓ **Interview Questions**

### **Advanced Distributed Systems Questions**

#### **1. CAP Theorem Implementation**

**Q: Explain CAP theorem and implement a system that can switch between CP and AP modes based on network conditions.**

**A: Dynamic CAP mode switching:**

```go
type DynamicCAPSystem struct {
    mode            CAPMode
    healthChecker   NetworkHealthChecker
    cpSystem        *CPSystem
    apSystem        *APSystem
    switchThreshold float64 // Network health threshold for switching
}

type CAPMode int
const (
    CPMode CAPMode = iota  // Consistency + Partition Tolerance
    APMode                 // Availability + Partition Tolerance
)

func (dcs *DynamicCAPSystem) adaptToNetworkConditions() {
    networkHealth := dcs.healthChecker.GetHealthScore()
    
    if networkHealth < dcs.switchThreshold && dcs.mode == CPMode {
        // Switch to AP mode when network is unhealthy
        dcs.mode = APMode
        dcs.logger.Info("Switched to AP mode due to network issues")
    } else if networkHealth >= dcs.switchThreshold && dcs.mode == APMode {
        // Switch back to CP mode when network recovers
        dcs.mode = CPMode
        dcs.logger.Info("Switched to CP mode - network recovered")
    }
}

func (dcs *DynamicCAPSystem) Write(ctx context.Context, key string, value interface{}) error {
    switch dcs.mode {
    case CPMode:
        return dcs.cpSystem.Write(ctx, key, value)
    case APMode:
        return dcs.apSystem.Write(ctx, key, value)
    default:
        return errors.New("unknown CAP mode")
    }
}
```

**Key considerations:**
- **CP systems** sacrifice availability during network partitions to maintain consistency
- **AP systems** sacrifice consistency to remain available during partitions
- Monitor network health and partition frequency to make switching decisions
- Implement graceful transition mechanisms between modes

#### **2. Consensus Algorithm Comparison**

**Q: Compare Raft, PBFT, and Paxos consensus algorithms. When would you use each?**

**A: Consensus algorithm selection matrix:**

| Algorithm | Use Case | Fault Model | Performance | Complexity |
|-----------|----------|-------------|-------------|------------|
| **Raft** | General distributed systems, replicated state machines | Crash failures | Good | Medium |
| **PBFT** | Byzantine fault tolerance, blockchain, untrusted environments | Byzantine failures | Lower | High |
| **Paxos** | Highly available systems, Google Spanner | Crash failures | Good | High |

```go
// Consensus algorithm factory
func NewConsensusAlgorithm(algorithmType string, config ConsensusConfig) ConsensusAlgorithm {
    switch algorithmType {
    case "raft":
        // Use for: replicated databases, distributed file systems
        return NewRaftConsensus(config)
    case "pbft":
        // Use for: blockchain, untrusted distributed systems
        return NewPBFTConsensus(config)
    case "paxos":
        // Use for: highly available systems requiring strong consistency
        return NewPaxosConsensus(config)
    default:
        panic("unsupported consensus algorithm")
    }
}
```

#### **3. Distributed Transaction Patterns**

**Q: Compare 2PC, Saga, and TCC patterns. Implement a transaction coordinator that can use any pattern.**

**A: Multi-pattern transaction coordinator:**

```go
type TransactionPattern int
const (
    TwoPhaseCommit TransactionPattern = iota
    SagaPattern
    TCCPattern  // Try-Confirm-Cancel
)

type UniversalTransactionCoordinator struct {
    pattern TransactionPattern
    tpcCoordinator *TransactionCoordinator
    sagaCoordinator *SagaCoordinator  
    tccCoordinator *TCCCoordinator
}

func (utc *UniversalTransactionCoordinator) ExecuteTransaction(ctx context.Context, txn DistributedTransaction) error {
    switch utc.pattern {
    case TwoPhaseCommit:
        // Use for: ACID requirements, short transactions, trusted network
        return utc.tpcCoordinator.ExecuteTransaction(ctx, txn.ID)
    case SagaPattern:
        // Use for: long-running transactions, loose coupling, eventual consistency
        return utc.sagaCoordinator.ExecuteSaga(ctx, txn.ID)
    case TCCPattern:
        // Use for: resource reservation scenarios, e-commerce
        return utc.tccCoordinator.ExecuteTCC(ctx, txn.ID)
    }
    return errors.New("unsupported transaction pattern")
}

// Pattern selection based on transaction characteristics
func (utc *UniversalTransactionCoordinator) SelectOptimalPattern(txn DistributedTransaction) TransactionPattern {
    if txn.Duration < 1*time.Second && txn.RequiresACID {
        return TwoPhaseCommit
    } else if txn.Duration > 30*time.Second || txn.CrossBoundaries {
        return SagaPattern
    } else if txn.RequiresReservation {
        return TCCPattern
    }
    return SagaPattern // Default to most flexible
}
```

#### **4. Event Sourcing vs Traditional CRUD**

**Q: When would you choose Event Sourcing over traditional CRUD? Implement a hybrid approach.**

**A: Hybrid Event Sourcing system:**

```go
type HybridDataStore struct {
    eventStore    EventStore
    crudStore     CRUDStore
    strategy      DataStrategy
}

type DataStrategy struct {
    UseEventSourcing func(aggregateType string) bool
    SnapshotFrequency int
}

func (hds *HybridDataStore) Save(ctx context.Context, aggregate AggregateRoot) error {
    aggregateType := reflect.TypeOf(aggregate).Name()
    
    if hds.strategy.UseEventSourcing(aggregateType) {
        // Use event sourcing for:
        // - Audit requirements
        // - Complex business logic
        // - Temporal queries
        // - Regulatory compliance
        return hds.saveWithEventSourcing(ctx, aggregate)
    } else {
        // Use CRUD for:
        // - Simple data models
        // - Performance-critical reads
        // - Third-party integrations
        return hds.saveWithCRUD(ctx, aggregate)
    }
}

// Event sourcing criteria
func DefaultEventSourcingStrategy() DataStrategy {
    return DataStrategy{
        UseEventSourcing: func(aggregateType string) bool {
            // Use event sourcing for financial and audit-critical aggregates
            eventSourcedTypes := []string{
                "Account", "Transaction", "Order", "Payment",
                "User", "Inventory", "Pricing",
            }
            
            for _, esType := range eventSourcedTypes {
                if strings.Contains(aggregateType, esType) {
                    return true
                }
            }
            return false
        },
        SnapshotFrequency: 10, // Snapshot every 10 events
    }
}
```

#### **5. Distributed Caching Strategy**

**Q: Design a multi-level distributed caching system with consistency guarantees.**

**A: Hierarchical distributed cache:**

```go
type MultiLevelCache struct {
    l1Cache    LocalCache     // In-memory cache
    l2Cache    DistributedCache // Redis cluster
    l3Cache    PersistentCache  // Database
    consistency ConsistencyLevel
}

type ConsistencyLevel int
const (
    WeakConsistency ConsistencyLevel = iota
    EventualConsistency
    StrongConsistency
)

func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, error) {
    // L1: Check local cache first
    if value, found := mlc.l1Cache.Get(key); found {
        return value, nil
    }
    
    // L2: Check distributed cache
    if value, err := mlc.l2Cache.Get(ctx, key); err == nil {
        // Populate L1 cache
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        return value, nil
    }
    
    // L3: Check persistent storage
    value, err := mlc.l3Cache.Get(ctx, key)
    if err != nil {
        return nil, err
    }
    
    // Populate both caches
    mlc.l2Cache.Set(ctx, key, value, 30*time.Minute)
    mlc.l1Cache.Set(key, value, 5*time.Minute)
    
    return value, nil
}

func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}) error {
    switch mlc.consistency {
    case StrongConsistency:
        // Update all levels synchronously
        return mlc.setStrongConsistency(ctx, key, value)
    case EventualConsistency:
        // Update L3 first, then propagate asynchronously
        return mlc.setEventualConsistency(ctx, key, value)
    case WeakConsistency:
        // Update locally, propagate best-effort
        return mlc.setWeakConsistency(ctx, key, value)
    }
    return nil
}

// Cache invalidation strategy
func (mlc *MultiLevelCache) InvalidatePattern(ctx context.Context, pattern string) error {
    // Invalidate across all cache levels
    var wg sync.WaitGroup
    errors := make(chan error, 3)
    
    wg.Add(3)
    go func() {
        defer wg.Done()
        errors <- mlc.l1Cache.InvalidatePattern(pattern)
    }()
    
    go func() {
        defer wg.Done() 
        errors <- mlc.l2Cache.InvalidatePattern(ctx, pattern)
    }()
    
    go func() {
        defer wg.Done()
        errors <- mlc.l3Cache.InvalidatePattern(ctx, pattern)
    }()
    
    wg.Wait()
    close(errors)
    
    // Check for any errors
    for err := range errors {
        if err != nil {
            return err
        }
    }
    
    return nil
}
```

This comprehensive Distributed Systems Engineering Guide provides advanced implementations of essential distributed systems patterns. The guide covers CAP theorem with practical CP/AP system implementations, Raft consensus algorithm, two-phase commit protocol, Saga pattern, event sourcing, CQRS, circuit breaker pattern, retry mechanisms, and bulkhead isolation, all with production-ready Go code that demonstrates the deep technical expertise expected from senior backend engineers in technical interviews.

## Service Mesh Architecture

<!-- AUTO-GENERATED ANCHOR: originally referenced as #service-mesh-architecture -->

Placeholder content. Please replace with proper section.


## Load Balancing Strategies

<!-- AUTO-GENERATED ANCHOR: originally referenced as #load-balancing-strategies -->

Placeholder content. Please replace with proper section.


## Event Driven Architecture

<!-- AUTO-GENERATED ANCHOR: originally referenced as #event-driven-architecture -->

Placeholder content. Please replace with proper section.


## Monitoring  Observability

<!-- AUTO-GENERATED ANCHOR: originally referenced as #monitoring--observability -->

Placeholder content. Please replace with proper section.
