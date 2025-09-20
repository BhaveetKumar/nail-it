# 🌐 Distributed Systems Comprehensive Guide

## Table of Contents
1. [Consensus Algorithms](#consensus-algorithms)
2. [Distributed Coordination](#distributed-coordination)
3. [Replication Strategies](#replication-strategies)
4. [Partitioning and Sharding](#partitioning-and-sharding)
5. [Distributed Caching](#distributed-caching)
6. [Message Queues and Event Streaming](#message-queues-and-event-streaming)
7. [Service Discovery](#service-discovery)
8. [Load Balancing](#load-balancing)
9. [Go Implementation Examples](#go-implementation-examples)
10. [Interview Questions](#interview-questions)

## Consensus Algorithms

### Raft Algorithm Implementation

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
    Peers       []int
    mutex       sync.RWMutex
    stopCh      chan bool
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

type RequestVoteRequest struct {
    Term         int
    CandidateID  int
    LastLogIndex int
    LastLogTerm  int
}

type RequestVoteResponse struct {
    Term        int
    VoteGranted bool
}

type AppendEntriesRequest struct {
    Term         int
    LeaderID     int
    PrevLogIndex int
    PrevLogTerm  int
    Entries      []LogEntry
    LeaderCommit int
}

type AppendEntriesResponse struct {
    Term    int
    Success bool
}

func NewRaftNode(id int, peers []int) *RaftNode {
    return &RaftNode{
        ID:          id,
        State:       Follower,
        CurrentTerm: 0,
        VotedFor:    -1,
        Log:         make([]LogEntry, 0),
        CommitIndex: -1,
        LastApplied: -1,
        NextIndex:   make(map[int]int),
        MatchIndex:  make(map[int]int),
        Peers:       peers,
        stopCh:      make(chan bool),
    }
}

func (rn *RaftNode) Start() {
    go rn.run()
}

func (rn *RaftNode) run() {
    for {
        select {
        case <-rn.stopCh:
            return
        default:
            rn.mutex.Lock()
            state := rn.State
            rn.mutex.Unlock()
            
            switch state {
            case Follower:
                rn.runFollower()
            case Candidate:
                rn.runCandidate()
            case Leader:
                rn.runLeader()
            }
        }
    }
}

func (rn *RaftNode) runFollower() {
    timeout := time.Duration(rand.Intn(300)+150) * time.Millisecond
    timer := time.NewTimer(timeout)
    
    select {
    case <-timer.C:
        rn.mutex.Lock()
        rn.State = Candidate
        rn.CurrentTerm++
        rn.VotedFor = rn.ID
        rn.mutex.Unlock()
    case <-rn.stopCh:
        timer.Stop()
        return
    }
}

func (rn *RaftNode) runCandidate() {
    rn.mutex.Lock()
    rn.CurrentTerm++
    rn.VotedFor = rn.ID
    votes := 1
    rn.mutex.Unlock()
    
    // Request votes from all peers
    for _, peerID := range rn.Peers {
        go rn.requestVote(peerID, &votes)
    }
    
    // Wait for majority or timeout
    timeout := time.Duration(rand.Intn(300)+150) * time.Millisecond
    timer := time.NewTimer(timeout)
    
    select {
    case <-timer.C:
        rn.mutex.Lock()
        if rn.State == Candidate {
            rn.State = Follower
        }
        rn.mutex.Unlock()
    case <-rn.stopCh:
        timer.Stop()
        return
    }
}

func (rn *RaftNode) runLeader() {
    // Send heartbeats to all peers
    for _, peerID := range rn.Peers {
        go rn.sendHeartbeat(peerID)
    }
    
    // Wait for heartbeat interval
    time.Sleep(50 * time.Millisecond)
}

func (rn *RaftNode) requestVote(peerID int, votes *int) {
    rn.mutex.RLock()
    lastLogIndex := len(rn.Log) - 1
    lastLogTerm := 0
    if lastLogIndex >= 0 {
        lastLogTerm = rn.Log[lastLogIndex].Term
    }
    term := rn.CurrentTerm
    rn.mutex.RUnlock()
    
    req := &RequestVoteRequest{
        Term:         term,
        CandidateID:  rn.ID,
        LastLogIndex: lastLogIndex,
        LastLogTerm:  lastLogTerm,
    }
    
    // Simulate network call
    resp := rn.handleRequestVote(req)
    
    if resp.VoteGranted {
        rn.mutex.Lock()
        (*votes)++
        if *votes > len(rn.Peers)/2 && rn.State == Candidate {
            rn.State = Leader
            rn.initializeLeader()
        }
        rn.mutex.Unlock()
    }
}

func (rn *RaftNode) handleRequestVote(req *RequestVoteRequest) *RequestVoteResponse {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    if req.Term < rn.CurrentTerm {
        return &RequestVoteResponse{
            Term:        rn.CurrentTerm,
            VoteGranted: false,
        }
    }
    
    if req.Term > rn.CurrentTerm {
        rn.CurrentTerm = req.Term
        rn.State = Follower
        rn.VotedFor = -1
    }
    
    if rn.VotedFor == -1 || rn.VotedFor == req.CandidateID {
        // Check if candidate's log is at least as up-to-date
        lastLogIndex := len(rn.Log) - 1
        lastLogTerm := 0
        if lastLogIndex >= 0 {
            lastLogTerm = rn.Log[lastLogIndex].Term
        }
        
        if req.LastLogTerm > lastLogTerm ||
           (req.LastLogTerm == lastLogTerm && req.LastLogIndex >= lastLogIndex) {
            rn.VotedFor = req.CandidateID
            return &RequestVoteResponse{
                Term:        rn.CurrentTerm,
                VoteGranted: true,
            }
        }
    }
    
    return &RequestVoteResponse{
        Term:        rn.CurrentTerm,
        VoteGranted: false,
    }
}

func (rn *RaftNode) sendHeartbeat(peerID int) {
    rn.mutex.RLock()
    term := rn.CurrentTerm
    leaderID := rn.ID
    prevLogIndex := rn.NextIndex[peerID] - 1
    prevLogTerm := 0
    if prevLogIndex >= 0 && prevLogIndex < len(rn.Log) {
        prevLogTerm = rn.Log[prevLogIndex].Term
    }
    entries := rn.Log[rn.NextIndex[peerID]:]
    leaderCommit := rn.CommitIndex
    rn.mutex.RUnlock()
    
    req := &AppendEntriesRequest{
        Term:         term,
        LeaderID:     leaderID,
        PrevLogIndex: prevLogIndex,
        PrevLogTerm:  prevLogTerm,
        Entries:      entries,
        LeaderCommit: leaderCommit,
    }
    
    resp := rn.handleAppendEntries(req)
    
    if resp.Success {
        rn.mutex.Lock()
        rn.NextIndex[peerID] = prevLogIndex + len(entries) + 1
        rn.MatchIndex[peerID] = rn.NextIndex[peerID] - 1
        rn.mutex.Unlock()
    } else {
        rn.mutex.Lock()
        if rn.NextIndex[peerID] > 0 {
            rn.NextIndex[peerID]--
        }
        rn.mutex.Unlock()
    }
}

func (rn *RaftNode) handleAppendEntries(req *AppendEntriesRequest) *AppendEntriesResponse {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    if req.Term < rn.CurrentTerm {
        return &AppendEntriesResponse{
            Term:    rn.CurrentTerm,
            Success: false,
        }
    }
    
    if req.Term > rn.CurrentTerm {
        rn.CurrentTerm = req.Term
        rn.State = Follower
        rn.VotedFor = -1
    }
    
    // Check if log contains entry at prevLogIndex with matching term
    if req.PrevLogIndex >= 0 && req.PrevLogIndex < len(rn.Log) {
        if rn.Log[req.PrevLogIndex].Term != req.PrevLogTerm {
            return &AppendEntriesResponse{
                Term:    rn.CurrentTerm,
                Success: false,
            }
        }
    }
    
    // Append new entries
    for i, entry := range req.Entries {
        if req.PrevLogIndex+1+i < len(rn.Log) {
            if rn.Log[req.PrevLogIndex+1+i].Term != entry.Term {
                // Remove conflicting entries
                rn.Log = rn.Log[:req.PrevLogIndex+1+i]
            }
        }
        if req.PrevLogIndex+1+i >= len(rn.Log) {
            rn.Log = append(rn.Log, entry)
        }
    }
    
    // Update commit index
    if req.LeaderCommit > rn.CommitIndex {
        rn.CommitIndex = min(req.LeaderCommit, len(rn.Log)-1)
    }
    
    return &AppendEntriesResponse{
        Term:    rn.CurrentTerm,
        Success: true,
    }
}

func (rn *RaftNode) initializeLeader() {
    for _, peerID := range rn.Peers {
        rn.NextIndex[peerID] = len(rn.Log)
        rn.MatchIndex[peerID] = -1
    }
}

func (rn *RaftNode) Stop() {
    close(rn.stopCh)
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Paxos Algorithm Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type PaxosNode struct {
    ID           int
    Proposers    []int
    Acceptors    []int
    Learners     []int
    Proposals    map[int]*Proposal
    Accepted     map[int]*Proposal
    mutex        sync.RWMutex
}

type Proposal struct {
    ProposalID int
    Value      interface{}
    AcceptorID int
}

type PrepareRequest struct {
    ProposalID int
    AcceptorID int
}

type PrepareResponse struct {
    ProposalID int
    Accepted   bool
    Value      interface{}
}

type AcceptRequest struct {
    ProposalID int
    Value      interface{}
    AcceptorID int
}

type AcceptResponse struct {
    ProposalID int
    Accepted   bool
}

func NewPaxosNode(id int, proposers, acceptors, learners []int) *PaxosNode {
    return &PaxosNode{
        ID:        id,
        Proposers: proposers,
        Acceptors: acceptors,
        Learners:  learners,
        Proposals: make(map[int]*Proposal),
        Accepted:  make(map[int]*Proposal),
    }
}

func (pn *PaxosNode) Propose(value interface{}) interface{} {
    proposalID := pn.generateProposalID()
    
    // Phase 1: Prepare
    prepareResponses := pn.prepare(proposalID)
    
    if len(prepareResponses) <= len(pn.Acceptors)/2 {
        return nil // Not enough responses
    }
    
    // Find highest accepted value
    highestValue := value
    for _, resp := range prepareResponses {
        if resp.Accepted && resp.Value != nil {
            highestValue = resp.Value
        }
    }
    
    // Phase 2: Accept
    acceptResponses := pn.accept(proposalID, highestValue)
    
    if len(acceptResponses) <= len(pn.Acceptors)/2 {
        return nil // Not enough responses
    }
    
    // Phase 3: Learn
    pn.learn(proposalID, highestValue)
    
    return highestValue
}

func (pn *PaxosNode) prepare(proposalID int) []*PrepareResponse {
    var responses []*PrepareResponse
    
    for _, acceptorID := range pn.Acceptors {
        req := &PrepareRequest{
            ProposalID: proposalID,
            AcceptorID: acceptorID,
        }
        
        resp := pn.handlePrepare(req)
        if resp != nil {
            responses = append(responses, resp)
        }
    }
    
    return responses
}

func (pn *PaxosNode) handlePrepare(req *PrepareRequest) *PrepareResponse {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    // Check if we've already accepted a higher proposal
    if accepted, exists := pn.Accepted[req.AcceptorID]; exists {
        if accepted.ProposalID >= req.ProposalID {
            return &PrepareResponse{
                ProposalID: req.ProposalID,
                Accepted:   false,
                Value:      nil,
            }
        }
    }
    
    // Accept the prepare request
    pn.Proposals[req.ProposalID] = &Proposal{
        ProposalID: req.ProposalID,
        Value:      nil,
        AcceptorID: req.AcceptorID,
    }
    
    // Return any previously accepted value
    var value interface{}
    if accepted, exists := pn.Accepted[req.AcceptorID]; exists {
        value = accepted.Value
    }
    
    return &PrepareResponse{
        ProposalID: req.ProposalID,
        Accepted:   true,
        Value:      value,
    }
}

func (pn *PaxosNode) accept(proposalID int, value interface{}) []*AcceptResponse {
    var responses []*AcceptResponse
    
    for _, acceptorID := range pn.Acceptors {
        req := &AcceptRequest{
            ProposalID: proposalID,
            Value:      value,
            AcceptorID: acceptorID,
        }
        
        resp := pn.handleAccept(req)
        if resp != nil {
            responses = append(responses, resp)
        }
    }
    
    return responses
}

func (pn *PaxosNode) handleAccept(req *AcceptRequest) *AcceptResponse {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    // Check if we've prepared this proposal
    if proposal, exists := pn.Proposals[req.ProposalID]; exists {
        if proposal.AcceptorID == req.AcceptorID {
            // Accept the proposal
            pn.Accepted[req.AcceptorID] = &Proposal{
                ProposalID: req.ProposalID,
                Value:      req.Value,
                AcceptorID: req.AcceptorID,
            }
            
            return &AcceptResponse{
                ProposalID: req.ProposalID,
                Accepted:   true,
            }
        }
    }
    
    return &AcceptResponse{
        ProposalID: req.ProposalID,
        Accepted:   false,
    }
}

func (pn *PaxosNode) learn(proposalID int, value interface{}) {
    // Notify all learners
    for _, learnerID := range pn.Learners {
        pn.notifyLearner(learnerID, proposalID, value)
    }
}

func (pn *PaxosNode) notifyLearner(learnerID, proposalID int, value interface{}) {
    // Simulate notifying learner
    fmt.Printf("Learner %d learned proposal %d with value %v\n", learnerID, proposalID, value)
}

func (pn *PaxosNode) generateProposalID() int {
    return int(time.Now().UnixNano())
}
```

## Distributed Coordination

### Distributed Lock Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type DistributedLock struct {
    key        string
    value      string
    ttl        time.Duration
    mutex      sync.Mutex
    acquired   bool
    stopCh     chan bool
}

type LockManager struct {
    locks map[string]*DistributedLock
    mutex sync.RWMutex
}

func NewLockManager() *LockManager {
    return &LockManager{
        locks: make(map[string]*DistributedLock),
    }
}

func (lm *LockManager) AcquireLock(key, value string, ttl time.Duration) (*DistributedLock, error) {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    if lock, exists := lm.locks[key]; exists {
        if lock.acquired {
            return nil, fmt.Errorf("lock already acquired")
        }
    }
    
    lock := &DistributedLock{
        key:      key,
        value:    value,
        ttl:      ttl,
        acquired: true,
        stopCh:   make(chan bool),
    }
    
    lm.locks[key] = lock
    
    // Start TTL timer
    go lock.startTTLTimer(lm)
    
    return lock, nil
}

func (dl *DistributedLock) startTTLTimer(lm *LockManager) {
    timer := time.NewTimer(dl.ttl)
    
    select {
    case <-timer.C:
        dl.Release()
    case <-dl.stopCh:
        timer.Stop()
    }
}

func (dl *DistributedLock) Release() {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    
    if dl.acquired {
        dl.acquired = false
        close(dl.stopCh)
    }
}

func (dl *DistributedLock) IsAcquired() bool {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    return dl.acquired
}

// Leader Election using Bully Algorithm
type BullyNode struct {
    ID        int
    Peers     []int
    Leader    int
    mutex     sync.RWMutex
    electionCh chan bool
}

func NewBullyNode(id int, peers []int) *BullyNode {
    return &BullyNode{
        ID:        id,
        Peers:     peers,
        Leader:    -1,
        electionCh: make(chan bool),
    }
}

func (bn *BullyNode) StartElection() {
    bn.mutex.Lock()
    bn.Leader = -1
    bn.mutex.Unlock()
    
    // Send election message to higher ID nodes
    higherNodes := bn.getHigherNodes()
    
    if len(higherNodes) == 0 {
        // This node is the highest, declare itself leader
        bn.declareLeader()
        return
    }
    
    // Wait for response from higher nodes
    timeout := time.After(5 * time.Second)
    
    select {
    case <-bn.electionCh:
        // Received response, wait for leader announcement
        bn.waitForLeader()
    case <-timeout:
        // No response, declare self as leader
        bn.declareLeader()
    }
}

func (bn *BullyNode) getHigherNodes() []int {
    var higher []int
    for _, peerID := range bn.Peers {
        if peerID > bn.ID {
            higher = append(higher, peerID)
        }
    }
    return higher
}

func (bn *BullyNode) declareLeader() {
    bn.mutex.Lock()
    bn.Leader = bn.ID
    bn.mutex.Unlock()
    
    // Notify all peers
    for _, peerID := range bn.Peers {
        go bn.notifyLeader(peerID)
    }
}

func (bn *BullyNode) notifyLeader(peerID int) {
    // Simulate network call
    fmt.Printf("Node %d notified peer %d that it's the leader\n", bn.ID, peerID)
}

func (bn *BullyNode) waitForLeader() {
    // Wait for leader announcement
    timeout := time.After(10 * time.Second)
    
    select {
    case <-timeout:
        // Timeout, start new election
        bn.StartElection()
    }
}

func (bn *BullyNode) HandleElectionMessage(fromID int) {
    if fromID < bn.ID {
        // Send response to lower ID node
        go bn.respondToElection(fromID)
        
        // Start own election
        bn.StartElection()
    }
}

func (bn *BullyNode) respondToElection(fromID int) {
    // Simulate network response
    fmt.Printf("Node %d responded to election from node %d\n", bn.ID, fromID)
}
```

## Replication Strategies

### Master-Slave Replication

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ReplicationManager struct {
    Master   *DatabaseNode
    Slaves   []*DatabaseNode
    mutex    sync.RWMutex
    log      []ReplicationLog
}

type DatabaseNode struct {
    ID       int
    Data     map[string]interface{}
    Log      []ReplicationLog
    mutex    sync.RWMutex
    isMaster bool
}

type ReplicationLog struct {
    ID        int
    Timestamp time.Time
    Operation string
    Key       string
    Value     interface{}
}

func NewReplicationManager(masterID int, slaveIDs []int) *ReplicationManager {
    master := &DatabaseNode{
        ID:       masterID,
        Data:     make(map[string]interface{}),
        Log:      make([]ReplicationLog, 0),
        isMaster: true,
    }
    
    var slaves []*DatabaseNode
    for _, id := range slaveIDs {
        slave := &DatabaseNode{
            ID:       id,
            Data:     make(map[string]interface{}),
            Log:      make([]ReplicationLog, 0),
            isMaster: false,
        }
        slaves = append(slaves, slave)
    }
    
    return &ReplicationManager{
        Master: master,
        Slaves: slaves,
        log:    make([]ReplicationLog, 0),
    }
}

func (rm *ReplicationManager) Write(key string, value interface{}) error {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    // Write to master
    rm.Master.mutex.Lock()
    rm.Master.Data[key] = value
    
    // Create replication log entry
    logEntry := ReplicationLog{
        ID:        len(rm.log) + 1,
        Timestamp: time.Now(),
        Operation: "WRITE",
        Key:       key,
        Value:     value,
    }
    
    rm.Master.Log = append(rm.Master.Log, logEntry)
    rm.log = append(rm.log, logEntry)
    rm.Master.mutex.Unlock()
    
    // Replicate to slaves
    for _, slave := range rm.Slaves {
        go rm.replicateToSlave(slave, logEntry)
    }
    
    return nil
}

func (rm *ReplicationManager) Read(key string) (interface{}, error) {
    // Read from master (in real implementation, could read from slaves for read scaling)
    rm.Master.mutex.RLock()
    defer rm.Master.mutex.RUnlock()
    
    value, exists := rm.Master.Data[key]
    if !exists {
        return nil, fmt.Errorf("key not found")
    }
    
    return value, nil
}

func (rm *ReplicationManager) replicateToSlave(slave *DatabaseNode, logEntry ReplicationLog) {
    slave.mutex.Lock()
    defer slave.mutex.Unlock()
    
    // Apply the log entry to slave
    slave.Data[logEntry.Key] = logEntry.Value
    slave.Log = append(slave.Log, logEntry)
    
    fmt.Printf("Replicated %s to slave %d\n", logEntry.Operation, slave.ID)
}

// Multi-Master Replication
type MultiMasterReplication struct {
    Nodes []*DatabaseNode
    mutex sync.RWMutex
}

func NewMultiMasterReplication(nodeIDs []int) *MultiMasterReplication {
    var nodes []*DatabaseNode
    for _, id := range nodeIDs {
        node := &DatabaseNode{
            ID:       id,
            Data:     make(map[string]interface{}),
            Log:      make([]ReplicationLog, 0),
            isMaster: true,
        }
        nodes = append(nodes, node)
    }
    
    return &MultiMasterReplication{
        Nodes: nodes,
    }
}

func (mmr *MultiMasterReplication) Write(nodeID int, key string, value interface{}) error {
    mmr.mutex.Lock()
    defer mmr.mutex.Unlock()
    
    // Find the node
    var node *DatabaseNode
    for _, n := range mmr.Nodes {
        if n.ID == nodeID {
            node = n
            break
        }
    }
    
    if node == nil {
        return fmt.Errorf("node not found")
    }
    
    // Write to local node
    node.mutex.Lock()
    node.Data[key] = value
    
    logEntry := ReplicationLog{
        ID:        len(node.Log) + 1,
        Timestamp: time.Now(),
        Operation: "WRITE",
        Key:       key,
        Value:     value,
    }
    
    node.Log = append(node.Log, logEntry)
    node.mutex.Unlock()
    
    // Replicate to other nodes
    for _, otherNode := range mmr.Nodes {
        if otherNode.ID != nodeID {
            go mmr.replicateToNode(otherNode, logEntry)
        }
    }
    
    return nil
}

func (mmr *MultiMasterReplication) replicateToNode(node *DatabaseNode, logEntry ReplicationLog) {
    node.mutex.Lock()
    defer node.mutex.Unlock()
    
    // Apply the log entry
    node.Data[logEntry.Key] = logEntry.Value
    node.Log = append(node.Log, logEntry)
    
    fmt.Printf("Replicated %s to node %d\n", logEntry.Operation, node.ID)
}
```

## Partitioning and Sharding

### Consistent Hashing Implementation

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
    replicas int
    mutex    sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        ring:     make(map[uint32]string),
        replicas: replicas,
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        key := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.ring[key] = node
    }
    
    ch.updateSortedKeys()
}

func (ch *ConsistentHash) RemoveNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        key := ch.hash(fmt.Sprintf("%s:%d", node, i))
        delete(ch.ring, key)
    }
    
    ch.updateSortedKeys()
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.ring) == 0 {
        return ""
    }
    
    hash := ch.hash(key)
    idx := sort.Search(len(ch.sortedKeys), func(i int) bool {
        return ch.sortedKeys[i] >= hash
    })
    
    if idx == len(ch.sortedKeys) {
        idx = 0
    }
    
    return ch.ring[ch.sortedKeys[idx]]
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (ch *ConsistentHash) updateSortedKeys() {
    ch.sortedKeys = make([]uint32, 0, len(ch.ring))
    for key := range ch.ring {
        ch.sortedKeys = append(ch.sortedKeys, key)
    }
    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
}

// Range-based Sharding
type RangeShard struct {
    StartKey string
    EndKey   string
    NodeID   int
}

type RangeSharding struct {
    shards []RangeShard
    mutex  sync.RWMutex
}

func NewRangeSharding() *RangeSharding {
    return &RangeSharding{
        shards: make([]RangeShard, 0),
    }
}

func (rs *RangeSharding) AddShard(startKey, endKey string, nodeID int) {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()
    
    shard := RangeShard{
        StartKey: startKey,
        EndKey:   endKey,
        NodeID:   nodeID,
    }
    
    rs.shards = append(rs.shards, shard)
    rs.sortShards()
}

func (rs *RangeSharding) GetNode(key string) int {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()
    
    for _, shard := range rs.shards {
        if key >= shard.StartKey && key < shard.EndKey {
            return shard.NodeID
        }
    }
    
    return -1
}

func (rs *RangeSharding) sortShards() {
    sort.Slice(rs.shards, func(i, j int) bool {
        return rs.shards[i].StartKey < rs.shards[j].StartKey
    })
}
```

## Interview Questions

### Basic Concepts
1. **What is the CAP theorem?**
2. **Explain the difference between strong and eventual consistency.**
3. **What are the challenges in distributed systems?**
4. **How does the Raft algorithm work?**
5. **What is the difference between horizontal and vertical scaling?**

### Advanced Topics
1. **How would you implement a distributed lock?**
2. **Explain the difference between Raft and Paxos.**
3. **How do you handle split-brain scenarios?**
4. **What are the trade-offs in different replication strategies?**
5. **How would you implement consistent hashing?**

### System Design
1. **Design a distributed key-value store.**
2. **How would you implement a distributed cache?**
3. **Design a distributed message queue.**
4. **How would you implement a distributed database?**
5. **Design a distributed file system.**

## Conclusion

Distributed systems are complex but essential for building scalable applications. Key areas to master:

- **Consensus Algorithms**: Raft, Paxos, PBFT
- **Replication**: Master-slave, multi-master, quorum-based
- **Partitioning**: Consistent hashing, range-based sharding
- **Coordination**: Distributed locks, leader election
- **Consistency**: Strong, eventual, causal consistency
- **Fault Tolerance**: Handling failures, split-brain scenarios

Understanding these concepts helps in:
- Designing scalable systems
- Handling distributed data
- Managing consistency
- Building fault-tolerant systems
- Preparing for technical interviews

This guide provides a comprehensive foundation for distributed systems concepts and their practical implementation in Go.


## Distributed Caching

<!-- AUTO-GENERATED ANCHOR: originally referenced as #distributed-caching -->

Placeholder content. Please replace with proper section.


## Message Queues And Event Streaming

<!-- AUTO-GENERATED ANCHOR: originally referenced as #message-queues-and-event-streaming -->

Placeholder content. Please replace with proper section.


## Service Discovery

<!-- AUTO-GENERATED ANCHOR: originally referenced as #service-discovery -->

Placeholder content. Please replace with proper section.


## Load Balancing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #load-balancing -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
