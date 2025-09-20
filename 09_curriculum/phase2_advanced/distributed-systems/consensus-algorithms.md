# Consensus Algorithms

## Overview

This module covers distributed consensus algorithms including Raft, PBFT (Practical Byzantine Fault Tolerance), and Proof of Work. These algorithms are essential for building reliable distributed systems that can tolerate failures and maintain consistency.

## Table of Contents

1. [Raft Consensus](#raft-consensus)
2. [PBFT (Practical Byzantine Fault Tolerance)](#pbft-practical-byzantine-fault-tolerance)
3. [Proof of Work](#proof-of-work)
4. [Applications](#applications)
5. [Complexity Analysis](#complexity-analysis)
6. [Follow-up Questions](#follow-up-questions)

## Raft Consensus

### Theory

Raft is a consensus algorithm designed to be easy to understand and implement. It achieves consensus through leader election and log replication, ensuring that all nodes in a cluster agree on the same sequence of operations.

### Key Components

- **Leader Election**: Nodes elect a leader to coordinate operations
- **Log Replication**: Leader replicates log entries to followers
- **Safety**: Ensures consistency even during failures

### Implementations

#### Golang Implementation

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
    electionTimeout time.Duration
    heartbeatTimeout time.Duration
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
        ID:              id,
        State:           "Follower",
        CurrentTerm:     0,
        VotedFor:        -1,
        Log:             make([]LogEntry, 0),
        CommitIndex:     -1,
        LastApplied:     -1,
        NextIndex:       make(map[int]int),
        MatchIndex:      make(map[int]int),
        Peers:           peers,
        electionTimeout: time.Duration(150+rand.Intn(150)) * time.Millisecond,
        heartbeatTimeout: 50 * time.Millisecond,
    }
}

func (rn *RaftNode) StartElection() {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    rn.State = "Candidate"
    rn.CurrentTerm++
    rn.VotedFor = rn.ID
    
    votes := 1 // Vote for self
    totalNodes := len(rn.Peers) + 1
    
    fmt.Printf("Node %d starting election for term %d\n", rn.ID, rn.CurrentTerm)
    
    // Request votes from peers
    for _, peerID := range rn.Peers {
        go func(peerID int) {
            if rn.requestVote(peerID) {
                rn.mutex.Lock()
                votes++
                rn.mutex.Unlock()
            }
        }(peerID)
    }
    
    // Check if we won the election
    if votes > totalNodes/2 {
        rn.becomeLeader()
    }
}

func (rn *RaftNode) requestVote(peerID int) bool {
    // Simulate network request
    time.Sleep(10 * time.Millisecond)
    
    // Simple voting logic (in real implementation, this would be an RPC)
    return rand.Float32() < 0.7 // 70% chance of getting vote
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
    ticker := time.NewTicker(rn.heartbeatTimeout)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            rn.mutex.RLock()
            if rn.State != "Leader" {
                rn.mutex.RUnlock()
                return
            }
            rn.mutex.RUnlock()
            
            rn.sendHeartbeat()
        }
    }
}

func (rn *RaftNode) sendHeartbeat() {
    rn.mutex.RLock()
    term := rn.CurrentTerm
    leaderID := rn.ID
    prevLogIndex := len(rn.Log) - 1
    prevLogTerm := -1
    if prevLogIndex >= 0 {
        prevLogTerm = rn.Log[prevLogIndex].Term
    }
    entries := make([]LogEntry, 0) // Empty for heartbeat
    leaderCommit := rn.CommitIndex
    rn.mutex.RUnlock()
    
    for _, peerID := range rn.Peers {
        go func(peerID int) {
            rn.appendEntries(peerID, term, leaderID, prevLogIndex, prevLogTerm, entries, leaderCommit)
        }(peerID)
    }
}

func (rn *RaftNode) appendEntries(peerID, term, leaderID, prevLogIndex, prevLogTerm int, entries []LogEntry, leaderCommit int) {
    // Simulate network request
    time.Sleep(5 * time.Millisecond)
    
    // Simple success response (in real implementation, this would be an RPC)
    success := rand.Float32() < 0.9 // 90% success rate
    
    if success {
        rn.mutex.Lock()
        if rn.NextIndex[peerID] < len(rn.Log) {
            rn.NextIndex[peerID]++
            rn.MatchIndex[peerID] = rn.NextIndex[peerID] - 1
        }
        rn.mutex.Unlock()
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
    fmt.Printf("Node %d appended command '%s' to log\n", rn.ID, command)
}

func (rn *RaftNode) Start() {
    go rn.run()
}

func (rn *RaftNode) run() {
    for {
        rn.mutex.RLock()
        state := rn.State
        rn.mutex.RUnlock()
        
        switch state {
        case "Follower":
            rn.runFollower()
        case "Candidate":
            rn.runCandidate()
        case "Leader":
            rn.runLeader()
        }
    }
}

func (rn *RaftNode) runFollower() {
    timeout := time.NewTimer(rn.electionTimeout)
    defer timeout.Stop()
    
    select {
    case <-timeout.C:
        rn.StartElection()
    }
}

func (rn *RaftNode) runCandidate() {
    timeout := time.NewTimer(rn.electionTimeout)
    defer timeout.Stop()
    
    select {
    case <-timeout.C:
        rn.StartElection()
    }
}

func (rn *RaftNode) runLeader() {
    // Leader logic is handled by heartbeat goroutine
    time.Sleep(100 * time.Millisecond)
}

func main() {
    // Create a 3-node Raft cluster
    peers1 := []int{2, 3}
    peers2 := []int{1, 3}
    peers3 := []int{1, 2}
    
    node1 := NewRaftNode(1, peers1)
    node2 := NewRaftNode(2, peers2)
    node3 := NewRaftNode(3, peers3)
    
    // Start all nodes
    go node1.Start()
    go node2.Start()
    go node3.Start()
    
    // Wait for leader election
    time.Sleep(2 * time.Second)
    
    // Append some commands
    node1.AppendCommand("command1")
    node2.AppendCommand("command2")
    node3.AppendCommand("command3")
    
    // Keep running
    time.Sleep(5 * time.Second)
}
```

#### Node.js Implementation

```javascript
class RaftNode {
    constructor(id, peers) {
        this.id = id;
        this.state = 'Follower';
        this.currentTerm = 0;
        this.votedFor = -1;
        this.log = [];
        this.commitIndex = -1;
        this.lastApplied = -1;
        this.nextIndex = new Map();
        this.matchIndex = new Map();
        this.peers = peers;
        this.electionTimeout = 150 + Math.random() * 150;
        this.heartbeatTimeout = 50;
    }

    startElection() {
        this.state = 'Candidate';
        this.currentTerm++;
        this.votedFor = this.id;
        
        let votes = 1; // Vote for self
        const totalNodes = this.peers.length + 1;
        
        console.log(`Node ${this.id} starting election for term ${this.currentTerm}`);
        
        // Request votes from peers
        this.peers.forEach(peerID => {
            if (this.requestVote(peerID)) {
                votes++;
            }
        });
        
        // Check if we won the election
        if (votes > totalNodes / 2) {
            this.becomeLeader();
        }
    }

    requestVote(peerID) {
        // Simulate network request
        return Math.random() < 0.7; // 70% chance of getting vote
    }

    becomeLeader() {
        this.state = 'Leader';
        console.log(`Node ${this.id} became leader for term ${this.currentTerm}`);
        
        // Initialize nextIndex and matchIndex
        this.peers.forEach(peerID => {
            this.nextIndex.set(peerID, this.log.length);
            this.matchIndex.set(peerID, -1);
        });
        
        // Start sending heartbeats
        this.sendHeartbeats();
    }

    sendHeartbeats() {
        setInterval(() => {
            if (this.state !== 'Leader') return;
            this.sendHeartbeat();
        }, this.heartbeatTimeout);
    }

    sendHeartbeat() {
        const term = this.currentTerm;
        const leaderID = this.id;
        const prevLogIndex = this.log.length - 1;
        const prevLogTerm = prevLogIndex >= 0 ? this.log[prevLogIndex].term : -1;
        const entries = []; // Empty for heartbeat
        const leaderCommit = this.commitIndex;
        
        this.peers.forEach(peerID => {
            this.appendEntries(peerID, term, leaderID, prevLogIndex, prevLogTerm, entries, leaderCommit);
        });
    }

    appendEntries(peerID, term, leaderID, prevLogIndex, prevLogTerm, entries, leaderCommit) {
        // Simulate network request
        const success = Math.random() < 0.9; // 90% success rate
        
        if (success) {
            if (this.nextIndex.get(peerID) < this.log.length) {
                this.nextIndex.set(peerID, this.nextIndex.get(peerID) + 1);
                this.matchIndex.set(peerID, this.nextIndex.get(peerID) - 1);
            }
        }
    }

    appendCommand(command) {
        if (this.state !== 'Leader') {
            console.log(`Node ${this.id} is not leader, cannot append command`);
            return;
        }
        
        const entry = {
            term: this.currentTerm,
            command: command
        };
        
        this.log.push(entry);
        console.log(`Node ${this.id} appended command '${command}' to log`);
    }

    start() {
        this.run();
    }

    run() {
        setInterval(() => {
            switch (this.state) {
                case 'Follower':
                    this.runFollower();
                    break;
                case 'Candidate':
                    this.runCandidate();
                    break;
                case 'Leader':
                    this.runLeader();
                    break;
            }
        }, 100);
    }

    runFollower() {
        setTimeout(() => {
            this.startElection();
        }, this.electionTimeout);
    }

    runCandidate() {
        setTimeout(() => {
            this.startElection();
        }, this.electionTimeout);
    }

    runLeader() {
        // Leader logic is handled by heartbeat
    }
}

// Example usage
const node1 = new RaftNode(1, [2, 3]);
const node2 = new RaftNode(2, [1, 3]);
const node3 = new RaftNode(3, [1, 2]);

node1.start();
node2.start();
node3.start();

// Wait for leader election
setTimeout(() => {
    node1.appendCommand('command1');
    node2.appendCommand('command2');
    node3.appendCommand('command3');
}, 2000);
```

## PBFT (Practical Byzantine Fault Tolerance)

### Theory

PBFT is a consensus algorithm that can tolerate Byzantine failures (arbitrary failures including malicious behavior) as long as less than one-third of the nodes are faulty.

### Key Properties

- **Safety**: Non-faulty nodes never decide on conflicting values
- **Liveness**: Non-faulty nodes eventually decide on a value
- **Byzantine Fault Tolerance**: Can handle up to (n-1)/3 faulty nodes

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type PBFTNode struct {
    ID          int
    State       string
    View        int
    Sequence    int
    PrePrepare  map[string]*PrePrepareMessage
    Prepare     map[string][]*PrepareMessage
    Commit      map[string][]*CommitMessage
    Checkpoint  map[int]*CheckpointMessage
    Peers       []int
    mutex       sync.RWMutex
}

type PrePrepareMessage struct {
    View     int
    Sequence int
    Digest   string
    Request  string
}

type PrepareMessage struct {
    View     int
    Sequence int
    Digest   string
    NodeID   int
}

type CommitMessage struct {
    View     int
    Sequence int
    Digest   string
    NodeID   int
}

type CheckpointMessage struct {
    Sequence int
    Digest   string
    NodeID   int
}

func NewPBFTNode(id int, peers []int) *PBFTNode {
    return &PBFTNode{
        ID:         id,
        State:      "Normal",
        View:       0,
        Sequence:   0,
        PrePrepare: make(map[string]*PrePrepareMessage),
        Prepare:    make(map[string][]*PrepareMessage),
        Commit:     make(map[string][]*CommitMessage),
        Checkpoint: make(map[int]*CheckpointMessage),
        Peers:      peers,
    }
}

func (pn *PBFTNode) PrePrepare(request string) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    if pn.State != "Normal" {
        return
    }
    
    pn.Sequence++
    digest := fmt.Sprintf("digest_%s_%d", request, pn.Sequence)
    
    prePrepare := &PrePrepareMessage{
        View:     pn.View,
        Sequence: pn.Sequence,
        Digest:   digest,
        Request:  request,
    }
    
    pn.PrePrepare[digest] = prePrepare
    
    fmt.Printf("Node %d: PrePrepare for request '%s' (seq: %d)\n", pn.ID, request, pn.Sequence)
    
    // Send PrePrepare to all peers
    for _, peerID := range pn.Peers {
        go pn.sendPrePrepare(peerID, prePrepare)
    }
}

func (pn *PBFTNode) sendPrePrepare(peerID int, prePrepare *PrePrepareMessage) {
    // Simulate network delay
    time.Sleep(10 * time.Millisecond)
    
    // In real implementation, this would be an RPC call
    fmt.Printf("Node %d -> Node %d: PrePrepare (seq: %d)\n", pn.ID, peerID, prePrepare.Sequence)
}

func (pn *PBFTNode) Prepare(digest string) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    if pn.State != "Normal" {
        return
    }
    
    prepare := &PrepareMessage{
        View:     pn.View,
        Sequence: pn.Sequence,
        Digest:   digest,
        NodeID:   pn.ID,
    }
    
    pn.Prepare[digest] = append(pn.Prepare[digest], prepare)
    
    fmt.Printf("Node %d: Prepare for digest '%s'\n", pn.ID, digest)
    
    // Send Prepare to all peers
    for _, peerID := range pn.Peers {
        go pn.sendPrepare(peerID, prepare)
    }
}

func (pn *PBFTNode) sendPrepare(peerID int, prepare *PrepareMessage) {
    // Simulate network delay
    time.Sleep(10 * time.Millisecond)
    
    // In real implementation, this would be an RPC call
    fmt.Printf("Node %d -> Node %d: Prepare (seq: %d)\n", pn.ID, peerID, prepare.Sequence)
}

func (pn *PBFTNode) Commit(digest string) {
    pn.mutex.Lock()
    defer pn.mutex.Unlock()
    
    if pn.State != "Normal" {
        return
    }
    
    commit := &CommitMessage{
        View:     pn.View,
        Sequence: pn.Sequence,
        Digest:   digest,
        NodeID:   pn.ID,
    }
    
    pn.Commit[digest] = append(pn.Commit[digest], commit)
    
    fmt.Printf("Node %d: Commit for digest '%s'\n", pn.ID, digest)
    
    // Send Commit to all peers
    for _, peerID := range pn.Peers {
        go pn.sendCommit(peerID, commit)
    }
}

func (pn *PBFTNode) sendCommit(peerID int, commit *CommitMessage) {
    // Simulate network delay
    time.Sleep(10 * time.Millisecond)
    
    // In real implementation, this would be an RPC call
    fmt.Printf("Node %d -> Node %d: Commit (seq: %d)\n", pn.ID, peerID, commit.Sequence)
}

func (pn *PBFTNode) CheckConsensus(digest string) bool {
    pn.mutex.RLock()
    defer pn.mutex.RUnlock()
    
    // Check if we have enough Prepare messages (2f + 1)
    prepareCount := len(pn.Prepare[digest])
    totalNodes := len(pn.Peers) + 1
    required := (2 * totalNodes / 3) + 1
    
    if prepareCount >= required {
        fmt.Printf("Node %d: Consensus reached for digest '%s' (prepares: %d, required: %d)\n", 
                   pn.ID, digest, prepareCount, required)
        return true
    }
    
    return false
}

func main() {
    // Create a 4-node PBFT cluster (can tolerate 1 Byzantine fault)
    peers1 := []int{2, 3, 4}
    peers2 := []int{1, 3, 4}
    peers3 := []int{1, 2, 4}
    peers4 := []int{1, 2, 3}
    
    node1 := NewPBFTNode(1, peers1)
    node2 := NewPBFTNode(2, peers2)
    node3 := NewPBFTNode(3, peers3)
    node4 := NewPBFTNode(4, peers4)
    
    // Simulate consensus process
    request := "update_database"
    node1.PrePrepare(request)
    
    // Wait for consensus
    time.Sleep(1 * time.Second)
    
    // Check consensus
    digest := fmt.Sprintf("digest_%s_%d", request, 1)
    node1.CheckConsensus(digest)
}
```

## Proof of Work

### Theory

Proof of Work is a consensus mechanism used in blockchain systems where nodes compete to solve a computationally difficult puzzle to validate transactions and create new blocks.

### Key Properties

- **Computational Difficulty**: Requires significant computational power
- **Probabilistic**: No guarantee of when a solution will be found
- **Energy Intensive**: Consumes large amounts of electricity

### Implementations

#### Golang Implementation

```go
package main

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type Block struct {
    Index     int
    Timestamp time.Time
    Data      string
    PrevHash  string
    Hash      string
    Nonce     int
}

type ProofOfWork struct {
    Block      *Block
    Target     string
    Difficulty int
}

func NewBlock(index int, data, prevHash string) *Block {
    return &Block{
        Index:     index,
        Timestamp: time.Now(),
        Data:      data,
        PrevHash:  prevHash,
        Nonce:     0,
    }
}

func NewProofOfWork(block *Block, difficulty int) *ProofOfWork {
    target := make([]byte, 32)
    for i := 0; i < difficulty; i++ {
        target[i] = 0
    }
    
    return &ProofOfWork{
        Block:      block,
        Target:     hex.EncodeToString(target),
        Difficulty: difficulty,
    }
}

func (pow *ProofOfWork) calculateHash(nonce int) string {
    data := fmt.Sprintf("%d%s%s%s%d", 
        pow.Block.Index, 
        pow.Block.Timestamp.Format(time.RFC3339), 
        pow.Block.Data, 
        pow.Block.PrevHash, 
        nonce)
    
    hash := sha256.Sum256([]byte(data))
    return hex.EncodeToString(hash[:])
}

func (pow *ProofOfWork) isValidHash(hash string) bool {
    return hash[:pow.Difficulty] == pow.Target[:pow.Difficulty]
}

func (pow *ProofOfWork) Mine() (int, string) {
    var nonce int
    var hash string
    
    fmt.Printf("Mining block %d...\n", pow.Block.Index)
    start := time.Now()
    
    for {
        hash = pow.calculateHash(nonce)
        if pow.isValidHash(hash) {
            break
        }
        nonce++
        
        // Print progress every 100000 attempts
        if nonce%100000 == 0 {
            fmt.Printf("Nonce: %d, Hash: %s\n", nonce, hash)
        }
    }
    
    duration := time.Since(start)
    fmt.Printf("Block %d mined in %v with nonce %d\n", pow.Block.Index, duration, nonce)
    fmt.Printf("Final hash: %s\n", hash)
    
    return nonce, hash
}

func (pow *ProofOfWork) MineParallel(workers int) (int, string) {
    var wg sync.WaitGroup
    var mu sync.Mutex
    var found bool
    var resultNonce int
    var resultHash string
    
    fmt.Printf("Mining block %d with %d workers...\n", pow.Block.Index, workers)
    start := time.Now()
    
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            
            nonce := workerID
            for {
                mu.Lock()
                if found {
                    mu.Unlock()
                    return
                }
                mu.Unlock()
                
                hash := pow.calculateHash(nonce)
                if pow.isValidHash(hash) {
                    mu.Lock()
                    if !found {
                        found = true
                        resultNonce = nonce
                        resultHash = hash
                    }
                    mu.Unlock()
                    return
                }
                nonce += workers
            }
        }(i)
    }
    
    wg.Wait()
    duration := time.Since(start)
    fmt.Printf("Block %d mined in %v with nonce %d\n", pow.Block.Index, duration, resultNonce)
    fmt.Printf("Final hash: %s\n", resultHash)
    
    return resultNonce, resultHash
}

func main() {
    // Create genesis block
    genesisBlock := NewBlock(0, "Genesis Block", "")
    pow := NewProofOfWork(genesisBlock, 4) // Difficulty 4 (4 leading zeros)
    
    // Mine the block
    nonce, hash := pow.Mine()
    genesisBlock.Nonce = nonce
    genesisBlock.Hash = hash
    
    fmt.Printf("Genesis Block: %+v\n", genesisBlock)
    
    // Create second block
    block2 := NewBlock(1, "Second Block", genesisBlock.Hash)
    pow2 := NewProofOfWork(block2, 4)
    
    // Mine with parallel workers
    nonce2, hash2 := pow2.MineParallel(4)
    block2.Nonce = nonce2
    block2.Hash = hash2
    
    fmt.Printf("Block 2: %+v\n", block2)
}
```

## Follow-up Questions

### 1. Consensus Algorithm Selection
**Q: When would you choose Raft over PBFT for a distributed system?**
A: Choose Raft when you have a trusted environment with crash failures only, as it's simpler to implement and understand. Choose PBFT when you need to tolerate Byzantine failures (malicious nodes) and can afford the higher complexity and communication overhead.

### 2. Performance Trade-offs
**Q: What are the performance implications of different consensus algorithms?**
A: Raft has lower latency but requires a majority of nodes to be available. PBFT has higher latency due to multiple rounds of communication but can tolerate Byzantine failures. Proof of Work has high energy consumption but provides strong security guarantees.

### 3. Fault Tolerance
**Q: How do you determine the minimum number of nodes needed for each consensus algorithm?**
A: Raft needs at least 3 nodes to tolerate 1 failure. PBFT needs at least 3f+1 nodes to tolerate f Byzantine failures. Proof of Work doesn't have a minimum node requirement but needs sufficient hash power to maintain security.

## Complexity Analysis

| Algorithm | Time Complexity | Communication Complexity | Fault Tolerance |
|-----------|----------------|-------------------------|-----------------|
| Raft | O(log n) | O(n) per operation | Crash failures |
| PBFT | O(1) | O(n²) per operation | Byzantine failures |
| Proof of Work | O(2^d) | O(1) | Byzantine failures |

## Applications

1. **Raft**: Distributed databases, configuration management
2. **PBFT**: Blockchain systems, financial applications
3. **Proof of Work**: Cryptocurrencies, blockchain consensus

---

**Next**: [Distributed Storage](distributed-storage.md) | **Previous**: [Distributed Systems](README.md) | **Up**: [Distributed Systems](README.md)
