# 🌐 **Distributed Systems - Asli Engineering Deep Dive**

## 📊 **Based on Arpit Bhyani's Distributed Systems Videos**

---

## 🎯 **Core Distributed Systems Concepts**

### **1. CAP Theorem Deep Dive**

#### **Understanding the Trade-offs**

##### **Consistency (C)**
All nodes see the same data at the same time.

```go
type ConsistentSystem struct {
    nodes []Node
    quorum int
    mutex sync.RWMutex
}

func (cs *ConsistentSystem) Write(key, value string) error {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    // Write to all nodes for consistency
    successCount := 0
    for _, node := range cs.nodes {
        if err := node.Write(key, value); err == nil {
            successCount++
        }
    }
    
    if successCount >= cs.quorum {
        return nil
    }
    
    return errors.New("failed to achieve consistency")
}

func (cs *ConsistentSystem) Read(key string) (string, error) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    // Read from majority for consistency
    values := make(map[string]int)
    for _, node := range cs.nodes {
        if value, err := node.Read(key); err == nil {
            values[value]++
        }
    }
    
    // Return most common value
    maxCount := 0
    var result string
    for value, count := range values {
        if count > maxCount {
            maxCount = count
            result = value
        }
    }
    
    if maxCount >= cs.quorum {
        return result, nil
    }
    
    return "", errors.New("failed to achieve consistency")
}
```

##### **Availability (A)**
System remains operational and accessible.

```go
type AvailableSystem struct {
    nodes []Node
    healthChecker *HealthChecker
    loadBalancer *LoadBalancer
}

func (as *AvailableSystem) HandleRequest(req *Request) (*Response, error) {
    // Try to find a healthy node
    for _, node := range as.nodes {
        if as.healthChecker.IsHealthy(node) {
            return node.Process(req)
        }
    }
    
    // If no healthy nodes, return cached response
    return as.getCachedResponse(req)
}

func (as *AvailableSystem) getCachedResponse(req *Request) (*Response, error) {
    // Return stale data if available
    if cached, exists := as.cache.Get(req.Key); exists {
        return cached, nil
    }
    
    return nil, errors.New("service unavailable")
}
```

##### **Partition Tolerance (P)**
System continues to work despite network failures.

```go
type PartitionTolerantSystem struct {
    partitions map[string][]Node
    replicas   map[string][]Node
}

func (pts *PartitionTolerantSystem) HandlePartition(partitionID string) error {
    // Replicate data across partitions
    for key, value := range pts.getPartitionData(partitionID) {
        for _, replica := range pts.replicas[partitionID] {
            go func(r Node) {
                r.Write(key, value)
            }(replica)
        }
    }
    
    return nil
}
```

### **2. Consensus Algorithms**

#### **Raft Algorithm Implementation**

```go
type RaftNode struct {
    ID          string
    state       NodeState
    currentTerm int
    votedFor    string
    log         []LogEntry
    commitIndex int
    lastApplied int
    nextIndex   map[string]int
    matchIndex  map[string]int
    peers       map[string]*RaftNode
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

type RequestVoteRequest struct {
    Term         int
    CandidateID  string
    LastLogIndex int
    LastLogTerm  int
}

type RequestVoteResponse struct {
    Term        int
    VoteGranted bool
}

func (rn *RaftNode) RequestVote(req *RequestVoteRequest) *RequestVoteResponse {
    rn.mutex.Lock()
    defer rn.mutex.Unlock()
    
    response := &RequestVoteResponse{
        Term:        rn.currentTerm,
        VoteGranted: false,
    }
    
    if req.Term > rn.currentTerm {
        rn.currentTerm = req.Term
        rn.state = Follower
        rn.votedFor = ""
    }
    
    if req.Term == rn.currentTerm && 
       (rn.votedFor == "" || rn.votedFor == req.CandidateID) &&
       rn.isUpToDate(req.LastLogIndex, req.LastLogTerm) {
        rn.votedFor = req.CandidateID
        response.VoteGranted = true
    }
    
    return response
}

func (rn *RaftNode) isUpToDate(lastLogIndex, lastLogTerm int) bool {
    if len(rn.log) == 0 {
        return true
    }
    
    lastEntry := rn.log[len(rn.log)-1]
    return lastLogTerm > lastEntry.Term || 
           (lastLogTerm == lastEntry.Term && lastLogIndex >= lastEntry.Index)
}

func (rn *RaftNode) StartElection() {
    rn.mutex.Lock()
    rn.state = Candidate
    rn.currentTerm++
    rn.votedFor = rn.ID
    rn.mutex.Unlock()
    
    votes := 1 // Vote for self
    totalVotes := len(rn.peers) + 1
    
    for peerID, peer := range rn.peers {
        go func(id string, p *RaftNode) {
            req := &RequestVoteRequest{
                Term:         rn.currentTerm,
                CandidateID:  rn.ID,
                LastLogIndex: rn.getLastLogIndex(),
                LastLogTerm:  rn.getLastLogTerm(),
            }
            
            resp := p.RequestVote(req)
            
            rn.mutex.Lock()
            if resp.Term > rn.currentTerm {
                rn.currentTerm = resp.Term
                rn.state = Follower
                rn.votedFor = ""
            } else if resp.VoteGranted {
                votes++
                if votes > totalVotes/2 {
                    rn.becomeLeader()
                }
            }
            rn.mutex.Unlock()
        }(peerID, peer)
    }
}

func (rn *RaftNode) becomeLeader() {
    rn.state = Leader
    
    // Initialize nextIndex and matchIndex
    for peerID := range rn.peers {
        rn.nextIndex[peerID] = rn.getLastLogIndex() + 1
        rn.matchIndex[peerID] = 0
    }
    
    // Start sending heartbeats
    go rn.sendHeartbeats()
}
```

#### **Paxos Algorithm Implementation**

```go
type PaxosNode struct {
    ID           string
    proposers    map[string]*Proposer
    acceptors    map[string]*Acceptor
    learners     map[string]*Learner
    mutex        sync.RWMutex
}

type Proposer struct {
    ID        string
    proposalNumber int
    value     interface{}
    quorum    int
}

type Acceptor struct {
    ID              string
    promisedNumber  int
    acceptedNumber  int
    acceptedValue   interface{}
    mutex           sync.RWMutex
}

type Learner struct {
    ID       string
    accepted map[string]interface{}
    mutex    sync.RWMutex
}

func (pn *PaxosNode) Propose(value interface{}) error {
    proposer := &Proposer{
        ID:     pn.ID,
        value:  value,
        quorum: (len(pn.acceptors) / 2) + 1,
    }
    
    // Phase 1: Prepare
    promises := pn.prepare(proposer)
    if len(promises) < proposer.quorum {
        return errors.New("failed to get majority promises")
    }
    
    // Phase 2: Accept
    accepts := pn.accept(proposer, promises)
    if len(accepts) < proposer.quorum {
        return errors.New("failed to get majority accepts")
    }
    
    return nil
}

func (pn *PaxosNode) prepare(proposer *Proposer) []Promise {
    promises := make([]Promise, 0)
    
    for acceptorID, acceptor := range pn.acceptors {
        promise := acceptor.Prepare(proposer.proposalNumber)
        if promise != nil {
            promises = append(promises, *promise)
        }
    }
    
    return promises
}

func (acceptor *Acceptor) Prepare(proposalNumber int) *Promise {
    acceptor.mutex.Lock()
    defer acceptor.mutex.Unlock()
    
    if proposalNumber > acceptor.promisedNumber {
        acceptor.promisedNumber = proposalNumber
        return &Promise{
            AcceptorID:      acceptor.ID,
            PromisedNumber:  proposalNumber,
            AcceptedNumber:  acceptor.acceptedNumber,
            AcceptedValue:   acceptor.acceptedValue,
        }
    }
    
    return nil
}
```

### **3. Data Replication Strategies**

#### **Master-Slave Replication**

```go
type MasterSlaveReplication struct {
    master *Database
    slaves []*Database
    mutex  sync.RWMutex
}

func (msr *MasterSlaveReplication) Write(key, value string) error {
    msr.mutex.Lock()
    defer msr.mutex.Unlock()
    
    // Write to master
    if err := msr.master.Write(key, value); err != nil {
        return err
    }
    
    // Replicate to slaves asynchronously
    for _, slave := range msr.slaves {
        go func(s *Database) {
            s.Write(key, value)
        }(slave)
    }
    
    return nil
}

func (msr *MasterSlaveReplication) Read(key string) (string, error) {
    msr.mutex.RLock()
    defer msr.mutex.RUnlock()
    
    // Read from master for consistency
    return msr.master.Read(key)
}

func (msr *MasterSlaveReplication) ReadFromSlave(key string) (string, error) {
    msr.mutex.RLock()
    defer msr.mutex.RUnlock()
    
    // Read from any available slave
    for _, slave := range msr.slaves {
        if value, err := slave.Read(key); err == nil {
            return value, nil
        }
    }
    
    // Fallback to master
    return msr.master.Read(key)
}
```

#### **Master-Master Replication**

```go
type MasterMasterReplication struct {
    masters []*Database
    conflictResolver *ConflictResolver
    mutex   sync.RWMutex
}

func (mmr *MasterMasterReplication) Write(key, value string) error {
    mmr.mutex.Lock()
    defer mmr.mutex.Unlock()
    
    // Write to local master
    if err := mmr.masters[0].Write(key, value); err != nil {
        return err
    }
    
    // Replicate to other masters
    for i := 1; i < len(mmr.masters); i++ {
        go func(master *Database) {
            master.Write(key, value)
        }(mmr.masters[i])
    }
    
    return nil
}

func (mmr *MasterMasterReplication) Read(key string) (string, error) {
    mmr.mutex.RLock()
    defer mmr.mutex.RUnlock()
    
    // Read from local master
    return mmr.masters[0].Read(key)
}

func (mmr *MasterMasterReplication) HandleConflict(key string, values []string) (string, error) {
    return mmr.conflictResolver.Resolve(key, values)
}
```

### **4. Sharding Strategies**

#### **Range-Based Sharding**

```go
type RangeSharding struct {
    shards []Shard
    ranges []Range
}

type Shard struct {
    ID    string
    Start int
    End   int
    DB    *Database
}

type Range struct {
    Start int
    End   int
    ShardID string
}

func (rs *RangeSharding) GetShard(key int) *Shard {
    for i, range_ := range rs.ranges {
        if key >= range_.Start && key < range_.End {
            return rs.shards[i]
        }
    }
    
    // Default to last shard
    return rs.shards[len(rs.shards)-1]
}

func (rs *RangeSharding) Write(key int, value string) error {
    shard := rs.GetShard(key)
    return shard.DB.Write(fmt.Sprintf("%d", key), value)
}

func (rs *RangeSharding) Read(key int) (string, error) {
    shard := rs.GetShard(key)
    return shard.DB.Read(fmt.Sprintf("%d", key))
}
```

#### **Hash-Based Sharding**

```go
type HashSharding struct {
    shards []Shard
    hashFunc func(string) int
}

func NewHashSharding(shardCount int) *HashSharding {
    shards := make([]Shard, shardCount)
    for i := 0; i < shardCount; i++ {
        shards[i] = Shard{
            ID: fmt.Sprintf("shard_%d", i),
            DB: NewDatabase(),
        }
    }
    
    return &HashSharding{
        shards: shards,
        hashFunc: func(key string) int {
            hash := 0
            for _, c := range key {
                hash += int(c)
            }
            return hash % shardCount
        },
    }
}

func (hs *HashSharding) GetShard(key string) *Shard {
    shardIndex := hs.hashFunc(key)
    return &hs.shards[shardIndex]
}

func (hs *HashSharding) Write(key, value string) error {
    shard := hs.GetShard(key)
    return shard.DB.Write(key, value)
}

func (hs *HashSharding) Read(key string) (string, error) {
    shard := hs.GetShard(key)
    return shard.DB.Read(key)
}
```

#### **Directory-Based Sharding**

```go
type DirectorySharding struct {
    shards     []Shard
    directory  map[string]string // key -> shard_id
    mutex      sync.RWMutex
}

func (ds *DirectorySharding) GetShard(key string) (*Shard, error) {
    ds.mutex.RLock()
    shardID, exists := ds.directory[key]
    ds.mutex.RUnlock()
    
    if !exists {
        return nil, errors.New("key not found in directory")
    }
    
    for _, shard := range ds.shards {
        if shard.ID == shardID {
            return &shard, nil
        }
    }
    
    return nil, errors.New("shard not found")
}

func (ds *DirectorySharding) Write(key, value string) error {
    shard, err := ds.GetShard(key)
    if err != nil {
        return err
    }
    
    return shard.DB.Write(key, value)
}

func (ds *DirectorySharding) Read(key string) (string, error) {
    shard, err := ds.GetShard(key)
    if err != nil {
        return "", err
    }
    
    return shard.DB.Read(key)
}

func (ds *DirectorySharding) AddKeyToShard(key, shardID string) {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    ds.directory[key] = shardID
}
```

### **5. Eventual Consistency**

#### **Vector Clocks Implementation**

```go
type VectorClock struct {
    clocks map[string]int
    mutex  sync.RWMutex
}

func NewVectorClock(nodeID string) *VectorClock {
    return &VectorClock{
        clocks: map[string]int{nodeID: 0},
    }
}

func (vc *VectorClock) Increment(nodeID string) {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    
    vc.clocks[nodeID]++
}

func (vc *VectorClock) Update(other *VectorClock) {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    
    for nodeID, clock := range other.clocks {
        if vc.clocks[nodeID] < clock {
            vc.clocks[nodeID] = clock
        }
    }
}

func (vc *VectorClock) Compare(other *VectorClock) int {
    vc.mutex.RLock()
    other.mutex.RLock()
    defer vc.mutex.RUnlock()
    defer other.mutex.RUnlock()
    
    allNodes := make(map[string]bool)
    for nodeID := range vc.clocks {
        allNodes[nodeID] = true
    }
    for nodeID := range other.clocks {
        allNodes[nodeID] = true
    }
    
    vcGreater := false
    otherGreater := false
    
    for nodeID := range allNodes {
        vcClock := vc.clocks[nodeID]
        otherClock := other.clocks[nodeID]
        
        if vcClock > otherClock {
            vcGreater = true
        } else if otherClock > vcClock {
            otherGreater = true
        }
    }
    
    if vcGreater && !otherGreater {
        return 1 // vc > other
    } else if otherGreater && !vcGreater {
        return -1 // other > vc
    } else if vcGreater && otherGreater {
        return 0 // concurrent
    } else {
        return 0 // equal
    }
}
```

#### **CRDT (Conflict-free Replicated Data Types)**

```go
type GSet struct {
    elements map[string]bool
    mutex    sync.RWMutex
}

func NewGSet() *GSet {
    return &GSet{
        elements: make(map[string]bool),
    }
}

func (gs *GSet) Add(element string) {
    gs.mutex.Lock()
    defer gs.mutex.Unlock()
    
    gs.elements[element] = true
}

func (gs *GSet) Contains(element string) bool {
    gs.mutex.RLock()
    defer gs.mutex.RUnlock()
    
    return gs.elements[element]
}

func (gs *GSet) Merge(other *GSet) {
    gs.mutex.Lock()
    other.mutex.RLock()
    defer gs.mutex.Unlock()
    defer other.mutex.RUnlock()
    
    for element := range other.elements {
        gs.elements[element] = true
    }
}

func (gs *GSet) GetElements() []string {
    gs.mutex.RLock()
    defer gs.mutex.RUnlock()
    
    elements := make([]string, 0, len(gs.elements))
    for element := range gs.elements {
        elements = append(elements, element)
    }
    
    return elements
}
```

### **6. Load Balancing Strategies**

#### **Round Robin Load Balancer**

```go
type RoundRobinLoadBalancer struct {
    servers []Server
    current int
    mutex   sync.Mutex
}

func (rrlb *RoundRobinLoadBalancer) GetServer() Server {
    rrlb.mutex.Lock()
    defer rrlb.mutex.Unlock()
    
    server := rrlb.servers[rrlb.current]
    rrlb.current = (rrlb.current + 1) % len(rrlb.servers)
    
    return server
}

func (rrlb *RoundRobinLoadBalancer) HandleRequest(req *Request) (*Response, error) {
    server := rrlb.GetServer()
    return server.Process(req)
}
```

#### **Least Connections Load Balancer**

```go
type LeastConnectionsLoadBalancer struct {
    servers []Server
    mutex   sync.RWMutex
}

func (lclb *LeastConnectionsLoadBalancer) GetServer() Server {
    lclb.mutex.RLock()
    defer lclb.mutex.RUnlock()
    
    minConnections := lclb.servers[0].GetConnectionCount()
    selectedServer := lclb.servers[0]
    
    for _, server := range lclb.servers[1:] {
        connections := server.GetConnectionCount()
        if connections < minConnections {
            minConnections = connections
            selectedServer = server
        }
    }
    
    return selectedServer
}
```

#### **Weighted Round Robin Load Balancer**

```go
type WeightedServer struct {
    Server   Server
    Weight   int
    Current  int
}

type WeightedRoundRobinLoadBalancer struct {
    servers []WeightedServer
    mutex   sync.Mutex
}

func (wrrlb *WeightedRoundRobinLoadBalancer) GetServer() Server {
    wrrlb.mutex.Lock()
    defer wrrlb.mutex.Unlock()
    
    totalWeight := 0
    for _, server := range wrrlb.servers {
        totalWeight += server.Weight
    }
    
    for i := range wrrlb.servers {
        wrrlb.servers[i].Current += wrrlb.servers[i].Weight
        if wrrlb.servers[i].Current >= totalWeight {
            wrrlb.servers[i].Current -= totalWeight
            return wrrlb.servers[i].Server
        }
    }
    
    // Fallback to first server
    return wrrlb.servers[0].Server
}
```

---

## 🎯 **Key Takeaways**

### **1. CAP Theorem Trade-offs**
- **CP Systems**: Consistency + Partition Tolerance (e.g., MongoDB)
- **AP Systems**: Availability + Partition Tolerance (e.g., Cassandra)
- **CA Systems**: Consistency + Availability (e.g., Single-node databases)

### **2. Consensus Algorithms**
- **Raft**: Easier to understand, leader-based
- **Paxos**: More complex, but more flexible
- **Both ensure**: Safety and liveness properties

### **3. Replication Strategies**
- **Master-Slave**: Read scaling, eventual consistency
- **Master-Master**: Write scaling, conflict resolution needed
- **Choose based on**: Read/write patterns and consistency requirements

### **4. Sharding Approaches**
- **Range-based**: Good for sequential access
- **Hash-based**: Even distribution, no range queries
- **Directory-based**: Flexible, but single point of failure

### **5. Eventual Consistency**
- **Vector clocks**: Track causality
- **CRDTs**: Conflict-free data structures
- **Acceptable for**: Many real-world applications

---

**🎉 This comprehensive distributed systems guide provides deep understanding of distributed system concepts essential for system design interviews! 🚀**
