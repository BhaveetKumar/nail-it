---
# Auto-generated front matter
Title: Arpit Bhyani System Design Concepts
LastUpdated: 2025-11-06T20:45:58.639115
Tags: []
Status: draft
---

# 🚀 **Arpit Bhyani System Design Concepts**

## 📊 **Based on Arpit Bhyani's "Asli Engineering" Channel & Courses**

---

## 🎯 **About Arpit Bhyani**

### **Background**

- **Renowned educator** specializing in system design and software architecture
- **YouTube Channel**: "Asli Engineering" - in-depth tutorials on system design
- **Courses**: System Design for Beginners & System Design Masterclass
- **Focus**: Database internals, distributed systems, and scalable architecture

### **Teaching Philosophy**

- **Deep dive** into fundamental concepts
- **Practical implementation** over theoretical knowledge
- **Real-world examples** and production scenarios
- **Step-by-step breakdown** of complex systems

---

## 🏗️ **1. Database Internals Deep Dive**

### **B-Tree Implementation**

#### **Core Concept**

B-trees are self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in O(log n) time.

#### **Key Properties**

- **All leaves** are at the same level
- **Minimum degree** 't' determines the structure
- **Root** has at least 2 children (unless it's a leaf)
- **Internal nodes** have at least 't' children
- **All keys** in a node are sorted

#### **Go Implementation**

```go
type BTreeNode struct {
    keys     []int
    children []*BTreeNode
    leaf     bool
    t        int // minimum degree
}

type BTree struct {
    root *BTreeNode
    t    int
}

func NewBTree(t int) *BTree {
    return &BTree{
        root: nil,
        t:    t,
    }
}

// Search operation
func (bt *BTree) Search(key int) (*BTreeNode, int) {
    if bt.root == nil {
        return nil, -1
    }
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) (*BTreeNode, int) {
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }

    if i < len(node.keys) && key == node.keys[i] {
        return node, i
    }

    if node.leaf {
        return nil, -1
    }

    return bt.searchNode(node.children[i], key)
}

// Insert operation
func (bt *BTree) Insert(key int) {
    if bt.root == nil {
        bt.root = &BTreeNode{
            keys:     []int{key},
            children: nil,
            leaf:     true,
            t:        bt.t,
        }
        return
    }

    if len(bt.root.keys) == 2*bt.t-1 {
        newRoot := &BTreeNode{
            keys:     []int{},
            children: []*BTreeNode{bt.root},
            leaf:     false,
            t:        bt.t,
        }
        bt.splitChild(newRoot, 0)
        bt.root = newRoot
    }

    bt.insertNonFull(bt.root, key)
}

func (bt *BTree) insertNonFull(node *BTreeNode, key int) {
    i := len(node.keys) - 1

    if node.leaf {
        // Insert key in sorted order
        node.keys = append(node.keys, 0)
        for i >= 0 && key < node.keys[i] {
            node.keys[i+1] = node.keys[i]
            i--
        }
        node.keys[i+1] = key
    } else {
        // Find child to insert into
        for i >= 0 && key < node.keys[i] {
            i--
        }
        i++

        if len(node.children[i].keys) == 2*bt.t-1 {
            bt.splitChild(node, i)
            if key > node.keys[i] {
                i++
            }
        }

        bt.insertNonFull(node.children[i], key)
    }
}

func (bt *BTree) splitChild(parent *BTreeNode, index int) {
    t := bt.t
    y := parent.children[index]
    z := &BTreeNode{
        keys:     make([]int, t-1),
        children: make([]*BTreeNode, t),
        leaf:     y.leaf,
        t:        t,
    }

    // Copy last t-1 keys to z
    for j := 0; j < t-1; j++ {
        z.keys[j] = y.keys[j+t]
    }

    // Copy last t children to z
    if !y.leaf {
        for j := 0; j < t; j++ {
            z.children[j] = y.children[j+t]
        }
    }

    // Reduce keys in y
    y.keys = y.keys[:t-1]
    y.children = y.children[:t]

    // Insert z as child of parent
    parent.children = append(parent.children, nil)
    copy(parent.children[index+2:], parent.children[index+1:])
    parent.children[index+1] = z

    // Move median key to parent
    parent.keys = append(parent.keys, 0)
    copy(parent.keys[index+1:], parent.keys[index:])
    parent.keys[index] = y.keys[t-1]
}
```

### **LSM Tree (Log-Structured Merge Tree)**

#### **Core Concept**

LSM trees are optimized for write-heavy workloads by batching writes in memory and periodically flushing to disk.

#### **Key Components**

- **MemTable**: In-memory structure for recent writes
- **SSTables**: Immutable files on disk
- **Compaction**: Process of merging SSTables

#### **Go Implementation**

```go
type MemTable struct {
    data map[string]string
    size int
    maxSize int
}

type SSTable struct {
    filename string
    level    int
    minKey   string
    maxKey   string
}

type LSMTree struct {
    memTable    *MemTable
    sstables    [][]*SSTable
    maxLevels   int
    levelSize   int
}

func NewLSMTree(maxMemSize, maxLevels, levelSize int) *LSMTree {
    return &LSMTree{
        memTable: &MemTable{
            data:    make(map[string]string),
            size:    0,
            maxSize: maxMemSize,
        },
        sstables:  make([][]*SSTable, maxLevels),
        maxLevels: maxLevels,
        levelSize: levelSize,
    }
}

func (lsm *LSMTree) Put(key, value string) {
    lsm.memTable.data[key] = value
    lsm.memTable.size += len(key) + len(value)

    if lsm.memTable.size >= lsm.memTable.maxSize {
        lsm.flushMemTable()
    }
}

func (lsm *LSMTree) Get(key string) (string, bool) {
    // Check memtable first
    if value, exists := lsm.memTable.data[key]; exists {
        return value, true
    }

    // Check SSTables from newest to oldest
    for level := 0; level < lsm.maxLevels; level++ {
        for i := len(lsm.sstables[level]) - 1; i >= 0; i-- {
            if value, exists := lsm.searchSSTable(lsm.sstables[level][i], key); exists {
                return value, true
            }
        }
    }

    return "", false
}

func (lsm *LSMTree) flushMemTable() {
    // Create SSTable from memtable
    sstable := lsm.createSSTable(lsm.memTable.data)

    // Add to level 0
    lsm.sstables[0] = append(lsm.sstables[0], sstable)

    // Trigger compaction if needed
    lsm.compactLevel(0)

    // Clear memtable
    lsm.memTable.data = make(map[string]string)
    lsm.memTable.size = 0
}

func (lsm *LSMTree) compactLevel(level int) {
    if len(lsm.sstables[level]) <= lsm.levelSize {
        return
    }

    // Merge SSTables at this level
    merged := lsm.mergeSSTables(lsm.sstables[level])

    // Clear current level
    lsm.sstables[level] = nil

    // Add merged SSTable to next level
    if level+1 < lsm.maxLevels {
        lsm.sstables[level+1] = append(lsm.sstables[level+1], merged...)
        lsm.compactLevel(level + 1)
    }
}
```

---

## 🔄 **2. Consistent Hashing Advanced**

### **Virtual Nodes Concept**

#### **Problem with Basic Consistent Hashing**

- **Uneven distribution** when nodes have different capacities
- **Hot spots** when data is not evenly distributed
- **Difficult to handle** node failures gracefully

#### **Solution: Virtual Nodes**

```go
type VirtualNode struct {
    ID       string
    PhysicalNode string
    Hash     uint32
}

type ConsistentHashWithVirtualNodes struct {
    virtualNodes []VirtualNode
    hashFunc     func(string) uint32
    replicas     int // number of virtual nodes per physical node
}

func NewConsistentHashWithVirtualNodes(replicas int) *ConsistentHashWithVirtualNodes {
    return &ConsistentHashWithVirtualNodes{
        virtualNodes: make([]VirtualNode, 0),
        hashFunc:     crc32.ChecksumIEEE,
        replicas:     replicas,
    }
}

func (ch *ConsistentHashWithVirtualNodes) AddNode(nodeID string) {
    for i := 0; i < ch.replicas; i++ {
        virtualNodeID := fmt.Sprintf("%s#%d", nodeID, i)
        hash := ch.hashFunc(virtualNodeID)

        virtualNode := VirtualNode{
            ID:           virtualNodeID,
            PhysicalNode: nodeID,
            Hash:         hash,
        }

        ch.virtualNodes = append(ch.virtualNodes, virtualNode)
    }

    // Sort by hash
    sort.Slice(ch.virtualNodes, func(i, j int) bool {
        return ch.virtualNodes[i].Hash < ch.virtualNodes[j].Hash
    })
}

func (ch *ConsistentHashWithVirtualNodes) GetNode(key string) string {
    if len(ch.virtualNodes) == 0 {
        return ""
    }

    keyHash := ch.hashFunc(key)

    // Binary search for first virtual node with hash >= keyHash
    idx := sort.Search(len(ch.virtualNodes), func(i int) bool {
        return ch.virtualNodes[i].Hash >= keyHash
    })

    // Wrap around if necessary
    if idx == len(ch.virtualNodes) {
        idx = 0
    }

    return ch.virtualNodes[idx].PhysicalNode
}

func (ch *ConsistentHashWithVirtualNodes) RemoveNode(nodeID string) {
    newVirtualNodes := make([]VirtualNode, 0)

    for _, vn := range ch.virtualNodes {
        if vn.PhysicalNode != nodeID {
            newVirtualNodes = append(newVirtualNodes, vn)
        }
    }

    ch.virtualNodes = newVirtualNodes
}
```

### **Weighted Consistent Hashing**

#### **Concept**: Nodes with different capacities get different numbers of virtual nodes

```go
type WeightedConsistentHash struct {
    virtualNodes []VirtualNode
    hashFunc     func(string) uint32
    nodeWeights  map[string]int
}

func NewWeightedConsistentHash() *WeightedConsistentHash {
    return &WeightedConsistentHash{
        virtualNodes: make([]VirtualNode, 0),
        hashFunc:     crc32.ChecksumIEEE,
        nodeWeights:  make(map[string]int),
    }
}

func (wch *WeightedConsistentHash) AddNode(nodeID string, weight int) {
    wch.nodeWeights[nodeID] = weight

    for i := 0; i < weight; i++ {
        virtualNodeID := fmt.Sprintf("%s#%d", nodeID, i)
        hash := wch.hashFunc(virtualNodeID)

        virtualNode := VirtualNode{
            ID:           virtualNodeID,
            PhysicalNode: nodeID,
            Hash:         hash,
        }

        wch.virtualNodes = append(wch.virtualNodes, virtualNode)
    }

    // Sort by hash
    sort.Slice(wch.virtualNodes, func(i, j int) bool {
        return wch.virtualNodes[i].Hash < wch.virtualNodes[j].Hash
    })
}
```

---

## ⚡ **3. Thundering Herd Problem**

### **Problem Definition**

When a cache expires, multiple clients simultaneously try to regenerate the same data, causing a sudden spike in load on the backend system.

### **Real-World Example**

```go
type CacheService struct {
    cache    map[string]*CacheEntry
    mutex    sync.RWMutex
    db       *Database
    lockMap  map[string]*sync.Mutex
    lockMutex sync.Mutex
}

type CacheEntry struct {
    Value     interface{}
    ExpiresAt time.Time
    Loading   bool
}

func (cs *CacheService) Get(key string) (interface{}, error) {
    cs.mutex.RLock()
    entry, exists := cs.cache[key]
    cs.mutex.RUnlock()

    if exists && time.Now().Before(entry.ExpiresAt) {
        return entry.Value, nil
    }

    // Cache miss or expired
    return cs.loadData(key)
}

func (cs *CacheService) loadData(key string) (interface{}, error) {
    // Get lock for this specific key
    lock := cs.getLock(key)
    lock.Lock()
    defer lock.Unlock()

    // Double-check after acquiring lock
    cs.mutex.RLock()
    entry, exists := cs.cache[key]
    cs.mutex.RUnlock()

    if exists && time.Now().Before(entry.ExpiresAt) {
        return entry.Value, nil
    }

    // Load from database
    data, err := cs.db.Get(key)
    if err != nil {
        return nil, err
    }

    // Update cache
    cs.mutex.Lock()
    cs.cache[key] = &CacheEntry{
        Value:     data,
        ExpiresAt: time.Now().Add(5 * time.Minute),
        Loading:   false,
    }
    cs.mutex.Unlock()

    return data, nil
}

func (cs *CacheService) getLock(key string) *sync.Mutex {
    cs.lockMutex.Lock()
    defer cs.lockMutex.Unlock()

    if lock, exists := cs.lockMap[key]; exists {
        return lock
    }

    lock := &sync.Mutex{}
    cs.lockMap[key] = lock
    return lock
}
```

### **Alternative Solutions**

#### **1. Cache-Aside with Lock**

```go
type CacheAsideService struct {
    cache     map[string]*CacheEntry
    mutex     sync.RWMutex
    db        *Database
    keyLocks  sync.Map // map[string]*sync.Mutex
}

func (cas *CacheAsideService) Get(key string) (interface{}, error) {
    // Check cache first
    cas.mutex.RLock()
    entry, exists := cas.cache[key]
    cas.mutex.RUnlock()

    if exists && time.Now().Before(entry.ExpiresAt) {
        return entry.Value, nil
    }

    // Get or create lock for this key
    lockInterface, _ := cas.keyLocks.LoadOrStore(key, &sync.Mutex{})
    lock := lockInterface.(*sync.Mutex)

    lock.Lock()
    defer lock.Unlock()

    // Double-check after acquiring lock
    cas.mutex.RLock()
    entry, exists = cas.cache[key]
    cas.mutex.RUnlock()

    if exists && time.Now().Before(entry.ExpiresAt) {
        return entry.Value, nil
    }

    // Load from database
    data, err := cas.db.Get(key)
    if err != nil {
        return nil, err
    }

    // Update cache
    cas.mutex.Lock()
    cas.cache[key] = &CacheEntry{
        Value:     data,
        ExpiresAt: time.Now().Add(5 * time.Minute),
    }
    cas.mutex.Unlock()

    return data, nil
}
```

#### **2. Write-Through Cache**

```go
type WriteThroughCache struct {
    cache map[string]*CacheEntry
    mutex sync.RWMutex
    db    *Database
}

func (wtc *WriteThroughCache) Get(key string) (interface{}, error) {
    wtc.mutex.RLock()
    entry, exists := wtc.cache[key]
    wtc.mutex.RUnlock()

    if exists && time.Now().Before(entry.ExpiresAt) {
        return entry.Value, nil
    }

    // Load from database
    data, err := wtc.db.Get(key)
    if err != nil {
        return nil, err
    }

    // Update cache
    wtc.mutex.Lock()
    wtc.cache[key] = &CacheEntry{
        Value:     data,
        ExpiresAt: time.Now().Add(5 * time.Minute),
    }
    wtc.mutex.Unlock()

    return data, nil
}

func (wtc *WriteThroughCache) Set(key string, value interface{}) error {
    // Write to database first
    err := wtc.db.Set(key, value)
    if err != nil {
        return err
    }

    // Then update cache
    wtc.mutex.Lock()
    wtc.cache[key] = &CacheEntry{
        Value:     value,
        ExpiresAt: time.Now().Add(5 * time.Minute),
    }
    wtc.mutex.Unlock()

    return nil
}
```

---

## 🔐 **4. Idempotent API Design**

### **Problem**: Ensuring API calls can be safely retried

### **Solution Patterns**

#### **1. Idempotency Keys**

```go
type IdempotencyService struct {
    cache map[string]*IdempotencyEntry
    mutex sync.RWMutex
    db    *Database
}

type IdempotencyEntry struct {
    RequestID string
    Response  interface{}
    Status    string
    CreatedAt time.Time
}

func (is *IdempotencyService) ProcessRequest(idempotencyKey string, request *Request) (*Response, error) {
    // Check if request already processed
    is.mutex.RLock()
    entry, exists := is.cache[idempotencyKey]
    is.mutex.RUnlock()

    if exists {
        if entry.Status == "completed" {
            return entry.Response.(*Response), nil
        }
        if entry.Status == "processing" {
            return nil, errors.New("request already being processed")
        }
    }

    // Create new entry
    is.mutex.Lock()
    is.cache[idempotencyKey] = &IdempotencyEntry{
        RequestID: idempotencyKey,
        Status:    "processing",
        CreatedAt: time.Now(),
    }
    is.mutex.Unlock()

    // Process request
    response, err := is.processPayment(request)

    // Update entry
    is.mutex.Lock()
    if err != nil {
        is.cache[idempotencyKey].Status = "failed"
    } else {
        is.cache[idempotencyKey].Status = "completed"
        is.cache[idempotencyKey].Response = response
    }
    is.mutex.Unlock()

    return response, err
}

func (is *IdempotencyService) processPayment(request *Request) (*Response, error) {
    // Simulate payment processing
    time.Sleep(100 * time.Millisecond)

    return &Response{
        TransactionID: generateTransactionID(),
        Status:        "success",
        Amount:        request.Amount,
    }, nil
}
```

#### **2. Database-Level Idempotency**

```go
type PaymentService struct {
    db *sql.DB
}

func (ps *PaymentService) ProcessPayment(request *PaymentRequest) (*PaymentResponse, error) {
    tx, err := ps.db.Begin()
    if err != nil {
        return nil, err
    }
    defer tx.Rollback()

    // Check if transaction already exists
    var existingID string
    err = tx.QueryRow(`
        SELECT transaction_id
        FROM payments
        WHERE idempotency_key = $1
    `, request.IdempotencyKey).Scan(&existingID)

    if err == nil {
        // Transaction already exists, return existing response
        return ps.getPaymentResponse(tx, existingID)
    }

    if err != sql.ErrNoRows {
        return nil, err
    }

    // Create new transaction
    transactionID := generateTransactionID()
    _, err = tx.Exec(`
        INSERT INTO payments (transaction_id, idempotency_key, amount, status, created_at)
        VALUES ($1, $2, $3, $4, $5)
    `, transactionID, request.IdempotencyKey, request.Amount, "processing", time.Now())

    if err != nil {
        return nil, err
    }

    // Process payment
    status := "success"
    if request.Amount > 1000 {
        status = "failed"
    }

    // Update status
    _, err = tx.Exec(`
        UPDATE payments
        SET status = $1, updated_at = $2
        WHERE transaction_id = $3
    `, status, time.Now(), transactionID)

    if err != nil {
        return nil, err
    }

    err = tx.Commit()
    if err != nil {
        return nil, err
    }

    return &PaymentResponse{
        TransactionID: transactionID,
        Status:        status,
        Amount:        request.Amount,
    }, nil
}
```

---

## 🚀 **5. High-Throughput System Design**

### **Message Queue Implementation**

#### **Basic Queue**

```go
type MessageQueue struct {
    messages chan Message
    capacity int
}

type Message struct {
    ID      string
    Payload interface{}
    Topic   string
}

func NewMessageQueue(capacity int) *MessageQueue {
    return &MessageQueue{
        messages: make(chan Message, capacity),
        capacity: capacity,
    }
}

func (mq *MessageQueue) Publish(message Message) error {
    select {
    case mq.messages <- message:
        return nil
    default:
        return errors.New("queue is full")
    }
}

func (mq *MessageQueue) Consume() <-chan Message {
    return mq.messages
}
```

#### **Partitioned Queue**

```go
type PartitionedQueue struct {
    partitions []chan Message
    partitionCount int
    hashFunc   func(string) int
}

func NewPartitionedQueue(partitionCount, capacity int) *PartitionedQueue {
    partitions := make([]chan Message, partitionCount)
    for i := 0; i < partitionCount; i++ {
        partitions[i] = make(chan Message, capacity)
    }

    return &PartitionedQueue{
        partitions:    partitions,
        partitionCount: partitionCount,
        hashFunc:     func(key string) int {
            hash := 0
            for _, c := range key {
                hash += int(c)
            }
            return hash % partitionCount
        },
    }
}

func (pq *PartitionedQueue) Publish(message Message, key string) error {
    partition := pq.hashFunc(key)

    select {
    case pq.partitions[partition] <- message:
        return nil
    default:
        return errors.New("partition is full")
    }
}

func (pq *PartitionedQueue) Consume(partition int) <-chan Message {
    return pq.partitions[partition]
}
```

### **Rate Limiting**

#### **Token Bucket Algorithm**

```go
type TokenBucket struct {
    capacity     int
    tokens       int
    refillRate   int
    lastRefill   time.Time
    mutex        sync.Mutex
}

func NewTokenBucket(capacity, refillRate int) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        tokens:     capacity,
        refillRate: refillRate,
        lastRefill: time.Now(),
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()

    // Refill tokens
    now := time.Now()
    tokensToAdd := int(now.Sub(tb.lastRefill).Seconds()) * tb.refillRate
    tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
    tb.lastRefill = now

    if tb.tokens > 0 {
        tb.tokens--
        return true
    }

    return false
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

#### **Sliding Window Rate Limiter**

```go
type SlidingWindowRateLimiter struct {
    requests    []time.Time
    windowSize  time.Duration
    maxRequests int
    mutex       sync.Mutex
}

func NewSlidingWindowRateLimiter(windowSize time.Duration, maxRequests int) *SlidingWindowRateLimiter {
    return &SlidingWindowRateLimiter{
        requests:    make([]time.Time, 0),
        windowSize:  windowSize,
        maxRequests: maxRequests,
    }
}

func (swrl *SlidingWindowRateLimiter) Allow() bool {
    swrl.mutex.Lock()
    defer swrl.mutex.Unlock()

    now := time.Now()

    // Remove old requests outside the window
    cutoff := now.Add(-swrl.windowSize)
    for len(swrl.requests) > 0 && swrl.requests[0].Before(cutoff) {
        swrl.requests = swrl.requests[1:]
    }

    // Check if we can add a new request
    if len(swrl.requests) < swrl.maxRequests {
        swrl.requests = append(swrl.requests, now)
        return true
    }

    return false
}
```

---

## 🎯 **6. Key Insights from Arpit Bhyani**

### **1. Deep Understanding Over Surface Knowledge**

- **Don't just memorize** - understand the underlying principles
- **Implement everything** - coding helps internalize concepts
- **Question assumptions** - why does this work this way?

### **2. Database Internals Matter**

- **B-trees** are fundamental to understanding databases
- **LSM trees** are crucial for write-heavy workloads
- **Indexing strategies** affect query performance significantly

### **3. Distributed Systems Patterns**

- **Consistent hashing** with virtual nodes for load balancing
- **Idempotency** is essential for reliable systems
- **Rate limiting** prevents system overload

### **4. Performance Optimization**

- **Cache strategies** can make or break system performance
- **Thundering herd** problems are common and solvable
- **Message queues** enable scalable architectures

### **5. Real-World Applications**

- **Payment systems** require idempotency and consistency
- **Caching layers** need careful design to avoid problems
- **Rate limiting** is essential for API protection

---

## 🚀 **7. Practical Implementation Tips**

### **1. Start Simple, Optimize Later**

```go
// Start with simple implementation
type SimpleCache struct {
    data map[string]interface{}
    mutex sync.RWMutex
}

// Add complexity only when needed
type AdvancedCache struct {
    data        map[string]*CacheEntry
    mutex       sync.RWMutex
    ttl         time.Duration
    maxSize     int
    evictionPolicy string
}
```

### **2. Measure Before Optimizing**

```go
func benchmarkCacheOperations() {
    cache := NewSimpleCache()

    start := time.Now()
    for i := 0; i < 1000000; i++ {
        cache.Set(fmt.Sprintf("key%d", i), fmt.Sprintf("value%d", i))
    }
    duration := time.Since(start)

    fmt.Printf("Set operations took: %v\n", duration)
}
```

### **3. Handle Edge Cases**

```go
func (cache *Cache) Get(key string) (interface{}, error) {
    if key == "" {
        return nil, errors.New("key cannot be empty")
    }

    cache.mutex.RLock()
    defer cache.mutex.RUnlock()

    value, exists := cache.data[key]
    if !exists {
        return nil, errors.New("key not found")
    }

    return value, nil
}
```

---

**🎉 This comprehensive guide based on Arpit Bhyani's concepts provides deep insights into system design fundamentals with practical Go implementations! 🚀**
