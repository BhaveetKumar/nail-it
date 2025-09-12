# 🚀 **Asli Engineering Complete System Design Guide**

## 📊 **Based on Arpit Bhyani's Asli Engineering YouTube Channel**

---

## 🎯 **Channel Overview**

### **About Asli Engineering**
- **Creator**: Arpit Bhyani
- **Focus**: System Design, Database Internals, Distributed Systems
- **Approach**: Deep technical understanding with practical implementations
- **Target Audience**: Software engineers preparing for interviews and building scalable systems

### **Key Teaching Philosophy**
- **Deep dive** into fundamental concepts
- **Real-world examples** and production scenarios
- **Implementation-focused** learning approach
- **Interview preparation** with practical coding examples

---

## 🏗️ **1. System Design Fundamentals**

### **Core Principles**

#### **1. Scalability**
**Definition**: The ability of a system to handle increased load by adding resources.

**Types of Scaling**:
- **Vertical Scaling (Scale Up)**: Adding more power to existing machines
- **Horizontal Scaling (Scale Out)**: Adding more machines to the system

```go
// Vertical Scaling Example
type VerticalScaling struct {
    CPU    int
    Memory int64
    Disk   int64
}

func (vs *VerticalScaling) ScaleUp() {
    vs.CPU *= 2
    vs.Memory *= 2
    vs.Disk *= 2
}

// Horizontal Scaling Example
type HorizontalScaling struct {
    servers []Server
}

func (hs *HorizontalScaling) AddServer(server Server) {
    hs.servers = append(hs.servers, server)
}
```

#### **2. Availability**
**Definition**: The percentage of time a system is operational and accessible.

**Availability Levels**:
- **99% (3.65 days downtime/year)**
- **99.9% (8.76 hours downtime/year)**
- **99.99% (52.56 minutes downtime/year)**
- **99.999% (5.26 minutes downtime/year)**

```go
type HighAvailabilitySystem struct {
    primary   *Server
    secondary *Server
    healthCheck *HealthChecker
}

func (has *HighAvailabilitySystem) HandleRequest(req *Request) *Response {
    if has.healthCheck.IsHealthy(has.primary) {
        return has.primary.Process(req)
    }
    
    // Failover to secondary
    return has.secondary.Process(req)
}
```

#### **3. Consistency**
**Definition**: Ensuring all nodes in a distributed system have the same data at the same time.

**Consistency Models**:
- **Strong Consistency**: All nodes see the same data simultaneously
- **Eventual Consistency**: All nodes will eventually have the same data
- **Weak Consistency**: No guarantee about when data will be consistent

```go
type ConsistencyManager struct {
    nodes []Node
    quorum int
}

func (cm *ConsistencyManager) Write(key, value string) error {
    // Write to majority of nodes for strong consistency
    successCount := 0
    for _, node := range cm.nodes {
        if err := node.Write(key, value); err == nil {
            successCount++
        }
    }
    
    if successCount >= cm.quorum {
        return nil
    }
    
    return errors.New("failed to achieve quorum")
}
```

### **CAP Theorem**

#### **The Three Properties**
- **Consistency (C)**: All nodes see the same data at the same time
- **Availability (A)**: System remains operational
- **Partition Tolerance (P)**: System continues to work despite network failures

#### **Trade-offs**
- **CP Systems**: Consistency + Partition Tolerance (e.g., MongoDB)
- **AP Systems**: Availability + Partition Tolerance (e.g., Cassandra)
- **CA Systems**: Consistency + Availability (e.g., Single-node databases)

```go
type CAPSystem struct {
    consistency bool
    availability bool
    partitionTolerance bool
}

func (cs *CAPSystem) ChooseTradeoff() string {
    if cs.partitionTolerance {
        if cs.consistency {
            return "CP System - Choose Consistency over Availability"
        } else {
            return "AP System - Choose Availability over Consistency"
        }
    }
    return "CA System - Not suitable for distributed systems"
}
```

---

## 🗄️ **2. Database Internals Deep Dive**

### **B-Tree Implementation**

#### **Core Concept**
B-trees are self-balancing tree data structures that maintain sorted data and allow efficient searches, insertions, and deletions.

#### **Key Properties**
- All leaves are at the same level
- Minimum degree 't' determines the structure
- Root has at least 2 children (unless it's a leaf)
- Internal nodes have at least 't' children

```go
type BTreeNode struct {
    keys     []int
    children []*BTreeNode
    leaf     bool
    t        int
}

type BTree struct {
    root *BTreeNode
    t    int
}

func NewBTree(t int) *BTree {
    return &BTree{root: nil, t: t}
}

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
```

### **LSM Tree (Log-Structured Merge Tree)**

#### **Core Concept**
LSM trees are optimized for write-heavy workloads by batching writes in memory and periodically flushing to disk.

#### **Key Components**
- **MemTable**: In-memory structure for recent writes
- **SSTables**: Immutable files on disk
- **Compaction**: Process of merging SSTables

```go
type MemTable struct {
    data    map[string]string
    size    int
    maxSize int
}

type SSTable struct {
    filename string
    level    int
    minKey   string
    maxKey   string
}

type LSMTree struct {
    memTable  *MemTable
    sstables  [][]*SSTable
    maxLevels int
    levelSize int
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
```

---

## 🔄 **3. Distributed Systems Concepts**

### **Consistent Hashing**

#### **Problem with Traditional Hashing**
When using `hash(key) % num_servers`, adding or removing servers causes most keys to be remapped.

#### **Solution: Consistent Hashing**
```go
type ConsistentHash struct {
    ring []Node
    hashFunc func(string) uint32
}

type Node struct {
    ID       string
    Position uint32
    Data     map[string]interface{}
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    position := ch.hashFunc(nodeID)
    node := Node{
        ID:       nodeID,
        Position: position,
        Data:     make(map[string]interface{}),
    }
    
    ch.ring = append(ch.ring, node)
    sort.Slice(ch.ring, func(i, j int) bool {
        return ch.ring[i].Position < ch.ring[j].Position
    })
}

func (ch *ConsistentHash) GetNode(key string) *Node {
    keyHash := ch.hashFunc(key)
    
    for _, node := range ch.ring {
        if node.Position >= keyHash {
            return &node
        }
    }
    
    // Wrap around to first node
    return &ch.ring[0]
}
```

### **Virtual Nodes for Load Balancing**

```go
type VirtualNode struct {
    ID           string
    PhysicalNode string
    Hash         uint32
}

type ConsistentHashWithVirtualNodes struct {
    virtualNodes []VirtualNode
    hashFunc     func(string) uint32
    replicas     int
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
    
    sort.Slice(ch.virtualNodes, func(i, j int) bool {
        return ch.virtualNodes[i].Hash < ch.virtualNodes[j].Hash
    })
}
```

---

## 🚀 **4. Real-World System Design Cases**

### **URL Shortener (like bit.ly)**

#### **Requirements**
- Shorten long URLs to short URLs
- Redirect short URLs to original URLs
- Handle 100M URLs per day
- 99.9% availability

#### **High-Level Design**
```go
type URLShortener struct {
    storage    Storage
    cache      Cache
    idGenerator IDGenerator
}

type ShortURL struct {
    ShortCode string
    LongURL   string
    CreatedAt time.Time
    ExpiresAt time.Time
}

func (us *URLShortener) ShortenURL(longURL string) (string, error) {
    // Generate short code
    shortCode := us.idGenerator.Generate()
    
    // Store mapping
    shortURL := &ShortURL{
        ShortCode: shortCode,
        LongURL:   longURL,
        CreatedAt: time.Now(),
        ExpiresAt: time.Now().Add(365 * 24 * time.Hour),
    }
    
    err := us.storage.Store(shortCode, shortURL)
    if err != nil {
        return "", err
    }
    
    // Cache for fast access
    us.cache.Set(shortCode, longURL, 24*time.Hour)
    
    return fmt.Sprintf("https://short.ly/%s", shortCode), nil
}

func (us *URLShortener) Redirect(shortCode string) (string, error) {
    // Check cache first
    if longURL, found := us.cache.Get(shortCode); found {
        return longURL.(string), nil
    }
    
    // Get from storage
    shortURL, err := us.storage.Get(shortCode)
    if err != nil {
        return "", err
    }
    
    // Cache for future requests
    us.cache.Set(shortCode, shortURL.LongURL, 24*time.Hour)
    
    return shortURL.LongURL, nil
}
```

### **Rate Limiter**

#### **Token Bucket Algorithm**
```go
type TokenBucket struct {
    capacity   int
    tokens     int
    refillRate int
    lastRefill time.Time
    mutex      sync.Mutex
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
```

### **Notification System**

#### **Requirements**
- Send notifications via email, SMS, push
- Handle 1M notifications per day
- Support multiple channels
- Retry failed notifications

```go
type NotificationService struct {
    channels map[string]NotificationChannel
    queue    *MessageQueue
    retry    *RetryManager
}

type Notification struct {
    ID       string
    UserID   string
    Type     string
    Message  string
    Channels []string
    Priority int
}

func (ns *NotificationService) SendNotification(notification *Notification) error {
    // Queue notification for processing
    return ns.queue.Publish(notification)
}

func (ns *NotificationService) ProcessNotification(notification *Notification) error {
    for _, channelType := range notification.Channels {
        channel, exists := ns.channels[channelType]
        if !exists {
            continue
        }
        
        err := channel.Send(notification)
        if err != nil {
            // Retry failed notifications
            ns.retry.Schedule(notification, channelType, err)
        }
    }
    
    return nil
}
```

---

## 🎯 **5. Interview Preparation Tips**

### **System Design Interview Structure**

#### **1. Requirements Clarification**
- **Functional Requirements**: What the system should do
- **Non-Functional Requirements**: Performance, scalability, availability
- **Constraints**: Users, requests per second, data size

#### **2. High-Level Design**
- **Draw diagrams** showing major components
- **Identify APIs** and their contracts
- **Consider data flow** between components

#### **3. Detailed Design**
- **Database schema** and indexing strategy
- **Caching strategy** and cache invalidation
- **Load balancing** and scaling approach

#### **4. Scale the Design**
- **Identify bottlenecks** and scaling solutions
- **Consider trade-offs** and alternatives
- **Discuss monitoring** and alerting

### **Common Interview Questions**

#### **1. Design a Chat System**
- **Requirements**: Real-time messaging, group chats, message history
- **Components**: WebSocket connections, message queues, databases
- **Scaling**: Horizontal scaling, message partitioning

#### **2. Design a Social Media Feed**
- **Requirements**: Timeline generation, real-time updates, personalization
- **Components**: Feed service, user service, content service
- **Scaling**: Caching, pre-computation, CDN

#### **3. Design a Search Engine**
- **Requirements**: Full-text search, ranking, autocomplete
- **Components**: Crawler, indexer, query processor
- **Scaling**: Distributed indexing, sharding

### **Key Concepts to Master**

#### **1. Load Balancing**
- **Round Robin**: Distribute requests evenly
- **Least Connections**: Route to server with fewest connections
- **IP Hash**: Route based on client IP
- **Weighted**: Route based on server capacity

#### **2. Caching Strategies**
- **Cache-Aside**: Application manages cache
- **Write-Through**: Write to cache and database
- **Write-Behind**: Write to cache, async to database
- **Refresh-Ahead**: Proactively refresh cache

#### **3. Database Sharding**
- **Horizontal Sharding**: Split data across multiple databases
- **Vertical Sharding**: Split by feature/table
- **Directory-Based**: Use lookup service for shard location
- **Hash-Based**: Use hash function to determine shard

---

## 🚀 **6. Advanced Topics**

### **Microservices Architecture**

#### **Benefits**
- **Independent deployment** and scaling
- **Technology diversity** across services
- **Fault isolation** and resilience

#### **Challenges**
- **Distributed system complexity**
- **Network latency** and reliability
- **Data consistency** across services

```go
type Microservice struct {
    name     string
    port     int
    database *Database
    cache    *Cache
    queue    *MessageQueue
}

func (ms *Microservice) Start() error {
    // Initialize service components
    if err := ms.database.Connect(); err != nil {
        return err
    }
    
    if err := ms.cache.Connect(); err != nil {
        return err
    }
    
    // Start HTTP server
    return ms.startHTTPServer()
}
```

### **Event-Driven Architecture**

#### **Components**
- **Event Producers**: Generate events
- **Event Bus**: Routes events to consumers
- **Event Consumers**: Process events

```go
type Event struct {
    ID        string
    Type      string
    Data      interface{}
    Timestamp time.Time
}

type EventBus struct {
    subscribers map[string][]EventHandler
    mutex       sync.RWMutex
}

type EventHandler interface {
    Handle(event *Event) error
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
}

func (eb *EventBus) Publish(event *Event) error {
    eb.mutex.RLock()
    handlers := eb.subscribers[event.Type]
    eb.mutex.RUnlock()
    
    for _, handler := range handlers {
        go func(h EventHandler) {
            h.Handle(event)
        }(handler)
    }
    
    return nil
}
```

---

## 🎯 **7. Key Takeaways**

### **1. Think in Systems**
- **Understand the big picture** before diving into details
- **Consider all stakeholders** and their requirements
- **Plan for failure** and edge cases

### **2. Master the Fundamentals**
- **Database internals** are crucial for system design
- **Distributed systems** concepts are essential
- **Caching strategies** can make or break performance

### **3. Practice with Real Examples**
- **Study existing systems** like Twitter, Facebook, Netflix
- **Implement small versions** of complex systems
- **Understand trade-offs** in different approaches

### **4. Interview Success Tips**
- **Ask clarifying questions** about requirements
- **Draw diagrams** to explain your thinking
- **Discuss trade-offs** and alternatives
- **Consider scaling** from the beginning

---

**🎉 This comprehensive guide based on Asli Engineering content provides deep insights into system design fundamentals with practical implementations for interview success! 🚀**
