# 📚 Designing Data-Intensive Applications - Comprehensive Summary

> **Complete guide to Martin Kleppmann's seminal work on building reliable, scalable, and maintainable data systems**

## 📋 Table of Contents

1. [Book Overview](#book-overview/)
2. [Part I: Foundations of Data Systems](#part-i-foundations-of-data-systems/)
3. [Part II: Distributed Data](#part-ii-distributed-data/)
4. [Part III: Derived Data](#part-iii-derived-data/)
5. [Database Types & Architectures](#database-types--architectures/)
6. [Error Handling & Scaling](#error-handling--scaling/)
7. [FAANG Interview Questions](#faang-interview-questions/)

---

## 📖 Book Overview

**"Designing Data-Intensive Applications"** by Martin Kleppmann is the definitive guide to building modern data systems. The book covers everything from basic data models to complex distributed systems, making it essential for backend engineers.

### **Key Themes:**

- **Reliability**: Systems continue working correctly even when things go wrong
- **Scalability**: Systems can handle increased load gracefully
- **Maintainability**: Systems are easy to understand, modify, and extend

---

## 🏗️ Part I: Foundations of Data Systems

### **1. Reliable, Scalable, and Maintainable Applications**

#### **Reliability**

```go
// Circuit Breaker Pattern - Prevents cascading failures
// Automatically stops calling failing services to allow recovery
type CircuitBreaker struct {
    maxFailures int           // Maximum failures before opening circuit
    timeout     time.Duration // Time to wait before trying again
    failures    int           // Current failure count
    lastFailure time.Time     // Timestamp of last failure
    state       State         // Current circuit state
}

// Circuit states represent different operational modes
type State int

const (
    Closed State = iota  // Normal operation, calls pass through
    Open                 // Circuit is open, calls are blocked
    HalfOpen             // Testing if service has recovered
)

// Call executes a function with circuit breaker protection
func (cb *CircuitBreaker) Call(fn func() error) error {
    // Check if circuit is open
    if cb.state == Open {
        // If timeout has passed, try half-open state
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = HalfOpen
        } else {
            // Circuit is still open, reject call immediately
            return ErrCircuitOpen
        }
    }

    // Execute the function
    err := fn()
    if err != nil {
        // Function failed, increment failure count
        cb.failures++
        cb.lastFailure = time.Now()
        
        // If we've hit the failure threshold, open the circuit
        if cb.failures >= cb.maxFailures {
            cb.state = Open
        }
        return err
    }

    // Function succeeded, reset circuit to closed state
    cb.failures = 0
    cb.state = Closed
    return nil
}
```

**Key Concepts Explained:**
- **Circuit Breaker**: Prevents cascading failures by blocking calls to failing services
- **Three States**: Closed (normal), Open (blocking), HalfOpen (testing)
- **Failure Threshold**: Opens circuit after specified number of failures
- **Recovery**: Automatically tries to recover after timeout period
- **Fast Failure**: Rejects calls immediately when circuit is open

#### **Scalability**

- **Load Parameters**: Throughput, response time, resource utilization
- **Scaling Approaches**: Vertical (scale up) vs Horizontal (scale out)
- **Load Balancing**: Distributing load across multiple servers

#### **Maintainability**

- **Operability**: Easy to monitor, debug, and operate
- **Simplicity**: Reducing complexity through good abstractions
- **Evolvability**: Easy to make changes and add features

### **2. Data Models and Query Languages**

#### **Relational Model**

```sql
-- Example: Relational data model
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
);
```

#### **Document Model**

```json
// Example: Document data model (MongoDB)
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "John Doe",
  "email": "john@example.com",
  "orders": [
    {
      "id": 1,
      "amount": 99.99,
      "status": "completed",
      "items": [
        {"product": "Laptop", "quantity": 1, "price": 99.99}
      ]
    }
  ],
  "created_at": ISODate("2023-01-01T00:00:00Z")
}
```

#### **Graph Model**

```cypher
// Example: Graph data model (Neo4j)
CREATE (u:User {name: "John Doe", email: "john@example.com"})
CREATE (p:Product {name: "Laptop", price: 999.99})
CREATE (o:Order {amount: 999.99, status: "completed"})
CREATE (u)-[:PURCHASED]->(o)
CREATE (o)-[:CONTAINS]->(p)
```

### **3. Storage and Retrieval**

#### **B-Trees**

```go
// Simplified B-Tree implementation
type BTreeNode struct {
    keys     []int
    values   []interface{}
    children []*BTreeNode
    isLeaf   bool
}

type BTree struct {
    root *BTreeNode
    t    int // minimum degree
}

func (bt *BTree) Search(key int) interface{} {
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) interface{} {
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }

    if i < len(node.keys) && key == node.keys[i] {
        return node.values[i]
    }

    if node.isLeaf {
        return nil
    }

    return bt.searchNode(node.children[i], key)
}
```

#### **LSM-Trees (Log-Structured Merge-Trees)**

```go
// Simplified LSM-Tree implementation
type LSMTree struct {
    memtable  map[string]string
    sstables  []*SSTable
    threshold int
}

type SSTable struct {
    data map[string]string
    size int
}

func (lsm *LSMTree) Put(key, value string) {
    lsm.memtable[key] = value

    if len(lsm.memtable) >= lsm.threshold {
        lsm.flushMemtable()
    }
}

func (lsm *LSMTree) Get(key string) (string, bool) {
    // Check memtable first
    if value, exists := lsm.memtable[key]; exists {
        return value, true
    }

    // Check SSTables (newest first)
    for i := len(lsm.sstables) - 1; i >= 0; i-- {
        if value, exists := lsm.sstables[i].data[key]; exists {
            return value, true
        }
    }

    return "", false
}
```

### **4. Encoding and Evolution**

#### **JSON Encoding**

```go
type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func encodeJSON(user User) ([]byte, error) {
    return json.Marshal(user)
}

func decodeJSON(data []byte) (User, error) {
    var user User
    err := json.Unmarshal(data, &user)
    return user, err
}
```

#### **Protocol Buffers**

```protobuf
// user.proto
syntax = "proto3";

message User {
    int32 id = 1;
    string name = 2;
    string email = 3;
    repeated Order orders = 4;
}

message Order {
    int32 id = 1;
    double amount = 2;
    string status = 3;
}
```

---

## 🌐 Part II: Distributed Data

### **5. Replication**

#### **Leader-Follower Replication**

```go
type ReplicationManager struct {
    leader   *Node
    followers []*Node
}

func (rm *ReplicationManager) Write(data []byte) error {
    // Write to leader
    if err := rm.leader.Write(data); err != nil {
        return err
    }

    // Replicate to followers
    for _, follower := range rm.followers {
        go func(f *Node) {
            f.Write(data)
        }(follower)
    }

    return nil
}

func (rm *ReplicationManager) Read() ([]byte, error) {
    // Can read from leader or followers
    return rm.leader.Read()
}
```

#### **Multi-Leader Replication**

```go
type MultiLeaderReplication struct {
    leaders []*Node
    conflictResolver ConflictResolver
}

func (mlr *MultiLeaderReplication) Write(nodeID int, data []byte) error {
    // Write to specific leader
    if err := mlr.leaders[nodeID].Write(data); err != nil {
        return err
    }

    // Replicate to other leaders
    for i, leader := range mlr.leaders {
        if i != nodeID {
            go func(l *Node) {
                l.Write(data)
            }(leader)
        }
    }

    return nil
}
```

### **6. Partitioning (Sharding)**

#### **Range-Based Partitioning**

```go
type RangePartitioner struct {
    ranges []Range
    nodes  []*Node
}

type Range struct {
    start int
    end   int
}

func (rp *RangePartitioner) GetNode(key int) *Node {
    for i, r := range rp.ranges {
        if key >= r.start && key < r.end {
            return rp.nodes[i]
        }
    }
    return rp.nodes[len(rp.nodes)-1] // Default to last node
}
```

#### **Hash-Based Partitioning**

```go
type HashPartitioner struct {
    nodes []*Node
}

func (hp *HashPartitioner) GetNode(key string) *Node {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    index := hash.Sum32() % uint32(len(hp.nodes))
    return hp.nodes[index]
}
```

### **7. Transactions**

#### **ACID Properties Implementation**

```go
type Transaction struct {
    id        string
    operations []Operation
    state     TransactionState
}

type TransactionState int

const (
    Active TransactionState = iota
    Committed
    Aborted
)

type Operation struct {
    type_ string // "read" or "write"
    key   string
    value interface{}
}

func (t *Transaction) Begin() {
    t.state = Active
    t.operations = make([]Operation, 0)
}

func (t *Transaction) Read(key string) (interface{}, error) {
    if t.state != Active {
        return nil, ErrTransactionNotActive
    }

    op := Operation{type_: "read", key: key}
    t.operations = append(t.operations, op)

    // Perform actual read
    return t.executeRead(key)
}

func (t *Transaction) Write(key string, value interface{}) error {
    if t.state != Active {
        return ErrTransactionNotActive
    }

    op := Operation{type_: "write", key: key, value: value}
    t.operations = append(t.operations, op)

    return nil
}

func (t *Transaction) Commit() error {
    if t.state != Active {
        return ErrTransactionNotActive
    }

    // Execute all write operations
    for _, op := range t.operations {
        if op.type_ == "write" {
            if err := t.executeWrite(op.key, op.value); err != nil {
                t.Abort()
                return err
            }
        }
    }

    t.state = Committed
    return nil
}

func (t *Transaction) Abort() {
    t.state = Aborted
    // Rollback any changes
}
```

### **8. The Trouble with Distributed Systems**

#### **Network Partitions**

```go
type NetworkPartition struct {
    nodes []*Node
    partition1 []*Node
    partition2 []*Node
}

func (np *NetworkPartition) HandlePartition() {
    // Detect partition
    if np.isPartitioned() {
        // Choose partition to continue serving
        if len(np.partition1) > len(np.partition2) {
            np.partition1[0].BecomeLeader()
        } else {
            np.partition2[0].BecomeLeader()
        }
    }
}
```

#### **Clock Synchronization**

```go
type ClockSync struct {
    localTime  time.Time
    serverTime time.Time
    offset     time.Duration
}

func (cs *ClockSync) SyncWithServer() error {
    // Send request with local time
    requestTime := time.Now()

    // Receive response with server time
    response := cs.getServerTime()
    responseTime := time.Now()

    // Calculate offset
    cs.offset = response.ServerTime.Sub(requestTime.Add(responseTime.Sub(requestTime) / 2))

    return nil
}
```

### **9. Consistency and Consensus**

#### **Raft Consensus Algorithm**

```go
type RaftNode struct {
    id          int
    state       NodeState
    currentTerm int
    votedFor    int
    log         []LogEntry
    commitIndex int
    lastApplied int
}

type NodeState int

const (
    Follower NodeState = iota
    Candidate
    Leader
)

type LogEntry struct {
    term    int
    command interface{}
}

func (rn *RaftNode) StartElection() {
    rn.state = Candidate
    rn.currentTerm++
    rn.votedFor = rn.id

    votes := 1 // Vote for self

    // Request votes from other nodes
    for _, node := range rn.getOtherNodes() {
        go func(n *RaftNode) {
            if n.RequestVote(rn.currentTerm, rn.id, rn.log[len(rn.log)-1]) {
                votes++
                if votes > len(rn.getOtherNodes())/2 {
                    rn.BecomeLeader()
                }
            }
        }(node)
    }
}
```

---

## 📊 Part III: Derived Data

### **10. Batch Processing**

#### **MapReduce Implementation**

```go
type MapReduce struct {
    mapper  func(string) []KeyValue
    reducer func(string, []string) string
}

type KeyValue struct {
    Key   string
    Value string
}

func (mr *MapReduce) Process(input []string) map[string]string {
    // Map phase
    intermediate := make(map[string][]string)
    for _, line := range input {
        kvs := mr.mapper(line)
        for _, kv := range kvs {
            intermediate[kv.Key] = append(intermediate[kv.Key], kv.Value)
        }
    }

    // Reduce phase
    result := make(map[string]string)
    for key, values := range intermediate {
        result[key] = mr.reducer(key, values)
    }

    return result
}

// Example: Word count
func wordCountMapper(line string) []KeyValue {
    words := strings.Fields(line)
    kvs := make([]KeyValue, len(words))
    for i, word := range words {
        kvs[i] = KeyValue{Key: word, Value: "1"}
    }
    return kvs
}

func wordCountReducer(key string, values []string) string {
    return strconv.Itoa(len(values))
}
```

### **11. Stream Processing**

#### **Event Stream Processing**

```go
type StreamProcessor struct {
    input  chan Event
    output chan ProcessedEvent
}

type Event struct {
    ID        string
    Timestamp time.Time
    Data      map[string]interface{}
}

type ProcessedEvent struct {
    Event
    ProcessedAt time.Time
    Result      interface{}
}

func (sp *StreamProcessor) Process() {
    for event := range sp.input {
        // Process event
        result := sp.processEvent(event)

        processed := ProcessedEvent{
            Event:       event,
            ProcessedAt: time.Now(),
            Result:      result,
        }

        sp.output <- processed
    }
}

func (sp *StreamProcessor) processEvent(event Event) interface{} {
    // Implement event processing logic
    return fmt.Sprintf("Processed: %s", event.ID)
}
```

---

## 🗄️ Database Types & Architectures

### **Relational Databases (SQL)**

#### **PostgreSQL Architecture**

```go
// Connection pooling example
type PostgresPool struct {
    pool *sql.DB
}

func NewPostgresPool(dsn string, maxConnections int) (*PostgresPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }

    db.SetMaxOpenConns(maxConnections)
    db.SetMaxIdleConns(maxConnections / 2)
    db.SetConnMaxLifetime(time.Hour)

    return &PostgresPool{pool: db}, nil
}

func (pp *PostgresPool) Query(query string, args ...interface{}) (*sql.Rows, error) {
    return pp.pool.Query(query, args...)
}
```

#### **MySQL Architecture**

```go
// MySQL replication setup
type MySQLReplication struct {
    master *sql.DB
    slaves []*sql.DB
}

func (mr *MySQLReplication) Write(query string, args ...interface{}) error {
    // Write to master
    _, err := mr.master.Exec(query, args...)
    return err
}

func (mr *MySQLReplication) Read(query string, args ...interface{}) (*sql.Rows, error) {
    // Read from slave (round-robin)
    slave := mr.slaves[rand.Intn(len(mr.slaves))]
    return slave.Query(query, args...)
}
```

### **NoSQL Databases**

#### **MongoDB (Document Database)**

```go
type MongoDBClient struct {
    client *mongo.Client
    db     *mongo.Database
}

func (mc *MongoDBClient) Insert(collection string, document interface{}) error {
    coll := mc.db.Collection(collection)
    _, err := coll.InsertOne(context.Background(), document)
    return err
}

func (mc *MongoDBClient) Find(collection string, filter bson.M) (*mongo.Cursor, error) {
    coll := mc.db.Collection(collection)
    return coll.Find(context.Background(), filter)
}
```

#### **Redis (Key-Value Store)**

```go
type RedisClient struct {
    client *redis.Client
}

func (rc *RedisClient) Set(key string, value interface{}, expiration time.Duration) error {
    return rc.client.Set(context.Background(), key, value, expiration).Err()
}

func (rc *RedisClient) Get(key string) (string, error) {
    return rc.client.Get(context.Background(), key).Result()
}

func (rc *RedisClient) HSet(key, field string, value interface{}) error {
    return rc.client.HSet(context.Background(), key, field, value).Err()
}
```

#### **Cassandra (Wide-Column Store)**

```go
type CassandraClient struct {
    session *gocql.Session
}

func (cc *CassandraClient) Insert(table string, columns []string, values []interface{}) error {
    query := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
        table,
        strings.Join(columns, ","),
        strings.Repeat("?,", len(values)-1)+"?")

    return cc.session.Query(query, values...).Exec()
}
```

#### **Neo4j (Graph Database)**

```go
type Neo4jClient struct {
    driver neo4j.Driver
}

func (nc *Neo4jClient) CreateNode(label string, properties map[string]interface{}) error {
    session := nc.driver.NewSession(neo4j.SessionConfig{})
    defer session.Close()

    query := fmt.Sprintf("CREATE (n:%s $props)", label)
    _, err := session.Run(query, map[string]interface{}{"props": properties})
    return err
}
```

### **NewSQL Databases**

#### **CockroachDB**

```go
type CockroachDBClient struct {
    db *sql.DB
}

func (cdb *CockroachDBClient) BeginTransaction() (*sql.Tx, error) {
    return cdb.db.Begin()
}

func (cdb *CockroachDBClient) ExecuteWithRetry(query string, args ...interface{}) error {
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        _, err := cdb.db.Exec(query, args...)
        if err == nil {
            return nil
        }

        if isRetryableError(err) {
            time.Sleep(time.Duration(i+1) * time.Second)
            continue
        }

        return err
    }
    return ErrMaxRetriesExceeded
}
```

---

## ⚠️ Error Handling & Scaling

### **Error Handling Patterns**

#### **Circuit Breaker**

```go
type CircuitBreaker struct {
    maxFailures int
    timeout     time.Duration
    failures    int
    lastFailure time.Time
    state       State
    mutex       sync.RWMutex
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()

    if cb.state == Open {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = HalfOpen
        } else {
            return ErrCircuitOpen
        }
    }

    err := fn()
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.maxFailures {
            cb.state = Open
        }
        return err
    }

    cb.failures = 0
    cb.state = Closed
    return nil
}
```

#### **Retry with Exponential Backoff**

```go
type RetryConfig struct {
    MaxRetries int
    BaseDelay  time.Duration
    MaxDelay   time.Duration
}

func RetryWithBackoff(config RetryConfig, fn func() error) error {
    var lastErr error

    for i := 0; i <= config.MaxRetries; i++ {
        if err := fn(); err == nil {
            return nil
        } else {
            lastErr = err
        }

        if i < config.MaxRetries {
            delay := time.Duration(1<<uint(i)) * config.BaseDelay
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
            time.Sleep(delay)
        }
    }

    return lastErr
}
```

### **Scaling Strategies**

#### **Horizontal Scaling**

```go
type LoadBalancer struct {
    servers []*Server
    current int
    mutex   sync.Mutex
}

func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()

    server := lb.servers[lb.current]
    lb.current = (lb.current + 1) % len(lb.servers)
    return server
}

func (lb *LoadBalancer) AddServer(server *Server) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()

    lb.servers = append(lb.servers, server)
}
```

#### **Database Sharding**

```go
type ShardManager struct {
    shards map[string]*Shard
    hashFunc func(string) string
}

type Shard struct {
    id    string
    db    *sql.DB
    range_ Range
}

func (sm *ShardManager) GetShard(key string) *Shard {
    shardKey := sm.hashFunc(key)
    return sm.shards[shardKey]
}

func (sm *ShardManager) Write(key string, value interface{}) error {
    shard := sm.GetShard(key)
    return shard.Write(key, value)
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Design a Distributed Cache System**

**Question**: "Design a distributed cache system that can handle 1 million requests per second with 99.9% availability."

**Answer Framework**:

```go
type DistributedCache struct {
    nodes     map[string]*CacheNode
    hashRing  *ConsistentHash
    replicas  int
    mutex     sync.RWMutex
}

type CacheNode struct {
    id       string
    data     map[string]interface{}
    capacity int
    mutex    sync.RWMutex
}

func (dc *DistributedCache) Get(key string) (interface{}, bool) {
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()

    nodeID := dc.hashRing.GetNode(key)
    node := dc.nodes[nodeID]

    node.mutex.RLock()
    defer node.mutex.RUnlock()

    value, exists := node.data[key]
    return value, exists
}

func (dc *DistributedCache) Set(key string, value interface{}) error {
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()

    nodeID := dc.hashRing.GetNode(key)
    node := dc.nodes[nodeID]

    node.mutex.Lock()
    defer node.mutex.Unlock()

    if len(node.data) >= node.capacity {
        dc.evictLRU(node)
    }

    node.data[key] = value
    return nil
}
```

#### **2. Implement a Distributed Lock**

**Question**: "Implement a distributed lock that can be used across multiple services."

**Answer Framework**:

```go
type DistributedLock struct {
    client    *redis.Client
    key       string
    value     string
    ttl       time.Duration
    mutex     sync.Mutex
}

func (dl *DistributedLock) Lock() error {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()

    success, err := dl.client.SetNX(context.Background(), dl.key, dl.value, dl.ttl).Result()
    if err != nil {
        return err
    }

    if !success {
        return ErrLockAcquisitionFailed
    }

    return nil
}

func (dl *DistributedLock) Unlock() error {
    script := `
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
    `

    result, err := dl.client.Eval(context.Background(), script, []string{dl.key}, dl.value).Result()
    if err != nil {
        return err
    }

    if result.(int64) == 0 {
        return ErrLockNotOwned
    }

    return nil
}
```

### **Meta Interview Questions**

#### **3. Design a News Feed System**

**Question**: "Design a news feed system similar to Facebook's that can handle millions of users."

**Answer Framework**:

```go
type NewsFeedService struct {
    userService    *UserService
    postService    *PostService
    feedService    *FeedService
    cacheService   *CacheService
}

type FeedItem struct {
    ID        string
    UserID    string
    Content   string
    Timestamp time.Time
    Score     float64
}

func (nfs *NewsFeedService) GetFeed(userID string, limit int) ([]FeedItem, error) {
    // Check cache first
    if cached, err := nfs.cacheService.Get(fmt.Sprintf("feed:%s", userID)); err == nil {
        return cached, nil
    }

    // Get user's friends
    friends, err := nfs.userService.GetFriends(userID)
    if err != nil {
        return nil, err
    }

    // Get posts from friends
    posts, err := nfs.postService.GetPostsByUsers(friends, limit)
    if err != nil {
        return nil, err
    }

    // Rank posts by score
    rankedPosts := nfs.rankPosts(posts)

    // Cache result
    nfs.cacheService.Set(fmt.Sprintf("feed:%s", userID), rankedPosts, time.Hour)

    return rankedPosts, nil
}

func (nfs *NewsFeedService) rankPosts(posts []Post) []FeedItem {
    // Implement ranking algorithm (recency, engagement, etc.)
    feedItems := make([]FeedItem, len(posts))

    for i, post := range posts {
        score := nfs.calculateScore(post)
        feedItems[i] = FeedItem{
            ID:        post.ID,
            UserID:    post.UserID,
            Content:   post.Content,
            Timestamp: post.Timestamp,
            Score:     score,
        }
    }

    // Sort by score
    sort.Slice(feedItems, func(i, j int) bool {
        return feedItems[i].Score > feedItems[j].Score
    })

    return feedItems
}
```

### **Amazon Interview Questions**

#### **4. Design a URL Shortener**

**Question**: "Design a URL shortener service like bit.ly that can handle billions of URLs."

**Answer Framework**:

```go
type URLShortener struct {
    db        *sql.DB
    cache     *redis.Client
    baseURL   string
    counter   int64
    mutex     sync.Mutex
}

func (us *URLShortener) ShortenURL(longURL string) (string, error) {
    // Check if URL already exists
    if short, err := us.getExistingShortURL(longURL); err == nil {
        return short, nil
    }

    // Generate new short URL
    us.mutex.Lock()
    us.counter++
    shortCode := us.encodeBase62(us.counter)
    us.mutex.Unlock()

    // Store in database
    if err := us.storeURL(shortCode, longURL); err != nil {
        return "", err
    }

    // Cache the mapping
    us.cache.Set(context.Background(), shortCode, longURL, time.Hour*24)

    return us.baseURL + shortCode, nil
}

func (us *URLShortener) ExpandURL(shortCode string) (string, error) {
    // Check cache first
    if longURL, err := us.cache.Get(context.Background(), shortCode).Result(); err == nil {
        return longURL, nil
    }

    // Check database
    longURL, err := us.getLongURL(shortCode)
    if err != nil {
        return "", err
    }

    // Cache the result
    us.cache.Set(context.Background(), shortCode, longURL, time.Hour*24)

    return longURL, nil
}

func (us *URLShortener) encodeBase62(num int64) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := ""

    for num > 0 {
        result = string(charset[num%62]) + result
        num /= 62
    }

    return result
}
```

### **Netflix Interview Questions**

#### **5. Design a Video Streaming System**

**Question**: "Design a video streaming system that can handle millions of concurrent users."

**Answer Framework**:

```go
type VideoStreamingService struct {
    cdnService     *CDNService
    userService    *UserService
    videoService   *VideoService
    analyticsService *AnalyticsService
}

type VideoStream struct {
    VideoID    string
    UserID     string
    Quality    string
    Bitrate    int
    StartTime  time.Time
    EndTime    time.Time
}

func (vss *VideoStreamingService) StreamVideo(userID, videoID string) (*VideoStream, error) {
    // Check user subscription
    if !vss.userService.HasValidSubscription(userID) {
        return nil, ErrSubscriptionRequired
    }

    // Get video metadata
    video, err := vss.videoService.GetVideo(videoID)
    if err != nil {
        return nil, err
    }

    // Determine optimal quality based on user's connection
    quality := vss.determineOptimalQuality(userID, video)

    // Get CDN URL
    cdnURL, err := vss.cdnService.GetStreamURL(videoID, quality)
    if err != nil {
        return nil, err
    }

    // Create stream session
    stream := &VideoStream{
        VideoID:   videoID,
        UserID:    userID,
        Quality:   quality,
        Bitrate:   video.Bitrates[quality],
        StartTime: time.Now(),
    }

    // Log analytics
    vss.analyticsService.LogStreamStart(stream)

    return stream, nil
}

func (vss *VideoStreamingService) determineOptimalQuality(userID string, video *Video) string {
    // Get user's connection speed
    connectionSpeed := vss.userService.GetConnectionSpeed(userID)

    // Determine best quality based on connection speed
    if connectionSpeed > 10000 { // 10 Mbps
        return "4K"
    } else if connectionSpeed > 5000 { // 5 Mbps
        return "1080p"
    } else if connectionSpeed > 2000 { // 2 Mbps
        return "720p"
    } else {
        return "480p"
    }
}
```

---

## 📚 Additional Resources

### **Books**

- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [Database Systems: The Complete Book](https://www.amazon.com/Database-Systems-Complete-Hector-Garcia-Molina/dp/0131873253/) - Hector Garcia-Molina
- [High Performance MySQL](https://www.oreilly.com/library/view/high-performance-mysql/9781449332471/) - Baron Schwartz

### **Online Resources**

- [High Scalability](https://highscalability.com/) - Real-world system design case studies
- [System Design Primer](https://github.com/donnemartin/system-design-primer/) - GitHub repository
- [Database Internals](https://www.databass.dev/) - Alex Petrov

### **Video Resources**

- [ByteByteGo](https://www.youtube.com/c/ByteByteGo/) - System design explanations
- [Gaurav Sen](https://www.youtube.com/c/GauravSensei/) - System design and algorithms
- [Exponent](https://www.youtube.com/c/ExponentTV/) - Mock interviews and system design

---

_This comprehensive guide covers all essential concepts from "Designing Data-Intensive Applications" along with practical implementations and real-world interview questions from top tech companies._
