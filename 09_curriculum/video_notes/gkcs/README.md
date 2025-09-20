# GKC's Video Notes

## Table of Contents

1. [Overview](#overview)
2. [System Design Videos](#system-design-videos)
3. [Algorithm Videos](#algorithm-videos)
4. [Database Videos](#database-videos)
5. [Distributed Systems Videos](#distributed-systems-videos)
6. [Implementation Examples](#implementation-examples)
7. [Cross-References](#cross-references)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)

## Overview

### Learning Objectives

- Extract key insights from GKC's engineering videos
- Convert video content into structured learning materials
- Create implementation examples based on video concepts
- Build comprehensive reference system for advanced topics
- Integrate video content with curriculum modules

### What is GKC's Content?

GKC provides in-depth technical content covering system design, algorithms, databases, and distributed systems with practical implementations and real-world examples.

## System Design Videos

### 1. Scalability Patterns

#### Video: "How to Design a Scalable System"
- **Duration**: 45 minutes
- **Key Concepts**: Horizontal scaling, load balancing, caching strategies
- **Implementation**: Golang and Node.js examples
- **Cross-Reference**: [Scalability Patterns](../../phase1_intermediate/system-design-basics/scalability-patterns.md)

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ScalableSystem struct {
    LoadBalancer *LoadBalancer
    Servers      []*Server
    Cache        *Cache
    mutex        sync.RWMutex
}

type LoadBalancer struct {
    servers []string
    current int
    mutex   sync.Mutex
}

func NewLoadBalancer(servers []string) *LoadBalancer {
    return &LoadBalancer{
        servers: servers,
        current: 0,
    }
}

func (lb *LoadBalancer) GetServer() string {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    server := lb.servers[lb.current]
    lb.current = (lb.current + 1) % len(lb.servers)
    return server
}

type Server struct {
    ID       string
    Capacity int
    Load     int
    mutex    sync.RWMutex
}

func (s *Server) HandleRequest() bool {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if s.Load < s.Capacity {
        s.Load++
        return true
    }
    return false
}

type Cache struct {
    data map[string]interface{}
    ttl  map[string]time.Time
    mutex sync.RWMutex
}

func NewCache() *Cache {
    return &Cache{
        data: make(map[string]interface{}),
        ttl:  make(map[string]time.Time),
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    value, exists := c.data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(c.ttl[key]) {
        return nil, false
    }
    
    return value, true
}

func (c *Cache) Set(key string, value interface{}, duration time.Duration) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    c.data[key] = value
    c.ttl[key] = time.Now().Add(duration)
}

func main() {
    servers := []string{"server1", "server2", "server3"}
    lb := NewLoadBalancer(servers)
    cache := NewCache()
    
    // Simulate requests
    for i := 0; i < 10; i++ {
        server := lb.GetServer()
        fmt.Printf("Request %d routed to %s\n", i+1, server)
        
        // Check cache first
        if value, exists := cache.Get("key1"); exists {
            fmt.Printf("Cache hit: %v\n", value)
        } else {
            fmt.Println("Cache miss, fetching from server")
            cache.Set("key1", "cached_value", 5*time.Minute)
        }
    }
}
```

#### Video: "Microservices Architecture Deep Dive"
- **Duration**: 60 minutes
- **Key Concepts**: Service discovery, API gateway, circuit breakers
- **Implementation**: Service mesh examples
- **Cross-Reference**: [Microservices Architecture](../../phase1_intermediate/system-design-basics/microservices-architecture.md)

### 2. Database Design

#### Video: "Database Sharding Strategies"
- **Duration**: 50 minutes
- **Key Concepts**: Horizontal partitioning, consistent hashing, data distribution
- **Implementation**: Sharding algorithms
- **Cross-Reference**: [Database Systems](../../../README.md)

```go
package main

import (
    "crypto/md5"
    "fmt"
    "strconv"
)

type ShardManager struct {
    shards []string
    count  int
}

func NewShardManager(shardCount int) *ShardManager {
    shards := make([]string, shardCount)
    for i := 0; i < shardCount; i++ {
        shards[i] = fmt.Sprintf("shard_%d", i)
    }
    
    return &ShardManager{
        shards: shards,
        count:  shardCount,
    }
}

func (sm *ShardManager) GetShard(key string) string {
    hash := md5.Sum([]byte(key))
    hashValue := int(hash[0]) + int(hash[1])*256 + int(hash[2])*65536
    index := hashValue % sm.count
    return sm.shards[index]
}

func (sm *ShardManager) GetShardsForRange(start, end string) []string {
    shardSet := make(map[string]bool)
    
    // Get shard for start key
    startShard := sm.GetShard(start)
    shardSet[startShard] = true
    
    // Get shard for end key
    endShard := sm.GetShard(end)
    shardSet[endShard] = true
    
    // In a real implementation, you might need to check intermediate shards
    result := make([]string, 0, len(shardSet))
    for shard := range shardSet {
        result = append(result, shard)
    }
    
    return result
}

func main() {
    manager := NewShardManager(4)
    
    keys := []string{"user:123", "user:456", "user:789", "user:101", "user:202"}
    
    fmt.Println("Key to Shard Mapping:")
    for _, key := range keys {
        shard := manager.GetShard(key)
        fmt.Printf("%s -> %s\n", key, shard)
    }
    
    fmt.Println("\nRange Query Shards:")
    shards := manager.GetShardsForRange("user:100", "user:300")
    for _, shard := range shards {
        fmt.Printf("Range query needs: %s\n", shard)
    }
}
```

## Algorithm Videos

### 1. Advanced Data Structures

#### Video: "Advanced Tree Structures"
- **Duration**: 40 minutes
- **Key Concepts**: B-trees, Red-black trees, Segment trees
- **Implementation**: Tree operations and algorithms
- **Cross-Reference**: [Advanced Trees](../../phase1_intermediate/advanced-dsa/advanced-trees.md)

```go
package main

import "fmt"

type BTreeNode struct {
    keys     []int
    children []*BTreeNode
    isLeaf   bool
    t        int // minimum degree
}

func NewBTreeNode(t int, isLeaf bool) *BTreeNode {
    return &BTreeNode{
        keys:     make([]int, 2*t-1),
        children: make([]*BTreeNode, 2*t),
        isLeaf:   isLeaf,
        t:        t,
    }
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

func (bt *BTree) Search(key int) *BTreeNode {
    if bt.root == nil {
        return nil
    }
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) *BTreeNode {
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }
    
    if i < len(node.keys) && node.keys[i] == key {
        return node
    }
    
    if node.isLeaf {
        return nil
    }
    
    return bt.searchNode(node.children[i], key)
}

func (bt *BTree) Insert(key int) {
    if bt.root == nil {
        bt.root = NewBTreeNode(bt.t, true)
        bt.root.keys[0] = key
        return
    }
    
    if len(bt.root.keys) == 2*bt.t-1 {
        newRoot := NewBTreeNode(bt.t, false)
        newRoot.children[0] = bt.root
        bt.splitChild(newRoot, 0)
        
        i := 0
        if newRoot.keys[0] < key {
            i++
        }
        bt.insertNonFull(newRoot.children[i], key)
        
        bt.root = newRoot
    } else {
        bt.insertNonFull(bt.root, key)
    }
}

func (bt *BTree) insertNonFull(node *BTreeNode, key int) {
    i := len(node.keys) - 1
    
    if node.isLeaf {
        for i >= 0 && node.keys[i] > key {
            node.keys[i+1] = node.keys[i]
            i--
        }
        node.keys[i+1] = key
    } else {
        for i >= 0 && node.keys[i] > key {
            i--
        }
        i++
        
        if len(node.children[i].keys) == 2*bt.t-1 {
            bt.splitChild(node, i)
            if node.keys[i] < key {
                i++
            }
        }
        bt.insertNonFull(node.children[i], key)
    }
}

func (bt *BTree) splitChild(parent *BTreeNode, index int) {
    t := bt.t
    y := parent.children[index]
    z := NewBTreeNode(t, y.isLeaf)
    
    for j := 0; j < t-1; j++ {
        z.keys[j] = y.keys[j+t]
    }
    
    if !y.isLeaf {
        for j := 0; j < t; j++ {
            z.children[j] = y.children[j+t]
        }
    }
    
    for j := len(parent.keys) - 1; j >= index; j-- {
        parent.keys[j+1] = parent.keys[j]
    }
    
    parent.keys[index] = y.keys[t-1]
    
    for j := len(parent.children) - 1; j > index; j-- {
        parent.children[j+1] = parent.children[j]
    }
    
    parent.children[index+1] = z
}

func main() {
    btree := NewBTree(3)
    
    keys := []int{10, 20, 5, 6, 12, 30, 7, 17}
    
    for _, key := range keys {
        btree.Insert(key)
        fmt.Printf("Inserted %d\n", key)
    }
    
    fmt.Println("\nSearching for key 6:")
    if node := btree.Search(6); node != nil {
        fmt.Println("Key 6 found!")
    } else {
        fmt.Println("Key 6 not found!")
    }
}
```

### 2. Graph Algorithms

#### Video: "Advanced Graph Algorithms"
- **Duration**: 55 minutes
- **Key Concepts**: Shortest path, minimum spanning tree, topological sort
- **Implementation**: Graph algorithms with optimizations
- **Cross-Reference**: [Graph Algorithms](../../phase1_intermediate/advanced-dsa/graph-algorithms.md)

## Database Videos

### 1. Query Optimization

#### Video: "Database Query Optimization Techniques"
- **Duration**: 45 minutes
- **Key Concepts**: Index optimization, query planning, execution strategies
- **Implementation**: Query optimization examples
- **Cross-Reference**: [Query Optimization](../../phase1_intermediate/database-systems/query-optimization.md)

```go
package main

import (
    "fmt"
    "sort"
    "time"
)

type QueryOptimizer struct {
    indexes map[string]*Index
    stats   map[string]*TableStats
}

type Index struct {
    name     string
    columns  []string
    type     string
    size     int
    selectivity float64
}

type TableStats struct {
    rowCount    int
    avgRowSize  int
    lastUpdated time.Time
}

type QueryPlan struct {
    operations []Operation
    cost       float64
    time       time.Duration
}

type Operation struct {
    type     string
    table    string
    cost     float64
    rows     int
    children []*Operation
}

func NewQueryOptimizer() *QueryOptimizer {
    return &QueryOptimizer{
        indexes: make(map[string]*Index),
        stats:   make(map[string]*TableStats),
    }
}

func (qo *QueryOptimizer) AddIndex(name string, columns []string, indexType string) {
    qo.indexes[name] = &Index{
        name:        name,
        columns:     columns,
        type:        indexType,
        size:        0, // Would be calculated in real implementation
        selectivity: 0.1, // Would be calculated based on data
    }
}

func (qo *QueryOptimizer) AddTableStats(table string, rowCount int, avgRowSize int) {
    qo.stats[table] = &TableStats{
        rowCount:    rowCount,
        avgRowSize:  avgRowSize,
        lastUpdated: time.Now(),
    }
}

func (qo *QueryOptimizer) OptimizeQuery(query string) *QueryPlan {
    // Simplified query optimization
    // In real implementation, this would parse SQL and create execution plan
    
    operations := []Operation{
        {
            type:  "Index Scan",
            table: "users",
            cost:  10.0,
            rows:  1000,
        },
        {
            type:  "Filter",
            table: "users",
            cost:  5.0,
            rows:  100,
        },
        {
            type:  "Sort",
            table: "users",
            cost:  15.0,
            rows:  100,
        },
    }
    
    totalCost := 0.0
    for _, op := range operations {
        totalCost += op.cost
    }
    
    return &QueryPlan{
        operations: operations,
        cost:       totalCost,
        time:       time.Duration(totalCost) * time.Millisecond,
    }
}

func (qo *QueryOptimizer) SuggestIndexes(query string) []string {
    // Simplified index suggestion
    // In real implementation, this would analyze query patterns
    
    suggestions := []string{
        "CREATE INDEX idx_users_email ON users(email)",
        "CREATE INDEX idx_orders_user_id ON orders(user_id)",
        "CREATE INDEX idx_products_category ON products(category)",
    }
    
    return suggestions
}

func main() {
    optimizer := NewQueryOptimizer()
    
    // Add some indexes
    optimizer.AddIndex("idx_users_email", []string{"email"}, "B-tree")
    optimizer.AddIndex("idx_users_name", []string{"first_name", "last_name"}, "B-tree")
    
    // Add table statistics
    optimizer.AddTableStats("users", 100000, 200)
    optimizer.AddTableStats("orders", 500000, 150)
    
    // Optimize a query
    query := "SELECT * FROM users WHERE email = 'test@example.com' ORDER BY created_at"
    plan := optimizer.OptimizeQuery(query)
    
    fmt.Println("Query Optimization Results:")
    fmt.Println("==========================")
    fmt.Printf("Query: %s\n", query)
    fmt.Printf("Total Cost: %.2f\n", plan.cost)
    fmt.Printf("Estimated Time: %v\n", plan.time)
    
    fmt.Println("\nOperations:")
    for i, op := range plan.operations {
        fmt.Printf("  %d. %s on %s (cost: %.2f, rows: %d)\n", 
            i+1, op.type, op.table, op.cost, op.rows)
    }
    
    fmt.Println("\nIndex Suggestions:")
    suggestions := optimizer.SuggestIndexes(query)
    for i, suggestion := range suggestions {
        fmt.Printf("  %d. %s\n", i+1, suggestion)
    }
}
```

## Distributed Systems Videos

### 1. Consensus Algorithms

#### Video: "Raft Consensus Algorithm Deep Dive"
- **Duration**: 60 minutes
- **Key Concepts**: Leader election, log replication, safety guarantees
- **Implementation**: Raft implementation
- **Cross-Reference**: [Consensus Algorithms](../../phase2_advanced/distributed-systems/consensus-algorithms.md)

### 2. Distributed Storage

#### Video: "Distributed Storage Systems"
- **Duration**: 50 minutes
- **Key Concepts**: Replication, consistency models, partition tolerance
- **Implementation**: Storage system design
- **Cross-Reference**: [Distributed Storage](../../phase2_advanced/distributed-systems/distributed-storage.md)

## Implementation Examples

### 1. System Design Patterns

#### Load Balancing Implementation
```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type LoadBalancer struct {
    servers    []*Server
    strategy   string
    current    int
    mutex      sync.Mutex
}

type Server struct {
    ID           string
    Health       bool
    ResponseTime time.Duration
    Connections  int
    mutex        sync.RWMutex
}

func NewServer(id string) *Server {
    return &Server{
        ID:           id,
        Health:       true,
        ResponseTime: time.Millisecond * 100,
        Connections:  0,
    }
}

func (s *Server) HandleRequest() bool {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if !s.Health {
        return false
    }
    
    s.Connections++
    time.Sleep(s.ResponseTime) // Simulate processing
    return true
}

func NewLoadBalancer(servers []*Server, strategy string) *LoadBalancer {
    return &LoadBalancer{
        servers:  servers,
        strategy: strategy,
        current:  0,
    }
}

func (lb *LoadBalancer) SelectServer() *Server {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    healthyServers := lb.getHealthyServers()
    if len(healthyServers) == 0 {
        return nil
    }
    
    switch lb.strategy {
    case "round_robin":
        return lb.roundRobin(healthyServers)
    case "random":
        return lb.random(healthyServers)
    case "least_connections":
        return lb.leastConnections(healthyServers)
    default:
        return lb.roundRobin(healthyServers)
    }
}

func (lb *LoadBalancer) getHealthyServers() []*Server {
    var healthy []*Server
    for _, server := range lb.servers {
        if server.Health {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

func (lb *LoadBalancer) roundRobin(servers []*Server) *Server {
    server := servers[lb.current%len(servers)]
    lb.current++
    return server
}

func (lb *LoadBalancer) random(servers []*Server) *Server {
    index := rand.Intn(len(servers))
    return servers[index]
}

func (lb *LoadBalancer) leastConnections(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }
    
    least := servers[0]
    for _, server := range servers[1:] {
        if server.Connections < least.Connections {
            least = server
        }
    }
    return least
}

func main() {
    // Create servers
    servers := []*Server{
        NewServer("server1"),
        NewServer("server2"),
        NewServer("server3"),
    }
    
    // Create load balancer
    lb := NewLoadBalancer(servers, "round_robin")
    
    // Simulate requests
    for i := 0; i < 10; i++ {
        server := lb.SelectServer()
        if server != nil {
            success := server.HandleRequest()
            fmt.Printf("Request %d: %s - %v\n", i+1, server.ID, success)
        } else {
            fmt.Printf("Request %d: No healthy servers available\n", i+1)
        }
        time.Sleep(time.Millisecond * 50)
    }
}
```

## Cross-References

### 1. Curriculum Integration

#### Phase 1 Integration
- **System Design Basics**: [Link to module](../../../README.md)
- **Database Systems**: [Link to module](../../../README.md)
- **Advanced DSA**: [Link to module](../../../README.md)

#### Phase 2 Integration
- **Distributed Systems**: [Link to module](../../../README.md)
- **Cloud Architecture**: [Link to module](../../../README.md)
- **Performance Engineering**: [Link to module](../../../README.md)

### 2. Video Content Mapping

#### Beginner Level Videos
- System Design Fundamentals
- Basic Algorithm Concepts
- Database Basics

#### Intermediate Level Videos
- Advanced Data Structures
- Query Optimization
- Caching Strategies

#### Advanced Level Videos
- Distributed Systems
- Consensus Algorithms
- High-Performance Systems

## Follow-up Questions

### 1. Content Extraction
**Q: How do you ensure video content is accurately captured?**
A: Use structured note-taking, timestamp references, and cross-validation with multiple sources.

### 2. Implementation Quality
**Q: What makes a good code example from video content?**
A: Clear, runnable code with proper error handling, comments, and real-world applicability.

### 3. Integration Strategy
**Q: How do you integrate video content with existing curriculum?**
A: Map video topics to curriculum modules, create cross-references, and ensure consistency.

## Sources

### Video Channels
- **GKC's Channel**: [YouTube Channel](https://www.youtube.com/@gkcs/)
- **System Design Playlist**: [Playlist](https://www.youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPcvoX/)
- **Algorithm Playlist**: [Playlist](https://www.youtube.com/playlist?list=PLMCXHnjXnTnvQzJ4qgJNQhY6x5vJ1MpnT/)

### Related Resources
- **System Design Primer**: [GitHub](https://github.com/donnemartin/system-design-primer/)
- **Algorithm Visualizations**: [VisuAlgo](https://visualgo.net/)
- **Database Internals**: [Database Internals](https://www.databass.dev/)

---

**Next**: [Asli Engineering](../../../README.md) | **Previous**: [Company Prep](../../../README.md) | **Up**: [Video Notes](README.md)
