# 🗄️ **Database Sharding Strategies - Complete Guide**

## 📊 **Comprehensive Guide to Database Sharding and Partitioning**

---

## 🎯 **1. Sharding Fundamentals**

### **What is Database Sharding?**

Database sharding is a method of horizontal partitioning where data is distributed across multiple database instances (shards) to improve performance, scalability, and availability.

**Benefits:**
- **Horizontal Scalability**: Add more shards as data grows
- **Performance**: Smaller datasets per shard
- **Fault Isolation**: Failure of one shard doesn't affect others
- **Geographic Distribution**: Place shards closer to users

**Challenges:**
- **Cross-shard Queries**: Complex queries across multiple shards
- **Data Rebalancing**: Moving data when adding/removing shards
- **Transaction Consistency**: Maintaining ACID across shards
- **Schema Changes**: Coordinating changes across all shards

### **Sharding Strategies**

#### **1. Range-Based Sharding**

```go
package main

import (
    "fmt"
    "sort"
    "sync"
)

type RangeSharding struct {
    shards []Shard
    ranges []ShardRange
    mutex  sync.RWMutex
}

type Shard struct {
    ID       string
    Database *Database
    StartKey int
    EndKey   int
}

type ShardRange struct {
    Start   int
    End     int
    ShardID string
}

type Database struct {
    ID     string
    Data   map[string]interface{}
    mutex  sync.RWMutex
}

func NewRangeSharding() *RangeSharding {
    return &RangeSharding{
        shards: make([]Shard, 0),
        ranges: make([]ShardRange, 0),
    }
}

func (rs *RangeSharding) AddShard(shardID string, startKey, endKey int) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()
    
    // Check for overlapping ranges
    for _, r := range rs.ranges {
        if (startKey >= r.Start && startKey <= r.End) ||
           (endKey >= r.Start && endKey <= r.End) ||
           (startKey <= r.Start && endKey >= r.End) {
            return fmt.Errorf("overlapping range detected")
        }
    }
    
    shard := Shard{
        ID:       shardID,
        Database: &Database{ID: shardID, Data: make(map[string]interface{})},
        StartKey: startKey,
        EndKey:   endKey,
    }
    
    rs.shards = append(rs.shards, shard)
    rs.ranges = append(rs.ranges, ShardRange{
        Start:   startKey,
        End:     endKey,
        ShardID: shardID,
    })
    
    // Sort ranges by start key
    sort.Slice(rs.ranges, func(i, j int) bool {
        return rs.ranges[i].Start < rs.ranges[j].Start
    })
    
    return nil
}

func (rs *RangeSharding) GetShard(key int) (*Shard, error) {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()
    
    for _, r := range rs.ranges {
        if key >= r.Start && key <= r.End {
            for _, shard := range rs.shards {
                if shard.ID == r.ShardID {
                    return &shard, nil
                }
            }
        }
    }
    
    return nil, fmt.Errorf("no shard found for key %d", key)
}

func (rs *RangeSharding) Write(key int, value interface{}) error {
    shard, err := rs.GetShard(key)
    if err != nil {
        return err
    }
    
    shard.Database.mutex.Lock()
    defer shard.Database.mutex.Unlock()
    
    shard.Database.Data[fmt.Sprintf("%d", key)] = value
    return nil
}

func (rs *RangeSharding) Read(key int) (interface{}, error) {
    shard, err := rs.GetShard(key)
    if err != nil {
        return nil, err
    }
    
    shard.Database.mutex.RLock()
    defer shard.Database.mutex.RUnlock()
    
    value, exists := shard.Database.Data[fmt.Sprintf("%d", key)]
    if !exists {
        return nil, fmt.Errorf("key %d not found", key)
    }
    
    return value, nil
}

func (rs *RangeSharding) RangeQuery(startKey, endKey int) (map[string]interface{}, error) {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()
    
    result := make(map[string]interface{})
    
    for _, r := range rs.ranges {
        // Check if range overlaps with query range
        if r.End >= startKey && r.Start <= endKey {
            for _, shard := range rs.shards {
                if shard.ID == r.ShardID {
                    shard.Database.mutex.RLock()
                    for k, v := range shard.Database.Data {
                        keyInt := parseInt(k)
                        if keyInt >= startKey && keyInt <= endKey {
                            result[k] = v
                        }
                    }
                    shard.Database.mutex.RUnlock()
                }
            }
        }
    }
    
    return result, nil
}

func parseInt(s string) int {
    // Simple implementation - in real code, use strconv.Atoi
    result := 0
    for _, c := range s {
        if c >= '0' && c <= '9' {
            result = result*10 + int(c-'0')
        }
    }
    return result
}

// Example usage
func main() {
    sharding := NewRangeSharding()
    
    // Add shards
    sharding.AddShard("shard1", 1, 1000)
    sharding.AddShard("shard2", 1001, 2000)
    sharding.AddShard("shard3", 2001, 3000)
    
    // Write data
    sharding.Write(500, "data1")
    sharding.Write(1500, "data2")
    sharding.Write(2500, "data3")
    
    // Read data
    if value, err := sharding.Read(500); err == nil {
        fmt.Printf("Read: %v\n", value)
    }
    
    // Range query
    if results, err := sharding.RangeQuery(400, 1600); err == nil {
        fmt.Printf("Range query results: %+v\n", results)
    }
}
```

#### **2. Hash-Based Sharding**

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sync"
)

type HashSharding struct {
    shards []Shard
    hashFunc func(string) int
    shardCount int
    mutex  sync.RWMutex
}

func NewHashSharding(shardCount int) *HashSharding {
    return &HashSharding{
        shards: make([]Shard, shardCount),
        hashFunc: func(key string) int {
            hash := md5.Sum([]byte(key))
            result := 0
            for _, b := range hash {
                result = (result << 8) | int(b)
            }
            if result < 0 {
                result = -result
            }
            return result
        },
        shardCount: shardCount,
    }
}

func (hs *HashSharding) InitializeShards() {
    hs.mutex.Lock()
    defer hs.mutex.Unlock()
    
    for i := 0; i < hs.shardCount; i++ {
        hs.shards[i] = Shard{
            ID:       fmt.Sprintf("shard_%d", i),
            Database: &Database{ID: fmt.Sprintf("shard_%d", i), Data: make(map[string]interface{})},
        }
    }
}

func (hs *HashSharding) GetShard(key string) *Shard {
    hs.mutex.RLock()
    defer hs.mutex.RUnlock()
    
    hash := hs.hashFunc(key)
    shardIndex := hash % hs.shardCount
    return &hs.shards[shardIndex]
}

func (hs *HashSharding) Write(key string, value interface{}) error {
    shard := hs.GetShard(key)
    
    shard.Database.mutex.Lock()
    defer shard.Database.mutex.Unlock()
    
    shard.Database.Data[key] = value
    return nil
}

func (hs *HashSharding) Read(key string) (interface{}, error) {
    shard := hs.GetShard(key)
    
    shard.Database.mutex.RLock()
    defer shard.Database.mutex.RUnlock()
    
    value, exists := shard.Database.Data[key]
    if !exists {
        return nil, fmt.Errorf("key %s not found", key)
    }
    
    return value, nil
}

func (hs *HashSharding) Delete(key string) error {
    shard := hs.GetShard(key)
    
    shard.Database.mutex.Lock()
    defer shard.Database.mutex.Unlock()
    
    delete(shard.Database.Data, key)
    return nil
}

func (hs *HashSharding) GetShardStats() map[string]int {
    hs.mutex.RLock()
    defer hs.mutex.RUnlock()
    
    stats := make(map[string]int)
    for _, shard := range hs.shards {
        shard.Database.mutex.RLock()
        stats[shard.ID] = len(shard.Database.Data)
        shard.Database.mutex.RUnlock()
    }
    
    return stats
}

// Example usage
func main() {
    sharding := NewHashSharding(3)
    sharding.InitializeShards()
    
    // Write data
    sharding.Write("user:1", "John Doe")
    sharding.Write("user:2", "Jane Smith")
    sharding.Write("user:3", "Bob Johnson")
    sharding.Write("user:4", "Alice Brown")
    
    // Read data
    if value, err := sharding.Read("user:1"); err == nil {
        fmt.Printf("Read: %v\n", value)
    }
    
    // Get shard statistics
    stats := sharding.GetShardStats()
    fmt.Printf("Shard stats: %+v\n", stats)
}
```

#### **3. Directory-Based Sharding**

```go
package main

import (
    "fmt"
    "sync"
)

type DirectorySharding struct {
    shards    []Shard
    directory map[string]string // key -> shard_id
    mutex     sync.RWMutex
}

func NewDirectorySharding() *DirectorySharding {
    return &DirectorySharding{
        shards:    make([]Shard, 0),
        directory: make(map[string]string),
    }
}

func (ds *DirectorySharding) AddShard(shardID string) error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    shard := Shard{
        ID:       shardID,
        Database: &Database{ID: shardID, Data: make(map[string]interface{})},
    }
    
    ds.shards = append(ds.shards, shard)
    return nil
}

func (ds *DirectorySharding) GetShard(key string) (*Shard, error) {
    ds.mutex.RLock()
    shardID, exists := ds.directory[key]
    ds.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("key %s not found in directory", key)
    }
    
    for _, shard := range ds.shards {
        if shard.ID == shardID {
            return &shard, nil
        }
    }
    
    return nil, fmt.Errorf("shard %s not found", shardID)
}

func (ds *DirectorySharding) Write(key string, value interface{}) error {
    // Find least loaded shard
    shard, err := ds.findLeastLoadedShard()
    if err != nil {
        return err
    }
    
    // Write to shard
    shard.Database.mutex.Lock()
    shard.Database.Data[key] = value
    shard.Database.mutex.Unlock()
    
    // Update directory
    ds.mutex.Lock()
    ds.directory[key] = shard.ID
    ds.mutex.Unlock()
    
    return nil
}

func (ds *DirectorySharding) Read(key string) (interface{}, error) {
    shard, err := ds.GetShard(key)
    if err != nil {
        return nil, err
    }
    
    shard.Database.mutex.RLock()
    defer shard.Database.mutex.RUnlock()
    
    value, exists := shard.Database.Data[key]
    if !exists {
        return nil, fmt.Errorf("key %s not found", key)
    }
    
    return value, nil
}

func (ds *DirectorySharding) findLeastLoadedShard() (*Shard, error) {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    if len(ds.shards) == 0 {
        return nil, fmt.Errorf("no shards available")
    }
    
    leastLoaded := &ds.shards[0]
    minCount := len(leastLoaded.Database.Data)
    
    for i := 1; i < len(ds.shards); i++ {
        count := len(ds.shards[i].Database.Data)
        if count < minCount {
            leastLoaded = &ds.shards[i]
            minCount = count
        }
    }
    
    return leastLoaded, nil
}

func (ds *DirectorySharding) Rebalance() error {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    // Calculate total data count
    totalCount := 0
    for _, shard := range ds.shards {
        shard.Database.mutex.RLock()
        totalCount += len(shard.Database.Data)
        shard.Database.mutex.RUnlock()
    }
    
    if len(ds.shards) == 0 {
        return fmt.Errorf("no shards available")
    }
    
    targetCount := totalCount / len(ds.shards)
    
    // Redistribute data
    for _, shard := range ds.shards {
        shard.Database.mutex.RLock()
        currentCount := len(shard.Database.Data)
        shard.Database.mutex.RUnlock()
        
        if currentCount > targetCount {
            // Move excess data to other shards
            ds.moveExcessData(&shard, currentCount-targetCount)
        }
    }
    
    return nil
}

func (ds *DirectorySharding) moveExcessData(shard *Shard, excessCount int) {
    // Implementation would move excess data to other shards
    // This is a simplified version
    fmt.Printf("Moving %d items from shard %s\n", excessCount, shard.ID)
}

// Example usage
func main() {
    sharding := NewDirectorySharding()
    
    // Add shards
    sharding.AddShard("shard1")
    sharding.AddShard("shard2")
    sharding.AddShard("shard3")
    
    // Write data
    sharding.Write("user:1", "John Doe")
    sharding.Write("user:2", "Jane Smith")
    sharding.Write("user:3", "Bob Johnson")
    
    // Read data
    if value, err := sharding.Read("user:1"); err == nil {
        fmt.Printf("Read: %v\n", value)
    }
    
    // Rebalance
    sharding.Rebalance()
}
```

---

## 🎯 **2. Consistent Hashing for Sharding**

### **Consistent Hashing Implementation**

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
)

type ConsistentHash struct {
    ring       []HashNode
    virtualNodes int
    mutex      sync.RWMutex
}

type HashNode struct {
    Hash     uint32
    NodeID   string
    IsVirtual bool
    RealNode string
}

func NewConsistentHash(virtualNodes int) *ConsistentHash {
    return &ConsistentHash{
        ring:        make([]HashNode, 0),
        virtualNodes: virtualNodes,
    }
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    // Add real node
    ch.addNodeToRing(nodeID, false, nodeID)
    
    // Add virtual nodes
    for i := 0; i < ch.virtualNodes; i++ {
        virtualNodeID := fmt.Sprintf("%s#%d", nodeID, i)
        ch.addNodeToRing(virtualNodeID, true, nodeID)
    }
    
    // Sort ring
    ch.sortRing()
}

func (ch *ConsistentHash) RemoveNode(nodeID string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    // Remove real node and all its virtual nodes
    newRing := make([]HashNode, 0)
    for _, node := range ch.ring {
        if node.NodeID != nodeID && node.RealNode != nodeID {
            newRing = append(newRing, node)
        }
    }
    ch.ring = newRing
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.ring) == 0 {
        return ""
    }
    
    hash := ch.hash(key)
    
    // Find first node with hash >= key hash
    for _, node := range ch.ring {
        if node.Hash >= hash {
            return node.RealNode
        }
    }
    
    // Wrap around to first node
    return ch.ring[0].RealNode
}

func (ch *ConsistentHash) addNodeToRing(nodeID string, isVirtual bool, realNode string) {
    hash := ch.hash(nodeID)
    node := HashNode{
        Hash:      hash,
        NodeID:    nodeID,
        IsVirtual: isVirtual,
        RealNode:  realNode,
    }
    ch.ring = append(ch.ring, node)
}

func (ch *ConsistentHash) sortRing() {
    sort.Slice(ch.ring, func(i, j int) bool {
        return ch.ring[i].Hash < ch.ring[j].Hash
    })
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (ch *ConsistentHash) GetRing() []HashNode {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    return append([]HashNode{}, ch.ring...)
}

// Example usage
func main() {
    ch := NewConsistentHash(3)
    
    // Add nodes
    ch.AddNode("node1")
    ch.AddNode("node2")
    ch.AddNode("node3")
    
    // Test key distribution
    keys := []string{"key1", "key2", "key3", "key4", "key5"}
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key %s -> Node %s\n", key, node)
    }
    
    // Remove a node
    ch.RemoveNode("node2")
    fmt.Println("\nAfter removing node2:")
    
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key %s -> Node %s\n", key, node)
    }
}
```

---

## 🎯 **3. Cross-Shard Queries and Transactions**

### **Cross-Shard Query Engine**

```go
package main

import (
    "fmt"
    "sync"
)

type CrossShardQueryEngine struct {
    shards map[string]*Shard
    mutex  sync.RWMutex
}

type QueryResult struct {
    ShardID string
    Data    []map[string]interface{}
    Error   error
}

type CrossShardQuery struct {
    SQL      string
    ShardIDs []string
    MergeFunc func([]QueryResult) interface{}
}

func NewCrossShardQueryEngine() *CrossShardQueryEngine {
    return &CrossShardQueryEngine{
        shards: make(map[string]*Shard),
    }
}

func (cqe *CrossShardQueryEngine) AddShard(shardID string, shard *Shard) {
    cqe.mutex.Lock()
    defer cqe.mutex.Unlock()
    
    cqe.shards[shardID] = shard
}

func (cqe *CrossShardQueryEngine) ExecuteQuery(query *CrossShardQuery) (interface{}, error) {
    cqe.mutex.RLock()
    defer cqe.mutex.RUnlock()
    
    var wg sync.WaitGroup
    results := make([]QueryResult, len(query.ShardIDs))
    
    // Execute query on each shard
    for i, shardID := range query.ShardIDs {
        shard, exists := cqe.shards[shardID]
        if !exists {
            results[i] = QueryResult{
                ShardID: shardID,
                Error:   fmt.Errorf("shard %s not found", shardID),
            }
            continue
        }
        
        wg.Add(1)
        go func(index int, s *Shard) {
            defer wg.Done()
            
            data, err := s.ExecuteQuery(query.SQL)
            results[index] = QueryResult{
                ShardID: query.ShardIDs[index],
                Data:    data,
                Error:   err,
            }
        }(i, shard)
    }
    
    wg.Wait()
    
    // Merge results
    return query.MergeFunc(results), nil
}

func (cqe *CrossShardQueryEngine) ExecuteAggregateQuery(query *CrossShardQuery) (interface{}, error) {
    cqe.mutex.RLock()
    defer cqe.mutex.RUnlock()
    
    var wg sync.WaitGroup
    results := make([]QueryResult, len(query.ShardIDs))
    
    // Execute query on each shard
    for i, shardID := range query.ShardIDs {
        shard, exists := cqe.shards[shardID]
        if !exists {
            results[i] = QueryResult{
                ShardID: shardID,
                Error:   fmt.Errorf("shard %s not found", shardID),
            }
            continue
        }
        
        wg.Add(1)
        go func(index int, s *Shard) {
            defer wg.Done()
            
            data, err := s.ExecuteQuery(query.SQL)
            results[index] = QueryResult{
                ShardID: query.ShardIDs[index],
                Data:    data,
                Error:   err,
            }
        }(i, shard)
    }
    
    wg.Wait()
    
    // Aggregate results
    return cqe.aggregateResults(results), nil
}

func (cqe *CrossShardQueryEngine) aggregateResults(results []QueryResult) map[string]interface{} {
    totalCount := 0
    totalSum := 0.0
    errors := make([]string, 0)
    
    for _, result := range results {
        if result.Error != nil {
            errors = append(errors, result.Error.Error())
            continue
        }
        
        for _, row := range result.Data {
            if count, ok := row["count"].(int); ok {
                totalCount += count
            }
            if sum, ok := row["sum"].(float64); ok {
                totalSum += sum
            }
        }
    }
    
    return map[string]interface{}{
        "total_count": totalCount,
        "total_sum":   totalSum,
        "errors":      errors,
    }
}

// Example usage
func main() {
    engine := NewCrossShardQueryEngine()
    
    // Add shards
    engine.AddShard("shard1", &Shard{ID: "shard1"})
    engine.AddShard("shard2", &Shard{ID: "shard2"})
    engine.AddShard("shard3", &Shard{ID: "shard3"})
    
    // Execute cross-shard query
    query := &CrossShardQuery{
        SQL:      "SELECT * FROM users WHERE age > 18",
        ShardIDs: []string{"shard1", "shard2", "shard3"},
        MergeFunc: func(results []QueryResult) interface{} {
            var allData []map[string]interface{}
            for _, result := range results {
                if result.Error == nil {
                    allData = append(allData, result.Data...)
                }
            }
            return allData
        },
    }
    
    result, err := engine.ExecuteQuery(query)
    if err != nil {
        fmt.Printf("Query error: %v\n", err)
    } else {
        fmt.Printf("Query result: %+v\n", result)
    }
}
```

---

## 🎯 **4. Data Rebalancing and Migration**

### **Data Rebalancing System**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RebalancingSystem struct {
    shards     map[string]*Shard
    rebalancer *DataRebalancer
    mutex      sync.RWMutex
}

type DataRebalancer struct {
    threshold float64 // Load imbalance threshold
    mutex     sync.RWMutex
}

type RebalancePlan struct {
    Moves []DataMove
}

type DataMove struct {
    FromShard string
    ToShard   string
    Keys      []string
    Size      int
}

func NewRebalancingSystem() *RebalancingSystem {
    return &RebalancingSystem{
        shards: make(map[string]*Shard),
        rebalancer: &DataRebalancer{
            threshold: 0.2, // 20% imbalance threshold
        },
    }
}

func (rs *RebalancingSystem) AddShard(shardID string, shard *Shard) {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()
    
    rs.shards[shardID] = shard
}

func (rs *RebalancingSystem) CheckRebalancingNeeded() bool {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()
    
    if len(rs.shards) < 2 {
        return false
    }
    
    // Calculate load for each shard
    loads := make(map[string]int)
    totalLoad := 0
    
    for shardID, shard := range rs.shards {
        shard.mutex.RLock()
        load := len(shard.Data)
        shard.mutex.RUnlock()
        
        loads[shardID] = load
        totalLoad += load
    }
    
    // Calculate average load
    avgLoad := float64(totalLoad) / float64(len(rs.shards))
    
    // Check if any shard is significantly imbalanced
    for _, load := range loads {
        imbalance := float64(load) - avgLoad
        if imbalance/avgLoad > rs.rebalancer.threshold {
            return true
        }
    }
    
    return false
}

func (rs *RebalancingSystem) CreateRebalancePlan() *RebalancePlan {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()
    
    plan := &RebalancePlan{
        Moves: make([]DataMove, 0),
    }
    
    // Calculate load for each shard
    loads := make(map[string]int)
    totalLoad := 0
    
    for shardID, shard := range rs.shards {
        shard.mutex.RLock()
        load := len(shard.Data)
        shard.mutex.RUnlock()
        
        loads[shardID] = load
        totalLoad += load
    }
    
    avgLoad := float64(totalLoad) / float64(len(rs.shards))
    
    // Find overloaded and underloaded shards
    var overloaded, underloaded []string
    
    for shardID, load := range loads {
        if float64(load) > avgLoad*1.2 {
            overloaded = append(overloaded, shardID)
        } else if float64(load) < avgLoad*0.8 {
            underloaded = append(underloaded, shardID)
        }
    }
    
    // Create moves from overloaded to underloaded shards
    for _, fromShard := range overloaded {
        for _, toShard := range underloaded {
            if len(plan.Moves) >= 10 { // Limit number of moves
                break
            }
            
            move := rs.createDataMove(fromShard, toShard, loads[fromShard], loads[toShard])
            if move != nil {
                plan.Moves = append(plan.Moves, *move)
            }
        }
    }
    
    return plan
}

func (rs *RebalancingSystem) createDataMove(fromShard, toShard string, fromLoad, toLoad int) *DataMove {
    // Calculate how much data to move
    targetLoad := (fromLoad + toLoad) / 2
    moveSize := fromLoad - targetLoad
    
    if moveSize <= 0 {
        return nil
    }
    
    // Get keys to move (simplified - in real implementation, this would be more sophisticated)
    shard := rs.shards[fromShard]
    shard.mutex.RLock()
    keys := make([]string, 0, moveSize)
    for key := range shard.Data {
        if len(keys) >= moveSize {
            break
        }
        keys = append(keys, key)
    }
    shard.mutex.RUnlock()
    
    return &DataMove{
        FromShard: fromShard,
        ToShard:   toShard,
        Keys:      keys,
        Size:      len(keys),
    }
}

func (rs *RebalancingSystem) ExecuteRebalancePlan(plan *RebalancePlan) error {
    for _, move := range plan.Moves {
        if err := rs.executeDataMove(move); err != nil {
            return fmt.Errorf("failed to move data from %s to %s: %v", move.FromShard, move.ToShard, err)
        }
    }
    
    return nil
}

func (rs *RebalancingSystem) executeDataMove(move DataMove) error {
    fromShard := rs.shards[move.FromShard]
    toShard := rs.shards[move.ToShard]
    
    // Move data atomically
    for _, key := range move.Keys {
        fromShard.mutex.Lock()
        value, exists := fromShard.Data[key]
        if !exists {
            fromShard.mutex.Unlock()
            continue
        }
        delete(fromShard.Data, key)
        fromShard.mutex.Unlock()
        
        toShard.mutex.Lock()
        toShard.Data[key] = value
        toShard.mutex.Unlock()
    }
    
    return nil
}

func (rs *RebalancingSystem) StartRebalancing() {
    ticker := time.NewTicker(5 * time.Minute)
    go func() {
        for range ticker.C {
            if rs.CheckRebalancingNeeded() {
                plan := rs.CreateRebalancePlan()
                if len(plan.Moves) > 0 {
                    fmt.Printf("Executing rebalance plan with %d moves\n", len(plan.Moves))
                    if err := rs.ExecuteRebalancePlan(plan); err != nil {
                        fmt.Printf("Rebalancing failed: %v\n", err)
                    } else {
                        fmt.Println("Rebalancing completed successfully")
                    }
                }
            }
        }
    }()
}

// Example usage
func main() {
    system := NewRebalancingSystem()
    
    // Add shards
    system.AddShard("shard1", &Shard{ID: "shard1", Data: make(map[string]interface{})})
    system.AddShard("shard2", &Shard{ID: "shard2", Data: make(map[string]interface{})})
    system.AddShard("shard3", &Shard{ID: "shard3", Data: make(map[string]interface{})})
    
    // Add some data to create imbalance
    shard1 := system.shards["shard1"]
    for i := 0; i < 100; i++ {
        shard1.Data[fmt.Sprintf("key%d", i)] = fmt.Sprintf("value%d", i)
    }
    
    // Check if rebalancing is needed
    if system.CheckRebalancingNeeded() {
        fmt.Println("Rebalancing needed")
        
        plan := system.CreateRebalancePlan()
        fmt.Printf("Rebalance plan: %+v\n", plan)
        
        if err := system.ExecuteRebalancePlan(plan); err != nil {
            fmt.Printf("Rebalancing failed: %v\n", err)
        } else {
            fmt.Println("Rebalancing completed")
        }
    } else {
        fmt.Println("No rebalancing needed")
    }
}
```

---

## 🎯 **Key Takeaways**

### **1. Sharding Strategies**
- **Range-based**: Good for sequential access and range queries
- **Hash-based**: Even distribution, no range queries
- **Directory-based**: Flexible, but single point of failure

### **2. Consistent Hashing**
- **Virtual nodes** for load balancing
- **Minimal data movement** when adding/removing nodes
- **Ring structure** for efficient lookups

### **3. Cross-Shard Operations**
- **Query aggregation** across multiple shards
- **Transaction coordination** for consistency
- **Result merging** for unified responses

### **4. Data Rebalancing**
- **Load monitoring** for imbalance detection
- **Migration planning** for data movement
- **Atomic operations** for consistency

### **5. Best Practices**
- **Choose sharding key carefully** for even distribution
- **Plan for cross-shard queries** and their performance impact
- **Implement monitoring** for shard health and load
- **Design for rebalancing** from the beginning

---

**🎉 This comprehensive guide provides deep insights into database sharding strategies with practical Go implementations! 🚀**
