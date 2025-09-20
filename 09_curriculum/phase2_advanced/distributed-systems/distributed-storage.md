# Distributed Storage

## Overview

This module covers distributed storage systems including consistent hashing, distributed hash tables (DHT), replication strategies, and data partitioning. These concepts are essential for building scalable and reliable storage systems.

## Table of Contents

1. [Consistent Hashing](#consistent-hashing)
2. [Distributed Hash Tables (DHT)](#distributed-hash-tables-dht)
3. [Replication Strategies](#replication-strategies)
4. [Data Partitioning](#data-partitioning)
5. [CAP Theorem](#cap-theorem)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Consistent Hashing

### Theory

Consistent hashing is a distributed hashing scheme that minimizes the number of keys that need to be remapped when nodes are added or removed from the system. It provides load balancing and fault tolerance.

### Key Properties

- **Load Balancing**: Keys are distributed evenly across nodes
- **Scalability**: Adding/removing nodes affects only a small number of keys
- **Fault Tolerance**: System continues to work when nodes fail

### Implementations

#### Golang Implementation

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "strconv"
)

type Node struct {
    ID   string
    Hash uint32
}

type ConsistentHash struct {
    nodes     []Node
    replicas  int
    hashRing  map[uint32]string
    sortedKeys []uint32
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        nodes:     make([]Node, 0),
        replicas:  replicas,
        hashRing:  make(map[uint32]string),
        sortedKeys: make([]uint32, 0),
    }
}

func (ch *ConsistentHash) hashKey(key string) uint32 {
    h := md5.New()
    h.Write([]byte(key))
    hash := h.Sum(nil)
    return uint32(hash[0])<<24 | uint32(hash[1])<<16 | uint32(hash[2])<<8 | uint32(hash[3])
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    for i := 0; i < ch.replicas; i++ {
        virtualNode := fmt.Sprintf("%s:%d", nodeID, i)
        hash := ch.hashKey(virtualNode)
        
        ch.hashRing[hash] = nodeID
        ch.sortedKeys = append(ch.sortedKeys, hash)
    }
    
    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
    
    ch.nodes = append(ch.nodes, Node{ID: nodeID, Hash: ch.hashKey(nodeID)})
    fmt.Printf("Added node %s with %d replicas\n", nodeID, ch.replicas)
}

func (ch *ConsistentHash) RemoveNode(nodeID string) {
    // Remove all virtual nodes for this node
    for i := 0; i < ch.replicas; i++ {
        virtualNode := fmt.Sprintf("%s:%d", nodeID, i)
        hash := ch.hashKey(virtualNode)
        
        delete(ch.hashRing, hash)
        
        // Remove from sorted keys
        for j, key := range ch.sortedKeys {
            if key == hash {
                ch.sortedKeys = append(ch.sortedKeys[:j], ch.sortedKeys[j+1:]...)
                break
            }
        }
    }
    
    // Remove from nodes list
    for i, node := range ch.nodes {
        if node.ID == nodeID {
            ch.nodes = append(ch.nodes[:i], ch.nodes[i+1:]...)
            break
        }
    }
    
    fmt.Printf("Removed node %s\n", nodeID)
}

func (ch *ConsistentHash) GetNode(key string) string {
    if len(ch.sortedKeys) == 0 {
        return ""
    }
    
    hash := ch.hashKey(key)
    
    // Find the first node with hash >= key hash
    for _, nodeHash := range ch.sortedKeys {
        if nodeHash >= hash {
            return ch.hashRing[nodeHash]
        }
    }
    
    // Wrap around to the first node
    return ch.hashRing[ch.sortedKeys[0]]
}

func (ch *ConsistentHash) GetNodes(key string, count int) []string {
    if len(ch.sortedKeys) == 0 {
        return []string{}
    }
    
    hash := ch.hashKey(key)
    nodes := make([]string, 0, count)
    seen := make(map[string]bool)
    
    // Find the first node with hash >= key hash
    startIndex := -1
    for i, nodeHash := range ch.sortedKeys {
        if nodeHash >= hash {
            startIndex = i
            break
        }
    }
    
    if startIndex == -1 {
        startIndex = 0
    }
    
    // Collect unique nodes
    for i := 0; i < len(ch.sortedKeys) && len(nodes) < count; i++ {
        index := (startIndex + i) % len(ch.sortedKeys)
        nodeHash := ch.sortedKeys[index]
        nodeID := ch.hashRing[nodeHash]
        
        if !seen[nodeID] {
            nodes = append(nodes, nodeID)
            seen[nodeID] = true
        }
    }
    
    return nodes
}

func (ch *ConsistentHash) PrintRing() {
    fmt.Println("Hash Ring:")
    for _, key := range ch.sortedKeys {
        fmt.Printf("Hash: %d, Node: %s\n", key, ch.hashRing[key])
    }
}

func main() {
    ch := NewConsistentHash(3) // 3 replicas per node
    
    // Add nodes
    ch.AddNode("node1")
    ch.AddNode("node2")
    ch.AddNode("node3")
    
    ch.PrintRing()
    
    // Test key distribution
    keys := []string{"key1", "key2", "key3", "key4", "key5"}
    fmt.Println("\nKey distribution:")
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key '%s' -> Node '%s'\n", key, node)
    }
    
    // Test multiple nodes for a key
    fmt.Println("\nMultiple nodes for 'key1':")
    nodes := ch.GetNodes("key1", 2)
    for i, node := range nodes {
        fmt.Printf("Replica %d: %s\n", i+1, node)
    }
    
    // Remove a node and test redistribution
    fmt.Println("\nAfter removing node2:")
    ch.RemoveNode("node2")
    
    for _, key := range keys {
        node := ch.GetNode(key)
        fmt.Printf("Key '%s' -> Node '%s'\n", key, node)
    }
}
```

#### Node.js Implementation

```javascript
const crypto = require('crypto');

class ConsistentHash {
    constructor(replicas = 3) {
        this.nodes = [];
        this.replicas = replicas;
        this.hashRing = new Map();
        this.sortedKeys = [];
    }

    hashKey(key) {
        const hash = crypto.createHash('md5').update(key).digest('hex');
        return parseInt(hash.substring(0, 8), 16);
    }

    addNode(nodeID) {
        for (let i = 0; i < this.replicas; i++) {
            const virtualNode = `${nodeID}:${i}`;
            const hash = this.hashKey(virtualNode);
            
            this.hashRing.set(hash, nodeID);
            this.sortedKeys.push(hash);
        }
        
        this.sortedKeys.sort((a, b) => a - b);
        this.nodes.push({ id: nodeID, hash: this.hashKey(nodeID) });
        console.log(`Added node ${nodeID} with ${this.replicas} replicas`);
    }

    removeNode(nodeID) {
        // Remove all virtual nodes for this node
        for (let i = 0; i < this.replicas; i++) {
            const virtualNode = `${nodeID}:${i}`;
            const hash = this.hashKey(virtualNode);
            
            this.hashRing.delete(hash);
            const index = this.sortedKeys.indexOf(hash);
            if (index > -1) {
                this.sortedKeys.splice(index, 1);
            }
        }
        
        // Remove from nodes list
        this.nodes = this.nodes.filter(node => node.id !== nodeID);
        console.log(`Removed node ${nodeID}`);
    }

    getNode(key) {
        if (this.sortedKeys.length === 0) {
            return '';
        }
        
        const hash = this.hashKey(key);
        
        // Find the first node with hash >= key hash
        for (const nodeHash of this.sortedKeys) {
            if (nodeHash >= hash) {
                return this.hashRing.get(nodeHash);
            }
        }
        
        // Wrap around to the first node
        return this.hashRing.get(this.sortedKeys[0]);
    }

    getNodes(key, count) {
        if (this.sortedKeys.length === 0) {
            return [];
        }
        
        const hash = this.hashKey(key);
        const nodes = [];
        const seen = new Set();
        
        // Find the first node with hash >= key hash
        let startIndex = -1;
        for (let i = 0; i < this.sortedKeys.length; i++) {
            if (this.sortedKeys[i] >= hash) {
                startIndex = i;
                break;
            }
        }
        
        if (startIndex === -1) {
            startIndex = 0;
        }
        
        // Collect unique nodes
        for (let i = 0; i < this.sortedKeys.length && nodes.length < count; i++) {
            const index = (startIndex + i) % this.sortedKeys.length;
            const nodeHash = this.sortedKeys[index];
            const nodeID = this.hashRing.get(nodeHash);
            
            if (!seen.has(nodeID)) {
                nodes.push(nodeID);
                seen.add(nodeID);
            }
        }
        
        return nodes;
    }

    printRing() {
        console.log('Hash Ring:');
        for (const key of this.sortedKeys) {
            console.log(`Hash: ${key}, Node: ${this.hashRing.get(key)}`);
        }
    }
}

// Example usage
const ch = new ConsistentHash(3);

// Add nodes
ch.addNode('node1');
ch.addNode('node2');
ch.addNode('node3');

ch.printRing();

// Test key distribution
const keys = ['key1', 'key2', 'key3', 'key4', 'key5'];
console.log('\nKey distribution:');
keys.forEach(key => {
    const node = ch.getNode(key);
    console.log(`Key '${key}' -> Node '${node}'`);
});

// Test multiple nodes for a key
console.log('\nMultiple nodes for "key1":');
const nodes = ch.getNodes('key1', 2);
nodes.forEach((node, i) => {
    console.log(`Replica ${i + 1}: ${node}`);
});

// Remove a node and test redistribution
console.log('\nAfter removing node2:');
ch.removeNode('node2');

keys.forEach(key => {
    const node = ch.getNode(key);
    console.log(`Key '${key}' -> Node '${node}'`);
});
```

## Distributed Hash Tables (DHT)

### Theory

A Distributed Hash Table (DHT) is a distributed system that provides a lookup service similar to a hash table, where (key, value) pairs are stored in a distributed network of nodes.

### Key Properties

- **Decentralized**: No central coordinator
- **Scalable**: Can handle large numbers of nodes
- **Fault Tolerant**: Continues to work when nodes fail

### Implementations

#### Golang Implementation

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
)

type DHTNode struct {
    ID       string
    Hash     uint32
    Data     map[string]string
    Successor *DHTNode
    Predecessor *DHTNode
    Finger   []*DHTNode
    mutex    sync.RWMutex
}

type DHT struct {
    nodes map[string]*DHTNode
    mutex sync.RWMutex
}

func NewDHT() *DHT {
    return &DHT{
        nodes: make(map[string]*DHTNode),
    }
}

func (dht *DHT) hashKey(key string) uint32 {
    h := md5.New()
    h.Write([]byte(key))
    hash := h.Sum(nil)
    return uint32(hash[0])<<24 | uint32(hash[1])<<16 | uint32(hash[2])<<8 | uint32(hash[3])
}

func (dht *DHT) AddNode(nodeID string) *DHTNode {
    dht.mutex.Lock()
    defer dht.mutex.Unlock()
    
    node := &DHTNode{
        ID:     nodeID,
        Hash:   dht.hashKey(nodeID),
        Data:   make(map[string]string),
        Finger: make([]*DHTNode, 32), // 32-bit hash space
    }
    
    dht.nodes[nodeID] = node
    
    // Update finger table
    dht.updateFingerTable(node)
    
    // Update successor and predecessor
    dht.updateSuccessorPredecessor(node)
    
    fmt.Printf("Added node %s with hash %d\n", nodeID, node.Hash)
    return node
}

func (dht *DHT) updateFingerTable(node *DHTNode) {
    for i := 0; i < 32; i++ {
        target := (node.Hash + (1 << i)) % (1 << 32)
        node.Finger[i] = dht.findSuccessor(target)
    }
}

func (dht *DHT) findSuccessor(hash uint32) *DHTNode {
    if len(dht.nodes) == 0 {
        return nil
    }
    
    // Find the node with the smallest hash >= target hash
    var closest *DHTNode
    minDistance := uint32(^uint32(0))
    
    for _, node := range dht.nodes {
        if node.Hash >= hash {
            distance := node.Hash - hash
            if distance < minDistance {
                minDistance = distance
                closest = node
            }
        }
    }
    
    // If no node found with hash >= target, wrap around
    if closest == nil {
        var minHash uint32 = ^uint32(0)
        for _, node := range dht.nodes {
            if node.Hash < minHash {
                minHash = node.Hash
                closest = node
            }
        }
    }
    
    return closest
}

func (dht *DHT) updateSuccessorPredecessor(node *DHTNode) {
    if len(dht.nodes) <= 1 {
        node.Successor = node
        node.Predecessor = node
        return
    }
    
    // Find successor
    node.Successor = dht.findSuccessor(node.Hash + 1)
    
    // Find predecessor
    var predecessor *DHTNode
    maxHash := uint32(0)
    
    for _, n := range dht.nodes {
        if n != node && n.Hash < node.Hash && n.Hash > maxHash {
            maxHash = n.Hash
            predecessor = n
        }
    }
    
    if predecessor == nil {
        // Wrap around
        for _, n := range dht.nodes {
            if n != node && n.Hash > maxHash {
                maxHash = n.Hash
                predecessor = n
            }
        }
    }
    
    node.Predecessor = predecessor
}

func (dht *DHT) Put(key, value string) {
    hash := dht.hashKey(key)
    node := dht.findSuccessor(hash)
    
    if node != nil {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
        fmt.Printf("Stored key '%s' with value '%s' on node %s\n", key, value, node.ID)
    }
}

func (dht *DHT) Get(key string) (string, bool) {
    hash := dht.hashKey(key)
    node := dht.findSuccessor(hash)
    
    if node != nil {
        node.mutex.RLock()
        value, exists := node.Data[key]
        node.mutex.RUnlock()
        
        if exists {
            fmt.Printf("Retrieved key '%s' with value '%s' from node %s\n", key, value, node.ID)
        } else {
            fmt.Printf("Key '%s' not found on node %s\n", key, node.ID)
        }
        
        return value, exists
    }
    
    return "", false
}

func (dht *DHT) RemoveNode(nodeID string) {
    dht.mutex.Lock()
    defer dht.mutex.Unlock()
    
    node, exists := dht.nodes[nodeID]
    if !exists {
        return
    }
    
    // Transfer data to successor
    if node.Successor != nil && node.Successor != node {
        node.mutex.Lock()
        for key, value := range node.Data {
            node.Successor.mutex.Lock()
            node.Successor.Data[key] = value
            node.Successor.mutex.Unlock()
        }
        node.mutex.Unlock()
    }
    
    // Update successor and predecessor pointers
    if node.Predecessor != nil {
        node.Predecessor.Successor = node.Successor
    }
    if node.Successor != nil {
        node.Successor.Predecessor = node.Predecessor
    }
    
    delete(dht.nodes, nodeID)
    fmt.Printf("Removed node %s\n", nodeID)
}

func (dht *DHT) PrintRing() {
    dht.mutex.RLock()
    defer dht.mutex.RUnlock()
    
    fmt.Println("DHT Ring:")
    for _, node := range dht.nodes {
        fmt.Printf("Node %s (hash: %d) -> Successor: %s, Predecessor: %s\n", 
                   node.ID, node.Hash, 
                   node.Successor.ID, node.Predecessor.ID)
    }
}

func main() {
    dht := NewDHT()
    
    // Add nodes
    dht.AddNode("node1")
    dht.AddNode("node2")
    dht.AddNode("node3")
    
    dht.PrintRing()
    
    // Store some data
    dht.Put("key1", "value1")
    dht.Put("key2", "value2")
    dht.Put("key3", "value3")
    
    // Retrieve data
    dht.Get("key1")
    dht.Get("key2")
    dht.Get("key3")
    
    // Remove a node
    dht.RemoveNode("node2")
    dht.PrintRing()
}
```

## Replication Strategies

### Theory

Replication strategies determine how data is copied across multiple nodes to ensure availability and fault tolerance.

### Common Strategies

1. **Master-Slave**: One master, multiple slaves
2. **Master-Master**: Multiple masters, bidirectional replication
3. **Chain Replication**: Data flows through a chain of nodes
4. **Quorum-based**: Requires majority agreement

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ReplicationStrategy interface {
    Write(key, value string) error
    Read(key string) (string, error)
    AddReplica(nodeID string)
    RemoveReplica(nodeID string)
}

type MasterSlaveReplication struct {
    master   *Node
    slaves   []*Node
    mutex    sync.RWMutex
}

type Node struct {
    ID   string
    Data map[string]string
    mutex sync.RWMutex
}

func NewNode(id string) *Node {
    return &Node{
        ID:   id,
        Data: make(map[string]string),
    }
}

func (n *Node) Write(key, value string) {
    n.mutex.Lock()
    defer n.mutex.Unlock()
    n.Data[key] = value
}

func (n *Node) Read(key string) (string, bool) {
    n.mutex.RLock()
    defer n.mutex.RUnlock()
    value, exists := n.Data[key]
    return value, exists
}

func NewMasterSlaveReplication(masterID string) *MasterSlaveReplication {
    return &MasterSlaveReplication{
        master: NewNode(masterID),
        slaves: make([]*Node, 0),
    }
}

func (ms *MasterSlaveReplication) Write(key, value string) error {
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    
    // Write to master
    ms.master.Write(key, value)
    fmt.Printf("Master %s: Wrote key '%s' with value '%s'\n", ms.master.ID, key, value)
    
    // Replicate to slaves
    for _, slave := range ms.slaves {
        slave.Write(key, value)
        fmt.Printf("Slave %s: Replicated key '%s' with value '%s'\n", slave.ID, key, value)
    }
    
    return nil
}

func (ms *MasterSlaveReplication) Read(key string) (string, error) {
    ms.mutex.RLock()
    defer ms.mutex.RUnlock()
    
    // Read from master
    value, exists := ms.master.Read(key)
    if exists {
        fmt.Printf("Master %s: Read key '%s' with value '%s'\n", ms.master.ID, key, value)
        return value, nil
    }
    
    return "", fmt.Errorf("key '%s' not found", key)
}

func (ms *MasterSlaveReplication) AddReplica(nodeID string) {
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    
    slave := NewNode(nodeID)
    ms.slaves = append(ms.slaves, slave)
    fmt.Printf("Added slave replica: %s\n", nodeID)
}

func (ms *MasterSlaveReplication) RemoveReplica(nodeID string) {
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    
    for i, slave := range ms.slaves {
        if slave.ID == nodeID {
            ms.slaves = append(ms.slaves[:i], ms.slaves[i+1:]...)
            fmt.Printf("Removed slave replica: %s\n", nodeID)
            break
        }
    }
}

type QuorumReplication struct {
    nodes   []*Node
    quorum  int
    mutex   sync.RWMutex
}

func NewQuorumReplication(nodeIDs []string, quorum int) *QuorumReplication {
    nodes := make([]*Node, len(nodeIDs))
    for i, id := range nodeIDs {
        nodes[i] = NewNode(id)
    }
    
    return &QuorumReplication{
        nodes:  nodes,
        quorum: quorum,
    }
}

func (qr *QuorumReplication) Write(key, value string) error {
    qr.mutex.Lock()
    defer qr.mutex.Unlock()
    
    // Write to all nodes
    for _, node := range qr.nodes {
        node.Write(key, value)
    }
    
    fmt.Printf("Quorum write: Wrote key '%s' with value '%s' to all %d nodes\n", 
               key, value, len(qr.nodes))
    
    return nil
}

func (qr *QuorumReplication) Read(key string) (string, error) {
    qr.mutex.RLock()
    defer qr.mutex.RUnlock()
    
    // Read from quorum number of nodes
    values := make(map[string]int)
    
    for _, node := range qr.nodes {
        if value, exists := node.Read(key); exists {
            values[value]++
        }
    }
    
    // Find the value with the most votes
    var bestValue string
    maxCount := 0
    
    for value, count := range values {
        if count > maxCount {
            maxCount = count
            bestValue = value
        }
    }
    
    if maxCount >= qr.quorum {
        fmt.Printf("Quorum read: Read key '%s' with value '%s' (votes: %d)\n", 
                   key, bestValue, maxCount)
        return bestValue, nil
    }
    
    return "", fmt.Errorf("quorum not reached for key '%s'", key)
}

func (qr *QuorumReplication) AddReplica(nodeID string) {
    qr.mutex.Lock()
    defer qr.mutex.Unlock()
    
    node := NewNode(nodeID)
    qr.nodes = append(qr.nodes, node)
    fmt.Printf("Added quorum replica: %s\n", nodeID)
}

func (qr *QuorumReplication) RemoveReplica(nodeID string) {
    qr.mutex.Lock()
    defer qr.mutex.Unlock()
    
    for i, node := range qr.nodes {
        if node.ID == nodeID {
            qr.nodes = append(qr.nodes[:i], qr.nodes[i+1:]...)
            fmt.Printf("Removed quorum replica: %s\n", nodeID)
            break
        }
    }
}

func main() {
    // Test Master-Slave replication
    fmt.Println("=== Master-Slave Replication ===")
    ms := NewMasterSlaveReplication("master1")
    ms.AddReplica("slave1")
    ms.AddReplica("slave2")
    
    ms.Write("key1", "value1")
    ms.Write("key2", "value2")
    
    ms.Read("key1")
    ms.Read("key2")
    
    // Test Quorum replication
    fmt.Println("\n=== Quorum Replication ===")
    qr := NewQuorumReplication([]string{"node1", "node2", "node3", "node4", "node5"}, 3)
    
    qr.Write("key1", "value1")
    qr.Write("key2", "value2")
    
    qr.Read("key1")
    qr.Read("key2")
}
```

## Follow-up Questions

### 1. Consistent Hashing
**Q: How do you handle node failures in consistent hashing?**
A: When a node fails, its virtual nodes are removed from the hash ring, and the keys that were mapped to that node are redistributed to the next available node. The system continues to work, but some keys may be temporarily unavailable until the failed node is replaced.

### 2. DHT Design
**Q: What are the trade-offs between different DHT topologies?**
A: Ring topology is simple but has O(n) lookup time. Tree-based topologies like Chord provide O(log n) lookup time but are more complex. The choice depends on the expected number of nodes and performance requirements.

### 3. Replication Strategies
**Q: When would you choose quorum-based replication over master-slave?**
A: Choose quorum-based replication when you need high availability and can tolerate some inconsistency. Choose master-slave when you need strong consistency and can accept single points of failure.

## Complexity Analysis

| Strategy | Read Complexity | Write Complexity | Consistency | Availability |
|----------|----------------|------------------|-------------|--------------|
| Consistent Hashing | O(log n) | O(log n) | Eventual | High |
| DHT | O(log n) | O(log n) | Eventual | High |
| Master-Slave | O(1) | O(n) | Strong | Medium |
| Quorum | O(n) | O(n) | Eventual | High |

## Applications

1. **Consistent Hashing**: Load balancers, distributed caches
2. **DHT**: Peer-to-peer networks, distributed file systems
3. **Replication**: Databases, distributed storage systems

---

**Next**: [Event Sourcing](event-sourcing.md) | **Previous**: [Distributed Systems](README.md) | **Up**: [Distributed Systems](README.md)


## Data Partitioning

<!-- AUTO-GENERATED ANCHOR: originally referenced as #data-partitioning -->

Placeholder content. Please replace with proper section.


## Cap Theorem

<!-- AUTO-GENERATED ANCHOR: originally referenced as #cap-theorem -->

Placeholder content. Please replace with proper section.
