---
# Auto-generated front matter
Title: Designing Data Intensive Applications Summary
LastUpdated: 2025-11-06T20:45:59.095685
Tags: []
Status: draft
---

# 📊 Designing Data Intensive Applications - Node.js Summary

> **Comprehensive summary of "Designing Data-Intensive Applications" with Node.js implementations**

## 🎯 **Overview**

This guide summarizes the key concepts from Martin Kleppmann's "Designing Data-Intensive Applications" book, adapted for Node.js developers. It covers reliability, scalability, maintainability, and the fundamental principles of building robust data systems.

## 📚 **Table of Contents**

1. [Reliability, Scalability, and Maintainability](#reliability-scalability-and-maintainability)
2. [Data Models and Query Languages](#data-models-and-query-languages)
3. [Storage and Retrieval](#storage-and-retrieval)
4. [Encoding and Evolution](#encoding-and-evolution)
5. [Replication](#replication)
6. [Partitioning](#partitioning)
7. [Transactions](#transactions)
8. [Consistency and Consensus](#consistency-and-consensus)
9. [Stream Processing](#stream-processing)
10. [Batch Processing](#batch-processing)

---

## 🛡️ **Reliability, Scalability, and Maintainability**

### **Reliability**

```javascript
// Reliability Patterns in Node.js
class ReliableService {
    constructor() {
        this.retryConfig = {
            maxRetries: 3,
            baseDelay: 1000,
            maxDelay: 10000,
            backoffFactor: 2
        };
        this.circuitBreaker = new CircuitBreaker();
    }
    
    async executeWithRetry(operation, context = {}) {
        let lastError;
        
        for (let attempt = 0; attempt <= this.retryConfig.maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error;
                
                if (!this.isRetryableError(error) || attempt === this.retryConfig.maxRetries) {
                    throw error;
                }
                
                const delay = this.calculateDelay(attempt);
                await this.sleep(delay);
            }
        }
        
        throw lastError;
    }
    
    isRetryableError(error) {
        const retryableCodes = ['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND'];
        return retryableCodes.includes(error.code) || error.status >= 500;
    }
    
    calculateDelay(attempt) {
        const delay = this.retryConfig.baseDelay * Math.pow(this.retryConfig.backoffFactor, attempt);
        return Math.min(delay, this.retryConfig.maxDelay);
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Circuit Breaker Pattern
class CircuitBreaker {
    constructor(options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.timeout = options.timeout || 60000;
        this.resetTimeout = options.resetTimeout || 30000;
        
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        this.failureCount = 0;
        this.lastFailureTime = null;
    }
    
    async execute(operation) {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.resetTimeout) {
                this.state = 'HALF_OPEN';
            } else {
                throw new Error('Circuit breaker is OPEN');
            }
        }
        
        try {
            const result = await operation();
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure();
            throw error;
        }
    }
    
    onSuccess() {
        this.failureCount = 0;
        this.state = 'CLOSED';
    }
    
    onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
        }
    }
}
```

### **Scalability**

```javascript
// Scalability Patterns
class ScalableArchitecture {
    constructor() {
        this.loadBalancer = new LoadBalancer();
        this.cache = new DistributedCache();
        this.database = new DatabaseCluster();
    }
    
    // Load Balancing
    async handleRequest(request) {
        const server = this.loadBalancer.selectServer();
        return await this.forwardRequest(server, request);
    }
    
    // Caching Strategy
    async getData(key) {
        // Try cache first
        let data = await this.cache.get(key);
        if (data) {
            return data;
        }
        
        // Fallback to database
        data = await this.database.get(key);
        
        // Cache for future requests
        await this.cache.set(key, data, 3600); // 1 hour TTL
        
        return data;
    }
    
    // Database Sharding
    async getShardedData(key) {
        const shard = this.getShardForKey(key);
        return await this.database.getFromShard(shard, key);
    }
    
    getShardForKey(key) {
        const hash = this.hash(key);
        return hash % this.database.shardCount;
    }
    
    hash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }
}
```

---

## 🗃️ **Data Models and Query Languages**

### **Relational vs Document Models**

```javascript
// Document Model Implementation
class DocumentStore {
    constructor() {
        this.documents = new Map();
    }
    
    async insert(collection, document) {
        const id = this.generateId();
        const doc = {
            _id: id,
            ...document,
            createdAt: new Date(),
            updatedAt: new Date()
        };
        
        if (!this.documents.has(collection)) {
            this.documents.set(collection, new Map());
        }
        
        this.documents.get(collection).set(id, doc);
        return doc;
    }
    
    async find(collection, query = {}) {
        const collectionDocs = this.documents.get(collection) || new Map();
        const results = [];
        
        for (const doc of collectionDocs.values()) {
            if (this.matchesQuery(doc, query)) {
                results.push(doc);
            }
        }
        
        return results;
    }
    
    matchesQuery(doc, query) {
        for (const [key, value] of Object.entries(query)) {
            if (doc[key] !== value) {
                return false;
            }
        }
        return true;
    }
    
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
}

// Graph Model Implementation
class GraphDatabase {
    constructor() {
        this.nodes = new Map();
        this.edges = new Map();
    }
    
    async addNode(id, properties = {}) {
        const node = {
            id,
            properties,
            createdAt: new Date()
        };
        
        this.nodes.set(id, node);
        return node;
    }
    
    async addEdge(from, to, relationship, properties = {}) {
        const edge = {
            from,
            to,
            relationship,
            properties,
            createdAt: new Date()
        };
        
        const edgeId = `${from}-${relationship}-${to}`;
        this.edges.set(edgeId, edge);
        
        return edge;
    }
    
    async findPath(start, end, maxDepth = 5) {
        const visited = new Set();
        const queue = [{ node: start, path: [start], depth: 0 }];
        
        while (queue.length > 0) {
            const { node, path, depth } = queue.shift();
            
            if (node === end) {
                return path;
            }
            
            if (depth >= maxDepth || visited.has(node)) {
                continue;
            }
            
            visited.add(node);
            
            // Find all connected nodes
            const connectedNodes = this.getConnectedNodes(node);
            
            for (const connectedNode of connectedNodes) {
                if (!visited.has(connectedNode)) {
                    queue.push({
                        node: connectedNode,
                        path: [...path, connectedNode],
                        depth: depth + 1
                    });
                }
            }
        }
        
        return null; // No path found
    }
    
    getConnectedNodes(nodeId) {
        const connected = [];
        
        for (const edge of this.edges.values()) {
            if (edge.from === nodeId) {
                connected.push(edge.to);
            } else if (edge.to === nodeId) {
                connected.push(edge.from);
            }
        }
        
        return connected;
    }
}
```

---

## 💾 **Storage and Retrieval**

### **LSM-Tree Implementation**

```javascript
// Log-Structured Merge Tree
class LSMTree {
    constructor() {
        this.memtable = new Map();
        this.sstables = [];
        this.maxMemtableSize = 1000;
    }
    
    async put(key, value) {
        this.memtable.set(key, { value, timestamp: Date.now() });
        
        if (this.memtable.size >= this.maxMemtableSize) {
            await this.flushMemtable();
        }
    }
    
    async get(key) {
        // Check memtable first
        if (this.memtable.has(key)) {
            return this.memtable.get(key).value;
        }
        
        // Check SSTables (newest first)
        for (let i = this.sstables.length - 1; i >= 0; i--) {
            const value = await this.sstables[i].get(key);
            if (value !== null) {
                return value;
            }
        }
        
        return null;
    }
    
    async flushMemtable() {
        const sstable = new SSTable(this.memtable);
        this.sstables.push(sstable);
        this.memtable.clear();
        
        // Compact if too many SSTables
        if (this.sstables.length > 3) {
            await this.compact();
        }
    }
    
    async compact() {
        // Merge and compact SSTables
        const merged = new Map();
        
        for (const sstable of this.sstables) {
            for (const [key, value] of sstable.entries()) {
                if (!merged.has(key) || merged.get(key).timestamp < value.timestamp) {
                    merged.set(key, value);
                }
            }
        }
        
        this.sstables = [new SSTable(merged)];
    }
}

class SSTable {
    constructor(data) {
        this.data = new Map(data);
        this.index = this.buildIndex();
    }
    
    buildIndex() {
        const index = new Map();
        let offset = 0;
        
        for (const [key, value] of this.data) {
            index.set(key, offset);
            offset += this.serializeEntry(key, value).length;
        }
        
        return index;
    }
    
    async get(key) {
        if (this.index.has(key)) {
            return this.data.get(key).value;
        }
        return null;
    }
    
    serializeEntry(key, value) {
        return JSON.stringify({ key, value }) + '\n';
    }
    
    *entries() {
        for (const [key, value] of this.data) {
            yield [key, value];
        }
    }
}
```

### **B-Tree Implementation**

```javascript
// B-Tree for Database Indexing
class BTreeNode {
    constructor(isLeaf = false) {
        this.keys = [];
        this.values = [];
        this.children = [];
        this.isLeaf = isLeaf;
        this.parent = null;
    }
    
    insert(key, value) {
        if (this.isLeaf) {
            this.insertInLeaf(key, value);
        } else {
            this.insertInInternal(key, value);
        }
    }
    
    insertInLeaf(key, value) {
        let index = 0;
        while (index < this.keys.length && this.keys[index] < key) {
            index++;
        }
        
        this.keys.splice(index, 0, key);
        this.values.splice(index, 0, value);
    }
    
    insertInInternal(key, value) {
        let index = 0;
        while (index < this.keys.length && this.keys[index] < key) {
            index++;
        }
        
        const child = this.children[index];
        child.insert(key, value);
        
        if (child.isOverflow()) {
            this.splitChild(index, child);
        }
    }
    
    isOverflow() {
        return this.keys.length > 2 * this.getMinKeys() - 1;
    }
    
    getMinKeys() {
        return 2; // Minimum degree
    }
    
    splitChild(index, child) {
        const newChild = new BTreeNode(child.isLeaf);
        const mid = Math.floor(child.keys.length / 2);
        
        // Move half of child's keys to new child
        newChild.keys = child.keys.splice(mid + 1);
        newChild.values = child.values.splice(mid + 1);
        
        if (!child.isLeaf) {
            newChild.children = child.children.splice(mid + 1);
        }
        
        // Insert middle key into this node
        this.keys.splice(index, 0, child.keys[mid]);
        this.values.splice(index, 0, child.values[mid]);
        this.children.splice(index + 1, 0, newChild);
    }
    
    search(key) {
        let index = 0;
        while (index < this.keys.length && this.keys[index] < key) {
            index++;
        }
        
        if (index < this.keys.length && this.keys[index] === key) {
            return this.values[index];
        }
        
        if (this.isLeaf) {
            return null;
        }
        
        return this.children[index].search(key);
    }
}

class BTree {
    constructor() {
        this.root = new BTreeNode(true);
    }
    
    insert(key, value) {
        this.root.insert(key, value);
        
        if (this.root.isOverflow()) {
            const newRoot = new BTreeNode(false);
            newRoot.children.push(this.root);
            newRoot.splitChild(0, this.root);
            this.root = newRoot;
        }
    }
    
    search(key) {
        return this.root.search(key);
    }
}
```

---

## 🔄 **Replication**

### **Leader-Follower Replication**

```javascript
// Leader-Follower Replication
class ReplicationManager {
    constructor() {
        this.leader = null;
        this.followers = [];
        this.replicationLag = new Map();
    }
    
    async write(data) {
        if (!this.leader) {
            throw new Error('No leader available');
        }
        
        // Write to leader
        const result = await this.leader.write(data);
        
        // Replicate to followers
        await this.replicateToFollowers(data);
        
        return result;
    }
    
    async read(key, consistencyLevel = 'eventual') {
        if (consistencyLevel === 'strong') {
            // Read from leader for strong consistency
            return await this.leader.read(key);
        } else {
            // Read from any available replica
            const replicas = [this.leader, ...this.followers].filter(r => r.isHealthy());
            const replica = this.selectReplica(replicas);
            return await replica.read(key);
        }
    }
    
    async replicateToFollowers(data) {
        const replicationPromises = this.followers.map(async (follower) => {
            try {
                await follower.replicate(data);
                this.replicationLag.set(follower.id, 0);
            } catch (error) {
                console.error(`Replication failed to follower ${follower.id}:`, error);
                this.replicationLag.set(follower.id, Date.now());
            }
        });
        
        await Promise.allSettled(replicationPromises);
    }
    
    selectReplica(replicas) {
        // Select replica with lowest lag
        let bestReplica = replicas[0];
        let lowestLag = this.replicationLag.get(bestReplica.id) || 0;
        
        for (const replica of replicas) {
            const lag = this.replicationLag.get(replica.id) || 0;
            if (lag < lowestLag) {
                bestReplica = replica;
                lowestLag = lag;
            }
        }
        
        return bestReplica;
    }
    
    async handleLeaderFailure() {
        // Elect new leader
        const newLeader = await this.electLeader();
        
        if (newLeader) {
            this.leader = newLeader;
            this.followers = this.followers.filter(f => f.id !== newLeader.id);
            console.log(`New leader elected: ${newLeader.id}`);
        } else {
            throw new Error('No suitable leader found');
        }
    }
    
    async electLeader() {
        // Simple leader election - in practice, use Raft or similar
        const candidates = this.followers.filter(f => f.isHealthy());
        
        if (candidates.length === 0) {
            return null;
        }
        
        // Select candidate with highest priority (lowest lag)
        return candidates.reduce((best, current) => {
            const bestLag = this.replicationLag.get(best.id) || Infinity;
            const currentLag = this.replicationLag.get(current.id) || Infinity;
            return currentLag < bestLag ? current : best;
        });
    }
}
```

---

## 🔀 **Partitioning**

### **Consistent Hashing**

```javascript
// Consistent Hashing for Partitioning
class ConsistentHash {
    constructor(nodes = [], replicas = 3) {
        this.replicas = replicas;
        this.ring = new Map();
        this.nodes = new Set();
        
        nodes.forEach(node => this.addNode(node));
    }
    
    addNode(node) {
        this.nodes.add(node);
        
        for (let i = 0; i < this.replicas; i++) {
            const hash = this.hash(`${node}:${i}`);
            this.ring.set(hash, node);
        }
        
        this.sortRing();
    }
    
    removeNode(node) {
        this.nodes.delete(node);
        
        for (let i = 0; i < this.replicas; i++) {
            const hash = this.hash(`${node}:${i}`);
            this.ring.delete(hash);
        }
        
        this.sortRing();
    }
    
    getNode(key) {
        if (this.ring.size === 0) {
            return null;
        }
        
        const hash = this.hash(key);
        const keys = Array.from(this.ring.keys()).sort((a, b) => a - b);
        
        for (const ringKey of keys) {
            if (hash <= ringKey) {
                return this.ring.get(ringKey);
            }
        }
        
        // Wrap around to first node
        return this.ring.get(keys[0]);
    }
    
    hash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }
    
    sortRing() {
        // Ring is automatically sorted when we iterate over keys
    }
    
    getNodesForKey(key, count = 1) {
        const nodes = new Set();
        const hash = this.hash(key);
        const keys = Array.from(this.ring.keys()).sort((a, b) => a - b);
        
        let startIndex = 0;
        for (let i = 0; i < keys.length; i++) {
            if (hash <= keys[i]) {
                startIndex = i;
                break;
            }
        }
        
        for (let i = 0; i < keys.length && nodes.size < count; i++) {
            const index = (startIndex + i) % keys.length;
            const node = this.ring.get(keys[index]);
            nodes.add(node);
        }
        
        return Array.from(nodes);
    }
}

// Partitioned Database
class PartitionedDatabase {
    constructor(partitions = 4) {
        this.partitions = [];
        this.hash = new ConsistentHash();
        
        for (let i = 0; i < partitions; i++) {
            const partition = new DatabasePartition(`partition-${i}`);
            this.partitions.push(partition);
            this.hash.addNode(partition.id);
        }
    }
    
    async write(key, value) {
        const partition = this.getPartition(key);
        return await partition.write(key, value);
    }
    
    async read(key) {
        const partition = this.getPartition(key);
        return await partition.read(key);
    }
    
    getPartition(key) {
        const nodeId = this.hash.getNode(key);
        return this.partitions.find(p => p.id === nodeId);
    }
    
    async rebalance() {
        // Implement partition rebalancing logic
        const data = await this.getAllData();
        const newPartitions = this.calculateOptimalPartitions();
        
        // Redistribute data
        for (const [key, value] of data) {
            const newPartition = this.getPartition(key);
            await newPartition.write(key, value);
        }
    }
    
    async getAllData() {
        const allData = new Map();
        
        for (const partition of this.partitions) {
            const partitionData = await partition.getAllData();
            for (const [key, value] of partitionData) {
                allData.set(key, value);
            }
        }
        
        return allData;
    }
    
    calculateOptimalPartitions() {
        // Calculate optimal partition distribution
        // This is a simplified implementation
        return this.partitions;
    }
}

class DatabasePartition {
    constructor(id) {
        this.id = id;
        this.data = new Map();
    }
    
    async write(key, value) {
        this.data.set(key, value);
        return { success: true, partition: this.id };
    }
    
    async read(key) {
        return this.data.get(key) || null;
    }
    
    async getAllData() {
        return new Map(this.data);
    }
}
```

---

## 🔒 **Transactions**

### **ACID Transactions**

```javascript
// ACID Transaction Implementation
class TransactionManager {
    constructor() {
        this.activeTransactions = new Map();
        this.lockManager = new LockManager();
        this.logManager = new LogManager();
    }
    
    async beginTransaction() {
        const transactionId = this.generateTransactionId();
        const transaction = {
            id: transactionId,
            startTime: Date.now(),
            status: 'active',
            operations: [],
            locks: new Set()
        };
        
        this.activeTransactions.set(transactionId, transaction);
        return transactionId;
    }
    
    async commit(transactionId) {
        const transaction = this.activeTransactions.get(transactionId);
        if (!transaction) {
            throw new Error('Transaction not found');
        }
        
        try {
            // Write-ahead logging
            await this.logManager.logCommit(transactionId);
            
            // Execute all operations
            for (const operation of transaction.operations) {
                await this.executeOperation(operation);
            }
            
            // Release locks
            await this.releaseLocks(transactionId);
            
            transaction.status = 'committed';
            this.activeTransactions.delete(transactionId);
            
            return { success: true };
        } catch (error) {
            await this.rollback(transactionId);
            throw error;
        }
    }
    
    async rollback(transactionId) {
        const transaction = this.activeTransactions.get(transactionId);
        if (!transaction) {
            throw new Error('Transaction not found');
        }
        
        try {
            // Undo all operations
            for (let i = transaction.operations.length - 1; i >= 0; i--) {
                await this.undoOperation(transaction.operations[i]);
            }
            
            // Release locks
            await this.releaseLocks(transactionId);
            
            transaction.status = 'rolled_back';
            this.activeTransactions.delete(transactionId);
            
            return { success: true };
        } catch (error) {
            console.error('Rollback failed:', error);
            throw error;
        }
    }
    
    async executeOperation(operation) {
        switch (operation.type) {
            case 'read':
                return await this.read(operation.key);
            case 'write':
                return await this.write(operation.key, operation.value);
            case 'delete':
                return await this.delete(operation.key);
            default:
                throw new Error(`Unknown operation type: ${operation.type}`);
        }
    }
    
    async undoOperation(operation) {
        // Implement undo logic based on operation type
        // This is a simplified implementation
        console.log(`Undoing operation: ${operation.type} on ${operation.key}`);
    }
    
    async releaseLocks(transactionId) {
        const transaction = this.activeTransactions.get(transactionId);
        if (transaction) {
            for (const lock of transaction.locks) {
                await this.lockManager.releaseLock(lock, transactionId);
            }
        }
    }
    
    generateTransactionId() {
        return `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

class LockManager {
    constructor() {
        this.locks = new Map(); // key -> { type, holder, waiters }
    }
    
    async acquireLock(key, transactionId, lockType = 'exclusive') {
        const lock = this.locks.get(key);
        
        if (!lock) {
            // No existing lock
            this.locks.set(key, {
                type: lockType,
                holder: transactionId,
                waiters: []
            });
            return true;
        }
        
        if (lock.holder === transactionId) {
            // Already holding the lock
            return true;
        }
        
        if (this.canGrantLock(lock, lockType)) {
            // Can grant lock immediately
            lock.holder = transactionId;
            lock.type = lockType;
            return true;
        } else {
            // Must wait
            lock.waiters.push({ transactionId, lockType });
            return false;
        }
    }
    
    async releaseLock(key, transactionId) {
        const lock = this.locks.get(key);
        
        if (!lock || lock.holder !== transactionId) {
            return false;
        }
        
        if (lock.waiters.length === 0) {
            this.locks.delete(key);
        } else {
            // Grant lock to next waiter
            const nextWaiter = lock.waiters.shift();
            lock.holder = nextWaiter.transactionId;
            lock.type = nextWaiter.lockType;
        }
        
        return true;
    }
    
    canGrantLock(lock, requestedType) {
        if (lock.type === 'shared' && requestedType === 'shared') {
            return true;
        }
        
        if (lock.type === 'exclusive' || requestedType === 'exclusive') {
            return false;
        }
        
        return true;
    }
}

class LogManager {
    constructor() {
        this.log = [];
    }
    
    async logCommit(transactionId) {
        const logEntry = {
            type: 'commit',
            transactionId,
            timestamp: Date.now()
        };
        
        this.log.push(logEntry);
        await this.flushLog();
    }
    
    async logOperation(transactionId, operation) {
        const logEntry = {
            type: 'operation',
            transactionId,
            operation,
            timestamp: Date.now()
        };
        
        this.log.push(logEntry);
        await this.flushLog();
    }
    
    async flushLog() {
        // In a real implementation, this would write to persistent storage
        console.log('Log flushed to disk');
    }
}
```

---

## 🎯 **Key Takeaways**

### **Reliability**
- Implement retry mechanisms with exponential backoff
- Use circuit breakers to prevent cascade failures
- Design for graceful degradation

### **Scalability**
- Use load balancing and caching strategies
- Implement proper partitioning and sharding
- Design for horizontal scaling

### **Data Models**
- Choose appropriate data models for your use case
- Consider the trade-offs between relational and document models
- Use graph models for complex relationships

### **Storage**
- Understand the trade-offs between different storage engines
- Use appropriate indexing strategies
- Implement proper data compression and encoding

### **Replication**
- Implement leader-follower replication for read scaling
- Handle replication lag and consistency
- Design for failover and recovery

### **Partitioning**
- Use consistent hashing for even distribution
- Implement proper rebalancing strategies
- Handle partition failures gracefully

### **Transactions**
- Implement ACID properties correctly
- Use appropriate locking mechanisms
- Design for deadlock prevention

---

**🎉 This comprehensive summary covers all key concepts from "Designing Data-Intensive Applications" with Node.js implementations!**


## Encoding And Evolution

<!-- AUTO-GENERATED ANCHOR: originally referenced as #encoding-and-evolution -->

Placeholder content. Please replace with proper section.


## Consistency And Consensus

<!-- AUTO-GENERATED ANCHOR: originally referenced as #consistency-and-consensus -->

Placeholder content. Please replace with proper section.


## Stream Processing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #stream-processing -->

Placeholder content. Please replace with proper section.


## Batch Processing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #batch-processing -->

Placeholder content. Please replace with proper section.
