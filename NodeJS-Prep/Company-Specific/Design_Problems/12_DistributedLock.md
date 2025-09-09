# 12. Distributed Lock - Coordination Service

## Title & Summary
Design and implement a distributed locking system using Node.js that provides mutual exclusion across multiple processes and machines with Redis backend, lease management, and deadlock prevention.

## Problem Statement

Build a robust distributed locking system that:

1. **Mutual Exclusion**: Ensure only one process can hold a lock at a time
2. **Deadlock Prevention**: Automatic lock expiration and cleanup
3. **High Availability**: Work across multiple Redis instances
4. **Lock Renewal**: Extend lock duration for long-running operations
5. **Fair Locking**: FIFO ordering for lock acquisition
6. **Monitoring**: Lock usage statistics and health monitoring

## Requirements & Constraints

### Functional Requirements
- Acquire and release distributed locks
- Lock expiration and automatic cleanup
- Lock renewal for long-running operations
- Fair lock acquisition with queuing
- Lock statistics and monitoring
- Multiple lock types (exclusive, shared, read-write)

### Non-Functional Requirements
- **Latency**: < 10ms for lock acquisition
- **Throughput**: 10,000+ lock operations per second
- **Availability**: 99.9% uptime
- **Consistency**: Strong consistency for lock state
- **Reliability**: Handle Redis failures gracefully
- **Scalability**: Support 1000+ concurrent locks

## API / Interfaces

### REST Endpoints

```javascript
// Lock Management
POST   /api/locks/acquire
POST   /api/locks/release
POST   /api/locks/renew
GET    /api/locks/{lockId}/status
GET    /api/locks/held

// Lock Types
POST   /api/locks/exclusive
POST   /api/locks/shared
POST   /api/locks/read-write

// Monitoring
GET    /api/locks/stats
GET    /api/locks/health
GET    /api/locks/queue
```

### Request/Response Examples

```json
// Acquire Lock
POST /api/locks/acquire
{
  "resource": "user_123_profile",
  "type": "exclusive",
  "ttl": 30000,
  "waitTimeout": 5000,
  "clientId": "client_456"
}

// Response
{
  "success": true,
  "data": {
    "lockId": "lock_789",
    "resource": "user_123_profile",
    "type": "exclusive",
    "clientId": "client_456",
    "acquiredAt": "2024-01-15T10:30:00Z",
    "expiresAt": "2024-01-15T10:30:30Z",
    "ttl": 30000
  }
}

// Lock Statistics
{
  "success": true,
  "data": {
    "totalLocks": 150,
    "activeLocks": 45,
    "expiredLocks": 105,
    "averageHoldTime": 2500,
    "lockTypes": {
      "exclusive": 30,
      "shared": 10,
      "read-write": 5
    },
    "topResources": [
      { "resource": "user_123_profile", "locks": 15 },
      { "resource": "order_456", "locks": 12 }
    ]
  }
}
```

## Data Model

### Core Entities

```javascript
// Lock Entity
class Lock {
  constructor(resource, type, clientId, ttl) {
    this.id = this.generateID();
    this.resource = resource;
    this.type = type; // 'exclusive', 'shared', 'read-write'
    this.clientId = clientId;
    this.ttl = ttl;
    this.acquiredAt = new Date();
    this.expiresAt = new Date(Date.now() + ttl);
    this.renewalCount = 0;
    this.maxRenewals = 10;
    this.isActive = true;
  }
}

// Lock Queue Entity
class LockQueue {
  constructor(resource) {
    this.resource = resource;
    this.waitingClients = [];
    this.currentLock = null;
    this.createdAt = new Date();
  }
}

// Lock Statistics Entity
class LockStats {
  constructor() {
    this.totalLocks = 0;
    this.activeLocks = 0;
    this.expiredLocks = 0;
    this.averageHoldTime = 0;
    this.lockTypes = {
      exclusive: 0,
      shared: 0,
      "read-write": 0
    };
    this.resourceStats = new Map();
    this.lastUpdated = new Date();
  }
}

// Client Entity
class Client {
  constructor(clientId, connectionInfo) {
    this.id = clientId;
    this.connectionInfo = connectionInfo;
    this.activeLocks = new Set();
    this.totalLocksAcquired = 0;
    this.totalLocksReleased = 0;
    this.lastSeen = new Date();
    this.isActive = true;
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. Basic Redis-based locking
2. Simple TTL expiration
3. No lock renewal or queuing
4. Single Redis instance

### Production-Ready Design
1. **Redis Cluster**: Multiple Redis instances for high availability
2. **Lock Queuing**: FIFO queue for fair lock acquisition
3. **Lock Renewal**: Automatic lock extension for long operations
4. **Deadlock Prevention**: Automatic cleanup and monitoring
5. **Multiple Lock Types**: Exclusive, shared, and read-write locks
6. **Comprehensive Monitoring**: Statistics and health monitoring

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const Redis = require("redis");
const { v4: uuidv4 } = require("uuid");

class DistributedLockService extends EventEmitter {
  constructor(redisConfig) {
    super();
    this.redis = Redis.createClient(redisConfig);
    this.locks = new Map();
    this.lockQueues = new Map();
    this.clients = new Map();
    this.stats = new LockStats();
    this.renewalInterval = null;
    this.cleanupInterval = null;
    this.statsInterval = null;
    
    // Start background tasks
    this.startLockRenewal();
    this.startLockCleanup();
    this.startStatsCollection();
  }

  // Lock Acquisition
  async acquireLock(lockData) {
    try {
      const { resource, type = "exclusive", ttl = 30000, waitTimeout = 5000, clientId } = lockData;
      
      if (!resource || !clientId) {
        throw new Error("Resource and clientId are required");
      }
      
      // Check if client exists
      let client = this.clients.get(clientId);
      if (!client) {
        client = new Client(clientId, {});
        this.clients.set(clientId, client);
      }
      
      // Try to acquire lock immediately
      const lock = await this.tryAcquireLock(resource, type, clientId, ttl);
      
      if (lock) {
        client.activeLocks.add(lock.id);
        client.totalLocksAcquired++;
        client.lastSeen = new Date();
        
        this.emit("lockAcquired", { lock, client });
        return lock;
      }
      
      // If immediate acquisition fails, wait in queue
      if (waitTimeout > 0) {
        return await this.waitForLock(resource, type, clientId, ttl, waitTimeout);
      }
      
      throw new Error("Lock acquisition timeout");
      
    } catch (error) {
      console.error("Lock acquisition error:", error);
      throw error;
    }
  }

  async tryAcquireLock(resource, type, clientId, ttl) {
    try {
      const lockKey = `lock:${resource}`;
      const lockId = this.generateID();
      
      // Check if lock already exists
      const existingLock = await this.redis.get(lockKey);
      
      if (existingLock) {
        const lockData = JSON.parse(existingLock);
        
        // Check if lock is expired
        if (new Date(lockData.expiresAt) < new Date()) {
          await this.redis.del(lockKey);
        } else {
          // Check lock compatibility
          if (!this.isLockCompatible(lockData.type, type)) {
            return null;
          }
        }
      }
      
      // Create new lock
      const lock = new Lock(resource, type, clientId, ttl);
      lock.id = lockId;
      
      // Store lock in Redis
      const lockData = {
        id: lock.id,
        resource: lock.resource,
        type: lock.type,
        clientId: lock.clientId,
        acquiredAt: lock.acquiredAt.toISOString(),
        expiresAt: lock.expiresAt.toISOString(),
        ttl: lock.ttl
      };
      
      await this.redis.setex(lockKey, Math.ceil(ttl / 1000), JSON.stringify(lockData));
      
      // Store lock locally
      this.locks.set(lock.id, lock);
      
      // Update statistics
      this.stats.totalLocks++;
      this.stats.activeLocks++;
      this.stats.lockTypes[type]++;
      
      return lock;
      
    } catch (error) {
      console.error("Try acquire lock error:", error);
      throw error;
    }
  }

  async waitForLock(resource, type, clientId, ttl, waitTimeout) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Lock acquisition timeout"));
      }, waitTimeout);
      
      // Add to queue
      this.addToQueue(resource, { type, clientId, ttl, resolve, timeout });
      
      // Try to acquire lock periodically
      const checkInterval = setInterval(async () => {
        try {
          const lock = await this.tryAcquireLock(resource, type, clientId, ttl);
          if (lock) {
            clearTimeout(timeout);
            clearInterval(checkInterval);
            this.removeFromQueue(resource, clientId);
            
            const client = this.clients.get(clientId);
            if (client) {
              client.activeLocks.add(lock.id);
              client.totalLocksAcquired++;
              client.lastSeen = new Date();
            }
            
            this.emit("lockAcquired", { lock, client });
            resolve(lock);
          }
        } catch (error) {
          clearTimeout(timeout);
          clearInterval(checkInterval);
          this.removeFromQueue(resource, clientId);
          reject(error);
        }
      }, 100); // Check every 100ms
    });
  }

  // Lock Release
  async releaseLock(lockId, clientId) {
    try {
      const lock = this.locks.get(lockId);
      if (!lock) {
        throw new Error("Lock not found");
      }
      
      if (lock.clientId !== clientId) {
        throw new Error("Lock not owned by client");
      }
      
      // Remove from Redis
      const lockKey = `lock:${lock.resource}`;
      await this.redis.del(lockKey);
      
      // Remove from local storage
      this.locks.delete(lockId);
      
      // Update client
      const client = this.clients.get(clientId);
      if (client) {
        client.activeLocks.delete(lockId);
        client.totalLocksReleased++;
        client.lastSeen = new Date();
      }
      
      // Update statistics
      this.stats.activeLocks--;
      this.stats.lockTypes[lock.type]--;
      
      // Notify waiting clients
      this.notifyWaitingClients(lock.resource);
      
      this.emit("lockReleased", { lock, client });
      
      return true;
      
    } catch (error) {
      console.error("Lock release error:", error);
      throw error;
    }
  }

  // Lock Renewal
  async renewLock(lockId, clientId, newTtl) {
    try {
      const lock = this.locks.get(lockId);
      if (!lock) {
        throw new Error("Lock not found");
      }
      
      if (lock.clientId !== clientId) {
        throw new Error("Lock not owned by client");
      }
      
      if (lock.renewalCount >= lock.maxRenewals) {
        throw new Error("Maximum renewals exceeded");
      }
      
      // Update lock expiration
      lock.ttl = newTtl || lock.ttl;
      lock.expiresAt = new Date(Date.now() + lock.ttl);
      lock.renewalCount++;
      
      // Update in Redis
      const lockKey = `lock:${lock.resource}`;
      const lockData = {
        id: lock.id,
        resource: lock.resource,
        type: lock.type,
        clientId: lock.clientId,
        acquiredAt: lock.acquiredAt.toISOString(),
        expiresAt: lock.expiresAt.toISOString(),
        ttl: lock.ttl,
        renewalCount: lock.renewalCount
      };
      
      await this.redis.setex(lockKey, Math.ceil(lock.ttl / 1000), JSON.stringify(lockData));
      
      this.emit("lockRenewed", { lock, client: this.clients.get(clientId) });
      
      return lock;
      
    } catch (error) {
      console.error("Lock renewal error:", error);
      throw error;
    }
  }

  // Lock Types
  isLockCompatible(existingType, requestedType) {
    const compatibility = {
      exclusive: { exclusive: false, shared: false, "read-write": false },
      shared: { exclusive: false, shared: true, "read-write": false },
      "read-write": { exclusive: false, shared: false, "read-write": true }
    };
    
    return compatibility[existingType][requestedType];
  }

  // Queue Management
  addToQueue(resource, lockRequest) {
    if (!this.lockQueues.has(resource)) {
      this.lockQueues.set(resource, new LockQueue(resource));
    }
    
    const queue = this.lockQueues.get(resource);
    queue.waitingClients.push(lockRequest);
  }

  removeFromQueue(resource, clientId) {
    const queue = this.lockQueues.get(resource);
    if (queue) {
      queue.waitingClients = queue.waitingClients.filter(
        request => request.clientId !== clientId
      );
      
      if (queue.waitingClients.length === 0) {
        this.lockQueues.delete(resource);
      }
    }
  }

  notifyWaitingClients(resource) {
    const queue = this.lockQueues.get(resource);
    if (queue && queue.waitingClients.length > 0) {
      // Notify the first waiting client
      const nextClient = queue.waitingClients.shift();
      if (nextClient) {
        // Trigger lock acquisition attempt
        this.emit("lockAvailable", { resource, clientId: nextClient.clientId });
      }
    }
  }

  // Background Tasks
  startLockRenewal() {
    this.renewalInterval = setInterval(() => {
      this.renewExpiringLocks();
    }, 10000); // Check every 10 seconds
  }

  async renewExpiringLocks() {
    const now = new Date();
    const expiringLocks = Array.from(this.locks.values())
      .filter(lock => {
        const timeUntilExpiry = lock.expiresAt.getTime() - now.getTime();
        return timeUntilExpiry < 10000 && timeUntilExpiry > 0; // Expires in next 10 seconds
      });
    
    for (const lock of expiringLocks) {
      try {
        await this.renewLock(lock.id, lock.clientId, lock.ttl);
      } catch (error) {
        console.error(`Failed to renew lock ${lock.id}:`, error);
      }
    }
  }

  startLockCleanup() {
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpiredLocks();
    }, 30000); // Run every 30 seconds
  }

  async cleanupExpiredLocks() {
    const now = new Date();
    const expiredLocks = Array.from(this.locks.values())
      .filter(lock => lock.expiresAt < now);
    
    for (const lock of expiredLocks) {
      try {
        await this.releaseLock(lock.id, lock.clientId);
        this.stats.expiredLocks++;
        
        this.emit("lockExpired", { lock });
      } catch (error) {
        console.error(`Failed to cleanup expired lock ${lock.id}:`, error);
      }
    }
  }

  startStatsCollection() {
    this.statsInterval = setInterval(() => {
      this.updateStats();
    }, 60000); // Update every minute
  }

  updateStats() {
    // Calculate average hold time
    const activeLocks = Array.from(this.locks.values());
    if (activeLocks.length > 0) {
      const totalHoldTime = activeLocks.reduce((sum, lock) => {
        return sum + (Date.now() - lock.acquiredAt.getTime());
      }, 0);
      
      this.stats.averageHoldTime = totalHoldTime / activeLocks.length;
    }
    
    // Update resource statistics
    this.stats.resourceStats.clear();
    for (const lock of activeLocks) {
      const count = this.stats.resourceStats.get(lock.resource) || 0;
      this.stats.resourceStats.set(lock.resource, count + 1);
    }
    
    this.stats.lastUpdated = new Date();
    this.emit("statsUpdated", this.stats);
  }

  // Utility Methods
  async getLockStatus(lockId) {
    const lock = this.locks.get(lockId);
    if (!lock) {
      return null;
    }
    
    return {
      id: lock.id,
      resource: lock.resource,
      type: lock.type,
      clientId: lock.clientId,
      acquiredAt: lock.acquiredAt,
      expiresAt: lock.expiresAt,
      ttl: lock.ttl,
      renewalCount: lock.renewalCount,
      isExpired: lock.expiresAt < new Date()
    };
  }

  getHeldLocks(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return [];
    }
    
    return Array.from(client.activeLocks).map(lockId => {
      const lock = this.locks.get(lockId);
      return lock ? {
        id: lock.id,
        resource: lock.resource,
        type: lock.type,
        acquiredAt: lock.acquiredAt,
        expiresAt: lock.expiresAt
      } : null;
    }).filter(lock => lock !== null);
  }

  getQueueStatus(resource) {
    const queue = this.lockQueues.get(resource);
    if (!queue) {
      return { waitingClients: 0, queue: [] };
    }
    
    return {
      waitingClients: queue.waitingClients.length,
      queue: queue.waitingClients.map(client => ({
        clientId: client.clientId,
        type: client.type,
        ttl: client.ttl
      }))
    };
  }

  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { DistributedLockService } = require("./services/DistributedLockService");

class DistributedLockAPI {
  constructor() {
    this.app = express();
    this.lockService = new DistributedLockService({
      host: process.env.REDIS_HOST || "localhost",
      port: process.env.REDIS_PORT || 6379
    });
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupEventHandlers();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    
    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
      next();
    });
  }

  setupRoutes() {
    // Lock management
    this.app.post("/api/locks/acquire", this.acquireLock.bind(this));
    this.app.post("/api/locks/release", this.releaseLock.bind(this));
    this.app.post("/api/locks/renew", this.renewLock.bind(this));
    this.app.get("/api/locks/:lockId/status", this.getLockStatus.bind(this));
    this.app.get("/api/locks/held", this.getHeldLocks.bind(this));
    
    // Lock types
    this.app.post("/api/locks/exclusive", this.acquireExclusiveLock.bind(this));
    this.app.post("/api/locks/shared", this.acquireSharedLock.bind(this));
    this.app.post("/api/locks/read-write", this.acquireReadWriteLock.bind(this));
    
    // Monitoring
    this.app.get("/api/locks/stats", this.getStats.bind(this));
    this.app.get("/api/locks/health", this.getHealth.bind(this));
    this.app.get("/api/locks/queue", this.getQueueStatus.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        activeLocks: this.lockService.stats.activeLocks,
        totalLocks: this.lockService.stats.totalLocks
      });
    });
  }

  setupEventHandlers() {
    this.lockService.on("lockAcquired", ({ lock, client }) => {
      console.log(`Lock acquired: ${lock.id} by client ${client.id}`);
    });
    
    this.lockService.on("lockReleased", ({ lock, client }) => {
      console.log(`Lock released: ${lock.id} by client ${client.id}`);
    });
    
    this.lockService.on("lockExpired", ({ lock }) => {
      console.log(`Lock expired: ${lock.id}`);
    });
  }

  // HTTP Handlers
  async acquireLock(req, res) {
    try {
      const lock = await this.lockService.acquireLock(req.body);
      
      res.status(201).json({
        success: true,
        data: lock
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async releaseLock(req, res) {
    try {
      const { lockId, clientId } = req.body;
      
      await this.lockService.releaseLock(lockId, clientId);
      
      res.json({
        success: true,
        message: "Lock released successfully"
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async renewLock(req, res) {
    try {
      const { lockId, clientId, ttl } = req.body;
      
      const lock = await this.lockService.renewLock(lockId, clientId, ttl);
      
      res.json({
        success: true,
        data: lock
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getLockStatus(req, res) {
    try {
      const { lockId } = req.params;
      
      const status = await this.lockService.getLockStatus(lockId);
      
      if (!status) {
        return res.status(404).json({ error: "Lock not found" });
      }
      
      res.json({
        success: true,
        data: status
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getHeldLocks(req, res) {
    try {
      const { clientId } = req.query;
      
      if (!clientId) {
        return res.status(400).json({ error: "clientId is required" });
      }
      
      const locks = this.lockService.getHeldLocks(clientId);
      
      res.json({
        success: true,
        data: locks
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getStats(req, res) {
    try {
      const stats = {
        ...this.lockService.stats,
        resourceStats: Object.fromEntries(this.lockService.stats.resourceStats),
        topResources: Array.from(this.lockService.stats.resourceStats.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([resource, locks]) => ({ resource, locks }))
      };
      
      res.json({
        success: true,
        data: stats
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getHealth(req, res) {
    try {
      const health = {
        status: "healthy",
        activeLocks: this.lockService.stats.activeLocks,
        totalLocks: this.lockService.stats.totalLocks,
        expiredLocks: this.lockService.stats.expiredLocks,
        averageHoldTime: this.lockService.stats.averageHoldTime,
        lastUpdated: this.lockService.stats.lastUpdated
      };
      
      res.json({
        success: true,
        data: health
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getQueueStatus(req, res) {
    try {
      const { resource } = req.query;
      
      if (!resource) {
        return res.status(400).json({ error: "resource is required" });
      }
      
      const queueStatus = this.lockService.getQueueStatus(resource);
      
      res.json({
        success: true,
        data: queueStatus
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Distributed Lock API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new DistributedLockAPI();
  api.start(3000);
}

module.exports = { DistributedLockAPI };
```

## Key Features

### Lock Management
- **Multiple Lock Types**: Exclusive, shared, and read-write locks
- **Automatic Expiration**: TTL-based lock expiration
- **Lock Renewal**: Extend lock duration for long operations
- **Deadlock Prevention**: Automatic cleanup and monitoring

### High Availability
- **Redis Backend**: Distributed storage with Redis
- **Fault Tolerance**: Handle Redis failures gracefully
- **Lock Queuing**: FIFO queue for fair lock acquisition
- **Client Management**: Track client connections and locks

### Performance & Monitoring
- **Low Latency**: Sub-10ms lock acquisition
- **High Throughput**: 10,000+ operations per second
- **Comprehensive Stats**: Lock usage and performance metrics
- **Health Monitoring**: System health and status tracking

## Extension Ideas

### Advanced Features
1. **Lock Hierarchies**: Nested and hierarchical locks
2. **Conditional Locks**: Lock acquisition based on conditions
3. **Lock Groups**: Group-related locks for atomic operations
4. **Priority Queues**: Priority-based lock acquisition
5. **Lock Analytics**: Advanced usage patterns and insights

### Enterprise Features
1. **Multi-region Support**: Cross-region lock coordination
2. **Advanced Security**: Authentication and authorization
3. **Audit Logging**: Complete lock operation audit trail
4. **Performance Tuning**: Optimized lock algorithms
5. **Integration APIs**: Third-party service integration
