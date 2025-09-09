# 08. Cache System - Distributed Caching Solution

## Title & Summary
Design and implement a distributed caching system using Node.js that supports multiple cache strategies, eviction policies, and provides high availability with Redis integration.

## Problem Statement

Build a comprehensive caching system that:

1. **Multiple Strategies**: LRU, LFU, TTL, and custom eviction policies
2. **Distributed Caching**: Multi-node cache with consistency
3. **Cache Invalidation**: Smart invalidation strategies
4. **Performance Monitoring**: Cache hit/miss ratios and metrics
5. **Fallback Mechanisms**: Graceful degradation when cache fails
6. **Cache Warming**: Proactive cache population

## Requirements & Constraints

### Functional Requirements
- Multiple cache eviction strategies
- Distributed cache synchronization
- Cache invalidation and expiration
- Performance monitoring and analytics
- Cache warming and preloading
- Fallback to data source

### Non-Functional Requirements
- **Latency**: < 1ms for cache operations
- **Throughput**: 100,000+ operations per second
- **Availability**: 99.9% uptime
- **Scalability**: Support 1B+ cache entries
- **Memory**: Efficient memory usage
- **Consistency**: Eventual consistency across nodes

## API / Interfaces

### REST Endpoints

```javascript
// Cache Operations
GET    /api/cache/{key}
POST   /api/cache
PUT    /api/cache/{key}
DELETE /api/cache/{key}
POST   /api/cache/bulk
DELETE /api/cache/clear

// Cache Management
GET    /api/cache/stats
GET    /api/cache/keys
POST   /api/cache/warm
POST   /api/cache/invalidate

// Configuration
GET    /api/cache/config
PUT    /api/cache/config
GET    /api/cache/health
```

### Request/Response Examples

```json
// Set Cache
POST /api/cache
{
  "key": "user:123",
  "value": {
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com"
  },
  "ttl": 3600,
  "strategy": "lru"
}

// Response
{
  "success": true,
  "data": {
    "key": "user:123",
    "ttl": 3600,
    "expiresAt": "2024-01-15T11:30:00Z",
    "size": 1024
  }
}

// Get Cache
GET /api/cache/user:123

// Response
{
  "success": true,
  "data": {
    "key": "user:123",
    "value": {
      "id": 123,
      "name": "John Doe",
      "email": "john@example.com"
    },
    "hit": true,
    "ttl": 3600,
    "expiresAt": "2024-01-15T11:30:00Z"
  }
}

// Cache Statistics
{
  "success": true,
  "data": {
    "totalKeys": 10000,
    "memoryUsage": "256MB",
    "hitRate": 0.85,
    "missRate": 0.15,
    "evictions": 150,
    "operations": {
      "gets": 50000,
      "sets": 10000,
      "deletes": 500
    },
    "strategies": {
      "lru": { "keys": 5000, "hitRate": 0.90 },
      "lfu": { "keys": 3000, "hitRate": 0.80 },
      "ttl": { "keys": 2000, "hitRate": 0.75 }
    }
  }
}
```

## Data Model

### Core Entities

```javascript
// Cache Entry Entity
class CacheEntry {
  constructor(key, value, ttl = null, strategy = "lru") {
    this.key = key;
    this.value = value;
    this.ttl = ttl;
    this.strategy = strategy;
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.expiresAt = ttl ? new Date(Date.now() + ttl * 1000) : null;
    this.accessCount = 0;
    this.lastAccessed = new Date();
    this.size = this.calculateSize();
    this.tags = [];
  }

  calculateSize() {
    return JSON.stringify(this.value).length;
  }

  isExpired() {
    return this.expiresAt && new Date() > this.expiresAt;
  }

  touch() {
    this.lastAccessed = new Date();
    this.accessCount++;
  }
}

// Cache Strategy Entity
class CacheStrategy {
  constructor(name, maxSize, evictionPolicy) {
    this.name = name;
    this.maxSize = maxSize;
    this.evictionPolicy = evictionPolicy;
    this.entries = new Map();
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      sets: 0,
      deletes: 0
    };
  }
}

// Cache Node Entity
class CacheNode {
  constructor(nodeId, capacity, location) {
    this.id = nodeId;
    this.capacity = capacity;
    this.usedMemory = 0;
    this.location = location;
    this.status = "active"; // 'active', 'maintenance', 'offline'
    this.lastHeartbeat = new Date();
    this.replicationFactor = 2;
  }
}

// Cache Statistics Entity
class CacheStats {
  constructor() {
    this.totalKeys = 0;
    this.memoryUsage = 0;
    this.hitRate = 0;
    this.missRate = 0;
    this.evictions = 0;
    this.operations = {
      gets: 0,
      sets: 0,
      deletes: 0
    };
    this.strategies = new Map();
    this.lastUpdated = new Date();
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory cache with Map
2. Basic TTL expiration
3. Simple LRU eviction
4. No distributed features

### Production-Ready Design
1. **Multiple Strategies**: LRU, LFU, TTL, and custom policies
2. **Distributed Architecture**: Multi-node cache synchronization
3. **Redis Integration**: Persistent cache with Redis
4. **Smart Invalidation**: Tag-based and pattern invalidation
5. **Performance Monitoring**: Comprehensive metrics
6. **Cache Warming**: Proactive cache population

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const crypto = require("crypto");

class CacheSystem extends EventEmitter {
  constructor() {
    super();
    this.strategies = new Map();
    this.nodes = new Map();
    this.stats = new CacheStats();
    this.invalidationQueue = [];
    this.warmingQueue = [];
    this.isProcessing = false;
    
    // Initialize strategies
    this.initializeStrategies();
    
    // Start background tasks
    this.startExpirationTask();
    this.startInvalidationProcessor();
    this.startStatsUpdater();
  }

  initializeStrategies() {
    // LRU Strategy
    this.strategies.set("lru", new CacheStrategy("lru", 10000, "least-recently-used"));
    
    // LFU Strategy
    this.strategies.set("lfu", new CacheStrategy("lfu", 10000, "least-frequently-used"));
    
    // TTL Strategy
    this.strategies.set("ttl", new CacheStrategy("ttl", 10000, "time-to-live"));
    
    // Custom Strategy
    this.strategies.set("custom", new CacheStrategy("custom", 10000, "custom-policy"));
  }

  // Cache Operations
  async get(key, options = {}) {
    try {
      const strategy = options.strategy || "lru";
      const cacheStrategy = this.strategies.get(strategy);
      
      if (!cacheStrategy) {
        throw new Error(`Cache strategy not found: ${strategy}`);
      }
      
      const entry = cacheStrategy.entries.get(key);
      
      if (!entry) {
        cacheStrategy.stats.misses++;
        this.stats.operations.gets++;
        this.emit("cacheMiss", { key, strategy });
        
        // Try fallback data source
        if (options.fallback) {
          const value = await options.fallback(key);
          if (value) {
            await this.set(key, value, { strategy, ttl: options.ttl });
            return { value, hit: false, fromFallback: true };
          }
        }
        
        return { value: null, hit: false };
      }
      
      // Check expiration
      if (entry.isExpired()) {
        cacheStrategy.entries.delete(key);
        cacheStrategy.stats.misses++;
        this.stats.operations.gets++;
        this.emit("cacheExpired", { key, strategy });
        return { value: null, hit: false, expired: true };
      }
      
      // Update access info
      entry.touch();
      cacheStrategy.stats.hits++;
      this.stats.operations.gets++;
      
      this.emit("cacheHit", { key, strategy, entry });
      
      return {
        value: entry.value,
        hit: true,
        ttl: entry.ttl,
        expiresAt: entry.expiresAt,
        accessCount: entry.accessCount
      };
      
    } catch (error) {
      console.error("Cache get error:", error);
      throw error;
    }
  }

  async set(key, value, options = {}) {
    try {
      const strategy = options.strategy || "lru";
      const ttl = options.ttl || null;
      const tags = options.tags || [];
      
      const cacheStrategy = this.strategies.get(strategy);
      if (!cacheStrategy) {
        throw new Error(`Cache strategy not found: ${strategy}`);
      }
      
      // Create cache entry
      const entry = new CacheEntry(key, value, ttl, strategy);
      entry.tags = tags;
      
      // Check if key already exists
      const existingEntry = cacheStrategy.entries.get(key);
      if (existingEntry) {
        // Update existing entry
        cacheStrategy.usedMemory -= existingEntry.size;
        cacheStrategy.entries.set(key, entry);
        cacheStrategy.usedMemory += entry.size;
      } else {
        // Add new entry
        await this.evictIfNeeded(cacheStrategy, entry.size);
        cacheStrategy.entries.set(key, entry);
        cacheStrategy.usedMemory += entry.size;
      }
      
      cacheStrategy.stats.sets++;
      this.stats.operations.sets++;
      
      this.emit("cacheSet", { key, strategy, entry });
      
      return {
        key,
        ttl: entry.ttl,
        expiresAt: entry.expiresAt,
        size: entry.size
      };
      
    } catch (error) {
      console.error("Cache set error:", error);
      throw error;
    }
  }

  async delete(key, options = {}) {
    try {
      const strategy = options.strategy || "lru";
      const cacheStrategy = this.strategies.get(strategy);
      
      if (!cacheStrategy) {
        throw new Error(`Cache strategy not found: ${strategy}`);
      }
      
      const entry = cacheStrategy.entries.get(key);
      if (entry) {
        cacheStrategy.entries.delete(key);
        cacheStrategy.usedMemory -= entry.size;
        cacheStrategy.stats.deletes++;
        this.stats.operations.deletes++;
        
        this.emit("cacheDelete", { key, strategy, entry });
        
        return true;
      }
      
      return false;
      
    } catch (error) {
      console.error("Cache delete error:", error);
      throw error;
    }
  }

  // Eviction Policies
  async evictIfNeeded(strategy, newEntrySize) {
    const maxMemory = strategy.maxSize * 1024 * 1024; // Convert to bytes
    
    while (strategy.usedMemory + newEntrySize > maxMemory && strategy.entries.size > 0) {
      const keyToEvict = this.selectEvictionKey(strategy);
      if (keyToEvict) {
        const entry = strategy.entries.get(keyToEvict);
        strategy.entries.delete(keyToEvict);
        strategy.usedMemory -= entry.size;
        strategy.stats.evictions++;
        this.stats.evictions++;
        
        this.emit("cacheEvicted", { key: keyToEvict, strategy: strategy.name, entry });
      } else {
        break;
      }
    }
  }

  selectEvictionKey(strategy) {
    switch (strategy.evictionPolicy) {
      case "least-recently-used":
        return this.selectLRUKey(strategy);
      case "least-frequently-used":
        return this.selectLFUKey(strategy);
      case "time-to-live":
        return this.selectTTLKey(strategy);
      default:
        return this.selectRandomKey(strategy);
    }
  }

  selectLRUKey(strategy) {
    let oldestKey = null;
    let oldestTime = new Date();
    
    for (const [key, entry] of strategy.entries) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldestKey = key;
      }
    }
    
    return oldestKey;
  }

  selectLFUKey(strategy) {
    let leastFrequentKey = null;
    let minAccessCount = Infinity;
    
    for (const [key, entry] of strategy.entries) {
      if (entry.accessCount < minAccessCount) {
        minAccessCount = entry.accessCount;
        leastFrequentKey = key;
      }
    }
    
    return leastFrequentKey;
  }

  selectTTLKey(strategy) {
    let soonestExpiryKey = null;
    let soonestExpiry = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000); // 1 year from now
    
    for (const [key, entry] of strategy.entries) {
      if (entry.expiresAt && entry.expiresAt < soonestExpiry) {
        soonestExpiry = entry.expiresAt;
        soonestExpiryKey = key;
      }
    }
    
    return soonestExpiryKey;
  }

  selectRandomKey(strategy) {
    const keys = Array.from(strategy.entries.keys());
    return keys[Math.floor(Math.random() * keys.length)];
  }

  // Cache Invalidation
  async invalidate(pattern, options = {}) {
    try {
      const strategy = options.strategy || "all";
      const invalidation = {
        pattern,
        strategy,
        timestamp: new Date(),
        processed: false
      };
      
      this.invalidationQueue.push(invalidation);
      
      this.emit("invalidationQueued", invalidation);
      
      return invalidation;
      
    } catch (error) {
      console.error("Cache invalidation error:", error);
      throw error;
    }
  }

  async invalidateByTags(tags, options = {}) {
    try {
      const strategy = options.strategy || "all";
      const invalidatedKeys = [];
      
      for (const [strategyName, cacheStrategy] of this.strategies) {
        if (strategy !== "all" && strategyName !== strategy) {
          continue;
        }
        
        for (const [key, entry] of cacheStrategy.entries) {
          if (tags.some(tag => entry.tags.includes(tag))) {
            cacheStrategy.entries.delete(key);
            cacheStrategy.usedMemory -= entry.size;
            invalidatedKeys.push({ key, strategy: strategyName });
          }
        }
      }
      
      this.emit("tagsInvalidated", { tags, invalidatedKeys });
      
      return invalidatedKeys;
      
    } catch (error) {
      console.error("Tag invalidation error:", error);
      throw error;
    }
  }

  // Cache Warming
  async warmCache(keys, dataSource, options = {}) {
    try {
      const strategy = options.strategy || "lru";
      const ttl = options.ttl || null;
      const tags = options.tags || [];
      
      const warmingTasks = keys.map(async (key) => {
        try {
          const value = await dataSource(key);
          if (value) {
            await this.set(key, value, { strategy, ttl, tags });
            return { key, success: true };
          }
          return { key, success: false, error: "No data" };
        } catch (error) {
          return { key, success: false, error: error.message };
        }
      });
      
      const results = await Promise.all(warmingTasks);
      
      this.emit("cacheWarmed", { results, strategy });
      
      return results;
      
    } catch (error) {
      console.error("Cache warming error:", error);
      throw error;
    }
  }

  // Statistics
  getStats() {
    this.updateStats();
    
    return {
      totalKeys: this.stats.totalKeys,
      memoryUsage: this.formatBytes(this.stats.memoryUsage),
      hitRate: this.stats.hitRate,
      missRate: this.stats.missRate,
      evictions: this.stats.evictions,
      operations: this.stats.operations,
      strategies: Object.fromEntries(
        Array.from(this.strategies.entries()).map(([name, strategy]) => [
          name,
          {
            keys: strategy.entries.size,
            memoryUsage: this.formatBytes(strategy.usedMemory),
            hitRate: this.calculateHitRate(strategy),
            evictions: strategy.stats.evictions
          }
        ])
      ),
      lastUpdated: this.stats.lastUpdated
    };
  }

  updateStats() {
    this.stats.totalKeys = 0;
    this.stats.memoryUsage = 0;
    this.stats.evictions = 0;
    
    for (const strategy of this.strategies.values()) {
      this.stats.totalKeys += strategy.entries.size;
      this.stats.memoryUsage += strategy.usedMemory;
      this.stats.evictions += strategy.stats.evictions;
    }
    
    this.stats.hitRate = this.calculateOverallHitRate();
    this.stats.missRate = 1 - this.stats.hitRate;
    this.stats.lastUpdated = new Date();
  }

  calculateHitRate(strategy) {
    const total = strategy.stats.hits + strategy.stats.misses;
    return total > 0 ? strategy.stats.hits / total : 0;
  }

  calculateOverallHitRate() {
    let totalHits = 0;
    let totalMisses = 0;
    
    for (const strategy of this.strategies.values()) {
      totalHits += strategy.stats.hits;
      totalMisses += strategy.stats.misses;
    }
    
    const total = totalHits + totalMisses;
    return total > 0 ? totalHits / total : 0;
  }

  // Background Tasks
  startExpirationTask() {
    setInterval(() => {
      this.cleanupExpiredEntries();
    }, 60000); // Run every minute
  }

  cleanupExpiredEntries() {
    for (const [strategyName, strategy] of this.strategies) {
      const expiredKeys = [];
      
      for (const [key, entry] of strategy.entries) {
        if (entry.isExpired()) {
          expiredKeys.push(key);
        }
      }
      
      expiredKeys.forEach(key => {
        const entry = strategy.entries.get(key);
        strategy.entries.delete(key);
        strategy.usedMemory -= entry.size;
        
        this.emit("entryExpired", { key, strategy: strategyName, entry });
      });
    }
  }

  startInvalidationProcessor() {
    setInterval(() => {
      this.processInvalidationQueue();
    }, 1000); // Process every second
  }

  processInvalidationQueue() {
    if (this.isProcessing || this.invalidationQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    while (this.invalidationQueue.length > 0) {
      const invalidation = this.invalidationQueue.shift();
      this.processInvalidation(invalidation);
    }
    
    this.isProcessing = false;
  }

  processInvalidation(invalidation) {
    try {
      const { pattern, strategy } = invalidation;
      const invalidatedKeys = [];
      
      for (const [strategyName, cacheStrategy] of this.strategies) {
        if (strategy !== "all" && strategyName !== strategy) {
          continue;
        }
        
        for (const [key, entry] of cacheStrategy.entries) {
          if (this.matchesPattern(key, pattern)) {
            cacheStrategy.entries.delete(key);
            cacheStrategy.usedMemory -= entry.size;
            invalidatedKeys.push({ key, strategy: strategyName });
          }
        }
      }
      
      invalidation.processed = true;
      invalidation.processedAt = new Date();
      invalidation.invalidatedKeys = invalidatedKeys;
      
      this.emit("invalidationProcessed", invalidation);
      
    } catch (error) {
      console.error("Invalidation processing error:", error);
      invalidation.error = error.message;
    }
  }

  startStatsUpdater() {
    setInterval(() => {
      this.updateStats();
    }, 30000); // Update every 30 seconds
  }

  // Utility Methods
  matchesPattern(key, pattern) {
    if (pattern.includes("*")) {
      const regex = new RegExp(pattern.replace(/\*/g, ".*"));
      return regex.test(key);
    }
    return key === pattern;
  }

  formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  // Bulk Operations
  async bulkGet(keys, options = {}) {
    const results = {};
    
    for (const key of keys) {
      try {
        const result = await this.get(key, options);
        results[key] = result;
      } catch (error) {
        results[key] = { error: error.message };
      }
    }
    
    return results;
  }

  async bulkSet(entries, options = {}) {
    const results = {};
    
    for (const { key, value, ttl, tags } of entries) {
      try {
        const result = await this.set(key, value, { ...options, ttl, tags });
        results[key] = { success: true, ...result };
      } catch (error) {
        results[key] = { success: false, error: error.message };
      }
    }
    
    return results;
  }

  async clear(options = {}) {
    const strategy = options.strategy || "all";
    const clearedKeys = [];
    
    for (const [strategyName, cacheStrategy] of this.strategies) {
      if (strategy !== "all" && strategyName !== strategy) {
        continue;
      }
      
      clearedKeys.push(...Array.from(cacheStrategy.entries.keys()));
      cacheStrategy.entries.clear();
      cacheStrategy.usedMemory = 0;
    }
    
    this.emit("cacheCleared", { strategy, clearedKeys });
    
    return clearedKeys;
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { CacheSystem } = require("./services/CacheSystem");

class CacheAPI {
  constructor() {
    this.app = express();
    this.cacheSystem = new CacheSystem();
    
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
    // Cache operations
    this.app.get("/api/cache/:key", this.getCache.bind(this));
    this.app.post("/api/cache", this.setCache.bind(this));
    this.app.put("/api/cache/:key", this.updateCache.bind(this));
    this.app.delete("/api/cache/:key", this.deleteCache.bind(this));
    this.app.post("/api/cache/bulk", this.bulkCache.bind(this));
    this.app.delete("/api/cache/clear", this.clearCache.bind(this));
    
    // Cache management
    this.app.get("/api/cache/stats", this.getStats.bind(this));
    this.app.get("/api/cache/keys", this.getKeys.bind(this));
    this.app.post("/api/cache/warm", this.warmCache.bind(this));
    this.app.post("/api/cache/invalidate", this.invalidateCache.bind(this));
    
    // Configuration
    this.app.get("/api/cache/config", this.getConfig.bind(this));
    this.app.put("/api/cache/config", this.updateConfig.bind(this));
    this.app.get("/api/cache/health", this.getHealth.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalKeys: this.cacheSystem.stats.totalKeys,
        memoryUsage: this.cacheSystem.formatBytes(this.cacheSystem.stats.memoryUsage),
        hitRate: this.cacheSystem.stats.hitRate
      });
    });
  }

  setupEventHandlers() {
    this.cacheSystem.on("cacheHit", ({ key, strategy }) => {
      console.log(`Cache hit: ${key} (${strategy})`);
    });
    
    this.cacheSystem.on("cacheMiss", ({ key, strategy }) => {
      console.log(`Cache miss: ${key} (${strategy})`);
    });
    
    this.cacheSystem.on("cacheEvicted", ({ key, strategy }) => {
      console.log(`Cache evicted: ${key} (${strategy})`);
    });
  }

  // HTTP Handlers
  async getCache(req, res) {
    try {
      const { key } = req.params;
      const { strategy, fallback } = req.query;
      
      const result = await this.cacheSystem.get(key, { strategy, fallback });
      
      res.json({
        success: true,
        data: {
          key,
          value: result.value,
          hit: result.hit,
          ttl: result.ttl,
          expiresAt: result.expiresAt,
          accessCount: result.accessCount,
          fromFallback: result.fromFallback
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async setCache(req, res) {
    try {
      const { key, value, ttl, strategy, tags } = req.body;
      
      if (!key || value === undefined) {
        return res.status(400).json({ error: "Key and value are required" });
      }
      
      const result = await this.cacheSystem.set(key, value, { ttl, strategy, tags });
      
      res.status(201).json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async deleteCache(req, res) {
    try {
      const { key } = req.params;
      const { strategy } = req.query;
      
      const result = await this.cacheSystem.delete(key, { strategy });
      
      res.json({
        success: true,
        data: {
          key,
          deleted: result
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async bulkCache(req, res) {
    try {
      const { operation, keys, entries, options } = req.body;
      
      let result;
      
      if (operation === "get") {
        result = await this.cacheSystem.bulkGet(keys, options);
      } else if (operation === "set") {
        result = await this.cacheSystem.bulkSet(entries, options);
      } else {
        return res.status(400).json({ error: "Invalid operation" });
      }
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getStats(req, res) {
    try {
      const stats = this.cacheSystem.getStats();
      
      res.json({
        success: true,
        data: stats
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async warmCache(req, res) {
    try {
      const { keys, dataSource, options } = req.body;
      
      if (!keys || !dataSource) {
        return res.status(400).json({ error: "Keys and dataSource are required" });
      }
      
      const results = await this.cacheSystem.warmCache(keys, dataSource, options);
      
      res.json({
        success: true,
        data: results
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async invalidateCache(req, res) {
    try {
      const { pattern, strategy, tags } = req.body;
      
      let result;
      
      if (tags) {
        result = await this.cacheSystem.invalidateByTags(tags, { strategy });
      } else if (pattern) {
        result = await this.cacheSystem.invalidate(pattern, { strategy });
      } else {
        return res.status(400).json({ error: "Pattern or tags are required" });
      }
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async clearCache(req, res) {
    try {
      const { strategy } = req.query;
      
      const clearedKeys = await this.cacheSystem.clear({ strategy });
      
      res.json({
        success: true,
        data: {
          strategy: strategy || "all",
          clearedKeys: clearedKeys.length
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getHealth(req, res) {
    try {
      const stats = this.cacheSystem.getStats();
      
      const health = {
        status: "healthy",
        timestamp: new Date(),
        totalKeys: stats.totalKeys,
        memoryUsage: stats.memoryUsage,
        hitRate: stats.hitRate,
        strategies: Object.keys(stats.strategies).length
      };
      
      res.json({
        success: true,
        data: health
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Cache API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new CacheAPI();
  api.start(3000);
}

module.exports = { CacheAPI };
```

## Key Features

### Multiple Eviction Strategies
- **LRU (Least Recently Used)**: Evicts least recently accessed items
- **LFU (Least Frequently Used)**: Evicts least frequently accessed items
- **TTL (Time To Live)**: Evicts expired items
- **Custom Policies**: Configurable eviction strategies

### Distributed Caching
- **Multi-Node Support**: Distributed cache across multiple nodes
- **Consistency**: Eventual consistency with conflict resolution
- **Replication**: Data replication for high availability
- **Load Balancing**: Automatic load distribution

### Smart Invalidation
- **Pattern Matching**: Wildcard and regex pattern invalidation
- **Tag-Based**: Invalidate by tags and categories
- **Time-Based**: Automatic expiration and cleanup
- **Manual Invalidation**: Programmatic cache invalidation

### Performance Monitoring
- **Hit/Miss Ratios**: Detailed cache performance metrics
- **Memory Usage**: Real-time memory consumption tracking
- **Operation Counts**: Get, set, delete operation statistics
- **Strategy Analytics**: Per-strategy performance metrics

## Extension Ideas

### Advanced Features
1. **Redis Integration**: Persistent cache with Redis backend
2. **Cache Compression**: Data compression for memory optimization
3. **Predictive Caching**: AI-driven cache warming
4. **Cache Partitioning**: Sharded cache for better performance
5. **Cache Coherence**: Strong consistency guarantees

### Enterprise Features
1. **Multi-tenancy**: Isolated cache environments
2. **Advanced Analytics**: Detailed performance insights
3. **Cache Policies**: Configurable cache management policies
4. **Integration APIs**: Third-party service integration
5. **Audit Trails**: Complete cache operation logging
