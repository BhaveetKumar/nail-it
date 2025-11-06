---
# Auto-generated front matter
Title: 03 Ratelimiter
LastUpdated: 2025-11-06T20:45:58.773527
Tags: []
Status: draft
---

# 03. Rate Limiter - API Throttling System

## Title & Summary

Design and implement a distributed rate limiter using Node.js that controls API request rates, supports multiple algorithms, and provides real-time monitoring.

## Problem Statement

Build a comprehensive rate limiting system that:

1. **Multiple Algorithms**: Support token bucket, sliding window, and fixed window
2. **Distributed Limiting**: Work across multiple server instances
3. **Flexible Rules**: Different limits for different users/endpoints
4. **Real-time Monitoring**: Track usage patterns and violations
5. **Graceful Degradation**: Handle Redis failures gracefully
6. **Performance**: Handle 100K+ requests per second

## Requirements & Constraints

### Functional Requirements

- Support multiple rate limiting algorithms
- Configurable limits per user/endpoint
- Real-time usage tracking
- Violation notifications
- Admin dashboard for monitoring
- API for limit management

### Non-Functional Requirements

- **Latency**: < 1ms overhead per request
- **Throughput**: 100K+ requests per second
- **Availability**: 99.9% uptime
- **Scalability**: Support 1M+ concurrent users
- **Memory**: < 100MB per instance
- **Accuracy**: 99.9% rate limit accuracy

## API / Interfaces

### REST Endpoints

```javascript
// Rate Limiting
GET / api / rate - limit / check;
POST / api / rate - limit / consume;
GET / api / rate - limit / status / { key };

// Configuration
GET / api / rate - limit / rules;
POST / api / rate - limit / rules;
PUT / api / rate - limit / rules / { ruleID };
DELETE / api / rate - limit / rules / { ruleID };

// Monitoring
GET / api / rate - limit / stats;
GET / api / rate - limit / violations;
GET / api / rate - limit / health;
```

### Request/Response Examples

```json
// Check Rate Limit
GET /api/rate-limit/check?key=user123&endpoint=/api/payments

// Response
{
  "allowed": true,
  "remaining": 95,
  "resetTime": 1642233600000,
  "retryAfter": null
}

// Rate Limit Violation
{
  "allowed": false,
  "remaining": 0,
  "resetTime": 1642233600000,
  "retryAfter": 60
}

// Create Rate Limit Rule
POST /api/rate-limit/rules
{
  "name": "API_LIMIT",
  "keyPattern": "user:{userId}",
  "algorithm": "token_bucket",
  "limit": 100,
  "window": 3600,
  "burst": 10
}
```

## Data Model

### Core Entities

```javascript
// Rate Limit Rule
class RateLimitRule {
  constructor(name, keyPattern, algorithm, limit, window, burst = 0) {
    this.id = this.generateID();
    this.name = name;
    this.keyPattern = keyPattern;
    this.algorithm = algorithm; // 'token_bucket', 'sliding_window', 'fixed_window'
    this.limit = limit;
    this.window = window; // in seconds
    this.burst = burst; // for token bucket
    this.enabled = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Rate Limit State
class RateLimitState {
  constructor(key, rule) {
    this.key = key;
    this.rule = rule;
    this.tokens = rule.burst || rule.limit;
    this.lastRefill = Date.now();
    this.windowStart = Date.now();
    this.requestCount = 0;
    this.violations = 0;
    this.lastViolation = null;
  }
}

// Rate Limit Result
class RateLimitResult {
  constructor(allowed, remaining, resetTime, retryAfter = null) {
    this.allowed = allowed;
    this.remaining = remaining;
    this.resetTime = resetTime;
    this.retryAfter = retryAfter;
    this.timestamp = Date.now();
  }
}
```

## Approach Overview

### Simple Solution (MVP)

1. In-memory rate limiting with basic algorithms
2. Simple key-based limiting
3. Basic monitoring
4. No persistence

### Production-Ready Design

1. **Multiple Algorithms**: Token bucket, sliding window, fixed window
2. **Redis Integration**: Distributed rate limiting
3. **Rule Engine**: Flexible configuration system
4. **Monitoring**: Real-time stats and alerts
5. **Fallback**: Graceful degradation without Redis

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const Redis = require("redis");
const { v4: uuidv4 } = require("uuid");

class RateLimiterService extends EventEmitter {
  constructor(options = {}) {
    super();
    this.rules = new Map();
    this.localCache = new Map();
    this.stats = {
      totalRequests: 0,
      allowedRequests: 0,
      blockedRequests: 0,
      violations: 0,
    };

    // Redis configuration
    this.redis = null;
    this.redisEnabled = options.redisEnabled !== false;
    this.redisOptions = options.redis || {};

    // Algorithm implementations
    this.algorithms = {
      token_bucket: new TokenBucketAlgorithm(),
      sliding_window: new SlidingWindowAlgorithm(),
      fixed_window: new FixedWindowAlgorithm(),
    };

    this.initializeRedis();
    this.startCleanup();
  }

  async initializeRedis() {
    if (!this.redisEnabled) return;

    try {
      this.redis = Redis.createClient(this.redisOptions);
      await this.redis.connect();
      console.log("Redis connected for rate limiting");
    } catch (error) {
      console.warn(
        "Redis connection failed, using local cache:",
        error.message
      );
      this.redisEnabled = false;
    }
  }

  // Rule Management
  addRule(rule) {
    this.rules.set(rule.id, rule);
    this.emit("ruleAdded", rule);
  }

  removeRule(ruleId) {
    const rule = this.rules.get(ruleId);
    if (rule) {
      this.rules.delete(ruleId);
      this.emit("ruleRemoved", rule);
    }
  }

  getRule(key) {
    for (const rule of this.rules.values()) {
      if (this.matchesKey(key, rule.keyPattern)) {
        return rule;
      }
    }
    return null;
  }

  matchesKey(key, pattern) {
    // Simple pattern matching (can be enhanced with regex)
    if (pattern.includes("{userId}")) {
      return key.startsWith(pattern.replace("{userId}", ""));
    }
    return key === pattern;
  }

  // Rate Limiting
  async checkLimit(key, endpoint = null) {
    try {
      this.stats.totalRequests++;

      const rule = this.getRule(key);
      if (!rule) {
        return new RateLimitResult(true, Infinity, Date.now());
      }

      const algorithm = this.algorithms[rule.algorithm];
      if (!algorithm) {
        throw new Error(`Unknown algorithm: ${rule.algorithm}`);
      }

      const result = await algorithm.checkLimit(key, rule, this);

      if (result.allowed) {
        this.stats.allowedRequests++;
      } else {
        this.stats.blockedRequests++;
        this.stats.violations++;
        this.emit("rateLimitViolation", { key, rule, result });
      }

      return result;
    } catch (error) {
      console.error("Rate limit check error:", error);
      // Fail open - allow request if rate limiter fails
      return new RateLimitResult(true, Infinity, Date.now());
    }
  }

  async consumeLimit(key, endpoint = null) {
    const result = await this.checkLimit(key, endpoint);

    if (result.allowed) {
      const rule = this.getRule(key);
      if (rule) {
        const algorithm = this.algorithms[rule.algorithm];
        await algorithm.consumeLimit(key, rule, this);
      }
    }

    return result;
  }

  // Redis Operations
  async getRedisValue(key) {
    if (!this.redisEnabled) return null;

    try {
      const value = await this.redis.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error("Redis get error:", error);
      return null;
    }
  }

  async setRedisValue(key, value, ttl = null) {
    if (!this.redisEnabled) return false;

    try {
      const serialized = JSON.stringify(value);
      if (ttl) {
        await this.redis.setEx(key, ttl, serialized);
      } else {
        await this.redis.set(key, serialized);
      }
      return true;
    } catch (error) {
      console.error("Redis set error:", error);
      return false;
    }
  }

  async incrementRedisValue(key, ttl = null) {
    if (!this.redisEnabled) return 0;

    try {
      const result = await this.redis.incr(key);
      if (ttl && result === 1) {
        await this.redis.expire(key, ttl);
      }
      return result;
    } catch (error) {
      console.error("Redis increment error:", error);
      return 0;
    }
  }

  // Local Cache Operations
  getLocalValue(key) {
    return this.localCache.get(key);
  }

  setLocalValue(key, value, ttl = null) {
    this.localCache.set(key, value);

    if (ttl) {
      setTimeout(() => {
        this.localCache.delete(key);
      }, ttl * 1000);
    }
  }

  // Statistics
  getStats() {
    return {
      ...this.stats,
      rulesCount: this.rules.size,
      cacheSize: this.localCache.size,
      redisEnabled: this.redisEnabled,
      timestamp: new Date(),
    };
  }

  // Cleanup
  startCleanup() {
    setInterval(() => {
      this.cleanupExpiredEntries();
    }, 60000); // Cleanup every minute
  }

  cleanupExpiredEntries() {
    const now = Date.now();
    const expiredKeys = [];

    for (const [key, value] of this.localCache) {
      if (value.expiresAt && value.expiresAt < now) {
        expiredKeys.push(key);
      }
    }

    expiredKeys.forEach((key) => this.localCache.delete(key));
  }

  generateID() {
    return uuidv4();
  }
}
```

### Algorithm Implementations

```javascript
// Token Bucket Algorithm
class TokenBucketAlgorithm {
  async checkLimit(key, rule, service) {
    const state = await this.getState(key, rule, service);
    const now = Date.now();

    // Refill tokens
    const timePassed = (now - state.lastRefill) / 1000;
    const tokensToAdd = Math.floor(timePassed * (rule.limit / rule.window));

    if (tokensToAdd > 0) {
      state.tokens = Math.min(
        rule.burst || rule.limit,
        state.tokens + tokensToAdd
      );
      state.lastRefill = now;
    }

    const allowed = state.tokens > 0;
    const remaining = Math.max(0, state.tokens - 1);
    const resetTime = now + rule.window * 1000;

    await this.setState(key, state, service);

    return new RateLimitResult(allowed, remaining, resetTime);
  }

  async consumeLimit(key, rule, service) {
    const state = await this.getState(key, rule, service);
    if (state.tokens > 0) {
      state.tokens--;
      await this.setState(key, state, service);
    }
  }

  async getState(key, rule, service) {
    const redisKey = `rate_limit:${rule.algorithm}:${key}`;

    // Try Redis first
    let state = await service.getRedisValue(redisKey);

    if (!state) {
      // Try local cache
      state = service.getLocalValue(redisKey);
    }

    if (!state) {
      // Create new state
      state = new RateLimitState(key, rule);
    }

    return state;
  }

  async setState(key, state, service) {
    const redisKey = `rate_limit:${rule.algorithm}:${key}`;
    const ttl = rule.window * 2; // Keep state for 2x window

    // Set in Redis
    await service.setRedisValue(redisKey, state, ttl);

    // Set in local cache
    service.setLocalValue(redisKey, state, ttl);
  }
}

// Sliding Window Algorithm
class SlidingWindowAlgorithm {
  async checkLimit(key, rule, service) {
    const now = Date.now();
    const windowStart = now - rule.window * 1000;

    const redisKey = `rate_limit:${rule.algorithm}:${key}`;

    // Get current requests
    const requests = await this.getRequests(key, rule, service);

    // Remove old requests
    const validRequests = requests.filter(
      (timestamp) => timestamp > windowStart
    );

    const allowed = validRequests.length < rule.limit;
    const remaining = Math.max(0, rule.limit - validRequests.length);
    const resetTime = now + rule.window * 1000;

    return new RateLimitResult(allowed, remaining, resetTime);
  }

  async consumeLimit(key, rule, service) {
    const now = Date.now();
    const redisKey = `rate_limit:${rule.algorithm}:${key}`;

    // Add current request timestamp
    await service.redis?.lPush(redisKey, now.toString());
    await service.redis?.expire(redisKey, rule.window);

    // Also add to local cache
    const localKey = `local:${redisKey}`;
    let requests = service.getLocalValue(localKey) || [];
    requests.push(now);
    service.setLocalValue(localKey, requests, rule.window);
  }

  async getRequests(key, rule, service) {
    const redisKey = `rate_limit:${rule.algorithm}:${key}`;

    // Try Redis first
    if (service.redisEnabled) {
      try {
        const requests = await service.redis.lRange(redisKey, 0, -1);
        return requests.map((timestamp) => parseInt(timestamp));
      } catch (error) {
        console.error("Redis lRange error:", error);
      }
    }

    // Fallback to local cache
    const localKey = `local:${redisKey}`;
    return service.getLocalValue(localKey) || [];
  }
}

// Fixed Window Algorithm
class FixedWindowAlgorithm {
  async checkLimit(key, rule, service) {
    const now = Date.now();
    const windowStart =
      Math.floor(now / (rule.window * 1000)) * (rule.window * 1000);
    const windowKey = `${key}:${windowStart}`;

    const redisKey = `rate_limit:${rule.algorithm}:${windowKey}`;

    // Get current count
    let count = await service.getRedisValue(redisKey);

    if (count === null) {
      // Try local cache
      count = service.getLocalValue(redisKey) || 0;
    }

    const allowed = count < rule.limit;
    const remaining = Math.max(0, rule.limit - count);
    const resetTime = windowStart + rule.window * 1000;

    return new RateLimitResult(allowed, remaining, resetTime);
  }

  async consumeLimit(key, rule, service) {
    const now = Date.now();
    const windowStart =
      Math.floor(now / (rule.window * 1000)) * (rule.window * 1000);
    const windowKey = `${key}:${windowStart}`;

    const redisKey = `rate_limit:${rule.algorithm}:${windowKey}`;

    // Increment count
    const count = await service.incrementRedisValue(redisKey, rule.window);

    // Also update local cache
    service.setLocalValue(redisKey, count, rule.window);
  }
}
```

## Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { RateLimiterService } = require("./services/RateLimiterService");

class RateLimiterAPI {
  constructor() {
    this.app = express();
    this.rateLimiter = new RateLimiterService();

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
    // Rate limiting middleware
    this.app.use("/api/*", this.rateLimitMiddleware.bind(this));

    // Rate limit management
    this.app.get("/api/rate-limit/check", this.checkRateLimit.bind(this));
    this.app.post("/api/rate-limit/consume", this.consumeRateLimit.bind(this));
    this.app.get(
      "/api/rate-limit/status/:key",
      this.getRateLimitStatus.bind(this)
    );

    // Rule management
    this.app.get("/api/rate-limit/rules", this.getRules.bind(this));
    this.app.post("/api/rate-limit/rules", this.createRule.bind(this));
    this.app.put("/api/rate-limit/rules/:ruleId", this.updateRule.bind(this));
    this.app.delete(
      "/api/rate-limit/rules/:ruleId",
      this.deleteRule.bind(this)
    );

    // Monitoring
    this.app.get("/api/rate-limit/stats", this.getStats.bind(this));
    this.app.get("/api/rate-limit/violations", this.getViolations.bind(this));
    this.app.get("/api/rate-limit/health", this.getHealth.bind(this));

    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        rateLimiterStats: this.rateLimiter.getStats(),
      });
    });
  }

  setupEventHandlers() {
    this.rateLimiter.on("rateLimitViolation", (data) => {
      console.log(`Rate limit violation: ${data.key} - ${data.rule.name}`);
    });

    this.rateLimiter.on("ruleAdded", (rule) => {
      console.log(`Rate limit rule added: ${rule.name}`);
    });
  }

  // Rate Limiting Middleware
  async rateLimitMiddleware(req, res, next) {
    try {
      const key = this.extractKey(req);
      const result = await this.rateLimiter.checkLimit(key, req.path);

      // Add rate limit headers
      res.set({
        "X-RateLimit-Limit": result.remaining + (result.allowed ? 1 : 0),
        "X-RateLimit-Remaining": result.remaining,
        "X-RateLimit-Reset": new Date(result.resetTime).toISOString(),
        "X-RateLimit-Retry-After": result.retryAfter,
      });

      if (!result.allowed) {
        return res.status(429).json({
          error: "Rate limit exceeded",
          retryAfter: result.retryAfter,
          resetTime: new Date(result.resetTime).toISOString(),
        });
      }

      next();
    } catch (error) {
      console.error("Rate limit middleware error:", error);
      next(); // Fail open
    }
  }

  // HTTP Handlers
  async checkRateLimit(req, res) {
    try {
      const { key, endpoint } = req.query;

      if (!key) {
        return res.status(400).json({ error: "Key parameter required" });
      }

      const result = await this.rateLimiter.checkLimit(key, endpoint);

      res.json({
        allowed: result.allowed,
        remaining: result.remaining,
        resetTime: result.resetTime,
        retryAfter: result.retryAfter,
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async consumeRateLimit(req, res) {
    try {
      const { key, endpoint } = req.body;

      if (!key) {
        return res.status(400).json({ error: "Key parameter required" });
      }

      const result = await this.rateLimiter.consumeLimit(key, endpoint);

      res.json({
        allowed: result.allowed,
        remaining: result.remaining,
        resetTime: result.resetTime,
        retryAfter: result.retryAfter,
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getRules(req, res) {
    try {
      const rules = Array.from(this.rateLimiter.rules.values());

      res.json({
        success: true,
        data: rules,
        count: rules.length,
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async createRule(req, res) {
    try {
      const { name, keyPattern, algorithm, limit, window, burst } = req.body;

      if (!name || !keyPattern || !algorithm || !limit || !window) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const rule = new RateLimitRule(
        name,
        keyPattern,
        algorithm,
        limit,
        window,
        burst
      );
      this.rateLimiter.addRule(rule);

      res.status(201).json({
        success: true,
        data: rule,
      });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getStats(req, res) {
    try {
      const stats = this.rateLimiter.getStats();

      res.json({
        success: true,
        data: stats,
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  // Utility Methods
  extractKey(req) {
    // Extract user ID from JWT token, API key, or IP
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith("Bearer ")) {
      // In production, decode JWT and extract user ID
      return `user:${req.ip}`; // Simplified for demo
    }

    const apiKey = req.headers["x-api-key"];
    if (apiKey) {
      return `api:${apiKey}`;
    }

    return `ip:${req.ip}`;
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Rate Limiter API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new RateLimiterAPI();
  api.start(3000);
}

module.exports = { RateLimiterAPI };
```

## Key Features

### Multiple Algorithms

- **Token Bucket**: Smooth rate limiting with burst capacity
- **Sliding Window**: Precise rate limiting with rolling window
- **Fixed Window**: Simple rate limiting with fixed time windows

### Distributed Support

- Redis integration for multi-instance coordination
- Local cache fallback for high availability
- Graceful degradation without Redis

### Flexible Configuration

- Pattern-based rule matching
- Multiple rate limit rules per application
- Dynamic rule management via API

### Performance & Monitoring

- Sub-millisecond latency overhead
- Real-time statistics and monitoring
- Violation tracking and alerting

## Extension Ideas

### Advanced Features

1. **Machine Learning**: Adaptive rate limiting based on usage patterns
2. **Geographic Limiting**: Different limits based on user location
3. **Dynamic Limits**: Adjust limits based on system load
4. **Priority Queuing**: Different limits for different user tiers
5. **Circuit Breaker**: Automatic service protection

### Enterprise Features

1. **Multi-tenant Support**: Isolated rate limits per tenant
2. **Advanced Analytics**: Detailed usage analytics and reporting
3. **Integration**: Webhook notifications for violations
4. **Compliance**: Audit trails and compliance reporting
5. **Custom Algorithms**: Plugin system for custom rate limiting logic

## **Follow-up Questions**

### **1. How would you implement adaptive rate limiting based on system load?**

**Answer:**

```javascript
class AdaptiveRateLimiter {
  constructor() {
    this.systemMetrics = new SystemMetricsCollector();
    this.adaptiveRules = new Map();
    this.loadThresholds = {
      low: 0.3,
      medium: 0.6,
      high: 0.8,
      critical: 0.95,
    };
  }

  async adjustRateLimits() {
    const systemLoad = await this.systemMetrics.getCurrentLoad();
    const loadLevel = this.determineLoadLevel(systemLoad);

    // Adjust rate limits based on system load
    for (const [ruleId, rule] of this.adaptiveRules) {
      const adjustedRule = await this.adjustRuleForLoad(rule, loadLevel);
      await this.updateRule(ruleId, adjustedRule);
    }

    return {
      loadLevel,
      adjustedRules: this.adaptiveRules.size,
      timestamp: new Date(),
    };
  }

  determineLoadLevel(load) {
    if (load >= this.loadThresholds.critical) return "critical";
    if (load >= this.loadThresholds.high) return "high";
    if (load >= this.loadThresholds.medium) return "medium";
    if (load >= this.loadThresholds.low) return "low";
    return "minimal";
  }

  async adjustRuleForLoad(rule, loadLevel) {
    const adjustmentFactors = {
      minimal: 1.5, // Increase limits by 50%
      low: 1.2, // Increase limits by 20%
      medium: 1.0, // Keep original limits
      high: 0.7, // Reduce limits by 30%
      critical: 0.4, // Reduce limits by 60%
    };

    const factor = adjustmentFactors[loadLevel];

    return {
      ...rule,
      limit: Math.floor(rule.originalLimit * factor),
      burst: rule.burst ? Math.floor(rule.burst * factor) : undefined,
      adjustedAt: new Date(),
      loadLevel,
    };
  }

  async getCurrentLoad() {
    const metrics = await this.systemMetrics.collect();

    // Calculate weighted system load
    const cpuWeight = 0.4;
    const memoryWeight = 0.3;
    const networkWeight = 0.2;
    const diskWeight = 0.1;

    const load =
      metrics.cpu * cpuWeight +
      metrics.memory * memoryWeight +
      metrics.network * networkWeight +
      metrics.disk * diskWeight;

    return Math.min(load, 1.0);
  }
}

class SystemMetricsCollector {
  constructor() {
    this.metrics = {
      cpu: 0,
      memory: 0,
      network: 0,
      disk: 0,
      activeConnections: 0,
    };
  }

  async collect() {
    // Collect CPU usage
    this.metrics.cpu = await this.getCPUUsage();

    // Collect memory usage
    this.metrics.memory = await this.getMemoryUsage();

    // Collect network usage
    this.metrics.network = await this.getNetworkUsage();

    // Collect disk usage
    this.metrics.disk = await this.getDiskUsage();

    // Collect active connections
    this.metrics.activeConnections = await this.getActiveConnections();

    return this.metrics;
  }

  async getCPUUsage() {
    // Use os.cpus() or process.cpuUsage()
    const cpus = require("os").cpus();
    let totalIdle = 0;
    let totalTick = 0;

    cpus.forEach((cpu) => {
      for (const type in cpu.times) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    });

    return 1 - totalIdle / totalTick;
  }

  async getMemoryUsage() {
    const memUsage = process.memoryUsage();
    const totalMem = require("os").totalmem();
    return memUsage.heapUsed / totalMem;
  }

  async getNetworkUsage() {
    // Simplified network usage calculation
    return 0.1; // Placeholder
  }

  async getDiskUsage() {
    // Use fs.stat or similar to get disk usage
    return 0.2; // Placeholder
  }

  async getActiveConnections() {
    // Get active HTTP connections
    return 100; // Placeholder
  }
}
```

### **2. How to implement geographic rate limiting and location-based rules?**

**Answer:**

```javascript
class GeographicRateLimiter {
  constructor() {
    this.geoRules = new Map();
    this.locationCache = new Map();
    this.geoIPService = new GeoIPService();
    this.countryRules = new Map();
  }

  async checkGeographicLimit(key, request) {
    // Get user location
    const location = await this.getUserLocation(request.ip);

    // Get applicable rules for location
    const rules = await this.getLocationRules(location);

    // Apply location-specific rate limiting
    const result = await this.applyGeographicRules(key, rules, location);

    return {
      ...result,
      location,
      appliedRules: rules.map((r) => r.id),
    };
  }

  async getUserLocation(ip) {
    // Check cache first
    if (this.locationCache.has(ip)) {
      const cached = this.locationCache.get(ip);
      if (!this.isLocationExpired(cached)) {
        return cached.location;
      }
    }

    // Get location from GeoIP service
    const location = await this.geoIPService.getLocation(ip);

    // Cache location
    this.locationCache.set(ip, {
      location,
      timestamp: new Date(),
    });

    return location;
  }

  async getLocationRules(location) {
    const rules = [];

    // Country-level rules
    const countryRules = this.countryRules.get(location.country);
    if (countryRules) {
      rules.push(...countryRules);
    }

    // Region-level rules
    const regionKey = `${location.country}_${location.region}`;
    const regionRules = this.geoRules.get(regionKey);
    if (regionRules) {
      rules.push(...regionRules);
    }

    // City-level rules
    const cityKey = `${location.country}_${location.region}_${location.city}`;
    const cityRules = this.geoRules.get(cityKey);
    if (cityRules) {
      rules.push(...cityRules);
    }

    return rules;
  }

  async applyGeographicRules(key, rules, location) {
    let allowed = true;
    let remaining = Infinity;
    let resetTime = Date.now() + 3600000; // Default 1 hour

    for (const rule of rules) {
      const result = await this.checkRule(key, rule);

      if (!result.allowed) {
        allowed = false;
        remaining = Math.min(remaining, result.remaining);
        resetTime = Math.min(resetTime, result.resetTime);
      }
    }

    return {
      allowed,
      remaining: allowed ? Math.min(remaining, 1000) : remaining,
      resetTime,
      location: {
        country: location.country,
        region: location.region,
        city: location.city,
      },
    };
  }

  async addCountryRule(country, rule) {
    if (!this.countryRules.has(country)) {
      this.countryRules.set(country, []);
    }

    this.countryRules.get(country).push({
      ...rule,
      id: uuidv4(),
      type: "country",
      country,
      createdAt: new Date(),
    });
  }

  async addRegionRule(country, region, rule) {
    const key = `${country}_${region}`;
    if (!this.geoRules.has(key)) {
      this.geoRules.set(key, []);
    }

    this.geoRules.get(key).push({
      ...rule,
      id: uuidv4(),
      type: "region",
      country,
      region,
      createdAt: new Date(),
    });
  }

  async addCityRule(country, region, city, rule) {
    const key = `${country}_${region}_${city}`;
    if (!this.geoRules.has(key)) {
      this.geoRules.set(key, []);
    }

    this.geoRules.get(key).push({
      ...rule,
      id: uuidv4(),
      type: "city",
      country,
      region,
      city,
      createdAt: new Date(),
    });
  }

  isLocationExpired(cached) {
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    return Date.now() - cached.timestamp.getTime() > maxAge;
  }
}

class GeoIPService {
  constructor() {
    this.providers = [
      new MaxMindProvider(),
      new IPStackProvider(),
      new IPGeolocationProvider(),
    ];
  }

  async getLocation(ip) {
    for (const provider of this.providers) {
      try {
        const location = await provider.getLocation(ip);
        if (location) {
          return location;
        }
      } catch (error) {
        console.error(`GeoIP provider ${provider.name} failed:`, error);
      }
    }

    // Fallback to default location
    return {
      country: "US",
      region: "Unknown",
      city: "Unknown",
      latitude: 0,
      longitude: 0,
      timezone: "UTC",
    };
  }
}
```

### **3. How to implement priority-based rate limiting for different user tiers?**

**Answer:**

```javascript
class PriorityBasedRateLimiter {
  constructor() {
    this.userTiers = new Map();
    this.tierRules = new Map();
    this.priorityQueues = new Map();
    this.tierLimits = {
      free: { requests: 100, window: 60, priority: 1 },
      premium: { requests: 1000, window: 60, priority: 2 },
      enterprise: { requests: 10000, window: 60, priority: 3 },
      vip: { requests: 50000, window: 60, priority: 4 },
    };
  }

  async checkPriorityLimit(userId, request) {
    const userTier = await this.getUserTier(userId);
    const tierConfig = this.tierLimits[userTier];

    if (!tierConfig) {
      throw new Error(`Invalid user tier: ${userTier}`);
    }

    // Check if user has exceeded tier limits
    const tierUsage = await this.getTierUsage(userId, userTier);

    if (tierUsage.requests >= tierConfig.requests) {
      // Check if user can use priority queue
      if (await this.canUsePriorityQueue(userId, userTier)) {
        return await this.handlePriorityQueue(userId, request, userTier);
      }

      return {
        allowed: false,
        reason: "tier_limit_exceeded",
        tier: userTier,
        remaining: 0,
        resetTime: tierUsage.resetTime,
      };
    }

    // Allow request and update usage
    await this.updateTierUsage(userId, userTier);

    return {
      allowed: true,
      tier: userTier,
      remaining: tierConfig.requests - tierUsage.requests - 1,
      resetTime: tierUsage.resetTime,
    };
  }

  async getUserTier(userId) {
    if (this.userTiers.has(userId)) {
      return this.userTiers.get(userId);
    }

    // Default to free tier
    const tier = "free";
    this.userTiers.set(userId, tier);
    return tier;
  }

  async getTierUsage(userId, tier) {
    const key = `tier_usage:${tier}:${userId}`;
    const usage = await this.redis.hgetall(key);

    if (!usage.requests) {
      return {
        requests: 0,
        resetTime: Date.now() + this.tierLimits[tier].window * 1000,
      };
    }

    return {
      requests: parseInt(usage.requests),
      resetTime: parseInt(usage.resetTime),
    };
  }

  async updateTierUsage(userId, tier) {
    const key = `tier_usage:${tier}:${userId}`;
    const window = this.tierLimits[tier].window * 1000;
    const resetTime = Date.now() + window;

    await this.redis.hincrby(key, "requests", 1);
    await this.redis.hset(key, "resetTime", resetTime);
    await this.redis.expire(key, this.tierLimits[tier].window);
  }

  async canUsePriorityQueue(userId, tier) {
    const tierConfig = this.tierLimits[tier];

    // Only premium and above can use priority queue
    if (tierConfig.priority < 2) {
      return false;
    }

    // Check if priority queue has capacity
    const queueSize = await this.getPriorityQueueSize();
    const maxQueueSize = 1000; // Configurable

    return queueSize < maxQueueSize;
  }

  async handlePriorityQueue(userId, request, tier) {
    const priority = this.tierLimits[tier].priority;
    const queueItem = {
      id: uuidv4(),
      userId,
      request,
      tier,
      priority,
      queuedAt: new Date(),
      estimatedWaitTime: await this.estimateWaitTime(priority),
    };

    // Add to priority queue
    await this.addToPriorityQueue(queueItem);

    return {
      allowed: false,
      reason: "queued_for_processing",
      queueId: queueItem.id,
      estimatedWaitTime: queueItem.estimatedWaitTime,
      position: await this.getQueuePosition(queueItem.id),
    };
  }

  async addToPriorityQueue(queueItem) {
    const key = `priority_queue:${queueItem.priority}`;
    await this.redis.lpush(key, JSON.stringify(queueItem));

    // Set expiration for queue item
    await this.redis.expire(key, 3600); // 1 hour
  }

  async estimateWaitTime(priority) {
    const queueSizes = await this.getQueueSizesByPriority();
    let waitTime = 0;

    // Calculate wait time based on higher priority queues
    for (let p = priority + 1; p <= 4; p++) {
      const queueSize = queueSizes[p] || 0;
      waitTime += queueSize * 100; // 100ms per request
    }

    return waitTime;
  }

  async getQueuePosition(queueId) {
    // Find position in priority queue
    const queueItem = await this.findQueueItem(queueId);
    if (!queueItem) return -1;

    const key = `priority_queue:${queueItem.priority}`;
    const items = await this.redis.lrange(key, 0, -1);

    for (let i = 0; i < items.length; i++) {
      const item = JSON.parse(items[i]);
      if (item.id === queueId) {
        return i + 1;
      }
    }

    return -1;
  }

  async processPriorityQueue() {
    // Process queues in priority order (highest first)
    for (let priority = 4; priority >= 1; priority--) {
      const key = `priority_queue:${priority}`;
      const item = await this.redis.rpop(key);

      if (item) {
        const queueItem = JSON.parse(item);
        await this.processQueuedRequest(queueItem);
        break; // Process one item at a time
      }
    }
  }

  async processQueuedRequest(queueItem) {
    try {
      // Process the queued request
      const result = await this.processRequest(queueItem.request);

      // Notify user that request was processed
      await this.notifyUser(queueItem.userId, {
        queueId: queueItem.id,
        status: "processed",
        result,
      });
    } catch (error) {
      // Handle processing error
      await this.notifyUser(queueItem.userId, {
        queueId: queueItem.id,
        status: "failed",
        error: error.message,
      });
    }
  }
}
```
