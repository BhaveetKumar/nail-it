---
# Auto-generated front matter
Title: System Design Concepts Guide
LastUpdated: 2025-11-06T20:45:59.108682
Tags: []
Status: draft
---

# 🏗️ System Design Concepts: Complete Guide with Node.js

> **Master system design for large-scale applications with Node.js**

## 🎯 **Learning Objectives**

- Master system design principles and patterns
- Design scalable and reliable systems with Node.js
- Understand microservices and distributed systems
- Learn database design and optimization
- Prepare for system design interviews at top companies

## 📚 **Table of Contents**

1. [System Design Fundamentals](#system-design-fundamentals)
2. [Scalability Patterns](#scalability-patterns)
3. [Microservices Architecture](#microservices-architecture)
4. [Database Design](#database-design)
5. [Caching Strategies](#caching-strategies)
6. [Load Balancing](#load-balancing)
7. [Message Queues](#message-queues)
8. [API Design](#api-design)
9. [Security & Authentication](#security--authentication)
10. [Monitoring & Observability](#monitoring--observability)
11. [Interview Questions](#interview-questions)

---

## 🚀 **System Design Fundamentals**

### **What is System Design?**

System design is the process of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. It involves making high-level decisions about the system's structure and behavior.

### **Key Principles**

1. **Scalability**: System can handle increased load
2. **Reliability**: System works correctly and consistently
3. **Availability**: System is accessible when needed
4. **Performance**: System responds quickly
5. **Maintainability**: System is easy to modify and extend
6. **Security**: System protects data and resources

### **System Design Process**

```javascript
// System Design Process Example
class SystemDesignProcess {
    constructor() {
        this.steps = [
            'Requirements Gathering',
            'Capacity Estimation',
            'High-Level Design',
            'Detailed Design',
            'Identify Bottlenecks',
            'Scale the Design'
        ];
    }
    
    gatherRequirements(useCase) {
        return {
            functional: this.getFunctionalRequirements(useCase),
            nonFunctional: this.getNonFunctionalRequirements(useCase),
            constraints: this.getConstraints(useCase)
        };
    }
    
    estimateCapacity(requirements) {
        const { users, requestsPerSecond, dataSize } = requirements;
        
        return {
            dailyActiveUsers: users,
            requestsPerSecond: requestsPerSecond,
            dataPerRequest: dataSize,
            totalDataPerDay: users * requestsPerSecond * 86400 * dataSize,
            storageNeeded: this.calculateStorage(requirements)
        };
    }
    
    calculateStorage(requirements) {
        // Calculate storage requirements
        const { users, dataPerUser, retentionDays } = requirements;
        return users * dataPerUser * retentionDays;
    }
}
```

---

## 📈 **Scalability Patterns**

### **Horizontal vs Vertical Scaling**

```javascript
// Horizontal Scaling with Node.js
class HorizontalScaling {
    constructor() {
        this.instances = [];
        this.loadBalancer = new LoadBalancer();
    }
    
    addInstance(instance) {
        this.instances.push(instance);
        this.loadBalancer.addServer(instance);
    }
    
    removeInstance(instanceId) {
        this.instances = this.instances.filter(inst => inst.id !== instanceId);
        this.loadBalancer.removeServer(instanceId);
    }
    
    distributeLoad(request) {
        return this.loadBalancer.route(request);
    }
}

// Vertical Scaling Example
class VerticalScaling {
    constructor() {
        this.cpu = 2;
        this.memory = 4; // GB
        this.storage = 100; // GB
    }
    
    scaleUp() {
        this.cpu *= 2;
        this.memory *= 2;
        this.storage *= 2;
    }
    
    scaleDown() {
        this.cpu = Math.max(1, this.cpu / 2);
        this.memory = Math.max(1, this.memory / 2);
        this.storage = Math.max(10, this.storage / 2);
    }
}
```

### **Database Sharding**

```javascript
// Database Sharding Implementation
class DatabaseSharding {
    constructor() {
        this.shards = new Map();
        this.shardCount = 4;
        this.initializeShards();
    }
    
    initializeShards() {
        for (let i = 0; i < this.shardCount; i++) {
            this.shards.set(i, new DatabaseShard(i));
        }
    }
    
    getShard(key) {
        const hash = this.hashFunction(key);
        const shardId = hash % this.shardCount;
        return this.shards.get(shardId);
    }
    
    hashFunction(key) {
        let hash = 0;
        for (let i = 0; i < key.length; i++) {
            const char = key.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }
    
    async write(key, value) {
        const shard = this.getShard(key);
        return await shard.write(key, value);
    }
    
    async read(key) {
        const shard = this.getShard(key);
        return await shard.read(key);
    }
}

class DatabaseShard {
    constructor(id) {
        this.id = id;
        this.data = new Map();
    }
    
    async write(key, value) {
        this.data.set(key, value);
        return { success: true, shardId: this.id };
    }
    
    async read(key) {
        return this.data.get(key);
    }
}
```

---

## 🏗️ **Microservices Architecture**

### **Microservices with Node.js**

```javascript
// User Service
class UserService {
    constructor() {
        this.express = require('express');
        this.app = this.express();
        this.port = process.env.USER_SERVICE_PORT || 3001;
        this.database = new UserDatabase();
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.use(this.express.json());
        
        // Get user by ID
        this.app.get('/users/:id', async (req, res) => {
            try {
                const user = await this.database.getUser(req.params.id);
                if (!user) {
                    return res.status(404).json({ error: 'User not found' });
                }
                res.json(user);
            } catch (error) {
                res.status(500).json({ error: 'Internal server error' });
            }
        });
        
        // Create user
        this.app.post('/users', async (req, res) => {
            try {
                const user = await this.database.createUser(req.body);
                res.status(201).json(user);
            } catch (error) {
                res.status(400).json({ error: error.message });
            }
        });
        
        // Update user
        this.app.put('/users/:id', async (req, res) => {
            try {
                const user = await this.database.updateUser(req.params.id, req.body);
                res.json(user);
            } catch (error) {
                res.status(400).json({ error: error.message });
            }
        });
        
        // Delete user
        this.app.delete('/users/:id', async (req, res) => {
            try {
                await this.database.deleteUser(req.params.id);
                res.status(204).send();
            } catch (error) {
                res.status(500).json({ error: 'Internal server error' });
            }
        });
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`User Service running on port ${this.port}`);
        });
    }
}

// Order Service
class OrderService {
    constructor() {
        this.express = require('express');
        this.app = this.express();
        this.port = process.env.ORDER_SERVICE_PORT || 3002;
        this.database = new OrderDatabase();
        this.userService = new UserServiceClient();
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.use(this.express.json());
        
        // Create order
        this.app.post('/orders', async (req, res) => {
            try {
                const { userId, items } = req.body;
                
                // Validate user exists
                const user = await this.userService.getUser(userId);
                if (!user) {
                    return res.status(400).json({ error: 'User not found' });
                }
                
                const order = await this.database.createOrder({ userId, items });
                res.status(201).json(order);
            } catch (error) {
                res.status(400).json({ error: error.message });
            }
        });
        
        // Get orders by user
        this.app.get('/orders/user/:userId', async (req, res) => {
            try {
                const orders = await this.database.getOrdersByUser(req.params.userId);
                res.json(orders);
            } catch (error) {
                res.status(500).json({ error: 'Internal server error' });
            }
        });
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`Order Service running on port ${this.port}`);
        });
    }
}

// API Gateway
class APIGateway {
    constructor() {
        this.express = require('express');
        this.app = this.express();
        this.port = process.env.API_GATEWAY_PORT || 3000;
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.use(this.express.json());
        
        // Route to user service
        this.app.use('/api/users', this.createProxy('http://localhost:3001'));
        
        // Route to order service
        this.app.use('/api/orders', this.createProxy('http://localhost:3002'));
        
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ status: 'healthy', timestamp: new Date().toISOString() });
        });
    }
    
    createProxy(target) {
        const { createProxyMiddleware } = require('http-proxy-middleware');
        return createProxyMiddleware({
            target,
            changeOrigin: true,
            onError: (err, req, res) => {
                res.status(500).json({ error: 'Service unavailable' });
            }
        });
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`API Gateway running on port ${this.port}`);
        });
    }
}
```

### **Service Discovery**

```javascript
// Service Discovery with Consul
class ServiceDiscovery {
    constructor() {
        this.consul = require('consul')();
        this.services = new Map();
    }
    
    async registerService(serviceName, serviceId, address, port) {
        const service = {
            name: serviceName,
            id: serviceId,
            address: address,
            port: port,
            check: {
                http: `http://${address}:${port}/health`,
                interval: '10s'
            }
        };
        
        await this.consul.agent.service.register(service);
        this.services.set(serviceId, service);
        console.log(`Service ${serviceName} registered`);
    }
    
    async discoverService(serviceName) {
        const services = await this.consul.health.service(serviceName);
        return services[0].Service;
    }
    
    async deregisterService(serviceId) {
        await this.consul.agent.service.deregister(serviceId);
        this.services.delete(serviceId);
        console.log(`Service ${serviceId} deregistered`);
    }
}
```

---

## 🗄️ **Database Design**

### **Database Selection Criteria**

```javascript
// Database Selection Helper
class DatabaseSelector {
    constructor() {
        this.criteria = {
            consistency: 'strong', // strong, eventual
            scalability: 'horizontal', // horizontal, vertical
            queryPattern: 'complex', // simple, complex
            dataStructure: 'relational', // relational, document, key-value, graph
            transactionSupport: true,
            readWriteRatio: 0.8 // 80% reads, 20% writes
        };
    }
    
    selectDatabase(requirements) {
        const { consistency, scalability, queryPattern, dataStructure } = requirements;
        
        if (consistency === 'strong' && dataStructure === 'relational') {
            return 'PostgreSQL';
        } else if (scalability === 'horizontal' && dataStructure === 'document') {
            return 'MongoDB';
        } else if (queryPattern === 'simple' && dataStructure === 'key-value') {
            return 'Redis';
        } else if (dataStructure === 'graph') {
            return 'Neo4j';
        } else if (scalability === 'horizontal' && dataStructure === 'column') {
            return 'Cassandra';
        }
        
        return 'PostgreSQL'; // Default
    }
}
```

### **Database Optimization**

```javascript
// Database Optimization Strategies
class DatabaseOptimizer {
    constructor(database) {
        this.database = database;
    }
    
    // Indexing
    async createIndexes() {
        // Single column index
        await this.database.createIndex('users', 'email');
        
        // Composite index
        await this.database.createIndex('orders', ['userId', 'createdAt']);
        
        // Partial index
        await this.database.createIndex('orders', 'status', { 
            where: 'status = "active"' 
        });
        
        // Text search index
        await this.database.createIndex('products', 'name', { 
            type: 'text' 
        });
    }
    
    // Query Optimization
    async optimizeQuery(query) {
        // Use EXPLAIN to analyze query
        const explain = await this.database.explain(query);
        
        // Check for missing indexes
        if (explain.missingIndexes.length > 0) {
            console.log('Missing indexes:', explain.missingIndexes);
        }
        
        // Check for full table scans
        if (explain.fullTableScans.length > 0) {
            console.log('Full table scans detected:', explain.fullTableScans);
        }
        
        return explain;
    }
    
    // Connection Pooling
    setupConnectionPool() {
        const pool = {
            min: 2,
            max: 10,
            acquireTimeoutMillis: 30000,
            createTimeoutMillis: 30000,
            destroyTimeoutMillis: 5000,
            idleTimeoutMillis: 30000,
            reapIntervalMillis: 1000,
            createRetryIntervalMillis: 200
        };
        
        return pool;
    }
    
    // Caching Strategy
    setupCaching() {
        return {
            queryCache: {
                enabled: true,
                size: 1000,
                ttl: 300 // 5 minutes
            },
            resultCache: {
                enabled: true,
                size: 5000,
                ttl: 600 // 10 minutes
            }
        };
    }
}
```

---

## 🚀 **Caching Strategies**

### **Multi-Level Caching**

```javascript
// Multi-Level Caching Implementation
class MultiLevelCache {
    constructor() {
        this.l1Cache = new Map(); // In-memory cache
        this.l2Cache = new RedisCache(); // Redis cache
        this.l3Cache = new DatabaseCache(); // Database cache
    }
    
    async get(key) {
        // L1 Cache (Fastest)
        if (this.l1Cache.has(key)) {
            console.log('L1 Cache hit');
            return this.l1Cache.get(key);
        }
        
        // L2 Cache (Fast)
        const l2Value = await this.l2Cache.get(key);
        if (l2Value) {
            console.log('L2 Cache hit');
            this.l1Cache.set(key, l2Value);
            return l2Value;
        }
        
        // L3 Cache (Slowest)
        const l3Value = await this.l3Cache.get(key);
        if (l3Value) {
            console.log('L3 Cache hit');
            this.l1Cache.set(key, l3Value);
            await this.l2Cache.set(key, l3Value);
            return l3Value;
        }
        
        console.log('Cache miss');
        return null;
    }
    
    async set(key, value, ttl = 3600) {
        // Set in all levels
        this.l1Cache.set(key, value);
        await this.l2Cache.set(key, value, ttl);
        await this.l3Cache.set(key, value, ttl);
    }
    
    async invalidate(key) {
        this.l1Cache.delete(key);
        await this.l2Cache.delete(key);
        await this.l3Cache.delete(key);
    }
}

// Cache-Aside Pattern
class CacheAsidePattern {
    constructor(cache, database) {
        this.cache = cache;
        this.database = database;
    }
    
    async get(key) {
        // Try cache first
        let value = await this.cache.get(key);
        
        if (!value) {
            // Cache miss - get from database
            value = await this.database.get(key);
            
            if (value) {
                // Store in cache for next time
                await this.cache.set(key, value);
            }
        }
        
        return value;
    }
    
    async set(key, value) {
        // Update database
        await this.database.set(key, value);
        
        // Update cache
        await this.cache.set(key, value);
    }
    
    async delete(key) {
        // Delete from database
        await this.database.delete(key);
        
        // Delete from cache
        await this.cache.delete(key);
    }
}
```

---

## ⚖️ **Load Balancing**

### **Load Balancing Algorithms**

```javascript
// Load Balancer Implementation
class LoadBalancer {
    constructor() {
        this.servers = [];
        this.currentIndex = 0;
        this.algorithm = 'roundRobin';
    }
    
    addServer(server) {
        this.servers.push(server);
    }
    
    removeServer(serverId) {
        this.servers = this.servers.filter(s => s.id !== serverId);
    }
    
    route(request) {
        switch (this.algorithm) {
            case 'roundRobin':
                return this.roundRobin(request);
            case 'leastConnections':
                return this.leastConnections(request);
            case 'weightedRoundRobin':
                return this.weightedRoundRobin(request);
            case 'ipHash':
                return this.ipHash(request);
            default:
                return this.roundRobin(request);
        }
    }
    
    roundRobin(request) {
        const server = this.servers[this.currentIndex];
        this.currentIndex = (this.currentIndex + 1) % this.servers.length;
        return server;
    }
    
    leastConnections(request) {
        return this.servers.reduce((min, server) => 
            server.connections < min.connections ? server : min
        );
    }
    
    weightedRoundRobin(request) {
        const totalWeight = this.servers.reduce((sum, server) => sum + server.weight, 0);
        let random = Math.random() * totalWeight;
        
        for (const server of this.servers) {
            random -= server.weight;
            if (random <= 0) {
                return server;
            }
        }
        
        return this.servers[0];
    }
    
    ipHash(request) {
        const ip = request.ip;
        const hash = this.hashFunction(ip);
        const index = hash % this.servers.length;
        return this.servers[index];
    }
    
    hashFunction(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
}

// Health Check Implementation
class HealthChecker {
    constructor(servers) {
        this.servers = servers;
        this.interval = 5000; // 5 seconds
        this.startHealthChecks();
    }
    
    startHealthChecks() {
        setInterval(() => {
            this.checkAllServers();
        }, this.interval);
    }
    
    async checkAllServers() {
        for (const server of this.servers) {
            try {
                const response = await fetch(`http://${server.host}:${server.port}/health`);
                if (response.ok) {
                    server.healthy = true;
                    server.lastCheck = new Date();
                } else {
                    server.healthy = false;
                }
            } catch (error) {
                server.healthy = false;
                console.error(`Health check failed for ${server.id}:`, error.message);
            }
        }
    }
    
    getHealthyServers() {
        return this.servers.filter(server => server.healthy);
    }
}
```

---

## 📨 **Message Queues**

### **Message Queue Implementation**

```javascript
// Message Queue with Redis
class MessageQueue {
    constructor(redisClient) {
        this.redis = redisClient;
        this.queues = new Map();
    }
    
    async createQueue(queueName) {
        this.queues.set(queueName, {
            name: queueName,
            consumers: [],
            messages: []
        });
    }
    
    async publish(queueName, message) {
        const queue = this.queues.get(queueName);
        if (!queue) {
            throw new Error(`Queue ${queueName} does not exist`);
        }
        
        const messageId = this.generateMessageId();
        const messageData = {
            id: messageId,
            data: message,
            timestamp: Date.now(),
            attempts: 0
        };
        
        // Store in Redis
        await this.redis.lpush(`queue:${queueName}`, JSON.stringify(messageData));
        
        // Notify consumers
        this.notifyConsumers(queueName);
        
        return messageId;
    }
    
    async subscribe(queueName, consumer) {
        const queue = this.queues.get(queueName);
        if (!queue) {
            throw new Error(`Queue ${queueName} does not exist`);
        }
        
        queue.consumers.push(consumer);
        
        // Start consuming messages
        this.startConsuming(queueName);
    }
    
    async startConsuming(queueName) {
        const queue = this.queues.get(queueName);
        
        while (true) {
            try {
                // Blocking pop with timeout
                const result = await this.redis.brpop(`queue:${queueName}`, 1);
                
                if (result) {
                    const messageData = JSON.parse(result[1]);
                    
                    // Process message with each consumer
                    for (const consumer of queue.consumers) {
                        try {
                            await consumer(messageData.data);
                        } catch (error) {
                            console.error('Consumer error:', error);
                            // Implement retry logic
                            await this.handleConsumerError(queueName, messageData, error);
                        }
                    }
                }
            } catch (error) {
                console.error('Queue error:', error);
                await this.sleep(1000); // Wait 1 second before retrying
            }
        }
    }
    
    async handleConsumerError(queueName, messageData, error) {
        messageData.attempts++;
        
        if (messageData.attempts < 3) {
            // Retry with exponential backoff
            const delay = Math.pow(2, messageData.attempts) * 1000;
            setTimeout(async () => {
                await this.redis.lpush(`queue:${queueName}`, JSON.stringify(messageData));
            }, delay);
        } else {
            // Move to dead letter queue
            await this.redis.lpush(`dlq:${queueName}`, JSON.stringify(messageData));
        }
    }
    
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Event-Driven Architecture
class EventBus {
    constructor() {
        this.events = new Map();
    }
    
    on(eventName, callback) {
        if (!this.events.has(eventName)) {
            this.events.set(eventName, []);
        }
        this.events.get(eventName).push(callback);
    }
    
    emit(eventName, data) {
        const callbacks = this.events.get(eventName) || [];
        callbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event handler for ${eventName}:`, error);
            }
        });
    }
    
    off(eventName, callback) {
        const callbacks = this.events.get(eventName) || [];
        const index = callbacks.indexOf(callback);
        if (index > -1) {
            callbacks.splice(index, 1);
        }
    }
}
```

---

## 🔐 **Security & Authentication**

### **JWT Authentication**

```javascript
// JWT Authentication Service
class JWTAuthService {
    constructor(secretKey) {
        this.jwt = require('jsonwebtoken');
        this.secretKey = secretKey;
        this.refreshTokens = new Map();
    }
    
    generateTokens(user) {
        const accessToken = this.jwt.sign(
            { userId: user.id, email: user.email },
            this.secretKey,
            { expiresIn: '15m' }
        );
        
        const refreshToken = this.jwt.sign(
            { userId: user.id, type: 'refresh' },
            this.secretKey,
            { expiresIn: '7d' }
        );
        
        // Store refresh token
        this.refreshTokens.set(refreshToken, {
            userId: user.id,
            createdAt: Date.now()
        });
        
        return { accessToken, refreshToken };
    }
    
    verifyToken(token) {
        try {
            return this.jwt.verify(token, this.secretKey);
        } catch (error) {
            throw new Error('Invalid token');
        }
    }
    
    refreshAccessToken(refreshToken) {
        if (!this.refreshTokens.has(refreshToken)) {
            throw new Error('Invalid refresh token');
        }
        
        const tokenData = this.refreshTokens.get(refreshToken);
        const user = { id: tokenData.userId };
        
        return this.generateTokens(user);
    }
    
    revokeRefreshToken(refreshToken) {
        this.refreshTokens.delete(refreshToken);
    }
}

// Rate Limiting
class RateLimiter {
    constructor() {
        this.requests = new Map();
        this.windows = new Map();
    }
    
    isAllowed(identifier, limit = 100, windowMs = 60000) {
        const now = Date.now();
        const windowStart = now - windowMs;
        
        // Clean old requests
        if (this.requests.has(identifier)) {
            const userRequests = this.requests.get(identifier);
            const validRequests = userRequests.filter(time => time > windowStart);
            this.requests.set(identifier, validRequests);
        } else {
            this.requests.set(identifier, []);
        }
        
        const userRequests = this.requests.get(identifier);
        
        if (userRequests.length >= limit) {
            return false;
        }
        
        userRequests.push(now);
        return true;
    }
}

// Input Validation
class InputValidator {
    constructor() {
        this.schemas = new Map();
    }
    
    addSchema(name, schema) {
        this.schemas.set(name, schema);
    }
    
    validate(data, schemaName) {
        const schema = this.schemas.get(schemaName);
        if (!schema) {
            throw new Error(`Schema ${schemaName} not found`);
        }
        
        const errors = [];
        
        for (const [field, rules] of Object.entries(schema)) {
            const value = data[field];
            
            if (rules.required && (value === undefined || value === null)) {
                errors.push(`${field} is required`);
                continue;
            }
            
            if (value !== undefined && value !== null) {
                if (rules.type && typeof value !== rules.type) {
                    errors.push(`${field} must be of type ${rules.type}`);
                }
                
                if (rules.minLength && value.length < rules.minLength) {
                    errors.push(`${field} must be at least ${rules.minLength} characters`);
                }
                
                if (rules.maxLength && value.length > rules.maxLength) {
                    errors.push(`${field} must be at most ${rules.maxLength} characters`);
                }
                
                if (rules.pattern && !rules.pattern.test(value)) {
                    errors.push(`${field} format is invalid`);
                }
            }
        }
        
        if (errors.length > 0) {
            throw new Error(`Validation failed: ${errors.join(', ')}`);
        }
        
        return true;
    }
}
```

---

## 📊 **Monitoring & Observability**

### **Logging System**

```javascript
// Structured Logging
class Logger {
    constructor(serviceName) {
        this.serviceName = serviceName;
        this.winston = require('winston');
        this.logger = this.winston.createLogger({
            level: process.env.LOG_LEVEL || 'info',
            format: this.winston.format.combine(
                this.winston.format.timestamp(),
                this.winston.format.errors({ stack: true }),
                this.winston.format.json()
            ),
            defaultMeta: { service: serviceName },
            transports: [
                new this.winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
                new this.winston.transports.File({ filename: 'logs/combined.log' }),
            ]
        });
        
        if (process.env.NODE_ENV !== 'production') {
            this.logger.add(new this.winston.transports.Console({
                format: this.winston.format.combine(
                    this.winston.format.colorize(),
                    this.winston.format.simple()
                )
            }));
        }
    }
    
    info(message, meta = {}) {
        this.logger.info(message, meta);
    }
    
    error(message, meta = {}) {
        this.logger.error(message, meta);
    }
    
    warn(message, meta = {}) {
        this.logger.warn(message, meta);
    }
    
    debug(message, meta = {}) {
        this.logger.debug(message, meta);
    }
}

// Metrics Collection
class MetricsCollector {
    constructor() {
        this.metrics = new Map();
        this.counters = new Map();
        this.gauges = new Map();
        this.histograms = new Map();
    }
    
    incrementCounter(name, value = 1, labels = {}) {
        const key = this.getKey(name, labels);
        const current = this.counters.get(key) || 0;
        this.counters.set(key, current + value);
    }
    
    setGauge(name, value, labels = {}) {
        const key = this.getKey(name, labels);
        this.gauges.set(key, value);
    }
    
    recordHistogram(name, value, labels = {}) {
        const key = this.getKey(name, labels);
        if (!this.histograms.has(key)) {
            this.histograms.set(key, []);
        }
        this.histograms.get(key).push(value);
    }
    
    getMetrics() {
        return {
            counters: Object.fromEntries(this.counters),
            gauges: Object.fromEntries(this.gauges),
            histograms: Object.fromEntries(this.histograms)
        };
    }
    
    getKey(name, labels) {
        const labelString = Object.entries(labels)
            .map(([k, v]) => `${k}=${v}`)
            .join(',');
        return labelString ? `${name}{${labelString}}` : name;
    }
}

// Health Check
class HealthChecker {
    constructor() {
        this.checks = new Map();
    }
    
    addCheck(name, checkFunction) {
        this.checks.set(name, checkFunction);
    }
    
    async runChecks() {
        const results = {};
        
        for (const [name, checkFunction] of this.checks) {
            try {
                const result = await checkFunction();
                results[name] = {
                    status: 'healthy',
                    result: result
                };
            } catch (error) {
                results[name] = {
                    status: 'unhealthy',
                    error: error.message
                };
            }
        }
        
        return results;
    }
    
    async getHealthStatus() {
        const checks = await this.runChecks();
        const allHealthy = Object.values(checks).every(check => check.status === 'healthy');
        
        return {
            status: allHealthy ? 'healthy' : 'unhealthy',
            checks: checks,
            timestamp: new Date().toISOString()
        };
    }
}
```

---

## 🎯 **Interview Questions**

### **1. How do you design a URL shortener like bit.ly?**

**Answer:**
- **Requirements**: Shorten URLs, redirect to original, handle high traffic
- **Capacity**: 100M URLs/day, 1B reads/day, 10 years storage
- **API Design**: POST /shorten, GET /{shortCode}
- **Database**: SQL for metadata, NoSQL for analytics
- **Caching**: Redis for hot URLs
- **Encoding**: Base62 encoding for short codes
- **Scaling**: Horizontal scaling with load balancers

### **2. How do you design a chat system like WhatsApp?**

**Answer:**
- **Requirements**: Real-time messaging, group chats, file sharing
- **Architecture**: Microservices with message queues
- **Real-time**: WebSocket connections
- **Storage**: Messages in database, files in object storage
- **Scaling**: Horizontal scaling with message queues
- **Security**: End-to-end encryption

### **3. How do you design a social media feed like Facebook?**

**Answer:**
- **Requirements**: Personalized feed, real-time updates, high read/write ratio
- **Architecture**: Microservices with event-driven architecture
- **Feed Generation**: Pre-computed feeds with caching
- **Storage**: User data in SQL, posts in NoSQL, media in CDN
- **Scaling**: Horizontal scaling with sharding
- **Caching**: Multi-level caching strategy

### **4. How do you handle database scaling in a high-traffic application?**

**Answer:**
- **Read Replicas**: Distribute read queries across replicas
- **Sharding**: Partition data across multiple databases
- **Caching**: Use Redis for frequently accessed data
- **Connection Pooling**: Optimize database connections
- **Query Optimization**: Index optimization and query analysis
- **Monitoring**: Track database performance metrics

### **5. How do you design a distributed system for handling millions of requests?**

**Answer:**
- **Load Balancing**: Distribute requests across multiple servers
- **Microservices**: Break down monolithic application
- **Message Queues**: Decouple services with async communication
- **Caching**: Multi-level caching strategy
- **Database Scaling**: Read replicas and sharding
- **Monitoring**: Comprehensive observability and alerting

---

**🎉 System design is crucial for building scalable and reliable applications!**


## Api Design

<!-- AUTO-GENERATED ANCHOR: originally referenced as #api-design -->

Placeholder content. Please replace with proper section.
