# Node.js Backend Engineering

Comprehensive Node.js backend engineering guide for senior developers.

## 🎯 Node.js Fundamentals

### Event Loop and Asynchronous Programming
```javascript
// Event Loop Deep Dive
const { performance } = require('perf_hooks');

console.log('1. Synchronous code starts');

setTimeout(() => {
    console.log('4. setTimeout callback');
}, 0);

setImmediate(() => {
    console.log('5. setImmediate callback');
});

process.nextTick(() => {
    console.log('3. nextTick callback');
});

Promise.resolve().then(() => {
    console.log('6. Promise resolve');
});

console.log('2. Synchronous code ends');

// Event Loop Phases:
// 1. Timer phase (setTimeout, setInterval)
// 2. Pending callbacks phase (I/O callbacks)
// 3. Idle, prepare phase (internal use)
// 4. Poll phase (fetch new I/O events)
// 5. Check phase (setImmediate callbacks)
// 6. Close callbacks phase (close event callbacks)
```

### Advanced Error Handling
```javascript
// Comprehensive Error Handling Strategy
class AppError extends Error {
    constructor(message, statusCode, isOperational = true) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = isOperational;
        this.status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
        
        Error.captureStackTrace(this, this.constructor);
    }
}

// Global Error Handler
const globalErrorHandler = (err, req, res, next) => {
    let error = { ...err };
    error.message = err.message;
    
    // Mongoose bad ObjectId
    if (err.name === 'CastError') {
        const message = 'Resource not found';
        error = new AppError(message, 404);
    }
    
    // Mongoose duplicate key
    if (err.code === 11000) {
        const value = err.errmsg.match(/(["'])(\\?.)*?\1/)[0];
        const message = `Duplicate field value: ${value}`;
        error = new AppError(message, 400);
    }
    
    // Mongoose validation error
    if (err.name === 'ValidationError') {
        const message = Object.values(err.errors).map(val => val.message);
        error = new AppError(message, 400);
    }
    
    // JWT errors
    if (err.name === 'JsonWebTokenError') {
        error = new AppError('Invalid token', 401);
    }
    
    if (err.name === 'TokenExpiredError') {
        error = new AppError('Token expired', 401);
    }
    
    res.status(error.statusCode || 500).json({
        success: false,
        error: error.message || 'Server Error',
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
    });
};

// Async Error Wrapper
const catchAsync = (fn) => {
    return (req, res, next) => {
        fn(req, res, next).catch(next);
    };
};

// Usage Example
const createUser = catchAsync(async (req, res, next) => {
    const user = await User.create(req.body);
    res.status(201).json({
        success: true,
        data: user
    });
});
```

## 🚀 Advanced Node.js Patterns

### Microservices Architecture
```javascript
// Service Discovery and Communication
class ServiceRegistry {
    constructor() {
        this.services = new Map();
        this.heartbeatInterval = 30000;
    }
    
    register(serviceName, serviceInfo) {
        this.services.set(serviceName, {
            ...serviceInfo,
            lastHeartbeat: Date.now(),
            status: 'healthy'
        });
        
        console.log(`Service ${serviceName} registered`);
    }
    
    discover(serviceName) {
        const service = this.services.get(serviceName);
        if (!service || service.status !== 'healthy') {
            throw new Error(`Service ${serviceName} not available`);
        }
        return service;
    }
    
    startHeartbeat() {
        setInterval(() => {
            this.services.forEach((service, name) => {
                if (Date.now() - service.lastHeartbeat > this.heartbeatInterval * 2) {
                    service.status = 'unhealthy';
                    console.log(`Service ${name} marked as unhealthy`);
                }
            });
        }, this.heartbeatInterval);
    }
}

// Circuit Breaker Pattern
class CircuitBreaker {
    constructor(threshold = 5, timeout = 60000) {
        this.threshold = threshold;
        this.timeout = timeout;
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    }
    
    async execute(fn) {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.timeout) {
                this.state = 'HALF_OPEN';
            } else {
                throw new Error('Circuit breaker is OPEN');
            }
        }
        
        try {
            const result = await fn();
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
        
        if (this.failureCount >= this.threshold) {
            this.state = 'OPEN';
        }
    }
}

// Service Communication
class ServiceClient {
    constructor(serviceRegistry, circuitBreaker) {
        this.registry = serviceRegistry;
        this.circuitBreaker = circuitBreaker;
    }
    
    async call(serviceName, endpoint, data) {
        const service = this.registry.discover(serviceName);
        const url = `http://${service.host}:${service.port}${endpoint}`;
        
        return this.circuitBreaker.execute(async () => {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`Service call failed: ${response.statusText}`);
            }
            
            return response.json();
        });
    }
}
```

### Advanced Caching Strategies
```javascript
// Multi-Level Cache Implementation
class MultiLevelCache {
    constructor() {
        this.l1Cache = new Map(); // In-memory cache
        this.l2Cache = null; // Redis cache
        this.l3Cache = null; // Database
        this.ttl = {
            l1: 5 * 60 * 1000,    // 5 minutes
            l2: 60 * 60 * 1000,   // 1 hour
            l3: 24 * 60 * 60 * 1000 // 24 hours
        };
    }
    
    async get(key) {
        // Try L1 cache first
        const l1Value = this.l1Cache.get(key);
        if (l1Value && !this.isExpired(l1Value)) {
            return l1Value.data;
        }
        
        // Try L2 cache (Redis)
        if (this.l2Cache) {
            try {
                const l2Value = await this.l2Cache.get(key);
                if (l2Value) {
                    // Store in L1 cache
                    this.l1Cache.set(key, {
                        data: l2Value,
                        timestamp: Date.now()
                    });
                    return l2Value;
                }
            } catch (error) {
                console.error('L2 cache error:', error);
            }
        }
        
        // Try L3 cache (Database)
        if (this.l3Cache) {
            try {
                const l3Value = await this.l3Cache.get(key);
                if (l3Value) {
                    // Store in both caches
                    this.l1Cache.set(key, {
                        data: l3Value,
                        timestamp: Date.now()
                    });
                    
                    if (this.l2Cache) {
                        await this.l2Cache.setex(key, this.ttl.l2 / 1000, l3Value);
                    }
                    
                    return l3Value;
                }
            } catch (error) {
                console.error('L3 cache error:', error);
            }
        }
        
        return null;
    }
    
    async set(key, value, ttl = this.ttl.l1) {
        // Store in L1 cache
        this.l1Cache.set(key, {
            data: value,
            timestamp: Date.now()
        });
        
        // Store in L2 cache
        if (this.l2Cache) {
            try {
                await this.l2Cache.setex(key, ttl / 1000, value);
            } catch (error) {
                console.error('L2 cache set error:', error);
            }
        }
        
        // Store in L3 cache
        if (this.l3Cache) {
            try {
                await this.l3Cache.set(key, value, ttl);
            } catch (error) {
                console.error('L3 cache set error:', error);
            }
        }
    }
    
    isExpired(cacheEntry) {
        return Date.now() - cacheEntry.timestamp > this.ttl.l1;
    }
    
    invalidate(key) {
        this.l1Cache.delete(key);
        
        if (this.l2Cache) {
            this.l2Cache.del(key);
        }
        
        if (this.l3Cache) {
            this.l3Cache.del(key);
        }
    }
}
```

## 🔧 Performance Optimization

### Memory Management
```javascript
// Memory Leak Prevention
class MemoryManager {
    constructor() {
        this.weakRefs = new WeakMap();
        this.cleanupTasks = new Set();
    }
    
    // Use WeakMap for automatic garbage collection
    createWeakReference(obj, metadata) {
        this.weakRefs.set(obj, metadata);
        return obj;
    }
    
    // Cleanup tasks when objects are garbage collected
    addCleanupTask(obj, cleanupFn) {
        const ref = new WeakRef(obj);
        this.cleanupTasks.add({
            ref,
            cleanup: cleanupFn
        });
        
        // Check for collected objects periodically
        setInterval(() => {
            this.cleanupTasks.forEach(task => {
                if (task.ref.deref() === undefined) {
                    task.cleanup();
                    this.cleanupTasks.delete(task);
                }
            });
        }, 60000); // Check every minute
    }
    
    // Memory usage monitoring
    getMemoryUsage() {
        const usage = process.memoryUsage();
        return {
            rss: Math.round(usage.rss / 1024 / 1024 * 100) / 100, // MB
            heapTotal: Math.round(usage.heapTotal / 1024 / 1024 * 100) / 100,
            heapUsed: Math.round(usage.heapUsed / 1024 / 1024 * 100) / 100,
            external: Math.round(usage.external / 1024 / 1024 * 100) / 100
        };
    }
}

// Stream Processing for Large Data
class StreamProcessor {
    constructor(options = {}) {
        this.batchSize = options.batchSize || 1000;
        this.flushInterval = options.flushInterval || 5000;
        this.buffer = [];
        this.isProcessing = false;
    }
    
    async process(data) {
        this.buffer.push(data);
        
        if (this.buffer.length >= this.batchSize) {
            await this.flush();
        }
    }
    
    async flush() {
        if (this.isProcessing || this.buffer.length === 0) {
            return;
        }
        
        this.isProcessing = true;
        const batch = this.buffer.splice(0, this.batchSize);
        
        try {
            await this.processBatch(batch);
        } catch (error) {
            console.error('Batch processing error:', error);
            // Re-add failed items to buffer
            this.buffer.unshift(...batch);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async processBatch(batch) {
        // Implement your batch processing logic here
        console.log(`Processing batch of ${batch.length} items`);
        
        // Simulate async processing
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    start() {
        setInterval(() => {
            this.flush();
        }, this.flushInterval);
    }
}
```

### Database Optimization
```javascript
// Advanced Database Connection Pooling
class DatabasePool {
    constructor(config) {
        this.config = config;
        this.pools = new Map();
        this.healthCheckInterval = 30000;
        this.startHealthChecks();
    }
    
    async getConnection(database) {
        if (!this.pools.has(database)) {
            await this.createPool(database);
        }
        
        const pool = this.pools.get(database);
        return pool.getConnection();
    }
    
    async createPool(database) {
        const config = this.config[database];
        const pool = mysql.createPool({
            ...config,
            acquireTimeout: 60000,
            timeout: 60000,
            reconnect: true,
            idleTimeout: 300000,
            queueLimit: 0
        });
        
        this.pools.set(database, pool);
    }
    
    async query(database, sql, params = []) {
        const connection = await this.getConnection(database);
        
        try {
            const [rows] = await connection.execute(sql, params);
            return rows;
        } finally {
            connection.release();
        }
    }
    
    startHealthChecks() {
        setInterval(async () => {
            for (const [database, pool] of this.pools) {
                try {
                    await pool.execute('SELECT 1');
                } catch (error) {
                    console.error(`Database ${database} health check failed:`, error);
                    // Implement reconnection logic
                }
            }
        }, this.healthCheckInterval);
    }
}

// Query Optimization
class QueryOptimizer {
    constructor() {
        this.queryCache = new Map();
        this.cacheTimeout = 300000; // 5 minutes
    }
    
    async executeQuery(sql, params, options = {}) {
        const cacheKey = this.generateCacheKey(sql, params);
        
        // Check cache first
        if (options.cache !== false) {
            const cached = this.queryCache.get(cacheKey);
            if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        // Execute query
        const start = Date.now();
        const result = await this.database.query(sql, params);
        const duration = Date.now() - start;
        
        // Log slow queries
        if (duration > 1000) {
            console.warn(`Slow query detected: ${duration}ms`, sql);
        }
        
        // Cache result
        if (options.cache !== false) {
            this.queryCache.set(cacheKey, {
                data: result,
                timestamp: Date.now()
            });
        }
        
        return result;
    }
    
    generateCacheKey(sql, params) {
        return `${sql}:${JSON.stringify(params)}`;
    }
    
    clearCache() {
        this.queryCache.clear();
    }
}
```

## 🔐 Security Best Practices

### Authentication and Authorization
```javascript
// JWT Authentication with Refresh Tokens
class AuthManager {
    constructor() {
        this.accessTokenSecret = process.env.JWT_ACCESS_SECRET;
        this.refreshTokenSecret = process.env.JWT_REFRESH_SECRET;
        this.accessTokenExpiry = '15m';
        this.refreshTokenExpiry = '7d';
        this.refreshTokens = new Set();
    }
    
    generateTokens(user) {
        const accessToken = jwt.sign(
            { userId: user.id, email: user.email },
            this.accessTokenSecret,
            { expiresIn: this.accessTokenExpiry }
        );
        
        const refreshToken = jwt.sign(
            { userId: user.id, type: 'refresh' },
            this.refreshTokenSecret,
            { expiresIn: this.refreshTokenExpiry }
        );
        
        this.refreshTokens.add(refreshToken);
        
        return { accessToken, refreshToken };
    }
    
    async refreshAccessToken(refreshToken) {
        if (!this.refreshTokens.has(refreshToken)) {
            throw new Error('Invalid refresh token');
        }
        
        try {
            const decoded = jwt.verify(refreshToken, this.refreshTokenSecret);
            
            if (decoded.type !== 'refresh') {
                throw new Error('Invalid token type');
            }
            
            const user = await User.findById(decoded.userId);
            if (!user) {
                throw new Error('User not found');
            }
            
            const accessToken = jwt.sign(
                { userId: user.id, email: user.email },
                this.accessTokenSecret,
                { expiresIn: this.accessTokenExpiry }
            );
            
            return { accessToken };
        } catch (error) {
            this.refreshTokens.delete(refreshToken);
            throw error;
        }
    }
    
    async revokeRefreshToken(refreshToken) {
        this.refreshTokens.delete(refreshToken);
    }
}

// Rate Limiting
class RateLimiter {
    constructor() {
        this.requests = new Map();
        this.cleanupInterval = 60000; // 1 minute
        this.startCleanup();
    }
    
    isAllowed(identifier, limit = 100, window = 60000) {
        const now = Date.now();
        const key = `${identifier}:${Math.floor(now / window)}`;
        
        const current = this.requests.get(key) || 0;
        
        if (current >= limit) {
            return false;
        }
        
        this.requests.set(key, current + 1);
        return true;
    }
    
    startCleanup() {
        setInterval(() => {
            const now = Date.now();
            for (const [key, timestamp] of this.requests) {
                if (now - timestamp > 300000) { // 5 minutes
                    this.requests.delete(key);
                }
            }
        }, this.cleanupInterval);
    }
}

// Input Validation and Sanitization
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
        
        const { error, value } = schema.validate(data, {
            abortEarly: false,
            stripUnknown: true
        });
        
        if (error) {
            throw new Error(`Validation error: ${error.details.map(d => d.message).join(', ')}`);
        }
        
        return value;
    }
    
    sanitize(input) {
        if (typeof input === 'string') {
            return input
                .replace(/[<>]/g, '') // Remove HTML tags
                .replace(/['"]/g, '') // Remove quotes
                .trim();
        }
        
        if (Array.isArray(input)) {
            return input.map(item => this.sanitize(item));
        }
        
        if (typeof input === 'object' && input !== null) {
            const sanitized = {};
            for (const [key, value] of Object.entries(input)) {
                sanitized[key] = this.sanitize(value);
            }
            return sanitized;
        }
        
        return input;
    }
}
```

## 🎯 Best Practices

### Code Organization
1. **Modular Architecture**: Use modules and services
2. **Error Handling**: Implement comprehensive error handling
3. **Logging**: Use structured logging
4. **Testing**: Write unit and integration tests
5. **Documentation**: Document your code and APIs

### Performance Tips
1. **Use Streams**: For large data processing
2. **Implement Caching**: Multiple levels of caching
3. **Database Optimization**: Connection pooling and query optimization
4. **Memory Management**: Prevent memory leaks
5. **Monitoring**: Implement performance monitoring

### Security Guidelines
1. **Input Validation**: Validate and sanitize all inputs
2. **Authentication**: Implement secure authentication
3. **Authorization**: Use proper authorization mechanisms
4. **Rate Limiting**: Implement rate limiting
5. **HTTPS**: Always use HTTPS in production

---

**Last Updated**: December 2024  
**Category**: Node.js Backend Engineering  
**Complexity**: Senior Level
