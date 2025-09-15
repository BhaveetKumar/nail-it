# ⚡ Node.js Performance Optimization: Complete Guide

> **Master Node.js performance optimization for production applications**

## 🎯 **Learning Objectives**

- Master Node.js performance optimization techniques
- Understand memory management and garbage collection
- Learn profiling and monitoring tools
- Implement caching and optimization strategies
- Build high-performance Node.js applications

## 📚 **Table of Contents**

1. [Performance Fundamentals](#performance-fundamentals)
2. [Memory Management](#memory-management)
3. [CPU Optimization](#cpu-optimization)
4. [I/O Optimization](#io-optimization)
5. [Caching Strategies](#caching-strategies)
6. [Database Optimization](#database-optimization)
7. [Profiling and Monitoring](#profiling-and-monitoring)
8. [Production Optimization](#production-optimization)
9. [Interview Questions](#interview-questions)

---

## 🚀 **Performance Fundamentals**

### **What is Performance?**

Performance in Node.js refers to how efficiently your application uses system resources (CPU, memory, I/O) to handle requests and process data. Key metrics include:

- **Throughput**: Requests per second
- **Latency**: Response time
- **Memory Usage**: RAM consumption
- **CPU Usage**: Processor utilization
- **I/O Efficiency**: Disk and network operations

### **Performance Bottlenecks**

```javascript
// Common Performance Bottlenecks
class PerformanceBottlenecks {
    constructor() {
        this.bottlenecks = {
            cpu: 'CPU-intensive operations blocking event loop',
            memory: 'Memory leaks and excessive allocation',
            io: 'Synchronous I/O operations',
            database: 'Inefficient queries and connections',
            network: 'Slow external API calls',
            garbage: 'Frequent garbage collection'
        };
    }
    
    identifyBottleneck(metrics) {
        if (metrics.cpu > 80) return 'cpu';
        if (metrics.memory > 90) return 'memory';
        if (metrics.io > 1000) return 'io';
        if (metrics.database > 500) return 'database';
        return 'unknown';
    }
}
```

### **Performance Measurement**

```javascript
// Performance Measurement Tools
class PerformanceMeasurer {
    constructor() {
        this.metrics = {
            startTime: Date.now(),
            requestCount: 0,
            totalResponseTime: 0,
            memoryUsage: process.memoryUsage(),
            cpuUsage: process.cpuUsage()
        };
    }
    
    startTimer() {
        return process.hrtime();
    }
    
    endTimer(startTime) {
        const diff = process.hrtime(startTime);
        return diff[0] * 1000 + diff[1] / 1000000; // Convert to milliseconds
    }
    
    measureFunction(fn, ...args) {
        const start = this.startTimer();
        const result = fn(...args);
        const duration = this.endTimer(start);
        
        console.log(`Function ${fn.name} took ${duration.toFixed(2)}ms`);
        return { result, duration };
    }
    
    async measureAsyncFunction(fn, ...args) {
        const start = this.startTimer();
        const result = await fn(...args);
        const duration = this.endTimer(start);
        
        console.log(`Async function ${fn.name} took ${duration.toFixed(2)}ms`);
        return { result, duration };
    }
    
    getMemoryUsage() {
        const usage = process.memoryUsage();
        return {
            rss: `${Math.round(usage.rss / 1024 / 1024)} MB`,
            heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)} MB`,
            heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)} MB`,
            external: `${Math.round(usage.external / 1024 / 1024)} MB`
        };
    }
}
```

---

## 🧠 **Memory Management**

### **Memory Optimization Techniques**

```javascript
// Memory Optimization Strategies
class MemoryOptimizer {
    constructor() {
        this.optimizations = [
            'Object pooling',
            'Streaming data',
            'Weak references',
            'Memory monitoring',
            'Garbage collection tuning'
        ];
    }
    
    // Object Pooling
    createObjectPool(createFn, resetFn, initialSize = 10) {
        const pool = [];
        
        // Pre-populate pool
        for (let i = 0; i < initialSize; i++) {
            pool.push(createFn());
        }
        
        return {
            acquire: () => {
                if (pool.length > 0) {
                    return pool.pop();
                }
                return createFn();
            },
            
            release: (obj) => {
                resetFn(obj);
                pool.push(obj);
            },
            
            size: () => pool.length
        };
    }
    
    // Streaming for large data
    async processLargeDataset(data, chunkSize = 1000) {
        const results = [];
        
        for (let i = 0; i < data.length; i += chunkSize) {
            const chunk = data.slice(i, i + chunkSize);
            const processed = await this.processChunk(chunk);
            results.push(...processed);
            
            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }
        }
        
        return results;
    }
    
    // Weak references for temporary data
    createWeakCache() {
        const cache = new WeakMap();
        
        return {
            set: (key, value) => cache.set(key, value),
            get: (key) => cache.get(key),
            has: (key) => cache.has(key)
        };
    }
    
    // Memory leak detection
    detectMemoryLeaks() {
        const initialMemory = process.memoryUsage();
        
        return {
            check: () => {
                const currentMemory = process.memoryUsage();
                const growth = currentMemory.heapUsed - initialMemory.heapUsed;
                
                if (growth > 100 * 1024 * 1024) { // 100MB growth
                    console.warn('Potential memory leak detected');
                    return true;
                }
                return false;
            }
        };
    }
}

// Usage example
const objectPool = new MemoryOptimizer().createObjectPool(
    () => ({ data: null, processed: false }),
    (obj) => { obj.data = null; obj.processed = false; }
);

const obj = objectPool.acquire();
// Use object
objectPool.release(obj);
```

### **Garbage Collection Optimization**

```javascript
// Garbage Collection Management
class GarbageCollectionManager {
    constructor() {
        this.gcStats = {
            totalCollections: 0,
            totalTime: 0,
            averageTime: 0
        };
    }
    
    // Force garbage collection
    forceGC() {
        if (global.gc) {
            const start = process.hrtime();
            global.gc();
            const end = process.hrtime(start);
            
            this.gcStats.totalCollections++;
            this.gcStats.totalTime += end[0] * 1000 + end[1] / 1000000;
            this.gcStats.averageTime = this.gcStats.totalTime / this.gcStats.totalCollections;
            
            console.log(`GC took ${(end[0] * 1000 + end[1] / 1000000).toFixed(2)}ms`);
        } else {
            console.log('Garbage collection not available. Run with --expose-gc flag');
        }
    }
    
    // Monitor garbage collection
    monitorGC() {
        if (process.memoryUsage) {
            setInterval(() => {
                const usage = process.memoryUsage();
                console.log('Memory usage:', {
                    rss: `${Math.round(usage.rss / 1024 / 1024)} MB`,
                    heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)} MB`,
                    heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)} MB`
                });
            }, 5000);
        }
    }
    
    // Optimize for garbage collection
    optimizeForGC() {
        // Use object pooling
        // Minimize object creation
        // Use primitive types when possible
        // Avoid closures that capture large objects
        // Use WeakMap/WeakSet for temporary references
    }
}
```

---

## ⚡ **CPU Optimization**

### **Event Loop Optimization**

```javascript
// Event Loop Optimization
class EventLoopOptimizer {
    constructor() {
        this.lagThreshold = 10; // 10ms
        this.monitoring = false;
    }
    
    // Monitor event loop lag
    startMonitoring() {
        this.monitoring = true;
        this.monitorLoop();
    }
    
    monitorLoop() {
        if (!this.monitoring) return;
        
        const start = process.hrtime();
        
        setImmediate(() => {
            const lag = process.hrtime(start);
            const lagMs = lag[0] * 1000 + lag[1] / 1000000;
            
            if (lagMs > this.lagThreshold) {
                console.warn(`Event loop lag: ${lagMs.toFixed(2)}ms`);
            }
            
            this.monitorLoop();
        });
    }
    
    // Break up CPU-intensive tasks
    async processLargeArray(array, chunkSize = 1000) {
        const results = [];
        
        for (let i = 0; i < array.length; i += chunkSize) {
            const chunk = array.slice(i, i + chunkSize);
            const processed = this.processChunk(chunk);
            results.push(...processed);
            
            // Yield to event loop
            await new Promise(resolve => setImmediate(resolve));
        }
        
        return results;
    }
    
    processChunk(chunk) {
        // Process chunk synchronously
        return chunk.map(item => item * 2);
    }
    
    // Use worker threads for CPU-intensive tasks
    async processWithWorker(data) {
        const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
        
        if (isMainThread) {
            return new Promise((resolve, reject) => {
                const worker = new Worker(__filename, {
                    workerData: data
                });
                
                worker.on('message', resolve);
                worker.on('error', reject);
                worker.on('exit', (code) => {
                    if (code !== 0) {
                        reject(new Error(`Worker stopped with exit code ${code}`));
                    }
                });
            });
        } else {
            // Worker thread
            const result = this.cpuIntensiveTask(workerData);
            parentPort.postMessage(result);
        }
    }
    
    cpuIntensiveTask(data) {
        // CPU-intensive computation
        let result = 0;
        for (let i = 0; i < data.length; i++) {
            result += Math.sqrt(data[i]);
        }
        return result;
    }
}
```

### **Algorithm Optimization**

```javascript
// Algorithm Optimization
class AlgorithmOptimizer {
    constructor() {
        this.cache = new Map();
    }
    
    // Memoization for expensive functions
    memoize(fn) {
        return (...args) => {
            const key = JSON.stringify(args);
            
            if (this.cache.has(key)) {
                return this.cache.get(key);
            }
            
            const result = fn(...args);
            this.cache.set(key, result);
            return result;
        };
    }
    
    // Debouncing for frequent calls
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Throttling for rate limiting
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    // Efficient data structures
    createEfficientSet() {
        return new Set();
    }
    
    createEfficientMap() {
        return new Map();
    }
    
    // Binary search for sorted arrays
    binarySearch(arr, target) {
        let left = 0;
        let right = arr.length - 1;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            
            if (arr[mid] === target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
}
```

---

## 📁 **I/O Optimization**

### **Asynchronous I/O Patterns**

```javascript
// I/O Optimization Patterns
class IOOptimizer {
    constructor() {
        this.connectionPool = new Map();
        this.requestQueue = [];
    }
    
    // Connection pooling
    createConnectionPool(maxConnections = 10) {
        const pool = {
            connections: [],
            maxConnections,
            available: 0,
            
            async acquire() {
                if (this.connections.length > 0) {
                    return this.connections.pop();
                }
                
                if (this.available < this.maxConnections) {
                    const connection = await this.createConnection();
                    this.available++;
                    return connection;
                }
                
                // Wait for available connection
                return new Promise(resolve => {
                    this.requestQueue.push(resolve);
                });
            },
            
            release(connection) {
                if (this.requestQueue.length > 0) {
                    const resolve = this.requestQueue.shift();
                    resolve(connection);
                } else {
                    this.connections.push(connection);
                }
            },
            
            async createConnection() {
                // Simulate connection creation
                return { id: Date.now(), connected: true };
            }
        };
        
        return pool;
    }
    
    // Batch processing for I/O operations
    async batchProcess(items, batchSize = 100, processor) {
        const results = [];
        
        for (let i = 0; i < items.length; i += batchSize) {
            const batch = items.slice(i, i + batchSize);
            const batchResults = await Promise.all(
                batch.map(item => processor(item))
            );
            results.push(...batchResults);
        }
        
        return results;
    }
    
    // Streaming for large files
    async streamLargeFile(filePath, processor) {
        const fs = require('fs');
        const readline = require('readline');
        
        const fileStream = fs.createReadStream(filePath);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });
        
        const results = [];
        
        for await (const line of rl) {
            const processed = await processor(line);
            results.push(processed);
        }
        
        return results;
    }
    
    // Parallel I/O operations
    async parallelIO(operations) {
        const results = await Promise.allSettled(operations);
        
        return results.map((result, index) => ({
            index,
            success: result.status === 'fulfilled',
            data: result.status === 'fulfilled' ? result.value : null,
            error: result.status === 'rejected' ? result.reason : null
        }));
    }
}
```

### **Database Optimization**

```javascript
// Database Optimization
class DatabaseOptimizer {
    constructor() {
        this.queryCache = new Map();
        this.connectionPool = null;
    }
    
    // Query optimization
    optimizeQuery(query, params) {
        // Use prepared statements
        // Add appropriate indexes
        // Limit result sets
        // Use pagination
        // Avoid SELECT *
        
        return {
            query: query.replace(/\s+/g, ' ').trim(),
            params: params,
            optimized: true
        };
    }
    
    // Connection pooling
    setupConnectionPool(config) {
        const { Pool } = require('pg');
        
        this.connectionPool = new Pool({
            user: config.user,
            host: config.host,
            database: config.database,
            password: config.password,
            port: config.port,
            max: 20,
            idleTimeoutMillis: 30000,
            connectionTimeoutMillis: 2000,
        });
        
        return this.connectionPool;
    }
    
    // Query caching
    async cachedQuery(query, params, ttl = 300000) { // 5 minutes
        const key = `${query}_${JSON.stringify(params)}`;
        
        if (this.queryCache.has(key)) {
            const cached = this.queryCache.get(key);
            if (Date.now() - cached.timestamp < ttl) {
                return cached.data;
            }
        }
        
        const result = await this.executeQuery(query, params);
        
        this.queryCache.set(key, {
            data: result,
            timestamp: Date.now()
        });
        
        return result;
    }
    
    async executeQuery(query, params) {
        const client = await this.connectionPool.connect();
        
        try {
            const result = await client.query(query, params);
            return result.rows;
        } finally {
            client.release();
        }
    }
    
    // Batch operations
    async batchInsert(table, data, batchSize = 1000) {
        const batches = [];
        
        for (let i = 0; i < data.length; i += batchSize) {
            const batch = data.slice(i, i + batchSize);
            batches.push(this.insertBatch(table, batch));
        }
        
        const results = await Promise.all(batches);
        return results.flat();
    }
    
    async insertBatch(table, batch) {
        const columns = Object.keys(batch[0]);
        const values = batch.map(row => 
            `(${columns.map(col => `'${row[col]}'`).join(', ')})`
        ).join(', ');
        
        const query = `INSERT INTO ${table} (${columns.join(', ')}) VALUES ${values}`;
        return await this.executeQuery(query);
    }
}
```

---

## 🚀 **Caching Strategies**

### **Multi-Level Caching**

```javascript
// Multi-Level Caching System
class MultiLevelCache {
    constructor() {
        this.l1Cache = new Map(); // In-memory cache
        this.l2Cache = null; // Redis cache
        this.l3Cache = null; // Database cache
        this.stats = {
            l1Hits: 0,
            l2Hits: 0,
            l3Hits: 0,
            misses: 0
        };
    }
    
    async get(key) {
        // L1 Cache (Fastest)
        if (this.l1Cache.has(key)) {
            this.stats.l1Hits++;
            return this.l1Cache.get(key);
        }
        
        // L2 Cache (Fast)
        if (this.l2Cache) {
            const l2Value = await this.l2Cache.get(key);
            if (l2Value) {
                this.stats.l2Hits++;
                this.l1Cache.set(key, l2Value);
                return l2Value;
            }
        }
        
        // L3 Cache (Slowest)
        if (this.l3Cache) {
            const l3Value = await this.l3Cache.get(key);
            if (l3Value) {
                this.stats.l3Hits++;
                this.l1Cache.set(key, l3Value);
                if (this.l2Cache) {
                    await this.l2Cache.set(key, l3Value);
                }
                return l3Value;
            }
        }
        
        this.stats.misses++;
        return null;
    }
    
    async set(key, value, ttl = 3600) {
        // Set in all levels
        this.l1Cache.set(key, value);
        
        if (this.l2Cache) {
            await this.l2Cache.set(key, value, ttl);
        }
        
        if (this.l3Cache) {
            await this.l3Cache.set(key, value, ttl);
        }
    }
    
    async invalidate(key) {
        this.l1Cache.delete(key);
        
        if (this.l2Cache) {
            await this.l2Cache.delete(key);
        }
        
        if (this.l3Cache) {
            await this.l3Cache.delete(key);
        }
    }
    
    getStats() {
        const total = this.stats.l1Hits + this.stats.l2Hits + this.stats.l3Hits + this.stats.misses;
        return {
            ...this.stats,
            hitRate: total > 0 ? (this.stats.l1Hits + this.stats.l2Hits + this.stats.l3Hits) / total : 0
        };
    }
}
```

### **Cache-Aside Pattern**

```javascript
// Cache-Aside Pattern Implementation
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
    
    async update(key, updater) {
        // Get current value
        const currentValue = await this.get(key);
        
        if (!currentValue) {
            throw new Error('Key not found');
        }
        
        // Update value
        const newValue = updater(currentValue);
        
        // Update both database and cache
        await this.set(key, newValue);
        
        return newValue;
    }
}
```

---

## 📊 **Profiling and Monitoring**

### **Performance Profiling**

```javascript
// Performance Profiler
class PerformanceProfiler {
    constructor() {
        this.profiles = new Map();
        this.activeProfiles = new Set();
    }
    
    startProfile(name) {
        this.activeProfiles.add(name);
        this.profiles.set(name, {
            startTime: process.hrtime(),
            startMemory: process.memoryUsage(),
            calls: 0
        });
    }
    
    endProfile(name) {
        if (!this.activeProfiles.has(name)) {
            throw new Error(`Profile ${name} not started`);
        }
        
        const profile = this.profiles.get(name);
        const endTime = process.hrtime();
        const endMemory = process.memoryUsage();
        
        const duration = (endTime[0] - profile.startTime[0]) * 1000 + 
                        (endTime[1] - profile.startTime[1]) / 1000000;
        
        const memoryDelta = endMemory.heapUsed - profile.startMemory.heapUsed;
        
        profile.duration = duration;
        profile.memoryDelta = memoryDelta;
        profile.calls++;
        
        this.activeProfiles.delete(name);
        
        console.log(`Profile ${name}: ${duration.toFixed(2)}ms, Memory: ${memoryDelta} bytes`);
        
        return profile;
    }
    
    getProfile(name) {
        return this.profiles.get(name);
    }
    
    getAllProfiles() {
        return Object.fromEntries(this.profiles);
    }
    
    // Auto-profiling decorator
    profile(name) {
        return (target, propertyName, descriptor) => {
            const method = descriptor.value;
            
            descriptor.value = async function(...args) {
                this.startProfile(name);
                try {
                    const result = await method.apply(this, args);
                    return result;
                } finally {
                    this.endProfile(name);
                }
            };
            
            return descriptor;
        };
    }
}
```

### **Real-time Monitoring**

```javascript
// Real-time Performance Monitor
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            requests: 0,
            errors: 0,
            responseTime: [],
            memoryUsage: [],
            cpuUsage: []
        };
        
        this.thresholds = {
            maxResponseTime: 1000, // 1 second
            maxMemoryUsage: 500 * 1024 * 1024, // 500MB
            maxCpuUsage: 80 // 80%
        };
    }
    
    startMonitoring() {
        // Monitor memory usage
        setInterval(() => {
            const memory = process.memoryUsage();
            this.metrics.memoryUsage.push({
                timestamp: Date.now(),
                rss: memory.rss,
                heapUsed: memory.heapUsed,
                heapTotal: memory.heapTotal
            });
            
            // Keep only last 100 measurements
            if (this.metrics.memoryUsage.length > 100) {
                this.metrics.memoryUsage.shift();
            }
        }, 1000);
        
        // Monitor CPU usage
        setInterval(() => {
            const cpuUsage = process.cpuUsage();
            this.metrics.cpuUsage.push({
                timestamp: Date.now(),
                user: cpuUsage.user,
                system: cpuUsage.system
            });
            
            if (this.metrics.cpuUsage.length > 100) {
                this.metrics.cpuUsage.shift();
            }
        }, 1000);
    }
    
    recordRequest(responseTime, success = true) {
        this.metrics.requests++;
        
        if (!success) {
            this.metrics.errors++;
        }
        
        this.metrics.responseTime.push({
            timestamp: Date.now(),
            duration: responseTime
        });
        
        // Keep only last 1000 measurements
        if (this.metrics.responseTime.length > 1000) {
            this.metrics.responseTime.shift();
        }
        
        // Check thresholds
        this.checkThresholds();
    }
    
    checkThresholds() {
        const avgResponseTime = this.getAverageResponseTime();
        const currentMemory = process.memoryUsage().heapUsed;
        const currentCpu = this.getCurrentCpuUsage();
        
        if (avgResponseTime > this.thresholds.maxResponseTime) {
            console.warn(`High response time: ${avgResponseTime.toFixed(2)}ms`);
        }
        
        if (currentMemory > this.thresholds.maxMemoryUsage) {
            console.warn(`High memory usage: ${(currentMemory / 1024 / 1024).toFixed(2)}MB`);
        }
        
        if (currentCpu > this.thresholds.maxCpuUsage) {
            console.warn(`High CPU usage: ${currentCpu.toFixed(2)}%`);
        }
    }
    
    getAverageResponseTime() {
        if (this.metrics.responseTime.length === 0) return 0;
        
        const sum = this.metrics.responseTime.reduce((acc, item) => acc + item.duration, 0);
        return sum / this.metrics.responseTime.length;
    }
    
    getCurrentCpuUsage() {
        if (this.metrics.cpuUsage.length < 2) return 0;
        
        const latest = this.metrics.cpuUsage[this.metrics.cpuUsage.length - 1];
        const previous = this.metrics.cpuUsage[this.metrics.cpuUsage.length - 2];
        
        const timeDelta = latest.timestamp - previous.timestamp;
        const cpuDelta = (latest.user + latest.system) - (previous.user + previous.system);
        
        return (cpuDelta / timeDelta) * 100;
    }
    
    getMetrics() {
        return {
            ...this.metrics,
            averageResponseTime: this.getAverageResponseTime(),
            errorRate: this.metrics.requests > 0 ? this.metrics.errors / this.metrics.requests : 0,
            currentMemory: process.memoryUsage(),
            currentCpu: this.getCurrentCpuUsage()
        };
    }
}
```

---

## 🎯 **Interview Questions**

### **1. How do you optimize Node.js performance?**

**Answer:**
- **Memory Management**: Object pooling, streaming, weak references
- **CPU Optimization**: Worker threads, event loop monitoring, algorithm optimization
- **I/O Optimization**: Connection pooling, batch processing, async patterns
- **Caching**: Multi-level caching, cache-aside pattern
- **Database**: Query optimization, connection pooling, indexing
- **Monitoring**: Profiling, real-time metrics, threshold alerts

### **2. What are the main performance bottlenecks in Node.js?**

**Answer:**
- **CPU-intensive operations**: Blocking the event loop
- **Memory leaks**: Objects not being garbage collected
- **Synchronous I/O**: Blocking operations
- **Inefficient algorithms**: O(n²) instead of O(n)
- **Database queries**: N+1 queries, missing indexes
- **External API calls**: Slow network requests

### **3. How do you monitor Node.js performance?**

**Answer:**
- **Built-in tools**: process.memoryUsage(), process.cpuUsage()
- **Profiling**: Node.js built-in profiler, clinic.js
- **APM tools**: New Relic, DataDog, AppDynamics
- **Custom monitoring**: Real-time metrics, threshold alerts
- **Logging**: Structured logging with performance data
- **Health checks**: Endpoint monitoring

### **4. What is the event loop and how does it affect performance?**

**Answer:**
The event loop is Node.js's mechanism for handling asynchronous operations. It affects performance because:
- **Single-threaded**: All operations run on one thread
- **Non-blocking**: I/O operations don't block the thread
- **Event-driven**: Callbacks are executed when events occur
- **Phases**: Timers, I/O callbacks, idle, poll, check, close
- **Lag**: Long-running operations can block the event loop

### **5. How do you handle memory leaks in Node.js?**

**Answer:**
- **Identify leaks**: Monitor memory usage over time
- **Use tools**: heapdump, clinic.js, Chrome DevTools
- **Common causes**: Global variables, event listeners, closures
- **Prevention**: Proper cleanup, weak references, object pooling
- **Debugging**: Memory snapshots, allocation timeline
- **Monitoring**: Set up alerts for memory growth

---

**🎉 Node.js performance optimization is crucial for production applications!**
