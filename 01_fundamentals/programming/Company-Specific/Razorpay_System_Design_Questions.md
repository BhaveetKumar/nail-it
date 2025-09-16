# 🏦 Razorpay System Design Questions & Machine Coding

> **Comprehensive system design and machine coding problems for Razorpay interviews**

## 🎯 **Overview**

This guide covers system design questions and machine coding problems commonly asked in Razorpay interviews, particularly for Round 2 (System Design + Machine Coding). Each problem includes detailed discussions, multiple approaches, and production-ready implementations.

## 📚 **Table of Contents**

1. [Payment Gateway System Design](#payment-gateway-system-design/)
2. [Real-time Notification System](#real-time-notification-system/)
3. [Distributed Rate Limiting](#distributed-rate-limiting/)
4. [Machine Coding Problems](#machine-coding-problems/)
5. [Interview Discussion Framework](#interview-discussion-framework/)

---

## 🏦 **Payment Gateway System Design**

### **Problem Statement**

Design a payment gateway system similar to Razorpay that can handle:
- 1M+ transactions per day
- 99.9% uptime
- Support for multiple payment methods (cards, UPI, net banking, wallets)
- Real-time transaction processing
- Fraud detection
- Refund processing
- Settlement and reconciliation

### **Requirements Gathering**

#### **Functional Requirements**
- Process payments from multiple sources
- Support various payment methods
- Handle refunds and chargebacks
- Generate transaction reports
- Real-time transaction status updates
- Merchant dashboard and analytics

#### **Non-Functional Requirements**
- **Scalability**: Handle 1M+ transactions/day
- **Availability**: 99.9% uptime
- **Latency**: < 2 seconds for payment processing
- **Consistency**: Strong consistency for financial data
- **Security**: PCI DSS compliance, encryption
- **Reliability**: Handle failures gracefully

### **Capacity Estimation**

```javascript
// Capacity Estimation
class PaymentGatewayCapacity {
    constructor() {
        this.estimates = {
            dailyTransactions: 1000000,
            peakTransactionsPerSecond: 1000,
            averageTransactionSize: 1000, // INR
            dataPerTransaction: 2, // KB
            retentionPeriod: 7 * 365 // 7 years
        };
    }
    
    calculateStorage() {
        const dailyData = this.estimates.dailyTransactions * this.estimates.dataPerTransaction;
        const yearlyData = dailyData * 365;
        const totalData = yearlyData * 7; // 7 years retention
        
        return {
            dailyData: `${(dailyData / 1024 / 1024).toFixed(2)} MB`,
            yearlyData: `${(yearlyData / 1024 / 1024 / 1024).toFixed(2)} GB`,
            totalData: `${(totalData / 1024 / 1024 / 1024).toFixed(2)} GB`
        };
    }
    
    calculateBandwidth() {
        const peakTPS = this.estimates.peakTransactionsPerSecond;
        const dataPerTransaction = this.estimates.dataPerTransaction;
        const peakBandwidth = peakTPS * dataPerTransaction;
        
        return {
            peakBandwidth: `${(peakBandwidth / 1024).toFixed(2)} KB/s`,
            averageBandwidth: `${(peakBandwidth / 10).toFixed(2)} KB/s`
        };
    }
}
```

### **High-Level Design**

```javascript
// Payment Gateway High-Level Architecture
class PaymentGatewayArchitecture {
    constructor() {
        this.components = {
            apiGateway: 'Load balancer and API gateway',
            paymentService: 'Core payment processing',
            fraudDetection: 'Real-time fraud analysis',
            notificationService: 'Real-time notifications',
            settlementService: 'Settlement and reconciliation',
            reportingService: 'Analytics and reporting',
            database: 'Transaction and user data storage',
            cache: 'Redis for session and rate limiting',
            messageQueue: 'Async processing and notifications'
        };
    }
    
    getArchitecture() {
        return {
            layers: [
                'Load Balancer (AWS ALB)',
                'API Gateway (Kong/AWS API Gateway)',
                'Microservices Layer',
                'Data Layer',
                'External Integrations'
            ],
            services: [
                'Payment Service',
                'Fraud Detection Service',
                'Notification Service',
                'Settlement Service',
                'Reporting Service',
                'User Management Service',
                'Merchant Service'
            ],
            databases: [
                'PostgreSQL (Primary)',
                'MongoDB (Analytics)',
                'Redis (Cache)',
                'Elasticsearch (Search)'
            ]
        };
    }
}
```

### **Detailed Design**

#### **Payment Processing Flow**

```javascript
// Payment Processing Service
class PaymentProcessingService {
    constructor() {
        this.paymentMethods = {
            card: new CardPaymentHandler(),
            upi: new UPIPaymentHandler(),
            netbanking: new NetBankingHandler(),
            wallet: new WalletHandler()
        };
        this.fraudDetector = new FraudDetectionService();
        this.notificationService = new NotificationService();
    }
    
    async processPayment(paymentRequest) {
        try {
            // 1. Validate payment request
            await this.validatePaymentRequest(paymentRequest);
            
            // 2. Check fraud
            const fraudCheck = await this.fraudDetector.checkFraud(paymentRequest);
            if (fraudCheck.isFraudulent) {
                throw new Error('Payment flagged as fraudulent');
            }
            
            // 3. Process payment based on method
            const paymentMethod = paymentRequest.paymentMethod;
            const handler = this.paymentMethods[paymentMethod];
            
            if (!handler) {
                throw new Error('Unsupported payment method');
            }
            
            const result = await handler.process(paymentRequest);
            
            // 4. Update transaction status
            await this.updateTransactionStatus(paymentRequest.transactionId, result.status);
            
            // 5. Send notifications
            await this.notificationService.sendPaymentNotification(paymentRequest, result);
            
            return result;
            
        } catch (error) {
            await this.handlePaymentError(paymentRequest, error);
            throw error;
        }
    }
    
    async validatePaymentRequest(request) {
        const required = ['amount', 'currency', 'paymentMethod', 'merchantId', 'customerId'];
        
        for (const field of required) {
            if (!request[field]) {
                throw new Error(`Missing required field: ${field}`);
            }
        }
        
        if (request.amount <= 0) {
            throw new Error('Invalid amount');
        }
        
        if (!['INR', 'USD'].includes(request.currency)) {
            throw new Error('Unsupported currency');
        }
    }
    
    async updateTransactionStatus(transactionId, status) {
        // Update in database
        const transaction = await this.database.updateTransaction(transactionId, {
            status: status,
            updatedAt: new Date()
        });
        
        // Publish event for other services
        await this.messageQueue.publish('transaction.updated', {
            transactionId,
            status,
            timestamp: new Date()
        });
    }
}
```

#### **Database Schema Design**

```sql
-- Core Tables for Payment Gateway

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Merchants table
CREATE TABLE merchants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    business_name VARCHAR(255) NOT NULL,
    business_type VARCHAR(100),
    gst_number VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID REFERENCES merchants(id),
    customer_id UUID REFERENCES users(id),
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    gateway_transaction_id VARCHAR(255),
    gateway_response JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payment methods table
CREATE TABLE payment_methods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    type VARCHAR(50) NOT NULL,
    details JSONB NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Refunds table
CREATE TABLE refunds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID REFERENCES transactions(id),
    amount DECIMAL(15,2) NOT NULL,
    reason VARCHAR(255),
    status VARCHAR(20) NOT NULL,
    gateway_refund_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_created_at ON transactions(created_at);
CREATE INDEX idx_transactions_gateway_id ON transactions(gateway_transaction_id);
```

#### **API Design**

```javascript
// Payment Gateway API Design
class PaymentGatewayAPI {
    constructor() {
        this.express = require('express');
        this.app = this.express();
        this.paymentService = new PaymentProcessingService();
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.use(this.express.json());
        
        // Payment endpoints
        this.app.post('/api/v1/payments', this.createPayment.bind(this));
        this.app.get('/api/v1/payments/:id', this.getPayment.bind(this));
        this.app.post('/api/v1/payments/:id/capture', this.capturePayment.bind(this));
        this.app.post('/api/v1/payments/:id/refund', this.refundPayment.bind(this));
        
        // Webhook endpoints
        this.app.post('/api/v1/webhooks/gateway', this.handleGatewayWebhook.bind(this));
        
        // Merchant endpoints
        this.app.get('/api/v1/merchants/:id/transactions', this.getMerchantTransactions.bind(this));
        this.app.get('/api/v1/merchants/:id/analytics', this.getMerchantAnalytics.bind(this));
    }
    
    async createPayment(req, res) {
        try {
            const paymentRequest = {
                amount: req.body.amount,
                currency: req.body.currency,
                paymentMethod: req.body.payment_method,
                merchantId: req.user.merchantId,
                customerId: req.body.customer_id,
                description: req.body.description,
                metadata: req.body.metadata
            };
            
            const result = await this.paymentService.processPayment(paymentRequest);
            
            res.status(201).json({
                success: true,
                data: {
                    transaction_id: result.transactionId,
                    status: result.status,
                    payment_url: result.paymentUrl,
                    expires_at: result.expiresAt
                }
            });
            
        } catch (error) {
            res.status(400).json({
                success: false,
                error: error.message
            });
        }
    }
    
    async getPayment(req, res) {
        try {
            const transactionId = req.params.id;
            const transaction = await this.paymentService.getTransaction(transactionId);
            
            if (!transaction) {
                return res.status(404).json({
                    success: false,
                    error: 'Transaction not found'
                });
            }
            
            res.json({
                success: true,
                data: transaction
            });
            
        } catch (error) {
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }
}
```

### **Scalability Considerations**

```javascript
// Scalability and Performance Optimizations
class PaymentGatewayScalability {
    constructor() {
        this.optimizations = {
            database: 'Read replicas, sharding, connection pooling',
            caching: 'Redis for session data, rate limiting',
            loadBalancing: 'Round robin, least connections',
            microservices: 'Independent scaling, fault isolation',
            messageQueues: 'Async processing, event-driven architecture',
            cdn: 'Static content delivery',
            monitoring: 'Real-time metrics, alerting'
        };
    }
    
    // Database Sharding Strategy
    implementSharding() {
        return {
            strategy: 'Shard by merchant_id',
            shards: 10,
            routing: 'Consistent hashing',
            rebalancing: 'Automated based on load'
        };
    }
    
    // Caching Strategy
    implementCaching() {
        return {
            sessionData: 'Redis with 1 hour TTL',
            merchantData: 'Redis with 24 hour TTL',
            rateLimits: 'Redis with sliding window',
            paymentMethods: 'Redis with 1 hour TTL'
        };
    }
    
    // Rate Limiting
    implementRateLimiting() {
        return {
            merchant: '1000 requests per minute',
            customer: '100 requests per minute',
            ip: '10000 requests per minute',
            algorithm: 'Sliding window counter'
        };
    }
}
```

---

## 📱 **Real-time Notification System**

### **Problem Statement**

Design a real-time notification system for payment updates that can:
- Send notifications to 1M+ users
- Handle different notification types (SMS, email, push, in-app)
- Support multiple channels and providers
- Ensure delivery guarantees
- Handle provider failures gracefully

### **System Design**

```javascript
// Real-time Notification System
class NotificationSystem {
    constructor() {
        this.channels = {
            sms: new SMSProvider(),
            email: new EmailProvider(),
            push: new PushNotificationProvider(),
            inapp: new InAppNotificationProvider()
        };
        this.messageQueue = new MessageQueue();
        this.deliveryTracker = new DeliveryTracker();
    }
    
    async sendNotification(notification) {
        try {
            // 1. Validate notification
            await this.validateNotification(notification);
            
            // 2. Determine channels
            const channels = this.determineChannels(notification);
            
            // 3. Create notification tasks
            const tasks = channels.map(channel => ({
                id: this.generateTaskId(),
                notification,
                channel,
                status: 'pending',
                createdAt: new Date()
            }));
            
            // 4. Queue tasks for processing
            await this.messageQueue.enqueue('notification.tasks', tasks);
            
            // 5. Track delivery
            await this.deliveryTracker.trackNotification(notification.id, tasks);
            
            return { success: true, taskIds: tasks.map(t => t.id) };
            
        } catch (error) {
            console.error('Notification sending failed:', error);
            throw error;
        }
    }
    
    determineChannels(notification) {
        const channels = [];
        
        if (notification.user.preferences.sms) {
            channels.push('sms');
        }
        
        if (notification.user.preferences.email) {
            channels.push('email');
        }
        
        if (notification.user.preferences.push) {
            channels.push('push');
        }
        
        if (notification.user.preferences.inapp) {
            channels.push('inapp');
        }
        
        return channels;
    }
}
```

### **Message Queue Implementation**

```javascript
// Message Queue for Notifications
class NotificationMessageQueue {
    constructor() {
        this.redis = require('redis');
        this.client = this.redis.createClient();
        this.workers = new Map();
    }
    
    async enqueue(queueName, data) {
        await this.client.lpush(queueName, JSON.stringify(data));
    }
    
    async dequeue(queueName) {
        const result = await this.client.brpop(queueName, 0);
        return JSON.parse(result[1]);
    }
    
    async processNotifications() {
        while (true) {
            try {
                const task = await this.dequeue('notification.tasks');
                await this.processNotificationTask(task);
            } catch (error) {
                console.error('Notification processing error:', error);
                await this.sleep(1000);
            }
        }
    }
    
    async processNotificationTask(task) {
        const { notification, channel } = task;
        const provider = this.getProvider(channel);
        
        try {
            const result = await provider.send(notification);
            await this.updateTaskStatus(task.id, 'completed', result);
        } catch (error) {
            await this.handleNotificationError(task, error);
        }
    }
    
    async handleNotificationError(task, error) {
        const retryCount = task.retryCount || 0;
        
        if (retryCount < 3) {
            // Retry with exponential backoff
            const delay = Math.pow(2, retryCount) * 1000;
            setTimeout(() => {
                this.enqueue('notification.tasks', {
                    ...task,
                    retryCount: retryCount + 1
                });
            }, delay);
        } else {
            // Move to dead letter queue
            await this.enqueue('notification.dlq', task);
        }
    }
}
```

---

## 🚦 **Distributed Rate Limiting**

### **Problem Statement**

Design a distributed rate limiting system that can:
- Handle rate limits across multiple servers
- Support different rate limiting algorithms
- Provide real-time rate limit status
- Handle high traffic with low latency

### **Implementation**

```javascript
// Distributed Rate Limiting System
class DistributedRateLimiter {
    constructor() {
        this.redis = require('redis');
        this.client = this.redis.createClient();
        this.algorithms = {
            slidingWindow: new SlidingWindowRateLimiter(),
            tokenBucket: new TokenBucketRateLimiter(),
            fixedWindow: new FixedWindowRateLimiter()
        };
    }
    
    async checkRateLimit(key, limit, windowMs, algorithm = 'slidingWindow') {
        const limiter = this.algorithms[algorithm];
        return await limiter.checkLimit(key, limit, windowMs);
    }
    
    async getRateLimitStatus(key, limit, windowMs) {
        const current = await this.client.get(`rate_limit:${key}`);
        const count = current ? parseInt(current) : 0;
        
        return {
            limit,
            remaining: Math.max(0, limit - count),
            resetTime: Date.now() + windowMs,
            retryAfter: count >= limit ? windowMs : 0
        };
    }
}

// Sliding Window Rate Limiter
class SlidingWindowRateLimiter {
    async checkLimit(key, limit, windowMs) {
        const now = Date.now();
        const windowStart = now - windowMs;
        
        // Remove old entries
        await this.client.zremrangebyscore(`rate_limit:${key}`, 0, windowStart);
        
        // Count current entries
        const current = await this.client.zcard(`rate_limit:${key}`);
        
        if (current >= limit) {
            return { allowed: false, remaining: 0 };
        }
        
        // Add current request
        await this.client.zadd(`rate_limit:${key}`, now, `${now}-${Math.random()}`);
        await this.client.expire(`rate_limit:${key}`, Math.ceil(windowMs / 1000));
        
        return { allowed: true, remaining: limit - current - 1 };
    }
}

// Token Bucket Rate Limiter
class TokenBucketRateLimiter {
    async checkLimit(key, capacity, refillRate) {
        const now = Date.now();
        const bucket = await this.getBucket(key);
        
        // Calculate tokens to add
        const timePassed = now - bucket.lastRefill;
        const tokensToAdd = Math.floor(timePassed * refillRate / 1000);
        
        bucket.tokens = Math.min(capacity, bucket.tokens + tokensToAdd);
        bucket.lastRefill = now;
        
        if (bucket.tokens >= 1) {
            bucket.tokens -= 1;
            await this.setBucket(key, bucket);
            return { allowed: true, remaining: bucket.tokens };
        }
        
        await this.setBucket(key, bucket);
        return { allowed: false, remaining: 0 };
    }
    
    async getBucket(key) {
        const data = await this.client.get(`bucket:${key}`);
        return data ? JSON.parse(data) : { tokens: 0, lastRefill: Date.now() };
    }
    
    async setBucket(key, bucket) {
        await this.client.setex(`bucket:${key}`, 3600, JSON.stringify(bucket));
    }
}
```

---

## 💻 **Machine Coding Problems**

### **Problem 1: Design a URL Shortener**

```javascript
// URL Shortener Implementation
class URLShortener {
    constructor() {
        this.database = new Map();
        this.counter = 1000000; // Start from 1M
        this.baseUrl = 'https://short.ly/';
        this.characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    }
    
    shortenUrl(originalUrl) {
        // Validate URL
        if (!this.isValidUrl(originalUrl)) {
            throw new Error('Invalid URL');
        }
        
        // Check if URL already exists
        const existing = this.findExistingUrl(originalUrl);
        if (existing) {
            return existing;
        }
        
        // Generate short code
        const shortCode = this.generateShortCode();
        
        // Store mapping
        this.database.set(shortCode, {
            originalUrl,
            shortCode,
            createdAt: new Date(),
            clickCount: 0
        });
        
        return {
            shortUrl: this.baseUrl + shortCode,
            originalUrl,
            shortCode
        };
    }
    
    expandUrl(shortCode) {
        const data = this.database.get(shortCode);
        if (!data) {
            throw new Error('Short URL not found');
        }
        
        // Increment click count
        data.clickCount++;
        this.database.set(shortCode, data);
        
        return data.originalUrl;
    }
    
    generateShortCode() {
        let code = '';
        let num = this.counter++;
        
        while (num > 0) {
            code = this.characters[num % this.characters.length] + code;
            num = Math.floor(num / this.characters.length);
        }
        
        return code;
    }
    
    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    }
    
    findExistingUrl(originalUrl) {
        for (const [shortCode, data] of this.database) {
            if (data.originalUrl === originalUrl) {
                return {
                    shortUrl: this.baseUrl + shortCode,
                    originalUrl,
                    shortCode
                };
            }
        }
        return null;
    }
    
    getStats(shortCode) {
        const data = this.database.get(shortCode);
        if (!data) {
            throw new Error('Short URL not found');
        }
        
        return {
            shortCode,
            originalUrl: data.originalUrl,
            clickCount: data.clickCount,
            createdAt: data.createdAt
        };
    }
}
```

### **Problem 2: Design a Distributed Cache**

```javascript
// Distributed Cache Implementation
class DistributedCache {
    constructor() {
        this.cache = new Map();
        this.maxSize = 1000;
        this.ttl = new Map(); // Time to live
        this.accessOrder = []; // For LRU
    }
    
    set(key, value, ttlMs = 3600000) { // Default 1 hour
        // Remove if exists
        if (this.cache.has(key)) {
            this.removeFromAccessOrder(key);
        }
        
        // Check size limit
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            this.evictLRU();
        }
        
        // Set value
        this.cache.set(key, value);
        this.ttl.set(key, Date.now() + ttlMs);
        this.accessOrder.push(key);
        
        // Set expiration
        setTimeout(() => {
            this.delete(key);
        }, ttlMs);
    }
    
    get(key) {
        if (!this.cache.has(key)) {
            return null;
        }
        
        // Check TTL
        if (this.ttl.get(key) < Date.now()) {
            this.delete(key);
            return null;
        }
        
        // Update access order
        this.removeFromAccessOrder(key);
        this.accessOrder.push(key);
        
        return this.cache.get(key);
    }
    
    delete(key) {
        this.cache.delete(key);
        this.ttl.delete(key);
        this.removeFromAccessOrder(key);
    }
    
    evictLRU() {
        if (this.accessOrder.length > 0) {
            const lruKey = this.accessOrder.shift();
            this.delete(lruKey);
        }
    }
    
    removeFromAccessOrder(key) {
        const index = this.accessOrder.indexOf(key);
        if (index > -1) {
            this.accessOrder.splice(index, 1);
        }
    }
    
    clear() {
        this.cache.clear();
        this.ttl.clear();
        this.accessOrder = [];
    }
    
    size() {
        return this.cache.size;
    }
    
    keys() {
        return Array.from(this.cache.keys());
    }
}
```

### **Problem 3: Design a Task Scheduler**

```javascript
// Task Scheduler Implementation
class TaskScheduler {
    constructor() {
        this.tasks = new Map();
        this.scheduledTasks = [];
        this.isRunning = false;
        this.intervalId = null;
    }
    
    scheduleTask(id, task, delayMs, repeat = false) {
        const scheduledTime = Date.now() + delayMs;
        
        const taskData = {
            id,
            task,
            scheduledTime,
            repeat,
            delayMs,
            createdAt: Date.now()
        };
        
        this.tasks.set(id, taskData);
        this.scheduledTasks.push(taskData);
        this.scheduledTasks.sort((a, b) => a.scheduledTime - b.scheduledTime);
        
        if (!this.isRunning) {
            this.start();
        }
        
        return id;
    }
    
    cancelTask(id) {
        const task = this.tasks.get(id);
        if (!task) {
            return false;
        }
        
        this.tasks.delete(id);
        const index = this.scheduledTasks.findIndex(t => t.id === id);
        if (index > -1) {
            this.scheduledTasks.splice(index, 1);
        }
        
        return true;
    }
    
    start() {
        if (this.isRunning) {
            return;
        }
        
        this.isRunning = true;
        this.intervalId = setInterval(() => {
            this.processTasks();
        }, 100); // Check every 100ms
    }
    
    stop() {
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
    
    processTasks() {
        const now = Date.now();
        
        while (this.scheduledTasks.length > 0 && this.scheduledTasks[0].scheduledTime <= now) {
            const task = this.scheduledTasks.shift();
            
            try {
                task.task();
            } catch (error) {
                console.error(`Task ${task.id} failed:`, error);
            }
            
            // Reschedule if repeat
            if (task.repeat) {
                task.scheduledTime = now + task.delayMs;
                this.scheduledTasks.push(task);
                this.scheduledTasks.sort((a, b) => a.scheduledTime - b.scheduledTime);
            } else {
                this.tasks.delete(task.id);
            }
        }
    }
    
    getTaskStatus(id) {
        const task = this.tasks.get(id);
        if (!task) {
            return null;
        }
        
        return {
            id: task.id,
            scheduledTime: task.scheduledTime,
            repeat: task.repeat,
            delayMs: task.delayMs,
            createdAt: task.createdAt,
            timeUntilExecution: Math.max(0, task.scheduledTime - Date.now())
        };
    }
    
    getAllTasks() {
        return Array.from(this.tasks.values());
    }
}
```

---

## 🎯 **Interview Discussion Framework**

### **System Design Discussion Structure**

1. **Requirements Clarification**
   - Ask clarifying questions
   - Understand functional and non-functional requirements
   - Estimate scale and constraints

2. **High-Level Design**
   - Draw system architecture
   - Identify main components
   - Show data flow

3. **Detailed Design**
   - Database schema
   - API design
   - Component interactions

4. **Scalability & Performance**
   - Identify bottlenecks
   - Discuss scaling strategies
   - Performance optimizations

5. **Security & Reliability**
   - Security considerations
   - Error handling
   - Monitoring and alerting

### **Machine Coding Discussion Points**

1. **Problem Understanding**
   - Clarify requirements
   - Identify edge cases
   - Discuss trade-offs

2. **Approach Selection**
   - Explain algorithm choice
   - Discuss time/space complexity
   - Consider alternatives

3. **Implementation**
   - Write clean, readable code
   - Handle edge cases
   - Add error handling

4. **Testing**
   - Discuss test cases
   - Consider edge cases
   - Performance testing

5. **Improvements**
   - Discuss optimizations
   - Scalability considerations
   - Production readiness

---

**🎉 This guide provides comprehensive coverage of Razorpay-style system design and machine coding problems!**
