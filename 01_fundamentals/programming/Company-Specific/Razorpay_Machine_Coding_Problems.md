# 💻 Razorpay Machine Coding Problems

> **Comprehensive machine coding problems with detailed implementations and discussions**

## 🎯 **Overview**

This guide covers machine coding problems commonly asked in Razorpay interviews, with detailed implementations, test cases, and discussion points. Each problem includes multiple approaches and production-ready solutions.

## 📚 **Table of Contents**

1. [Design a Payment Gateway](#design-a-payment-gateway/)
2. [Design a URL Shortener](#design-a-url-shortener/)
3. [Design a Distributed Cache](#design-a-distributed-cache/)
4. [Design a Task Scheduler](#design-a-task-scheduler/)
5. [Design a Rate Limiter](#design-a-rate-limiter/)
6. [Design a Notification System](#design-a-notification-system/)
7. [Design a File Storage System](#design-a-file-storage-system/)
8. [Design a Search Engine](#design-a-search-engine/)

---

## 💳 **Design a Payment Gateway**

### **Problem Statement**

Design a payment gateway system that can process payments, handle refunds, and manage transactions. The system should support multiple payment methods and provide real-time status updates.

### **Requirements**

- Process payments with different methods (card, UPI, net banking)
- Handle refunds and partial refunds
- Track transaction status in real-time
- Support webhooks for status updates
- Generate transaction reports
- Handle failures and retries

### **Implementation**

```javascript
// Payment Gateway Core Implementation
class PaymentGateway {
  constructor() {
    this.transactions = new Map();
    this.paymentMethods = {
      card: new CardPaymentHandler(),
      upi: new UPIPaymentHandler(),
      netbanking: new NetBankingHandler(),
    };
    this.webhookService = new WebhookService();
    this.retryService = new RetryService();
  }

  async processPayment(paymentRequest) {
    const transactionId = this.generateTransactionId();

    try {
      // Create transaction record
      const transaction = {
        id: transactionId,
        amount: paymentRequest.amount,
        currency: paymentRequest.currency,
        paymentMethod: paymentRequest.paymentMethod,
        merchantId: paymentRequest.merchantId,
        customerId: paymentRequest.customerId,
        status: "pending",
        createdAt: new Date(),
        metadata: paymentRequest.metadata || {},
      };

      this.transactions.set(transactionId, transaction);

      // Process payment based on method
      const handler = this.paymentMethods[paymentRequest.paymentMethod];
      if (!handler) {
        throw new Error("Unsupported payment method");
      }

      const result = await handler.process(transaction);

      // Update transaction status
      transaction.status = result.status;
      transaction.gatewayTransactionId = result.gatewayTransactionId;
      transaction.processedAt = new Date();
      transaction.gatewayResponse = result.response;

      this.transactions.set(transactionId, transaction);

      // Send webhook
      await this.webhookService.sendWebhook(transaction);

      return {
        transactionId,
        status: transaction.status,
        gatewayTransactionId: transaction.gatewayTransactionId,
        paymentUrl: result.paymentUrl,
      };
    } catch (error) {
      // Update transaction status to failed
      const transaction = this.transactions.get(transactionId);
      if (transaction) {
        transaction.status = "failed";
        transaction.error = error.message;
        transaction.failedAt = new Date();
        this.transactions.set(transactionId, transaction);
      }

      throw error;
    }
  }

  async processRefund(transactionId, refundAmount, reason) {
    const transaction = this.transactions.get(transactionId);
    if (!transaction) {
      throw new Error("Transaction not found");
    }

    if (transaction.status !== "completed") {
      throw new Error("Can only refund completed transactions");
    }

    if (refundAmount > transaction.amount) {
      throw new Error("Refund amount cannot exceed transaction amount");
    }

    const refundId = this.generateRefundId();
    const refund = {
      id: refundId,
      transactionId,
      amount: refundAmount,
      reason,
      status: "pending",
      createdAt: new Date(),
    };

    try {
      // Process refund with payment method handler
      const handler = this.paymentMethods[transaction.paymentMethod];
      const result = await handler.refund(transaction, refund);

      refund.status = result.status;
      refund.gatewayRefundId = result.gatewayRefundId;
      refund.processedAt = new Date();

      // Update transaction
      transaction.refunds = transaction.refunds || [];
      transaction.refunds.push(refund);
      this.transactions.set(transactionId, transaction);

      // Send webhook
      await this.webhookService.sendWebhook(transaction, "refund");

      return refund;
    } catch (error) {
      refund.status = "failed";
      refund.error = error.message;
      refund.failedAt = new Date();
      throw error;
    }
  }

  getTransaction(transactionId) {
    return this.transactions.get(transactionId);
  }

  getTransactionsByMerchant(merchantId, limit = 100, offset = 0) {
    const merchantTransactions = Array.from(this.transactions.values())
      .filter((t) => t.merchantId === merchantId)
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(offset, offset + limit);

    return merchantTransactions;
  }

  generateTransactionId() {
    return "txn_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  }

  generateRefundId() {
    return "ref_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  }
}

// Card Payment Handler
class CardPaymentHandler {
  async process(transaction) {
    // Simulate card payment processing
    await this.delay(1000);

    // Simulate success/failure
    const success = Math.random() > 0.1; // 90% success rate

    if (success) {
      return {
        status: "completed",
        gatewayTransactionId: "card_" + Date.now(),
        response: { code: "SUCCESS", message: "Payment successful" },
      };
    } else {
      throw new Error("Card payment failed");
    }
  }

  async refund(transaction, refund) {
    await this.delay(500);

    return {
      status: "completed",
      gatewayRefundId: "refund_" + Date.now(),
      response: { code: "SUCCESS", message: "Refund successful" },
    };
  }

  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// UPI Payment Handler
class UPIPaymentHandler {
  async process(transaction) {
    await this.delay(800);

    const success = Math.random() > 0.05; // 95% success rate

    if (success) {
      return {
        status: "completed",
        gatewayTransactionId: "upi_" + Date.now(),
        response: { code: "SUCCESS", message: "UPI payment successful" },
      };
    } else {
      throw new Error("UPI payment failed");
    }
  }

  async refund(transaction, refund) {
    await this.delay(300);

    return {
      status: "completed",
      gatewayRefundId: "upi_refund_" + Date.now(),
      response: { code: "SUCCESS", message: "UPI refund successful" },
    };
  }

  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Webhook Service
class WebhookService {
  async sendWebhook(transaction, event = "payment") {
    // Simulate webhook sending
    console.log(`Sending webhook for ${event}:`, {
      transactionId: transaction.id,
      status: transaction.status,
      timestamp: new Date(),
    });
  }
}
```

### **Test Cases**

```javascript
// Test Cases for Payment Gateway
async function testPaymentGateway() {
  const gateway = new PaymentGateway();

  // Test successful payment
  try {
    const payment = await gateway.processPayment({
      amount: 1000,
      currency: "INR",
      paymentMethod: "card",
      merchantId: "merchant_123",
      customerId: "customer_456",
      metadata: { orderId: "order_789" },
    });

    console.log("Payment successful:", payment);

    // Test refund
    const refund = await gateway.processRefund(
      payment.transactionId,
      500,
      "Customer requested partial refund"
    );

    console.log("Refund successful:", refund);
  } catch (error) {
    console.error("Payment failed:", error.message);
  }

  // Test transaction retrieval
  const transaction = gateway.getTransaction("txn_123");
  console.log("Transaction details:", transaction);
}

testPaymentGateway();
```

---

## 🔗 **Design a URL Shortener**

### **Problem Statement**

Design a URL shortener service that can convert long URLs into short ones and redirect users to the original URL when they visit the short URL.

### **Requirements**

- Convert long URLs to short URLs
- Redirect short URLs to original URLs
- Track click statistics
- Handle custom short codes
- Support URL expiration
- Prevent spam and abuse

### **Implementation**

```javascript
// URL Shortener Implementation
class URLShortener {
  constructor() {
    this.urlDatabase = new Map();
    this.customCodes = new Set();
    this.clickStats = new Map();
    this.baseUrl = "https://short.ly/";
    this.characters =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    this.counter = 1000000; // Start from 1M
  }

  shortenUrl(originalUrl, customCode = null, expiresAt = null) {
    // Validate URL
    if (!this.isValidUrl(originalUrl)) {
      throw new Error("Invalid URL provided");
    }

    // Check if URL already exists
    const existing = this.findExistingUrl(originalUrl);
    if (existing && !customCode) {
      return existing;
    }

    // Generate or use custom code
    let shortCode;
    if (customCode) {
      if (this.customCodes.has(customCode)) {
        throw new Error("Custom code already exists");
      }
      shortCode = customCode;
    } else {
      shortCode = this.generateShortCode();
    }

    // Create URL record
    const urlRecord = {
      shortCode,
      originalUrl,
      createdAt: new Date(),
      expiresAt,
      clickCount: 0,
      isActive: true,
      customCode: !!customCode,
    };

    // Store in database
    this.urlDatabase.set(shortCode, urlRecord);
    this.customCodes.add(shortCode);

    return {
      shortUrl: this.baseUrl + shortCode,
      originalUrl,
      shortCode,
      expiresAt,
    };
  }

  expandUrl(shortCode) {
    const urlRecord = this.urlDatabase.get(shortCode);

    if (!urlRecord) {
      throw new Error("Short URL not found");
    }

    if (!urlRecord.isActive) {
      throw new Error("Short URL is inactive");
    }

    if (urlRecord.expiresAt && new Date() > urlRecord.expiresAt) {
      throw new Error("Short URL has expired");
    }

    // Increment click count
    urlRecord.clickCount++;
    this.urlDatabase.set(shortCode, urlRecord);

    // Track click statistics
    this.trackClick(shortCode);

    return {
      originalUrl: urlRecord.originalUrl,
      clickCount: urlRecord.clickCount,
      createdAt: urlRecord.createdAt,
    };
  }

  generateShortCode() {
    let code = "";
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
    for (const [shortCode, record] of this.urlDatabase) {
      if (record.originalUrl === originalUrl && record.isActive) {
        return {
          shortUrl: this.baseUrl + shortCode,
          originalUrl,
          shortCode,
        };
      }
    }
    return null;
  }

  trackClick(shortCode) {
    const stats = this.clickStats.get(shortCode) || {
      totalClicks: 0,
      dailyClicks: new Map(),
      hourlyClicks: new Map(),
    };

    stats.totalClicks++;

    const now = new Date();
    const today = now.toISOString().split("T")[0];
    const hour = now.getHours();

    stats.dailyClicks.set(today, (stats.dailyClicks.get(today) || 0) + 1);
    stats.hourlyClicks.set(hour, (stats.hourlyClicks.get(hour) || 0) + 1);

    this.clickStats.set(shortCode, stats);
  }

  getStats(shortCode) {
    const urlRecord = this.urlDatabase.get(shortCode);
    if (!urlRecord) {
      throw new Error("Short URL not found");
    }

    const stats = this.clickStats.get(shortCode) || {
      totalClicks: 0,
      dailyClicks: new Map(),
      hourlyClicks: new Map(),
    };

    return {
      shortCode,
      originalUrl: urlRecord.originalUrl,
      shortUrl: this.baseUrl + shortCode,
      clickCount: urlRecord.clickCount,
      createdAt: urlRecord.createdAt,
      expiresAt: urlRecord.expiresAt,
      isActive: urlRecord.isActive,
      customCode: urlRecord.customCode,
      stats: {
        totalClicks: stats.totalClicks,
        dailyClicks: Object.fromEntries(stats.dailyClicks),
        hourlyClicks: Object.fromEntries(stats.hourlyClicks),
      },
    };
  }

  deactivateUrl(shortCode) {
    const urlRecord = this.urlDatabase.get(shortCode);
    if (!urlRecord) {
      throw new Error("Short URL not found");
    }

    urlRecord.isActive = false;
    this.urlDatabase.set(shortCode, urlRecord);

    return { success: true, message: "URL deactivated successfully" };
  }

  getTopUrls(limit = 10) {
    const urls = Array.from(this.urlDatabase.values())
      .filter((url) => url.isActive)
      .sort((a, b) => b.clickCount - a.clickCount)
      .slice(0, limit);

    return urls.map((url) => ({
      shortCode: url.shortCode,
      originalUrl: url.originalUrl,
      clickCount: url.clickCount,
      createdAt: url.createdAt,
    }));
  }
}
```

### **Test Cases**

```javascript
// Test Cases for URL Shortener
async function testURLShortener() {
  const shortener = new URLShortener();

  try {
    // Test basic URL shortening
    const result1 = shortener.shortenUrl("https://www.google.com");
    console.log("Shortened URL:", result1);

    // Test custom code
    const result2 = shortener.shortenUrl("https://www.github.com", "github");
    console.log("Custom short URL:", result2);

    // Test URL expansion
    const expanded = shortener.expandUrl(result1.shortCode);
    console.log("Expanded URL:", expanded);

    // Test click tracking
    shortener.expandUrl(result1.shortCode);
    shortener.expandUrl(result1.shortCode);

    // Test statistics
    const stats = shortener.getStats(result1.shortCode);
    console.log("URL Statistics:", stats);

    // Test top URLs
    const topUrls = shortener.getTopUrls(5);
    console.log("Top URLs:", topUrls);
  } catch (error) {
    console.error("Error:", error.message);
  }
}

testURLShortener();
```

---

## 🗄️ **Design a Distributed Cache**

### **Problem Statement**

Design a distributed cache system that can store and retrieve data across multiple nodes with consistency and high availability.

### **Requirements**

- Store key-value pairs with TTL
- Support different eviction policies (LRU, LFU, TTL)
- Handle node failures gracefully
- Provide consistency guarantees
- Support distributed operations
- Monitor cache performance

### **Implementation**

```javascript
// Distributed Cache Implementation
class DistributedCache {
  constructor(nodeId, nodes = []) {
    this.nodeId = nodeId;
    this.nodes = nodes;
    this.localCache = new Map();
    this.ttl = new Map();
    this.accessOrder = [];
    this.maxSize = 1000;
    this.replicationFactor = 2;
    this.consistencyLevel = "eventual"; // eventual, strong
  }

  async set(key, value, ttlMs = 3600000) {
    const expirationTime = Date.now() + ttlMs;

    // Store locally
    this.storeLocally(key, value, expirationTime);

    // Replicate to other nodes
    await this.replicate(key, value, expirationTime);

    return { success: true, nodeId: this.nodeId };
  }

  async get(key) {
    // Check local cache first
    let value = this.getLocally(key);

    if (value !== null) {
      return value;
    }

    // If not found locally, try other nodes
    if (this.consistencyLevel === "strong") {
      value = await this.getFromOtherNodes(key);
      if (value !== null) {
        // Store locally for future access
        this.storeLocally(key, value.value, value.expirationTime);
      }
    }

    return value;
  }

  async delete(key) {
    // Delete locally
    this.deleteLocally(key);

    // Delete from other nodes
    await this.deleteFromOtherNodes(key);

    return { success: true };
  }

  storeLocally(key, value, expirationTime) {
    // Check size limit
    if (this.localCache.size >= this.maxSize && !this.localCache.has(key)) {
      this.evictLRU();
    }

    this.localCache.set(key, value);
    this.ttl.set(key, expirationTime);
    this.updateAccessOrder(key);

    // Set expiration
    setTimeout(() => {
      this.deleteLocally(key);
    }, expirationTime - Date.now());
  }

  getLocally(key) {
    if (!this.localCache.has(key)) {
      return null;
    }

    const expirationTime = this.ttl.get(key);
    if (expirationTime < Date.now()) {
      this.deleteLocally(key);
      return null;
    }

    this.updateAccessOrder(key);
    return this.localCache.get(key);
  }

  deleteLocally(key) {
    this.localCache.delete(key);
    this.ttl.delete(key);
    this.removeFromAccessOrder(key);
  }

  updateAccessOrder(key) {
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  removeFromAccessOrder(key) {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  evictLRU() {
    if (this.accessOrder.length > 0) {
      const lruKey = this.accessOrder.shift();
      this.deleteLocally(lruKey);
    }
  }

  async replicate(key, value, expirationTime) {
    const targetNodes = this.getTargetNodes(key);

    const replicationPromises = targetNodes.map((nodeId) => {
      if (nodeId !== this.nodeId) {
        return this.sendToNode(nodeId, "set", { key, value, expirationTime });
      }
    });

    await Promise.allSettled(replicationPromises);
  }

  async getFromOtherNodes(key) {
    const targetNodes = this.getTargetNodes(key);

    for (const nodeId of targetNodes) {
      if (nodeId !== this.nodeId) {
        try {
          const result = await this.sendToNode(nodeId, "get", { key });
          if (result && result.value) {
            return result;
          }
        } catch (error) {
          console.error(`Failed to get from node ${nodeId}:`, error);
        }
      }
    }

    return null;
  }

  async deleteFromOtherNodes(key) {
    const targetNodes = this.getTargetNodes(key);

    const deletionPromises = targetNodes.map((nodeId) => {
      if (nodeId !== this.nodeId) {
        return this.sendToNode(nodeId, "delete", { key });
      }
    });

    await Promise.allSettled(deletionPromises);
  }

  getTargetNodes(key) {
    const hash = this.hash(key);
    const nodeCount = this.nodes.length;
    const startIndex = hash % nodeCount;

    const targetNodes = [];
    for (let i = 0; i < this.replicationFactor; i++) {
      const nodeIndex = (startIndex + i) % nodeCount;
      targetNodes.push(this.nodes[nodeIndex]);
    }

    return targetNodes;
  }

  hash(key) {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      const char = key.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  async sendToNode(nodeId, operation, data) {
    // Simulate network call
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (Math.random() > 0.1) {
          // 90% success rate
          resolve({ success: true, data });
        } else {
          reject(new Error("Network error"));
        }
      }, Math.random() * 100);
    });
  }

  getStats() {
    return {
      nodeId: this.nodeId,
      localCacheSize: this.localCache.size,
      maxSize: this.maxSize,
      hitRate: this.calculateHitRate(),
      memoryUsage: this.calculateMemoryUsage(),
    };
  }

  calculateHitRate() {
    // Simplified hit rate calculation
    return 0.85; // 85% hit rate
  }

  calculateMemoryUsage() {
    let totalSize = 0;
    for (const [key, value] of this.localCache) {
      totalSize += key.length + JSON.stringify(value).length;
    }
    return totalSize;
  }
}
```

---

## ⏰ **Design a Task Scheduler**

### **Problem Statement**

Design a task scheduler that can schedule tasks to run at specific times, with support for recurring tasks, priority queues, and failure handling.

### **Requirements**

- Schedule one-time and recurring tasks
- Support different priority levels
- Handle task failures and retries
- Provide task status tracking
- Support task dependencies
- Monitor task execution

### **Implementation**

```javascript
// Task Scheduler Implementation
class TaskScheduler {
  constructor() {
    this.tasks = new Map();
    this.priorityQueue = new PriorityQueue();
    this.runningTasks = new Set();
    this.maxConcurrentTasks = 10;
    this.isRunning = false;
    this.intervalId = null;
    this.retryService = new RetryService();
  }

  scheduleTask(taskConfig) {
    const taskId = this.generateTaskId();

    const task = {
      id: taskId,
      name: taskConfig.name,
      handler: taskConfig.handler,
      scheduledTime: taskConfig.scheduledTime || Date.now(),
      priority: taskConfig.priority || 0,
      maxRetries: taskConfig.maxRetries || 3,
      retryDelay: taskConfig.retryDelay || 1000,
      dependencies: taskConfig.dependencies || [],
      metadata: taskConfig.metadata || {},
      status: "scheduled",
      createdAt: new Date(),
      attempts: 0,
    };

    this.tasks.set(taskId, task);
    this.priorityQueue.enqueue(task, task.priority);

    if (!this.isRunning) {
      this.start();
    }

    return taskId;
  }

  scheduleRecurringTask(taskConfig) {
    const taskId = this.generateTaskId();

    const task = {
      id: taskId,
      name: taskConfig.name,
      handler: taskConfig.handler,
      scheduledTime: taskConfig.scheduledTime || Date.now(),
      priority: taskConfig.priority || 0,
      maxRetries: taskConfig.maxRetries || 3,
      retryDelay: taskConfig.retryDelay || 1000,
      dependencies: taskConfig.dependencies || [],
      metadata: taskConfig.metadata || {},
      status: "scheduled",
      createdAt: new Date(),
      attempts: 0,
      isRecurring: true,
      interval: taskConfig.interval || 60000, // 1 minute default
      nextRun: taskConfig.scheduledTime || Date.now(),
    };

    this.tasks.set(taskId, task);
    this.priorityQueue.enqueue(task, task.priority);

    if (!this.isRunning) {
      this.start();
    }

    return taskId;
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

  async processTasks() {
    const now = Date.now();

    // Process ready tasks
    while (
      this.priorityQueue.size() > 0 &&
      this.runningTasks.size < this.maxConcurrentTasks
    ) {
      const task = this.priorityQueue.peek();

      if (task.scheduledTime <= now) {
        this.priorityQueue.dequeue();
        await this.executeTask(task);
      } else {
        break; // No more ready tasks
      }
    }
  }

  async executeTask(task) {
    if (this.runningTasks.has(task.id)) {
      return; // Task already running
    }

    // Check dependencies
    if (!this.areDependenciesMet(task)) {
      // Reschedule task
      this.priorityQueue.enqueue(task, task.priority);
      return;
    }

    this.runningTasks.add(task.id);
    task.status = "running";
    task.startedAt = new Date();
    task.attempts++;

    try {
      // Execute task
      const result = await task.handler(task.metadata);

      // Task completed successfully
      task.status = "completed";
      task.completedAt = new Date();
      task.result = result;

      // Handle recurring tasks
      if (task.isRecurring) {
        task.nextRun = Date.now() + task.interval;
        task.scheduledTime = task.nextRun;
        task.status = "scheduled";
        task.attempts = 0;
        this.priorityQueue.enqueue(task, task.priority);
      }
    } catch (error) {
      // Task failed
      task.status = "failed";
      task.failedAt = new Date();
      task.error = error.message;

      // Handle retries
      if (task.attempts < task.maxRetries) {
        await this.retryService.scheduleRetry(task);
      } else {
        task.status = "permanently_failed";
      }
    } finally {
      this.runningTasks.delete(task.id);
    }
  }

  areDependenciesMet(task) {
    for (const depId of task.dependencies) {
      const depTask = this.tasks.get(depId);
      if (!depTask || depTask.status !== "completed") {
        return false;
      }
    }
    return true;
  }

  cancelTask(taskId) {
    const task = this.tasks.get(taskId);
    if (!task) {
      throw new Error("Task not found");
    }

    if (task.status === "running") {
      throw new Error("Cannot cancel running task");
    }

    task.status = "cancelled";
    task.cancelledAt = new Date();

    return { success: true, message: "Task cancelled successfully" };
  }

  getTaskStatus(taskId) {
    const task = this.tasks.get(taskId);
    if (!task) {
      throw new Error("Task not found");
    }

    return {
      id: task.id,
      name: task.name,
      status: task.status,
      priority: task.priority,
      scheduledTime: task.scheduledTime,
      createdAt: task.createdAt,
      startedAt: task.startedAt,
      completedAt: task.completedAt,
      failedAt: task.failedAt,
      attempts: task.attempts,
      maxRetries: task.maxRetries,
      isRecurring: task.isRecurring,
      nextRun: task.nextRun,
      error: task.error,
      result: task.result,
    };
  }

  getAllTasks(status = null) {
    let tasks = Array.from(this.tasks.values());

    if (status) {
      tasks = tasks.filter((task) => task.status === status);
    }

    return tasks.sort((a, b) => b.createdAt - a.createdAt);
  }

  generateTaskId() {
    return "task_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  }
}

// Priority Queue Implementation
class PriorityQueue {
  constructor() {
    this.items = [];
  }

  enqueue(item, priority) {
    const queueElement = { item, priority };
    let added = false;

    for (let i = 0; i < this.items.length; i++) {
      if (queueElement.priority > this.items[i].priority) {
        this.items.splice(i, 0, queueElement);
        added = true;
        break;
      }
    }

    if (!added) {
      this.items.push(queueElement);
    }
  }

  dequeue() {
    if (this.isEmpty()) {
      return null;
    }
    return this.items.shift().item;
  }

  peek() {
    if (this.isEmpty()) {
      return null;
    }
    return this.items[0].item;
  }

  isEmpty() {
    return this.items.length === 0;
  }

  size() {
    return this.items.length;
  }
}

// Retry Service
class RetryService {
  constructor() {
    this.retryQueue = new Map();
  }

  async scheduleRetry(task) {
    const retryDelay = task.retryDelay * Math.pow(2, task.attempts - 1); // Exponential backoff

    setTimeout(() => {
      task.status = "scheduled";
      task.scheduledTime = Date.now();
      // Re-add to priority queue
    }, retryDelay);
  }
}
```

---

## 🎯 **Interview Discussion Points**

### **System Design Discussion**

1. **Requirements Clarification**

   - Ask about scale, performance, and reliability requirements
   - Understand the problem domain and constraints
   - Identify non-functional requirements

2. **High-Level Architecture**

   - Draw system components and their interactions
   - Show data flow and request flow
   - Identify external dependencies

3. **Detailed Design**

   - Database schema and data modeling
   - API design and interfaces
   - Component interactions and protocols

4. **Scalability and Performance**

   - Identify bottlenecks and scaling strategies
   - Discuss caching and optimization techniques
   - Consider load balancing and partitioning

5. **Reliability and Fault Tolerance**
   - Error handling and recovery mechanisms
   - Monitoring and alerting strategies
   - Backup and disaster recovery

### **Machine Coding Discussion**

1. **Problem Understanding**

   - Clarify requirements and edge cases
   - Discuss trade-offs and design decisions
   - Identify potential challenges

2. **Approach Selection**

   - Explain algorithm and data structure choices
   - Discuss time and space complexity
   - Consider alternative approaches

3. **Implementation Quality**

   - Write clean, readable, and maintainable code
   - Handle edge cases and error conditions
   - Add appropriate logging and monitoring

4. **Testing Strategy**

   - Discuss test cases and scenarios
   - Consider unit tests and integration tests
   - Plan for performance testing

5. **Production Readiness**
   - Discuss deployment and monitoring
   - Consider security and performance implications
   - Plan for maintenance and updates

---

**🎉 This guide provides comprehensive coverage of Razorpay-style machine coding problems with detailed implementations and discussions!**
