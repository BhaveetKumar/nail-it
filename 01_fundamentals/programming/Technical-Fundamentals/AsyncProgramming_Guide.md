# ⚡ Node.js Async Programming Complete Guide

> **Master asynchronous programming in Node.js with Promises, async/await, and event loops**

## 📚 Overview

Asynchronous programming is fundamental to Node.js. This guide covers all aspects of async programming including callbacks, Promises, async/await, event loops, and advanced patterns.

## 🎯 Table of Contents

1. [Event Loop Fundamentals](#event-loop-fundamentals/)
2. [Callbacks](#callbacks/)
3. [Promises](#promises/)
4. [Async/Await](#asyncawait/)
5. [Error Handling](#error-handling/)
6. [Advanced Patterns](#advanced-patterns/)
7. [Performance Optimization](#performance-optimization/)
8. [Best Practices](#best-practices/)

## 🔄 Event Loop Fundamentals

### **Event Loop Architecture**

```javascript
// Event Loop Phases
console.log("1. Start");

setTimeout(() => console.log("2. Timer"), 0);
setImmediate(() => console.log("3. Immediate"));

process.nextTick(() => console.log("4. Next Tick"));

Promise.resolve().then(() => console.log("5. Promise"));

console.log("6. End");

// Output:
// 1. Start
// 6. End
// 4. Next Tick
// 5. Promise
// 2. Timer
// 3. Immediate
```

### **Event Loop Phases**

```javascript
const { performance } = require("perf_hooks");

function demonstrateEventLoop() {
  console.log("=== Event Loop Phases Demo ===");

  // Phase 1: Timer Phase
  setTimeout(() => {
    console.log("Timer Phase - setTimeout");
  }, 0);

  // Phase 2: Pending Callbacks
  setImmediate(() => {
    console.log("Check Phase - setImmediate");
  });

  // Phase 3: Idle, Prepare (internal)

  // Phase 4: Poll Phase
  const fs = require("fs");
  fs.readFile(__filename, () => {
    console.log("Poll Phase - I/O callback");

    // Phase 5: Check Phase
    setImmediate(() => {
      console.log("Check Phase - setImmediate in I/O");
    });
  });

  // Phase 6: Close Callbacks

  // Microtasks (higher priority)
  process.nextTick(() => {
    console.log("Microtask - process.nextTick");
  });

  Promise.resolve().then(() => {
    console.log("Microtask - Promise");
  });
}

demonstrateEventLoop();
```

### **Event Loop Monitoring**

```javascript
const { performance, PerformanceObserver } = require("perf_hooks");

class EventLoopMonitor {
  constructor() {
    this.observations = [];
    this.observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.name === "eventloop") {
          this.observations.push({
            duration: entry.duration,
            timestamp: Date.now(),
          });
        }
      });
    });
    this.observer.observe({ entryTypes: ["measure"] });
  }

  startMonitoring() {
    setInterval(() => {
      const start = performance.now();

      setImmediate(() => {
        const end = performance.now();
        performance.mark("eventloop-start");
        performance.mark("eventloop-end");
        performance.measure("eventloop", "eventloop-start", "eventloop-end");
      });
    }, 1000);
  }

  getStats() {
    if (this.observations.length === 0) return null;

    const durations = this.observations.map((obs) => obs.duration);
    return {
      average: durations.reduce((a, b) => a + b, 0) / durations.length,
      min: Math.min(...durations),
      max: Math.max(...durations),
      count: durations.length,
    };
  }
}

// Usage
const monitor = new EventLoopMonitor();
monitor.startMonitoring();

setTimeout(() => {
  console.log("Event Loop Stats:", monitor.getStats());
}, 5000);
```

## 📞 Callbacks

### **Basic Callback Pattern**

```javascript
// Callback Hell Example
function getUserData(userId, callback) {
  setTimeout(() => {
    callback(null, { id: userId, name: "John Doe" });
  }, 100);
}

function getUserPosts(userId, callback) {
  setTimeout(() => {
    callback(null, [
      { id: 1, title: "Post 1", userId: userId },
      { id: 2, title: "Post 2", userId: userId },
    ]);
  }, 150);
}

function getUserComments(userId, callback) {
  setTimeout(() => {
    callback(null, [
      { id: 1, text: "Comment 1", userId: userId },
      { id: 2, text: "Comment 2", userId: userId },
    ]);
  }, 200);
}

// Callback Hell
function getUserProfile(userId, callback) {
  getUserData(userId, (err, user) => {
    if (err) return callback(err);

    getUserPosts(userId, (err, posts) => {
      if (err) return callback(err);

      getUserComments(userId, (err, comments) => {
        if (err) return callback(err);

        callback(null, {
          user,
          posts,
          comments,
        });
      });
    });
  });
}
```

### **Callback Utilities**

```javascript
// Callback Utilities
class CallbackUtils {
  // Promisify a callback-based function
  static promisify(fn) {
    return function (...args) {
      return new Promise((resolve, reject) => {
        fn.call(this, ...args, (err, result) => {
          if (err) reject(err);
          else resolve(result);
        });
      });
    };
  }

  // Callback with timeout
  static withTimeout(fn, timeout = 5000) {
    return function (...args) {
      const callback = args[args.length - 1];
      let completed = false;

      const timer = setTimeout(() => {
        if (!completed) {
          completed = true;
          callback(new Error("Operation timeout"));
        }
      }, timeout);

      const wrappedCallback = (...cbArgs) => {
        if (!completed) {
          completed = true;
          clearTimeout(timer);
          callback(...cbArgs);
        }
      };

      args[args.length - 1] = wrappedCallback;
      fn.apply(this, args);
    };
  }

  // Retry with callback
  static withRetry(fn, maxRetries = 3, delay = 1000) {
    return function (...args) {
      const callback = args[args.length - 1];
      let attempts = 0;

      const attempt = () => {
        attempts++;
        fn.apply(this, [
          ...args.slice(0, -1),
          (err, result) => {
            if (err && attempts < maxRetries) {
              setTimeout(attempt, delay * attempts);
            } else {
              callback(err, result);
            }
          },
        ]);
      };

      attempt();
    };
  }
}

// Usage examples
const fs = require("fs");
const readFileAsync = CallbackUtils.promisify(fs.readFile);
const readFileWithTimeout = CallbackUtils.withTimeout(fs.readFile, 2000);
const readFileWithRetry = CallbackUtils.withRetry(fs.readFile, 3, 1000);
```

## 🔮 Promises

### **Promise Fundamentals**

```javascript
// Basic Promise Creation
function createPromise(value, shouldReject = false) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (shouldReject) {
        reject(new Error("Promise rejected"));
      } else {
        resolve(value);
      }
    }, 100);
  });
}

// Promise Chaining
function demonstratePromiseChaining() {
  return createPromise(1)
    .then((value) => {
      console.log("First then:", value);
      return value * 2;
    })
    .then((value) => {
      console.log("Second then:", value);
      return createPromise(value + 1);
    })
    .then((value) => {
      console.log("Third then:", value);
      return value;
    })
    .catch((error) => {
      console.error("Error in chain:", error.message);
      throw error;
    });
}

// Promise.all - Wait for all promises
function demonstratePromiseAll() {
  const promises = [
    createPromise("Result 1"),
    createPromise("Result 2"),
    createPromise("Result 3"),
  ];

  return Promise.all(promises)
    .then((results) => {
      console.log("All promises resolved:", results);
      return results;
    })
    .catch((error) => {
      console.error("One or more promises rejected:", error);
    });
}

// Promise.allSettled - Wait for all promises to settle
function demonstratePromiseAllSettled() {
  const promises = [
    createPromise("Success 1"),
    createPromise("Success 2", true), // This will reject
    createPromise("Success 3"),
  ];

  return Promise.allSettled(promises).then((results) => {
    console.log("All promises settled:", results);
    return results;
  });
}

// Promise.race - First promise to settle wins
function demonstratePromiseRace() {
  const promises = [
    createPromise("Fast", false),
    createPromise("Slow", false),
    createPromise("Medium", false),
  ];

  return Promise.race(promises).then((result) => {
    console.log("Race winner:", result);
    return result;
  });
}
```

### **Advanced Promise Patterns**

```javascript
// Promise Pool - Limit concurrent promises
class PromisePool {
  constructor(concurrency = 5) {
    this.concurrency = concurrency;
    this.running = 0;
    this.queue = [];
  }

  async add(promiseFactory) {
    return new Promise((resolve, reject) => {
      this.queue.push({
        promiseFactory,
        resolve,
        reject,
      });
      this.process();
    });
  }

  async process() {
    if (this.running >= this.concurrency || this.queue.length === 0) {
      return;
    }

    this.running++;
    const { promiseFactory, resolve, reject } = this.queue.shift();

    try {
      const result = await promiseFactory();
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.running--;
      this.process();
    }
  }
}

// Usage
async function demonstratePromisePool() {
  const pool = new PromisePool(3);

  const tasks = Array.from({ length: 10 }, (_, i) =>
    pool.add(() => createPromise(`Task ${i + 1}`))
  );

  const results = await Promise.all(tasks);
  console.log("Pool results:", results);
}

// Promise Retry Pattern
function withRetry(fn, maxRetries = 3, delay = 1000) {
  return async function (...args) {
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await fn(...args);
      } catch (error) {
        lastError = error;
        console.log(`Attempt ${attempt} failed:`, error.message);

        if (attempt < maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, delay * attempt));
        }
      }
    }

    throw lastError;
  };
}

// Promise Timeout Pattern
function withTimeout(promise, timeout = 5000) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Promise timeout")), timeout)
    ),
  ]);
}

// Promise Memoization
function memoize(fn, ttl = 60000) {
  const cache = new Map();

  return async function (...args) {
    const key = JSON.stringify(args);
    const cached = cache.get(key);

    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.value;
    }

    const result = await fn(...args);
    cache.set(key, {
      value: result,
      timestamp: Date.now(),
    });

    return result;
  };
}
```

## ⏳ Async/Await

### **Basic Async/Await**

```javascript
// Converting callbacks to async/await
async function getUserProfileAsync(userId) {
  try {
    const user = await getUserDataAsync(userId);
    const posts = await getUserPostsAsync(userId);
    const comments = await getUserCommentsAsync(userId);

    return {
      user,
      posts,
      comments,
    };
  } catch (error) {
    console.error("Error fetching user profile:", error);
    throw error;
  }
}

// Parallel execution with async/await
async function getUserProfileParallel(userId) {
  try {
    const [user, posts, comments] = await Promise.all([
      getUserDataAsync(userId),
      getUserPostsAsync(userId),
      getUserCommentsAsync(userId),
    ]);

    return { user, posts, comments };
  } catch (error) {
    console.error("Error fetching user profile:", error);
    throw error;
  }
}

// Sequential vs Parallel execution
async function demonstrateExecutionPatterns() {
  console.log("=== Sequential Execution ===");
  const startSequential = Date.now();

  await createPromise("Task 1");
  await createPromise("Task 2");
  await createPromise("Task 3");

  const sequentialTime = Date.now() - startSequential;
  console.log(`Sequential time: ${sequentialTime}ms`);

  console.log("=== Parallel Execution ===");
  const startParallel = Date.now();

  await Promise.all([
    createPromise("Task 1"),
    createPromise("Task 2"),
    createPromise("Task 3"),
  ]);

  const parallelTime = Date.now() - startParallel;
  console.log(`Parallel time: ${parallelTime}ms`);
}
```

### **Advanced Async/Await Patterns**

```javascript
// Async Generator
async function* asyncGenerator() {
  for (let i = 0; i < 5; i++) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    yield i;
  }
}

async function consumeAsyncGenerator() {
  for await (const value of asyncGenerator()) {
    console.log("Generated value:", value);
  }
}

// Async Iterator
class AsyncDataStream {
  constructor(data) {
    this.data = data;
    this.index = 0;
  }

  async *[Symbol.asyncIterator]() {
    while (this.index < this.data.length) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      yield this.data[this.index++];
    }
  }
}

async function consumeAsyncIterator() {
  const stream = new AsyncDataStream([1, 2, 3, 4, 5]);

  for await (const value of stream) {
    console.log("Stream value:", value);
  }
}

// Async Queue
class AsyncQueue {
  constructor() {
    this.queue = [];
    this.resolvers = [];
  }

  async enqueue(item) {
    if (this.resolvers.length > 0) {
      const resolve = this.resolvers.shift();
      resolve(item);
    } else {
      this.queue.push(item);
    }
  }

  async dequeue() {
    if (this.queue.length > 0) {
      return this.queue.shift();
    } else {
      return new Promise((resolve) => {
        this.resolvers.push(resolve);
      });
    }
  }
}

// Async Semaphore
class AsyncSemaphore {
  constructor(permits) {
    this.permits = permits;
    this.waiting = [];
  }

  async acquire() {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise((resolve) => {
      this.waiting.push(resolve);
    });
  }

  release() {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift();
      resolve();
    } else {
      this.permits++;
    }
  }
}

// Usage
async function demonstrateAsyncSemaphore() {
  const semaphore = new AsyncSemaphore(2);

  async function worker(id) {
    await semaphore.acquire();
    console.log(`Worker ${id} started`);

    await new Promise((resolve) => setTimeout(resolve, 1000));

    console.log(`Worker ${id} finished`);
    semaphore.release();
  }

  // Start 5 workers with semaphore limit of 2
  await Promise.all([worker(1), worker(2), worker(3), worker(4), worker(5)]);
}
```

## ❌ Error Handling

### **Error Handling Patterns**

```javascript
// Try-Catch with Async/Await
async function robustAsyncFunction() {
  try {
    const result = await riskyAsyncOperation();
    return result;
  } catch (error) {
    console.error("Operation failed:", error.message);

    // Handle specific error types
    if (error.code === "ENOENT") {
      throw new Error("File not found");
    } else if (error.code === "ETIMEDOUT") {
      throw new Error("Operation timeout");
    } else {
      throw new Error("Unknown error occurred");
    }
  }
}

// Error Boundary Pattern
class AsyncErrorBoundary {
  constructor(fn, fallback) {
    this.fn = fn;
    this.fallback = fallback;
  }

  async execute(...args) {
    try {
      return await this.fn(...args);
    } catch (error) {
      console.error("Error caught by boundary:", error);
      return this.fallback(error, ...args);
    }
  }
}

// Usage
const safeOperation = new AsyncErrorBoundary(
  async (data) => {
    // Risky operation
    if (Math.random() > 0.5) {
      throw new Error("Random failure");
    }
    return `Processed: ${data}`;
  },
  (error, data) => {
    return `Fallback result for: ${data}`;
  }
);

// Error Aggregation
async function aggregateErrors(operations) {
  const results = await Promise.allSettled(operations);
  const errors = results
    .filter((result) => result.status === "rejected")
    .map((result) => result.reason);

  if (errors.length > 0) {
    console.error("Multiple errors occurred:", errors);
    throw new Error(`Failed operations: ${errors.length}`);
  }

  return results.map((result) => result.value);
}

// Retry with Exponential Backoff
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
  let lastError;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`Attempt ${attempt + 1} failed, retrying in ${delay}ms`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}
```

## 🚀 Advanced Patterns

### **Stream Processing**

```javascript
const { Readable, Writable, Transform, pipeline } = require("stream");
const { promisify } = require("util");

const pipelineAsync = promisify(pipeline);

// Async Stream Processing
class AsyncDataProcessor extends Transform {
  constructor(options = {}) {
    super({ objectMode: true, ...options });
  }

  async _transform(chunk, encoding, callback) {
    try {
      // Simulate async processing
      await new Promise((resolve) => setTimeout(resolve, 100));

      const processed = {
        ...chunk,
        processed: true,
        timestamp: Date.now(),
      };

      callback(null, processed);
    } catch (error) {
      callback(error);
    }
  }
}

// Usage
async function processDataStream() {
  const data = Array.from({ length: 10 }, (_, i) => ({
    id: i,
    data: `item-${i}`,
  }));

  const readable = new Readable({
    objectMode: true,
    read() {
      const item = data.shift();
      if (item) {
        this.push(item);
      } else {
        this.push(null);
      }
    },
  });

  const processor = new AsyncDataProcessor();

  const writable = new Writable({
    objectMode: true,
    write(chunk, encoding, callback) {
      console.log("Processed:", chunk);
      callback();
    },
  });

  await pipelineAsync(readable, processor, writable);
}
```

### **Event-Driven Architecture**

```javascript
const EventEmitter = require("events");

class AsyncEventEmitter extends EventEmitter {
  async emitAsync(event, ...args) {
    const listeners = this.listeners(event);

    if (listeners.length === 0) {
      return false;
    }

    // Execute listeners in parallel
    const promises = listeners.map((listener) => {
      try {
        const result = listener(...args);
        return Promise.resolve(result);
      } catch (error) {
        return Promise.reject(error);
      }
    });

    try {
      await Promise.all(promises);
      return true;
    } catch (error) {
      this.emit("error", error);
      return false;
    }
  }

  async emitSequential(event, ...args) {
    const listeners = this.listeners(event);

    if (listeners.length === 0) {
      return false;
    }

    // Execute listeners sequentially
    for (const listener of listeners) {
      try {
        await listener(...args);
      } catch (error) {
        this.emit("error", error);
        return false;
      }
    }

    return true;
  }
}

// Usage
const emitter = new AsyncEventEmitter();

emitter.on("process", async (data) => {
  console.log("Processing:", data);
  await new Promise((resolve) => setTimeout(resolve, 100));
  console.log("Processed:", data);
});

emitter.on("process", async (data) => {
  console.log("Post-processing:", data);
  await new Promise((resolve) => setTimeout(resolve, 50));
  console.log("Post-processed:", data);
});

async function demonstrateAsyncEvents() {
  await emitter.emitAsync("process", "test-data");
  console.log("All processing complete");
}
```

## ⚡ Performance Optimization

### **Performance Monitoring**

```javascript
const { performance, PerformanceObserver } = require("perf_hooks");

class AsyncPerformanceMonitor {
  constructor() {
    this.marks = new Map();
    this.observations = [];
    this.observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        this.observations.push({
          name: entry.name,
          duration: entry.duration,
          startTime: entry.startTime,
          timestamp: Date.now(),
        });
      });
    });
    this.observer.observe({ entryTypes: ["measure"] });
  }

  startTiming(name) {
    this.marks.set(name, performance.now());
    performance.mark(`${name}-start`);
  }

  endTiming(name) {
    const startTime = this.marks.get(name);
    if (startTime) {
      const endTime = performance.now();
      performance.mark(`${name}-end`);
      performance.measure(name, `${name}-start`, `${name}-end`);

      this.marks.delete(name);
      return endTime - startTime;
    }
    return null;
  }

  getStats() {
    const stats = {};

    this.observations.forEach((obs) => {
      if (!stats[obs.name]) {
        stats[obs.name] = {
          count: 0,
          totalDuration: 0,
          minDuration: Infinity,
          maxDuration: 0,
        };
      }

      const stat = stats[obs.name];
      stat.count++;
      stat.totalDuration += obs.duration;
      stat.minDuration = Math.min(stat.minDuration, obs.duration);
      stat.maxDuration = Math.max(stat.maxDuration, obs.duration);
    });

    // Calculate averages
    Object.keys(stats).forEach((name) => {
      const stat = stats[name];
      stat.averageDuration = stat.totalDuration / stat.count;
    });

    return stats;
  }
}

// Usage
const monitor = new AsyncPerformanceMonitor();

async function monitoredAsyncOperation() {
  monitor.startTiming("async-operation");

  await new Promise((resolve) => setTimeout(resolve, 100));

  const duration = monitor.endTiming("async-operation");
  console.log(`Operation took ${duration}ms`);
}

// Run multiple operations
async function runPerformanceTest() {
  for (let i = 0; i < 10; i++) {
    await monitoredAsyncOperation();
  }

  console.log("Performance Stats:", monitor.getStats());
}
```

## 🎯 Best Practices

### **1. Error Handling**

```javascript
// Good: Proper error handling
async function goodAsyncFunction() {
  try {
    const result = await riskyOperation();
    return { success: true, data: result };
  } catch (error) {
    console.error("Operation failed:", error);
    return { success: false, error: error.message };
  }
}

// Bad: Unhandled promise rejection
async function badAsyncFunction() {
  const result = await riskyOperation(); // No error handling
  return result;
}
```

### **2. Resource Management**

```javascript
// Good: Proper resource cleanup
async function withResourceCleanup() {
  const resource = await acquireResource();

  try {
    return await useResource(resource);
  } finally {
    await cleanupResource(resource);
  }
}

// Good: Using async iterators for large datasets
async function* processLargeDataset() {
  for (let i = 0; i < 1000000; i++) {
    yield await processItem(i);
  }
}
```

### **3. Concurrency Control**

```javascript
// Good: Controlled concurrency
async function processWithConcurrency(items, concurrency = 5) {
  const results = [];

  for (let i = 0; i < items.length; i += concurrency) {
    const batch = items.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map((item) => processItem(item))
    );
    results.push(...batchResults);
  }

  return results;
}
```

### **4. Timeout Handling**

```javascript
// Good: Always use timeouts for external operations
async function fetchWithTimeout(url, timeout = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}
```

---

**🎉 Master these async programming patterns to build robust, performant Node.js applications!**

**Good luck with your Node.js async programming journey! 🚀**
