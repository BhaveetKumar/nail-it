# 🚀 Node.js Expertise: From Zero to Production-Ready

> **Complete guide to mastering Node.js for backend engineering roles**

## 🎯 **Learning Objectives**

- Master Node.js fundamentals and advanced concepts
- Understand event-driven architecture and async programming
- Build scalable and performant Node.js applications
- Implement production-ready patterns and best practices
- Prepare for Node.js interviews at top tech companies

## 📚 **Table of Contents**

1. [Node.js Fundamentals](#nodejs-fundamentals)
2. [Event Loop and Async Programming](#event-loop-and-async-programming)
3. [Modules and Package Management](#modules-and-package-management)
4. [File System and Streams](#file-system-and-streams)
5. [HTTP and Web Servers](#http-and-web-servers)
6. [Database Integration](#database-integration)
7. [Testing and Debugging](#testing-and-debugging)
8. [Performance Optimization](#performance-optimization)
9. [Security Best Practices](#security-best-practices)
10. [Production Deployment](#production-deployment)
11. [Interview Questions](#interview-questions)

---

## 🚀 **Node.js Fundamentals**

### **What is Node.js?**

Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine. It allows you to run JavaScript on the server-side, enabling full-stack JavaScript development.

### **Key Features**

- **Event-driven**: Non-blocking I/O operations
- **Single-threaded**: Uses event loop for concurrency
- **Cross-platform**: Runs on Windows, macOS, and Linux
- **NPM ecosystem**: Largest package registry
- **Fast execution**: V8 engine optimization

### **Node.js Architecture**

```javascript
// Node.js Architecture Overview
const fs = require('fs');
const http = require('http');

// Event-driven architecture example
const EventEmitter = require('events');

class MyEmitter extends EventEmitter {}

const myEmitter = new MyEmitter();

myEmitter.on('event', (data) => {
  console.log('Event received:', data);
});

myEmitter.emit('event', 'Hello Node.js!');
```

---

## ⚡ **Event Loop and Async Programming**

### **Event Loop Concept**

The event loop is the core of Node.js's non-blocking I/O operations. It allows Node.js to perform non-blocking I/O operations despite being single-threaded.

### **Event Loop Phases**

```javascript
// Event Loop Phases Example
console.log('1. Start');

setTimeout(() => {
  console.log('2. Timer');
}, 0);

setImmediate(() => {
  console.log('3. Immediate');
});

process.nextTick(() => {
  console.log('4. Next Tick');
});

console.log('5. End');

// Output:
// 1. Start
// 5. End
// 4. Next Tick
// 2. Timer
// 3. Immediate
```

### **Promises and Async/Await**

```javascript
// Promises Example
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve('Data fetched successfully');
    }, 1000);
  });
}

// Using Promises
fetchData()
  .then(data => console.log(data))
  .catch(error => console.error(error));

// Using Async/Await
async function handleData() {
  try {
    const data = await fetchData();
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

handleData();
```

### **Advanced Async Patterns**

```javascript
// Promise.all for concurrent operations
async function fetchMultipleData() {
  const promises = [
    fetch('https://api.example.com/data1'),
    fetch('https://api.example.com/data2'),
    fetch('https://api.example.com/data3')
  ];
  
  try {
    const results = await Promise.all(promises);
    return results.map(response => response.json());
  } catch (error) {
    console.error('One or more requests failed:', error);
  }
}

// Promise.allSettled for handling partial failures
async function fetchWithFallback() {
  const promises = [
    fetch('https://api.example.com/primary'),
    fetch('https://api.example.com/fallback')
  ];
  
  const results = await Promise.allSettled(promises);
  
  for (const result of results) {
    if (result.status === 'fulfilled') {
      return result.value;
    }
  }
  
  throw new Error('All requests failed');
}
```

---

## 📦 **Modules and Package Management**

### **CommonJS Modules**

```javascript
// math.js - Module definition
const add = (a, b) => a + b;
const subtract = (a, b) => a - b;
const multiply = (a, b) => a * b;
const divide = (a, b) => b !== 0 ? a / b : null;

module.exports = {
  add,
  subtract,
  multiply,
  divide
};

// app.js - Module usage
const math = require('./math');

console.log(math.add(5, 3)); // 8
console.log(math.multiply(4, 6)); // 24
```

### **ES6 Modules**

```javascript
// utils.js - ES6 Module
export const formatDate = (date) => {
  return new Date(date).toLocaleDateString();
};

export const validateEmail = (email) => {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
};

export default class Utils {
  static capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
}

// app.js - ES6 Module usage
import { formatDate, validateEmail } from './utils.js';
import Utils from './utils.js';

console.log(formatDate(new Date()));
console.log(validateEmail('test@example.com'));
console.log(Utils.capitalize('hello'));
```

### **Package.json and NPM**

```json
{
  "name": "nodejs-example",
  "version": "1.0.0",
  "description": "Node.js example application",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest",
    "lint": "eslint .",
    "build": "webpack"
  },
  "dependencies": {
    "express": "^4.18.0",
    "mongoose": "^6.0.0",
    "redis": "^4.0.0"
  },
  "devDependencies": {
    "nodemon": "^2.0.0",
    "jest": "^27.0.0",
    "eslint": "^8.0.0"
  },
  "engines": {
    "node": ">=16.0.0"
  }
}
```

---

## 📁 **File System and Streams**

### **File System Operations**

```javascript
const fs = require('fs').promises;
const path = require('path');

// Async file operations
async function fileOperations() {
  try {
    // Read file
    const data = await fs.readFile('input.txt', 'utf8');
    console.log('File content:', data);
    
    // Write file
    await fs.writeFile('output.txt', 'Hello World!', 'utf8');
    
    // Append to file
    await fs.appendFile('output.txt', '\nNew line', 'utf8');
    
    // Check if file exists
    const exists = await fs.access('input.txt').then(() => true).catch(() => false);
    console.log('File exists:', exists);
    
    // Get file stats
    const stats = await fs.stat('input.txt');
    console.log('File size:', stats.size);
    console.log('Created:', stats.birthtime);
    
  } catch (error) {
    console.error('File operation error:', error);
  }
}

fileOperations();
```

### **Streams**

```javascript
const fs = require('fs');
const { Transform, PassThrough } = require('stream');

// Readable Stream
const readableStream = fs.createReadStream('large-file.txt', { encoding: 'utf8' });

readableStream.on('data', (chunk) => {
  console.log('Received chunk:', chunk.length, 'bytes');
});

readableStream.on('end', () => {
  console.log('Stream ended');
});

readableStream.on('error', (error) => {
  console.error('Stream error:', error);
});

// Writable Stream
const writableStream = fs.createWriteStream('output.txt');

writableStream.write('Hello ');
writableStream.write('World!');
writableStream.end();

// Transform Stream
const transformStream = new Transform({
  transform(chunk, encoding, callback) {
    const transformed = chunk.toString().toUpperCase();
    callback(null, transformed);
  }
});

// Pipe streams
readableStream.pipe(transformStream).pipe(writableStream);
```

### **Advanced Stream Patterns**

```javascript
const { pipeline } = require('stream');
const { promisify } = require('util');
const pipelineAsync = promisify(pipeline);

// Error handling with streams
async function processFile(inputFile, outputFile) {
  try {
    await pipelineAsync(
      fs.createReadStream(inputFile),
      new Transform({
        transform(chunk, encoding, callback) {
          // Process chunk
          const processed = chunk.toString().replace(/old/g, 'new');
          callback(null, processed);
        }
      }),
      fs.createWriteStream(outputFile)
    );
    console.log('File processed successfully');
  } catch (error) {
    console.error('Pipeline error:', error);
  }
}
```

---

## 🌐 **HTTP and Web Servers**

### **Built-in HTTP Module**

```javascript
const http = require('http');
const url = require('url');

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const path = parsedUrl.pathname;
  const method = req.method;
  
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  // Route handling
  if (method === 'GET' && path === '/api/users') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ users: ['John', 'Jane', 'Bob'] }));
  } else if (method === 'POST' && path === '/api/users') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', () => {
      try {
        const userData = JSON.parse(body);
        res.writeHead(201, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ message: 'User created', user: userData }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### **Express.js Framework**

```javascript
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.get('/api/users', async (req, res) => {
  try {
    const users = await getUsers();
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/users', async (req, res) => {
  try {
    const userData = req.body;
    const newUser = await createUser(userData);
    res.status(201).json(newUser);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Express server running on port ${PORT}`);
});
```

---

## 🗄️ **Database Integration**

### **MongoDB with Mongoose**

```javascript
const mongoose = require('mongoose');

// Schema definition
const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  age: {
    type: Number,
    min: 0,
    max: 120
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Indexes
userSchema.index({ email: 1 });
userSchema.index({ createdAt: -1 });

// Virtual fields
userSchema.virtual('isAdult').get(function() {
  return this.age >= 18;
});

// Instance methods
userSchema.methods.getDisplayName = function() {
  return `${this.name} (${this.email})`;
};

// Static methods
userSchema.statics.findByEmail = function(email) {
  return this.findOne({ email: email.toLowerCase() });
};

// Middleware
userSchema.pre('save', function(next) {
  this.email = this.email.toLowerCase();
  next();
});

const User = mongoose.model('User', userSchema);

// Database operations
class UserService {
  async createUser(userData) {
    try {
      const user = new User(userData);
      await user.save();
      return user;
    } catch (error) {
      if (error.code === 11000) {
        throw new Error('Email already exists');
      }
      throw error;
    }
  }
  
  async getUserById(id) {
    return await User.findById(id);
  }
  
  async updateUser(id, updateData) {
    return await User.findByIdAndUpdate(id, updateData, { new: true });
  }
  
  async deleteUser(id) {
    return await User.findByIdAndDelete(id);
  }
  
  async getUsersWithPagination(page = 1, limit = 10) {
    const skip = (page - 1) * limit;
    const users = await User.find()
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });
    
    const total = await User.countDocuments();
    
    return {
      users,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    };
  }
}

module.exports = { User, UserService };
```

### **PostgreSQL with pg**

```javascript
const { Pool } = require('pg');

const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

class DatabaseService {
  async query(text, params) {
    const start = Date.now();
    try {
      const res = await pool.query(text, params);
      const duration = Date.now() - start;
      console.log('Executed query', { text, duration, rows: res.rowCount });
      return res;
    } catch (error) {
      console.error('Database query error:', error);
      throw error;
    }
  }
  
  async getClient() {
    return await pool.connect();
  }
  
  async transaction(callback) {
    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
}

const db = new DatabaseService();

// User operations
class UserRepository {
  async create(userData) {
    const query = `
      INSERT INTO users (name, email, age, created_at)
      VALUES ($1, $2, $3, $4)
      RETURNING *
    `;
    const values = [userData.name, userData.email, userData.age, new Date()];
    const result = await db.query(query, values);
    return result.rows[0];
  }
  
  async findById(id) {
    const query = 'SELECT * FROM users WHERE id = $1';
    const result = await db.query(query, [id]);
    return result.rows[0];
  }
  
  async findByEmail(email) {
    const query = 'SELECT * FROM users WHERE email = $1';
    const result = await db.query(query, [email]);
    return result.rows[0];
  }
  
  async update(id, updateData) {
    const fields = Object.keys(updateData);
    const values = Object.values(updateData);
    const setClause = fields.map((field, index) => `${field} = $${index + 2}`).join(', ');
    
    const query = `
      UPDATE users 
      SET ${setClause}, updated_at = $1
      WHERE id = $${fields.length + 2}
      RETURNING *
    `;
    
    const result = await db.query(query, [new Date(), ...values, id]);
    return result.rows[0];
  }
  
  async delete(id) {
    const query = 'DELETE FROM users WHERE id = $1 RETURNING *';
    const result = await db.query(query, [id]);
    return result.rows[0];
  }
}

module.exports = { DatabaseService, UserRepository };
```

---

## 🧪 **Testing and Debugging**

### **Jest Testing Framework**

```javascript
// user.test.js
const { UserService } = require('./userService');
const { User } = require('./models/user');

// Mock mongoose
jest.mock('mongoose');

describe('UserService', () => {
  let userService;
  
  beforeEach(() => {
    userService = new UserService();
    jest.clearAllMocks();
  });
  
  describe('createUser', () => {
    it('should create a user successfully', async () => {
      const userData = {
        name: 'John Doe',
        email: 'john@example.com',
        age: 30
      };
      
      const mockUser = { ...userData, _id: '123', createdAt: new Date() };
      User.prototype.save = jest.fn().mockResolvedValue(mockUser);
      
      const result = await userService.createUser(userData);
      
      expect(result).toEqual(mockUser);
      expect(User.prototype.save).toHaveBeenCalledTimes(1);
    });
    
    it('should throw error for duplicate email', async () => {
      const userData = {
        name: 'John Doe',
        email: 'john@example.com',
        age: 30
      };
      
      const error = new Error('Duplicate key');
      error.code = 11000;
      User.prototype.save = jest.fn().mockRejectedValue(error);
      
      await expect(userService.createUser(userData))
        .rejects.toThrow('Email already exists');
    });
  });
  
  describe('getUserById', () => {
    it('should return user by id', async () => {
      const mockUser = {
        _id: '123',
        name: 'John Doe',
        email: 'john@example.com'
      };
      
      User.findById = jest.fn().mockResolvedValue(mockUser);
      
      const result = await userService.getUserById('123');
      
      expect(result).toEqual(mockUser);
      expect(User.findById).toHaveBeenCalledWith('123');
    });
  });
});
```

### **Integration Testing**

```javascript
// app.test.js
const request = require('supertest');
const app = require('./app');
const { connectDB, disconnectDB } = require('./config/database');

describe('API Integration Tests', () => {
  beforeAll(async () => {
    await connectDB();
  });
  
  afterAll(async () => {
    await disconnectDB();
  });
  
  describe('GET /api/users', () => {
    it('should return list of users', async () => {
      const response = await request(app)
        .get('/api/users')
        .expect(200);
      
      expect(response.body).toHaveProperty('users');
      expect(Array.isArray(response.body.users)).toBe(true);
    });
  });
  
  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const userData = {
        name: 'Test User',
        email: 'test@example.com',
        age: 25
      };
      
      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201);
      
      expect(response.body).toHaveProperty('name', userData.name);
      expect(response.body).toHaveProperty('email', userData.email);
    });
    
    it('should return 400 for invalid data', async () => {
      const invalidData = {
        name: '',
        email: 'invalid-email'
      };
      
      await request(app)
        .post('/api/users')
        .send(invalidData)
        .expect(400);
    });
  });
});
```

### **Debugging Techniques**

```javascript
// Debugging with console methods
console.log('Basic logging');
console.error('Error message');
console.warn('Warning message');
console.info('Info message');
console.debug('Debug message');

// Debugging with util.inspect
const util = require('util');

const complexObject = {
  name: 'John',
  age: 30,
  address: {
    street: '123 Main St',
    city: 'New York'
  },
  hobbies: ['reading', 'coding', 'gaming']
};

console.log(util.inspect(complexObject, { 
  showHidden: false, 
  depth: null, 
  colors: true 
}));

// Performance debugging
console.time('operation');
// ... some operation
console.timeEnd('operation');

// Memory usage debugging
const used = process.memoryUsage();
console.log('Memory usage:', {
  rss: `${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB`,
  heapTotal: `${Math.round(used.heapTotal / 1024 / 1024 * 100) / 100} MB`,
  heapUsed: `${Math.round(used.heapUsed / 1024 / 1024 * 100) / 100} MB`,
  external: `${Math.round(used.external / 1024 / 1024 * 100) / 100} MB`
});
```

---

## ⚡ **Performance Optimization**

### **Memory Management**

```javascript
// Memory optimization techniques
class MemoryOptimizedService {
  constructor() {
    this.cache = new Map();
    this.maxCacheSize = 1000;
  }
  
  // Implement LRU cache
  set(key, value) {
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }
  
  get(key) {
    const value = this.cache.get(key);
    if (value) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }
  
  // Clear cache periodically
  clearCache() {
    this.cache.clear();
  }
}

// Garbage collection optimization
function optimizeGarbageCollection() {
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  // Clear large objects
  const largeObject = null;
  
  // Use weak references for temporary data
  const weakMap = new WeakMap();
  const tempData = { data: 'temporary' };
  weakMap.set(tempData, 'value');
}

// Memory leak prevention
class LeakPreventer {
  constructor() {
    this.listeners = new Map();
  }
  
  addListener(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }
  
  removeAllListeners() {
    this.listeners.clear();
  }
}
```

### **CPU Optimization**

```javascript
// Worker threads for CPU-intensive tasks
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (isMainThread) {
  // Main thread
  function runCPUIntensiveTask(data) {
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
  }
  
  // Usage
  async function main() {
    try {
      const result = await runCPUIntensiveTask([1, 2, 3, 4, 5]);
      console.log('Result:', result);
    } catch (error) {
      console.error('Error:', error);
    }
  }
  
  main();
} else {
  // Worker thread
  const data = workerData;
  
  // CPU-intensive computation
  function fibonacci(n) {
    if (n < 2) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
  }
  
  const result = data.map(n => fibonacci(n));
  parentPort.postMessage(result);
}
```

### **I/O Optimization**

```javascript
// Connection pooling
const { Pool } = require('pg');

const pool = new Pool({
  max: 20, // Maximum number of clients in the pool
  idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
  connectionTimeoutMillis: 2000, // Return an error after 2 seconds if connection could not be established
});

// Batch operations
async function batchInsert(records) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    
    const values = records.map((record, index) => 
      `($${index * 3 + 1}, $${index * 3 + 2}, $${index * 3 + 3})`
    ).join(', ');
    
    const query = `
      INSERT INTO users (name, email, age) 
      VALUES ${values}
    `;
    
    const params = records.flatMap(record => [record.name, record.email, record.age]);
    
    await client.query(query, params);
    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

// Streaming large datasets
async function streamLargeDataset() {
  const client = await pool.connect();
  try {
    const query = 'SELECT * FROM large_table';
    const result = client.query(query);
    
    result.on('row', (row) => {
      // Process each row as it comes
      processRow(row);
    });
    
    result.on('end', () => {
      console.log('Streaming completed');
    });
    
    result.on('error', (error) => {
      console.error('Streaming error:', error);
    });
  } finally {
    client.release();
  }
}
```

---

## 🔒 **Security Best Practices**

### **Input Validation and Sanitization**

```javascript
const Joi = require('joi');
const validator = require('validator');
const xss = require('xss');

// Input validation with Joi
const userSchema = Joi.object({
  name: Joi.string().min(2).max(50).required(),
  email: Joi.string().email().required(),
  age: Joi.number().integer().min(0).max(120),
  password: Joi.string().min(8).pattern(new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#\$%\^&\*])'))
});

function validateUserInput(data) {
  const { error, value } = userSchema.validate(data);
  if (error) {
    throw new Error(`Validation error: ${error.details[0].message}`);
  }
  return value;
}

// XSS prevention
function sanitizeInput(input) {
  if (typeof input === 'string') {
    return xss(input, {
      whiteList: {}, // No HTML tags allowed
      stripIgnoreTag: true,
      stripIgnoreTagBody: ['script']
    });
  }
  return input;
}

// SQL injection prevention
function buildSafeQuery(table, conditions) {
  const whereClause = Object.keys(conditions)
    .map((key, index) => `${key} = $${index + 1}`)
    .join(' AND ');
  
  const values = Object.values(conditions);
  const query = `SELECT * FROM ${table} WHERE ${whereClause}`;
  
  return { query, values };
}
```

### **Authentication and Authorization**

```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const rateLimit = require('express-rate-limit');

// Password hashing
async function hashPassword(password) {
  const saltRounds = 12;
  return await bcrypt.hash(password, saltRounds);
}

async function comparePassword(password, hash) {
  return await bcrypt.compare(password, hash);
}

// JWT token management
class TokenManager {
  constructor(secretKey) {
    this.secretKey = secretKey;
    this.accessTokenExpiry = '15m';
    this.refreshTokenExpiry = '7d';
  }
  
  generateAccessToken(payload) {
    return jwt.sign(payload, this.secretKey, { 
      expiresIn: this.accessTokenExpiry 
    });
  }
  
  generateRefreshToken(payload) {
    return jwt.sign(payload, this.secretKey, { 
      expiresIn: this.refreshTokenExpiry 
    });
  }
  
  verifyToken(token) {
    try {
      return jwt.verify(token, this.secretKey);
    } catch (error) {
      throw new Error('Invalid token');
    }
  }
  
  decodeToken(token) {
    return jwt.decode(token);
  }
}

// Authentication middleware
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }
  
  try {
    const tokenManager = new TokenManager(process.env.JWT_SECRET);
    const decoded = tokenManager.verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(403).json({ error: 'Invalid token' });
  }
}

// Role-based authorization
function authorize(roles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
    
    next();
  };
}

// Rate limiting for authentication
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // limit each IP to 5 requests per windowMs
  message: 'Too many authentication attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});
```

### **Security Headers and CORS**

```javascript
const helmet = require('helmet');
const cors = require('cors');

// Security headers with Helmet
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    const allowedOrigins = [
      'https://example.com',
      'https://www.example.com',
      'http://localhost:3000'
    ];
    
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));

// Additional security middleware
app.use((req, res, next) => {
  // Remove X-Powered-By header
  res.removeHeader('X-Powered-By');
  
  // Set security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  
  next();
});
```

---

## 🚀 **Production Deployment**

### **Environment Configuration**

```javascript
// config/environment.js
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables
dotenv.config({ path: path.join(__dirname, `../.env.${process.env.NODE_ENV || 'development'}`) });

const config = {
  development: {
    port: process.env.PORT || 3000,
    database: {
      url: process.env.DATABASE_URL || 'mongodb://localhost:27017/dev_db',
      options: {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      }
    },
    redis: {
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
    },
    jwt: {
      secret: process.env.JWT_SECRET || 'dev-secret',
      expiresIn: '24h'
    }
  },
  
  production: {
    port: process.env.PORT || 3000,
    database: {
      url: process.env.DATABASE_URL,
      options: {
        useNewUrlParser: true,
        useUnifiedTopology: true,
        ssl: true,
        sslValidate: true
      }
    },
    redis: {
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT,
      password: process.env.REDIS_PASSWORD
    },
    jwt: {
      secret: process.env.JWT_SECRET,
      expiresIn: '15m'
    }
  }
};

module.exports = config[process.env.NODE_ENV || 'development'];
```

### **Logging and Monitoring**

```javascript
const winston = require('winston');
const morgan = require('morgan');

// Winston logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'nodejs-app' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
  ],
});

// Add console transport for development
if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.simple()
    )
  }));
}

// Morgan HTTP logging
const morganFormat = process.env.NODE_ENV === 'production' 
  ? 'combined' 
  : 'dev';

app.use(morgan(morganFormat, {
  stream: {
    write: (message) => logger.info(message.trim())
  }
}));

// Custom logging middleware
function requestLogger(req, res, next) {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info('HTTP Request', {
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip
    });
  });
  
  next();
}

app.use(requestLogger);

// Error logging
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});
```

### **Health Checks and Graceful Shutdown**

```javascript
// Health check endpoint
app.get('/health', async (req, res) => {
  const healthCheck = {
    uptime: process.uptime(),
    message: 'OK',
    timestamp: Date.now(),
    checks: {}
  };
  
  try {
    // Check database connection
    await mongoose.connection.db.admin().ping();
    healthCheck.checks.database = 'OK';
  } catch (error) {
    healthCheck.checks.database = 'ERROR';
    healthCheck.message = 'ERROR';
  }
  
  try {
    // Check Redis connection
    await redis.ping();
    healthCheck.checks.redis = 'OK';
  } catch (error) {
    healthCheck.checks.redis = 'ERROR';
    healthCheck.message = 'ERROR';
  }
  
  const status = healthCheck.message === 'OK' ? 200 : 503;
  res.status(status).json(healthCheck);
});

// Graceful shutdown
function gracefulShutdown(signal) {
  logger.info(`Received ${signal}. Starting graceful shutdown...`);
  
  server.close(() => {
    logger.info('HTTP server closed');
    
    // Close database connections
    mongoose.connection.close(false, () => {
      logger.info('MongoDB connection closed');
      
      // Close Redis connection
      redis.quit(() => {
        logger.info('Redis connection closed');
        process.exit(0);
      });
    });
  });
  
  // Force close after 30 seconds
  setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 30000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
```

---

## 🎯 **Interview Questions**

### **1. What is the Node.js event loop and how does it work?**

**Answer:**
The event loop is the core of Node.js's non-blocking I/O operations. It consists of several phases:
- **Timers**: Executes callbacks scheduled by setTimeout() and setInterval()
- **Pending callbacks**: Executes I/O callbacks deferred to the next loop iteration
- **Idle, prepare**: Internal use only
- **Poll**: Retrieves new I/O events and executes I/O related callbacks
- **Check**: setImmediate() callbacks are invoked here
- **Close callbacks**: Some close callbacks (e.g., socket.on('close', ...))

### **2. Explain the difference between process.nextTick() and setImmediate().**

**Answer:**
- **process.nextTick()**: Executes before any other phase of the event loop. It has the highest priority.
- **setImmediate()**: Executes in the Check phase of the event loop, after the Poll phase.

### **3. How do you handle memory leaks in Node.js?**

**Answer:**
- **Monitor memory usage**: Use process.memoryUsage() and tools like clinic.js
- **Avoid global variables**: Keep variables in appropriate scope
- **Clear timers and intervals**: Always clear setTimeout/setInterval
- **Remove event listeners**: Use removeListener() or removeAllListeners()
- **Use weak references**: WeakMap and WeakSet for temporary data
- **Stream properly**: Handle stream end events and close connections

### **4. What are the benefits and drawbacks of using Node.js?**

**Answer:**
**Benefits:**
- Fast execution with V8 engine
- Non-blocking I/O for high concurrency
- Large npm ecosystem
- Full-stack JavaScript development
- Good for real-time applications

**Drawbacks:**
- Single-threaded (CPU-intensive tasks can block)
- Callback hell (though async/await helps)
- Less mature than other server technologies
- Memory usage can be high for large applications

### **5. How do you implement authentication in Node.js?**

**Answer:**
- **JWT tokens**: For stateless authentication
- **Session-based**: Using express-session with Redis
- **OAuth**: For third-party authentication
- **Password hashing**: Using bcrypt
- **Rate limiting**: Prevent brute force attacks
- **HTTPS**: Secure communication
- **Input validation**: Sanitize user inputs

---

**🎉 Node.js expertise is essential for modern backend development!**
