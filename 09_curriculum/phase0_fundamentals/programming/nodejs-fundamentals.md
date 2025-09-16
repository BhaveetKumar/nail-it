# Node.js Fundamentals

## Table of Contents

1. [Overview](#overview)
2. [JavaScript ES6+ Features](#javascript-es6-features)
3. [Node.js Runtime](#nodejs-runtime)
4. [Asynchronous Programming](#asynchronous-programming)
5. [Modules and NPM](#modules-and-npm)
6. [File System Operations](#file-system-operations)
7. [HTTP and Web Servers](#http-and-web-servers)
8. [Event-Driven Programming](#event-driven-programming)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)

## Overview

### Learning Objectives

- Master JavaScript ES6+ features
- Understand Node.js runtime and event loop
- Learn asynchronous programming with Promises and async/await
- Build web servers and APIs
- Apply Node.js best practices

### What is Node.js?

Node.js is a JavaScript runtime built on Chrome's V8 engine that allows you to run JavaScript on the server side. It's designed for building scalable network applications.

## JavaScript ES6+ Features

### 1. Modern JavaScript Syntax

```javascript
// Arrow functions
const add = (a, b) => a + b;
const square = x => x * x;

// Template literals
const name = 'John';
const age = 30;
const message = `Hello, ${name}! You are ${age} years old.`;

// Destructuring
const person = { name: 'Alice', age: 25, city: 'New York' };
const { name, age, city } = person;

const numbers = [1, 2, 3, 4, 5];
const [first, second, ...rest] = numbers;

// Spread operator
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];

const obj1 = { a: 1, b: 2 };
const obj2 = { c: 3, d: 4 };
const merged = { ...obj1, ...obj2 };

// Default parameters
function greet(name = 'World') {
    return `Hello, ${name}!`;
}

// Rest parameters
function sum(...numbers) {
    return numbers.reduce((total, num) => total + num, 0);
}

// Classes
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    greet() {
        return `Hello, I'm ${this.name}`;
    }
    
    static createAdult(name) {
        return new Person(name, 18);
    }
}

// Inheritance
class Student extends Person {
    constructor(name, age, grade) {
        super(name, age);
        this.grade = grade;
    }
    
    study() {
        return `${this.name} is studying`;
    }
}

// Modules
export const PI = 3.14159;
export function circleArea(radius) {
    return PI * radius * radius;
}

export default class Calculator {
    add(a, b) {
        return a + b;
    }
}
```

### 2. Advanced JavaScript Features

```javascript
// Promises
function fetchData(url) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (url) {
                resolve({ data: 'Some data', url });
            } else {
                reject(new Error('Invalid URL'));
            }
        }, 1000);
    });
}

// Async/await
async function processData() {
    try {
        const result = await fetchData('https://api.example.com');
        console.log('Data:', result);
        return result;
    } catch (error) {
        console.error('Error:', error.message);
        throw error;
    }
}

// Generators
function* numberGenerator() {
    let num = 1;
    while (true) {
        yield num++;
    }
}

const gen = numberGenerator();
console.log(gen.next().value); // 1
console.log(gen.next().value); // 2

// Map and Set
const map = new Map();
map.set('key1', 'value1');
map.set('key2', 'value2');
console.log(map.get('key1')); // value1

const set = new Set([1, 2, 3, 3, 4]);
console.log(set.size); // 4

// WeakMap and WeakSet
const weakMap = new WeakMap();
const obj = {};
weakMap.set(obj, 'some value');

// Symbol
const sym1 = Symbol('description');
const sym2 = Symbol('description');
console.log(sym1 === sym2); // false

// Proxy
const target = { name: 'John' };
const proxy = new Proxy(target, {
    get(target, prop) {
        console.log(`Getting ${prop}`);
        return target[prop];
    },
    set(target, prop, value) {
        console.log(`Setting ${prop} to ${value}`);
        target[prop] = value;
        return true;
    }
});
```

## Node.js Runtime

### 1. Event Loop and V8 Engine

```javascript
// Understanding the event loop
console.log('1. Synchronous code');

setTimeout(() => {
    console.log('2. setTimeout callback');
}, 0);

setImmediate(() => {
    console.log('3. setImmediate callback');
});

process.nextTick(() => {
    console.log('4. nextTick callback');
});

console.log('5. More synchronous code');

// Output:
// 1. Synchronous code
// 5. More synchronous code
// 4. nextTick callback
// 2. setTimeout callback
// 3. setImmediate callback

// Process object
console.log('Process ID:', process.pid);
console.log('Node version:', process.version);
console.log('Platform:', process.platform);
console.log('Architecture:', process.arch);

// Environment variables
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('PORT:', process.env.PORT || 3000);

// Command line arguments
console.log('Arguments:', process.argv);

// Memory usage
const memUsage = process.memoryUsage();
console.log('Memory usage:', {
    rss: Math.round(memUsage.rss / 1024 / 1024) + ' MB',
    heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024) + ' MB',
    heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + ' MB',
    external: Math.round(memUsage.external / 1024 / 1024) + ' MB'
});

// Exit handlers
process.on('exit', (code) => {
    console.log(`Process exiting with code: ${code}`);
});

process.on('SIGINT', () => {
    console.log('Received SIGINT. Graceful shutdown...');
    process.exit(0);
});
```

## Asynchronous Programming

### 1. Callbacks, Promises, and Async/Await

```javascript
const fs = require('fs').promises;
const path = require('path');

// Callback pattern (older style)
function readFileCallback(filename, callback) {
    fs.readFile(filename, 'utf8', (err, data) => {
        if (err) {
            callback(err, null);
        } else {
            callback(null, data);
        }
    });
}

// Promise pattern
function readFilePromise(filename) {
    return fs.readFile(filename, 'utf8')
        .then(data => {
            console.log('File read successfully');
            return data;
        })
        .catch(err => {
            console.error('Error reading file:', err);
            throw err;
        });
}

// Async/await pattern
async function readFileAsync(filename) {
    try {
        const data = await fs.readFile(filename, 'utf8');
        console.log('File read successfully');
        return data;
    } catch (err) {
        console.error('Error reading file:', err);
        throw err;
    }
}

// Promise.all for parallel execution
async function readMultipleFiles(filenames) {
    try {
        const promises = filenames.map(filename => fs.readFile(filename, 'utf8'));
        const results = await Promise.all(promises);
        return results;
    } catch (err) {
        console.error('Error reading files:', err);
        throw err;
    }
}

// Promise.allSettled for handling partial failures
async function readFilesWithPartialFailure(filenames) {
    const promises = filenames.map(filename => 
        fs.readFile(filename, 'utf8')
            .then(data => ({ status: 'fulfilled', value: data, filename }))
            .catch(err => ({ status: 'rejected', reason: err, filename }))
    );
    
    const results = await Promise.allSettled(promises);
    return results;
}

// Race condition handling
async function fetchWithTimeout(url, timeout = 5000) {
    const fetchPromise = fetch(url);
    const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), timeout)
    );
    
    return Promise.race([fetchPromise, timeoutPromise]);
}
```

### 2. Advanced Async Patterns

```javascript
// Async generator
async function* asyncGenerator() {
    for (let i = 0; i < 5; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        yield i;
    }
}

async function consumeAsyncGenerator() {
    for await (const value of asyncGenerator()) {
        console.log('Generated value:', value);
    }
}

// Semaphore for limiting concurrent operations
class Semaphore {
    constructor(permits) {
        this.permits = permits;
        this.waiting = [];
    }
    
    async acquire() {
        if (this.permits > 0) {
            this.permits--;
            return;
        }
        
        return new Promise(resolve => {
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

// Throttling function calls
function throttle(func, delay) {
    let timeoutId;
    let lastExecTime = 0;
    
    return function (...args) {
        const currentTime = Date.now();
        
        if (currentTime - lastExecTime > delay) {
            func.apply(this, args);
            lastExecTime = currentTime;
        } else {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func.apply(this, args);
                lastExecTime = Date.now();
            }, delay - (currentTime - lastExecTime));
        }
    };
}

// Debouncing function calls
function debounce(func, delay) {
    let timeoutId;
    
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}
```

## Modules and NPM

### 1. CommonJS and ES Modules

```javascript
// math.js (CommonJS)
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

module.exports = {
    add,
    subtract
};

// math.mjs (ES Modules)
export function add(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}

export default {
    add,
    subtract
};

// index.js (using CommonJS)
const math = require('./math');
const { add, subtract } = require('./math');

console.log(add(5, 3)); // 8
console.log(subtract(10, 4)); // 6

// index.mjs (using ES Modules)
import math, { add, subtract } from './math.mjs';

console.log(add(5, 3)); // 8
console.log(subtract(10, 4)); // 6

// package.json
{
    "name": "my-node-app",
    "version": "1.0.0",
    "type": "module",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "dev": "nodemon index.js",
        "test": "jest"
    },
    "dependencies": {
        "express": "^4.18.2",
        "lodash": "^4.17.21"
    },
    "devDependencies": {
        "nodemon": "^3.0.1",
        "jest": "^29.6.2"
    }
}
```

### 2. NPM and Package Management

```javascript
// Installing packages
// npm install express
// npm install --save-dev nodemon
// npm install -g typescript

// Using installed packages
const express = require('express');
const _ = require('lodash');

const app = express();

// Custom module with dependencies
// utils/logger.js
const winston = require('winston');

const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'error.log', level: 'error' }),
        new winston.transports.File({ filename: 'combined.log' }),
        new winston.transports.Console()
    ]
});

module.exports = logger;

// Using the logger
const logger = require('./utils/logger');

logger.info('Application started');
logger.error('Something went wrong');
```

## File System Operations

### 1. File and Directory Operations

```javascript
const fs = require('fs').promises;
const path = require('path');

// Reading files
async function readFileExample() {
    try {
        const data = await fs.readFile('example.txt', 'utf8');
        console.log('File content:', data);
    } catch (err) {
        console.error('Error reading file:', err);
    }
}

// Writing files
async function writeFileExample() {
    try {
        const content = 'Hello, Node.js!';
        await fs.writeFile('output.txt', content, 'utf8');
        console.log('File written successfully');
    } catch (err) {
        console.error('Error writing file:', err);
    }
}

// Directory operations
async function directoryExample() {
    try {
        // Create directory
        await fs.mkdir('new-directory', { recursive: true });
        
        // Read directory
        const files = await fs.readdir('.');
        console.log('Files in current directory:', files);
        
        // Get file stats
        const stats = await fs.stat('package.json');
        console.log('File stats:', {
            isFile: stats.isFile(),
            isDirectory: stats.isDirectory(),
            size: stats.size,
            modified: stats.mtime
        });
        
        // Watch file changes
        const watcher = fs.watch('.', (eventType, filename) => {
            console.log(`File ${filename} ${eventType}`);
        });
        
        // Stop watching after 10 seconds
        setTimeout(() => {
            watcher.close();
        }, 10000);
        
    } catch (err) {
        console.error('Error with directory operations:', err);
    }
}

// Stream operations
const { createReadStream, createWriteStream } = require('fs');

function copyFileStream(source, destination) {
    const readStream = createReadStream(source);
    const writeStream = createWriteStream(destination);
    
    readStream.pipe(writeStream);
    
    readStream.on('error', (err) => {
        console.error('Read error:', err);
    });
    
    writeStream.on('error', (err) => {
        console.error('Write error:', err);
    });
    
    writeStream.on('finish', () => {
        console.log('File copied successfully');
    });
}
```

## HTTP and Web Servers

### 1. Built-in HTTP Module

```javascript
const http = require('http');
const url = require('url');

// Basic HTTP server
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const path = parsedUrl.pathname;
    const method = req.method;
    
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    // Route handling
    if (method === 'GET' && path === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<h1>Hello, Node.js!</h1>');
    } else if (method === 'GET' && path === '/api/users') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify([
            { id: 1, name: 'John Doe' },
            { id: 2, name: 'Jane Smith' }
        ]));
    } else if (method === 'POST' && path === '/api/users') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        
        req.on('end', () => {
            try {
                const user = JSON.parse(body);
                res.writeHead(201, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ message: 'User created', user }));
            } catch (err) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON' }));
            }
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
    }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

// HTTP client
function makeHttpRequest() {
    const options = {
        hostname: 'jsonplaceholder.typicode.com',
        port: 80,
        path: '/posts/1',
        method: 'GET'
    };
    
    const req = http.request(options, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
            data += chunk;
        });
        
        res.on('end', () => {
            console.log('Response:', JSON.parse(data));
        });
    });
    
    req.on('error', (err) => {
        console.error('Request error:', err);
    });
    
    req.end();
}
```

### 2. Express.js Framework

```javascript
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'Hello, Express!' });
});

app.get('/api/users', (req, res) => {
    const users = [
        { id: 1, name: 'John Doe', email: 'john@example.com' },
        { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
    ];
    res.json(users);
});

app.post('/api/users', (req, res) => {
    const { name, email } = req.body;
    
    if (!name || !email) {
        return res.status(400).json({ error: 'Name and email are required' });
    }
    
    const user = { id: Date.now(), name, email };
    res.status(201).json(user);
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Express server running on port ${PORT}`);
});
```

## Event-Driven Programming

### 1. EventEmitter

```javascript
const EventEmitter = require('events');

class MyEmitter extends EventEmitter {}

const myEmitter = new MyEmitter();

// Event listeners
myEmitter.on('event', (data) => {
    console.log('Event received:', data);
});

myEmitter.once('once-event', (data) => {
    console.log('This will only fire once:', data);
});

// Emit events
myEmitter.emit('event', { message: 'Hello World' });
myEmitter.emit('once-event', { message: 'First time' });
myEmitter.emit('once-event', { message: 'Second time' }); // Won't fire

// Custom event emitter
class Logger extends EventEmitter {
    log(message) {
        this.emit('log', { message, timestamp: new Date() });
    }
    
    error(message) {
        this.emit('error', { message, timestamp: new Date() });
    }
}

const logger = new Logger();

logger.on('log', (data) => {
    console.log(`[LOG] ${data.timestamp}: ${data.message}`);
});

logger.on('error', (data) => {
    console.error(`[ERROR] ${data.timestamp}: ${data.message}`);
});

logger.log('Application started');
logger.error('Something went wrong');
```

## Error Handling

### 1. Error Handling Patterns

```javascript
// Custom error class
class AppError extends Error {
    constructor(message, statusCode) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = true;
        
        Error.captureStackTrace(this, this.constructor);
    }
}

// Error handling middleware
function errorHandler(err, req, res, next) {
    let error = { ...err };
    error.message = err.message;
    
    // Log error
    console.error(err);
    
    // Mongoose bad ObjectId
    if (err.name === 'CastError') {
        const message = 'Resource not found';
        error = new AppError(message, 404);
    }
    
    // Mongoose duplicate key
    if (err.code === 11000) {
        const message = 'Duplicate field value entered';
        error = new AppError(message, 400);
    }
    
    // Mongoose validation error
    if (err.name === 'ValidationError') {
        const message = Object.values(err.errors).map(val => val.message);
        error = new AppError(message, 400);
    }
    
    res.status(error.statusCode || 500).json({
        success: false,
        error: error.message || 'Server Error'
    });
}

// Async error wrapper
const asyncHandler = (fn) => (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
};

// Usage example
app.get('/api/users/:id', asyncHandler(async (req, res, next) => {
    const user = await User.findById(req.params.id);
    
    if (!user) {
        return next(new AppError('User not found', 404));
    }
    
    res.json({
        success: true,
        data: user
    });
}));

// Global error handler
process.on('unhandledRejection', (err, promise) => {
    console.log('Unhandled Rejection at:', promise, 'reason:', err);
    // Close server & exit process
    server.close(() => {
        process.exit(1);
    });
});

process.on('uncaughtException', (err) => {
    console.log('Uncaught Exception:', err);
    process.exit(1);
});
```

## Best Practices

### 1. Code Organization

```javascript
// Project structure
// src/
//   controllers/
//     userController.js
//   models/
//     User.js
//   routes/
//     userRoutes.js
//   middleware/
//     auth.js
//   utils/
//     logger.js
//   app.js
//   server.js

// userController.js
const User = require('../models/User');
const AppError = require('../utils/AppError');

exports.getAllUsers = async (req, res, next) => {
    try {
        const users = await User.find();
        res.json({
            success: true,
            count: users.length,
            data: users
        });
    } catch (err) {
        next(err);
    }
};

exports.getUser = async (req, res, next) => {
    try {
        const user = await User.findById(req.params.id);
        
        if (!user) {
            return next(new AppError('User not found', 404));
        }
        
        res.json({
            success: true,
            data: user
        });
    } catch (err) {
        next(err);
    }
};

// Environment configuration
const config = {
    development: {
        port: 3000,
        db: 'mongodb://localhost:27017/myapp_dev'
    },
    production: {
        port: process.env.PORT || 3000,
        db: process.env.MONGODB_URI
    }
};

const env = process.env.NODE_ENV || 'development';
module.exports = config[env];
```

## Follow-up Questions

### 1. JavaScript Features
**Q: What's the difference between `let`, `const`, and `var`?**
A: `let` and `const` are block-scoped, while `var` is function-scoped. `const` cannot be reassigned, `let` can be reassigned, and `var` can be reassigned and is hoisted.

### 2. Asynchronous Programming
**Q: When should you use Promises vs async/await?**
A: Use async/await for cleaner, more readable code when dealing with sequential async operations. Use Promises for parallel operations or when you need more control over the execution flow.

### 3. Node.js Runtime
**Q: What's the difference between `setImmediate` and `setTimeout`?**
A: `setImmediate` executes in the check phase of the event loop, while `setTimeout` executes in the timer phase. `setImmediate` is generally faster for I/O operations.

## Sources

### Books
- **Node.js in Action** by Mike Cantelon, Marc Harter, T.J. Holowaychuk, Nathan Rajlich
- **You Don't Know JS** by Kyle Simpson
- **JavaScript: The Good Parts** by Douglas Crockford

### Online Resources
- **Node.js Documentation** - https://nodejs.org/docs/
- **MDN Web Docs** - https://developer.mozilla.org/
- **Node.js Best Practices** - https://github.com/goldbergyoni/nodebestpractices

## Projects

### 1. REST API
**Objective**: Build a RESTful API
**Requirements**: Express.js, MongoDB, authentication
**Deliverables**: Complete API with CRUD operations

### 2. Real-time Chat
**Objective**: Create a real-time chat application
**Requirements**: Socket.io, WebSockets, rooms
**Deliverables**: Multi-room chat application

### 3. File Upload Service
**Objective**: Build a file upload and management service
**Requirements**: Multer, file validation, cloud storage
**Deliverables**: File upload API with storage

---

**Next**: [Data Structures & Algorithms](./dsa-questions-golang-nodejs.md) | **Previous**: [Go Fundamentals](./go-fundamentals.md) | **Up**: [Phase 0](../README.md)


continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue

continue