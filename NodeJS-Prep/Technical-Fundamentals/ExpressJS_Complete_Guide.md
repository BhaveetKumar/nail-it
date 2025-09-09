# 🚀 Express.js Complete Guide: From Basics to Production

> **Master Express.js for building scalable web applications and APIs**

## 🎯 **Learning Objectives**

- Master Express.js framework fundamentals
- Build RESTful APIs and web applications
- Implement authentication and authorization
- Learn middleware patterns and best practices
- Deploy production-ready Express applications

## 📚 **Table of Contents**

1. [Express.js Fundamentals](#expressjs-fundamentals)
2. [Routing and Middleware](#routing-and-middleware)
3. [Request and Response Handling](#request-and-response-handling)
4. [Authentication & Security](#authentication--security)
5. [Database Integration](#database-integration)
6. [Error Handling](#error-handling)
7. [Testing](#testing)
8. [Performance Optimization](#performance-optimization)
9. [Production Deployment](#production-deployment)
10. [Interview Questions](#interview-questions)

---

## 🚀 **Express.js Fundamentals**

### **What is Express.js?**

Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for web and mobile applications. It's built on top of Node.js's HTTP module and provides a thin layer of fundamental web application features.

### **Key Features**

- **Minimal and Flexible**: Lightweight framework with minimal overhead
- **Middleware Support**: Extensive middleware ecosystem
- **Routing**: Powerful routing system
- **Template Engines**: Support for various template engines
- **Static Files**: Built-in static file serving
- **HTTP Helpers**: Redirection, caching, and more

### **Basic Express Application**

```javascript
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

// Basic route
app.get('/', (req, res) => {
    res.send('Hello World!');
});

// JSON response
app.get('/api/users', (req, res) => {
    res.json([
        { id: 1, name: 'John Doe', email: 'john@example.com' },
        { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
    ]);
});

// Route parameters
app.get('/api/users/:id', (req, res) => {
    const userId = req.params.id;
    res.json({ id: userId, name: 'User', email: 'user@example.com' });
});

// Query parameters
app.get('/api/search', (req, res) => {
    const { q, page = 1, limit = 10 } = req.query;
    res.json({
        query: q,
        page: parseInt(page),
        limit: parseInt(limit),
        results: []
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
```

---

## 🛣️ **Routing and Middleware**

### **Basic Routing**

```javascript
const express = require('express');
const app = express();

// HTTP Methods
app.get('/users', (req, res) => {
    res.send('GET /users');
});

app.post('/users', (req, res) => {
    res.send('POST /users');
});

app.put('/users/:id', (req, res) => {
    res.send(`PUT /users/${req.params.id}`);
});

app.delete('/users/:id', (req, res) => {
    res.send(`DELETE /users/${req.params.id}`);
});

// Route with multiple handlers
app.get('/users/:id', 
    (req, res, next) => {
        console.log('First handler');
        next();
    },
    (req, res) => {
        console.log('Second handler');
        res.send(`User ID: ${req.params.id}`);
    }
);

// Route groups
app.route('/users/:id')
    .get((req, res) => {
        res.send(`Get user ${req.params.id}`);
    })
    .put((req, res) => {
        res.send(`Update user ${req.params.id}`);
    })
    .delete((req, res) => {
        res.send(`Delete user ${req.params.id}`);
    });
```

### **Router Module**

```javascript
// routes/users.js
const express = require('express');
const router = express.Router();

// Middleware specific to this router
router.use((req, res, next) => {
    console.log('Users router middleware');
    next();
});

// GET /users
router.get('/', (req, res) => {
    res.json({ message: 'Get all users' });
});

// GET /users/:id
router.get('/:id', (req, res) => {
    res.json({ message: `Get user ${req.params.id}` });
});

// POST /users
router.post('/', (req, res) => {
    res.json({ message: 'Create user' });
});

module.exports = router;

// app.js
const express = require('express');
const userRoutes = require('./routes/users');

const app = express();

app.use('/api/users', userRoutes);

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

### **Middleware Patterns**

```javascript
// Application-level middleware
app.use((req, res, next) => {
    console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
    next();
});

// Built-in middleware
app.use(express.json()); // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies
app.use(express.static('public')); // Serve static files

// Custom middleware
const logger = (req, res, next) => {
    console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
    next();
};

const authenticate = (req, res, next) => {
    const token = req.headers.authorization;
    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }
    // Verify token logic here
    next();
};

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});
```

---

## 📨 **Request and Response Handling**

### **Request Object**

```javascript
app.get('/api/users/:id', (req, res) => {
    // Request properties
    console.log('Method:', req.method);
    console.log('URL:', req.url);
    console.log('Path:', req.path);
    console.log('Query:', req.query);
    console.log('Params:', req.params);
    console.log('Headers:', req.headers);
    console.log('Body:', req.body);
    console.log('IP:', req.ip);
    console.log('Protocol:', req.protocol);
    console.log('Secure:', req.secure);
    
    res.json({ message: 'Request received' });
});

// POST request with body
app.post('/api/users', (req, res) => {
    const { name, email, age } = req.body;
    
    // Validate required fields
    if (!name || !email) {
        return res.status(400).json({ error: 'Name and email are required' });
    }
    
    // Create user logic here
    const user = { id: Date.now(), name, email, age };
    
    res.status(201).json(user);
});
```

### **Response Object**

```javascript
app.get('/api/response-examples', (req, res) => {
    // Send JSON response
    res.json({ message: 'JSON response' });
    
    // Send HTML response
    res.send('<h1>HTML response</h1>');
    
    // Send file
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
    
    // Redirect
    res.redirect('/api/users');
    
    // Set status code
    res.status(201).json({ message: 'Created' });
    
    // Set headers
    res.set('Content-Type', 'application/json');
    res.set('X-Custom-Header', 'Custom Value');
    
    // Set multiple headers
    res.set({
        'Content-Type': 'application/json',
        'X-Custom-Header': 'Custom Value'
    });
    
    // Send response with cookies
    res.cookie('name', 'value', { maxAge: 900000, httpOnly: true });
    res.json({ message: 'Response with cookie' });
});

// Download file
app.get('/api/download', (req, res) => {
    res.download('path/to/file.pdf', 'custom-filename.pdf');
});

// Stream response
app.get('/api/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');
    
    const interval = setInterval(() => {
        res.write('chunk\n');
    }, 1000);
    
    setTimeout(() => {
        clearInterval(interval);
        res.end();
    }, 10000);
});
```

---

## 🔐 **Authentication & Security**

### **JWT Authentication**

```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

// JWT Secret (should be in environment variables)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// User model (simplified)
const users = [
    { id: 1, username: 'admin', password: '$2b$10$...' }, // hashed password
    { id: 2, username: 'user', password: '$2b$10$...' }
];

// Login endpoint
app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        
        // Find user
        const user = users.find(u => u.username === username);
        if (!user) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        
        // Verify password
        const isValidPassword = await bcrypt.compare(password, user.password);
        if (!isValidPassword) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        
        // Generate JWT token
        const token = jwt.sign(
            { userId: user.id, username: user.username },
            JWT_SECRET,
            { expiresIn: '1h' }
        );
        
        res.json({ token, user: { id: user.id, username: user.username } });
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Authentication middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }
    
    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            return res.status(403).json({ error: 'Invalid token' });
        }
        req.user = user;
        next();
    });
};

// Protected route
app.get('/api/profile', authenticateToken, (req, res) => {
    res.json({ user: req.user });
});
```

### **Password Hashing**

```javascript
const bcrypt = require('bcrypt');

// Hash password
async function hashPassword(password) {
    const saltRounds = 10;
    return await bcrypt.hash(password, saltRounds);
}

// Compare password
async function comparePassword(password, hash) {
    return await bcrypt.compare(password, hash);
}

// Register endpoint
app.post('/api/register', async (req, res) => {
    try {
        const { username, password, email } = req.body;
        
        // Check if user already exists
        const existingUser = users.find(u => u.username === username);
        if (existingUser) {
            return res.status(400).json({ error: 'Username already exists' });
        }
        
        // Hash password
        const hashedPassword = await hashPassword(password);
        
        // Create user
        const user = {
            id: users.length + 1,
            username,
            email,
            password: hashedPassword
        };
        
        users.push(user);
        
        res.status(201).json({ message: 'User created successfully' });
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});
```

### **Security Middleware**

```javascript
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

// Security headers
app.use(helmet());

// CORS configuration
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.'
});

app.use('/api/', limiter);

// Input validation
const { body, validationResult } = require('express-validator');

app.post('/api/users',
    [
        body('name').trim().isLength({ min: 2 }).withMessage('Name must be at least 2 characters'),
        body('email').isEmail().withMessage('Must be a valid email'),
        body('age').isInt({ min: 0, max: 120 }).withMessage('Age must be between 0 and 120')
    ],
    (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }
        
        // Process valid data
        res.json({ message: 'User created successfully' });
    }
);
```

---

## 🗄️ **Database Integration**

### **MongoDB with Mongoose**

```javascript
const mongoose = require('mongoose');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/myapp', {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

// User Schema
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

const User = mongoose.model('User', userSchema);

// User routes
app.get('/api/users', async (req, res) => {
    try {
        const users = await User.find();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/api/users', async (req, res) => {
    try {
        const user = new User(req.body);
        await user.save();
        res.status(201).json(user);
    } catch (error) {
        if (error.code === 11000) {
            res.status(400).json({ error: 'Email already exists' });
        } else {
            res.status(400).json({ error: error.message });
        }
    }
});
```

### **PostgreSQL with pg**

```javascript
const { Pool } = require('pg');

// Database connection
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

// User operations
class UserService {
    async createUser(userData) {
        const query = `
            INSERT INTO users (name, email, age, created_at)
            VALUES ($1, $2, $3, $4)
            RETURNING *
        `;
        const values = [userData.name, userData.email, userData.age, new Date()];
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    
    async getUserById(id) {
        const query = 'SELECT * FROM users WHERE id = $1';
        const result = await pool.query(query, [id]);
        return result.rows[0];
    }
    
    async updateUser(id, updateData) {
        const fields = Object.keys(updateData);
        const values = Object.values(updateData);
        const setClause = fields.map((field, index) => `${field} = $${index + 2}`).join(', ');
        
        const query = `
            UPDATE users 
            SET ${setClause}, updated_at = $1
            WHERE id = $${fields.length + 2}
            RETURNING *
        `;
        
        const result = await pool.query(query, [new Date(), ...values, id]);
        return result.rows[0];
    }
    
    async deleteUser(id) {
        const query = 'DELETE FROM users WHERE id = $1 RETURNING *';
        const result = await pool.query(query, [id]);
        return result.rows[0];
    }
}

const userService = new UserService();

// User routes
app.get('/api/users/:id', async (req, res) => {
    try {
        const user = await userService.getUserById(req.params.id);
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json(user);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});
```

---

## ❌ **Error Handling**

### **Custom Error Classes**

```javascript
// Custom error classes
class AppError extends Error {
    constructor(message, statusCode) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = true;
        
        Error.captureStackTrace(this, this.constructor);
    }
}

class ValidationError extends AppError {
    constructor(message) {
        super(message, 400);
    }
}

class NotFoundError extends AppError {
    constructor(message) {
        super(message, 404);
    }
}

class UnauthorizedError extends AppError {
    constructor(message) {
        super(message, 401);
    }
}

// Error handling middleware
const errorHandler = (err, req, res, next) => {
    let error = { ...err };
    error.message = err.message;
    
    // Log error
    console.error(err);
    
    // Mongoose bad ObjectId
    if (err.name === 'CastError') {
        const message = 'Resource not found';
        error = new NotFoundError(message);
    }
    
    // Mongoose duplicate key
    if (err.code === 11000) {
        const message = 'Duplicate field value entered';
        error = new ValidationError(message);
    }
    
    // Mongoose validation error
    if (err.name === 'ValidationError') {
        const message = Object.values(err.errors).map(val => val.message).join(', ');
        error = new ValidationError(message);
    }
    
    res.status(error.statusCode || 500).json({
        success: false,
        error: error.message || 'Server Error'
    });
};

// Async error handler wrapper
const asyncHandler = (fn) => (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
};

// Usage with async handler
app.get('/api/users/:id', asyncHandler(async (req, res) => {
    const user = await userService.getUserById(req.params.id);
    if (!user) {
        throw new NotFoundError('User not found');
    }
    res.json(user);
}));

// Global error handler
app.use(errorHandler);
```

---

## 🧪 **Testing**

### **Unit Testing with Jest**

```javascript
// userService.test.js
const request = require('supertest');
const app = require('../app');
const User = require('../models/User');

describe('User API', () => {
    beforeEach(async () => {
        await User.deleteMany({});
    });
    
    describe('GET /api/users', () => {
        it('should get all users', async () => {
            const users = [
                { name: 'John Doe', email: 'john@example.com' },
                { name: 'Jane Smith', email: 'jane@example.com' }
            ];
            
            await User.insertMany(users);
            
            const res = await request(app)
                .get('/api/users')
                .expect(200);
            
            expect(res.body).toHaveLength(2);
            expect(res.body[0]).toHaveProperty('name', 'John Doe');
        });
    });
    
    describe('POST /api/users', () => {
        it('should create a new user', async () => {
            const userData = {
                name: 'John Doe',
                email: 'john@example.com',
                age: 30
            };
            
            const res = await request(app)
                .post('/api/users')
                .send(userData)
                .expect(201);
            
            expect(res.body).toHaveProperty('name', userData.name);
            expect(res.body).toHaveProperty('email', userData.email);
        });
        
        it('should return 400 for invalid data', async () => {
            const invalidData = {
                name: '',
                email: 'invalid-email'
            };
            
            const res = await request(app)
                .post('/api/users')
                .send(invalidData)
                .expect(400);
            
            expect(res.body).toHaveProperty('error');
        });
    });
});
```

### **Integration Testing**

```javascript
// integration.test.js
const request = require('supertest');
const app = require('../app');
const { connectDB, disconnectDB } = require('../config/database');

describe('Integration Tests', () => {
    beforeAll(async () => {
        await connectDB();
    });
    
    afterAll(async () => {
        await disconnectDB();
    });
    
    describe('User Flow', () => {
        it('should complete user registration and login flow', async () => {
            // Register user
            const userData = {
                name: 'John Doe',
                email: 'john@example.com',
                password: 'password123'
            };
            
            const registerRes = await request(app)
                .post('/api/register')
                .send(userData)
                .expect(201);
            
            expect(registerRes.body).toHaveProperty('message', 'User created successfully');
            
            // Login user
            const loginRes = await request(app)
                .post('/api/login')
                .send({
                    email: userData.email,
                    password: userData.password
                })
                .expect(200);
            
            expect(loginRes.body).toHaveProperty('token');
            expect(loginRes.body).toHaveProperty('user');
        });
    });
});
```

---

## ⚡ **Performance Optimization**

### **Caching with Redis**

```javascript
const redis = require('redis');
const client = redis.createClient(process.env.REDIS_URL);

// Cache middleware
const cache = (duration) => {
    return async (req, res, next) => {
        const key = `cache:${req.originalUrl}`;
        
        try {
            const cached = await client.get(key);
            if (cached) {
                return res.json(JSON.parse(cached));
            }
            
            // Store original res.json
            const originalJson = res.json;
            res.json = function(data) {
                // Cache the response
                client.setex(key, duration, JSON.stringify(data));
                originalJson.call(this, data);
            };
            
            next();
        } catch (error) {
            next();
        }
    };
};

// Usage
app.get('/api/users', cache(300), async (req, res) => {
    const users = await User.find();
    res.json(users);
});
```

### **Compression and Optimization**

```javascript
const compression = require('compression');
const helmet = require('helmet');

// Compression middleware
app.use(compression());

// Security headers
app.use(helmet());

// Request size limit
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Response time middleware
app.use((req, res, next) => {
    const start = Date.now();
    
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`${req.method} ${req.path} - ${duration}ms`);
    });
    
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
            url: process.env.DATABASE_URL || 'mongodb://localhost:27017/dev_db'
        },
        jwt: {
            secret: process.env.JWT_SECRET || 'dev-secret',
            expiresIn: '24h'
        }
    },
    
    production: {
        port: process.env.PORT || 3000,
        database: {
            url: process.env.DATABASE_URL
        },
        jwt: {
            secret: process.env.JWT_SECRET,
            expiresIn: '1h'
        }
    }
};

module.exports = config[process.env.NODE_ENV || 'development'];
```

### **Docker Configuration**

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
```

### **Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=mongodb://mongo:27017/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - mongo
      - redis
    restart: unless-stopped

  mongo:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  mongo_data:
```

---

## 🎯 **Interview Questions**

### **1. What is Express.js and how does it work?**

**Answer:**
Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for web and mobile applications. It's built on top of Node.js's HTTP module and provides:
- Middleware support for request processing
- Routing system for handling different endpoints
- Template engine support for rendering views
- Static file serving capabilities
- HTTP utility methods and middleware

### **2. What is middleware in Express.js?**

**Answer:**
Middleware functions are functions that have access to the request object (req), response object (res), and the next middleware function in the application's request-response cycle. They can:
- Execute code
- Make changes to request and response objects
- End the request-response cycle
- Call the next middleware function

### **3. How do you handle errors in Express.js?**

**Answer:**
- **Try-catch blocks**: For async operations
- **Error handling middleware**: Global error handler
- **Custom error classes**: For different error types
- **Async error wrapper**: For cleaner async error handling
- **Validation errors**: Using express-validator

### **4. What are the different types of middleware in Express?**

**Answer:**
- **Application-level middleware**: `app.use()`
- **Router-level middleware**: `router.use()`
- **Error-handling middleware**: `app.use((err, req, res, next) => {})`
- **Built-in middleware**: `express.json()`, `express.static()`
- **Third-party middleware**: `helmet`, `cors`, `morgan`

### **5. How do you implement authentication in Express.js?**

**Answer:**
- **JWT tokens**: For stateless authentication
- **Session-based**: Using express-session
- **OAuth**: For third-party authentication
- **Password hashing**: Using bcrypt
- **Middleware**: For protecting routes
- **Rate limiting**: For preventing brute force attacks

---

**🎉 Express.js is a powerful framework for building web applications and APIs!**
