# 🗄️ Node.js Database Design Complete Guide

> **Master database design, optimization, and integration with Node.js applications**

## 📚 Overview

This comprehensive guide covers database design principles, SQL and NoSQL databases, optimization techniques, and best practices for Node.js applications.

## 🎯 Table of Contents

1. [Database Design Principles](#database-design-principles)
2. [SQL Databases](#sql-databases)
3. [NoSQL Databases](#nosql-databases)
4. [Database Optimization](#database-optimization)
5. [Connection Management](#connection-management)
6. [Transactions and ACID](#transactions-and-acid)
7. [Database Security](#database-security)
8. [Performance Monitoring](#performance-monitoring)

## 🏗️ Database Design Principles

### **Normalization**

```javascript
// Example: User and Post entities with proper normalization

// Users Table (1NF, 2NF, 3NF)
const usersTable = {
    id: 'PRIMARY KEY',
    email: 'UNIQUE NOT NULL',
    username: 'UNIQUE NOT NULL',
    first_name: 'VARCHAR(50)',
    last_name: 'VARCHAR(50)',
    created_at: 'TIMESTAMP',
    updated_at: 'TIMESTAMP'
};

// Posts Table
const postsTable = {
    id: 'PRIMARY KEY',
    user_id: 'FOREIGN KEY REFERENCES users(id)',
    title: 'VARCHAR(200) NOT NULL',
    content: 'TEXT',
    status: 'ENUM("draft", "published", "archived")',
    created_at: 'TIMESTAMP',
    updated_at: 'TIMESTAMP'
};

// Categories Table (Many-to-Many with Posts)
const categoriesTable = {
    id: 'PRIMARY KEY',
    name: 'VARCHAR(100) UNIQUE',
    description: 'TEXT',
    created_at: 'TIMESTAMP'
};

// Post_Categories Junction Table
const postCategoriesTable = {
    post_id: 'FOREIGN KEY REFERENCES posts(id)',
    category_id: 'FOREIGN KEY REFERENCES categories(id)',
    PRIMARY_KEY: '(post_id, category_id)'
};
```

### **Entity Relationship Design**

```javascript
// Database schema design with relationships
class DatabaseSchema {
    static createTables() {
        return {
            // User Management
            users: {
                columns: {
                    id: 'SERIAL PRIMARY KEY',
                    email: 'VARCHAR(255) UNIQUE NOT NULL',
                    username: 'VARCHAR(50) UNIQUE NOT NULL',
                    password_hash: 'VARCHAR(255) NOT NULL',
                    first_name: 'VARCHAR(50)',
                    last_name: 'VARCHAR(50)',
                    avatar_url: 'VARCHAR(500)',
                    is_active: 'BOOLEAN DEFAULT true',
                    email_verified: 'BOOLEAN DEFAULT false',
                    created_at: 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    updated_at: 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                indexes: [
                    'CREATE INDEX idx_users_email ON users(email)',
                    'CREATE INDEX idx_users_username ON users(username)',
                    'CREATE INDEX idx_users_created_at ON users(created_at)'
                ]
            },
            
            // Content Management
            posts: {
                columns: {
                    id: 'SERIAL PRIMARY KEY',
                    user_id: 'INTEGER REFERENCES users(id) ON DELETE CASCADE',
                    title: 'VARCHAR(200) NOT NULL',
                    slug: 'VARCHAR(200) UNIQUE NOT NULL',
                    content: 'TEXT',
                    excerpt: 'VARCHAR(500)',
                    featured_image: 'VARCHAR(500)',
                    status: 'ENUM("draft", "published", "archived") DEFAULT "draft"',
                    published_at: 'TIMESTAMP',
                    view_count: 'INTEGER DEFAULT 0',
                    like_count: 'INTEGER DEFAULT 0',
                    created_at: 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    updated_at: 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                indexes: [
                    'CREATE INDEX idx_posts_user_id ON posts(user_id)',
                    'CREATE INDEX idx_posts_status ON posts(status)',
                    'CREATE INDEX idx_posts_published_at ON posts(published_at)',
                    'CREATE INDEX idx_posts_slug ON posts(slug)'
                ]
            },
            
            // Categories and Tags
            categories: {
                columns: {
                    id: 'SERIAL PRIMARY KEY',
                    name: 'VARCHAR(100) UNIQUE NOT NULL',
                    slug: 'VARCHAR(100) UNIQUE NOT NULL',
                    description: 'TEXT',
                    parent_id: 'INTEGER REFERENCES categories(id)',
                    created_at: 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                indexes: [
                    'CREATE INDEX idx_categories_slug ON categories(slug)',
                    'CREATE INDEX idx_categories_parent_id ON categories(parent_id)'
                ]
            },
            
            // Many-to-Many relationships
            post_categories: {
                columns: {
                    post_id: 'INTEGER REFERENCES posts(id) ON DELETE CASCADE',
                    category_id: 'INTEGER REFERENCES categories(id) ON DELETE CASCADE',
                    PRIMARY_KEY: '(post_id, category_id)'
                }
            }
        };
    }
}
```

## 🗃️ SQL Databases

### **PostgreSQL with Node.js**

```javascript
const { Pool } = require('pg');
const { promisify } = require('util');

class PostgreSQLDatabase {
    constructor(config) {
        this.pool = new Pool({
            host: config.host || 'localhost',
            port: config.port || 5432,
            database: config.database,
            user: config.user,
            password: config.password,
            max: config.maxConnections || 20,
            idleTimeoutMillis: config.idleTimeout || 30000,
            connectionTimeoutMillis: config.connectionTimeout || 2000,
            ssl: config.ssl || false
        });
        
        this.pool.on('error', (err) => {
            console.error('Unexpected error on idle client', err);
        });
    }
    
    async query(text, params = []) {
        const start = Date.now();
        try {
            const res = await this.pool.query(text, params);
            const duration = Date.now() - start;
            console.log('Executed query', { text, duration, rows: res.rowCount });
            return res;
        } catch (error) {
            console.error('Database query error:', error);
            throw error;
        }
    }
    
    async getClient() {
        return await this.pool.connect();
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
    
    async close() {
        await this.pool.end();
    }
}

// Usage example
const db = new PostgreSQLDatabase({
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD
});

// User service with database operations
class UserService {
    constructor(database) {
        this.db = database;
    }
    
    async createUser(userData) {
        const { email, username, passwordHash, firstName, lastName } = userData;
        
        const query = `
            INSERT INTO users (email, username, password_hash, first_name, last_name)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, email, username, first_name, last_name, created_at
        `;
        
        const result = await this.db.query(query, [
            email, username, passwordHash, firstName, lastName
        ]);
        
        return result.rows[0];
    }
    
    async getUserById(id) {
        const query = `
            SELECT id, email, username, first_name, last_name, 
                   avatar_url, is_active, email_verified, created_at, updated_at
            FROM users 
            WHERE id = $1
        `;
        
        const result = await this.db.query(query, [id]);
        return result.rows[0] || null;
    }
    
    async getUserByEmail(email) {
        const query = `
            SELECT id, email, username, password_hash, first_name, last_name,
                   is_active, email_verified, created_at
            FROM users 
            WHERE email = $1
        `;
        
        const result = await this.db.query(query, [email]);
        return result.rows[0] || null;
    }
    
    async updateUser(id, updateData) {
        const fields = [];
        const values = [];
        let paramCount = 1;
        
        Object.entries(updateData).forEach(([key, value]) => {
            if (value !== undefined) {
                fields.push(`${key} = $${paramCount}`);
                values.push(value);
                paramCount++;
            }
        });
        
        if (fields.length === 0) {
            throw new Error('No fields to update');
        }
        
        fields.push(`updated_at = CURRENT_TIMESTAMP`);
        values.push(id);
        
        const query = `
            UPDATE users 
            SET ${fields.join(', ')}
            WHERE id = $${paramCount}
            RETURNING id, email, username, first_name, last_name, updated_at
        `;
        
        const result = await this.db.query(query, values);
        return result.rows[0] || null;
    }
    
    async deleteUser(id) {
        const query = 'DELETE FROM users WHERE id = $1';
        const result = await this.db.query(query, [id]);
        return result.rowCount > 0;
    }
    
    async getUsersWithPagination(options = {}) {
        const { page = 1, limit = 10, sortBy = 'created_at', sortOrder = 'DESC' } = options;
        const offset = (page - 1) * limit;
        
        const query = `
            SELECT id, email, username, first_name, last_name, 
                   is_active, email_verified, created_at
            FROM users
            ORDER BY ${sortBy} ${sortOrder}
            LIMIT $1 OFFSET $2
        `;
        
        const countQuery = 'SELECT COUNT(*) FROM users';
        
        const [usersResult, countResult] = await Promise.all([
            this.db.query(query, [limit, offset]),
            this.db.query(countQuery)
        ]);
        
        return {
            users: usersResult.rows,
            pagination: {
                page,
                limit,
                total: parseInt(countResult.rows[0].count),
                totalPages: Math.ceil(parseInt(countResult.rows[0].count) / limit)
            }
        };
    }
}
```

### **MySQL with Node.js**

```javascript
const mysql = require('mysql2/promise');

class MySQLDatabase {
    constructor(config) {
        this.config = {
            host: config.host || 'localhost',
            port: config.port || 3306,
            user: config.user,
            password: config.password,
            database: config.database,
            connectionLimit: config.connectionLimit || 10,
            acquireTimeout: config.acquireTimeout || 60000,
            timeout: config.timeout || 60000,
            reconnect: true,
            charset: 'utf8mb4'
        };
        
        this.pool = mysql.createPool(this.config);
    }
    
    async query(sql, params = []) {
        const start = Date.now();
        try {
            const [rows, fields] = await this.pool.execute(sql, params);
            const duration = Date.now() - start;
            console.log('Executed query', { sql, duration, rows: rows.length });
            return { rows, fields };
        } catch (error) {
            console.error('Database query error:', error);
            throw error;
        }
    }
    
    async transaction(callback) {
        const connection = await this.pool.getConnection();
        try {
            await connection.beginTransaction();
            const result = await callback(connection);
            await connection.commit();
            return result;
        } catch (error) {
            await connection.rollback();
            throw error;
        } finally {
            connection.release();
        }
    }
    
    async close() {
        await this.pool.end();
    }
}

// MySQL-specific user service
class MySQLUserService {
    constructor(database) {
        this.db = database;
    }
    
    async createUser(userData) {
        const { email, username, passwordHash, firstName, lastName } = userData;
        
        const query = `
            INSERT INTO users (email, username, password_hash, first_name, last_name)
            VALUES (?, ?, ?, ?, ?)
        `;
        
        const result = await this.db.query(query, [
            email, username, passwordHash, firstName, lastName
        ]);
        
        return { id: result.rows.insertId, ...userData };
    }
    
    async getUserById(id) {
        const query = `
            SELECT id, email, username, first_name, last_name, 
                   avatar_url, is_active, email_verified, created_at, updated_at
            FROM users 
            WHERE id = ?
        `;
        
        const result = await this.db.query(query, [id]);
        return result.rows[0] || null;
    }
}
```

## 🍃 NoSQL Databases

### **MongoDB with Node.js**

```javascript
const { MongoClient, ObjectId } = require('mongodb');

class MongoDBDatabase {
    constructor(connectionString, options = {}) {
        this.connectionString = connectionString;
        this.options = {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            maxPoolSize: options.maxPoolSize || 10,
            serverSelectionTimeoutMS: options.serverSelectionTimeoutMS || 5000,
            socketTimeoutMS: options.socketTimeoutMS || 45000,
            ...options
        };
        this.client = null;
        this.db = null;
    }
    
    async connect(databaseName) {
        try {
            this.client = new MongoClient(this.connectionString, this.options);
            await this.client.connect();
            this.db = this.client.db(databaseName);
            console.log('Connected to MongoDB');
        } catch (error) {
            console.error('MongoDB connection error:', error);
            throw error;
        }
    }
    
    getCollection(collectionName) {
        if (!this.db) {
            throw new Error('Database not connected');
        }
        return this.db.collection(collectionName);
    }
    
    async close() {
        if (this.client) {
            await this.client.close();
            console.log('MongoDB connection closed');
        }
    }
}

// MongoDB User Service
class MongoDBUserService {
    constructor(database) {
        this.db = database;
        this.collection = database.getCollection('users');
    }
    
    async createUser(userData) {
        const user = {
            ...userData,
            createdAt: new Date(),
            updatedAt: new Date(),
            isActive: true,
            emailVerified: false
        };
        
        const result = await this.collection.insertOne(user);
        return { id: result.insertedId, ...user };
    }
    
    async getUserById(id) {
        const user = await this.collection.findOne({ _id: new ObjectId(id) });
        return user;
    }
    
    async getUserByEmail(email) {
        const user = await this.collection.findOne({ email });
        return user;
    }
    
    async updateUser(id, updateData) {
        const update = {
            ...updateData,
            updatedAt: new Date()
        };
        
        const result = await this.collection.findOneAndUpdate(
            { _id: new ObjectId(id) },
            { $set: update },
            { returnDocument: 'after' }
        );
        
        return result.value;
    }
    
    async deleteUser(id) {
        const result = await this.collection.deleteOne({ _id: new ObjectId(id) });
        return result.deletedCount > 0;
    }
    
    async getUsersWithPagination(options = {}) {
        const { page = 1, limit = 10, sortBy = 'createdAt', sortOrder = -1 } = options;
        const skip = (page - 1) * limit;
        
        const sort = {};
        sort[sortBy] = sortOrder;
        
        const [users, total] = await Promise.all([
            this.collection
                .find({})
                .sort(sort)
                .skip(skip)
                .limit(limit)
                .toArray(),
            this.collection.countDocuments({})
        ]);
        
        return {
            users,
            pagination: {
                page,
                limit,
                total,
                totalPages: Math.ceil(total / limit)
            }
        };
    }
    
    async searchUsers(searchTerm, options = {}) {
        const { page = 1, limit = 10 } = options;
        const skip = (page - 1) * limit;
        
        const query = {
            $or: [
                { email: { $regex: searchTerm, $options: 'i' } },
                { username: { $regex: searchTerm, $options: 'i' } },
                { firstName: { $regex: searchTerm, $options: 'i' } },
                { lastName: { $regex: searchTerm, $options: 'i' } }
            ]
        };
        
        const [users, total] = await Promise.all([
            this.collection
                .find(query)
                .skip(skip)
                .limit(limit)
                .toArray(),
            this.collection.countDocuments(query)
        ]);
        
        return {
            users,
            pagination: {
                page,
                limit,
                total,
                totalPages: Math.ceil(total / limit)
            }
        };
    }
}
```

### **Redis with Node.js**

```javascript
const redis = require('redis');

class RedisDatabase {
    constructor(config) {
        this.config = {
            host: config.host || 'localhost',
            port: config.port || 6379,
            password: config.password,
            db: config.db || 0,
            retryDelayOnFailover: 100,
            enableReadyCheck: false,
            maxRetriesPerRequest: null,
            ...config
        };
        
        this.client = redis.createClient(this.config);
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.client.on('connect', () => {
            console.log('Connected to Redis');
        });
        
        this.client.on('error', (err) => {
            console.error('Redis error:', err);
        });
        
        this.client.on('end', () => {
            console.log('Redis connection ended');
        });
    }
    
    async connect() {
        await this.client.connect();
    }
    
    async get(key) {
        return await this.client.get(key);
    }
    
    async set(key, value, ttl = null) {
        if (ttl) {
            return await this.client.setEx(key, ttl, value);
        }
        return await this.client.set(key, value);
    }
    
    async del(key) {
        return await this.client.del(key);
    }
    
    async exists(key) {
        return await this.client.exists(key);
    }
    
    async expire(key, seconds) {
        return await this.client.expire(key, seconds);
    }
    
    async hGet(key, field) {
        return await this.client.hGet(key, field);
    }
    
    async hSet(key, field, value) {
        return await this.client.hSet(key, field, value);
    }
    
    async hGetAll(key) {
        return await this.client.hGetAll(key);
    }
    
    async lPush(key, ...values) {
        return await this.client.lPush(key, values);
    }
    
    async rPop(key) {
        return await this.client.rPop(key);
    }
    
    async sAdd(key, ...members) {
        return await this.client.sAdd(key, members);
    }
    
    async sMembers(key) {
        return await this.client.sMembers(key);
    }
    
    async close() {
        await this.client.quit();
    }
}

// Redis caching service
class CacheService {
    constructor(redisClient) {
        this.redis = redisClient;
    }
    
    async get(key) {
        try {
            const value = await this.redis.get(key);
            return value ? JSON.parse(value) : null;
        } catch (error) {
            console.error('Cache get error:', error);
            return null;
        }
    }
    
    async set(key, value, ttl = 3600) {
        try {
            await this.redis.set(key, JSON.stringify(value), ttl);
        } catch (error) {
            console.error('Cache set error:', error);
        }
    }
    
    async del(key) {
        try {
            await this.redis.del(key);
        } catch (error) {
            console.error('Cache delete error:', error);
        }
    }
    
    async invalidatePattern(pattern) {
        try {
            const keys = await this.redis.keys(pattern);
            if (keys.length > 0) {
                await this.redis.del(...keys);
            }
        } catch (error) {
            console.error('Cache invalidation error:', error);
        }
    }
}
```

## ⚡ Database Optimization

### **Query Optimization**

```javascript
// Query optimization techniques
class QueryOptimizer {
    constructor(database) {
        this.db = database;
    }
    
    // Use prepared statements
    async getUserWithPosts(userId) {
        const query = `
            SELECT u.id, u.email, u.username, u.first_name, u.last_name,
                   p.id as post_id, p.title, p.content, p.created_at as post_created_at
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id
            WHERE u.id = $1
            ORDER BY p.created_at DESC
        `;
        
        return await this.db.query(query, [userId]);
    }
    
    // Use proper indexing
    async searchPosts(searchTerm, limit = 10) {
        const query = `
            SELECT p.id, p.title, p.content, p.created_at,
                   u.username, u.first_name, u.last_name
            FROM posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.title ILIKE $1 OR p.content ILIKE $1
            ORDER BY p.created_at DESC
            LIMIT $2
        `;
        
        return await this.db.query(query, [`%${searchTerm}%`, limit]);
    }
    
    // Use pagination with cursor-based approach for large datasets
    async getPostsWithCursor(cursor = null, limit = 20) {
        let query, params;
        
        if (cursor) {
            query = `
                SELECT id, title, content, created_at
                FROM posts
                WHERE created_at < $1
                ORDER BY created_at DESC
                LIMIT $2
            `;
            params = [cursor, limit];
        } else {
            query = `
                SELECT id, title, content, created_at
                FROM posts
                ORDER BY created_at DESC
                LIMIT $1
            `;
            params = [limit];
        }
        
        const result = await this.db.query(query, params);
        return result.rows;
    }
    
    // Use batch operations
    async createMultiplePosts(posts) {
        const values = posts.map((post, index) => {
            const baseIndex = index * 4;
            return `($${baseIndex + 1}, $${baseIndex + 2}, $${baseIndex + 3}, $${baseIndex + 4})`;
        }).join(', ');
        
        const params = posts.flatMap(post => [
            post.userId, post.title, post.content, new Date()
        ]);
        
        const query = `
            INSERT INTO posts (user_id, title, content, created_at)
            VALUES ${values}
            RETURNING id, title, created_at
        `;
        
        return await this.db.query(query, params);
    }
}
```

### **Connection Pooling**

```javascript
// Advanced connection pooling configuration
class DatabasePool {
    constructor(config) {
        this.config = {
            // Connection pool settings
            max: config.maxConnections || 20,
            min: config.minConnections || 5,
            idleTimeoutMillis: config.idleTimeout || 30000,
            connectionTimeoutMillis: config.connectionTimeout || 2000,
            
            // Health check settings
            healthCheckInterval: config.healthCheckInterval || 30000,
            
            // Retry settings
            maxRetries: config.maxRetries || 3,
            retryDelay: config.retryDelay || 1000,
            
            ...config
        };
        
        this.pool = null;
        this.healthCheckTimer = null;
    }
    
    async initialize() {
        this.pool = new Pool(this.config);
        this.startHealthCheck();
    }
    
    startHealthCheck() {
        this.healthCheckTimer = setInterval(async () => {
            try {
                await this.pool.query('SELECT 1');
                console.log('Database health check passed');
            } catch (error) {
                console.error('Database health check failed:', error);
            }
        }, this.config.healthCheckInterval);
    }
    
    async query(text, params = []) {
        let retries = 0;
        
        while (retries < this.config.maxRetries) {
            try {
                return await this.pool.query(text, params);
            } catch (error) {
                retries++;
                if (retries >= this.config.maxRetries) {
                    throw error;
                }
                
                console.log(`Query failed, retrying (${retries}/${this.config.maxRetries})`);
                await this.delay(this.config.retryDelay * retries);
            }
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    async close() {
        if (this.healthCheckTimer) {
            clearInterval(this.healthCheckTimer);
        }
        await this.pool.end();
    }
}
```

## 🔒 Database Security

### **SQL Injection Prevention**

```javascript
// Secure database operations
class SecureDatabaseService {
    constructor(database) {
        this.db = database;
    }
    
    // Always use parameterized queries
    async getUserByEmail(email) {
        // ✅ Good: Parameterized query
        const query = 'SELECT * FROM users WHERE email = $1';
        return await this.db.query(query, [email]);
        
        // ❌ Bad: String concatenation (vulnerable to SQL injection)
        // const query = `SELECT * FROM users WHERE email = '${email}'`;
    }
    
    // Input validation and sanitization
    async createUser(userData) {
        // Validate input
        const { email, username, password } = userData;
        
        if (!this.isValidEmail(email)) {
            throw new Error('Invalid email format');
        }
        
        if (!this.isValidUsername(username)) {
            throw new Error('Invalid username format');
        }
        
        if (!this.isStrongPassword(password)) {
            throw new Error('Password does not meet requirements');
        }
        
        // Sanitize input
        const sanitizedData = {
            email: email.toLowerCase().trim(),
            username: username.toLowerCase().trim(),
            password: await this.hashPassword(password)
        };
        
        const query = `
            INSERT INTO users (email, username, password_hash)
            VALUES ($1, $2, $3)
            RETURNING id, email, username, created_at
        `;
        
        return await this.db.query(query, [
            sanitizedData.email,
            sanitizedData.username,
            sanitizedData.password
        ]);
    }
    
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    isValidUsername(username) {
        const usernameRegex = /^[a-zA-Z0-9_]{3,20}$/;
        return usernameRegex.test(username);
    }
    
    isStrongPassword(password) {
        // At least 8 characters, 1 uppercase, 1 lowercase, 1 number, 1 special char
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        return passwordRegex.test(password);
    }
    
    async hashPassword(password) {
        const bcrypt = require('bcrypt');
        const saltRounds = 12;
        return await bcrypt.hash(password, saltRounds);
    }
}
```

## 📊 Performance Monitoring

### **Database Performance Monitoring**

```javascript
// Database performance monitoring
class DatabaseMonitor {
    constructor(database) {
        this.db = database;
        this.metrics = {
            queryCount: 0,
            totalQueryTime: 0,
            slowQueries: [],
            errorCount: 0
        };
    }
    
    async monitoredQuery(text, params = []) {
        const start = Date.now();
        this.metrics.queryCount++;
        
        try {
            const result = await this.db.query(text, params);
            const duration = Date.now() - start;
            this.metrics.totalQueryTime += duration;
            
            // Log slow queries
            if (duration > 1000) { // Queries taking more than 1 second
                this.metrics.slowQueries.push({
                    query: text,
                    duration,
                    timestamp: new Date()
                });
                console.warn(`Slow query detected: ${duration}ms`, text);
            }
            
            return result;
        } catch (error) {
            this.metrics.errorCount++;
            console.error('Database query error:', error);
            throw error;
        }
    }
    
    getMetrics() {
        return {
            ...this.metrics,
            averageQueryTime: this.metrics.queryCount > 0 
                ? this.metrics.totalQueryTime / this.metrics.queryCount 
                : 0
        };
    }
    
    resetMetrics() {
        this.metrics = {
            queryCount: 0,
            totalQueryTime: 0,
            slowQueries: [],
            errorCount: 0
        };
    }
}
```

---

**🎉 Master database design and optimization to build scalable Node.js applications!**

**Good luck with your database development journey! 🚀**
