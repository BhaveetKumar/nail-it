# 🗄️ Databases Complete Guide - Node.js Perspective

> **Comprehensive guide to database systems with Node.js implementations and best practices**

## 🎯 **Overview**

This guide covers all major database systems, their architectures, use cases, and Node.js implementations. From relational databases to NoSQL, vector databases, and time-series databases, this guide provides production-ready knowledge for backend engineers.

## 📚 **Table of Contents**

1. [Relational Databases](#relational-databases)
2. [NoSQL Databases](#nosql-databases)
3. [Vector Databases](#vector-databases)
4. [Time-Series Databases](#time-series-databases)
5. [Database Design Patterns](#database-design-patterns)
6. [Performance Optimization](#performance-optimization)
7. [Scaling Strategies](#scaling-strategies)
8. [Interview Questions](#interview-questions)

---

## 🗃️ **Relational Databases**

### **MySQL with Node.js**

```javascript
// MySQL Connection and Operations
const mysql = require('mysql2/promise');
const { Pool } = require('mysql2/promise');

class MySQLDatabase {
    constructor(config) {
        this.config = {
            host: config.host || 'localhost',
            user: config.user || 'root',
            password: config.password || '',
            database: config.database || 'test',
            port: config.port || 3306,
            connectionLimit: config.connectionLimit || 10,
            acquireTimeout: config.acquireTimeout || 60000,
            timeout: config.timeout || 60000,
            reconnect: config.reconnect || true,
            ...config
        };
        
        this.pool = null;
        this.initializePool();
    }
    
    initializePool() {
        this.pool = mysql.createPool(this.config);
        
        this.pool.on('connection', (connection) => {
            console.log('New MySQL connection established');
        });
        
        this.pool.on('error', (error) => {
            console.error('MySQL pool error:', error);
            if (error.code === 'PROTOCOL_CONNECTION_LOST') {
                this.initializePool();
            }
        });
    }
    
    async query(sql, params = []) {
        try {
            const [rows] = await this.pool.execute(sql, params);
            return {
                success: true,
                data: rows,
                affectedRows: rows.affectedRows || 0,
                insertId: rows.insertId || null
            };
        } catch (error) {
            console.error('MySQL query error:', error);
            return {
                success: false,
                error: error.message,
                code: error.code
            };
        }
    }
    
    async transaction(operations) {
        const connection = await this.pool.getConnection();
        
        try {
            await connection.beginTransaction();
            
            const results = [];
            for (const operation of operations) {
                const result = await connection.execute(operation.sql, operation.params || []);
                results.push(result);
            }
            
            await connection.commit();
            return { success: true, results };
            
        } catch (error) {
            await connection.rollback();
            throw error;
        } finally {
            connection.release();
        }
    }
    
    async close() {
        if (this.pool) {
            await this.pool.end();
        }
    }
}

// MySQL Schema Design
class MySQLSchemaDesigner {
    constructor(database) {
        this.db = database;
    }
    
    async createUserTable() {
        const sql = `
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                phone VARCHAR(20),
                is_active BOOLEAN DEFAULT TRUE,
                email_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_email (email),
                INDEX idx_username (username),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        `;
        
        return await this.db.query(sql);
    }
    
    async createProductTable() {
        const sql = `
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                price DECIMAL(10,2) NOT NULL,
                category_id INT,
                sku VARCHAR(100) UNIQUE,
                stock_quantity INT DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL,
                INDEX idx_name (name),
                INDEX idx_category (category_id),
                INDEX idx_sku (sku),
                INDEX idx_price (price),
                FULLTEXT idx_search (name, description)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        `;
        
        return await this.db.query(sql);
    }
    
    async createOrderTable() {
        const sql = `
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                order_number VARCHAR(50) UNIQUE NOT NULL,
                status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
                total_amount DECIMAL(10,2) NOT NULL,
                shipping_address JSON,
                billing_address JSON,
                payment_method VARCHAR(50),
                payment_status ENUM('pending', 'paid', 'failed', 'refunded') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                INDEX idx_order_number (order_number),
                INDEX idx_status (status),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        `;
        
        return await this.db.query(sql);
    }
}

// MySQL Query Optimization
class MySQLQueryOptimizer {
    constructor(database) {
        this.db = database;
    }
    
    async analyzeQuery(sql, params = []) {
        const explainSql = `EXPLAIN ${sql}`;
        const result = await this.db.query(explainSql, params);
        
        if (result.success) {
            return this.analyzeExplainResult(result.data);
        }
        
        return { error: 'Failed to analyze query' };
    }
    
    analyzeExplainResult(explainData) {
        const analysis = {
            recommendations: [],
            warnings: [],
            performance: 'good'
        };
        
        for (const row of explainData) {
            // Check for full table scan
            if (row.type === 'ALL') {
                analysis.warnings.push('Full table scan detected');
                analysis.recommendations.push('Add appropriate indexes');
                analysis.performance = 'poor';
            }
            
            // Check for temporary table
            if (row.Extra && row.Extra.includes('Using temporary')) {
                analysis.warnings.push('Temporary table used');
                analysis.recommendations.push('Optimize ORDER BY or GROUP BY clauses');
            }
            
            // Check for filesort
            if (row.Extra && row.Extra.includes('Using filesort')) {
                analysis.warnings.push('Filesort used');
                analysis.recommendations.push('Add index for ORDER BY clause');
            }
            
            // Check key usage
            if (!row.key && row.rows > 1000) {
                analysis.warnings.push('No index used for large result set');
                analysis.recommendations.push('Consider adding indexes');
            }
        }
        
        return analysis;
    }
    
    async createOptimizedIndexes() {
        const indexes = [
            'CREATE INDEX idx_users_email_active ON users(email, is_active)',
            'CREATE INDEX idx_products_category_price ON products(category_id, price)',
            'CREATE INDEX idx_orders_user_status ON orders(user_id, status)',
            'CREATE INDEX idx_orders_created_status ON orders(created_at, status)'
        ];
        
        const results = [];
        for (const indexSql of indexes) {
            const result = await this.db.query(indexSql);
            results.push(result);
        }
        
        return results;
    }
}
```

### **PostgreSQL with Node.js**

```javascript
// PostgreSQL Connection and Operations
const { Pool } = require('pg');

class PostgreSQLDatabase {
    constructor(config) {
        this.config = {
            host: config.host || 'localhost',
            user: config.user || 'postgres',
            password: config.password || '',
            database: config.database || 'test',
            port: config.port || 5432,
            max: config.max || 20,
            idleTimeoutMillis: config.idleTimeoutMillis || 30000,
            connectionTimeoutMillis: config.connectionTimeoutMillis || 2000,
            ...config
        };
        
        this.pool = new Pool(this.config);
        
        this.pool.on('connect', (client) => {
            console.log('New PostgreSQL connection established');
        });
        
        this.pool.on('error', (error) => {
            console.error('PostgreSQL pool error:', error);
        });
    }
    
    async query(sql, params = []) {
        const client = await this.pool.connect();
        
        try {
            const result = await client.query(sql, params);
            return {
                success: true,
                data: result.rows,
                rowCount: result.rowCount,
                command: result.command
            };
        } catch (error) {
            console.error('PostgreSQL query error:', error);
            return {
                success: false,
                error: error.message,
                code: error.code
            };
        } finally {
            client.release();
        }
    }
    
    async transaction(operations) {
        const client = await this.pool.connect();
        
        try {
            await client.query('BEGIN');
            
            const results = [];
            for (const operation of operations) {
                const result = await client.query(operation.sql, operation.params || []);
                results.push(result);
            }
            
            await client.query('COMMIT');
            return { success: true, results };
            
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

// PostgreSQL Advanced Features
class PostgreSQLAdvanced {
    constructor(database) {
        this.db = database;
    }
    
    async createJsonTable() {
        const sql = `
            CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                profile_data JSONB NOT NULL,
                preferences JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        `;
        
        return await this.db.query(sql);
    }
    
    async createJsonIndexes() {
        const indexes = [
            'CREATE INDEX idx_profile_data_gin ON user_profiles USING GIN (profile_data)',
            'CREATE INDEX idx_preferences_gin ON user_profiles USING GIN (preferences)',
            'CREATE INDEX idx_profile_email ON user_profiles USING GIN ((profile_data->>\'email\'))',
            'CREATE INDEX idx_profile_name ON user_profiles USING GIN ((profile_data->>\'name\'))'
        ];
        
        const results = [];
        for (const indexSql of indexes) {
            const result = await this.db.query(indexSql);
            results.push(result);
        }
        
        return results;
    }
    
    async searchJsonData(searchTerm) {
        const sql = `
            SELECT id, user_id, profile_data, preferences
            FROM user_profiles
            WHERE profile_data @> $1
               OR profile_data->>'name' ILIKE $2
               OR profile_data->>'email' ILIKE $2
        `;
        
        const params = [
            JSON.stringify({ name: searchTerm }),
            `%${searchTerm}%`
        ];
        
        return await this.db.query(sql, params);
    }
    
    async createFullTextSearch() {
        const sql = `
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                search_vector TSVECTOR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        `;
        
        await this.db.query(sql);
        
        // Create trigger for automatic vector update
        const triggerSql = `
            CREATE OR REPLACE FUNCTION update_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.search_vector := to_tsvector('english', NEW.title || ' ' || NEW.content);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            CREATE TRIGGER update_documents_search_vector
                BEFORE INSERT OR UPDATE ON documents
                FOR EACH ROW EXECUTE FUNCTION update_search_vector();
        `;
        
        return await this.db.query(triggerSql);
    }
    
    async fullTextSearch(query) {
        const sql = `
            SELECT id, title, content, ts_rank(search_vector, query) as rank
            FROM documents, to_tsquery('english', $1) query
            WHERE search_vector @@ query
            ORDER BY rank DESC
        `;
        
        return await this.db.query(sql, [query]);
    }
}
```

---

## 🍃 **NoSQL Databases**

### **MongoDB with Node.js**

```javascript
// MongoDB Connection and Operations
const { MongoClient, ObjectId } = require('mongodb');

class MongoDBDatabase {
    constructor(connectionString, dbName) {
        this.connectionString = connectionString;
        this.dbName = dbName;
        this.client = null;
        this.db = null;
    }
    
    async connect() {
        try {
            this.client = new MongoClient(this.connectionString, {
                useNewUrlParser: true,
                useUnifiedTopology: true,
                maxPoolSize: 10,
                serverSelectionTimeoutMS: 5000,
                socketTimeoutMS: 45000,
            });
            
            await this.client.connect();
            this.db = this.client.db(this.dbName);
            
            console.log('Connected to MongoDB');
            return { success: true };
        } catch (error) {
            console.error('MongoDB connection error:', error);
            return { success: false, error: error.message };
        }
    }
    
    async insertOne(collection, document) {
        try {
            const result = await this.db.collection(collection).insertOne(document);
            return {
                success: true,
                insertedId: result.insertedId,
                acknowledged: result.acknowledged
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async insertMany(collection, documents) {
        try {
            const result = await this.db.collection(collection).insertMany(documents);
            return {
                success: true,
                insertedIds: result.insertedIds,
                insertedCount: result.insertedCount
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async findOne(collection, filter, options = {}) {
        try {
            const result = await this.db.collection(collection).findOne(filter, options);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async find(collection, filter = {}, options = {}) {
        try {
            const cursor = this.db.collection(collection).find(filter, options);
            const results = await cursor.toArray();
            return { success: true, data: results };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async updateOne(collection, filter, update, options = {}) {
        try {
            const result = await this.db.collection(collection).updateOne(filter, update, options);
            return {
                success: true,
                matchedCount: result.matchedCount,
                modifiedCount: result.modifiedCount,
                acknowledged: result.acknowledged
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async deleteOne(collection, filter) {
        try {
            const result = await this.db.collection(collection).deleteOne(filter);
            return {
                success: true,
                deletedCount: result.deletedCount,
                acknowledged: result.acknowledged
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async aggregate(collection, pipeline) {
        try {
            const cursor = this.db.collection(collection).aggregate(pipeline);
            const results = await cursor.toArray();
            return { success: true, data: results };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async createIndex(collection, indexSpec, options = {}) {
        try {
            const result = await this.db.collection(collection).createIndex(indexSpec, options);
            return { success: true, indexName: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async close() {
        if (this.client) {
            await this.client.close();
        }
    }
}

// MongoDB Schema Design and Patterns
class MongoDBSchemaDesigner {
    constructor(database) {
        this.db = database;
    }
    
    async createUserSchema() {
        const userSchema = {
            email: { type: 'string', required: true, unique: true },
            username: { type: 'string', required: true, unique: true },
            passwordHash: { type: 'string', required: true },
            profile: {
                firstName: { type: 'string' },
                lastName: { type: 'string' },
                phone: { type: 'string' },
                avatar: { type: 'string' }
            },
            preferences: {
                theme: { type: 'string', default: 'light' },
                notifications: { type: 'boolean', default: true },
                language: { type: 'string', default: 'en' }
            },
            isActive: { type: 'boolean', default: true },
            emailVerified: { type: 'boolean', default: false },
            createdAt: { type: 'date', default: Date.now },
            updatedAt: { type: 'date', default: Date.now }
        };
        
        // Create indexes
        await this.db.createIndex('users', { email: 1 }, { unique: true });
        await this.db.createIndex('users', { username: 1 }, { unique: true });
        await this.db.createIndex('users', { 'profile.firstName': 1, 'profile.lastName': 1 });
        await this.db.createIndex('users', { createdAt: -1 });
        
        return userSchema;
    }
    
    async createProductSchema() {
        const productSchema = {
            name: { type: 'string', required: true },
            description: { type: 'string' },
            price: { type: 'number', required: true },
            category: { type: 'string', required: true },
            sku: { type: 'string', unique: true },
            inventory: {
                quantity: { type: 'number', default: 0 },
                reserved: { type: 'number', default: 0 },
                lowStockThreshold: { type: 'number', default: 10 }
            },
            attributes: { type: 'object' },
            images: [{ type: 'string' }],
            tags: [{ type: 'string' }],
            isActive: { type: 'boolean', default: true },
            createdAt: { type: 'date', default: Date.now },
            updatedAt: { type: 'date', default: Date.now }
        };
        
        // Create indexes
        await this.db.createIndex('products', { name: 'text', description: 'text' });
        await this.db.createIndex('products', { category: 1, price: 1 });
        await this.db.createIndex('products', { sku: 1 }, { unique: true });
        await this.db.createIndex('products', { tags: 1 });
        await this.db.createIndex('products', { 'inventory.quantity': 1 });
        
        return productSchema;
    }
    
    async createOrderSchema() {
        const orderSchema = {
            orderNumber: { type: 'string', required: true, unique: true },
            userId: { type: 'ObjectId', required: true, ref: 'users' },
            items: [{
                productId: { type: 'ObjectId', required: true, ref: 'products' },
                quantity: { type: 'number', required: true },
                price: { type: 'number', required: true },
                total: { type: 'number', required: true }
            }],
            totals: {
                subtotal: { type: 'number', required: true },
                tax: { type: 'number', default: 0 },
                shipping: { type: 'number', default: 0 },
                total: { type: 'number', required: true }
            },
            shipping: {
                address: { type: 'object', required: true },
                method: { type: 'string', required: true },
                trackingNumber: { type: 'string' }
            },
            payment: {
                method: { type: 'string', required: true },
                status: { type: 'string', enum: ['pending', 'paid', 'failed', 'refunded'], default: 'pending' },
                transactionId: { type: 'string' }
            },
            status: { type: 'string', enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'], default: 'pending' },
            createdAt: { type: 'date', default: Date.now },
            updatedAt: { type: 'date', default: Date.now }
        };
        
        // Create indexes
        await this.db.createIndex('orders', { orderNumber: 1 }, { unique: true });
        await this.db.createIndex('orders', { userId: 1, createdAt: -1 });
        await this.db.createIndex('orders', { status: 1 });
        await this.db.createIndex('orders', { 'payment.status': 1 });
        await this.db.createIndex('orders', { createdAt: -1 });
        
        return orderSchema;
    }
}

// MongoDB Aggregation Examples
class MongoDBAggregation {
    constructor(database) {
        this.db = database;
    }
    
    async getSalesReport(startDate, endDate) {
        const pipeline = [
            {
                $match: {
                    createdAt: {
                        $gte: new Date(startDate),
                        $lte: new Date(endDate)
                    },
                    status: { $in: ['delivered', 'shipped'] }
                }
            },
            {
                $unwind: '$items'
            },
            {
                $lookup: {
                    from: 'products',
                    localField: 'items.productId',
                    foreignField: '_id',
                    as: 'product'
                }
            },
            {
                $unwind: '$product'
            },
            {
                $group: {
                    _id: {
                        category: '$product.category',
                        month: { $month: '$createdAt' },
                        year: { $year: '$createdAt' }
                    },
                    totalSales: { $sum: '$items.total' },
                    totalQuantity: { $sum: '$items.quantity' },
                    orderCount: { $sum: 1 }
                }
            },
            {
                $sort: { '_id.year': -1, '_id.month': -1, totalSales: -1 }
            }
        ];
        
        return await this.db.aggregate('orders', pipeline);
    }
    
    async getUserActivityReport() {
        const pipeline = [
            {
                $lookup: {
                    from: 'orders',
                    localField: '_id',
                    foreignField: 'userId',
                    as: 'orders'
                }
            },
            {
                $addFields: {
                    totalOrders: { $size: '$orders' },
                    totalSpent: { $sum: '$orders.totals.total' },
                    lastOrderDate: { $max: '$orders.createdAt' }
                }
            },
            {
                $project: {
                    email: 1,
                    username: 1,
                    'profile.firstName': 1,
                    'profile.lastName': 1,
                    totalOrders: 1,
                    totalSpent: 1,
                    lastOrderDate: 1,
                    createdAt: 1
                }
            },
            {
                $sort: { totalSpent: -1 }
            }
        ];
        
        return await this.db.aggregate('users', pipeline);
    }
}
```

### **Redis with Node.js**

```javascript
// Redis Connection and Operations
const redis = require('redis');

class RedisDatabase {
    constructor(config) {
        this.config = {
            host: config.host || 'localhost',
            port: config.port || 6379,
            password: config.password || null,
            db: config.db || 0,
            retryDelayOnFailover: 100,
            enableReadyCheck: false,
            maxRetriesPerRequest: null,
            ...config
        };
        
        this.client = null;
        this.subscriber = null;
        this.publisher = null;
    }
    
    async connect() {
        try {
            this.client = redis.createClient(this.config);
            this.subscriber = redis.createClient(this.config);
            this.publisher = redis.createClient(this.config);
            
            await this.client.connect();
            await this.subscriber.connect();
            await this.publisher.connect();
            
            console.log('Connected to Redis');
            return { success: true };
        } catch (error) {
            console.error('Redis connection error:', error);
            return { success: false, error: error.message };
        }
    }
    
    // String operations
    async set(key, value, options = {}) {
        try {
            const result = await this.client.set(key, value, options);
            return { success: true, result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async get(key) {
        try {
            const result = await this.client.get(key);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async del(key) {
        try {
            const result = await this.client.del(key);
            return { success: true, deleted: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async exists(key) {
        try {
            const result = await this.client.exists(key);
            return { success: true, exists: result === 1 };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Hash operations
    async hset(key, field, value) {
        try {
            const result = await this.client.hSet(key, field, value);
            return { success: true, result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async hget(key, field) {
        try {
            const result = await this.client.hGet(key, field);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async hgetall(key) {
        try {
            const result = await this.client.hGetAll(key);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // List operations
    async lpush(key, ...values) {
        try {
            const result = await this.client.lPush(key, values);
            return { success: true, length: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async rpop(key) {
        try {
            const result = await this.client.rPop(key);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async lrange(key, start, stop) {
        try {
            const result = await this.client.lRange(key, start, stop);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Set operations
    async sadd(key, ...members) {
        try {
            const result = await this.client.sAdd(key, members);
            return { success: true, added: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async smembers(key) {
        try {
            const result = await this.client.sMembers(key);
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Sorted set operations
    async zadd(key, score, member) {
        try {
            const result = await this.client.zAdd(key, { score, value: member });
            return { success: true, result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async zrange(key, start, stop, withScores = false) {
        try {
            const result = await this.client.zRange(key, start, stop, { withScores });
            return { success: true, data: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Pub/Sub operations
    async publish(channel, message) {
        try {
            const result = await this.publisher.publish(channel, message);
            return { success: true, subscribers: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async subscribe(channel, callback) {
        try {
            await this.subscriber.subscribe(channel, callback);
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Expiration operations
    async expire(key, seconds) {
        try {
            const result = await this.client.expire(key, seconds);
            return { success: true, set: result === 1 };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async ttl(key) {
        try {
            const result = await this.client.ttl(key);
            return { success: true, ttl: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async close() {
        if (this.client) await this.client.quit();
        if (this.subscriber) await this.subscriber.quit();
        if (this.publisher) await this.publisher.quit();
    }
}

// Redis Caching Patterns
class RedisCacheManager {
    constructor(redisClient) {
        this.redis = redisClient;
        this.defaultTTL = 3600; // 1 hour
    }
    
    async cache(key, data, ttl = this.defaultTTL) {
        const serializedData = JSON.stringify(data);
        return await this.redis.set(key, serializedData, { EX: ttl });
    }
    
    async getCached(key) {
        const result = await this.redis.get(key);
        if (result.success && result.data) {
            try {
                return { success: true, data: JSON.parse(result.data) };
            } catch (error) {
                return { success: false, error: 'Failed to parse cached data' };
            }
        }
        return { success: false, error: 'Cache miss' };
    }
    
    async cacheWithFallback(key, fallbackFunction, ttl = this.defaultTTL) {
        // Try to get from cache first
        const cached = await this.getCached(key);
        if (cached.success) {
            return cached.data;
        }
        
        // Execute fallback function
        const data = await fallbackFunction();
        
        // Cache the result
        await this.cache(key, data, ttl);
        
        return data;
    }
    
    async invalidatePattern(pattern) {
        const keys = await this.redis.client.keys(pattern);
        if (keys.length > 0) {
            return await this.redis.client.del(keys);
        }
        return 0;
    }
}
```

---

## 🔍 **Vector Databases**

### **Pinecone with Node.js**

```javascript
// Pinecone Vector Database
const { Pinecone } = require('@pinecone-database/pinecone');

class PineconeVectorDB {
    constructor(apiKey, environment) {
        this.pinecone = new Pinecone({
            apiKey: apiKey,
            environment: environment
        });
        this.index = null;
    }
    
    async initialize(indexName) {
        try {
            this.index = this.pinecone.Index(indexName);
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async upsert(vectors) {
        try {
            const result = await this.index.upsert(vectors);
            return { success: true, upsertedCount: result.upsertedCount };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async query(queryVector, options = {}) {
        try {
            const result = await this.index.query({
                vector: queryVector,
                topK: options.topK || 10,
                includeMetadata: options.includeMetadata || true,
                includeValues: options.includeValues || false,
                filter: options.filter
            });
            
            return {
                success: true,
                matches: result.matches,
                namespace: result.namespace
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async delete(ids) {
        try {
            const result = await this.index.deleteMany(ids);
            return { success: true, deletedCount: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async describeIndexStats() {
        try {
            const result = await this.index.describeIndexStats();
            return { success: true, stats: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
}

// Vector Search Implementation
class VectorSearchService {
    constructor(vectorDB) {
        this.vectorDB = vectorDB;
        this.embeddingService = new EmbeddingService();
    }
    
    async indexDocument(id, text, metadata = {}) {
        try {
            // Generate embedding
            const embedding = await this.embeddingService.generateEmbedding(text);
            
            // Upsert to vector database
            const result = await this.vectorDB.upsert([{
                id: id,
                values: embedding,
                metadata: {
                    text: text,
                    ...metadata
                }
            }]);
            
            return result;
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async searchSimilar(query, options = {}) {
        try {
            // Generate query embedding
            const queryEmbedding = await this.embeddingService.generateEmbedding(query);
            
            // Search vector database
            const result = await this.vectorDB.query(queryEmbedding, options);
            
            return result;
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async batchIndex(documents) {
        const vectors = [];
        
        for (const doc of documents) {
            const embedding = await this.embeddingService.generateEmbedding(doc.text);
            vectors.push({
                id: doc.id,
                values: embedding,
                metadata: doc.metadata || {}
            });
        }
        
        return await this.vectorDB.upsert(vectors);
    }
}

// Embedding Service
class EmbeddingService {
    constructor() {
        // This would typically use OpenAI, Cohere, or local embedding models
        this.model = 'text-embedding-ada-002';
    }
    
    async generateEmbedding(text) {
        // Simulate embedding generation
        // In production, this would call an embedding API
        const embedding = Array.from({ length: 1536 }, () => Math.random() - 0.5);
        return embedding;
    }
    
    async generateBatchEmbeddings(texts) {
        const embeddings = [];
        for (const text of texts) {
            const embedding = await this.generateEmbedding(text);
            embeddings.push(embedding);
        }
        return embeddings;
    }
}
```

---

## ⏰ **Time-Series Databases**

### **InfluxDB with Node.js**

```javascript
// InfluxDB Time-Series Database
const { InfluxDB, Point } = require('@influxdata/influxdb-client');

class InfluxDBTimeSeries {
    constructor(url, token, org, bucket) {
        this.client = new InfluxDB({ url, token });
        this.org = org;
        this.bucket = bucket;
        this.writeApi = this.client.getWriteApi(org, bucket);
        this.queryApi = this.client.getQueryApi(org);
    }
    
    async writePoint(measurement, fields, tags = {}, timestamp = null) {
        try {
            const point = new Point(measurement);
            
            // Add tags
            Object.entries(tags).forEach(([key, value]) => {
                point.tag(key, value);
            });
            
            // Add fields
            Object.entries(fields).forEach(([key, value]) => {
                if (typeof value === 'number') {
                    point.floatField(key, value);
                } else if (typeof value === 'boolean') {
                    point.booleanField(key, value);
                } else {
                    point.stringField(key, value);
                }
            });
            
            // Set timestamp
            if (timestamp) {
                point.timestamp(timestamp);
            }
            
            this.writeApi.writePoint(point);
            await this.writeApi.flush();
            
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async writePoints(points) {
        try {
            for (const pointData of points) {
                const point = new Point(pointData.measurement);
                
                Object.entries(pointData.tags || {}).forEach(([key, value]) => {
                    point.tag(key, value);
                });
                
                Object.entries(pointData.fields || {}).forEach(([key, value]) => {
                    if (typeof value === 'number') {
                        point.floatField(key, value);
                    } else if (typeof value === 'boolean') {
                        point.booleanField(key, value);
                    } else {
                        point.stringField(key, value);
                    }
                });
                
                if (pointData.timestamp) {
                    point.timestamp(pointData.timestamp);
                }
                
                this.writeApi.writePoint(point);
            }
            
            await this.writeApi.flush();
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async query(fluxQuery) {
        try {
            const results = [];
            
            await this.queryApi.queryRows(fluxQuery, {
                next(row, tableMeta) {
                    const record = tableMeta.toObject(row);
                    results.push(record);
                },
                error(error) {
                    throw error;
                },
                complete() {
                    // Query completed
                }
            });
            
            return { success: true, data: results };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async getMetrics(measurement, startTime, endTime, tags = {}) {
        let fluxQuery = `
            from(bucket: "${this.bucket}")
            |> range(start: ${startTime}, stop: ${endTime})
            |> filter(fn: (r) => r._measurement == "${measurement}")
        `;
        
        // Add tag filters
        Object.entries(tags).forEach(([key, value]) => {
            fluxQuery += `|> filter(fn: (r) => r.${key} == "${value}")`;
        });
        
        fluxQuery += `|> aggregateWindow(every: 1m, fn: mean, createEmpty: false)`;
        
        return await this.query(fluxQuery);
    }
    
    async getAggregatedData(measurement, startTime, endTime, aggregation = 'mean') {
        const fluxQuery = `
            from(bucket: "${this.bucket}")
            |> range(start: ${startTime}, stop: ${endTime})
            |> filter(fn: (r) => r._measurement == "${measurement}")
            |> aggregateWindow(every: 1h, fn: ${aggregation}, createEmpty: false)
            |> yield(name: "aggregated")
        `;
        
        return await this.query(fluxQuery);
    }
    
    close() {
        this.writeApi.close();
    }
}

// Time-Series Analytics
class TimeSeriesAnalytics {
    constructor(influxDB) {
        this.influxDB = influxDB;
    }
    
    async recordMetric(metricName, value, tags = {}) {
        return await this.influxDB.writePoint(metricName, { value }, tags);
    }
    
    async recordSystemMetrics() {
        const os = require('os');
        const timestamp = new Date();
        
        const metrics = [
            {
                measurement: 'system_cpu',
                fields: { usage: process.cpuUsage().user / 1000000 },
                tags: { host: os.hostname(), type: 'user' },
                timestamp
            },
            {
                measurement: 'system_memory',
                fields: { 
                    total: os.totalmem(),
                    free: os.freemem(),
                    used: os.totalmem() - os.freemem()
                },
                tags: { host: os.hostname() },
                timestamp
            },
            {
                measurement: 'system_load',
                fields: { 
                    load1: os.loadavg()[0],
                    load5: os.loadavg()[1],
                    load15: os.loadavg()[2]
                },
                tags: { host: os.hostname() },
                timestamp
            }
        ];
        
        return await this.influxDB.writePoints(metrics);
    }
    
    async getPerformanceReport(startTime, endTime) {
        const queries = [
            this.influxDB.getMetrics('system_cpu', startTime, endTime),
            this.influxDB.getMetrics('system_memory', startTime, endTime),
            this.influxDB.getMetrics('system_load', startTime, endTime)
        ];
        
        const results = await Promise.all(queries);
        
        return {
            cpu: results[0],
            memory: results[1],
            load: results[2]
        };
    }
}
```

---

## 🎯 **Interview Questions**

### **Database Design Questions**

1. **How would you design a database for an e-commerce platform?**
   - Discuss normalization vs denormalization
   - Explain indexing strategies
   - Cover scalability considerations

2. **What's the difference between ACID and BASE properties?**
   - ACID: Atomicity, Consistency, Isolation, Durability
   - BASE: Basically Available, Soft state, Eventual consistency

3. **How do you handle database migrations in production?**
   - Blue-green deployments
   - Rolling updates
   - Backward compatibility

### **Performance Optimization Questions**

1. **How would you optimize a slow query?**
   - Query analysis and profiling
   - Index optimization
   - Query rewriting

2. **What are the different types of database indexes?**
   - B-tree, Hash, Bitmap, Composite
   - When to use each type

3. **How do you handle database connection pooling?**
   - Connection pool sizing
   - Connection lifecycle management
   - Monitoring and alerting

### **Scaling Questions**

1. **How would you scale a database to handle 1M+ users?**
   - Read replicas
   - Sharding strategies
   - Caching layers

2. **What's the difference between horizontal and vertical scaling?**
   - Vertical: More powerful hardware
   - Horizontal: More servers

3. **How do you handle database consistency in distributed systems?**
   - CAP theorem
   - Eventual consistency
   - Conflict resolution

---

**🎉 This comprehensive guide covers all major database systems with Node.js implementations!**
