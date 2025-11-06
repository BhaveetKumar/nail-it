---
# Auto-generated front matter
Title: Advanced Database Interviews
LastUpdated: 2025-11-06T20:45:58.351145
Tags: []
Status: draft
---

# Advanced Database Interviews

## Table of Contents
- [Introduction](#introduction)
- [Database Design](#database-design)
- [Query Optimization](#query-optimization)
- [Indexing Strategies](#indexing-strategies)
- [Transaction Management](#transaction-management)
- [Replication and Sharding](#replication-and-sharding)
- [Performance Tuning](#performance-tuning)
- [NoSQL Databases](#nosql-databases)

## Introduction

Advanced database interviews test your understanding of complex database concepts, optimization techniques, and distributed database systems.

## Database Design

### Normalization and Denormalization

```sql
-- Normalized design (3NF)
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_profiles (
    user_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    phone VARCHAR(20),
    address TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2),
    status VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

-- Denormalized design for read optimization
CREATE TABLE user_orders_denormalized (
    user_id INT,
    username VARCHAR(50),
    email VARCHAR(100),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    order_id INT,
    order_date TIMESTAMP,
    product_id INT,
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    order_status VARCHAR(20),
    INDEX idx_user_id (user_id),
    INDEX idx_order_date (order_date),
    INDEX idx_product_id (product_id)
);
```

### Database Schema Design Patterns

```sql
-- Single Table Inheritance
CREATE TABLE vehicles (
    id INT PRIMARY KEY,
    type ENUM('car', 'truck', 'motorcycle'),
    make VARCHAR(50),
    model VARCHAR(50),
    year INT,
    -- Car-specific fields
    doors INT,
    -- Truck-specific fields
    cargo_capacity DECIMAL(10,2),
    -- Motorcycle-specific fields
    engine_cc INT
);

-- Class Table Inheritance
CREATE TABLE vehicles (
    id INT PRIMARY KEY,
    make VARCHAR(50),
    model VARCHAR(50),
    year INT
);

CREATE TABLE cars (
    vehicle_id INT PRIMARY KEY,
    doors INT,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
);

CREATE TABLE trucks (
    vehicle_id INT PRIMARY KEY,
    cargo_capacity DECIMAL(10,2),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
);

-- Concrete Table Inheritance
CREATE TABLE cars (
    id INT PRIMARY KEY,
    make VARCHAR(50),
    model VARCHAR(50),
    year INT,
    doors INT
);

CREATE TABLE trucks (
    id INT PRIMARY KEY,
    make VARCHAR(50),
    model VARCHAR(50),
    year INT,
    cargo_capacity DECIMAL(10,2)
);

-- Audit Trail Pattern
CREATE TABLE audit_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(50) NOT NULL,
    record_id INT NOT NULL,
    action ENUM('INSERT', 'UPDATE', 'DELETE') NOT NULL,
    old_values JSON,
    new_values JSON,
    changed_by INT,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_record (table_name, record_id),
    INDEX idx_changed_at (changed_at)
);

-- Soft Delete Pattern
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2),
    deleted_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_deleted_at (deleted_at)
);

-- Event Sourcing Pattern
CREATE TABLE events (
    id INT PRIMARY KEY AUTO_INCREMENT,
    aggregate_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSON NOT NULL,
    version INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_aggregate_version (aggregate_id, version)
);
```

## Query Optimization

### Query Analysis and Optimization

```sql
-- Query execution plan analysis
EXPLAIN SELECT u.username, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
GROUP BY u.id, u.username
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC
LIMIT 10;

-- Optimized query with proper indexing
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Query with covering index
CREATE INDEX idx_orders_covering ON orders(user_id, created_at, status);

-- Optimized query using covering index
SELECT u.username, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
  AND o.status = 'completed'
GROUP BY u.id, u.username
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC
LIMIT 10;

-- Window functions for complex analytics
SELECT 
    user_id,
    order_date,
    total_amount,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) as order_sequence,
    LAG(total_amount) OVER (PARTITION BY user_id ORDER BY order_date) as previous_amount,
    LEAD(total_amount) OVER (PARTITION BY user_id ORDER BY order_date) as next_amount,
    AVG(total_amount) OVER (PARTITION BY user_id ORDER BY order_date 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) as moving_avg
FROM orders
WHERE order_date >= '2023-01-01';

-- Common Table Expressions (CTEs) for complex queries
WITH monthly_sales AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') as month,
        SUM(total_amount) as total_sales,
        COUNT(*) as order_count
    FROM orders
    WHERE order_date >= '2023-01-01'
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
),
sales_growth AS (
    SELECT 
        month,
        total_sales,
        order_count,
        LAG(total_sales) OVER (ORDER BY month) as previous_month_sales,
        ROUND(((total_sales - LAG(total_sales) OVER (ORDER BY month)) / 
               LAG(total_sales) OVER (ORDER BY month)) * 100, 2) as growth_percentage
    FROM monthly_sales
)
SELECT * FROM sales_growth;
```

### Advanced Query Techniques

```sql
-- Recursive CTEs for hierarchical data
WITH RECURSIVE category_tree AS (
    -- Base case: root categories
    SELECT id, name, parent_id, 0 as level, CAST(name AS CHAR(1000)) as path
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- Recursive case: child categories
    SELECT c.id, c.name, c.parent_id, ct.level + 1, 
           CONCAT(ct.path, ' > ', c.name)
    FROM categories c
    INNER JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree ORDER BY path;

-- Pivot queries for reporting
SELECT 
    user_id,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_orders,
    SUM(CASE WHEN status = 'pending' THEN total_amount ELSE 0 END) as pending_amount,
    SUM(CASE WHEN status = 'completed' THEN total_amount ELSE 0 END) as completed_amount
FROM orders
GROUP BY user_id;

-- Advanced aggregation with ROLLUP
SELECT 
    COALESCE(category, 'Total') as category,
    COALESCE(subcategory, 'Subtotal') as subcategory,
    COUNT(*) as product_count,
    AVG(price) as avg_price,
    SUM(price) as total_value
FROM products
GROUP BY category, subcategory WITH ROLLUP;

-- Query with EXISTS vs IN optimization
-- Less efficient
SELECT u.* FROM users u
WHERE u.id IN (
    SELECT DISTINCT user_id FROM orders 
    WHERE order_date >= '2023-01-01'
);

-- More efficient
SELECT u.* FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.user_id = u.id 
    AND o.order_date >= '2023-01-01'
);
```

## Indexing Strategies

### Advanced Indexing

```sql
-- Composite indexes
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);
CREATE INDEX idx_orders_date_status ON orders(order_date, status);

-- Partial indexes (MySQL 8.0+)
CREATE INDEX idx_active_orders ON orders(user_id, order_date) 
WHERE status = 'active';

-- Functional indexes
CREATE INDEX idx_orders_year ON orders((YEAR(order_date)));
CREATE INDEX idx_users_email_domain ON users((SUBSTRING(email, LOCATE('@', email) + 1)));

-- Full-text search indexes
CREATE FULLTEXT INDEX idx_products_search ON products(name, description);
CREATE FULLTEXT INDEX idx_products_name ON products(name) WITH PARSER ngram;

-- Spatial indexes
CREATE SPATIAL INDEX idx_locations_coords ON locations(coordinates);

-- Covering indexes
CREATE INDEX idx_orders_covering ON orders(user_id, order_date, status, total_amount);

-- Index hints
SELECT /*+ USE_INDEX(orders, idx_orders_user_date) */ 
    u.username, o.order_date, o.total_amount
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.user_id = 123 AND o.order_date >= '2023-01-01';
```

### Index Maintenance and Monitoring

```sql
-- Index usage statistics
SELECT 
    TABLE_SCHEMA,
    TABLE_NAME,
    INDEX_NAME,
    CARDINALITY,
    SUB_PART,
    PACKED,
    NULLABLE,
    INDEX_TYPE
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'your_database'
ORDER BY TABLE_NAME, INDEX_NAME;

-- Unused indexes
SELECT 
    t.TABLE_SCHEMA,
    t.TABLE_NAME,
    s.INDEX_NAME,
    s.CARDINALITY
FROM information_schema.TABLES t
JOIN information_schema.STATISTICS s ON t.TABLE_NAME = s.TABLE_NAME
LEFT JOIN performance_schema.table_io_waits_summary_by_index_usage p 
    ON s.TABLE_SCHEMA = p.OBJECT_SCHEMA 
    AND s.TABLE_NAME = p.OBJECT_NAME 
    AND s.INDEX_NAME = p.INDEX_NAME
WHERE t.TABLE_SCHEMA = 'your_database'
    AND p.INDEX_NAME IS NULL
    AND s.INDEX_NAME != 'PRIMARY';

-- Index fragmentation analysis
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) AS 'Size (MB)',
    ROUND((DATA_FREE / 1024 / 1024), 2) AS 'Free Space (MB)',
    ROUND((DATA_FREE / (DATA_LENGTH + INDEX_LENGTH)) * 100, 2) AS 'Fragmentation %'
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'your_database'
    AND DATA_FREE > 0
ORDER BY Fragmentation DESC;

-- Rebuild fragmented indexes
ALTER TABLE orders ENGINE=InnoDB;
OPTIMIZE TABLE orders;
```

## Transaction Management

### ACID Properties Implementation

```sql
-- Transaction isolation levels
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Explicit transaction control
START TRANSACTION;

INSERT INTO orders (user_id, total_amount, status) 
VALUES (123, 99.99, 'pending');

INSERT INTO order_items (order_id, product_id, quantity, price)
VALUES (LAST_INSERT_ID(), 456, 2, 49.99);

UPDATE products 
SET stock_quantity = stock_quantity - 2 
WHERE id = 456;

-- Check for sufficient stock
IF (SELECT stock_quantity FROM products WHERE id = 456) < 0 THEN
    ROLLBACK;
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Insufficient stock';
ELSE
    COMMIT;
END IF;

-- Savepoints for partial rollback
START TRANSACTION;

INSERT INTO orders (user_id, total_amount, status) 
VALUES (123, 99.99, 'pending');

SAVEPOINT order_created;

INSERT INTO order_items (order_id, product_id, quantity, price)
VALUES (LAST_INSERT_ID(), 456, 2, 49.99);

-- If item insertion fails, rollback to savepoint
IF @@ERROR != 0 THEN
    ROLLBACK TO SAVEPOINT order_created;
    -- Continue with order without items
END IF;

COMMIT;
```

### Deadlock Prevention

```sql
-- Deadlock detection and prevention
-- Always access tables in the same order
-- Use appropriate indexes to minimize lock time
-- Keep transactions short
-- Use lower isolation levels when possible

-- Example of deadlock-prone code (DON'T DO THIS)
-- Transaction 1: UPDATE users SET ... WHERE id = 1; UPDATE orders SET ... WHERE user_id = 2;
-- Transaction 2: UPDATE users SET ... WHERE id = 2; UPDATE orders SET ... WHERE user_id = 1;

-- Better approach: consistent ordering
-- Transaction 1: UPDATE users SET ... WHERE id = 1; UPDATE orders SET ... WHERE user_id = 1;
-- Transaction 2: UPDATE users SET ... WHERE id = 2; UPDATE orders SET ... WHERE user_id = 2;

-- Lock timeout configuration
SET innodb_lock_wait_timeout = 50;
SET lock_wait_timeout = 50;

-- Deadlock detection
SHOW ENGINE INNODB STATUS;

-- Lock monitoring
SELECT 
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;
```

## Replication and Sharding

### Master-Slave Replication

```sql
-- Master server configuration
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog-format = ROW
gtid-mode = ON
enforce-gtid-consistency = ON

-- Slave server configuration
[mysqld]
server-id = 2
relay-log = mysql-relay-bin
read-only = 1
gtid-mode = ON
enforce-gtid-consistency = ON

-- Setup replication
-- On master
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';

-- On slave
CHANGE MASTER TO
    MASTER_HOST='master_host',
    MASTER_USER='repl',
    MASTER_PASSWORD='password',
    MASTER_AUTO_POSITION=1;

START SLAVE;

-- Monitor replication
SHOW SLAVE STATUS\G
SHOW MASTER STATUS\G

-- Read/write splitting
-- Write queries go to master
INSERT INTO orders (user_id, total_amount) VALUES (123, 99.99);

-- Read queries can go to slave
SELECT * FROM orders WHERE user_id = 123;
```

### Database Sharding

```sql
-- Horizontal sharding by user_id
-- Shard 1: user_id % 4 = 0
CREATE TABLE orders_shard_1 LIKE orders;
ALTER TABLE orders_shard_1 ADD CONSTRAINT chk_user_id CHECK (user_id % 4 = 0);

-- Shard 2: user_id % 4 = 1
CREATE TABLE orders_shard_2 LIKE orders;
ALTER TABLE orders_shard_2 ADD CONSTRAINT chk_user_id CHECK (user_id % 4 = 1);

-- Shard 3: user_id % 4 = 2
CREATE TABLE orders_shard_3 LIKE orders;
ALTER TABLE orders_shard_3 ADD CONSTRAINT chk_user_id CHECK (user_id % 4 = 2);

-- Shard 4: user_id % 4 = 3
CREATE TABLE orders_shard_4 LIKE orders;
ALTER TABLE orders_shard_4 ADD CONSTRAINT chk_user_id CHECK (user_id % 4 = 3);

-- Sharding function
DELIMITER //
CREATE FUNCTION get_shard_table(user_id INT) 
RETURNS VARCHAR(50)
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE shard_num INT;
    SET shard_num = user_id % 4;
    RETURN CONCAT('orders_shard_', shard_num + 1);
END//
DELIMITER ;

-- Cross-shard queries using UNION
SELECT * FROM orders_shard_1 WHERE order_date >= '2023-01-01'
UNION ALL
SELECT * FROM orders_shard_2 WHERE order_date >= '2023-01-01'
UNION ALL
SELECT * FROM orders_shard_3 WHERE order_date >= '2023-01-01'
UNION ALL
SELECT * FROM orders_shard_4 WHERE order_date >= '2023-01-01';
```

## Performance Tuning

### Query Performance Optimization

```sql
-- Query cache configuration
SET query_cache_type = ON;
SET query_cache_size = 64M;
SET query_cache_limit = 2M;

-- Buffer pool optimization
SET innodb_buffer_pool_size = 1G;
SET innodb_buffer_pool_instances = 4;

-- Connection optimization
SET max_connections = 200;
SET thread_cache_size = 16;
SET table_open_cache = 2000;

-- Slow query log
SET slow_query_log = ON;
SET slow_query_log_file = '/var/log/mysql/slow.log';
SET long_query_time = 2;

-- Query profiling
SET profiling = ON;
SELECT * FROM orders WHERE user_id = 123;
SHOW PROFILES;
SHOW PROFILE FOR QUERY 1;

-- Performance schema analysis
SELECT 
    EVENT_NAME,
    COUNT_STAR,
    SUM_TIMER_WAIT/1000000000 as total_time_sec,
    AVG_TIMER_WAIT/1000000000 as avg_time_sec
FROM performance_schema.events_waits_summary_global_by_event_name
WHERE COUNT_STAR > 0
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 10;

-- Index usage analysis
SELECT 
    OBJECT_SCHEMA,
    OBJECT_NAME,
    INDEX_NAME,
    COUNT_FETCH,
    COUNT_INSERT,
    COUNT_UPDATE,
    COUNT_DELETE
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE OBJECT_SCHEMA = 'your_database'
ORDER BY COUNT_FETCH DESC;
```

### Database Monitoring

```sql
-- Database size monitoring
SELECT 
    table_schema AS 'Database',
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'Size (MB)'
FROM information_schema.tables
GROUP BY table_schema
ORDER BY SUM(data_length + index_length) DESC;

-- Table size monitoring
SELECT 
    table_name AS 'Table',
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)',
    table_rows AS 'Rows'
FROM information_schema.tables
WHERE table_schema = 'your_database'
ORDER BY (data_length + index_length) DESC;

-- Connection monitoring
SELECT 
    ID,
    USER,
    HOST,
    DB,
    COMMAND,
    TIME,
    STATE,
    INFO
FROM information_schema.processlist
WHERE COMMAND != 'Sleep'
ORDER BY TIME DESC;

-- Lock monitoring
SELECT 
    r.trx_id AS waiting_trx_id,
    r.trx_mysql_thread_id AS waiting_thread,
    r.trx_query AS waiting_query,
    b.trx_id AS blocking_trx_id,
    b.trx_mysql_thread_id AS blocking_thread,
    b.trx_query AS blocking_query
FROM information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;
```

## NoSQL Databases

### MongoDB Operations

```javascript
// MongoDB connection and basic operations
const { MongoClient } = require('mongodb');

const client = new MongoClient('mongodb://localhost:27017');
await client.connect();
const db = client.db('ecommerce');

// Document design for orders
const orderSchema = {
    _id: ObjectId,
    user_id: Number,
    order_date: Date,
    status: String,
    items: [{
        product_id: Number,
        name: String,
        quantity: Number,
        price: Number
    }],
    total_amount: Number,
    shipping_address: {
        street: String,
        city: String,
        state: String,
        zip: String
    },
    payment_info: {
        method: String,
        transaction_id: String
    }
};

// Insert operations
await db.collection('orders').insertOne({
    user_id: 123,
    order_date: new Date(),
    status: 'pending',
    items: [
        { product_id: 456, name: 'Laptop', quantity: 1, price: 999.99 },
        { product_id: 789, name: 'Mouse', quantity: 2, price: 29.99 }
    ],
    total_amount: 1059.97,
    shipping_address: {
        street: '123 Main St',
        city: 'New York',
        state: 'NY',
        zip: '10001'
    },
    payment_info: {
        method: 'credit_card',
        transaction_id: 'txn_123456'
    }
});

// Query operations
// Find orders by user
const userOrders = await db.collection('orders')
    .find({ user_id: 123 })
    .sort({ order_date: -1 })
    .toArray();

// Aggregation pipeline
const orderStats = await db.collection('orders').aggregate([
    {
        $match: {
            order_date: { $gte: new Date('2023-01-01') }
        }
    },
    {
        $group: {
            _id: '$status',
            count: { $sum: 1 },
            total_amount: { $sum: '$total_amount' },
            avg_amount: { $avg: '$total_amount' }
        }
    },
    {
        $sort: { count: -1 }
    }
]).toArray();

// Update operations
await db.collection('orders').updateOne(
    { _id: ObjectId('...') },
    { 
        $set: { status: 'completed' },
        $push: { 
            status_history: {
                status: 'completed',
                timestamp: new Date(),
                note: 'Order processed successfully'
            }
        }
    }
);

// Indexing
await db.collection('orders').createIndex({ user_id: 1, order_date: -1 });
await db.collection('orders').createIndex({ status: 1 });
await db.collection('orders').createIndex({ 'items.product_id': 1 });
await db.collection('orders').createIndex({ 
    'shipping_address.city': 1, 
    'shipping_address.state': 1 
});

// Text search
await db.collection('orders').createIndex({
    'items.name': 'text',
    'shipping_address.city': 'text'
});

const searchResults = await db.collection('orders')
    .find({ $text: { $search: 'laptop new york' } })
    .toArray();
```

### Redis Operations

```javascript
// Redis operations
const redis = require('redis');
const client = redis.createClient();

// String operations
await client.set('user:123:name', 'John Doe');
await client.setex('user:123:session', 3600, 'session_token_123');
const userName = await client.get('user:123:name');

// Hash operations
await client.hset('user:123', {
    name: 'John Doe',
    email: 'john@example.com',
    age: 30
});
const userData = await client.hgetall('user:123');

// List operations
await client.lpush('recent_orders', 'order_456');
await client.lpush('recent_orders', 'order_789');
const recentOrders = await client.lrange('recent_orders', 0, 9);

// Set operations
await client.sadd('product_categories', 'electronics', 'books', 'clothing');
await client.sadd('user_123_interests', 'electronics', 'gaming');
const commonInterests = await client.sinter('product_categories', 'user_123_interests');

// Sorted set operations
await client.zadd('product_ratings', 4.5, 'product_123', 4.2, 'product_456');
await client.zadd('product_ratings', 4.8, 'product_789');
const topProducts = await client.zrevrange('product_ratings', 0, 9, 'WITHSCORES');

// Pub/Sub
const subscriber = redis.createClient();
const publisher = redis.createClient();

subscriber.on('message', (channel, message) => {
    console.log(`Received: ${message} from ${channel}`);
});

await subscriber.subscribe('order_updates');
await publisher.publish('order_updates', 'Order 123 completed');

// Caching patterns
// Cache-aside pattern
async function getProduct(productId) {
    const cacheKey = `product:${productId}`;
    let product = await client.get(cacheKey);
    
    if (!product) {
        // Cache miss - get from database
        product = await db.products.findById(productId);
        if (product) {
            await client.setex(cacheKey, 3600, JSON.stringify(product));
        }
    } else {
        product = JSON.parse(product);
    }
    
    return product;
}

// Write-through pattern
async function updateProduct(productId, updates) {
    // Update database
    const product = await db.products.update(productId, updates);
    
    // Update cache
    const cacheKey = `product:${productId}`;
    await client.setex(cacheKey, 3600, JSON.stringify(product));
    
    return product;
}

// Write-behind pattern
async function createOrder(orderData) {
    // Add to cache immediately
    const orderId = generateId();
    const cacheKey = `order:${orderId}`;
    await client.setex(cacheKey, 3600, JSON.stringify(orderData));
    
    // Queue for database write
    await client.lpush('order_write_queue', JSON.stringify({
        id: orderId,
        data: orderData,
        timestamp: Date.now()
    }));
    
    return orderId;
}
```

## Conclusion

Advanced database interviews test:

1. **Database Design**: Normalization, denormalization, and schema patterns
2. **Query Optimization**: Performance tuning and execution plan analysis
3. **Indexing Strategies**: Advanced indexing techniques and maintenance
4. **Transaction Management**: ACID properties and concurrency control
5. **Replication and Sharding**: Distributed database systems
6. **Performance Tuning**: Monitoring and optimization techniques
7. **NoSQL Databases**: Document and key-value store operations

Mastering these advanced database concepts demonstrates your readiness for senior engineering roles and complex data management challenges.

## Additional Resources

- [Database Design Patterns](https://www.databasedesignpatterns.com/)
- [Query Optimization](https://www.queryoptimization.com/)
- [Indexing Strategies](https://www.indexingstrategies.com/)
- [Transaction Management](https://www.transactionmanagement.com/)
- [Database Replication](https://www.databasereplication.com/)
- [Performance Tuning](https://www.performance-tuning.com/)
- [NoSQL Databases](https://www.nosqldatabases.com/)
