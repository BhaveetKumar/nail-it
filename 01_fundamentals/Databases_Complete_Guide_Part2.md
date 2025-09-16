# 🗄️ Databases Complete Guide - Part 2

## ⏰ Time Series Databases

### **InfluxDB Deep Dive**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/influxdata/influxdb-client-go/v2"
    "github.com/influxdata/influxdb-client-go/v2/api"
)

// InfluxDBExample demonstrates InfluxDB features
type InfluxDBExample struct {
    client influxdb2.Client
    writeAPI api.WriteAPI
    queryAPI api.QueryAPI
}

// NewInfluxDBExample creates a new InfluxDB example
func NewInfluxDBExample() (*InfluxDBExample, error) {
    client := influxdb2.NewClient("http://localhost:8086", "your-token")
    
    writeAPI := client.WriteAPI("your-org", "your-bucket")
    queryAPI := client.QueryAPI("your-org")
    
    return &InfluxDBExample{
        client: client,
        writeAPI: writeAPI,
        queryAPI: queryAPI,
    }, nil
}

// WriteTimeSeriesData writes time series data
func (ie *InfluxDBExample) WriteTimeSeriesData() error {
    // Create point
    p := influxdb2.NewPointWithMeasurement("cpu_usage").
        AddTag("host", "server1").
        AddTag("region", "us-west").
        AddField("usage", 45.2).
        AddField("temperature", 65.1).
        SetTime(time.Now())
    
    // Write point
    ie.writeAPI.WritePoint(p)
    
    // Create another point
    p2 := influxdb2.NewPointWithMeasurement("memory_usage").
        AddTag("host", "server1").
        AddTag("region", "us-west").
        AddField("usage", 78.5).
        AddField("available", 21.5).
        SetTime(time.Now().Add(time.Minute))
    
    ie.writeAPI.WritePoint(p2)
    
    // Flush writes
    ie.writeAPI.Flush()
    
    return nil
}

// QueryTimeSeriesData queries time series data
func (ie *InfluxDBExample) QueryTimeSeriesData() error {
    query := `from(bucket: "your-bucket")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "cpu_usage")
        |> filter(fn: (r) => r.host == "server1")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        |> yield(name: "mean")`
    
    result, err := ie.queryAPI.Query(context.Background(), query)
    if err != nil {
        return err
    }
    
    fmt.Println("Time Series Query Results:")
    for result.Next() {
        record := result.Record()
        fmt.Printf("Time: %s, Measurement: %s, Field: %s, Value: %v\n",
            record.Time(), record.Measurement(), record.Field(), record.Value())
    }
    
    return nil
}

func main() {
    ie, err := NewInfluxDBExample()
    if err != nil {
        log.Fatal(err)
    }
    defer ie.client.Close()
    
    // Write time series data
    if err := ie.WriteTimeSeriesData(); err != nil {
        log.Fatal(err)
    }
    
    // Query time series data
    if err := ie.QueryTimeSeriesData(); err != nil {
        log.Fatal(err)
    }
}
```

---

## 🏗️ Database Architecture Patterns

### **Database Sharding**

```go
package main

import (
    "crypto/md5"
    "fmt"
    "hash/crc32"
    "sync"
)

// ShardManager manages database shards
type ShardManager struct {
    shards []*Shard
    mutex  sync.RWMutex
}

// Shard represents a database shard
type Shard struct {
    ID     int
    DB     interface{} // Database connection
    mutex  sync.RWMutex
}

// NewShardManager creates a new shard manager
func NewShardManager(shardCount int) *ShardManager {
    shards := make([]*Shard, shardCount)
    for i := 0; i < shardCount; i++ {
        shards[i] = &Shard{
            ID: i,
            DB: nil, // Initialize with actual DB connection
        }
    }
    
    return &ShardManager{shards: shards}
}

// GetShardByHash returns shard based on hash
func (sm *ShardManager) GetShardByHash(key string) *Shard {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    hash := crc32.ChecksumIEEE([]byte(key))
    shardIndex := int(hash) % len(sm.shards)
    return sm.shards[shardIndex]
}

// GetShardByRange returns shard based on range
func (sm *ShardManager) GetShardByRange(key string) *Shard {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    hash := md5.Sum([]byte(key))
    shardIndex := int(hash[0]) % len(sm.shards)
    return sm.shards[shardIndex]
}

// ConsistentHashSharding demonstrates consistent hashing
type ConsistentHashSharding struct {
    ring map[uint32]*Shard
    sortedKeys []uint32
    mutex sync.RWMutex
}

// NewConsistentHashSharding creates consistent hash sharding
func NewConsistentHashSharding(shards []*Shard, replicas int) *ConsistentHashSharding {
    chs := &ConsistentHashSharding{
        ring: make(map[uint32]*Shard),
    }
    
    for _, shard := range shards {
        for i := 0; i < replicas; i++ {
            hash := crc32.ChecksumIEEE([]byte(fmt.Sprintf("%d:%d", shard.ID, i)))
            chs.ring[hash] = shard
        }
    }
    
    // Sort keys
    for hash := range chs.ring {
        chs.sortedKeys = append(chs.sortedKeys, hash)
    }
    
    return chs
}

// GetShard returns shard for key using consistent hashing
func (chs *ConsistentHashSharding) GetShard(key string) *Shard {
    chs.mutex.RLock()
    defer chs.mutex.RUnlock()
    
    hash := crc32.ChecksumIEEE([]byte(key))
    
    // Find first shard with hash >= key hash
    for _, shardHash := range chs.sortedKeys {
        if shardHash >= hash {
            return chs.ring[shardHash]
        }
    }
    
    // Wrap around to first shard
    return chs.ring[chs.sortedKeys[0]]
}

func main() {
    // Create shards
    shards := make([]*Shard, 3)
    for i := 0; i < 3; i++ {
        shards[i] = &Shard{ID: i}
    }
    
    // Test hash-based sharding
    sm := NewShardManager(3)
    fmt.Println("Hash-based Sharding:")
    for i := 0; i < 10; i++ {
        key := fmt.Sprintf("user%d", i)
        shard := sm.GetShardByHash(key)
        fmt.Printf("Key: %s -> Shard: %d\n", key, shard.ID)
    }
    
    // Test consistent hashing
    chs := NewConsistentHashSharding(shards, 3)
    fmt.Println("\nConsistent Hashing:")
    for i := 0; i < 10; i++ {
        key := fmt.Sprintf("user%d", i)
        shard := chs.GetShard(key)
        fmt.Printf("Key: %s -> Shard: %d\n", key, shard.ID)
    }
}
```

---

## ⚡ Performance Optimization

### **Query Optimization**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/go-sql-driver/mysql"
)

// QueryOptimizer demonstrates query optimization techniques
type QueryOptimizer struct {
    db *sql.DB
}

// NewQueryOptimizer creates a new query optimizer
func NewQueryOptimizer() (*QueryOptimizer, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    return &QueryOptimizer{db: db}, nil
}

// CreateOptimizedTable creates a table with proper indexes
func (qo *QueryOptimizer) CreateOptimizedTable() error {
    createTableSQL := `
    CREATE TABLE IF NOT EXISTS orders (
        id INT PRIMARY KEY AUTO_INCREMENT,
        customer_id INT NOT NULL,
        product_id INT NOT NULL,
        order_date DATE NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        status ENUM('pending', 'completed', 'cancelled') NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        INDEX idx_customer_id (customer_id),
        INDEX idx_product_id (product_id),
        INDEX idx_order_date (order_date),
        INDEX idx_status (status),
        INDEX idx_customer_date (customer_id, order_date),
        INDEX idx_status_date (status, order_date)
    ) ENGINE=InnoDB`
    
    _, err := qo.db.Exec(createTableSQL)
    return err
}

// OptimizedQuery demonstrates optimized query
func (qo *QueryOptimizer) OptimizedQuery() error {
    start := time.Now()
    
    // Use proper indexes and avoid SELECT *
    query := `
    SELECT 
        o.id,
        o.customer_id,
        o.amount,
        o.order_date
    FROM orders o
    WHERE o.customer_id = ? 
        AND o.order_date >= ?
        AND o.status = 'completed'
    ORDER BY o.order_date DESC
    LIMIT 10`
    
    rows, err := qo.db.Query(query, 123, "2024-01-01")
    if err != nil {
        return err
    }
    defer rows.Close()
    
    var count int
    for rows.Next() {
        count++
    }
    
    duration := time.Since(start)
    fmt.Printf("Optimized query took: %v, returned %d rows\n", duration, count)
    
    return nil
}

// BatchInsert demonstrates batch insert optimization
func (qo *QueryOptimizer) BatchInsert() error {
    start := time.Now()
    
    // Prepare statement for batch insert
    stmt, err := qo.db.Prepare(`
        INSERT INTO orders (customer_id, product_id, order_date, amount, status) 
        VALUES (?, ?, ?, ?, ?)`)
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    // Batch insert
    for i := 0; i < 1000; i++ {
        _, err = stmt.Exec(
            100+i%10,           // customer_id
            200+i%20,           // product_id
            "2024-01-01",       // order_date
            50.0+float64(i%100), // amount
            "completed",        // status
        )
        if err != nil {
            return err
        }
    }
    
    duration := time.Since(start)
    fmt.Printf("Batch insert took: %v\n", duration)
    
    return nil
}

// ConnectionPooling demonstrates connection pooling
func (qo *QueryOptimizer) ConnectionPooling() error {
    // Configure connection pool
    qo.db.SetMaxOpenConns(25)
    qo.db.SetMaxIdleConns(5)
    qo.db.SetConnMaxLifetime(5 * time.Minute)
    
    // Test concurrent queries
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            
            start := time.Now()
            _, err := qo.db.Query("SELECT COUNT(*) FROM orders WHERE customer_id = ?", id)
            if err != nil {
                fmt.Printf("Query %d failed: %v\n", id, err)
                return
            }
            
            duration := time.Since(start)
            fmt.Printf("Query %d took: %v\n", id, duration)
        }(i)
    }
    
    wg.Wait()
    return nil
}

func main() {
    qo, err := NewQueryOptimizer()
    if err != nil {
        log.Fatal(err)
    }
    defer qo.db.Close()
    
    // Create optimized table
    if err := qo.CreateOptimizedTable(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate optimized query
    if err := qo.OptimizedQuery(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate batch insert
    if err := qo.BatchInsert(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate connection pooling
    if err := qo.ConnectionPooling(); err != nil {
        log.Fatal(err)
    }
}
```

---

## 📈 Scaling Strategies

### **Read Replicas**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "sync"
    _ "github.com/go-sql-driver/mysql"
)

// ReadReplicaManager manages read replicas
type ReadReplicaManager struct {
    master *sql.DB
    replicas []*sql.DB
    currentReplica int
    mutex sync.RWMutex
}

// NewReadReplicaManager creates a new read replica manager
func NewReadReplicaManager() (*ReadReplicaManager, error) {
    // Master database
    master, err := sql.Open("mysql", "user:password@tcp(master:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    // Read replicas
    replica1, err := sql.Open("mysql", "user:password@tcp(replica1:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    replica2, err := sql.Open("mysql", "user:password@tcp(replica2:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    return &ReadReplicaManager{
        master: master,
        replicas: []*sql.DB{replica1, replica2},
        currentReplica: 0,
    }, nil
}

// Write executes write operations on master
func (rrm *ReadReplicaManager) Write(query string, args ...interface{}) (sql.Result, error) {
    return rrm.master.Exec(query, args...)
}

// Read executes read operations on replicas
func (rrm *ReadReplicaManager) Read(query string, args ...interface{}) (*sql.Rows, error) {
    rrm.mutex.Lock()
    replica := rrm.replicas[rrm.currentReplica]
    rrm.currentReplica = (rrm.currentReplica + 1) % len(rrm.replicas)
    rrm.mutex.Unlock()
    
    return replica.Query(query, args...)
}

// ReadWithFallback executes read with fallback to master
func (rrm *ReadReplicaManager) ReadWithFallback(query string, args ...interface{}) (*sql.Rows, error) {
    // Try replica first
    rows, err := rrm.Read(query, args...)
    if err != nil {
        // Fallback to master
        return rrm.master.Query(query, args...)
    }
    
    return rows, nil
}

func main() {
    rrm, err := NewReadReplicaManager()
    if err != nil {
        log.Fatal(err)
    }
    defer rrm.master.Close()
    for _, replica := range rrm.replicas {
        defer replica.Close()
    }
    
    // Write to master
    _, err = rrm.Write("INSERT INTO users (name, email) VALUES (?, ?)", "John Doe", "john@example.com")
    if err != nil {
        log.Fatal(err)
    }
    
    // Read from replica
    rows, err := rrm.Read("SELECT * FROM users WHERE name = ?", "John Doe")
    if err != nil {
        log.Fatal(err)
    }
    defer rows.Close()
    
    for rows.Next() {
        var id int
        var name, email string
        err := rows.Scan(&id, &name, &email)
        if err != nil {
            log.Fatal(err)
        }
        fmt.Printf("ID: %d, Name: %s, Email: %s\n", id, name, email)
    }
}
```

---

## 🎯 Interview Questions & Solutions

### **Database Design Questions**

#### **1. Design a Database for a Social Media Platform**

**Question**: Design a database schema for a social media platform like Facebook with users, posts, comments, likes, and friendships.

**Solution**:

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/go-sql-driver/mysql"
)

// SocialMediaDB represents the social media database
type SocialMediaDB struct {
    db *sql.DB
}

// NewSocialMediaDB creates a new social media database
func NewSocialMediaDB() (*SocialMediaDB, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/socialmedia")
    if err != nil {
        return nil, err
    }
    
    return &SocialMediaDB{db: db}, nil
}

// CreateSchema creates the database schema
func (smdb *SocialMediaDB) CreateSchema() error {
    // Users table
    usersTable := `
    CREATE TABLE IF NOT EXISTS users (
        id INT PRIMARY KEY AUTO_INCREMENT,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        date_of_birth DATE,
        profile_picture_url VARCHAR(255),
        bio TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        INDEX idx_username (username),
        INDEX idx_email (email),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB`
    
    // Posts table
    postsTable := `
    CREATE TABLE IF NOT EXISTS posts (
        id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        content TEXT NOT NULL,
        image_url VARCHAR(255),
        video_url VARCHAR(255),
        privacy ENUM('public', 'friends', 'private') DEFAULT 'public',
        is_deleted BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_user_id (user_id),
        INDEX idx_created_at (created_at),
        INDEX idx_privacy (privacy),
        INDEX idx_user_created (user_id, created_at)
    ) ENGINE=InnoDB`
    
    // Comments table
    commentsTable := `
    CREATE TABLE IF NOT EXISTS comments (
        id INT PRIMARY KEY AUTO_INCREMENT,
        post_id INT NOT NULL,
        user_id INT NOT NULL,
        parent_comment_id INT NULL,
        content TEXT NOT NULL,
        is_deleted BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_comment_id) REFERENCES comments(id) ON DELETE CASCADE,
        INDEX idx_post_id (post_id),
        INDEX idx_user_id (user_id),
        INDEX idx_parent_comment_id (parent_comment_id),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB`
    
    // Likes table
    likesTable := `
    CREATE TABLE IF NOT EXISTS likes (
        id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        post_id INT NULL,
        comment_id INT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE,
        UNIQUE KEY unique_post_like (user_id, post_id),
        UNIQUE KEY unique_comment_like (user_id, comment_id),
        INDEX idx_user_id (user_id),
        INDEX idx_post_id (post_id),
        INDEX idx_comment_id (comment_id)
    ) ENGINE=InnoDB`
    
    // Friendships table
    friendshipsTable := `
    CREATE TABLE IF NOT EXISTS friendships (
        id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        friend_id INT NOT NULL,
        status ENUM('pending', 'accepted', 'blocked') DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (friend_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE KEY unique_friendship (user_id, friend_id),
        INDEX idx_user_id (user_id),
        INDEX idx_friend_id (friend_id),
        INDEX idx_status (status)
    ) ENGINE=InnoDB`
    
    // Execute table creation
    tables := []string{usersTable, postsTable, commentsTable, likesTable, friendshipsTable}
    for _, table := range tables {
        _, err := smdb.db.Exec(table)
        if err != nil {
            return err
        }
    }
    
    return nil
}

// GetUserFeed retrieves user's news feed
func (smdb *SocialMediaDB) GetUserFeed(userID int, limit int) error {
    query := `
    SELECT 
        p.id,
        p.user_id,
        p.content,
        p.image_url,
        p.created_at,
        u.username,
        u.first_name,
        u.last_name,
        u.profile_picture_url,
        COUNT(DISTINCT l.id) as like_count,
        COUNT(DISTINCT c.id) as comment_count
    FROM posts p
    JOIN users u ON p.user_id = u.id
    LEFT JOIN likes l ON p.id = l.post_id
    LEFT JOIN comments c ON p.id = c.post_id
    WHERE p.user_id IN (
        SELECT friend_id FROM friendships 
        WHERE user_id = ? AND status = 'accepted'
        UNION
        SELECT user_id FROM friendships 
        WHERE friend_id = ? AND status = 'accepted'
        UNION
        SELECT ? -- Include user's own posts
    )
    AND p.is_deleted = FALSE
    AND p.privacy IN ('public', 'friends')
    GROUP BY p.id
    ORDER BY p.created_at DESC
    LIMIT ?`
    
    rows, err := smdb.db.Query(query, userID, userID, userID, limit)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("User Feed:")
    for rows.Next() {
        var postID, postUserID, likeCount, commentCount int
        var content, imageURL, username, firstName, lastName, profilePictureURL string
        var createdAt time.Time
        
        err := rows.Scan(&postID, &postUserID, &content, &imageURL, &createdAt,
            &username, &firstName, &lastName, &profilePictureURL, &likeCount, &commentCount)
        if err != nil {
            return err
        }
        
        fmt.Printf("Post ID: %d, Author: %s %s, Content: %s, Likes: %d, Comments: %d\n",
            postID, firstName, lastName, content, likeCount, commentCount)
    }
    
    return nil
}

// GetPostComments retrieves comments for a post
func (smdb *SocialMediaDB) GetPostComments(postID int, limit int) error {
    query := `
    SELECT 
        c.id,
        c.user_id,
        c.content,
        c.parent_comment_id,
        c.created_at,
        u.username,
        u.first_name,
        u.last_name,
        u.profile_picture_url,
        COUNT(DISTINCT l.id) as like_count
    FROM comments c
    JOIN users u ON c.user_id = u.id
    LEFT JOIN likes l ON c.id = l.comment_id
    WHERE c.post_id = ? AND c.is_deleted = FALSE
    GROUP BY c.id
    ORDER BY c.created_at ASC
    LIMIT ?`
    
    rows, err := smdb.db.Query(query, postID, limit)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Post Comments:")
    for rows.Next() {
        var commentID, commentUserID, likeCount int
        var content, username, firstName, lastName, profilePictureURL string
        var parentCommentID sql.NullInt64
        var createdAt time.Time
        
        err := rows.Scan(&commentID, &commentUserID, &content, &parentCommentID, &createdAt,
            &username, &firstName, &lastName, &profilePictureURL, &likeCount)
        if err != nil {
            return err
        }
        
        fmt.Printf("Comment ID: %d, Author: %s %s, Content: %s, Likes: %d\n",
            commentID, firstName, lastName, content, likeCount)
    }
    
    return nil
}

func main() {
    smdb, err := NewSocialMediaDB()
    if err != nil {
        log.Fatal(err)
    }
    defer smdb.db.Close()
    
    // Create schema
    if err := smdb.CreateSchema(); err != nil {
        log.Fatal(err)
    }
    
    // Get user feed
    if err := smdb.GetUserFeed(1, 10); err != nil {
        log.Fatal(err)
    }
    
    // Get post comments
    if err := smdb.GetPostComments(1, 20); err != nil {
        log.Fatal(err)
    }
}
```

#### **2. Design a Database for an E-commerce Platform**

**Question**: Design a database schema for an e-commerce platform with products, categories, orders, customers, and inventory management.

**Solution**:

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/go-sql-driver/mysql"
)

// EcommerceDB represents the e-commerce database
type EcommerceDB struct {
    db *sql.DB
}

// NewEcommerceDB creates a new e-commerce database
func NewEcommerceDB() (*EcommerceDB, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/ecommerce")
    if err != nil {
        return nil, err
    }
    
    return &EcommerceDB{db: db}, nil
}

// CreateSchema creates the e-commerce database schema
func (edb *EcommerceDB) CreateSchema() error {
    // Categories table
    categoriesTable := `
    CREATE TABLE IF NOT EXISTS categories (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        parent_id INT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (parent_id) REFERENCES categories(id) ON DELETE SET NULL,
        INDEX idx_name (name),
        INDEX idx_parent_id (parent_id),
        INDEX idx_is_active (is_active)
    ) ENGINE=InnoDB`
    
    // Products table
    productsTable := `
    CREATE TABLE IF NOT EXISTS products (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        sku VARCHAR(100) UNIQUE NOT NULL,
        price DECIMAL(10,2) NOT NULL,
        compare_price DECIMAL(10,2),
        cost_price DECIMAL(10,2),
        category_id INT NOT NULL,
        brand VARCHAR(100),
        weight DECIMAL(8,2),
        dimensions VARCHAR(100),
        is_active BOOLEAN DEFAULT TRUE,
        is_digital BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT,
        INDEX idx_name (name),
        INDEX idx_sku (sku),
        INDEX idx_category_id (category_id),
        INDEX idx_price (price),
        INDEX idx_is_active (is_active),
        FULLTEXT idx_search (name, description)
    ) ENGINE=InnoDB`
    
    // Customers table
    customersTable := `
    CREATE TABLE IF NOT EXISTS customers (
        id INT PRIMARY KEY AUTO_INCREMENT,
        email VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        phone VARCHAR(20),
        date_of_birth DATE,
        is_active BOOLEAN DEFAULT TRUE,
        email_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        INDEX idx_email (email),
        INDEX idx_is_active (is_active)
    ) ENGINE=InnoDB`
    
    // Orders table
    ordersTable := `
    CREATE TABLE IF NOT EXISTS orders (
        id INT PRIMARY KEY AUTO_INCREMENT,
        order_number VARCHAR(50) UNIQUE NOT NULL,
        customer_id INT NOT NULL,
        status ENUM('pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'refunded') DEFAULT 'pending',
        subtotal DECIMAL(10,2) NOT NULL,
        tax_amount DECIMAL(10,2) DEFAULT 0,
        shipping_amount DECIMAL(10,2) DEFAULT 0,
        discount_amount DECIMAL(10,2) DEFAULT 0,
        total_amount DECIMAL(10,2) NOT NULL,
        payment_status ENUM('pending', 'paid', 'failed', 'refunded') DEFAULT 'pending',
        payment_method VARCHAR(50),
        shipping_address JSON,
        billing_address JSON,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE RESTRICT,
        INDEX idx_order_number (order_number),
        INDEX idx_customer_id (customer_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB`
    
    // Order items table
    orderItemsTable := `
    CREATE TABLE IF NOT EXISTS order_items (
        id INT PRIMARY KEY AUTO_INCREMENT,
        order_id INT NOT NULL,
        product_id INT NOT NULL,
        quantity INT NOT NULL,
        unit_price DECIMAL(10,2) NOT NULL,
        total_price DECIMAL(10,2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE RESTRICT,
        INDEX idx_order_id (order_id),
        INDEX idx_product_id (product_id)
    ) ENGINE=InnoDB`
    
    // Inventory table
    inventoryTable := `
    CREATE TABLE IF NOT EXISTS inventory (
        id INT PRIMARY KEY AUTO_INCREMENT,
        product_id INT NOT NULL,
        quantity INT NOT NULL DEFAULT 0,
        reserved_quantity INT NOT NULL DEFAULT 0,
        available_quantity INT GENERATED ALWAYS AS (quantity - reserved_quantity) STORED,
        low_stock_threshold INT DEFAULT 10,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
        UNIQUE KEY unique_product (product_id),
        INDEX idx_available_quantity (available_quantity)
    ) ENGINE=InnoDB`
    
    // Execute table creation
    tables := []string{categoriesTable, productsTable, customersTable, ordersTable, orderItemsTable, inventoryTable}
    for _, table := range tables {
        _, err := edb.db.Exec(table)
        if err != nil {
            return err
        }
    }
    
    return nil
}

// GetProductCatalog retrieves products with pagination and filtering
func (edb *EcommerceDB) GetProductCatalog(categoryID int, page, limit int) error {
    offset := (page - 1) * limit
    
    query := `
    SELECT 
        p.id,
        p.name,
        p.description,
        p.sku,
        p.price,
        p.compare_price,
        p.brand,
        c.name as category_name,
        i.available_quantity,
        CASE 
            WHEN i.available_quantity <= i.low_stock_threshold THEN 'low_stock'
            WHEN i.available_quantity = 0 THEN 'out_of_stock'
            ELSE 'in_stock'
        END as stock_status
    FROM products p
    JOIN categories c ON p.category_id = c.id
    LEFT JOIN inventory i ON p.id = i.product_id
    WHERE p.is_active = TRUE
        AND c.is_active = TRUE
        AND (? = 0 OR p.category_id = ?)
    ORDER BY p.created_at DESC
    LIMIT ? OFFSET ?`
    
    rows, err := edb.db.Query(query, categoryID, categoryID, limit, offset)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Product Catalog:")
    for rows.Next() {
        var productID int
        var name, description, sku, brand, categoryName, stockStatus string
        var price, comparePrice sql.NullFloat64
        var availableQuantity sql.NullInt64
        
        err := rows.Scan(&productID, &name, &description, &sku, &price, &comparePrice,
            &brand, &categoryName, &availableQuantity, &stockStatus)
        if err != nil {
            return err
        }
        
        fmt.Printf("Product ID: %d, Name: %s, SKU: %s, Price: %.2f, Stock: %s\n",
            productID, name, sku, price.Float64, stockStatus)
    }
    
    return nil
}

// GetOrderSummary retrieves order summary with items
func (edb *EcommerceDB) GetOrderSummary(orderID int) error {
    query := `
    SELECT 
        o.id,
        o.order_number,
        o.status,
        o.total_amount,
        o.payment_status,
        c.first_name,
        c.last_name,
        c.email,
        o.created_at
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    WHERE o.id = ?`
    
    var orderNumber, status, paymentStatus, firstName, lastName, email string
    var totalAmount float64
    var createdAt time.Time
    
    err := edb.db.QueryRow(query, orderID).Scan(
        &orderID, &orderNumber, &status, &totalAmount, &paymentStatus,
        &firstName, &lastName, &email, &createdAt)
    if err != nil {
        return err
    }
    
    fmt.Printf("Order: %s, Customer: %s %s, Total: %.2f, Status: %s\n",
        orderNumber, firstName, lastName, totalAmount, status)
    
    // Get order items
    itemsQuery := `
    SELECT 
        oi.id,
        p.name,
        p.sku,
        oi.quantity,
        oi.unit_price,
        oi.total_price
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = ?`
    
    rows, err := edb.db.Query(itemsQuery, orderID)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Order Items:")
    for rows.Next() {
        var itemID, quantity int
        var productName, sku string
        var unitPrice, totalPrice float64
        
        err := rows.Scan(&itemID, &productName, &sku, &quantity, &unitPrice, &totalPrice)
        if err != nil {
            return err
        }
        
        fmt.Printf("  Item: %s (%s), Qty: %d, Price: %.2f, Total: %.2f\n",
            productName, sku, quantity, unitPrice, totalPrice)
    }
    
    return nil
}

func main() {
    edb, err := NewEcommerceDB()
    if err != nil {
        log.Fatal(err)
    }
    defer edb.db.Close()
    
    // Create schema
    if err := edb.CreateSchema(); err != nil {
        log.Fatal(err)
    }
    
    // Get product catalog
    if err := edb.GetProductCatalog(0, 1, 10); err != nil {
        log.Fatal(err)
    }
    
    // Get order summary
    if err := edb.GetOrderSummary(1); err != nil {
        log.Fatal(err)
    }
}
```

---

## 📚 Additional Resources

### **Books**
- [Database System Concepts](https://www.db-book.com/) - Silberschatz, Korth, Sudarshan
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [High Performance MySQL](https://www.oreilly.com/library/view/high-performance-mysql/9780596101718/) - Baron Schwartz

### **Online Resources**
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Redis Documentation](https://redis.io/documentation/)

### **Video Resources**
- [Database Design Course](https://www.youtube.com/watch?v=ztHopE5Wnpc/) - freeCodeCamp
- [SQL Tutorial](https://www.youtube.com/watch?v=HXV3zeQKqGY/) - freeCodeCamp
- [NoSQL Databases](https://www.youtube.com/watch?v=uD3p_rZPBcQ/) - freeCodeCamp

---

*This comprehensive guide covers all major database concepts, from fundamentals to advanced topics, with practical Go examples and real-world interview questions. Each section includes detailed explanations, code examples, and best practices for database design and optimization.*
