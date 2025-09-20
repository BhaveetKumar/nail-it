# 🗄️ Databases Complete Guide - From Fundamentals to Advanced

## 📋 Table of Contents

1. [Database Fundamentals](#database-fundamentals)
2. [Relational Databases (MySQL, PostgreSQL)](#relational-databases)
3. [NoSQL Databases (MongoDB, DynamoDB)](#nosql-databases)
4. [In-Memory Databases (Redis)](#in-memory-databases)
5. [Search Engines (Elasticsearch)](#search-engines)
6. [Vector Databases](#vector-databases)
7. [Data Warehouses (Snowflake)](#data-warehouses)
8. [Time Series Databases](#time-series-databases)
9. [Database Architecture Patterns](#database-architecture-patterns)
10. [Performance Optimization](#performance-optimization)
11. [Scaling Strategies](#scaling-strategies)
12. [Interview Questions & Solutions](#interview-questions--solutions)

---

## 🏗️ Database Fundamentals

### **ACID Properties**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    _ "github.com/go-sql-driver/mysql"
)

// ACIDExample demonstrates ACID properties
type ACIDExample struct {
    db *sql.DB
}

// NewACIDExample creates a new ACID example
func NewACIDExample() (*ACIDExample, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    return &ACIDExample{db: db}, nil
}

// AtomicityExample demonstrates atomicity
func (ae *ACIDExample) AtomicityExample() error {
    // Start transaction
    tx, err := ae.db.Begin()
    if err != nil {
        return err
    }
    
    // Defer rollback in case of error
    defer func() {
        if err != nil {
            tx.Rollback()
        }
    }()
    
    // Execute multiple operations
    _, err = tx.Exec("INSERT INTO accounts (id, balance) VALUES (?, ?)", 1, 1000)
    if err != nil {
        return err
    }
    
    _, err = tx.Exec("INSERT INTO accounts (id, balance) VALUES (?, ?)", 2, 500)
    if err != nil {
        return err
    }
    
    // Transfer money
    _, err = tx.Exec("UPDATE accounts SET balance = balance - ? WHERE id = ?", 100, 1)
    if err != nil {
        return err
    }
    
    _, err = tx.Exec("UPDATE accounts SET balance = balance + ? WHERE id = ?", 100, 2)
    if err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}

// ConsistencyExample demonstrates consistency
func (ae *ACIDExample) ConsistencyExample() error {
    // Check constraints before operation
    var count int
    err := ae.db.QueryRow("SELECT COUNT(*) FROM accounts WHERE balance < 0").Scan(&count)
    if err != nil {
        return err
    }
    
    if count > 0 {
        return fmt.Errorf("consistency violation: negative balance detected")
    }
    
    return nil
}

// IsolationExample demonstrates isolation
func (ae *ACIDExample) IsolationExample() error {
    // Set isolation level
    _, err := ae.db.Exec("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    if err != nil {
        return err
    }
    
    // Start transaction
    tx, err := ae.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    // Read data
    var balance int
    err = tx.QueryRow("SELECT balance FROM accounts WHERE id = ?", 1).Scan(&balance)
    if err != nil {
        return err
    }
    
    // Update data
    _, err = tx.Exec("UPDATE accounts SET balance = ? WHERE id = ?", balance+100, 1)
    if err != nil {
        return err
    }
    
    return tx.Commit()
}

// DurabilityExample demonstrates durability
func (ae *ACIDExample) DurabilityExample() error {
    // Ensure WAL (Write-Ahead Logging) is enabled
    _, err := ae.db.Exec("SET innodb_flush_log_at_trx_commit = 1")
    if err != nil {
        return err
    }
    
    // Perform transaction
    tx, err := ae.db.Begin()
    if err != nil {
        return err
    }
    
    _, err = tx.Exec("INSERT INTO accounts (id, balance) VALUES (?, ?)", 3, 2000)
    if err != nil {
        tx.Rollback()
        return err
    }
    
    // Commit ensures data is written to disk
    return tx.Commit()
}

func main() {
    acid, err := NewACIDExample()
    if err != nil {
        log.Fatal(err)
    }
    defer acid.db.Close()
    
    // Demonstrate ACID properties
    fmt.Println("Demonstrating ACID Properties:")
    
    if err := acid.AtomicityExample(); err != nil {
        fmt.Printf("Atomicity error: %v\n", err)
    } else {
        fmt.Println("✓ Atomicity: Transaction completed successfully")
    }
    
    if err := acid.ConsistencyExample(); err != nil {
        fmt.Printf("Consistency error: %v\n", err)
    } else {
        fmt.Println("✓ Consistency: Data integrity maintained")
    }
    
    if err := acid.IsolationExample(); err != nil {
        fmt.Printf("Isolation error: %v\n", err)
    } else {
        fmt.Println("✓ Isolation: Concurrent transactions handled properly")
    }
    
    if err := acid.DurabilityExample(); err != nil {
        fmt.Printf("Durability error: %v\n", err)
    } else {
        fmt.Println("✓ Durability: Data persisted to disk")
    }
}
```

**ACID Properties Explained:**

- **Atomicity**: All operations in a transaction succeed or all fail
- **Consistency**: Database remains in valid state after transaction
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed data survives system failures

### **Database Indexing**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/go-sql-driver/mysql"
)

// IndexExample demonstrates database indexing
type IndexExample struct {
    db *sql.DB
}

// NewIndexExample creates a new index example
func NewIndexExample() (*IndexExample, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    return &IndexExample{db: db}, nil
}

// CreateTableWithIndexes creates a table with various indexes
func (ie *IndexExample) CreateTableWithIndexes() error {
    // Create users table
    createTableSQL := `
    CREATE TABLE IF NOT EXISTS users (
        id INT PRIMARY KEY AUTO_INCREMENT,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        age INT,
        city VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_email (email),
        INDEX idx_name (name),
        INDEX idx_age (age),
        INDEX idx_city (city),
        INDEX idx_created_at (created_at),
        INDEX idx_composite (city, age)
    )`
    
    _, err := ie.db.Exec(createTableSQL)
    return err
}

// InsertSampleData inserts sample data
func (ie *IndexExample) InsertSampleData() error {
    cities := []string{"New York", "London", "Tokyo", "Paris", "Sydney"}
    
    for i := 0; i < 10000; i++ {
        email := fmt.Sprintf("user%d@example.com", i)
        name := fmt.Sprintf("User %d", i)
        age := 20 + (i % 50)
        city := cities[i%len(cities)]
        
        _, err := ie.db.Exec(
            "INSERT INTO users (email, name, age, city) VALUES (?, ?, ?, ?)",
            email, name, age, city,
        )
        if err != nil {
            return err
        }
    }
    
    return nil
}

// QueryWithIndex demonstrates index usage
func (ie *IndexExample) QueryWithIndex() error {
    start := time.Now()
    
    // Query using indexed column
    rows, err := ie.db.Query("SELECT * FROM users WHERE email = ?", "user5000@example.com")
    if err != nil {
        return err
    }
    defer rows.Close()
    
    var count int
    for rows.Next() {
        count++
    }
    
    duration := time.Since(start)
    fmt.Printf("Indexed query took: %v, found %d rows\n", duration, count)
    
    return nil
}

// QueryWithoutIndex demonstrates query without index
func (ie *IndexExample) QueryWithoutIndex() error {
    start := time.Now()
    
    // Query using non-indexed column (assuming we remove the index)
    rows, err := ie.db.Query("SELECT * FROM users WHERE name LIKE ?", "%User 5000%")
    if err != nil {
        return err
    }
    defer rows.Close()
    
    var count int
    for rows.Next() {
        count++
    }
    
    duration := time.Since(start)
    fmt.Printf("Non-indexed query took: %v, found %d rows\n", duration, count)
    
    return nil
}

// ExplainQuery shows query execution plan
func (ie *IndexExample) ExplainQuery() error {
    rows, err := ie.db.Query("EXPLAIN SELECT * FROM users WHERE city = ? AND age > ?", "New York", 30)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Query Execution Plan:")
    fmt.Println("id | select_type | table | type | possible_keys | key | key_len | ref | rows | Extra")
    fmt.Println("---|-------------|-------|------|---------------|-----|---------|-----|------|------")
    
    for rows.Next() {
        var id, selectType, table, queryType, possibleKeys, key, keyLen, ref, rowsCount, extra string
        err := rows.Scan(&id, &selectType, &table, &queryType, &possibleKeys, &key, &keyLen, &ref, &rowsCount, &extra)
        if err != nil {
            return err
        }
        fmt.Printf("%s | %s | %s | %s | %s | %s | %s | %s | %s | %s\n",
            id, selectType, table, queryType, possibleKeys, key, keyLen, ref, rowsCount, extra)
    }
    
    return nil
}

func main() {
    ie, err := NewIndexExample()
    if err != nil {
        log.Fatal(err)
    }
    defer ie.db.Close()
    
    // Create table with indexes
    if err := ie.CreateTableWithIndexes(); err != nil {
        log.Fatal(err)
    }
    
    // Insert sample data
    if err := ie.InsertSampleData(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate index performance
    fmt.Println("Database Indexing Examples:")
    
    if err := ie.QueryWithIndex(); err != nil {
        log.Fatal(err)
    }
    
    if err := ie.QueryWithoutIndex(); err != nil {
        log.Fatal(err)
    }
    
    if err := ie.ExplainQuery(); err != nil {
        log.Fatal(err)
    }
}
```

**Index Types Explained:**

- **Primary Index**: Automatically created for primary key
- **Unique Index**: Ensures uniqueness, automatically created for unique constraints
- **Composite Index**: Index on multiple columns
- **Partial Index**: Index on subset of rows
- **Covering Index**: Index that contains all columns needed for query

---

## 🗃️ Relational Databases

### **MySQL Deep Dive**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/go-sql-driver/mysql"
)

// MySQLExample demonstrates MySQL features
type MySQLExample struct {
    db *sql.DB
}

// NewMySQLExample creates a new MySQL example
func NewMySQLExample() (*MySQLExample, error) {
    db, err := sql.Open("mysql", "user:password@tcp(localhost:3306)/testdb")
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    return &MySQLExample{db: db}, nil
}

// CreateAdvancedTable creates a table with advanced MySQL features
func (me *MySQLExample) CreateAdvancedTable() error {
    createTableSQL := `
    CREATE TABLE IF NOT EXISTS products (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        price DECIMAL(10,2) NOT NULL,
        category_id INT,
        stock_quantity INT DEFAULT 0,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        INDEX idx_name (name),
        INDEX idx_category (category_id),
        INDEX idx_price (price),
        INDEX idx_active (is_active),
        INDEX idx_created_at (created_at),
        
        CONSTRAINT chk_price CHECK (price > 0),
        CONSTRAINT chk_stock CHECK (stock_quantity >= 0)
    ) ENGINE=InnoDB`
    
    _, err := me.db.Exec(createTableSQL)
    return err
}

// ComplexQuery demonstrates complex MySQL queries
func (me *MySQLExample) ComplexQuery() error {
    // Window functions
    windowQuery := `
    SELECT 
        name,
        price,
        category_id,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY price DESC) as price_rank,
        RANK() OVER (ORDER BY price DESC) as overall_rank,
        LAG(price, 1) OVER (ORDER BY price) as prev_price,
        LEAD(price, 1) OVER (ORDER BY price) as next_price
    FROM products 
    WHERE is_active = TRUE
    ORDER BY price DESC`
    
    rows, err := me.db.Query(windowQuery)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Window Functions Results:")
    fmt.Println("Name | Price | Category | Price Rank | Overall Rank | Prev Price | Next Price")
    fmt.Println("-----|-------|----------|------------|--------------|------------|-----------")
    
    for rows.Next() {
        var name string
        var price float64
        var categoryID sql.NullInt64
        var priceRank, overallRank int
        var prevPrice, nextPrice sql.NullFloat64
        
        err := rows.Scan(&name, &price, &categoryID, &priceRank, &overallRank, &prevPrice, &nextPrice)
        if err != nil {
            return err
        }
        
        fmt.Printf("%s | %.2f | %v | %d | %d | %v | %v\n",
            name, price, categoryID, priceRank, overallRank, prevPrice, nextPrice)
    }
    
    return nil
}

// StoredProcedureExample demonstrates stored procedures
func (me *MySQLExample) StoredProcedureExample() error {
    // Create stored procedure
    createProcSQL := `
    DELIMITER //
    CREATE PROCEDURE GetProductsByCategory(IN category_id INT)
    BEGIN
        SELECT 
            p.id,
            p.name,
            p.price,
            p.stock_quantity,
            CASE 
                WHEN p.stock_quantity = 0 THEN 'Out of Stock'
                WHEN p.stock_quantity < 10 THEN 'Low Stock'
                ELSE 'In Stock'
            END as stock_status
        FROM products p
        WHERE p.category_id = category_id AND p.is_active = TRUE
        ORDER BY p.price DESC;
    END //
    DELIMITER ;`
    
    _, err := me.db.Exec(createProcSQL)
    if err != nil {
        return err
    }
    
    // Call stored procedure
    rows, err := me.db.Query("CALL GetProductsByCategory(?)", 1)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Stored Procedure Results:")
    for rows.Next() {
        var id int
        var name string
        var price float64
        var stockQuantity int
        var stockStatus string
        
        err := rows.Scan(&id, &name, &price, &stockQuantity, &stockStatus)
        if err != nil {
            return err
        }
        
        fmt.Printf("ID: %d, Name: %s, Price: %.2f, Stock: %d, Status: %s\n",
            id, name, price, stockQuantity, stockStatus)
    }
    
    return nil
}

// TransactionExample demonstrates MySQL transactions
func (me *MySQLExample) TransactionExample() error {
    tx, err := me.db.Begin()
    if err != nil {
        return err
    }
    
    defer func() {
        if err != nil {
            tx.Rollback()
        }
    }()
    
    // Insert product
    result, err := tx.Exec(
        "INSERT INTO products (name, description, price, category_id, stock_quantity) VALUES (?, ?, ?, ?, ?)",
        "New Product", "A great product", 99.99, 1, 100,
    )
    if err != nil {
        return err
    }
    
    productID, err := result.LastInsertId()
    if err != nil {
        return err
    }
    
    // Update stock
    _, err = tx.Exec("UPDATE products SET stock_quantity = stock_quantity - ? WHERE id = ?", 10, productID)
    if err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}

func main() {
    me, err := NewMySQLExample()
    if err != nil {
        log.Fatal(err)
    }
    defer me.db.Close()
    
    // Create advanced table
    if err := me.CreateAdvancedTable(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate complex queries
    if err := me.ComplexQuery(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate stored procedures
    if err := me.StoredProcedureExample(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate transactions
    if err := me.TransactionExample(); err != nil {
        log.Fatal(err)
    }
}
```

### **PostgreSQL Deep Dive**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    _ "github.com/lib/pq"
)

// PostgreSQLExample demonstrates PostgreSQL features
type PostgreSQLExample struct {
    db *sql.DB
}

// NewPostgreSQLExample creates a new PostgreSQL example
func NewPostgreSQLExample() (*PostgreSQLExample, error) {
    db, err := sql.Open("postgres", "user=postgres password=password dbname=testdb sslmode=disable")
    if err != nil {
        return nil, err
    }
    
    return &PostgreSQLExample{db: db}, nil
}

// CreateAdvancedTable creates a table with advanced PostgreSQL features
func (pe *PostgreSQLExample) CreateAdvancedTable() error {
    createTableSQL := `
    CREATE TABLE IF NOT EXISTS employees (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        salary DECIMAL(10,2),
        department_id INTEGER,
        hire_date DATE,
        skills TEXT[],
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )`
    
    _, err := pe.db.Exec(createTableSQL)
    if err != nil {
        return err
    }
    
    // Create indexes
    indexes := []string{
        "CREATE INDEX IF NOT EXISTS idx_employees_name ON employees USING gin(to_tsvector('english', name))",
        "CREATE INDEX IF NOT EXISTS idx_employees_email ON employees (email)",
        "CREATE INDEX IF NOT EXISTS idx_employees_department ON employees (department_id)",
        "CREATE INDEX IF NOT EXISTS idx_employees_metadata ON employees USING gin(metadata)",
        "CREATE INDEX IF NOT EXISTS idx_employees_skills ON employees USING gin(skills)",
    }
    
    for _, indexSQL := range indexes {
        _, err := pe.db.Exec(indexSQL)
        if err != nil {
            return err
        }
    }
    
    return nil
}

// FullTextSearch demonstrates PostgreSQL full-text search
func (pe *PostgreSQLExample) FullTextSearch() error {
    searchQuery := `
    SELECT 
        name,
        email,
        ts_rank(to_tsvector('english', name), plainto_tsquery('english', $1)) as rank
    FROM employees 
    WHERE to_tsvector('english', name) @@ plainto_tsquery('english', $1)
    ORDER BY rank DESC`
    
    rows, err := pe.db.Query(searchQuery, "john")
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Full-Text Search Results:")
    for rows.Next() {
        var name, email string
        var rank float64
        
        err := rows.Scan(&name, &email, &rank)
        if err != nil {
            return err
        }
        
        fmt.Printf("Name: %s, Email: %s, Rank: %.4f\n", name, email, rank)
    }
    
    return nil
}

// JSONQuery demonstrates JSON operations
func (pe *PostgreSQLExample) JSONQuery() error {
    // Insert sample data with JSON
    _, err := pe.db.Exec(`
        INSERT INTO employees (name, email, salary, department_id, skills, metadata) 
        VALUES ($1, $2, $3, $4, $5, $6)`,
        "John Doe", "john@example.com", 75000, 1,
        []string{"Go", "PostgreSQL", "Docker"},
        `{"experience": 5, "certifications": ["AWS", "Kubernetes"]}`,
    )
    if err != nil {
        return err
    }
    
    // Query JSON data
    jsonQuery := `
    SELECT 
        name,
        metadata->>'experience' as experience,
        metadata->'certifications' as certifications,
        skills
    FROM employees 
    WHERE metadata->>'experience' > '3'`
    
    rows, err := pe.db.Query(jsonQuery)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("JSON Query Results:")
    for rows.Next() {
        var name, experience string
        var certifications, skills []string
        
        err := rows.Scan(&name, &experience, &certifications, &skills)
        if err != nil {
            return err
        }
        
        fmt.Printf("Name: %s, Experience: %s, Certifications: %v, Skills: %v\n",
            name, experience, certifications, skills)
    }
    
    return nil
}

// ArrayOperations demonstrates array operations
func (pe *PostgreSQLExample) ArrayOperations() error {
    // Query using array operations
    arrayQuery := `
    SELECT 
        name,
        skills,
        array_length(skills, 1) as skill_count,
        'Go' = ANY(skills) as knows_go
    FROM employees 
    WHERE 'PostgreSQL' = ANY(skills)`
    
    rows, err := pe.db.Query(arrayQuery)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    fmt.Println("Array Operations Results:")
    for rows.Next() {
        var name string
        var skills []string
        var skillCount int
        var knowsGo bool
        
        err := rows.Scan(&name, &skills, &skillCount, &knowsGo)
        if err != nil {
            return err
        }
        
        fmt.Printf("Name: %s, Skills: %v, Count: %d, Knows Go: %t\n",
            name, skills, skillCount, knowsGo)
    }
    
    return nil
}

func main() {
    pe, err := NewPostgreSQLExample()
    if err != nil {
        log.Fatal(err)
    }
    defer pe.db.Close()
    
    // Create advanced table
    if err := pe.CreateAdvancedTable(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate full-text search
    if err := pe.FullTextSearch(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate JSON operations
    if err := pe.JSONQuery(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate array operations
    if err := pe.ArrayOperations(); err != nil {
        log.Fatal(err)
    }
}
```

---

## 🍃 NoSQL Databases

### **MongoDB Deep Dive**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "go.mongodb.org/mongo-driver/bson"
    "go.mongodb.org/mongo-driver/bson/primitive"
    "go.mongodb.org/mongo-driver/mongo"
    "go.mongodb.org/mongo-driver/mongo/options"
)

// MongoDBExample demonstrates MongoDB features
type MongoDBExample struct {
    client *mongo.Client
    db     *mongo.Database
}

// NewMongoDBExample creates a new MongoDB example
func NewMongoDBExample() (*MongoDBExample, error) {
    client, err := mongo.Connect(context.TODO(), options.Client().ApplyURI("mongodb://localhost:27017"))
    if err != nil {
        return nil, err
    }
    
    db := client.Database("testdb")
    
    return &MongoDBExample{client: client, db: db}, nil
}

// User represents a user document
type User struct {
    ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
    Name      string             `bson:"name" json:"name"`
    Email     string             `bson:"email" json:"email"`
    Age       int                `bson:"age" json:"age"`
    Address   Address            `bson:"address" json:"address"`
    Skills    []string           `bson:"skills" json:"skills"`
    CreatedAt time.Time          `bson:"created_at" json:"created_at"`
    UpdatedAt time.Time          `bson:"updated_at" json:"updated_at"`
}

// Address represents an address embedded document
type Address struct {
    Street  string `bson:"street" json:"street"`
    City    string `bson:"city" json:"city"`
    Country string `bson:"country" json:"country"`
    ZipCode string `bson:"zip_code" json:"zip_code"`
}

// CreateIndexes creates indexes on the users collection
func (me *MongoDBExample) CreateIndexes() error {
    collection := me.db.Collection("users")
    
    // Create single field index
    _, err := collection.Indexes().CreateOne(
        context.TODO(),
        mongo.IndexModel{
            Keys: bson.D{{"email", 1}},
            Options: options.Index().SetUnique(true),
        },
    )
    if err != nil {
        return err
    }
    
    // Create compound index
    _, err = collection.Indexes().CreateOne(
        context.TODO(),
        mongo.IndexModel{
            Keys: bson.D{{"age", 1}, {"created_at", -1}},
        },
    )
    if err != nil {
        return err
    }
    
    // Create text index
    _, err = collection.Indexes().CreateOne(
        context.TODO(),
        mongo.IndexModel{
            Keys: bson.D{{"name", "text"}, {"skills", "text"}},
        },
    )
    if err != nil {
        return err
    }
    
    return nil
}

// InsertUser inserts a user document
func (me *MongoDBExample) InsertUser(user *User) error {
    collection := me.db.Collection("users")
    
    user.CreatedAt = time.Now()
    user.UpdatedAt = time.Now()
    
    result, err := collection.InsertOne(context.TODO(), user)
    if err != nil {
        return err
    }
    
    user.ID = result.InsertedID.(primitive.ObjectID)
    return nil
}

// FindUsers demonstrates various query operations
func (me *MongoDBExample) FindUsers() error {
    collection := me.db.Collection("users")
    
    // Find users with age greater than 25
    filter := bson.M{"age": bson.M{"$gt": 25}}
    cursor, err := collection.Find(context.TODO(), filter)
    if err != nil {
        return err
    }
    defer cursor.Close(context.TODO())
    
    fmt.Println("Users with age > 25:")
    for cursor.Next(context.TODO()) {
        var user User
        if err := cursor.Decode(&user); err != nil {
            return err
        }
        fmt.Printf("Name: %s, Age: %d, Email: %s\n", user.Name, user.Age, user.Email)
    }
    
    // Find users with specific skills
    skillFilter := bson.M{"skills": bson.M{"$in": []string{"Go", "MongoDB"}}}
    cursor, err = collection.Find(context.TODO(), skillFilter)
    if err != nil {
        return err
    }
    defer cursor.Close(context.TODO())
    
    fmt.Println("\nUsers with Go or MongoDB skills:")
    for cursor.Next(context.TODO()) {
        var user User
        if err := cursor.Decode(&user); err != nil {
            return err
        }
        fmt.Printf("Name: %s, Skills: %v\n", user.Name, user.Skills)
    }
    
    return nil
}

// AggregateExample demonstrates aggregation pipeline
func (me *MongoDBExample) AggregateExample() error {
    collection := me.db.Collection("users")
    
    pipeline := mongo.Pipeline{
        {{"$match", bson.M{"age": bson.M{"$gte": 18}}}},
        {{"$group", bson.M{
            "_id": "$address.city",
            "count": bson.M{"$sum": 1},
            "avgAge": bson.M{"$avg": "$age"},
        }}},
        {{"$sort", bson.M{"count": -1}}},
        {{"$limit", 5}},
    }
    
    cursor, err := collection.Aggregate(context.TODO(), pipeline)
    if err != nil {
        return err
    }
    defer cursor.Close(context.TODO())
    
    fmt.Println("Aggregation Results - Users by City:")
    for cursor.Next(context.TODO()) {
        var result bson.M
        if err := cursor.Decode(&result); err != nil {
            return err
        }
        fmt.Printf("City: %s, Count: %d, Avg Age: %.2f\n",
            result["_id"], result["count"], result["avgAge"])
    }
    
    return nil
}

// UpdateUser demonstrates update operations
func (me *MongoDBExample) UpdateUser(userID primitive.ObjectID) error {
    collection := me.db.Collection("users")
    
    filter := bson.M{"_id": userID}
    update := bson.M{
        "$set": bson.M{
            "updated_at": time.Now(),
            "skills": []string{"Go", "MongoDB", "Docker", "Kubernetes"},
        },
        "$inc": bson.M{"age": 1},
    }
    
    result, err := collection.UpdateOne(context.TODO(), filter, update)
    if err != nil {
        return err
    }
    
    fmt.Printf("Updated %d user(s)\n", result.ModifiedCount)
    return nil
}

func main() {
    me, err := NewMongoDBExample()
    if err != nil {
        log.Fatal(err)
    }
    defer me.client.Disconnect(context.TODO())
    
    // Create indexes
    if err := me.CreateIndexes(); err != nil {
        log.Fatal(err)
    }
    
    // Insert sample user
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
        Age:   30,
        Address: Address{
            Street:  "123 Main St",
            City:    "New York",
            Country: "USA",
            ZipCode: "10001",
        },
        Skills: []string{"Go", "MongoDB", "Docker"},
    }
    
    if err := me.InsertUser(user); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate queries
    if err := me.FindUsers(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate aggregation
    if err := me.AggregateExample(); err != nil {
        log.Fatal(err)
    }
    
    // Demonstrate updates
    if err := me.UpdateUser(user.ID); err != nil {
        log.Fatal(err)
    }
}
```

---

*This guide continues with Redis, Elasticsearch, Vector Databases, Snowflake, Time Series Databases, and comprehensive interview questions. Each section includes detailed Go examples, architecture diagrams, and real-world use cases.*


## In Memory Databases

<!-- AUTO-GENERATED ANCHOR: originally referenced as #in-memory-databases -->

Placeholder content. Please replace with proper section.


## Search Engines

<!-- AUTO-GENERATED ANCHOR: originally referenced as #search-engines -->

Placeholder content. Please replace with proper section.


## Vector Databases

<!-- AUTO-GENERATED ANCHOR: originally referenced as #vector-databases -->

Placeholder content. Please replace with proper section.


## Data Warehouses

<!-- AUTO-GENERATED ANCHOR: originally referenced as #data-warehouses -->

Placeholder content. Please replace with proper section.


## Time Series Databases

<!-- AUTO-GENERATED ANCHOR: originally referenced as #time-series-databases -->

Placeholder content. Please replace with proper section.


## Database Architecture Patterns

<!-- AUTO-GENERATED ANCHOR: originally referenced as #database-architecture-patterns -->

Placeholder content. Please replace with proper section.


## Performance Optimization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #performance-optimization -->

Placeholder content. Please replace with proper section.


## Scaling Strategies

<!-- AUTO-GENERATED ANCHOR: originally referenced as #scaling-strategies -->

Placeholder content. Please replace with proper section.


## Interview Questions  Solutions

<!-- AUTO-GENERATED ANCHOR: originally referenced as #interview-questions--solutions -->

Placeholder content. Please replace with proper section.
