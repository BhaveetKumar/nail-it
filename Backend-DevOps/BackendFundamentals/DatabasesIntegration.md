# 🗄️ Databases Integration: SQL vs NoSQL, Connection Pooling, Migrations

> **Master database integration patterns for scalable backend systems**

## 📚 Concept

Database integration is crucial for backend systems. Understanding SQL vs NoSQL, connection pooling, and migrations is essential for building scalable applications.

### Database Types

- **SQL**: Relational databases (PostgreSQL, MySQL, SQLite)
- **NoSQL**: Document (MongoDB), Key-Value (Redis), Column (Cassandra), Graph (Neo4j)

## 🛠️ Hands-on Example

### SQL Database Integration (Go with GORM)

```go
package main

import (
    "fmt"
    "log"
    "time"

    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/logger"
)

type User struct {
    ID        uint      `gorm:"primaryKey" json:"id"`
    Username  string    `gorm:"uniqueIndex;not null" json:"username"`
    Email     string    `gorm:"uniqueIndex;not null" json:"email"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
    Posts     []Post    `gorm:"foreignKey:UserID" json:"posts,omitempty"`
}

type Post struct {
    ID        uint      `gorm:"primaryKey" json:"id"`
    Title     string    `gorm:"not null" json:"title"`
    Content   string    `gorm:"type:text" json:"content"`
    UserID    uint      `gorm:"not null" json:"user_id"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
    User      User      `gorm:"foreignKey:UserID" json:"user,omitempty"`
}

type DatabaseService struct {
    db *gorm.DB
}

func NewDatabaseService(dsn string) (*DatabaseService, error) {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
    })
    if err != nil {
        return nil, err
    }

    // Configure connection pool
    sqlDB, err := db.DB()
    if err != nil {
        return nil, err
    }

    sqlDB.SetMaxIdleConns(10)
    sqlDB.SetMaxOpenConns(100)
    sqlDB.SetConnMaxLifetime(time.Hour)

    return &DatabaseService{db: db}, nil
}

func (ds *DatabaseService) AutoMigrate() error {
    return ds.db.AutoMigrate(&User{}, &Post{})
}

func (ds *DatabaseService) CreateUser(user *User) error {
    return ds.db.Create(user).Error
}

func (ds *DatabaseService) GetUser(id uint) (*User, error) {
    var user User
    err := ds.db.Preload("Posts").First(&user, id).Error
    return &user, err
}

func (ds *DatabaseService) GetUsers(limit, offset int) ([]User, error) {
    var users []User
    err := ds.db.Limit(limit).Offset(offset).Find(&users).Error
    return users, err
}

func (ds *DatabaseService) UpdateUser(id uint, updates map[string]interface{}) error {
    return ds.db.Model(&User{}).Where("id = ?", id).Updates(updates).Error
}

func (ds *DatabaseService) DeleteUser(id uint) error {
    return ds.db.Delete(&User{}, id).Error
}

func (ds *DatabaseService) CreatePost(post *Post) error {
    return ds.db.Create(post).Error
}

func (ds *DatabaseService) GetPostsByUser(userID uint) ([]Post, error) {
    var posts []Post
    err := ds.db.Where("user_id = ?", userID).Find(&posts).Error
    return posts, err
}

// Transaction example
func (ds *DatabaseService) CreateUserWithPost(user *User, post *Post) error {
    return ds.db.Transaction(func(tx *gorm.DB) error {
        if err := tx.Create(user).Error; err != nil {
            return err
        }

        post.UserID = user.ID
        if err := tx.Create(post).Error; err != nil {
            return err
        }

        return nil
    })
}

// Raw SQL example
func (ds *DatabaseService) GetUserStats() (map[string]interface{}, error) {
    var result struct {
        TotalUsers int `json:"total_users"`
        TotalPosts int `json:"total_posts"`
    }

    err := ds.db.Raw(`
        SELECT
            (SELECT COUNT(*) FROM users) as total_users,
            (SELECT COUNT(*) FROM posts) as total_posts
    `).Scan(&result).Error

    return map[string]interface{}{
        "total_users": result.TotalUsers,
        "total_posts": result.TotalPosts,
    }, err
}
```

### NoSQL Database Integration (MongoDB)

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

type User struct {
    ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
    Username  string             `bson:"username" json:"username"`
    Email     string             `bson:"email" json:"email"`
    CreatedAt time.Time          `bson:"created_at" json:"created_at"`
    UpdatedAt time.Time          `bson:"updated_at" json:"updated_at"`
}

type Post struct {
    ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
    Title     string             `bson:"title" json:"title"`
    Content   string             `bson:"content" json:"content"`
    UserID    primitive.ObjectID `bson:"user_id" json:"user_id"`
    CreatedAt time.Time          `bson:"created_at" json:"created_at"`
    UpdatedAt time.Time          `bson:"updated_at" json:"updated_at"`
}

type MongoService struct {
    client   *mongo.Client
    database *mongo.Database
}

func NewMongoService(uri string) (*MongoService, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
    if err != nil {
        return nil, err
    }

    database := client.Database("blog")

    return &MongoService{
        client:   client,
        database: database,
    }, nil
}

func (ms *MongoService) CreateUser(user *User) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    user.CreatedAt = time.Now()
    user.UpdatedAt = time.Now()

    collection := ms.database.Collection("users")
    _, err := collection.InsertOne(ctx, user)
    return err
}

func (ms *MongoService) GetUser(id primitive.ObjectID) (*User, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    var user User
    collection := ms.database.Collection("users")
    err := collection.FindOne(ctx, bson.M{"_id": id}).Decode(&user)
    return &user, err
}

func (ms *MongoService) GetUsers(limit, skip int64) ([]User, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    var users []User
    collection := ms.database.Collection("users")

    opts := options.Find().SetLimit(limit).SetSkip(skip)
    cursor, err := collection.Find(ctx, bson.M{}, opts)
    if err != nil {
        return nil, err
    }
    defer cursor.Close(ctx)

    err = cursor.All(ctx, &users)
    return users, err
}

func (ms *MongoService) UpdateUser(id primitive.ObjectID, updates bson.M) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    updates["updated_at"] = time.Now()
    collection := ms.database.Collection("users")
    _, err := collection.UpdateOne(ctx, bson.M{"_id": id}, bson.M{"$set": updates})
    return err
}

func (ms *MongoService) DeleteUser(id primitive.ObjectID) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    collection := ms.database.Collection("users")
    _, err := collection.DeleteOne(ctx, bson.M{"_id": id})
    return err
}

func (ms *MongoService) CreatePost(post *Post) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    post.CreatedAt = time.Now()
    post.UpdatedAt = time.Now()

    collection := ms.database.Collection("posts")
    _, err := collection.InsertOne(ctx, post)
    return err
}

func (ms *MongoService) GetPostsByUser(userID primitive.ObjectID) ([]Post, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    var posts []Post
    collection := ms.database.Collection("posts")

    cursor, err := collection.Find(ctx, bson.M{"user_id": userID})
    if err != nil {
        return nil, err
    }
    defer cursor.Close(ctx)

    err = cursor.All(ctx, &posts)
    return posts, err
}

// Aggregation example
func (ms *MongoService) GetUserPostStats() ([]bson.M, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    collection := ms.database.Collection("posts")

    pipeline := []bson.M{
        {
            "$group": bson.M{
                "_id":   "$user_id",
                "count": bson.M{"$sum": 1},
            },
        },
        {
            "$lookup": bson.M{
                "from":         "users",
                "localField":   "_id",
                "foreignField": "_id",
                "as":           "user",
            },
        },
        {
            "$unwind": "$user",
        },
        {
            "$project": bson.M{
                "username": "$user.username",
                "post_count": "$count",
            },
        },
    }

    cursor, err := collection.Aggregate(ctx, pipeline)
    if err != nil {
        return nil, err
    }
    defer cursor.Close(ctx)

    var results []bson.M
    err = cursor.All(ctx, &results)
    return results, err
}
```

### Connection Pooling

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "sync"
    "time"

    _ "github.com/lib/pq"
)

type ConnectionPool struct {
    db     *sql.DB
    mutex  sync.RWMutex
    config PoolConfig
}

type PoolConfig struct {
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

func NewConnectionPool(dsn string, config PoolConfig) (*ConnectionPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }

    // Configure connection pool
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)

    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }

    return &ConnectionPool{
        db:     db,
        config: config,
    }, nil
}

func (cp *ConnectionPool) GetDB() *sql.DB {
    cp.mutex.RLock()
    defer cp.mutex.RUnlock()
    return cp.db
}

func (cp *ConnectionPool) GetStats() sql.DBStats {
    return cp.db.Stats()
}

func (cp *ConnectionPool) Close() error {
    return cp.db.Close()
}

// Health check
func (cp *ConnectionPool) HealthCheck() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    return cp.db.PingContext(ctx)
}
```

### Database Migrations

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "os"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
    "time"

    "github.com/golang-migrate/migrate/v4"
    "github.com/golang-migrate/migrate/v4/database/postgres"
    _ "github.com/golang-migrate/migrate/v4/source/file"
)

type MigrationService struct {
    db     *sql.DB
    migrate *migrate.Migrate
}

func NewMigrationService(dsn string) (*MigrationService, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }

    driver, err := postgres.WithInstance(db, &postgres.Config{})
    if err != nil {
        return nil, err
    }

    m, err := migrate.NewWithDatabaseInstance(
        "file://migrations",
        "postgres", driver)
    if err != nil {
        return nil, err
    }

    return &MigrationService{
        db:      db,
        migrate: m,
    }, nil
}

func (ms *MigrationService) Up() error {
    return ms.migrate.Up()
}

func (ms *MigrationService) Down() error {
    return ms.migrate.Down()
}

func (ms *MigrationService) Steps(n int) error {
    return ms.migrate.Steps(n)
}

func (ms *MigrationService) Version() (uint, bool, error) {
    return ms.migrate.Version()
}

func (ms *MigrationService) Force(version int) error {
    return ms.migrate.Force(version)
}

// Create migration files
func CreateMigration(name string) error {
    timestamp := time.Now().Format("20060102150405")

    upFile := fmt.Sprintf("migrations/%s_%s.up.sql", timestamp, name)
    downFile := fmt.Sprintf("migrations/%s_%s.down.sql", timestamp, name)

    // Create up migration
    upContent := fmt.Sprintf("-- +migrate Up\n-- Add your up migration here\n")
    if err := os.WriteFile(upFile, []byte(upContent), 0644); err != nil {
        return err
    }

    // Create down migration
    downContent := fmt.Sprintf("-- +migrate Down\n-- Add your down migration here\n")
    if err := os.WriteFile(downFile, []byte(downContent), 0644); err != nil {
        return err
    }

    log.Printf("Created migration files: %s, %s", upFile, downFile)
    return nil
}
```

## 🚀 Best Practices

### 1. Connection Pooling

```go
// Configure connection pool based on load
func ConfigurePool(db *sql.DB, maxConns, maxIdle int) {
    db.SetMaxOpenConns(maxConns)
    db.SetMaxIdleConns(maxIdle)
    db.SetConnMaxLifetime(time.Hour)
    db.SetConnMaxIdleTime(time.Minute * 30)
}
```

### 2. Query Optimization

```go
// Use prepared statements
func (ds *DatabaseService) GetUserByEmail(email string) (*User, error) {
    stmt, err := ds.db.Prepare("SELECT * FROM users WHERE email = $1")
    if err != nil {
        return nil, err
    }
    defer stmt.Close()

    var user User
    err = stmt.QueryRow(email).Scan(&user.ID, &user.Username, &user.Email, &user.CreatedAt, &user.UpdatedAt)
    return &user, err
}
```

### 3. Transaction Management

```go
func (ds *DatabaseService) TransferMoney(fromID, toID uint, amount float64) error {
    tx, err := ds.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // Deduct from sender
    if _, err := tx.Exec("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, fromID); err != nil {
        return err
    }

    // Add to receiver
    if _, err := tx.Exec("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, toID); err != nil {
        return err
    }

    return tx.Commit()
}
```

## 🏢 Industry Insights

### Meta's Database Strategy

- **MySQL**: Primary database for user data
- **Cassandra**: Time-series data
- **HBase**: Big data analytics
- **Memcached**: Caching layer

### Google's Database Strategy

- **Spanner**: Global distributed database
- **Bigtable**: NoSQL for large-scale data
- **Cloud SQL**: Managed relational databases
- **Firestore**: Document database

### Amazon's Database Strategy

- **RDS**: Managed relational databases
- **DynamoDB**: NoSQL database
- **Redshift**: Data warehouse
- **ElastiCache**: In-memory caching

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between SQL and NoSQL?**

   - SQL: Relational, ACID properties, structured data
   - NoSQL: Non-relational, flexible schema, horizontal scaling

2. **What is connection pooling?**

   - Reuse database connections
   - Reduce connection overhead
   - Control concurrent connections
   - Improve performance

3. **What are database migrations?**
   - Version control for database schema
   - Automated schema changes
   - Rollback capabilities
   - Team collaboration

### Intermediate Level

4. **How do you handle database transactions?**

   ```go
   func (ds *DatabaseService) TransferMoney(fromID, toID uint, amount float64) error {
       tx, err := ds.db.Begin()
       if err != nil {
           return err
       }
       defer tx.Rollback()

       // Perform operations
       if err := tx.Exec("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, fromID); err != nil {
           return err
       }

       return tx.Commit()
   }
   ```

5. **How do you optimize database queries?**

   - Use indexes
   - Avoid N+1 queries
   - Use prepared statements
   - Query analysis and optimization

6. **How do you handle database failures?**
   - Connection retry logic
   - Circuit breaker pattern
   - Health checks
   - Failover mechanisms

### Advanced Level

7. **How do you implement database sharding?**

   - Horizontal partitioning
   - Consistent hashing
   - Shard key selection
   - Cross-shard queries

8. **How do you handle database replication?**

   - Master-slave replication
   - Read replicas
   - Consistency models
   - Failover strategies

9. **How do you implement database backup and recovery?**
   - Point-in-time recovery
   - Incremental backups
   - Cross-region replication
   - Disaster recovery

---

**Next**: [Scaling Microservices](./ScalingMicroservices.md) - Service mesh, load balancing, circuit breakers
