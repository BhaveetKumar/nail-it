# Database Design Patterns - Success in Tech Methodology

## Table of Contents
1. [Introduction](#introduction/)
2. [Database Design Fundamentals](#database-design-fundamentals/)
3. [Normalization Patterns](#normalization-patterns/)
4. [Denormalization Strategies](#denormalization-strategies/)
5. [Indexing Patterns](#indexing-patterns/)
6. [Partitioning Strategies](#partitioning-strategies/)
7. [Replication Patterns](#replication-patterns/)
8. [Sharding Strategies](#sharding-strategies/)
9. [Caching Patterns](#caching-patterns/)
10. [Golang Implementation](#golang-implementation/)

## Introduction

This guide is based on Success in Tech's comprehensive approach to database design. It focuses on practical patterns, real-world implementations, and common challenges in building scalable database systems.

### Key Principles
- **Data Integrity**: Ensure data consistency and accuracy
- **Performance**: Optimize for read and write operations
- **Scalability**: Design for growth and high load
- **Availability**: Ensure data is always accessible
- **Consistency**: Balance consistency with performance

## Database Design Fundamentals

### Entity-Relationship Modeling
```go
// E-commerce Database Schema
type DatabaseSchema struct {
    // Core Entities
    Users        *UserTable
    Products     *ProductTable
    Categories   *CategoryTable
    Orders       *OrderTable
    OrderItems   *OrderItemTable
    Payments     *PaymentTable
    Addresses    *AddressTable
    
    // Relationships
    UserAddresses    *UserAddressTable
    ProductCategories *ProductCategoryTable
    UserOrders       *UserOrderTable
}

// User Table
type UserTable struct {
    ID           int64     `db:"id" json:"id"`
    Email        string    `db:"email" json:"email"`
    PasswordHash string    `db:"password_hash" json:"password_hash"`
    FirstName    string    `db:"first_name" json:"first_name"`
    LastName     string    `db:"last_name" json:"last_name"`
    Phone        string    `db:"phone" json:"phone"`
    CreatedAt    time.Time `db:"created_at" json:"created_at"`
    UpdatedAt    time.Time `db:"updated_at" json:"updated_at"`
    IsActive     bool      `db:"is_active" json:"is_active"`
}

// Product Table
type ProductTable struct {
    ID          int64     `db:"id" json:"id"`
    Name        string    `db:"name" json:"name"`
    Description string    `db:"description" json:"description"`
    Price       float64   `db:"price" json:"price"`
    SKU         string    `db:"sku" json:"sku"`
    Stock       int       `db:"stock" json:"stock"`
    CreatedAt   time.Time `db:"created_at" json:"created_at"`
    UpdatedAt   time.Time `db:"updated_at" json:"updated_at"`
    IsActive    bool      `db:"is_active" json:"is_active"`
}

// Order Table
type OrderTable struct {
    ID           int64     `db:"id" json:"id"`
    UserID       int64     `db:"user_id" json:"user_id"`
    TotalAmount  float64   `db:"total_amount" json:"total_amount"`
    Status       string    `db:"status" json:"status"`
    CreatedAt    time.Time `db:"created_at" json:"created_at"`
    UpdatedAt    time.Time `db:"updated_at" json:"updated_at"`
    ShippingAddress *AddressTable `db:"shipping_address" json:"shipping_address"`
}

// Order Item Table
type OrderItemTable struct {
    ID        int64   `db:"id" json:"id"`
    OrderID   int64   `db:"order_id" json:"order_id"`
    ProductID int64   `db:"product_id" json:"product_id"`
    Quantity  int     `db:"quantity" json:"quantity"`
    Price     float64 `db:"price" json:"price"`
    Total     float64 `db:"total" json:"total"`
}
```

### Database Constraints
```go
// Primary Key Constraints
type PrimaryKeyConstraint struct {
    TableName string
    ColumnName string
    IsAutoIncrement bool
}

// Foreign Key Constraints
type ForeignKeyConstraint struct {
    TableName      string
    ColumnName     string
    ReferencedTable string
    ReferencedColumn string
    OnDelete       string // CASCADE, SET NULL, RESTRICT
    OnUpdate       string // CASCADE, SET NULL, RESTRICT
}

// Unique Constraints
type UniqueConstraint struct {
    TableName string
    ColumnNames []string
    IsComposite bool
}

// Check Constraints
type CheckConstraint struct {
    TableName string
    ColumnName string
    Condition string
}

// Database Schema Builder
type SchemaBuilder struct {
    constraints []Constraint
    indexes     []Index
    tables      []Table
}

type Constraint interface {
    GetType() string
    GetTableName() string
    GetColumnName() string
}

func (sb *SchemaBuilder) AddPrimaryKey(tableName, columnName string, autoIncrement bool) *SchemaBuilder {
    constraint := &PrimaryKeyConstraint{
        TableName: tableName,
        ColumnName: columnName,
        IsAutoIncrement: autoIncrement,
    }
    sb.constraints = append(sb.constraints, constraint)
    return sb
}

func (sb *SchemaBuilder) AddForeignKey(tableName, columnName, referencedTable, referencedColumn string) *SchemaBuilder {
    constraint := &ForeignKeyConstraint{
        TableName: tableName,
        ColumnName: columnName,
        ReferencedTable: referencedTable,
        ReferencedColumn: referencedColumn,
        OnDelete: "CASCADE",
        OnUpdate: "CASCADE",
    }
    sb.constraints = append(sb.constraints, constraint)
    return sb
}

func (sb *SchemaBuilder) AddUniqueConstraint(tableName string, columnNames []string) *SchemaBuilder {
    constraint := &UniqueConstraint{
        TableName: tableName,
        ColumnNames: columnNames,
        IsComposite: len(columnNames) > 1,
    }
    sb.constraints = append(sb.constraints, constraint)
    return sb
}
```

## Normalization Patterns

### First Normal Form (1NF)
```go
// Before 1NF - Violates atomicity
type BadOrderTable struct {
    ID        int64  `db:"id"`
    UserID    int64  `db:"user_id"`
    Products  string `db:"products"` // "Product1,Product2,Product3" - Violates 1NF
    Quantities string `db:"quantities"` // "1,2,3" - Violates 1NF
}

// After 1NF - Atomic values
type GoodOrderTable struct {
    ID        int64  `db:"id"`
    UserID    int64  `db:"user_id"`
    ProductID int64  `db:"product_id"` // Atomic value
    Quantity  int    `db:"quantity"`   // Atomic value
}
```

### Second Normal Form (2NF)
```go
// Before 2NF - Partial dependency
type BadOrderItemTable struct {
    OrderID     int64   `db:"order_id"`
    ProductID   int64   `db:"product_id"`
    ProductName string  `db:"product_name"` // Depends only on ProductID, not OrderID
    Quantity    int     `db:"quantity"`
    Price       float64 `db:"price"`
}

// After 2NF - Remove partial dependencies
type GoodOrderItemTable struct {
    OrderID   int64   `db:"order_id"`
    ProductID int64   `db:"product_id"`
    Quantity  int     `db:"quantity"`
    Price     float64 `db:"price"`
}

type ProductTable struct {
    ID   int64  `db:"id"`
    Name string `db:"name"`
    // Other product attributes
}
```

### Third Normal Form (3NF)
```go
// Before 3NF - Transitive dependency
type BadUserTable struct {
    ID       int64  `db:"id"`
    Email    string `db:"email"`
    City     string `db:"city"`     // Depends on UserID
    State    string `db:"state"`    // Depends on City, not UserID
    Country  string `db:"country"`  // Depends on State, not UserID
}

// After 3NF - Remove transitive dependencies
type GoodUserTable struct {
    ID        int64 `db:"id"`
    Email     string `db:"email"`
    AddressID int64 `db:"address_id"`
}

type AddressTable struct {
    ID      int64  `db:"id"`
    City    string `db:"city"`
    StateID int64  `db:"state_id"`
}

type StateTable struct {
    ID      int64  `db:"id"`
    Name    string `db:"name"`
    CountryID int64 `db:"country_id"`
}

type CountryTable struct {
    ID   int64  `db:"id"`
    Name string `db:"name"`
}
```

### Boyce-Codd Normal Form (BCNF)
```go
// Before BCNF - Candidate key dependency
type BadOrderStatusTable struct {
    OrderID   int64  `db:"order_id"`
    Status    string `db:"status"`
    StatusID  int64  `db:"status_id"`
    Timestamp time.Time `db:"timestamp"`
}

// After BCNF - Every determinant is a candidate key
type GoodOrderStatusTable struct {
    OrderID   int64     `db:"order_id"`
    StatusID  int64     `db:"status_id"`
    Timestamp time.Time `db:"timestamp"`
}

type StatusTable struct {
    ID   int64  `db:"id"`
    Name string `db:"name"`
}
```

## Denormalization Strategies

### Read Optimization
```go
// Denormalized User Profile for Read Performance
type DenormalizedUserProfile struct {
    UserID      int64     `db:"user_id"`
    Email       string    `db:"email"`
    FirstName   string    `db:"first_name"`
    LastName    string    `db:"last_name"`
    FullName    string    `db:"full_name"`    // Denormalized
    City        string    `db:"city"`         // Denormalized
    State       string    `db:"state"`        // Denormalized
    Country     string    `db:"country"`      // Denormalized
    OrderCount  int       `db:"order_count"`  // Denormalized
    TotalSpent  float64   `db:"total_spent"`  // Denormalized
    LastOrderAt time.Time `db:"last_order_at"` // Denormalized
}

// Materialized View for Analytics
type UserAnalyticsView struct {
    UserID      int64     `db:"user_id"`
    OrderCount  int       `db:"order_count"`
    TotalSpent  float64   `db:"total_spent"`
    AvgOrderValue float64 `db:"avg_order_value"`
    LastOrderAt time.Time `db:"last_order_at"`
    CreatedAt   time.Time `db:"created_at"`
    UpdatedAt   time.Time `db:"updated_at"`
}

// Update Materialized View
func (db *Database) UpdateUserAnalytics(userID int64) error {
    query := `
        INSERT INTO user_analytics_view (user_id, order_count, total_spent, avg_order_value, last_order_at, created_at, updated_at)
        SELECT 
            u.id,
            COUNT(o.id) as order_count,
            COALESCE(SUM(o.total_amount), 0) as total_spent,
            COALESCE(AVG(o.total_amount), 0) as avg_order_value,
            MAX(o.created_at) as last_order_at,
            NOW() as created_at,
            NOW() as updated_at
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.id = $1
        GROUP BY u.id
        ON CONFLICT (user_id) DO UPDATE SET
            order_count = EXCLUDED.order_count,
            total_spent = EXCLUDED.total_spent,
            avg_order_value = EXCLUDED.avg_order_value,
            last_order_at = EXCLUDED.last_order_at,
            updated_at = NOW()
    `
    
    _, err := db.Exec(query, userID)
    return err
}
```

### Write Optimization
```go
// Write-Through Cache Pattern
type WriteThroughCache struct {
    cache  *redis.Client
    db     *sql.DB
    mutex  sync.RWMutex
}

func (wtc *WriteThroughCache) Set(key string, value interface{}) error {
    wtc.mutex.Lock()
    defer wtc.mutex.Unlock()
    
    // Write to database first
    if err := wtc.writeToDB(key, value); err != nil {
        return err
    }
    
    // Then write to cache
    return wtc.cache.Set(key, value, 0).Err()
}

func (wtc *WriteThroughCache) Get(key string) (interface{}, error) {
    wtc.mutex.RLock()
    defer wtc.mutex.RUnlock()
    
    // Try cache first
    if value, err := wtc.cache.Get(key).Result(); err == nil {
        return value, nil
    }
    
    // Cache miss - read from database
    value, err := wtc.readFromDB(key)
    if err != nil {
        return nil, err
    }
    
    // Write to cache for next time
    wtc.cache.Set(key, value, 0)
    
    return value, nil
}

// Write-Behind Cache Pattern
type WriteBehindCache struct {
    cache     *redis.Client
    db        *sql.DB
    writeQueue chan WriteOperation
    mutex     sync.RWMutex
}

type WriteOperation struct {
    Key   string
    Value interface{}
    Time  time.Time
}

func (wbc *WriteBehindCache) Set(key string, value interface{}) error {
    wbc.mutex.Lock()
    defer wbc.mutex.Unlock()
    
    // Write to cache immediately
    if err := wbc.cache.Set(key, value, 0).Err(); err != nil {
        return err
    }
    
    // Queue for database write
    select {
    case wbc.writeQueue <- WriteOperation{Key: key, Value: value, Time: time.Now()}:
        return nil
    default:
        return errors.New("write queue full")
    }
}

func (wbc *WriteBehindCache) processWriteQueue() {
    for writeOp := range wbc.writeQueue {
        if err := wbc.writeToDB(writeOp.Key, writeOp.Value); err != nil {
            log.Printf("Failed to write to database: %v", err)
        }
    }
}
```

## Indexing Patterns

### B-Tree Indexes
```go
// Single Column Index
type SingleColumnIndex struct {
    TableName  string
    ColumnName string
    IsUnique   bool
    IsPrimary  bool
}

// Composite Index
type CompositeIndex struct {
    TableName   string
    ColumnNames []string
    IsUnique    bool
    Order       []string // ASC, DESC
}

// Partial Index
type PartialIndex struct {
    TableName  string
    ColumnName string
    Condition  string
    IsUnique   bool
}

// Index Manager
type IndexManager struct {
    db *sql.DB
}

func (im *IndexManager) CreateIndex(index Index) error {
    var query string
    
    switch idx := index.(type) {
    case *SingleColumnIndex:
        query = im.buildSingleColumnIndexQuery(idx)
    case *CompositeIndex:
        query = im.buildCompositeIndexQuery(idx)
    case *PartialIndex:
        query = im.buildPartialIndexQuery(idx)
    default:
        return fmt.Errorf("unknown index type")
    }
    
    _, err := im.db.Exec(query)
    return err
}

func (im *IndexManager) buildSingleColumnIndexQuery(idx *SingleColumnIndex) string {
    unique := ""
    if idx.IsUnique {
        unique = "UNIQUE "
    }
    
    return fmt.Sprintf("CREATE %sINDEX idx_%s_%s ON %s (%s)",
        unique, idx.TableName, idx.ColumnName, idx.TableName, idx.ColumnName)
}

func (im *IndexManager) buildCompositeIndexQuery(idx *CompositeIndex) string {
    unique := ""
    if idx.IsUnique {
        unique = "UNIQUE "
    }
    
    columns := strings.Join(idx.ColumnNames, ", ")
    return fmt.Sprintf("CREATE %sINDEX idx_%s_%s ON %s (%s)",
        unique, idx.TableName, strings.Join(idx.ColumnNames, "_"), idx.TableName, columns)
}

func (im *IndexManager) buildPartialIndexQuery(idx *PartialIndex) string {
    unique := ""
    if idx.IsUnique {
        unique = "UNIQUE "
    }
    
    return fmt.Sprintf("CREATE %sINDEX idx_%s_%s ON %s (%s) WHERE %s",
        unique, idx.TableName, idx.ColumnName, idx.TableName, idx.ColumnName, idx.Condition)
}
```

### Hash Indexes
```go
// Hash Index for Equality Queries
type HashIndex struct {
    TableName  string
    ColumnName string
    IsUnique   bool
}

func (im *IndexManager) CreateHashIndex(idx *HashIndex) error {
    unique := ""
    if idx.IsUnique {
        unique = "UNIQUE "
    }
    
    query := fmt.Sprintf("CREATE %sINDEX idx_%s_%s_hash ON %s USING HASH (%s)",
        unique, idx.TableName, idx.ColumnName, idx.TableName, idx.ColumnName)
    
    _, err := im.db.Exec(query)
    return err
}
```

### Bitmap Indexes
```go
// Bitmap Index for Low Cardinality Columns
type BitmapIndex struct {
    TableName  string
    ColumnName string
    Values     []interface{}
}

func (im *IndexManager) CreateBitmapIndex(idx *BitmapIndex) error {
    // Create bitmap index for each value
    for _, value := range idx.Values {
        query := fmt.Sprintf("CREATE INDEX idx_%s_%s_%v_bitmap ON %s (%s) WHERE %s = %v",
            idx.TableName, idx.ColumnName, value, idx.TableName, idx.ColumnName, idx.ColumnName, value)
        
        if _, err := im.db.Exec(query); err != nil {
            return err
        }
    }
    
    return nil
}
```

## Partitioning Strategies

### Horizontal Partitioning (Sharding)
```go
// Range Partitioning
type RangePartition struct {
    TableName string
    ColumnName string
    Partitions []PartitionRange
}

type PartitionRange struct {
    Name      string
    MinValue  interface{}
    MaxValue  interface{}
    IsDefault bool
}

func (rp *RangePartition) CreatePartitions() error {
    for _, partition := range rp.Partitions {
        query := fmt.Sprintf("CREATE TABLE %s_%s PARTITION OF %s FOR VALUES FROM (%v) TO (%v)",
            rp.TableName, partition.Name, rp.TableName, partition.MinValue, partition.MaxValue)
        
        if _, err := rp.db.Exec(query); err != nil {
            return err
        }
    }
    
    return nil
}

// Hash Partitioning
type HashPartition struct {
    TableName string
    ColumnName string
    PartitionCount int
}

func (hp *HashPartition) CreatePartitions() error {
    for i := 0; i < hp.PartitionCount; i++ {
        query := fmt.Sprintf("CREATE TABLE %s_p%d PARTITION OF %s FOR VALUES WITH (MODULUS %d, REMAINDER %d)",
            hp.TableName, i, hp.TableName, hp.PartitionCount, i)
        
        if _, err := hp.db.Exec(query); err != nil {
            return err
        }
    }
    
    return nil
}

// List Partitioning
type ListPartition struct {
    TableName string
    ColumnName string
    Partitions []PartitionList
}

type PartitionList struct {
    Name   string
    Values []interface{}
}

func (lp *ListPartition) CreatePartitions() error {
    for _, partition := range lp.Partitions {
        values := make([]string, len(partition.Values))
        for i, value := range partition.Values {
            values[i] = fmt.Sprintf("'%v'", value)
        }
        
        query := fmt.Sprintf("CREATE TABLE %s_%s PARTITION OF %s FOR VALUES IN (%s)",
            lp.TableName, partition.Name, lp.TableName, strings.Join(values, ", "))
        
        if _, err := lp.db.Exec(query); err != nil {
            return err
        }
    }
    
    return nil
}
```

### Vertical Partitioning
```go
// Vertical Partitioning by Access Patterns
type VerticalPartition struct {
    TableName string
    Partitions []PartitionColumn
}

type PartitionColumn struct {
    Name      string
    Columns   []string
    AccessPattern string // READ_HEAVY, WRITE_HEAVY, MIXED
}

func (vp *VerticalPartition) CreatePartitions() error {
    for _, partition := range vp.Partitions {
        columns := strings.Join(partition.Columns, ", ")
        query := fmt.Sprintf("CREATE TABLE %s_%s AS SELECT %s FROM %s",
            vp.TableName, partition.Name, columns, vp.TableName)
        
        if _, err := vp.db.Exec(query); err != nil {
            return err
        }
    }
    
    return nil
}
```

## Replication Patterns

### Master-Slave Replication
```go
// Master-Slave Replication Manager
type MasterSlaveReplication struct {
    Master *sql.DB
    Slaves []*sql.DB
    LoadBalancer *LoadBalancer
}

func (msr *MasterSlaveReplication) Write(query string, args ...interface{}) (*sql.Result, error) {
    // All writes go to master
    return msr.Master.Exec(query, args...)
}

func (msr *MasterSlaveReplication) Read(query string, args ...interface{}) (*sql.Rows, error) {
    // Reads can go to any slave
    slave := msr.LoadBalancer.GetSlave()
    return slave.Query(query, args...)
}

func (msr *MasterSlaveReplication) ReadOne(query string, args ...interface{}) *sql.Row {
    // Reads can go to any slave
    slave := msr.LoadBalancer.GetSlave()
    return slave.QueryRow(query, args...)
}

// Load Balancer for Slaves
type LoadBalancer struct {
    Slaves []*sql.DB
    Current int
    mutex  sync.Mutex
}

func (lb *LoadBalancer) GetSlave() *sql.DB {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    slave := lb.Slaves[lb.Current]
    lb.Current = (lb.Current + 1) % len(lb.Slaves)
    return slave
}
```

### Master-Master Replication
```go
// Master-Master Replication Manager
type MasterMasterReplication struct {
    Masters []*sql.DB
    LoadBalancer *LoadBalancer
    ConflictResolver *ConflictResolver
}

func (mmr *MasterMasterReplication) Write(query string, args ...interface{}) (*sql.Result, error) {
    // Writes can go to any master
    master := mmr.LoadBalancer.GetMaster()
    return master.Exec(query, args...)
}

func (mmr *MasterMasterReplication) Read(query string, args ...interface{}) (*sql.Rows, error) {
    // Reads can go to any master
    master := mmr.LoadBalancer.GetMaster()
    return master.Query(query, args...)
}

// Conflict Resolution
type ConflictResolver struct {
    strategies map[string]ConflictStrategy
}

type ConflictStrategy interface {
    Resolve(conflict Conflict) interface{}
}

type Conflict struct {
    TableName string
    ColumnName string
    Value1    interface{}
    Value2    interface{}
    Timestamp1 time.Time
    Timestamp2 time.Time
}

// Last Write Wins Strategy
type LastWriteWinsStrategy struct{}

func (lwws *LastWriteWinsStrategy) Resolve(conflict Conflict) interface{} {
    if conflict.Timestamp1.After(conflict.Timestamp2) {
        return conflict.Value1
    }
    return conflict.Value2
}

// First Write Wins Strategy
type FirstWriteWinsStrategy struct{}

func (fwws *FirstWriteWinsStrategy) Resolve(conflict Conflict) interface{} {
    if conflict.Timestamp1.Before(conflict.Timestamp2) {
        return conflict.Value1
    }
    return conflict.Value2
}
```

## Sharding Strategies

### Consistent Hashing
```go
// Consistent Hash Ring
type ConsistentHashRing struct {
    nodes    []HashNode
    replicas int
    hashFunc func(string) uint32
    mutex    sync.RWMutex
}

type HashNode struct {
    ID     string
    DB     *sql.DB
    Weight int
}

func (chr *ConsistentHashRing) AddNode(node HashNode) {
    chr.mutex.Lock()
    defer chr.mutex.Unlock()
    
    for i := 0; i < node.Weight*chr.replicas; i++ {
        hash := chr.hashFunc(fmt.Sprintf("%s:%d", node.ID, i))
        chr.nodes = append(chr.nodes, HashNode{
            ID:     node.ID,
            DB:     node.DB,
            Weight: node.Weight,
        })
    }
    
    sort.Slice(chr.nodes, func(i, j int) bool {
        return chr.hashFunc(chr.nodes[i].ID) < chr.hashFunc(chr.nodes[j].ID)
    })
}

func (chr *ConsistentHashRing) GetNode(key string) *HashNode {
    chr.mutex.RLock()
    defer chr.mutex.RUnlock()
    
    if len(chr.nodes) == 0 {
        return nil
    }
    
    hash := chr.hashFunc(key)
    
    for _, node := range chr.nodes {
        if chr.hashFunc(node.ID) >= hash {
            return &node
        }
    }
    
    // Wrap around to first node
    return &chr.nodes[0]
}

// Shard Manager
type ShardManager struct {
    hashRing *ConsistentHashRing
    shards   map[string]*Shard
}

type Shard struct {
    ID       string
    DB       *sql.DB
    Range    ShardRange
    Replicas []*Shard
}

type ShardRange struct {
    Start int64
    End   int64
}

func (sm *ShardManager) GetShard(key string) *Shard {
    node := sm.hashRing.GetNode(key)
    if node == nil {
        return nil
    }
    
    return sm.shards[node.ID]
}

func (sm *ShardManager) Write(key string, query string, args ...interface{}) (*sql.Result, error) {
    shard := sm.GetShard(key)
    if shard == nil {
        return nil, errors.New("no shard available")
    }
    
    return shard.DB.Exec(query, args...)
}

func (sm *ShardManager) Read(key string, query string, args ...interface{}) (*sql.Rows, error) {
    shard := sm.GetShard(key)
    if shard == nil {
        return nil, errors.New("no shard available")
    }
    
    return shard.DB.Query(query, args...)
}
```

### Directory-Based Sharding
```go
// Shard Directory
type ShardDirectory struct {
    shards map[string]*Shard
    mutex  sync.RWMutex
}

func (sd *ShardDirectory) GetShard(key string) *Shard {
    sd.mutex.RLock()
    defer sd.mutex.RUnlock()
    
    // Simple hash-based shard selection
    hash := crc32.ChecksumIEEE([]byte(key))
    shardID := fmt.Sprintf("shard_%d", hash%int32(len(sd.shards)))
    
    return sd.shards[shardID]
}

func (sd *ShardDirectory) AddShard(shard *Shard) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    sd.shards[shard.ID] = shard
}

func (sd *ShardDirectory) RemoveShard(shardID string) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    delete(sd.shards, shardID)
}
```

## Caching Patterns

### Multi-Level Caching
```go
// Multi-Level Cache
type MultiLevelCache struct {
    L1Cache *sync.Map // In-memory cache
    L2Cache *redis.Client // Redis cache
    L3Cache *CDNClient // CDN cache
    DB      *sql.DB // Database
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // L1: Check in-memory cache
    if value, found := mlc.L1Cache.Load(key); found {
        return value, nil
    }
    
    // L2: Check Redis cache
    if value, err := mlc.L2Cache.Get(key).Result(); err == nil {
        // Store in L1 cache
        mlc.L1Cache.Store(key, value)
        return value, nil
    }
    
    // L3: Check CDN cache
    if value, err := mlc.L3Cache.Get(key); err == nil {
        // Store in L2 and L1 caches
        mlc.L2Cache.Set(key, value, 0)
        mlc.L1Cache.Store(key, value)
        return value, nil
    }
    
    // Cache miss - read from database
    value, err := mlc.readFromDB(key)
    if err != nil {
        return nil, err
    }
    
    // Store in all caches
    mlc.L1Cache.Store(key, value)
    mlc.L2Cache.Set(key, value, 0)
    mlc.L3Cache.Set(key, value)
    
    return value, nil
}

func (mlc *MultiLevelCache) Set(key string, value interface{}) error {
    // Write to database first
    if err := mlc.writeToDB(key, value); err != nil {
        return err
    }
    
    // Then write to all caches
    mlc.L1Cache.Store(key, value)
    mlc.L2Cache.Set(key, value, 0)
    mlc.L3Cache.Set(key, value)
    
    return nil
}
```

### Cache-Aside Pattern
```go
// Cache-Aside Implementation
type CacheAside struct {
    cache *redis.Client
    db    *sql.DB
    mutex sync.RWMutex
}

func (ca *CacheAside) Get(key string) (interface{}, error) {
    ca.mutex.RLock()
    defer ca.mutex.RUnlock()
    
    // Try cache first
    if value, err := ca.cache.Get(key).Result(); err == nil {
        return value, nil
    }
    
    // Cache miss - read from database
    value, err := ca.readFromDB(key)
    if err != nil {
        return nil, err
    }
    
    // Write to cache for next time
    ca.cache.Set(key, value, 0)
    
    return value, nil
}

func (ca *CacheAside) Set(key string, value interface{}) error {
    ca.mutex.Lock()
    defer ca.mutex.Unlock()
    
    // Write to database
    if err := ca.writeToDB(key, value); err != nil {
        return err
    }
    
    // Invalidate cache
    ca.cache.Del(key)
    
    return nil
}
```

## Golang Implementation

### Database Connection Pool
```go
// Connection Pool Manager
type ConnectionPool struct {
    db     *sql.DB
    config *PoolConfig
    stats  *PoolStats
    mutex  sync.RWMutex
}

type PoolConfig struct {
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

type PoolStats struct {
    OpenConnections int
    InUse           int
    Idle            int
    WaitCount       int64
    WaitDuration    time.Duration
}

func NewConnectionPool(dsn string, config *PoolConfig) (*ConnectionPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    return &ConnectionPool{
        db:     db,
        config: config,
        stats:  &PoolStats{},
    }, nil
}

func (cp *ConnectionPool) GetStats() *PoolStats {
    cp.mutex.RLock()
    defer cp.mutex.RUnlock()
    
    stats := db.Stats()
    cp.stats.OpenConnections = stats.OpenConnections
    cp.stats.InUse = stats.InUse
    cp.stats.Idle = stats.Idle
    cp.stats.WaitCount = stats.WaitCount
    cp.stats.WaitDuration = stats.WaitDuration
    
    return cp.stats
}
```

### Query Builder
```go
// SQL Query Builder
type QueryBuilder struct {
    table     string
    columns   []string
    where     []WhereCondition
    joins     []JoinCondition
    groupBy   []string
    having    []WhereCondition
    orderBy   []OrderCondition
    limit     int
    offset    int
}

type WhereCondition struct {
    Column   string
    Operator string
    Value    interface{}
    Logic    string // AND, OR
}

type JoinCondition struct {
    Table    string
    On       string
    Type     string // INNER, LEFT, RIGHT, FULL
}

type OrderCondition struct {
    Column string
    Direction string // ASC, DESC
}

func NewQueryBuilder(table string) *QueryBuilder {
    return &QueryBuilder{
        table: table,
        where: make([]WhereCondition, 0),
        joins: make([]JoinCondition, 0),
        groupBy: make([]string, 0),
        having: make([]WhereCondition, 0),
        orderBy: make([]OrderCondition, 0),
    }
}

func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.columns = columns
    return qb
}

func (qb *QueryBuilder) Where(column, operator string, value interface{}) *QueryBuilder {
    qb.where = append(qb.where, WhereCondition{
        Column:   column,
        Operator: operator,
        Value:    value,
        Logic:    "AND",
    })
    return qb
}

func (qb *QueryBuilder) OrWhere(column, operator string, value interface{}) *QueryBuilder {
    qb.where = append(qb.where, WhereCondition{
        Column:   column,
        Operator: operator,
        Value:    value,
        Logic:    "OR",
    })
    return qb
}

func (qb *QueryBuilder) Join(table, on string) *QueryBuilder {
    qb.joins = append(qb.joins, JoinCondition{
        Table: table,
        On:    on,
        Type:  "INNER",
    })
    return qb
}

func (qb *QueryBuilder) LeftJoin(table, on string) *QueryBuilder {
    qb.joins = append(qb.joins, JoinCondition{
        Table: table,
        On:    on,
        Type:  "LEFT",
    })
    return qb
}

func (qb *QueryBuilder) GroupBy(columns ...string) *QueryBuilder {
    qb.groupBy = append(qb.groupBy, columns...)
    return qb
}

func (qb *QueryBuilder) Having(column, operator string, value interface{}) *QueryBuilder {
    qb.having = append(qb.having, WhereCondition{
        Column:   column,
        Operator: operator,
        Value:    value,
        Logic:    "AND",
    })
    return qb
}

func (qb *QueryBuilder) OrderBy(column, direction string) *QueryBuilder {
    qb.orderBy = append(qb.orderBy, OrderCondition{
        Column:    column,
        Direction: direction,
    })
    return qb
}

func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
    qb.limit = limit
    return qb
}

func (qb *QueryBuilder) Offset(offset int) *QueryBuilder {
    qb.offset = offset
    return qb
}

func (qb *QueryBuilder) Build() (string, []interface{}) {
    var query strings.Builder
    var args []interface{}
    
    // SELECT clause
    if len(qb.columns) > 0 {
        query.WriteString("SELECT " + strings.Join(qb.columns, ", "))
    } else {
        query.WriteString("SELECT *")
    }
    
    // FROM clause
    query.WriteString(" FROM " + qb.table)
    
    // JOIN clauses
    for _, join := range qb.joins {
        query.WriteString(fmt.Sprintf(" %s JOIN %s ON %s", join.Type, join.Table, join.On))
    }
    
    // WHERE clause
    if len(qb.where) > 0 {
        query.WriteString(" WHERE ")
        for i, condition := range qb.where {
            if i > 0 {
                query.WriteString(" " + condition.Logic + " ")
            }
            query.WriteString(fmt.Sprintf("%s %s $%d", condition.Column, condition.Operator, len(args)+1))
            args = append(args, condition.Value)
        }
    }
    
    // GROUP BY clause
    if len(qb.groupBy) > 0 {
        query.WriteString(" GROUP BY " + strings.Join(qb.groupBy, ", "))
    }
    
    // HAVING clause
    if len(qb.having) > 0 {
        query.WriteString(" HAVING ")
        for i, condition := range qb.having {
            if i > 0 {
                query.WriteString(" " + condition.Logic + " ")
            }
            query.WriteString(fmt.Sprintf("%s %s $%d", condition.Column, condition.Operator, len(args)+1))
            args = append(args, condition.Value)
        }
    }
    
    // ORDER BY clause
    if len(qb.orderBy) > 0 {
        query.WriteString(" ORDER BY ")
        for i, condition := range qb.orderBy {
            if i > 0 {
                query.WriteString(", ")
            }
            query.WriteString(fmt.Sprintf("%s %s", condition.Column, condition.Direction))
        }
    }
    
    // LIMIT clause
    if qb.limit > 0 {
        query.WriteString(fmt.Sprintf(" LIMIT %d", qb.limit))
    }
    
    // OFFSET clause
    if qb.offset > 0 {
        query.WriteString(fmt.Sprintf(" OFFSET %d", qb.offset))
    }
    
    return query.String(), args
}
```

## Conclusion

Database design patterns are essential for building scalable, performant, and maintainable systems. Key success factors:

1. **Proper Normalization**: Balance between normalization and performance
2. **Strategic Denormalization**: Optimize for read performance when needed
3. **Effective Indexing**: Choose the right index type for your use case
4. **Smart Partitioning**: Distribute data efficiently across multiple nodes
5. **Reliable Replication**: Ensure data availability and consistency
6. **Efficient Sharding**: Scale horizontally while maintaining performance
7. **Intelligent Caching**: Reduce database load and improve response times

By following these patterns and best practices, you can build robust database systems that meet the demands of modern applications.
