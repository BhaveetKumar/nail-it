---
# Auto-generated front matter
Title: Advanced Database Patterns
LastUpdated: 2025-11-06T20:45:58.663823
Tags: []
Status: draft
---

# Advanced Database Patterns

Advanced database patterns and practices for backend systems.

## 🎯 Database Architecture Patterns

### Multi-Tenant Database Design
```sql
-- Schema per tenant approach
CREATE SCHEMA tenant_123;
CREATE TABLE tenant_123.users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Shared schema with tenant isolation
CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, email)
);

-- Row-level security
CREATE POLICY tenant_isolation ON users
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

### Database Sharding
```go
type ShardManager struct {
    shards    map[string]*sql.DB
    router    ShardRouter
    balancer  LoadBalancer
}

type ShardRouter struct {
    hashFunction func(string) int
    shardCount   int
}

func (sr *ShardRouter) GetShard(key string) string {
    hash := sr.hashFunction(key)
    shardIndex := hash % sr.shardCount
    return fmt.Sprintf("shard_%d", shardIndex)
}

func (sm *ShardManager) ExecuteQuery(query string, args ...interface{}) (*sql.Rows, error) {
    // Determine shard based on query
    shardKey := sm.extractShardKey(query, args...)
    shardName := sm.router.GetShard(shardKey)
    
    // Get shard connection
    shard, exists := sm.shards[shardName]
    if !exists {
        return nil, errors.New("shard not found")
    }
    
    // Execute query
    return shard.Query(query, args...)
}

func (sm *ShardManager) ExecuteCrossShardQuery(query string, args ...interface{}) ([]*sql.Rows, error) {
    var allRows []*sql.Rows
    
    // Execute query on all shards
    for shardName, shard := range sm.shards {
        rows, err := shard.Query(query, args...)
        if err != nil {
            return nil, err
        }
        allRows = append(allRows, rows)
    }
    
    return allRows, nil
}
```

## 🔄 Data Replication Patterns

### Master-Slave Replication
```go
type ReplicationManager struct {
    master    *sql.DB
    slaves    []*sql.DB
    router    QueryRouter
    monitor   ReplicationMonitor
}

type QueryRouter struct {
    readWriteSplit bool
    slaveCount     int
    currentSlave   int
    mutex          sync.Mutex
}

func (qr *QueryRouter) GetConnection(query string) *sql.DB {
    if qr.isReadQuery(query) && qr.readWriteSplit {
        qr.mutex.Lock()
        defer qr.mutex.Unlock()
        
        // Round-robin for read queries
        slave := qr.slaves[qr.currentSlave]
        qr.currentSlave = (qr.currentSlave + 1) % qr.slaveCount
        return slave
    }
    
    // Write queries go to master
    return qr.master
}

func (qr *QueryRouter) isReadQuery(query string) bool {
    query = strings.ToUpper(strings.TrimSpace(query))
    return strings.HasPrefix(query, "SELECT") || 
           strings.HasPrefix(query, "SHOW") ||
           strings.HasPrefix(query, "DESCRIBE")
}

func (rm *ReplicationManager) ExecuteQuery(query string, args ...interface{}) (*sql.Rows, error) {
    conn := rm.router.GetConnection(query)
    return conn.Query(query, args...)
}

func (rm *ReplicationManager) ExecuteTransaction(queries []Query) error {
    // All transaction queries must go to master
    tx, err := rm.master.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    for _, query := range queries {
        if _, err := tx.Exec(query.SQL, query.Args...); err != nil {
            return err
        }
    }
    
    return tx.Commit()
}
```

### Multi-Master Replication
```go
type MultiMasterReplication struct {
    masters   []*sql.DB
    conflictResolver ConflictResolver
    vectorClock VectorClock
}

type ConflictResolver struct {
    strategies map[string]ConflictStrategy
}

type ConflictStrategy interface {
    Resolve(conflict Conflict) (interface{}, error)
}

type LastWriteWinsStrategy struct{}

func (lww *LastWriteWinsStrategy) Resolve(conflict Conflict) (interface{}, error) {
    // Compare timestamps
    if conflict.Left.Timestamp.After(conflict.Right.Timestamp) {
        return conflict.Left.Value, nil
    }
    return conflict.Right.Value, nil
}

type VectorClock struct {
    clocks map[string]int64
    mutex  sync.RWMutex
}

func (vc *VectorClock) Increment(nodeID string) {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    vc.clocks[nodeID]++
}

func (vc *VectorClock) Update(other VectorClock) {
    vc.mutex.Lock()
    defer vc.mutex.Unlock()
    
    for nodeID, clock := range other.clocks {
        if vc.clocks[nodeID] < clock {
            vc.clocks[nodeID] = clock
        }
    }
}

func (mmr *MultiMasterReplication) Write(key string, value interface{}) error {
    // Increment vector clock
    mmr.vectorClock.Increment("local")
    
    // Write to all masters
    for _, master := range mmr.masters {
        if err := mmr.writeToMaster(master, key, value); err != nil {
            return err
        }
    }
    
    return nil
}
```

## 📊 Data Partitioning

### Horizontal Partitioning
```go
type HorizontalPartitioner struct {
    partitions map[string]*sql.DB
    partitioner PartitionStrategy
}

type PartitionStrategy interface {
    GetPartition(key interface{}) string
}

type HashPartitioner struct {
    partitionCount int
}

func (hp *HashPartitioner) GetPartition(key interface{}) string {
    hash := hashFunction(key)
    partitionIndex := hash % hp.partitionCount
    return fmt.Sprintf("partition_%d", partitionIndex)
}

type RangePartitioner struct {
    ranges []PartitionRange
}

type PartitionRange struct {
    Name  string
    Min   interface{}
    Max   interface{}
}

func (rp *RangePartitioner) GetPartition(key interface{}) string {
    for _, r := range rp.ranges {
        if r.contains(key) {
            return r.Name
        }
    }
    return "default"
}

func (r PartitionRange) contains(key interface{}) bool {
    // Implement range checking logic
    return key >= r.Min && key < r.Max
}

func (hp *HorizontalPartitioner) Insert(table string, data map[string]interface{}) error {
    // Determine partition
    partitionKey := hp.extractPartitionKey(data)
    partition := hp.partitioner.GetPartition(partitionKey)
    
    // Get partition connection
    conn, exists := hp.partitions[partition]
    if !exists {
        return errors.New("partition not found")
    }
    
    // Insert data
    return hp.insertToPartition(conn, table, data)
}
```

### Vertical Partitioning
```go
type VerticalPartitioner struct {
    tables map[string]*sql.DB
    schema SchemaManager
}

type SchemaManager struct {
    tables map[string]TableSchema
}

type TableSchema struct {
    Name       string
    Columns    []Column
    Partitions []Partition
}

type Column struct {
    Name     string
    Type     string
    Required bool
}

type Partition struct {
    Name    string
    Columns []string
    DB      *sql.DB
}

func (vp *VerticalPartitioner) Insert(table string, data map[string]interface{}) error {
    schema := vp.schema.GetTableSchema(table)
    
    // Insert into each partition
    for _, partition := range schema.Partitions {
        partitionData := vp.filterDataForPartition(data, partition.Columns)
        if len(partitionData) > 0 {
            if err := vp.insertToPartition(partition.DB, table, partitionData); err != nil {
                return err
            }
        }
    }
    
    return nil
}

func (vp *VerticalPartitioner) filterDataForPartition(data map[string]interface{}, columns []string) map[string]interface{} {
    filtered := make(map[string]interface{})
    for _, col := range columns {
        if value, exists := data[col]; exists {
            filtered[col] = value
        }
    }
    return filtered
}
```

## 🔍 Query Optimization

### Query Plan Analysis
```go
type QueryAnalyzer struct {
    db *sql.DB
}

type QueryPlan struct {
    Query     string
    Cost      float64
    Rows      int64
    Time      time.Duration
    Plan      string
    Indexes   []string
}

func (qa *QueryAnalyzer) AnalyzeQuery(query string) (*QueryPlan, error) {
    // Get query plan
    plan, err := qa.getQueryPlan(query)
    if err != nil {
        return nil, err
    }
    
    // Analyze cost
    cost := qa.calculateCost(plan)
    
    // Analyze rows
    rows := qa.estimateRows(plan)
    
    // Analyze time
    time := qa.estimateTime(plan)
    
    // Extract indexes
    indexes := qa.extractIndexes(plan)
    
    return &QueryPlan{
        Query:   query,
        Cost:    cost,
        Rows:    rows,
        Time:    time,
        Plan:    plan,
        Indexes: indexes,
    }, nil
}

func (qa *QueryAnalyzer) getQueryPlan(query string) (string, error) {
    rows, err := qa.db.Query("EXPLAIN " + query)
    if err != nil {
        return "", err
    }
    defer rows.Close()
    
    var plan strings.Builder
    for rows.Next() {
        var line string
        if err := rows.Scan(&line); err != nil {
            return "", err
        }
        plan.WriteString(line + "\n")
    }
    
    return plan.String(), nil
}
```

### Index Optimization
```go
type IndexOptimizer struct {
    db        *sql.DB
    analyzer  QueryAnalyzer
    monitor   IndexMonitor
}

type IndexRecommendation struct {
    Table     string
    Columns   []string
    Type      string
    Reason    string
    Priority  int
}

func (io *IndexOptimizer) AnalyzeIndexes() ([]IndexRecommendation, error) {
    var recommendations []IndexRecommendation
    
    // Get slow queries
    slowQueries, err := io.monitor.GetSlowQueries()
    if err != nil {
        return nil, err
    }
    
    // Analyze each slow query
    for _, query := range slowQueries {
        plan, err := io.analyzer.AnalyzeQuery(query)
        if err != nil {
            continue
        }
        
        // Check if index would help
        if io.wouldIndexHelp(plan) {
            rec := io.generateIndexRecommendation(plan)
            recommendations = append(recommendations, rec)
        }
    }
    
    // Sort by priority
    sort.Slice(recommendations, func(i, j int) bool {
        return recommendations[i].Priority > recommendations[j].Priority
    })
    
    return recommendations, nil
}

func (io *IndexOptimizer) wouldIndexHelp(plan *QueryPlan) bool {
    // Check if query uses full table scan
    if strings.Contains(plan.Plan, "Seq Scan") {
        return true
    }
    
    // Check if query has high cost
    if plan.Cost > 1000 {
        return true
    }
    
    // Check if query returns many rows
    if plan.Rows > 10000 {
        return true
    }
    
    return false
}
```

## 🔄 Data Migration

### Zero-Downtime Migration
```go
type MigrationManager struct {
    sourceDB  *sql.DB
    targetDB  *sql.DB
    replicator DataReplicator
    validator DataValidator
}

type DataReplicator struct {
    batchSize int
    interval  time.Duration
}

func (mm *MigrationManager) MigrateTable(tableName string) error {
    // Start replication
    if err := mm.replicator.StartReplication(tableName); err != nil {
        return err
    }
    
    // Wait for initial sync
    if err := mm.waitForSync(tableName); err != nil {
        return err
    }
    
    // Switch traffic
    if err := mm.switchTraffic(tableName); err != nil {
        return err
    }
    
    // Stop replication
    if err := mm.replicator.StopReplication(tableName); err != nil {
        return err
    }
    
    return nil
}

func (mm *MigrationManager) waitForSync(tableName string) error {
    for {
        // Check if source and target are in sync
        if mm.validator.AreInSync(tableName) {
            return nil
        }
        
        time.Sleep(1 * time.Second)
    }
}

func (mm *MigrationManager) switchTraffic(tableName string) error {
    // Update application configuration
    // This would typically involve updating load balancer config
    // or application configuration to point to new database
    
    return nil
}
```

### Schema Migration
```go
type SchemaMigrator struct {
    db        *sql.DB
    migrations []Migration
    version   int
}

type Migration struct {
    Version   int
    Name      string
    Up        string
    Down      string
    Checksum  string
}

func (sm *SchemaMigrator) Migrate() error {
    // Get current version
    currentVersion, err := sm.getCurrentVersion()
    if err != nil {
        return err
    }
    
    // Apply pending migrations
    for _, migration := range sm.migrations {
        if migration.Version > currentVersion {
            if err := sm.applyMigration(migration); err != nil {
                return err
            }
        }
    }
    
    return nil
}

func (sm *SchemaMigrator) applyMigration(migration Migration) error {
    // Start transaction
    tx, err := sm.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    // Apply migration
    if _, err := tx.Exec(migration.Up); err != nil {
        return err
    }
    
    // Update version
    if _, err := tx.Exec("UPDATE schema_migrations SET version = ?", migration.Version); err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}
```

## 📊 Data Consistency

### Eventual Consistency
```go
type EventualConsistencyManager struct {
    replicas    []*sql.DB
    conflictResolver ConflictResolver
    vectorClock VectorClock
}

func (ecm *EventualConsistencyManager) Write(key string, value interface{}) error {
    // Write to all replicas
    for _, replica := range ecm.replicas {
        if err := ecm.writeToReplica(replica, key, value); err != nil {
            // Log error but continue
            log.Printf("Failed to write to replica: %v", err)
        }
    }
    
    return nil
}

func (ecm *EventualConsistencyManager) Read(key string) (interface{}, error) {
    // Read from any replica
    for _, replica := range ecm.replicas {
        value, err := ecm.readFromReplica(replica, key)
        if err == nil {
            return value, nil
        }
    }
    
    return nil, errors.New("all replicas failed")
}

func (ecm *EventualConsistencyManager) ReadConsistent(key string) (interface{}, error) {
    // Read from all replicas and resolve conflicts
    var values []interface{}
    
    for _, replica := range ecm.replicas {
        value, err := ecm.readFromReplica(replica, key)
        if err == nil {
            values = append(values, value)
        }
    }
    
    if len(values) == 0 {
        return nil, errors.New("no replicas available")
    }
    
    // Resolve conflicts
    return ecm.conflictResolver.Resolve(values), nil
}
```

### Strong Consistency
```go
type StrongConsistencyManager struct {
    master    *sql.DB
    slaves    []*sql.DB
    quorum    int
}

func (scm *StrongConsistencyManager) Write(key string, value interface{}) error {
    // Write to master
    if err := scm.writeToMaster(key, value); err != nil {
        return err
    }
    
    // Wait for replication to slaves
    if err := scm.waitForReplication(); err != nil {
        return err
    }
    
    return nil
}

func (scm *StrongConsistencyManager) Read(key string) (interface{}, error) {
    // Read from master for strong consistency
    return scm.readFromMaster(key)
}

func (scm *StrongConsistencyManager) waitForReplication() error {
    // Wait for at least quorum number of slaves to be in sync
    for {
        syncedSlaves := 0
        for _, slave := range scm.slaves {
            if scm.isSlaveInSync(slave) {
                syncedSlaves++
            }
        }
        
        if syncedSlaves >= scm.quorum {
            return nil
        }
        
        time.Sleep(100 * time.Millisecond)
    }
}
```

## 🎯 Best Practices

### Database Design
1. **Normalization**: Proper normalization for data integrity
2. **Indexing**: Strategic indexing for performance
3. **Partitioning**: Appropriate partitioning strategy
4. **Replication**: Proper replication setup
5. **Monitoring**: Comprehensive database monitoring

### Performance
1. **Query Optimization**: Optimize queries for performance
2. **Connection Pooling**: Use connection pooling
3. **Caching**: Implement appropriate caching
4. **Load Balancing**: Distribute load across replicas
5. **Monitoring**: Monitor performance metrics

### Security
1. **Access Control**: Implement proper access control
2. **Encryption**: Encrypt sensitive data
3. **Audit Logging**: Log all database operations
4. **Backup**: Regular backups and testing
5. **Compliance**: Ensure regulatory compliance

---

**Last Updated**: December 2024  
**Category**: Advanced Database Patterns  
**Complexity**: Senior Level
