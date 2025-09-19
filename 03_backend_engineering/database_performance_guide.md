# 🚀 Database Performance Optimization Guide

> **Advanced database performance tuning and optimization for senior backend engineers**

## 🎯 **Overview**

Database performance is critical for scalable backend systems. This guide covers query optimization, indexing strategies, connection pooling, monitoring, partitioning, and database-specific optimizations for PostgreSQL, MySQL, and MongoDB with practical implementations.

## 📚 **Table of Contents**

1. [Query Optimization Fundamentals](#query-optimization-fundamentals)
2. [Indexing Strategies](#indexing-strategies)
3. [Connection Pooling](#connection-pooling)
4. [Database Monitoring](#database-monitoring)
5. [Partitioning & Sharding](#partitioning--sharding)
6. [PostgreSQL Optimization](#postgresql-optimization)
7. [MySQL Optimization](#mysql-optimization)
8. [MongoDB Optimization](#mongodb-optimization)
9. [Caching Strategies](#caching-strategies)
10. [Performance Testing](#performance-testing)
11. [Interview Questions](#interview-questions)

---

## 🔍 **Query Optimization Fundamentals**

### **Query Analysis and Execution Plans**

```go
package dbperf

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "strings"
    "time"
    
    "github.com/lib/pq"
    _ "github.com/lib/pq"
)

// Query Performance Analyzer
type QueryAnalyzer struct {
    db     *sql.DB
    logger *log.Logger
}

func NewQueryAnalyzer(db *sql.DB) *QueryAnalyzer {
    return &QueryAnalyzer{
        db:     db,
        logger: log.New(os.Stdout, "[QueryAnalyzer] ", log.LstdFlags),
    }
}

// Execution Plan Analysis
type ExecutionPlan struct {
    Query            string        `json:"query"`
    ExecutionTime    time.Duration `json:"execution_time"`
    PlanningTime     time.Duration `json:"planning_time"`
    TotalCost        float64       `json:"total_cost"`
    ActualRows       int64         `json:"actual_rows"`
    EstimatedRows    int64         `json:"estimated_rows"`
    IndexesUsed      []string      `json:"indexes_used"`
    TableScans       []string      `json:"table_scans"`
    Recommendations  []string      `json:"recommendations"`
}

// Analyze query performance
func (qa *QueryAnalyzer) AnalyzeQuery(ctx context.Context, query string, args ...interface{}) (*ExecutionPlan, error) {
    // Get execution plan
    explainQuery := "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query
    
    start := time.Now()
    rows, err := qa.db.QueryContext(ctx, explainQuery, args...)
    if err != nil {
        return nil, fmt.Errorf("failed to execute EXPLAIN: %w", err)
    }
    defer rows.Close()
    
    executionTime := time.Since(start)
    
    // Parse execution plan
    plan, err := qa.parseExecutionPlan(rows)
    if err != nil {
        return nil, fmt.Errorf("failed to parse execution plan: %w", err)
    }
    
    plan.Query = query
    plan.ExecutionTime = executionTime
    
    // Generate recommendations
    plan.Recommendations = qa.generateRecommendations(plan)
    
    return plan, nil
}

// Parse PostgreSQL execution plan
func (qa *QueryAnalyzer) parseExecutionPlan(rows *sql.Rows) (*ExecutionPlan, error) {
    var planJSON string
    if rows.Next() {
        if err := rows.Scan(&planJSON); err != nil {
            return nil, err
        }
    }
    
    // Parse JSON plan (simplified implementation)
    plan := &ExecutionPlan{
        IndexesUsed: []string{},
        TableScans:  []string{},
    }
    
    // Extract key metrics from JSON plan
    // This would involve proper JSON parsing in a real implementation
    if strings.Contains(planJSON, "Seq Scan") {
        plan.TableScans = append(plan.TableScans, "Sequential scan detected")
    }
    
    if strings.Contains(planJSON, "Index Scan") {
        plan.IndexesUsed = append(plan.IndexesUsed, "Index scan used")
    }
    
    return plan, nil
}

// Generate optimization recommendations
func (qa *QueryAnalyzer) generateRecommendations(plan *ExecutionPlan) []string {
    var recommendations []string
    
    if len(plan.TableScans) > 0 {
        recommendations = append(recommendations, "Consider adding indexes to avoid table scans")
    }
    
    if plan.ActualRows > 0 && plan.EstimatedRows > 0 {
        ratio := float64(plan.ActualRows) / float64(plan.EstimatedRows)
        if ratio > 10 || ratio < 0.1 {
            recommendations = append(recommendations, "Statistics may be outdated - consider running ANALYZE")
        }
    }
    
    if plan.ExecutionTime > 100*time.Millisecond {
        recommendations = append(recommendations, "Query execution time is high - consider optimization")
    }
    
    return recommendations
}

// Query optimization patterns
type QueryOptimizer struct {
    analyzer *QueryAnalyzer
}

func NewQueryOptimizer(db *sql.DB) *QueryOptimizer {
    return &QueryOptimizer{
        analyzer: NewQueryAnalyzer(db),
    }
}

// Optimize SELECT queries
func (qo *QueryOptimizer) OptimizeSelect(table string, conditions map[string]interface{}, columns []string) string {
    var query strings.Builder
    
    // SELECT clause optimization
    if len(columns) == 0 {
        query.WriteString("SELECT *")
    } else {
        query.WriteString("SELECT ")
        query.WriteString(strings.Join(columns, ", "))
    }
    
    query.WriteString(" FROM ")
    query.WriteString(table)
    
    // WHERE clause optimization
    if len(conditions) > 0 {
        query.WriteString(" WHERE ")
        var conditionParts []string
        
        for column, value := range conditions {
            switch v := value.(type) {
            case []interface{}:
                // Use IN clause for arrays
                placeholders := make([]string, len(v))
                for i := range v {
                    placeholders[i] = "$" + fmt.Sprintf("%d", i+1)
                }
                conditionParts = append(conditionParts, fmt.Sprintf("%s IN (%s)", column, strings.Join(placeholders, ", ")))
            default:
                conditionParts = append(conditionParts, fmt.Sprintf("%s = $1", column))
            }
        }
        
        query.WriteString(strings.Join(conditionParts, " AND "))
    }
    
    return query.String()
}

// Batch query optimization
type BatchQueryOptimizer struct {
    db        *sql.DB
    batchSize int
}

func NewBatchQueryOptimizer(db *sql.DB, batchSize int) *BatchQueryOptimizer {
    return &BatchQueryOptimizer{
        db:        db,
        batchSize: batchSize,
    }
}

// Batch insert optimization
func (bqo *BatchQueryOptimizer) BatchInsert(ctx context.Context, table string, columns []string, values [][]interface{}) error {
    if len(values) == 0 {
        return nil
    }
    
    // Process in batches
    for i := 0; i < len(values); i += bqo.batchSize {
        end := i + bqo.batchSize
        if end > len(values) {
            end = len(values)
        }
        
        batch := values[i:end]
        if err := bqo.executeBatchInsert(ctx, table, columns, batch); err != nil {
            return fmt.Errorf("batch insert failed at index %d: %w", i, err)
        }
    }
    
    return nil
}

func (bqo *BatchQueryOptimizer) executeBatchInsert(ctx context.Context, table string, columns []string, batch [][]interface{}) error {
    // Build VALUES clause for batch insert
    var query strings.Builder
    query.WriteString("INSERT INTO ")
    query.WriteString(table)
    query.WriteString(" (")
    query.WriteString(strings.Join(columns, ", "))
    query.WriteString(") VALUES ")
    
    valuePlaceholders := make([]string, len(batch))
    var args []interface{}
    argIndex := 1
    
    for i, row := range batch {
        rowPlaceholders := make([]string, len(row))
        for j, value := range row {
            rowPlaceholders[j] = fmt.Sprintf("$%d", argIndex)
            args = append(args, value)
            argIndex++
        }
        valuePlaceholders[i] = "(" + strings.Join(rowPlaceholders, ", ") + ")"
    }
    
    query.WriteString(strings.Join(valuePlaceholders, ", "))
    
    _, err := bqo.db.ExecContext(ctx, query.String(), args...)
    return err
}

// Prepared statement manager for performance
type PreparedStatementManager struct {
    db         *sql.DB
    statements map[string]*sql.Stmt
    mu         sync.RWMutex
}

func NewPreparedStatementManager(db *sql.DB) *PreparedStatementManager {
    return &PreparedStatementManager{
        db:         db,
        statements: make(map[string]*sql.Stmt),
    }
}

func (psm *PreparedStatementManager) GetStatement(key, query string) (*sql.Stmt, error) {
    psm.mu.RLock()
    if stmt, exists := psm.statements[key]; exists {
        psm.mu.RUnlock()
        return stmt, nil
    }
    psm.mu.RUnlock()
    
    psm.mu.Lock()
    defer psm.mu.Unlock()
    
    // Double-check after acquiring write lock
    if stmt, exists := psm.statements[key]; exists {
        return stmt, nil
    }
    
    stmt, err := psm.db.Prepare(query)
    if err != nil {
        return nil, fmt.Errorf("failed to prepare statement: %w", err)
    }
    
    psm.statements[key] = stmt
    return stmt, nil
}

func (psm *PreparedStatementManager) Close() error {
    psm.mu.Lock()
    defer psm.mu.Unlock()
    
    for _, stmt := range psm.statements {
        stmt.Close()
    }
    psm.statements = make(map[string]*sql.Stmt)
    return nil
}
```

---

## 🔗 **Indexing Strategies**

### **Advanced Index Design**

```go
package indexing

import (
    "context"
    "database/sql"
    "fmt"
    "strings"
    "time"
)

// Index Analyzer
type IndexAnalyzer struct {
    db *sql.DB
}

func NewIndexAnalyzer(db *sql.DB) *IndexAnalyzer {
    return &IndexAnalyzer{db: db}
}

// Index Usage Statistics
type IndexUsage struct {
    SchemaName    string    `json:"schema_name"`
    TableName     string    `json:"table_name"`
    IndexName     string    `json:"index_name"`
    IndexType     string    `json:"index_type"`
    Columns       []string  `json:"columns"`
    ScanCount     int64     `json:"scan_count"`
    TupleReads    int64     `json:"tuple_reads"`
    TupleFetches  int64     `json:"tuple_fetches"`
    Size          int64     `json:"size_bytes"`
    LastUsed      *time.Time `json:"last_used"`
    Selectivity   float64   `json:"selectivity"`
    Efficiency    float64   `json:"efficiency"`
}

// Analyze index usage
func (ia *IndexAnalyzer) AnalyzeIndexUsage(ctx context.Context) ([]IndexUsage, error) {
    query := `
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch,
            pg_relation_size(indexrelid) as size_bytes
        FROM pg_stat_user_indexes
        ORDER BY idx_scan DESC, idx_tup_read DESC
    `
    
    rows, err := ia.db.QueryContext(ctx, query)
    if err != nil {
        return nil, fmt.Errorf("failed to query index usage: %w", err)
    }
    defer rows.Close()
    
    var usages []IndexUsage
    for rows.Next() {
        var usage IndexUsage
        err := rows.Scan(
            &usage.SchemaName,
            &usage.TableName,
            &usage.IndexName,
            &usage.ScanCount,
            &usage.TupleReads,
            &usage.TupleFetches,
            &usage.Size,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan index usage: %w", err)
        }
        
        // Calculate efficiency
        if usage.ScanCount > 0 {
            usage.Efficiency = float64(usage.TupleReads) / float64(usage.ScanCount)
        }
        
        usages = append(usages, usage)
    }
    
    return usages, nil
}

// Find unused indexes
func (ia *IndexAnalyzer) FindUnusedIndexes(ctx context.Context) ([]IndexUsage, error) {
    query := `
        SELECT 
            schemaname,
            tablename,
            indexname,
            pg_relation_size(indexrelid) as size_bytes
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
        AND indexname NOT LIKE '%_pkey'  -- Exclude primary keys
        ORDER BY size_bytes DESC
    `
    
    rows, err := ia.db.QueryContext(ctx, query)
    if err != nil {
        return nil, fmt.Errorf("failed to query unused indexes: %w", err)
    }
    defer rows.Close()
    
    var unusedIndexes []IndexUsage
    for rows.Next() {
        var usage IndexUsage
        err := rows.Scan(
            &usage.SchemaName,
            &usage.TableName,
            &usage.IndexName,
            &usage.Size,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan unused index: %w", err)
        }
        
        unusedIndexes = append(unusedIndexes, usage)
    }
    
    return unusedIndexes, nil
}

// Index Recommendation Engine
type IndexRecommendationEngine struct {
    db       *sql.DB
    analyzer *IndexAnalyzer
}

func NewIndexRecommendationEngine(db *sql.DB) *IndexRecommendationEngine {
    return &IndexRecommendationEngine{
        db:       db,
        analyzer: NewIndexAnalyzer(db),
    }
}

type IndexRecommendation struct {
    Table       string    `json:"table"`
    Columns     []string  `json:"columns"`
    IndexType   string    `json:"index_type"`
    Reason      string    `json:"reason"`
    Priority    string    `json:"priority"`
    CreateSQL   string    `json:"create_sql"`
    EstimatedBenefit string `json:"estimated_benefit"`
}

// Generate index recommendations based on slow queries
func (ire *IndexRecommendationEngine) GenerateRecommendations(ctx context.Context) ([]IndexRecommendation, error) {
    // Find slow queries from pg_stat_statements
    slowQueries, err := ire.findSlowQueries(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to find slow queries: %w", err)
    }
    
    var recommendations []IndexRecommendation
    
    for _, query := range slowQueries {
        recs := ire.analyzeQueryForIndexes(query)
        recommendations = append(recommendations, recs...)
    }
    
    return recommendations, nil
}

type SlowQuery struct {
    Query        string        `json:"query"`
    Calls        int64         `json:"calls"`
    TotalTime    time.Duration `json:"total_time"`
    MeanTime     time.Duration `json:"mean_time"`
    Rows         int64         `json:"rows"`
}

func (ire *IndexRecommendationEngine) findSlowQueries(ctx context.Context) ([]SlowQuery, error) {
    query := `
        SELECT 
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            rows
        FROM pg_stat_statements
        WHERE mean_exec_time > 100  -- Queries taking more than 100ms on average
        ORDER BY mean_exec_time DESC
        LIMIT 50
    `
    
    rows, err := ire.db.QueryContext(ctx, query)
    if err != nil {
        return nil, fmt.Errorf("failed to query slow queries: %w", err)
    }
    defer rows.Close()
    
    var slowQueries []SlowQuery
    for rows.Next() {
        var sq SlowQuery
        var totalTimeMs, meanTimeMs float64
        
        err := rows.Scan(
            &sq.Query,
            &sq.Calls,
            &totalTimeMs,
            &meanTimeMs,
            &sq.Rows,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan slow query: %w", err)
        }
        
        sq.TotalTime = time.Duration(totalTimeMs * float64(time.Millisecond))
        sq.MeanTime = time.Duration(meanTimeMs * float64(time.Millisecond))
        
        slowQueries = append(slowQueries, sq)
    }
    
    return slowQueries, nil
}

func (ire *IndexRecommendationEngine) analyzeQueryForIndexes(query SlowQuery) []IndexRecommendation {
    var recommendations []IndexRecommendation
    
    // Simple heuristic-based analysis (in practice, you'd use a query parser)
    queryLower := strings.ToLower(query.Query)
    
    // Look for WHERE clauses
    if strings.Contains(queryLower, "where") {
        recommendations = append(recommendations, IndexRecommendation{
            Table:     extractTableName(queryLower),
            Columns:   extractWhereColumns(queryLower),
            IndexType: "btree",
            Reason:    "Query has WHERE clause conditions",
            Priority:  "high",
            CreateSQL: generateCreateIndexSQL("btree", extractTableName(queryLower), extractWhereColumns(queryLower)),
            EstimatedBenefit: fmt.Sprintf("Could reduce execution time from %v", query.MeanTime),
        })
    }
    
    // Look for ORDER BY clauses
    if strings.Contains(queryLower, "order by") {
        recommendations = append(recommendations, IndexRecommendation{
            Table:     extractTableName(queryLower),
            Columns:   extractOrderByColumns(queryLower),
            IndexType: "btree",
            Reason:    "Query has ORDER BY clause",
            Priority:  "medium",
            CreateSQL: generateCreateIndexSQL("btree", extractTableName(queryLower), extractOrderByColumns(queryLower)),
            EstimatedBenefit: "Could eliminate sorting step",
        })
    }
    
    // Look for JOIN conditions
    if strings.Contains(queryLower, "join") {
        recommendations = append(recommendations, IndexRecommendation{
            Table:     extractTableName(queryLower),
            Columns:   extractJoinColumns(queryLower),
            IndexType: "btree",
            Reason:    "Query performs JOINs",
            Priority:  "high",
            CreateSQL: generateCreateIndexSQL("btree", extractTableName(queryLower), extractJoinColumns(queryLower)),
            EstimatedBenefit: "Could improve JOIN performance",
        })
    }
    
    return recommendations
}

// Composite Index Optimizer
type CompositeIndexOptimizer struct {
    db *sql.DB
}

func NewCompositeIndexOptimizer(db *sql.DB) *CompositeIndexOptimizer {
    return &CompositeIndexOptimizer{db: db}
}

// Optimize column order in composite indexes
func (cio *CompositeIndexOptimizer) OptimizeColumnOrder(table string, columns []string) ([]string, error) {
    // Rule 1: Most selective columns first
    selectivities := make(map[string]float64)
    
    for _, column := range columns {
        selectivity, err := cio.calculateSelectivity(table, column)
        if err != nil {
            return nil, fmt.Errorf("failed to calculate selectivity for %s: %w", column, err)
        }
        selectivities[column] = selectivity
    }
    
    // Sort columns by selectivity (most selective first)
    optimizedColumns := make([]string, len(columns))
    copy(optimizedColumns, columns)
    
    // Simple bubble sort by selectivity
    for i := 0; i < len(optimizedColumns)-1; i++ {
        for j := 0; j < len(optimizedColumns)-i-1; j++ {
            if selectivities[optimizedColumns[j]] > selectivities[optimizedColumns[j+1]] {
                optimizedColumns[j], optimizedColumns[j+1] = optimizedColumns[j+1], optimizedColumns[j]
            }
        }
    }
    
    return optimizedColumns, nil
}

func (cio *CompositeIndexOptimizer) calculateSelectivity(table, column string) (float64, error) {
    query := `
        SELECT 
            (SELECT COUNT(DISTINCT %s) FROM %s)::float / 
            (SELECT COUNT(*) FROM %s)::float as selectivity
    `
    
    var selectivity float64
    err := cio.db.QueryRow(fmt.Sprintf(query, column, table, table)).Scan(&selectivity)
    if err != nil {
        return 0, fmt.Errorf("failed to calculate selectivity: %w", err)
    }
    
    return selectivity, nil
}

// Partial Index Manager
type PartialIndexManager struct {
    db *sql.DB
}

func NewPartialIndexManager(db *sql.DB) *PartialIndexManager {
    return &PartialIndexManager{db: db}
}

// Create partial index with condition
func (pim *PartialIndexManager) CreatePartialIndex(table, indexName string, columns []string, condition string) error {
    query := fmt.Sprintf(
        "CREATE INDEX CONCURRENTLY %s ON %s (%s) WHERE %s",
        indexName,
        table,
        strings.Join(columns, ", "),
        condition,
    )
    
    _, err := pim.db.Exec(query)
    if err != nil {
        return fmt.Errorf("failed to create partial index: %w", err)
    }
    
    return nil
}

// Suggest partial indexes based on data distribution
func (pim *PartialIndexManager) SuggestPartialIndexes(ctx context.Context, table string) ([]IndexRecommendation, error) {
    // Find columns with skewed distributions
    skewedColumns, err := pim.findSkewedColumns(ctx, table)
    if err != nil {
        return nil, fmt.Errorf("failed to find skewed columns: %w", err)
    }
    
    var recommendations []IndexRecommendation
    
    for _, column := range skewedColumns {
        // Suggest partial index for non-null values if many nulls exist
        nullRatio, err := pim.calculateNullRatio(table, column.Name)
        if err != nil {
            continue
        }
        
        if nullRatio > 0.1 { // More than 10% nulls
            recommendations = append(recommendations, IndexRecommendation{
                Table:     table,
                Columns:   []string{column.Name},
                IndexType: "btree",
                Reason:    fmt.Sprintf("Column %s has %.1f%% null values", column.Name, nullRatio*100),
                Priority:  "medium",
                CreateSQL: fmt.Sprintf("CREATE INDEX CONCURRENTLY idx_%s_%s_partial ON %s (%s) WHERE %s IS NOT NULL", table, column.Name, table, column.Name, column.Name),
                EstimatedBenefit: "Reduces index size and improves performance for non-null queries",
            })
        }
    }
    
    return recommendations, nil
}

type SkewedColumn struct {
    Name        string  `json:"name"`
    Cardinality int64   `json:"cardinality"`
    TotalRows   int64   `json:"total_rows"`
    Selectivity float64 `json:"selectivity"`
}

func (pim *PartialIndexManager) findSkewedColumns(ctx context.Context, table string) ([]SkewedColumn, error) {
    // This would involve analyzing column statistics
    // Simplified implementation
    query := `
        SELECT 
            column_name,
            n_distinct,
            null_frac
        FROM pg_stats 
        WHERE tablename = $1
        AND n_distinct > 0
        ORDER BY n_distinct ASC
    `
    
    rows, err := pim.db.QueryContext(ctx, query, table)
    if err != nil {
        return nil, fmt.Errorf("failed to query column stats: %w", err)
    }
    defer rows.Close()
    
    var columns []SkewedColumn
    for rows.Next() {
        var col SkewedColumn
        var nDistinct float64
        var nullFrac float64
        
        err := rows.Scan(&col.Name, &nDistinct, &nullFrac)
        if err != nil {
            continue
        }
        
        col.Cardinality = int64(nDistinct)
        col.Selectivity = nDistinct / 100000 // Approximate, would need actual row count
        
        columns = append(columns, col)
    }
    
    return columns, nil
}

func (pim *PartialIndexManager) calculateNullRatio(table, column string) (float64, error) {
    query := fmt.Sprintf(`
        SELECT 
            COUNT(CASE WHEN %s IS NULL THEN 1 END)::float / COUNT(*)::float as null_ratio
        FROM %s
    `, column, table)
    
    var nullRatio float64
    err := pim.db.QueryRow(query).Scan(&nullRatio)
    if err != nil {
        return 0, fmt.Errorf("failed to calculate null ratio: %w", err)
    }
    
    return nullRatio, nil
}

// Helper functions for query parsing (simplified implementations)
func extractTableName(query string) string {
    // Simplified table name extraction
    words := strings.Fields(query)
    for i, word := range words {
        if strings.ToLower(word) == "from" && i+1 < len(words) {
            return words[i+1]
        }
    }
    return "unknown_table"
}

func extractWhereColumns(query string) []string {
    // Simplified WHERE column extraction
    return []string{"extracted_column"}
}

func extractOrderByColumns(query string) []string {
    // Simplified ORDER BY column extraction
    return []string{"order_column"}
}

func extractJoinColumns(query string) []string {
    // Simplified JOIN column extraction
    return []string{"join_column"}
}

func generateCreateIndexSQL(indexType, table string, columns []string) string {
    indexName := fmt.Sprintf("idx_%s_%s", table, strings.Join(columns, "_"))
    return fmt.Sprintf("CREATE INDEX CONCURRENTLY %s ON %s USING %s (%s)", indexName, table, indexType, strings.Join(columns, ", "))
}
```

---

## 🏊 **Connection Pooling**

### **Advanced Connection Pool Implementation**

```go
package connpool

import (
    "context"
    "database/sql"
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// Connection Pool Configuration
type PoolConfig struct {
    MaxOpenConns        int           `json:"max_open_conns"`
    MaxIdleConns        int           `json:"max_idle_conns"`
    ConnMaxLifetime     time.Duration `json:"conn_max_lifetime"`
    ConnMaxIdleTime     time.Duration `json:"conn_max_idle_time"`
    HealthCheckInterval time.Duration `json:"health_check_interval"`
    ConnectionTimeout   time.Duration `json:"connection_timeout"`
    QueryTimeout        time.Duration `json:"query_timeout"`
    RetryAttempts       int           `json:"retry_attempts"`
    RetryDelay          time.Duration `json:"retry_delay"`
}

// Default configuration
func DefaultPoolConfig() PoolConfig {
    return PoolConfig{
        MaxOpenConns:        25,
        MaxIdleConns:        5,
        ConnMaxLifetime:     1 * time.Hour,
        ConnMaxIdleTime:     15 * time.Minute,
        HealthCheckInterval: 30 * time.Second,
        ConnectionTimeout:   10 * time.Second,
        QueryTimeout:        30 * time.Second,
        RetryAttempts:       3,
        RetryDelay:          1 * time.Second,
    }
}

// Connection Pool Manager
type ConnectionPoolManager struct {
    config     PoolConfig
    db         *sql.DB
    stats      *PoolStats
    healthTicker *time.Ticker
    stopChan   chan struct{}
    mu         sync.RWMutex
}

// Pool Statistics
type PoolStats struct {
    OpenConnections     int64 `json:"open_connections"`
    InUseConnections    int64 `json:"in_use_connections"`
    IdleConnections     int64 `json:"idle_connections"`
    WaitCount           int64 `json:"wait_count"`
    WaitDuration        int64 `json:"wait_duration_ms"`
    MaxIdleClosed       int64 `json:"max_idle_closed"`
    MaxLifetimeClosed   int64 `json:"max_lifetime_closed"`
    TotalQueries        int64 `json:"total_queries"`
    FailedQueries       int64 `json:"failed_queries"`
    AvgQueryTime        int64 `json:"avg_query_time_ms"`
    ConnectionErrors    int64 `json:"connection_errors"`
    HealthCheckFailures int64 `json:"health_check_failures"`
}

func NewConnectionPoolManager(dsn string, config PoolConfig) (*ConnectionPoolManager, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    cpm := &ConnectionPoolManager{
        config:   config,
        db:       db,
        stats:    &PoolStats{},
        stopChan: make(chan struct{}),
    }
    
    // Start health monitoring
    cpm.startHealthMonitoring()
    
    return cpm, nil
}

// Get database connection
func (cpm *ConnectionPoolManager) GetDB() *sql.DB {
    return cpm.db
}

// Execute query with retry logic and statistics tracking
func (cpm *ConnectionPoolManager) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    start := time.Now()
    atomic.AddInt64(&cpm.stats.TotalQueries, 1)
    
    var rows *sql.Rows
    var err error
    
    for attempt := 0; attempt <= cpm.config.RetryAttempts; attempt++ {
        // Apply query timeout
        queryCtx, cancel := context.WithTimeout(ctx, cpm.config.QueryTimeout)
        
        rows, err = cpm.db.QueryContext(queryCtx, query, args...)
        cancel()
        
        if err == nil {
            break
        }
        
        // Check if error is retryable
        if !isRetryableError(err) {
            break
        }
        
        if attempt < cpm.config.RetryAttempts {
            time.Sleep(cpm.config.RetryDelay * time.Duration(attempt+1))
        }
    }
    
    // Update statistics
    queryDuration := time.Since(start)
    cpm.updateQueryStats(queryDuration, err)
    
    if err != nil {
        atomic.AddInt64(&cpm.stats.FailedQueries, 1)
        return nil, fmt.Errorf("query failed after %d attempts: %w", cpm.config.RetryAttempts+1, err)
    }
    
    return rows, nil
}

// Execute query with single result
func (cpm *ConnectionPoolManager) QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row {
    start := time.Now()
    atomic.AddInt64(&cpm.stats.TotalQueries, 1)
    
    queryCtx, cancel := context.WithTimeout(ctx, cpm.config.QueryTimeout)
    defer cancel()
    
    row := cpm.db.QueryRowContext(queryCtx, query, args...)
    
    queryDuration := time.Since(start)
    cpm.updateQueryStats(queryDuration, nil)
    
    return row
}

// Execute statement
func (cpm *ConnectionPoolManager) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    start := time.Now()
    atomic.AddInt64(&cpm.stats.TotalQueries, 1)
    
    var result sql.Result
    var err error
    
    for attempt := 0; attempt <= cpm.config.RetryAttempts; attempt++ {
        queryCtx, cancel := context.WithTimeout(ctx, cpm.config.QueryTimeout)
        
        result, err = cpm.db.ExecContext(queryCtx, query, args...)
        cancel()
        
        if err == nil {
            break
        }
        
        if !isRetryableError(err) {
            break
        }
        
        if attempt < cpm.config.RetryAttempts {
            time.Sleep(cpm.config.RetryDelay * time.Duration(attempt+1))
        }
    }
    
    queryDuration := time.Since(start)
    cpm.updateQueryStats(queryDuration, err)
    
    if err != nil {
        atomic.AddInt64(&cpm.stats.FailedQueries, 1)
        return nil, fmt.Errorf("exec failed after %d attempts: %w", cpm.config.RetryAttempts+1, err)
    }
    
    return result, nil
}

// Get pool statistics
func (cpm *ConnectionPoolManager) GetStats() PoolStats {
    cpm.mu.RLock()
    defer cpm.mu.RUnlock()
    
    dbStats := cpm.db.Stats()
    
    return PoolStats{
        OpenConnections:     int64(dbStats.OpenConnections),
        InUseConnections:    int64(dbStats.InUse),
        IdleConnections:     int64(dbStats.Idle),
        WaitCount:           dbStats.WaitCount,
        WaitDuration:        dbStats.WaitDuration.Nanoseconds() / 1000000, // Convert to ms
        MaxIdleClosed:       dbStats.MaxIdleClosed,
        MaxLifetimeClosed:   dbStats.MaxLifetimeClosed,
        TotalQueries:        atomic.LoadInt64(&cpm.stats.TotalQueries),
        FailedQueries:       atomic.LoadInt64(&cpm.stats.FailedQueries),
        AvgQueryTime:        atomic.LoadInt64(&cpm.stats.AvgQueryTime),
        ConnectionErrors:    atomic.LoadInt64(&cpm.stats.ConnectionErrors),
        HealthCheckFailures: atomic.LoadInt64(&cpm.stats.HealthCheckFailures),
    }
}

// Start health monitoring
func (cpm *ConnectionPoolManager) startHealthMonitoring() {
    cpm.healthTicker = time.NewTicker(cpm.config.HealthCheckInterval)
    
    go func() {
        for {
            select {
            case <-cpm.healthTicker.C:
                cpm.performHealthCheck()
            case <-cpm.stopChan:
                cpm.healthTicker.Stop()
                return
            }
        }
    }()
}

// Perform health check
func (cpm *ConnectionPoolManager) performHealthCheck() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := cpm.db.PingContext(ctx); err != nil {
        atomic.AddInt64(&cpm.stats.HealthCheckFailures, 1)
        atomic.AddInt64(&cpm.stats.ConnectionErrors, 1)
    }
}

// Update query statistics
func (cpm *ConnectionPoolManager) updateQueryStats(duration time.Duration, err error) {
    durationMs := duration.Nanoseconds() / 1000000
    
    // Update average query time using exponential moving average
    oldAvg := atomic.LoadInt64(&cpm.stats.AvgQueryTime)
    newAvg := (oldAvg*9 + durationMs) / 10 // Simple EMA with alpha = 0.1
    atomic.StoreInt64(&cpm.stats.AvgQueryTime, newAvg)
    
    if err != nil {
        atomic.AddInt64(&cpm.stats.ConnectionErrors, 1)
    }
}

// Close connection pool
func (cpm *ConnectionPoolManager) Close() error {
    close(cpm.stopChan)
    if cpm.healthTicker != nil {
        cpm.healthTicker.Stop()
    }
    return cpm.db.Close()
}

// Check if error is retryable
func isRetryableError(err error) bool {
    if err == nil {
        return false
    }
    
    errStr := err.Error()
    // Common retryable database errors
    retryableErrors := []string{
        "connection refused",
        "connection reset",
        "timeout",
        "temporary failure",
        "deadlock",
        "lock timeout",
    }
    
    for _, retryableErr := range retryableErrors {
        if strings.Contains(strings.ToLower(errStr), retryableErr) {
            return true
        }
    }
    
    return false
}

// Connection Pool Monitor
type PoolMonitor struct {
    pools map[string]*ConnectionPoolManager
    mu    sync.RWMutex
}

func NewPoolMonitor() *PoolMonitor {
    return &PoolMonitor{
        pools: make(map[string]*ConnectionPoolManager),
    }
}

func (pm *PoolMonitor) RegisterPool(name string, pool *ConnectionPoolManager) {
    pm.mu.Lock()
    defer pm.mu.Unlock()
    pm.pools[name] = pool
}

func (pm *PoolMonitor) GetAllStats() map[string]PoolStats {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    stats := make(map[string]PoolStats)
    for name, pool := range pm.pools {
        stats[name] = pool.GetStats()
    }
    
    return stats
}

// Pool health assessment
func (pm *PoolMonitor) AssessPoolHealth() map[string]string {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    health := make(map[string]string)
    
    for name, pool := range pm.pools {
        stats := pool.GetStats()
        
        var status string
        var issues []string
        
        // Check various health indicators
        if stats.FailedQueries > 0 && float64(stats.FailedQueries)/float64(stats.TotalQueries) > 0.05 {
            issues = append(issues, "high failure rate")
        }
        
        if stats.AvgQueryTime > 1000 { // 1 second
            issues = append(issues, "slow queries")
        }
        
        if stats.WaitCount > 0 && stats.WaitDuration > 1000 { // 1 second total wait
            issues = append(issues, "connection contention")
        }
        
        if stats.HealthCheckFailures > 0 {
            issues = append(issues, "connectivity issues")
        }
        
        if len(issues) == 0 {
            status = "healthy"
        } else {
            status = "warning: " + strings.Join(issues, ", ")
        }
        
        health[name] = status
    }
    
    return health
}
```

---

## 📊 **Database Monitoring**

### **Performance Metrics Collection**

```go
package monitoring

import (
    "context"
    "database/sql"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Database Performance Metrics
type DatabaseMetrics struct {
    Timestamp           time.Time `json:"timestamp"`
    ActiveConnections   int64     `json:"active_connections"`
    IdleConnections     int64     `json:"idle_connections"`
    TotalConnections    int64     `json:"total_connections"`
    QueriesPerSecond    float64   `json:"queries_per_second"`
    AvgQueryTime        float64   `json:"avg_query_time_ms"`
    SlowQueries         int64     `json:"slow_queries"`
    CacheHitRatio       float64   `json:"cache_hit_ratio"`
    BufferHitRatio      float64   `json:"buffer_hit_ratio"`
    IndexHitRatio       float64   `json:"index_hit_ratio"`
    LockWaits           int64     `json:"lock_waits"`
    Deadlocks           int64     `json:"deadlocks"`
    TempFiles           int64     `json:"temp_files"`
    TempBytes           int64     `json:"temp_bytes"`
    CheckpointWriteTime float64   `json:"checkpoint_write_time"`
    WALSize             int64     `json:"wal_size"`
}

// Database Monitor
type DatabaseMonitor struct {
    db          *sql.DB
    metrics     []DatabaseMetrics
    mu          sync.RWMutex
    stopChan    chan struct{}
    interval    time.Duration
    retention   time.Duration
}

func NewDatabaseMonitor(db *sql.DB, interval, retention time.Duration) *DatabaseMonitor {
    return &DatabaseMonitor{
        db:        db,
        metrics:   make([]DatabaseMetrics, 0),
        stopChan:  make(chan struct{}),
        interval:  interval,
        retention: retention,
    }
}

// Start monitoring
func (dm *DatabaseMonitor) Start() {
    go dm.monitorLoop()
}

func (dm *DatabaseMonitor) monitorLoop() {
    ticker := time.NewTicker(dm.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            if metrics, err := dm.collectMetrics(); err == nil {
                dm.addMetrics(metrics)
                dm.cleanupOldMetrics()
            }
        case <-dm.stopChan:
            return
        }
    }
}

// Collect current metrics
func (dm *DatabaseMonitor) collectMetrics() (DatabaseMetrics, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    metrics := DatabaseMetrics{
        Timestamp: time.Now(),
    }
    
    // Collect PostgreSQL-specific metrics
    if err := dm.collectPostgreSQLMetrics(ctx, &metrics); err != nil {
        return metrics, fmt.Errorf("failed to collect PostgreSQL metrics: %w", err)
    }
    
    return metrics, nil
}

func (dm *DatabaseMonitor) collectPostgreSQLMetrics(ctx context.Context, metrics *DatabaseMetrics) error {
    // Active connections
    var activeConns sql.NullInt64
    err := dm.db.QueryRowContext(ctx, "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'").Scan(&activeConns)
    if err != nil {
        return fmt.Errorf("failed to get active connections: %w", err)
    }
    if activeConns.Valid {
        metrics.ActiveConnections = activeConns.Int64
    }
    
    // Total connections
    var totalConns sql.NullInt64
    err = dm.db.QueryRowContext(ctx, "SELECT count(*) FROM pg_stat_activity").Scan(&totalConns)
    if err != nil {
        return fmt.Errorf("failed to get total connections: %w", err)
    }
    if totalConns.Valid {
        metrics.TotalConnections = totalConns.Int64
        metrics.IdleConnections = totalConns.Int64 - metrics.ActiveConnections
    }
    
    // Cache hit ratio
    var bufferHits, bufferReads sql.NullInt64
    err = dm.db.QueryRowContext(ctx, `
        SELECT 
            sum(blks_hit) as buffer_hits,
            sum(blks_read) as buffer_reads
        FROM pg_stat_database
    `).Scan(&bufferHits, &bufferReads)
    
    if err != nil {
        return fmt.Errorf("failed to get buffer stats: %w", err)
    }
    
    if bufferHits.Valid && bufferReads.Valid && (bufferHits.Int64+bufferReads.Int64) > 0 {
        metrics.BufferHitRatio = float64(bufferHits.Int64) / float64(bufferHits.Int64+bufferReads.Int64) * 100
    }
    
    // Query statistics
    var totalQueries, slowQueries sql.NullInt64
    var avgQueryTime sql.NullFloat64
    
    err = dm.db.QueryRowContext(ctx, `
        SELECT 
            sum(calls) as total_queries,
            avg(mean_exec_time) as avg_time,
            sum(CASE WHEN mean_exec_time > 1000 THEN calls ELSE 0 END) as slow_queries
        FROM pg_stat_statements
    `).Scan(&totalQueries, &avgQueryTime, &slowQueries)
    
    if err == nil {
        if slowQueries.Valid {
            metrics.SlowQueries = slowQueries.Int64
        }
        if avgQueryTime.Valid {
            metrics.AvgQueryTime = avgQueryTime.Float64
        }
    }
    
    // Lock statistics
    var lockWaits sql.NullInt64
    err = dm.db.QueryRowContext(ctx, `
        SELECT count(*) 
        FROM pg_stat_activity 
        WHERE wait_event_type = 'Lock'
    `).Scan(&lockWaits)
    
    if err == nil && lockWaits.Valid {
        metrics.LockWaits = lockWaits.Int64
    }
    
    // WAL size
    var walSize sql.NullInt64
    err = dm.db.QueryRowContext(ctx, `
        SELECT sum(size) 
        FROM pg_ls_waldir()
    `).Scan(&walSize)
    
    if err == nil && walSize.Valid {
        metrics.WALSize = walSize.Int64
    }
    
    return nil
}

// Add metrics to history
func (dm *DatabaseMonitor) addMetrics(metrics DatabaseMetrics) {
    dm.mu.Lock()
    defer dm.mu.Unlock()
    
    dm.metrics = append(dm.metrics, metrics)
}

// Clean up old metrics
func (dm *DatabaseMonitor) cleanupOldMetrics() {
    dm.mu.Lock()
    defer dm.mu.Unlock()
    
    cutoff := time.Now().Add(-dm.retention)
    
    // Find first metric to keep
    keepIndex := 0
    for i, metric := range dm.metrics {
        if metric.Timestamp.After(cutoff) {
            keepIndex = i
            break
        }
    }
    
    if keepIndex > 0 {
        dm.metrics = dm.metrics[keepIndex:]
    }
}

// Get recent metrics
func (dm *DatabaseMonitor) GetMetrics(duration time.Duration) []DatabaseMetrics {
    dm.mu.RLock()
    defer dm.mu.RUnlock()
    
    since := time.Now().Add(-duration)
    var result []DatabaseMetrics
    
    for _, metric := range dm.metrics {
        if metric.Timestamp.After(since) {
            result = append(result, metric)
        }
    }
    
    return result
}

// Performance Alert System
type AlertThreshold struct {
    MetricName string  `json:"metric_name"`
    Operator   string  `json:"operator"` // "gt", "lt", "eq"
    Value      float64 `json:"value"`
    Duration   time.Duration `json:"duration"`
}

type Alert struct {
    ID          string                 `json:"id"`
    Timestamp   time.Time             `json:"timestamp"`
    Threshold   AlertThreshold        `json:"threshold"`
    ActualValue float64               `json:"actual_value"`
    Severity    string                `json:"severity"`
    Message     string                `json:"message"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type AlertManager struct {
    thresholds []AlertThreshold
    alerts     []Alert
    mu         sync.RWMutex
    callbacks  []func(Alert)
}

func NewAlertManager() *AlertManager {
    return &AlertManager{
        thresholds: make([]AlertThreshold, 0),
        alerts:     make([]Alert, 0),
        callbacks:  make([]func(Alert), 0),
    }
}

// Add alert threshold
func (am *AlertManager) AddThreshold(threshold AlertThreshold) {
    am.mu.Lock()
    defer am.mu.Unlock()
    am.thresholds = append(am.thresholds, threshold)
}

// Add alert callback
func (am *AlertManager) AddCallback(callback func(Alert)) {
    am.mu.Lock()
    defer am.mu.Unlock()
    am.callbacks = append(am.callbacks, callback)
}

// Check metrics against thresholds
func (am *AlertManager) CheckMetrics(metrics DatabaseMetrics) {
    am.mu.Lock()
    defer am.mu.Unlock()
    
    for _, threshold := range am.thresholds {
        if am.evaluateThreshold(threshold, metrics) {
            alert := Alert{
                ID:        fmt.Sprintf("alert_%d", time.Now().UnixNano()),
                Timestamp: time.Now(),
                Threshold: threshold,
                Severity:  am.determineSeverity(threshold, metrics),
                Message:   am.generateAlertMessage(threshold, metrics),
            }
            
            am.alerts = append(am.alerts, alert)
            
            // Trigger callbacks
            for _, callback := range am.callbacks {
                go callback(alert)
            }
        }
    }
}

func (am *AlertManager) evaluateThreshold(threshold AlertThreshold, metrics DatabaseMetrics) bool {
    var actualValue float64
    
    switch threshold.MetricName {
    case "active_connections":
        actualValue = float64(metrics.ActiveConnections)
    case "avg_query_time":
        actualValue = metrics.AvgQueryTime
    case "buffer_hit_ratio":
        actualValue = metrics.BufferHitRatio
    case "slow_queries":
        actualValue = float64(metrics.SlowQueries)
    case "lock_waits":
        actualValue = float64(metrics.LockWaits)
    default:
        return false
    }
    
    switch threshold.Operator {
    case "gt":
        return actualValue > threshold.Value
    case "lt":
        return actualValue < threshold.Value
    case "eq":
        return actualValue == threshold.Value
    default:
        return false
    }
}

func (am *AlertManager) determineSeverity(threshold AlertThreshold, metrics DatabaseMetrics) string {
    // Simple severity determination logic
    switch threshold.MetricName {
    case "active_connections":
        if metrics.ActiveConnections > 80 {
            return "critical"
        } else if metrics.ActiveConnections > 60 {
            return "warning"
        }
    case "avg_query_time":
        if metrics.AvgQueryTime > 5000 {
            return "critical"
        } else if metrics.AvgQueryTime > 1000 {
            return "warning"
        }
    case "buffer_hit_ratio":
        if metrics.BufferHitRatio < 80 {
            return "critical"
        } else if metrics.BufferHitRatio < 90 {
            return "warning"
        }
    }
    return "info"
}

func (am *AlertManager) generateAlertMessage(threshold AlertThreshold, metrics DatabaseMetrics) string {
    return fmt.Sprintf("Database metric %s exceeded threshold: %s %v", 
        threshold.MetricName, threshold.Operator, threshold.Value)
}

// Get recent alerts
func (am *AlertManager) GetAlerts(duration time.Duration) []Alert {
    am.mu.RLock()
    defer am.mu.RUnlock()
    
    since := time.Now().Add(-duration)
    var result []Alert
    
    for _, alert := range am.alerts {
        if alert.Timestamp.After(since) {
            result = append(result, alert)
        }
    }
    
    return result
}
```

---

## 📈 **Partitioning & Sharding**

### **Table Partitioning Strategies**

```go
package partitioning

import (
    "context"
    "database/sql"
    "fmt"
    "strings"
    "time"
)

// Partition Strategy Types
type PartitionStrategy string

const (
    RangePartitioning PartitionStrategy = "range"
    HashPartitioning  PartitionStrategy = "hash"
    ListPartitioning  PartitionStrategy = "list"
)

// Partition Configuration
type PartitionConfig struct {
    Strategy    PartitionStrategy `json:"strategy"`
    Column      string           `json:"column"`
    Partitions  int              `json:"partitions"`
    Interval    string           `json:"interval"` // For time-based partitioning
    Values      []string         `json:"values"`   // For list partitioning
}

// Partition Manager
type PartitionManager struct {
    db *sql.DB
}

func NewPartitionManager(db *sql.DB) *PartitionManager {
    return &PartitionManager{db: db}
}

// Create partitioned table
func (pm *PartitionManager) CreatePartitionedTable(tableName string, columns []string, config PartitionConfig) error {
    // Create main partitioned table
    createTableSQL := fmt.Sprintf(`
        CREATE TABLE %s (
            %s
        ) PARTITION BY %s (%s)
    `, tableName, strings.Join(columns, ", "), 
       strings.ToUpper(string(config.Strategy)), config.Column)
    
    if _, err := pm.db.Exec(createTableSQL); err != nil {
        return fmt.Errorf("failed to create partitioned table: %w", err)
    }
    
    // Create partitions based on strategy
    switch config.Strategy {
    case RangePartitioning:
        return pm.createRangePartitions(tableName, config)
    case HashPartitioning:
        return pm.createHashPartitions(tableName, config)
    case ListPartitioning:
        return pm.createListPartitions(tableName, config)
    default:
        return fmt.Errorf("unsupported partition strategy: %s", config.Strategy)
    }
}

// Create range partitions (typically for dates)
func (pm *PartitionManager) createRangePartitions(tableName string, config PartitionConfig) error {
    // Example: Monthly partitions for a year
    if config.Interval == "monthly" {
        currentTime := time.Now()
        for i := 0; i < 12; i++ {
            start := currentTime.AddDate(0, i, 0)
            end := start.AddDate(0, 1, 0)
            
            partitionName := fmt.Sprintf("%s_%s", tableName, start.Format("2006_01"))
            
            createPartitionSQL := fmt.Sprintf(`
                CREATE TABLE %s PARTITION OF %s
                FOR VALUES FROM ('%s') TO ('%s')
            `, partitionName, tableName, 
               start.Format("2006-01-02"), end.Format("2006-01-02"))
            
            if _, err := pm.db.Exec(createPartitionSQL); err != nil {
                return fmt.Errorf("failed to create partition %s: %w", partitionName, err)
            }
        }
    }
    return nil
}

// Create hash partitions
func (pm *PartitionManager) createHashPartitions(tableName string, config PartitionConfig) error {
    for i := 0; i < config.Partitions; i++ {
        partitionName := fmt.Sprintf("%s_hash_%d", tableName, i)
        
        createPartitionSQL := fmt.Sprintf(`
            CREATE TABLE %s PARTITION OF %s
            FOR VALUES WITH (modulus %d, remainder %d)
        `, partitionName, tableName, config.Partitions, i)
        
        if _, err := pm.db.Exec(createPartitionSQL); err != nil {
            return fmt.Errorf("failed to create hash partition %s: %w", partitionName, err)
        }
    }
    return nil
}

// Create list partitions
func (pm *PartitionManager) createListPartitions(tableName string, config PartitionConfig) error {
    for i, value := range config.Values {
        partitionName := fmt.Sprintf("%s_list_%d", tableName, i)
        
        createPartitionSQL := fmt.Sprintf(`
            CREATE TABLE %s PARTITION OF %s
            FOR VALUES IN (%s)
        `, partitionName, tableName, value)
        
        if _, err := pm.db.Exec(createPartitionSQL); err != nil {
            return fmt.Errorf("failed to create list partition %s: %w", partitionName, err)
        }
    }
    return nil
}

// Automatic partition maintenance
type PartitionMaintainer struct {
    db       *sql.DB
    interval time.Duration
    stopChan chan struct{}
}

func NewPartitionMaintainer(db *sql.DB, interval time.Duration) *PartitionMaintainer {
    return &PartitionMaintainer{
        db:       db,
        interval: interval,
        stopChan: make(chan struct{}),
    }
}

func (pm *PartitionMaintainer) Start() {
    go pm.maintenanceLoop()
}

func (pm *PartitionMaintainer) maintenanceLoop() {
    ticker := time.NewTicker(pm.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            pm.performMaintenance()
        case <-pm.stopChan:
            return
        }
    }
}

func (pm *PartitionMaintainer) performMaintenance() {
    // Create new partitions for future dates
    pm.createFuturePartitions()
    
    // Drop old partitions
    pm.dropOldPartitions()
    
    // Update partition statistics
    pm.updatePartitionStatistics()
}

func (pm *PartitionMaintainer) createFuturePartitions() {
    // Implementation for creating future time-based partitions
    tables, err := pm.getPartitionedTables()
    if err != nil {
        return
    }
    
    for _, table := range tables {
        // Check if we need new partitions for next month
        nextMonth := time.Now().AddDate(0, 1, 0)
        partitionName := fmt.Sprintf("%s_%s", table, nextMonth.Format("2006_01"))
        
        if exists, _ := pm.partitionExists(partitionName); !exists {
            pm.createMonthlyPartition(table, nextMonth)
        }
    }
}

func (pm *PartitionMaintainer) dropOldPartitions() {
    // Drop partitions older than retention period
    cutoffDate := time.Now().AddDate(0, -12, 0) // Keep 12 months
    
    tables, err := pm.getPartitionedTables()
    if err != nil {
        return
    }
    
    for _, table := range tables {
        oldPartitions, err := pm.getOldPartitions(table, cutoffDate)
        if err != nil {
            continue
        }
        
        for _, partition := range oldPartitions {
            pm.dropPartition(partition)
        }
    }
}

func (pm *PartitionMaintainer) updatePartitionStatistics() {
    // Update statistics for all partitions
    _, err := pm.db.Exec("ANALYZE")
    if err != nil {
        fmt.Printf("Failed to update partition statistics: %v\n", err)
    }
}

func (pm *PartitionMaintainer) getPartitionedTables() ([]string, error) {
    query := `
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename IN (
            SELECT schemaname||'.'||tablename 
            FROM pg_partitioned_tables
        )
    `
    
    rows, err := pm.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var tables []string
    for rows.Next() {
        var table string
        if err := rows.Scan(&table); err != nil {
            continue
        }
        tables = append(tables, table)
    }
    
    return tables, nil
}

func (pm *PartitionMaintainer) partitionExists(partitionName string) (bool, error) {
    var exists bool
    query := "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE tablename = $1)"
    err := pm.db.QueryRow(query, partitionName).Scan(&exists)
    return exists, err
}

func (pm *PartitionMaintainer) createMonthlyPartition(tableName string, month time.Time) error {
    start := time.Date(month.Year(), month.Month(), 1, 0, 0, 0, 0, time.UTC)
    end := start.AddDate(0, 1, 0)
    
    partitionName := fmt.Sprintf("%s_%s", tableName, start.Format("2006_01"))
    
    createSQL := fmt.Sprintf(`
        CREATE TABLE %s PARTITION OF %s
        FOR VALUES FROM ('%s') TO ('%s')
    `, partitionName, tableName,
       start.Format("2006-01-02"), end.Format("2006-01-02"))
    
    _, err := pm.db.Exec(createSQL)
    return err
}

func (pm *PartitionMaintainer) getOldPartitions(tableName string, cutoffDate time.Time) ([]string, error) {
    // Simplified implementation - would need more sophisticated logic
    // to parse partition names and determine age
    return []string{}, nil
}

func (pm *PartitionMaintainer) dropPartition(partitionName string) error {
    _, err := pm.db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", partitionName))
    return err
}

func (pm *PartitionMaintainer) Stop() {
    close(pm.stopChan)
}
```

---

## 🏎️ **Performance Testing**

### **Database Load Testing Framework**

```go
package loadtest

import (
    "context"
    "database/sql"
    "fmt"
    "math/rand"
    "sync"
    "sync/atomic"
    "time"
)

// Load Test Configuration
type LoadTestConfig struct {
    ConcurrentUsers    int           `json:"concurrent_users"`
    TestDuration       time.Duration `json:"test_duration"`
    RampUpTime         time.Duration `json:"ramp_up_time"`
    QueryDistribution  map[string]int `json:"query_distribution"` // query_name -> percentage
    DataSetSize        int           `json:"data_set_size"`
    ThinkTime          time.Duration `json:"think_time"`
    ConnectionStrategy string        `json:"connection_strategy"` // "shared", "per_user"
}

// Load Test Results
type LoadTestResults struct {
    TotalRequests      int64         `json:"total_requests"`
    SuccessfulRequests int64         `json:"successful_requests"`
    FailedRequests     int64         `json:"failed_requests"`
    AverageLatency     time.Duration `json:"average_latency"`
    MinLatency         time.Duration `json:"min_latency"`
    MaxLatency         time.Duration `json:"max_latency"`
    P50Latency         time.Duration `json:"p50_latency"`
    P95Latency         time.Duration `json:"p95_latency"`
    P99Latency         time.Duration `json:"p99_latency"`
    ThroughputRPS      float64       `json:"throughput_rps"`
    ErrorRate          float64       `json:"error_rate"`
    ConnectionErrors   int64         `json:"connection_errors"`
    QueryResults       map[string]QueryResult `json:"query_results"`
}

type QueryResult struct {
    QueryName      string        `json:"query_name"`
    ExecutionCount int64         `json:"execution_count"`
    AverageLatency time.Duration `json:"average_latency"`
    ErrorCount     int64         `json:"error_count"`
}

// Database Load Tester
type DatabaseLoadTester struct {
    db        *sql.DB
    config    LoadTestConfig
    results   LoadTestResults
    stopChan  chan struct{}
    wg        sync.WaitGroup
    queries   map[string]string
    latencies []time.Duration
    mu        sync.Mutex
}

func NewDatabaseLoadTester(db *sql.DB, config LoadTestConfig) *DatabaseLoadTester {
    return &DatabaseLoadTester{
        db:       db,
        config:   config,
        stopChan: make(chan struct{}),
        queries:  make(map[string]string),
        results: LoadTestResults{
            QueryResults: make(map[string]QueryResult),
        },
    }
}

// Add test query
func (dlt *DatabaseLoadTester) AddQuery(name, query string) {
    dlt.queries[name] = query
}

// Run load test
func (dlt *DatabaseLoadTester) RunLoadTest(ctx context.Context) (*LoadTestResults, error) {
    startTime := time.Now()
    
    // Initialize results
    dlt.results = LoadTestResults{
        QueryResults: make(map[string]QueryResult),
        MinLatency:   time.Hour, // Initialize with high value
    }
    
    // Start users gradually (ramp-up)
    userStartInterval := dlt.config.RampUpTime / time.Duration(dlt.config.ConcurrentUsers)
    
    for i := 0; i < dlt.config.ConcurrentUsers; i++ {
        dlt.wg.Add(1)
        go dlt.simulateUser(ctx, i)
        
        if userStartInterval > 0 {
            time.Sleep(userStartInterval)
        }
    }
    
    // Wait for test duration
    select {
    case <-time.After(dlt.config.TestDuration):
        close(dlt.stopChan)
    case <-ctx.Done():
        close(dlt.stopChan)
        return nil, ctx.Err()
    }
    
    // Wait for all users to finish
    dlt.wg.Wait()
    
    // Calculate final results
    dlt.calculateResults(time.Since(startTime))
    
    return &dlt.results, nil
}

// Simulate single user behavior
func (dlt *DatabaseLoadTester) simulateUser(ctx context.Context, userID int) {
    defer dlt.wg.Done()
    
    rand.Seed(time.Now().UnixNano() + int64(userID))
    
    for {
        select {
        case <-dlt.stopChan:
            return
        case <-ctx.Done():
            return
        default:
            // Execute random query based on distribution
            queryName := dlt.selectRandomQuery()
            query, exists := dlt.queries[queryName]
            if !exists {
                continue
            }
            
            // Execute query and measure latency
            start := time.Now()
            err := dlt.executeQuery(ctx, query)
            latency := time.Since(start)
            
            // Record results
            dlt.recordResult(queryName, latency, err)
            
            // Think time between requests
            if dlt.config.ThinkTime > 0 {
                time.Sleep(dlt.config.ThinkTime)
            }
        }
    }
}

// Select query based on distribution
func (dlt *DatabaseLoadTester) selectRandomQuery() string {
    totalWeight := 0
    for _, weight := range dlt.config.QueryDistribution {
        totalWeight += weight
    }
    
    randomValue := rand.Intn(totalWeight)
    currentWeight := 0
    
    for queryName, weight := range dlt.config.QueryDistribution {
        currentWeight += weight
        if randomValue < currentWeight {
            return queryName
        }
    }
    
    // Fallback to first query
    for queryName := range dlt.queries {
        return queryName
    }
    
    return ""
}

// Execute database query
func (dlt *DatabaseLoadTester) executeQuery(ctx context.Context, query string) error {
    queryCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    
    // Generate random parameters for queries
    args := dlt.generateQueryArgs(query)
    
    rows, err := dlt.db.QueryContext(queryCtx, query, args...)
    if err != nil {
        return err
    }
    defer rows.Close()
    
    // Consume all rows to simulate real usage
    for rows.Next() {
        // Process row (simplified)
    }
    
    return rows.Err()
}

// Generate random arguments for parameterized queries
func (dlt *DatabaseLoadTester) generateQueryArgs(query string) []interface{} {
    // Count parameter placeholders
    paramCount := strings.Count(query, "$")
    if paramCount == 0 {
        return nil
    }
    
    args := make([]interface{}, paramCount)
    for i := 0; i < paramCount; i++ {
        args[i] = rand.Intn(dlt.config.DataSetSize) + 1
    }
    
    return args
}

// Record query execution result
func (dlt *DatabaseLoadTester) recordResult(queryName string, latency time.Duration, err error) {
    dlt.mu.Lock()
    defer dlt.mu.Unlock()
    
    // Update total counters
    atomic.AddInt64(&dlt.results.TotalRequests, 1)
    
    if err != nil {
        atomic.AddInt64(&dlt.results.FailedRequests, 1)
        
        // Track connection errors separately
        if strings.Contains(err.Error(), "connection") {
            atomic.AddInt64(&dlt.results.ConnectionErrors, 1)
        }
    } else {
        atomic.AddInt64(&dlt.results.SuccessfulRequests, 1)
    }
    
    // Record latency
    dlt.latencies = append(dlt.latencies, latency)
    
    // Update min/max latency
    if latency < dlt.results.MinLatency {
        dlt.results.MinLatency = latency
    }
    if latency > dlt.results.MaxLatency {
        dlt.results.MaxLatency = latency
    }
    
    // Update query-specific results
    queryResult, exists := dlt.results.QueryResults[queryName]
    if !exists {
        queryResult = QueryResult{QueryName: queryName}
    }
    
    queryResult.ExecutionCount++
    if err != nil {
        queryResult.ErrorCount++
    }
    
    // Update average latency (simple moving average)
    if queryResult.ExecutionCount == 1 {
        queryResult.AverageLatency = latency
    } else {
        queryResult.AverageLatency = (queryResult.AverageLatency*time.Duration(queryResult.ExecutionCount-1) + latency) / time.Duration(queryResult.ExecutionCount)
    }
    
    dlt.results.QueryResults[queryName] = queryResult
}

// Calculate final test results
func (dlt *DatabaseLoadTester) calculateResults(totalDuration time.Duration) {
    if len(dlt.latencies) == 0 {
        return
    }
    
    // Sort latencies for percentile calculations
    sort.Slice(dlt.latencies, func(i, j int) bool {
        return dlt.latencies[i] < dlt.latencies[j]
    })
    
    // Calculate average latency
    var totalLatency time.Duration
    for _, latency := range dlt.latencies {
        totalLatency += latency
    }
    dlt.results.AverageLatency = totalLatency / time.Duration(len(dlt.latencies))
    
    // Calculate percentiles
    dlt.results.P50Latency = dlt.calculatePercentile(50)
    dlt.results.P95Latency = dlt.calculatePercentile(95)
    dlt.results.P99Latency = dlt.calculatePercentile(99)
    
    // Calculate throughput
    dlt.results.ThroughputRPS = float64(dlt.results.TotalRequests) / totalDuration.Seconds()
    
    // Calculate error rate
    if dlt.results.TotalRequests > 0 {
        dlt.results.ErrorRate = float64(dlt.results.FailedRequests) / float64(dlt.results.TotalRequests) * 100
    }
}

func (dlt *DatabaseLoadTester) calculatePercentile(percentile int) time.Duration {
    if len(dlt.latencies) == 0 {
        return 0
    }
    
    index := int(float64(len(dlt.latencies)) * float64(percentile) / 100.0)
    if index >= len(dlt.latencies) {
        index = len(dlt.latencies) - 1
    }
    
    return dlt.latencies[index]
}

// Benchmark specific query patterns
type QueryBenchmark struct {
    db *sql.DB
}

func NewQueryBenchmark(db *sql.DB) *QueryBenchmark {
    return &QueryBenchmark{db: db}
}

// Benchmark JOIN performance
func (qb *QueryBenchmark) BenchmarkJoin(ctx context.Context, iterations int) (time.Duration, error) {
    query := `
        SELECT u.id, u.name, p.title, c.name as company
        FROM users u
        JOIN profiles p ON u.id = p.user_id
        JOIN companies c ON p.company_id = c.id
        WHERE u.created_at > $1
        LIMIT 100
    `
    
    start := time.Now()
    
    for i := 0; i < iterations; i++ {
        rows, err := qb.db.QueryContext(ctx, query, time.Now().AddDate(0, -1, 0))
        if err != nil {
            return 0, err
        }
        
        for rows.Next() {
            // Process rows
        }
        rows.Close()
    }
    
    return time.Since(start) / time.Duration(iterations), nil
}

// Benchmark aggregation queries
func (qb *QueryBenchmark) BenchmarkAggregation(ctx context.Context, iterations int) (time.Duration, error) {
    query := `
        SELECT 
            DATE_TRUNC('day', created_at) as day,
            COUNT(*) as count,
            AVG(amount) as avg_amount,
            SUM(amount) as total_amount
        FROM transactions
        WHERE created_at >= $1
        GROUP BY DATE_TRUNC('day', created_at)
        ORDER BY day DESC
    `
    
    start := time.Now()
    
    for i := 0; i < iterations; i++ {
        rows, err := qb.db.QueryContext(ctx, query, time.Now().AddDate(0, -3, 0))
        if err != nil {
            return 0, err
        }
        
        for rows.Next() {
            // Process rows
        }
        rows.Close()
    }
    
    return time.Since(start) / time.Duration(iterations), nil
}

// Benchmark index usage
func (qb *QueryBenchmark) BenchmarkIndexedQuery(ctx context.Context, iterations int) (time.Duration, error) {
    query := `
        SELECT id, email, created_at
        FROM users
        WHERE email = $1
    `
    
    emails := []string{
        "user1@example.com",
        "user2@example.com",
        "user3@example.com",
    }
    
    start := time.Now()
    
    for i := 0; i < iterations; i++ {
        email := emails[i%len(emails)]
        rows, err := qb.db.QueryContext(ctx, query, email)
        if err != nil {
            return 0, err
        }
        
        for rows.Next() {
            // Process rows
        }
        rows.Close()
    }
    
    return time.Since(start) / time.Duration(iterations), nil
}
```

---

## ❓ **Interview Questions**

### **Advanced Database Performance Questions**

#### **1. Query Optimization**

**Q: How would you optimize a slow query that joins multiple large tables?**

**A: Comprehensive optimization approach:**

```sql
-- Original slow query
SELECT u.name, p.title, c.company_name, COUNT(o.id) as order_count
FROM users u
JOIN profiles p ON u.id = p.user_id
JOIN companies c ON p.company_id = c.id
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2023-01-01'
GROUP BY u.id, u.name, p.title, c.company_name
ORDER BY order_count DESC;

-- Optimization steps:
-- 1. Add indexes
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_profiles_user_company ON profiles(user_id, company_id);
CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);

-- 2. Rewrite with better structure
WITH user_orders AS (
    SELECT user_id, COUNT(*) as order_count
    FROM orders o
    WHERE EXISTS (
        SELECT 1 FROM users u 
        WHERE u.id = o.user_id 
        AND u.created_at > '2023-01-01'
    )
    GROUP BY user_id
)
SELECT u.name, p.title, c.company_name, 
       COALESCE(uo.order_count, 0) as order_count
FROM users u
JOIN profiles p ON u.id = p.user_id
JOIN companies c ON p.company_id = c.id
LEFT JOIN user_orders uo ON u.id = uo.user_id
WHERE u.created_at > '2023-01-01'
ORDER BY COALESCE(uo.order_count, 0) DESC;
```

**Key optimization techniques:**
- Add appropriate indexes on join and filter columns
- Use EXISTS instead of JOIN when possible
- Pre-aggregate data in CTEs to reduce computation
- Consider materialized views for frequently accessed aggregations
- Use EXPLAIN (ANALYZE, BUFFERS) to verify improvements

#### **2. Index Design Strategy**

**Q: Design an indexing strategy for a high-traffic e-commerce database.**

**A: Multi-layered indexing approach:**

```sql
-- 1. Primary and unique indexes (automatic)
-- users: id (PK), email (UNIQUE)
-- products: id (PK), sku (UNIQUE)
-- orders: id (PK)

-- 2. Query-pattern based indexes
-- User queries by email (login)
CREATE INDEX CONCURRENTLY idx_users_email_active ON users(email) WHERE status = 'active';

-- Product searches
CREATE INDEX CONCURRENTLY idx_products_category_price ON products(category_id, price);
CREATE INDEX CONCURRENTLY idx_products_name_search ON products USING gin(to_tsvector('english', name));

-- Order queries
CREATE INDEX CONCURRENTLY idx_orders_user_status_date ON orders(user_id, status, created_at);
CREATE INDEX CONCURRENTLY idx_orders_date_status ON orders(created_at DESC, status) 
WHERE status IN ('pending', 'processing');

-- 3. Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_order_items_order_product ON order_items(order_id, product_id);

-- 4. Partial indexes for specific conditions
CREATE INDEX CONCURRENTLY idx_users_premium ON users(created_at) WHERE subscription_type = 'premium';

-- 5. Covering indexes for common queries
CREATE INDEX CONCURRENTLY idx_products_catalog ON products(category_id, price) 
INCLUDE (name, description, image_url);
```

#### **3. Connection Pool Optimization**

**Q: How would you tune connection pool settings for maximum performance?**

**A: Systematic connection pool tuning:**

```go
// Connection pool configuration based on workload analysis
func OptimizeConnectionPool(db *sql.DB, workloadType string) {
    switch workloadType {
    case "high_throughput_oltp":
        // High concurrent short transactions
        db.SetMaxOpenConns(100)    // 2-3x CPU cores
        db.SetMaxIdleConns(20)     // 20% of max open
        db.SetConnMaxLifetime(1 * time.Hour)
        db.SetConnMaxIdleTime(15 * time.Minute)
        
    case "analytical_olap":
        // Long-running analytical queries
        db.SetMaxOpenConns(20)     // Fewer connections for long queries
        db.SetMaxIdleConns(5)      // Minimal idle connections
        db.SetConnMaxLifetime(4 * time.Hour)
        db.SetConnMaxIdleTime(30 * time.Minute)
        
    case "mixed_workload":
        // Balanced configuration
        db.SetMaxOpenConns(50)
        db.SetMaxIdleConns(10)
        db.SetConnMaxLifetime(2 * time.Hour)
        db.SetConnMaxIdleTime(20 * time.Minute)
    }
}

// Dynamic connection pool monitoring and adjustment
type PoolOptimizer struct {
    db *sql.DB
    targetLatency time.Duration
    maxConnections int
}

func (po *PoolOptimizer) AutoTune() {
    stats := po.db.Stats()
    
    // If wait count is high, increase connections
    if stats.WaitCount > 0 && stats.WaitDuration > po.targetLatency {
        newMax := min(stats.OpenConnections + 5, po.maxConnections)
        po.db.SetMaxOpenConns(newMax)
    }
    
    // If idle connections are consistently high, reduce max idle
    if stats.Idle > stats.InUse*2 {
        newIdle := max(stats.InUse, 5)
        po.db.SetMaxIdleConns(newIdle)
    }
}
```

#### **4. Database Partitioning Strategy**

**Q: When and how would you implement table partitioning?**

**A: Strategic partitioning implementation:**

```sql
-- Scenario: Large orders table with time-based queries
-- 1. Analyze query patterns
SELECT 
    DATE_TRUNC('month', created_at) as month,
    COUNT(*) as order_count,
    AVG(total_amount) as avg_amount
FROM orders 
WHERE created_at >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;

-- 2. Implement range partitioning by date
CREATE TABLE orders_partitioned (
    id BIGSERIAL,
    user_id BIGINT NOT NULL,
    total_amount DECIMAL(10,2),
    status VARCHAR(20),
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 3. Create monthly partitions
CREATE TABLE orders_2024_01 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 4. Add indexes to each partition
CREATE INDEX idx_orders_2024_01_user_id ON orders_2024_01(user_id);
CREATE INDEX idx_orders_2024_01_status ON orders_2024_01(status);

-- 5. Benefits analysis:
-- - Partition pruning: Queries with date filters only scan relevant partitions
-- - Parallel maintenance: VACUUM, ANALYZE can run on partitions concurrently
-- - Easier archival: Drop old partitions instead of DELETE operations
-- - Improved performance: Smaller indexes per partition
```

#### **5. Database Monitoring and Alerting**

**Q: Design a comprehensive database monitoring system.**

**A: Multi-layered monitoring architecture:**

```go
// Monitoring metrics hierarchy
type DatabaseMonitoring struct {
    // Infrastructure level
    CPUUsage        float64 `json:"cpu_usage"`
    MemoryUsage     float64 `json:"memory_usage"`
    DiskIO          DiskIOMetrics `json:"disk_io"`
    NetworkIO       NetworkIOMetrics `json:"network_io"`
    
    // Database level
    Connections     ConnectionMetrics `json:"connections"`
    QueryPerf       QueryMetrics `json:"query_performance"`
    IndexUsage      IndexMetrics `json:"index_usage"`
    LockContention  LockMetrics `json:"lock_contention"`
    
    // Application level
    TransactionRate float64 `json:"transaction_rate"`
    ErrorRate       float64 `json:"error_rate"`
    ResponseTime    ResponseTimeMetrics `json:"response_time"`
}

// Alert thresholds with escalation
var AlertThresholds = map[string]AlertConfig{
    "connection_saturation": {
        Warning:  80,  // 80% of max connections
        Critical: 95,  // 95% of max connections
        Duration: 2 * time.Minute,
    },
    "query_latency": {
        Warning:  1000, // 1 second average
        Critical: 5000, // 5 seconds average
        Duration: 5 * time.Minute,
    },
    "cache_hit_ratio": {
        Warning:  85, // Below 85%
        Critical: 70, // Below 70%
        Duration: 10 * time.Minute,
    },
    "disk_space": {
        Warning:  80, // 80% full
        Critical: 95, // 95% full
        Duration: 1 * time.Minute,
    },
}

// Proactive monitoring queries
const (
    SlowQueryMonitoring = `
        SELECT query, calls, mean_exec_time, total_exec_time,
               rows, 100.0 * shared_blks_hit /
               nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
        FROM pg_stat_statements
        ORDER BY mean_exec_time DESC
        LIMIT 10;
    `
    
    BlockingQueriesMonitoring = `
        SELECT blocked_locks.pid AS blocked_pid,
               blocked_activity.usename AS blocked_user,
               blocking_locks.pid AS blocking_pid,
               blocking_activity.usename AS blocking_user,
               blocked_activity.query AS blocked_statement,
               blocking_activity.query AS blocking_statement
        FROM pg_catalog.pg_locks blocked_locks
        JOIN pg_catalog.pg_stat_activity blocked_activity 
             ON blocked_activity.pid = blocked_locks.pid
        JOIN pg_catalog.pg_locks blocking_locks 
             ON blocking_locks.locktype = blocked_locks.locktype
        JOIN pg_catalog.pg_stat_activity blocking_activity 
             ON blocking_activity.pid = blocking_locks.pid
        WHERE NOT blocked_locks.granted;
    `
)
```

This comprehensive Database Performance Optimization Guide provides advanced techniques for optimizing database performance across different database systems. It covers query optimization, indexing strategies, connection pooling, monitoring, partitioning, and performance testing with practical Go implementations that demonstrate the expertise expected from senior backend engineers in technical interviews.