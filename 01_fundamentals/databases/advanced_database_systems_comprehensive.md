---
# Auto-generated front matter
Title: Advanced Database Systems Comprehensive
LastUpdated: 2025-11-06T20:45:58.663268
Tags: []
Status: draft
---

# Advanced Database Systems Comprehensive

Comprehensive guide to advanced database systems for senior backend engineers.

## 🎯 Advanced Database Architecture

### Multi-Model Database Design
```go
// Advanced Multi-Model Database Implementation
package database

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/neo4j/neo4j-go-driver/v5/neo4j"
    "go.mongodb.org/mongo-driver/mongo"
    "go.mongodb.org/mongo-driver/mongo/options"
    "gorm.io/gorm"
)

type MultiModelDatabase struct {
    relational *gorm.DB
    document   *mongo.Database
    graph      neo4j.Driver
    cache      *RedisCache
    search     *ElasticsearchClient
}

type DatabaseConfig struct {
    Relational RelationalConfig `json:"relational"`
    Document   DocumentConfig   `json:"document"`
    Graph      GraphConfig      `json:"graph"`
    Cache      CacheConfig      `json:"cache"`
    Search     SearchConfig     `json:"search"`
}

type RelationalConfig struct {
    Host     string `json:"host"`
    Port     int    `json:"port"`
    Database string `json:"database"`
    Username string `json:"username"`
    Password string `json:"password"`
    SSLMode  string `json:"ssl_mode"`
}

type DocumentConfig struct {
    URI      string `json:"uri"`
    Database string `json:"database"`
}

type GraphConfig struct {
    URI      string `json:"uri"`
    Username string `json:"username"`
    Password string `json:"password"`
}

type CacheConfig struct {
    Host     string `json:"host"`
    Port     int    `json:"port"`
    Password string `json:"password"`
    DB       int    `json:"db"`
}

type SearchConfig struct {
    Hosts []string `json:"hosts"`
    Index  string   `json:"index"`
}

func NewMultiModelDatabase(config *DatabaseConfig) (*MultiModelDatabase, error) {
    // Initialize relational database
    relational, err := initRelationalDB(config.Relational)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize relational DB: %w", err)
    }
    
    // Initialize document database
    document, err := initDocumentDB(config.Document)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize document DB: %w", err)
    }
    
    // Initialize graph database
    graph, err := initGraphDB(config.Graph)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize graph DB: %w", err)
    }
    
    // Initialize cache
    cache, err := initCache(config.Cache)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize cache: %w", err)
    }
    
    // Initialize search
    search, err := initSearch(config.Search)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize search: %w", err)
    }
    
    return &MultiModelDatabase{
        relational: relational,
        document:   document,
        graph:      graph,
        cache:      cache,
        search:     search,
    }, nil
}

// User Management with Multi-Model Support
type User struct {
    ID          string    `json:"id" gorm:"primaryKey"`
    Email       string    `json:"email" gorm:"uniqueIndex"`
    Username    string    `json:"username" gorm:"uniqueIndex"`
    Profile     UserProfile `json:"profile" gorm:"embedded"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
    LastLoginAt *time.Time `json:"last_login_at"`
}

type UserProfile struct {
    FirstName string `json:"first_name"`
    LastName  string `json:"last_name"`
    Avatar    string `json:"avatar"`
    Bio       string `json:"bio"`
    Location  string `json:"location"`
    Website   string `json:"website"`
}

func (mmdb *MultiModelDatabase) CreateUser(ctx context.Context, user *User) error {
    // Start transaction
    tx := mmdb.relational.Begin()
    defer func() {
        if r := recover(); r != nil {
            tx.Rollback()
        }
    }()
    
    // Create in relational database
    if err := tx.Create(user).Error; err != nil {
        tx.Rollback()
        return fmt.Errorf("failed to create user in relational DB: %w", err)
    }
    
    // Create in document database
    if err := mmdb.createUserDocument(ctx, user); err != nil {
        tx.Rollback()
        return fmt.Errorf("failed to create user document: %w", err)
    }
    
    // Create in graph database
    if err := mmdb.createUserNode(ctx, user); err != nil {
        tx.Rollback()
        return fmt.Errorf("failed to create user node: %w", err)
    }
    
    // Cache user data
    if err := mmdb.cacheUser(ctx, user); err != nil {
        // Log error but don't fail the transaction
        log.Printf("Failed to cache user: %v", err)
    }
    
    // Index for search
    if err := mmdb.indexUser(ctx, user); err != nil {
        // Log error but don't fail the transaction
        log.Printf("Failed to index user: %v", err)
    }
    
    return tx.Commit().Error
}

func (mmdb *MultiModelDatabase) GetUser(ctx context.Context, userID string) (*User, error) {
    // Try cache first
    if user, err := mmdb.getCachedUser(ctx, userID); err == nil && user != nil {
        return user, nil
    }
    
    // Get from relational database
    var user User
    if err := mmdb.relational.First(&user, "id = ?", userID).Error; err != nil {
        return nil, fmt.Errorf("user not found: %w", err)
    }
    
    // Cache the result
    go mmdb.cacheUser(ctx, &user)
    
    return &user, nil
}

func (mmdb *MultiModelDatabase) SearchUsers(ctx context.Context, query string, filters map[string]interface{}) ([]*User, error) {
    // Search in Elasticsearch
    searchResults, err := mmdb.searchUsers(ctx, query, filters)
    if err != nil {
        return nil, fmt.Errorf("search failed: %w", err)
    }
    
    // Get full user data from relational database
    var users []*User
    for _, userID := range searchResults {
        user, err := mmdb.GetUser(ctx, userID)
        if err != nil {
            log.Printf("Failed to get user %s: %v", userID, err)
            continue
        }
        users = append(users, user)
    }
    
    return users, nil
}

func (mmdb *MultiModelDatabase) GetUserRelationships(ctx context.Context, userID string) ([]*User, error) {
    // Query graph database for relationships
    session := mmdb.graph.NewSession(ctx, neo4j.SessionConfig{})
    defer session.Close(ctx)
    
    query := `
        MATCH (u:User {id: $userID})-[:FOLLOWS]->(f:User)
        RETURN f.id as id, f.email as email, f.username as username
    `
    
    result, err := session.Run(ctx, query, map[string]interface{}{
        "userID": userID,
    })
    if err != nil {
        return nil, fmt.Errorf("graph query failed: %w", err)
    }
    
    var users []*User
    for result.Next(ctx) {
        record := result.Record()
        user := &User{
            ID:       record.Values[0].(string),
            Email:    record.Values[1].(string),
            Username: record.Values[2].(string),
        }
        users = append(users, user)
    }
    
    return users, nil
}

func (mmdb *MultiModelDatabase) createUserDocument(ctx context.Context, user *User) error {
    collection := mmdb.document.Collection("users")
    
    userDoc := map[string]interface{}{
        "id":           user.ID,
        "email":        user.Email,
        "username":     user.Username,
        "profile":      user.Profile,
        "created_at":   user.CreatedAt,
        "updated_at":   user.UpdatedAt,
        "last_login_at": user.LastLoginAt,
    }
    
    _, err := collection.InsertOne(ctx, userDoc)
    return err
}

func (mmdb *MultiModelDatabase) createUserNode(ctx context.Context, user *User) error {
    session := mmdb.graph.NewSession(ctx, neo4j.SessionConfig{})
    defer session.Close(ctx)
    
    query := `
        CREATE (u:User {
            id: $id,
            email: $email,
            username: $username,
            created_at: $created_at
        })
    `
    
    _, err := session.Run(ctx, query, map[string]interface{}{
        "id":         user.ID,
        "email":      user.Email,
        "username":   user.Username,
        "created_at": user.CreatedAt,
    })
    
    return err
}

func (mmdb *MultiModelDatabase) cacheUser(ctx context.Context, user *User) error {
    key := fmt.Sprintf("user:%s", user.ID)
    data, err := json.Marshal(user)
    if err != nil {
        return err
    }
    
    return mmdb.cache.Set(ctx, key, data, 24*time.Hour)
}

func (mmdb *MultiModelDatabase) getCachedUser(ctx context.Context, userID string) (*User, error) {
    key := fmt.Sprintf("user:%s", userID)
    data, err := mmdb.cache.Get(ctx, key)
    if err != nil {
        return nil, err
    }
    
    var user User
    if err := json.Unmarshal(data, &user); err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (mmdb *MultiModelDatabase) indexUser(ctx context.Context, user *User) error {
    doc := map[string]interface{}{
        "id":       user.ID,
        "email":    user.Email,
        "username": user.Username,
        "profile":  user.Profile,
    }
    
    return mmdb.search.IndexDocument(ctx, "users", user.ID, doc)
}

func (mmdb *MultiModelDatabase) searchUsers(ctx context.Context, query string, filters map[string]interface{}) ([]string, error) {
    searchQuery := map[string]interface{}{
        "query": map[string]interface{}{
            "bool": map[string]interface{}{
                "must": []map[string]interface{}{
                    {
                        "multi_match": map[string]interface{}{
                            "query":  query,
                            "fields": []string{"email", "username", "profile.first_name", "profile.last_name"},
                        },
                    },
                },
            },
        },
    }
    
    // Add filters
    if len(filters) > 0 {
        filterClauses := []map[string]interface{}{}
        for key, value := range filters {
            filterClauses = append(filterClauses, map[string]interface{}{
                "term": map[string]interface{}{
                    key: value,
                },
            })
        }
        
        searchQuery["query"].(map[string]interface{})["bool"].(map[string]interface{})["filter"] = filterClauses
    }
    
    results, err := mmdb.search.Search(ctx, "users", searchQuery)
    if err != nil {
        return nil, err
    }
    
    var userIDs []string
    for _, hit := range results.Hits {
        userIDs = append(userIDs, hit.ID)
    }
    
    return userIDs, nil
}
```

### Advanced Query Optimization
```go
// Advanced Query Optimization Engine
package database

import (
    "context"
    "fmt"
    "strings"
    "time"
)

type QueryOptimizer struct {
    db           *gorm.DB
    queryCache    *QueryCache
    statsCollector *StatsCollector
    indexManager  *IndexManager
}

type QueryPlan struct {
    ID          string        `json:"id"`
    Query       string        `json:"query"`
    ExecutionTime time.Duration `json:"execution_time"`
    Cost        float64       `json:"cost"`
    Plan        string        `json:"plan"`
    Indexes     []string      `json:"indexes"`
    Warnings    []string      `json:"warnings"`
}

type QueryCache struct {
    cache map[string]*CachedQuery
    mutex sync.RWMutex
}

type CachedQuery struct {
    Result      interface{}
    ExpiresAt   time.Time
    HitCount    int
    LastUsed    time.Time
}

type StatsCollector struct {
    stats map[string]*QueryStats
    mutex sync.RWMutex
}

type QueryStats struct {
    Query         string        `json:"query"`
    ExecutionTime time.Duration `json:"execution_time"`
    RowCount      int64         `json:"row_count"`
    IndexUsed     string        `json:"index_used"`
    CacheHit      bool          `json:"cache_hit"`
    Timestamp     time.Time     `json:"timestamp"`
}

func NewQueryOptimizer(db *gorm.DB) *QueryOptimizer {
    return &QueryOptimizer{
        db:            db,
        queryCache:    NewQueryCache(),
        statsCollector: NewStatsCollector(),
        indexManager:  NewIndexManager(db),
    }
}

func (qo *QueryOptimizer) OptimizeQuery(ctx context.Context, query string, args ...interface{}) (*QueryPlan, error) {
    // Check cache first
    if cached, found := qo.queryCache.Get(query); found {
        qo.statsCollector.RecordQuery(query, 0, 0, "", true)
        return cached, nil
    }
    
    // Analyze query
    analysis := qo.analyzeQuery(query)
    
    // Generate execution plan
    plan, err := qo.generateExecutionPlan(ctx, query, analysis)
    if err != nil {
        return nil, fmt.Errorf("failed to generate execution plan: %w", err)
    }
    
    // Optimize plan
    optimizedPlan := qo.optimizePlan(plan, analysis)
    
    // Cache the plan
    qo.queryCache.Set(query, optimizedPlan)
    
    // Record statistics
    qo.statsCollector.RecordQuery(query, optimizedPlan.ExecutionTime, 0, "", false)
    
    return optimizedPlan, nil
}

func (qo *QueryOptimizer) analyzeQuery(query string) *QueryAnalysis {
    analysis := &QueryAnalysis{
        Query: query,
        Tables: []string{},
        Joins: []JoinInfo{},
        Filters: []FilterInfo{},
        Sorts: []SortInfo{},
        Aggregations: []AggregationInfo{},
    }
    
    // Parse query to extract information
    // This is a simplified version - in practice, you'd use a proper SQL parser
    
    // Extract tables
    if strings.Contains(strings.ToLower(query), "from") {
        // Simple table extraction
        parts := strings.Split(strings.ToLower(query), "from")
        if len(parts) > 1 {
            tablePart := strings.Split(parts[1], " ")[0]
            analysis.Tables = append(analysis.Tables, strings.TrimSpace(tablePart))
        }
    }
    
    // Extract joins
    if strings.Contains(strings.ToLower(query), "join") {
        // Simple join extraction
        joinParts := strings.Split(strings.ToLower(query), "join")
        for i := 1; i < len(joinParts); i++ {
            tablePart := strings.Split(joinParts[i], " ")[0]
            analysis.Joins = append(analysis.Joins, JoinInfo{
                Table: strings.TrimSpace(tablePart),
            })
        }
    }
    
    // Extract filters
    if strings.Contains(strings.ToLower(query), "where") {
        // Simple filter extraction
        wherePart := strings.Split(strings.ToLower(query), "where")[1]
        if strings.Contains(wherePart, "order by") {
            wherePart = strings.Split(wherePart, "order by")[0]
        }
        if strings.Contains(wherePart, "group by") {
            wherePart = strings.Split(wherePart, "group by")[0]
        }
        
        analysis.Filters = append(analysis.Filters, FilterInfo{
            Condition: strings.TrimSpace(wherePart),
        })
    }
    
    // Extract sorts
    if strings.Contains(strings.ToLower(query), "order by") {
        orderPart := strings.Split(strings.ToLower(query), "order by")[1]
        if strings.Contains(orderPart, "limit") {
            orderPart = strings.Split(orderPart, "limit")[0]
        }
        
        analysis.Sorts = append(analysis.Sorts, SortInfo{
            Column: strings.TrimSpace(orderPart),
        })
    }
    
    return analysis
}

func (qo *QueryOptimizer) generateExecutionPlan(ctx context.Context, query string, analysis *QueryAnalysis) (*QueryPlan, error) {
    start := time.Now()
    
    // Execute EXPLAIN to get the actual plan
    var result []map[string]interface{}
    if err := qo.db.Raw("EXPLAIN (FORMAT JSON) "+query).Scan(&result).Error; err != nil {
        return nil, fmt.Errorf("failed to explain query: %w", err)
    }
    
    executionTime := time.Since(start)
    
    // Parse the execution plan
    plan := &QueryPlan{
        ID:           generatePlanID(),
        Query:        query,
        ExecutionTime: executionTime,
        Plan:         fmt.Sprintf("%v", result),
    }
    
    // Calculate cost based on execution time and complexity
    plan.Cost = qo.calculateCost(analysis, executionTime)
    
    // Identify recommended indexes
    plan.Indexes = qo.recommendIndexes(analysis)
    
    // Generate warnings
    plan.Warnings = qo.generateWarnings(analysis, executionTime)
    
    return plan, nil
}

func (qo *QueryOptimizer) optimizePlan(plan *QueryPlan, analysis *QueryAnalysis) *QueryPlan {
    // Apply optimizations
    optimized := *plan
    
    // Optimize based on analysis
    if len(analysis.Filters) > 0 {
        // Add index recommendations for filters
        for _, filter := range analysis.Filters {
            if strings.Contains(filter.Condition, "=") {
                column := strings.Split(filter.Condition, "=")[0]
                optimized.Indexes = append(optimized.Indexes, fmt.Sprintf("idx_%s", strings.TrimSpace(column)))
            }
        }
    }
    
    // Optimize joins
    if len(analysis.Joins) > 0 {
        // Add foreign key index recommendations
        for _, join := range analysis.Joins {
            optimized.Indexes = append(optimized.Indexes, fmt.Sprintf("idx_%s_id", join.Table))
        }
    }
    
    // Optimize sorts
    if len(analysis.Sorts) > 0 {
        for _, sort := range analysis.Sorts {
            optimized.Indexes = append(optimized.Indexes, fmt.Sprintf("idx_%s", strings.TrimSpace(sort.Column)))
        }
    }
    
    return &optimized
}

func (qo *QueryOptimizer) calculateCost(analysis *QueryAnalysis, executionTime time.Duration) float64 {
    cost := float64(executionTime.Nanoseconds()) / 1000000.0 // Convert to milliseconds
    
    // Add complexity factors
    cost += float64(len(analysis.Tables)) * 10
    cost += float64(len(analysis.Joins)) * 50
    cost += float64(len(analysis.Filters)) * 5
    cost += float64(len(analysis.Sorts)) * 20
    cost += float64(len(analysis.Aggregations)) * 30
    
    return cost
}

func (qo *QueryOptimizer) recommendIndexes(analysis *QueryAnalysis) []string {
    var indexes []string
    
    // Recommend indexes for filters
    for _, filter := range analysis.Filters {
        if strings.Contains(filter.Condition, "=") {
            column := strings.Split(filter.Condition, "=")[0]
            indexes = append(indexes, fmt.Sprintf("idx_%s", strings.TrimSpace(column)))
        }
    }
    
    // Recommend indexes for joins
    for _, join := range analysis.Joins {
        indexes = append(indexes, fmt.Sprintf("idx_%s_id", join.Table))
    }
    
    // Recommend indexes for sorts
    for _, sort := range analysis.Sorts {
        indexes = append(indexes, fmt.Sprintf("idx_%s", strings.TrimSpace(sort.Column)))
    }
    
    return indexes
}

func (qo *QueryOptimizer) generateWarnings(analysis *QueryAnalysis, executionTime time.Duration) []string {
    var warnings []string
    
    // Check for missing indexes
    if len(analysis.Filters) > 0 && len(qo.recommendIndexes(analysis)) == 0 {
        warnings = append(warnings, "No indexes found for WHERE conditions")
    }
    
    // Check for expensive operations
    if executionTime > 100*time.Millisecond {
        warnings = append(warnings, "Query execution time is high")
    }
    
    // Check for cartesian products
    if len(analysis.Tables) > 1 && len(analysis.Joins) == 0 {
        warnings = append(warnings, "Potential cartesian product detected")
    }
    
    // Check for missing LIMIT
    if !strings.Contains(strings.ToLower(analysis.Query), "limit") {
        warnings = append(warnings, "Query missing LIMIT clause")
    }
    
    return warnings
}

// Query Cache Implementation
func NewQueryCache() *QueryCache {
    return &QueryCache{
        cache: make(map[string]*CachedQuery),
    }
}

func (qc *QueryCache) Get(query string) (*QueryPlan, bool) {
    qc.mutex.RLock()
    defer qc.mutex.RUnlock()
    
    cached, exists := qc.cache[query]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(cached.ExpiresAt) {
        delete(qc.cache, query)
        return nil, false
    }
    
    cached.HitCount++
    cached.LastUsed = time.Now()
    
    return cached.Result.(*QueryPlan), true
}

func (qc *QueryCache) Set(query string, plan *QueryPlan) {
    qc.mutex.Lock()
    defer qc.mutex.Unlock()
    
    qc.cache[query] = &CachedQuery{
        Result:    plan,
        ExpiresAt: time.Now().Add(1 * time.Hour),
        HitCount:  0,
        LastUsed:  time.Now(),
    }
}

// Stats Collector Implementation
func NewStatsCollector() *StatsCollector {
    return &StatsCollector{
        stats: make(map[string]*QueryStats),
    }
}

func (sc *StatsCollector) RecordQuery(query string, executionTime time.Duration, rowCount int64, indexUsed string, cacheHit bool) {
    sc.mutex.Lock()
    defer sc.mutex.Unlock()
    
    stats := &QueryStats{
        Query:         query,
        ExecutionTime: executionTime,
        RowCount:      rowCount,
        IndexUsed:     indexUsed,
        CacheHit:      cacheHit,
        Timestamp:     time.Now(),
    }
    
    sc.stats[query] = stats
}

func (sc *StatsCollector) GetStats() map[string]*QueryStats {
    sc.mutex.RLock()
    defer sc.mutex.RUnlock()
    
    result := make(map[string]*QueryStats)
    for k, v := range sc.stats {
        result[k] = v
    }
    return result
}
```

## 🚀 Advanced Data Engineering

### Real-Time Data Pipeline
```go
// Real-Time Data Pipeline with Apache Kafka
package data

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"
    
    "github.com/Shopify/sarama"
    "github.com/redis/go-redis/v9"
    "go.mongodb.org/mongo-driver/mongo"
)

type RealTimeDataPipeline struct {
    kafkaProducer sarama.SyncProducer
    kafkaConsumer sarama.ConsumerGroup
    redisClient   *redis.Client
    mongoClient   *mongo.Client
    processors    map[string]DataProcessor
}

type DataProcessor interface {
    Process(ctx context.Context, data []byte) error
    GetTopic() string
    GetGroup() string
}

type UserEventProcessor struct {
    mongoClient *mongo.Client
    redisClient *redis.Client
}

func (uep *UserEventProcessor) Process(ctx context.Context, data []byte) error {
    var event UserEvent
    if err := json.Unmarshal(data, &event); err != nil {
        return fmt.Errorf("failed to unmarshal user event: %w", err)
    }
    
    // Process the event
    switch event.Type {
    case "user_created":
        return uep.processUserCreated(ctx, &event)
    case "user_updated":
        return uep.processUserUpdated(ctx, &event)
    case "user_deleted":
        return uep.processUserDeleted(ctx, &event)
    default:
        return fmt.Errorf("unknown event type: %s", event.Type)
    }
}

func (uep *UserEventProcessor) GetTopic() string {
    return "user-events"
}

func (uep *UserEventProcessor) GetGroup() string {
    return "user-processor"
}

func (uep *UserEventProcessor) processUserCreated(ctx context.Context, event *UserEvent) error {
    // Store in MongoDB
    collection := uep.mongoClient.Database("analytics").Collection("users")
    _, err := collection.InsertOne(ctx, event.Data)
    if err != nil {
        return fmt.Errorf("failed to insert user in MongoDB: %w", err)
    }
    
    // Update Redis cache
    key := fmt.Sprintf("user:%s", event.Data["id"])
    userData, err := json.Marshal(event.Data)
    if err != nil {
        return fmt.Errorf("failed to marshal user data: %w", err)
    }
    
    if err := uep.redisClient.Set(ctx, key, userData, 24*time.Hour).Err(); err != nil {
        return fmt.Errorf("failed to cache user in Redis: %w", err)
    }
    
    // Update user count
    if err := uep.redisClient.Incr(ctx, "user_count").Err(); err != nil {
        return fmt.Errorf("failed to update user count: %w", err)
    }
    
    return nil
}

func (uep *UserEventProcessor) processUserUpdated(ctx context.Context, event *UserEvent) error {
    // Update in MongoDB
    collection := uep.mongoClient.Database("analytics").Collection("users")
    filter := map[string]interface{}{"id": event.Data["id"]}
    update := map[string]interface{}{"$set": event.Data}
    
    _, err := collection.UpdateOne(ctx, filter, update)
    if err != nil {
        return fmt.Errorf("failed to update user in MongoDB: %w", err)
    }
    
    // Update Redis cache
    key := fmt.Sprintf("user:%s", event.Data["id"])
    userData, err := json.Marshal(event.Data)
    if err != nil {
        return fmt.Errorf("failed to marshal user data: %w", err)
    }
    
    if err := uep.redisClient.Set(ctx, key, userData, 24*time.Hour).Err(); err != nil {
        return fmt.Errorf("failed to cache user in Redis: %w", err)
    }
    
    return nil
}

func (uep *UserEventProcessor) processUserDeleted(ctx context.Context, event *UserEvent) error {
    // Delete from MongoDB
    collection := uep.mongoClient.Database("analytics").Collection("users")
    filter := map[string]interface{}{"id": event.Data["id"]}
    
    _, err := collection.DeleteOne(ctx, filter)
    if err != nil {
        return fmt.Errorf("failed to delete user from MongoDB: %w", err)
    }
    
    // Delete from Redis cache
    key := fmt.Sprintf("user:%s", event.Data["id"])
    if err := uep.redisClient.Del(ctx, key).Err(); err != nil {
        return fmt.Errorf("failed to delete user from Redis: %w", err)
    }
    
    // Decrease user count
    if err := uep.redisClient.Decr(ctx, "user_count").Err(); err != nil {
        return fmt.Errorf("failed to update user count: %w", err)
    }
    
    return nil
}

type UserEvent struct {
    ID        string                 `json:"id"`
    Type      string                 `json:"type"`
    Data      map[string]interface{} `json:"data"`
    Timestamp time.Time              `json:"timestamp"`
    Source    string                 `json:"source"`
}

func NewRealTimeDataPipeline(kafkaBrokers []string, redisClient *redis.Client, mongoClient *mongo.Client) (*RealTimeDataPipeline, error) {
    // Create Kafka producer
    producerConfig := sarama.NewConfig()
    producerConfig.Producer.RequiredAcks = sarama.WaitForAll
    producerConfig.Producer.Retry.Max = 3
    producerConfig.Producer.Return.Successes = true
    
    producer, err := sarama.NewSyncProducer(kafkaBrokers, producerConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create Kafka producer: %w", err)
    }
    
    // Create Kafka consumer
    consumerConfig := sarama.NewConfig()
    consumerConfig.Consumer.Group.Rebalance.Strategy = sarama.BalanceStrategyRoundRobin
    consumerConfig.Consumer.Offsets.Initial = sarama.OffsetNewest
    
    consumer, err := sarama.NewConsumerGroup(kafkaBrokers, "data-pipeline", consumerConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create Kafka consumer: %w", err)
    }
    
    pipeline := &RealTimeDataPipeline{
        kafkaProducer: producer,
        kafkaConsumer: consumer,
        redisClient:   redisClient,
        mongoClient:   mongoClient,
        processors:    make(map[string]DataProcessor),
    }
    
    // Register processors
    pipeline.RegisterProcessor(&UserEventProcessor{
        mongoClient: mongoClient,
        redisClient: redisClient,
    })
    
    return pipeline, nil
}

func (rtdp *RealTimeDataPipeline) RegisterProcessor(processor DataProcessor) {
    rtdp.processors[processor.GetTopic()] = processor
}

func (rtdp *RealTimeDataPipeline) PublishEvent(ctx context.Context, topic string, event interface{}) error {
    data, err := json.Marshal(event)
    if err != nil {
        return fmt.Errorf("failed to marshal event: %w", err)
    }
    
    message := &sarama.ProducerMessage{
        Topic: topic,
        Value: sarama.ByteEncoder(data),
    }
    
    partition, offset, err := rtdp.kafkaProducer.SendMessage(message)
    if err != nil {
        return fmt.Errorf("failed to send message: %w", err)
    }
    
    log.Printf("Message sent to topic %s, partition %d, offset %d", topic, partition, offset)
    return nil
}

func (rtdp *RealTimeDataPipeline) StartConsuming(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            if err := rtdp.kafkaConsumer.Consume(ctx, rtdp.getTopics(), rtdp); err != nil {
                log.Printf("Error consuming messages: %v", err)
                time.Sleep(1 * time.Second)
            }
        }
    }
}

func (rtdp *RealTimeDataPipeline) getTopics() []string {
    var topics []string
    for topic := range rtdp.processors {
        topics = append(topics, topic)
    }
    return topics
}

// Implement sarama.ConsumerGroupHandler
func (rtdp *RealTimeDataPipeline) Setup(sarama.ConsumerGroupSession) error {
    return nil
}

func (rtdp *RealTimeDataPipeline) Cleanup(sarama.ConsumerGroupSession) error {
    return nil
}

func (rtdp *RealTimeDataPipeline) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            processor, exists := rtdp.processors[message.Topic]
            if !exists {
                log.Printf("No processor found for topic: %s", message.Topic)
                continue
            }
            
            if err := processor.Process(context.Background(), message.Value); err != nil {
                log.Printf("Error processing message: %v", err)
                continue
            }
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}
```

## 🎯 Best Practices

### Database Design Principles
1. **Normalization**: Properly normalize data to reduce redundancy
2. **Indexing**: Create appropriate indexes for query performance
3. **Partitioning**: Partition large tables for better performance
4. **Sharding**: Distribute data across multiple servers
5. **Caching**: Implement multi-level caching strategies

### Performance Optimization
1. **Query Optimization**: Use query analyzers and optimizers
2. **Connection Pooling**: Implement proper connection pooling
3. **Read Replicas**: Use read replicas for read-heavy workloads
4. **Batch Operations**: Use batch operations for bulk data operations
5. **Monitoring**: Implement comprehensive database monitoring

### Data Consistency
1. **ACID Properties**: Ensure ACID compliance for critical operations
2. **Eventual Consistency**: Use eventual consistency for distributed systems
3. **Conflict Resolution**: Implement conflict resolution strategies
4. **Data Validation**: Validate data at multiple levels
5. **Backup and Recovery**: Implement robust backup and recovery procedures

---

**Last Updated**: December 2024  
**Category**: Advanced Database Systems Comprehensive  
**Complexity**: Expert Level
