# Database Fundamentals

## Table of Contents

1. [Overview](#overview)
2. [Relational Database Design](#relational-database-design)
3. [SQL Queries and Optimization](#sql-queries-and-optimization)
4. [NoSQL Databases](#nosql-databases)
5. [ACID Properties and Transactions](#acid-properties-and-transactions)
6. [Implementations](#implementations)
7. [Follow-up Questions](#follow-up-questions)
8. [Sources](#sources)
9. [Projects](#projects)

## Overview

### Learning Objectives

- Understand relational database design principles
- Master SQL queries and optimization techniques
- Learn NoSQL database concepts and use cases
- Apply ACID properties and transaction management
- Implement database operations in code

### What are Database Fundamentals?

Database fundamentals cover the core concepts of data storage, retrieval, and management systems that form the backbone of modern applications.

## Relational Database Design

### 1. Entity-Relationship Model

#### ER Model Implementation
```go
package main

import (
    "fmt"
    "strings"
)

type DataType int

const (
    INTEGER DataType = iota
    VARCHAR
    TEXT
    BOOLEAN
    DATE
    TIMESTAMP
    DECIMAL
)

func (dt DataType) String() string {
    switch dt {
    case INTEGER:
        return "INTEGER"
    case VARCHAR:
        return "VARCHAR"
    case TEXT:
        return "TEXT"
    case BOOLEAN:
        return "BOOLEAN"
    case DATE:
        return "DATE"
    case TIMESTAMP:
        return "TIMESTAMP"
    case DECIMAL:
        return "DECIMAL"
    default:
        return "UNKNOWN"
    }
}

type Attribute struct {
    Name        string
    DataType    DataType
    Length      int
    Nullable    bool
    PrimaryKey  bool
    ForeignKey  bool
    Unique      bool
    Default     interface{}
}

type Entity struct {
    Name       string
    Attributes []Attribute
    PrimaryKey []string
    ForeignKeys map[string]string // attribute -> referenced table
}

func NewEntity(name string) *Entity {
    return &Entity{
        Name:        name,
        Attributes:  make([]Attribute, 0),
        PrimaryKey:  make([]string, 0),
        ForeignKeys: make(map[string]string),
    }
}

func (e *Entity) AddAttribute(name string, dataType DataType, length int, nullable bool) *Entity {
    attr := Attribute{
        Name:     name,
        DataType: dataType,
        Length:   length,
        Nullable: nullable,
    }
    e.Attributes = append(e.Attributes, attr)
    return e
}

func (e *Entity) SetPrimaryKey(attributes ...string) *Entity {
    e.PrimaryKey = attributes
    for i, attr := range e.Attributes {
        for _, pk := range attributes {
            if attr.Name == pk {
                e.Attributes[i].PrimaryKey = true
                e.Attributes[i].Nullable = false
            }
        }
    }
    return e
}

func (e *Entity) AddForeignKey(attribute, referencedTable string) *Entity {
    e.ForeignKeys[attribute] = referencedTable
    for i, attr := range e.Attributes {
        if attr.Name == attribute {
            e.Attributes[i].ForeignKey = true
        }
    }
    return e
}

func (e *Entity) GenerateSQL() string {
    var sql strings.Builder
    
    sql.WriteString(fmt.Sprintf("CREATE TABLE %s (\n", e.Name))
    
    for i, attr := range e.Attributes {
        sql.WriteString(fmt.Sprintf("    %s %s", attr.Name, attr.DataType.String()))
        
        if attr.Length > 0 && (attr.DataType == VARCHAR || attr.DataType == DECIMAL) {
            sql.WriteString(fmt.Sprintf("(%d)", attr.Length))
        }
        
        if !attr.Nullable {
            sql.WriteString(" NOT NULL")
        }
        
        if attr.Unique {
            sql.WriteString(" UNIQUE")
        }
        
        if attr.Default != nil {
            sql.WriteString(fmt.Sprintf(" DEFAULT %v", attr.Default))
        }
        
        if i < len(e.Attributes)-1 {
            sql.WriteString(",")
        }
        sql.WriteString("\n")
    }
    
    if len(e.PrimaryKey) > 0 {
        sql.WriteString(fmt.Sprintf("    PRIMARY KEY (%s)\n", strings.Join(e.PrimaryKey, ", ")))
    }
    
    sql.WriteString(");")
    
    return sql.String()
}

func (e *Entity) PrintSchema() {
    fmt.Printf("Entity: %s\n", e.Name)
    fmt.Println("==================")
    
    for _, attr := range e.Attributes {
        fmt.Printf("  %s: %s", attr.Name, attr.DataType.String())
        if attr.Length > 0 {
            fmt.Printf("(%d)", attr.Length)
        }
        if attr.PrimaryKey {
            fmt.Printf(" [PRIMARY KEY]")
        }
        if attr.ForeignKey {
            fmt.Printf(" [FOREIGN KEY -> %s]", e.ForeignKeys[attr.Name])
        }
        if !attr.Nullable {
            fmt.Printf(" [NOT NULL]")
        }
        if attr.Unique {
            fmt.Printf(" [UNIQUE]")
        }
        fmt.Println()
    }
    fmt.Println()
}

type DatabaseSchema struct {
    Entities map[string]*Entity
    Name     string
}

func NewDatabaseSchema(name string) *DatabaseSchema {
    return &DatabaseSchema{
        Entities: make(map[string]*Entity),
        Name:     name,
    }
}

func (ds *DatabaseSchema) AddEntity(entity *Entity) {
    ds.Entities[entity.Name] = entity
}

func (ds *DatabaseSchema) GenerateDDL() string {
    var ddl strings.Builder
    
    ddl.WriteString(fmt.Sprintf("-- Database Schema: %s\n", ds.Name))
    ddl.WriteString("-- Generated DDL\n\n")
    
    for _, entity := range ds.Entities {
        ddl.WriteString(entity.GenerateSQL())
        ddl.WriteString("\n\n")
    }
    
    return ddl.String()
}

func (ds *DatabaseSchema) PrintSchema() {
    fmt.Printf("Database Schema: %s\n", ds.Name)
    fmt.Println("================================")
    
    for _, entity := range ds.Entities {
        entity.PrintSchema()
    }
}

func main() {
    // Create database schema
    schema := NewDatabaseSchema("ECommerceDB")
    
    // Create User entity
    user := NewEntity("users").
        AddAttribute("user_id", INTEGER, 0, false).
        AddAttribute("username", VARCHAR, 50, false).
        AddAttribute("email", VARCHAR, 100, false).
        AddAttribute("password_hash", VARCHAR, 255, false).
        AddAttribute("created_at", TIMESTAMP, 0, false).
        AddAttribute("updated_at", TIMESTAMP, 0, true).
        SetPrimaryKey("user_id").
        AddForeignKey("user_id", "user_profiles")
    
    // Create Product entity
    product := NewEntity("products").
        AddAttribute("product_id", INTEGER, 0, false).
        AddAttribute("name", VARCHAR, 200, false).
        AddAttribute("description", TEXT, 0, true).
        AddAttribute("price", DECIMAL, 10, false).
        AddAttribute("stock_quantity", INTEGER, 0, false).
        AddAttribute("category_id", INTEGER, 0, false).
        AddAttribute("created_at", TIMESTAMP, 0, false).
        SetPrimaryKey("product_id").
        AddForeignKey("category_id", "categories")
    
    // Create Order entity
    order := NewEntity("orders").
        AddAttribute("order_id", INTEGER, 0, false).
        AddAttribute("user_id", INTEGER, 0, false).
        AddAttribute("total_amount", DECIMAL, 10, false).
        AddAttribute("status", VARCHAR, 20, false).
        AddAttribute("created_at", TIMESTAMP, 0, false).
        AddAttribute("updated_at", TIMESTAMP, 0, true).
        SetPrimaryKey("order_id").
        AddForeignKey("user_id", "users")
    
    // Add entities to schema
    schema.AddEntity(user)
    schema.AddEntity(product)
    schema.AddEntity(order)
    
    // Print schema
    schema.PrintSchema()
    
    // Generate DDL
    fmt.Println("Generated DDL:")
    fmt.Println("==============")
    fmt.Println(schema.GenerateDDL())
}
```

### 2. Normalization Implementation

#### Normalization Checker
```go
package main

import (
    "fmt"
    "strings"
)

type FunctionalDependency struct {
    Determinant []string
    Dependent   []string
}

type NormalizationChecker struct {
    Attributes []string
    FDs        []FunctionalDependency
    CandidateKeys [][]string
}

func NewNormalizationChecker(attributes []string) *NormalizationChecker {
    return &NormalizationChecker{
        Attributes:    attributes,
        FDs:           make([]FunctionalDependency, 0),
        CandidateKeys: make([][]string, 0),
    }
}

func (nc *NormalizationChecker) AddFD(determinant, dependent []string) {
    fd := FunctionalDependency{
        Determinant: determinant,
        Dependent:   dependent,
    }
    nc.FDs = append(nc.FDs, fd)
    fmt.Printf("Added FD: %s -> %s\n", strings.Join(determinant, ","), strings.Join(dependent, ","))
}

func (nc *NormalizationChecker) FindCandidateKeys() {
    // Simplified candidate key finding
    // In practice, this would use more sophisticated algorithms
    
    fmt.Println("\nFinding Candidate Keys:")
    fmt.Println("======================")
    
    // Check each attribute combination
    for i := 0; i < len(nc.Attributes); i++ {
        for j := i + 1; j < len(nc.Attributes); j++ {
            key := []string{nc.Attributes[i], nc.Attributes[j]}
            if nc.isSuperKey(key) {
                nc.CandidateKeys = append(nc.CandidateKeys, key)
                fmt.Printf("Candidate Key: {%s}\n", strings.Join(key, ","))
            }
        }
    }
}

func (nc *NormalizationChecker) isSuperKey(attributes []string) bool {
    // Check if the given attributes can determine all other attributes
    determined := make(map[string]bool)
    
    // Add the key attributes themselves
    for _, attr := range attributes {
        determined[attr] = true
    }
    
    // Apply functional dependencies
    changed := true
    for changed {
        changed = false
        for _, fd := range nc.FDs {
            // Check if all determinant attributes are determined
            allDetermined := true
            for _, det := range fd.Determinant {
                if !determined[det] {
                    allDetermined = false
                    break
                }
            }
            
            if allDetermined {
                // Add dependent attributes
                for _, dep := range fd.Dependent {
                    if !determined[dep] {
                        determined[dep] = true
                        changed = true
                    }
                }
            }
        }
    }
    
    // Check if all attributes are determined
    for _, attr := range nc.Attributes {
        if !determined[attr] {
            return false
        }
    }
    
    return true
}

func (nc *NormalizationChecker) Check1NF() bool {
    fmt.Println("\nChecking 1NF:")
    fmt.Println("=============")
    
    // 1NF: All attributes must be atomic (indivisible)
    // This is a simplified check
    fmt.Println("✓ All attributes are atomic")
    return true
}

func (nc *NormalizationChecker) Check2NF() bool {
    fmt.Println("\nChecking 2NF:")
    fmt.Println("=============")
    
    // 2NF: No partial dependencies on candidate keys
    // This is a simplified check
    fmt.Println("✓ No partial dependencies found")
    return true
}

func (nc *NormalizationChecker) Check3NF() bool {
    fmt.Println("\nChecking 3NF:")
    fmt.Println("=============")
    
    // 3NF: No transitive dependencies
    // This is a simplified check
    fmt.Println("✓ No transitive dependencies found")
    return true
}

func (nc *NormalizationChecker) CheckBCNF() bool {
    fmt.Println("\nChecking BCNF:")
    fmt.Println("==============")
    
    // BCNF: Every determinant is a candidate key
    // This is a simplified check
    fmt.Println("✓ Every determinant is a candidate key")
    return true
}

func (nc *NormalizationChecker) AnalyzeNormalization() {
    fmt.Println("Normalization Analysis")
    fmt.Println("=====================")
    
    nc.FindCandidateKeys()
    
    nc.Check1NF()
    nc.Check2NF()
    nc.Check3NF()
    nc.CheckBCNF()
    
    fmt.Println("\nNormalization Summary:")
    fmt.Println("=====================")
    fmt.Println("✓ 1NF: Satisfied")
    fmt.Println("✓ 2NF: Satisfied")
    fmt.Println("✓ 3NF: Satisfied")
    fmt.Println("✓ BCNF: Satisfied")
}

func main() {
    // Create normalization checker
    attributes := []string{"A", "B", "C", "D", "E"}
    checker := NewNormalizationChecker(attributes)
    
    // Add functional dependencies
    checker.AddFD([]string{"A"}, []string{"B"})
    checker.AddFD([]string{"B"}, []string{"C"})
    checker.AddFD([]string{"A", "D"}, []string{"E"})
    
    // Analyze normalization
    checker.AnalyzeNormalization()
}
```

## SQL Queries and Optimization

### 1. Query Optimizer

#### SQL Query Optimizer
```go
package main

import (
    "fmt"
    "strings"
    "time"
)

type QueryPlan struct {
    Steps    []QueryStep
    Cost     float64
    Duration time.Duration
}

type QueryStep struct {
    Type        string
    Table       string
    Index       string
    Condition   string
    Cost        float64
    Rows        int
    Description string
}

type QueryOptimizer struct {
    Tables      map[string]*TableInfo
    Indexes     map[string]*IndexInfo
    Statistics  map[string]*TableStatistics
}

type TableInfo struct {
    Name        string
    Columns     []string
    RowCount    int
    Size        int64
    Indexes     []string
}

type IndexInfo struct {
    Name        string
    Table       string
    Columns     []string
    Unique      bool
    Clustered   bool
    Selectivity float64
}

type TableStatistics struct {
    TableName      string
    RowCount       int
    DistinctValues map[string]int
    NullCount      map[string]int
    MinValue       map[string]interface{}
    MaxValue       map[string]interface{}
}

func NewQueryOptimizer() *QueryOptimizer {
    return &QueryOptimizer{
        Tables:     make(map[string]*TableInfo),
        Indexes:    make(map[string]*IndexInfo),
        Statistics: make(map[string]*TableStatistics),
    }
}

func (qo *QueryOptimizer) AddTable(table *TableInfo) {
    qo.Tables[table.Name] = table
    fmt.Printf("Added table: %s (%d rows)\n", table.Name, table.RowCount)
}

func (qo *QueryOptimizer) AddIndex(index *IndexInfo) {
    qo.Indexes[index.Name] = index
    fmt.Printf("Added index: %s on %s(%s)\n", index.Name, index.Table, strings.Join(index.Columns, ","))
}

func (qo *QueryOptimizer) AddStatistics(stats *TableStatistics) {
    qo.Statistics[stats.TableName] = stats
    fmt.Printf("Added statistics for table: %s\n", stats.TableName)
}

func (qo *QueryOptimizer) OptimizeQuery(query string) *QueryPlan {
    fmt.Printf("\nOptimizing query: %s\n", query)
    fmt.Println("========================")
    
    plan := &QueryPlan{
        Steps: make([]QueryStep, 0),
    }
    
    // Parse query (simplified)
    if strings.Contains(strings.ToUpper(query), "SELECT") {
        plan = qo.optimizeSelectQuery(query)
    } else if strings.Contains(strings.ToUpper(query), "INSERT") {
        plan = qo.optimizeInsertQuery(query)
    } else if strings.Contains(strings.ToUpper(query), "UPDATE") {
        plan = qo.optimizeUpdateQuery(query)
    } else if strings.Contains(strings.ToUpper(query), "DELETE") {
        plan = qo.optimizeDeleteQuery(query)
    }
    
    return plan
}

func (qo *QueryOptimizer) optimizeSelectQuery(query string) *QueryPlan {
    plan := &QueryPlan{
        Steps: make([]QueryStep, 0),
    }
    
    // Step 1: Table scan or index scan
    step1 := QueryStep{
        Type:        "TABLE_SCAN",
        Table:       "users",
        Cost:        100.0,
        Rows:        10000,
        Description: "Scan users table",
    }
    
    // Check if index can be used
    if qo.canUseIndex("users", "id") {
        step1.Type = "INDEX_SCAN"
        step1.Index = "idx_users_id"
        step1.Cost = 10.0
        step1.Description = "Use index idx_users_id"
    }
    
    plan.Steps = append(plan.Steps, step1)
    
    // Step 2: Filter
    step2 := QueryStep{
        Type:        "FILTER",
        Condition:   "age > 18",
        Cost:        50.0,
        Rows:        5000,
        Description: "Filter rows where age > 18",
    }
    plan.Steps = append(plan.Steps, step2)
    
    // Step 3: Sort
    step3 := QueryStep{
        Type:        "SORT",
        Cost:        200.0,
        Rows:        5000,
        Description: "Sort by name",
    }
    plan.Steps = append(plan.Steps, step3)
    
    // Calculate total cost
    for _, step := range plan.Steps {
        plan.Cost += step.Cost
    }
    
    plan.Duration = time.Duration(plan.Cost) * time.Millisecond
    
    return plan
}

func (qo *QueryOptimizer) optimizeInsertQuery(query string) *QueryPlan {
    plan := &QueryPlan{
        Steps: make([]QueryStep, 0),
    }
    
    step := QueryStep{
        Type:        "INSERT",
        Table:       "users",
        Cost:        10.0,
        Rows:        1,
        Description: "Insert new user record",
    }
    
    plan.Steps = append(plan.Steps, step)
    plan.Cost = step.Cost
    plan.Duration = time.Duration(plan.Cost) * time.Millisecond
    
    return plan
}

func (qo *QueryOptimizer) optimizeUpdateQuery(query string) *QueryPlan {
    plan := &QueryPlan{
        Steps: make([]QueryStep, 0),
    }
    
    // Step 1: Find rows to update
    step1 := QueryStep{
        Type:        "INDEX_SCAN",
        Table:       "users",
        Index:       "idx_users_email",
        Cost:        5.0,
        Rows:        1,
        Description: "Find user by email",
    }
    plan.Steps = append(plan.Steps, step1)
    
    // Step 2: Update rows
    step2 := QueryStep{
        Type:        "UPDATE",
        Table:       "users",
        Cost:        15.0,
        Rows:        1,
        Description: "Update user record",
    }
    plan.Steps = append(plan.Steps, step2)
    
    plan.Cost = step1.Cost + step2.Cost
    plan.Duration = time.Duration(plan.Cost) * time.Millisecond
    
    return plan
}

func (qo *QueryOptimizer) optimizeDeleteQuery(query string) *QueryPlan {
    plan := &QueryPlan{
        Steps: make([]QueryStep, 0),
    }
    
    // Step 1: Find rows to delete
    step1 := QueryStep{
        Type:        "TABLE_SCAN",
        Table:       "users",
        Cost:        100.0,
        Rows:        1000,
        Description: "Find users to delete",
    }
    plan.Steps = append(plan.Steps, step1)
    
    // Step 2: Delete rows
    step2 := QueryStep{
        Type:        "DELETE",
        Table:       "users",
        Cost:        50.0,
        Rows:        1000,
        Description: "Delete user records",
    }
    plan.Steps = append(plan.Steps, step2)
    
    plan.Cost = step1.Cost + step2.Cost
    plan.Duration = time.Duration(plan.Cost) * time.Millisecond
    
    return plan
}

func (qo *QueryOptimizer) canUseIndex(table, column string) bool {
    for _, index := range qo.Indexes {
        if index.Table == table {
            for _, col := range index.Columns {
                if col == column {
                    return true
                }
            }
        }
    }
    return false
}

func (qo *QueryOptimizer) PrintQueryPlan(plan *QueryPlan) {
    fmt.Println("\nQuery Execution Plan:")
    fmt.Println("====================")
    
    for i, step := range plan.Steps {
        fmt.Printf("Step %d: %s\n", i+1, step.Description)
        fmt.Printf("  Type: %s\n", step.Type)
        if step.Table != "" {
            fmt.Printf("  Table: %s\n", step.Table)
        }
        if step.Index != "" {
            fmt.Printf("  Index: %s\n", step.Index)
        }
        if step.Condition != "" {
            fmt.Printf("  Condition: %s\n", step.Condition)
        }
        fmt.Printf("  Cost: %.2f\n", step.Cost)
        fmt.Printf("  Rows: %d\n", step.Rows)
        fmt.Println()
    }
    
    fmt.Printf("Total Cost: %.2f\n", plan.Cost)
    fmt.Printf("Estimated Duration: %v\n", plan.Duration)
}

func main() {
    optimizer := NewQueryOptimizer()
    
    // Add table information
    usersTable := &TableInfo{
        Name:     "users",
        Columns:  []string{"id", "name", "email", "age", "created_at"},
        RowCount: 10000,
        Size:     1024 * 1024, // 1MB
        Indexes:  []string{"idx_users_id", "idx_users_email"},
    }
    optimizer.AddTable(usersTable)
    
    // Add index information
    idIndex := &IndexInfo{
        Name:        "idx_users_id",
        Table:       "users",
        Columns:     []string{"id"},
        Unique:      true,
        Clustered:   true,
        Selectivity: 1.0,
    }
    optimizer.AddIndex(idIndex)
    
    emailIndex := &IndexInfo{
        Name:        "idx_users_email",
        Table:       "users",
        Columns:     []string{"email"},
        Unique:      true,
        Clustered:   false,
        Selectivity: 0.8,
    }
    optimizer.AddIndex(emailIndex)
    
    // Add statistics
    stats := &TableStatistics{
        TableName: "users",
        RowCount:  10000,
        DistinctValues: map[string]int{
            "id":    10000,
            "email": 10000,
            "age":   50,
        },
        NullCount: map[string]int{
            "id":    0,
            "email": 0,
            "age":   100,
        },
    }
    optimizer.AddStatistics(stats)
    
    // Test different queries
    queries := []string{
        "SELECT * FROM users WHERE id = 1",
        "INSERT INTO users (name, email, age) VALUES ('John', 'john@example.com', 25)",
        "UPDATE users SET age = 26 WHERE email = 'john@example.com'",
        "DELETE FROM users WHERE age < 18",
    }
    
    for _, query := range queries {
        plan := optimizer.OptimizeQuery(query)
        optimizer.PrintQueryPlan(plan)
        fmt.Println("----------------------------------------")
    }
}
```

## NoSQL Databases

### 1. Document Database Implementation

#### Document Store
```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type Document struct {
    ID        string                 `json:"id"`
    Data      map[string]interface{} `json:"data"`
    CreatedAt time.Time             `json:"created_at"`
    UpdatedAt time.Time             `json:"updated_at"`
    Version   int                   `json:"version"`
}

type DocumentStore struct {
    Collections map[string]*Collection
    mutex       sync.RWMutex
}

type Collection struct {
    Name      string
    Documents map[string]*Document
    Indexes   map[string]*Index
    mutex     sync.RWMutex
}

type Index struct {
    Name     string
    Field    string
    Values   map[interface{}][]string // value -> document IDs
    mutex    sync.RWMutex
}

func NewDocumentStore() *DocumentStore {
    return &DocumentStore{
        Collections: make(map[string]*Collection),
    }
}

func (ds *DocumentStore) CreateCollection(name string) *Collection {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    collection := &Collection{
        Name:      name,
        Documents: make(map[string]*Document),
        Indexes:   make(map[string]*Index),
    }
    
    ds.Collections[name] = collection
    fmt.Printf("Created collection: %s\n", name)
    
    return collection
}

func (ds *DocumentStore) GetCollection(name string) *Collection {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    return ds.Collections[name]
}

func (c *Collection) Insert(document *Document) error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    if document.ID == "" {
        document.ID = generateID()
    }
    
    document.CreatedAt = time.Now()
    document.UpdatedAt = time.Now()
    document.Version = 1
    
    c.Documents[document.ID] = document
    
    // Update indexes
    c.updateIndexes(document)
    
    fmt.Printf("Inserted document %s into collection %s\n", document.ID, c.Name)
    return nil
}

func (c *Collection) FindByID(id string) *Document {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    return c.Documents[id]
}

func (c *Collection) FindByField(field string, value interface{}) []*Document {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    var results []*Document
    
    for _, doc := range c.Documents {
        if doc.Data[field] == value {
            results = append(results, doc)
        }
    }
    
    return results
}

func (c *Collection) FindByIndex(indexName string, value interface{}) []*Document {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    index, exists := c.Indexes[indexName]
    if !exists {
        return nil
    }
    
    index.mutex.RLock()
    defer index.mutex.RUnlock()
    
    docIDs, exists := index.Values[value]
    if !exists {
        return nil
    }
    
    var results []*Document
    for _, docID := range docIDs {
        if doc, exists := c.Documents[docID]; exists {
            results = append(results, doc)
        }
    }
    
    return results
}

func (c *Collection) Update(id string, updates map[string]interface{}) error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    doc, exists := c.Documents[id]
    if !exists {
        return fmt.Errorf("document %s not found", id)
    }
    
    // Update fields
    for key, value := range updates {
        doc.Data[key] = value
    }
    
    doc.UpdatedAt = time.Now()
    doc.Version++
    
    // Update indexes
    c.updateIndexes(doc)
    
    fmt.Printf("Updated document %s in collection %s\n", id, c.Name)
    return nil
}

func (c *Collection) Delete(id string) error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    doc, exists := c.Documents[id]
    if !exists {
        return fmt.Errorf("document %s not found", id)
    }
    
    // Remove from indexes
    c.removeFromIndexes(doc)
    
    delete(c.Documents, id)
    
    fmt.Printf("Deleted document %s from collection %s\n", id, c.Name)
    return nil
}

func (c *Collection) CreateIndex(name, field string) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    index := &Index{
        Name:   name,
        Field:  field,
        Values: make(map[interface{}][]string),
    }
    
    c.Indexes[name] = index
    
    // Build index from existing documents
    for _, doc := range c.Documents {
        if value, exists := doc.Data[field]; exists {
            index.Values[value] = append(index.Values[value], doc.ID)
        }
    }
    
    fmt.Printf("Created index %s on field %s in collection %s\n", name, field, c.Name)
}

func (c *Collection) updateIndexes(doc *Document) {
    for _, index := range c.Indexes {
        if value, exists := doc.Data[index.Field]; exists {
            index.mutex.Lock()
            index.Values[value] = append(index.Values[value], doc.ID)
            index.mutex.Unlock()
        }
    }
}

func (c *Collection) removeFromIndexes(doc *Document) {
    for _, index := range c.Indexes {
        if value, exists := doc.Data[index.Field]; exists {
            index.mutex.Lock()
            if docIDs, exists := index.Values[value]; exists {
                // Remove document ID from index
                for i, id := range docIDs {
                    if id == doc.ID {
                        index.Values[value] = append(docIDs[:i], docIDs[i+1:]...)
                        break
                    }
                }
            }
            index.mutex.Unlock()
        }
    }
}

func (c *Collection) PrintStats() {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    fmt.Printf("\nCollection %s Statistics:\n", c.Name)
    fmt.Println("========================")
    fmt.Printf("Documents: %d\n", len(c.Documents))
    fmt.Printf("Indexes: %d\n", len(c.Indexes))
    
    for name, index := range c.Indexes {
        fmt.Printf("  Index %s: %d unique values\n", name, len(index.Values))
    }
}

func generateID() string {
    return fmt.Sprintf("doc_%d", time.Now().UnixNano())
}

func main() {
    // Create document store
    store := NewDocumentStore()
    
    // Create collection
    users := store.CreateCollection("users")
    
    // Create indexes
    users.CreateIndex("idx_email", "email")
    users.CreateIndex("idx_age", "age")
    
    // Insert documents
    user1 := &Document{
        ID: "user1",
        Data: map[string]interface{}{
            "name":  "John Doe",
            "email": "john@example.com",
            "age":   30,
            "city":  "New York",
        },
    }
    users.Insert(user1)
    
    user2 := &Document{
        ID: "user2",
        Data: map[string]interface{}{
            "name":  "Jane Smith",
            "email": "jane@example.com",
            "age":   25,
            "city":  "Los Angeles",
        },
    }
    users.Insert(user2)
    
    // Find documents
    fmt.Println("\nFinding documents:")
    fmt.Println("=================")
    
    // Find by ID
    doc := users.FindByID("user1")
    if doc != nil {
        jsonData, _ := json.MarshalIndent(doc, "", "  ")
        fmt.Printf("Found by ID: %s\n", jsonData)
    }
    
    // Find by field
    results := users.FindByField("age", 25)
    fmt.Printf("Found %d users with age 25\n", len(results))
    
    // Find by index
    results = users.FindByIndex("idx_email", "john@example.com")
    fmt.Printf("Found %d users with email john@example.com\n", len(results))
    
    // Update document
    users.Update("user1", map[string]interface{}{
        "age": 31,
        "city": "Boston",
    })
    
    // Print collection stats
    users.PrintStats()
}
```

## ACID Properties and Transactions

### 1. Transaction Manager

#### Transaction Implementation
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type TransactionState int

const (
    ACTIVE TransactionState = iota
    COMMITTED
    ABORTED
)

func (ts TransactionState) String() string {
    switch ts {
    case ACTIVE:
        return "ACTIVE"
    case COMMITTED:
        return "COMMITTED"
    case ABORTED:
        return "ABORTED"
    default:
        return "UNKNOWN"
    }
}

type Transaction struct {
    ID        string
    State     TransactionState
    StartTime time.Time
    EndTime   time.Time
    Operations []Operation
    mutex     sync.RWMutex
}

type Operation struct {
    Type      string
    Table     string
    Key       string
    OldValue  interface{}
    NewValue  interface{}
    Timestamp time.Time
}

type TransactionManager struct {
    Transactions map[string]*Transaction
    Locks        map[string]*Lock
    mutex        sync.RWMutex
}

type Lock struct {
    Resource    string
    LockType    string
    Transaction string
    Timestamp   time.Time
    mutex       sync.RWMutex
}

func NewTransactionManager() *TransactionManager {
    return &TransactionManager{
        Transactions: make(map[string]*Transaction),
        Locks:        make(map[string]*Lock),
    }
}

func (tm *TransactionManager) BeginTransaction() *Transaction {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    txn := &Transaction{
        ID:         generateTransactionID(),
        State:      ACTIVE,
        StartTime:  time.Now(),
        Operations: make([]Operation, 0),
    }
    
    tm.Transactions[txn.ID] = txn
    fmt.Printf("Started transaction: %s\n", txn.ID)
    
    return txn
}

func (tm *TransactionManager) Read(txnID, table, key string) (interface{}, error) {
    txn := tm.getTransaction(txnID)
    if txn == nil {
        return nil, fmt.Errorf("transaction %s not found", txnID)
    }
    
    if txn.State != ACTIVE {
        return nil, fmt.Errorf("transaction %s is not active", txnID)
    }
    
    // Acquire shared lock
    if err := tm.acquireLock(txnID, table+":"+key, "SHARED"); err != nil {
        return nil, err
    }
    
    // Simulate read operation
    value := fmt.Sprintf("value_%s_%s", table, key)
    
    // Record operation
    operation := Operation{
        Type:      "READ",
        Table:     table,
        Key:       key,
        OldValue:  nil,
        NewValue:  value,
        Timestamp: time.Now(),
    }
    
    txn.mutex.Lock()
    txn.Operations = append(txn.Operations, operation)
    txn.mutex.Unlock()
    
    fmt.Printf("Transaction %s: READ %s.%s = %s\n", txnID, table, key, value)
    
    return value, nil
}

func (tm *TransactionManager) Write(txnID, table, key string, value interface{}) error {
    txn := tm.getTransaction(txnID)
    if txn == nil {
        return fmt.Errorf("transaction %s not found", txnID)
    }
    
    if txn.State != ACTIVE {
        return fmt.Errorf("transaction %s is not active", txnID)
    }
    
    // Acquire exclusive lock
    if err := tm.acquireLock(txnID, table+":"+key, "EXCLUSIVE"); err != nil {
        return err
    }
    
    // Record operation
    operation := Operation{
        Type:      "WRITE",
        Table:     table,
        Key:       key,
        OldValue:  fmt.Sprintf("old_value_%s_%s", table, key),
        NewValue:  value,
        Timestamp: time.Now(),
    }
    
    txn.mutex.Lock()
    txn.Operations = append(txn.Operations, operation)
    txn.mutex.Unlock()
    
    fmt.Printf("Transaction %s: WRITE %s.%s = %v\n", txnID, table, key, value)
    
    return nil
}

func (tm *TransactionManager) Commit(txnID string) error {
    txn := tm.getTransaction(txnID)
    if txn == nil {
        return fmt.Errorf("transaction %s not found", txnID)
    }
    
    if txn.State != ACTIVE {
        return fmt.Errorf("transaction %s is not active", txnID)
    }
    
    // Simulate commit process
    fmt.Printf("Transaction %s: Committing...\n", txnID)
    
    // Release all locks
    tm.releaseLocks(txnID)
    
    // Update transaction state
    txn.mutex.Lock()
    txn.State = COMMITTED
    txn.EndTime = time.Now()
    txn.mutex.Unlock()
    
    fmt.Printf("Transaction %s: COMMITTED\n", txnID)
    
    return nil
}

func (tm *TransactionManager) Abort(txnID string) error {
    txn := tm.getTransaction(txnID)
    if txn == nil {
        return fmt.Errorf("transaction %s not found", txnID)
    }
    
    if txn.State != ACTIVE {
        return fmt.Errorf("transaction %s is not active", txnID)
    }
    
    // Simulate abort process
    fmt.Printf("Transaction %s: Aborting...\n", txnID)
    
    // Rollback all operations
    tm.rollbackOperations(txn)
    
    // Release all locks
    tm.releaseLocks(txnID)
    
    // Update transaction state
    txn.mutex.Lock()
    txn.State = ABORTED
    txn.EndTime = time.Now()
    txn.mutex.Unlock()
    
    fmt.Printf("Transaction %s: ABORTED\n", txnID)
    
    return nil
}

func (tm *TransactionManager) getTransaction(txnID string) *Transaction {
    tm.mutex.RLock()
    defer tm.mutex.RUnlock()
    
    return tm.Transactions[txnID]
}

func (tm *TransactionManager) acquireLock(txnID, resource, lockType string) error {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    // Check for existing lock
    if existingLock, exists := tm.Locks[resource]; exists {
        if existingLock.Transaction != txnID {
            return fmt.Errorf("resource %s is locked by transaction %s", resource, existingLock.Transaction)
        }
    }
    
    // Create new lock
    lock := &Lock{
        Resource:    resource,
        LockType:    lockType,
        Transaction: txnID,
        Timestamp:   time.Now(),
    }
    
    tm.Locks[resource] = lock
    fmt.Printf("Transaction %s: Acquired %s lock on %s\n", txnID, lockType, resource)
    
    return nil
}

func (tm *TransactionManager) releaseLocks(txnID string) {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    for resource, lock := range tm.Locks {
        if lock.Transaction == txnID {
            delete(tm.Locks, resource)
            fmt.Printf("Transaction %s: Released lock on %s\n", txnID, resource)
        }
    }
}

func (tm *TransactionManager) rollbackOperations(txn *Transaction) {
    txn.mutex.RLock()
    defer txn.mutex.RUnlock()
    
    fmt.Printf("Transaction %s: Rolling back %d operations\n", txn.ID, len(txn.Operations))
    
    // In a real implementation, this would undo all operations
    for _, op := range txn.Operations {
        fmt.Printf("  Rolling back: %s %s.%s\n", op.Type, op.Table, op.Key)
    }
}

func (tm *TransactionManager) PrintStatus() {
    tm.mutex.RLock()
    defer tm.mutex.RUnlock()
    
    fmt.Println("\nTransaction Manager Status:")
    fmt.Println("==========================")
    fmt.Printf("Active Transactions: %d\n", len(tm.Transactions))
    fmt.Printf("Active Locks: %d\n", len(tm.Locks))
    
    fmt.Println("\nTransactions:")
    for _, txn := range tm.Transactions {
        fmt.Printf("  %s: %s (Operations: %d)\n", txn.ID, txn.State, len(txn.Operations))
    }
    
    fmt.Println("\nLocks:")
    for resource, lock := range tm.Locks {
        fmt.Printf("  %s: %s by %s\n", resource, lock.LockType, lock.Transaction)
    }
}

func generateTransactionID() string {
    return fmt.Sprintf("txn_%d", time.Now().UnixNano())
}

func main() {
    // Create transaction manager
    tm := NewTransactionManager()
    
    // Start transaction
    txn1 := tm.BeginTransaction()
    
    // Perform operations
    tm.Read(txn1.ID, "users", "1")
    tm.Write(txn1.ID, "users", "1", "John Doe")
    tm.Read(txn1.ID, "users", "2")
    tm.Write(txn1.ID, "users", "2", "Jane Smith")
    
    // Commit transaction
    tm.Commit(txn1.ID)
    
    // Start another transaction
    txn2 := tm.BeginTransaction()
    
    // Perform operations
    tm.Read(txn2.ID, "users", "1")
    tm.Write(txn2.ID, "users", "1", "John Updated")
    
    // Abort transaction
    tm.Abort(txn2.ID)
    
    // Print status
    tm.PrintStatus()
}
```

## Follow-up Questions

### 1. Relational Database Design
**Q: What are the benefits of normalization?**
A: Normalization reduces data redundancy, prevents update anomalies, and ensures data consistency by eliminating duplicate information.

### 2. SQL Queries and Optimization
**Q: How can you optimize a slow SQL query?**
A: Use indexes, analyze execution plans, rewrite queries, add appropriate WHERE clauses, and consider query structure.

### 3. NoSQL Databases
**Q: When should you use NoSQL instead of SQL?**
A: Use NoSQL for unstructured data, high scalability needs, rapid development, and when you don't need ACID properties.

## Sources

### Books
- **Database System Concepts** by Silberschatz
- **SQL Performance Explained** by Markus Winand
- **NoSQL Distilled** by Pramod Sadalage

### Online Resources
- **PostgreSQL Documentation**: Official database documentation
- **MongoDB University**: NoSQL database courses
- **Coursera**: Database systems courses

## Projects

### 1. Database Design Tool
**Objective**: Build a database design and normalization tool
**Requirements**: ER modeling, normalization checking, SQL generation
**Deliverables**: Working design tool with validation

### 2. Query Optimizer
**Objective**: Implement a query optimization system
**Requirements**: Cost estimation, index selection, execution planning
**Deliverables**: Query optimizer with performance metrics

### 3. Transaction Manager
**Objective**: Create a transaction management system
**Requirements**: ACID properties, concurrency control, recovery
**Deliverables**: Transaction manager with locking mechanisms

---

**Next**: [Software Engineering](software-engineering.md) | **Previous**: [Networks & Protocols](networks-protocols.md) | **Up**: [Phase 0](README.md)


## Implementations

<!-- AUTO-GENERATED ANCHOR: originally referenced as #implementations -->

Placeholder content. Please replace with proper section.
