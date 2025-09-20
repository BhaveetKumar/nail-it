# NoSQL Databases

## Overview

This module covers NoSQL database concepts including document stores, key-value stores, column-family stores, and graph databases. These concepts are essential for understanding modern database systems that handle unstructured and semi-structured data.

## Table of Contents

1. [Document Stores](#document-stores)
2. [Key-Value Stores](#key-value-stores)
3. [Column-Family Stores](#column-family-stores)
4. [Graph Databases](#graph-databases)
5. [CAP Theorem](#cap-theorem)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Document Stores

### Theory

Document stores store data as documents (typically JSON, BSON, or XML) and provide rich query capabilities. They are schema-flexible and well-suited for content management and real-time web applications.

### Document Store Implementation

#### Golang Implementation

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
    Timestamp time.Time             `json:"timestamp"`
    Version   int                   `json:"version"`
}

type Query struct {
    Field    string
    Operator string
    Value    interface{}
}

type DocumentStore struct {
    documents map[string]*Document
    indexes   map[string]map[interface{}][]string
    mutex     sync.RWMutex
}

func NewDocumentStore() *DocumentStore {
    return &DocumentStore{
        documents: make(map[string]*Document),
        indexes:   make(map[string]map[interface{}][]string),
    }
}

func (ds *DocumentStore) Insert(id string, data map[string]interface{}) bool {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    if _, exists := ds.documents[id]; exists {
        fmt.Printf("Document with ID %s already exists\n", id)
        return false
    }
    
    document := &Document{
        ID:        id,
        Data:      data,
        Timestamp: time.Now(),
        Version:   1,
    }
    
    ds.documents[id] = document
    ds.updateIndexes(document)
    
    fmt.Printf("Inserted document with ID %s\n", id)
    return true
}

func (ds *DocumentStore) Update(id string, data map[string]interface{}) bool {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    document, exists := ds.documents[id]
    if !exists {
        fmt.Printf("Document with ID %s not found\n", id)
        return false
    }
    
    // Remove old indexes
    ds.removeIndexes(document)
    
    // Update document
    document.Data = data
    document.Timestamp = time.Now()
    document.Version++
    
    // Update indexes
    ds.updateIndexes(document)
    
    fmt.Printf("Updated document with ID %s\n", id)
    return true
}

func (ds *DocumentStore) Delete(id string) bool {
    ds.mutex.Lock()
    defer ds.mutex.Unlock()
    
    document, exists := ds.documents[id]
    if !exists {
        fmt.Printf("Document with ID %s not found\n", id)
        return false
    }
    
    // Remove indexes
    ds.removeIndexes(document)
    
    // Delete document
    delete(ds.documents, id)
    
    fmt.Printf("Deleted document with ID %s\n", id)
    return true
}

func (ds *DocumentStore) FindByID(id string) *Document {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    if document, exists := ds.documents[id]; exists {
        return document
    }
    return nil
}

func (ds *DocumentStore) FindByField(field string, value interface{}) []*Document {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    var results []*Document
    
    if index, exists := ds.indexes[field]; exists {
        if ids, exists := index[value]; exists {
            for _, id := range ids {
                if document, exists := ds.documents[id]; exists {
                    results = append(results, document)
                }
            }
        }
    }
    
    return results
}

func (ds *DocumentStore) FindByQuery(queries []Query) []*Document {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    var results []*Document
    
    for _, query := range queries {
        var matchingDocs []*Document
        
        if index, exists := ds.indexes[query.Field]; exists {
            for value, ids := range index {
                if ds.matchesQuery(value, query) {
                    for _, id := range ids {
                        if document, exists := ds.documents[id]; exists {
                            matchingDocs = append(matchingDocs, document)
                        }
                    }
                }
            }
        }
        
        if len(results) == 0 {
            results = matchingDocs
        } else {
            results = ds.intersect(results, matchingDocs)
        }
    }
    
    return results
}

func (ds *DocumentStore) matchesQuery(value interface{}, query Query) bool {
    switch query.Operator {
    case "=":
        return value == query.Value
    case "!=":
        return value != query.Value
    case ">":
        return ds.compareValues(value, query.Value) > 0
    case ">=":
        return ds.compareValues(value, query.Value) >= 0
    case "<":
        return ds.compareValues(value, query.Value) < 0
    case "<=":
        return ds.compareValues(value, query.Value) <= 0
    default:
        return false
    }
}

func (ds *DocumentStore) compareValues(v1, v2 interface{}) int {
    switch val1 := v1.(type) {
    case int:
        val2 := v2.(int)
        if val1 < val2 {
            return -1
        } else if val1 > val2 {
            return 1
        }
        return 0
    case string:
        val2 := v2.(string)
        if val1 < val2 {
            return -1
        } else if val1 > val2 {
            return 1
        }
        return 0
    default:
        return 0
    }
}

func (ds *DocumentStore) intersect(docs1, docs2 []*Document) []*Document {
    var result []*Document
    docMap := make(map[string]bool)
    
    for _, doc := range docs1 {
        docMap[doc.ID] = true
    }
    
    for _, doc := range docs2 {
        if docMap[doc.ID] {
            result = append(result, doc)
        }
    }
    
    return result
}

func (ds *DocumentStore) updateIndexes(document *Document) {
    for field, value := range document.Data {
        if ds.indexes[field] == nil {
            ds.indexes[field] = make(map[interface{}][]string)
        }
        
        if ids, exists := ds.indexes[field][value]; exists {
            ds.indexes[field][value] = append(ids, document.ID)
        } else {
            ds.indexes[field][value] = []string{document.ID}
        }
    }
}

func (ds *DocumentStore) removeIndexes(document *Document) {
    for field, value := range document.Data {
        if index, exists := ds.indexes[field]; exists {
            if ids, exists := index[value]; exists {
                var newIds []string
                for _, id := range ids {
                    if id != document.ID {
                        newIds = append(newIds, id)
                    }
                }
                if len(newIds) == 0 {
                    delete(index, value)
                } else {
                    ds.indexes[field][value] = newIds
                }
            }
        }
    }
}

func (ds *DocumentStore) GetStats() {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    fmt.Printf("Document Store Statistics:\n")
    fmt.Printf("  Total Documents: %d\n", len(ds.documents))
    fmt.Printf("  Indexed Fields: %d\n", len(ds.indexes))
    
    for field, index := range ds.indexes {
        fmt.Printf("  Field '%s': %d unique values\n", field, len(index))
    }
}

func (ds *DocumentStore) PrintDocument(id string) {
    document := ds.FindByID(id)
    if document == nil {
        fmt.Printf("Document with ID %s not found\n", id)
        return
    }
    
    jsonData, _ := json.MarshalIndent(document, "", "  ")
    fmt.Printf("Document %s:\n%s\n", id, string(jsonData))
}

func main() {
    ds := NewDocumentStore()
    
    fmt.Println("Document Store Demo:")
    
    // Insert some documents
    ds.Insert("user1", map[string]interface{}{
        "name":    "Alice",
        "age":     25,
        "city":    "New York",
        "active":  true,
    })
    
    ds.Insert("user2", map[string]interface{}{
        "name":    "Bob",
        "age":     30,
        "city":    "San Francisco",
        "active":  true,
    })
    
    ds.Insert("user3", map[string]interface{}{
        "name":    "Charlie",
        "age":     35,
        "city":    "New York",
        "active":  false,
    })
    
    // Find documents by field
    fmt.Println("\nFinding users by city 'New York':")
    nyUsers := ds.FindByField("city", "New York")
    for _, user := range nyUsers {
        fmt.Printf("  %s: %s\n", user.ID, user.Data["name"])
    }
    
    // Find documents by query
    fmt.Println("\nFinding active users:")
    activeUsers := ds.FindByQuery([]Query{
        {Field: "active", Operator: "=", Value: true},
    })
    for _, user := range activeUsers {
        fmt.Printf("  %s: %s\n", user.ID, user.Data["name"])
    }
    
    // Update a document
    ds.Update("user1", map[string]interface{}{
        "name":    "Alice Smith",
        "age":     26,
        "city":    "New York",
        "active":  true,
    })
    
    // Print document
    ds.PrintDocument("user1")
    
    // Get statistics
    ds.GetStats()
}
```

## Key-Value Stores

### Theory

Key-value stores are the simplest NoSQL databases, storing data as key-value pairs. They provide fast access to data but limited query capabilities, making them ideal for caching and session storage.

### Key-Value Store Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type KeyValueStore struct {
    data      map[string]interface{}
    ttl       map[string]time.Time
    mutex     sync.RWMutex
}

func NewKeyValueStore() *KeyValueStore {
    return &KeyValueStore{
        data: make(map[string]interface{}),
        ttl:  make(map[string]time.Time),
    }
}

func (kvs *KeyValueStore) Set(key string, value interface{}) {
    kvs.mutex.Lock()
    defer kvs.mutex.Unlock()
    
    kvs.data[key] = value
    delete(kvs.ttl, key) // Remove TTL if set
    
    fmt.Printf("Set key '%s' to value '%v'\n", key, value)
}

func (kvs *KeyValueStore) SetWithTTL(key string, value interface{}, ttl time.Duration) {
    kvs.mutex.Lock()
    defer kvs.mutex.Unlock()
    
    kvs.data[key] = value
    kvs.ttl[key] = time.Now().Add(ttl)
    
    fmt.Printf("Set key '%s' to value '%v' with TTL %v\n", key, value, ttl)
}

func (kvs *KeyValueStore) Get(key string) (interface{}, bool) {
    kvs.mutex.RLock()
    defer kvs.mutex.RUnlock()
    
    // Check if key has expired
    if expiry, exists := kvs.ttl[key]; exists {
        if time.Now().After(expiry) {
            return nil, false
        }
    }
    
    value, exists := kvs.data[key]
    return value, exists
}

func (kvs *KeyValueStore) Delete(key string) bool {
    kvs.mutex.Lock()
    defer kvs.mutex.Unlock()
    
    if _, exists := kvs.data[key]; exists {
        delete(kvs.data, key)
        delete(kvs.ttl, key)
        fmt.Printf("Deleted key '%s'\n", key)
        return true
    }
    
    fmt.Printf("Key '%s' not found\n", key)
    return false
}

func (kvs *KeyValueStore) Exists(key string) bool {
    kvs.mutex.RLock()
    defer kvs.mutex.RUnlock()
    
    // Check if key has expired
    if expiry, exists := kvs.ttl[key]; exists {
        if time.Now().After(expiry) {
            return false
        }
    }
    
    _, exists := kvs.data[key]
    return exists
}

func (kvs *KeyValueStore) Keys() []string {
    kvs.mutex.RLock()
    defer kvs.mutex.RUnlock()
    
    var keys []string
    now := time.Now()
    
    for key := range kvs.data {
        // Check if key has expired
        if expiry, exists := kvs.ttl[key]; exists {
            if now.After(expiry) {
                continue
            }
        }
        keys = append(keys, key)
    }
    
    return keys
}

func (kvs *KeyValueStore) Clear() {
    kvs.mutex.Lock()
    defer kvs.mutex.Unlock()
    
    kvs.data = make(map[string]interface{})
    kvs.ttl = make(map[string]time.Time)
    
    fmt.Println("Cleared all keys")
}

func (kvs *KeyValueStore) GetStats() {
    kvs.mutex.RLock()
    defer kvs.mutex.RUnlock()
    
    totalKeys := len(kvs.data)
    expiredKeys := 0
    now := time.Now()
    
    for _, expiry := range kvs.ttl {
        if now.After(expiry) {
            expiredKeys++
        }
    }
    
    fmt.Printf("Key-Value Store Statistics:\n")
    fmt.Printf("  Total Keys: %d\n", totalKeys)
    fmt.Printf("  Expired Keys: %d\n", expiredKeys)
    fmt.Printf("  Active Keys: %d\n", totalKeys-expiredKeys)
}

func (kvs *KeyValueStore) CleanupExpired() {
    kvs.mutex.Lock()
    defer kvs.mutex.Unlock()
    
    now := time.Now()
    var expiredKeys []string
    
    for key, expiry := range kvs.ttl {
        if now.After(expiry) {
            expiredKeys = append(expiredKeys, key)
        }
    }
    
    for _, key := range expiredKeys {
        delete(kvs.data, key)
        delete(kvs.ttl, key)
    }
    
    if len(expiredKeys) > 0 {
        fmt.Printf("Cleaned up %d expired keys\n", len(expiredKeys))
    }
}

func main() {
    kvs := NewKeyValueStore()
    
    fmt.Println("Key-Value Store Demo:")
    
    // Set some values
    kvs.Set("name", "Alice")
    kvs.Set("age", 25)
    kvs.Set("city", "New York")
    
    // Set with TTL
    kvs.SetWithTTL("session", "abc123", 5*time.Second)
    
    // Get values
    if value, exists := kvs.Get("name"); exists {
        fmt.Printf("Name: %v\n", value)
    }
    
    if value, exists := kvs.Get("age"); exists {
        fmt.Printf("Age: %v\n", value)
    }
    
    // Check if key exists
    if kvs.Exists("city") {
        fmt.Println("City key exists")
    }
    
    // List all keys
    fmt.Printf("All keys: %v\n", kvs.Keys())
    
    // Wait for TTL to expire
    fmt.Println("Waiting for session to expire...")
    time.Sleep(6 * time.Second)
    
    // Check expired key
    if value, exists := kvs.Get("session"); exists {
        fmt.Printf("Session: %v\n", value)
    } else {
        fmt.Println("Session expired")
    }
    
    // Cleanup expired keys
    kvs.CleanupExpired()
    
    // Get statistics
    kvs.GetStats()
}
```

## Column-Family Stores

### Theory

Column-family stores organize data into column families (similar to tables) and columns. They are optimized for write-heavy workloads and provide good performance for analytical queries.

### Column-Family Store Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "sync"
    "time"
)

type Column struct {
    Name  string
    Value interface{}
    Timestamp time.Time
}

type Row struct {
    Key      string
    Columns  map[string]*Column
    Timestamp time.Time
}

type ColumnFamily struct {
    Name string
    Rows map[string]*Row
    mutex sync.RWMutex
}

type ColumnFamilyStore struct {
    families map[string]*ColumnFamily
    mutex    sync.RWMutex
}

func NewColumnFamilyStore() *ColumnFamilyStore {
    return &ColumnFamilyStore{
        families: make(map[string]*ColumnFamily),
    }
}

func (cfs *ColumnFamilyStore) CreateColumnFamily(name string) *ColumnFamily {
    cfs.mutex.Lock()
    defer cfs.mutex.Unlock()
    
    family := &ColumnFamily{
        Name: name,
        Rows: make(map[string]*Row),
    }
    
    cfs.families[name] = family
    fmt.Printf("Created column family: %s\n", name)
    return family
}

func (cfs *ColumnFamilyStore) Insert(familyName, key string, columns map[string]interface{}) bool {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        fmt.Printf("Column family %s not found\n", familyName)
        return false
    }
    
    family.mutex.Lock()
    defer family.mutex.Unlock()
    
    // Get or create row
    row, exists := family.Rows[key]
    if !exists {
        row = &Row{
            Key:      key,
            Columns:  make(map[string]*Column),
            Timestamp: time.Now(),
        }
        family.Rows[key] = row
    }
    
    // Update columns
    for name, value := range columns {
        row.Columns[name] = &Column{
            Name:      name,
            Value:     value,
            Timestamp: time.Now(),
        }
    }
    
    fmt.Printf("Inserted data into %s.%s\n", familyName, key)
    return true
}

func (cfs *ColumnFamilyStore) Get(familyName, key string) *Row {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        return nil
    }
    
    family.mutex.RLock()
    defer family.mutex.RUnlock()
    
    if row, exists := family.Rows[key]; exists {
        return row
    }
    
    return nil
}

func (cfs *ColumnFamilyStore) GetColumn(familyName, key, columnName string) *Column {
    row := cfs.Get(familyName, key)
    if row == nil {
        return nil
    }
    
    if column, exists := row.Columns[columnName]; exists {
        return column
    }
    
    return nil
}

func (cfs *ColumnFamilyStore) Update(familyName, key string, columns map[string]interface{}) bool {
    return cfs.Insert(familyName, key, columns)
}

func (cfs *ColumnFamilyStore) Delete(familyName, key string) bool {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        return false
    }
    
    family.mutex.Lock()
    defer family.mutex.Unlock()
    
    if _, exists := family.Rows[key]; exists {
        delete(family.Rows, key)
        fmt.Printf("Deleted row %s from %s\n", key, familyName)
        return true
    }
    
    return false
}

func (cfs *ColumnFamilyStore) DeleteColumn(familyName, key, columnName string) bool {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        return false
    }
    
    family.mutex.Lock()
    defer family.mutex.Unlock()
    
    if row, exists := family.Rows[key]; exists {
        if _, exists := row.Columns[columnName]; exists {
            delete(row.Columns, columnName)
            fmt.Printf("Deleted column %s from %s.%s\n", columnName, familyName, key)
            return true
        }
    }
    
    return false
}

func (cfs *ColumnFamilyStore) Scan(familyName string, startKey, endKey string) []*Row {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        return nil
    }
    
    family.mutex.RLock()
    defer family.mutex.RUnlock()
    
    var results []*Row
    var keys []string
    
    // Collect all keys
    for key := range family.Rows {
        keys = append(keys, key)
    }
    
    // Sort keys
    sort.Strings(keys)
    
    // Find keys in range
    for _, key := range keys {
        if key >= startKey && key <= endKey {
            results = append(results, family.Rows[key])
        }
    }
    
    return results
}

func (cfs *ColumnFamilyStore) GetFamilyStats(familyName string) {
    cfs.mutex.RLock()
    family, exists := cfs.families[familyName]
    cfs.mutex.RUnlock()
    
    if !exists {
        fmt.Printf("Column family %s not found\n", familyName)
        return
    }
    
    family.mutex.RLock()
    defer family.mutex.RUnlock()
    
    fmt.Printf("Column Family %s Statistics:\n", familyName)
    fmt.Printf("  Rows: %d\n", len(family.Rows))
    
    // Count columns
    columnCount := 0
    for _, row := range family.Rows {
        columnCount += len(row.Columns)
    }
    fmt.Printf("  Total Columns: %d\n", columnCount)
}

func (cfs *ColumnFamilyStore) PrintRow(familyName, key string) {
    row := cfs.Get(familyName, key)
    if row == nil {
        fmt.Printf("Row %s not found in family %s\n", key, familyName)
        return
    }
    
    fmt.Printf("Row %s in family %s:\n", key, familyName)
    for name, column := range row.Columns {
        fmt.Printf("  %s: %v (timestamp: %s)\n", name, column.Value, column.Timestamp.Format(time.RFC3339))
    }
}

func main() {
    cfs := NewColumnFamilyStore()
    
    fmt.Println("Column-Family Store Demo:")
    
    // Create column families
    cfs.CreateColumnFamily("Users")
    cfs.CreateColumnFamily("Posts")
    
    // Insert data
    cfs.Insert("Users", "user1", map[string]interface{}{
        "name":    "Alice",
        "email":   "alice@example.com",
        "age":     25,
    })
    
    cfs.Insert("Users", "user2", map[string]interface{}{
        "name":    "Bob",
        "email":   "bob@example.com",
        "age":     30,
    })
    
    cfs.Insert("Posts", "post1", map[string]interface{}{
        "title":   "Hello World",
        "content": "This is my first post",
        "author":  "user1",
    })
    
    // Get data
    user := cfs.Get("Users", "user1")
    if user != nil {
        fmt.Printf("User: %s\n", user.Columns["name"].Value)
    }
    
    // Get specific column
    email := cfs.GetColumn("Users", "user1", "email")
    if email != nil {
        fmt.Printf("Email: %s\n", email.Value)
    }
    
    // Update data
    cfs.Update("Users", "user1", map[string]interface{}{
        "age": 26,
        "city": "New York",
    })
    
    // Print row
    cfs.PrintRow("Users", "user1")
    
    // Scan rows
    fmt.Println("\nScanning users:")
    users := cfs.Scan("Users", "user1", "user2")
    for _, user := range users {
        fmt.Printf("  %s: %s\n", user.Key, user.Columns["name"].Value)
    }
    
    // Get statistics
    cfs.GetFamilyStats("Users")
    cfs.GetFamilyStats("Posts")
}
```

## Follow-up Questions

### 1. Document Stores
**Q: When would you choose a document store over a relational database?**
A: Choose document stores when you have semi-structured data, need schema flexibility, want to store complex nested objects, or require horizontal scaling. They're ideal for content management, user profiles, and real-time applications.

### 2. Key-Value Stores
**Q: What are the main use cases for key-value stores?**
A: Key-value stores are ideal for caching, session storage, configuration management, and simple data storage where you only need to access data by key. They provide the fastest read/write performance for simple operations.

### 3. Column-Family Stores
**Q: How do column-family stores differ from relational databases?**
A: Column-family stores store data in columns rather than rows, which improves write performance and compression. They're optimized for analytical workloads and can handle sparse data efficiently, unlike relational databases.

## Complexity Analysis

| Operation | Document Store | Key-Value Store | Column-Family Store |
|-----------|----------------|-----------------|---------------------|
| Insert | O(1) | O(1) | O(1) |
| Read | O(1) | O(1) | O(1) |
| Update | O(1) | O(1) | O(1) |
| Delete | O(1) | O(1) | O(1) |
| Query | O(n) | O(n) | O(n) |

## Applications

1. **Document Stores**: Content management, user profiles, real-time applications
2. **Key-Value Stores**: Caching, session storage, configuration management
3. **Column-Family Stores**: Analytics, time-series data, big data processing
4. **Graph Databases**: Social networks, recommendation systems, fraud detection

---

**Next**: [Web Development](../../../README.md) | **Previous**: [Database Systems](README.md) | **Up**: [Phase 1](README.md)
