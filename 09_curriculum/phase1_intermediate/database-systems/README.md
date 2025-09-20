# Database Systems

## Table of Contents

1. [Overview](#overview)
2. [Database Design](#database-design)
3. [Query Optimization](#query-optimization)
4. [Transaction Management](#transaction-management)
5. [Indexing Strategies](#indexing-strategies)
6. [Concurrency Control](#concurrency-control)
7. [NoSQL Databases](#nosql-databases)
8. [Database Performance](#database-performance)
9. [Implementations](#implementations)
10. [Follow-up Questions](#follow-up-questions)
11. [Sources](#sources)
12. [Projects](#projects)

## Overview

### Learning Objectives

- Master database design and normalization
- Understand query optimization techniques
- Learn transaction management and ACID properties
- Apply indexing strategies for performance
- Master concurrency control mechanisms
- Work with both SQL and NoSQL databases

### What is Database Systems?

Database Systems covers the design, implementation, and optimization of database management systems, including relational databases, NoSQL databases, and advanced concepts like transactions, indexing, and concurrency control.

## Database Design

### 1. Normalization

#### First Normal Form (1NF)
```sql
-- Before 1NF (denormalized)
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(100),
    courses VARCHAR(500) -- Comma-separated values
);

-- After 1NF
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(100)
);

CREATE TABLE StudentCourses (
    student_id INT,
    course_id INT,
    course_name VARCHAR(100),
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);
```

#### Second Normal Form (2NF)
```sql
-- Before 2NF (partial dependency)
CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    customer_name VARCHAR(100),
    product_id INT,
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2)
);

-- After 2NF
CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100)
);

CREATE TABLE Products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

CREATE TABLE OrderItems (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);
```

#### Third Normal Form (3NF)
```sql
-- Before 3NF (transitive dependency)
CREATE TABLE Employees (
    employee_id INT PRIMARY KEY,
    employee_name VARCHAR(100),
    department_id INT,
    department_name VARCHAR(100),
    manager_id INT
);

-- After 3NF
CREATE TABLE Departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(100)
);

CREATE TABLE Employees (
    employee_id INT PRIMARY KEY,
    employee_name VARCHAR(100),
    department_id INT,
    manager_id INT,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id),
    FOREIGN KEY (manager_id) REFERENCES Employees(employee_id)
);
```

### 2. Entity-Relationship Modeling

#### ER Diagram Implementation
```go
package main

import "fmt"

type Entity struct {
    Name       string
    Attributes []Attribute
    PrimaryKey string
}

type Attribute struct {
    Name     string
    Type     string
    Required bool
    Unique   bool
}

type Relationship struct {
    Name         string
    Entity1      string
    Entity2      string
    Cardinality1 string // One, Many
    Cardinality2 string // One, Many
}

type ERModel struct {
    Entities     []Entity
    Relationships []Relationship
}

func NewERModel() *ERModel {
    return &ERModel{
        Entities:     make([]Entity, 0),
        Relationships: make([]Relationship, 0),
    }
}

func (er *ERModel) AddEntity(name string, attributes []Attribute, primaryKey string) {
    entity := Entity{
        Name:       name,
        Attributes: attributes,
        PrimaryKey: primaryKey,
    }
    er.Entities = append(er.Entities, entity)
}

func (er *ERModel) AddRelationship(name, entity1, entity2, card1, card2 string) {
    relationship := Relationship{
        Name:         name,
        Entity1:      entity1,
        Entity2:      entity2,
        Cardinality1: card1,
        Cardinality2: card2,
    }
    er.Relationships = append(er.Relationships, relationship)
}

func (er *ERModel) GenerateSQL() []string {
    var sqlStatements []string
    
    for _, entity := range er.Entities {
        sql := fmt.Sprintf("CREATE TABLE %s (", entity.Name)
        
        for i, attr := range entity.Attributes {
            if i > 0 {
                sql += ", "
            }
            sql += fmt.Sprintf("%s %s", attr.Name, attr.Type)
            
            if attr.Required {
                sql += " NOT NULL"
            }
            if attr.Unique {
                sql += " UNIQUE"
            }
        }
        
        sql += fmt.Sprintf(", PRIMARY KEY (%s)", entity.PrimaryKey)
        sql += ");"
        
        sqlStatements = append(sqlStatements, sql)
    }
    
    return sqlStatements
}

func main() {
    er := NewERModel()
    
    // Define entities
    er.AddEntity("Student", []Attribute{
        {Name: "student_id", Type: "INT", Required: true, Unique: true},
        {Name: "name", Type: "VARCHAR(100)", Required: true, Unique: false},
        {Name: "email", Type: "VARCHAR(100)", Required: true, Unique: true},
        {Name: "enrollment_date", Type: "DATE", Required: true, Unique: false},
    }, "student_id")
    
    er.AddEntity("Course", []Attribute{
        {Name: "course_id", Type: "INT", Required: true, Unique: true},
        {Name: "title", Type: "VARCHAR(100)", Required: true, Unique: false},
        {Name: "credits", Type: "INT", Required: true, Unique: false},
    }, "course_id")
    
    er.AddEntity("Enrollment", []Attribute{
        {Name: "student_id", Type: "INT", Required: true, Unique: false},
        {Name: "course_id", Type: "INT", Required: true, Unique: false},
        {Name: "grade", Type: "CHAR(2)", Required: false, Unique: false},
        {Name: "enrollment_date", Type: "DATE", Required: true, Unique: false},
    }, "student_id, course_id")
    
    // Define relationships
    er.AddRelationship("enrolls", "Student", "Enrollment", "One", "Many")
    er.AddRelationship("enrolled_in", "Course", "Enrollment", "One", "Many")
    
    // Generate SQL
    sqlStatements := er.GenerateSQL()
    for _, sql := range sqlStatements {
        fmt.Println(sql)
    }
}
```

## Query Optimization

### 1. Query Execution Plans

#### Cost-Based Optimization
```go
package main

import (
    "fmt"
    "math"
)

type Table struct {
    Name      string
    RowCount  int
    PageCount int
    Indexes   []Index
}

type Index struct {
    Name        string
    Column      string
    Selectivity float64
    Height      int
}

type Query struct {
    Tables     []string
    Conditions []Condition
    Joins      []Join
    Projections []string
}

type Condition struct {
    Column   string
    Operator string
    Value    interface{}
    Selectivity float64
}

type Join struct {
    Table1    string
    Table2    string
    Condition string
    Type      string // INNER, LEFT, RIGHT, FULL
}

type QueryOptimizer struct {
    tables map[string]*Table
    stats  map[string]TableStats
}

type TableStats struct {
    RowCount    int
    PageCount   int
    ColumnStats map[string]ColumnStats
}

type ColumnStats struct {
    DistinctValues int
    MinValue       interface{}
    MaxValue       interface{}
    NullCount      int
}

func NewQueryOptimizer() *QueryOptimizer {
    return &QueryOptimizer{
        tables: make(map[string]*Table),
        stats:  make(map[string]TableStats),
    }
}

func (qo *QueryOptimizer) AddTable(table *Table) {
    qo.tables[table.Name] = table
}

func (qo *QueryOptimizer) EstimateCost(query Query) float64 {
    totalCost := 0.0
    
    // Estimate cost for each table access
    for _, tableName := range query.Tables {
        table := qo.tables[tableName]
        cost := qo.estimateTableCost(table, query.Conditions)
        totalCost += cost
    }
    
    // Estimate cost for joins
    for _, join := range query.Joins {
        cost := qo.estimateJoinCost(join)
        totalCost += cost
    }
    
    return totalCost
}

func (qo *QueryOptimizer) estimateTableCost(table *Table, conditions []Condition) float64 {
    // Base cost: sequential scan
    baseCost := float64(table.PageCount)
    
    // Check if we can use an index
    for _, condition := range conditions {
        for _, index := range table.Indexes {
            if index.Column == condition.Column {
                // Index scan cost
                indexCost := float64(index.Height) + float64(table.RowCount) * condition.Selectivity
                if indexCost < baseCost {
                    baseCost = indexCost
                }
            }
        }
    }
    
    return baseCost
}

func (qo *QueryOptimizer) estimateJoinCost(join Join) float64 {
    table1 := qo.tables[join.Table1]
    table2 := qo.tables[join.Table2]
    
    // Nested loop join cost
    nestedLoopCost := float64(table1.RowCount * table2.RowCount)
    
    // Hash join cost
    hashJoinCost := float64(table1.RowCount + table2.RowCount)
    
    // Choose the cheaper option
    if hashJoinCost < nestedLoopCost {
        return hashJoinCost
    }
    return nestedLoopCost
}

func (qo *QueryOptimizer) OptimizeQuery(query Query) Query {
    // Simple optimization: reorder joins based on cost
    optimized := query
    
    // Sort joins by estimated cost
    for i := 0; i < len(optimized.Joins)-1; i++ {
        for j := i + 1; j < len(optimized.Joins); j++ {
            costI := qo.estimateJoinCost(optimized.Joins[i])
            costJ := qo.estimateJoinCost(optimized.Joins[j])
            
            if costJ < costI {
                optimized.Joins[i], optimized.Joins[j] = optimized.Joins[j], optimized.Joins[i]
            }
        }
    }
    
    return optimized
}

func main() {
    optimizer := NewQueryOptimizer()
    
    // Add tables
    optimizer.AddTable(&Table{
        Name:      "users",
        RowCount:  10000,
        PageCount: 100,
        Indexes: []Index{
            {Name: "idx_user_id", Column: "user_id", Selectivity: 0.1, Height: 3},
            {Name: "idx_email", Column: "email", Selectivity: 0.05, Height: 2},
        },
    })
    
    optimizer.AddTable(&Table{
        Name:      "orders",
        RowCount:  50000,
        PageCount: 500,
        Indexes: []Index{
            {Name: "idx_order_id", Column: "order_id", Selectivity: 0.1, Height: 4},
            {Name: "idx_user_id", Column: "user_id", Selectivity: 0.2, Height: 3},
        },
    })
    
    // Create query
    query := Query{
        Tables: []string{"users", "orders"},
        Conditions: []Condition{
            {Column: "email", Operator: "=", Value: "user@example.com", Selectivity: 0.05},
            {Column: "order_date", Operator: ">", Value: "2023-01-01", Selectivity: 0.3},
        },
        Joins: []Join{
            {Table1: "users", Table2: "orders", Condition: "users.user_id = orders.user_id", Type: "INNER"},
        },
        Projections: []string{"user_id", "name", "order_id", "total"},
    }
    
    // Estimate cost
    cost := optimizer.EstimateCost(query)
    fmt.Printf("Estimated query cost: %.2f\n", cost)
    
    // Optimize query
    optimized := optimizer.OptimizeQuery(query)
    fmt.Printf("Optimized query: %+v\n", optimized)
}
```

### 2. Index Selection

#### B-Tree Index Implementation
```go
package main

import (
    "fmt"
    "sort"
)

type BTreeNode struct {
    keys     []int
    values   []interface{}
    children []*BTreeNode
    isLeaf   bool
    parent   *BTreeNode
}

type BTree struct {
    root   *BTreeNode
    degree int // Minimum degree
}

func NewBTree(degree int) *BTree {
    return &BTree{
        root:   nil,
        degree: degree,
    }
}

func (bt *BTree) Insert(key int, value interface{}) {
    if bt.root == nil {
        bt.root = &BTreeNode{
            keys:   []int{key},
            values: []interface{}{value},
            isLeaf: true,
        }
        return
    }
    
    if len(bt.root.keys) == 2*bt.degree-1 {
        // Root is full, need to split
        newRoot := &BTreeNode{
            children: []*BTreeNode{bt.root},
            isLeaf:   false,
        }
        bt.splitChild(newRoot, 0)
        bt.root = newRoot
    }
    
    bt.insertNonFull(bt.root, key, value)
}

func (bt *BTree) insertNonFull(node *BTreeNode, key int, value interface{}) {
    if node.isLeaf {
        // Insert into leaf node
        i := bt.findInsertPosition(node.keys, key)
        node.keys = append(node.keys[:i], append([]int{key}, node.keys[i:]...)...)
        node.values = append(node.values[:i], append([]interface{}{value}, node.values[i:]...)...)
    } else {
        // Find child to insert into
        i := bt.findInsertPosition(node.keys, key)
        if len(node.children[i].keys) == 2*bt.degree-1 {
            bt.splitChild(node, i)
            if key > node.keys[i] {
                i++
            }
        }
        bt.insertNonFull(node.children[i], key, value)
    }
}

func (bt *BTree) splitChild(parent *BTreeNode, index int) {
    child := parent.children[index]
    newChild := &BTreeNode{
        isLeaf: child.isLeaf,
        parent: parent,
    }
    
    // Move half of keys and values to new child
    mid := bt.degree - 1
    newChild.keys = make([]int, bt.degree-1)
    newChild.values = make([]interface{}, bt.degree-1)
    copy(newChild.keys, child.keys[bt.degree:])
    copy(newChild.values, child.values[bt.degree:])
    
    // Update child's keys and values
    child.keys = child.keys[:bt.degree-1]
    child.values = child.values[:bt.degree-1]
    
    // Move children if not leaf
    if !child.isLeaf {
        newChild.children = make([]*BTreeNode, bt.degree)
        copy(newChild.children, child.children[bt.degree:])
        child.children = child.children[:bt.degree]
        
        // Update parent references
        for _, c := range newChild.children {
            c.parent = newChild
        }
    }
    
    // Insert middle key into parent
    parent.keys = append(parent.keys[:index], append([]int{child.keys[bt.degree-1]}, parent.keys[index:]...)...)
    parent.values = append(parent.values[:index], append([]interface{}{child.values[bt.degree-1]}, parent.values[index:]...)...)
    
    // Add new child to parent
    parent.children = append(parent.children[:index+1], append([]*BTreeNode{newChild}, parent.children[index+1:]...)...)
    
    // Remove middle key from child
    child.keys = child.keys[:bt.degree-1]
    child.values = child.values[:bt.degree-1]
}

func (bt *BTree) findInsertPosition(keys []int, key int) int {
    i := 0
    for i < len(keys) && key > keys[i] {
        i++
    }
    return i
}

func (bt *BTree) Search(key int) (interface{}, bool) {
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) (interface{}, bool) {
    if node == nil {
        return nil, false
    }
    
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }
    
    if i < len(node.keys) && key == node.keys[i] {
        return node.values[i], true
    }
    
    if node.isLeaf {
        return nil, false
    }
    
    return bt.searchNode(node.children[i], key)
}

func (bt *BTree) Print() {
    bt.printNode(bt.root, 0)
}

func (bt *BTree) printNode(node *BTreeNode, level int) {
    if node == nil {
        return
    }
    
    for i := 0; i < level; i++ {
        fmt.Print("  ")
    }
    
    fmt.Printf("Keys: %v\n", node.keys)
    
    if !node.isLeaf {
        for _, child := range node.children {
            bt.printNode(child, level+1)
        }
    }
}

func main() {
    bt := NewBTree(3)
    
    // Insert some keys
    keys := []int{10, 20, 5, 6, 12, 30, 7, 17}
    for _, key := range keys {
        bt.Insert(key, fmt.Sprintf("value_%d", key))
    }
    
    fmt.Println("B-Tree structure:")
    bt.Print()
    
    // Search for a key
    value, found := bt.Search(12)
    if found {
        fmt.Printf("Found key 12: %v\n", value)
    } else {
        fmt.Println("Key 12 not found")
    }
}
```

## Transaction Management

### 1. ACID Properties Implementation

#### Transaction Manager
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type TransactionState int

const (
    Active TransactionState = iota
    Committed
    Aborted
)

type Transaction struct {
    ID        int
    State     TransactionState
    StartTime time.Time
    Locks     map[string]bool
    Log       []LogEntry
}

type LogEntry struct {
    TransactionID int
    Operation     string
    Key           string
    OldValue      interface{}
    NewValue      interface{}
    Timestamp     time.Time
}

type LockType int

const (
    SharedLock LockType = iota
    ExclusiveLock
)

type Lock struct {
    Key        string
    Type       LockType
    TransactionID int
    Granted    bool
    WaitQueue  []int
}

type TransactionManager struct {
    transactions map[int]*Transaction
    locks        map[string]*Lock
    nextTxnID    int
    mutex        sync.RWMutex
}

func NewTransactionManager() *TransactionManager {
    return &TransactionManager{
        transactions: make(map[int]*Transaction),
        locks:        make(map[string]*Lock),
        nextTxnID:    1,
    }
}

func (tm *TransactionManager) BeginTransaction() int {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    txnID := tm.nextTxnID
    tm.nextTxnID++
    
    transaction := &Transaction{
        ID:        txnID,
        State:     Active,
        StartTime: time.Now(),
        Locks:     make(map[string]bool),
        Log:       make([]LogEntry, 0),
    }
    
    tm.transactions[txnID] = transaction
    return txnID
}

func (tm *TransactionManager) AcquireLock(txnID int, key string, lockType LockType) bool {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    lock, exists := tm.locks[key]
    if !exists {
        // No existing lock, grant immediately
        tm.locks[key] = &Lock{
            Key:          key,
            Type:         lockType,
            TransactionID: txnID,
            Granted:      true,
            WaitQueue:    make([]int, 0),
        }
        transaction.Locks[key] = true
        return true
    }
    
    if lock.TransactionID == txnID {
        // Already holds a lock, upgrade if necessary
        if lockType == ExclusiveLock && lock.Type == SharedLock {
            lock.Type = ExclusiveLock
        }
        return true
    }
    
    // Check compatibility
    if tm.isCompatible(lock.Type, lockType) {
        // Grant shared lock
        if lockType == SharedLock {
            lock.WaitQueue = append(lock.WaitQueue, txnID)
            transaction.Locks[key] = true
            return true
        }
    }
    
    // Add to wait queue
    lock.WaitQueue = append(lock.WaitQueue, txnID)
    return false
}

func (tm *TransactionManager) isCompatible(held, requested LockType) bool {
    if held == SharedLock && requested == SharedLock {
        return true
    }
    return false
}

func (tm *TransactionManager) ReleaseLock(txnID int, key string) {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists {
        return
    }
    
    delete(transaction.Locks, key)
    
    lock, exists := tm.locks[key]
    if !exists {
        return
    }
    
    if lock.TransactionID == txnID {
        // Remove from wait queue
        for i, waitingTxn := range lock.WaitQueue {
            if waitingTxn == txnID {
                lock.WaitQueue = append(lock.WaitQueue[:i], lock.WaitQueue[i+1:]...)
                break
            }
        }
        
        // Grant lock to next waiting transaction
        if len(lock.WaitQueue) > 0 {
            nextTxn := lock.WaitQueue[0]
            lock.WaitQueue = lock.WaitQueue[1:]
            lock.TransactionID = nextTxn
            lock.Granted = true
            
            if waitingTxn, exists := tm.transactions[nextTxn]; exists {
                waitingTxn.Locks[key] = true
            }
        } else {
            delete(tm.locks, key)
        }
    }
}

func (tm *TransactionManager) Commit(txnID int) bool {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    // Write log entry
    logEntry := LogEntry{
        TransactionID: txnID,
        Operation:     "COMMIT",
        Timestamp:     time.Now(),
    }
    transaction.Log = append(transaction.Log, logEntry)
    
    // Release all locks
    for key := range transaction.Locks {
        tm.ReleaseLock(txnID, key)
    }
    
    transaction.State = Committed
    return true
}

func (tm *TransactionManager) Abort(txnID int) bool {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    // Write log entry
    logEntry := LogEntry{
        TransactionID: txnID,
        Operation:     "ABORT",
        Timestamp:     time.Now(),
    }
    transaction.Log = append(transaction.Log, logEntry)
    
    // Release all locks
    for key := range transaction.Locks {
        tm.ReleaseLock(txnID, key)
    }
    
    transaction.State = Aborted
    return true
}

func main() {
    tm := NewTransactionManager()
    
    // Begin transaction
    txnID := tm.BeginTransaction()
    fmt.Printf("Started transaction %d\n", txnID)
    
    // Acquire locks
    if tm.AcquireLock(txnID, "key1", SharedLock) {
        fmt.Println("Acquired shared lock on key1")
    }
    
    if tm.AcquireLock(txnID, "key2", ExclusiveLock) {
        fmt.Println("Acquired exclusive lock on key2")
    }
    
    // Commit transaction
    if tm.Commit(txnID) {
        fmt.Println("Transaction committed successfully")
    }
}
```

### 2. Two-Phase Locking

#### 2PL Implementation
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type TwoPhaseLocking struct {
    transactions map[int]*Transaction
    locks        map[string]*Lock
    nextTxnID    int
    mutex        sync.RWMutex
}

type Transaction struct {
    ID        int
    State     TransactionState
    StartTime time.Time
    Locks     map[string]LockType
    Phase     string // "Growing" or "Shrinking"
}

type Lock struct {
    Key          string
    Type         LockType
    TransactionID int
    Granted      bool
    WaitQueue    []int
}

func NewTwoPhaseLocking() *TwoPhaseLocking {
    return &TwoPhaseLocking{
        transactions: make(map[int]*Transaction),
        locks:        make(map[string]*Lock),
        nextTxnID:    1,
    }
}

func (tpl *TwoPhaseLocking) BeginTransaction() int {
    tpl.mutex.Lock()
    defer tpl.mutex.Unlock()
    
    txnID := tpl.nextTxnID
    tpl.nextTxnID++
    
    transaction := &Transaction{
        ID:        txnID,
        State:     Active,
        StartTime: time.Now(),
        Locks:     make(map[string]LockType),
        Phase:     "Growing",
    }
    
    tpl.transactions[txnID] = transaction
    return txnID
}

func (tpl *TwoPhaseLocking) AcquireLock(txnID int, key string, lockType LockType) bool {
    tpl.mutex.Lock()
    defer tpl.mutex.Unlock()
    
    transaction, exists := tpl.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    // Check if already in shrinking phase
    if transaction.Phase == "Shrinking" {
        return false
    }
    
    lock, exists := tpl.locks[key]
    if !exists {
        // No existing lock, grant immediately
        tpl.locks[key] = &Lock{
            Key:          key,
            Type:         lockType,
            TransactionID: txnID,
            Granted:      true,
            WaitQueue:    make([]int, 0),
        }
        transaction.Locks[key] = lockType
        return true
    }
    
    if lock.TransactionID == txnID {
        // Already holds a lock, upgrade if necessary
        if lockType == ExclusiveLock && lock.Type == SharedLock {
            lock.Type = ExclusiveLock
            transaction.Locks[key] = lockType
        }
        return true
    }
    
    // Check compatibility
    if tpl.isCompatible(lock.Type, lockType) {
        // Grant shared lock
        if lockType == SharedLock {
            lock.WaitQueue = append(lock.WaitQueue, txnID)
            transaction.Locks[key] = lockType
            return true
        }
    }
    
    // Add to wait queue
    lock.WaitQueue = append(lock.WaitQueue, txnID)
    return false
}

func (tpl *TwoPhaseLocking) isCompatible(held, requested LockType) bool {
    if held == SharedLock && requested == SharedLock {
        return true
    }
    return false
}

func (tpl *TwoPhaseLocking) ReleaseLock(txnID int, key string) {
    tpl.mutex.Lock()
    defer tpl.mutex.Unlock()
    
    transaction, exists := tpl.transactions[txnID]
    if !exists {
        return
    }
    
    // Move to shrinking phase
    if transaction.Phase == "Growing" {
        transaction.Phase = "Shrinking"
    }
    
    delete(transaction.Locks, key)
    
    lock, exists := tpl.locks[key]
    if !exists {
        return
    }
    
    if lock.TransactionID == txnID {
        // Remove from wait queue
        for i, waitingTxn := range lock.WaitQueue {
            if waitingTxn == txnID {
                lock.WaitQueue = append(lock.WaitQueue[:i], lock.WaitQueue[i+1:]...)
                break
            }
        }
        
        // Grant lock to next waiting transaction
        if len(lock.WaitQueue) > 0 {
            nextTxn := lock.WaitQueue[0]
            lock.WaitQueue = lock.WaitQueue[1:]
            lock.TransactionID = nextTxn
            lock.Granted = true
            
            if waitingTxn, exists := tpl.transactions[nextTxn]; exists {
                waitingTxn.Locks[key] = lock.Type
            }
        } else {
            delete(tpl.locks, key)
        }
    }
}

func (tpl *TwoPhaseLocking) Commit(txnID int) bool {
    tpl.mutex.Lock()
    defer tpl.mutex.Unlock()
    
    transaction, exists := tpl.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    // Release all locks
    for key := range transaction.Locks {
        tpl.ReleaseLock(txnID, key)
    }
    
    transaction.State = Committed
    return true
}

func (tpl *TwoPhaseLocking) Abort(txnID int) bool {
    tpl.mutex.Lock()
    defer tpl.mutex.Unlock()
    
    transaction, exists := tpl.transactions[txnID]
    if !exists || transaction.State != Active {
        return false
    }
    
    // Release all locks
    for key := range transaction.Locks {
        tpl.ReleaseLock(txnID, key)
    }
    
    transaction.State = Aborted
    return true
}

func main() {
    tpl := NewTwoPhaseLocking()
    
    // Begin transaction
    txnID := tpl.BeginTransaction()
    fmt.Printf("Started transaction %d\n", txnID)
    
    // Acquire locks in growing phase
    if tpl.AcquireLock(txnID, "key1", SharedLock) {
        fmt.Println("Acquired shared lock on key1")
    }
    
    if tpl.AcquireLock(txnID, "key2", ExclusiveLock) {
        fmt.Println("Acquired exclusive lock on key2")
    }
    
    // Release a lock (moves to shrinking phase)
    tpl.ReleaseLock(txnID, "key1")
    fmt.Println("Released lock on key1 (moved to shrinking phase)")
    
    // Try to acquire another lock (should fail in shrinking phase)
    if tpl.AcquireLock(txnID, "key3", SharedLock) {
        fmt.Println("Acquired lock on key3")
    } else {
        fmt.Println("Failed to acquire lock on key3 (in shrinking phase)")
    }
    
    // Commit transaction
    if tpl.Commit(txnID) {
        fmt.Println("Transaction committed successfully")
    }
}
```

## Follow-up Questions

### 1. Database Design
**Q: What's the difference between 2NF and 3NF?**
A: 2NF eliminates partial dependencies (non-key attributes depend on part of a composite key), while 3NF eliminates transitive dependencies (non-key attributes depend on other non-key attributes).

### 2. Query Optimization
**Q: What factors affect query performance?**
A: Index usage, join order, table size, data distribution, statistics accuracy, hardware resources, and concurrency level all significantly impact query performance.

### 3. Transaction Management
**Q: What's the difference between 2PL and strict 2PL?**
A: In 2PL, locks can be released during the shrinking phase, while strict 2PL holds all locks until transaction commit, preventing cascading aborts.

## Sources

### Books
- **Database System Concepts** by Silberschatz, Korth, Sudarshan
- **Database Management Systems** by Ramakrishnan and Gehrke
- **Transaction Processing** by Gray and Reuter

### Online Resources
- **PostgreSQL Documentation** - Advanced database concepts
- **MySQL Documentation** - Database optimization
- **MongoDB University** - NoSQL database concepts

## Projects

### 1. Database Engine
**Objective**: Build a simple database engine
**Requirements**: SQL parser, query optimizer, transaction manager
**Deliverables**: Working database with basic SQL support

### 2. Query Optimizer
**Objective**: Implement query optimization techniques
**Requirements**: Cost estimation, join reordering, index selection
**Deliverables**: Optimizer with performance benchmarks

### 3. NoSQL Database
**Objective**: Design and implement a NoSQL database
**Requirements**: Document storage, indexing, query language
**Deliverables**: Complete NoSQL database system

---

**Next**: [Web Development](../../../README.md) | **Previous**: [OS Deep Dive](../../../README.md) | **Up**: [Phase 1](README.md)

