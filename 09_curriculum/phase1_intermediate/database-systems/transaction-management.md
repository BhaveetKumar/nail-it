# Transaction Management

## Overview

This module covers transaction management concepts including ACID properties, concurrency control, locking mechanisms, and transaction isolation levels. These concepts are essential for ensuring data consistency and integrity in database systems.

## Table of Contents

1. [ACID Properties](#acid-properties)
2. [Concurrency Control](#concurrency-control)
3. [Locking Mechanisms](#locking-mechanisms)
4. [Transaction Isolation Levels](#transaction-isolation-levels)
5. [Deadlock Detection](#deadlock-detection)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## ACID Properties

### Theory

ACID properties ensure reliable database transactions:
- **Atomicity**: All operations in a transaction succeed or all fail
- **Consistency**: Database remains in a valid state before and after transaction
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed changes persist even after system failure

### Transaction Manager Implementation

#### Golang Implementation

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
    Preparing
)

type Transaction struct {
    ID        int
    State     TransactionState
    StartTime time.Time
    Locks     map[string]bool
    Changes   []Change
    mutex     sync.RWMutex
}

type Change struct {
    Table    string
    RowID    int
    Column   string
    OldValue interface{}
    NewValue interface{}
}

type TransactionManager struct {
    transactions map[int]*Transaction
    nextTxnID    int
    mutex        sync.RWMutex
}

func NewTransactionManager() *TransactionManager {
    return &TransactionManager{
        transactions: make(map[int]*Transaction),
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
        Changes:   make([]Change, 0),
    }
    
    tm.transactions[txnID] = transaction
    fmt.Printf("Started transaction %d\n", txnID)
    return txnID
}

func (tm *TransactionManager) CommitTransaction(txnID int) bool {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists {
        fmt.Printf("Transaction %d not found\n", txnID)
        return false
    }
    
    if transaction.State != Active {
        fmt.Printf("Transaction %d is not active\n", txnID)
        return false
    }
    
    // Atomicity: Apply all changes or none
    if tm.applyChanges(transaction) {
        transaction.State = Committed
        fmt.Printf("Transaction %d committed successfully\n", txnID)
        return true
    } else {
        transaction.State = Aborted
        fmt.Printf("Transaction %d aborted due to failure\n", txnID)
        return false
    }
}

func (tm *TransactionManager) AbortTransaction(txnID int) bool {
    tm.mutex.Lock()
    defer tm.mutex.Unlock()
    
    transaction, exists := tm.transactions[txnID]
    if !exists {
        fmt.Printf("Transaction %d not found\n", txnID)
        return false
    }
    
    if transaction.State != Active {
        fmt.Printf("Transaction %d is not active\n", txnID)
        return false
    }
    
    // Rollback changes
    tm.rollbackChanges(transaction)
    transaction.State = Aborted
    fmt.Printf("Transaction %d aborted\n", txnID)
    return true
}

func (tm *TransactionManager) applyChanges(transaction *Transaction) bool {
    // Simulate applying changes
    // In a real implementation, this would write to the database
    for _, change := range transaction.Changes {
        fmt.Printf("Applying change: %s.%s[%d] = %v\n", 
                   change.Table, change.Column, change.RowID, change.NewValue)
        
        // Simulate potential failure
        if change.NewValue == "FAIL" {
            fmt.Printf("Change failed for %s.%s[%d]\n", 
                       change.Table, change.Column, change.RowID)
            return false
        }
    }
    
    return true
}

func (tm *TransactionManager) rollbackChanges(transaction *Transaction) {
    // Simulate rolling back changes
    for _, change := range transaction.Changes {
        fmt.Printf("Rolling back change: %s.%s[%d] = %v\n", 
                   change.Table, change.Column, change.RowID, change.OldValue)
    }
}

func (tm *TransactionManager) AddChange(txnID int, table string, rowID int, column string, oldValue, newValue interface{}) bool {
    tm.mutex.RLock()
    transaction, exists := tm.transactions[txnID]
    tm.mutex.RUnlock()
    
    if !exists {
        fmt.Printf("Transaction %d not found\n", txnID)
        return false
    }
    
    if transaction.State != Active {
        fmt.Printf("Transaction %d is not active\n", txnID)
        return false
    }
    
    transaction.mutex.Lock()
    defer transaction.mutex.Unlock()
    
    change := Change{
        Table:    table,
        RowID:    rowID,
        Column:   column,
        OldValue: oldValue,
        NewValue: newValue,
    }
    
    transaction.Changes = append(transaction.Changes, change)
    fmt.Printf("Added change to transaction %d: %s.%s[%d] = %v\n", 
               txnID, table, column, rowID, newValue)
    
    return true
}

func (tm *TransactionManager) GetTransactionStatus(txnID int) {
    tm.mutex.RLock()
    transaction, exists := tm.transactions[txnID]
    tm.mutex.RUnlock()
    
    if !exists {
        fmt.Printf("Transaction %d not found\n", txnID)
        return
    }
    
    transaction.mutex.RLock()
    defer transaction.mutex.RUnlock()
    
    fmt.Printf("Transaction %d Status:\n", txnID)
    fmt.Printf("  State: %v\n", transaction.State)
    fmt.Printf("  Start Time: %s\n", transaction.StartTime.Format(time.RFC3339))
    fmt.Printf("  Changes: %d\n", len(transaction.Changes))
    fmt.Printf("  Locks: %d\n", len(transaction.Locks))
}

func main() {
    tm := NewTransactionManager()
    
    fmt.Println("Transaction Management Demo:")
    
    // Begin a transaction
    txnID := tm.BeginTransaction()
    
    // Add some changes
    tm.AddChange(txnID, "Student", 1, "Name", "OldName", "NewName")
    tm.AddChange(txnID, "Student", 1, "Age", 20, 21)
    tm.AddChange(txnID, "Course", 1, "Title", "OldTitle", "NewTitle")
    
    // Check transaction status
    tm.GetTransactionStatus(txnID)
    
    // Commit the transaction
    success := tm.CommitTransaction(txnID)
    if success {
        fmt.Println("Transaction committed successfully")
    } else {
        fmt.Println("Transaction failed to commit")
    }
    
    // Test transaction failure
    txnID2 := tm.BeginTransaction()
    tm.AddChange(txnID2, "Student", 2, "Name", "OldName", "FAIL")
    
    success = tm.CommitTransaction(txnID2)
    if !success {
        fmt.Println("Transaction failed as expected")
    }
}
```

## Concurrency Control

### Theory

Concurrency control ensures that multiple transactions can execute simultaneously without compromising data integrity. Common techniques include locking, timestamp ordering, and optimistic concurrency control.

### Concurrency Controller Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type LockType int

const (
    SharedLock LockType = iota
    ExclusiveLock
)

type Lock struct {
    Resource    string
    Type        LockType
    Transaction int
    Timestamp   time.Time
}

type ConcurrencyController struct {
    locks       map[string][]Lock
    waitQueue   map[string][]int
    transactions map[int]*Transaction
    mutex       sync.RWMutex
}

func NewConcurrencyController() *ConcurrencyController {
    return &ConcurrencyController{
        locks:       make(map[string][]Lock),
        waitQueue:   make(map[string][]int),
        transactions: make(map[int]*Transaction),
    }
}

func (cc *ConcurrencyController) AcquireLock(txnID int, resource string, lockType LockType) bool {
    cc.mutex.Lock()
    defer cc.mutex.Unlock()
    
    // Check if transaction exists
    if _, exists := cc.transactions[txnID]; !exists {
        fmt.Printf("Transaction %d not found\n", txnID)
        return false
    }
    
    // Check if lock is already held by this transaction
    if cc.hasLock(txnID, resource, lockType) {
        fmt.Printf("Transaction %d already holds %v lock on %s\n", txnID, lockType, resource)
        return true
    }
    
    // Check for lock conflicts
    if cc.hasLockConflict(resource, lockType) {
        fmt.Printf("Lock conflict detected for resource %s\n", resource)
        cc.addToWaitQueue(txnID, resource)
        return false
    }
    
    // Acquire the lock
    lock := Lock{
        Resource:    resource,
        Type:        lockType,
        Transaction: txnID,
        Timestamp:   time.Now(),
    }
    
    cc.locks[resource] = append(cc.locks[resource], lock)
    fmt.Printf("Transaction %d acquired %v lock on %s\n", txnID, lockType, resource)
    
    return true
}

func (cc *ConcurrencyController) ReleaseLock(txnID int, resource string) bool {
    cc.mutex.Lock()
    defer cc.mutex.Unlock()
    
    if locks, exists := cc.locks[resource]; exists {
        for i, lock := range locks {
            if lock.Transaction == txnID {
                // Remove the lock
                cc.locks[resource] = append(locks[:i], locks[i+1:]...)
                fmt.Printf("Transaction %d released lock on %s\n", txnID, resource)
                
                // Process waiting transactions
                cc.processWaitQueue(resource)
                return true
            }
        }
    }
    
    fmt.Printf("Transaction %d does not hold lock on %s\n", txnID, resource)
    return false
}

func (cc *ConcurrencyController) hasLock(txnID int, resource string, lockType LockType) bool {
    if locks, exists := cc.locks[resource]; exists {
        for _, lock := range locks {
            if lock.Transaction == txnID && lock.Type == lockType {
                return true
            }
        }
    }
    return false
}

func (cc *ConcurrencyController) hasLockConflict(resource string, lockType LockType) bool {
    if locks, exists := cc.locks[resource]; exists {
        for _, lock := range locks {
            // Exclusive lock conflicts with any other lock
            if lockType == ExclusiveLock || lock.Type == ExclusiveLock {
                return true
            }
        }
    }
    return false
}

func (cc *ConcurrencyController) addToWaitQueue(txnID int, resource string) {
    cc.waitQueue[resource] = append(cc.waitQueue[resource], txnID)
    fmt.Printf("Transaction %d added to wait queue for %s\n", txnID, resource)
}

func (cc *ConcurrencyController) processWaitQueue(resource string) {
    if queue, exists := cc.waitQueue[resource]; exists && len(queue) > 0 {
        // Process waiting transactions in FIFO order
        for len(queue) > 0 {
            txnID := queue[0]
            queue = queue[1:]
            
            // Try to acquire lock for waiting transaction
            if cc.tryAcquireLock(txnID, resource) {
                fmt.Printf("Transaction %d acquired lock on %s from wait queue\n", txnID, resource)
                break
            }
        }
        
        cc.waitQueue[resource] = queue
    }
}

func (cc *ConcurrencyController) tryAcquireLock(txnID int, resource string) bool {
    // This is a simplified version - in practice, you'd need to know the lock type
    // For now, assume shared lock
    if cc.hasLockConflict(resource, SharedLock) {
        return false
    }
    
    lock := Lock{
        Resource:    resource,
        Type:        SharedLock,
        Transaction: txnID,
        Timestamp:   time.Now(),
    }
    
    cc.locks[resource] = append(cc.locks[resource], lock)
    return true
}

func (cc *ConcurrencyController) RegisterTransaction(txnID int, transaction *Transaction) {
    cc.mutex.Lock()
    defer cc.mutex.Unlock()
    
    cc.transactions[txnID] = transaction
    fmt.Printf("Registered transaction %d\n", txnID)
}

func (cc *ConcurrencyController) GetLockStatus(resource string) {
    cc.mutex.RLock()
    defer cc.mutex.RUnlock()
    
    if locks, exists := cc.locks[resource]; exists {
        fmt.Printf("Locks on resource %s:\n", resource)
        for _, lock := range locks {
            fmt.Printf("  Transaction %d: %v (since %s)\n", 
                       lock.Transaction, lock.Type, lock.Timestamp.Format(time.RFC3339))
        }
    } else {
        fmt.Printf("No locks on resource %s\n", resource)
    }
    
    if queue, exists := cc.waitQueue[resource]; exists && len(queue) > 0 {
        fmt.Printf("Waiting transactions: %v\n", queue)
    }
}

func main() {
    cc := NewConcurrencyController()
    
    fmt.Println("Concurrency Control Demo:")
    
    // Create some transactions
    txn1 := &Transaction{ID: 1, State: Active, Locks: make(map[string]bool)}
    txn2 := &Transaction{ID: 2, State: Active, Locks: make(map[string]bool)}
    txn3 := &Transaction{ID: 3, State: Active, Locks: make(map[string]bool)}
    
    cc.RegisterTransaction(1, txn1)
    cc.RegisterTransaction(2, txn2)
    cc.RegisterTransaction(3, txn3)
    
    // Test lock acquisition
    cc.AcquireLock(1, "Student", SharedLock)
    cc.AcquireLock(2, "Student", SharedLock)
    cc.AcquireLock(3, "Student", ExclusiveLock) // This should wait
    
    cc.GetLockStatus("Student")
    
    // Release locks
    cc.ReleaseLock(1, "Student")
    cc.ReleaseLock(2, "Student")
    
    cc.GetLockStatus("Student")
}
```

## Locking Mechanisms

### Theory

Locking mechanisms prevent concurrent access to shared resources. Common lock types include shared locks (for reading) and exclusive locks (for writing), with various granularities from row-level to table-level.

### Lock Manager Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type LockGranularity int

const (
    RowLevel LockGranularity = iota
    PageLevel
    TableLevel
)

type LockRequest struct {
    Transaction int
    Resource    string
    Type        LockType
    Granularity LockGranularity
    Timestamp   time.Time
}

type LockManager struct {
    locks       map[string][]Lock
    waitQueue   map[string][]LockRequest
    mutex       sync.RWMutex
    nextLockID  int
}

func NewLockManager() *LockManager {
    return &LockManager{
        locks:     make(map[string][]Lock),
        waitQueue: make(map[string][]LockRequest),
        nextLockID: 1,
    }
}

func (lm *LockManager) RequestLock(txnID int, resource string, lockType LockType, granularity LockGranularity) bool {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    // Check if lock can be granted immediately
    if lm.canGrantLock(resource, lockType, granularity) {
        lm.grantLock(txnID, resource, lockType, granularity)
        return true
    }
    
    // Add to wait queue
    request := LockRequest{
        Transaction: txnID,
        Resource:    resource,
        Type:        lockType,
        Granularity: granularity,
        Timestamp:   time.Now(),
    }
    
    lm.waitQueue[resource] = append(lm.waitQueue[resource], request)
    fmt.Printf("Transaction %d waiting for %v lock on %s\n", txnID, lockType, resource)
    return false
}

func (lm *LockManager) ReleaseLock(txnID int, resource string) bool {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    if locks, exists := lm.locks[resource]; exists {
        for i, lock := range locks {
            if lock.Transaction == txnID {
                // Remove the lock
                lm.locks[resource] = append(locks[:i], locks[i+1:]...)
                fmt.Printf("Transaction %d released lock on %s\n", txnID, resource)
                
                // Process waiting transactions
                lm.processWaitQueue(resource)
                return true
            }
        }
    }
    
    fmt.Printf("Transaction %d does not hold lock on %s\n", txnID, resource)
    return false
}

func (lm *LockManager) canGrantLock(resource string, lockType LockType, granularity LockGranularity) bool {
    if locks, exists := lm.locks[resource]; exists {
        for _, lock := range locks {
            // Check for lock conflicts
            if lm.hasConflict(lockType, lock.Type) {
                return false
            }
        }
    }
    return true
}

func (lm *LockManager) hasConflict(requestType, existingType LockType) bool {
    // Exclusive lock conflicts with any other lock
    if requestType == ExclusiveLock || existingType == ExclusiveLock {
        return true
    }
    return false
}

func (lm *LockManager) grantLock(txnID int, resource string, lockType LockType, granularity LockGranularity) {
    lock := Lock{
        Resource:    resource,
        Type:        lockType,
        Transaction: txnID,
        Timestamp:   time.Now(),
    }
    
    lm.locks[resource] = append(lm.locks[resource], lock)
    fmt.Printf("Transaction %d granted %v lock on %s\n", txnID, lockType, resource)
}

func (lm *LockManager) processWaitQueue(resource string) {
    if queue, exists := lm.waitQueue[resource]; exists && len(queue) > 0 {
        // Process waiting transactions in timestamp order
        for i, request := range queue {
            if lm.canGrantLock(resource, request.Type, request.Granularity) {
                lm.grantLock(request.Transaction, resource, request.Type, request.Granularity)
                
                // Remove from queue
                lm.waitQueue[resource] = append(queue[:i], queue[i+1:]...)
                break
            }
        }
    }
}

func (lm *LockManager) UpgradeLock(txnID int, resource string, newType LockType) bool {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    if locks, exists := lm.locks[resource]; exists {
        for i, lock := range locks {
            if lock.Transaction == txnID {
                // Check if upgrade is possible
                if lm.canUpgradeLock(lock.Type, newType) {
                    lm.locks[resource][i].Type = newType
                    lm.locks[resource][i].Timestamp = time.Now()
                    fmt.Printf("Transaction %d upgraded lock on %s to %v\n", txnID, resource, newType)
                    return true
                } else {
                    fmt.Printf("Cannot upgrade lock for transaction %d on %s\n", txnID, resource)
                    return false
                }
            }
        }
    }
    
    fmt.Printf("Transaction %d does not hold lock on %s\n", txnID, resource)
    return false
}

func (lm *LockManager) canUpgradeLock(currentType, newType LockType) bool {
    // Can upgrade from shared to exclusive
    return currentType == SharedLock && newType == ExclusiveLock
}

func (lm *LockManager) GetLockStatus(resource string) {
    lm.mutex.RLock()
    defer lm.mutex.RUnlock()
    
    if locks, exists := lm.locks[resource]; exists {
        fmt.Printf("Locks on resource %s:\n", resource)
        for _, lock := range locks {
            fmt.Printf("  Transaction %d: %v (since %s)\n", 
                       lock.Transaction, lock.Type, lock.Timestamp.Format(time.RFC3339))
        }
    } else {
        fmt.Printf("No locks on resource %s\n", resource)
    }
    
    if queue, exists := lm.waitQueue[resource]; exists && len(queue) > 0 {
        fmt.Printf("Waiting transactions: %d\n", len(queue))
        for _, request := range queue {
            fmt.Printf("  Transaction %d: %v (since %s)\n", 
                       request.Transaction, request.Type, request.Timestamp.Format(time.RFC3339))
        }
    }
}

func main() {
    lm := NewLockManager()
    
    fmt.Println("Lock Manager Demo:")
    
    // Test lock requests
    lm.RequestLock(1, "Student", SharedLock, RowLevel)
    lm.RequestLock(2, "Student", SharedLock, RowLevel)
    lm.RequestLock(3, "Student", ExclusiveLock, RowLevel) // This should wait
    
    lm.GetLockStatus("Student")
    
    // Test lock upgrade
    lm.UpgradeLock(1, "Student", ExclusiveLock)
    
    lm.GetLockStatus("Student")
    
    // Release locks
    lm.ReleaseLock(1, "Student")
    lm.ReleaseLock(2, "Student")
    
    lm.GetLockStatus("Student")
}
```

## Follow-up Questions

### 1. ACID Properties
**Q: How do you ensure durability in a database system?**
A: Durability is ensured through write-ahead logging (WAL), where all changes are logged to persistent storage before being applied to the database. This ensures that committed transactions survive system failures.

### 2. Concurrency Control
**Q: What are the trade-offs between pessimistic and optimistic concurrency control?**
A: Pessimistic locking prevents conflicts but can reduce concurrency and cause deadlocks. Optimistic concurrency control allows higher concurrency but may require transaction rollbacks when conflicts occur.

### 3. Locking Mechanisms
**Q: When would you use row-level locking versus table-level locking?**
A: Use row-level locking for high-concurrency applications where different transactions access different rows. Use table-level locking for operations that affect the entire table or when row-level locking overhead is too high.

## Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Lock Acquisition | O(1) | O(1) | Hash table lookup |
| Lock Release | O(1) | O(1) | Hash table removal |
| Deadlock Detection | O(V + E) | O(V) | Graph traversal |
| Lock Upgrade | O(1) | O(1) | Direct access |

## Applications

1. **ACID Properties**: Database systems, transaction processing
2. **Concurrency Control**: Multi-user database systems, distributed systems
3. **Locking Mechanisms**: Database engines, file systems, operating systems
4. **Transaction Isolation**: Database systems, financial systems

---

**Next**: [NoSQL Databases](nosql-databases.md/) | **Previous**: [Database Systems](README.md/) | **Up**: [Database Systems](README.md/)
