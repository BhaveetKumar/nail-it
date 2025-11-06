---
# Auto-generated front matter
Title: Asli Engineering Database Internals
LastUpdated: 2025-11-06T20:45:58.286572
Tags: []
Status: draft
---

# 🗄️ **Database Internals - Asli Engineering Deep Dive**

## 📊 **Based on Arpit Bhyani's Database Internals Videos**

---

## 🎯 **Core Database Concepts**

### **1. Storage Engines**

#### **InnoDB (MySQL)**
- **ACID compliant** transaction support
- **Row-level locking** for concurrency
- **Clustered indexes** for primary key
- **MVCC (Multi-Version Concurrency Control)**

```go
type InnoDBStorage struct {
    bufferPool *BufferPool
    logBuffer  *LogBuffer
    undoLog    *UndoLog
    redoLog    *RedoLog
}

type BufferPool struct {
    pages    map[uint64]*Page
    dirty    map[uint64]bool
    capacity int
    mutex    sync.RWMutex
}

func (bp *BufferPool) GetPage(pageID uint64) (*Page, error) {
    bp.mutex.RLock()
    page, exists := bp.pages[pageID]
    bp.mutex.RUnlock()
    
    if exists {
        return page, nil
    }
    
    // Page not in buffer pool, load from disk
    return bp.loadPageFromDisk(pageID)
}
```

#### **MyISAM (MySQL)**
- **Table-level locking**
- **Faster reads** for read-heavy workloads
- **No transaction support**
- **Compressed storage** for better space utilization

```go
type MyISAMStorage struct {
    dataFile   *os.File
    indexFile  *os.File
    tableLock  sync.RWMutex
}

func (ms *MyISAMStorage) Read(key string) ([]byte, error) {
    ms.tableLock.RLock()
    defer ms.tableLock.RUnlock()
    
    // Read from data file using index
    return ms.readFromDataFile(key)
}
```

### **2. Index Structures**

#### **B-Tree Index**
- **Balanced tree** structure
- **O(log n)** search, insert, delete
- **Range queries** supported
- **Sequential access** efficient

```go
type BTreeIndex struct {
    root   *BTreeNode
    degree int
    height int
}

type BTreeNode struct {
    keys     []int
    values   []interface{}
    children []*BTreeNode
    leaf     bool
    degree   int
}

func (bt *BTreeIndex) Search(key int) (interface{}, bool) {
    return bt.searchNode(bt.root, key)
}

func (bt *BTreeIndex) searchNode(node *BTreeNode, key int) (interface{}, bool) {
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
    
    if node.leaf {
        return nil, false
    }
    
    return bt.searchNode(node.children[i], key)
}

func (bt *BTreeIndex) Insert(key int, value interface{}) {
    if bt.root == nil {
        bt.root = &BTreeNode{
            keys:     []int{key},
            values:   []interface{}{value},
            children: nil,
            leaf:     true,
            degree:   bt.degree,
        }
        return
    }
    
    if len(bt.root.keys) == 2*bt.degree-1 {
        // Root is full, need to split
        newRoot := &BTreeNode{
            keys:     []int{},
            values:   []interface{}{},
            children: []*BTreeNode{bt.root},
            leaf:     false,
            degree:   bt.degree,
        }
        
        bt.splitChild(newRoot, 0)
        bt.root = newRoot
    }
    
    bt.insertNonFull(bt.root, key, value)
}
```

#### **Hash Index**
- **O(1)** average case lookup
- **No range queries** supported
- **Memory efficient** for exact matches
- **Collision handling** required

```go
type HashIndex struct {
    buckets []*Bucket
    size    int
    hashFunc func(int) int
}

type Bucket struct {
    entries []*Entry
    mutex   sync.RWMutex
}

type Entry struct {
    key   int
    value interface{}
}

func NewHashIndex(size int) *HashIndex {
    buckets := make([]*Bucket, size)
    for i := 0; i < size; i++ {
        buckets[i] = &Bucket{
            entries: make([]*Entry, 0),
        }
    }
    
    return &HashIndex{
        buckets:  buckets,
        size:     size,
        hashFunc: func(key int) int { return key % size },
    }
}

func (hi *HashIndex) Get(key int) (interface{}, bool) {
    bucketIndex := hi.hashFunc(key)
    bucket := hi.buckets[bucketIndex]
    
    bucket.mutex.RLock()
    defer bucket.mutex.RUnlock()
    
    for _, entry := range bucket.entries {
        if entry.key == key {
            return entry.value, true
        }
    }
    
    return nil, false
}

func (hi *HashIndex) Put(key int, value interface{}) {
    bucketIndex := hi.hashFunc(key)
    bucket := hi.buckets[bucketIndex]
    
    bucket.mutex.Lock()
    defer bucket.mutex.Unlock()
    
    // Check if key already exists
    for i, entry := range bucket.entries {
        if entry.key == key {
            bucket.entries[i].value = value
            return
        }
    }
    
    // Add new entry
    bucket.entries = append(bucket.entries, &Entry{
        key:   key,
        value: value,
    })
}
```

### **3. Query Processing**

#### **Query Parser**
```go
type QueryParser struct {
    lexer *Lexer
}

type SQLQuery struct {
    Type      string // SELECT, INSERT, UPDATE, DELETE
    Table     string
    Columns   []string
    Where     *Condition
    OrderBy   []string
    Limit     int
    Offset    int
}

type Condition struct {
    Column    string
    Operator  string
    Value     interface{}
    LogicalOp string // AND, OR
    Left      *Condition
    Right     *Condition
}

func (qp *QueryParser) Parse(query string) (*SQLQuery, error) {
    tokens := qp.lexer.Tokenize(query)
    return qp.parseTokens(tokens)
}

func (qp *QueryParser) parseTokens(tokens []Token) (*SQLQuery, error) {
    if len(tokens) == 0 {
        return nil, errors.New("empty query")
    }
    
    queryType := tokens[0].Value
    query := &SQLQuery{Type: queryType}
    
    switch queryType {
    case "SELECT":
        return qp.parseSelect(tokens[1:], query)
    case "INSERT":
        return qp.parseInsert(tokens[1:], query)
    case "UPDATE":
        return qp.parseUpdate(tokens[1:], query)
    case "DELETE":
        return qp.parseDelete(tokens[1:], query)
    default:
        return nil, fmt.Errorf("unsupported query type: %s", queryType)
    }
}
```

#### **Query Optimizer**
```go
type QueryOptimizer struct {
    statistics *Statistics
    indexes    map[string]*Index
}

type ExecutionPlan struct {
    Steps []ExecutionStep
    Cost  float64
}

type ExecutionStep struct {
    Type        string
    Table       string
    Index       *Index
    Filter      *Condition
    JoinType    string
    LeftChild   *ExecutionStep
    RightChild  *ExecutionStep
}

func (qo *QueryOptimizer) Optimize(query *SQLQuery) *ExecutionPlan {
    // Generate multiple execution plans
    plans := qo.generatePlans(query)
    
    // Choose the plan with lowest cost
    bestPlan := plans[0]
    for _, plan := range plans[1:] {
        if plan.Cost < bestPlan.Cost {
            bestPlan = plan
        }
    }
    
    return bestPlan
}

func (qo *QueryOptimizer) generatePlans(query *SQLQuery) []*ExecutionPlan {
    plans := make([]*ExecutionPlan, 0)
    
    // Plan 1: Full table scan
    plans = append(plans, &ExecutionPlan{
        Steps: []ExecutionStep{
            {
                Type:   "TableScan",
                Table:  query.Table,
                Filter: query.Where,
            },
        },
        Cost: qo.calculateTableScanCost(query.Table),
    })
    
    // Plan 2: Index scan if applicable
    if index, exists := qo.indexes[query.Table]; exists {
        plans = append(plans, &ExecutionPlan{
            Steps: []ExecutionStep{
                {
                    Type:   "IndexScan",
                    Table:  query.Table,
                    Index:  index,
                    Filter: query.Where,
                },
            },
            Cost: qo.calculateIndexScanCost(index, query.Where),
        })
    }
    
    return plans
}
```

### **4. Transaction Management**

#### **ACID Properties**

##### **Atomicity**
```go
type Transaction struct {
    ID        string
    operations []Operation
    state     TransactionState
    mutex     sync.RWMutex
}

type Operation struct {
    Type string // READ, WRITE, DELETE
    Table string
    Key   string
    Value interface{}
}

func (t *Transaction) Commit() error {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if t.state != ACTIVE {
        return errors.New("transaction not active")
    }
    
    // Execute all operations
    for _, op := range t.operations {
        if err := t.executeOperation(op); err != nil {
            // Rollback on error
            t.rollback()
            return err
        }
    }
    
    t.state = COMMITTED
    return nil
}

func (t *Transaction) Rollback() error {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if t.state != ACTIVE {
        return errors.New("transaction not active")
    }
    
    t.rollback()
    t.state = ABORTED
    return nil
}
```

##### **Consistency**
```go
type ConsistencyChecker struct {
    constraints []Constraint
}

type Constraint struct {
    Type     string // UNIQUE, FOREIGN_KEY, CHECK
    Table    string
    Column   string
    Value    interface{}
    RefTable string
    RefColumn string
}

func (cc *ConsistencyChecker) Validate(transaction *Transaction) error {
    for _, op := range transaction.operations {
        for _, constraint := range cc.constraints {
            if constraint.Table == op.Table {
                if err := cc.validateConstraint(constraint, op); err != nil {
                    return err
                }
            }
        }
    }
    return nil
}
```

##### **Isolation**
```go
type IsolationLevel int

const (
    READ_UNCOMMITTED IsolationLevel = iota
    READ_COMMITTED
    REPEATABLE_READ
    SERIALIZABLE
)

type LockManager struct {
    locks map[string]*Lock
    mutex sync.RWMutex
}

type Lock struct {
    resource string
    type     LockType
    holders  map[string]*Transaction
    waiters  []*Transaction
}

type LockType int

const (
    SHARED_LOCK LockType = iota
    EXCLUSIVE_LOCK
)

func (lm *LockManager) AcquireLock(transaction *Transaction, resource string, lockType LockType) error {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    lock, exists := lm.locks[resource]
    if !exists {
        lock = &Lock{
            resource: resource,
            type:     lockType,
            holders:  make(map[string]*Transaction),
            waiters:  make([]*Transaction, 0),
        }
        lm.locks[resource] = lock
    }
    
    // Check if lock can be acquired
    if lm.canAcquireLock(lock, transaction, lockType) {
        lock.holders[transaction.ID] = transaction
        return nil
    }
    
    // Add to waiters
    lock.waiters = append(lock.waiters, transaction)
    return errors.New("lock not available")
}
```

##### **Durability**
```go
type WAL struct {
    logFile *os.File
    mutex   sync.Mutex
}

type LogEntry struct {
    LSN       uint64 // Log Sequence Number
    Type      string // COMMIT, ABORT, WRITE
    TransactionID string
    Table     string
    Key       string
    OldValue  interface{}
    NewValue  interface{}
    Timestamp time.Time
}

func (wal *WAL) WriteLog(entry *LogEntry) error {
    wal.mutex.Lock()
    defer wal.mutex.Unlock()
    
    // Write to log file
    data, err := json.Marshal(entry)
    if err != nil {
        return err
    }
    
    _, err = wal.logFile.Write(data)
    if err != nil {
        return err
    }
    
    // Force write to disk
    return wal.logFile.Sync()
}

func (wal *WAL) Recover() error {
    wal.mutex.Lock()
    defer wal.mutex.Unlock()
    
    // Read log entries and replay
    scanner := bufio.NewScanner(wal.logFile)
    for scanner.Scan() {
        var entry LogEntry
        if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
            continue
        }
        
        if err := wal.replayEntry(&entry); err != nil {
            return err
        }
    }
    
    return scanner.Err()
}
```

### **5. Concurrency Control**

#### **MVCC (Multi-Version Concurrency Control)**
```go
type MVCCManager struct {
    versions map[string][]*Version
    mutex    sync.RWMutex
}

type Version struct {
    Value     interface{}
    CreatedBy string
    CreatedAt time.Time
    DeletedAt *time.Time
}

func (mvcc *MVCCManager) Read(key string, transactionID string, timestamp time.Time) (interface{}, error) {
    mvcc.mutex.RLock()
    versions, exists := mvcc.versions[key]
    mvcc.mutex.RUnlock()
    
    if !exists {
        return nil, errors.New("key not found")
    }
    
    // Find the latest version visible to this transaction
    for i := len(versions) - 1; i >= 0; i-- {
        version := versions[i]
        if version.CreatedAt.Before(timestamp) && 
           (version.DeletedAt == nil || version.DeletedAt.After(timestamp)) {
            return version.Value, nil
        }
    }
    
    return nil, errors.New("no visible version found")
}

func (mvcc *MVCCManager) Write(key string, value interface{}, transactionID string) error {
    mvcc.mutex.Lock()
    defer mvcc.mutex.Unlock()
    
    version := &Version{
        Value:     value,
        CreatedBy: transactionID,
        CreatedAt: time.Now(),
        DeletedAt: nil,
    }
    
    if versions, exists := mvcc.versions[key]; exists {
        mvcc.versions[key] = append(versions, version)
    } else {
        mvcc.versions[key] = []*Version{version}
    }
    
    return nil
}
```

### **6. Buffer Pool Management**

```go
type BufferPool struct {
    pages     map[uint64]*Page
    dirty     map[uint64]bool
    capacity  int
    lru       *LRUCache
    mutex     sync.RWMutex
}

type Page struct {
    ID       uint64
    Data     []byte
    Dirty    bool
    PinCount int
    mutex    sync.RWMutex
}

func (bp *BufferPool) GetPage(pageID uint64) (*Page, error) {
    bp.mutex.RLock()
    page, exists := bp.pages[pageID]
    bp.mutex.RUnlock()
    
    if exists {
        page.mutex.Lock()
        page.PinCount++
        page.mutex.Unlock()
        return page, nil
    }
    
    // Page not in buffer pool
    return bp.loadPageFromDisk(pageID)
}

func (bp *BufferPool) loadPageFromDisk(pageID uint64) (*Page, error) {
    // Check if buffer pool is full
    if len(bp.pages) >= bp.capacity {
        if err := bp.evictPage(); err != nil {
            return nil, err
        }
    }
    
    // Load page from disk
    data, err := bp.readPageFromDisk(pageID)
    if err != nil {
        return nil, err
    }
    
    page := &Page{
        ID:       pageID,
        Data:     data,
        Dirty:    false,
        PinCount: 1,
    }
    
    bp.mutex.Lock()
    bp.pages[pageID] = page
    bp.mutex.Unlock()
    
    return page, nil
}

func (bp *BufferPool) evictPage() error {
    // Find a page to evict (LRU)
    pageID := bp.lru.GetLRU()
    if pageID == 0 {
        return errors.New("no page to evict")
    }
    
    bp.mutex.Lock()
    page, exists := bp.pages[pageID]
    if !exists {
        bp.mutex.Unlock()
        return errors.New("page not found")
    }
    
    // Check if page is pinned
    if page.PinCount > 0 {
        bp.mutex.Unlock()
        return errors.New("page is pinned")
    }
    
    // Write dirty page to disk
    if page.Dirty {
        if err := bp.writePageToDisk(page); err != nil {
            bp.mutex.Unlock()
            return err
        }
    }
    
    // Remove from buffer pool
    delete(bp.pages, pageID)
    delete(bp.dirty, pageID)
    bp.mutex.Unlock()
    
    return nil
}
```

---

## 🎯 **Key Takeaways**

### **1. Storage Engine Selection**
- **InnoDB**: ACID transactions, row-level locking
- **MyISAM**: Fast reads, table-level locking
- **Memory**: In-memory storage, fast but volatile

### **2. Index Strategy**
- **B-Tree**: Range queries, sequential access
- **Hash**: Exact matches, O(1) lookup
- **Composite**: Multiple columns, covering indexes

### **3. Query Optimization**
- **Statistics**: Cardinality, selectivity
- **Execution plans**: Cost-based optimization
- **Index usage**: Proper index selection

### **4. Concurrency Control**
- **Locking**: Pessimistic concurrency control
- **MVCC**: Optimistic concurrency control
- **Isolation levels**: Balance consistency vs performance

### **5. Transaction Management**
- **ACID properties**: Atomicity, Consistency, Isolation, Durability
- **WAL**: Write-ahead logging for durability
- **Recovery**: Crash recovery and rollback

---

**🎉 This comprehensive database internals guide provides deep understanding of how databases work internally, essential for system design interviews! 🚀**
