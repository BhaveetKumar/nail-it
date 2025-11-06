---
# Auto-generated front matter
Title: Database Internals Detailed Theory And Examples
LastUpdated: 2025-11-06T20:45:58.286150
Tags: []
Status: draft
---

# 🗄️ **Database Internals - Detailed Theory & Examples**

## 📊 **Comprehensive Guide with Theory, Examples, and Practical Implementations**

---

## 🎯 **1. B-Tree Index - Deep Dive with Examples**

### **Theory: What is a B-Tree?**

A B-tree is a self-balancing tree data structure that maintains sorted data and allows efficient searches, insertions, and deletions.

**Key Properties:**
- All leaves are at the same level
- Internal nodes have between `t` and `2t-1` keys
- Root has between 1 and `2t-1` keys
- Each node has between `t` and `2t` children

**Why B-Trees?**
- **Disk I/O Optimization**: Minimizes disk reads by keeping nodes large
- **Balanced Height**: O(log n) search time
- **Range Queries**: Efficient for range-based searches
- **Sequential Access**: Good for pagination and sorting

### **Real-World Example: Database Index for E-commerce**

```go
type BTreeIndex struct {
    root   *BTreeNode
    degree int
    height int
    mutex  sync.RWMutex
}

type BTreeNode struct {
    keys     []int
    values   []interface{}
    children []*BTreeNode
    leaf     bool
    degree   int
    mutex    sync.RWMutex
}

// E-commerce product index
type ProductIndex struct {
    btree *BTreeIndex
}

type Product struct {
    ID       int
    Name     string
    Price    float64
    Category string
    Stock    int
}

func NewProductIndex() *ProductIndex {
    return &ProductIndex{
        btree: NewBTree(3), // Degree 3 for demonstration
    }
}

func (pi *ProductIndex) AddProduct(product *Product) error {
    return pi.btree.Insert(product.ID, product)
}

func (pi *ProductIndex) FindProduct(productID int) (*Product, error) {
    value, exists := pi.btree.Search(productID)
    if !exists {
        return nil, errors.New("product not found")
    }
    
    product, ok := value.(*Product)
    if !ok {
        return nil, errors.New("invalid product data")
    }
    
    return product, nil
}

func (pi *ProductIndex) FindProductsInRange(minID, maxID int) ([]*Product, error) {
    return pi.btree.RangeSearch(minID, maxID)
}

// B-Tree implementation
func NewBTree(degree int) *BTreeIndex {
    return &BTreeIndex{
        root:   nil,
        degree: degree,
        height: 0,
    }
}

func (bt *BTreeIndex) Search(key int) (interface{}, bool) {
    bt.mutex.RLock()
    defer bt.mutex.RUnlock()
    
    if bt.root == nil {
        return nil, false
    }
    
    return bt.searchNode(bt.root, key)
}

func (bt *BTreeIndex) searchNode(node *BTreeNode, key int) (interface{}, bool) {
    if node == nil {
        return nil, false
    }
    
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    // Find the key in current node
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }
    
    // Key found in current node
    if i < len(node.keys) && key == node.keys[i] {
        return node.values[i], true
    }
    
    // If leaf node, key not found
    if node.leaf {
        return nil, false
    }
    
    // Search in appropriate child
    return bt.searchNode(node.children[i], key)
}

func (bt *BTreeIndex) Insert(key int, value interface{}) error {
    bt.mutex.Lock()
    defer bt.mutex.Unlock()
    
    if bt.root == nil {
        bt.root = &BTreeNode{
            keys:     []int{key},
            values:   []interface{}{value},
            children: nil,
            leaf:     true,
            degree:   bt.degree,
        }
        return nil
    }
    
    // If root is full, split it
    if len(bt.root.keys) == 2*bt.degree-1 {
        oldRoot := bt.root
        bt.root = &BTreeNode{
            keys:     []int{},
            values:   []interface{}{},
            children: []*BTreeNode{oldRoot},
            leaf:     false,
            degree:   bt.degree,
        }
        
        bt.splitChild(bt.root, 0)
        bt.height++
    }
    
    return bt.insertNonFull(bt.root, key, value)
}

func (bt *BTreeIndex) insertNonFull(node *BTreeNode, key int, value interface{}) error {
    node.mutex.Lock()
    defer node.mutex.Unlock()
    
    i := len(node.keys) - 1
    
    if node.leaf {
        // Insert into leaf node
        node.keys = append(node.keys, 0)
        node.values = append(node.values, nil)
        
        // Shift elements to make room
        for i >= 0 && key < node.keys[i] {
            node.keys[i+1] = node.keys[i]
            node.values[i+1] = node.values[i]
            i--
        }
        
        node.keys[i+1] = key
        node.values[i+1] = value
        return nil
    }
    
    // Find child to insert into
    for i >= 0 && key < node.keys[i] {
        i--
    }
    i++
    
    // Check if child is full
    if len(node.children[i].keys) == 2*bt.degree-1 {
        bt.splitChild(node, i)
        if key > node.keys[i] {
            i++
        }
    }
    
    return bt.insertNonFull(node.children[i], key, value)
}

func (bt *BTreeIndex) splitChild(parent *BTreeNode, index int) {
    child := parent.children[index]
    newChild := &BTreeNode{
        keys:     make([]int, bt.degree-1),
        values:   make([]interface{}, bt.degree-1),
        children: make([]*BTreeNode, bt.degree),
        leaf:     child.leaf,
        degree:   bt.degree,
    }
    
    // Copy second half of child to new child
    for i := 0; i < bt.degree-1; i++ {
        newChild.keys[i] = child.keys[i+bt.degree]
        newChild.values[i] = child.values[i+bt.degree]
    }
    
    if !child.leaf {
        for i := 0; i < bt.degree; i++ {
            newChild.children[i] = child.children[i+bt.degree]
        }
    }
    
    // Resize child
    child.keys = child.keys[:bt.degree-1]
    child.values = child.values[:bt.degree-1]
    child.children = child.children[:bt.degree]
    
    // Insert new child into parent
    parent.children = append(parent.children, nil)
    copy(parent.children[index+2:], parent.children[index+1:])
    parent.children[index+1] = newChild
    
    // Insert middle key into parent
    parent.keys = append(parent.keys, 0)
    parent.values = append(parent.values, nil)
    copy(parent.keys[index+1:], parent.keys[index:])
    copy(parent.values[index+1:], parent.values[index:])
    parent.keys[index] = child.keys[bt.degree-1]
    parent.values[index] = child.values[bt.degree-1]
    
    // Remove middle key from child
    child.keys = child.keys[:bt.degree-1]
    child.values = child.values[:bt.degree-1]
}

func (bt *BTreeIndex) RangeSearch(minKey, maxKey int) ([]interface{}, error) {
    bt.mutex.RLock()
    defer bt.mutex.RUnlock()
    
    if bt.root == nil {
        return nil, errors.New("tree is empty")
    }
    
    var results []interface{}
    bt.rangeSearchNode(bt.root, minKey, maxKey, &results)
    return results, nil
}

func (bt *BTreeIndex) rangeSearchNode(node *BTreeNode, minKey, maxKey int, results *[]interface{}) {
    if node == nil {
        return
    }
    
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    i := 0
    for i < len(node.keys) && minKey > node.keys[i] {
        i++
    }
    
    // Search in children
    if !node.leaf {
        bt.rangeSearchNode(node.children[i], minKey, maxKey, results)
    }
    
    // Check keys in current node
    for i < len(node.keys) && node.keys[i] <= maxKey {
        if node.keys[i] >= minKey {
            *results = append(*results, node.values[i])
        }
        i++
    }
    
    // Search in remaining children
    if !node.leaf {
        for j := i; j < len(node.children); j++ {
            bt.rangeSearchNode(node.children[j], minKey, maxKey, results)
        }
    }
}

// Example usage
func main() {
    productIndex := NewProductIndex()
    
    // Add products
    products := []*Product{
        {ID: 1, Name: "Laptop", Price: 999.99, Category: "Electronics", Stock: 10},
        {ID: 2, Name: "Mouse", Price: 29.99, Category: "Electronics", Stock: 100},
        {ID: 3, Name: "Keyboard", Price: 79.99, Category: "Electronics", Stock: 50},
        {ID: 4, Name: "Monitor", Price: 299.99, Category: "Electronics", Stock: 25},
        {ID: 5, Name: "Headphones", Price: 149.99, Category: "Electronics", Stock: 75},
    }
    
    for _, product := range products {
        productIndex.AddProduct(product)
    }
    
    // Search for specific product
    if product, err := productIndex.FindProduct(3); err == nil {
        fmt.Printf("Found product: %s - $%.2f\n", product.Name, product.Price)
    }
    
    // Search for products in range
    if products, err := productIndex.FindProductsInRange(2, 4); err == nil {
        fmt.Println("Products in range 2-4:")
        for _, product := range products {
            fmt.Printf("  %d: %s - $%.2f\n", product.ID, product.Name, product.Price)
        }
    }
}
```

---

## 🗄️ **2. LSM Tree - Deep Dive with Examples**

### **Theory: What is an LSM Tree?**

Log-Structured Merge Tree (LSM Tree) is optimized for write-heavy workloads by batching writes in memory and periodically flushing to disk.

**Key Components:**
- **MemTable**: In-memory structure for recent writes
- **SSTables**: Immutable files on disk
- **Compaction**: Process of merging SSTables

**Why LSM Trees?**
- **Write Optimization**: Batched writes are more efficient
- **Sequential I/O**: Better for SSDs and HDDs
- **Compression**: SSTables can be compressed
- **Bloom Filters**: Fast non-existence checks

### **Real-World Example: Time-Series Database**

```go
type TimeSeriesDB struct {
    lsmTree *LSMTree
    mutex   sync.RWMutex
}

type TimeSeriesData struct {
    Timestamp time.Time
    Metric    string
    Value     float64
    Tags      map[string]string
}

type LSMTree struct {
    memTable  *MemTable
    sstables  [][]*SSTable
    maxLevels int
    levelSize int
    mutex     sync.RWMutex
}

type MemTable struct {
    data    map[string]*TimeSeriesData
    size    int
    maxSize int
    mutex   sync.RWMutex
}

type SSTable struct {
    filename string
    level    int
    minKey   string
    maxKey   string
    data     map[string]*TimeSeriesData
    mutex    sync.RWMutex
}

func NewTimeSeriesDB() *TimeSeriesDB {
    return &TimeSeriesDB{
        lsmTree: NewLSMTree(1000, 4, 2), // 1000 max mem size, 4 levels, 2 files per level
    }
}

func (tsdb *TimeSeriesDB) WriteData(data *TimeSeriesData) error {
    key := fmt.Sprintf("%d_%s", data.Timestamp.Unix(), data.Metric)
    return tsdb.lsmTree.Put(key, data)
}

func (tsdb *TimeSeriesDB) ReadData(metric string, startTime, endTime time.Time) ([]*TimeSeriesData, error) {
    var results []*TimeSeriesData
    
    // Search in memtable first
    memResults := tsdb.lsmTree.SearchInMemTable(metric, startTime, endTime)
    results = append(results, memResults...)
    
    // Search in SSTables
    sstResults, err := tsdb.lsmTree.SearchInSSTables(metric, startTime, endTime)
    if err != nil {
        return nil, err
    }
    
    results = append(results, sstResults...)
    
    // Sort by timestamp
    sort.Slice(results, func(i, j int) bool {
        return results[i].Timestamp.Before(results[j].Timestamp)
    })
    
    return results, nil
}

func NewLSMTree(maxMemSize, maxLevels, levelSize int) *LSMTree {
    return &LSMTree{
        memTable: &MemTable{
            data:    make(map[string]*TimeSeriesData),
            size:    0,
            maxSize: maxMemSize,
        },
        sstables:  make([][]*SSTable, maxLevels),
        maxLevels: maxLevels,
        levelSize: levelSize,
    }
}

func (lsm *LSMTree) Put(key string, data *TimeSeriesData) error {
    lsm.mutex.Lock()
    defer lsm.mutex.Unlock()
    
    // Add to memtable
    lsm.memTable.mutex.Lock()
    lsm.memTable.data[key] = data
    lsm.memTable.size += len(key) + 8 // Approximate size
    lsm.memTable.mutex.Unlock()
    
    // Check if memtable is full
    if lsm.memTable.size >= lsm.memTable.maxSize {
        return lsm.flushMemTable()
    }
    
    return nil
}

func (lsm *LSMTree) flushMemTable() error {
    // Create new SSTable from memtable
    sstable := &SSTable{
        filename: fmt.Sprintf("sstable_%d_%d.sst", time.Now().Unix(), rand.Intn(1000)),
        level:    0,
        data:     make(map[string]*TimeSeriesData),
    }
    
    // Copy data from memtable
    lsm.memTable.mutex.RLock()
    for key, data := range lsm.memTable.data {
        sstable.data[key] = data
    }
    lsm.memTable.mutex.RUnlock()
    
    // Add to level 0
    lsm.sstables[0] = append(lsm.sstables[0], sstable)
    
    // Clear memtable
    lsm.memTable.mutex.Lock()
    lsm.memTable.data = make(map[string]*TimeSeriesData)
    lsm.memTable.size = 0
    lsm.memTable.mutex.Unlock()
    
    // Trigger compaction if needed
    if len(lsm.sstables[0]) > lsm.levelSize {
        return lsm.compactLevel(0)
    }
    
    return nil
}

func (lsm *LSMTree) compactLevel(level int) error {
    if level >= lsm.maxLevels-1 {
        return nil // Cannot compact last level
    }
    
    // Get all SSTables in current level
    sstables := lsm.sstables[level]
    if len(sstables) <= lsm.levelSize {
        return nil // No compaction needed
    }
    
    // Merge SSTables
    mergedData := make(map[string]*TimeSeriesData)
    for _, sstable := range sstables {
        sstable.mutex.RLock()
        for key, data := range sstable.data {
            mergedData[key] = data
        }
        sstable.mutex.RUnlock()
    }
    
    // Create new SSTable for next level
    newSSTable := &SSTable{
        filename: fmt.Sprintf("sstable_%d_%d.sst", time.Now().Unix(), rand.Intn(1000)),
        level:    level + 1,
        data:     mergedData,
    }
    
    // Add to next level
    lsm.sstables[level+1] = append(lsm.sstables[level+1], newSSTable)
    
    // Clear current level
    lsm.sstables[level] = make([]*SSTable, 0)
    
    // Recursively compact next level if needed
    if len(lsm.sstables[level+1]) > lsm.levelSize {
        return lsm.compactLevel(level + 1)
    }
    
    return nil
}

func (lsm *LSMTree) SearchInMemTable(metric string, startTime, endTime time.Time) []*TimeSeriesData {
    var results []*TimeSeriesData
    
    lsm.memTable.mutex.RLock()
    defer lsm.memTable.mutex.RUnlock()
    
    for _, data := range lsm.memTable.data {
        if data.Metric == metric && 
           data.Timestamp.After(startTime) && 
           data.Timestamp.Before(endTime) {
            results = append(results, data)
        }
    }
    
    return results
}

func (lsm *LSMTree) SearchInSSTables(metric string, startTime, endTime time.Time) ([]*TimeSeriesData, error) {
    var results []*TimeSeriesData
    
    // Search from level 0 to max level
    for level := 0; level < lsm.maxLevels; level++ {
        for _, sstable := range lsm.sstables[level] {
            sstable.mutex.RLock()
            for _, data := range sstable.data {
                if data.Metric == metric && 
                   data.Timestamp.After(startTime) && 
                   data.Timestamp.Before(endTime) {
                    results = append(results, data)
                }
            }
            sstable.mutex.RUnlock()
        }
    }
    
    return results, nil
}

// Example usage
func main() {
    db := NewTimeSeriesDB()
    
    // Write some time series data
    now := time.Now()
    for i := 0; i < 100; i++ {
        data := &TimeSeriesData{
            Timestamp: now.Add(time.Duration(i) * time.Minute),
            Metric:    "cpu_usage",
            Value:     float64(50 + rand.Intn(50)),
            Tags: map[string]string{
                "host": "server1",
                "dc":   "us-east-1",
            },
        }
        db.WriteData(data)
    }
    
    // Read data for a time range
    startTime := now.Add(-1 * time.Hour)
    endTime := now.Add(1 * time.Hour)
    
    results, err := db.ReadData("cpu_usage", startTime, endTime)
    if err == nil {
        fmt.Printf("Found %d data points\n", len(results))
        for i, data := range results {
            if i < 5 { // Show first 5
                fmt.Printf("  %s: %.2f\n", data.Timestamp.Format("15:04:05"), data.Value)
            }
        }
    }
}
```

---

## 🔄 **3. Transaction Management - Deep Dive with Examples**

### **Theory: ACID Properties**

**Atomicity**: All operations in a transaction succeed or all fail
**Consistency**: Database remains in a valid state
**Isolation**: Concurrent transactions don't interfere
**Durability**: Committed changes survive system failures

### **Real-World Example: Banking System with Transactions**

```go
type BankingSystem struct {
    accounts map[string]*Account
    mutex    sync.RWMutex
    log      *TransactionLog
}

type Account struct {
    ID      string
    Balance int
    mutex   sync.RWMutex
}

type Transaction struct {
    ID        string
    operations []Operation
    state     TransactionState
    mutex     sync.RWMutex
}

type Operation struct {
    Type    string // DEBIT, CREDIT, TRANSFER
    Account string
    Amount  int
    ToAccount string // For transfers
}

type TransactionState int

const (
    ACTIVE TransactionState = iota
    COMMITTED
    ABORTED
)

type TransactionLog struct {
    entries []LogEntry
    mutex   sync.RWMutex
}

type LogEntry struct {
    LSN       uint64
    Type      string
    TransactionID string
    Data      interface{}
    Timestamp time.Time
}

func NewBankingSystem() *BankingSystem {
    return &BankingSystem{
        accounts: make(map[string]*Account),
        log:      &TransactionLog{entries: make([]LogEntry, 0)},
    }
}

func (bs *BankingSystem) CreateAccount(accountID string, initialBalance int) error {
    bs.mutex.Lock()
    defer bs.mutex.Unlock()
    
    if _, exists := bs.accounts[accountID]; exists {
        return errors.New("account already exists")
    }
    
    bs.accounts[accountID] = &Account{
        ID:      accountID,
        Balance: initialBalance,
    }
    
    return nil
}

func (bs *BankingSystem) BeginTransaction() *Transaction {
    transaction := &Transaction{
        ID:        generateTransactionID(),
        operations: make([]Operation, 0),
        state:     ACTIVE,
    }
    
    // Log transaction begin
    bs.log.LogEntry(LogEntry{
        LSN:           bs.log.getNextLSN(),
        Type:          "BEGIN",
        TransactionID: transaction.ID,
        Data:          nil,
        Timestamp:     time.Now(),
    })
    
    return transaction
}

func (bs *BankingSystem) TransferMoney(transaction *Transaction, from, to string, amount int) error {
    if transaction.state != ACTIVE {
        return errors.New("transaction not active")
    }
    
    // Add operations to transaction
    transaction.mutex.Lock()
    transaction.operations = append(transaction.operations, Operation{
        Type:       "TRANSFER",
        Account:    from,
        Amount:     amount,
        ToAccount:  to,
    })
    transaction.mutex.Unlock()
    
    // Log the operation
    bs.log.LogEntry(LogEntry{
        LSN:           bs.log.getNextLSN(),
        Type:          "TRANSFER",
        TransactionID: transaction.ID,
        Data:          map[string]interface{}{"from": from, "to": to, "amount": amount},
        Timestamp:     time.Now(),
    })
    
    return nil
}

func (bs *BankingSystem) CommitTransaction(transaction *Transaction) error {
    if transaction.state != ACTIVE {
        return errors.New("transaction not active")
    }
    
    // Two-phase commit
    // Phase 1: Prepare
    if err := bs.prepareTransaction(transaction); err != nil {
        bs.abortTransaction(transaction)
        return err
    }
    
    // Phase 2: Commit
    if err := bs.commitTransaction(transaction); err != nil {
        bs.abortTransaction(transaction)
        return err
    }
    
    return nil
}

func (bs *BankingSystem) prepareTransaction(transaction *Transaction) error {
    // Check if all operations can be performed
    for _, op := range transaction.operations {
        if op.Type == "TRANSFER" {
            account := bs.accounts[op.Account]
            if account == nil {
                return errors.New("account not found")
            }
            
            account.mutex.RLock()
            if account.Balance < op.Amount {
                account.mutex.RUnlock()
                return errors.New("insufficient funds")
            }
            account.mutex.RUnlock()
        }
    }
    
    return nil
}

func (bs *BankingSystem) commitTransaction(transaction *Transaction) error {
    // Acquire locks on all affected accounts
    accounts := make([]*Account, 0)
    for _, op := range transaction.operations {
        if account := bs.accounts[op.Account]; account != nil {
            accounts = append(accounts, account)
        }
        if op.ToAccount != "" {
            if account := bs.accounts[op.ToAccount]; account != nil {
                accounts = append(accounts, account)
            }
        }
    }
    
    // Sort accounts to avoid deadlock
    sort.Slice(accounts, func(i, j int) bool {
        return accounts[i].ID < accounts[j].ID
    })
    
    // Lock all accounts
    for _, account := range accounts {
        account.mutex.Lock()
    }
    
    // Perform operations
    for _, op := range transaction.operations {
        switch op.Type {
        case "TRANSFER":
            fromAccount := bs.accounts[op.Account]
            toAccount := bs.accounts[op.ToAccount]
            
            fromAccount.Balance -= op.Amount
            toAccount.Balance += op.Amount
        }
    }
    
    // Unlock all accounts
    for _, account := range accounts {
        account.mutex.Unlock()
    }
    
    // Update transaction state
    transaction.mutex.Lock()
    transaction.state = COMMITTED
    transaction.mutex.Unlock()
    
    // Log commit
    bs.log.LogEntry(LogEntry{
        LSN:           bs.log.getNextLSN(),
        Type:          "COMMIT",
        TransactionID: transaction.ID,
        Data:          nil,
        Timestamp:     time.Now(),
    })
    
    return nil
}

func (bs *BankingSystem) abortTransaction(transaction *Transaction) {
    transaction.mutex.Lock()
    transaction.state = ABORTED
    transaction.mutex.Unlock()
    
    // Log abort
    bs.log.LogEntry(LogEntry{
        LSN:           bs.log.getNextLSN(),
        Type:          "ABORT",
        TransactionID: transaction.ID,
        Data:          nil,
        Timestamp:     time.Now(),
    })
}

func (bs *BankingSystem) GetAccountBalance(accountID string) (int, error) {
    bs.mutex.RLock()
    account, exists := bs.accounts[accountID]
    bs.mutex.RUnlock()
    
    if !exists {
        return 0, errors.New("account not found")
    }
    
    account.mutex.RLock()
    defer account.mutex.RUnlock()
    
    return account.Balance, nil
}

func (tl *TransactionLog) LogEntry(entry LogEntry) {
    tl.mutex.Lock()
    defer tl.mutex.Unlock()
    
    tl.entries = append(tl.entries, entry)
}

func (tl *TransactionLog) getNextLSN() uint64 {
    tl.mutex.RLock()
    defer tl.mutex.RUnlock()
    
    return uint64(len(tl.entries) + 1)
}

// Example usage
func main() {
    bank := NewBankingSystem()
    
    // Create accounts
    bank.CreateAccount("alice", 1000)
    bank.CreateAccount("bob", 500)
    
    // Begin transaction
    txn := bank.BeginTransaction()
    
    // Transfer money
    bank.TransferMoney(txn, "alice", "bob", 200)
    
    // Commit transaction
    if err := bank.CommitTransaction(txn); err != nil {
        fmt.Printf("Transaction failed: %v\n", err)
        return
    }
    
    // Check balances
    aliceBalance, _ := bank.GetAccountBalance("alice")
    bobBalance, _ := bank.GetAccountBalance("bob")
    
    fmt.Printf("Alice balance: %d\n", aliceBalance)
    fmt.Printf("Bob balance: %d\n", bobBalance)
}
```

---

## 🔒 **4. Concurrency Control - Deep Dive with Examples**

### **Theory: Concurrency Control**

Concurrency control ensures that concurrent transactions don't interfere with each other while maintaining data consistency.

**Techniques:**
- **Locking**: Pessimistic concurrency control
- **MVCC**: Optimistic concurrency control
- **Timestamp Ordering**: Order transactions by timestamp

### **Real-World Example: Multi-User Document Editor**

```go
type DocumentEditor struct {
    documents map[string]*Document
    mutex     sync.RWMutex
    lockManager *LockManager
}

type Document struct {
    ID        string
    Content   string
    Version   int
    mutex     sync.RWMutex
    lastModified time.Time
}

type LockManager struct {
    locks map[string]*Lock
    mutex sync.RWMutex
}

type Lock struct {
    resource    string
    type        LockType
    holders     map[string]*User
    waiters     []*User
    mutex       sync.RWMutex
}

type LockType int

const (
    SHARED_LOCK LockType = iota
    EXCLUSIVE_LOCK
)

type User struct {
    ID   string
    Name string
}

func NewDocumentEditor() *DocumentEditor {
    return &DocumentEditor{
        documents:   make(map[string]*Document),
        lockManager: &LockManager{locks: make(map[string]*Lock)},
    }
}

func (de *DocumentEditor) CreateDocument(docID, content string) error {
    de.mutex.Lock()
    defer de.mutex.Unlock()
    
    if _, exists := de.documents[docID]; exists {
        return errors.New("document already exists")
    }
    
    de.documents[docID] = &Document{
        ID:          docID,
        Content:     content,
        Version:     1,
        lastModified: time.Now(),
    }
    
    return nil
}

func (de *DocumentEditor) ReadDocument(user *User, docID string) (string, error) {
    // Acquire shared lock
    if err := de.lockManager.AcquireLock(user, docID, SHARED_LOCK); err != nil {
        return "", err
    }
    defer de.lockManager.ReleaseLock(user, docID)
    
    de.mutex.RLock()
    doc, exists := de.documents[docID]
    de.mutex.RUnlock()
    
    if !exists {
        return "", errors.New("document not found")
    }
    
    doc.mutex.RLock()
    defer doc.mutex.RUnlock()
    
    return doc.Content, nil
}

func (de *DocumentEditor) WriteDocument(user *User, docID, content string) error {
    // Acquire exclusive lock
    if err := de.lockManager.AcquireLock(user, docID, EXCLUSIVE_LOCK); err != nil {
        return err
    }
    defer de.lockManager.ReleaseLock(user, docID)
    
    de.mutex.RLock()
    doc, exists := de.documents[docID]
    de.mutex.RUnlock()
    
    if !exists {
        return errors.New("document not found")
    }
    
    doc.mutex.Lock()
    defer doc.mutex.Unlock()
    
    doc.Content = content
    doc.Version++
    doc.lastModified = time.Now()
    
    return nil
}

func (lm *LockManager) AcquireLock(user *User, resource string, lockType LockType) error {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    lock, exists := lm.locks[resource]
    if !exists {
        lock = &Lock{
            resource: resource,
            type:     lockType,
            holders:  make(map[string]*User),
            waiters:  make([]*User, 0),
        }
        lm.locks[resource] = lock
    }
    
    lock.mutex.Lock()
    defer lock.mutex.Unlock()
    
    // Check if lock can be acquired
    if lm.canAcquireLock(lock, user, lockType) {
        lock.holders[user.ID] = user
        return nil
    }
    
    // Add to waiters
    lock.waiters = append(lock.waiters, user)
    return errors.New("lock not available")
}

func (lm *LockManager) canAcquireLock(lock *Lock, user *User, lockType LockType) bool {
    // If no holders, can acquire
    if len(lock.holders) == 0 {
        return true
    }
    
    // If requesting shared lock and all holders have shared locks
    if lockType == SHARED_LOCK {
        for _, holder := range lock.holders {
            if holder.ID != user.ID {
                return false
            }
        }
        return true
    }
    
    // If requesting exclusive lock, no other holders allowed
    if lockType == EXCLUSIVE_LOCK {
        return len(lock.holders) == 0
    }
    
    return false
}

func (lm *LockManager) ReleaseLock(user *User, resource string) error {
    lm.mutex.Lock()
    defer lm.mutex.Unlock()
    
    lock, exists := lm.locks[resource]
    if !exists {
        return errors.New("lock not found")
    }
    
    lock.mutex.Lock()
    defer lock.mutex.Unlock()
    
    // Remove user from holders
    delete(lock.holders, user.ID)
    
    // If no more holders, remove lock
    if len(lock.holders) == 0 {
        delete(lm.locks, resource)
    }
    
    return nil
}

// Example usage
func main() {
    editor := NewDocumentEditor()
    
    // Create a document
    editor.CreateDocument("doc1", "Hello, World!")
    
    // Create users
    alice := &User{ID: "alice", Name: "Alice"}
    bob := &User{ID: "bob", Name: "Bob"}
    
    // Alice reads the document
    if content, err := editor.ReadDocument(alice, "doc1"); err == nil {
        fmt.Printf("Alice reads: %s\n", content)
    }
    
    // Bob tries to write (will wait for Alice to finish reading)
    go func() {
        if err := editor.WriteDocument(bob, "doc1", "Hello, Bob!"); err != nil {
            fmt.Printf("Bob write error: %v\n", err)
        } else {
            fmt.Println("Bob successfully wrote to document")
        }
    }()
    
    // Alice reads again
    if content, err := editor.ReadDocument(alice, "doc1"); err == nil {
        fmt.Printf("Alice reads again: %s\n", content)
    }
    
    // Wait for Bob's write to complete
    time.Sleep(100 * time.Millisecond)
}
```

---

## 🎯 **Key Takeaways**

### **1. B-Tree Indexes**
- **Balanced height**: O(log n) search time
- **Disk I/O optimization**: Large nodes minimize disk reads
- **Range queries**: Efficient for sequential access
- **Use cases**: Primary keys, range queries, sorting

### **2. LSM Trees**
- **Write optimization**: Batched writes are more efficient
- **Sequential I/O**: Better for SSDs and HDDs
- **Compression**: SSTables can be compressed
- **Use cases**: Time-series databases, write-heavy workloads

### **3. Transaction Management**
- **ACID properties**: Atomicity, Consistency, Isolation, Durability
- **Two-phase commit**: Prepare and commit phases
- **Logging**: Write-ahead logging for durability
- **Use cases**: Banking systems, financial transactions

### **4. Concurrency Control**
- **Locking**: Pessimistic concurrency control
- **MVCC**: Optimistic concurrency control
- **Deadlock prevention**: Order locks to avoid deadlocks
- **Use cases**: Multi-user systems, document editors

---

**🎉 This comprehensive guide provides deep understanding of database internals with practical examples and implementations! 🚀**
