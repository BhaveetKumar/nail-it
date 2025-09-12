# 🌐 **Distributed Systems - Detailed Theory & Examples**

## 📊 **Comprehensive Guide with Theory, Examples, and Practical Implementations**

---

## 🎯 **1. CAP Theorem - Deep Dive with Examples**

### **Theory: Understanding the CAP Theorem**

The CAP Theorem, formulated by Eric Brewer, states that in a distributed system, you can only guarantee **two out of three** properties:

- **Consistency (C)**: All nodes see the same data at the same time
- **Availability (A)**: System remains operational and accessible
- **Partition Tolerance (P)**: System continues to work despite network failures

### **Why Only Two Out of Three?**

When a network partition occurs (nodes can't communicate), you must choose:
- **Consistency**: Reject requests to maintain data consistency
- **Availability**: Accept requests but may serve stale data

### **Real-World Examples**

#### **Example 1: Banking System (CP System)**

**Scenario**: A bank's distributed database with account balances.

```go
type BankingSystem struct {
    nodes []BankNode
    quorum int
}

type BankNode struct {
    ID       string
    accounts map[string]int
    mutex    sync.RWMutex
}

type AccountBalance struct {
    AccountID string
    Balance   int
    Timestamp time.Time
}

// CP System: Prioritizes Consistency over Availability
func (bs *BankingSystem) TransferMoney(from, to string, amount int) error {
    // Step 1: Acquire locks on all nodes for consistency
    locks := make([]sync.Locker, len(bs.nodes))
    for i, node := range bs.nodes {
        locks[i] = &node.mutex
        locks[i].Lock()
    }
    defer func() {
        for _, lock := range locks {
            lock.Unlock()
        }
    }()
    
    // Step 2: Check if all nodes have consistent data
    if !bs.isConsistent() {
        return errors.New("system inconsistent, rejecting transaction")
    }
    
    // Step 3: Perform transfer atomically
    for _, node := range bs.nodes {
        if node.accounts[from] < amount {
            return errors.New("insufficient funds")
        }
        node.accounts[from] -= amount
        node.accounts[to] += amount
    }
    
    return nil
}

func (bs *BankingSystem) isConsistent() bool {
    // Check if all nodes have the same account balances
    reference := bs.nodes[0].accounts
    for i := 1; i < len(bs.nodes); i++ {
        if !reflect.DeepEqual(reference, bs.nodes[i].accounts) {
            return false
        }
    }
    return true
}
```

**Trade-off**: If network partition occurs, the system becomes unavailable to maintain consistency.

#### **Example 2: Social Media Feed (AP System)**

**Scenario**: A social media platform showing user feeds.

```go
type SocialMediaSystem struct {
    nodes []FeedNode
    cache *Cache
}

type FeedNode struct {
    ID     string
    posts  []Post
    mutex  sync.RWMutex
}

type Post struct {
    ID        string
    UserID    string
    Content   string
    Timestamp time.Time
}

// AP System: Prioritizes Availability over Consistency
func (sms *SocialMediaSystem) GetUserFeed(userID string) ([]Post, error) {
    // Try to get from any available node
    for _, node := range sms.nodes {
        if sms.isNodeHealthy(node) {
            posts, err := sms.getFeedFromNode(node, userID)
            if err == nil {
                // Cache the result for future requests
                sms.cache.Set(fmt.Sprintf("feed_%s", userID), posts, 5*time.Minute)
                return posts, nil
            }
        }
    }
    
    // If no nodes available, return cached data
    if cached, exists := sms.cache.Get(fmt.Sprintf("feed_%s", userID)); exists {
        return cached.([]Post), nil
    }
    
    return nil, errors.New("no nodes available")
}

func (sms *SocialMediaSystem) AddPost(userID, content string) error {
    post := Post{
        ID:        generateID(),
        UserID:    userID,
        Content:   content,
        Timestamp: time.Now(),
    }
    
    // Write to any available node
    for _, node := range sms.nodes {
        if sms.isNodeHealthy(node) {
            node.mutex.Lock()
            node.posts = append(node.posts, post)
            node.mutex.Unlock()
            
            // Invalidate cache
            sms.cache.Delete(fmt.Sprintf("feed_%s", userID))
            return nil
        }
    }
    
    return errors.New("no nodes available for write")
}
```

**Trade-off**: Users might see slightly different feeds, but the system remains available.

### **Practical Implementation: CAP System Chooser**

```go
type CAPSystemChooser struct {
    consistencyRequired bool
    availabilityRequired bool
    partitionToleranceRequired bool
}

func (csc *CAPSystemChooser) ChooseSystem() string {
    if !csc.partitionToleranceRequired {
        return "CA System - Single node database (not suitable for distributed systems)"
    }
    
    if csc.consistencyRequired && csc.availabilityRequired {
        return "Impossible - Cannot have all three in distributed systems"
    }
    
    if csc.consistencyRequired {
        return "CP System - Choose Consistency over Availability (e.g., MongoDB, PostgreSQL)"
    }
    
    if csc.availabilityRequired {
        return "AP System - Choose Availability over Consistency (e.g., Cassandra, DynamoDB)"
    }
    
    return "Unknown configuration"
}

// Example usage
func main() {
    // Banking system - needs consistency
    banking := &CAPSystemChooser{
        consistencyRequired: true,
        availabilityRequired: false,
        partitionToleranceRequired: true,
    }
    fmt.Println("Banking System:", banking.ChooseSystem()) // CP System
    
    // Social media - needs availability
    social := &CAPSystemChooser{
        consistencyRequired: false,
        availabilityRequired: true,
        partitionToleranceRequired: true,
    }
    fmt.Println("Social Media:", social.ChooseSystem()) // AP System
}
```

---

## 🔄 **2. Consensus Algorithms - Theory and Examples**

### **Theory: What is Consensus?**

Consensus is the process of getting all nodes in a distributed system to agree on a single value, even in the presence of failures.

**Properties of Consensus:**
- **Safety**: All nodes agree on the same value
- **Liveness**: Eventually, a decision is made

### **Raft Algorithm - Detailed Example**

#### **Theory: Raft Algorithm**

Raft divides consensus into three sub-problems:
1. **Leader Election**: Select a leader when current leader fails
2. **Log Replication**: Leader replicates log entries to followers
3. **Safety**: Ensure safety properties are maintained

#### **Practical Implementation: Raft with Real Example**

```go
type RaftNode struct {
    ID          string
    state       NodeState
    currentTerm int
    votedFor    string
    log         []LogEntry
    commitIndex int
    lastApplied int
    nextIndex   map[string]int
    matchIndex  map[string]int
    peers       map[string]*RaftNode
    mutex       sync.RWMutex
    electionTimeout time.Duration
    heartbeatInterval time.Duration
}

type LogEntry struct {
    Term    int
    Index   int
    Command interface{}
}

type NodeState int

const (
    Follower NodeState = iota
    Candidate
    Leader
)

// Real-world example: Distributed Key-Value Store
type DistributedKVStore struct {
    raftNode *RaftNode
    data     map[string]string
    mutex    sync.RWMutex
}

func NewDistributedKVStore(nodeID string, peers []string) *DistributedKVStore {
    raftNode := &RaftNode{
        ID:               nodeID,
        state:            Follower,
        currentTerm:      0,
        votedFor:        "",
        log:             make([]LogEntry, 0),
        commitIndex:     -1,
        lastApplied:     -1,
        nextIndex:       make(map[string]int),
        matchIndex:      make(map[string]int),
        peers:           make(map[string]*RaftNode),
        electionTimeout:  time.Duration(150+rand.Intn(150)) * time.Millisecond,
        heartbeatInterval: 50 * time.Millisecond,
    }
    
    // Initialize peer connections
    for _, peerID := range peers {
        if peerID != nodeID {
            raftNode.peers[peerID] = &RaftNode{ID: peerID}
        }
    }
    
    return &DistributedKVStore{
        raftNode: raftNode,
        data:     make(map[string]string),
    }
}

// Client operations
func (dkvs *DistributedKVStore) Set(key, value string) error {
    // Only leader can accept writes
    if dkvs.raftNode.state != Leader {
        return errors.New("not leader, redirect to leader")
    }
    
    // Create log entry
    logEntry := LogEntry{
        Term:    dkvs.raftNode.currentTerm,
        Index:   len(dkvs.raftNode.log),
        Command: map[string]string{"action": "set", "key": key, "value": value},
    }
    
    // Append to log
    dkvs.raftNode.mutex.Lock()
    dkvs.raftNode.log = append(dkvs.raftNode.log, logEntry)
    dkvs.raftNode.mutex.Unlock()
    
    // Replicate to followers
    return dkvs.replicateLogEntry(logEntry)
}

func (dkvs *DistributedKVStore) Get(key string) (string, error) {
    dkvs.mutex.RLock()
    defer dkvs.mutex.RUnlock()
    
    value, exists := dkvs.data[key]
    if !exists {
        return "", errors.New("key not found")
    }
    
    return value, nil
}

// Raft implementation
func (dkvs *DistributedKVStore) replicateLogEntry(entry LogEntry) error {
    successCount := 1 // Leader counts as success
    
    for peerID, peer := range dkvs.raftNode.peers {
        go func(id string, p *RaftNode) {
            // Send AppendEntries RPC
            success := dkvs.sendAppendEntries(id, entry)
            if success {
                dkvs.raftNode.mutex.Lock()
                dkvs.raftNode.matchIndex[id] = entry.Index
                dkvs.raftNode.mutex.Unlock()
                successCount++
                
                // Check if majority has replicated
                if successCount > len(dkvs.raftNode.peers)/2 {
                    dkvs.commitLogEntry(entry)
                }
            }
        }(peerID, peer)
    }
    
    return nil
}

func (dkvs *DistributedKVStore) commitLogEntry(entry LogEntry) {
    dkvs.raftNode.mutex.Lock()
    dkvs.raftNode.commitIndex = entry.Index
    dkvs.raftNode.mutex.Unlock()
    
    // Apply to state machine
    dkvs.applyLogEntry(entry)
}

func (dkvs *DistributedKVStore) applyLogEntry(entry LogEntry) {
    command := entry.Command.(map[string]string)
    
    dkvs.mutex.Lock()
    defer dkvs.mutex.Unlock()
    
    switch command["action"] {
    case "set":
        dkvs.data[command["key"]] = command["value"]
    case "delete":
        delete(dkvs.data, command["key"])
    }
}

// Leader election
func (dkvs *DistributedKVStore) startElection() {
    dkvs.raftNode.mutex.Lock()
    dkvs.raftNode.state = Candidate
    dkvs.raftNode.currentTerm++
    dkvs.raftNode.votedFor = dkvs.raftNode.ID
    dkvs.raftNode.mutex.Unlock()
    
    votes := 1 // Vote for self
    totalVotes := len(dkvs.raftNode.peers) + 1
    
    for peerID, peer := range dkvs.raftNode.peers {
        go func(id string, p *RaftNode) {
            req := &RequestVoteRequest{
                Term:         dkvs.raftNode.currentTerm,
                CandidateID:  dkvs.raftNode.ID,
                LastLogIndex: len(dkvs.raftNode.log) - 1,
                LastLogTerm:  dkvs.getLastLogTerm(),
            }
            
            resp := p.RequestVote(req)
            
            dkvs.raftNode.mutex.Lock()
            if resp.Term > dkvs.raftNode.currentTerm {
                dkvs.raftNode.currentTerm = resp.Term
                dkvs.raftNode.state = Follower
                dkvs.raftNode.votedFor = ""
            } else if resp.VoteGranted {
                votes++
                if votes > totalVotes/2 {
                    dkvs.becomeLeader()
                }
            }
            dkvs.raftNode.mutex.Unlock()
        }(peerID, peer)
    }
}

// Example usage
func main() {
    // Create a 3-node cluster
    node1 := NewDistributedKVStore("node1", []string{"node2", "node3"})
    node2 := NewDistributedKVStore("node2", []string{"node1", "node3"})
    node3 := NewDistributedKVStore("node3", []string{"node1", "node2"})
    
    // Start the nodes
    go node1.startRaft()
    go node2.startRaft()
    go node3.startRaft()
    
    // Wait for leader election
    time.Sleep(1 * time.Second)
    
    // Perform operations
    if err := node1.Set("key1", "value1"); err != nil {
        fmt.Println("Error setting key:", err)
    }
    
    if value, err := node1.Get("key1"); err == nil {
        fmt.Println("Retrieved value:", value)
    }
}
```

---

## 🔄 **3. Data Replication - Theory and Examples**

### **Theory: Why Replication?**

**Benefits:**
- **Availability**: System continues working if some nodes fail
- **Performance**: Read operations can be distributed
- **Scalability**: Handle more read requests

**Challenges:**
- **Consistency**: Keeping replicas in sync
- **Network partitions**: Handling split-brain scenarios
- **Write conflicts**: Resolving concurrent writes

### **Master-Slave Replication - Detailed Example**

#### **Theory: Master-Slave Pattern**

- **Master**: Handles all writes
- **Slaves**: Handle reads and replicate from master
- **Consistency**: Eventually consistent
- **Availability**: High read availability

#### **Practical Implementation: E-commerce Product Catalog**

```go
type ProductCatalog struct {
    master *ProductDatabase
    slaves []*ProductDatabase
    mutex  sync.RWMutex
}

type ProductDatabase struct {
    ID       string
    products map[string]*Product
    mutex    sync.RWMutex
    isMaster bool
}

type Product struct {
    ID          string
    Name        string
    Price       float64
    Stock       int
    LastUpdated time.Time
}

func NewProductCatalog() *ProductCatalog {
    master := &ProductDatabase{
        ID:       "master",
        products: make(map[string]*Product),
        isMaster: true,
    }
    
    slaves := []*ProductDatabase{
        {ID: "slave1", products: make(map[string]*Product), isMaster: false},
        {ID: "slave2", products: make(map[string]*Product), isMaster: false},
        {ID: "slave3", products: make(map[string]*Product), isMaster: false},
    }
    
    return &ProductCatalog{
        master: master,
        slaves: slaves,
    }
}

// Write operations go to master
func (pc *ProductCatalog) AddProduct(product *Product) error {
    pc.mutex.Lock()
    defer pc.mutex.Unlock()
    
    // Write to master
    pc.master.mutex.Lock()
    pc.master.products[product.ID] = product
    pc.master.mutex.Unlock()
    
    // Replicate to slaves asynchronously
    go pc.replicateToSlaves(product)
    
    return nil
}

func (pc *ProductCatalog) UpdateProduct(productID string, updates map[string]interface{}) error {
    pc.mutex.Lock()
    defer pc.mutex.Unlock()
    
    // Update master
    pc.master.mutex.Lock()
    if product, exists := pc.master.products[productID]; exists {
        // Apply updates
        if name, ok := updates["name"].(string); ok {
            product.Name = name
        }
        if price, ok := updates["price"].(float64); ok {
            product.Price = price
        }
        if stock, ok := updates["stock"].(int); ok {
            product.Stock = stock
        }
        product.LastUpdated = time.Now()
    }
    pc.master.mutex.Unlock()
    
    // Replicate to slaves
    go pc.replicateToSlaves(pc.master.products[productID])
    
    return nil
}

// Read operations can go to any slave
func (pc *ProductCatalog) GetProduct(productID string) (*Product, error) {
    // Try slaves first for better performance
    for _, slave := range pc.slaves {
        if pc.isSlaveHealthy(slave) {
            slave.mutex.RLock()
            if product, exists := slave.products[productID]; exists {
                slave.mutex.RUnlock()
                return product, nil
            }
            slave.mutex.RUnlock()
        }
    }
    
    // Fallback to master
    pc.master.mutex.RLock()
    defer pc.master.mutex.RUnlock()
    
    if product, exists := pc.master.products[productID]; exists {
        return product, nil
    }
    
    return nil, errors.New("product not found")
}

func (pc *ProductCatalog) SearchProducts(query string) ([]*Product, error) {
    var results []*Product
    
    // Use any healthy slave
    for _, slave := range pc.slaves {
        if pc.isSlaveHealthy(slave) {
            slave.mutex.RLock()
            for _, product := range slave.products {
                if strings.Contains(strings.ToLower(product.Name), strings.ToLower(query)) {
                    results = append(results, product)
                }
            }
            slave.mutex.RUnlock()
            break
        }
    }
    
    return results, nil
}

func (pc *ProductCatalog) replicateToSlaves(product *Product) {
    for _, slave := range pc.slaves {
        go func(s *ProductDatabase) {
            s.mutex.Lock()
            s.products[product.ID] = product
            s.mutex.Unlock()
        }(slave)
    }
}

func (pc *ProductCatalog) isSlaveHealthy(slave *ProductDatabase) bool {
    // In real implementation, this would check actual health
    return true
}

// Example usage
func main() {
    catalog := NewProductCatalog()
    
    // Add products
    product1 := &Product{
        ID:          "p1",
        Name:        "Laptop",
        Price:       999.99,
        Stock:       10,
        LastUpdated: time.Now(),
    }
    
    product2 := &Product{
        ID:          "p2",
        Name:        "Mouse",
        Price:       29.99,
        Stock:       100,
        LastUpdated: time.Now(),
    }
    
    catalog.AddProduct(product1)
    catalog.AddProduct(product2)
    
    // Search products
    results, err := catalog.SearchProducts("laptop")
    if err == nil {
        for _, product := range results {
            fmt.Printf("Found: %s - $%.2f\n", product.Name, product.Price)
        }
    }
    
    // Update product
    updates := map[string]interface{}{
        "price": 899.99,
        "stock": 5,
    }
    catalog.UpdateProduct("p1", updates)
}
```

### **Master-Master Replication - Detailed Example**

#### **Theory: Master-Master Pattern**

- **Multiple Masters**: Each can handle writes
- **Conflict Resolution**: Handle concurrent writes
- **Eventual Consistency**: All masters eventually agree

#### **Practical Implementation: Distributed User Profiles**

```go
type UserProfileSystem struct {
    masters []*UserDatabase
    conflictResolver *ConflictResolver
    mutex   sync.RWMutex
}

type UserDatabase struct {
    ID       string
    users    map[string]*User
    mutex    sync.RWMutex
    isMaster bool
}

type User struct {
    ID        string
    Username  string
    Email     string
    Profile   map[string]interface{}
    Version   int
    LastModified time.Time
    ModifiedBy   string
}

type ConflictResolver struct {
    strategies map[string]ConflictStrategy
}

type ConflictStrategy interface {
    Resolve(user1, user2 *User) *User
}

// Last-Write-Wins strategy
type LastWriteWinsStrategy struct{}

func (lww *LastWriteWinsStrategy) Resolve(user1, user2 *User) *User {
    if user1.LastModified.After(user2.LastModified) {
        return user1
    }
    return user2
}

// Field-level merge strategy
type FieldMergeStrategy struct{}

func (fm *FieldMergeStrategy) Resolve(user1, user2 *User) *User {
    merged := &User{
        ID:           user1.ID,
        Username:     user1.Username,
        Email:        user1.Email,
        Profile:      make(map[string]interface{}),
        Version:      max(user1.Version, user2.Version) + 1,
        LastModified: time.Now(),
        ModifiedBy:   "conflict_resolver",
    }
    
    // Merge profiles field by field
    for key, value := range user1.Profile {
        merged.Profile[key] = value
    }
    for key, value := range user2.Profile {
        if _, exists := merged.Profile[key]; !exists {
            merged.Profile[key] = value
        }
    }
    
    return merged
}

func NewUserProfileSystem() *UserProfileSystem {
    masters := []*UserDatabase{
        {ID: "master1", users: make(map[string]*User), isMaster: true},
        {ID: "master2", users: make(map[string]*User), isMaster: true},
        {ID: "master3", users: make(map[string]*User), isMaster: true},
    }
    
    conflictResolver := &ConflictResolver{
        strategies: map[string]ConflictStrategy{
            "last_write_wins": &LastWriteWinsStrategy{},
            "field_merge":     &FieldMergeStrategy{},
        },
    }
    
    return &UserProfileSystem{
        masters:         masters,
        conflictResolver: conflictResolver,
    }
}

func (ups *UserProfileSystem) UpdateUser(user *User) error {
    ups.mutex.Lock()
    defer ups.mutex.Unlock()
    
    // Update local master
    localMaster := ups.masters[0] // Assume this is the local master
    localMaster.mutex.Lock()
    localMaster.users[user.ID] = user
    localMaster.mutex.Unlock()
    
    // Replicate to other masters
    go ups.replicateToOtherMasters(user)
    
    return nil
}

func (ups *UserProfileSystem) GetUser(userID string) (*User, error) {
    // Try to get from any master
    for _, master := range ups.masters {
        master.mutex.RLock()
        if user, exists := master.users[userID]; exists {
            master.mutex.RUnlock()
            return user, nil
        }
        master.mutex.RUnlock()
    }
    
    return nil, errors.New("user not found")
}

func (ups *UserProfileSystem) replicateToOtherMasters(user *User) {
    for i := 1; i < len(ups.masters); i++ {
        go func(master *UserDatabase) {
            master.mutex.Lock()
            defer master.mutex.Unlock()
            
            if existingUser, exists := master.users[user.ID]; exists {
                // Check for conflicts
                if existingUser.Version != user.Version {
                    // Resolve conflict
                    resolvedUser := ups.conflictResolver.strategies["last_write_wins"].Resolve(existingUser, user)
                    master.users[user.ID] = resolvedUser
                } else {
                    master.users[user.ID] = user
                }
            } else {
                master.users[user.ID] = user
            }
        }(ups.masters[i])
    }
}

// Example usage
func main() {
    system := NewUserProfileSystem()
    
    // Create user
    user := &User{
        ID:       "u1",
        Username: "john_doe",
        Email:    "john@example.com",
        Profile: map[string]interface{}{
            "age":     30,
            "city":    "New York",
            "bio":     "Software engineer",
        },
        Version:      1,
        LastModified: time.Now(),
        ModifiedBy:   "user",
    }
    
    // Update user
    system.UpdateUser(user)
    
    // Retrieve user
    retrievedUser, err := system.GetUser("u1")
    if err == nil {
        fmt.Printf("User: %s, Email: %s\n", retrievedUser.Username, retrievedUser.Email)
    }
}
```

---

## 🔀 **4. Sharding Strategies - Theory and Examples**

### **Theory: What is Sharding?**

Sharding is the process of splitting a large database into smaller, more manageable pieces called shards.

**Benefits:**
- **Scalability**: Handle more data and requests
- **Performance**: Faster queries on smaller datasets
- **Fault Isolation**: Failure of one shard doesn't affect others

**Challenges:**
- **Cross-shard queries**: Difficult to query across shards
- **Data rebalancing**: Moving data when adding/removing shards
- **Transaction consistency**: Maintaining ACID across shards

### **Range-Based Sharding - Detailed Example**

#### **Theory: Range-Based Sharding**

Data is partitioned based on a range of values (e.g., user IDs 1-1000 go to shard 1, 1001-2000 go to shard 2).

**Advantages:**
- **Range queries**: Efficient for queries like "users between 1000-2000"
- **Sequential access**: Good for pagination
- **Simple implementation**: Easy to understand and implement

**Disadvantages:**
- **Hot spots**: Some ranges might be more popular
- **Rebalancing**: Difficult to rebalance data

#### **Practical Implementation: E-commerce Order System**

```go
type OrderShardingSystem struct {
    shards []OrderShard
    ranges []ShardRange
}

type OrderShard struct {
    ID       string
    orders   map[string]*Order
    mutex    sync.RWMutex
    database *sql.DB
}

type Order struct {
    ID        string
    UserID    int
    ProductID string
    Amount    float64
    Status    string
    CreatedAt time.Time
}

type ShardRange struct {
    Start   int
    End     int
    ShardID string
}

func NewOrderShardingSystem() *OrderShardingSystem {
    // Define shard ranges based on user ID
    ranges := []ShardRange{
        {Start: 1, End: 10000, ShardID: "shard1"},
        {Start: 10001, End: 20000, ShardID: "shard2"},
        {Start: 20001, End: 30000, ShardID: "shard3"},
    }
    
    shards := make([]OrderShard, len(ranges))
    for i, r := range ranges {
        shards[i] = OrderShard{
            ID:     r.ShardID,
            orders: make(map[string]*Order),
        }
    }
    
    return &OrderShardingSystem{
        shards: shards,
        ranges: ranges,
    }
}

func (oss *OrderShardingSystem) CreateOrder(order *Order) error {
    shard := oss.getShardForUser(order.UserID)
    if shard == nil {
        return errors.New("no shard found for user")
    }
    
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    shard.orders[order.ID] = order
    
    // In real implementation, also write to database
    return oss.writeToDatabase(shard, order)
}

func (oss *OrderShardingSystem) GetOrder(orderID string, userID int) (*Order, error) {
    shard := oss.getShardForUser(userID)
    if shard == nil {
        return nil, errors.New("no shard found for user")
    }
    
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()
    
    if order, exists := shard.orders[orderID]; exists {
        return order, nil
    }
    
    return nil, errors.New("order not found")
}

func (oss *OrderShardingSystem) GetOrdersByUser(userID int) ([]*Order, error) {
    shard := oss.getShardForUser(userID)
    if shard == nil {
        return nil, errors.New("no shard found for user")
    }
    
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()
    
    var orders []*Order
    for _, order := range shard.orders {
        if order.UserID == userID {
            orders = append(orders, order)
        }
    }
    
    return orders, nil
}

func (oss *OrderShardingSystem) getShardForUser(userID int) *OrderShard {
    for i, r := range oss.ranges {
        if userID >= r.Start && userID <= r.End {
            return &oss.shards[i]
        }
    }
    return nil
}

func (oss *OrderShardingSystem) writeToDatabase(shard *OrderShard, order *Order) error {
    // In real implementation, this would write to the actual database
    fmt.Printf("Writing order %s to shard %s\n", order.ID, shard.ID)
    return nil
}

// Example usage
func main() {
    system := NewOrderShardingSystem()
    
    // Create orders for different users
    order1 := &Order{
        ID:        "o1",
        UserID:    5000,  // Goes to shard1
        ProductID: "p1",
        Amount:    99.99,
        Status:    "pending",
        CreatedAt: time.Now(),
    }
    
    order2 := &Order{
        ID:        "o2",
        UserID:    15000, // Goes to shard2
        ProductID: "p2",
        Amount:    199.99,
        Status:    "pending",
        CreatedAt: time.Now(),
    }
    
    system.CreateOrder(order1)
    system.CreateOrder(order2)
    
    // Retrieve orders
    if order, err := system.GetOrder("o1", 5000); err == nil {
        fmt.Printf("Order: %s, Amount: $%.2f\n", order.ID, order.Amount)
    }
}
```

### **Hash-Based Sharding - Detailed Example**

#### **Theory: Hash-Based Sharding**

Data is partitioned based on a hash function applied to a key.

**Advantages:**
- **Even distribution**: Hash function distributes data evenly
- **No hot spots**: All shards get similar load
- **Simple routing**: Easy to determine which shard to use

**Disadvantages:**
- **No range queries**: Cannot query ranges efficiently
- **Rebalancing**: Adding/removing shards requires data movement

#### **Practical Implementation: Distributed Cache System**

```go
type DistributedCache struct {
    shards []CacheShard
    hashFunc func(string) int
    shardCount int
}

type CacheShard struct {
    ID       string
    data     map[string]*CacheEntry
    mutex    sync.RWMutex
    maxSize  int
    evictionPolicy string
}

type CacheEntry struct {
    Key       string
    Value     interface{}
    ExpiresAt time.Time
    CreatedAt time.Time
    AccessCount int
}

func NewDistributedCache(shardCount int) *DistributedCache {
    shards := make([]CacheShard, shardCount)
    for i := 0; i < shardCount; i++ {
        shards[i] = CacheShard{
            ID:       fmt.Sprintf("shard_%d", i),
            data:     make(map[string]*CacheEntry),
            maxSize:  1000,
            evictionPolicy: "LRU",
        }
    }
    
    return &DistributedCache{
        shards:     shards,
        hashFunc:   func(key string) int {
            hash := 0
            for _, c := range key {
                hash += int(c)
            }
            return hash % shardCount
        },
        shardCount: shardCount,
    }
}

func (dc *DistributedCache) Set(key string, value interface{}, ttl time.Duration) error {
    shard := dc.getShardForKey(key)
    
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    // Check if shard is full
    if len(shard.data) >= shard.maxSize {
        dc.evictEntry(shard)
    }
    
    entry := &CacheEntry{
        Key:        key,
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        CreatedAt:  time.Now(),
        AccessCount: 1,
    }
    
    shard.data[key] = entry
    return nil
}

func (dc *DistributedCache) Get(key string) (interface{}, error) {
    shard := dc.getShardForKey(key)
    
    shard.mutex.RLock()
    entry, exists := shard.data[key]
    shard.mutex.RUnlock()
    
    if !exists {
        return nil, errors.New("key not found")
    }
    
    // Check if expired
    if time.Now().After(entry.ExpiresAt) {
        shard.mutex.Lock()
        delete(shard.data, key)
        shard.mutex.Unlock()
        return nil, errors.New("key expired")
    }
    
    // Update access count
    shard.mutex.Lock()
    entry.AccessCount++
    shard.mutex.Unlock()
    
    return entry.Value, nil
}

func (dc *DistributedCache) Delete(key string) error {
    shard := dc.getShardForKey(key)
    
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    delete(shard.data, key)
    return nil
}

func (dc *DistributedCache) getShardForKey(key string) *CacheShard {
    shardIndex := dc.hashFunc(key)
    return &dc.shards[shardIndex]
}

func (dc *DistributedCache) evictEntry(shard *CacheShard) {
    // LRU eviction
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range shard.data {
        if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.CreatedAt
        }
    }
    
    if oldestKey != "" {
        delete(shard.data, oldestKey)
    }
}

// Example usage
func main() {
    cache := NewDistributedCache(3)
    
    // Set some values
    cache.Set("user:1", "John Doe", 5*time.Minute)
    cache.Set("user:2", "Jane Smith", 5*time.Minute)
    cache.Set("product:1", "Laptop", 10*time.Minute)
    
    // Get values
    if value, err := cache.Get("user:1"); err == nil {
        fmt.Printf("User 1: %s\n", value)
    }
    
    if value, err := cache.Get("product:1"); err == nil {
        fmt.Printf("Product 1: %s\n", value)
    }
}
```

---

## 🎯 **5. Eventual Consistency - Theory and Examples**

### **Theory: What is Eventual Consistency?**

Eventual consistency is a consistency model where the system will eventually become consistent, but there's no guarantee about when.

**Use Cases:**
- **Social media feeds**: Slight delays in seeing new posts are acceptable
- **Product catalogs**: Minor inconsistencies in product information are tolerable
- **Analytics systems**: Approximate data is often sufficient

### **Vector Clocks - Detailed Example**

#### **Theory: Vector Clocks**

Vector clocks are used to track causality in distributed systems. Each node maintains a vector of logical clocks.

#### **Practical Implementation: Distributed Chat System**

```go
type ChatSystem struct {
    nodes map[string]*ChatNode
    mutex sync.RWMutex
}

type ChatNode struct {
    ID       string
    messages []*ChatMessage
    vectorClock map[string]int
    mutex    sync.RWMutex
}

type ChatMessage struct {
    ID        string
    Content   string
    Sender    string
    Timestamp time.Time
    VectorClock map[string]int
}

func NewChatSystem() *ChatSystem {
    return &ChatSystem{
        nodes: make(map[string]*ChatNode),
    }
}

func (cs *ChatSystem) AddNode(nodeID string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    cs.nodes[nodeID] = &ChatNode{
        ID:          nodeID,
        messages:    make([]*ChatMessage, 0),
        vectorClock: make(map[string]int),
    }
}

func (cs *ChatSystem) SendMessage(senderID, content string) error {
    cs.mutex.RLock()
    sender, exists := cs.nodes[senderID]
    cs.mutex.RUnlock()
    
    if !exists {
        return errors.New("sender not found")
    }
    
    sender.mutex.Lock()
    defer sender.mutex.Unlock()
    
    // Increment sender's clock
    sender.vectorClock[senderID]++
    
    // Create message
    message := &ChatMessage{
        ID:          generateMessageID(),
        Content:     content,
        Sender:      senderID,
        Timestamp:   time.Now(),
        VectorClock: make(map[string]int),
    }
    
    // Copy vector clock
    for nodeID, clock := range sender.vectorClock {
        message.VectorClock[nodeID] = clock
    }
    
    // Add to sender's messages
    sender.messages = append(sender.messages, message)
    
    // Replicate to other nodes
    go cs.replicateMessage(message)
    
    return nil
}

func (cs *ChatSystem) replicateMessage(message *ChatMessage) {
    cs.mutex.RLock()
    nodes := make([]*ChatNode, 0, len(cs.nodes))
    for _, node := range cs.nodes {
        nodes = append(nodes, node)
    }
    cs.mutex.RUnlock()
    
    for _, node := range nodes {
        if node.ID != message.Sender {
            go func(n *ChatNode) {
                n.mutex.Lock()
                defer n.mutex.Unlock()
                
                // Update vector clock
                for nodeID, clock := range message.VectorClock {
                    if n.vectorClock[nodeID] < clock {
                        n.vectorClock[nodeID] = clock
                    }
                }
                
                // Add message
                n.messages = append(n.messages, message)
            }(node)
        }
    }
}

func (cs *ChatSystem) GetMessages(nodeID string) ([]*ChatMessage, error) {
    cs.mutex.RLock()
    node, exists := cs.nodes[nodeID]
    cs.mutex.RUnlock()
    
    if !exists {
        return nil, errors.New("node not found")
    }
    
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    // Return a copy of messages
    messages := make([]*ChatMessage, len(node.messages))
    copy(messages, node.messages)
    
    return messages, nil
}

// Example usage
func main() {
    chat := NewChatSystem()
    
    // Add nodes
    chat.AddNode("alice")
    chat.AddNode("bob")
    chat.AddNode("charlie")
    
    // Send messages
    chat.SendMessage("alice", "Hello everyone!")
    chat.SendMessage("bob", "Hi Alice!")
    chat.SendMessage("charlie", "Hey guys!")
    
    // Wait for replication
    time.Sleep(100 * time.Millisecond)
    
    // Get messages for each node
    for _, nodeID := range []string{"alice", "bob", "charlie"} {
        messages, err := chat.GetMessages(nodeID)
        if err == nil {
            fmt.Printf("\nMessages for %s:\n", nodeID)
            for _, msg := range messages {
                fmt.Printf("  %s: %s\n", msg.Sender, msg.Content)
            }
        }
    }
}
```

---

## 🎯 **Key Takeaways**

### **1. CAP Theorem Trade-offs**
- **CP Systems**: Consistency + Partition Tolerance (e.g., MongoDB, PostgreSQL)
- **AP Systems**: Availability + Partition Tolerance (e.g., Cassandra, DynamoDB)
- **CA Systems**: Consistency + Availability (e.g., Single-node databases)

### **2. Consensus Algorithms**
- **Raft**: Easier to understand, leader-based, good for most use cases
- **Paxos**: More complex, but more flexible, used in Google's systems
- **Both ensure**: Safety and liveness properties

### **3. Replication Strategies**
- **Master-Slave**: Read scaling, eventual consistency, good for read-heavy workloads
- **Master-Master**: Write scaling, conflict resolution needed, good for write-heavy workloads
- **Choose based on**: Read/write patterns and consistency requirements

### **4. Sharding Approaches**
- **Range-based**: Good for sequential access, range queries
- **Hash-based**: Even distribution, no range queries
- **Directory-based**: Flexible, but single point of failure

### **5. Eventual Consistency**
- **Vector clocks**: Track causality in distributed systems
- **CRDTs**: Conflict-free replicated data types
- **Acceptable for**: Many real-world applications where slight delays are tolerable

---

**🎉 This comprehensive guide provides deep understanding of distributed systems concepts with practical examples and implementations! 🚀**
