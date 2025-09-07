# 🏗️ System Design Concepts - Complete Guide

> **Comprehensive guide to system design with Go implementations and FAANG interview questions**

## 📋 Table of Contents

1. [System Design Fundamentals](#system-design-fundamentals)
2. [Load Balancing](#load-balancing)
3. [Caching Strategies](#caching-strategies)
4. [Database Design](#database-design)
5. [Microservices Architecture](#microservices-architecture)
6. [Message Queues](#message-queues)
7. [API Design](#api-design)
8. [Security & Authentication](#security--authentication)
9. [FAANG Interview Questions](#faang-interview-questions)

---

## 🎯 System Design Fundamentals

### **1. Scalability Patterns**

#### **Horizontal vs Vertical Scaling**
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Vertical Scaling - Single powerful machine
type VerticalScaler struct {
    capacity int
    currentLoad int
    mutex sync.Mutex
}

func NewVerticalScaler(capacity int) *VerticalScaler {
    return &VerticalScaler{capacity: capacity}
}

func (vs *VerticalScaler) ProcessRequest() bool {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()
    
    if vs.currentLoad < vs.capacity {
        vs.currentLoad++
        time.Sleep(100 * time.Millisecond) // Simulate processing
        vs.currentLoad--
        return true
    }
    return false
}

// Horizontal Scaling - Multiple machines
type HorizontalScaler struct {
    servers []*Server
    loadBalancer *LoadBalancer
}

type Server struct {
    ID int
    capacity int
    currentLoad int
    mutex sync.Mutex
}

func NewServer(id, capacity int) *Server {
    return &Server{ID: id, capacity: capacity}
}

func (s *Server) ProcessRequest() bool {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if s.currentLoad < s.capacity {
        s.currentLoad++
        time.Sleep(100 * time.Millisecond) // Simulate processing
        s.currentLoad--
        return true
    }
    return false
}

type LoadBalancer struct {
    servers []*Server
    current int
    mutex sync.Mutex
}

func NewLoadBalancer(servers []*Server) *LoadBalancer {
    return &LoadBalancer{servers: servers}
}

func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    server := lb.servers[lb.current]
    lb.current = (lb.current + 1) % len(lb.servers)
    return server
}

func (hs *HorizontalScaler) ProcessRequest() bool {
    server := hs.loadBalancer.GetServer()
    return server.ProcessRequest()
}

func main() {
    // Vertical scaling example
    vs := NewVerticalScaler(10)
    fmt.Println("Vertical Scaling:")
    for i := 0; i < 15; i++ {
        if vs.ProcessRequest() {
            fmt.Printf("Request %d processed\n", i)
        } else {
            fmt.Printf("Request %d rejected (capacity exceeded)\n", i)
        }
    }
    
    // Horizontal scaling example
    servers := []*Server{
        NewServer(1, 5),
        NewServer(2, 5),
        NewServer(3, 5),
    }
    lb := NewLoadBalancer(servers)
    hs := &HorizontalScaler{servers: servers, loadBalancer: lb}
    
    fmt.Println("\nHorizontal Scaling:")
    for i := 0; i < 15; i++ {
        if hs.ProcessRequest() {
            fmt.Printf("Request %d processed\n", i)
        } else {
            fmt.Printf("Request %d rejected (capacity exceeded)\n", i)
        }
    }
}
```

### **2. CAP Theorem Implementation**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Consistency, Availability, Partition Tolerance
type CAPSystem struct {
    nodes map[string]*Node
    consistencyLevel string
    mutex sync.RWMutex
}

type Node struct {
    ID string
    Data map[string]string
    mutex sync.RWMutex
}

func NewCAPSystem(consistencyLevel string) *CAPSystem {
    return &CAPSystem{
        nodes: make(map[string]*Node),
        consistencyLevel: consistencyLevel,
    }
}

func (cs *CAPSystem) AddNode(id string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    cs.nodes[id] = &Node{
        ID: id,
        Data: make(map[string]string),
    }
}

func (cs *CAPSystem) Write(key, value string) error {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    switch cs.consistencyLevel {
    case "strong":
        return cs.strongConsistencyWrite(key, value)
    case "eventual":
        return cs.eventualConsistencyWrite(key, value)
    default:
        return fmt.Errorf("unknown consistency level")
    }
}

func (cs *CAPSystem) strongConsistencyWrite(key, value string) error {
    // Write to all nodes synchronously
    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
    }
    return nil
}

func (cs *CAPSystem) eventualConsistencyWrite(key, value string) error {
    // Write to one node, replicate asynchronously
    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
        break // Write to first available node
    }
    
    // Replicate asynchronously
    go cs.replicateAsync(key, value)
    return nil
}

func (cs *CAPSystem) replicateAsync(key, value string) {
    time.Sleep(100 * time.Millisecond) // Simulate network delay
    
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
    }
}

func (cs *CAPSystem) Read(key string) (string, error) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    switch cs.consistencyLevel {
    case "strong":
        return cs.strongConsistencyRead(key)
    case "eventual":
        return cs.eventualConsistencyRead(key)
    default:
        return "", fmt.Errorf("unknown consistency level")
    }
}

func (cs *CAPSystem) strongConsistencyRead(key string) (string, error) {
    // Read from all nodes and ensure consistency
    var values []string
    for _, node := range cs.nodes {
        node.mutex.RLock()
        if value, exists := node.Data[key]; exists {
            values = append(values, value)
        }
        node.mutex.RUnlock()
    }
    
    if len(values) == 0 {
        return "", fmt.Errorf("key not found")
    }
    
    // Check if all values are the same
    for _, value := range values {
        if value != values[0] {
            return "", fmt.Errorf("inconsistent data")
        }
    }
    
    return values[0], nil
}

func (cs *CAPSystem) eventualConsistencyRead(key string) (string, error) {
    // Read from any available node
    for _, node := range cs.nodes {
        node.mutex.RLock()
        if value, exists := node.Data[key]; exists {
            node.mutex.RUnlock()
            return value, nil
        }
        node.mutex.RUnlock()
    }
    
    return "", fmt.Errorf("key not found")
}

func main() {
    // Strong consistency system
    strongSystem := NewCAPSystem("strong")
    strongSystem.AddNode("node1")
    strongSystem.AddNode("node2")
    strongSystem.AddNode("node3")
    
    strongSystem.Write("key1", "value1")
    value, _ := strongSystem.Read("key1")
    fmt.Printf("Strong consistency read: %s\n", value)
    
    // Eventual consistency system
    eventualSystem := NewCAPSystem("eventual")
    eventualSystem.AddNode("node1")
    eventualSystem.AddNode("node2")
    eventualSystem.AddNode("node3")
    
    eventualSystem.Write("key1", "value1")
    value, _ = eventualSystem.Read("key1")
    fmt.Printf("Eventual consistency read: %s\n", value)
}
```

---

## ⚖️ Load Balancing

### **3. Load Balancing Algorithms**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type LoadBalancer interface {
    GetServer() *Server
}

type Server struct {
    ID string
    Weight int
    CurrentConnections int
    mutex sync.Mutex
}

func (s *Server) ProcessRequest() {
    s.mutex.Lock()
    s.CurrentConnections++
    s.mutex.Unlock()
    
    time.Sleep(100 * time.Millisecond) // Simulate processing
    
    s.mutex.Lock()
    s.CurrentConnections--
    s.mutex.Unlock()
}

// Round Robin Load Balancer
type RoundRobinLB struct {
    servers []*Server
    current int
    mutex sync.Mutex
}

func NewRoundRobinLB(servers []*Server) *RoundRobinLB {
    return &RoundRobinLB{servers: servers}
}

func (rr *RoundRobinLB) GetServer() *Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    server := rr.servers[rr.current]
    rr.current = (rr.current + 1) % len(rr.servers)
    return server
}

// Weighted Round Robin Load Balancer
type WeightedRoundRobinLB struct {
    servers []*Server
    current int
    currentWeight int
    mutex sync.Mutex
}

func NewWeightedRoundRobinLB(servers []*Server) *WeightedRoundRobinLB {
    return &WeightedRoundRobinLB{servers: servers}
}

func (wrr *WeightedRoundRobinLB) GetServer() *Server {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()
    
    for {
        wrr.current = (wrr.current + 1) % len(wrr.servers)
        if wrr.current == 0 {
            wrr.currentWeight -= wrr.gcd()
            if wrr.currentWeight <= 0 {
                wrr.currentWeight = wrr.maxWeight()
            }
        }
        
        if wrr.servers[wrr.current].Weight >= wrr.currentWeight {
            return wrr.servers[wrr.current]
        }
    }
}

func (wrr *WeightedRoundRobinLB) gcd() int {
    weights := make([]int, len(wrr.servers))
    for i, server := range wrr.servers {
        weights[i] = server.Weight
    }
    
    result := weights[0]
    for i := 1; i < len(weights); i++ {
        result = gcd(result, weights[i])
    }
    return result
}

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func (wrr *WeightedRoundRobinLB) maxWeight() int {
    max := wrr.servers[0].Weight
    for _, server := range wrr.servers {
        if server.Weight > max {
            max = server.Weight
        }
    }
    return max
}

// Least Connections Load Balancer
type LeastConnectionsLB struct {
    servers []*Server
    mutex sync.Mutex
}

func NewLeastConnectionsLB(servers []*Server) *LeastConnectionsLB {
    return &LeastConnectionsLB{servers: servers}
}

func (lc *LeastConnectionsLB) GetServer() *Server {
    lc.mutex.Lock()
    defer lc.mutex.Unlock()
    
    minConnections := lc.servers[0].CurrentConnections
    selectedServer := lc.servers[0]
    
    for _, server := range lc.servers {
        if server.CurrentConnections < minConnections {
            minConnections = server.CurrentConnections
            selectedServer = server
        }
    }
    
    return selectedServer
}

func main() {
    servers := []*Server{
        {ID: "server1", Weight: 3},
        {ID: "server2", Weight: 2},
        {ID: "server3", Weight: 1},
    }
    
    // Test Round Robin
    fmt.Println("Round Robin Load Balancing:")
    rrLB := NewRoundRobinLB(servers)
    for i := 0; i < 6; i++ {
        server := rrLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
    }
    
    // Test Weighted Round Robin
    fmt.Println("\nWeighted Round Robin Load Balancing:")
    wrrLB := NewWeightedRoundRobinLB(servers)
    for i := 0; i < 6; i++ {
        server := wrrLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
    }
    
    // Test Least Connections
    fmt.Println("\nLeast Connections Load Balancing:")
    lcLB := NewLeastConnectionsLB(servers)
    for i := 0; i < 6; i++ {
        server := lcLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
        server.ProcessRequest()
    }
}
```

---

## 💾 Caching Strategies

### **4. Cache Implementation**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheItem struct {
    Key string
    Value interface{}
    Expiry time.Time
    AccessCount int
    LastAccessed time.Time
}

type Cache struct {
    items map[string]*CacheItem
    mutex sync.RWMutex
    maxSize int
    ttl time.Duration
}

func NewCache(maxSize int, ttl time.Duration) *Cache {
    cache := &Cache{
        items: make(map[string]*CacheItem),
        maxSize: maxSize,
        ttl: ttl,
    }
    
    // Start cleanup goroutine
    go cache.cleanup()
    
    return cache
}

func (c *Cache) Set(key string, value interface{}) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // Check if cache is full
    if len(c.items) >= c.maxSize {
        c.evictLRU()
    }
    
    c.items[key] = &CacheItem{
        Key: key,
        Value: value,
        Expiry: time.Now().Add(c.ttl),
        AccessCount: 1,
        LastAccessed: time.Now(),
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    item, exists := c.items[key]
    if !exists {
        return nil, false
    }
    
    // Check if expired
    if time.Now().After(item.Expiry) {
        delete(c.items, key)
        return nil, false
    }
    
    // Update access information
    item.AccessCount++
    item.LastAccessed = time.Now()
    
    return item.Value, true
}

func (c *Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range c.items {
        if oldestKey == "" || item.LastAccessed.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccessed
        }
    }
    
    if oldestKey != "" {
        delete(c.items, oldestKey)
    }
}

func (c *Cache) cleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        c.mutex.Lock()
        now := time.Now()
        for key, item := range c.items {
            if now.After(item.Expiry) {
                delete(c.items, key)
            }
        }
        c.mutex.Unlock()
    }
}

func (c *Cache) Size() int {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    return len(c.items)
}

func main() {
    cache := NewCache(3, 5*time.Second)
    
    // Set some values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")
    
    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("key1: %v\n", value)
    }
    
    // Test LRU eviction
    cache.Set("key4", "value4") // This should evict key1
    
    if _, exists := cache.Get("key1"); !exists {
        fmt.Println("key1 was evicted (LRU)")
    }
    
    // Test expiration
    time.Sleep(6 * time.Second)
    if _, exists := cache.Get("key2"); !exists {
        fmt.Println("key2 expired")
    }
    
    fmt.Printf("Cache size: %d\n", cache.Size())
}
```

---

## 🗄️ Database Design

### **5. Database Sharding**

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
)

type Shard struct {
    ID string
    Data map[string]interface{}
    mutex sync.RWMutex
}

type ShardManager struct {
    shards []*Shard
    hashRing []string
    mutex sync.RWMutex
}

func NewShardManager(shardCount int) *ShardManager {
    sm := &ShardManager{
        shards: make([]*Shard, shardCount),
        hashRing: make([]string, 0),
    }
    
    for i := 0; i < shardCount; i++ {
        shardID := fmt.Sprintf("shard%d", i)
        sm.shards[i] = &Shard{
            ID: shardID,
            Data: make(map[string]interface{}),
        }
        sm.hashRing = append(sm.hashRing, shardID)
    }
    
    sort.Strings(sm.hashRing)
    return sm
}

func (sm *ShardManager) GetShard(key string) *Shard {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    hash := sm.hash(key)
    
    // Find the first shard with hash >= key hash
    for _, shardID := range sm.hashRing {
        shardHash := sm.hash(shardID)
        if shardHash >= hash {
            for _, shard := range sm.shards {
                if shard.ID == shardID {
                    return shard
                }
            }
        }
    }
    
    // If no shard found, return the first one
    return sm.shards[0]
}

func (sm *ShardManager) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (sm *ShardManager) Set(key string, value interface{}) {
    shard := sm.GetShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    shard.Data[key] = value
    fmt.Printf("Set %s = %v in %s\n", key, value, shard.ID)
}

func (sm *ShardManager) Get(key string) (interface{}, bool) {
    shard := sm.GetShard(key)
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()
    
    value, exists := shard.Data[key]
    if exists {
        fmt.Printf("Get %s = %v from %s\n", key, value, shard.ID)
    }
    return value, exists
}

func (sm *ShardManager) Delete(key string) {
    shard := sm.GetShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    delete(shard.Data, key)
    fmt.Printf("Deleted %s from %s\n", key, shard.ID)
}

func main() {
    sm := NewShardManager(3)
    
    // Set some data
    sm.Set("user1", "John Doe")
    sm.Set("user2", "Jane Smith")
    sm.Set("user3", "Bob Johnson")
    sm.Set("user4", "Alice Brown")
    
    // Get data
    if value, exists := sm.Get("user1"); exists {
        fmt.Printf("Retrieved: %v\n", value)
    }
    
    // Delete data
    sm.Delete("user2")
    
    // Try to get deleted data
    if _, exists := sm.Get("user2"); !exists {
        fmt.Println("user2 not found (deleted)")
    }
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Design a URL Shortener**
**Question**: "Design a URL shortener service like bit.ly that can handle billions of URLs."

**Answer**:
```go
package main

import (
    "crypto/md5"
    "fmt"
    "sync"
    "time"
)

type URLShortener struct {
    urls map[string]string
    shortToLong map[string]string
    mutex sync.RWMutex
    counter int64
}

func NewURLShortener() *URLShortener {
    return &URLShortener{
        urls: make(map[string]string),
        shortToLong: make(map[string]string),
    }
}

func (us *URLShortener) ShortenURL(longURL string) string {
    us.mutex.Lock()
    defer us.mutex.Unlock()
    
    // Check if URL already exists
    if shortURL, exists := us.urls[longURL]; exists {
        return shortURL
    }
    
    // Generate short URL
    us.counter++
    shortURL := us.encodeBase62(us.counter)
    
    // Store mapping
    us.urls[longURL] = shortURL
    us.shortToLong[shortURL] = longURL
    
    return shortURL
}

func (us *URLShortener) ExpandURL(shortURL string) (string, bool) {
    us.mutex.RLock()
    defer us.mutex.RUnlock()
    
    longURL, exists := us.shortToLong[shortURL]
    return longURL, exists
}

func (us *URLShortener) encodeBase62(num int64) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := ""
    
    for num > 0 {
        result = string(charset[num%62]) + result
        num /= 62
    }
    
    return result
}

func main() {
    shortener := NewURLShortener()
    
    // Shorten URLs
    short1 := shortener.ShortenURL("https://www.google.com")
    short2 := shortener.ShortenURL("https://www.github.com")
    short3 := shortener.ShortenURL("https://www.stackoverflow.com")
    
    fmt.Printf("Shortened URLs:\n")
    fmt.Printf("https://www.google.com -> %s\n", short1)
    fmt.Printf("https://www.github.com -> %s\n", short2)
    fmt.Printf("https://www.stackoverflow.com -> %s\n", short3)
    
    // Expand URLs
    if longURL, exists := shortener.ExpandURL(short1); exists {
        fmt.Printf("Expanded %s -> %s\n", short1, longURL)
    }
}
```

### **Meta Interview Questions**

#### **2. Design a News Feed System**
**Question**: "Design a news feed system similar to Facebook's that can handle millions of users."

**Answer**:
```go
package main

import (
    "fmt"
    "sort"
    "sync"
    "time"
)

type Post struct {
    ID string
    UserID string
    Content string
    Timestamp time.Time
    Likes int
    Comments int
}

type User struct {
    ID string
    Friends []string
    Posts []string
}

type NewsFeedService struct {
    users map[string]*User
    posts map[string]*Post
    feeds map[string][]*Post
    mutex sync.RWMutex
}

func NewNewsFeedService() *NewsFeedService {
    return &NewsFeedService{
        users: make(map[string]*User),
        posts: make(map[string]*Post),
        feeds: make(map[string][]*Post),
    }
}

func (nfs *NewsFeedService) AddUser(userID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()
    
    nfs.users[userID] = &User{
        ID: userID,
        Friends: make([]string, 0),
        Posts: make([]string, 0),
    }
}

func (nfs *NewsFeedService) AddFriend(userID, friendID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()
    
    if user, exists := nfs.users[userID]; exists {
        user.Friends = append(user.Friends, friendID)
    }
}

func (nfs *NewsFeedService) CreatePost(userID, content string) string {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()
    
    postID := fmt.Sprintf("post_%d", time.Now().UnixNano())
    post := &Post{
        ID: postID,
        UserID: userID,
        Content: content,
        Timestamp: time.Now(),
        Likes: 0,
        Comments: 0,
    }
    
    nfs.posts[postID] = post
    
    if user, exists := nfs.users[userID]; exists {
        user.Posts = append(user.Posts, postID)
    }
    
    // Update feeds of friends
    nfs.updateFeeds(userID, post)
    
    return postID
}

func (nfs *NewsFeedService) updateFeeds(userID string, post *Post) {
    if user, exists := nfs.users[userID]; exists {
        for _, friendID := range user.Friends {
            nfs.feeds[friendID] = append(nfs.feeds[friendID], post)
        }
    }
}

func (nfs *NewsFeedService) GetNewsFeed(userID string, limit int) []*Post {
    nfs.mutex.RLock()
    defer nfs.mutex.RUnlock()
    
    feed, exists := nfs.feeds[userID]
    if !exists {
        return []*Post{}
    }
    
    // Sort by timestamp (newest first)
    sortedFeed := make([]*Post, len(feed))
    copy(sortedFeed, feed)
    sort.Slice(sortedFeed, func(i, j int) bool {
        return sortedFeed[i].Timestamp.After(sortedFeed[j].Timestamp)
    })
    
    // Return limited results
    if len(sortedFeed) > limit {
        return sortedFeed[:limit]
    }
    
    return sortedFeed
}

func (nfs *NewsFeedService) LikePost(postID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()
    
    if post, exists := nfs.posts[postID]; exists {
        post.Likes++
    }
}

func main() {
    nfs := NewNewsFeedService()
    
    // Add users
    nfs.AddUser("user1")
    nfs.AddUser("user2")
    nfs.AddUser("user3")
    
    // Add friendships
    nfs.AddFriend("user1", "user2")
    nfs.AddFriend("user1", "user3")
    nfs.AddFriend("user2", "user1")
    nfs.AddFriend("user3", "user1")
    
    // Create posts
    nfs.CreatePost("user1", "Hello, world!")
    nfs.CreatePost("user2", "Go is awesome!")
    nfs.CreatePost("user3", "Learning system design")
    
    // Get news feed for user1
    feed := nfs.GetNewsFeed("user1", 10)
    fmt.Println("News Feed for user1:")
    for _, post := range feed {
        fmt.Printf("- %s: %s (Likes: %d)\n", post.UserID, post.Content, post.Likes)
    }
    
    // Like a post
    nfs.LikePost("post_1")
    
    // Get updated feed
    feed = nfs.GetNewsFeed("user1", 10)
    fmt.Println("\nUpdated News Feed for user1:")
    for _, post := range feed {
        fmt.Printf("- %s: %s (Likes: %d)\n", post.UserID, post.Content, post.Likes)
    }
}
```

---

## 📚 Additional Resources

### **Books**
- [System Design Interview](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF) - Alex Xu
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [Building Microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/) - Sam Newman

### **Online Resources**
- [System Design Primer](https://github.com/donnemartin/system-design-primer) - GitHub repository
- [High Scalability](https://highscalability.com/) - Real-world system design case studies
- [AWS Architecture Center](https://aws.amazon.com/architecture/) - Cloud architecture patterns

### **Video Resources**
- [ByteByteGo](https://www.youtube.com/c/ByteByteGo) - System design explanations
- [Gaurav Sen](https://www.youtube.com/c/GauravSensei) - System design and algorithms
- [Exponent](https://www.youtube.com/c/ExponentTV) - Mock interviews and system design

---

*This comprehensive guide covers essential system design concepts with practical Go implementations and real-world interview questions from top tech companies.*
