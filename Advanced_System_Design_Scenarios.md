# 🏗️ Advanced System Design Scenarios

> **Master complex system design scenarios for senior engineering interviews**

## 📚 Overview

This guide covers advanced system design scenarios that are commonly asked in senior engineering interviews at top tech companies. These scenarios test your ability to design large-scale, distributed systems with complex requirements.

## 🎯 Table of Contents

1. [Design a Distributed Cache System](#design-a-distributed-cache-system)
2. [Design a Real-time Chat System](#design-a-real-time-chat-system)
3. [Design a Video Streaming Platform](#design-a-video-streaming-platform)
4. [Design a Social Media Feed System](#design-a-social-media-feed-system)
5. [Design a Payment Processing System](#design-a-payment-processing-system)
6. [Design a Search Engine](#design-a-search-engine)
7. [Design a URL Shortener](#design-a-url-shortener)
8. [Design a Distributed File System](#design-a-distributed-file-system)
9. [Design a Recommendation System](#design-a-recommendation-system)
10. [Design a Multi-tenant SaaS Platform](#design-a-multi-tenant-saas-platform)

## 🗄️ Design a Distributed Cache System

### **Requirements**
- **Scale**: 100M requests/second
- **Data**: 1TB cache capacity
- **Latency**: < 1ms for cache hits
- **Availability**: 99.99% uptime
- **Consistency**: Eventual consistency acceptable

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                Distributed Cache System                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Client    │  │   Client    │  │   Client    │     │
│  │   App 1     │  │   App 2     │  │   App 3     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Load Balancer                         │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Cache     │  │   Cache     │  │   Cache     │     │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Consistent Hash Ring                  │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Storage   │  │   Storage   │  │   Storage   │     │
│  │   Backend   │  │   Backend   │  │   Backend   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Cache Node**
```go
type CacheNode struct {
    ID       string
    Address  string
    Capacity int64
    Data     map[string]CacheItem
    mutex    sync.RWMutex
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int64
    LastAccess  time.Time
}

func (cn *CacheNode) Get(key string) (interface{}, bool) {
    cn.mutex.RLock()
    defer cn.mutex.RUnlock()
    
    item, exists := cn.Data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(item.ExpiresAt) {
        delete(cn.Data, key)
        return nil, false
    }
    
    // Update access statistics
    item.AccessCount++
    item.LastAccess = time.Now()
    cn.Data[key] = item
    
    return item.Value, true
}

func (cn *CacheNode) Set(key string, value interface{}, ttl time.Duration) {
    cn.mutex.Lock()
    defer cn.mutex.Unlock()
    
    cn.Data[key] = CacheItem{
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess:  time.Now(),
    }
}
```

#### **2. Consistent Hash Ring**
```go
type ConsistentHash struct {
    ring       map[uint32]string
    sortedKeys []uint32
    replicas   int
    mutex      sync.RWMutex
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", nodeID, i))
        ch.ring[hash] = nodeID
        ch.sortedKeys = append(ch.sortedKeys, hash)
    }
    
    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    hash := ch.hash(key)
    
    for _, nodeHash := range ch.sortedKeys {
        if nodeHash >= hash {
            return ch.ring[nodeHash]
        }
    }
    
    return ch.ring[ch.sortedKeys[0]]
}
```

#### **3. Cache Client**
```go
type CacheClient struct {
    nodes    map[string]*CacheNode
    hashRing *ConsistentHash
    mutex    sync.RWMutex
}

func (cc *CacheClient) Get(key string) (interface{}, error) {
    nodeID := cc.hashRing.GetNode(key)
    
    cc.mutex.RLock()
    node, exists := cc.nodes[nodeID]
    cc.mutex.RUnlock()
    
    if !exists {
        return nil, errors.New("node not found")
    }
    
    value, found := node.Get(key)
    if !found {
        return nil, errors.New("key not found")
    }
    
    return value, nil
}

func (cc *CacheClient) Set(key string, value interface{}, ttl time.Duration) error {
    nodeID := cc.hashRing.GetNode(key)
    
    cc.mutex.RLock()
    node, exists := cc.nodes[nodeID]
    cc.mutex.RUnlock()
    
    if !exists {
        return errors.New("node not found")
    }
    
    node.Set(key, value, ttl)
    return nil
}
```

### **Advanced Features**

#### **1. Cache Eviction (LRU)**
```go
func (cn *CacheNode) EvictLRU() {
    cn.mutex.Lock()
    defer cn.mutex.Unlock()
    
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range cn.Data {
        if oldestKey == "" || item.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(cn.Data, oldestKey)
    }
}
```

#### **2. Cache Warming**
```go
func (cc *CacheClient) WarmCache(keys []string) {
    for _, key := range keys {
        // Preload frequently accessed data
        go func(k string) {
            value, err := cc.loadFromBackend(k)
            if err == nil {
                cc.Set(k, value, 1*time.Hour)
            }
        }(key)
    }
}
```

## 💬 Design a Real-time Chat System

### **Requirements**
- **Scale**: 10M concurrent users
- **Messages**: 1M messages/second
- **Latency**: < 100ms for message delivery
- **Features**: 1-on-1 chat, group chat, presence, typing indicators

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                Real-time Chat System                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Mobile    │  │   Web       │  │   Desktop   │     │
│  │   Client    │  │   Client    │  │   Client    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Load Balancer                         │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Chat      │  │   Chat      │  │   Chat      │     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Message Queue                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Kafka     │  │   Kafka     │  │   Kafka     │ │ │
│  │  │   Topic 1   │  │   Topic 2   │  │   Topic 3   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   WebSocket │  │   WebSocket │  │   WebSocket │     │
│  │   Gateway   │  │   Gateway   │  │   Gateway   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Database  │  │   Cache     │  │   Presence  │     │
│  │   (MongoDB) │  │   (Redis)   │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Chat Service**
```go
type ChatService struct {
    messageQueue MessageQueue
    db          Database
    cache       Cache
    presence    PresenceService
    wsGateway   WebSocketGateway
}

type Message struct {
    ID        string    `json:"id"`
    SenderID  string    `json:"sender_id"`
    ChatID    string    `json:"chat_id"`
    Content   string    `json:"content"`
    Type      string    `json:"type"`
    Timestamp time.Time `json:"timestamp"`
}

func (cs *ChatService) SendMessage(message Message) error {
    // Validate message
    if err := cs.validateMessage(message); err != nil {
        return err
    }
    
    // Store in database
    if err := cs.db.StoreMessage(message); err != nil {
        return err
    }
    
    // Publish to message queue
    if err := cs.messageQueue.Publish("messages", message); err != nil {
        return err
    }
    
    // Update cache
    cs.cache.Set(fmt.Sprintf("message:%s", message.ID), message, 1*time.Hour)
    
    return nil
}

func (cs *ChatService) GetMessages(chatID string, limit int, offset int) ([]Message, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("messages:%s:%d:%d", chatID, limit, offset)
    if cached, found := cs.cache.Get(cacheKey); found {
        return cached.([]Message), nil
    }
    
    // Load from database
    messages, err := cs.db.GetMessages(chatID, limit, offset)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    cs.cache.Set(cacheKey, messages, 5*time.Minute)
    
    return messages, nil
}
```

#### **2. WebSocket Gateway**
```go
type WebSocketGateway struct {
    connections map[string]*websocket.Conn
    mutex       sync.RWMutex
    messageQueue MessageQueue
}

func (wsg *WebSocketGateway) HandleConnection(conn *websocket.Conn, userID string) {
    wsg.mutex.Lock()
    wsg.connections[userID] = conn
    wsg.mutex.Unlock()
    
    defer func() {
        wsg.mutex.Lock()
        delete(wsg.connections, userID)
        wsg.mutex.Unlock()
        conn.Close()
    }()
    
    // Subscribe to user's message queue
    go wsg.subscribeToMessages(userID)
    
    // Handle incoming messages
    for {
        var message Message
        if err := conn.ReadJSON(&message); err != nil {
            break
        }
        
        // Process incoming message
        wsg.processIncomingMessage(message)
    }
}

func (wsg *WebSocketGateway) subscribeToMessages(userID string) {
    consumer := wsg.messageQueue.Subscribe(fmt.Sprintf("user:%s", userID))
    
    for message := range consumer {
        wsg.mutex.RLock()
        conn, exists := wsg.connections[userID]
        wsg.mutex.RUnlock()
        
        if exists {
            conn.WriteJSON(message)
        }
    }
}
```

#### **3. Presence Service**
```go
type PresenceService struct {
    cache Cache
    mutex sync.RWMutex
}

type UserPresence struct {
    UserID    string    `json:"user_id"`
    Status    string    `json:"status"`
    LastSeen  time.Time `json:"last_seen"`
    Device    string    `json:"device"`
}

func (ps *PresenceService) UpdatePresence(userID string, status string, device string) {
    presence := UserPresence{
        UserID:   userID,
        Status:   status,
        LastSeen: time.Now(),
        Device:   device,
    }
    
    ps.cache.Set(fmt.Sprintf("presence:%s", userID), presence, 5*time.Minute)
}

func (ps *PresenceService) GetPresence(userID string) (*UserPresence, error) {
    if cached, found := ps.cache.Get(fmt.Sprintf("presence:%s", userID)); found {
        return cached.(*UserPresence), nil
    }
    
    return nil, errors.New("presence not found")
}
```

## 🎬 Design a Video Streaming Platform

### **Requirements**
- **Scale**: 100M users, 1B hours of content
- **Quality**: 4K video streaming
- **Latency**: < 2 seconds for video start
- **Features**: Live streaming, on-demand, recommendations

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                Video Streaming Platform                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Mobile    │  │   Web       │  │   Smart TV  │     │
│  │   App       │  │   Player    │  │   App       │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              CDN Network                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Edge      │  │   Edge      │  │   Edge      │ │ │
│  │  │   Server    │  │   Server    │  │   Server    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              API Gateway                           │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Video     │  │   User      │  │   Content   │     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Video     │  │   User      │  │   Content   │     │
│  │   Storage   │  │   Database  │  │   Database  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Video Service**
```go
type VideoService struct {
    storage    VideoStorage
    encoder    VideoEncoder
    cdn        CDNService
    db         Database
    cache      Cache
}

type Video struct {
    ID          string    `json:"id"`
    Title       string    `json:"title"`
    Description string    `json:"description"`
    Duration    int       `json:"duration"`
    Quality     []string  `json:"quality"`
    Thumbnail   string    `json:"thumbnail"`
    UploadedAt  time.Time `json:"uploaded_at"`
}

func (vs *VideoService) UploadVideo(file io.Reader, metadata VideoMetadata) (*Video, error) {
    // Generate video ID
    videoID := vs.generateVideoID()
    
    // Store original video
    if err := vs.storage.StoreOriginal(videoID, file); err != nil {
        return nil, err
    }
    
    // Create video record
    video := &Video{
        ID:          videoID,
        Title:       metadata.Title,
        Description: metadata.Description,
        UploadedAt:  time.Now(),
    }
    
    // Store in database
    if err := vs.db.CreateVideo(video); err != nil {
        return nil, err
    }
    
    // Start encoding process
    go vs.encodeVideo(videoID)
    
    return video, nil
}

func (vs *VideoService) encodeVideo(videoID string) {
    // Get original video
    originalVideo, err := vs.storage.GetOriginal(videoID)
    if err != nil {
        return
    }
    
    // Encode to different qualities
    qualities := []string{"360p", "720p", "1080p", "4K"}
    for _, quality := range qualities {
        encodedVideo, err := vs.encoder.Encode(originalVideo, quality)
        if err != nil {
            continue
        }
        
        // Store encoded video
        if err := vs.storage.StoreEncoded(videoID, quality, encodedVideo); err != nil {
            continue
        }
        
        // Upload to CDN
        vs.cdn.Upload(videoID, quality, encodedVideo)
    }
}
```

#### **2. CDN Service**
```go
type CDNService struct {
    edgeServers []EdgeServer
    mutex       sync.RWMutex
}

type EdgeServer struct {
    ID      string
    Region  string
    URL     string
    Load    float64
}

func (cs *CDNService) GetVideoURL(videoID string, quality string, userLocation string) string {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    // Find best edge server for user location
    bestServer := cs.findBestServer(userLocation)
    
    return fmt.Sprintf("%s/videos/%s/%s", bestServer.URL, videoID, quality)
}

func (cs *CDNService) findBestServer(userLocation string) *EdgeServer {
    // Simple implementation - in reality, use geographic routing
    var bestServer *EdgeServer
    minLoad := float64(1.0)
    
    for i := range cs.edgeServers {
        if cs.edgeServers[i].Load < minLoad {
            minLoad = cs.edgeServers[i].Load
            bestServer = &cs.edgeServers[i]
        }
    }
    
    return bestServer
}
```

## 📱 Design a Social Media Feed System

### **Requirements**
- **Scale**: 1B users, 100M posts per day
- **Latency**: < 200ms for feed generation
- **Features**: Timeline, stories, live updates, recommendations

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                Social Media Feed System                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Mobile    │  │   Web       │  │   Desktop   │     │
│  │   App       │  │   Client    │  │   Client    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Load Balancer                         │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Feed      │  │   Post      │  │   User      │     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Message Queue                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Feed      │  │   Post      │  │   User      │ │ │
│  │  │   Updates   │  │   Events    │  │   Events    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Feed      │  │   Post      │  │   User      │     │
│  │   Cache     │  │   Database  │  │   Database  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Feed Service**
```go
type FeedService struct {
    postService PostService
    userService UserService
    cache       Cache
    db          Database
    messageQueue MessageQueue
}

type FeedItem struct {
    PostID    string    `json:"post_id"`
    UserID    string    `json:"user_id"`
    Content   string    `json:"content"`
    Timestamp time.Time `json:"timestamp"`
    Score     float64   `json:"score"`
}

func (fs *FeedService) GetFeed(userID string, limit int) ([]FeedItem, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("feed:%s", userID)
    if cached, found := fs.cache.Get(cacheKey); found {
        return cached.([]FeedItem), nil
    }
    
    // Get user's following list
    following, err := fs.userService.GetFollowing(userID)
    if err != nil {
        return nil, err
    }
    
    // Get recent posts from following users
    posts, err := fs.postService.GetRecentPosts(following, limit)
    if err != nil {
        return nil, err
    }
    
    // Rank posts by relevance
    rankedPosts := fs.rankPosts(posts, userID)
    
    // Cache the result
    fs.cache.Set(cacheKey, rankedPosts, 5*time.Minute)
    
    return rankedPosts, nil
}

func (fs *FeedService) rankPosts(posts []Post, userID string) []FeedItem {
    var feedItems []FeedItem
    
    for _, post := range posts {
        score := fs.calculateScore(post, userID)
        feedItems = append(feedItems, FeedItem{
            PostID:    post.ID,
            UserID:    post.UserID,
            Content:   post.Content,
            Timestamp: post.CreatedAt,
            Score:     score,
        })
    }
    
    // Sort by score
    sort.Slice(feedItems, func(i, j int) bool {
        return feedItems[i].Score > feedItems[j].Score
    })
    
    return feedItems
}
```

## 💳 Design a Payment Processing System

### **Requirements**
- **Scale**: 1M transactions per second
- **Latency**: < 100ms for payment processing
- **Security**: PCI DSS compliance
- **Features**: Multiple payment methods, fraud detection, refunds

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                Payment Processing System               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   E-commerce│  │   Mobile    │  │   POS       │     │
│  │   Website   │  │   App       │  │   Terminal  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              API Gateway                           │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Payment   │  │   Fraud     │  │   Risk      │     │
│  │   Service   │  │   Detection │  │   Assessment│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Payment Gateway                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Credit    │  │   Debit     │  │   Digital   │ │ │
│  │  │   Card      │  │   Card      │  │   Wallet    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Transaction│  │   Audit    │  │   Compliance│     │
│  │   Database  │  │   Logs     │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Payment Service**
```go
type PaymentService struct {
    gateway      PaymentGateway
    fraudDetector FraudDetector
    riskAssessor RiskAssessor
    db          Database
    cache       Cache
    auditLogger AuditLogger
}

type PaymentRequest struct {
    ID            string  `json:"id"`
    Amount        float64 `json:"amount"`
    Currency      string  `json:"currency"`
    PaymentMethod string  `json:"payment_method"`
    CardNumber    string  `json:"card_number"`
    CVV           string  `json:"cvv"`
    ExpiryDate    string  `json:"expiry_date"`
    MerchantID    string  `json:"merchant_id"`
    CustomerID    string  `json:"customer_id"`
}

type PaymentResponse struct {
    ID            string    `json:"id"`
    Status        string    `json:"status"`
    TransactionID string    `json:"transaction_id"`
    Timestamp     time.Time `json:"timestamp"`
    Error         string    `json:"error,omitempty"`
}

func (ps *PaymentService) ProcessPayment(request PaymentRequest) (*PaymentResponse, error) {
    // Validate request
    if err := ps.validateRequest(request); err != nil {
        return nil, err
    }
    
    // Risk assessment
    riskScore, err := ps.riskAssessor.AssessRisk(request)
    if err != nil {
        return nil, err
    }
    
    // Fraud detection
    isFraud, err := ps.fraudDetector.DetectFraud(request)
    if err != nil {
        return nil, err
    }
    
    if isFraud {
        return &PaymentResponse{
            ID:        request.ID,
            Status:    "declined",
            Timestamp: time.Now(),
            Error:     "fraud detected",
        }, nil
    }
    
    // Process payment
    result, err := ps.gateway.ProcessPayment(request)
    if err != nil {
        return nil, err
    }
    
    // Store transaction
    transaction := &Transaction{
        ID:            request.ID,
        Amount:        request.Amount,
        Currency:      request.Currency,
        Status:        result.Status,
        TransactionID: result.TransactionID,
        Timestamp:     time.Now(),
    }
    
    if err := ps.db.StoreTransaction(transaction); err != nil {
        return nil, err
    }
    
    // Audit log
    ps.auditLogger.LogPayment(request, result)
    
    return result, nil
}
```

## 🔍 Design a Search Engine

### **Requirements**
- **Scale**: 1B web pages, 10M queries per second
- **Latency**: < 100ms for search results
- **Features**: Full-text search, autocomplete, spell correction

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────┐
│                    Search Engine                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Web       │  │   Mobile    │  │   API       │     │
│  │   Client    │  │   Client    │  │   Client    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Load Balancer                         │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Search    │  │   Index     │  │   Crawler   │     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Search    │  │   Inverted  │  │   Web       │     │
│  │   Index     │  │   Index     │  │   Pages     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **Key Components**

#### **1. Search Service**
```go
type SearchService struct {
    indexer    Indexer
    ranker     Ranker
    cache      Cache
    db         Database
}

type SearchResult struct {
    URL         string  `json:"url"`
    Title       string  `json:"title"`
    Snippet     string  `json:"snippet"`
    Score       float64 `json:"score"`
    LastCrawled time.Time `json:"last_crawled"`
}

func (ss *SearchService) Search(query string, limit int) ([]SearchResult, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("search:%s:%d", query, limit)
    if cached, found := ss.cache.Get(cacheKey); found {
        return cached.([]SearchResult), nil
    }
    
    // Parse query
    terms := ss.parseQuery(query)
    
    // Search index
    documents, err := ss.indexer.Search(terms)
    if err != nil {
        return nil, err
    }
    
    // Rank results
    rankedResults := ss.ranker.Rank(documents, query)
    
    // Limit results
    if len(rankedResults) > limit {
        rankedResults = rankedResults[:limit]
    }
    
    // Cache results
    ss.cache.Set(cacheKey, rankedResults, 5*time.Minute)
    
    return rankedResults, nil
}
```

## 🎯 Best Practices

### **1. Scalability**
- Design for horizontal scaling
- Use appropriate data structures
- Implement caching strategies
- Consider database sharding

### **2. Performance**
- Optimize for latency
- Use CDNs for static content
- Implement proper indexing
- Monitor and profile

### **3. Reliability**
- Implement circuit breakers
- Use message queues for decoupling
- Design for failure
- Implement proper error handling

### **4. Security**
- Encrypt sensitive data
- Implement proper authentication
- Use HTTPS everywhere
- Follow security best practices

---

**🎉 Master these advanced scenarios to excel in senior system design interviews!**

**Good luck with your system design journey! 🚀**
