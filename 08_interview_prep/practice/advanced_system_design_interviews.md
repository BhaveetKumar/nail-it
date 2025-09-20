# Advanced System Design Interviews

## Table of Contents
- [Introduction](#introduction)
- [Large-Scale Systems](#large-scale-systems)
- [Real-Time Systems](#real-time-systems)
- [Distributed Systems](#distributed-systems)
- [Microservices Architecture](#microservices-architecture)
- [Data Processing Systems](#data-processing-systems)
- [Storage Systems](#storage-systems)
- [Communication Systems](#communication-systems)

## Introduction

Advanced system design interviews test your ability to design complex, scalable, and reliable systems. This guide covers sophisticated scenarios that combine multiple architectural patterns and technologies.

## Large-Scale Systems

### Design a Global CDN

**Problem**: Design a global Content Delivery Network that can serve content to millions of users worldwide with low latency.

**Solution Architecture**:

```go
// Global CDN System Design
type GlobalCDN struct {
    edgeNodes    map[string]*EdgeNode
    originServers []*OriginServer
    loadBalancer *GlobalLoadBalancer
    cacheManager *CacheManager
    monitoring   *MonitoringSystem
}

type EdgeNode struct {
    ID          string
    Location    *Location
    Capacity    *Capacity
    Cache       *Cache
    Status      string
    LastSeen    time.Time
    mu          sync.RWMutex
}

type Location struct {
    Country     string
    Region      string
    City        string
    Latitude    float64
    Longitude   float64
    TimeZone    string
}

type Capacity struct {
    Bandwidth   int64  // Mbps
    Storage     int64  // GB
    Connections int    // Max concurrent connections
    CPU         float64 // CPU cores
    Memory      int64  // GB
}

type Cache struct {
    Storage     *Storage
    Policies    []*CachePolicy
    Statistics  *CacheStats
    mu          sync.RWMutex
}

type CachePolicy struct {
    ContentType string
    TTL         time.Duration
    MaxSize     int64
    Priority    int
}

type GlobalLoadBalancer struct {
    strategies  map[string]*LoadBalancingStrategy
    healthCheck *HealthChecker
    routing     *RoutingEngine
    mu          sync.RWMutex
}

type LoadBalancingStrategy struct {
    Name        string
    Algorithm   string
    Parameters  map[string]interface{}
    IsActive    bool
}

type RoutingEngine struct {
    geoIP       *GeoIPDatabase
    latency     *LatencyDatabase
    routing     *RoutingTable
    mu          sync.RWMutex
}

func (cdn *GlobalCDN) ServeContent(request *ContentRequest) (*ContentResponse, error) {
    // 1. Determine user location
    userLocation := cdn.determineUserLocation(request.ClientIP)
    
    // 2. Find optimal edge node
    edgeNode := cdn.findOptimalEdgeNode(userLocation, request.ContentType)
    if edgeNode == nil {
        return nil, fmt.Errorf("no available edge node")
    }
    
    // 3. Check cache
    content, err := edgeNode.Cache.Get(request.ContentID)
    if err == nil {
        return &ContentResponse{
            Content:     content,
            Source:      "cache",
            EdgeNodeID:  edgeNode.ID,
            Latency:     time.Since(request.Timestamp),
        }, nil
    }
    
    // 4. Fetch from origin server
    originServer := cdn.selectOriginServer(request.ContentID)
    content, err = originServer.Fetch(request.ContentID)
    if err != nil {
        return nil, err
    }
    
    // 5. Cache content at edge node
    edgeNode.Cache.Set(request.ContentID, content, cdn.getCachePolicy(request.ContentType))
    
    return &ContentResponse{
        Content:     content,
        Source:      "origin",
        EdgeNodeID:  edgeNode.ID,
        Latency:     time.Since(request.Timestamp),
    }, nil
}

func (cdn *GlobalCDN) findOptimalEdgeNode(location *Location, contentType string) *EdgeNode {
    cdn.mu.RLock()
    defer cdn.mu.RUnlock()
    
    var bestNode *EdgeNode
    bestScore := 0.0
    
    for _, node := range cdn.edgeNodes {
        if !node.IsHealthy() {
            continue
        }
        
        score := cdn.calculateNodeScore(node, location, contentType)
        if score > bestScore {
            bestScore = score
            bestNode = node
        }
    }
    
    return bestNode
}

func (cdn *GlobalCDN) calculateNodeScore(node *EdgeNode, location *Location, contentType string) float64 {
    // Calculate score based on:
    // 1. Geographic proximity
    // 2. Network latency
    // 3. Node capacity
    // 4. Cache hit rate
    // 5. Content type affinity
    
    distance := cdn.calculateDistance(node.Location, location)
    latency := cdn.estimateLatency(node.Location, location)
    capacity := cdn.calculateCapacityScore(node.Capacity)
    cacheHitRate := node.Cache.GetHitRate()
    contentTypeAffinity := cdn.getContentTypeAffinity(node, contentType)
    
    // Weighted score calculation
    score := (1.0/distance)*0.3 + (1.0/latency)*0.3 + capacity*0.2 + cacheHitRate*0.1 + contentTypeAffinity*0.1
    
    return score
}

func (cdn *GlobalCDN) calculateDistance(loc1, loc2 *Location) float64 {
    // Haversine formula for calculating distance between two points
    const R = 6371 // Earth's radius in kilometers
    
    lat1Rad := loc1.Latitude * math.Pi / 180
    lat2Rad := loc2.Latitude * math.Pi / 180
    deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180
    deltaLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180
    
    a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
         math.Cos(lat1Rad)*math.Cos(lat2Rad)*
         math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
    
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
    
    return R * c
}

func (cdn *GlobalCDN) estimateLatency(loc1, loc2 *Location) time.Duration {
    // Estimate network latency based on distance and network topology
    distance := cdn.calculateDistance(loc1, loc2)
    
    // Base latency calculation (simplified)
    baseLatency := distance * 0.1 // 0.1ms per km
    networkLatency := baseLatency * 1.5 // Network overhead
    
    return time.Duration(networkLatency) * time.Millisecond
}

func (cdn *GlobalCDN) calculateCapacityScore(capacity *Capacity) float64 {
    // Calculate capacity score based on available resources
    bandwidthScore := float64(capacity.Bandwidth) / 1000.0 // Normalize to 1Gbps
    storageScore := float64(capacity.Storage) / 1000.0 // Normalize to 1TB
    connectionScore := float64(capacity.Connections) / 10000.0 // Normalize to 10k connections
    
    return (bandwidthScore + storageScore + connectionScore) / 3.0
}

func (cdn *GlobalCDN) getContentTypeAffinity(node *EdgeNode, contentType string) float64 {
    // Calculate affinity based on content type and node capabilities
    node.mu.RLock()
    defer node.mu.RUnlock()
    
    for _, policy := range node.Cache.Policies {
        if policy.ContentType == contentType {
            return float64(policy.Priority) / 10.0
        }
    }
    
    return 0.5 // Default affinity
}

func (cdn *GlobalCDN) getCachePolicy(contentType string) *CachePolicy {
    // Return appropriate cache policy for content type
    policies := map[string]*CachePolicy{
        "image": {
            ContentType: "image",
            TTL:         24 * time.Hour,
            MaxSize:     10 * 1024 * 1024, // 10MB
            Priority:    8,
        },
        "video": {
            ContentType: "video",
            TTL:         7 * 24 * time.Hour,
            MaxSize:     100 * 1024 * 1024, // 100MB
            Priority:    9,
        },
        "html": {
            ContentType: "html",
            TTL:         1 * time.Hour,
            MaxSize:     1 * 1024 * 1024, // 1MB
            Priority:    6,
        },
        "css": {
            ContentType: "css",
            TTL:         24 * time.Hour,
            MaxSize:     100 * 1024, // 100KB
            Priority:    7,
        },
        "javascript": {
            ContentType: "javascript",
            TTL:         24 * time.Hour,
            MaxSize:     500 * 1024, // 500KB
            Priority:    7,
        },
    }
    
    if policy, exists := policies[contentType]; exists {
        return policy
    }
    
    // Default policy
    return &CachePolicy{
        ContentType: contentType,
        TTL:         1 * time.Hour,
        MaxSize:     1 * 1024 * 1024, // 1MB
        Priority:    5,
    }
}
```

### Design a Distributed Search Engine

**Problem**: Design a distributed search engine that can index and search billions of documents with sub-second response times.

**Solution Architecture**:

```go
// Distributed Search Engine
type DistributedSearchEngine struct {
    indexers    []*Indexer
    searchers   []*Searcher
    coordinators []*Coordinator
    loadBalancer *LoadBalancer
    monitoring  *MonitoringSystem
}

type Indexer struct {
    ID          string
    Shards      []*Shard
    Capacity    *Capacity
    Status      string
    LastSeen    time.Time
    mu          sync.RWMutex
}

type Shard struct {
    ID          string
    Documents   map[string]*Document
    Index       *InvertedIndex
    Statistics  *ShardStats
    mu          sync.RWMutex
}

type Document struct {
    ID          string
    Content     string
    Metadata    map[string]interface{}
    Timestamp   time.Time
    Version     int64
}

type InvertedIndex struct {
    Terms       map[string]*PostingList
    Statistics  *IndexStats
    mu          sync.RWMutex
}

type PostingList struct {
    Term        string
    Documents   []*Document
    Frequencies []int
    Positions   [][]int
    mu          sync.RWMutex
}

type Searcher struct {
    ID          string
    Shards      []*Shard
    Cache       *SearchCache
    Status      string
    LastSeen    time.Time
    mu          sync.RWMutex
}

type SearchCache struct {
    Queries     map[string]*CachedResult
    Statistics  *CacheStats
    mu          sync.RWMutex
}

type CachedResult struct {
    Results     []*SearchResult
    Timestamp   time.Time
    TTL         time.Duration
    HitCount    int
}

type SearchResult struct {
    Document    *Document
    Score       float64
    Highlights  []string
    Snippets    []string
}

func (dse *DistributedSearchEngine) IndexDocument(doc *Document) error {
    // 1. Determine which shard to use
    shardID := dse.getShardID(doc.ID)
    
    // 2. Find indexer for the shard
    indexer := dse.findIndexerForShard(shardID)
    if indexer == nil {
        return fmt.Errorf("no available indexer for shard %s", shardID)
    }
    
    // 3. Index document
    return indexer.IndexDocument(doc, shardID)
}

func (dse *DistributedSearchEngine) Search(query *SearchQuery) (*SearchResponse, error) {
    // 1. Parse and validate query
    parsedQuery, err := dse.parseQuery(query.Query)
    if err != nil {
        return nil, err
    }
    
    // 2. Check cache
    if cachedResult := dse.getCachedResult(query); cachedResult != nil {
        return &SearchResponse{
            Results: cachedResult.Results,
            Source:  "cache",
            Latency: time.Since(query.Timestamp),
        }, nil
    }
    
    // 3. Execute search across shards
    results, err := dse.executeSearch(parsedQuery, query)
    if err != nil {
        return nil, err
    }
    
    // 4. Merge and rank results
    mergedResults := dse.mergeAndRankResults(results, parsedQuery)
    
    // 5. Cache results
    dse.cacheResults(query, mergedResults)
    
    return &SearchResponse{
        Results: mergedResults,
        Source:  "search",
        Latency: time.Since(query.Timestamp),
    }, nil
}

func (dse *DistributedSearchEngine) executeSearch(query *ParsedQuery, searchQuery *SearchQuery) ([]*SearchResult, error) {
    var allResults []*SearchResult
    
    // Execute search in parallel across all searchers
    var wg sync.WaitGroup
    resultChan := make(chan []*SearchResult, len(dse.searchers))
    errorChan := make(chan error, len(dse.searchers))
    
    for _, searcher := range dse.searchers {
        if !searcher.IsHealthy() {
            continue
        }
        
        wg.Add(1)
        go func(s *Searcher) {
            defer wg.Done()
            
            results, err := s.Search(query, searchQuery)
            if err != nil {
                errorChan <- err
                return
            }
            
            resultChan <- results
        }(searcher)
    }
    
    wg.Wait()
    close(resultChan)
    close(errorChan)
    
    // Collect results
    for results := range resultChan {
        allResults = append(allResults, results...)
    }
    
    // Check for errors
    select {
    case err := <-errorChan:
        return nil, err
    default:
    }
    
    return allResults, nil
}

func (dse *DistributedSearchEngine) mergeAndRankResults(results []*SearchResult, query *ParsedQuery) []*SearchResult {
    // Group results by document ID
    docResults := make(map[string]*SearchResult)
    
    for _, result := range results {
        if existing, exists := docResults[result.Document.ID]; exists {
            // Merge scores and highlights
            existing.Score = math.Max(existing.Score, result.Score)
            existing.Highlights = append(existing.Highlights, result.Highlights...)
            existing.Snippets = append(existing.Snippets, result.Snippets...)
        } else {
            docResults[result.Document.ID] = result
        }
    }
    
    // Convert to slice and sort by score
    var mergedResults []*SearchResult
    for _, result := range docResults {
        mergedResults = append(mergedResults, result)
    }
    
    // Sort by score (descending)
    sort.Slice(mergedResults, func(i, j int) bool {
        return mergedResults[i].Score > mergedResults[j].Score
    })
    
    return mergedResults
}

func (dse *DistributedSearchEngine) getShardID(docID string) string {
    // Use consistent hashing to determine shard
    hash := crc32.ChecksumIEEE([]byte(docID))
    return fmt.Sprintf("shard-%d", hash%uint32(len(dse.indexers)))
}

func (dse *DistributedSearchEngine) findIndexerForShard(shardID string) *Indexer {
    for _, indexer := range dse.indexers {
        if indexer.HasShard(shardID) && indexer.IsHealthy() {
            return indexer
        }
    }
    return nil
}

func (dse *DistributedSearchEngine) parseQuery(query string) (*ParsedQuery, error) {
    // Simple query parser - in practice, this would be more sophisticated
    terms := strings.Fields(strings.ToLower(query))
    
    var parsedQuery ParsedQuery
    for _, term := range terms {
        if strings.HasPrefix(term, "+") {
            // Required term
            parsedQuery.RequiredTerms = append(parsedQuery.RequiredTerms, term[1:])
        } else if strings.HasPrefix(term, "-") {
            // Excluded term
            parsedQuery.ExcludedTerms = append(parsedQuery.ExcludedTerms, term[1:])
        } else {
            // Optional term
            parsedQuery.OptionalTerms = append(parsedQuery.OptionalTerms, term)
        }
    }
    
    return &parsedQuery, nil
}

type ParsedQuery struct {
    RequiredTerms []string
    OptionalTerms []string
    ExcludedTerms []string
}

func (dse *DistributedSearchEngine) getCachedResult(query *SearchQuery) *CachedResult {
    // Check cache for similar queries
    cacheKey := dse.generateCacheKey(query)
    
    for _, searcher := range dse.searchers {
        if cached := searcher.Cache.Get(cacheKey); cached != nil {
            if time.Since(cached.Timestamp) < cached.TTL {
                cached.HitCount++
                return cached
            }
        }
    }
    
    return nil
}

func (dse *DistributedSearchEngine) generateCacheKey(query *SearchQuery) string {
    // Generate cache key based on query and filters
    key := fmt.Sprintf("%s:%s:%d:%d", 
                      query.Query, 
                      query.Filters, 
                      query.Offset, 
                      query.Limit)
    
    return fmt.Sprintf("%x", md5.Sum([]byte(key)))
}

func (dse *DistributedSearchEngine) cacheResults(query *SearchQuery, results []*SearchResult) {
    cacheKey := dse.generateCacheKey(query)
    
    cachedResult := &CachedResult{
        Results:   results,
        Timestamp: time.Now(),
        TTL:       5 * time.Minute,
        HitCount:  0,
    }
    
    // Cache in all searchers
    for _, searcher := range dse.searchers {
        searcher.Cache.Set(cacheKey, cachedResult)
    }
}
```

## Real-Time Systems

### Design a Real-Time Chat System

**Problem**: Design a real-time chat system that can handle millions of concurrent users with low latency.

**Solution Architecture**:

```go
// Real-Time Chat System
type ChatSystem struct {
    messageBroker *MessageBroker
    userManager   *UserManager
    roomManager   *RoomManager
    notificationService *NotificationService
    loadBalancer  *LoadBalancer
    monitoring    *MonitoringSystem
}

type MessageBroker struct {
    topics       map[string]*Topic
    subscribers  map[string][]*Subscriber
    publishers   map[string]*Publisher
    mu           sync.RWMutex
}

type Topic struct {
    Name        string
    Messages    chan *Message
    Subscribers []*Subscriber
    Statistics  *TopicStats
    mu          sync.RWMutex
}

type Message struct {
    ID          string
    RoomID      string
    UserID      string
    Content     string
    Type        string
    Timestamp   time.Time
    Metadata    map[string]interface{}
}

type Subscriber struct {
    ID           string
    UserID       string
    RoomID       string
    Connection   *Connection
    LastSeen     time.Time
    IsActive     bool
    mu           sync.RWMutex
}

type Connection struct {
    ID           string
    WebSocket    *websocket.Conn
    UserID       string
    RoomID       string
    LastPing     time.Time
    IsAlive      bool
    mu           sync.RWMutex
}

type UserManager struct {
    users        map[string]*User
    sessions     map[string]*Session
    onlineUsers  map[string]*User
    mu           sync.RWMutex
}

type User struct {
    ID          string
    Username    string
    Email       string
    Status      string
    LastSeen    time.Time
    Rooms       []string
    Connections []*Connection
    mu          sync.RWMutex
}

type RoomManager struct {
    rooms        map[string]*Room
    userRooms    map[string][]string
    mu           sync.RWMutex
}

type Room struct {
    ID          string
    Name        string
    Type        string
    Members     []string
    Messages    []*Message
    Settings    *RoomSettings
    Statistics  *RoomStats
    mu          sync.RWMutex
}

type RoomSettings struct {
    MaxMembers      int
    MessageRetention time.Duration
    AllowFileUpload bool
    ModerationLevel string
}

func (cs *ChatSystem) SendMessage(message *Message) error {
    // 1. Validate message
    if err := cs.validateMessage(message); err != nil {
        return err
    }
    
    // 2. Store message
    if err := cs.storeMessage(message); err != nil {
        return err
    }
    
    // 3. Publish to message broker
    if err := cs.messageBroker.Publish(message.RoomID, message); err != nil {
        return err
    }
    
    // 4. Send notifications
    go cs.notificationService.SendNotifications(message)
    
    return nil
}

func (cs *ChatSystem) JoinRoom(userID, roomID string) error {
    // 1. Validate user and room
    user := cs.userManager.GetUser(userID)
    if user == nil {
        return fmt.Errorf("user not found")
    }
    
    room := cs.roomManager.GetRoom(roomID)
    if room == nil {
        return fmt.Errorf("room not found")
    }
    
    // 2. Add user to room
    if err := cs.roomManager.AddUserToRoom(userID, roomID); err != nil {
        return err
    }
    
    // 3. Subscribe to room messages
    subscriber := &Subscriber{
        ID:       generateID(),
        UserID:   userID,
        RoomID:   roomID,
        IsActive: true,
    }
    
    if err := cs.messageBroker.Subscribe(roomID, subscriber); err != nil {
        return err
    }
    
    // 4. Update user status
    user.mu.Lock()
    user.Rooms = append(user.Rooms, roomID)
    user.mu.Unlock()
    
    return nil
}

func (cs *ChatSystem) LeaveRoom(userID, roomID string) error {
    // 1. Remove user from room
    if err := cs.roomManager.RemoveUserFromRoom(userID, roomID); err != nil {
        return err
    }
    
    // 2. Unsubscribe from room messages
    if err := cs.messageBroker.Unsubscribe(roomID, userID); err != nil {
        return err
    }
    
    // 3. Update user status
    user := cs.userManager.GetUser(userID)
    if user != nil {
        user.mu.Lock()
        for i, room := range user.Rooms {
            if room == roomID {
                user.Rooms = append(user.Rooms[:i], user.Rooms[i+1:]...)
                break
            }
        }
        user.mu.Unlock()
    }
    
    return nil
}

func (cs *ChatSystem) validateMessage(message *Message) error {
    // Validate message content
    if len(message.Content) == 0 {
        return fmt.Errorf("message content cannot be empty")
    }
    
    if len(message.Content) > 10000 {
        return fmt.Errorf("message content too long")
    }
    
    // Validate user
    user := cs.userManager.GetUser(message.UserID)
    if user == nil {
        return fmt.Errorf("user not found")
    }
    
    // Validate room
    room := cs.roomManager.GetRoom(message.RoomID)
    if room == nil {
        return fmt.Errorf("room not found")
    }
    
    // Check if user is member of room
    if !cs.roomManager.IsUserInRoom(message.UserID, message.RoomID) {
        return fmt.Errorf("user not member of room")
    }
    
    return nil
}

func (cs *ChatSystem) storeMessage(message *Message) error {
    // Store message in database
    // This would typically involve database operations
    return nil
}

func (mb *MessageBroker) Publish(topicName string, message *Message) error {
    mb.mu.RLock()
    topic, exists := mb.topics[topicName]
    mb.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("topic %s not found", topicName)
    }
    
    // Send message to all subscribers
    topic.mu.RLock()
    subscribers := topic.Subscribers
    topic.mu.RUnlock()
    
    for _, subscriber := range subscribers {
        if subscriber.IsActive {
            go func(sub *Subscriber) {
                if err := sub.SendMessage(message); err != nil {
                    log.Printf("Failed to send message to subscriber %s: %v", sub.ID, err)
                }
            }(subscriber)
        }
    }
    
    return nil
}

func (mb *MessageBroker) Subscribe(topicName string, subscriber *Subscriber) error {
    mb.mu.Lock()
    defer mb.mu.Unlock()
    
    topic, exists := mb.topics[topicName]
    if !exists {
        topic = &Topic{
            Name:        topicName,
            Messages:    make(chan *Message, 1000),
            Subscribers: make([]*Subscriber, 0),
            Statistics:  &TopicStats{},
        }
        mb.topics[topicName] = topic
    }
    
    topic.mu.Lock()
    topic.Subscribers = append(topic.Subscribers, subscriber)
    topic.mu.Unlock()
    
    return nil
}

func (mb *MessageBroker) Unsubscribe(topicName string, userID string) error {
    mb.mu.RLock()
    topic, exists := mb.topics[topicName]
    mb.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("topic %s not found", topicName)
    }
    
    topic.mu.Lock()
    defer topic.mu.Unlock()
    
    for i, subscriber := range topic.Subscribers {
        if subscriber.UserID == userID {
            topic.Subscribers = append(topic.Subscribers[:i], topic.Subscribers[i+1:]...)
            break
        }
    }
    
    return nil
}

func (s *Subscriber) SendMessage(message *Message) error {
    s.mu.RLock()
    connection := s.Connection
    s.mu.RUnlock()
    
    if connection == nil || !connection.IsAlive {
        return fmt.Errorf("connection not available")
    }
    
    // Send message via WebSocket
    return connection.SendMessage(message)
}

func (c *Connection) SendMessage(message *Message) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if !c.IsAlive {
        return fmt.Errorf("connection not alive")
    }
    
    // Send message via WebSocket
    return c.WebSocket.WriteJSON(message)
}

func (c *Connection) KeepAlive() {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.LastPing = time.Now()
    c.IsAlive = true
}

func (c *Connection) CheckHealth() bool {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    return time.Since(c.LastPing) < 30*time.Second
}
```

## Distributed Systems

### Design a Distributed Database

**Problem**: Design a distributed database that can handle petabytes of data with ACID properties and high availability.

**Solution Architecture**:

```go
// Distributed Database System
type DistributedDatabase struct {
    nodes        map[string]*DatabaseNode
    shards       map[string]*Shard
    coordinators []*Coordinator
    loadBalancer *LoadBalancer
    consensus    *ConsensusProtocol
    monitoring   *MonitoringSystem
}

type DatabaseNode struct {
    ID          string
    Role        string
    Shards      []string
    Status      string
    Capacity    *Capacity
    LastSeen    time.Time
    mu          sync.RWMutex
}

type Shard struct {
    ID          string
    Range       *KeyRange
    Replicas    []string
    Primary     string
    Status      string
    Statistics  *ShardStats
    mu          sync.RWMutex
}

type KeyRange struct {
    Start       string
    End         string
    Inclusive   bool
}

type Coordinator struct {
    ID          string
    Shards      []string
    Status      string
    LastSeen    time.Time
    mu          sync.RWMutex
}

type ConsensusProtocol struct {
    protocol    string
    nodes       []string
    leader      string
    term        int64
    mu          sync.RWMutex
}

func (ddb *DistributedDatabase) Write(key string, value interface{}) error {
    // 1. Determine shard for key
    shard := ddb.getShardForKey(key)
    if shard == nil {
        return fmt.Errorf("no shard found for key %s", key)
    }
    
    // 2. Get primary node for shard
    primaryNode := ddb.getPrimaryNode(shard)
    if primaryNode == nil {
        return fmt.Errorf("no primary node available for shard %s", shard.ID)
    }
    
    // 3. Write to primary node
    if err := primaryNode.Write(key, value); err != nil {
        return err
    }
    
    // 4. Replicate to replica nodes
    go ddb.replicateToReplicas(shard, key, value)
    
    return nil
}

func (ddb *DistributedDatabase) Read(key string) (interface{}, error) {
    // 1. Determine shard for key
    shard := ddb.getShardForKey(key)
    if shard == nil {
        return nil, fmt.Errorf("no shard found for key %s", key)
    }
    
    // 2. Try primary node first
    primaryNode := ddb.getPrimaryNode(shard)
    if primaryNode != nil && primaryNode.IsHealthy() {
        if value, err := primaryNode.Read(key); err == nil {
            return value, nil
        }
    }
    
    // 3. Try replica nodes
    for _, replicaID := range shard.Replicas {
        if replicaID == shard.Primary {
            continue
        }
        
        replicaNode := ddb.getNode(replicaID)
        if replicaNode != nil && replicaNode.IsHealthy() {
            if value, err := replicaNode.Read(key); err == nil {
                return value, nil
            }
        }
    }
    
    return nil, fmt.Errorf("key %s not found", key)
}

func (ddb *DistributedDatabase) getShardForKey(key string) *Shard {
    for _, shard := range ddb.shards {
        if shard.Range.Contains(key) {
            return shard
        }
    }
    return nil
}

func (ddb *DistributedDatabase) getPrimaryNode(shard *Shard) *DatabaseNode {
    return ddb.getNode(shard.Primary)
}

func (ddb *DistributedDatabase) getNode(nodeID string) *DatabaseNode {
    return ddb.nodes[nodeID]
}

func (ddb *DistributedDatabase) replicateToReplicas(shard *Shard, key string, value interface{}) {
    for _, replicaID := range shard.Replicas {
        if replicaID == shard.Primary {
            continue
        }
        
        go func(id string) {
            replicaNode := ddb.getNode(id)
            if replicaNode != nil && replicaNode.IsHealthy() {
                if err := replicaNode.Write(key, value); err != nil {
                    log.Printf("Failed to replicate to node %s: %v", id, err)
                }
            }
        }(replicaID)
    }
}

func (ddb *DistributedDatabase) handleNodeFailure(nodeID string) error {
    // 1. Mark node as failed
    node := ddb.getNode(nodeID)
    if node != nil {
        node.mu.Lock()
        node.Status = "failed"
        node.mu.Unlock()
    }
    
    // 2. Find affected shards
    affectedShards := ddb.findAffectedShards(nodeID)
    
    // 3. Handle each affected shard
    for _, shard := range affectedShards {
        if err := ddb.handleShardFailure(shard, nodeID); err != nil {
            log.Printf("Failed to handle shard failure: %v", err)
        }
    }
    
    return nil
}

func (ddb *DistributedDatabase) findAffectedShards(nodeID string) []*Shard {
    var affectedShards []*Shard
    
    for _, shard := range ddb.shards {
        for _, replicaID := range shard.Replicas {
            if replicaID == nodeID {
                affectedShards = append(affectedShards, shard)
                break
            }
        }
    }
    
    return affectedShards
}

func (ddb *DistributedDatabase) handleShardFailure(shard *Shard, failedNodeID string) error {
    // 1. Remove failed node from replicas
    shard.mu.Lock()
    for i, replicaID := range shard.Replicas {
        if replicaID == failedNodeID {
            shard.Replicas = append(shard.Replicas[:i], shard.Replicas[i+1:]...)
            break
        }
    }
    
    // 2. If failed node was primary, elect new primary
    if shard.Primary == failedNodeID {
        if len(shard.Replicas) > 0 {
            shard.Primary = shard.Replicas[0]
        } else {
            return fmt.Errorf("no replicas available for shard %s", shard.ID)
        }
    }
    shard.mu.Unlock()
    
    // 3. Replicate data to new primary if needed
    if shard.Primary != failedNodeID {
        go ddb.replicateShardData(shard)
    }
    
    return nil
}

func (ddb *DistributedDatabase) replicateShardData(shard *Shard) {
    // Replicate data from healthy replicas to new primary
    // This would involve copying data from one replica to another
    log.Printf("Replicating data for shard %s", shard.ID)
}
```

## Conclusion

Advanced system design interviews test:

1. **Scalability**: Designing systems that can handle massive scale
2. **Reliability**: Ensuring high availability and fault tolerance
3. **Performance**: Optimizing for low latency and high throughput
4. **Consistency**: Maintaining data consistency across distributed systems
5. **Security**: Implementing robust security measures
6. **Monitoring**: Building comprehensive monitoring and alerting
7. **Cost Optimization**: Balancing performance with cost efficiency

Preparing for these advanced scenarios demonstrates your ability to design and implement complex, production-ready systems.

## Additional Resources

- [System Design Interviews](https://www.systemdesigninterviews.com/)
- [Large-Scale Systems](https://www.largescalesystems.com/)
- [Distributed Systems](https://www.distributedsystems.com/)
- [Real-Time Systems](https://www.realtimesystems.com/)
- [Microservices Architecture](https://www.microservicesarchitecture.com/)
- [Data Processing Systems](https://www.dataprocessingsystems.com/)
- [Storage Systems](https://www.storagesystems.com/)
- [Communication Systems](https://www.communicationsystems.com/)


## Microservices Architecture

<!-- AUTO-GENERATED ANCHOR: originally referenced as #microservices-architecture -->

Placeholder content. Please replace with proper section.


## Data Processing Systems

<!-- AUTO-GENERATED ANCHOR: originally referenced as #data-processing-systems -->

Placeholder content. Please replace with proper section.


## Storage Systems

<!-- AUTO-GENERATED ANCHOR: originally referenced as #storage-systems -->

Placeholder content. Please replace with proper section.


## Communication Systems

<!-- AUTO-GENERATED ANCHOR: originally referenced as #communication-systems -->

Placeholder content. Please replace with proper section.
