# Advanced System Design Scenarios

Comprehensive advanced system design scenarios for senior engineering interviews.

## 🎯 Large-Scale Distributed Systems

### Scenario 1: Design a Global CDN
**Scale**: 1 billion users, 100TB content, 99.99% availability  
**Requirements**: Sub-100ms latency globally, real-time content updates

#### Architecture Design
```
Global CDN Architecture:
├── Origin Servers (AWS S3, Google Cloud Storage)
├── Regional Distribution Centers (50+ locations)
├── Edge Servers (1000+ locations)
├── DNS Load Balancer (Route 53, CloudFlare)
├── Content Management System
├── Analytics and Monitoring
└── Security and Access Control
```

#### Key Components
```go
// CDN Edge Server Implementation
type EdgeServer struct {
    ID          string
    Location    GeoLocation
    Cache       *ContentCache
    LoadBalancer *LoadBalancer
    HealthCheck *HealthChecker
    Metrics     *MetricsCollector
}

type ContentCache struct {
    memoryCache *MemoryCache
    diskCache   *DiskCache
    ttl         time.Duration
    maxSize     int64
}

func (es *EdgeServer) ServeContent(request *ContentRequest) (*ContentResponse, error) {
    // Check local cache
    if content := es.Cache.Get(request.ContentID); content != nil {
        return &ContentResponse{
            Content: content,
            Source:  "cache",
            Latency: time.Since(request.Timestamp),
        }, nil
    }
    
    // Check parent edge server
    if content := es.getFromParent(request); content != nil {
        es.Cache.Set(request.ContentID, content)
        return &ContentResponse{
            Content: content,
            Source:  "parent",
            Latency: time.Since(request.Timestamp),
        }, nil
    }
    
    // Fetch from origin
    content, err := es.fetchFromOrigin(request)
    if err != nil {
        return nil, err
    }
    
    es.Cache.Set(request.ContentID, content)
    return &ContentResponse{
        Content: content,
        Source:  "origin",
        Latency: time.Since(request.Timestamp),
    }, nil
}
```

#### Content Distribution Strategy
```go
// Content Distribution Manager
type ContentDistributionManager struct {
    originServers []*OriginServer
    edgeServers   []*EdgeServer
    routingTable  *RoutingTable
    analytics     *AnalyticsService
}

func (cdm *ContentDistributionManager) DistributeContent(content *Content) error {
    // Determine distribution strategy based on content type
    strategy := cdm.selectDistributionStrategy(content)
    
    switch strategy {
    case "immediate":
        return cdm.distributeImmediately(content)
    case "scheduled":
        return cdm.distributeScheduled(content)
    case "on-demand":
        return cdm.distributeOnDemand(content)
    default:
        return errors.New("unknown distribution strategy")
    }
}

func (cdm *ContentDistributionManager) distributeImmediately(content *Content) error {
    // Distribute to all edge servers immediately
    var wg sync.WaitGroup
    errors := make(chan error, len(cdm.edgeServers))
    
    for _, edgeServer := range cdm.edgeServers {
        wg.Add(1)
        go func(es *EdgeServer) {
            defer wg.Done()
            if err := es.Cache.Set(content.ID, content); err != nil {
                errors <- err
            }
        }(edgeServer)
    }
    
    wg.Wait()
    close(errors)
    
    // Check for errors
    for err := range errors {
        if err != nil {
            return err
        }
    }
    
    return nil
}
```

### Scenario 2: Design a Real-Time Gaming Platform
**Scale**: 10 million concurrent players, 1000+ games, 50ms latency  
**Requirements**: Real-time synchronization, anti-cheat, matchmaking

#### Architecture Design
```
Gaming Platform Architecture:
├── Game Servers (Dedicated instances per game)
├── Matchmaking Service (Player matching algorithm)
├── Anti-Cheat System (Behavioral analysis)
├── Real-Time Communication (WebSocket, UDP)
├── Game State Management (Redis, Database)
├── Player Management (Authentication, Profiles)
├── Analytics and Telemetry
└── Load Balancing and Scaling
```

#### Game Server Implementation
```go
// Game Server for Real-Time Gaming
type GameServer struct {
    ID          string
    GameID      string
    MaxPlayers  int
    Players     map[string]*Player
    GameState   *GameState
    TickRate    int
    mutex       sync.RWMutex
}

type Player struct {
    ID       string
    Username string
    Position Vector3
    Health   int
    Score    int
    LastSeen time.Time
}

type GameState struct {
    GameID      string
    Status      string
    StartTime   time.Time
    EndTime     time.Time
    WorldState  map[string]interface{}
    Events      []GameEvent
}

func (gs *GameServer) AddPlayer(player *Player) error {
    gs.mutex.Lock()
    defer gs.mutex.Unlock()
    
    if len(gs.Players) >= gs.MaxPlayers {
        return errors.New("server full")
    }
    
    gs.Players[player.ID] = player
    gs.broadcastPlayerJoined(player)
    
    return nil
}

func (gs *GameServer) UpdatePlayerState(playerID string, state *PlayerState) error {
    gs.mutex.Lock()
    defer gs.mutex.Unlock()
    
    player, exists := gs.Players[playerID]
    if !exists {
        return errors.New("player not found")
    }
    
    // Validate state update
    if !gs.validatePlayerState(player, state) {
        return errors.New("invalid state update")
    }
    
    // Update player state
    player.Position = state.Position
    player.Health = state.Health
    player.LastSeen = time.Now()
    
    // Broadcast to other players
    gs.broadcastPlayerUpdate(playerID, state)
    
    return nil
}

func (gs *GameServer) validatePlayerState(player *Player, state *PlayerState) bool {
    // Implement anti-cheat validation
    // Check for impossible movements, health changes, etc.
    
    // Check movement speed
    if gs.calculateDistance(player.Position, state.Position) > gs.maxMovementSpeed {
        return false
    }
    
    // Check health changes
    if state.Health > player.Health {
        return false // Health can't increase without items
    }
    
    return true
}
```

## 🚀 Machine Learning Systems

### Scenario 3: Design a Recommendation System
**Scale**: 100 million users, 1 billion items, 1000 recommendations/second  
**Requirements**: Real-time recommendations, A/B testing, personalization

#### Architecture Design
```
Recommendation System Architecture:
├── Data Pipeline (ETL, Feature Engineering)
├── Model Training (Offline, Online)
├── Model Serving (Real-time inference)
├── Feature Store (User, Item, Context features)
├── A/B Testing Framework
├── Feedback Loop (Implicit, Explicit)
├── Monitoring and Analytics
└── API Gateway and Load Balancing
```

#### Recommendation Engine
```go
// Recommendation Engine Implementation
type RecommendationEngine struct {
    models        map[string]Model
    featureStore  *FeatureStore
    userProfiles  *UserProfileStore
    itemCatalog   *ItemCatalog
    abTestManager *ABTestManager
}

type Model interface {
    Predict(userID string, itemID string, context map[string]interface{}) (float64, error)
    BatchPredict(userID string, itemIDs []string, context map[string]interface{}) (map[string]float64, error)
}

type CollaborativeFilteringModel struct {
    userItemMatrix map[string]map[string]float64
    itemSimilarity map[string]map[string]float64
}

func (cfm *CollaborativeFilteringModel) Predict(userID string, itemID string, context map[string]interface{}) (float64, error) {
    // Get user's rating history
    userRatings, exists := cfm.userItemMatrix[userID]
    if !exists {
        return 0, errors.New("user not found")
    }
    
    // Find similar items
    similarItems := cfm.itemSimilarity[itemID]
    if similarItems == nil {
        return 0, errors.New("item not found")
    }
    
    // Calculate weighted average
    var weightedSum, totalWeight float64
    for similarItem, similarity := range similarItems {
        if rating, exists := userRatings[similarItem]; exists {
            weightedSum += rating * similarity
            totalWeight += similarity
        }
    }
    
    if totalWeight == 0 {
        return 0, errors.New("no similar items found")
    }
    
    return weightedSum / totalWeight, nil
}

func (re *RecommendationEngine) GetRecommendations(userID string, limit int) ([]Recommendation, error) {
    // Get user profile
    profile, err := re.userProfiles.Get(userID)
    if err != nil {
        return nil, err
    }
    
    // Get context features
    context := re.getContextFeatures(userID)
    
    // Get candidate items
    candidates := re.getCandidateItems(userID, limit*3)
    
    // Score candidates
    scores := make(map[string]float64)
    for _, itemID := range candidates {
        score, err := re.scoreItem(userID, itemID, context)
        if err != nil {
            continue
        }
        scores[itemID] = score
    }
    
    // Sort by score and return top recommendations
    return re.rankRecommendations(scores, limit), nil
}

func (re *RecommendationEngine) scoreItem(userID string, itemID string, context map[string]interface{}) (float64, error) {
    // Get A/B test variant
    variant := re.abTestManager.GetVariant(userID)
    
    // Get model for variant
    model, exists := re.models[variant]
    if !exists {
        return 0, errors.New("model not found")
    }
    
    // Get features
    features := re.featureStore.GetFeatures(userID, itemID, context)
    
    // Make prediction
    score, err := model.Predict(userID, itemID, features)
    if err != nil {
        return 0, err
    }
    
    // Apply business rules
    score = re.applyBusinessRules(score, userID, itemID)
    
    return score, nil
}
```

## 🔧 Financial Systems

### Scenario 4: Design a High-Frequency Trading System
**Scale**: 1 million trades/second, 1ms latency, 99.99% uptime  
**Requirements**: Ultra-low latency, fault tolerance, regulatory compliance

#### Architecture Design
```
Trading System Architecture:
├── Market Data Feed (Real-time price feeds)
├── Order Management System (Order routing, execution)
├── Risk Management (Position limits, risk checks)
├── Matching Engine (Order matching algorithm)
├── Settlement System (Trade settlement, clearing)
├── Compliance Engine (Regulatory reporting)
├── Monitoring and Alerting
└── Disaster Recovery
```

#### Trading Engine
```go
// High-Frequency Trading Engine
type TradingEngine struct {
    orderBook    *OrderBook
    riskManager  *RiskManager
    marketData   *MarketDataFeed
    execution    *ExecutionEngine
    compliance   *ComplianceEngine
    latency      *LatencyMonitor
}

type Order struct {
    ID          string
    Symbol      string
    Side        string // "buy" or "sell"
    Quantity    int64
    Price       float64
    OrderType   string // "market", "limit", "stop"
    Timestamp   time.Time
    ClientID    string
}

type OrderBook struct {
    Symbol    string
    BuyOrders []*Order
    SellOrders []*Order
    mutex     sync.RWMutex
}

func (ob *OrderBook) AddOrder(order *Order) error {
    ob.mutex.Lock()
    defer ob.mutex.Unlock()
    
    if order.Side == "buy" {
        ob.BuyOrders = append(ob.BuyOrders, order)
        sort.Slice(ob.BuyOrders, func(i, j int) bool {
            return ob.BuyOrders[i].Price > ob.BuyOrders[j].Price // Descending for buy orders
        })
    } else {
        ob.SellOrders = append(ob.SellOrders, order)
        sort.Slice(ob.SellOrders, func(i, j int) bool {
            return ob.SellOrders[i].Price < ob.SellOrders[j].Price // Ascending for sell orders
        })
    }
    
    return nil
}

func (ob *OrderBook) MatchOrders() []Trade {
    ob.mutex.Lock()
    defer ob.mutex.Unlock()
    
    var trades []Trade
    
    for len(ob.BuyOrders) > 0 && len(ob.SellOrders) > 0 {
        buyOrder := ob.BuyOrders[0]
        sellOrder := ob.SellOrders[0]
        
        if buyOrder.Price >= sellOrder.Price {
            // Match found
            quantity := min(buyOrder.Quantity, sellOrder.Quantity)
            price := (buyOrder.Price + sellOrder.Price) / 2
            
            trade := Trade{
                ID:        generateTradeID(),
                Symbol:    buyOrder.Symbol,
                Quantity:  quantity,
                Price:     price,
                BuyOrder:  buyOrder.ID,
                SellOrder: sellOrder.ID,
                Timestamp: time.Now(),
            }
            
            trades = append(trades, trade)
            
            // Update quantities
            buyOrder.Quantity -= quantity
            sellOrder.Quantity -= quantity
            
            // Remove filled orders
            if buyOrder.Quantity == 0 {
                ob.BuyOrders = ob.BuyOrders[1:]
            }
            if sellOrder.Quantity == 0 {
                ob.SellOrders = ob.SellOrders[1:]
            }
        } else {
            break // No more matches possible
        }
    }
    
    return trades
}

func (te *TradingEngine) ProcessOrder(order *Order) error {
    start := time.Now()
    defer func() {
        te.latency.RecordLatency("order_processing", time.Since(start))
    }()
    
    // Risk checks
    if err := te.riskManager.ValidateOrder(order); err != nil {
        return err
    }
    
    // Compliance checks
    if err := te.compliance.ValidateOrder(order); err != nil {
        return err
    }
    
    // Add to order book
    if err := te.orderBook.AddOrder(order); err != nil {
        return err
    }
    
    // Match orders
    trades := te.orderBook.MatchOrders()
    
    // Execute trades
    for _, trade := range trades {
        if err := te.execution.ExecuteTrade(trade); err != nil {
            return err
        }
    }
    
    return nil
}
```

## 🎯 Best Practices

### Design Principles
1. **Scalability**: Design for horizontal scaling
2. **Availability**: Implement redundancy and failover
3. **Performance**: Optimize for latency and throughput
4. **Consistency**: Choose appropriate consistency models
5. **Security**: Implement comprehensive security measures

### Common Patterns
1. **Microservices**: Break down into independent services
2. **Event-Driven**: Use events for loose coupling
3. **Caching**: Implement multiple levels of caching
4. **Load Balancing**: Distribute load across instances
5. **Database Sharding**: Partition data for horizontal scaling

### Performance Considerations
1. **Latency**: Minimize end-to-end latency
2. **Throughput**: Design for high request rates
3. **Memory**: Optimize memory usage
4. **CPU**: Efficient CPU utilization
5. **Network**: Minimize network overhead

---

**Last Updated**: December 2024  
**Category**: Advanced System Design Scenarios  
**Complexity**: Expert Level
