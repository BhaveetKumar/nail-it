# Advanced System Design Case Studies

Comprehensive advanced system design case studies for senior engineering roles.

## 🎯 Case Study 1: Netflix Streaming Platform

### Business Requirements
- **Scale**: 200+ million subscribers globally
- **Content**: 15,000+ titles, 100+ TB of video content
- **Traffic**: 15% of global internet traffic
- **Availability**: 99.99% uptime
- **Performance**: <2 seconds to start playback

### Technical Challenges
1. **Global Content Distribution**
2. **Adaptive Bitrate Streaming**
3. **Real-time Analytics**
4. **Personalization Engine**
5. **Content Recommendation**

### Architecture Design

#### 1. Content Delivery Network (CDN)
```
Global Edge Locations:
├── North America (50+ locations)
├── Europe (30+ locations)
├── Asia Pacific (25+ locations)
└── Latin America (15+ locations)

Content Distribution:
├── Origin Servers (AWS S3)
├── Regional Caches (Open Connect)
├── Edge Caches (ISP partnerships)
└── Client Caches (Device storage)
```

#### 2. Adaptive Bitrate Streaming
```go
// Adaptive Bitrate Streaming Implementation
type AdaptiveStreaming struct {
    segments    []VideoSegment
    qualities   []QualityLevel
    algorithm   *AdaptiveAlgorithm
    player      *VideoPlayer
}

type VideoSegment struct {
    ID          string
    Quality     QualityLevel
    Bitrate     int
    Resolution  string
    Duration    time.Duration
    URL         string
}

type QualityLevel struct {
    Name        string
    Bitrate     int
    Resolution  string
    Bandwidth   int
    BufferSize  int
}

func (as *AdaptiveStreaming) SelectQuality(networkSpeed int, bufferLevel float64) QualityLevel {
    // Network-based selection
    if networkSpeed < 1000 { // < 1 Mbps
        return as.qualities[0] // Lowest quality
    } else if networkSpeed < 5000 { // < 5 Mbps
        return as.qualities[1] // Medium quality
    } else {
        return as.qualities[2] // Highest quality
    }
}

func (as *AdaptiveStreaming) StreamVideo(segmentID string) error {
    // Get current network conditions
    networkSpeed := as.getNetworkSpeed()
    bufferLevel := as.player.GetBufferLevel()
    
    // Select appropriate quality
    quality := as.SelectQuality(networkSpeed, bufferLevel)
    
    // Download segment
    segment, err := as.downloadSegment(segmentID, quality)
    if err != nil {
        return err
    }
    
    // Add to player buffer
    return as.player.AddSegment(segment)
}
```

#### 3. Personalization Engine
```go
// Personalization Engine Implementation
type PersonalizationEngine struct {
    userProfiles    map[string]*UserProfile
    contentCatalog  *ContentCatalog
    recommendation  *RecommendationEngine
    analytics       *AnalyticsService
}

type UserProfile struct {
    UserID          string
    ViewingHistory  []ViewingEvent
    Preferences     map[string]float64
    Demographics    Demographics
    DeviceInfo      DeviceInfo
    LastUpdated     time.Time
}

type ViewingEvent struct {
    ContentID       string
    Timestamp       time.Time
    Duration        time.Duration
    CompletionRate  float64
    Quality         string
    Device          string
}

func (pe *PersonalizationEngine) GetRecommendations(userID string, limit int) ([]Content, error) {
    // Get user profile
    profile, exists := pe.userProfiles[userID]
    if !exists {
        return pe.getDefaultRecommendations(limit)
    }
    
    // Calculate content scores
    scores := make(map[string]float64)
    for contentID, content := range pe.contentCatalog.GetAll() {
        score := pe.calculateContentScore(profile, content)
        scores[contentID] = score
    }
    
    // Sort by score and return top recommendations
    return pe.getTopRecommendations(scores, limit)
}

func (pe *PersonalizationEngine) calculateContentScore(profile *UserProfile, content *Content) float64 {
    score := 0.0
    
    // Genre preference
    for genre, preference := range profile.Preferences {
        if content.HasGenre(genre) {
            score += preference * 0.3
        }
    }
    
    // Viewing history similarity
    similarContent := pe.findSimilarContent(profile.ViewingHistory, content)
    score += similarContent * 0.4
    
    // Trending content boost
    if content.IsTrending() {
        score += 0.2
    }
    
    // Recency boost
    if content.IsRecent() {
        score += 0.1
    }
    
    return score
}
```

### Key Design Decisions
1. **Microservices Architecture**: Each service handles specific functionality
2. **Event-Driven Architecture**: Asynchronous communication between services
3. **Caching Strategy**: Multi-level caching for performance
4. **Database Sharding**: Horizontal partitioning for scalability
5. **Global Distribution**: Edge locations for low latency

## 🎯 Case Study 2: Uber Real-time System

### Business Requirements
- **Scale**: 100+ million users, 5+ million drivers
- **Real-time**: <3 seconds for ride matching
- **Global**: 10,000+ cities worldwide
- **Availability**: 99.9% uptime
- **Performance**: Handle 1M+ requests per second

### Technical Challenges
1. **Real-time Location Tracking**
2. **Dynamic Pricing**
3. **Ride Matching Algorithm**
4. **Global Scale**
5. **High Availability**

### Architecture Design

#### 1. Real-time Location Tracking
```go
// Location Tracking Service
type LocationTrackingService struct {
    redis        *redis.Client
    kafka        *kafka.Producer
    geohash      *GeohashService
    subscribers  map[string][]LocationSubscriber
    mutex        sync.RWMutex
}

type LocationUpdate struct {
    UserID      string
    Latitude    float64
    Longitude   float64
    Timestamp   time.Time
    Accuracy    float64
    Speed       float64
    Bearing     float64
}

func (lts *LocationTrackingService) UpdateLocation(update *LocationUpdate) error {
    // Store in Redis with TTL
    key := fmt.Sprintf("location:%s", update.UserID)
    data, _ := json.Marshal(update)
    
    err := lts.redis.Set(key, data, 5*time.Minute).Err()
    if err != nil {
        return err
    }
    
    // Update geohash
    geohash := lts.geohash.Encode(update.Latitude, update.Longitude)
    lts.geohash.AddLocation(geohash, update.UserID)
    
    // Publish to Kafka for real-time processing
    return lts.kafka.Produce("location-updates", update)
}

func (lts *LocationTrackingService) GetNearbyUsers(lat, lng float64, radius float64) ([]string, error) {
    // Get geohash for the location
    geohash := lts.geohash.Encode(lat, lng)
    
    // Get nearby geohashes
    nearbyHashes := lts.geohash.GetNearby(geohash, radius)
    
    var userIDs []string
    for _, hash := range nearbyHashes {
        users := lts.geohash.GetUsers(hash)
        userIDs = append(userIDs, users...)
    }
    
    return userIDs, nil
}
```

#### 2. Dynamic Pricing Engine
```go
// Dynamic Pricing Engine
type PricingEngine struct {
    basePrice    float64
    multipliers  map[string]float64
    analytics    *AnalyticsService
    mutex        sync.RWMutex
}

type PricingFactors struct {
    Demand       float64
    Supply       float64
    TimeOfDay    int
    DayOfWeek    int
    Weather      string
    Events       []string
    Traffic      float64
}

func (pe *PricingEngine) CalculatePrice(ride *Ride, factors *PricingFactors) float64 {
    pe.mutex.RLock()
    defer pe.mutex.RUnlock()
    
    price := pe.basePrice
    
    // Demand multiplier
    if factors.Demand > 1.5 {
        price *= 1.5
    } else if factors.Demand > 1.2 {
        price *= 1.2
    }
    
    // Supply multiplier
    if factors.Supply < 0.5 {
        price *= 1.3
    } else if factors.Supply < 0.8 {
        price *= 1.1
    }
    
    // Time-based multiplier
    if factors.TimeOfDay >= 22 || factors.TimeOfDay <= 6 {
        price *= 1.2 // Night surcharge
    }
    
    // Weather multiplier
    if factors.Weather == "rain" || factors.Weather == "snow" {
        price *= 1.3
    }
    
    // Event multiplier
    for _, event := range factors.Events {
        if event == "concert" || event == "sports" {
            price *= 1.4
        }
    }
    
    // Traffic multiplier
    if factors.Traffic > 0.7 {
        price *= 1.2
    }
    
    return price
}
```

#### 3. Ride Matching Algorithm
```go
// Ride Matching Service
type RideMatchingService struct {
    locationService *LocationTrackingService
    pricingEngine   *PricingEngine
    matchingQueue   chan *RideRequest
    driverPool      map[string]*Driver
    mutex           sync.RWMutex
}

type RideRequest struct {
    ID            string
    UserID        string
    PickupLat     float64
    PickupLng     float64
    DropoffLat    float64
    DropoffLng    float64
    RequestTime   time.Time
    MaxWaitTime   time.Duration
    Preferences   map[string]interface{}
}

type Driver struct {
    ID            string
    CurrentLat    float64
    CurrentLng    float64
    Status        string
    Rating        float64
    VehicleType   string
    LastUpdate    time.Time
}

func (rms *RideMatchingService) FindDriver(request *RideRequest) (*Driver, error) {
    // Get nearby drivers
    nearbyDrivers, err := rms.locationService.GetNearbyUsers(
        request.PickupLat, request.PickupLng, 5.0) // 5km radius
    if err != nil {
        return nil, err
    }
    
    // Filter available drivers
    availableDrivers := rms.filterAvailableDrivers(nearbyDrivers)
    if len(availableDrivers) == 0 {
        return nil, errors.New("no available drivers")
    }
    
    // Calculate scores for each driver
    scores := make(map[string]float64)
    for _, driverID := range availableDrivers {
        driver := rms.driverPool[driverID]
        score := rms.calculateDriverScore(request, driver)
        scores[driverID] = score
    }
    
    // Select best driver
    bestDriverID := rms.selectBestDriver(scores)
    return rms.driverPool[bestDriverID], nil
}

func (rms *RideMatchingService) calculateDriverScore(request *RideRequest, driver *Driver) float64 {
    score := 0.0
    
    // Distance factor (closer is better)
    distance := rms.calculateDistance(
        request.PickupLat, request.PickupLng,
        driver.CurrentLat, driver.CurrentLng)
    score += 1.0 / (1.0 + distance) // Inverse distance
    
    // Rating factor
    score += driver.Rating * 0.3
    
    // Response time factor
    responseTime := time.Since(driver.LastUpdate)
    if responseTime < 30*time.Second {
        score += 0.2
    }
    
    // Vehicle type preference
    if request.Preferences["vehicle_type"] == driver.VehicleType {
        score += 0.1
    }
    
    return score
}
```

### Key Design Decisions
1. **Event-Driven Architecture**: Real-time updates via Kafka
2. **Geospatial Indexing**: Efficient location-based queries
3. **Caching Strategy**: Redis for fast location lookups
4. **Microservices**: Separate services for different functionalities
5. **Global Distribution**: Multi-region deployment

## 🎯 Case Study 3: WhatsApp Messaging System

### Business Requirements
- **Scale**: 2+ billion users globally
- **Messages**: 100+ billion messages per day
- **Real-time**: <1 second message delivery
- **Availability**: 99.9% uptime
- **Security**: End-to-end encryption

### Technical Challenges
1. **Message Delivery**
2. **End-to-End Encryption**
3. **Group Messaging**
4. **Media Sharing**
5. **Global Scale**

### Architecture Design

#### 1. Message Delivery System
```go
// Message Delivery Service
type MessageDeliveryService struct {
    messageQueue  *MessageQueue
    userSessions  map[string]*UserSession
    webSocketHub  *WebSocketHub
    mutex         sync.RWMutex
}

type Message struct {
    ID          string
    From        string
    To          string
    Content     string
    Type        string
    Timestamp   time.Time
    Status      string
    Encryption  *EncryptionInfo
}

type UserSession struct {
    UserID      string
    Connection  *websocket.Conn
    LastSeen    time.Time
    Status      string
}

func (mds *MessageDeliveryService) SendMessage(message *Message) error {
    // Store message in queue
    if err := mds.messageQueue.Enqueue(message); err != nil {
        return err
    }
    
    // Check if recipient is online
    session, exists := mds.userSessions[message.To]
    if exists && session.Status == "online" {
        // Send via WebSocket
        return mds.sendViaWebSocket(session, message)
    } else {
        // Send via push notification
        return mds.sendViaPushNotification(message)
    }
}

func (mds *MessageDeliveryService) sendViaWebSocket(session *UserSession, message *Message) error {
    data, err := json.Marshal(message)
    if err != nil {
        return err
    }
    
    return session.Connection.WriteMessage(websocket.TextMessage, data)
}
```

#### 2. End-to-End Encryption
```go
// End-to-End Encryption Service
type EncryptionService struct {
    keyStore     *KeyStore
    crypto       *CryptoService
    keyExchange  *KeyExchangeService
}

type EncryptionInfo struct {
    Algorithm   string
    KeyID       string
    IV          []byte
    Ciphertext  []byte
    Signature   []byte
}

func (es *EncryptionService) EncryptMessage(message *Message, recipientID string) (*EncryptionInfo, error) {
    // Get recipient's public key
    publicKey, err := es.keyStore.GetPublicKey(recipientID)
    if err != nil {
        return nil, err
    }
    
    // Generate random key for this message
    messageKey := es.crypto.GenerateRandomKey(32)
    
    // Encrypt message content
    ciphertext, iv, err := es.crypto.EncryptAES([]byte(message.Content), messageKey)
    if err != nil {
        return nil, err
    }
    
    // Encrypt message key with recipient's public key
    encryptedKey, err := es.crypto.EncryptRSA(messageKey, publicKey)
    if err != nil {
        return nil, err
    }
    
    // Create signature
    signature, err := es.crypto.Sign(ciphertext, es.keyStore.GetPrivateKey(message.From))
    if err != nil {
        return nil, err
    }
    
    return &EncryptionInfo{
        Algorithm:  "AES-256-GCM",
        KeyID:      publicKey.ID,
        IV:         iv,
        Ciphertext: ciphertext,
        Signature:  signature,
    }, nil
}

func (es *EncryptionService) DecryptMessage(encryptionInfo *EncryptionInfo, recipientID string) (string, error) {
    // Get recipient's private key
    privateKey, err := es.keyStore.GetPrivateKey(recipientID)
    if err != nil {
        return "", err
    }
    
    // Decrypt message key
    messageKey, err := es.crypto.DecryptRSA(encryptionInfo.Ciphertext, privateKey)
    if err != nil {
        return "", err
    }
    
    // Decrypt message content
    plaintext, err := es.crypto.DecryptAES(encryptionInfo.Ciphertext, messageKey, encryptionInfo.IV)
    if err != nil {
        return "", err
    }
    
    // Verify signature
    if !es.crypto.Verify(plaintext, encryptionInfo.Signature, es.keyStore.GetPublicKey(encryptionInfo.KeyID)) {
        return "", errors.New("signature verification failed")
    }
    
    return string(plaintext), nil
}
```

#### 3. Group Messaging
```go
// Group Messaging Service
type GroupMessagingService struct {
    groupStore   *GroupStore
    memberStore  *MemberStore
    messageQueue *MessageQueue
    mutex        sync.RWMutex
}

type Group struct {
    ID          string
    Name        string
    Description string
    Admin       string
    Members     []string
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

type GroupMessage struct {
    ID        string
    GroupID   string
    From      string
    Content   string
    Type      string
    Timestamp time.Time
    Status    string
}

func (gms *GroupMessagingService) SendGroupMessage(message *GroupMessage) error {
    // Get group members
    group, err := gms.groupStore.GetGroup(message.GroupID)
    if err != nil {
        return err
    }
    
    // Send to all members except sender
    for _, memberID := range group.Members {
        if memberID != message.From {
            // Create individual message
            individualMessage := &Message{
                ID:        generateMessageID(),
                From:      message.From,
                To:        memberID,
                Content:   message.Content,
                Type:      message.Type,
                Timestamp: message.Timestamp,
                Status:    "pending",
            }
            
            // Send message
            if err := gms.messageQueue.Enqueue(individualMessage); err != nil {
                return err
            }
        }
    }
    
    return nil
}
```

### Key Design Decisions
1. **WebSocket Connections**: Real-time bidirectional communication
2. **Message Queuing**: Reliable message delivery
3. **End-to-End Encryption**: Signal protocol for security
4. **Database Sharding**: Horizontal partitioning for scale
5. **Global Distribution**: Multi-region deployment

## 🎯 Best Practices for System Design Case Studies

### Design Principles
1. **Start with Requirements**: Clarify functional and non-functional requirements
2. **Think About Scale**: Consider current and future scale requirements
3. **Design for Failure**: Plan for component failures and recovery
4. **Security First**: Consider security implications from the start
5. **Monitor Everything**: Design for observability and monitoring

### Common Patterns
1. **Microservices**: Break down into independent services
2. **Event-Driven**: Use events for loose coupling
3. **Caching**: Implement multiple levels of caching
4. **Load Balancing**: Distribute load across multiple instances
5. **Database Sharding**: Partition data for horizontal scaling

### Performance Considerations
1. **Latency**: Minimize end-to-end latency
2. **Throughput**: Design for high request rates
3. **Availability**: Ensure high uptime
4. **Scalability**: Plan for growth
5. **Efficiency**: Optimize resource usage

---

**Last Updated**: December 2024  
**Category**: Advanced System Design Case Studies  
**Complexity**: Expert Level
