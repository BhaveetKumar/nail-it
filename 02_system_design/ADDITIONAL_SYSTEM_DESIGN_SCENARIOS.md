# 🏗️ Additional System Design Scenarios

> **Comprehensive system design scenarios for technical interviews**

## 📚 Table of Contents

1. [Content Delivery Network (CDN)](#1-content-delivery-network-cdn)
2. [Code Collaboration Platform](#2-code-collaboration-platform)
3. [URL Shortening Service](#3-url-shortening-service)
4. [Traffic Control System](#4-traffic-control-system)
5. [Search Typeahead System](#5-search-typeahead-system)
6. [Web Crawler](#6-web-crawler)
7. [Elevator System](#7-elevator-system)
8. [Clustered Caching System](#8-clustered-caching-system)
9. [Train Search Functionality for IRCTC](#9-train-search-functionality-for-irctc)
10. [Image Quality Analysis from URL](#10-image-quality-analysis-from-url)
11. [Ridesharing Platform like Uber](#11-ridesharing-platform-like-uber)
12. [LRU Cache](#12-lru-cache)
13. [Chat System for Amazon Returns](#13-chat-system-for-amazon-returns)
14. [Modular Snake and Ladder Game](#14-modular-snake-and-ladder-game)
15. [Yelp-like Review System](#15-yelp-like-review-system)
16. [WhatsApp-like Messaging Service](#16-whatsapp-like-messaging-service)
17. [Class Diagram for Chess Board Game](#17-class-diagram-for-chess-board-game)

---

## 1. Design a Content Delivery Network (CDN)

### Problem Statement
Design a CDN system that can serve static content (images, videos, CSS, JS) to users globally with low latency and high availability.

### Requirements
- **Functional Requirements**
  - Serve static content (images, videos, CSS, JS)
  - Global distribution with edge servers
  - Cache management and invalidation
  - Content compression and optimization
  - SSL/TLS termination

- **Non-Functional Requirements**
  - Low latency (< 100ms)
  - High availability (99.99%)
  - Global scale (millions of users)
  - High throughput (millions of requests/second)

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    CDN Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Users  →  Edge Servers  →  Origin Servers  →  Storage    │
│    │           │              │                │           │
│    │           │              │                │           │
│    ▼           ▼              ▼                ▼           │
│  Global    Cache Layer    Load Balancer    S3/CloudFS     │
│  Users     (POPs)         Origin Servers   Content Store   │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Edge Servers (Points of Presence)
```go
type EdgeServer struct {
    ID          string
    Location    string
    Cache       *Cache
    LoadBalancer *LoadBalancer
    SSLTerminator *SSLTerminator
}

type Cache struct {
    Memory      map[string]*CachedContent
    Disk        *DiskCache
    TTL         time.Duration
    MaxSize     int64
    EvictionPolicy string
}

type CachedContent struct {
    Data        []byte
    ContentType string
    LastModified time.Time
    ETag        string
    TTL         time.Duration
    Size        int64
}
```

#### 2. Origin Servers
```go
type OriginServer struct {
    ID          string
    LoadBalancer *LoadBalancer
    HealthCheck *HealthCheck
    SSLConfig   *SSLConfig
}

type LoadBalancer struct {
    Algorithm   string // round-robin, least-connections, weighted
    Servers     []*Server
    HealthCheck *HealthCheck
}
```

#### 3. Cache Management
```go
type CacheManager struct {
    EdgeServers map[string]*EdgeServer
    InvalidationQueue chan *InvalidationRequest
    Analytics   *Analytics
}

type InvalidationRequest struct {
    URL         string
    Pattern     string
    Timestamp   time.Time
    Priority    int
}

func (cm *CacheManager) InvalidateContent(url string) error {
    // Invalidate content across all edge servers
    for _, server := range cm.EdgeServers {
        server.Cache.Invalidate(url)
    }
    return nil
}
```

### Detailed Design

#### 1. Request Flow
```go
func (es *EdgeServer) HandleRequest(req *http.Request) (*http.Response, error) {
    // 1. Check cache
    if cached := es.Cache.Get(req.URL.Path); cached != nil {
        return es.ServeFromCache(cached), nil
    }
    
    // 2. Check if content exists in other edge servers
    if content := es.GetFromPeerEdgeServers(req.URL.Path); content != nil {
        es.Cache.Set(req.URL.Path, content)
        return es.ServeFromCache(content), nil
    }
    
    // 3. Fetch from origin server
    content, err := es.FetchFromOrigin(req)
    if err != nil {
        return nil, err
    }
    
    // 4. Cache and serve
    es.Cache.Set(req.URL.Path, content)
    return es.ServeFromCache(content), nil
}
```

#### 2. Cache Invalidation
```go
type InvalidationService struct {
    MessageQueue *MessageQueue
    EdgeServers  map[string]*EdgeServer
}

func (is *InvalidationService) InvalidateContent(url string) error {
    invalidationReq := &InvalidationRequest{
        URL:       url,
        Timestamp: time.Now(),
        Priority:  1,
    }
    
    // Send to all edge servers
    for _, server := range is.EdgeServers {
        server.InvalidationQueue <- invalidationReq
    }
    
    return nil
}
```

#### 3. Content Optimization
```go
type ContentOptimizer struct {
    Compressors map[string]Compressor
    ImageOptimizer *ImageOptimizer
    Minifier    *Minifier
}

func (co *ContentOptimizer) OptimizeContent(content []byte, contentType string) ([]byte, error) {
    switch contentType {
    case "image/jpeg", "image/png":
        return co.ImageOptimizer.Optimize(content)
    case "text/css", "application/javascript":
        return co.Minifier.Minify(content)
    default:
        return content, nil
    }
}
```

### Scalability Considerations

#### 1. Edge Server Distribution
- Deploy edge servers in major cities worldwide
- Use Anycast routing for automatic failover
- Implement health checks and automatic replacement

#### 2. Cache Management
- Use LRU eviction policy for memory cache
- Implement cache warming for popular content
- Use cache hierarchies (L1, L2, L3)

#### 3. Load Balancing
- Use geographic load balancing
- Implement health checks
- Use weighted round-robin based on server capacity

### Security Considerations

#### 1. SSL/TLS Termination
```go
type SSLTerminator struct {
    Certificates map[string]*tls.Certificate
    SNIEnabled   bool
}

func (st *SSLTerminator) GetCertificate(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error) {
    if cert, exists := st.Certificates[clientHello.ServerName]; exists {
        return cert, nil
    }
    return st.Certificates["default"], nil
}
```

#### 2. DDoS Protection
- Rate limiting per IP
- CAPTCHA for suspicious requests
- Blacklisting malicious IPs

### Monitoring and Analytics

```go
type CDNAnalytics struct {
    Metrics     *MetricsCollector
    Logs        *LogCollector
    Alerts      *AlertManager
}

type Metrics struct {
    RequestCount    int64
    CacheHitRate   float64
    ResponseTime   time.Duration
    ErrorRate      float64
    BandwidthUsage int64
}
```

---

## 2. Design a Code Collaboration Platform

### Problem Statement
Design a platform like GitHub that allows developers to collaborate on code, manage repositories, and track changes.

### Requirements
- **Functional Requirements**
  - Repository management (create, clone, fork)
  - Version control (Git operations)
  - Code review and pull requests
  - Issue tracking and project management
  - User authentication and authorization
  - Real-time collaboration features

- **Non-Functional Requirements**
  - High availability (99.9%)
  - Scalable to millions of repositories
  - Low latency for Git operations
  - Data consistency and integrity

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│              Code Collaboration Platform                   │
├─────────────────────────────────────────────────────────────┤
│  Web UI  →  API Gateway  →  Microservices  →  Databases   │
│    │           │              │                │           │
│    │           │              │                │           │
│    ▼           ▼              ▼                ▼           │
│  React     Load Balancer   Auth Service    PostgreSQL     │
│  Angular   Rate Limiter    Repo Service    Redis Cache    │
│  Vue.js    API Gateway     Git Service     File Storage   │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Repository Service
```go
type RepositoryService struct {
    DB          *gorm.DB
    GitStorage  *GitStorage
    Cache       *redis.Client
    FileStorage *FileStorage
}

type Repository struct {
    ID          uint      `json:"id"`
    Name        string    `json:"name"`
    Description string    `json:"description"`
    OwnerID     uint      `json:"owner_id"`
    IsPrivate   bool      `json:"is_private"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

func (rs *RepositoryService) CreateRepository(userID uint, req *CreateRepoRequest) (*Repository, error) {
    repo := &Repository{
        Name:        req.Name,
        Description: req.Description,
        OwnerID:     userID,
        IsPrivate:   req.IsPrivate,
    }
    
    if err := rs.DB.Create(repo).Error; err != nil {
        return nil, err
    }
    
    // Initialize Git repository
    if err := rs.GitStorage.InitRepository(repo.ID); err != nil {
        return nil, err
    }
    
    return repo, nil
}
```

#### 2. Git Service
```go
type GitService struct {
    Storage     *GitStorage
    LFS         *GitLFS
    Hooks       *GitHooks
}

type GitOperation struct {
    RepositoryID uint
    Operation    string // push, pull, clone, fetch
    UserID       uint
    Timestamp    time.Time
    Branch       string
    CommitHash   string
}

func (gs *GitService) Push(repoID uint, userID uint, branch string, commits []Commit) error {
    // Validate permissions
    if !gs.HasWritePermission(repoID, userID) {
        return errors.New("insufficient permissions")
    }
    
    // Process commits
    for _, commit := range commits {
        if err := gs.Storage.StoreCommit(repoID, commit); err != nil {
            return err
        }
    }
    
    // Update branch reference
    if err := gs.Storage.UpdateBranch(repoID, branch, commits[len(commits)-1].Hash); err != nil {
        return err
    }
    
    // Trigger webhooks
    gs.Hooks.TriggerPush(repoID, branch, commits)
    
    return nil
}
```

#### 3. Pull Request Service
```go
type PullRequestService struct {
    DB          *gorm.DB
    GitService  *GitService
    Notifier    *NotificationService
}

type PullRequest struct {
    ID          uint      `json:"id"`
    RepositoryID uint     `json:"repository_id"`
    Title       string    `json:"title"`
    Description string    `json:"description"`
    SourceBranch string   `json:"source_branch"`
    TargetBranch string   `json:"target_branch"`
    AuthorID    uint      `json:"author_id"`
    Status      string    `json:"status"` // open, closed, merged
    CreatedAt   time.Time `json:"created_at"`
}

func (prs *PullRequestService) CreatePullRequest(req *CreatePRRequest) (*PullRequest, error) {
    pr := &PullRequest{
        RepositoryID:  req.RepositoryID,
        Title:        req.Title,
        Description:  req.Description,
        SourceBranch: req.SourceBranch,
        TargetBranch: req.TargetBranch,
        AuthorID:     req.AuthorID,
        Status:       "open",
    }
    
    if err := prs.DB.Create(pr).Error; err != nil {
        return nil, err
    }
    
    // Notify reviewers
    prs.Notifier.NotifyPullRequestCreated(pr)
    
    return pr, nil
}
```

### Detailed Design

#### 1. Real-time Collaboration
```go
type CollaborationService struct {
    Hub         *Hub
    Redis       *redis.Client
    GitService  *GitService
}

type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
}

func (h *Hub) Run() {
    for {
        select {
        case client := <-h.register:
            h.clients[client] = true
            
        case client := <-h.unregister:
            if _, ok := h.clients[client]; ok {
                delete(h.clients, client)
                close(client.send)
            }
            
        case message := <-h.broadcast:
            for client := range h.clients {
                select {
                case client.send <- message:
                default:
                    close(client.send)
                    delete(h.clients, client)
                }
            }
        }
    }
}
```

#### 2. Code Review System
```go
type CodeReviewService struct {
    DB          *gorm.DB
    GitService  *GitService
    Notifier    *NotificationService
}

type CodeReview struct {
    ID            uint      `json:"id"`
    PullRequestID uint      `json:"pull_request_id"`
    ReviewerID    uint      `json:"reviewer_id"`
    FilePath      string    `json:"file_path"`
    LineNumber    int       `json:"line_number"`
    Comment       string    `json:"comment"`
    Status        string    `json:"status"` // pending, approved, changes_requested
    CreatedAt     time.Time `json:"created_at"`
}

func (crs *CodeReviewService) AddReviewComment(prID uint, req *AddCommentRequest) error {
    review := &CodeReview{
        PullRequestID: prID,
        ReviewerID:    req.ReviewerID,
        FilePath:      req.FilePath,
        LineNumber:    req.LineNumber,
        Comment:       req.Comment,
        Status:        "pending",
    }
    
    if err := crs.DB.Create(review).Error; err != nil {
        return err
    }
    
    // Notify PR author
    crs.Notifier.NotifyReviewComment(prID, review)
    
    return nil
}
```

### Scalability Considerations

#### 1. Repository Storage
- Use distributed file system (HDFS, S3)
- Implement Git LFS for large files
- Use CDN for static assets

#### 2. Database Sharding
- Shard by repository ID
- Use consistent hashing
- Implement cross-shard queries

#### 3. Caching Strategy
- Cache repository metadata in Redis
- Use CDN for static content
- Implement query result caching

---

## 3. Design a URL Shortening Service

### Problem Statement
Design a service like bit.ly that can shorten long URLs and redirect users to the original URL when they visit the shortened version.

### Requirements
- **Functional Requirements**
  - Shorten long URLs to short URLs
  - Redirect short URLs to original URLs
  - Custom short URLs (optional)
  - Analytics and click tracking
  - URL expiration (optional)

- **Non-Functional Requirements**
  - High availability (99.9%)
  - Low latency (< 100ms)
  - Scalable to billions of URLs
  - URL uniqueness guarantee

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                URL Shortening Service                      │
├─────────────────────────────────────────────────────────────┤
│  Users  →  Load Balancer  →  API Service  →  Database     │
│    │           │              │                │           │
│    │           │              │                │           │
│    ▼           ▼              ▼                ▼           │
│  Browser   Rate Limiter    Shortener      PostgreSQL      │
│  Mobile    Cache Layer     Analytics      Redis Cache     │
│  API       URL Service     Click Tracker  Click Analytics │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. URL Shortening Service
```go
type URLShortener struct {
    DB          *gorm.DB
    Cache       *redis.Client
    Analytics   *AnalyticsService
    Counter     *CounterService
}

type ShortURL struct {
    ID          uint      `json:"id"`
    ShortCode   string    `json:"short_code"`
    OriginalURL string    `json:"original_url"`
    UserID      uint      `json:"user_id"`
    CreatedAt   time.Time `json:"created_at"`
    ExpiresAt   *time.Time `json:"expires_at"`
    ClickCount  int64     `json:"click_count"`
}

func (us *URLShortener) ShortenURL(originalURL string, userID uint) (*ShortURL, error) {
    // Generate unique short code
    shortCode, err := us.generateShortCode()
    if err != nil {
        return nil, err
    }
    
    shortURL := &ShortURL{
        ShortCode:   shortCode,
        OriginalURL: originalURL,
        UserID:      userID,
        CreatedAt:   time.Now(),
    }
    
    // Save to database
    if err := us.DB.Create(shortURL).Error; err != nil {
        return nil, err
    }
    
    // Cache the mapping
    us.Cache.Set(fmt.Sprintf("short:%s", shortCode), originalURL, 24*time.Hour)
    
    return shortURL, nil
}
```

#### 2. Short Code Generation
```go
type CounterService struct {
    DB    *gorm.DB
    Cache *redis.Client
}

func (cs *CounterService) GetNextCounter() (int64, error) {
    // Use Redis atomic increment for performance
    counter, err := cs.Cache.Incr("url_counter").Result()
    if err != nil {
        return 0, err
    }
    
    return counter, nil
}

func (us *URLShortener) generateShortCode() (string, error) {
    counter, err := us.Counter.GetNextCounter()
    if err != nil {
        return "", err
    }
    
    // Convert counter to base62 string
    return us.encodeBase62(counter), nil
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
```

#### 3. URL Redirection Service
```go
type RedirectService struct {
    Cache       *redis.Client
    DB          *gorm.DB
    Analytics   *AnalyticsService
}

func (rs *RedirectService) Redirect(shortCode string) (string, error) {
    // Check cache first
    if originalURL := rs.Cache.Get(fmt.Sprintf("short:%s", shortCode)).Val(); originalURL != "" {
        rs.Analytics.TrackClick(shortCode)
        return originalURL, nil
    }
    
    // Check database
    var shortURL ShortURL
    if err := rs.DB.Where("short_code = ?", shortCode).First(&shortURL).Error; err != nil {
        return "", errors.New("URL not found")
    }
    
    // Check expiration
    if shortURL.ExpiresAt != nil && time.Now().After(*shortURL.ExpiresAt) {
        return "", errors.New("URL expired")
    }
    
    // Cache the result
    rs.Cache.Set(fmt.Sprintf("short:%s", shortCode), shortURL.OriginalURL, 24*time.Hour)
    
    // Track click
    rs.Analytics.TrackClick(shortCode)
    
    return shortURL.OriginalURL, nil
}
```

### Detailed Design

#### 1. Analytics Service
```go
type AnalyticsService struct {
    DB      *gorm.DB
    Queue   *MessageQueue
}

type ClickEvent struct {
    ShortCode   string    `json:"short_code"`
    IPAddress   string    `json:"ip_address"`
    UserAgent   string    `json:"user_agent"`
    Referer     string    `json:"referer"`
    Timestamp   time.Time `json:"timestamp"`
    Country     string    `json:"country"`
    City        string    `json:"city"`
}

func (as *AnalyticsService) TrackClick(shortCode string) {
    event := &ClickEvent{
        ShortCode: shortCode,
        Timestamp: time.Now(),
    }
    
    // Send to message queue for async processing
    as.Queue.Publish("click_events", event)
}
```

#### 2. Rate Limiting
```go
type RateLimiter struct {
    Cache *redis.Client
}

func (rl *RateLimiter) IsAllowed(userID uint, limit int, window time.Duration) bool {
    key := fmt.Sprintf("rate_limit:%d", userID)
    
    // Use sliding window counter
    current := rl.Cache.Incr(key).Val()
    if current == 1 {
        rl.Cache.Expire(key, window)
    }
    
    return current <= int64(limit)
}
```

### Scalability Considerations

#### 1. Database Sharding
- Shard by short code hash
- Use consistent hashing
- Implement cross-shard queries

#### 2. Caching Strategy
- Cache popular URLs in Redis
- Use CDN for global distribution
- Implement cache warming

#### 3. Load Balancing
- Use round-robin load balancing
- Implement health checks
- Use geographic load balancing

---

## 4. Design a Traffic Control System

### Problem Statement
Design a traffic control system that can manage traffic lights, monitor traffic flow, and optimize traffic patterns in real-time.

### Requirements
- **Functional Requirements**
  - Control traffic lights at intersections
  - Monitor traffic flow and density
  - Optimize traffic patterns based on real-time data
  - Emergency vehicle priority
  - Pedestrian crossing management
  - Traffic violation detection

- **Non-Functional Requirements**
  - Real-time response (< 1 second)
  - High availability (99.99%)
  - Scalable to thousands of intersections
  - Fault tolerance and redundancy

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                Traffic Control System                      │
├─────────────────────────────────────────────────────────────┤
│  Sensors  →  Edge Gateway  →  Control Center  →  Actuators │
│    │           │              │                │           │
│    │           │              │                │           │
│    ▼           ▼              ▼                ▼           │
│  Cameras    Data Processor  AI Controller   Traffic Lights │
│  Detectors  Local Cache     ML Models       Sign Boards    │
│  GPS Data   Message Queue   Analytics       Barriers       │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Traffic Light Controller
```go
type TrafficLightController struct {
    ID          string
    IntersectionID string
    Lights      map[string]*TrafficLight
    Sensors     []*Sensor
    State       *TrafficState
    Timer       *Timer
}

type TrafficLight struct {
    ID          string
    Direction   string // north, south, east, west
    Color       string // red, yellow, green
    Duration    time.Duration
    Priority    int
}

type TrafficState struct {
    CurrentPhase string
    PhaseStart   time.Time
    PhaseDuration time.Duration
    NextPhase    string
    EmergencyMode bool
}

func (tlc *TrafficLightController) UpdateTrafficLights() {
    if tlc.State.EmergencyMode {
        tlc.handleEmergency()
        return
    }
    
    // Check if phase should change
    if time.Since(tlc.State.PhaseStart) >= tlc.State.PhaseDuration {
        tlc.changePhase()
    }
    
    // Update light colors based on current phase
    tlc.updateLightColors()
}
```

#### 2. Sensor Data Processing
```go
type SensorDataProcessor struct {
    Sensors     map[string]*Sensor
    DataQueue   chan *SensorData
    Analytics   *TrafficAnalytics
}

type SensorData struct {
    SensorID    string
    Timestamp   time.Time
    VehicleCount int
    Speed       float64
    Density     float64
    Direction   string
}

type Sensor struct {
    ID          string
    Type        string // camera, radar, inductive_loop
    Location    *Location
    Status      string
    LastUpdate  time.Time
}

func (sdp *SensorDataProcessor) ProcessSensorData() {
    for data := range sdp.DataQueue {
        // Validate sensor data
        if !sdp.validateData(data) {
            continue
        }
        
        // Update analytics
        sdp.Analytics.UpdateTrafficData(data)
        
        // Check for anomalies
        if sdp.Analytics.DetectAnomaly(data) {
            sdp.handleAnomaly(data)
        }
        
        // Update traffic model
        sdp.updateTrafficModel(data)
    }
}
```

#### 3. AI Traffic Optimizer
```go
type TrafficOptimizer struct {
    MLModel     *MLModel
    Analytics   *TrafficAnalytics
    Controllers map[string]*TrafficLightController
}

type MLModel struct {
    Model       interface{} // TensorFlow/PyTorch model
    Features    []string
    Predictions map[string]float64
}

func (to *TrafficOptimizer) OptimizeTrafficFlow() {
    // Collect current traffic data
    trafficData := to.Analytics.GetCurrentTrafficData()
    
    // Generate features for ML model
    features := to.extractFeatures(trafficData)
    
    // Predict optimal timing
    predictions := to.MLModel.Predict(features)
    
    // Update traffic light timings
    for intersectionID, optimalTiming := range predictions {
        if controller, exists := to.Controllers[intersectionID]; exists {
            controller.UpdateTiming(optimalTiming)
        }
    }
}
```

### Detailed Design

#### 1. Emergency Vehicle Detection
```go
type EmergencyVehicleDetector struct {
    Cameras     []*Camera
    AudioSensors []*AudioSensor
    MLModel     *EmergencyDetectionModel
}

func (evd *EmergencyVehicleDetector) DetectEmergencyVehicle() bool {
    // Analyze camera feeds
    for _, camera := range evd.Cameras {
        frame := camera.CaptureFrame()
        if evd.MLModel.DetectEmergencyVehicle(frame) {
            return true
        }
    }
    
    // Analyze audio for sirens
    for _, sensor := range evd.AudioSensors {
        audio := sensor.CaptureAudio()
        if evd.MLModel.DetectSiren(audio) {
            return true
        }
    }
    
    return false
}
```

#### 2. Traffic Violation Detection
```go
type ViolationDetector struct {
    Cameras     []*Camera
    MLModel     *ViolationDetectionModel
    Database    *gorm.DB
}

type TrafficViolation struct {
    ID          uint
    Type        string // red_light, speeding, wrong_lane
    VehicleID   string
    Timestamp   time.Time
    Location    *Location
    Evidence    []byte // image/video data
    Status      string // pending, processed, dismissed
}

func (vd *ViolationDetector) DetectViolations() {
    for _, camera := range vd.Cameras {
        frame := camera.CaptureFrame()
        violations := vd.MLModel.DetectViolations(frame)
        
        for _, violation := range violations {
            vd.recordViolation(violation)
        }
    }
}
```

### Scalability Considerations

#### 1. Edge Computing
- Deploy processing at intersection level
- Use local caching for real-time decisions
- Implement edge-to-cloud synchronization

#### 2. Data Processing
- Use message queues for sensor data
- Implement batch processing for analytics
- Use stream processing for real-time decisions

#### 3. Fault Tolerance
- Implement redundant controllers
- Use backup communication channels
- Implement automatic failover

---

## 5. Design a Search Typeahead System

### Problem Statement
Design a search typeahead system like Google's autocomplete that provides real-time search suggestions as users type.

### Requirements
- **Functional Requirements**
  - Real-time search suggestions
  - Personalized suggestions based on user history
  - Trending/popular suggestions
  - Multi-language support
  - Search result ranking

- **Non-Functional Requirements**
  - Low latency (< 100ms)
  - High availability (99.9%)
  - Scalable to millions of users
  - Handle high query volume

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                Search Typeahead System                     │
├─────────────────────────────────────────────────────────────┤
│  Users  →  Load Balancer  →  API Gateway  →  Services     │
│    │           │              │                │           │
│    │           │              │                │           │
│    ▼           ▼              ▼                ▼           │
│  Browser   Rate Limiter    Search API      Search Service  │
│  Mobile    Cache Layer     Analytics       Ranking Service │
│  Desktop   CDN            Personalization  Suggestion DB  │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Search Service
```go
type SearchService struct {
    Trie        *Trie
    Cache       *redis.Client
    Analytics   *AnalyticsService
    Personalizer *PersonalizationService
}

type Trie struct {
    Root        *TrieNode
    MaxResults  int
}

type TrieNode struct {
    Children    map[rune]*TrieNode
    IsEnd       bool
    Suggestions []*Suggestion
    Frequency   int64
}

type Suggestion struct {
    Text        string
    Frequency   int64
    Category    string
    Metadata    map[string]interface{}
}

func (ss *SearchService) GetSuggestions(query string, userID uint) ([]*Suggestion, error) {
    // Check cache first
    cacheKey := fmt.Sprintf("suggestions:%s:%d", query, userID)
    if cached := ss.Cache.Get(cacheKey).Val(); cached != "" {
        var suggestions []*Suggestion
        json.Unmarshal([]byte(cached), &suggestions)
        return suggestions, nil
    }
    
    // Get base suggestions from trie
    baseSuggestions := ss.Trie.Search(query)
    
    // Apply personalization
    personalizedSuggestions := ss.Personalizer.Personalize(baseSuggestions, userID)
    
    // Cache results
    data, _ := json.Marshal(personalizedSuggestions)
    ss.Cache.Set(cacheKey, data, 5*time.Minute)
    
    return personalizedSuggestions, nil
}
```

#### 2. Personalization Service
```go
type PersonalizationService struct {
    DB          *gorm.DB
    MLModel     *MLModel
    UserHistory *UserHistoryService
}

type UserSearchHistory struct {
    UserID      uint
    Query       string
    ClickedSuggestion string
    Timestamp   time.Time
    Category    string
}

func (ps *PersonalizationService) Personalize(suggestions []*Suggestion, userID uint) []*Suggestion {
    // Get user search history
    history := ps.UserHistory.GetRecentSearches(userID, 100)
    
    // Calculate personalized scores
    for _, suggestion := range suggestions {
        score := ps.calculatePersonalizedScore(suggestion, history)
        suggestion.Score = score
    }
    
    // Sort by personalized score
    sort.Slice(suggestions, func(i, j int) bool {
        return suggestions[i].Score > suggestions[j].Score
    })
    
    return suggestions[:min(len(suggestions), 10)]
}
```

#### 3. Analytics Service
```go
type AnalyticsService struct {
    DB      *gorm.DB
    Queue   *MessageQueue
}

type SearchEvent struct {
    UserID      uint
    Query       string
    Suggestion  string
    Clicked     bool
    Timestamp   time.Time
    SessionID   string
}

func (as *AnalyticsService) TrackSearchEvent(event *SearchEvent) {
    // Store in database
    as.DB.Create(event)
    
    // Send to message queue for real-time processing
    as.Queue.Publish("search_events", event)
    
    // Update trending queries
    as.updateTrendingQueries(event.Query)
}
```

### Detailed Design

#### 1. Trie Implementation
```go
func (t *Trie) Insert(suggestion *Suggestion) {
    node := t.Root
    for _, char := range suggestion.Text {
        if node.Children[char] == nil {
            node.Children[char] = &TrieNode{
                Children: make(map[rune]*TrieNode),
            }
        }
        node = node.Children[char]
    }
    
    node.IsEnd = true
    node.Suggestions = append(node.Suggestions, suggestion)
    node.Frequency += suggestion.Frequency
}

func (t *Trie) Search(query string) []*Suggestion {
    node := t.Root
    for _, char := range query {
        if node.Children[char] == nil {
            return []*Suggestion{}
        }
        node = node.Children[char]
    }
    
    // Collect all suggestions from this node and its children
    suggestions := t.collectSuggestions(node)
    
    // Sort by frequency
    sort.Slice(suggestions, func(i, j int) bool {
        return suggestions[i].Frequency > suggestions[j].Frequency
    })
    
    return suggestions[:min(len(suggestions), t.MaxResults)]
}
```

#### 2. Real-time Updates
```go
type RealTimeUpdater struct {
    Trie        *Trie
    Queue       *MessageQueue
    Cache       *redis.Client
}

func (rtu *RealTimeUpdater) ProcessUpdates() {
    for update := range rtu.Queue.Subscribe("suggestion_updates") {
        switch update.Type {
        case "insert":
            rtu.Trie.Insert(update.Suggestion)
        case "update":
            rtu.Trie.Update(update.Suggestion)
        case "delete":
            rtu.Trie.Delete(update.Suggestion.Text)
        }
        
        // Invalidate related cache entries
        rtu.invalidateCache(update.Suggestion.Text)
    }
}
```

### Scalability Considerations

#### 1. Caching Strategy
- Cache popular queries in Redis
- Use CDN for global distribution
- Implement cache warming

#### 2. Database Optimization
- Use read replicas for search queries
- Implement database sharding
- Use full-text search indexes

#### 3. Load Balancing
- Use round-robin load balancing
- Implement health checks
- Use geographic load balancing

---

This covers the first 5 system design scenarios. The file is getting quite large, so I'll continue with the remaining scenarios in the next part. Would you like me to continue with the remaining 12 scenarios?
