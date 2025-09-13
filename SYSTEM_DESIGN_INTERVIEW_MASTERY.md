# 🎯 **System Design Interview Mastery**

## 📊 **Complete Guide to System Design Interviews**

---

## 🎯 **1. Interview Structure and Approach**

### **The 4-Step System Design Process**

```go
package main

import (
    "fmt"
    "time"
)

// System Design Interview Framework
type SystemDesignInterview struct {
    Problem     *Problem
    Requirements *Requirements
    Design      *SystemDesign
    Discussion  *Discussion
}

type Problem struct {
    Description string
    Scale       *Scale
    Features    []string
    Constraints []string
}

type Scale struct {
    Users       int64
    Requests    int64
    Data        int64
    Storage     int64
    Bandwidth   int64
}

type Requirements struct {
    Functional    []string
    NonFunctional []string
    Performance   *Performance
    Availability  *Availability
    Scalability   *Scalability
}

type Performance struct {
    Latency    time.Duration
    Throughput int64
    ResponseTime time.Duration
}

type Availability struct {
    Uptime      float64
    MTBF        time.Duration
    MTTR        time.Duration
}

type Scalability struct {
    Horizontal bool
    Vertical   bool
    AutoScale  bool
}

type SystemDesign struct {
    Architecture *Architecture
    Components   []*Component
    DataFlow     *DataFlow
    APIs         []*API
    Database     *DatabaseDesign
    Caching      *CachingStrategy
    Security     *SecurityDesign
}

type Architecture struct {
    Type        string
    Patterns    []string
    Technologies []string
    Deployment  string
}

type Component struct {
    Name        string
    Type        string
    Responsibility string
    Technologies []string
    Scale       *Scale
}

type DataFlow struct {
    Steps       []*FlowStep
    Protocols   []string
    Formats     []string
}

type FlowStep struct {
    From        string
    To          string
    Data        string
    Protocol    string
    Latency     time.Duration
}

type API struct {
    Name        string
    Method      string
    Endpoint    string
    Parameters  []string
    Response    string
    RateLimit   int64
}

type DatabaseDesign struct {
    Type        string
    Schema      *Schema
    Sharding    *ShardingStrategy
    Replication *ReplicationStrategy
    Indexing    *IndexingStrategy
}

type Schema struct {
    Tables      []*Table
    Relationships []*Relationship
}

type Table struct {
    Name        string
    Columns     []*Column
    Indexes     []*Index
    Constraints []*Constraint
}

type Column struct {
    Name        string
    Type        string
    Nullable    bool
    Unique      bool
    PrimaryKey  bool
}

type Index struct {
    Name        string
    Columns     []string
    Type        string
    Unique      bool
}

type Constraint struct {
    Name        string
    Type        string
    Columns     []string
    Reference   string
}

type Relationship struct {
    From        string
    To          string
    Type        string
    Cardinality string
}

type ShardingStrategy struct {
    Type        string
    Key         string
    Shards      int
    Algorithm   string
}

type ReplicationStrategy struct {
    Type        string
    Replicas    int
    Consistency string
}

type IndexingStrategy struct {
    Primary     []string
    Secondary   []string
    Composite   []string
    FullText    []string
}

type CachingStrategy struct {
    Levels      []*CacheLevel
    Eviction    string
    TTL         time.Duration
    Invalidation string
}

type CacheLevel struct {
    Name        string
    Type        string
    Size        int64
    Latency     time.Duration
}

type SecurityDesign struct {
    Authentication *Authentication
    Authorization  *Authorization
    Encryption     *Encryption
    Network        *NetworkSecurity
}

type Authentication struct {
    Methods    []string
    Tokens     string
    Sessions   string
    MFA        bool
}

type Authorization struct {
    Model      string
    Roles      []string
    Permissions []string
    Policies   []string
}

type Encryption struct {
    InTransit  string
    AtRest     string
    Keys       string
    Rotation   time.Duration
}

type NetworkSecurity struct {
    Firewall   bool
    VPN        bool
    DDoS       bool
    WAF        bool
}

type Discussion struct {
    Tradeoffs  []*Tradeoff
    Alternatives []*Alternative
    Improvements []*Improvement
    Questions  []string
}

type Tradeoff struct {
    Aspect     string
    Option1    string
    Option2    string
    Pros       []string
    Cons       []string
    Decision   string
}

type Alternative struct {
    Approach   string
    Pros       []string
    Cons       []string
    UseCase    string
}

type Improvement struct {
    Area       string
    Current    string
    Proposed   string
    Impact     string
}

// Step 1: Clarify Requirements
func (sdi *SystemDesignInterview) ClarifyRequirements() {
    fmt.Println("=== STEP 1: CLARIFY REQUIREMENTS ===")
    
    // Ask clarifying questions
    questions := []string{
        "What is the expected scale?",
        "What are the key features?",
        "What are the performance requirements?",
        "What are the availability requirements?",
        "What are the security requirements?",
        "What are the constraints?",
    }
    
    for _, question := range questions {
        fmt.Printf("Q: %s\n", question)
        // In real interview, wait for answer
        time.Sleep(100 * time.Millisecond)
    }
    
    // Set example requirements
    sdi.Requirements = &Requirements{
        Functional: []string{
            "User registration and authentication",
            "Create and manage posts",
            "Follow other users",
            "View timeline/feed",
            "Like and comment on posts",
        },
        NonFunctional: []string{
            "High availability (99.9%)",
            "Low latency (<200ms)",
            "High throughput (1M requests/second)",
            "Scalable to 100M users",
        },
        Performance: &Performance{
            Latency:     200 * time.Millisecond,
            Throughput:  1000000,
            ResponseTime: 200 * time.Millisecond,
        },
        Availability: &Availability{
            Uptime: 99.9,
            MTBF:   24 * time.Hour,
            MTTR:   1 * time.Hour,
        },
        Scalability: &Scalability{
            Horizontal: true,
            Vertical:   true,
            AutoScale:  true,
        },
    }
    
    fmt.Printf("Requirements: %+v\n", sdi.Requirements)
}

// Step 2: High-Level Design
func (sdi *SystemDesignInterview) HighLevelDesign() {
    fmt.Println("\n=== STEP 2: HIGH-LEVEL DESIGN ===")
    
    // Create high-level architecture
    sdi.Design = &SystemDesign{
        Architecture: &Architecture{
            Type: "Microservices",
            Patterns: []string{
                "API Gateway",
                "Load Balancer",
                "Database Sharding",
                "Caching",
                "CDN",
            },
            Technologies: []string{
                "Go",
                "PostgreSQL",
                "Redis",
                "Kafka",
                "AWS",
            },
            Deployment: "Kubernetes",
        },
        Components: []*Component{
            {
                Name: "API Gateway",
                Type: "Gateway",
                Responsibility: "Request routing, authentication, rate limiting",
                Technologies: []string{"Kong", "Envoy"},
                Scale: &Scale{Users: 100000000, Requests: 1000000},
            },
            {
                Name: "User Service",
                Type: "Microservice",
                Responsibility: "User management, authentication",
                Technologies: []string{"Go", "PostgreSQL"},
                Scale: &Scale{Users: 100000000, Requests: 100000},
            },
            {
                Name: "Post Service",
                Type: "Microservice",
                Responsibility: "Post creation, retrieval, management",
                Technologies: []string{"Go", "PostgreSQL"},
                Scale: &Scale{Users: 100000000, Requests: 500000},
            },
            {
                Name: "Timeline Service",
                Type: "Microservice",
                Responsibility: "Timeline generation, feed management",
                Technologies: []string{"Go", "Redis", "Kafka"},
                Scale: &Scale{Users: 100000000, Requests: 1000000},
            },
        },
    }
    
    fmt.Printf("High-Level Design: %+v\n", sdi.Design)
}

// Step 3: Detailed Design
func (sdi *SystemDesignInterview) DetailedDesign() {
    fmt.Println("\n=== STEP 3: DETAILED DESIGN ===")
    
    // Design APIs
    sdi.Design.APIs = []*API{
        {
            Name: "Create Post",
            Method: "POST",
            Endpoint: "/api/v1/posts",
            Parameters: []string{"content", "user_id"},
            Response: "post_id, created_at",
            RateLimit: 100,
        },
        {
            Name: "Get Timeline",
            Method: "GET",
            Endpoint: "/api/v1/timeline",
            Parameters: []string{"user_id", "page", "limit"},
            Response: "posts[]",
            RateLimit: 1000,
        },
    }
    
    // Design Database
    sdi.Design.Database = &DatabaseDesign{
        Type: "PostgreSQL",
        Schema: &Schema{
            Tables: []*Table{
                {
                    Name: "users",
                    Columns: []*Column{
                        {Name: "id", Type: "BIGINT", PrimaryKey: true},
                        {Name: "username", Type: "VARCHAR", Unique: true},
                        {Name: "email", Type: "VARCHAR", Unique: true},
                        {Name: "created_at", Type: "TIMESTAMP"},
                    },
                    Indexes: []*Index{
                        {Name: "idx_username", Columns: []string{"username"}, Type: "BTREE"},
                        {Name: "idx_email", Columns: []string{"email"}, Type: "BTREE"},
                    },
                },
                {
                    Name: "posts",
                    Columns: []*Column{
                        {Name: "id", Type: "BIGINT", PrimaryKey: true},
                        {Name: "user_id", Type: "BIGINT"},
                        {Name: "content", Type: "TEXT"},
                        {Name: "created_at", Type: "TIMESTAMP"},
                    },
                    Indexes: []*Index{
                        {Name: "idx_user_id", Columns: []string{"user_id"}, Type: "BTREE"},
                        {Name: "idx_created_at", Columns: []string{"created_at"}, Type: "BTREE"},
                    },
                },
            },
        },
        Sharding: &ShardingStrategy{
            Type: "Hash-based",
            Key: "user_id",
            Shards: 100,
            Algorithm: "Consistent Hashing",
        },
        Replication: &ReplicationStrategy{
            Type: "Master-Slave",
            Replicas: 3,
            Consistency: "Eventual",
        },
    }
    
    // Design Caching
    sdi.Design.Caching = &CachingStrategy{
        Levels: []*CacheLevel{
            {
                Name: "L1",
                Type: "In-Memory",
                Size: 1000000,
                Latency: 1 * time.Millisecond,
            },
            {
                Name: "L2",
                Type: "Redis",
                Size: 10000000,
                Latency: 5 * time.Millisecond,
            },
        },
        Eviction: "LRU",
        TTL: 1 * time.Hour,
        Invalidation: "Write-through",
    }
    
    fmt.Printf("Detailed Design: %+v\n", sdi.Design)
}

// Step 4: Scale and Optimize
func (sdi *SystemDesignInterview) ScaleAndOptimize() {
    fmt.Println("\n=== STEP 4: SCALE AND OPTIMIZE ===")
    
    // Discuss scaling strategies
    scalingStrategies := []string{
        "Horizontal scaling with load balancers",
        "Database sharding and read replicas",
        "Caching at multiple levels",
        "CDN for static content",
        "Message queues for async processing",
        "Microservices for independent scaling",
    }
    
    for _, strategy := range scalingStrategies {
        fmt.Printf("Scaling Strategy: %s\n", strategy)
    }
    
    // Discuss optimizations
    optimizations := []string{
        "Database query optimization",
        "Connection pooling",
        "Compression and serialization",
        "Batch processing",
        "Precomputation and materialized views",
        "Monitoring and alerting",
    }
    
    for _, optimization := range optimizations {
        fmt.Printf("Optimization: %s\n", optimization)
    }
}

// Run complete interview
func (sdi *SystemDesignInterview) RunInterview() {
    fmt.Println("🎯 SYSTEM DESIGN INTERVIEW MASTERY")
    fmt.Println("=====================================")
    
    sdi.ClarifyRequirements()
    sdi.HighLevelDesign()
    sdi.DetailedDesign()
    sdi.ScaleAndOptimize()
    
    fmt.Println("\n✅ Interview completed successfully!")
}

// Example usage
func main() {
    interview := &SystemDesignInterview{
        Problem: &Problem{
            Description: "Design a social media platform like Twitter",
            Scale: &Scale{
                Users:    100000000,
                Requests: 1000000,
                Data:     1000000000,
                Storage:  10000000000,
                Bandwidth: 1000000000,
            },
        },
    }
    
    interview.RunInterview()
}
```

---

## 🎯 **2. Common System Design Questions**

### **Question 1: Design a URL Shortener (like bit.ly)**

```go
// URL Shortener System Design
type URLShortener struct {
    shortURL    string
    longURL     string
    createdAt   time.Time
    expiresAt   time.Time
    clickCount  int64
    userID      string
}

type URLShortenerService struct {
    db          *Database
    cache       *Cache
    counter     *Counter
    analytics   *Analytics
}

func (us *URLShortenerService) ShortenURL(longURL string) (string, error) {
    // Generate short code
    shortCode := us.generateShortCode()
    
    // Store in database
    shortener := &URLShortener{
        shortURL:   shortCode,
        longURL:    longURL,
        createdAt:  time.Now(),
        expiresAt:  time.Now().Add(365 * 24 * time.Hour),
        clickCount: 0,
    }
    
    if err := us.db.Store(shortener); err != nil {
        return "", err
    }
    
    // Cache for fast access
    us.cache.Set(shortCode, longURL, 24*time.Hour)
    
    return shortCode, nil
}

func (us *URLShortenerService) Redirect(shortCode string) (string, error) {
    // Try cache first
    if longURL, found := us.cache.Get(shortCode); found {
        us.analytics.TrackClick(shortCode)
        return longURL.(string), nil
    }
    
    // Get from database
    shortener, err := us.db.Get(shortCode)
    if err != nil {
        return "", err
    }
    
    // Cache for future requests
    us.cache.Set(shortCode, shortener.longURL, 24*time.Hour)
    
    // Track click
    us.analytics.TrackClick(shortCode)
    
    return shortener.longURL, nil
}

func (us *URLShortenerService) generateShortCode() string {
    // Use base62 encoding
    counter := us.counter.Increment()
    return us.encodeBase62(counter)
}

func (us *URLShortenerService) encodeBase62(num int64) string {
    chars := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result := ""
    
    for num > 0 {
        result = string(chars[num%62]) + result
        num /= 62
    }
    
    return result
}
```

### **Question 2: Design a Chat System (like WhatsApp)**

```go
// Chat System Design
type ChatSystem struct {
    users       map[string]*User
    rooms       map[string]*Room
    messages    map[string]*Message
    db          *Database
    cache       *Cache
    websocket   *WebSocketManager
}

type User struct {
    ID       string
    Username string
    Status   string
    LastSeen time.Time
}

type Room struct {
    ID       string
    Name     string
    Type     string // "private", "group"
    Members  []string
    CreatedAt time.Time
}

type Message struct {
    ID        string
    RoomID    string
    UserID    string
    Content   string
    Type      string // "text", "image", "file"
    Timestamp time.Time
    Status    string // "sent", "delivered", "read"
}

func (cs *ChatSystem) SendMessage(roomID, userID, content string) error {
    // Create message
    message := &Message{
        ID:        generateMessageID(),
        RoomID:    roomID,
        UserID:    userID,
        Content:   content,
        Type:      "text",
        Timestamp: time.Now(),
        Status:    "sent",
    }
    
    // Store in database
    if err := cs.db.StoreMessage(message); err != nil {
        return err
    }
    
    // Send to room members
    room := cs.rooms[roomID]
    for _, memberID := range room.Members {
        if memberID != userID {
            cs.websocket.SendToUser(memberID, message)
        }
    }
    
    return nil
}

func (cs *ChatSystem) GetMessages(roomID string, limit int) ([]*Message, error) {
    // Try cache first
    if messages, found := cs.cache.Get(roomID); found {
        return messages.([]*Message), nil
    }
    
    // Get from database
    messages, err := cs.db.GetMessages(roomID, limit)
    if err != nil {
        return nil, err
    }
    
    // Cache for future requests
    cs.cache.Set(roomID, messages, 5*time.Minute)
    
    return messages, nil
}
```

---

## 🎯 **3. Interview Tips and Best Practices**

### **Communication Tips**

```go
// Interview Communication Framework
type InterviewCommunication struct {
    Clarity     bool
    Structure   bool
    Tradeoffs   bool
    Questions   bool
    Confidence  bool
}

func (ic *InterviewCommunication) CommunicateEffectively() {
    fmt.Println("=== COMMUNICATION BEST PRACTICES ===")
    
    practices := []string{
        "Think out loud - explain your thought process",
        "Ask clarifying questions early",
        "Start with high-level design, then dive deep",
        "Discuss tradeoffs and alternatives",
        "Be open to feedback and suggestions",
        "Use diagrams and visual aids",
        "Estimate numbers and do back-of-envelope calculations",
        "Consider edge cases and failure scenarios",
    }
    
    for i, practice := range practices {
        fmt.Printf("%d. %s\n", i+1, practice)
    }
}

func (ic *InterviewCommunication) HandleQuestions() {
    fmt.Println("\n=== HANDLING QUESTIONS ===")
    
    questionTypes := map[string][]string{
        "Clarification": {
            "What is the expected scale?",
            "What are the key features?",
            "What are the performance requirements?",
        },
        "Design": {
            "How would you handle this scenario?",
            "What if the load increases 10x?",
            "How would you ensure data consistency?",
        },
        "Tradeoffs": {
            "What are the pros and cons of this approach?",
            "Why did you choose this over that?",
            "How would you handle this tradeoff?",
        },
        "Optimization": {
            "How would you optimize this?",
            "What are the bottlenecks?",
            "How would you improve performance?",
        },
    }
    
    for category, questions := range questionTypes {
        fmt.Printf("\n%s Questions:\n", category)
        for _, question := range questions {
            fmt.Printf("- %s\n", question)
        }
    }
}
```

---

## 🎯 **Key Takeaways from System Design Interview Mastery**

### **1. Interview Structure**
- **4-Step Process**: Clarify, Design, Detail, Scale
- **Communication**: Think out loud, ask questions, discuss tradeoffs
- **Visualization**: Use diagrams and visual aids
- **Estimation**: Do back-of-envelope calculations

### **2. Common Questions**
- **URL Shortener**: Hash functions, database design, caching
- **Chat System**: WebSockets, message queues, real-time updates
- **Social Media**: Timeline generation, feed algorithms, scaling
- **E-commerce**: Inventory management, payment processing, recommendations

### **3. Design Patterns**
- **Microservices**: Service decomposition and communication
- **Load Balancing**: Traffic distribution and health checks
- **Caching**: Multi-level caching strategies
- **Database**: Sharding, replication, and consistency

### **4. Production Considerations**
- **Scalability**: Horizontal and vertical scaling strategies
- **Availability**: Fault tolerance and disaster recovery
- **Performance**: Latency optimization and throughput
- **Security**: Authentication, authorization, and encryption

---

**🎉 This comprehensive guide provides complete system design interview mastery with practical frameworks and real-world examples! 🚀**
