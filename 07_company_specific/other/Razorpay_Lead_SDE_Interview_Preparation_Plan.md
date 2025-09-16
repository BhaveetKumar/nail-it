# Razorpay Lead Software Development Engineer - Interview Preparation Plan

## 📅 5-Day Study Schedule (Monday Interview)

### Day 1 (Today): System Design & Architecture Fundamentals

**Time: 4-5 hours**

#### Morning (2 hours): System Design Basics

- **High-Level Design (HLD) Concepts**
  - Scalability patterns (Horizontal vs Vertical scaling)
  - Load balancing strategies
  - Database design (SQL vs NoSQL)
  - Caching strategies (Redis, Memcached)
  - Message queues (Kafka, RabbitMQ)

#### Afternoon (2-3 hours): Practice System Design

- **Focus Areas:**
  - Distributed systems fundamentals
  - Microservices architecture
  - Event-driven architecture
  - CAP theorem and consistency models

### Day 2: Coding & Problem Solving

**Time: 4-5 hours**

#### Morning (2 hours): Coding Fundamentals

- **Data Structures Review:**
  - Arrays, Linked Lists, Stacks, Queues
  - Trees, Graphs, Hash Tables
  - Time and space complexity analysis

#### Afternoon (2-3 hours): Practice Coding

- **Focus Areas:**
  - Object-oriented design
  - Design patterns (Singleton, Factory, Observer, etc.)
  - API design and implementation
  - Code modularity and extensibility

### Day 3: Technical Deep Dive & Experience Preparation

**Time: 3-4 hours**

#### Morning (2 hours): Technical Deep Dive

- **Review Your Past Projects:**
  - System architecture decisions
  - Technology choices and trade-offs
  - Performance optimizations
  - Challenges faced and solutions

#### Afternoon (1-2 hours): Behavioral Preparation

- **Prepare STAR Method Examples:**
  - Situation, Task, Action, Result
  - Leadership experiences
  - Conflict resolution
  - Technical challenges overcome

### Day 4: Leadership & Management Preparation

**Time: 3-4 hours**

#### Morning (2 hours): Leadership Scenarios

- **Team Management:**
  - Mentoring junior developers
  - Code review processes
  - Technical decision making
  - Cross-functional collaboration

#### Afternoon (1-2 hours): Company Research

- **Razorpay Specific:**
  - Company culture and values
  - Product portfolio
  - Recent news and developments
  - Engineering blog insights

### Day 5: Final Review & Mock Practice

**Time: 2-3 hours**

#### Morning (1-2 hours): Final Review

- **Quick Review:**
  - Key concepts from all rounds
  - Your prepared examples
  - Questions to ask interviewers

#### Afternoon (1 hour): Mock Interview

- **Self-Practice:**
  - Time yourself on coding problems
  - Practice explaining system designs
  - Review behavioral answers

---

## 🎯 Round-by-Round Preparation

### Round 1: Machine Coding Round (90 Minutes)

#### Sample Questions & Solutions:

**1. Implement a Messaging API**

```go
package main

import (
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Message represents a single message in the system
type Message struct {
	MessageID  string    `json:"message_id"`
	SenderID   string    `json:"sender_id"`
	ReceiverID string    `json:"receiver_id"`
	Content    string    `json:"content"`
	Timestamp  time.Time `json:"timestamp"`
	IsRead     bool      `json:"is_read"`
}

// MessageResponse represents the response format for messages
type MessageResponse struct {
	MessageID  string    `json:"message_id"`
	SenderID   string    `json:"sender_id"`
	ReceiverID string    `json:"receiver_id"`
	Content    string    `json:"content"`
	Timestamp  time.Time `json:"timestamp"`
	IsRead     bool      `json:"is_read"`
}

// MessagingAPI handles all messaging operations
type MessagingAPI struct {
	messages     map[string]*Message
	userMessages map[string][]string
	mutex        sync.RWMutex
}

// NewMessagingAPI creates a new instance of MessagingAPI
func NewMessagingAPI() *MessagingAPI {
	return &MessagingAPI{
		messages:     make(map[string]*Message),
		userMessages: make(map[string][]string),
	}
}

// SendMessage sends a message from sender to receiver
func (api *MessagingAPI) SendMessage(senderID, receiverID, content string) (string, error) {
	if strings.TrimSpace(content) == "" {
		return "", errors.New("message content cannot be empty")
	}

	api.mutex.Lock()
	defer api.mutex.Unlock()

	message := &Message{
		MessageID:  uuid.New().String(),
		SenderID:   senderID,
		ReceiverID: receiverID,
		Content:    content,
		Timestamp:  time.Now(),
		IsRead:     false,
	}

	api.messages[message.MessageID] = message

	// Add to user's message lists
	api.userMessages[senderID] = append(api.userMessages[senderID], message.MessageID)
	api.userMessages[receiverID] = append(api.userMessages[receiverID], message.MessageID)

	return message.MessageID, nil
}

// GetMessages retrieves messages for a user with pagination
func (api *MessagingAPI) GetMessages(userID string, limit int) []MessageResponse {
	api.mutex.RLock()
	defer api.mutex.RUnlock()

	messageIDs, exists := api.userMessages[userID]
	if !exists {
		return []MessageResponse{}
	}

	// Get the most recent messages (last 'limit' messages)
	start := 0
	if len(messageIDs) > limit {
		start = len(messageIDs) - limit
	}

	var messages []MessageResponse
	for i := start; i < len(messageIDs); i++ {
		msgID := messageIDs[i]
		if msg, exists := api.messages[msgID]; exists {
			messages = append(messages, MessageResponse{
				MessageID:  msg.MessageID,
				SenderID:   msg.SenderID,
				ReceiverID: msg.ReceiverID,
				Content:    msg.Content,
				Timestamp:  msg.Timestamp,
				IsRead:     msg.IsRead,
			})
		}
	}

	return messages
}

// MarkAsRead marks a message as read for a specific user
func (api *MessagingAPI) MarkAsRead(userID, messageID string) error {
	api.mutex.Lock()
	defer api.mutex.Unlock()

	message, exists := api.messages[messageID]
	if !exists {
		return errors.New("message not found")
	}

	if message.ReceiverID != userID {
		return errors.New("user not authorized to mark this message as read")
	}

	message.IsRead = true
	return nil
}

// GetUnreadCount returns the count of unread messages for a user
func (api *MessagingAPI) GetUnreadCount(userID string) int {
	api.mutex.RLock()
	defer api.mutex.RUnlock()

	messageIDs, exists := api.userMessages[userID]
	if !exists {
		return 0
	}

	unreadCount := 0
	for _, msgID := range messageIDs {
		if msg, exists := api.messages[msgID]; exists {
			if msg.ReceiverID == userID && !msg.IsRead {
				unreadCount++
			}
		}
	}

	return unreadCount
}

// GetMessage retrieves a specific message by ID
func (api *MessagingAPI) GetMessage(messageID string) (*Message, error) {
	api.mutex.RLock()
	defer api.mutex.RUnlock()

	message, exists := api.messages[messageID]
	if !exists {
		return nil, errors.New("message not found")
	}

	return message, nil
}
```

**Key Design Decisions:**

- Used UUID for message IDs to ensure uniqueness
- Stored messages in a map for O(1) access
- Maintained user message lists for efficient retrieval
- Added proper error handling with Go idioms
- Included thread-safe operations with sync.RWMutex
- Used JSON tags for API serialization
- Implemented proper Go naming conventions

**2. Implement Entities for Price Comparison Website**

```go
package main

import (
	"errors"
	"sort"
	"strings"
	"sync"
	"time"
)

// Product represents a product in the system
type Product struct {
	ProductID      string            `json:"product_id"`
	Name           string            `json:"name"`
	Category       string            `json:"category"`
	Brand          string            `json:"brand"`
	Description    string            `json:"description"`
	Specifications map[string]string `json:"specifications"`
}

// NewProduct creates a new product instance
func NewProduct(productID, name, category, brand string) *Product {
	return &Product{
		ProductID:      productID,
		Name:           name,
		Category:       category,
		Brand:          brand,
		Description:    "",
		Specifications: make(map[string]string),
	}
}

// AddSpecification adds a specification to the product
func (p *Product) AddSpecification(key, value string) {
	p.Specifications[key] = value
}

// Price represents a price entry for a product from a seller
type Price struct {
	ProductID          string    `json:"product_id"`
	SellerID           string    `json:"seller_id"`
	Price              float64   `json:"price"`
	Currency           string    `json:"currency"`
	Availability       bool      `json:"availability"`
	LastUpdated        time.Time `json:"last_updated"`
	ShippingCost       float64   `json:"shipping_cost"`
	DiscountPercentage float64   `json:"discount_percentage"`
}

// NewPrice creates a new price instance
func NewPrice(productID, sellerID string, price float64, currency string) *Price {
	return &Price{
		ProductID:          productID,
		SellerID:           sellerID,
		Price:              price,
		Currency:           currency,
		Availability:       true,
		LastUpdated:        time.Now(),
		ShippingCost:       0.0,
		DiscountPercentage: 0.0,
	}
}

// Seller represents a seller in the system
type Seller struct {
	SellerID      string  `json:"seller_id"`
	Name          string  `json:"name"`
	Website       string  `json:"website"`
	Rating        float64 `json:"rating"`
	ReviewCount   int     `json:"review_count"`
	ShippingPolicy string `json:"shipping_policy"`
	ReturnPolicy  string  `json:"return_policy"`
}

// NewSeller creates a new seller instance
func NewSeller(sellerID, name, website string) *Seller {
	return &Seller{
		SellerID:        sellerID,
		Name:            name,
		Website:         website,
		Rating:          0.0,
		ReviewCount:     0,
		ShippingPolicy:  "",
		ReturnPolicy:    "",
	}
}

// PriceComparisonService handles all price comparison operations
type PriceComparisonService struct {
	products map[string]*Product
	prices   map[string][]*Price
	sellers  map[string]*Seller
	mutex    sync.RWMutex
}

// NewPriceComparisonService creates a new service instance
func NewPriceComparisonService() *PriceComparisonService {
	return &PriceComparisonService{
		products: make(map[string]*Product),
		prices:   make(map[string][]*Price),
		sellers:  make(map[string]*Seller),
	}
}

// AddProduct adds a product to the service
func (s *PriceComparisonService) AddProduct(product *Product) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.products[product.ProductID] = product
	s.prices[product.ProductID] = []*Price{}
}

// AddSeller adds a seller to the service
func (s *PriceComparisonService) AddSeller(seller *Seller) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.sellers[seller.SellerID] = seller
}

// AddPrice adds a price entry for a product
func (s *PriceComparisonService) AddPrice(price *Price) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if _, exists := s.prices[price.ProductID]; !exists {
		return errors.New("product not found")
	}

	s.prices[price.ProductID] = append(s.prices[price.ProductID], price)
	return nil
}

// GetBestPrice returns the lowest available price for a product
func (s *PriceComparisonService) GetBestPrice(productID string) (*Price, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	prices, exists := s.prices[productID]
	if !exists {
		return nil, errors.New("product not found")
	}

	var availablePrices []*Price
	for _, price := range prices {
		if price.Availability {
			availablePrices = append(availablePrices, price)
		}
	}

	if len(availablePrices) == 0 {
		return nil, errors.New("no available prices found")
	}

	// Sort by price and return the lowest
	sort.Slice(availablePrices, func(i, j int) bool {
		return availablePrices[i].Price < availablePrices[j].Price
	})

	return availablePrices[0], nil
}

// ComparePrices returns all available prices for a product, sorted by price
func (s *PriceComparisonService) ComparePrices(productID string) ([]*Price, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	prices, exists := s.prices[productID]
	if !exists {
		return nil, errors.New("product not found")
	}

	var availablePrices []*Price
	for _, price := range prices {
		if price.Availability {
			availablePrices = append(availablePrices, price)
		}
	}

	// Sort by price
	sort.Slice(availablePrices, func(i, j int) bool {
		return availablePrices[i].Price < availablePrices[j].Price
	})

	return availablePrices, nil
}

// SearchProducts searches for products by name or brand
func (s *PriceComparisonService) SearchProducts(query string, category string) []*Product {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	queryLower := strings.ToLower(query)
	var results []*Product

	for _, product := range s.products {
		// Check if query matches name or brand
		if strings.Contains(strings.ToLower(product.Name), queryLower) ||
			strings.Contains(strings.ToLower(product.Brand), queryLower) {

			// Filter by category if specified
			if category == "" || product.Category == category {
				results = append(results, product)
			}
		}
	}

	return results
}

// GetProduct retrieves a product by ID
func (s *PriceComparisonService) GetProduct(productID string) (*Product, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	product, exists := s.products[productID]
	if !exists {
		return nil, errors.New("product not found")
	}

	return product, nil
}

// GetSeller retrieves a seller by ID
func (s *PriceComparisonService) GetSeller(sellerID string) (*Seller, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	seller, exists := s.sellers[sellerID]
	if !exists {
		return nil, errors.New("seller not found")
	}

	return seller, nil
}
```

**Key Design Decisions:**

- Used float64 for price calculations (consider using decimal package for production)
- Abstracted seller information for extensibility
- Implemented search functionality with category filtering
- Added availability tracking for prices
- Included seller ratings and policies
- Used proper Go error handling patterns
- Implemented thread-safe operations with mutex
- Used constructor functions for better encapsulation

### Round 2: System Design (60 Minutes)

#### Sample Questions & Solutions:

**1. Design a Distributed Cache**

**Requirements:**

- Store key-value pairs
- Handle high read/write throughput
- Ensure data consistency
- Handle node failures
- Support TTL (Time To Live)

**High-Level Design:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   Client App    │    │   Client App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Load Balancer        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Cache Cluster         │
                    │  ┌─────┐ ┌─────┐ ┌─────┐  │
                    │  │Node1│ │Node2│ │Node3│  │
                    │  └─────┘ └─────┘ └─────┘  │
                    └───────────────────────────┘
```

**Key Components:**

1. **Cache Nodes**: Multiple cache servers for redundancy
2. **Consistent Hashing**: Distribute keys across nodes
3. **Replication**: Each key stored on multiple nodes
4. **Load Balancer**: Distribute requests across nodes

**Low-Level Design:**

```go
package main

import (
	"crypto/md5"
	"fmt"
	"sort"
	"sync"
	"time"
)

// CacheNode represents a single cache node
type CacheNode struct {
	NodeID     string
	Data       map[string]interface{}
	Timestamps map[string]time.Time
	mutex      sync.RWMutex
}

// NewCacheNode creates a new cache node
func NewCacheNode(nodeID string) *CacheNode {
	return &CacheNode{
		NodeID:     nodeID,
		Data:       make(map[string]interface{}),
		Timestamps: make(map[string]time.Time),
	}
}

// Get retrieves a value from the cache node
func (cn *CacheNode) Get(key string) (interface{}, bool) {
	cn.mutex.RLock()
	defer cn.mutex.RUnlock()

	value, exists := cn.Data[key]
	if !exists {
		return nil, false
	}

	// Check TTL
	if timestamp, hasTimestamp := cn.Timestamps[key]; hasTimestamp {
		ttl := time.Hour // 1 hour default TTL
		if time.Since(timestamp) > ttl {
			cn.mutex.RUnlock()
			cn.mutex.Lock()
			delete(cn.Data, key)
			delete(cn.Timestamps, key)
			cn.mutex.Unlock()
			cn.mutex.RLock()
			return nil, false
		}
	}

	return value, true
}

// Set stores a value in the cache node
func (cn *CacheNode) Set(key string, value interface{}, ttl *time.Duration) {
	cn.mutex.Lock()
	defer cn.mutex.Unlock()

	cn.Data[key] = value
	if ttl != nil {
		cn.Timestamps[key] = time.Now()
	}
}

// Delete removes a key from the cache node
func (cn *CacheNode) Delete(key string) {
	cn.mutex.Lock()
	defer cn.mutex.Unlock()

	delete(cn.Data, key)
	delete(cn.Timestamps, key)
}

// ConsistentHash implements consistent hashing for distributed cache
type ConsistentHash struct {
	replicas   int
	ring       map[uint32]string
	sortedKeys []uint32
	mutex      sync.RWMutex
}

// NewConsistentHash creates a new consistent hash instance
func NewConsistentHash(replicas int) *ConsistentHash {
	return &ConsistentHash{
		replicas:   replicas,
		ring:       make(map[uint32]string),
		sortedKeys: make([]uint32, 0),
	}
}

// AddNode adds a node to the hash ring
func (ch *ConsistentHash) AddNode(node string) {
	ch.mutex.Lock()
	defer ch.mutex.Unlock()

	for i := 0; i < ch.replicas; i++ {
		key := ch.hash(fmt.Sprintf("%s:%d", node, i))
		ch.ring[key] = node
		ch.sortedKeys = append(ch.sortedKeys, key)
	}

	sort.Slice(ch.sortedKeys, func(i, j int) bool {
		return ch.sortedKeys[i] < ch.sortedKeys[j]
	})
}

// RemoveNode removes a node from the hash ring
func (ch *ConsistentHash) RemoveNode(node string) {
	ch.mutex.Lock()
	defer ch.mutex.Unlock()

	for i := 0; i < ch.replicas; i++ {
		key := ch.hash(fmt.Sprintf("%s:%d", node, i))
		delete(ch.ring, key)

		// Remove from sorted keys
		for j, sortedKey := range ch.sortedKeys {
			if sortedKey == key {
				ch.sortedKeys = append(ch.sortedKeys[:j], ch.sortedKeys[j+1:]...)
				break
			}
		}
	}
}

// GetNode returns the node responsible for a given key
func (ch *ConsistentHash) GetNode(key string) (string, bool) {
	ch.mutex.RLock()
	defer ch.mutex.RUnlock()

	if len(ch.ring) == 0 {
		return "", false
	}

	hashKey := ch.hash(key)

	// Find the first node with hash >= hashKey
	for _, sortedKey := range ch.sortedKeys {
		if hashKey <= sortedKey {
			return ch.ring[sortedKey], true
		}
	}

	// Wrap around to the first node
	return ch.ring[ch.sortedKeys[0]], true
}

// hash generates a hash for the given key
func (ch *ConsistentHash) hash(key string) uint32 {
	hash := md5.Sum([]byte(key))
	return uint32(hash[0])<<24 | uint32(hash[1])<<16 | uint32(hash[2])<<8 | uint32(hash[3])
}

// DistributedCache represents the main distributed cache system
type DistributedCache struct {
	nodes     map[string]*CacheNode
	hashRing  *ConsistentHash
	replicas  int
	mutex     sync.RWMutex
}

// NewDistributedCache creates a new distributed cache instance
func NewDistributedCache(nodes []string, replicas int) *DistributedCache {
	dc := &DistributedCache{
		nodes:    make(map[string]*CacheNode),
		hashRing: NewConsistentHash(replicas),
		replicas: replicas,
	}

	// Initialize nodes
	for _, nodeID := range nodes {
		dc.nodes[nodeID] = NewCacheNode(nodeID)
		dc.hashRing.AddNode(nodeID)
	}

	return dc
}

// Get retrieves a value from the distributed cache
func (dc *DistributedCache) Get(key string) (interface{}, bool) {
	dc.mutex.RLock()
	defer dc.mutex.RUnlock()

	nodeID, exists := dc.hashRing.GetNode(key)
	if !exists {
		return nil, false
	}

	node, exists := dc.nodes[nodeID]
	if !exists {
		return nil, false
	}

	return node.Get(key)
}

// Set stores a value in the distributed cache
func (dc *DistributedCache) Set(key string, value interface{}, ttl *time.Duration) bool {
	dc.mutex.RLock()
	defer dc.mutex.RUnlock()

	nodeID, exists := dc.hashRing.GetNode(key)
	if !exists {
		return false
	}

	node, exists := dc.nodes[nodeID]
	if !exists {
		return false
	}

	// Store on primary node
	node.Set(key, value, ttl)

	// Replicate to other nodes
	for i := 1; i < dc.replicas; i++ {
		replicaKey := fmt.Sprintf("%s:replica:%d", key, i)
		replicaNodeID, exists := dc.hashRing.GetNode(replicaKey)
		if exists && replicaNodeID != nodeID {
			if replicaNode, exists := dc.nodes[replicaNodeID]; exists {
				replicaNode.Set(key, value, ttl)
			}
		}
	}

	return true
}

// Delete removes a key from the distributed cache
func (dc *DistributedCache) Delete(key string) {
	dc.mutex.RLock()
	defer dc.mutex.RUnlock()

	nodeID, exists := dc.hashRing.GetNode(key)
	if !exists {
		return
	}

	if node, exists := dc.nodes[nodeID]; exists {
		node.Delete(key)
	}
}

// AddNode adds a new node to the distributed cache
func (dc *DistributedCache) AddNode(nodeID string) {
	dc.mutex.Lock()
	defer dc.mutex.Unlock()

	dc.nodes[nodeID] = NewCacheNode(nodeID)
	dc.hashRing.AddNode(nodeID)
}

// RemoveNode removes a node from the distributed cache
func (dc *DistributedCache) RemoveNode(nodeID string) {
	dc.mutex.Lock()
	defer dc.mutex.Unlock()

	delete(dc.nodes, nodeID)
	dc.hashRing.RemoveNode(nodeID)
}
```

**Key Design Decisions:**

- **Consistent Hashing**: Ensures even distribution and minimal rehashing
- **Replication**: Provides fault tolerance and high availability
- **TTL Support**: Prevents stale data accumulation
- **Load Balancing**: Distributes requests across nodes

**2. Design an Event Throttling Framework**

**Requirements:**

- Rate limit events based on user/API key
- Support different throttling strategies
- Handle burst traffic
- Provide real-time monitoring

**High-Level Design:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │   Client    │    │   Client    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
              ┌───────────┴───────────┐
              │  Throttling Service   │
              │  ┌─────────────────┐  │
              │  │  Rate Limiter   │  │
              │  │  ┌─────────────┐│  │
              │  │  │Token Bucket ││  │
              │  │  └─────────────┘│  │
              │  │  ┌─────────────┐│  │
              │  │  │Sliding Win. ││  │
              │  │  └─────────────┘│  │
              │  └─────────────────┘  │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │     Redis Cache       │
              └───────────────────────┘
```

**Implementation:**

```go
package main

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ThrottlingStrategy defines the interface for throttling strategies
type ThrottlingStrategy interface {
	IsAllowed(key string, limit int, window time.Duration) bool
}

// TokenBucketStrategy implements token bucket throttling
type TokenBucketStrategy struct {
	buckets map[string]*TokenBucket
	mutex   sync.RWMutex
}

// TokenBucket represents a token bucket for a specific key
type TokenBucket struct {
	Tokens    float64
	LastRefill time.Time
	Limit     int
	Window    time.Duration
}

// NewTokenBucketStrategy creates a new token bucket strategy
func NewTokenBucketStrategy() *TokenBucketStrategy {
	return &TokenBucketStrategy{
		buckets: make(map[string]*TokenBucket),
	}
}

// IsAllowed checks if a request is allowed using token bucket algorithm
func (tbs *TokenBucketStrategy) IsAllowed(key string, limit int, window time.Duration) bool {
	tbs.mutex.Lock()
	defer tbs.mutex.Unlock()

	currentTime := time.Now()
	bucket, exists := tbs.buckets[key]

	if !exists {
		// Initialize new bucket
		bucket = &TokenBucket{
			Tokens:     float64(limit),
			LastRefill: currentTime,
			Limit:      limit,
			Window:     window,
		}
		tbs.buckets[key] = bucket
	}

	// Refill tokens based on time passed
	timePassed := currentTime.Sub(bucket.LastRefill)
	tokensToAdd := (timePassed.Seconds() / window.Seconds()) * float64(limit)
	bucket.Tokens = min(float64(limit), bucket.Tokens+tokensToAdd)
	bucket.LastRefill = currentTime

	if bucket.Tokens >= 1 {
		// Consume one token
		bucket.Tokens--
		return true
	}

	return false
}

// SlidingWindowStrategy implements sliding window throttling
type SlidingWindowStrategy struct {
	windows map[string]*SlidingWindow
	mutex   sync.RWMutex
}

// SlidingWindow represents a sliding window for a specific key
type SlidingWindow struct {
	Requests []time.Time
	Limit    int
	Window   time.Duration
}

// NewSlidingWindowStrategy creates a new sliding window strategy
func NewSlidingWindowStrategy() *SlidingWindowStrategy {
	return &SlidingWindowStrategy{
		windows: make(map[string]*SlidingWindow),
	}
}

// IsAllowed checks if a request is allowed using sliding window algorithm
func (sws *SlidingWindowStrategy) IsAllowed(key string, limit int, window time.Duration) bool {
	sws.mutex.Lock()
	defer sws.mutex.Unlock()

	currentTime := time.Now()
	windowStart := currentTime.Add(-window)

	slidingWindow, exists := sws.windows[key]
	if !exists {
		slidingWindow = &SlidingWindow{
			Requests: make([]time.Time, 0),
			Limit:    limit,
			Window:   window,
		}
		sws.windows[key] = slidingWindow
	}

	// Remove old requests outside the window
	var validRequests []time.Time
	for _, requestTime := range slidingWindow.Requests {
		if requestTime.After(windowStart) {
			validRequests = append(validRequests, requestTime)
		}
	}
	slidingWindow.Requests = validRequests

	// Check if we can add a new request
	if len(slidingWindow.Requests) < limit {
		slidingWindow.Requests = append(slidingWindow.Requests, currentTime)
		return true
	}

	return false
}

// EventThrottlingFramework manages different throttling strategies
type EventThrottlingFramework struct {
	strategies map[string]ThrottlingStrategy
	mutex      sync.RWMutex
}

// NewEventThrottlingFramework creates a new throttling framework
func NewEventThrottlingFramework() *EventThrottlingFramework {
	return &EventThrottlingFramework{
		strategies: map[string]ThrottlingStrategy{
			"token_bucket":   NewTokenBucketStrategy(),
			"sliding_window": NewSlidingWindowStrategy(),
		},
	}
}

// Throttle checks if a request is allowed based on the specified strategy
func (etf *EventThrottlingFramework) Throttle(key string, limit int, window time.Duration, strategy string) (bool, error) {
	etf.mutex.RLock()
	defer etf.mutex.RUnlock()

	throttlingStrategy, exists := etf.strategies[strategy]
	if !exists {
		return false, errors.New("unknown strategy: " + strategy)
	}

	return throttlingStrategy.IsAllowed(key, limit, window), nil
}

// GetUsageStats returns usage statistics for a key (simplified version)
func (etf *EventThrottlingFramework) GetUsageStats(key string) map[string]interface{} {
	etf.mutex.RLock()
	defer etf.mutex.RUnlock()

	stats := make(map[string]interface{})

	// Get token bucket stats
	if tokenBucketStrategy, exists := etf.strategies["token_bucket"].(*TokenBucketStrategy); exists {
		tokenBucketStrategy.mutex.RLock()
		if bucket, exists := tokenBucketStrategy.buckets[key]; exists {
			stats["bucket_tokens"] = bucket.Tokens
			stats["last_refill"] = bucket.LastRefill
		}
		tokenBucketStrategy.mutex.RUnlock()
	}

	// Get sliding window stats
	if slidingWindowStrategy, exists := etf.strategies["sliding_window"].(*SlidingWindowStrategy); exists {
		slidingWindowStrategy.mutex.RLock()
		if window, exists := slidingWindowStrategy.windows[key]; exists {
			stats["window_requests"] = len(window.Requests)
		}
		slidingWindowStrategy.mutex.RUnlock()
	}

	return stats
}

// AddStrategy adds a new throttling strategy
func (etf *EventThrottlingFramework) AddStrategy(name string, strategy ThrottlingStrategy) {
	etf.mutex.Lock()
	defer etf.mutex.Unlock()

	etf.strategies[name] = strategy
}

// min returns the minimum of two float64 values
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// Example usage
func main() {
	// Create throttling framework
	framework := NewEventThrottlingFramework()

	// Test token bucket strategy
	allowed, err := framework.Throttle("user123", 10, time.Minute, "token_bucket")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if allowed {
		fmt.Println("Request allowed")
	} else {
		fmt.Println("Request throttled")
	}

	// Test sliding window strategy
	allowed, err = framework.Throttle("user123", 5, time.Minute, "sliding_window")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if allowed {
		fmt.Println("Request allowed")
	} else {
		fmt.Println("Request throttled")
	}

	// Get usage stats
	stats := framework.GetUsageStats("user123")
	fmt.Printf("Usage stats: %+v\n", stats)
}
```

**Key Design Decisions:**

- **Multiple Strategies**: Token bucket for burst handling, sliding window for strict limits
- **Interface-based Design**: Extensible strategy pattern for different throttling algorithms
- **Thread-safe Operations**: Proper mutex usage for concurrent access
- **Configurable Limits**: Different limits per user/API key
- **Real-time Monitoring**: Usage statistics and monitoring capabilities
- **Go Idioms**: Proper error handling, constructor functions, and naming conventions

### Round 3: Technical Deep Dive (60 Minutes)

#### Key Areas to Prepare:

**1. Past Project Deep Dive**

**Sample Questions & Answers:**

**Q: Walk me through the most complex system you've designed and built.**

**A: "I designed and built a real-time analytics platform that processed 10M+ events per day. Here's the breakdown:**

**Problem:** We needed to process user behavior events in real-time and provide analytics dashboards with sub-second latency.

**Architecture Decisions:**

- **Event Ingestion**: Used Apache Kafka for event streaming
- **Processing**: Apache Storm for real-time stream processing
- **Storage**: ClickHouse for time-series analytics, Redis for caching
- **API**: RESTful APIs with GraphQL for complex queries

**Technical Challenges:**

1. **Data Consistency**: Implemented eventual consistency with idempotent processing
2. **Scalability**: Horizontal scaling with auto-partitioning
3. **Latency**: Optimized queries and implemented multi-level caching

**Results:**

- Reduced dashboard load time from 5s to 200ms
- Handled 10x traffic increase without performance degradation
- 99.9% uptime with proper monitoring and alerting"

**Q: What was your biggest technical failure and what did you learn?**

**A: "Early in my career, I designed a system that failed during a major product launch. Here's what happened:**

**Situation:** We were launching a new feature that required processing user data in real-time.

**What Went Wrong:**

- Underestimated data volume (10x more than expected)
- Used synchronous processing instead of async
- No proper error handling or circuit breakers
- Database became the bottleneck

**Impact:** System crashed within 2 hours of launch, affecting 50K+ users.

**What I Learned:**

1. **Always plan for 10x scale**: Design for worst-case scenarios
2. **Async processing**: Use message queues for decoupling
3. **Monitoring**: Implement comprehensive monitoring from day 1
4. **Graceful degradation**: Design systems to fail gracefully

**How I Applied This:**

- Implemented proper load testing before any major release
- Added circuit breakers and retry mechanisms
- Established monitoring and alerting systems
- Created runbooks for incident response"

**2. System Design Trade-offs**

**Q: How do you decide between SQL and NoSQL databases?**

**A: "I consider several factors:**

**Use SQL when:**

- ACID compliance is critical (financial transactions)
- Complex relationships between entities
- Strong consistency requirements
- Mature ecosystem and tooling

**Use NoSQL when:**

- High write throughput requirements
- Flexible schema needs
- Horizontal scaling is priority
- Eventual consistency is acceptable

**Example from my experience:**
For a user profile system, I chose PostgreSQL because:

- User data has complex relationships
- ACID compliance for profile updates
- Strong consistency for authentication
- Rich querying capabilities for analytics

For a logging system, I chose MongoDB because:

- High write volume (millions of logs/day)
- Flexible schema for different log types
- Easy horizontal scaling
- Eventual consistency was acceptable"

### Round 4: HM Round (60 Minutes)

#### Sample Questions & Answers:

**Q: Explain a situation where you felt overwhelmed with the project at hand.**

**A: "I was leading a team of 5 developers to migrate our monolithic application to microservices within 3 months while maintaining 99.9% uptime.**

**Situation:** The project involved breaking down a 500K+ line monolithic application into 15 microservices, each with different technology stacks and deployment pipelines.

**My Initial Reaction:** I felt overwhelmed because:

- Tight timeline with business pressure
- Complex interdependencies between services
- Team had limited microservices experience
- Risk of service disruptions during migration

**Actions I Took:**

1. **Broke down the problem**: Created a detailed migration plan with phases
2. **Leveraged team expertise**: Assigned services based on team members' strengths
3. **Implemented risk mitigation**: Created feature flags and rollback strategies
4. **Regular communication**: Daily standups and weekly stakeholder updates
5. **Continuous learning**: Organized microservices training sessions

**Result:** Successfully migrated 12 out of 15 services on time, with only 2 minor incidents. The remaining 3 services were completed in the following month. The team gained valuable microservices experience and the system became more maintainable.

**Key Learning:** Breaking down complex problems into manageable pieces and leveraging team strengths is crucial for success."

**Q: Why are you looking for a job change right now?**

**A: "I'm looking for new challenges and growth opportunities. Here's why Razorpay specifically interests me:**

**Current Situation:** I've been at my current company for 3 years, where I've grown from Senior Developer to Tech Lead. I've successfully led several major projects and built a strong team.

**Why I'm Looking to Move:**

1. **Scale and Impact**: Razorpay's scale (300M+ users) offers opportunities to work on systems that impact millions of users
2. **Technical Challenges**: The fintech domain presents unique challenges around security, compliance, and high availability
3. **Growth Opportunities**: The role offers both technical leadership and people management responsibilities
4. **Innovation**: Razorpay's focus on building India's financial infrastructure aligns with my passion for impactful technology

**What I Bring:**

- 6+ years of experience in building scalable systems
- Proven track record of leading technical teams
- Experience in fintech and payment systems
- Strong problem-solving and communication skills

**My Goals:**

- Contribute to Razorpay's mission of simplifying payments
- Lead technical initiatives that drive business growth
- Mentor and develop engineering talent
- Build systems that can handle India's scale"

**Q: What are you specifically interested in doing?**

**A: "I'm particularly interested in three areas:**

**1. Technical Leadership:**

- Leading architecture decisions for complex systems
- Building scalable, reliable, and maintainable software
- Driving technical excellence and best practices
- Mentoring engineers and building strong teams

**2. Fintech Innovation:**

- Working on payment processing systems
- Building fraud detection and risk management systems
- Implementing compliance and security measures
- Creating seamless user experiences

**3. System Design at Scale:**

- Designing distributed systems that handle millions of transactions
- Implementing real-time processing and analytics
- Building resilient systems with proper monitoring
- Optimizing for performance and cost efficiency

**Specific Interests at Razorpay:**

- Payment gateway architecture and optimization
- Risk management and fraud detection systems
- Developer experience and API design
- Data engineering and analytics platforms

**Long-term Vision:**
I want to be a technical leader who not only builds great products but also develops the next generation of engineers. I'm excited about the opportunity to work on systems that power India's digital economy."

### Round 5: HR Round (30 Minutes)

#### Sample Questions & Answers:

**Q: Tell me about your experience mentoring junior developers.**

**A: "I've been mentoring junior developers for the past 3 years, and it's one of the most rewarding aspects of my role.**

**My Mentoring Philosophy:**

- **Growth Mindset**: Focus on continuous learning and improvement
- **Hands-on Approach**: Pair programming and code reviews
- **Career Development**: Help them identify growth opportunities
- **Psychological Safety**: Create an environment where they feel comfortable asking questions

**Specific Examples:**

**1. Code Review Process:**
I established a structured code review process where:

- Every PR gets reviewed within 4 hours
- Reviews focus on learning, not just finding bugs
- I provide detailed explanations for suggestions
- Junior developers also review each other's code

**2. Technical Growth:**

- Created a "Tech Talk Friday" where team members present on new technologies
- Organized internal hackathons to encourage innovation
- Provided learning budgets for courses and conferences
- Implemented a buddy system for new joiners

**3. Career Development:**

- Regular 1:1s to discuss career goals and challenges
- Created individual development plans
- Provided opportunities to lead small projects
- Connected them with senior engineers for specific expertise

**Results:**

- 3 junior developers I mentored were promoted to mid-level within 2 years
- Team satisfaction scores improved from 7.2 to 9.1
- Reduced onboarding time for new developers from 3 weeks to 1 week
- Increased code quality metrics by 40%

**What I Learned:**
Mentoring is a two-way street. I've learned as much from my mentees as they've learned from me. It's taught me patience, communication skills, and the importance of adapting my teaching style to different learning preferences."

**Q: How do you approach building a positive team culture?**

**A: "Building a positive team culture is fundamental to team success. Here's my approach:**

**Core Values I Promote:**

1. **Transparency**: Open communication about challenges and successes
2. **Collaboration**: Team success over individual achievements
3. **Continuous Learning**: Encouraging experimentation and growth
4. **Ownership**: Taking responsibility for outcomes
5. **Inclusion**: Valuing diverse perspectives and backgrounds

**Practical Implementation:**

**1. Communication:**

- Weekly team retrospectives to discuss what's working and what's not
- Open-door policy for any team member to discuss concerns
- Regular team building activities (both technical and social)
- Transparent sharing of company and team goals

**2. Recognition and Feedback:**

- Peer recognition program where team members can nominate each other
- Regular feedback sessions (not just during performance reviews)
- Celebrating both technical achievements and personal milestones
- Constructive feedback focused on growth, not criticism

**3. Work-Life Balance:**

- Flexible working hours and remote work options
- No expectation of working late or weekends unless critical
- Encouraging team members to take their full vacation time
- Supporting personal development and learning

**4. Technical Culture:**

- Code review culture focused on learning and improvement
- Knowledge sharing sessions and tech talks
- Encouraging experimentation with new technologies
- Post-mortems for incidents focused on learning, not blame

**Results:**

- Team retention rate of 95% over 2 years
- High employee satisfaction scores (9.2/10)
- Increased productivity and code quality
- Strong cross-functional collaboration

**Key Learning:**
Culture is built through consistent actions, not just words. It requires ongoing effort and attention to maintain a positive environment where everyone can thrive."

---

## 🎯 Key Preparation Tips

### General Interview Tips:

1. **Think Out Loud**: Always verbalize your thought process
2. **Ask Clarifying Questions**: Understand requirements before jumping to solutions
3. **Start Simple**: Begin with basic solutions and iterate
4. **Consider Trade-offs**: Discuss pros and cons of different approaches
5. **Be Specific**: Use concrete examples from your experience

### Technical Preparation:

1. **Review Your Resume**: Be ready to discuss every project in detail
2. **Practice Coding**: Focus on clean, modular, and extensible code
3. **System Design**: Practice drawing diagrams and explaining architecture
4. **Know Your Tools**: Be familiar with technologies you've used
5. **Prepare Questions**: Have thoughtful questions about the role and company

### Behavioral Preparation:

1. **STAR Method**: Structure your answers with Situation, Task, Action, Result
2. **Prepare Examples**: Have 5-7 detailed examples ready
3. **Be Honest**: Don't exaggerate or make up experiences
4. **Show Growth**: Demonstrate learning from failures and challenges
5. **Research Company**: Understand Razorpay's mission and values

---

## 📚 Additional Resources

### Technical Resources:

- [System Design Primer](https://github.com/donnemartin/system-design-primer/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [High Scalability Blog](http://highscalability.com/)

### Razorpay Resources:

- [Razorpay Engineering Blog](https://engineering.razorpay.com/)
- [Razorpay LinkedIn](https://www.linkedin.com/company/razorpay/)
- [Razorpay Website](https://razorpay.com/)

### Practice Platforms:

- LeetCode for coding practice
- HackerRank for algorithm problems
- System Design Interview preparation

---

## 🎯 Final Checklist

### Day Before Interview:

- [ ] Review all prepared examples
- [ ] Practice coding problems (2-3)
- [ ] Review system design concepts
- [ ] Prepare questions for interviewers
- [ ] Get good sleep

### Interview Day:

- [ ] Arrive 10 minutes early
- [ ] Bring multiple copies of resume
- [ ] Have questions ready
- [ ] Stay calm and confident
- [ ] Think out loud
- [ ] Ask clarifying questions

### Post-Interview:

- [ ] Send thank you email
- [ ] Reflect on performance
- [ ] Note areas for improvement
- [ ] Follow up appropriately

---

**Good luck with your interview! Remember to be yourself, stay confident, and demonstrate your passion for technology and leadership. You've got this! 🚀**
