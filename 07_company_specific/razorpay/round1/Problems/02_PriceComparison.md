---
# Auto-generated front matter
Title: 02 Pricecomparison
LastUpdated: 2025-11-06T20:45:58.509099
Tags: []
Status: draft
---

# 02. Price Comparison - Multi-Vendor Price Aggregation Service

## Title & Summary
Design and implement a price comparison service that aggregates product prices from multiple vendors and provides real-time price updates with caching and rate limiting.

## Problem Statement

Build a price comparison system that:

1. **Vendor Integration**: Connect to multiple vendor APIs to fetch product prices
2. **Product Catalog**: Maintain a unified product catalog with vendor mappings
3. **Price Aggregation**: Collect and compare prices from different sources
4. **Real-time Updates**: Provide live price updates with WebSocket connections
5. **Search & Filter**: Allow users to search products and filter by price range
6. **Price Alerts**: Notify users when prices drop below their target

## Requirements & Constraints

### Functional Requirements
- Fetch prices from 3+ vendor APIs
- Unified product search across vendors
- Price comparison with historical data
- Real-time price updates
- User price alerts and notifications
- Product availability tracking

### Non-Functional Requirements
- **Latency**: < 200ms for price queries
- **Consistency**: Eventually consistent price data
- **Memory**: Cache 100K products with prices
- **Scalability**: Handle 10K concurrent users
- **Reliability**: 99.5% uptime for price data

## API / Interfaces

### REST Endpoints

```go
// Product Management
GET    /api/products/search?q={query}&category={category}
GET    /api/products/{productID}
GET    /api/products/{productID}/prices
GET    /api/products/{productID}/history

// Price Comparison
GET    /api/compare/{productID}
POST   /api/compare/bulk
GET    /api/categories/{categoryID}/products

// Price Alerts
POST   /api/alerts
GET    /api/alerts/{userID}
PUT    /api/alerts/{alertID}
DELETE /api/alerts/{alertID}

// WebSocket
WS     /ws/prices?productIDs={ids}
```

### Request/Response Examples

```json
// Search Products
GET /api/products/search?q=laptop&category=electronics
{
  "products": [
    {
      "id": "prod123",
      "name": "MacBook Pro 13\"",
      "category": "electronics",
      "vendors": [
        {
          "vendorID": "amazon",
          "price": 1299.99,
          "availability": "in_stock",
          "lastUpdated": "2024-01-15T10:30:00Z"
        }
      ]
    }
  ]
}

// Price Comparison
GET /api/compare/prod123
{
  "productID": "prod123",
  "productName": "MacBook Pro 13\"",
  "vendors": [
    {
      "vendorID": "amazon",
      "price": 1299.99,
      "availability": "in_stock",
      "url": "https://amazon.com/product/123"
    },
    {
      "vendorID": "bestbuy",
      "price": 1349.99,
      "availability": "in_stock",
      "url": "https://bestbuy.com/product/123"
    }
  ],
  "lowestPrice": 1299.99,
  "priceDifference": 50.00
}
```

## Data Model

### Core Entities

```go
type Product struct {
    ID          string            `json:"id"`
    Name        string            `json:"name"`
    Description string            `json:"description"`
    Category    string            `json:"category"`
    Brand       string            `json:"brand"`
    Model       string            `json:"model"`
    Images      []string          `json:"images"`
    Attributes  map[string]string `json:"attributes"`
    CreatedAt   time.Time         `json:"createdAt"`
    UpdatedAt   time.Time         `json:"updatedAt"`
}

type Vendor struct {
    ID       string `json:"id"`
    Name     string `json:"name"`
    BaseURL  string `json:"baseURL"`
    APIKey   string `json:"apiKey"`
    RateLimit int   `json:"rateLimit"` // requests per minute
    Active   bool   `json:"active"`
}

type ProductPrice struct {
    ProductID   string    `json:"productID"`
    VendorID    string    `json:"vendorID"`
    Price       float64   `json:"price"`
    Currency    string    `json:"currency"`
    Availability string   `json:"availability"`
    URL         string    `json:"url"`
    LastUpdated time.Time `json:"lastUpdated"`
    ValidUntil  time.Time `json:"validUntil"`
}

type PriceAlert struct {
    ID        string    `json:"id"`
    UserID    string    `json:"userID"`
    ProductID string    `json:"productID"`
    TargetPrice float64 `json:"targetPrice"`
    IsActive  bool      `json:"isActive"`
    CreatedAt time.Time `json:"createdAt"`
    TriggeredAt *time.Time `json:"triggeredAt,omitempty"`
}

type PriceHistory struct {
    ProductID string    `json:"productID"`
    VendorID  string    `json:"vendorID"`
    Price     float64   `json:"price"`
    Timestamp time.Time `json:"timestamp"`
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory product catalog with basic price storage
2. Simple HTTP client for vendor API calls
3. Basic price comparison logic
4. No caching or rate limiting

### Production-Ready Design
1. **Microservices Architecture**: Separate services for products, prices, alerts
2. **Caching Layer**: Redis for price data and product catalog
3. **Message Queue**: Async price updates and alert processing
4. **Rate Limiting**: Per-vendor API rate limiting
5. **Circuit Breaker**: Handle vendor API failures gracefully
6. **Data Pipeline**: ETL for price data processing

## Detailed Design

### Modular Decomposition

```go
pricecomparison/
├── products/        # Product catalog management
├── vendors/         # Vendor API integration
├── prices/          # Price aggregation and comparison
├── alerts/          # Price alert system
├── cache/           # Caching layer
├── websocket/       # Real-time updates
└── workers/         # Background job processing
```

### Concurrency Model

```go
type PriceService struct {
    products    map[string]*Product
    vendors     map[string]*Vendor
    prices      map[string][]ProductPrice
    alerts      map[string][]PriceAlert
    cache       *redis.Client
    httpClient  *http.Client
    rateLimiter *rate.Limiter
    mutex       sync.RWMutex
    priceChan   chan PriceUpdate
    alertChan   chan AlertCheck
}

// Goroutines for:
// 1. Price fetching from vendors
// 2. Alert processing
// 3. Cache warming
// 4. WebSocket broadcasting
```

### Persistence Strategy

```go
// Redis for caching
type CacheService struct {
    client *redis.Client
    ttl    time.Duration
}

// Database for persistence
CREATE TABLE products (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    brand VARCHAR(100),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE product_prices (
    id VARCHAR(36) PRIMARY KEY,
    product_id VARCHAR(36),
    vendor_id VARCHAR(36),
    price DECIMAL(10,2),
    availability VARCHAR(20),
    last_updated TIMESTAMP,
    valid_until TIMESTAMP
);
```

## Optimal Golang Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/go-redis/redis/v8"
    "github.com/gorilla/websocket"
    "golang.org/x/time/rate"
)

type Product struct {
    ID          string            `json:"id"`
    Name        string            `json:"name"`
    Description string            `json:"description"`
    Category    string            `json:"category"`
    Brand       string            `json:"brand"`
    Model       string            `json:"model"`
    Images      []string          `json:"images"`
    Attributes  map[string]string `json:"attributes"`
    CreatedAt   time.Time         `json:"createdAt"`
    UpdatedAt   time.Time         `json:"updatedAt"`
}

type Vendor struct {
    ID       string `json:"id"`
    Name     string `json:"name"`
    BaseURL  string `json:"baseURL"`
    APIKey   string `json:"apiKey"`
    RateLimit int   `json:"rateLimit"`
    Active   bool   `json:"active"`
}

type ProductPrice struct {
    ProductID   string    `json:"productID"`
    VendorID    string    `json:"vendorID"`
    Price       float64   `json:"price"`
    Currency    string    `json:"currency"`
    Availability string   `json:"availability"`
    URL         string    `json:"url"`
    LastUpdated time.Time `json:"lastUpdated"`
    ValidUntil  time.Time `json:"validUntil"`
}

type PriceAlert struct {
    ID        string    `json:"id"`
    UserID    string    `json:"userID"`
    ProductID string    `json:"productID"`
    TargetPrice float64 `json:"targetPrice"`
    IsActive  bool      `json:"isActive"`
    CreatedAt time.Time `json:"createdAt"`
    TriggeredAt *time.Time `json:"triggeredAt,omitempty"`
}

type PriceComparison struct {
    ProductID   string         `json:"productID"`
    ProductName string         `json:"productName"`
    Vendors     []ProductPrice `json:"vendors"`
    LowestPrice float64        `json:"lowestPrice"`
    PriceDifference float64    `json:"priceDifference"`
}

type PriceService struct {
    products    map[string]*Product
    vendors     map[string]*Vendor
    prices      map[string][]ProductPrice
    alerts      map[string][]PriceAlert
    cache       *redis.Client
    httpClient  *http.Client
    rateLimiter *rate.Limiter
    mutex       sync.RWMutex
    priceChan   chan PriceUpdate
    alertChan   chan AlertCheck
}

type PriceUpdate struct {
    ProductID string
    VendorID  string
    Price     float64
    Timestamp time.Time
}

type AlertCheck struct {
    ProductID string
    Price     float64
}

func NewPriceService() *PriceService {
    rdb := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
    })

    return &PriceService{
        products:    make(map[string]*Product),
        vendors:     make(map[string]*Vendor),
        prices:      make(map[string][]ProductPrice),
        alerts:      make(map[string][]PriceAlert),
        cache:       rdb,
        httpClient:  &http.Client{Timeout: 10 * time.Second},
        rateLimiter: rate.NewLimiter(rate.Every(time.Minute), 100),
        priceChan:   make(chan PriceUpdate, 1000),
        alertChan:   make(chan AlertCheck, 1000),
    }
}

func (ps *PriceService) SearchProducts(query, category string) ([]Product, error) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    var results []Product
    for _, product := range ps.products {
        if matchesQuery(product, query, category) {
            results = append(results, *product)
        }
    }

    return results, nil
}

func (ps *PriceService) GetProductPrices(productID string) ([]ProductPrice, error) {
    ps.mutex.RLock()
    prices, exists := ps.prices[productID]
    ps.mutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("product not found")
    }

    return prices, nil
}

func (ps *PriceService) ComparePrices(productID string) (*PriceComparison, error) {
    ps.mutex.RLock()
    product, exists := ps.products[productID]
    prices, priceExists := ps.prices[productID]
    ps.mutex.RUnlock()

    if !exists || !priceExists {
        return nil, fmt.Errorf("product not found")
    }

    if len(prices) == 0 {
        return nil, fmt.Errorf("no prices available")
    }

    lowestPrice := prices[0].Price
    highestPrice := prices[0].Price

    for _, price := range prices {
        if price.Price < lowestPrice {
            lowestPrice = price.Price
        }
        if price.Price > highestPrice {
            highestPrice = price.Price
        }
    }

    return &PriceComparison{
        ProductID:      productID,
        ProductName:    product.Name,
        Vendors:        prices,
        LowestPrice:    lowestPrice,
        PriceDifference: highestPrice - lowestPrice,
    }, nil
}

func (ps *PriceService) CreatePriceAlert(userID, productID string, targetPrice float64) (*PriceAlert, error) {
    alert := &PriceAlert{
        ID:        fmt.Sprintf("alert_%d", time.Now().UnixNano()),
        UserID:    userID,
        ProductID: productID,
        TargetPrice: targetPrice,
        IsActive:  true,
        CreatedAt: time.Now(),
    }

    ps.mutex.Lock()
    ps.alerts[userID] = append(ps.alerts[userID], *alert)
    ps.mutex.Unlock()

    return alert, nil
}

func (ps *PriceService) FetchPricesFromVendor(vendorID, productID string) error {
    ps.mutex.RLock()
    vendor, exists := ps.vendors[vendorID]
    ps.mutex.RUnlock()

    if !exists || !vendor.Active {
        return fmt.Errorf("vendor not available")
    }

    // Rate limiting
    if !ps.rateLimiter.Allow() {
        return fmt.Errorf("rate limit exceeded")
    }

    // Simulate API call
    url := fmt.Sprintf("%s/products/%s/price", vendor.BaseURL, productID)
    req, err := http.NewRequest("GET", url, nil)
    if err != nil {
        return err
    }

    req.Header.Set("Authorization", "Bearer "+vendor.APIKey)
    req.Header.Set("X-API-Key", vendor.APIKey)

    resp, err := ps.httpClient.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("vendor API error: %d", resp.StatusCode)
    }

    var priceData struct {
        Price       float64 `json:"price"`
        Availability string `json:"availability"`
        URL         string `json:"url"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&priceData); err != nil {
        return err
    }

    // Update price
    price := ProductPrice{
        ProductID:   productID,
        VendorID:    vendorID,
        Price:       priceData.Price,
        Currency:    "USD",
        Availability: priceData.Availability,
        URL:         priceData.URL,
        LastUpdated: time.Now(),
        ValidUntil:  time.Now().Add(1 * time.Hour),
    }

    ps.mutex.Lock()
    ps.prices[productID] = append(ps.prices[productID], price)
    ps.mutex.Unlock()

    // Send to price update channel
    ps.priceChan <- PriceUpdate{
        ProductID: productID,
        VendorID:  vendorID,
        Price:     priceData.Price,
        Timestamp: time.Now(),
    }

    return nil
}

func (ps *PriceService) ProcessPriceUpdates() {
    for update := range ps.priceChan {
        // Check alerts for this product
        ps.mutex.RLock()
        for userID, alerts := range ps.alerts {
            for _, alert := range alerts {
                if alert.ProductID == update.ProductID && alert.IsActive {
                    if update.Price <= alert.TargetPrice {
                        // Trigger alert
                        ps.alertChan <- AlertCheck{
                            ProductID: update.ProductID,
                            Price:     update.Price,
                        }
                    }
                }
            }
        }
        ps.mutex.RUnlock()

        // Update cache
        cacheKey := fmt.Sprintf("price:%s", update.ProductID)
        ps.cache.Set(context.Background(), cacheKey, update.Price, time.Hour)
    }
}

func (ps *PriceService) ProcessAlerts() {
    for alert := range ps.alertChan {
        // Send notification (email, push, etc.)
        log.Printf("Price alert triggered for product %s at price %.2f", 
            alert.ProductID, alert.Price)
    }
}

// HTTP Handlers
func (ps *PriceService) SearchHandler(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    category := r.URL.Query().Get("category")

    products, err := ps.SearchProducts(query, category)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "products": products,
    })
}

func (ps *PriceService) CompareHandler(w http.ResponseWriter, r *http.Request) {
    productID := r.URL.Path[len("/api/compare/"):]

    comparison, err := ps.ComparePrices(productID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(comparison)
}

func (ps *PriceService) CreateAlertHandler(w http.ResponseWriter, r *http.Request) {
    var req struct {
        UserID      string  `json:"userID"`
        ProductID   string  `json:"productID"`
        TargetPrice float64 `json:"targetPrice"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    alert, err := ps.CreatePriceAlert(req.UserID, req.ProductID, req.TargetPrice)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(alert)
}

func main() {
    service := NewPriceService()

    // Initialize sample data
    service.products["prod1"] = &Product{
        ID: "prod1", Name: "MacBook Pro 13\"", Category: "electronics",
        Brand: "Apple", CreatedAt: time.Now(),
    }
    service.vendors["amazon"] = &Vendor{
        ID: "amazon", Name: "Amazon", BaseURL: "https://api.amazon.com",
        APIKey: "amazon_key", RateLimit: 100, Active: true,
    }

    // Start background workers
    go service.ProcessPriceUpdates()
    go service.ProcessAlerts()

    // HTTP routes
    http.HandleFunc("/api/products/search", service.SearchHandler)
    http.HandleFunc("/api/compare/", service.CompareHandler)
    http.HandleFunc("/api/alerts", service.CreateAlertHandler)

    log.Println("Price comparison service starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func matchesQuery(product *Product, query, category string) bool {
    if category != "" && product.Category != category {
        return false
    }
    if query != "" {
        return contains(product.Name, query) || contains(product.Brand, query)
    }
    return true
}

func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}
```

## Unit Tests

```go
func TestPriceService_SearchProducts(t *testing.T) {
    service := NewPriceService()
    
    // Add test products
    service.products["prod1"] = &Product{
        ID: "prod1", Name: "MacBook Pro", Category: "electronics",
        Brand: "Apple", CreatedAt: time.Now(),
    }
    service.products["prod2"] = &Product{
        ID: "prod2", Name: "iPhone 15", Category: "electronics",
        Brand: "Apple", CreatedAt: time.Now(),
    }

    tests := []struct {
        name     string
        query    string
        category string
        expected int
    }{
        {"search by name", "MacBook", "", 1},
        {"search by category", "", "electronics", 2},
        {"search by brand", "Apple", "", 2},
        {"no matches", "Samsung", "", 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            products, err := service.SearchProducts(tt.query, tt.category)
            if err != nil {
                t.Fatalf("SearchProducts() error = %v", err)
            }
            if len(products) != tt.expected {
                t.Errorf("SearchProducts() returned %d products, want %d", 
                    len(products), tt.expected)
            }
        })
    }
}

func TestPriceService_ComparePrices(t *testing.T) {
    service := NewPriceService()
    
    // Add test product and prices
    service.products["prod1"] = &Product{
        ID: "prod1", Name: "Test Product", Category: "electronics",
    }
    service.prices["prod1"] = []ProductPrice{
        {ProductID: "prod1", VendorID: "vendor1", Price: 100.0},
        {ProductID: "prod1", VendorID: "vendor2", Price: 120.0},
        {ProductID: "prod1", VendorID: "vendor3", Price: 90.0},
    }

    comparison, err := service.ComparePrices("prod1")
    if err != nil {
        t.Fatalf("ComparePrices() error = %v", err)
    }

    if comparison.LowestPrice != 90.0 {
        t.Errorf("ComparePrices() lowestPrice = %v, want 90.0", comparison.LowestPrice)
    }

    if comparison.PriceDifference != 30.0 {
        t.Errorf("ComparePrices() priceDifference = %v, want 30.0", comparison.PriceDifference)
    }
}
```

## Complexity Analysis

### Time Complexity
- **Search Products**: O(n) - Linear scan through products
- **Compare Prices**: O(m) - Linear scan through prices for a product
- **Create Alert**: O(1) - Hash map insertion
- **Fetch Prices**: O(1) - HTTP request + hash map update

### Space Complexity
- **Product Storage**: O(P) where P is number of products
- **Price Storage**: O(P × V) where V is number of vendors per product
- **Alert Storage**: O(A) where A is number of alerts
- **Total**: O(P + P×V + A)

## Edge Cases & Validation

### Input Validation
- Empty search queries
- Invalid product IDs
- Negative target prices
- Invalid vendor configurations
- Malformed API responses

### Error Scenarios
- Vendor API timeouts
- Network connectivity issues
- Rate limit exceeded
- Invalid product data
- Cache failures

### Boundary Conditions
- Maximum search results (1000)
- Price alert limits per user (50)
- Vendor API rate limits
- Cache TTL expiration
- WebSocket connection limits

## Extension Ideas (Scaling)

### Horizontal Scaling
1. **Load Balancing**: Multiple service instances
2. **Database Sharding**: Partition by product category
3. **Cache Clustering**: Redis cluster for high availability
4. **Message Queue**: Kafka for price updates

### Performance Optimization
1. **Price Caching**: Redis with TTL for price data
2. **Search Indexing**: Elasticsearch for product search
3. **CDN Integration**: Static product data delivery
4. **Connection Pooling**: HTTP client optimization

### Advanced Features
1. **Machine Learning**: Price prediction and trend analysis
2. **Personalization**: User-specific recommendations
3. **Price History**: Historical price tracking and charts
4. **Vendor Analytics**: Performance metrics and insights

## 20 Follow-up Questions

### 1. How would you handle vendor API rate limits?
**Answer**: Implement token bucket algorithm with per-vendor rate limiters. Use circuit breaker pattern for failed vendors. Queue requests when rate limits are exceeded and process them as tokens become available.

### 2. What happens if a vendor API is down?
**Answer**: Use circuit breaker pattern to fail fast. Implement fallback to cached data with TTL. Use health checks to detect vendor recovery. Consider vendor redundancy for critical products.

### 3. How do you ensure price data consistency?
**Answer**: Use eventual consistency with timestamps. Implement data validation rules and outlier detection. Use consensus algorithms for critical price updates. Consider using event sourcing for audit trails.

### 4. What's your strategy for handling price spikes?
**Answer**: Implement price change thresholds and validation rules. Use moving averages to detect anomalies. Implement manual review for significant price changes. Consider using machine learning for fraud detection.

### 5. How would you implement price alerts efficiently?
**Answer**: Use Redis sorted sets for price thresholds. Implement batch processing for alert checks. Use WebSocket for real-time notifications. Consider using message queues for reliable delivery.

### 6. What happens if the cache is down?
**Answer**: Implement fallback to database queries. Use multiple cache layers (L1, L2). Implement cache warming strategies. Consider using in-memory caches as backup.

### 7. How do you handle product catalog updates?
**Answer**: Use event-driven architecture with message queues. Implement incremental updates with change detection. Use database triggers for real-time updates. Consider using CDC (Change Data Capture) for synchronization.

### 8. What's your approach to vendor onboarding?
**Answer**: Implement vendor API abstraction layer. Use configuration-driven vendor setup. Implement API testing and validation. Consider using vendor-specific adapters for different API formats.

### 9. How would you implement price history tracking?
**Answer**: Use time-series database for price data. Implement data retention policies. Use compression for historical data. Consider using data lakes for long-term storage.

### 10. What's your strategy for handling duplicate products?
**Answer**: Implement product deduplication algorithms. Use fuzzy matching for product names. Implement manual review for ambiguous cases. Consider using machine learning for product matching.

### 11. How do you handle product availability changes?
**Answer**: Implement real-time availability updates. Use WebSocket for instant notifications. Implement availability caching with TTL. Consider using vendor webhooks for availability changes.

### 12. What's your approach to price validation?
**Answer**: Implement price range validation rules. Use statistical analysis for outlier detection. Implement manual review for suspicious prices. Consider using machine learning for price validation.

### 13. How would you implement product recommendations?
**Answer**: Use collaborative filtering algorithms. Implement content-based recommendations. Use machine learning for personalized suggestions. Consider using A/B testing for recommendation algorithms.

### 14. What's your strategy for handling seasonal price changes?
**Answer**: Implement seasonal price tracking. Use historical data for price predictions. Implement seasonal alert adjustments. Consider using time-series forecasting for price trends.

### 15. How do you handle product image updates?
**Answer**: Use CDN for image delivery. Implement image optimization and compression. Use vendor APIs for image updates. Consider using image recognition for product matching.

### 16. What's your approach to vendor performance monitoring?
**Answer**: Implement vendor API response time monitoring. Use health checks for vendor availability. Implement vendor performance metrics. Consider using alerting for vendor issues.

### 17. How would you implement price comparison analytics?
**Answer**: Use data warehouse for analytics. Implement real-time dashboards. Use machine learning for price trend analysis. Consider using business intelligence tools for reporting.

### 18. What's your strategy for handling product reviews?
**Answer**: Integrate with review APIs. Implement review aggregation and scoring. Use sentiment analysis for review processing. Consider using review data for product recommendations.

### 19. How do you handle product specifications?
**Answer**: Implement structured product data model. Use vendor APIs for specification updates. Implement specification comparison features. Consider using machine learning for specification matching.

### 20. What's your approach to international pricing?
**Answer**: Implement multi-currency support. Use exchange rate APIs for currency conversion. Implement region-specific pricing. Consider using local vendor APIs for regional pricing.

## Evaluation Checklist

### Code Quality (25%)
- [ ] Clean, readable Go code with proper error handling
- [ ] Appropriate use of interfaces and structs
- [ ] Proper concurrency patterns (goroutines, channels)
- [ ] Good separation of concerns

### Architecture (25%)
- [ ] Scalable design with caching and rate limiting
- [ ] Proper vendor API integration
- [ ] Efficient data structures and algorithms
- [ ] Background job processing

### Functionality (25%)
- [ ] Product search and comparison working
- [ ] Price alert system functional
- [ ] Vendor API integration working
- [ ] Real-time updates implemented

### Testing (15%)
- [ ] Unit tests for core functionality
- [ ] Integration tests for API endpoints
- [ ] Mock vendor API responses
- [ ] Edge case testing

### Discussion (10%)
- [ ] Clear explanation of design decisions
- [ ] Understanding of scaling challenges
- [ ] Knowledge of caching strategies
- [ ] Ability to discuss trade-offs

## Discussion Pointers

### Key Points to Highlight
1. **Caching Strategy**: Explain Redis usage for price data and product catalog
2. **Rate Limiting**: Discuss per-vendor rate limiting and circuit breaker patterns
3. **Data Consistency**: Explain eventual consistency and validation strategies
4. **Scalability**: Discuss horizontal scaling and performance optimization
5. **Error Handling**: Explain graceful degradation and fallback strategies

### Trade-offs to Discuss
1. **Consistency vs Performance**: Real-time updates vs caching trade-offs
2. **Accuracy vs Speed**: Price validation vs response time trade-offs
3. **Storage vs Computation**: Caching vs real-time calculation trade-offs
4. **Reliability vs Cost**: Vendor redundancy vs infrastructure cost trade-offs
5. **Features vs Complexity**: Advanced features vs system complexity trade-offs

### Extension Scenarios
1. **Multi-region Deployment**: How to handle geographic distribution
2. **Machine Learning Integration**: Price prediction and recommendation systems
3. **Real-time Analytics**: Live dashboards and monitoring
4. **Mobile App Support**: Push notifications and offline capabilities
5. **Enterprise Features**: Admin controls and vendor management
