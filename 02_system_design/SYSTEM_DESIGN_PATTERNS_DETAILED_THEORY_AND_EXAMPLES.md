# 🏗️ **System Design Patterns - Detailed Theory & Examples**

## 📊 **Comprehensive Guide with Theory, Examples, and Practical Implementations**

---

## 🎯 **1. Load Balancing Patterns - Deep Dive with Examples**

### **Theory: What is Load Balancing?**

Load balancing distributes incoming requests across multiple servers to ensure no single server is overwhelmed.

**Benefits:**

- **High Availability**: System continues working if some servers fail
- **Scalability**: Handle more requests by adding more servers
- **Performance**: Distribute load evenly across servers
- **Fault Tolerance**: Isolate failures to specific servers

### **Real-World Example: E-commerce API Gateway**

```go
type APIGateway struct {
    loadBalancer LoadBalancer
    servers      []Server
    healthChecker *HealthChecker
    mutex        sync.RWMutex
}

type LoadBalancer interface {
    SelectServer(servers []Server) Server
}

type Server struct {
    ID           string
    Address      string
    Port         int
    Weight       int
    Connections  int
    IsHealthy    bool
    mutex        sync.RWMutex
}

type HealthChecker struct {
    checkInterval time.Duration
    timeout       time.Duration
}

// Round Robin Load Balancer
type RoundRobinLoadBalancer struct {
    current int
    mutex   sync.Mutex
}

func (rrlb *RoundRobinLoadBalancer) SelectServer(servers []Server) Server {
    rrlb.mutex.Lock()
    defer rrlb.mutex.Unlock()

    if len(servers) == 0 {
        return Server{}
    }

    server := servers[rrlb.current]
    rrlb.current = (rrlb.current + 1) % len(servers)

    return server
}

// Least Connections Load Balancer
type LeastConnectionsLoadBalancer struct{}

func (lclb *LeastConnectionsLoadBalancer) SelectServer(servers []Server) Server {
    if len(servers) == 0 {
        return Server{}
    }

    minConnections := servers[0].GetConnectionCount()
    selectedServer := servers[0]

    for _, server := range servers[1:] {
        connections := server.GetConnectionCount()
        if connections < minConnections {
            minConnections = connections
            selectedServer = server
        }
    }

    return selectedServer
}

// Weighted Round Robin Load Balancer
type WeightedRoundRobinLoadBalancer struct {
    current int
    mutex   sync.Mutex
}

func (wrrlb *WeightedRoundRobinLoadBalancer) SelectServer(servers []Server) Server {
    wrrlb.mutex.Lock()
    defer wrrlb.mutex.Unlock()

    if len(servers) == 0 {
        return Server{}
    }

    totalWeight := 0
    for _, server := range servers {
        totalWeight += server.Weight
    }

    for i := range servers {
        servers[i].Connections += servers[i].Weight
        if servers[i].Connections >= totalWeight {
            servers[i].Connections -= totalWeight
            return servers[i]
        }
    }

    return servers[0]
}

func (s *Server) GetConnectionCount() int {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Connections
}

func (s *Server) IncrementConnections() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Connections++
}

func (s *Server) DecrementConnections() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Connections--
}

func NewAPIGateway() *APIGateway {
    return &APIGateway{
        loadBalancer: &RoundRobinLoadBalancer{},
        servers:      make([]Server, 0),
        healthChecker: &HealthChecker{
            checkInterval: 30 * time.Second,
            timeout:       5 * time.Second,
        },
    }
}

func (ag *APIGateway) AddServer(server Server) {
    ag.mutex.Lock()
    defer ag.mutex.Unlock()

    ag.servers = append(ag.servers, server)
}

func (ag *APIGateway) HandleRequest(req *Request) (*Response, error) {
    // Get healthy servers
    healthyServers := ag.getHealthyServers()
    if len(healthyServers) == 0 {
        return nil, errors.New("no healthy servers available")
    }

    // Select server using load balancer
    server := ag.loadBalancer.SelectServer(healthyServers)

    // Increment connection count
    server.IncrementConnections()
    defer server.DecrementConnections()

    // Forward request to selected server
    return ag.forwardRequest(server, req)
}

func (ag *APIGateway) getHealthyServers() []Server {
    ag.mutex.RLock()
    defer ag.mutex.RUnlock()

    var healthyServers []Server
    for _, server := range ag.servers {
        if server.IsHealthy {
            healthyServers = append(healthyServers, server)
        }
    }

    return healthyServers
}

func (ag *APIGateway) forwardRequest(server Server, req *Request) (*Response, error) {
    // In real implementation, this would make HTTP request to server
    fmt.Printf("Forwarding request to server %s\n", server.ID)

    // Simulate response
    return &Response{
        StatusCode: 200,
        Body:       "Response from " + server.ID,
    }, nil
}

func (ag *APIGateway) StartHealthChecking() {
    ticker := time.NewTicker(ag.healthChecker.checkInterval)
    go func() {
        for range ticker.C {
            ag.checkServerHealth()
        }
    }()
}

func (ag *APIGateway) checkServerHealth() {
    ag.mutex.RLock()
    servers := make([]Server, len(ag.servers))
    copy(servers, ag.servers)
    ag.mutex.RUnlock()

    for i := range servers {
        go func(server *Server) {
            // Simulate health check
            isHealthy := ag.performHealthCheck(server)

            ag.mutex.Lock()
            for j := range ag.servers {
                if ag.servers[j].ID == server.ID {
                    ag.servers[j].IsHealthy = isHealthy
                    break
                }
            }
            ag.mutex.Unlock()
        }(&servers[i])
    }
}

func (ag *APIGateway) performHealthCheck(server *Server) bool {
    // In real implementation, this would make HTTP request to health endpoint
    // For simulation, randomly return true/false
    return rand.Float32() > 0.1 // 90% chance of being healthy
}

// Example usage
func main() {
    gateway := NewAPIGateway()

    // Add servers
    gateway.AddServer(Server{ID: "server1", Address: "192.168.1.1", Port: 8080, Weight: 1, IsHealthy: true})
    gateway.AddServer(Server{ID: "server2", Address: "192.168.1.2", Port: 8080, Weight: 2, IsHealthy: true})
    gateway.AddServer(Server{ID: "server3", Address: "192.168.1.3", Port: 8080, Weight: 1, IsHealthy: true})

    // Start health checking
    gateway.StartHealthChecking()

    // Handle some requests
    for i := 0; i < 10; i++ {
        req := &Request{ID: fmt.Sprintf("req_%d", i)}
        resp, err := gateway.HandleRequest(req)
        if err != nil {
            fmt.Printf("Request failed: %v\n", err)
        } else {
            fmt.Printf("Request %d: %s\n", i, resp.Body)
        }
    }
}
```

---

## 🗄️ **2. Caching Patterns - Deep Dive with Examples**

### **Theory: What is Caching?**

Caching stores frequently accessed data in fast storage to reduce latency and improve performance.

**Benefits:**

- **Reduced Latency**: Faster access to frequently used data
- **Reduced Load**: Less pressure on backend systems
- **Better Performance**: Improved user experience
- **Cost Savings**: Reduced infrastructure costs

### **Real-World Example: E-commerce Product Catalog with Caching**

```go
type ProductCatalog struct {
    cache      Cache
    database   Database
    mutex      sync.RWMutex
}

type Cache interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{}, ttl time.Duration) error
    Delete(key string) error
    Clear() error
}

type Database interface {
    GetProduct(productID string) (*Product, error)
    GetProducts(category string) ([]*Product, error)
    UpdateProduct(product *Product) error
}

type Product struct {
    ID          string
    Name        string
    Price       float64
    Category    string
    Description string
    Stock       int
    LastUpdated time.Time
}

// In-Memory Cache Implementation
type InMemoryCache struct {
    data    map[string]*CacheEntry
    mutex   sync.RWMutex
    maxSize int
    evictionPolicy string
}

type CacheEntry struct {
    Value     interface{}
    ExpiresAt time.Time
    CreatedAt time.Time
    AccessCount int
}

func NewInMemoryCache(maxSize int, evictionPolicy string) *InMemoryCache {
    return &InMemoryCache{
        data:          make(map[string]*CacheEntry),
        maxSize:       maxSize,
        evictionPolicy: evictionPolicy,
    }
}

func (cache *InMemoryCache) Get(key string) (interface{}, bool) {
    cache.mutex.RLock()
    entry, exists := cache.data[key]
    cache.mutex.RUnlock()

    if !exists {
        return nil, false
    }

    // Check if expired
    if time.Now().After(entry.ExpiresAt) {
        cache.mutex.Lock()
        delete(cache.data, key)
        cache.mutex.Unlock()
        return nil, false
    }

    // Update access count
    cache.mutex.Lock()
    entry.AccessCount++
    cache.mutex.Unlock()

    return entry.Value, true
}

func (cache *InMemoryCache) Set(key string, value interface{}, ttl time.Duration) error {
    cache.mutex.Lock()
    defer cache.mutex.Unlock()

    // Check if cache is full
    if len(cache.data) >= cache.maxSize {
        cache.evictEntry()
    }

    entry := &CacheEntry{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        CreatedAt:  time.Now(),
        AccessCount: 1,
    }

    cache.data[key] = entry
    return nil
}

func (cache *InMemoryCache) Delete(key string) error {
    cache.mutex.Lock()
    defer cache.mutex.Unlock()

    delete(cache.data, key)
    return nil
}

func (cache *InMemoryCache) Clear() error {
    cache.mutex.Lock()
    defer cache.mutex.Unlock()

    cache.data = make(map[string]*CacheEntry)
    return nil
}

func (cache *InMemoryCache) evictEntry() {
    switch cache.evictionPolicy {
    case "LRU":
        cache.evictLRU()
    case "LFU":
        cache.evictLFU()
    case "FIFO":
        cache.evictFIFO()
    default:
        cache.evictLRU()
    }
}

func (cache *InMemoryCache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time

    for key, entry := range cache.data {
        if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.CreatedAt
        }
    }

    if oldestKey != "" {
        delete(cache.data, oldestKey)
    }
}

func (cache *InMemoryCache) evictLFU() {
    var leastUsedKey string
    var leastUsedCount int

    for key, entry := range cache.data {
        if leastUsedKey == "" || entry.AccessCount < leastUsedCount {
            leastUsedKey = key
            leastUsedCount = entry.AccessCount
        }
    }

    if leastUsedKey != "" {
        delete(cache.data, leastUsedKey)
    }
}

func (cache *InMemoryCache) evictFIFO() {
    // Remove first inserted entry
    for key := range cache.data {
        delete(cache.data, key)
        break
    }
}

// Redis Cache Implementation
type RedisCache struct {
    client *redis.Client
}

func NewRedisCache(addr string) *RedisCache {
    client := redis.NewClient(&redis.Options{
        Addr: addr,
    })

    return &RedisCache{client: client}
}

func (cache *RedisCache) Get(key string) (interface{}, bool) {
    val, err := cache.client.Get(key).Result()
    if err != nil {
        return nil, false
    }

    // Deserialize value
    var product Product
    if err := json.Unmarshal([]byte(val), &product); err != nil {
        return nil, false
    }

    return &product, true
}

func (cache *RedisCache) Set(key string, value interface{}, ttl time.Duration) error {
    // Serialize value
    data, err := json.Marshal(value)
    if err != nil {
        return err
    }

    return cache.client.Set(key, data, ttl).Err()
}

func (cache *RedisCache) Delete(key string) error {
    return cache.client.Del(key).Err()
}

func (cache *RedisCache) Clear() error {
    return cache.client.FlushDB().Err()
}

// Product Catalog with Caching
func NewProductCatalog(cache Cache, database Database) *ProductCatalog {
    return &ProductCatalog{
        cache:    cache,
        database: database,
    }
}

func (pc *ProductCatalog) GetProduct(productID string) (*Product, error) {
    // Try cache first
    if value, found := pc.cache.Get("product:" + productID); found {
        if product, ok := value.(*Product); ok {
            return product, nil
        }
    }

    // Cache miss, get from database
    product, err := pc.database.GetProduct(productID)
    if err != nil {
        return nil, err
    }

    // Cache the product
    pc.cache.Set("product:"+productID, product, 5*time.Minute)

    return product, nil
}

func (pc *ProductCatalog) GetProducts(category string) ([]*Product, error) {
    // Try cache first
    if value, found := pc.cache.Get("products:" + category); found {
        if products, ok := value.([]*Product); ok {
            return products, nil
        }
    }

    // Cache miss, get from database
    products, err := pc.database.GetProducts(category)
    if err != nil {
        return nil, err
    }

    // Cache the products
    pc.cache.Set("products:"+category, products, 10*time.Minute)

    return products, nil
}

func (pc *ProductCatalog) UpdateProduct(product *Product) error {
    // Update database
    if err := pc.database.UpdateProduct(product); err != nil {
        return err
    }

    // Invalidate cache
    pc.cache.Delete("product:" + product.ID)
    pc.cache.Delete("products:" + product.Category)

    return nil
}

// Example usage
func main() {
    // Create cache and database
    cache := NewInMemoryCache(1000, "LRU")
    database := &MockDatabase{}

    // Create product catalog
    catalog := NewProductCatalog(cache, database)

    // Get product (will be cached)
    product, err := catalog.GetProduct("p1")
    if err == nil {
        fmt.Printf("Product: %s - $%.2f\n", product.Name, product.Price)
    }

    // Get same product again (will come from cache)
    product, err = catalog.GetProduct("p1")
    if err == nil {
        fmt.Printf("Product from cache: %s - $%.2f\n", product.Name, product.Price)
    }

    // Update product (will invalidate cache)
    product.Price = 999.99
    catalog.UpdateProduct(product)

    // Get product again (will be fetched from database and cached)
    product, err = catalog.GetProduct("p1")
    if err == nil {
        fmt.Printf("Updated product: %s - $%.2f\n", product.Name, product.Price)
    }
}
```

---

## 🔄 **3. Message Queue Patterns - Deep Dive with Examples**

### **Theory: What are Message Queues?**

Message queues enable asynchronous communication between services by storing messages in a queue until they can be processed.

**Benefits:**

- **Decoupling**: Services don't need to know about each other
- **Reliability**: Messages are persisted until processed
- **Scalability**: Can handle bursts of messages
- **Fault Tolerance**: Messages can be retried on failure

### **Real-World Example: Order Processing System**

```go
type OrderProcessingSystem struct {
    orderQueue    MessageQueue
    paymentQueue  MessageQueue
    inventoryQueue MessageQueue
    notificationQueue MessageQueue
    processors    map[string]*MessageProcessor
    mutex         sync.RWMutex
}

type MessageQueue interface {
    Publish(topic string, message interface{}) error
    Subscribe(topic string, handler MessageHandler) error
    Close() error
}

type MessageHandler func(message interface{}) error

type MessageProcessor struct {
    queue   MessageQueue
    topic   string
    handler MessageHandler
    workers int
    mutex   sync.RWMutex
}

type Order struct {
    ID        string
    UserID    string
    Items     []OrderItem
    Total     float64
    Status    string
    CreatedAt time.Time
}

type OrderItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

type Payment struct {
    OrderID   string
    Amount    float64
    Method    string
    Status    string
    CreatedAt time.Time
}

// In-Memory Message Queue Implementation
type InMemoryQueue struct {
    topics  map[string][]interface{}
    mutex   sync.RWMutex
    handlers map[string][]MessageHandler
}

func NewInMemoryQueue() *InMemoryQueue {
    return &InMemoryQueue{
        topics:   make(map[string][]interface{}),
        handlers: make(map[string][]MessageHandler),
    }
}

func (q *InMemoryQueue) Publish(topic string, message interface{}) error {
    q.mutex.Lock()
    defer q.mutex.Unlock()

    q.topics[topic] = append(q.topics[topic], message)

    // Notify handlers
    if handlers, exists := q.handlers[topic]; exists {
        for _, handler := range handlers {
            go func(h MessageHandler) {
                h(message)
            }(handler)
        }
    }

    return nil
}

func (q *InMemoryQueue) Subscribe(topic string, handler MessageHandler) error {
    q.mutex.Lock()
    defer q.mutex.Unlock()

    q.handlers[topic] = append(q.handlers[topic], handler)
    return nil
}

func (q *InMemoryQueue) Close() error {
    q.mutex.Lock()
    defer q.mutex.Unlock()

    q.topics = make(map[string][]interface{})
    q.handlers = make(map[string][]MessageHandler)
    return nil
}

// Order Processing System
func NewOrderProcessingSystem() *OrderProcessingSystem {
    return &OrderProcessingSystem{
        orderQueue:        NewInMemoryQueue(),
        paymentQueue:      NewInMemoryQueue(),
        inventoryQueue:    NewInMemoryQueue(),
        notificationQueue: NewInMemoryQueue(),
        processors:        make(map[string]*MessageProcessor),
    }
}

func (ops *OrderProcessingSystem) CreateOrder(order *Order) error {
    // Publish order to queue
    return ops.orderQueue.Publish("orders", order)
}

func (ops *OrderProcessingSystem) StartProcessing() {
    // Start order processor
    ops.startProcessor("orders", ops.processOrder, 3)

    // Start payment processor
    ops.startProcessor("payments", ops.processPayment, 2)

    // Start inventory processor
    ops.startProcessor("inventory", ops.processInventory, 2)

    // Start notification processor
    ops.startProcessor("notifications", ops.processNotification, 1)
}

func (ops *OrderProcessingSystem) startProcessor(topic string, handler MessageHandler, workers int) {
    processor := &MessageProcessor{
        queue:   ops.getQueueForTopic(topic),
        topic:   topic,
        handler: handler,
        workers: workers,
    }

    ops.mutex.Lock()
    ops.processors[topic] = processor
    ops.mutex.Unlock()

    // Start workers
    for i := 0; i < workers; i++ {
        go processor.startWorker()
    }
}

func (ops *OrderProcessingSystem) getQueueForTopic(topic string) MessageQueue {
    switch topic {
    case "orders":
        return ops.orderQueue
    case "payments":
        return ops.paymentQueue
    case "inventory":
        return ops.inventoryQueue
    case "notifications":
        return ops.notificationQueue
    default:
        return ops.orderQueue
    }
}

func (mp *MessageProcessor) startWorker() {
    // Subscribe to topic
    mp.queue.Subscribe(mp.topic, mp.handler)
}

func (ops *OrderProcessingSystem) processOrder(message interface{}) error {
    order, ok := message.(*Order)
    if !ok {
        return errors.New("invalid order message")
    }

    fmt.Printf("Processing order: %s\n", order.ID)

    // Update order status
    order.Status = "processing"

    // Check inventory
    for _, item := range order.Items {
        if !ops.checkInventory(item.ProductID, item.Quantity) {
            order.Status = "failed"
            return errors.New("insufficient inventory")
        }
    }

    // Reserve inventory
    for _, item := range order.Items {
        ops.reserveInventory(item.ProductID, item.Quantity)
    }

    // Create payment
    payment := &Payment{
        OrderID:   order.ID,
        Amount:    order.Total,
        Method:    "credit_card",
        Status:    "pending",
        CreatedAt: time.Now(),
    }

    // Publish payment to queue
    return ops.paymentQueue.Publish("payments", payment)
}

func (ops *OrderProcessingSystem) processPayment(message interface{}) error {
    payment, ok := message.(*Payment)
    if !ok {
        return errors.New("invalid payment message")
    }

    fmt.Printf("Processing payment for order: %s\n", payment.OrderID)

    // Simulate payment processing
    if rand.Float32() > 0.1 { // 90% success rate
        payment.Status = "completed"

        // Update order status
        order := &Order{ID: payment.OrderID, Status: "paid"}

        // Publish to inventory queue
        ops.inventoryQueue.Publish("inventory", order)

        // Publish to notification queue
        ops.notificationQueue.Publish("notifications", order)
    } else {
        payment.Status = "failed"
        return errors.New("payment failed")
    }

    return nil
}

func (ops *OrderProcessingSystem) processInventory(message interface{}) error {
    order, ok := message.(*Order)
    if !ok {
        return errors.New("invalid order message")
    }

    fmt.Printf("Updating inventory for order: %s\n", order.ID)

    // Simulate inventory update
    time.Sleep(100 * time.Millisecond)

    return nil
}

func (ops *OrderProcessingSystem) processNotification(message interface{}) error {
    order, ok := message.(*Order)
    if !ok {
        return errors.New("invalid order message")
    }

    fmt.Printf("Sending notification for order: %s\n", order.ID)

    // Simulate sending notification
    time.Sleep(50 * time.Millisecond)

    return nil
}

func (ops *OrderProcessingSystem) checkInventory(productID string, quantity int) bool {
    // Simulate inventory check
    return rand.Float32() > 0.2 // 80% chance of having inventory
}

func (ops *OrderProcessingSystem) reserveInventory(productID string, quantity int) {
    // Simulate inventory reservation
    fmt.Printf("Reserving %d units of product %s\n", quantity, productID)
}

// Example usage
func main() {
    // Create order processing system
    ops := NewOrderProcessingSystem()

    // Start processing
    ops.StartProcessing()

    // Create some orders
    orders := []*Order{
        {
            ID:   "order1",
            UserID: "user1",
            Items: []OrderItem{
                {ProductID: "p1", Quantity: 2, Price: 29.99},
                {ProductID: "p2", Quantity: 1, Price: 99.99},
            },
            Total:     159.97,
            Status:    "pending",
            CreatedAt: time.Now(),
        },
        {
            ID:   "order2",
            UserID: "user2",
            Items: []OrderItem{
                {ProductID: "p3", Quantity: 1, Price: 199.99},
            },
            Total:     199.99,
            Status:    "pending",
            CreatedAt: time.Now(),
        },
    }

    // Process orders
    for _, order := range orders {
        if err := ops.CreateOrder(order); err != nil {
            fmt.Printf("Failed to create order %s: %v\n", order.ID, err)
        }
    }

    // Wait for processing to complete
    time.Sleep(2 * time.Second)
}
```

---

## 🔄 **4. Microservices Patterns - Deep Dive with Examples**

### **Theory: What are Microservices?**

Microservices are an architectural approach where applications are built as a collection of loosely coupled services.

**Benefits:**

- **Independent Deployment**: Deploy services independently
- **Technology Diversity**: Use different technologies for different services
- **Fault Isolation**: Failure of one service doesn't affect others
- **Scalability**: Scale services independently

### **Real-World Example: E-commerce Microservices Architecture**

```go
type MicroservicesArchitecture struct {
    services map[string]Service
    registry *ServiceRegistry
    gateway  *APIGateway
    mutex    sync.RWMutex
}

type Service interface {
    Start() error
    Stop() error
    GetName() string
    GetPort() int
    GetHealth() bool
}

type ServiceRegistry struct {
    services map[string]*ServiceInfo
    mutex    sync.RWMutex
}

type ServiceInfo struct {
    Name     string
    Address  string
    Port     int
    Health   bool
    LastSeen time.Time
}

// User Service
type UserService struct {
    name    string
    port    int
    users   map[string]*User
    mutex   sync.RWMutex
    health  bool
}

type User struct {
    ID       string
    Username string
    Email    string
    Profile  map[string]interface{}
}

func NewUserService(port int) *UserService {
    return &UserService{
        name:   "user-service",
        port:   port,
        users:  make(map[string]*User),
        health: true,
    }
}

func (us *UserService) Start() error {
    fmt.Printf("Starting %s on port %d\n", us.name, us.port)
    // In real implementation, this would start HTTP server
    return nil
}

func (us *UserService) Stop() error {
    fmt.Printf("Stopping %s\n", us.name)
    us.health = false
    return nil
}

func (us *UserService) GetName() string {
    return us.name
}

func (us *UserService) GetPort() int {
    return us.port
}

func (us *UserService) GetHealth() bool {
    return us.health
}

func (us *UserService) CreateUser(user *User) error {
    us.mutex.Lock()
    defer us.mutex.Unlock()

    if _, exists := us.users[user.ID]; exists {
        return errors.New("user already exists")
    }

    us.users[user.ID] = user
    return nil
}

func (us *UserService) GetUser(userID string) (*User, error) {
    us.mutex.RLock()
    defer us.mutex.RUnlock()

    user, exists := us.users[userID]
    if !exists {
        return nil, errors.New("user not found")
    }

    return user, nil
}

// Product Service
type ProductService struct {
    name     string
    port     int
    products map[string]*Product
    mutex    sync.RWMutex
    health   bool
}

func NewProductService(port int) *ProductService {
    return &ProductService{
        name:     "product-service",
        port:     port,
        products: make(map[string]*Product),
        health:   true,
    }
}

func (ps *ProductService) Start() error {
    fmt.Printf("Starting %s on port %d\n", ps.name, ps.port)
    return nil
}

func (ps *ProductService) Stop() error {
    fmt.Printf("Stopping %s\n", ps.name)
    ps.health = false
    return nil
}

func (ps *ProductService) GetName() string {
    return ps.name
}

func (ps *ProductService) GetPort() int {
    return ps.port
}

func (ps *ProductService) GetHealth() bool {
    return ps.health
}

func (ps *ProductService) CreateProduct(product *Product) error {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()

    if _, exists := ps.products[product.ID]; exists {
        return errors.New("product already exists")
    }

    ps.products[product.ID] = product
    return nil
}

func (ps *ProductService) GetProduct(productID string) (*Product, error) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    product, exists := ps.products[productID]
    if !exists {
        return nil, errors.New("product not found")
    }

    return product, nil
}

func (ps *ProductService) GetProducts(category string) ([]*Product, error) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    var products []*Product
    for _, product := range ps.products {
        if product.Category == category {
            products = append(products, product)
        }
    }

    return products, nil
}

// Order Service
type OrderService struct {
    name     string
    port     int
    orders   map[string]*Order
    mutex    sync.RWMutex
    health   bool
}

func NewOrderService(port int) *OrderService {
    return &OrderService{
        name:   "order-service",
        port:   port,
        orders: make(map[string]*Order),
        health: true,
    }
}

func (os *OrderService) Start() error {
    fmt.Printf("Starting %s on port %d\n", os.name, os.port)
    return nil
}

func (os *OrderService) Stop() error {
    fmt.Printf("Stopping %s\n", os.name)
    os.health = false
    return nil
}

func (os *OrderService) GetName() string {
    return os.name
}

func (os *OrderService) GetPort() int {
    return os.port
}

func (os *OrderService) GetHealth() bool {
    return os.health
}

func (os *OrderService) CreateOrder(order *Order) error {
    os.mutex.Lock()
    defer os.mutex.Unlock()

    if _, exists := os.orders[order.ID]; exists {
        return errors.New("order already exists")
    }

    os.orders[order.ID] = order
    return nil
}

func (os *OrderService) GetOrder(orderID string) (*Order, error) {
    os.mutex.RLock()
    defer os.mutex.RUnlock()

    order, exists := os.orders[orderID]
    if !exists {
        return nil, errors.New("order not found")
    }

    return order, nil
}

// Service Registry
func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string]*ServiceInfo),
    }
}

func (sr *ServiceRegistry) Register(service Service) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()

    info := &ServiceInfo{
        Name:     service.GetName(),
        Address:  "localhost",
        Port:     service.GetPort(),
        Health:   service.GetHealth(),
        LastSeen: time.Now(),
    }

    sr.services[service.GetName()] = info
    return nil
}

func (sr *ServiceRegistry) GetService(name string) (*ServiceInfo, error) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()

    info, exists := sr.services[name]
    if !exists {
        return nil, errors.New("service not found")
    }

    return info, nil
}

func (sr *ServiceRegistry) GetHealthyServices() []*ServiceInfo {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()

    var healthy []*ServiceInfo
    for _, info := range sr.services {
        if info.Health {
            healthy = append(healthy, info)
        }
    }

    return healthy
}

// Microservices Architecture
func NewMicroservicesArchitecture() *MicroservicesArchitecture {
    return &MicroservicesArchitecture{
        services: make(map[string]Service),
        registry: NewServiceRegistry(),
        gateway:  NewAPIGateway(),
    }
}

func (ma *MicroservicesArchitecture) AddService(service Service) error {
    ma.mutex.Lock()
    defer ma.mutex.Unlock()

    ma.services[service.GetName()] = service

    // Register with service registry
    return ma.registry.Register(service)
}

func (ma *MicroservicesArchitecture) StartAllServices() error {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()

    for _, service := range ma.services {
        if err := service.Start(); err != nil {
            return err
        }
    }

    return nil
}

func (ma *MicroservicesArchitecture) StopAllServices() error {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()

    for _, service := range ma.services {
        if err := service.Stop(); err != nil {
            return err
        }
    }

    return nil
}

func (ma *MicroservicesArchitecture) GetService(name string) (Service, error) {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()

    service, exists := ma.services[name]
    if !exists {
        return nil, errors.New("service not found")
    }

    return service, nil
}

// Example usage
func main() {
    // Create microservices architecture
    ma := NewMicroservicesArchitecture()

    // Create services
    userService := NewUserService(8081)
    productService := NewProductService(8082)
    orderService := NewOrderService(8083)

    // Add services to architecture
    ma.AddService(userService)
    ma.AddService(productService)
    ma.AddService(orderService)

    // Start all services
    if err := ma.StartAllServices(); err != nil {
        fmt.Printf("Failed to start services: %v\n", err)
        return
    }

    // Create some test data
    user := &User{
        ID:       "user1",
        Username: "john_doe",
        Email:    "john@example.com",
        Profile:  map[string]interface{}{"age": 30, "city": "New York"},
    }

    product := &Product{
        ID:          "product1",
        Name:        "Laptop",
        Price:       999.99,
        Category:    "Electronics",
        Description: "High-performance laptop",
        Stock:       10,
    }

    order := &Order{
        ID:     "order1",
        UserID: "user1",
        Items: []OrderItem{
            {ProductID: "product1", Quantity: 1, Price: 999.99},
        },
        Total:     999.99,
        Status:    "pending",
        CreatedAt: time.Now(),
    }

    // Use services
    userService.CreateUser(user)
    productService.CreateProduct(product)
    orderService.CreateOrder(order)

    // Get service info
    if info, err := ma.registry.GetService("user-service"); err == nil {
        fmt.Printf("Service: %s, Port: %d, Health: %t\n", info.Name, info.Port, info.Health)
    }

    // Stop all services
    ma.StopAllServices()
}
```

---

## 🎯 **Key Takeaways**

### **1. Load Balancing Patterns**

- **Round Robin**: Simple, even distribution
- **Least Connections**: Good for long-running connections
- **Weighted**: Consider server capacity
- **Health Checking**: Essential for reliability

### **2. Caching Patterns**

- **Cache-Aside**: Application manages cache
- **Write-Through**: Write to cache and database
- **Write-Behind**: Write to cache, async to database
- **Eviction Policies**: LRU, LFU, FIFO

### **3. Message Queue Patterns**

- **Asynchronous Processing**: Decouple services
- **Reliability**: Persist messages until processed
- **Scalability**: Handle message bursts
- **Fault Tolerance**: Retry failed messages

### **4. Microservices Patterns**

- **Service Registry**: Discover services
- **API Gateway**: Single entry point
- **Independent Deployment**: Deploy services separately
- **Fault Isolation**: Isolate failures

---

**🎉 This comprehensive guide provides deep understanding of system design patterns with practical examples and implementations! 🚀**
