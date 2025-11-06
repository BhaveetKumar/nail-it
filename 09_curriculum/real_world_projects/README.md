---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.476253
Tags: []
Status: draft
---

# Real-World Projects & Case Studies

## Table of Contents

1. [Overview](#overview)
2. [E-commerce Platform](#e-commerce-platform)
3. [Social Media Application](#social-media-application)
4. [Payment Processing System](#payment-processing-system)
5. [Content Management System](#content-management-system)
6. [Data Analytics Platform](#data-analytics-platform)
7. [Microservices Architecture](#microservices-architecture)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)

## Overview

### Learning Objectives

- Apply theoretical knowledge to real-world problems
- Build complete, production-ready applications
- Understand system design and architecture patterns
- Develop problem-solving and debugging skills
- Create portfolio-worthy projects

### What are Real-World Projects?

Real-world projects are comprehensive, production-ready applications that demonstrate practical engineering skills and solve actual business problems.

## E-commerce Platform

### 1. Project Overview

#### E-commerce System Architecture
```go
package main

import (
    "fmt"
    "time"
)

type ECommercePlatform struct {
    ProductService    *ProductService
    UserService       *UserService
    OrderService      *OrderService
    PaymentService    *PaymentService
    InventoryService  *InventoryService
    NotificationService *NotificationService
}

type ProductService struct {
    products map[string]*Product
    categories map[string][]string
}

type Product struct {
    ID          string
    Name        string
    Description string
    Price       float64
    Category    string
    SKU         string
    Stock       int
    Images      []string
    Attributes  map[string]string
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

type UserService struct {
    users map[string]*User
    sessions map[string]*Session
}

type User struct {
    ID        string
    Email     string
    Name      string
    Address   *Address
    Phone     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

type Address struct {
    Street    string
    City      string
    State     string
    ZipCode   string
    Country   string
}

type Session struct {
    UserID    string
    Token     string
    ExpiresAt time.Time
}

type OrderService struct {
    orders map[string]*Order
    cartService *CartService
}

type Order struct {
    ID           string
    UserID       string
    Items        []*OrderItem
    Total        float64
    Status       string
    ShippingAddress *Address
    BillingAddress  *Address
    CreatedAt    time.Time
    UpdatedAt    time.Time
}

type OrderItem struct {
    ProductID string
    Quantity  int
    Price     float64
    Total     float64
}

type CartService struct {
    carts map[string]*Cart
}

type Cart struct {
    UserID string
    Items  []*CartItem
    Total  float64
}

type CartItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

type PaymentService struct {
    payments map[string]*Payment
    gateways map[string]PaymentGateway
}

type Payment struct {
    ID        string
    OrderID   string
    Amount    float64
    Method    string
    Status    string
    Gateway   string
    CreatedAt time.Time
}

type PaymentGateway interface {
    ProcessPayment(amount float64, method string) (*PaymentResult, error)
    RefundPayment(paymentID string, amount float64) error
}

type PaymentResult struct {
    TransactionID string
    Status        string
    Message       string
}

type InventoryService struct {
    inventory map[string]int
    reservations map[string]int
}

type NotificationService struct {
    emailService *EmailService
    smsService   *SMSService
}

type EmailService struct {
    smtpHost string
    smtpPort int
    username string
    password string
}

type SMSService struct {
    apiKey    string
    apiSecret string
}

func NewECommercePlatform() *ECommercePlatform {
    return &ECommercePlatform{
        ProductService:    NewProductService(),
        UserService:       NewUserService(),
        OrderService:      NewOrderService(),
        PaymentService:    NewPaymentService(),
        InventoryService:  NewInventoryService(),
        NotificationService: NewNotificationService(),
    }
}

func NewProductService() *ProductService {
    return &ProductService{
        products:   make(map[string]*Product),
        categories: make(map[string][]string),
    }
}

func (ps *ProductService) AddProduct(product *Product) {
    ps.products[product.ID] = product
    ps.categories[product.Category] = append(ps.categories[product.Category], product.ID)
}

func (ps *ProductService) GetProduct(id string) *Product {
    return ps.products[id]
}

func (ps *ProductService) SearchProducts(query string, category string) []*Product {
    var results []*Product
    
    for _, product := range ps.products {
        if category != "" && product.Category != category {
            continue
        }
        
        if query == "" || 
           contains(product.Name, query) || 
           contains(product.Description, query) {
            results = append(results, product)
        }
    }
    
    return results
}

func NewUserService() *UserService {
    return &UserService{
        users:    make(map[string]*User),
        sessions: make(map[string]*Session),
    }
}

func (us *UserService) RegisterUser(user *User) error {
    if _, exists := us.users[user.Email]; exists {
        return fmt.Errorf("user already exists")
    }
    
    us.users[user.Email] = user
    return nil
}

func (us *UserService) LoginUser(email, password string) (*Session, error) {
    user, exists := us.users[email]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    
    // In real implementation, verify password hash
    session := &Session{
        UserID:    user.ID,
        Token:     generateToken(),
        ExpiresAt: time.Now().Add(24 * time.Hour),
    }
    
    us.sessions[session.Token] = session
    return session, nil
}

func NewOrderService() *OrderService {
    return &OrderService{
        orders:      make(map[string]*Order),
        cartService: NewCartService(),
    }
}

func (os *OrderService) CreateOrder(userID string, cart *Cart) *Order {
    order := &Order{
        ID:        generateOrderID(),
        UserID:    userID,
        Items:     make([]*OrderItem, 0),
        Total:     0,
        Status:    "pending",
        CreatedAt: time.Now(),
    }
    
    for _, cartItem := range cart.Items {
        orderItem := &OrderItem{
            ProductID: cartItem.ProductID,
            Quantity:  cartItem.Quantity,
            Price:     cartItem.Price,
            Total:     cartItem.Price * float64(cartItem.Quantity),
        }
        order.Items = append(order.Items, orderItem)
        order.Total += orderItem.Total
    }
    
    os.orders[order.ID] = order
    return order
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        payments: make(map[string]*Payment),
        gateways: make(map[string]PaymentGateway),
    }
}

func (ps *PaymentService) ProcessPayment(orderID string, amount float64, method string) (*Payment, error) {
    payment := &Payment{
        ID:        generatePaymentID(),
        OrderID:   orderID,
        Amount:    amount,
        Method:    method,
        Status:    "processing",
        Gateway:   "stripe",
        CreatedAt: time.Now(),
    }
    
    // Process payment through gateway
    gateway := ps.gateways[payment.Gateway]
    if gateway == nil {
        return nil, fmt.Errorf("payment gateway not found")
    }
    
    result, err := gateway.ProcessPayment(amount, method)
    if err != nil {
        payment.Status = "failed"
        return payment, err
    }
    
    payment.Status = result.Status
    ps.payments[payment.ID] = payment
    
    return payment, nil
}

func NewInventoryService() *InventoryService {
    return &InventoryService{
        inventory:    make(map[string]int),
        reservations: make(map[string]int),
    }
}

func (is *InventoryService) ReserveStock(productID string, quantity int) bool {
    available := is.inventory[productID] - is.reservations[productID]
    if available >= quantity {
        is.reservations[productID] += quantity
        return true
    }
    return false
}

func (is *InventoryService) ReleaseStock(productID string, quantity int) {
    if is.reservations[productID] >= quantity {
        is.reservations[productID] -= quantity
    }
}

func NewCartService() *CartService {
    return &CartService{
        carts: make(map[string]*Cart),
    }
}

func (cs *CartService) AddToCart(userID, productID string, quantity int, price float64) {
    cart, exists := cs.carts[userID]
    if !exists {
        cart = &Cart{
            UserID: userID,
            Items:  make([]*CartItem, 0),
            Total:  0,
        }
        cs.carts[userID] = cart
    }
    
    // Check if item already in cart
    for _, item := range cart.Items {
        if item.ProductID == productID {
            item.Quantity += quantity
            item.Price = price
            cart.Total += price * float64(quantity)
            return
        }
    }
    
    // Add new item
    cartItem := &CartItem{
        ProductID: productID,
        Quantity:  quantity,
        Price:     price,
    }
    cart.Items = append(cart.Items, cartItem)
    cart.Total += price * float64(quantity)
}

func NewNotificationService() *NotificationService {
    return &NotificationService{
        emailService: NewEmailService(),
        smsService:   NewSMSService(),
    }
}

func NewEmailService() *EmailService {
    return &EmailService{
        smtpHost: "smtp.gmail.com",
        smtpPort: 587,
        username: "noreply@ecommerce.com",
        password: "password",
    }
}

func NewSMSService() *SMSService {
    return &SMSService{
        apiKey:    "api_key",
        apiSecret: "api_secret",
    }
}

// Helper functions
func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}

func generateToken() string {
    return fmt.Sprintf("token_%d", time.Now().UnixNano())
}

func generateOrderID() string {
    return fmt.Sprintf("order_%d", time.Now().UnixNano())
}

func generatePaymentID() string {
    return fmt.Sprintf("payment_%d", time.Now().UnixNano())
}

func main() {
    platform := NewECommercePlatform()
    
    // Add sample products
    product1 := &Product{
        ID:          "prod_001",
        Name:        "Laptop",
        Description: "High-performance laptop",
        Price:       999.99,
        Category:    "Electronics",
        SKU:         "LAPTOP001",
        Stock:       10,
        Images:      []string{"laptop1.jpg", "laptop2.jpg"},
        Attributes:  map[string]string{"brand": "Dell", "model": "XPS 13"},
        CreatedAt:   time.Now(),
    }
    platform.ProductService.AddProduct(product1)
    
    product2 := &Product{
        ID:          "prod_002",
        Name:        "Mouse",
        Description: "Wireless mouse",
        Price:       29.99,
        Category:    "Electronics",
        SKU:         "MOUSE001",
        Stock:       50,
        Images:      []string{"mouse1.jpg"},
        Attributes:  map[string]string{"brand": "Logitech", "model": "MX Master 3"},
        CreatedAt:   time.Now(),
    }
    platform.ProductService.AddProduct(product2)
    
    // Register user
    user := &User{
        ID:    "user_001",
        Email: "test@example.com",
        Name:  "John Doe",
        Address: &Address{
            Street:  "123 Main St",
            City:    "New York",
            State:   "NY",
            ZipCode: "10001",
            Country: "USA",
        },
        Phone:     "555-1234",
        CreatedAt: time.Now(),
    }
    platform.UserService.RegisterUser(user)
    
    // Add items to cart
    platform.OrderService.cartService.AddToCart("user_001", "prod_001", 1, 999.99)
    platform.OrderService.cartService.AddToCart("user_001", "prod_002", 2, 29.99)
    
    // Create order
    cart := platform.OrderService.cartService.carts["user_001"]
    order := platform.OrderService.CreateOrder("user_001", cart)
    
    fmt.Printf("E-commerce Platform Demo:\n")
    fmt.Printf("========================\n")
    fmt.Printf("Order Created: %s\n", order.ID)
    fmt.Printf("Total: $%.2f\n", order.Total)
    fmt.Printf("Items: %d\n", len(order.Items))
    
    for _, item := range order.Items {
        fmt.Printf("- Product %s: %d x $%.2f = $%.2f\n", 
            item.ProductID, item.Quantity, item.Price, item.Total)
    }
}
```

### 2. Database Schema

#### Database Design
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Addresses table
CREATE TABLE addresses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    street VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    zip_code VARCHAR(20) NOT NULL,
    country VARCHAR(100) NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories table
CREATE TABLE categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category_id UUID REFERENCES categories(id),
    sku VARCHAR(100) UNIQUE NOT NULL,
    stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Product images table
CREATE TABLE product_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    image_url VARCHAR(500) NOT NULL,
    alt_text VARCHAR(255),
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Product attributes table
CREATE TABLE product_attributes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    attribute_name VARCHAR(100) NOT NULL,
    attribute_value VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    status VARCHAR(50) DEFAULT 'pending',
    total DECIMAL(10,2) NOT NULL,
    shipping_address_id UUID REFERENCES addresses(id),
    billing_address_id UUID REFERENCES addresses(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order items table
CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payments table
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id),
    amount DECIMAL(10,2) NOT NULL,
    method VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    gateway VARCHAR(50) NOT NULL,
    transaction_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cart table
CREATE TABLE cart (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_id)
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_payments_order_id ON payments(order_id);
CREATE INDEX idx_cart_user_id ON cart(user_id);
```

## Social Media Application

### 1. Project Overview

#### Social Media Platform Architecture
```go
package main

import (
    "fmt"
    "time"
)

type SocialMediaPlatform struct {
    UserService      *UserService
    PostService      *PostService
    CommentService   *CommentService
    LikeService      *LikeService
    FollowService    *FollowService
    FeedService      *FeedService
    NotificationService *NotificationService
}

type UserService struct {
    users map[string]*User
    profiles map[string]*UserProfile
}

type User struct {
    ID        string
    Username  string
    Email     string
    Name      string
    Bio       string
    Avatar    string
    CreatedAt time.Time
    UpdatedAt time.Time
}

type UserProfile struct {
    UserID      string
    Followers   int
    Following   int
    Posts       int
    IsPrivate   bool
    IsVerified  bool
}

type PostService struct {
    posts map[string]*Post
}

type Post struct {
    ID        string
    UserID    string
    Content   string
    Images    []string
    Likes     int
    Comments  int
    Shares    int
    CreatedAt time.Time
    UpdatedAt time.Time
}

type CommentService struct {
    comments map[string]*Comment
}

type Comment struct {
    ID        string
    PostID    string
    UserID    string
    Content   string
    Likes     int
    CreatedAt time.Time
    UpdatedAt time.Time
}

type LikeService struct {
    likes map[string]*Like
}

type Like struct {
    ID        string
    UserID    string
    PostID    string
    CommentID string
    CreatedAt time.Time
}

type FollowService struct {
    follows map[string]*Follow
}

type Follow struct {
    ID          string
    FollowerID  string
    FollowingID string
    CreatedAt   time.Time
}

type FeedService struct {
    feeds map[string]*Feed
}

type Feed struct {
    UserID string
    Posts  []*Post
    Cursor string
}

type NotificationService struct {
    notifications map[string]*Notification
}

type Notification struct {
    ID        string
    UserID    string
    Type      string
    Message   string
    Data      map[string]interface{}
    Read      bool
    CreatedAt time.Time
}

func NewSocialMediaPlatform() *SocialMediaPlatform {
    return &SocialMediaPlatform{
        UserService:      NewUserService(),
        PostService:      NewPostService(),
        CommentService:   NewCommentService(),
        LikeService:      NewLikeService(),
        FollowService:    NewFollowService(),
        FeedService:      NewFeedService(),
        NotificationService: NewNotificationService(),
    }
}

func NewUserService() *UserService {
    return &UserService{
        users:    make(map[string]*User),
        profiles: make(map[string]*UserProfile),
    }
}

func (us *UserService) CreateUser(user *User) error {
    if _, exists := us.users[user.Username]; exists {
        return fmt.Errorf("username already exists")
    }
    
    us.users[user.Username] = user
    us.profiles[user.ID] = &UserProfile{
        UserID:     user.ID,
        Followers:  0,
        Following:  0,
        Posts:      0,
        IsPrivate:  false,
        IsVerified: false,
    }
    
    return nil
}

func (us *UserService) GetUser(username string) *User {
    return us.users[username]
}

func (us *UserService) GetUserProfile(userID string) *UserProfile {
    return us.profiles[userID]
}

func NewPostService() *PostService {
    return &PostService{
        posts: make(map[string]*Post),
    }
}

func (ps *PostService) CreatePost(post *Post) {
    ps.posts[post.ID] = post
}

func (ps *PostService) GetPost(id string) *Post {
    return ps.posts[id]
}

func (ps *PostService) GetUserPosts(userID string) []*Post {
    var posts []*Post
    
    for _, post := range ps.posts {
        if post.UserID == userID {
            posts = append(posts, post)
        }
    }
    
    return posts
}

func NewCommentService() *CommentService {
    return &CommentService{
        comments: make(map[string]*Comment),
    }
}

func (cs *CommentService) CreateComment(comment *Comment) {
    cs.comments[comment.ID] = comment
}

func (cs *CommentService) GetPostComments(postID string) []*Comment {
    var comments []*Comment
    
    for _, comment := range cs.comments {
        if comment.PostID == postID {
            comments = append(comments, comment)
        }
    }
    
    return comments
}

func NewLikeService() *LikeService {
    return &LikeService{
        likes: make(map[string]*Like),
    }
}

func (ls *LikeService) LikePost(userID, postID string) {
    like := &Like{
        ID:        generateLikeID(),
        UserID:    userID,
        PostID:    postID,
        CreatedAt: time.Now(),
    }
    
    ls.likes[like.ID] = like
}

func (ls *LikeService) UnlikePost(userID, postID string) {
    for id, like := range ls.likes {
        if like.UserID == userID && like.PostID == postID {
            delete(ls.likes, id)
            break
        }
    }
}

func NewFollowService() *FollowService {
    return &FollowService{
        follows: make(map[string]*Follow),
    }
}

func (fs *FollowService) FollowUser(followerID, followingID string) {
    follow := &Follow{
        ID:          generateFollowID(),
        FollowerID:  followerID,
        FollowingID: followingID,
        CreatedAt:   time.Now(),
    }
    
    fs.follows[follow.ID] = follow
}

func (fs *FollowService) UnfollowUser(followerID, followingID string) {
    for id, follow := range fs.follows {
        if follow.FollowerID == followerID && follow.FollowingID == followingID {
            delete(fs.follows, id)
            break
        }
    }
}

func NewFeedService() *FeedService {
    return &FeedService{
        feeds: make(map[string]*Feed),
    }
}

func (fs *FeedService) GetUserFeed(userID string) *Feed {
    feed, exists := fs.feeds[userID]
    if !exists {
        feed = &Feed{
            UserID: userID,
            Posts:  make([]*Post, 0),
            Cursor: "",
        }
        fs.feeds[userID] = feed
    }
    
    return feed
}

func NewNotificationService() *NotificationService {
    return &NotificationService{
        notifications: make(map[string]*Notification),
    }
}

func (ns *NotificationService) CreateNotification(notification *Notification) {
    ns.notifications[notification.ID] = notification
}

func (ns *NotificationService) GetUserNotifications(userID string) []*Notification {
    var notifications []*Notification
    
    for _, notification := range ns.notifications {
        if notification.UserID == userID {
            notifications = append(notifications, notification)
        }
    }
    
    return notifications
}

// Helper functions
func generateLikeID() string {
    return fmt.Sprintf("like_%d", time.Now().UnixNano())
}

func generateFollowID() string {
    return fmt.Sprintf("follow_%d", time.Now().UnixNano())
}

func main() {
    platform := NewSocialMediaPlatform()
    
    // Create users
    user1 := &User{
        ID:        "user_001",
        Username:  "john_doe",
        Email:     "john@example.com",
        Name:      "John Doe",
        Bio:       "Software engineer and tech enthusiast",
        Avatar:    "avatar1.jpg",
        CreatedAt: time.Now(),
    }
    platform.UserService.CreateUser(user1)
    
    user2 := &User{
        ID:        "user_002",
        Username:  "jane_smith",
        Email:     "jane@example.com",
        Name:      "Jane Smith",
        Bio:       "Designer and artist",
        Avatar:    "avatar2.jpg",
        CreatedAt: time.Now(),
    }
    platform.UserService.CreateUser(user2)
    
    // Create posts
    post1 := &Post{
        ID:        "post_001",
        UserID:    "user_001",
        Content:   "Just finished building an amazing e-commerce platform! #coding #golang",
        Images:    []string{"screenshot1.jpg"},
        Likes:     0,
        Comments:  0,
        Shares:    0,
        CreatedAt: time.Now(),
    }
    platform.PostService.CreatePost(post1)
    
    post2 := &Post{
        ID:        "post_002",
        UserID:    "user_002",
        Content:   "Beautiful sunset today! 🌅",
        Images:    []string{"sunset1.jpg", "sunset2.jpg"},
        Likes:     0,
        Comments:  0,
        Shares:    0,
        CreatedAt: time.Now(),
    }
    platform.PostService.CreatePost(post2)
    
    // Follow users
    platform.FollowService.FollowUser("user_001", "user_002")
    platform.FollowService.FollowUser("user_002", "user_001")
    
    // Like posts
    platform.LikeService.LikePost("user_002", "post_001")
    platform.LikeService.LikePost("user_001", "post_002")
    
    // Create comments
    comment1 := &Comment{
        ID:        "comment_001",
        PostID:    "post_001",
        UserID:    "user_002",
        Content:   "Great work! What technologies did you use?",
        Likes:     0,
        CreatedAt: time.Now(),
    }
    platform.CommentService.CreateComment(comment1)
    
    // Create notifications
    notification1 := &Notification{
        ID:        "notif_001",
        UserID:    "user_001",
        Type:      "like",
        Message:   "jane_smith liked your post",
        Data:      map[string]interface{}{"post_id": "post_001"},
        Read:      false,
        CreatedAt: time.Now(),
    }
    platform.NotificationService.CreateNotification(notification1)
    
    fmt.Printf("Social Media Platform Demo:\n")
    fmt.Printf("==========================\n")
    fmt.Printf("Users: %d\n", len(platform.UserService.users))
    fmt.Printf("Posts: %d\n", len(platform.PostService.posts))
    fmt.Printf("Comments: %d\n", len(platform.CommentService.comments))
    fmt.Printf("Likes: %d\n", len(platform.LikeService.likes))
    fmt.Printf("Follows: %d\n", len(platform.FollowService.follows))
    fmt.Printf("Notifications: %d\n", len(platform.NotificationService.notifications))
}
```

## Payment Processing System

### 1. Project Overview

#### Payment System Architecture
```go
package main

import (
    "fmt"
    "time"
)

type PaymentSystem struct {
    PaymentService    *PaymentService
    TransactionService *TransactionService
    RefundService     *RefundService
    WebhookService    *WebhookService
    FraudDetectionService *FraudDetectionService
}

type PaymentService struct {
    payments map[string]*Payment
    gateways map[string]PaymentGateway
}

type Payment struct {
    ID            string
    OrderID       string
    Amount        float64
    Currency      string
    Method        string
    Status        string
    Gateway       string
    TransactionID string
    CreatedAt     time.Time
    UpdatedAt     time.Time
}

type PaymentGateway interface {
    ProcessPayment(amount float64, currency string, method string) (*PaymentResult, error)
    RefundPayment(transactionID string, amount float64) (*RefundResult, error)
    GetPaymentStatus(transactionID string) (*PaymentStatus, error)
}

type PaymentResult struct {
    TransactionID string
    Status        string
    Message       string
    GatewayData   map[string]interface{}
}

type RefundResult struct {
    RefundID      string
    Status        string
    Message       string
    Amount        float64
}

type PaymentStatus struct {
    TransactionID string
    Status        string
    Amount        float64
    Currency      string
    CreatedAt     time.Time
}

type TransactionService struct {
    transactions map[string]*Transaction
}

type Transaction struct {
    ID            string
    PaymentID     string
    Type          string
    Amount        float64
    Currency      string
    Status        string
    Gateway       string
    GatewayData   map[string]interface{}
    CreatedAt     time.Time
}

type RefundService struct {
    refunds map[string]*Refund
}

type Refund struct {
    ID            string
    PaymentID     string
    Amount        float64
    Reason        string
    Status        string
    Gateway       string
    TransactionID string
    CreatedAt     time.Time
}

type WebhookService struct {
    webhooks map[string]*Webhook
}

type Webhook struct {
    ID        string
    URL       string
    Events    []string
    Secret    string
    Active    bool
    CreatedAt time.Time
}

type FraudDetectionService struct {
    rules map[string]*FraudRule
}

type FraudRule struct {
    ID          string
    Name        string
    Description string
    Conditions  []string
    Action      string
    Enabled     bool
}

func NewPaymentSystem() *PaymentSystem {
    return &PaymentSystem{
        PaymentService:    NewPaymentService(),
        TransactionService: NewTransactionService(),
        RefundService:     NewRefundService(),
        WebhookService:    NewWebhookService(),
        FraudDetectionService: NewFraudDetectionService(),
    }
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        payments: make(map[string]*Payment),
        gateways: make(map[string]PaymentGateway),
    }
}

func (ps *PaymentService) AddGateway(name string, gateway PaymentGateway) {
    ps.gateways[name] = gateway
}

func (ps *PaymentService) ProcessPayment(orderID string, amount float64, currency string, method string, gateway string) (*Payment, error) {
    gatewayImpl, exists := ps.gateways[gateway]
    if !exists {
        return nil, fmt.Errorf("payment gateway not found")
    }
    
    result, err := gatewayImpl.ProcessPayment(amount, currency, method)
    if err != nil {
        return nil, err
    }
    
    payment := &Payment{
        ID:            generatePaymentID(),
        OrderID:       orderID,
        Amount:        amount,
        Currency:      currency,
        Method:        method,
        Status:        result.Status,
        Gateway:       gateway,
        TransactionID: result.TransactionID,
        CreatedAt:     time.Now(),
        UpdatedAt:     time.Now(),
    }
    
    ps.payments[payment.ID] = payment
    return payment, nil
}

func (ps *PaymentService) GetPayment(id string) *Payment {
    return ps.payments[id]
}

func NewTransactionService() *TransactionService {
    return &TransactionService{
        transactions: make(map[string]*Transaction),
    }
}

func (ts *TransactionService) CreateTransaction(transaction *Transaction) {
    ts.transactions[transaction.ID] = transaction
}

func (ts *TransactionService) GetTransaction(id string) *Transaction {
    return ts.transactions[transaction.ID]
}

func NewRefundService() *RefundService {
    return &RefundService{
        refunds: make(map[string]*Refund),
    }
}

func (rs *RefundService) CreateRefund(paymentID string, amount float64, reason string) (*Refund, error) {
    refund := &Refund{
        ID:        generateRefundID(),
        PaymentID: paymentID,
        Amount:    amount,
        Reason:    reason,
        Status:    "pending",
        CreatedAt: time.Now(),
    }
    
    rs.refunds[refund.ID] = refund
    return refund, nil
}

func NewWebhookService() *WebhookService {
    return &WebhookService{
        webhooks: make(map[string]*Webhook),
    }
}

func (ws *WebhookService) CreateWebhook(url string, events []string, secret string) *Webhook {
    webhook := &Webhook{
        ID:        generateWebhookID(),
        URL:       url,
        Events:    events,
        Secret:    secret,
        Active:    true,
        CreatedAt: time.Now(),
    }
    
    ws.webhooks[webhook.ID] = webhook
    return webhook
}

func NewFraudDetectionService() *FraudDetectionService {
    return &FraudDetectionService{
        rules: make(map[string]*FraudRule),
    }
}

func (fds *FraudDetectionService) AddRule(rule *FraudRule) {
    fds.rules[rule.ID] = rule
}

func (fds *FraudDetectionService) CheckFraud(payment *Payment) bool {
    for _, rule := range fds.rules {
        if !rule.Enabled {
            continue
        }
        
        // Simple fraud detection logic
        if payment.Amount > 10000 { // High amount
            return true
        }
        
        if payment.Method == "card" && payment.Amount > 5000 { // High card amount
            return true
        }
    }
    
    return false
}

// Helper functions
func generatePaymentID() string {
    return fmt.Sprintf("pay_%d", time.Now().UnixNano())
}

func generateRefundID() string {
    return fmt.Sprintf("refund_%d", time.Now().UnixNano())
}

func generateWebhookID() string {
    return fmt.Sprintf("webhook_%d", time.Now().UnixNano())
}

func main() {
    system := NewPaymentSystem()
    
    // Add payment gateway
    stripeGateway := &StripeGateway{}
    system.PaymentService.AddGateway("stripe", stripeGateway)
    
    // Process payment
    payment, err := system.PaymentService.ProcessPayment(
        "order_123",
        99.99,
        "USD",
        "card",
        "stripe",
    )
    
    if err != nil {
        fmt.Printf("Payment failed: %v\n", err)
        return
    }
    
    fmt.Printf("Payment System Demo:\n")
    fmt.Printf("===================\n")
    fmt.Printf("Payment ID: %s\n", payment.ID)
    fmt.Printf("Amount: $%.2f %s\n", payment.Amount, payment.Currency)
    fmt.Printf("Status: %s\n", payment.Status)
    fmt.Printf("Gateway: %s\n", payment.Gateway)
    fmt.Printf("Transaction ID: %s\n", payment.TransactionID)
    
    // Check fraud
    isFraud := system.FraudDetectionService.CheckFraud(payment)
    if isFraud {
        fmt.Println("⚠️  Fraud detected!")
    } else {
        fmt.Println("✅ Payment approved")
    }
}

// Mock Stripe Gateway
type StripeGateway struct{}

func (sg *StripeGateway) ProcessPayment(amount float64, currency string, method string) (*PaymentResult, error) {
    return &PaymentResult{
        TransactionID: fmt.Sprintf("stripe_%d", time.Now().UnixNano()),
        Status:        "succeeded",
        Message:       "Payment processed successfully",
        GatewayData:   map[string]interface{}{"stripe_id": "pi_1234567890"},
    }, nil
}

func (sg *StripeGateway) RefundPayment(transactionID string, amount float64) (*RefundResult, error) {
    return &RefundResult{
        RefundID: fmt.Sprintf("re_%d", time.Now().UnixNano()),
        Status:   "succeeded",
        Message:  "Refund processed successfully",
        Amount:   amount,
    }, nil
}

func (sg *StripeGateway) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    return &PaymentStatus{
        TransactionID: transactionID,
        Status:        "succeeded",
        Amount:        99.99,
        Currency:      "USD",
        CreatedAt:     time.Now(),
    }, nil
}
```

## Follow-up Questions

### 1. Project Selection
**Q: How do you choose the right project for your portfolio?**
A: Select projects that demonstrate relevant skills, solve real problems, show technical depth, and align with your career goals.

### 2. Architecture Design
**Q: What principles guide your system architecture decisions?**
A: Scalability, maintainability, security, performance, and user experience should drive architectural choices.

### 3. Implementation Strategy
**Q: How do you approach building complex systems?**
A: Start with MVP, iterate incrementally, use proven patterns, implement proper testing, and focus on user value.

## Sources

### Project Ideas
- **GitHub**: [Open Source Projects](https://github.com/)
- **Devpost**: [Hackathon Projects](https://devpost.com/)
- **Kaggle**: [Data Science Projects](https://www.kaggle.com/)

### Architecture Resources
- **AWS Architecture Center**: [Best Practices](https://aws.amazon.com/architecture/)
- **Google Cloud Architecture**: [Design Patterns](https://cloud.google.com/architecture/)
- **Microsoft Azure Architecture**: [Reference Architectures](https://docs.microsoft.com/en-us/azure/architecture/)

### Development Tools
- **Docker**: [Containerization](https://www.docker.com/)
- **Kubernetes**: [Container Orchestration](https://kubernetes.io/)
- **Terraform**: [Infrastructure as Code](https://www.terraform.io/)

---

**Next**: [Assessment Tools](../../README.md) | **Previous**: [Practice Exercises](../../README.md) | **Up**: [Real-World Projects](README.md)


## Content Management System

<!-- AUTO-GENERATED ANCHOR: originally referenced as #content-management-system -->

Placeholder content. Please replace with proper section.


## Data Analytics Platform

<!-- AUTO-GENERATED ANCHOR: originally referenced as #data-analytics-platform -->

Placeholder content. Please replace with proper section.


## Microservices Architecture

<!-- AUTO-GENERATED ANCHOR: originally referenced as #microservices-architecture -->

Placeholder content. Please replace with proper section.
