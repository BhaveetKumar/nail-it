# Practice Exercises & Coding Challenges

## Table of Contents

1. [Overview](#overview/)
2. [Algorithm Practice](#algorithm-practice/)
3. [System Design Practice](#system-design-practice/)
4. [Database Practice](#database-practice/)
5. [Distributed Systems Practice](#distributed-systems-practice/)
6. [Real-World Projects](#real-world-projects/)
7. [Assessment Tools](#assessment-tools/)
8. [Follow-up Questions](#follow-up-questions/)
9. [Sources](#sources/)

## Overview

### Learning Objectives

- Master problem-solving techniques through practice
- Build confidence with coding challenges
- Develop system design skills through projects
- Prepare for technical interviews
- Apply theoretical knowledge to real problems

### What are Practice Exercises?

Practice exercises are hands-on coding challenges, system design problems, and real-world projects that help reinforce learning and build practical skills.

## Algorithm Practice

### 1. Data Structures & Algorithms

#### Array Problems
```go
package main

import "fmt"

// Problem 1: Two Sum
func twoSum(nums []int, target int) []int {
    numMap := make(map[int]int)
    
    for i, num := range nums {
        complement := target - num
        if index, exists := numMap[complement]; exists {
            return []int{index, i}
        }
        numMap[num] = i
    }
    
    return []int{}
}

// Problem 2: Maximum Subarray (Kadane's Algorithm)
func maxSubArray(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxSoFar := nums[0]
    maxEndingHere := nums[0]
    
    for i := 1; i < len(nums); i++ {
        maxEndingHere = max(nums[i], maxEndingHere+nums[i])
        maxSoFar = max(maxSoFar, maxEndingHere)
    }
    
    return maxSoFar
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// Problem 3: Container With Most Water
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0
    
    for left < right {
        width := right - left
        currentHeight := min(height[left], height[right])
        area := width * currentHeight
        
        if area > maxArea {
            maxArea = area
        }
        
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxArea
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    // Test Two Sum
    nums1 := []int{2, 7, 11, 15}
    target1 := 9
    result1 := twoSum(nums1, target1)
    fmt.Printf("Two Sum: %v\n", result1)
    
    // Test Maximum Subarray
    nums2 := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
    result2 := maxSubArray(nums2)
    fmt.Printf("Maximum Subarray: %d\n", result2)
    
    // Test Container With Most Water
    height := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
    result3 := maxArea(height)
    fmt.Printf("Container With Most Water: %d\n", result3)
}
```

#### Tree Problems
```go
package main

import "fmt"

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Problem 1: Maximum Depth of Binary Tree
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    return max(leftDepth, rightDepth) + 1
}

// Problem 2: Validate Binary Search Tree
func isValidBST(root *TreeNode) bool {
    return validateBST(root, nil, nil)
}

func validateBST(node, min, max *TreeNode) bool {
    if node == nil {
        return true
    }
    
    if min != nil && node.Val <= min.Val {
        return false
    }
    
    if max != nil && node.Val >= max.Val {
        return false
    }
    
    return validateBST(node.Left, min, node) && 
           validateBST(node.Right, node, max)
}

// Problem 3: Lowest Common Ancestor
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    
    if left != nil && right != nil {
        return root
    }
    
    if left != nil {
        return left
    }
    
    return right
}

func main() {
    // Create a sample tree
    root := &TreeNode{Val: 3}
    root.Left = &TreeNode{Val: 5}
    root.Right = &TreeNode{Val: 1}
    root.Left.Left = &TreeNode{Val: 6}
    root.Left.Right = &TreeNode{Val: 2}
    root.Right.Left = &TreeNode{Val: 0}
    root.Right.Right = &TreeNode{Val: 8}
    root.Left.Right.Left = &TreeNode{Val: 7}
    root.Left.Right.Right = &TreeNode{Val: 4}
    
    // Test Maximum Depth
    depth := maxDepth(root)
    fmt.Printf("Maximum Depth: %d\n", depth)
    
    // Test BST Validation
    isValid := isValidBST(root)
    fmt.Printf("Is Valid BST: %v\n", isValid)
    
    // Test LCA
    p := root.Left
    q := root.Right
    lca := lowestCommonAncestor(root, p, q)
    fmt.Printf("LCA of %d and %d: %d\n", p.Val, q.Val, lca.Val)
}
```

### 2. Dynamic Programming

#### Classic DP Problems
```go
package main

import "fmt"

// Problem 1: Fibonacci with Memoization
func fibonacci(n int) int {
    memo := make(map[int]int)
    return fibMemo(n, memo)
}

func fibMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo)
    return memo[n]
}

// Problem 2: Longest Increasing Subsequence
func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    dp := make([]int, len(nums))
    for i := range dp {
        dp[i] = 1
    }
    
    maxLen := 1
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
        maxLen = max(maxLen, dp[i])
    }
    
    return maxLen
}

// Problem 3: Coin Change
func coinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for i := 1; i <= amount; i++ {
        dp[i] = amount + 1
    }
    dp[0] = 0
    
    for i := 1; i <= amount; i++ {
        for _, coin := range coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i-coin]+1)
            }
        }
    }
    
    if dp[amount] > amount {
        return -1
    }
    
    return dp[amount]
}

func main() {
    // Test Fibonacci
    fmt.Printf("Fibonacci(10): %d\n", fibonacci(10))
    
    // Test LIS
    nums := []int{10, 9, 2, 5, 3, 7, 101, 18}
    lis := lengthOfLIS(nums)
    fmt.Printf("Length of LIS: %d\n", lis)
    
    // Test Coin Change
    coins := []int{1, 3, 4}
    amount := 6
    minCoins := coinChange(coins, amount)
    fmt.Printf("Minimum coins for %d: %d\n", amount, minCoins)
}
```

## System Design Practice

### 1. Design Problems

#### URL Shortener
```go
package main

import (
    "crypto/md5"
    "fmt"
    "math/rand"
    "time"
)

type URLShortener struct {
    urlMap    map[string]string
    shortMap  map[string]string
    baseURL   string
    counter   int64
}

func NewURLShortener(baseURL string) *URLShortener {
    return &URLShortener{
        urlMap:   make(map[string]string),
        shortMap: make(map[string]string),
        baseURL:  baseURL,
        counter:  0,
    }
}

func (us *URLShortener) ShortenURL(longURL string) string {
    // Check if URL already exists
    if shortURL, exists := us.urlMap[longURL]; exists {
        return shortURL
    }
    
    // Generate short code
    shortCode := us.generateShortCode()
    shortURL := us.baseURL + "/" + shortCode
    
    // Store mappings
    us.urlMap[longURL] = shortURL
    us.shortMap[shortCode] = longURL
    
    return shortURL
}

func (us *URLShortener) ExpandURL(shortURL string) string {
    shortCode := us.extractShortCode(shortURL)
    if longURL, exists := us.shortMap[shortCode]; exists {
        return longURL
    }
    return ""
}

func (us *URLShortener) generateShortCode() string {
    // Simple counter-based approach
    // In production, use more sophisticated encoding
    us.counter++
    
    // Convert counter to base62
    chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := ""
    num := us.counter
    
    for num > 0 {
        result = string(chars[num%62]) + result
        num /= 62
    }
    
    return result
}

func (us *URLShortener) extractShortCode(shortURL string) string {
    // Extract the short code from the full URL
    // This is a simplified version
    return shortURL[len(us.baseURL)+1:]
}

func main() {
    shortener := NewURLShortener("https://short.ly")
    
    // Test URL shortening
    longURL := "https://www.example.com/very/long/url/with/many/segments"
    shortURL := shortener.ShortenURL(longURL)
    fmt.Printf("Original: %s\n", longURL)
    fmt.Printf("Shortened: %s\n", shortURL)
    
    // Test URL expansion
    expanded := shortener.ExpandURL(shortURL)
    fmt.Printf("Expanded: %s\n", expanded)
    
    // Test with another URL
    longURL2 := "https://www.google.com/search?q=golang"
    shortURL2 := shortener.ShortenURL(longURL2)
    fmt.Printf("\nOriginal: %s\n", longURL2)
    fmt.Printf("Shortened: %s\n", shortURL2)
}
```

#### Chat System
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Message struct {
    ID        string
    UserID    string
    Content   string
    Timestamp time.Time
    RoomID    string
}

type ChatRoom struct {
    ID       string
    Name     string
    Users    map[string]*User
    Messages []*Message
    mutex    sync.RWMutex
}

type User struct {
    ID       string
    Username string
    Rooms    map[string]*ChatRoom
}

type ChatSystem struct {
    rooms map[string]*ChatRoom
    users map[string]*User
    mutex sync.RWMutex
}

func NewChatSystem() *ChatSystem {
    return &ChatSystem{
        rooms: make(map[string]*ChatRoom),
        users: make(map[string]*User),
    }
}

func (cs *ChatSystem) CreateRoom(roomID, roomName string) *ChatRoom {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    room := &ChatRoom{
        ID:       roomID,
        Name:     roomName,
        Users:    make(map[string]*User),
        Messages: make([]*Message, 0),
    }
    
    cs.rooms[roomID] = room
    return room
}

func (cs *ChatSystem) JoinRoom(userID, roomID string) bool {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    room, exists := cs.rooms[roomID]
    if !exists {
        return false
    }
    
    user, exists := cs.users[userID]
    if !exists {
        return false
    }
    
    room.mutex.Lock()
    room.Users[userID] = user
    user.Rooms[roomID] = room
    room.mutex.Unlock()
    
    return true
}

func (cs *ChatSystem) SendMessage(userID, roomID, content string) *Message {
    cs.mutex.RLock()
    room, exists := cs.rooms[roomID]
    cs.mutex.RUnlock()
    
    if !exists {
        return nil
    }
    
    room.mutex.Lock()
    defer room.mutex.Unlock()
    
    message := &Message{
        ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
        UserID:    userID,
        Content:   content,
        Timestamp: time.Now(),
        RoomID:    roomID,
    }
    
    room.Messages = append(room.Messages, message)
    return message
}

func (cs *ChatSystem) GetMessages(roomID string, limit int) []*Message {
    cs.mutex.RLock()
    room, exists := cs.rooms[roomID]
    cs.mutex.RUnlock()
    
    if !exists {
        return nil
    }
    
    room.mutex.RLock()
    defer room.mutex.RUnlock()
    
    start := len(room.Messages) - limit
    if start < 0 {
        start = 0
    }
    
    return room.Messages[start:]
}

func main() {
    chat := NewChatSystem()
    
    // Create users
    user1 := &User{ID: "user1", Username: "alice", Rooms: make(map[string]*ChatRoom)}
    user2 := &User{ID: "user2", Username: "bob", Rooms: make(map[string]*ChatRoom)}
    
    chat.users["user1"] = user1
    chat.users["user2"] = user2
    
    // Create room
    room := chat.CreateRoom("room1", "General Chat")
    fmt.Printf("Created room: %s\n", room.Name)
    
    // Join room
    chat.JoinRoom("user1", "room1")
    chat.JoinRoom("user2", "room1")
    fmt.Println("Users joined room")
    
    // Send messages
    chat.SendMessage("user1", "room1", "Hello everyone!")
    chat.SendMessage("user2", "room1", "Hi Alice!")
    chat.SendMessage("user1", "room1", "How are you?")
    
    // Get messages
    messages := chat.GetMessages("room1", 10)
    fmt.Println("\nRecent messages:")
    for _, msg := range messages {
        fmt.Printf("[%s] %s: %s\n", 
            msg.Timestamp.Format("15:04:05"), 
            chat.users[msg.UserID].Username, 
            msg.Content)
    }
}
```

## Database Practice

### 1. Query Optimization

#### Database Design Problems
```go
package main

import (
    "fmt"
    "time"
)

type DatabaseOptimizer struct {
    tables map[string]*Table
    indexes map[string]*Index
}

type Table struct {
    Name    string
    Columns []*Column
    Rows    int
    Size    int64
}

type Column struct {
    Name     string
    Type     string
    Nullable bool
    Indexed  bool
}

type Index struct {
    Name    string
    Table   string
    Columns []string
    Type    string
    Size    int64
}

type Query struct {
    SQL      string
    Tables   []string
    Columns  []string
    Filters  []string
    Joins    []string
    OrderBy  []string
    GroupBy  []string
}

func NewDatabaseOptimizer() *DatabaseOptimizer {
    return &DatabaseOptimizer{
        tables:  make(map[string]*Table),
        indexes: make(map[string]*Index),
    }
}

func (dbo *DatabaseOptimizer) AddTable(name string, columns []*Column, rows int) {
    dbo.tables[name] = &Table{
        Name:    name,
        Columns: columns,
        Rows:    rows,
        Size:    int64(rows * len(columns) * 100), // Estimated size
    }
}

func (dbo *DatabaseOptimizer) AddIndex(name, table string, columns []string, indexType string) {
    dbo.indexes[name] = &Index{
        Name:    name,
        Table:   table,
        Columns: columns,
        Type:    indexType,
        Size:    int64(len(columns) * 50), // Estimated size
    }
}

func (dbo *DatabaseOptimizer) AnalyzeQuery(query *Query) *QueryAnalysis {
    analysis := &QueryAnalysis{
        Query:     query,
        Cost:      0,
        Indexes:   []string{},
        Suggestions: []string{},
    }
    
    // Analyze table access
    for _, table := range query.Tables {
        if t, exists := dbo.tables[table]; exists {
            analysis.Cost += float64(t.Rows)
        }
    }
    
    // Check for available indexes
    for _, filter := range query.Filters {
        for _, index := range dbo.indexes {
            for _, col := range index.Columns {
                if contains(query.Columns, col) {
                    analysis.Indexes = append(analysis.Indexes, index.Name)
                    analysis.Cost *= 0.1 // Index reduces cost
                }
            }
        }
    }
    
    // Generate suggestions
    analysis.Suggestions = dbo.generateSuggestions(query)
    
    return analysis
}

func (dbo *DatabaseOptimizer) generateSuggestions(query *Query) []string {
    suggestions := []string{}
    
    // Check for missing indexes
    for _, filter := range query.Filters {
        if !dbo.hasIndexForColumn(filter) {
            suggestions = append(suggestions, 
                fmt.Sprintf("Consider adding index on %s", filter))
        }
    }
    
    // Check for unnecessary columns
    if len(query.Columns) > 10 {
        suggestions = append(suggestions, 
            "Consider selecting only necessary columns")
    }
    
    // Check for missing WHERE clause
    if len(query.Filters) == 0 {
        suggestions = append(suggestions, 
            "Add WHERE clause to limit result set")
    }
    
    return suggestions
}

func (dbo *DatabaseOptimizer) hasIndexForColumn(column string) bool {
    for _, index := range dbo.indexes {
        for _, col := range index.Columns {
            if col == column {
                return true
            }
        }
    }
    return false
}

func contains(slice []string, item string) bool {
    for _, s := range slice {
        if s == item {
            return true
        }
    }
    return false
}

type QueryAnalysis struct {
    Query       *Query
    Cost        float64
    Indexes     []string
    Suggestions []string
}

func main() {
    optimizer := NewDatabaseOptimizer()
    
    // Add sample tables
    usersColumns := []*Column{
        {Name: "id", Type: "INT", Nullable: false, Indexed: true},
        {Name: "email", Type: "VARCHAR", Nullable: false, Indexed: true},
        {Name: "name", Type: "VARCHAR", Nullable: false, Indexed: false},
        {Name: "created_at", Type: "TIMESTAMP", Nullable: false, Indexed: false},
    }
    optimizer.AddTable("users", usersColumns, 100000)
    
    ordersColumns := []*Column{
        {Name: "id", Type: "INT", Nullable: false, Indexed: true},
        {Name: "user_id", Type: "INT", Nullable: false, Indexed: true},
        {Name: "amount", Type: "DECIMAL", Nullable: false, Indexed: false},
        {Name: "status", Type: "VARCHAR", Nullable: false, Indexed: false},
        {Name: "created_at", Type: "TIMESTAMP", Nullable: false, Indexed: false},
    }
    optimizer.AddTable("orders", ordersColumns, 500000)
    
    // Add indexes
    optimizer.AddIndex("idx_users_email", "users", []string{"email"}, "B-tree")
    optimizer.AddIndex("idx_orders_user_id", "orders", []string{"user_id"}, "B-tree")
    optimizer.AddIndex("idx_orders_status", "orders", []string{"status"}, "B-tree")
    
    // Analyze a query
    query := &Query{
        SQL:     "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id WHERE u.email = 'test@example.com' AND o.status = 'completed'",
        Tables:  []string{"users", "orders"},
        Columns: []string{"name", "amount"},
        Filters: []string{"email", "status"},
        Joins:   []string{"users.id = orders.user_id"},
    }
    
    analysis := optimizer.AnalyzeQuery(query)
    
    fmt.Println("Query Analysis:")
    fmt.Println("===============")
    fmt.Printf("Query: %s\n", query.SQL)
    fmt.Printf("Estimated Cost: %.2f\n", analysis.Cost)
    fmt.Printf("Used Indexes: %v\n", analysis.Indexes)
    fmt.Println("Suggestions:")
    for i, suggestion := range analysis.Suggestions {
        fmt.Printf("  %d. %s\n", i+1, suggestion)
    }
}
```

## Real-World Projects

### 1. E-commerce System

#### Project Overview
```go
package main

import (
    "fmt"
    "time"
)

type ECommerceSystem struct {
    products  map[string]*Product
    users     map[string]*User
    orders    map[string]*Order
    inventory map[string]int
}

type Product struct {
    ID          string
    Name        string
    Price       float64
    Description string
    Category    string
    Stock       int
}

type User struct {
    ID       string
    Email    string
    Name     string
    Address  string
    Cart     []*CartItem
}

type CartItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

type Order struct {
    ID         string
    UserID     string
    Items      []*OrderItem
    Total      float64
    Status     string
    CreatedAt  time.Time
}

type OrderItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

func NewECommerceSystem() *ECommerceSystem {
    return &ECommerceSystem{
        products:  make(map[string]*Product),
        users:     make(map[string]*User),
        orders:    make(map[string]*Order),
        inventory: make(map[string]int),
    }
}

func (ecs *ECommerceSystem) AddProduct(product *Product) {
    ecs.products[product.ID] = product
    ecs.inventory[product.ID] = product.Stock
}

func (ecs *ECommerceSystem) AddUser(user *User) {
    ecs.users[user.ID] = user
}

func (ecs *ECommerceSystem) AddToCart(userID, productID string, quantity int) bool {
    user, exists := ecs.users[userID]
    if !exists {
        return false
    }
    
    product, exists := ecs.products[productID]
    if !exists {
        return false
    }
    
    if ecs.inventory[productID] < quantity {
        return false
    }
    
    // Check if item already in cart
    for _, item := range user.Cart {
        if item.ProductID == productID {
            item.Quantity += quantity
            return true
        }
    }
    
    // Add new item to cart
    cartItem := &CartItem{
        ProductID: productID,
        Quantity:  quantity,
        Price:     product.Price,
    }
    
    user.Cart = append(user.Cart, cartItem)
    return true
}

func (ecs *ECommerceSystem) CreateOrder(userID string) *Order {
    user, exists := ecs.users[userID]
    if !exists || len(user.Cart) == 0 {
        return nil
    }
    
    orderID := fmt.Sprintf("order_%d", time.Now().UnixNano())
    order := &Order{
        ID:        orderID,
        UserID:    userID,
        Items:     make([]*OrderItem, 0),
        Total:     0,
        Status:    "pending",
        CreatedAt: time.Now(),
    }
    
    // Convert cart items to order items
    for _, cartItem := range user.Cart {
        orderItem := &OrderItem{
            ProductID: cartItem.ProductID,
            Quantity:  cartItem.Quantity,
            Price:     cartItem.Price,
        }
        order.Items = append(order.Items, orderItem)
        order.Total += cartItem.Price * float64(cartItem.Quantity)
        
        // Update inventory
        ecs.inventory[cartItem.ProductID] -= cartItem.Quantity
    }
    
    // Clear cart
    user.Cart = make([]*CartItem, 0)
    
    ecs.orders[orderID] = order
    return order
}

func main() {
    ecs := NewECommerceSystem()
    
    // Add products
    product1 := &Product{
        ID:          "prod1",
        Name:        "Laptop",
        Price:       999.99,
        Description: "High-performance laptop",
        Category:    "Electronics",
        Stock:       10,
    }
    ecs.AddProduct(product1)
    
    product2 := &Product{
        ID:          "prod2",
        Name:        "Mouse",
        Price:       29.99,
        Description: "Wireless mouse",
        Category:    "Electronics",
        Stock:       50,
    }
    ecs.AddProduct(product2)
    
    // Add user
    user := &User{
        ID:      "user1",
        Email:   "test@example.com",
        Name:    "John Doe",
        Address: "123 Main St",
        Cart:    make([]*CartItem, 0),
    }
    ecs.AddUser(user)
    
    // Add items to cart
    ecs.AddToCart("user1", "prod1", 1)
    ecs.AddToCart("user1", "prod2", 2)
    
    // Create order
    order := ecs.CreateOrder("user1")
    if order != nil {
        fmt.Printf("Order created: %s\n", order.ID)
        fmt.Printf("Total: $%.2f\n", order.Total)
        fmt.Printf("Items: %d\n", len(order.Items))
    }
}
```

## Assessment Tools

### 1. Coding Challenge Generator

#### Problem Generator
```go
package main

import (
    "fmt"
    "math/rand"
    "time"
)

type ProblemGenerator struct {
    categories []string
    problems   map[string][]Problem
}

type Problem struct {
    Title       string
    Description string
    Difficulty  string
    Category    string
    Examples    []Example
    Constraints []string
}

type Example struct {
    Input    string
    Output   string
    Explanation string
}

func NewProblemGenerator() *ProblemGenerator {
    return &ProblemGenerator{
        categories: []string{"Arrays", "Strings", "Trees", "Graphs", "DP"},
        problems: map[string][]Problem{
            "Arrays": {
                {
                    Title: "Two Sum",
                    Description: "Given an array of integers and a target sum, return indices of the two numbers that add up to the target.",
                    Difficulty: "Easy",
                    Category: "Arrays",
                    Examples: []Example{
                        {
                            Input: "nums = [2,7,11,15], target = 9",
                            Output: "[0,1]",
                            Explanation: "Because nums[0] + nums[1] == 9, we return [0, 1].",
                        },
                    },
                    Constraints: []string{
                        "2 <= nums.length <= 10^4",
                        "-10^9 <= nums[i] <= 10^9",
                        "-10^9 <= target <= 10^9",
                    },
                },
            },
            "Strings": {
                {
                    Title: "Longest Substring Without Repeating Characters",
                    Description: "Given a string, find the length of the longest substring without repeating characters.",
                    Difficulty: "Medium",
                    Category: "Strings",
                    Examples: []Example{
                        {
                            Input: "s = \"abcabcbb\"",
                            Output: "3",
                            Explanation: "The answer is \"abc\", with the length of 3.",
                        },
                    },
                    Constraints: []string{
                        "0 <= s.length <= 5 * 10^4",
                        "s consists of English letters, digits, symbols and spaces.",
                    },
                },
            },
        },
    }
}

func (pg *ProblemGenerator) GetRandomProblem() Problem {
    rand.Seed(time.Now().UnixNano())
    
    category := pg.categories[rand.Intn(len(pg.categories))]
    problems := pg.problems[category]
    
    if len(problems) == 0 {
        return Problem{}
    }
    
    return problems[rand.Intn(len(problems))]
}

func (pg *ProblemGenerator) GetProblemsByDifficulty(difficulty string) []Problem {
    var result []Problem
    
    for _, problems := range pg.problems {
        for _, problem := range problems {
            if problem.Difficulty == difficulty {
                result = append(result, problem)
            }
        }
    }
    
    return result
}

func main() {
    generator := NewProblemGenerator()
    
    // Get random problem
    problem := generator.GetRandomProblem()
    fmt.Printf("Random Problem: %s\n", problem.Title)
    fmt.Printf("Difficulty: %s\n", problem.Difficulty)
    fmt.Printf("Description: %s\n", problem.Description)
    
    // Get easy problems
    easyProblems := generator.GetProblemsByDifficulty("Easy")
    fmt.Printf("\nEasy Problems: %d\n", len(easyProblems))
    
    for _, p := range easyProblems {
        fmt.Printf("- %s\n", p.Title)
    }
}
```

## Follow-up Questions

### 1. Practice Strategy
**Q: How do you structure your practice sessions?**
A: Start with easy problems, gradually increase difficulty, focus on understanding patterns, and practice regularly.

### 2. Problem-Solving Approach
**Q: What's the best way to approach a new coding problem?**
A: Read carefully, understand constraints, think of examples, design algorithm, code solution, test with examples.

### 3. System Design Practice
**Q: How do you practice system design effectively?**
A: Start with basic components, think about scalability, consider trade-offs, practice with real-world examples.

## Sources

### Practice Platforms
- **LeetCode**: [Coding Challenges](https://leetcode.com/)
- **HackerRank**: [Programming Challenges](https://www.hackerrank.com/)
- **CodeSignal**: [Technical Assessments](https://codesignal.com/)

### System Design Resources
- **System Design Primer**: [GitHub](https://github.com/donnemartin/system-design-primer/)
- **High Scalability**: [Blog](http://highscalability.com/)
- **AWS Architecture Center**: [Documentation](https://aws.amazon.com/architecture/)

### Project Ideas
- **GitHub**: [Open Source Projects](https://github.com/)
- **Kaggle**: [Data Science Projects](https://www.kaggle.com/)
- **Devpost**: [Hackathon Projects](https://devpost.com/)

---

**Next**: [Assessment Tools](../../README.md) | **Previous**: [Video Notes](../../README.md) | **Up**: [Practice Exercises](README.md/)
