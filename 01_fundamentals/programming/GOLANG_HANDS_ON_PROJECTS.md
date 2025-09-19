# 🚀 Go Hands-on Projects for Learning

> **Complete projects to build your Go programming skills from beginner to advanced**

## 📚 Project List

1. [Project 1: Personal Finance Tracker](#project-1-personal-finance-tracker)
2. [Project 2: URL Shortener Service](#project-2-url-shortener-service)
3. [Project 3: Chat Application](#project-3-chat-application)
4. [Project 4: File Backup Tool](#project-4-file-backup-tool)
5. [Project 5: Weather API Client](#project-5-weather-api-client)
6. [Project 6: Task Management System](#project-6-task-management-system)
7. [Project 7: Simple Web Server](#project-7-simple-web-server)
8. [Project 8: Log Analyzer](#project-8-log-analyzer)

---

## 💰 Project 1: Personal Finance Tracker

### Project Description
A command-line application to track personal income, expenses, and generate reports.

### Features
- Add income and expenses
- Categorize transactions
- Generate monthly/yearly reports
- Export data to CSV
- Budget tracking

### Implementation

#### Step 1: Project Structure
```
finance-tracker/
├── go.mod
├── main.go
├── models/
│   └── transaction.go
├── services/
│   └── finance.go
└── utils/
    └── helpers.go
```

#### Step 2: Models

**models/transaction.go**
```go
package models

import (
    "time"
    "fmt"
)

type TransactionType string

const (
    Income  TransactionType = "income"
    Expense TransactionType = "expense"
)

type Category string

const (
    Salary     Category = "salary"
    Freelance  Category = "freelance"
    Investment Category = "investment"
    Food       Category = "food"
    Transport  Category = "transport"
    Utilities  Category = "utilities"
    Entertainment Category = "entertainment"
    Other      Category = "other"
)

type Transaction struct {
    ID          int             `json:"id"`
    Type        TransactionType `json:"type"`
    Amount      float64         `json:"amount"`
    Category    Category        `json:"category"`
    Description string          `json:"description"`
    Date        time.Time       `json:"date"`
}

func (t Transaction) String() string {
    return fmt.Sprintf("[%s] %s: $%.2f - %s (%s)", 
        t.Type, t.Category, t.Amount, t.Description, t.Date.Format("2006-01-02"))
}
```

#### Step 3: Services

**services/finance.go**
```go
package services

import (
    "encoding/csv"
    "fmt"
    "os"
    "sort"
    "time"
    "finance-tracker/models"
)

type FinanceService struct {
    transactions []models.Transaction
    nextID       int
}

func NewFinanceService() *FinanceService {
    return &FinanceService{
        transactions: make([]models.Transaction, 0),
        nextID:       1,
    }
}

func (fs *FinanceService) AddTransaction(transactionType models.TransactionType, 
    amount float64, category models.Category, description string) {
    
    transaction := models.Transaction{
        ID:          fs.nextID,
        Type:        transactionType,
        Amount:      amount,
        Category:    category,
        Description: description,
        Date:        time.Now(),
    }
    
    fs.transactions = append(fs.transactions, transaction)
    fs.nextID++
    
    fmt.Printf("Added transaction: %s\n", transaction.String())
}

func (fs *FinanceService) GetTransactions() []models.Transaction {
    return fs.transactions
}

func (fs *FinanceService) GetBalance() float64 {
    var balance float64
    for _, t := range fs.transactions {
        if t.Type == models.Income {
            balance += t.Amount
        } else {
            balance -= t.Amount
        }
    }
    return balance
}

func (fs *FinanceService) GetMonthlyReport(month int, year int) map[models.Category]float64 {
    report := make(map[models.Category]float64)
    
    for _, t := range fs.transactions {
        if int(t.Date.Month()) == month && t.Date.Year() == year {
            report[t.Category] += t.Amount
        }
    }
    
    return report
}

func (fs *FinanceService) ExportToCSV(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := csv.NewWriter(file)
    defer writer.Flush()
    
    // Write header
    writer.Write([]string{"ID", "Type", "Amount", "Category", "Description", "Date"})
    
    // Write transactions
    for _, t := range fs.transactions {
        writer.Write([]string{
            fmt.Sprintf("%d", t.ID),
            string(t.Type),
            fmt.Sprintf("%.2f", t.Amount),
            string(t.Category),
            t.Description,
            t.Date.Format("2006-01-02 15:04:05"),
        })
    }
    
    return nil
}

func (fs *FinanceService) GetTopCategories(limit int) []models.Category {
    categoryTotals := make(map[models.Category]float64)
    
    for _, t := range fs.transactions {
        if t.Type == models.Expense {
            categoryTotals[t.Category] += t.Amount
        }
    }
    
    // Sort categories by total amount
    type categoryTotal struct {
        category models.Category
        total    float64
    }
    
    var sorted []categoryTotal
    for category, total := range categoryTotals {
        sorted = append(sorted, categoryTotal{category, total})
    }
    
    sort.Slice(sorted, func(i, j int) bool {
        return sorted[i].total > sorted[j].total
    })
    
    var result []models.Category
    for i, ct := range sorted {
        if i >= limit {
            break
        }
        result = append(result, ct.category)
    }
    
    return result
}
```

#### Step 4: Main Application

**main.go**
```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
    "time"
    "finance-tracker/models"
    "finance-tracker/services"
)

func main() {
    financeService := services.NewFinanceService()
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        showMenu()
        fmt.Print("Choose an option: ")
        scanner.Scan()
        choice := strings.TrimSpace(scanner.Text())
        
        switch choice {
        case "1":
            addTransaction(financeService, scanner)
        case "2":
            showTransactions(financeService)
        case "3":
            showBalance(financeService)
        case "4":
            showMonthlyReport(financeService, scanner)
        case "5":
            exportToCSV(financeService, scanner)
        case "6":
            showTopCategories(financeService)
        case "7":
            fmt.Println("Goodbye!")
            return
        default:
            fmt.Println("Invalid option!")
        }
        
        fmt.Println("\nPress Enter to continue...")
        scanner.Scan()
    }
}

func showMenu() {
    fmt.Println("\n=== Personal Finance Tracker ===")
    fmt.Println("1. Add Transaction")
    fmt.Println("2. View All Transactions")
    fmt.Println("3. View Balance")
    fmt.Println("4. Monthly Report")
    fmt.Println("5. Export to CSV")
    fmt.Println("6. Top Spending Categories")
    fmt.Println("7. Exit")
}

func addTransaction(fs *services.FinanceService, scanner *bufio.Scanner) {
    fmt.Print("Transaction type (income/expense): ")
    scanner.Scan()
    typeStr := strings.ToLower(strings.TrimSpace(scanner.Text()))
    
    var transactionType models.TransactionType
    switch typeStr {
    case "income":
        transactionType = models.Income
    case "expense":
        transactionType = models.Expense
    default:
        fmt.Println("Invalid transaction type!")
        return
    }
    
    fmt.Print("Amount: $")
    scanner.Scan()
    amountStr := strings.TrimSpace(scanner.Text())
    amount, err := strconv.ParseFloat(amountStr, 64)
    if err != nil {
        fmt.Println("Invalid amount!")
        return
    }
    
    fmt.Print("Category: ")
    scanner.Scan()
    categoryStr := strings.ToLower(strings.TrimSpace(scanner.Text()))
    category := models.Category(categoryStr)
    
    fmt.Print("Description: ")
    scanner.Scan()
    description := strings.TrimSpace(scanner.Text())
    
    fs.AddTransaction(transactionType, amount, category, description)
}

func showTransactions(fs *services.FinanceService) {
    transactions := fs.GetTransactions()
    if len(transactions) == 0 {
        fmt.Println("No transactions found!")
        return
    }
    
    fmt.Println("\n=== All Transactions ===")
    for _, t := range transactions {
        fmt.Println(t.String())
    }
}

func showBalance(fs *services.FinanceService) {
    balance := fs.GetBalance()
    fmt.Printf("\nCurrent Balance: $%.2f\n", balance)
}

func showMonthlyReport(fs *services.FinanceService, scanner *bufio.Scanner) {
    fmt.Print("Enter month (1-12): ")
    scanner.Scan()
    monthStr := strings.TrimSpace(scanner.Text())
    month, err := strconv.Atoi(monthStr)
    if err != nil || month < 1 || month > 12 {
        fmt.Println("Invalid month!")
        return
    }
    
    fmt.Print("Enter year: ")
    scanner.Scan()
    yearStr := strings.TrimSpace(scanner.Text())
    year, err := strconv.Atoi(yearStr)
    if err != nil {
        fmt.Println("Invalid year!")
        return
    }
    
    report := fs.GetMonthlyReport(month, year)
    if len(report) == 0 {
        fmt.Println("No transactions found for this month!")
        return
    }
    
    fmt.Printf("\n=== Monthly Report for %d/%d ===\n", month, year)
    for category, amount := range report {
        fmt.Printf("%s: $%.2f\n", category, amount)
    }
}

func exportToCSV(fs *services.FinanceService, scanner *bufio.Scanner) {
    fmt.Print("Enter filename (e.g., transactions.csv): ")
    scanner.Scan()
    filename := strings.TrimSpace(scanner.Text())
    
    if filename == "" {
        filename = "transactions.csv"
    }
    
    err := fs.ExportToCSV(filename)
    if err != nil {
        fmt.Printf("Error exporting to CSV: %v\n", err)
    } else {
        fmt.Printf("Data exported to %s successfully!\n", filename)
    }
}

func showTopCategories(fs *services.FinanceService) {
    fmt.Print("Enter number of top categories to show: ")
    scanner := bufio.NewScanner(os.Stdin)
    scanner.Scan()
    limitStr := strings.TrimSpace(scanner.Text())
    limit, err := strconv.Atoi(limitStr)
    if err != nil || limit <= 0 {
        limit = 5
    }
    
    topCategories := fs.GetTopCategories(limit)
    if len(topCategories) == 0 {
        fmt.Println("No spending categories found!")
        return
    }
    
    fmt.Printf("\n=== Top %d Spending Categories ===\n", limit)
    for i, category := range topCategories {
        fmt.Printf("%d. %s\n", i+1, category)
    }
}
```

#### Step 5: Initialize Module
```bash
go mod init finance-tracker
go run main.go
```

---

## 🔗 Project 2: URL Shortener Service

### Project Description
A web service that shortens long URLs and redirects users to the original URLs.

### Features
- Shorten URLs
- Redirect to original URLs
- Track click counts
- REST API
- Web interface

### Implementation

#### Step 1: Project Structure
```
url-shortener/
├── go.mod
├── main.go
├── handlers/
│   └── url.go
├── models/
│   └── url.go
├── services/
│   └── shortener.go
└── templates/
    └── index.html
```

#### Step 2: Models

**models/url.go**
```go
package models

import (
    "time"
    "crypto/rand"
    "encoding/base64"
)

type URL struct {
    ID          int       `json:"id"`
    OriginalURL string    `json:"original_url"`
    ShortCode   string    `json:"short_code"`
    Clicks      int       `json:"clicks"`
    CreatedAt   time.Time `json:"created_at"`
}

func NewURL(originalURL string) *URL {
    return &URL{
        OriginalURL: originalURL,
        ShortCode:   generateShortCode(),
        Clicks:      0,
        CreatedAt:   time.Now(),
    }
}

func generateShortCode() string {
    bytes := make([]byte, 6)
    rand.Read(bytes)
    return base64.URLEncoding.EncodeToString(bytes)[:8]
}
```

#### Step 3: Services

**services/shortener.go**
```go
package services

import (
    "fmt"
    "net/url"
    "sync"
    "url-shortener/models"
)

type ShortenerService struct {
    urls  map[string]*models.URL
    mutex sync.RWMutex
    nextID int
}

func NewShortenerService() *ShortenerService {
    return &ShortenerService{
        urls:   make(map[string]*models.URL),
        nextID: 1,
    }
}

func (ss *ShortenerService) ShortenURL(originalURL string) (*models.URL, error) {
    // Validate URL
    if !isValidURL(originalURL) {
        return nil, fmt.Errorf("invalid URL")
    }
    
    ss.mutex.Lock()
    defer ss.mutex.Unlock()
    
    // Check if URL already exists
    for _, u := range ss.urls {
        if u.OriginalURL == originalURL {
            return u, nil
        }
    }
    
    // Create new URL
    url := models.NewURL(originalURL)
    url.ID = ss.nextID
    ss.nextID++
    
    ss.urls[url.ShortCode] = url
    
    return url, nil
}

func (ss *ShortenerService) GetURL(shortCode string) (*models.URL, error) {
    ss.mutex.RLock()
    defer ss.mutex.RUnlock()
    
    url, exists := ss.urls[shortCode]
    if !exists {
        return nil, fmt.Errorf("URL not found")
    }
    
    return url, nil
}

func (ss *ShortenerService) IncrementClicks(shortCode string) error {
    ss.mutex.Lock()
    defer ss.mutex.Unlock()
    
    url, exists := ss.urls[shortCode]
    if !exists {
        return fmt.Errorf("URL not found")
    }
    
    url.Clicks++
    return nil
}

func (ss *ShortenerService) GetAllURLs() []*models.URL {
    ss.mutex.RLock()
    defer ss.mutex.RUnlock()
    
    urls := make([]*models.URL, 0, len(ss.urls))
    for _, url := range ss.urls {
        urls = append(urls, url)
    }
    
    return urls
}

func isValidURL(rawURL string) bool {
    _, err := url.ParseRequestURI(rawURL)
    return err == nil
}
```

#### Step 4: Handlers

**handlers/url.go**
```go
package handlers

import (
    "encoding/json"
    "fmt"
    "html/template"
    "net/http"
    "url-shortener/services"
)

type URLHandler struct {
    service *services.ShortenerService
}

func NewURLHandler(service *services.ShortenerService) *URLHandler {
    return &URLHandler{service: service}
}

func (uh *URLHandler) ShortenURL(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var request struct {
        URL string `json:"url"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    url, err := uh.service.ShortenURL(request.URL)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(url)
}

func (uh *URLHandler) RedirectURL(w http.ResponseWriter, r *http.Request) {
    shortCode := r.URL.Path[1:] // Remove leading slash
    
    url, err := uh.service.GetURL(shortCode)
    if err != nil {
        http.Error(w, "URL not found", http.StatusNotFound)
        return
    }
    
    // Increment click count
    uh.service.IncrementClicks(shortCode)
    
    // Redirect to original URL
    http.Redirect(w, r, url.OriginalURL, http.StatusMovedPermanently)
}

func (uh *URLHandler) GetStats(w http.ResponseWriter, r *http.Request) {
    urls := uh.service.GetAllURLs()
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(urls)
}

func (uh *URLHandler) HomePage(w http.ResponseWriter, r *http.Request) {
    tmpl := `
<!DOCTYPE html>
<html>
<head>
    <title>URL Shortener</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 10px; background: #e9ecef; border-radius: 4px; }
        .url-list { margin-top: 20px; }
        .url-item { padding: 10px; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>URL Shortener</h1>
        <form id="shortenForm">
            <input type="text" id="urlInput" placeholder="Enter URL to shorten" required>
            <button type="submit">Shorten URL</button>
        </form>
        <div id="result" class="result" style="display: none;"></div>
        <div class="url-list">
            <h3>Recent URLs</h3>
            <div id="urlList"></div>
        </div>
    </div>

    <script>
        document.getElementById('shortenForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = document.getElementById('urlInput').value;
            
            try {
                const response = await fetch('/api/shorten', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                if (response.ok) {
                    document.getElementById('result').innerHTML = 
                        '<strong>Shortened URL:</strong> <a href="' + data.short_code + '">' + 
                        window.location.origin + '/' + data.short_code + '</a>';
                    document.getElementById('result').style.display = 'block';
                    loadURLs();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });

        async function loadURLs() {
            try {
                const response = await fetch('/api/stats');
                const urls = await response.json();
                
                const urlList = document.getElementById('urlList');
                urlList.innerHTML = urls.map(url => 
                    '<div class="url-item">' +
                    '<strong>' + url.short_code + '</strong> - ' + url.original_url + 
                    ' (Clicks: ' + url.clicks + ')' +
                    '</div>'
                ).join('');
            } catch (error) {
                console.error('Error loading URLs:', error);
            }
        }

        loadURLs();
    </script>
</body>
</html>`
    
    w.Header().Set("Content-Type", "text/html")
    fmt.Fprint(w, tmpl)
}
```

#### Step 5: Main Application

**main.go**
```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "url-shortener/handlers"
    "url-shortener/services"
)

func main() {
    // Create services
    shortenerService := services.NewShortenerService()
    
    // Create handlers
    urlHandler := handlers.NewURLHandler(shortenerService)
    
    // Setup routes
    http.HandleFunc("/", urlHandler.HomePage)
    http.HandleFunc("/api/shorten", urlHandler.ShortenURL)
    http.HandleFunc("/api/stats", urlHandler.GetStats)
    http.HandleFunc("/", urlHandler.RedirectURL) // This will handle short URLs
    
    fmt.Println("URL Shortener server starting on http://localhost:8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

---

## 💬 Project 3: Chat Application

### Project Description
A real-time chat application using WebSockets for communication.

### Features
- Real-time messaging
- Multiple rooms
- User authentication
- Message history
- Online users list

### Implementation

#### Step 1: Project Structure
```
chat-app/
├── go.mod
├── main.go
├── models/
│   ├── user.go
│   └── message.go
├── handlers/
│   └── websocket.go
└── services/
    └── chat.go
```

#### Step 2: Models

**models/user.go**
```go
package models

import (
    "time"
    "crypto/rand"
    "encoding/hex"
)

type User struct {
    ID       string    `json:"id"`
    Username string    `json:"username"`
    Room     string    `json:"room"`
    JoinedAt time.Time `json:"joined_at"`
}

func NewUser(username, room string) *User {
    return &User{
        ID:       generateID(),
        Username: username,
        Room:     room,
        JoinedAt: time.Now(),
    }
}

func generateID() string {
    bytes := make([]byte, 16)
    rand.Read(bytes)
    return hex.EncodeToString(bytes)
}
```

**models/message.go**
```go
package models

import "time"

type Message struct {
    ID        string    `json:"id"`
    UserID    string    `json:"user_id"`
    Username  string    `json:"username"`
    Room      string    `json:"room"`
    Content   string    `json:"content"`
    Timestamp time.Time `json:"timestamp"`
}

type MessageType string

const (
    MessageTypeText     MessageType = "text"
    MessageTypeJoin     MessageType = "join"
    MessageTypeLeave    MessageType = "leave"
    MessageTypeError    MessageType = "error"
)

type ChatMessage struct {
    Type      MessageType `json:"type"`
    Message   *Message    `json:"message,omitempty"`
    Users     []*User     `json:"users,omitempty"`
    Error     string      `json:"error,omitempty"`
}
```

#### Step 3: Services

**services/chat.go**
```go
package services

import (
    "fmt"
    "sync"
    "time"
    "chat-app/models"
)

type ChatService struct {
    rooms      map[string]*Room
    users      map[string]*models.User
    mutex      sync.RWMutex
    messageID  int
}

type Room struct {
    Name     string
    Users    map[string]*models.User
    Messages []*models.Message
    mutex    sync.RWMutex
}

func NewChatService() *ChatService {
    return &ChatService{
        rooms: make(map[string]*Room),
        users: make(map[string]*models.User),
    }
}

func (cs *ChatService) CreateRoom(name string) *Room {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    room := &Room{
        Name:     name,
        Users:    make(map[string]*models.User),
        Messages: make([]*models.Message, 0),
    }
    
    cs.rooms[name] = room
    return room
}

func (cs *ChatService) GetRoom(name string) *Room {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    return cs.rooms[name]
}

func (cs *ChatService) JoinRoom(user *models.User) error {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    room, exists := cs.rooms[user.Room]
    if !exists {
        room = cs.CreateRoom(user.Room)
    }
    
    room.mutex.Lock()
    room.Users[user.ID] = user
    room.mutex.Unlock()
    
    cs.users[user.ID] = user
    
    return nil
}

func (cs *ChatService) LeaveRoom(userID string) error {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    
    user, exists := cs.users[userID]
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    room, exists := cs.rooms[user.Room]
    if exists {
        room.mutex.Lock()
        delete(room.Users, userID)
        room.mutex.Unlock()
    }
    
    delete(cs.users, userID)
    return nil
}

func (cs *ChatService) SendMessage(userID, content string) (*models.Message, error) {
    cs.mutex.RLock()
    user, exists := cs.users[userID]
    if !exists {
        cs.mutex.RUnlock()
        return nil, fmt.Errorf("user not found")
    }
    cs.mutex.RUnlock()
    
    room := cs.GetRoom(user.Room)
    if room == nil {
        return nil, fmt.Errorf("room not found")
    }
    
    cs.mutex.Lock()
    cs.messageID++
    messageID := cs.messageID
    cs.mutex.Unlock()
    
    message := &models.Message{
        ID:        fmt.Sprintf("%d", messageID),
        UserID:    userID,
        Username:  user.Username,
        Room:      user.Room,
        Content:   content,
        Timestamp: time.Now(),
    }
    
    room.mutex.Lock()
    room.Messages = append(room.Messages, message)
    room.mutex.Unlock()
    
    return message, nil
}

func (cs *ChatService) GetRoomUsers(roomName string) []*models.User {
    room := cs.GetRoom(roomName)
    if room == nil {
        return nil
    }
    
    room.mutex.RLock()
    defer room.mutex.RUnlock()
    
    users := make([]*models.User, 0, len(room.Users))
    for _, user := range room.Users {
        users = append(users, user)
    }
    
    return users
}

func (cs *ChatService) GetRoomMessages(roomName string) []*models.Message {
    room := cs.GetRoom(roomName)
    if room == nil {
        return nil
    }
    
    room.mutex.RLock()
    defer room.mutex.RUnlock()
    
    return room.Messages
}
```

#### Step 4: WebSocket Handler

**handlers/websocket.go**
```go
package handlers

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "chat-app/models"
    "chat-app/services"
    "github.com/gorilla/websocket"
)

type WebSocketHandler struct {
    service *services.ChatService
    upgrader websocket.Upgrader
    clients  map[*websocket.Conn]*models.User
}

func NewWebSocketHandler(service *services.ChatService) *WebSocketHandler {
    return &WebSocketHandler{
        service: service,
        upgrader: websocket.Upgrader{
            CheckOrigin: func(r *http.Request) bool {
                return true // Allow all origins in development
            },
        },
        clients: make(map[*websocket.Conn]*models.User),
    }
}

func (wsh *WebSocketHandler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    conn, err := wsh.upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Printf("WebSocket upgrade error: %v", err)
        return
    }
    defer conn.Close()
    
    // Read initial message to get user info
    var initMessage struct {
        Username string `json:"username"`
        Room     string `json:"room"`
    }
    
    if err := conn.ReadJSON(&initMessage); err != nil {
        log.Printf("Error reading initial message: %v", err)
        return
    }
    
    // Create user
    user := models.NewUser(initMessage.Username, initMessage.Room)
    
    // Join room
    if err := wsh.service.JoinRoom(user); err != nil {
        conn.WriteJSON(models.ChatMessage{
            Type:  models.MessageTypeError,
            Error: err.Error(),
        })
        return
    }
    
    // Store client
    wsh.clients[conn] = user
    
    // Send join message
    joinMessage := models.ChatMessage{
        Type: models.MessageTypeJoin,
        Message: &models.Message{
            Username:  user.Username,
            Room:      user.Room,
            Content:   fmt.Sprintf("%s joined the room", user.Username),
            Timestamp: user.JoinedAt,
        },
    }
    
    wsh.broadcastToRoom(user.Room, joinMessage)
    
    // Send current users
    users := wsh.service.GetRoomUsers(user.Room)
    conn.WriteJSON(models.ChatMessage{
        Type:  models.MessageTypeJoin,
        Users: users,
    })
    
    // Handle messages
    for {
        var message struct {
            Content string `json:"content"`
        }
        
        if err := conn.ReadJSON(&message); err != nil {
            log.Printf("Error reading message: %v", err)
            break
        }
        
        // Send message
        msg, err := wsh.service.SendMessage(user.ID, message.Content)
        if err != nil {
            conn.WriteJSON(models.ChatMessage{
                Type:  models.MessageTypeError,
                Error: err.Error(),
            })
            continue
        }
        
        // Broadcast message
        chatMessage := models.ChatMessage{
            Type:    models.MessageTypeText,
            Message: msg,
        }
        
        wsh.broadcastToRoom(user.Room, chatMessage)
    }
    
    // Clean up
    delete(wsh.clients, conn)
    wsh.service.LeaveRoom(user.ID)
    
    // Send leave message
    leaveMessage := models.ChatMessage{
        Type: models.MessageTypeLeave,
        Message: &models.Message{
            Username:  user.Username,
            Room:      user.Room,
            Content:   fmt.Sprintf("%s left the room", user.Username),
            Timestamp: user.JoinedAt,
        },
    }
    
    wsh.broadcastToRoom(user.Room, leaveMessage)
}

func (wsh *WebSocketHandler) broadcastToRoom(roomName string, message models.ChatMessage) {
    users := wsh.service.GetRoomUsers(roomName)
    
    for conn, user := range wsh.clients {
        if user.Room == roomName {
            if err := conn.WriteJSON(message); err != nil {
                log.Printf("Error broadcasting message: %v", err)
                conn.Close()
                delete(wsh.clients, conn)
            }
        }
    }
}

func (wsh *WebSocketHandler) ServeHome(w http.ResponseWriter, r *http.Request) {
    html := `
<!DOCTYPE html>
<html>
<head>
    <title>Chat App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0; background: white; }
        .message { margin: 5px 0; padding: 5px; border-radius: 4px; }
        .message.system { background: #e9ecef; font-style: italic; }
        .message.user { background: #d4edda; }
        .users { background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chat Application</h1>
        <div id="loginForm">
            <input type="text" id="username" placeholder="Enter your username" required>
            <input type="text" id="room" placeholder="Enter room name" required>
            <button onclick="connect()">Join Chat</button>
        </div>
        <div id="chatForm" style="display: none;">
            <div class="users">
                <h3>Online Users</h3>
                <div id="usersList"></div>
            </div>
            <div id="messages"></div>
            <input type="text" id="messageInput" placeholder="Type your message..." required>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws;
        let currentUser;

        function connect() {
            const username = document.getElementById('username').value;
            const room = document.getElementById('room').value;
            
            if (!username || !room) {
                alert('Please enter username and room');
                return;
            }
            
            ws = new WebSocket('ws://localhost:8080/ws');
            
            ws.onopen = function() {
                ws.send(JSON.stringify({ username: username, room: room }));
                currentUser = { username: username, room: room };
                document.getElementById('loginForm').style.display = 'none';
                document.getElementById('chatForm').style.display = 'block';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'join' && data.users) {
                    updateUsersList(data.users);
                } else if (data.message) {
                    addMessage(data.message);
                }
            };
            
            ws.onclose = function() {
                alert('Connection closed');
                document.getElementById('loginForm').style.display = 'block';
                document.getElementById('chatForm').style.display = 'none';
            };
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message && ws) {
                ws.send(JSON.stringify({ content: message }));
                input.value = '';
            }
        }
        
        function addMessage(message) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (message.username === currentUser.username ? 'user' : 'system');
            messageDiv.innerHTML = '<strong>' + message.username + ':</strong> ' + message.content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function updateUsersList(users) {
            const usersList = document.getElementById('usersList');
            usersList.innerHTML = users.map(user => 
                '<div>' + user.username + '</div>'
            ).join('');
        }
        
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>`
    
    w.Header().Set("Content-Type", "text/html")
    fmt.Fprint(w, html)
}
```

#### Step 5: Main Application

**main.go**
```go
package main

import (
    "log"
    "net/http"
    "chat-app/handlers"
    "chat-app/services"
)

func main() {
    // Create services
    chatService := services.NewChatService()
    
    // Create handlers
    wsHandler := handlers.NewWebSocketHandler(chatService)
    
    // Setup routes
    http.HandleFunc("/", wsHandler.ServeHome)
    http.HandleFunc("/ws", wsHandler.HandleWebSocket)
    
    log.Println("Chat server starting on http://localhost:8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

---

## 🎯 Project Summary

These projects demonstrate:

1. **Personal Finance Tracker**: File I/O, data structures, CSV handling
2. **URL Shortener**: Web services, REST APIs, HTML templates
3. **Chat Application**: WebSockets, real-time communication, concurrency

Each project includes:
- Complete source code
- Step-by-step implementation
- Error handling
- User interfaces
- Testing considerations

**🚀 Start with Project 1 and work your way up to build your Go skills!**
