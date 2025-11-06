---
# Auto-generated front matter
Title: Golang Practical Examples
LastUpdated: 2025-11-06T20:45:58.761521
Tags: []
Status: draft
---

# 🚀 Go Practical Examples with Detailed Explanations

> **Real-world Go code examples with step-by-step explanations for beginners**

## 📚 Table of Contents

1. [File Operations](#file-operations)
2. [HTTP Server](#http-server)
3. [JSON Handling](#json-handling)
4. [Database Operations](#database-operations)
5. [Concurrent Programming](#concurrent-programming)
6. [Web Scraping](#web-scraping)
7. [CLI Application](#cli-application)
8. [Configuration Management](#configuration-management)
9. [Logging](#logging)
10. [Testing](#testing)

---

## 📁 File Operations

### Reading and Writing Files

```go
package main

import (
    "fmt"
    "io/ioutil"
    "os"
    "path/filepath"
)

func main() {
    // Write data to a file
    data := []byte("Hello, Go!\nThis is a sample file.")
    err := ioutil.WriteFile("sample.txt", data, 0644)
    if err != nil {
        fmt.Printf("Error writing file: %v\n", err)
        return
    }
    fmt.Println("File written successfully!")

    // Read data from a file
    content, err := ioutil.ReadFile("sample.txt")
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fmt.Printf("File content:\n%s\n", string(content))

    // Check if file exists
    if _, err := os.Stat("sample.txt"); os.IsNotExist(err) {
        fmt.Println("File does not exist")
    } else {
        fmt.Println("File exists")
    }

    // Get file information
    fileInfo, err := os.Stat("sample.txt")
    if err != nil {
        fmt.Printf("Error getting file info: %v\n", err)
        return
    }
    fmt.Printf("File size: %d bytes\n", fileInfo.Size())
    fmt.Printf("File permissions: %v\n", fileInfo.Mode())
    fmt.Printf("Last modified: %v\n", fileInfo.ModTime())

    // List files in directory
    files, err := ioutil.ReadDir(".")
    if err != nil {
        fmt.Printf("Error reading directory: %v\n", err)
        return
    }
    fmt.Println("Files in current directory:")
    for _, file := range files {
        fmt.Printf("  %s (size: %d bytes)\n", file.Name(), file.Size())
    }
}
```

**Explanation:**
- `ioutil.WriteFile()` writes data to a file with specified permissions
- `ioutil.ReadFile()` reads entire file content into memory
- `os.Stat()` gets file information
- `ioutil.ReadDir()` lists directory contents
- File permissions `0644` means: owner can read/write, group and others can read

### Working with CSV Files

```go
package main

import (
    "encoding/csv"
    "fmt"
    "os"
)

type Person struct {
    Name  string
    Age   int
    Email string
}

func main() {
    // Create sample data
    people := []Person{
        {"John Doe", 30, "john@example.com"},
        {"Jane Smith", 25, "jane@example.com"},
        {"Bob Johnson", 35, "bob@example.com"},
    }

    // Write to CSV
    file, err := os.Create("people.csv")
    if err != nil {
        fmt.Printf("Error creating file: %v\n", err)
        return
    }
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    // Write header
    writer.Write([]string{"Name", "Age", "Email"})

    // Write data
    for _, person := range people {
        writer.Write([]string{
            person.Name,
            fmt.Sprintf("%d", person.Age),
            person.Email,
        })
    }

    fmt.Println("CSV file written successfully!")

    // Read from CSV
    file, err = os.Open("people.csv")
    if err != nil {
        fmt.Printf("Error opening file: %v\n", err)
        return
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        fmt.Printf("Error reading CSV: %v\n", err)
        return
    }

    fmt.Println("CSV file content:")
    for i, record := range records {
        if i == 0 {
            fmt.Printf("Header: %v\n", record)
        } else {
            fmt.Printf("Row %d: %v\n", i, record)
        }
    }
}
```

**Explanation:**
- `csv.NewWriter()` creates a CSV writer
- `writer.Write()` writes a single row
- `writer.Flush()` ensures all data is written
- `csv.NewReader()` creates a CSV reader
- `reader.ReadAll()` reads all records at once

---

## 🌐 HTTP Server

### Basic HTTP Server

```go
package main

import (
    "fmt"
    "html/template"
    "net/http"
    "time"
)

// Handler function for root path
func homeHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Welcome to Go HTTP Server!\n")
    fmt.Fprintf(w, "Current time: %s\n", time.Now().Format("2006-01-02 15:04:05"))
    fmt.Fprintf(w, "Request method: %s\n", r.Method)
    fmt.Fprintf(w, "Request URL: %s\n", r.URL.Path)
}

// Handler function for /hello path
func helloHandler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("name")
    if name == "" {
        name = "World"
    }
    fmt.Fprintf(w, "Hello, %s!\n", name)
}

// Handler function for /info path
func infoHandler(w http.ResponseWriter, r *http.Request) {
    info := struct {
        Method     string
        URL        string
        Headers    map[string][]string
        RemoteAddr string
    }{
        Method:     r.Method,
        URL:        r.URL.String(),
        Headers:    r.Header,
        RemoteAddr: r.RemoteAddr,
    }

    // Set content type to JSON
    w.Header().Set("Content-Type", "application/json")
    
    // Simple JSON response (in real app, use json.Marshal)
    fmt.Fprintf(w, `{
        "method": "%s",
        "url": "%s",
        "remote_addr": "%s",
        "headers": {
            "user_agent": "%s",
            "accept": "%s"
        }
    }`, info.Method, info.URL, info.RemoteAddr, 
       r.Header.Get("User-Agent"), r.Header.Get("Accept"))
}

func main() {
    // Register handlers
    http.HandleFunc("/", homeHandler)
    http.HandleFunc("/hello", helloHandler)
    http.HandleFunc("/info", infoHandler)

    // Start server
    fmt.Println("Server starting on http://localhost:8080")
    fmt.Println("Available endpoints:")
    fmt.Println("  GET  /")
    fmt.Println("  GET  /hello?name=YourName")
    fmt.Println("  GET  /info")
    
    err := http.ListenAndServe(":8080", nil)
    if err != nil {
        fmt.Printf("Server error: %v\n", err)
    }
}
```

**Explanation:**
- `http.HandleFunc()` registers a handler function for a path
- `http.ResponseWriter` is used to write responses
- `*http.Request` contains request information
- `r.URL.Query().Get()` gets query parameters
- `w.Header().Set()` sets response headers
- `http.ListenAndServe()` starts the server

### REST API with JSON

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "time"

    "github.com/gorilla/mux"
)

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

type UserService struct {
    users []User
    nextID int
}

func NewUserService() *UserService {
    return &UserService{
        users:  []User{},
        nextID: 1,
    }
}

func (us *UserService) GetUsers(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(us.users)
}

func (us *UserService) GetUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    for _, user := range us.users {
        if user.ID == id {
            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(user)
            return
        }
    }

    http.Error(w, "User not found", http.StatusNotFound)
}

func (us *UserService) CreateUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    user.ID = us.nextID
    user.CreatedAt = time.Now()
    us.nextID++

    us.users = append(us.users, user)

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}

func (us *UserService) UpdateUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    var updatedUser User
    if err := json.NewDecoder(r.Body).Decode(&updatedUser); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    for i, user := range us.users {
        if user.ID == id {
            updatedUser.ID = id
            updatedUser.CreatedAt = user.CreatedAt
            us.users[i] = updatedUser

            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(updatedUser)
            return
        }
    }

    http.Error(w, "User not found", http.StatusNotFound)
}

func (us *UserService) DeleteUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    for i, user := range us.users {
        if user.ID == id {
            us.users = append(us.users[:i], us.users[i+1:]...)
            w.WriteHeader(http.StatusNoContent)
            return
        }
    }

    http.Error(w, "User not found", http.StatusNotFound)
}

func main() {
    userService := NewUserService()

    r := mux.NewRouter()
    
    // API routes
    api := r.PathPrefix("/api/v1").Subrouter()
    api.HandleFunc("/users", userService.GetUsers).Methods("GET")
    api.HandleFunc("/users", userService.CreateUser).Methods("POST")
    api.HandleFunc("/users/{id}", userService.GetUser).Methods("GET")
    api.HandleFunc("/users/{id}", userService.UpdateUser).Methods("PUT")
    api.HandleFunc("/users/{id}", userService.DeleteUser).Methods("DELETE")

    // Health check endpoint
    r.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        fmt.Fprint(w, "OK")
    }).Methods("GET")

    fmt.Println("Server starting on http://localhost:8080")
    fmt.Println("Available endpoints:")
    fmt.Println("  GET    /api/v1/users")
    fmt.Println("  POST   /api/v1/users")
    fmt.Println("  GET    /api/v1/users/{id}")
    fmt.Println("  PUT    /api/v1/users/{id}")
    fmt.Println("  DELETE /api/v1/users/{id}")
    fmt.Println("  GET    /health")

    log.Fatal(http.ListenAndServe(":8080", r))
}
```

**Explanation:**
- `gorilla/mux` provides URL routing and parameter extraction
- JSON struct tags define field names in JSON
- `json.NewEncoder().Encode()` converts struct to JSON
- `json.NewDecoder().Decode()` converts JSON to struct
- HTTP status codes indicate success/failure
- `mux.Vars()` extracts URL parameters

---

## 📄 JSON Handling

### Working with JSON

```go
package main

import (
    "encoding/json"
    "fmt"
    "os"
)

type Person struct {
    Name    string `json:"name"`
    Age     int    `json:"age"`
    Email   string `json:"email"`
    Address Address `json:"address"`
}

type Address struct {
    Street string `json:"street"`
    City   string `json:"city"`
    State  string `json:"state"`
    Zip    string `json:"zip"`
}

func main() {
    // Create a person
    person := Person{
        Name:  "John Doe",
        Age:   30,
        Email: "john@example.com",
        Address: Address{
            Street: "123 Main St",
            City:   "New York",
            State:  "NY",
            Zip:    "10001",
        },
    }

    // Convert struct to JSON
    jsonData, err := json.Marshal(person)
    if err != nil {
        fmt.Printf("Error marshaling JSON: %v\n", err)
        return
    }
    fmt.Printf("JSON: %s\n", string(jsonData))

    // Pretty print JSON
    jsonDataPretty, err := json.MarshalIndent(person, "", "  ")
    if err != nil {
        fmt.Printf("Error marshaling JSON: %v\n", err)
        return
    }
    fmt.Printf("Pretty JSON:\n%s\n", string(jsonDataPretty))

    // Save JSON to file
    err = os.WriteFile("person.json", jsonDataPretty, 0644)
    if err != nil {
        fmt.Printf("Error writing file: %v\n", err)
        return
    }
    fmt.Println("JSON saved to person.json")

    // Read JSON from file
    fileData, err := os.ReadFile("person.json")
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }

    // Convert JSON to struct
    var personFromFile Person
    err = json.Unmarshal(fileData, &personFromFile)
    if err != nil {
        fmt.Printf("Error unmarshaling JSON: %v\n", err)
        return
    }
    fmt.Printf("Person from file: %+v\n", personFromFile)

    // Working with JSON arrays
    people := []Person{
        {Name: "Alice", Age: 25, Email: "alice@example.com"},
        {Name: "Bob", Age: 30, Email: "bob@example.com"},
        {Name: "Charlie", Age: 35, Email: "charlie@example.com"},
    }

    jsonArray, err := json.MarshalIndent(people, "", "  ")
    if err != nil {
        fmt.Printf("Error marshaling JSON array: %v\n", err)
        return
    }
    fmt.Printf("People array:\n%s\n", string(jsonArray))

    // Parse JSON array
    var peopleFromJSON []Person
    err = json.Unmarshal(jsonArray, &peopleFromJSON)
    if err != nil {
        fmt.Printf("Error unmarshaling JSON array: %v\n", err)
        return
    }
    fmt.Printf("People from JSON: %+v\n", peopleFromJSON)
}
```

**Explanation:**
- `json.Marshal()` converts struct to JSON bytes
- `json.MarshalIndent()` creates pretty-printed JSON
- `json.Unmarshal()` converts JSON to struct
- JSON struct tags control field names and behavior
- `os.WriteFile()` and `os.ReadFile()` handle file operations

---

## 🗄️ Database Operations

### SQLite Database

```go
package main

import (
    "database/sql"
    "fmt"
    "log"

    _ "github.com/mattn/go-sqlite3"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func main() {
    // Open database connection
    db, err := sql.Open("sqlite3", "users.db")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create table
    createTableSQL := `
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL
    );`

    _, err = db.Exec(createTableSQL)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Table created successfully!")

    // Insert data
    insertSQL := `INSERT INTO users (name, email) VALUES (?, ?)`
    _, err = db.Exec(insertSQL, "John Doe", "john@example.com")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("User inserted successfully!")

    // Query single row
    var user User
    querySQL := `SELECT id, name, email FROM users WHERE id = ?`
    row := db.QueryRow(querySQL, 1)
    err = row.Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("User: %+v\n", user)

    // Query multiple rows
    rows, err := db.Query("SELECT id, name, email FROM users")
    if err != nil {
        log.Fatal(err)
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        err = rows.Scan(&u.ID, &u.Name, &u.Email)
        if err != nil {
            log.Fatal(err)
        }
        users = append(users, u)
    }

    fmt.Println("All users:")
    for _, u := range users {
        fmt.Printf("  %+v\n", u)
    }

    // Update data
    updateSQL := `UPDATE users SET name = ? WHERE id = ?`
    _, err = db.Exec(updateSQL, "John Smith", 1)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("User updated successfully!")

    // Delete data
    deleteSQL := `DELETE FROM users WHERE id = ?`
    _, err = db.Exec(deleteSQL, 1)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("User deleted successfully!")
}
```

**Explanation:**
- `sql.Open()` opens a database connection
- `db.Exec()` executes SQL statements that don't return rows
- `db.QueryRow()` queries a single row
- `db.Query()` queries multiple rows
- `rows.Scan()` scans row data into variables
- `defer` ensures resources are cleaned up

---

## ⚡ Concurrent Programming

### Worker Pool Pattern

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID       int
    Data     string
    Duration time.Duration
}

type Worker struct {
    ID       int
    JobQueue chan Job
    Quit     chan bool
    WG       *sync.WaitGroup
}

func NewWorker(id int, jobQueue chan Job, wg *sync.WaitGroup) *Worker {
    return &Worker{
        ID:       id,
        JobQueue: jobQueue,
        Quit:     make(chan bool),
        WG:       wg,
    }
}

func (w *Worker) Start() {
    go func() {
        for {
            select {
            case job := <-w.JobQueue:
                fmt.Printf("Worker %d processing job %d: %s\n", w.ID, job.ID, job.Data)
                time.Sleep(job.Duration)
                fmt.Printf("Worker %d completed job %d\n", w.ID, job.ID)
                w.WG.Done()
            case <-w.Quit:
                fmt.Printf("Worker %d quitting\n", w.ID)
                return
            }
        }
    }()
}

func (w *Worker) Stop() {
    go func() {
        w.Quit <- true
    }()
}

func main() {
    // Create job queue
    jobQueue := make(chan Job, 10)
    
    // Create wait group
    var wg sync.WaitGroup

    // Create workers
    numWorkers := 3
    workers := make([]*Worker, numWorkers)
    
    for i := 0; i < numWorkers; i++ {
        workers[i] = NewWorker(i+1, jobQueue, &wg)
        workers[i].Start()
    }

    // Create jobs
    jobs := []Job{
        {ID: 1, Data: "Process data 1", Duration: 2 * time.Second},
        {ID: 2, Data: "Process data 2", Duration: 1 * time.Second},
        {ID: 3, Data: "Process data 3", Duration: 3 * time.Second},
        {ID: 4, Data: "Process data 4", Duration: 1 * time.Second},
        {ID: 5, Data: "Process data 5", Duration: 2 * time.Second},
    }

    // Send jobs to queue
    for _, job := range jobs {
        wg.Add(1)
        jobQueue <- job
    }

    // Wait for all jobs to complete
    wg.Wait()
    fmt.Println("All jobs completed!")

    // Stop workers
    for _, worker := range workers {
        worker.Stop()
    }

    // Wait a bit for workers to stop
    time.Sleep(1 * time.Second)
    fmt.Println("All workers stopped!")
}
```

**Explanation:**
- Worker pool pattern manages concurrent job processing
- `sync.WaitGroup` waits for all goroutines to complete
- `select` statement handles multiple channel operations
- Channels provide communication between goroutines
- Each worker processes jobs from a shared queue

### Rate Limiting

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RateLimiter struct {
    requests chan time.Time
    rate     time.Duration
    burst    int
    mu       sync.Mutex
}

func NewRateLimiter(rate time.Duration, burst int) *RateLimiter {
    rl := &RateLimiter{
        requests: make(chan time.Time, burst),
        rate:     rate,
        burst:    burst,
    }
    
    // Start cleanup goroutine
    go rl.cleanup()
    
    return rl
}

func (rl *RateLimiter) cleanup() {
    ticker := time.NewTicker(rl.rate)
    defer ticker.Stop()
    
    for range ticker.C {
        select {
        case <-rl.requests:
            // Remove old request
        default:
            // No requests to remove
        }
    }
}

func (rl *RateLimiter) Allow() bool {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    select {
    case rl.requests <- time.Now():
        return true
    default:
        return false
    }
}

func (rl *RateLimiter) Wait() {
    for !rl.Allow() {
        time.Sleep(rl.rate)
    }
}

func main() {
    // Create rate limiter: 2 requests per second, burst of 5
    limiter := NewRateLimiter(time.Second, 5)
    
    // Simulate requests
    for i := 0; i < 10; i++ {
        if limiter.Allow() {
            fmt.Printf("Request %d: Allowed\n", i+1)
        } else {
            fmt.Printf("Request %d: Rate limited\n", i+1)
            limiter.Wait()
            fmt.Printf("Request %d: Allowed after wait\n", i+1)
        }
        time.Sleep(100 * time.Millisecond)
    }
}
```

**Explanation:**
- Rate limiter controls request frequency
- Token bucket algorithm implementation
- `sync.Mutex` protects shared state
- Goroutine cleans up old requests
- `select` with `default` provides non-blocking check

---

## 🕷️ Web Scraping

### Simple Web Scraper

```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "regexp"
    "strings"
    "time"
)

type Scraper struct {
    Client  *http.Client
    BaseURL string
}

func NewScraper(baseURL string) *Scraper {
    return &Scraper{
        Client: &http.Client{
            Timeout: 10 * time.Second,
        },
        BaseURL: baseURL,
    }
}

func (s *Scraper) Fetch(url string) (string, error) {
    resp, err := s.Client.Get(url)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return "", fmt.Errorf("HTTP error: %d", resp.StatusCode)
    }

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", err
    }

    return string(body), nil
}

func (s *Scraper) ExtractLinks(html string) []string {
    // Simple regex to extract links (not production-ready)
    linkRegex := regexp.MustCompile(`<a[^>]+href="([^"]+)"`)
    matches := linkRegex.FindAllStringSubmatch(html, -1)
    
    var links []string
    for _, match := range matches {
        if len(match) > 1 {
            link := match[1]
            if strings.HasPrefix(link, "http") {
                links = append(links, link)
            } else if strings.HasPrefix(link, "/") {
                links = append(links, s.BaseURL+link)
            }
        }
    }
    
    return links
}

func (s *Scraper) ExtractText(html string) string {
    // Remove HTML tags (simple approach)
    tagRegex := regexp.MustCompile(`<[^>]*>`)
    text := tagRegex.ReplaceAllString(html, " ")
    
    // Clean up whitespace
    whitespaceRegex := regexp.MustCompile(`\s+`)
    text = whitespaceRegex.ReplaceAllString(text, " ")
    
    return strings.TrimSpace(text)
}

func main() {
    scraper := NewScraper("https://example.com")
    
    // Fetch webpage
    html, err := scraper.Fetch("https://example.com")
    if err != nil {
        fmt.Printf("Error fetching page: %v\n", err)
        return
    }
    
    // Extract links
    links := scraper.ExtractLinks(html)
    fmt.Printf("Found %d links:\n", len(links))
    for i, link := range links {
        if i < 5 { // Show first 5 links
            fmt.Printf("  %s\n", link)
        }
    }
    
    // Extract text
    text := scraper.ExtractText(html)
    fmt.Printf("\nPage text (first 200 chars):\n%s...\n", text[:min(200, len(text))])
}
```

**Explanation:**
- `http.Client` handles HTTP requests with timeout
- Regular expressions extract data from HTML
- `io.ReadAll()` reads response body
- Error handling for network and parsing errors
- Simple text extraction by removing HTML tags

---

## 💻 CLI Application

### Command Line Tool

```go
package main

import (
    "flag"
    "fmt"
    "os"
    "strings"
)

type CLI struct {
    commands map[string]func([]string)
}

func NewCLI() *CLI {
    cli := &CLI{
        commands: make(map[string]func([]string)),
    }
    
    // Register commands
    cli.RegisterCommand("hello", cli.helloCommand)
    cli.RegisterCommand("calc", cli.calcCommand)
    cli.RegisterCommand("file", cli.fileCommand)
    
    return cli
}

func (cli *CLI) RegisterCommand(name string, handler func([]string)) {
    cli.commands[name] = handler
}

func (cli *CLI) Run(args []string) {
    if len(args) < 1 {
        cli.showHelp()
        return
    }
    
    command := args[0]
    if handler, exists := cli.commands[command]; exists {
        handler(args[1:])
    } else {
        fmt.Printf("Unknown command: %s\n", command)
        cli.showHelp()
    }
}

func (cli *CLI) showHelp() {
    fmt.Println("Available commands:")
    fmt.Println("  hello [name]     - Say hello")
    fmt.Println("  calc <operation> - Calculator")
    fmt.Println("  file <path>      - File operations")
    fmt.Println("  help             - Show this help")
}

func (cli *CLI) helloCommand(args []string) {
    name := "World"
    if len(args) > 0 {
        name = strings.Join(args, " ")
    }
    fmt.Printf("Hello, %s!\n", name)
}

func (cli *CLI) calcCommand(args []string) {
    if len(args) < 3 {
        fmt.Println("Usage: calc <num1> <operation> <num2>")
        fmt.Println("Operations: +, -, *, /")
        return
    }
    
    var num1, num2 float64
    var operation string
    
    if _, err := fmt.Sscanf(args[0], "%f", &num1); err != nil {
        fmt.Printf("Invalid number: %s\n", args[0])
        return
    }
    
    operation = args[1]
    
    if _, err := fmt.Sscanf(args[2], "%f", &num2); err != nil {
        fmt.Printf("Invalid number: %s\n", args[2])
        return
    }
    
    var result float64
    switch operation {
    case "+":
        result = num1 + num2
    case "-":
        result = num1 - num2
    case "*":
        result = num1 * num2
    case "/":
        if num2 == 0 {
            fmt.Println("Division by zero!")
            return
        }
        result = num1 / num2
    default:
        fmt.Printf("Unknown operation: %s\n", operation)
        return
    }
    
    fmt.Printf("%.2f %s %.2f = %.2f\n", num1, operation, num2, result)
}

func (cli *CLI) fileCommand(args []string) {
    if len(args) < 1 {
        fmt.Println("Usage: file <path>")
        return
    }
    
    path := args[0]
    info, err := os.Stat(path)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("File: %s\n", path)
    fmt.Printf("Size: %d bytes\n", info.Size())
    fmt.Printf("Mode: %v\n", info.Mode())
    fmt.Printf("Modified: %v\n", info.ModTime())
}

func main() {
    cli := NewCLI()
    cli.Run(os.Args[1:])
}
```

**Explanation:**
- `flag` package for command-line argument parsing
- Command pattern for handling different operations
- `os.Args` contains command-line arguments
- `fmt.Sscanf()` parses formatted input
- Error handling for invalid inputs

---

## ⚙️ Configuration Management

### Configuration with Environment Variables

```go
package main

import (
    "fmt"
    "os"
    "strconv"
    "time"
)

type Config struct {
    ServerPort    int
    DatabaseURL   string
    RedisURL      string
    LogLevel      string
    Timeout       time.Duration
    MaxConnections int
    Debug         bool
}

func LoadConfig() *Config {
    config := &Config{
        ServerPort:    8080,
        DatabaseURL:   "localhost:5432",
        RedisURL:      "localhost:6379",
        LogLevel:      "info",
        Timeout:       30 * time.Second,
        MaxConnections: 100,
        Debug:         false,
    }
    
    // Load from environment variables
    if port := os.Getenv("SERVER_PORT"); port != "" {
        if p, err := strconv.Atoi(port); err == nil {
            config.ServerPort = p
        }
    }
    
    if dbURL := os.Getenv("DATABASE_URL"); dbURL != "" {
        config.DatabaseURL = dbURL
    }
    
    if redisURL := os.Getenv("REDIS_URL"); redisURL != "" {
        config.RedisURL = redisURL
    }
    
    if logLevel := os.Getenv("LOG_LEVEL"); logLevel != "" {
        config.LogLevel = logLevel
    }
    
    if timeout := os.Getenv("TIMEOUT"); timeout != "" {
        if t, err := time.ParseDuration(timeout); err == nil {
            config.Timeout = t
        }
    }
    
    if maxConn := os.Getenv("MAX_CONNECTIONS"); maxConn != "" {
        if mc, err := strconv.Atoi(maxConn); err == nil {
            config.MaxConnections = mc
        }
    }
    
    if debug := os.Getenv("DEBUG"); debug != "" {
        if d, err := strconv.ParseBool(debug); err == nil {
            config.Debug = d
        }
    }
    
    return config
}

func main() {
    config := LoadConfig()
    
    fmt.Printf("Configuration:\n")
    fmt.Printf("  Server Port: %d\n", config.ServerPort)
    fmt.Printf("  Database URL: %s\n", config.DatabaseURL)
    fmt.Printf("  Redis URL: %s\n", config.RedisURL)
    fmt.Printf("  Log Level: %s\n", config.LogLevel)
    fmt.Printf("  Timeout: %v\n", config.Timeout)
    fmt.Printf("  Max Connections: %d\n", config.MaxConnections)
    fmt.Printf("  Debug: %t\n", config.Debug)
}
```

**Explanation:**
- Environment variables provide configuration
- `os.Getenv()` reads environment variables
- `strconv` package converts strings to other types
- Default values for missing environment variables
- Type conversion with error handling

---

## 📝 Logging

### Structured Logging

```go
package main

import (
    "fmt"
    "log"
    "os"
    "time"
)

type Logger struct {
    infoLogger  *log.Logger
    errorLogger *log.Logger
    debugLogger *log.Logger
}

func NewLogger() *Logger {
    return &Logger{
        infoLogger:  log.New(os.Stdout, "INFO: ", log.LstdFlags|log.Lshortfile),
        errorLogger: log.New(os.Stderr, "ERROR: ", log.LstdFlags|log.Lshortfile),
        debugLogger: log.New(os.Stdout, "DEBUG: ", log.LstdFlags|log.Lshortfile),
    }
}

func (l *Logger) Info(format string, v ...interface{}) {
    l.infoLogger.Printf(format, v...)
}

func (l *Logger) Error(format string, v ...interface{}) {
    l.errorLogger.Printf(format, v...)
}

func (l *Logger) Debug(format string, v ...interface{}) {
    l.debugLogger.Printf(format, v...)
}

func (l *Logger) Fatal(format string, v ...interface{}) {
    l.errorLogger.Fatalf(format, v...)
}

func main() {
    logger := NewLogger()
    
    // Different log levels
    logger.Info("Application started")
    logger.Debug("Debug information: %s", "some debug data")
    logger.Error("An error occurred: %s", "something went wrong")
    
    // Structured logging example
    userID := 123
    action := "login"
    timestamp := time.Now()
    
    logger.Info("User %d performed action '%s' at %v", userID, action, timestamp)
    
    // Log to file
    file, err := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    if err != nil {
        log.Fatal("Failed to open log file:", err)
    }
    defer file.Close()
    
    fileLogger := log.New(file, "APP: ", log.LstdFlags|log.Lshortfile)
    fileLogger.Println("This message goes to the log file")
    
    fmt.Println("Logging examples completed!")
}
```

**Explanation:**
- `log.New()` creates custom loggers
- Different log levels (Info, Error, Debug, Fatal)
- `log.LstdFlags` adds timestamp
- `log.Lshortfile` adds file and line number
- Logging to both console and file

---

## 🧪 Testing

### Unit Testing

```go
package main

import (
    "testing"
)

// Function to test
func Add(a, b int) int {
    return a + b
}

func Multiply(a, b int) int {
    return a * b
}

func Divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Test functions
func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5
    if result != expected {
        t.Errorf("Add(2, 3) = %d; expected %d", result, expected)
    }
}

func TestMultiply(t *testing.T) {
    tests := []struct {
        a, b, expected int
    }{
        {2, 3, 6},
        {0, 5, 0},
        {-2, 3, -6},
        {10, 10, 100},
    }
    
    for _, test := range tests {
        result := Multiply(test.a, test.b)
        if result != test.expected {
            t.Errorf("Multiply(%d, %d) = %d; expected %d", 
                test.a, test.b, result, test.expected)
        }
    }
}

func TestDivide(t *testing.T) {
    // Test normal division
    result, err := Divide(10, 2)
    if err != nil {
        t.Errorf("Divide(10, 2) returned error: %v", err)
    }
    if result != 5.0 {
        t.Errorf("Divide(10, 2) = %f; expected 5.0", result)
    }
    
    // Test division by zero
    _, err = Divide(10, 0)
    if err == nil {
        t.Error("Divide(10, 0) should return error")
    }
}

// Benchmark test
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(2, 3)
    }
}

// Example test
func ExampleAdd() {
    result := Add(2, 3)
    fmt.Println(result)
    // Output: 5
}
```

**Explanation:**
- Test functions start with `Test`
- `testing.T` provides testing utilities
- Table-driven tests for multiple cases
- Benchmark tests start with `Benchmark`
- Example tests start with `Example`
- Run tests with `go test`

---

## 🎯 Summary

These practical examples demonstrate:

1. **File Operations**: Reading, writing, and managing files
2. **HTTP Server**: Building web servers and REST APIs
3. **JSON Handling**: Working with JSON data
4. **Database Operations**: SQLite database operations
5. **Concurrent Programming**: Goroutines, channels, and worker pools
6. **Web Scraping**: Extracting data from web pages
7. **CLI Applications**: Command-line tools
8. **Configuration**: Environment-based configuration
9. **Logging**: Structured logging
10. **Testing**: Unit tests and benchmarks

Each example includes detailed explanations and can be run independently. Practice these examples to build your Go programming skills!

---

**🚀 Keep practicing and building projects to master Go!**
