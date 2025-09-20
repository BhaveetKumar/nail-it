# 🌐 Web Development Comprehensive Guide

## Table of Contents
1. [Frontend Fundamentals](#frontend-fundamentals)
2. [Backend Development](#backend-development)
3. [Full-Stack Integration](#full-stack-integration)
4. [Web Security](#web-security)
5. [Performance Optimization](#performance-optimization)
6. [Modern Web Technologies](#modern-web-technologies)
7. [Testing Strategies](#testing-strategies)
8. [Deployment & DevOps](#deployment--devops)
9. [Go Implementation Examples](#go-implementation-examples)
10. [Interview Questions](#interview-questions)

## Frontend Fundamentals

### HTML5 & CSS3

**Modern HTML Structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Web App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="home">
            <h1>Welcome to Our App</h1>
            <p>Modern web development with Go backend</p>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 Web App. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>
```

**CSS3 with Flexbox and Grid:**
```css
/* Modern CSS with Flexbox and Grid */
.container {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas: 
        "header header header"
        "sidebar main aside"
        "footer footer footer";
    min-height: 100vh;
    gap: 1rem;
}

.header {
    grid-area: header;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
}

.sidebar {
    grid-area: sidebar;
    background: #f8f9fa;
    padding: 1rem;
}

.main {
    grid-area: main;
    padding: 1rem;
}

.aside {
    grid-area: aside;
    background: #e9ecef;
    padding: 1rem;
}

.footer {
    grid-area: footer;
    background: #343a40;
    color: white;
    text-align: center;
    padding: 1rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        grid-template-columns: 1fr;
        grid-template-areas: 
            "header"
            "main"
            "sidebar"
            "aside"
            "footer";
    }
}

/* CSS Variables */
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --danger-color: #dc3545;
    --warning-color: #ffc107;
    --info-color: #17a2b8;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background-color: #0056b3;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
```

### JavaScript ES6+

**Modern JavaScript Features:**
```javascript
// ES6+ Features
class WebApp {
    constructor(name) {
        this.name = name;
        this.data = [];
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadData();
    }
    
    setupEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.bindEvents();
        });
    }
    
    bindEvents() {
        const form = document.getElementById('dataForm');
        form.addEventListener('submit', (e) => this.handleSubmit(e));
    }
    
    async loadData() {
        try {
            const response = await fetch('/api/data');
            this.data = await response.json();
            this.renderData();
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }
    
    async handleSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const data = Object.fromEntries(formData);
        
        try {
            const response = await fetch('/api/data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                this.loadData(); // Reload data
                this.showNotification('Data saved successfully!', 'success');
            } else {
                throw new Error('Failed to save data');
            }
        } catch (error) {
            this.showNotification('Error saving data', 'error');
        }
    }
    
    renderData() {
        const container = document.getElementById('dataContainer');
        container.innerHTML = this.data
            .map(item => this.createDataItem(item))
            .join('');
    }
    
    createDataItem(item) {
        return `
            <div class="data-item">
                <h3>${item.title}</h3>
                <p>${item.description}</p>
                <button onclick="app.deleteItem(${item.id})">Delete</button>
            </div>
        `;
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize the app
const app = new WebApp('My Web App');
```

### React.js Fundamentals

**React Component Example:**
```jsx
import React, { useState, useEffect } from 'react';

const DataTable = ({ data, onDelete, onEdit }) => {
    const [filteredData, setFilteredData] = useState(data);
    const [searchTerm, setSearchTerm] = useState('');
    const [sortField, setSortField] = useState('id');
    const [sortDirection, setSortDirection] = useState('asc');

    useEffect(() => {
        let filtered = data.filter(item =>
            item.name.toLowerCase().includes(searchTerm.toLowerCase())
        );

        filtered.sort((a, b) => {
            const aVal = a[sortField];
            const bVal = b[sortField];
            
            if (sortDirection === 'asc') {
                return aVal > bVal ? 1 : -1;
            } else {
                return aVal < bVal ? 1 : -1;
            }
        });

        setFilteredData(filtered);
    }, [data, searchTerm, sortField, sortDirection]);

    const handleSort = (field) => {
        if (sortField === field) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortDirection('asc');
        }
    };

    return (
        <div className="data-table">
            <div className="table-controls">
                <input
                    type="text"
                    placeholder="Search..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th onClick={() => handleSort('id')}>
                            ID {sortField === 'id' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th onClick={() => handleSort('name')}>
                            Name {sortField === 'name' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th onClick={() => handleSort('email')}>
                            Email {sortField === 'email' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {filteredData.map(item => (
                        <tr key={item.id}>
                            <td>{item.id}</td>
                            <td>{item.name}</td>
                            <td>{item.email}</td>
                            <td>
                                <button onClick={() => onEdit(item)}>Edit</button>
                                <button onClick={() => onDelete(item.id)}>Delete</button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default DataTable;
```

## Backend Development

### RESTful API Design

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
    UpdatedAt time.Time `json:"updated_at"`
}

type UserService struct {
    users map[int]*User
    nextID int
}

func NewUserService() *UserService {
    return &UserService{
        users:  make(map[int]*User),
        nextID: 1,
    }
}

func (us *UserService) CreateUser(user *User) *User {
    user.ID = us.nextID
    user.CreatedAt = time.Now()
    user.UpdatedAt = time.Now()
    us.users[user.ID] = user
    us.nextID++
    return user
}

func (us *UserService) GetUser(id int) (*User, error) {
    user, exists := us.users[id]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    return user, nil
}

func (us *UserService) GetAllUsers() []*User {
    users := make([]*User, 0, len(us.users))
    for _, user := range us.users {
        users = append(users, user)
    }
    return users
}

func (us *UserService) UpdateUser(id int, user *User) error {
    existingUser, exists := us.users[id]
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    user.ID = id
    user.CreatedAt = existingUser.CreatedAt
    user.UpdatedAt = time.Now()
    us.users[id] = user
    return nil
}

func (us *UserService) DeleteUser(id int) error {
    _, exists := us.users[id]
    if !exists {
        return fmt.Errorf("user not found")
    }
    delete(us.users, id)
    return nil
}

type APIHandler struct {
    userService *UserService
}

func NewAPIHandler() *APIHandler {
    return &APIHandler{
        userService: NewUserService(),
    }
}

func (h *APIHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    createdUser := h.userService.CreateUser(&user)
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(createdUser)
}

func (h *APIHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }
    
    user, err := h.userService.GetUser(id)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (h *APIHandler) GetAllUsers(w http.ResponseWriter, r *http.Request) {
    users := h.userService.GetAllUsers()
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(users)
}

func (h *APIHandler) UpdateUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }
    
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    if err := h.userService.UpdateUser(id, &user); err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (h *APIHandler) DeleteUser(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id, err := strconv.Atoi(vars["id"])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }
    
    if err := h.userService.DeleteUser(id); err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }
    
    w.WriteHeader(http.StatusNoContent)
}

func (h *APIHandler) SetupRoutes() *mux.Router {
    router := mux.NewRouter()
    
    // API routes
    api := router.PathPrefix("/api/v1").Subrouter()
    api.HandleFunc("/users", h.CreateUser).Methods("POST")
    api.HandleFunc("/users", h.GetAllUsers).Methods("GET")
    api.HandleFunc("/users/{id}", h.GetUser).Methods("GET")
    api.HandleFunc("/users/{id}", h.UpdateUser).Methods("PUT")
    api.HandleFunc("/users/{id}", h.DeleteUser).Methods("DELETE")
    
    // CORS middleware
    router.Use(func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Access-Control-Allow-Origin", "*")
            w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
            
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    })
    
    return router
}

func main() {
    handler := NewAPIHandler()
    router := handler.SetupRoutes()
    
    fmt.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", router))
}
```

### GraphQL API

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    
    "github.com/graphql-go/graphql"
)

type Product struct {
    ID          int     `json:"id"`
    Name        string  `json:"name"`
    Description string  `json:"description"`
    Price       float64 `json:"price"`
    Category    string  `json:"category"`
}

type ProductService struct {
    products map[int]*Product
    nextID   int
}

func NewProductService() *ProductService {
    return &ProductService{
        products: make(map[int]*Product),
        nextID:   1,
    }
}

func (ps *ProductService) GetProduct(id int) *Product {
    return ps.products[id]
}

func (ps *ProductService) GetAllProducts() []*Product {
    products := make([]*Product, 0, len(ps.products))
    for _, product := range ps.products {
        products = append(products, product)
    }
    return products
}

func (ps *ProductService) CreateProduct(product *Product) *Product {
    product.ID = ps.nextID
    ps.products[product.ID] = product
    ps.nextID++
    return product
}

func (ps *ProductService) UpdateProduct(id int, product *Product) *Product {
    product.ID = id
    ps.products[id] = product
    return product
}

func (ps *ProductService) DeleteProduct(id int) bool {
    if _, exists := ps.products[id]; exists {
        delete(ps.products, id)
        return true
    }
    return false
}

func setupGraphQLSchema(productService *ProductService) *graphql.Schema {
    productType := graphql.NewObject(graphql.ObjectConfig{
        Name: "Product",
        Fields: graphql.Fields{
            "id": &graphql.Field{
                Type: graphql.Int,
            },
            "name": &graphql.Field{
                Type: graphql.String,
            },
            "description": &graphql.Field{
                Type: graphql.String,
            },
            "price": &graphql.Field{
                Type: graphql.Float,
            },
            "category": &graphql.Field{
                Type: graphql.String,
            },
        },
    })
    
    queryType := graphql.NewObject(graphql.ObjectConfig{
        Name: "Query",
        Fields: graphql.Fields{
            "product": &graphql.Field{
                Type: productType,
                Args: graphql.FieldConfigArgument{
                    "id": &graphql.ArgumentConfig{
                        Type: graphql.Int,
                    },
                },
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    id, ok := p.Args["id"].(int)
                    if ok {
                        return productService.GetProduct(id), nil
                    }
                    return nil, nil
                },
            },
            "products": &graphql.Field{
                Type: graphql.NewList(productType),
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    return productService.GetAllProducts(), nil
                },
            },
        },
    })
    
    mutationType := graphql.NewObject(graphql.ObjectConfig{
        Name: "Mutation",
        Fields: graphql.Fields{
            "createProduct": &graphql.Field{
                Type: productType,
                Args: graphql.FieldConfigArgument{
                    "name": &graphql.ArgumentConfig{
                        Type: graphql.NewNonNull(graphql.String),
                    },
                    "description": &graphql.ArgumentConfig{
                        Type: graphql.String,
                    },
                    "price": &graphql.ArgumentConfig{
                        Type: graphql.NewNonNull(graphql.Float),
                    },
                    "category": &graphql.ArgumentConfig{
                        Type: graphql.String,
                    },
                },
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    product := &Product{
                        Name:        p.Args["name"].(string),
                        Description: p.Args["description"].(string),
                        Price:       p.Args["price"].(float64),
                        Category:    p.Args["category"].(string),
                    }
                    return productService.CreateProduct(product), nil
                },
            },
        },
    })
    
    schema, _ := graphql.NewSchema(graphql.SchemaConfig{
        Query:    queryType,
        Mutation: mutationType,
    })
    
    return &schema
}

func graphqlHandler(schema *graphql.Schema) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        var result graphql.Result
        
        if r.Method == "POST" {
            var params struct {
                Query         string                 `json:"query"`
                OperationName string                 `json:"operationName"`
                Variables     map[string]interface{} `json:"variables"`
            }
            
            if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
                http.Error(w, "Invalid JSON", http.StatusBadRequest)
                return
            }
            
            result = graphql.Do(graphql.Params{
                Schema:         *schema,
                RequestString:  params.Query,
                OperationName:  params.OperationName,
                VariableValues: params.Variables,
            })
        } else {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(result)
    }
}

func main() {
    productService := NewProductService()
    schema := setupGraphQLSchema(productService)
    
    http.HandleFunc("/graphql", graphqlHandler(schema))
    
    fmt.Println("GraphQL server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Full-Stack Integration

### WebSocket Real-time Communication

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    
    "github.com/gorilla/websocket"
)

type Client struct {
    conn     *websocket.Conn
    send     chan []byte
    hub      *Hub
    username string
}

type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
}

type Message struct {
    Type     string `json:"type"`
    Username string `json:"username"`
    Content  string `json:"content"`
    Timestamp string `json:"timestamp"`
}

func newHub() *Hub {
    return &Hub{
        clients:    make(map[*Client]bool),
        broadcast:  make(chan []byte),
        register:   make(chan *Client),
        unregister: make(chan *Client),
    }
}

func (h *Hub) run() {
    for {
        select {
        case client := <-h.register:
            h.clients[client] = true
            log.Printf("Client connected: %s", client.username)
            
        case client := <-h.unregister:
            if _, ok := h.clients[client]; ok {
                delete(h.clients, client)
                close(client.send)
                log.Printf("Client disconnected: %s", client.username)
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

func (c *Client) readPump() {
    defer func() {
        c.hub.unregister <- c
        c.conn.Close()
    }()
    
    for {
        var msg Message
        err := c.conn.ReadJSON(&msg)
        if err != nil {
            if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
                log.Printf("error: %v", err)
            }
            break
        }
        
        msg.Username = c.username
        msg.Type = "message"
        
        data, _ := json.Marshal(msg)
        c.hub.broadcast <- data
    }
}

func (c *Client) writePump() {
    defer c.conn.Close()
    
    for {
        select {
        case message, ok := <-c.send:
            if !ok {
                c.conn.WriteMessage(websocket.CloseMessage, []byte{})
                return
            }
            
            c.conn.WriteMessage(websocket.TextMessage, message)
        }
    }
}

var upgrader = websocket.Upgrader{
    CheckOrigin: func(r *http.Request) bool {
        return true
    },
}

func serveWS(hub *Hub, w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println(err)
        return
    }
    
    username := r.URL.Query().Get("username")
    if username == "" {
        username = "Anonymous"
    }
    
    client := &Client{
        conn:     conn,
        send:     make(chan []byte, 256),
        hub:      hub,
        username: username,
    }
    
    client.hub.register <- client
    
    go client.writePump()
    go client.readPump()
}

func main() {
    hub := newHub()
    go hub.run()
    
    http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
        serveWS(hub, w, r)
    })
    
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        http.ServeFile(w, r, "index.html")
    })
    
    fmt.Println("WebSocket server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### Frontend WebSocket Client

```javascript
class ChatApp {
    constructor() {
        this.socket = null;
        this.username = '';
        this.init();
    }
    
    init() {
        this.setupUI();
        this.connectWebSocket();
    }
    
    setupUI() {
        this.username = prompt('Enter your username:') || 'Anonymous';
        
        this.messageContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }
    
    connectWebSocket() {
        this.socket = new WebSocket(`ws://localhost:8080/ws?username=${this.username}`);
        
        this.socket.onopen = () => {
            console.log('Connected to WebSocket');
            this.addSystemMessage('Connected to chat');
        };
        
        this.socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.displayMessage(message);
        };
        
        this.socket.onclose = () => {
            console.log('Disconnected from WebSocket');
            this.addSystemMessage('Disconnected from chat');
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    sendMessage() {
        const content = this.messageInput.value.trim();
        if (content && this.socket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'message',
                username: this.username,
                content: content,
                timestamp: new Date().toISOString()
            };
            
            this.socket.send(JSON.stringify(message));
            this.messageInput.value = '';
        }
    }
    
    displayMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message';
        
        const timestamp = new Date(message.timestamp).toLocaleTimeString();
        
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="username">${message.username}</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${message.content}</div>
        `;
        
        this.messageContainer.appendChild(messageElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    addSystemMessage(text) {
        const messageElement = document.createElement('div');
        messageElement.className = 'system-message';
        messageElement.textContent = text;
        
        this.messageContainer.appendChild(messageElement);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}

// Initialize the chat app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
```

## Web Security

### Authentication & Authorization

```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "log"
    "net/http"
    "strings"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
    "golang.org/x/crypto/bcrypt"
)

type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Email    string `json:"email"`
    Password string `json:"-"`
    Role     string `json:"role"`
}

type Claims struct {
    UserID   int    `json:"user_id"`
    Username string `json:"username"`
    Role     string `json:"role"`
    jwt.RegisteredClaims
}

type AuthService struct {
    users     map[string]*User
    jwtSecret []byte
}

func NewAuthService() *AuthService {
    secret := make([]byte, 32)
    rand.Read(secret)
    
    return &AuthService{
        users:     make(map[string]*User),
        jwtSecret: secret,
    }
}

func (as *AuthService) Register(username, email, password, role string) (*User, error) {
    if _, exists := as.users[username]; exists {
        return nil, fmt.Errorf("username already exists")
    }
    
    hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    if err != nil {
        return nil, err
    }
    
    user := &User{
        ID:       len(as.users) + 1,
        Username: username,
        Email:    email,
        Password: string(hashedPassword),
        Role:     role,
    }
    
    as.users[username] = user
    return user, nil
}

func (as *AuthService) Login(username, password string) (string, error) {
    user, exists := as.users[username]
    if !exists {
        return "", fmt.Errorf("invalid credentials")
    }
    
    err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password))
    if err != nil {
        return "", fmt.Errorf("invalid credentials")
    }
    
    claims := &Claims{
        UserID:   user.ID,
        Username: user.Username,
        Role:     user.Role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    tokenString, err := token.SignedString(as.jwtSecret)
    if err != nil {
        return "", err
    }
    
    return tokenString, nil
}

func (as *AuthService) ValidateToken(tokenString string) (*Claims, error) {
    claims := &Claims{}
    
    token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
        return as.jwtSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if !token.Valid {
        return nil, fmt.Errorf("invalid token")
    }
    
    return claims, nil
}

func (as *AuthService) RequireAuth(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        authHeader := r.Header.Get("Authorization")
        if authHeader == "" {
            http.Error(w, "Authorization header required", http.StatusUnauthorized)
            return
        }
        
        tokenString := strings.TrimPrefix(authHeader, "Bearer ")
        claims, err := as.ValidateToken(tokenString)
        if err != nil {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }
        
        // Add user info to request context
        ctx := context.WithValue(r.Context(), "user", claims)
        next.ServeHTTP(w, r.WithContext(ctx))
    }
}

func (as *AuthService) RequireRole(role string) func(http.HandlerFunc) http.HandlerFunc {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return as.RequireAuth(func(w http.ResponseWriter, r *http.Request) {
            claims := r.Context().Value("user").(*Claims)
            if claims.Role != role {
                http.Error(w, "Insufficient permissions", http.StatusForbidden)
                return
            }
            next.ServeHTTP(w, r)
        })
    }
}

func (as *AuthService) RegisterHandler(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Username string `json:"username"`
        Email    string `json:"email"`
        Password string `json:"password"`
        Role     string `json:"role"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    user, err := as.Register(req.Username, req.Email, req.Password, req.Role)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (as *AuthService) LoginHandler(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Username string `json:"username"`
        Password string `json:"password"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    token, err := as.Login(req.Username, req.Password)
    if err != nil {
        http.Error(w, err.Error(), http.StatusUnauthorized)
        return
    }
    
    response := map[string]string{
        "token": token,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (as *AuthService) ProfileHandler(w http.ResponseWriter, r *http.Request) {
    claims := r.Context().Value("user").(*Claims)
    
    user := as.users[claims.Username]
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func main() {
    authService := NewAuthService()
    
    http.HandleFunc("/register", authService.RegisterHandler)
    http.HandleFunc("/login", authService.LoginHandler)
    http.HandleFunc("/profile", authService.RequireAuth(authService.ProfileHandler))
    http.HandleFunc("/admin", authService.RequireRole("admin")(func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Admin only content"))
    }))
    
    fmt.Println("Auth server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Performance Optimization

### Caching Strategies

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
    
    "github.com/patrickmn/go-cache"
)

type CacheService struct {
    cache *cache.Cache
    mutex sync.RWMutex
}

func NewCacheService() *CacheService {
    return &CacheService{
        cache: cache.New(5*time.Minute, 10*time.Minute),
    }
}

func (cs *CacheService) Get(key string) (interface{}, bool) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    return cs.cache.Get(key)
}

func (cs *CacheService) Set(key string, value interface{}, expiration time.Duration) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.cache.Set(key, value, expiration)
}

func (cs *CacheService) Delete(key string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.cache.Delete(key)
}

type DataService struct {
    cache *CacheService
}

func NewDataService() *DataService {
    return &DataService{
        cache: NewCacheService(),
    }
}

func (ds *DataService) GetData(id string) (map[string]interface{}, error) {
    // Check cache first
    if cached, found := ds.cache.Get("data_" + id); found {
        return cached.(map[string]interface{}), nil
    }
    
    // Simulate expensive database operation
    time.Sleep(100 * time.Millisecond)
    
    data := map[string]interface{}{
        "id":      id,
        "name":    "Sample Data",
        "value":   "Cached value",
        "created": time.Now(),
    }
    
    // Cache the result
    ds.cache.Set("data_"+id, data, 5*time.Minute)
    
    return data, nil
}

func (ds *DataService) InvalidateCache(id string) {
    ds.cache.Delete("data_" + id)
}

func (ds *DataService) GetDataHandler(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    if id == "" {
        http.Error(w, "ID parameter required", http.StatusBadRequest)
        return
    }
    
    data, err := ds.GetData(id)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(data)
}

func (ds *DataService) InvalidateCacheHandler(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    if id == "" {
        http.Error(w, "ID parameter required", http.StatusBadRequest)
        return
    }
    
    ds.InvalidateCache(id)
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("Cache invalidated"))
}

func main() {
    dataService := NewDataService()
    
    http.HandleFunc("/data", dataService.GetDataHandler)
    http.HandleFunc("/invalidate", dataService.InvalidateCacheHandler)
    
    fmt.Println("Cache server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Interview Questions

### Basic Concepts
1. **What is the difference between HTTP and HTTPS?**
2. **Explain the concept of RESTful APIs.**
3. **What are the benefits of using a CDN?**
4. **How do you handle CORS in web applications?**
5. **What is the difference between authentication and authorization?**

### Advanced Topics
1. **How would you implement real-time communication in a web application?**
2. **Explain the concept of progressive web apps (PWAs).**
3. **How do you optimize web application performance?**
4. **What are the security best practices for web applications?**
5. **How would you implement a caching strategy for a web API?**

### System Design
1. **Design a real-time chat application.**
2. **How would you build a scalable e-commerce platform?**
3. **Design a content management system.**
4. **How would you implement a social media feed?**
5. **Design a file upload and sharing system.**

## Conclusion

Web development is a broad field that encompasses:

- **Frontend**: HTML, CSS, JavaScript, React, Vue.js, Angular
- **Backend**: Go, Node.js, Python, Java, C#
- **Databases**: PostgreSQL, MongoDB, Redis
- **Caching**: Redis, Memcached, CDN
- **Security**: Authentication, authorization, encryption
- **Performance**: Optimization, monitoring, scaling
- **DevOps**: Deployment, CI/CD, monitoring

Key skills for web developers:
- Strong understanding of web technologies
- Experience with modern frameworks
- Knowledge of security best practices
- Ability to design scalable systems
- Understanding of performance optimization
- Experience with testing and deployment

This guide provides a comprehensive foundation for web development and preparing for web development interviews.


## Modern Web Technologies

<!-- AUTO-GENERATED ANCHOR: originally referenced as #modern-web-technologies -->

Placeholder content. Please replace with proper section.


## Testing Strategies

<!-- AUTO-GENERATED ANCHOR: originally referenced as #testing-strategies -->

Placeholder content. Please replace with proper section.


## Deployment  Devops

<!-- AUTO-GENERATED ANCHOR: originally referenced as #deployment--devops -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
