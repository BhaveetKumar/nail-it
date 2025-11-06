---
# Auto-generated front matter
Title: 01 Messagingapi
LastUpdated: 2025-11-06T20:45:58.509769
Tags: []
Status: draft
---

# 01. Messaging API - Real-time Communication System

## Title & Summary
Design and implement a real-time messaging API that supports user-to-user and group messaging with WebSocket connections and message persistence.

## Problem Statement

Build a messaging system that allows users to send and receive messages in real-time. The system should support:

1. **User Management**: Register users and manage authentication
2. **Direct Messaging**: Send messages between two users
3. **Group Messaging**: Create groups and send messages to multiple users
4. **Real-time Delivery**: Use WebSocket connections for instant message delivery
5. **Message History**: Store and retrieve message history
6. **Online Status**: Track user online/offline status

## Requirements & Constraints

### Functional Requirements
- User registration and authentication
- Create/join/leave groups
- Send/receive messages in real-time
- Retrieve message history with pagination
- Track user online status
- Message delivery confirmation

### Non-Functional Requirements
- **Latency**: < 100ms for message delivery
- **Consistency**: Eventually consistent for message ordering
- **Memory**: Support 10,000 concurrent users
- **Scalability**: Handle 1M messages per day
- **Reliability**: 99.9% message delivery success rate

## API / Interfaces

### REST Endpoints

```go
// User Management
POST   /api/users/register
POST   /api/users/login
GET    /api/users/{userID}/status

// Group Management
POST   /api/groups
GET    /api/groups/{groupID}/members
POST   /api/groups/{groupID}/join
DELETE /api/groups/{groupID}/leave

// Messaging
POST   /api/messages/send
GET    /api/messages/history/{conversationID}
GET    /api/messages/unread

// WebSocket
WS     /ws?token={authToken}
```

### Request/Response Examples

```json
// Register User
POST /api/users/register
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure123"
}

// Send Message
POST /api/messages/send
{
  "recipientID": "user123",
  "content": "Hello, how are you?",
  "messageType": "text"
}

// WebSocket Message
{
  "type": "message",
  "data": {
    "id": "msg456",
    "senderID": "user123",
    "recipientID": "user789",
    "content": "Hello!",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Data Model

### Core Entities

```go
type User struct {
    ID       string    `json:"id"`
    Username string    `json:"username"`
    Email    string    `json:"email"`
    Status   UserStatus `json:"status"`
    LastSeen time.Time `json:"lastSeen"`
}

type Message struct {
    ID           string      `json:"id"`
    SenderID     string      `json:"senderID"`
    RecipientID  string      `json:"recipientID"`
    GroupID      *string     `json:"groupID,omitempty"`
    Content      string      `json:"content"`
    MessageType  MessageType `json:"messageType"`
    Timestamp    time.Time   `json:"timestamp"`
    Delivered    bool        `json:"delivered"`
    Read         bool        `json:"read"`
}

type Group struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Description string    `json:"description"`
    CreatedBy   string    `json:"createdBy"`
    Members     []string  `json:"members"`
    CreatedAt   time.Time `json:"createdAt"`
}

type Conversation struct {
    ID       string `json:"id"`
    Type     ConversationType `json:"type"`
    UserIDs  []string `json:"userIDs"`
    GroupID  *string  `json:"groupID,omitempty"`
    LastMessage *Message `json:"lastMessage,omitempty"`
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory storage with maps and slices
2. Basic WebSocket connection per user
3. Simple message broadcasting
4. No persistence or advanced features

### Production-Ready Design
1. **Modular Architecture**: Separate concerns (auth, messaging, persistence)
2. **Event-Driven**: Use channels for message routing
3. **Persistence Layer**: Database for message history
4. **Connection Management**: WebSocket pool with heartbeat
5. **Message Queue**: Redis for reliable message delivery
6. **Caching**: User status and recent messages

## Detailed Design

### Modular Decomposition

```go
// Core packages
messaging/
├── auth/           # Authentication and authorization
├── models/         # Data structures
├── storage/        # Persistence layer
├── websocket/      # WebSocket connection management
├── handlers/       # HTTP and WebSocket handlers
└── services/       # Business logic
```

### Concurrency Model

```go
type MessageService struct {
    users        map[string]*User
    connections  map[string]*websocket.Conn
    groups       map[string]*Group
    messages     []Message
    userMutex    sync.RWMutex
    connMutex    sync.RWMutex
    messageChan  chan Message
    broadcastChan chan BroadcastMessage
}

// Goroutines for:
// 1. Message processing
// 2. WebSocket heartbeat
// 3. Connection cleanup
// 4. Message persistence
```

### Persistence Strategy

```go
// In-memory for demo, with clear DB interface
type MessageRepository interface {
    SaveMessage(msg Message) error
    GetMessages(conversationID string, limit, offset int) ([]Message, error)
    GetUnreadCount(userID string) (int, error)
    MarkAsRead(messageID string) error
}

// Database schema (PostgreSQL)
CREATE TABLE messages (
    id VARCHAR(36) PRIMARY KEY,
    sender_id VARCHAR(36) NOT NULL,
    recipient_id VARCHAR(36),
    group_id VARCHAR(36),
    content TEXT NOT NULL,
    message_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    delivered BOOLEAN DEFAULT FALSE,
    read_status BOOLEAN DEFAULT FALSE
);
```

### Error Handling & Validation

```go
type APIError struct {
    Code    int    `json:"code"`
    Message string `json:"message"`
    Details string `json:"details,omitempty"`
}

// Validation rules:
// - Username: 3-20 chars, alphanumeric
// - Message content: 1-1000 chars
// - Group name: 1-50 chars
// - Rate limiting: 100 messages/minute per user
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

    "github.com/gorilla/websocket"
    "github.com/google/uuid"
)

// Types
type UserStatus string
const (
    StatusOnline  UserStatus = "online"
    StatusOffline UserStatus = "offline"
    StatusAway    UserStatus = "away"
)

type MessageType string
const (
    MessageTypeText MessageType = "text"
    MessageTypeImage MessageType = "image"
    MessageTypeFile  MessageType = "file"
)

type ConversationType string
const (
    ConversationTypeDirect ConversationType = "direct"
    ConversationTypeGroup  ConversationType = "group"
)

type User struct {
    ID       string    `json:"id"`
    Username string    `json:"username"`
    Email    string    `json:"email"`
    Status   UserStatus `json:"status"`
    LastSeen time.Time `json:"lastSeen"`
}

type Message struct {
    ID          string      `json:"id"`
    SenderID    string      `json:"senderID"`
    RecipientID string      `json:"recipientID"`
    GroupID     *string     `json:"groupID,omitempty"`
    Content     string      `json:"content"`
    MessageType MessageType `json:"messageType"`
    Timestamp   time.Time   `json:"timestamp"`
    Delivered   bool        `json:"delivered"`
    Read        bool        `json:"read"`
}

type Group struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Description string    `json:"description"`
    CreatedBy   string    `json:"createdBy"`
    Members     []string  `json:"members"`
    CreatedAt   time.Time `json:"createdAt"`
}

type WebSocketMessage struct {
    Type string      `json:"type"`
    Data interface{} `json:"data"`
}

type MessageService struct {
    users       map[string]*User
    connections map[string]*websocket.Conn
    groups      map[string]*Group
    messages    []Message
    userMutex   sync.RWMutex
    connMutex   sync.RWMutex
    groupMutex  sync.RWMutex
    msgMutex    sync.RWMutex
    messageChan chan Message
    upgrader    websocket.Upgrader
}

func NewMessageService() *MessageService {
    return &MessageService{
        users:       make(map[string]*User),
        connections: make(map[string]*websocket.Conn),
        groups:      make(map[string]*Group),
        messages:    make([]Message, 0),
        messageChan: make(chan Message, 1000),
        upgrader: websocket.Upgrader{
            CheckOrigin: func(r *http.Request) bool {
                return true // In production, validate origin
            },
        },
    }
}

func (ms *MessageService) RegisterUser(username, email string) (*User, error) {
    ms.userMutex.Lock()
    defer ms.userMutex.Unlock()

    // Check if user exists
    for _, user := range ms.users {
        if user.Username == username || user.Email == email {
            return nil, fmt.Errorf("user already exists")
        }
    }

    user := &User{
        ID:       uuid.New().String(),
        Username: username,
        Email:    email,
        Status:   StatusOffline,
        LastSeen: time.Now(),
    }

    ms.users[user.ID] = user
    return user, nil
}

func (ms *MessageService) SendMessage(senderID, recipientID, content string) (*Message, error) {
    ms.userMutex.RLock()
    sender, exists := ms.users[senderID]
    ms.userMutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("sender not found")
    }

    ms.userMutex.RLock()
    _, exists = ms.users[recipientID]
    ms.userMutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("recipient not found")
    }

    message := Message{
        ID:          uuid.New().String(),
        SenderID:    senderID,
        RecipientID: recipientID,
        Content:     content,
        MessageType: MessageTypeText,
        Timestamp:   time.Now(),
        Delivered:   false,
        Read:        false,
    }

    // Store message
    ms.msgMutex.Lock()
    ms.messages = append(ms.messages, message)
    ms.msgMutex.Unlock()

    // Send via WebSocket if recipient is online
    ms.connMutex.RLock()
    conn, online := ms.connections[recipientID]
    ms.connMutex.RUnlock()

    if online {
        wsMsg := WebSocketMessage{
            Type: "message",
            Data: message,
        }
        
        if err := conn.WriteJSON(wsMsg); err != nil {
            log.Printf("Failed to send message via WebSocket: %v", err)
        } else {
            message.Delivered = true
        }
    }

    return &message, nil
}

func (ms *MessageService) GetMessageHistory(conversationID string, limit, offset int) ([]Message, error) {
    ms.msgMutex.RLock()
    defer ms.msgMutex.RUnlock()

    var conversationMessages []Message
    for _, msg := range ms.messages {
        if msg.SenderID == conversationID || msg.RecipientID == conversationID {
            conversationMessages = append(conversationMessages, msg)
        }
    }

    // Simple pagination
    start := offset
    end := offset + limit
    if start >= len(conversationMessages) {
        return []Message{}, nil
    }
    if end > len(conversationMessages) {
        end = len(conversationMessages)
    }

    return conversationMessages[start:end], nil
}

func (ms *MessageService) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    conn, err := ms.upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Printf("WebSocket upgrade failed: %v", err)
        return
    }
    defer conn.Close()

    // Get user ID from query params (in production, use JWT)
    userID := r.URL.Query().Get("userID")
    if userID == "" {
        return
    }

    // Register connection
    ms.connMutex.Lock()
    ms.connections[userID] = conn
    ms.connMutex.Unlock()

    // Update user status
    ms.userMutex.Lock()
    if user, exists := ms.users[userID]; exists {
        user.Status = StatusOnline
        user.LastSeen = time.Now()
    }
    ms.userMutex.Unlock()

    // Cleanup on disconnect
    defer func() {
        ms.connMutex.Lock()
        delete(ms.connections, userID)
        ms.connMutex.Unlock()

        ms.userMutex.Lock()
        if user, exists := ms.users[userID]; exists {
            user.Status = StatusOffline
            user.LastSeen = time.Now()
        }
        ms.userMutex.Unlock()
    }()

    // Handle incoming messages
    for {
        var wsMsg WebSocketMessage
        if err := conn.ReadJSON(&wsMsg); err != nil {
            log.Printf("WebSocket read error: %v", err)
            break
        }

        // Process different message types
        switch wsMsg.Type {
        case "ping":
            conn.WriteJSON(WebSocketMessage{Type: "pong", Data: "ok"})
        case "message":
            // Handle incoming message
            if data, ok := wsMsg.Data.(map[string]interface{}); ok {
                content, _ := data["content"].(string)
                recipientID, _ := data["recipientID"].(string)
                ms.SendMessage(userID, recipientID, content)
            }
        }
    }
}

// HTTP Handlers
func (ms *MessageService) RegisterHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req struct {
        Username string `json:"username"`
        Email    string `json:"email"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    user, err := ms.RegisterUser(req.Username, req.Email)
    if err != nil {
        http.Error(w, err.Error(), http.StatusConflict)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (ms *MessageService) SendMessageHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req struct {
        SenderID    string `json:"senderID"`
        RecipientID string `json:"recipientID"`
        Content     string `json:"content"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    message, err := ms.SendMessage(req.SenderID, req.RecipientID, req.Content)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(message)
}

func (ms *MessageService) GetHistoryHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    conversationID := r.URL.Query().Get("conversationID")
    if conversationID == "" {
        http.Error(w, "conversationID required", http.StatusBadRequest)
        return
    }

    messages, err := ms.GetMessageHistory(conversationID, 50, 0)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(messages)
}

func main() {
    service := NewMessageService()

    // HTTP routes
    http.HandleFunc("/api/users/register", service.RegisterHandler)
    http.HandleFunc("/api/messages/send", service.SendMessageHandler)
    http.HandleFunc("/api/messages/history", service.GetHistoryHandler)
    http.HandleFunc("/ws", service.HandleWebSocket)

    log.Println("Messaging API server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Unit Tests

```go
package main

import (
    "testing"
    "time"
)

func TestMessageService_RegisterUser(t *testing.T) {
    service := NewMessageService()

    tests := []struct {
        name     string
        username string
        email    string
        wantErr  bool
    }{
        {
            name:     "valid user",
            username: "testuser",
            email:    "test@example.com",
            wantErr:  false,
        },
        {
            name:     "duplicate username",
            username: "testuser",
            email:    "test2@example.com",
            wantErr:  true,
        },
        {
            name:     "duplicate email",
            username: "testuser2",
            email:    "test@example.com",
            wantErr:  true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            user, err := service.RegisterUser(tt.username, tt.email)
            if (err != nil) != tt.wantErr {
                t.Errorf("RegisterUser() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if !tt.wantErr && user == nil {
                t.Error("RegisterUser() returned nil user")
            }
        })
    }
}

func TestMessageService_SendMessage(t *testing.T) {
    service := NewMessageService()

    // Register users
    sender, _ := service.RegisterUser("sender", "sender@example.com")
    recipient, _ := service.RegisterUser("recipient", "recipient@example.com")

    message, err := service.SendMessage(sender.ID, recipient.ID, "Hello!")
    if err != nil {
        t.Fatalf("SendMessage() error = %v", err)
    }

    if message.SenderID != sender.ID {
        t.Errorf("SendMessage() senderID = %v, want %v", message.SenderID, sender.ID)
    }

    if message.RecipientID != recipient.ID {
        t.Errorf("SendMessage() recipientID = %v, want %v", message.RecipientID, recipient.ID)
    }

    if message.Content != "Hello!" {
        t.Errorf("SendMessage() content = %v, want %v", message.Content, "Hello!")
    }
}

func TestMessageService_GetMessageHistory(t *testing.T) {
    service := NewMessageService()

    // Register users
    user1, _ := service.RegisterUser("user1", "user1@example.com")
    user2, _ := service.RegisterUser("user2", "user2@example.com")

    // Send messages
    service.SendMessage(user1.ID, user2.ID, "Message 1")
    service.SendMessage(user2.ID, user1.ID, "Message 2")
    service.SendMessage(user1.ID, user2.ID, "Message 3")

    // Get history for user1
    messages, err := service.GetMessageHistory(user1.ID, 10, 0)
    if err != nil {
        t.Fatalf("GetMessageHistory() error = %v", err)
    }

    if len(messages) != 3 {
        t.Errorf("GetMessageHistory() returned %d messages, want 3", len(messages))
    }
}
```

## Complexity Analysis

### Time Complexity
- **Register User**: O(1) - Hash map insertion
- **Send Message**: O(1) - Hash map lookup + slice append
- **Get Message History**: O(n) - Linear scan through messages
- **WebSocket Message**: O(1) - Direct connection lookup

### Space Complexity
- **User Storage**: O(U) where U is number of users
- **Message Storage**: O(M) where M is number of messages
- **Connection Storage**: O(C) where C is concurrent connections
- **Total**: O(U + M + C)

## Edge Cases & Validation

### Input Validation
- Empty usernames/emails
- Invalid email format
- Message content length limits
- Duplicate user registration
- Non-existent user references

### Error Scenarios
- WebSocket connection drops
- Database connection failures
- Message delivery failures
- Concurrent user registration
- Invalid message formats

### Boundary Conditions
- Maximum message length (1000 chars)
- Maximum group size (100 members)
- Rate limiting (100 messages/minute)
- Connection timeout (30 seconds)
- Message history pagination limits

## Extension Ideas (Scaling)

### Horizontal Scaling
1. **Load Balancing**: Multiple server instances
2. **Message Sharding**: Partition messages by user ID
3. **Database Scaling**: Read replicas, sharding
4. **Redis Clustering**: Distributed message queue

### Performance Optimization
1. **Message Batching**: Group multiple messages
2. **Connection Pooling**: Reuse WebSocket connections
3. **Caching Layer**: Redis for user status and recent messages
4. **CDN Integration**: File/media message delivery

### Advanced Features
1. **Message Encryption**: End-to-end encryption
2. **Push Notifications**: Mobile app integration
3. **Message Search**: Full-text search capabilities
4. **Analytics**: Message delivery metrics
5. **Bot Integration**: Automated message handling

## 20 Follow-up Questions

### 1. How would you handle message ordering in a distributed system?
**Answer**: Use vector clocks or logical timestamps. Each message gets a timestamp from the sender's local clock, and we resolve conflicts using user ID as tiebreaker. For production, consider using a distributed consensus algorithm like Raft.

### 2. What happens if a user goes offline while receiving messages?
**Answer**: Messages are stored in the database and delivered when the user reconnects. We maintain an "unread count" and send a batch of missed messages on reconnection. Consider implementing push notifications for critical messages.

### 3. How would you implement message encryption?
**Answer**: Use end-to-end encryption with public/private key pairs. Each user has a key pair, and messages are encrypted with the recipient's public key. Store encrypted messages in the database, decrypt on the client side.

### 4. What's your strategy for handling large group messages?
**Answer**: For groups > 100 members, use a fan-out pattern with message queues. Instead of sending to all members directly, publish to a queue and let workers handle delivery. Consider rate limiting per group.

### 5. How do you prevent spam and abuse?
**Answer**: Implement rate limiting (messages per minute), content filtering, and user reporting. Use machine learning for spam detection. Consider implementing user reputation scores and temporary bans.

### 6. What happens if the WebSocket connection is unstable?
**Answer**: Implement exponential backoff for reconnection, heartbeat mechanism to detect dead connections, and fallback to HTTP polling. Store connection state and resume from last known position.

### 7. How would you implement message search?
**Answer**: Use Elasticsearch for full-text search. Index message content with metadata (sender, timestamp, conversation). Implement search filters and pagination. Consider privacy implications for group messages.

### 8. What's your approach to message persistence and cleanup?
**Answer**: Store all messages in database with TTL policies. Implement message archiving for old conversations. Use database partitioning by date for better performance. Consider GDPR compliance for message deletion.

### 9. How do you handle file uploads in messages?
**Answer**: Use a separate file service with signed URLs. Store file metadata in message content, actual files in object storage (S3). Implement file type validation and size limits. Use CDN for delivery.

### 10. What's your strategy for handling message delivery confirmation?
**Answer**: Implement read receipts with timestamps. Use WebSocket acknowledgments for delivery confirmation. Store delivery status in database. Consider implementing "typing indicators" for better UX.

### 11. How would you implement message reactions/emojis?
**Answer**: Store reactions as separate entities linked to messages. Use Redis for real-time reaction updates. Implement reaction limits per user per message. Consider reaction analytics.

### 12. What happens if the database is down?
**Answer**: Implement circuit breaker pattern, fallback to in-memory storage with message queuing. Use database connection pooling and retry logic. Consider implementing message queuing for reliability.

### 13. How do you handle message editing and deletion?
**Answer**: Implement soft deletes with timestamps. For editing, store message versions. Use event sourcing for audit trail. Consider implementing "edit history" for transparency.

### 14. What's your approach to message threading/replies?
**Answer**: Add parent message ID to message structure. Implement thread navigation and reply chains. Use database queries with recursive CTEs for thread traversal. Consider thread notification settings.

### 15. How would you implement message forwarding?
**Answer**: Create new message instances with original message metadata. Implement forwarding permissions and rate limits. Consider implementing "forwarded from" indicators for transparency.

### 16. What's your strategy for handling message translation?
**Answer**: Integrate with translation APIs (Google Translate, AWS Translate). Store original and translated content. Implement language detection and user preferences. Consider caching translated messages.

### 17. How do you handle message scheduling?
**Answer**: Implement a job scheduler (Redis, database-based). Store scheduled messages with delivery time. Use background workers for message delivery. Consider implementing message templates.

### 18. What's your approach to message analytics?
**Answer**: Implement event tracking for message delivery, read rates, response times. Use time-series databases for analytics. Consider implementing A/B testing for message features.

### 19. How would you implement message moderation?
**Answer**: Implement content filtering with keyword detection. Use machine learning for inappropriate content detection. Implement human moderation workflows. Consider implementing message reporting and appeals.

### 20. What's your strategy for handling message backup and recovery?
**Answer**: Implement database backups with point-in-time recovery. Use message replication across regions. Implement disaster recovery procedures. Consider implementing message export for users.

## Evaluation Checklist

### Code Quality (25%)
- [ ] Clean, readable Go code with proper naming
- [ ] Proper error handling and validation
- [ ] Use of Go idioms and best practices
- [ ] Appropriate use of interfaces and structs

### Architecture (25%)
- [ ] Clear separation of concerns
- [ ] Modular design with testable components
- [ ] Proper use of concurrency (goroutines, channels)
- [ ] Scalable data structures and algorithms

### Functionality (25%)
- [ ] All requirements implemented correctly
- [ ] Edge cases handled appropriately
- [ ] WebSocket integration working
- [ ] Message persistence and retrieval

### Testing (15%)
- [ ] Unit tests for core functionality
- [ ] Test coverage for edge cases
- [ ] Integration tests for API endpoints
- [ ] WebSocket connection testing

### Discussion (10%)
- [ ] Clear explanation of design decisions
- [ ] Understanding of trade-offs and alternatives
- [ ] Ability to discuss scaling and extensions
- [ ] Knowledge of related technologies

## Discussion Pointers

### Key Points to Highlight
1. **Concurrency Model**: Explain the use of goroutines for message processing and WebSocket handling
2. **Data Structures**: Justify the choice of maps for O(1) lookups and slices for message storage
3. **Error Handling**: Discuss graceful degradation and user experience during failures
4. **Scalability**: Explain how the design can be extended for higher loads
5. **Security**: Discuss authentication, authorization, and data protection measures

### Trade-offs to Discuss
1. **In-memory vs Database**: Performance vs persistence trade-offs
2. **Synchronous vs Asynchronous**: Real-time delivery vs reliability
3. **Connection Management**: WebSocket vs HTTP polling trade-offs
4. **Message Ordering**: Consistency vs availability trade-offs
5. **Caching Strategy**: Memory usage vs performance benefits

### Extension Scenarios
1. **Multi-region Deployment**: How to handle geographic distribution
2. **Message Queuing**: Integration with external message brokers
3. **Analytics Integration**: Real-time metrics and monitoring
4. **Mobile App Support**: Push notifications and offline sync
5. **Enterprise Features**: Admin controls and compliance requirements
