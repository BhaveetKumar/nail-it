# 01. Messaging API - Real-time Communication System

## Title & Summary

Design and implement a real-time messaging API using Node.js that supports user-to-user and group messaging with WebSocket connections and message persistence.

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

```javascript
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

```javascript
// User Entity
class User {
  constructor(id, username, email, status = "offline") {
    this.id = id;
    this.username = username;
    this.email = email;
    this.status = status; // 'online', 'offline', 'away'
    this.lastSeen = new Date();
    this.createdAt = new Date();
  }
}

// Message Entity
class Message {
  constructor(id, senderID, recipientID, content, messageType = "text") {
    this.id = id;
    this.senderID = senderID;
    this.recipientID = recipientID;
    this.groupID = null; // For group messages
    this.content = content;
    this.messageType = messageType; // 'text', 'image', 'file'
    this.timestamp = new Date();
    this.delivered = false;
    this.read = false;
  }
}

// Group Entity
class Group {
  constructor(id, name, description, createdBy) {
    this.id = id;
    this.name = name;
    this.description = description;
    this.createdBy = createdBy;
    this.members = [createdBy];
    this.createdAt = new Date();
  }
}

// Conversation Entity
class Conversation {
  constructor(id, type, userIDs, groupID = null) {
    this.id = id;
    this.type = type; // 'direct', 'group'
    this.userIDs = userIDs;
    this.groupID = groupID;
    this.lastMessage = null;
    this.updatedAt = new Date();
  }
}
```

## Approach Overview

### Simple Solution (MVP)

1. In-memory storage with Maps and Arrays
2. Basic WebSocket connection per user
3. Simple message broadcasting
4. No persistence or advanced features

### Production-Ready Design

1. **Modular Architecture**: Separate concerns (auth, messaging, persistence)
2. **Event-Driven**: Use EventEmitter for message routing
3. **Persistence Layer**: Database for message history
4. **Connection Management**: WebSocket pool with heartbeat
5. **Message Queue**: Redis for reliable message delivery
6. **Caching**: User status and recent messages

## Detailed Design

### Modular Decomposition

```javascript
// Project structure
messaging-api/
├── src/
│   ├── controllers/     # HTTP and WebSocket handlers
│   ├── services/        # Business logic
│   ├── models/          # Data structures
│   ├── repositories/    # Data access layer
│   ├── middleware/      # Authentication, validation
│   ├── utils/           # Helper functions
│   └── config/          # Configuration
├── tests/               # Unit and integration tests
└── docs/                # API documentation
```

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const WebSocket = require("ws");
const { v4: uuidv4 } = require("uuid");

class MessagingService extends EventEmitter {
  constructor() {
    super();
    this.users = new Map();
    this.connections = new Map();
    this.groups = new Map();
    this.messages = [];
    this.conversations = new Map();

    // Message processing queue
    this.messageQueue = [];
    this.isProcessing = false;

    // Start message processor
    this.startMessageProcessor();
  }

  // User Management
  registerUser(username, email, password) {
    // Check if user already exists
    for (const user of this.users.values()) {
      if (user.username === username || user.email === email) {
        throw new Error("User already exists");
      }
    }

    const user = new User(uuidv4(), username, email, "offline");
    this.users.set(user.id, user);

    this.emit("userRegistered", user);
    return user;
  }

  // Message Sending
  async sendMessage(senderID, recipientID, content, messageType = "text") {
    // Validate users exist
    if (!this.users.has(senderID) || !this.users.has(recipientID)) {
      throw new Error("Invalid sender or recipient");
    }

    const message = new Message(
      uuidv4(),
      senderID,
      recipientID,
      content,
      messageType
    );

    // Store message
    this.messages.push(message);

    // Add to processing queue
    this.messageQueue.push(message);

    // Emit event for real-time delivery
    this.emit("messageSent", message);

    return message;
  }

  // Group Message Sending
  async sendGroupMessage(senderID, groupID, content, messageType = "text") {
    const group = this.groups.get(groupID);
    if (!group) {
      throw new Error("Group not found");
    }

    if (!group.members.includes(senderID)) {
      throw new Error("User not a member of this group");
    }

    const message = new Message(
      uuidv4(),
      senderID,
      null, // No direct recipient for group messages
      content,
      messageType
    );
    message.groupID = groupID;

    this.messages.push(message);
    this.messageQueue.push(message);

    this.emit("groupMessageSent", { message, group });
    return message;
  }

  // WebSocket Connection Management
  handleWebSocketConnection(ws, userID) {
    // Validate user
    if (!this.users.has(userID)) {
      ws.close(1008, "Invalid user");
      return;
    }

    // Store connection
    this.connections.set(userID, ws);

    // Update user status
    const user = this.users.get(userID);
    user.status = "online";
    user.lastSeen = new Date();

    // Send connection confirmation
    ws.send(
      JSON.stringify({
        type: "connected",
        data: { userID, timestamp: new Date() },
      })
    );

    // Handle incoming messages
    ws.on("message", (data) => {
      try {
        const message = JSON.parse(data);
        this.handleWebSocketMessage(userID, message);
      } catch (error) {
        console.error("Invalid WebSocket message:", error);
      }
    });

    // Handle disconnection
    ws.on("close", () => {
      this.connections.delete(userID);
      user.status = "offline";
      user.lastSeen = new Date();

      this.emit("userDisconnected", user);
    });

    // Handle errors
    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
      this.connections.delete(userID);
    });

    this.emit("userConnected", user);
  }

  // WebSocket Message Handling
  handleWebSocketMessage(userID, message) {
    switch (message.type) {
      case "ping":
        const ws = this.connections.get(userID);
        if (ws) {
          ws.send(JSON.stringify({ type: "pong", data: "ok" }));
        }
        break;

      case "send_message":
        this.sendMessage(
          userID,
          message.data.recipientID,
          message.data.content,
          message.data.messageType
        );
        break;

      case "join_group":
        this.joinGroup(userID, message.data.groupID);
        break;

      case "leave_group":
        this.leaveGroup(userID, message.data.groupID);
        break;
    }
  }

  // Message Processing
  startMessageProcessor() {
    setInterval(() => {
      if (this.messageQueue.length > 0 && !this.isProcessing) {
        this.processMessageQueue();
      }
    }, 100); // Process every 100ms
  }

  async processMessageQueue() {
    this.isProcessing = true;

    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      await this.deliverMessage(message);
    }

    this.isProcessing = false;
  }

  async deliverMessage(message) {
    try {
      if (message.groupID) {
        // Group message
        const group = this.groups.get(message.groupID);
        if (group) {
          for (const memberID of group.members) {
            if (memberID !== message.senderID) {
              this.sendToUser(memberID, {
                type: "group_message",
                data: message,
              });
            }
          }
        }
      } else {
        // Direct message
        this.sendToUser(message.recipientID, {
          type: "direct_message",
          data: message,
        });
      }

      message.delivered = true;
    } catch (error) {
      console.error("Message delivery failed:", error);
    }
  }

  // Send message to specific user
  sendToUser(userID, message) {
    const ws = this.connections.get(userID);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  // Group Management
  createGroup(name, description, createdBy) {
    const group = new Group(uuidv4(), name, description, createdBy);
    this.groups.set(group.id, group);

    this.emit("groupCreated", group);
    return group;
  }

  joinGroup(userID, groupID) {
    const group = this.groups.get(groupID);
    if (!group) {
      throw new Error("Group not found");
    }

    if (!group.members.includes(userID)) {
      group.members.push(userID);
      this.emit("userJoinedGroup", { userID, groupID });
    }
  }

  leaveGroup(userID, groupID) {
    const group = this.groups.get(groupID);
    if (!group) {
      throw new Error("Group not found");
    }

    const index = group.members.indexOf(userID);
    if (index > -1) {
      group.members.splice(index, 1);
      this.emit("userLeftGroup", { userID, groupID });
    }
  }

  // Message History
  getMessageHistory(conversationID, limit = 50, offset = 0) {
    const conversationMessages = this.messages.filter((msg) => {
      if (msg.groupID) {
        return msg.groupID === conversationID;
      } else {
        return (
          msg.senderID === conversationID || msg.recipientID === conversationID
        );
      }
    });

    // Sort by timestamp (newest first)
    conversationMessages.sort((a, b) => b.timestamp - a.timestamp);

    // Apply pagination
    return conversationMessages.slice(offset, offset + limit);
  }

  // Get unread messages count
  getUnreadCount(userID) {
    return this.messages.filter((msg) => {
      return (
        (msg.recipientID === userID ||
          (msg.groupID &&
            this.groups.get(msg.groupID)?.members.includes(userID))) &&
        !msg.read
      );
    }).length;
  }

  // Mark message as read
  markAsRead(messageID, userID) {
    const message = this.messages.find((msg) => msg.id === messageID);
    if (
      message &&
      (message.recipientID === userID ||
        (message.groupID &&
          this.groups.get(message.groupID)?.members.includes(userID)))
    ) {
      message.read = true;
      return true;
    }
    return false;
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const cors = require("cors");
const { MessagingService } = require("./services/MessagingService");

class MessagingAPI {
  constructor() {
    this.app = express();
    this.server = http.createServer(this.app);
    this.wss = new WebSocket.Server({ server: this.server });
    this.messagingService = new MessagingService();

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));

    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
      next();
    });
  }

  setupRoutes() {
    // User routes
    this.app.post("/api/users/register", this.registerUser.bind(this));
    this.app.get("/api/users/:userID/status", this.getUserStatus.bind(this));

    // Group routes
    this.app.post("/api/groups", this.createGroup.bind(this));
    this.app.get(
      "/api/groups/:groupID/members",
      this.getGroupMembers.bind(this)
    );
    this.app.post("/api/groups/:groupID/join", this.joinGroup.bind(this));
    this.app.delete("/api/groups/:groupID/leave", this.leaveGroup.bind(this));

    // Message routes
    this.app.post("/api/messages/send", this.sendMessage.bind(this));
    this.app.get(
      "/api/messages/history/:conversationID",
      this.getMessageHistory.bind(this)
    );
    this.app.get(
      "/api/messages/unread/:userID",
      this.getUnreadCount.bind(this)
    );
    this.app.put("/api/messages/:messageID/read", this.markAsRead.bind(this));

    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        activeConnections: this.wss.clients.size,
        totalUsers: this.messagingService.users.size,
      });
    });
  }

  setupWebSocket() {
    this.wss.on("connection", (ws, req) => {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const userID = url.searchParams.get("userID");

      if (!userID) {
        ws.close(1008, "User ID required");
        return;
      }

      this.messagingService.handleWebSocketConnection(ws, userID);
    });
  }

  // HTTP Handlers
  async registerUser(req, res) {
    try {
      const { username, email, password } = req.body;

      if (!username || !email || !password) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const user = this.messagingService.registerUser(
        username,
        email,
        password
      );

      res.status(201).json({
        success: true,
        data: {
          id: user.id,
          username: user.username,
          email: user.email,
          status: user.status,
          createdAt: user.createdAt,
        },
      });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getUserStatus(req, res) {
    try {
      const { userID } = req.params;
      const user = this.messagingService.users.get(userID);

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      res.json({
        success: true,
        data: {
          id: user.id,
          username: user.username,
          status: user.status,
          lastSeen: user.lastSeen,
        },
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async createGroup(req, res) {
    try {
      const { name, description, createdBy } = req.body;

      if (!name || !createdBy) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const group = this.messagingService.createGroup(
        name,
        description,
        createdBy
      );

      res.status(201).json({
        success: true,
        data: group,
      });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async sendMessage(req, res) {
    try {
      const { senderID, recipientID, content, messageType } = req.body;

      if (!senderID || !recipientID || !content) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const message = await this.messagingService.sendMessage(
        senderID,
        recipientID,
        content,
        messageType
      );

      res.status(201).json({
        success: true,
        data: message,
      });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getMessageHistory(req, res) {
    try {
      const { conversationID } = req.params;
      const { limit = 50, offset = 0 } = req.query;

      const messages = this.messagingService.getMessageHistory(
        conversationID,
        parseInt(limit),
        parseInt(offset)
      );

      res.json({
        success: true,
        data: messages,
        pagination: {
          limit: parseInt(limit),
          offset: parseInt(offset),
          total: messages.length,
        },
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getUnreadCount(req, res) {
    try {
      const { userID } = req.params;
      const count = this.messagingService.getUnreadCount(userID);

      res.json({
        success: true,
        data: { unreadCount: count },
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async markAsRead(req, res) {
    try {
      const { messageID } = req.params;
      const { userID } = req.body;

      const success = this.messagingService.markAsRead(messageID, userID);

      if (success) {
        res.json({ success: true, message: "Message marked as read" });
      } else {
        res.status(404).json({ error: "Message not found or access denied" });
      }
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.server.listen(port, () => {
      console.log(`Messaging API server running on port ${port}`);
      console.log(`WebSocket server running on ws://localhost:${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new MessagingAPI();
  api.start(3000);
}

module.exports = { MessagingAPI };
```

## Testing Implementation

```javascript
const { MessagingAPI } = require("./MessagingAPI");
const WebSocket = require("ws");

describe("MessagingAPI", () => {
  let api;
  let server;

  beforeEach(() => {
    api = new MessagingAPI();
    server = api.server;
  });

  afterEach(() => {
    server.close();
  });

  describe("User Registration", () => {
    test("should register a new user successfully", async () => {
      const userData = {
        username: "testuser",
        email: "test@example.com",
        password: "password123",
      };

      const response = await fetch("http://localhost:3000/api/users/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(userData),
      });

      const result = await response.json();

      expect(response.status).toBe(201);
      expect(result.success).toBe(true);
      expect(result.data.username).toBe(userData.username);
      expect(result.data.email).toBe(userData.email);
    });

    test("should reject duplicate username", async () => {
      const userData = {
        username: "testuser",
        email: "test@example.com",
        password: "password123",
      };

      // Register first user
      await fetch("http://localhost:3000/api/users/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(userData),
      });

      // Try to register duplicate
      const response = await fetch("http://localhost:3000/api/users/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(userData),
      });

      expect(response.status).toBe(400);
    });
  });

  describe("Message Sending", () => {
    test("should send message between users", async () => {
      // Register users
      const user1 = await registerUser("user1", "user1@example.com");
      const user2 = await registerUser("user2", "user2@example.com");

      const messageData = {
        senderID: user1.id,
        recipientID: user2.id,
        content: "Hello, World!",
        messageType: "text",
      };

      const response = await fetch("http://localhost:3000/api/messages/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(messageData),
      });

      const result = await response.json();

      expect(response.status).toBe(201);
      expect(result.success).toBe(true);
      expect(result.data.content).toBe(messageData.content);
      expect(result.data.senderID).toBe(user1.id);
      expect(result.data.recipientID).toBe(user2.id);
    });
  });

  describe("WebSocket Connection", () => {
    test("should establish WebSocket connection", (done) => {
      const ws = new WebSocket("ws://localhost:3000?userID=testuser");

      ws.on("open", () => {
        expect(ws.readyState).toBe(WebSocket.OPEN);
        ws.close();
        done();
      });

      ws.on("error", (error) => {
        done(error);
      });
    });

    test("should receive messages via WebSocket", (done) => {
      const ws = new WebSocket("ws://localhost:3000?userID=testuser");

      ws.on("message", (data) => {
        const message = JSON.parse(data);
        if (message.type === "direct_message") {
          expect(message.data.content).toBe("Test message");
          ws.close();
          done();
        }
      });

      ws.on("open", () => {
        // Send a message to this user
        setTimeout(() => {
          api.messagingService.sendMessage(
            "sender",
            "testuser",
            "Test message"
          );
        }, 100);
      });
    });
  });
});

async function registerUser(username, email) {
  const response = await fetch("http://localhost:3000/api/users/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username,
      email,
      password: "password123",
    }),
  });

  const result = await response.json();
  return result.data;
}
```

## Complexity Analysis

### Time Complexity

- **Register User**: O(1) - Map insertion
- **Send Message**: O(1) - Array append + Map lookup
- **Get Message History**: O(n) - Linear scan through messages
- **WebSocket Message**: O(1) - Direct connection lookup

### Space Complexity

- **User Storage**: O(U) where U is number of users
- **Message Storage**: O(M) where M is number of messages
- **Connection Storage**: O(C) where C is concurrent connections
- **Total**: O(U + M + C)

## Key Features

### Real-time Communication

- WebSocket connections for instant message delivery
- Event-driven architecture for message routing
- Connection management with heartbeat

### Scalability Considerations

- Message queuing for reliable delivery
- Connection pooling and management
- Efficient data structures for fast lookups

### Production Readiness

- Error handling and validation
- Comprehensive testing
- Health monitoring endpoints
- Modular architecture for easy extension

## Extension Ideas

### Advanced Features

1. **Message Encryption**: End-to-end encryption
2. **File Sharing**: Support for file uploads
3. **Message Reactions**: Emoji reactions to messages
4. **Message Threading**: Reply to specific messages
5. **Push Notifications**: Mobile app integration

### Performance Optimizations

1. **Database Integration**: Persistent storage
2. **Redis Caching**: User status and recent messages
3. **Message Batching**: Group multiple messages
4. **CDN Integration**: File delivery optimization

### Enterprise Features

1. **Admin Controls**: User management and moderation
2. **Analytics**: Message delivery metrics
3. **Compliance**: Message retention policies
4. **Multi-tenancy**: Organization-based messaging

## 💬 **Discussion Points**

### **Scalability Considerations**

**Q: How would you handle 1 million concurrent users?**
**A:**

- **Horizontal Scaling**: Deploy multiple server instances behind a load balancer
- **WebSocket Connection Management**: Use Redis for connection state sharing across servers
- **Database Sharding**: Partition users and messages by user ID or geographic region
- **Message Queuing**: Use Redis Streams or Apache Kafka for message queuing and processing
- **CDN Integration**: Use CloudFlare or AWS CloudFront for global message delivery
- **Connection Pooling**: Implement connection pooling for database and Redis connections

**Q: What happens when a server crashes with active WebSocket connections?**
**A:**

- **Connection Recovery**: Clients implement exponential backoff reconnection logic
- **State Synchronization**: Use Redis to store connection state and message delivery status
- **Message Replay**: Implement message acknowledgment and replay mechanism
- **Health Checks**: Load balancer removes unhealthy servers from rotation
- **Graceful Shutdown**: Implement graceful shutdown to notify clients before server restart

### **Data Consistency & Reliability**

**Q: How do you ensure message delivery guarantees?**
**A:**

- **Message Acknowledgments**: Implement ACK/NACK mechanism for message delivery
- **Retry Logic**: Exponential backoff retry for failed message deliveries
- **Message Persistence**: Store messages in database before sending
- **Delivery Status Tracking**: Track message status (sent, delivered, read)
- **Dead Letter Queue**: Handle permanently failed messages
- **Idempotency**: Use message IDs to prevent duplicate processing

**Q: How do you handle message ordering in group chats?**
**A:**

- **Sequence Numbers**: Assign incremental sequence numbers to messages
- **Vector Clocks**: Use vector clocks for distributed message ordering
- **Client-side Ordering**: Sort messages by timestamp and sequence number
- **Conflict Resolution**: Use last-write-wins or custom conflict resolution
- **Message Buffering**: Buffer out-of-order messages until gaps are filled

### **Security & Privacy**

**Q: How do you implement end-to-end encryption?**
**A:**

- **Key Exchange**: Use Diffie-Hellman key exchange for group chats
- **Message Encryption**: Encrypt messages with AES-256 before sending
- **Key Rotation**: Implement periodic key rotation for security
- **Forward Secrecy**: Use ephemeral keys for each message
- **Key Storage**: Store encrypted keys on server, decryption keys on client
- **Metadata Protection**: Minimize metadata exposure

**Q: How do you prevent message spoofing and tampering?**
**A:**

- **Digital Signatures**: Sign messages with sender's private key
- **Message Integrity**: Use HMAC for message integrity verification
- **Authentication**: Verify user identity with JWT tokens
- **Rate Limiting**: Prevent message flooding and spam
- **Content Validation**: Validate message content and format
- **Audit Logging**: Log all message operations for security auditing

## ❓ **Follow-up Questions**

### **System Design Deep Dive**

**Q1: How would you implement message search functionality?**
**A:**

```javascript
// Elasticsearch Integration for Message Search
class MessageSearchService {
  async searchMessages(query, userId, options = {}) {
    const searchQuery = {
      index: "messages",
      body: {
        query: {
          bool: {
            must: [
              { match: { content: query } },
              {
                bool: {
                  should: [
                    { term: { senderId: userId } },
                    { term: { recipientId: userId } },
                    { term: { groupMembers: userId } },
                  ],
                },
              },
            ],
          },
        },
        sort: [{ timestamp: { order: "desc" } }],
        from: options.offset || 0,
        size: options.limit || 20,
      },
    };

    return await this.elasticsearch.search(searchQuery);
  }
}
```

**Q2: How do you handle file sharing in messages?**
**A:**

```javascript
// File Upload and Sharing Implementation
class FileSharingService {
  async uploadFile(file, userId) {
    // Generate unique file ID
    const fileId = this.generateFileId();

    // Upload to cloud storage (AWS S3, Google Cloud Storage)
    const uploadResult = await this.cloudStorage.upload(file, fileId);

    // Store file metadata in database
    const fileMetadata = {
      fileId,
      originalName: file.originalname,
      size: file.size,
      mimeType: file.mimetype,
      uploaderId: userId,
      uploadUrl: uploadResult.url,
      createdAt: new Date(),
    };

    await this.database.files.insert(fileMetadata);

    return fileMetadata;
  }

  async shareFile(fileId, recipientIds, messageId) {
    // Create file share record
    const fileShare = {
      fileId,
      messageId,
      recipientIds,
      sharedAt: new Date(),
    };

    await this.database.fileShares.insert(fileShare);

    // Send file share notification
    this.notificationService.notifyFileShare(recipientIds, fileId);
  }
}
```

**Q3: How do you implement message reactions and threading?**
**A:**

```javascript
// Message Reactions and Threading
class MessageInteractionService {
  async addReaction(messageId, userId, emoji) {
    const reaction = {
      messageId,
      userId,
      emoji,
      createdAt: new Date(),
    };

    // Store reaction in database
    await this.database.reactions.insert(reaction);

    // Broadcast reaction to all connected clients
    this.broadcastReaction(messageId, reaction);

    return reaction;
  }

  async createThread(parentMessageId, replyMessage) {
    const thread = {
      parentMessageId,
      threadId: this.generateThreadId(),
      createdAt: new Date(),
    };

    // Store thread metadata
    await this.database.threads.insert(thread);

    // Send reply message
    const message = await this.sendMessage(replyMessage);

    // Link message to thread
    await this.database.messages.update(message.id, {
      threadId: thread.threadId,
    });

    return { thread, message };
  }
}
```

### **Performance & Optimization**

**Q4: How do you optimize for mobile clients with poor connectivity?**
**A:**

```javascript
// Mobile Optimization Service
class MobileOptimizationService {
  async handleOfflineMessages(userId) {
    // Store messages locally when offline
    const offlineMessages = await this.localStorage.getOfflineMessages(userId);

    // Sync when connection is restored
    for (const message of offlineMessages) {
      try {
        await this.messageService.sendMessage(message);
        await this.localStorage.removeOfflineMessage(message.id);
      } catch (error) {
        // Keep message for retry
        console.error("Failed to sync offline message:", error);
      }
    }
  }

  async optimizeMessageDelivery(message, recipientConnection) {
    // Check connection quality
    const connectionQuality = this.assessConnectionQuality(recipientConnection);

    if (connectionQuality === "poor") {
      // Compress message content
      message.content = await this.compressContent(message.content);

      // Send in smaller chunks
      return this.sendInChunks(message, recipientConnection);
    }

    return this.sendMessage(message, recipientConnection);
  }
}
```

**Q5: How do you implement message read receipts?**
**A:**

```javascript
// Read Receipts Implementation
class ReadReceiptService {
  async markAsRead(messageId, userId) {
    const readReceipt = {
      messageId,
      userId,
      readAt: new Date(),
    };

    // Store read receipt
    await this.database.readReceipts.insert(readReceipt);

    // Update message read status
    await this.database.messages.updateReadStatus(messageId, userId);

    // Notify sender about read receipt
    const message = await this.database.messages.findById(messageId);
    this.notifyReadReceipt(message.senderId, messageId, userId);

    return readReceipt;
  }

  async getReadReceipts(messageId) {
    return await this.database.readReceipts.findByMessageId(messageId);
  }
}
```

### **Advanced Features**

**Q6: How do you implement message encryption for group chats?**
**A:**

```javascript
// End-to-End Encryption for Group Chats
class GroupEncryptionService {
  async createGroupKey(groupId, members) {
    // Generate group encryption key
    const groupKey = this.generateEncryptionKey();

    // Encrypt group key for each member
    const encryptedKeys = {};
    for (const memberId of members) {
      const memberPublicKey = await this.getMemberPublicKey(memberId);
      encryptedKeys[memberId] = await this.encrypt(groupKey, memberPublicKey);
    }

    // Store encrypted keys
    await this.database.groupKeys.insert({
      groupId,
      encryptedKeys,
      createdAt: new Date(),
    });

    return groupKey;
  }

  async encryptMessage(message, groupId) {
    const groupKey = await this.getGroupKey(groupId);
    const encryptedContent = await this.encrypt(message.content, groupKey);

    return {
      ...message,
      content: encryptedContent,
      encrypted: true,
    };
  }
}
```

**Q7: How do you handle message moderation and content filtering?**
**A:**

```javascript
// Message Moderation Service
class MessageModerationService {
  async moderateMessage(message) {
    // Check for inappropriate content
    const contentCheck = await this.contentFilter.check(message.content);

    if (contentCheck.flagged) {
      // Store flagged message for review
      await this.database.flaggedMessages.insert({
        messageId: message.id,
        reason: contentCheck.reason,
        confidence: contentCheck.confidence,
        flaggedAt: new Date(),
      });

      // Notify moderators
      this.notifyModerators(message, contentCheck);

      // Block message if confidence is high
      if (contentCheck.confidence > 0.8) {
        return { blocked: true, reason: contentCheck.reason };
      }
    }

    return { blocked: false };
  }

  async autoModerate(message) {
    // Use AI/ML for content moderation
    const moderationResult = await this.aiModeration.analyze(message.content);

    if (moderationResult.violation) {
      // Apply automatic actions
      switch (moderationResult.severity) {
        case "low":
          return { action: "warn", message: "Content flagged for review" };
        case "medium":
          return {
            action: "block",
            message: "Message blocked due to policy violation",
          };
        case "high":
          return { action: "ban", message: "User temporarily banned" };
      }
    }

    return { action: "allow" };
  }
}
```

## **Follow-up Questions**

### **1. How would you implement message encryption and end-to-end security?**

**Answer:**

```javascript
class MessageEncryption {
  constructor() {
    this.encryptionKey = process.env.ENCRYPTION_KEY;
    this.algorithm = "aes-256-gcm";
  }

  async encryptMessage(message, recipientPublicKey) {
    try {
      // Generate random IV for each message
      const iv = crypto.randomBytes(16);

      // Create cipher
      const cipher = crypto.createCipher(this.algorithm, this.encryptionKey);
      cipher.setAAD(Buffer.from(message.id)); // Additional authenticated data

      // Encrypt message content
      let encrypted = cipher.update(message.content, "utf8", "hex");
      encrypted += cipher.final("hex");

      // Get authentication tag
      const authTag = cipher.getAuthTag();

      // Encrypt metadata separately
      const metadata = {
        timestamp: message.timestamp,
        senderId: message.senderId,
        messageType: message.type,
      };

      const encryptedMetadata = await this.encryptMetadata(
        metadata,
        recipientPublicKey
      );

      return {
        id: message.id,
        encryptedContent: encrypted,
        iv: iv.toString("hex"),
        authTag: authTag.toString("hex"),
        encryptedMetadata: encryptedMetadata,
        algorithm: this.algorithm,
      };
    } catch (error) {
      throw new Error(`Encryption failed: ${error.message}`);
    }
  }

  async decryptMessage(encryptedMessage, recipientPrivateKey) {
    try {
      const decipher = crypto.createDecipher(
        this.algorithm,
        this.encryptionKey
      );

      decipher.setAAD(Buffer.from(encryptedMessage.id));
      decipher.setAuthTag(Buffer.from(encryptedMessage.authTag, "hex"));

      let decrypted = decipher.update(
        encryptedMessage.encryptedContent,
        "hex",
        "utf8"
      );
      decrypted += decipher.final("utf8");

      // Decrypt metadata
      const metadata = await this.decryptMetadata(
        encryptedMessage.encryptedMetadata,
        recipientPrivateKey
      );

      return {
        id: encryptedMessage.id,
        content: decrypted,
        ...metadata,
      };
    } catch (error) {
      throw new Error(`Decryption failed: ${error.message}`);
    }
  }

  async encryptMetadata(metadata, publicKey) {
    const metadataString = JSON.stringify(metadata);
    const encrypted = crypto.publicEncrypt(
      {
        key: publicKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
      },
      Buffer.from(metadataString)
    );

    return encrypted.toString("base64");
  }

  async decryptMetadata(encryptedMetadata, privateKey) {
    const decrypted = crypto.privateDecrypt(
      {
        key: privateKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
      },
      Buffer.from(encryptedMetadata, "base64")
    );

    return JSON.parse(decrypted.toString());
  }
}

class EndToEndSecurity {
  constructor() {
    this.keyExchange = new KeyExchange();
    this.messageEncryption = new MessageEncryption();
    this.forwardSecrecy = new ForwardSecrecy();
  }

  async establishSecureChannel(userId1, userId2) {
    // Generate ephemeral keys for forward secrecy
    const ephemeralKey1 = this.forwardSecrecy.generateEphemeralKey();
    const ephemeralKey2 = this.forwardSecrecy.generateEphemeralKey();

    // Perform key exchange
    const sharedSecret = await this.keyExchange.performKeyExchange(
      ephemeralKey1,
      ephemeralKey2
    );

    // Store session keys
    await this.storeSessionKey(userId1, userId2, sharedSecret);

    return {
      sessionId: uuidv4(),
      establishedAt: new Date(),
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
    };
  }

  async sendSecureMessage(senderId, recipientId, message) {
    // Get session key
    const sessionKey = await this.getSessionKey(senderId, recipientId);

    if (!sessionKey) {
      throw new Error("No secure channel established");
    }

    // Encrypt message
    const encryptedMessage = await this.messageEncryption.encryptMessage(
      message,
      sessionKey
    );

    // Add security headers
    encryptedMessage.securityHeaders = {
      senderId,
      recipientId,
      timestamp: new Date(),
      nonce: crypto.randomBytes(16).toString("hex"),
    };

    return encryptedMessage;
  }
}
```

### **2. How to handle message delivery failures and retry mechanisms?**

**Answer:**

```javascript
class MessageDeliveryManager {
  constructor() {
    this.deliveryQueue = new Map();
    this.retryPolicies = new Map();
    this.dlq = new DeadLetterQueue();
    this.circuitBreaker = new CircuitBreaker();
  }

  async sendMessage(message) {
    const deliveryAttempt = {
      id: uuidv4(),
      messageId: message.id,
      recipientId: message.recipientId,
      attempts: 0,
      maxAttempts: 3,
      nextRetryAt: new Date(),
      status: "pending",
    };

    this.deliveryQueue.set(deliveryAttempt.id, deliveryAttempt);

    try {
      await this.attemptDelivery(deliveryAttempt, message);
    } catch (error) {
      await this.handleDeliveryFailure(deliveryAttempt, error);
    }
  }

  async attemptDelivery(deliveryAttempt, message) {
    // Check circuit breaker
    if (this.circuitBreaker.isOpen(message.recipientId)) {
      throw new Error("Circuit breaker is open");
    }

    deliveryAttempt.attempts++;
    deliveryAttempt.lastAttemptAt = new Date();

    try {
      // Attempt delivery
      const result = await this.deliverToRecipient(message);

      if (result.success) {
        deliveryAttempt.status = "delivered";
        deliveryAttempt.deliveredAt = new Date();
        this.deliveryQueue.delete(deliveryAttempt.id);
        this.circuitBreaker.recordSuccess(message.recipientId);
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.circuitBreaker.recordFailure(message.recipientId);
      throw error;
    }
  }

  async handleDeliveryFailure(deliveryAttempt, error) {
    if (deliveryAttempt.attempts >= deliveryAttempt.maxAttempts) {
      // Move to dead letter queue
      await this.dlq.addMessage(deliveryAttempt, error);
      deliveryAttempt.status = "failed";
      this.deliveryQueue.delete(deliveryAttempt.id);
    } else {
      // Schedule retry
      const retryDelay = this.calculateRetryDelay(deliveryAttempt.attempts);
      deliveryAttempt.nextRetryAt = new Date(Date.now() + retryDelay);
      deliveryAttempt.status = "retrying";

      // Schedule retry
      setTimeout(() => {
        this.retryDelivery(deliveryAttempt);
      }, retryDelay);
    }
  }

  calculateRetryDelay(attemptNumber) {
    // Exponential backoff with jitter
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const jitter = Math.random() * 0.1; // 10% jitter

    const delay = Math.min(
      baseDelay * Math.pow(2, attemptNumber - 1),
      maxDelay
    );

    return delay * (1 + jitter);
  }

  async retryDelivery(deliveryAttempt) {
    const message = await this.getMessage(deliveryAttempt.messageId);
    if (!message) {
      this.deliveryQueue.delete(deliveryAttempt.id);
      return;
    }

    try {
      await this.attemptDelivery(deliveryAttempt, message);
    } catch (error) {
      await this.handleDeliveryFailure(deliveryAttempt, error);
    }
  }
}

class CircuitBreaker {
  constructor() {
    this.states = new Map(); // recipientId -> state
    this.failureCounts = new Map();
    this.lastFailureTimes = new Map();
    this.threshold = 5; // failures before opening
    this.timeout = 60000; // 1 minute
  }

  isOpen(recipientId) {
    const state = this.states.get(recipientId) || "closed";
    return state === "open";
  }

  recordSuccess(recipientId) {
    this.states.set(recipientId, "closed");
    this.failureCounts.set(recipientId, 0);
  }

  recordFailure(recipientId) {
    const count = (this.failureCounts.get(recipientId) || 0) + 1;
    this.failureCounts.set(recipientId, count);
    this.lastFailureTimes.set(recipientId, Date.now());

    if (count >= this.threshold) {
      this.states.set(recipientId, "open");
      // Auto-close after timeout
      setTimeout(() => {
        this.states.set(recipientId, "half-open");
      }, this.timeout);
    }
  }
}
```

### **3. How to implement message reactions and replies?**

**Answer:**

```javascript
class MessageReactions {
  constructor() {
    this.reactions = new Map(); // messageId -> reactions
    this.reactionTypes = ["like", "love", "laugh", "angry", "sad", "wow"];
  }

  async addReaction(messageId, userId, reactionType) {
    if (!this.reactionTypes.includes(reactionType)) {
      throw new Error("Invalid reaction type");
    }

    const messageReactions = this.reactions.get(messageId) || new Map();

    // Remove existing reaction from user
    for (const [type, users] of messageReactions) {
      const userIndex = users.indexOf(userId);
      if (userIndex !== -1) {
        users.splice(userIndex, 1);
        if (users.length === 0) {
          messageReactions.delete(type);
        }
      }
    }

    // Add new reaction
    if (!messageReactions.has(reactionType)) {
      messageReactions.set(reactionType, []);
    }
    messageReactions.get(reactionType).push(userId);

    this.reactions.set(messageId, messageReactions);

    // Notify message sender
    await this.notifyReaction(messageId, userId, reactionType);

    return {
      messageId,
      reactionType,
      userId,
      timestamp: new Date(),
      totalReactions: this.getTotalReactions(messageId),
    };
  }

  async removeReaction(messageId, userId) {
    const messageReactions = this.reactions.get(messageId);
    if (!messageReactions) return;

    let removed = false;
    for (const [type, users] of messageReactions) {
      const userIndex = users.indexOf(userId);
      if (userIndex !== -1) {
        users.splice(userIndex, 1);
        removed = true;
        if (users.length === 0) {
          messageReactions.delete(type);
        }
      }
    }

    if (removed) {
      this.reactions.set(messageId, messageReactions);
    }

    return { messageId, userId, removed };
  }

  getReactions(messageId) {
    const messageReactions = this.reactions.get(messageId) || new Map();
    const result = {};

    for (const [type, users] of messageReactions) {
      result[type] = {
        count: users.length,
        users: users,
      };
    }

    return result;
  }

  getTotalReactions(messageId) {
    const messageReactions = this.reactions.get(messageId) || new Map();
    let total = 0;

    for (const [type, users] of messageReactions) {
      total += users.length;
    }

    return total;
  }
}

class MessageReplies {
  constructor() {
    this.replies = new Map(); // parentMessageId -> replies
    this.threads = new Map(); // messageId -> thread info
  }

  async addReply(parentMessageId, replyMessage) {
    const reply = {
      id: uuidv4(),
      parentMessageId,
      content: replyMessage.content,
      senderId: replyMessage.senderId,
      timestamp: new Date(),
      threadId: this.getThreadId(parentMessageId),
    };

    // Add to replies
    const messageReplies = this.replies.get(parentMessageId) || [];
    messageReplies.push(reply);
    this.replies.set(parentMessageId, messageReplies);

    // Update thread info
    const threadInfo = this.threads.get(parentMessageId) || {
      id: reply.threadId,
      messageCount: 0,
      lastActivity: new Date(),
      participants: new Set(),
    };

    threadInfo.messageCount++;
    threadInfo.lastActivity = new Date();
    threadInfo.participants.add(reply.senderId);
    this.threads.set(parentMessageId, threadInfo);

    // Notify parent message sender
    await this.notifyReply(parentMessageId, reply);

    return reply;
  }

  getReplies(parentMessageId, limit = 50, offset = 0) {
    const messageReplies = this.replies.get(parentMessageId) || [];
    return messageReplies
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(offset, offset + limit);
  }

  getThreadInfo(parentMessageId) {
    return this.threads.get(parentMessageId);
  }

  getThreadId(parentMessageId) {
    const threadInfo = this.threads.get(parentMessageId);
    return threadInfo ? threadInfo.id : uuidv4();
  }
}
```

### **4. How to handle file attachments in messages?**

**Answer:**

```javascript
class MessageAttachments {
  constructor() {
    this.attachments = new Map(); // messageId -> attachments
    this.fileStorage = new FileStorage();
    this.maxFileSize = 10 * 1024 * 1024; // 10MB
    this.allowedTypes = ["image", "video", "audio", "document"];
  }

  async addAttachment(messageId, fileData) {
    // Validate file
    await this.validateFile(fileData);

    // Generate unique file ID
    const fileId = uuidv4();
    const fileName = `${fileId}_${fileData.originalName}`;

    // Store file
    const fileUrl = await this.fileStorage.store(fileName, fileData.buffer);

    const attachment = {
      id: fileId,
      messageId,
      fileName: fileData.originalName,
      fileUrl,
      fileSize: fileData.size,
      mimeType: fileData.mimeType,
      fileType: this.getFileType(fileData.mimeType),
      uploadedAt: new Date(),
      metadata: await this.extractMetadata(fileData),
    };

    // Add to message attachments
    const messageAttachments = this.attachments.get(messageId) || [];
    messageAttachments.push(attachment);
    this.attachments.set(messageId, messageAttachments);

    return attachment;
  }

  async validateFile(fileData) {
    // Check file size
    if (fileData.size > this.maxFileSize) {
      throw new Error("File size exceeds limit");
    }

    // Check file type
    const fileType = this.getFileType(fileData.mimeType);
    if (!this.allowedTypes.includes(fileType)) {
      throw new Error("File type not allowed");
    }

    // Scan for malware
    const scanResult = await this.scanFile(fileData.buffer);
    if (scanResult.threats.length > 0) {
      throw new Error("File contains threats");
    }
  }

  getFileType(mimeType) {
    if (mimeType.startsWith("image/")) return "image";
    if (mimeType.startsWith("video/")) return "video";
    if (mimeType.startsWith("audio/")) return "audio";
    if (mimeType.startsWith("application/")) return "document";
    return "unknown";
  }

  async extractMetadata(fileData) {
    const metadata = {
      size: fileData.size,
      mimeType: fileData.mimeType,
      uploadedAt: new Date(),
    };

    // Extract image metadata
    if (fileData.mimeType.startsWith("image/")) {
      metadata.imageInfo = await this.extractImageMetadata(fileData.buffer);
    }

    // Extract video metadata
    if (fileData.mimeType.startsWith("video/")) {
      metadata.videoInfo = await this.extractVideoMetadata(fileData.buffer);
    }

    return metadata;
  }

  async extractImageMetadata(buffer) {
    // Use sharp or similar library to extract image metadata
    return {
      width: 1920,
      height: 1080,
      format: "jpeg",
      colorSpace: "sRGB",
    };
  }

  async extractVideoMetadata(buffer) {
    // Use ffprobe or similar to extract video metadata
    return {
      duration: 120,
      width: 1920,
      height: 1080,
      format: "mp4",
      bitrate: 5000000,
    };
  }

  async scanFile(buffer) {
    // Integrate with antivirus service
    return {
      threats: [],
      scanTime: new Date(),
      engine: "clamav",
    };
  }

  getAttachments(messageId) {
    return this.attachments.get(messageId) || [];
  }

  async deleteAttachment(messageId, attachmentId) {
    const messageAttachments = this.attachments.get(messageId) || [];
    const attachmentIndex = messageAttachments.findIndex(
      (a) => a.id === attachmentId
    );

    if (attachmentIndex === -1) {
      throw new Error("Attachment not found");
    }

    const attachment = messageAttachments[attachmentIndex];

    // Delete from storage
    await this.fileStorage.delete(attachment.fileUrl);

    // Remove from message
    messageAttachments.splice(attachmentIndex, 1);
    this.attachments.set(messageId, messageAttachments);

    return { deleted: true, attachmentId };
  }
}
```

### **5. How to implement message search and filtering?**

**Answer:**

```javascript
class MessageSearch {
  constructor() {
    this.searchIndex = new Map(); // userId -> search index
    this.fullTextSearch = new FullTextSearch();
    this.filters = new MessageFilters();
  }

  async indexMessage(message) {
    const searchableContent = {
      id: message.id,
      content: message.content,
      senderId: message.senderId,
      recipientId: message.recipientId,
      timestamp: message.timestamp,
      messageType: message.type,
      tags: this.extractTags(message.content),
    };

    // Index for sender
    await this.addToIndex(message.senderId, searchableContent);

    // Index for recipient
    await this.addToIndex(message.recipientId, searchableContent);
  }

  async addToIndex(userId, content) {
    const userIndex = this.searchIndex.get(userId) || new Map();

    // Add to full-text search
    await this.fullTextSearch.index(content.id, content.content);

    // Add to user index
    userIndex.set(content.id, content);
    this.searchIndex.set(userId, userIndex);
  }

  async searchMessages(userId, query, filters = {}) {
    const userIndex = this.searchIndex.get(userId) || new Map();
    let results = Array.from(userIndex.values());

    // Apply text search
    if (query.text) {
      const textResults = await this.fullTextSearch.search(query.text);
      const textResultIds = new Set(textResults.map((r) => r.id));
      results = results.filter((msg) => textResultIds.has(msg.id));
    }

    // Apply filters
    if (filters.senderId) {
      results = results.filter((msg) => msg.senderId === filters.senderId);
    }

    if (filters.recipientId) {
      results = results.filter(
        (msg) => msg.recipientId === filters.recipientId
      );
    }

    if (filters.messageType) {
      results = results.filter(
        (msg) => msg.messageType === filters.messageType
      );
    }

    if (filters.dateRange) {
      results = results.filter(
        (msg) =>
          msg.timestamp >= filters.dateRange.start &&
          msg.timestamp <= filters.dateRange.end
      );
    }

    if (filters.tags) {
      results = results.filter((msg) =>
        filters.tags.some((tag) => msg.tags.includes(tag))
      );
    }

    // Sort results
    results.sort((a, b) => b.timestamp - a.timestamp);

    // Pagination
    const page = filters.page || 1;
    const limit = filters.limit || 20;
    const offset = (page - 1) * limit;

    return {
      results: results.slice(offset, offset + limit),
      total: results.length,
      page,
      limit,
      hasMore: offset + limit < results.length,
    };
  }

  extractTags(content) {
    // Extract hashtags and mentions
    const hashtags = content.match(/#\w+/g) || [];
    const mentions = content.match(/@\w+/g) || [];

    return [
      ...hashtags.map((tag) => tag.substring(1)),
      ...mentions.map((mention) => mention.substring(1)),
    ];
  }

  async getConversationHistory(userId1, userId2, limit = 50) {
    const userIndex = this.searchIndex.get(userId1) || new Map();
    const results = Array.from(userIndex.values())
      .filter(
        (msg) =>
          (msg.senderId === userId1 && msg.recipientId === userId2) ||
          (msg.senderId === userId2 && msg.recipientId === userId1)
      )
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(-limit);

    return results;
  }

  async getMessageStats(userId, timeRange) {
    const userIndex = this.searchIndex.get(userId) || new Map();
    const messages = Array.from(userIndex.values()).filter(
      (msg) =>
        msg.timestamp >= timeRange.start && msg.timestamp <= timeRange.end
    );

    const stats = {
      totalMessages: messages.length,
      sentMessages: messages.filter((msg) => msg.senderId === userId).length,
      receivedMessages: messages.filter((msg) => msg.recipientId === userId)
        .length,
      uniqueContacts: new Set(
        messages.map((msg) =>
          msg.senderId === userId ? msg.recipientId : msg.senderId
        )
      ).size,
      messageTypes: this.groupByType(messages),
      dailyActivity: this.getDailyActivity(messages),
    };

    return stats;
  }

  groupByType(messages) {
    const types = {};
    messages.forEach((msg) => {
      types[msg.messageType] = (types[msg.messageType] || 0) + 1;
    });
    return types;
  }

  getDailyActivity(messages) {
    const daily = {};
    messages.forEach((msg) => {
      const date = msg.timestamp.toISOString().split("T")[0];
      daily[date] = (daily[date] || 0) + 1;
    });
    return daily;
  }
}

class FullTextSearch {
  constructor() {
    this.index = new Map(); // term -> messageIds
    this.stopWords = new Set([
      "the",
      "a",
      "an",
      "and",
      "or",
      "but",
      "in",
      "on",
      "at",
      "to",
      "for",
      "of",
      "with",
      "by",
    ]);
  }

  async index(messageId, content) {
    const terms = this.tokenize(content);

    terms.forEach((term) => {
      if (!this.index.has(term)) {
        this.index.set(term, new Set());
      }
      this.index.get(term).add(messageId);
    });
  }

  async search(query) {
    const terms = this.tokenize(query);
    const results = new Map(); // messageId -> score

    terms.forEach((term) => {
      if (this.index.has(term)) {
        this.index.get(term).forEach((messageId) => {
          results.set(messageId, (results.get(messageId) || 0) + 1);
        });
      }
    });

    // Sort by score
    return Array.from(results.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([messageId, score]) => ({ id: messageId, score }));
  }

  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((term) => term.length > 2 && !this.stopWords.has(term));
  }
}
```

### **6. How to handle 1 trillion users and extreme scale messaging?**

**Answer:**

```javascript
class ExtremeScaleMessagingSystem {
  constructor() {
    this.shardManager = new ShardManager();
    this.regionManager = new RegionManager();
    this.messageRouter = new MessageRouter();
    this.cacheManager = new CacheManager();
    this.compressionEngine = new CompressionEngine();
    this.analyticsEngine = new AnalyticsEngine();
  }

  async sendMessage(message) {
    // 1. Determine user shard
    const userShard = this.shardManager.getShard(message.senderId);
    const recipientShard = this.shardManager.getShard(message.recipientId);

    // 2. Compress message for storage efficiency
    const compressedMessage = await this.compressionEngine.compress(message);

    // 3. Route message based on shard location
    if (userShard === recipientShard) {
      return await this.sendLocalMessage(compressedMessage, userShard);
    } else {
      return await this.sendCrossShardMessage(
        compressedMessage,
        userShard,
        recipientShard
      );
    }
  }

  async sendLocalMessage(message, shardId) {
    const shard = this.shardManager.getShardInstance(shardId);

    // Store in shard-local database
    await shard.storeMessage(message);

    // Update recipient's message queue
    await shard.addToRecipientQueue(message.recipientId, message.id);

    // Real-time delivery if user is online
    const isOnline = await this.checkUserOnlineStatus(message.recipientId);
    if (isOnline) {
      await this.deliverRealtimeMessage(message);
    }

    return { success: true, shardId, local: true };
  }

  async sendCrossShardMessage(message, senderShard, recipientShard) {
    // Use message queue for cross-shard communication
    const routingKey = `shard.${recipientShard}.messages`;

    await this.messageRouter.routeMessage(routingKey, {
      message,
      senderShard,
      recipientShard,
      timestamp: Date.now(),
    });

    return { success: true, crossShard: true, routingKey };
  }
}

class ShardManager {
  constructor() {
    this.totalShards = 1000000; // 1 million shards
    this.shardsPerRegion = 10000; // 10k shards per region
    this.regions = [
      "us-east",
      "us-west",
      "eu-west",
      "ap-south",
      "ap-northeast",
    ];
    this.shardInstances = new Map();
  }

  getShard(userId) {
    // Consistent hashing for shard assignment
    const hash = this.hashUserId(userId);
    return hash % this.totalShards;
  }

  getShardInstance(shardId) {
    if (!this.shardInstances.has(shardId)) {
      const region = this.getRegionForShard(shardId);
      const shard = new ShardInstance(shardId, region);
      this.shardInstances.set(shardId, shard);
    }
    return this.shardInstances.get(shardId);
  }

  getRegionForShard(shardId) {
    const regionIndex = Math.floor(shardId / this.shardsPerRegion);
    return this.regions[regionIndex % this.regions.length];
  }

  hashUserId(userId) {
    // Use xxHash for fast, consistent hashing
    return this.xxHash(userId);
  }

  xxHash(input) {
    // Simplified xxHash implementation
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      hash = ((hash << 5) - hash + input.charCodeAt(i)) & 0xffffffff;
    }
    return Math.abs(hash);
  }
}

class ShardInstance {
  constructor(shardId, region) {
    this.shardId = shardId;
    this.region = region;
    this.database = new ShardDatabase(shardId);
    this.cache = new ShardCache(shardId);
    this.messageQueues = new Map(); // userId -> message queue
  }

  async storeMessage(message) {
    // Store in shard-specific database
    await this.database.insertMessage(message);

    // Update recipient's message queue
    await this.addToRecipientQueue(message.recipientId, message.id);
  }

  async addToRecipientQueue(userId, messageId) {
    if (!this.messageQueues.has(userId)) {
      this.messageQueues.set(userId, []);
    }

    const queue = this.messageQueues.get(userId);
    queue.push({
      messageId,
      timestamp: Date.now(),
      priority: this.calculatePriority(messageId),
    });

    // Keep queue size manageable
    if (queue.length > 1000) {
      queue.splice(0, queue.length - 1000);
    }
  }

  calculatePriority(messageId) {
    // Priority based on message type, sender importance, etc.
    return Math.random(); // Simplified
  }
}

class MessageRouter {
  constructor() {
    this.messageQueues = new Map();
    this.routingRules = new Map();
    this.loadBalancer = new LoadBalancer();
  }

  async routeMessage(routingKey, message) {
    // Get target shard from routing key
    const targetShard = this.extractShardFromKey(routingKey);

    // Route to appropriate shard instance
    const shardInstance = this.shardManager.getShardInstance(targetShard);

    // Process message in target shard
    await shardInstance.processIncomingMessage(message);
  }

  extractShardFromKey(routingKey) {
    const match = routingKey.match(/shard\.(\d+)\.messages/);
    return match ? parseInt(match[1]) : 0;
  }
}

class CompressionEngine {
  constructor() {
    this.compressionAlgorithms = {
      lz4: new LZ4Compressor(),
      zstd: new ZstdCompressor(),
      brotli: new BrotliCompressor(),
    };
    this.defaultAlgorithm = "zstd";
  }

  async compress(message) {
    const algorithm = this.selectAlgorithm(message);
    const compressor = this.compressionAlgorithms[algorithm];

    const compressed = await compressor.compress(JSON.stringify(message));

    return {
      ...message,
      content: compressed,
      compression: {
        algorithm,
        originalSize: JSON.stringify(message).length,
        compressedSize: compressed.length,
        ratio: compressed.length / JSON.stringify(message).length,
      },
    };
  }

  selectAlgorithm(message) {
    // Select compression algorithm based on message characteristics
    const contentSize = JSON.stringify(message).length;

    if (contentSize < 1024) return "lz4"; // Fast for small messages
    if (contentSize < 10240) return "zstd"; // Balanced for medium messages
    return "brotli"; // Best compression for large messages
  }
}

class CacheManager {
  constructor() {
    this.redisCluster = new RedisCluster();
    this.localCache = new Map();
    this.cacheLayers = [
      new L1Cache(), // In-memory cache
      new L2Cache(), // Redis cluster
      new L3Cache(), // Persistent cache
    ];
  }

  async get(key) {
    // Try each cache layer
    for (const layer of this.cacheLayers) {
      const value = await layer.get(key);
      if (value) {
        // Populate higher layers
        await this.populateHigherLayers(key, value);
        return value;
      }
    }

    return null;
  }

  async set(key, value, ttl = 3600) {
    // Set in all cache layers
    const promises = this.cacheLayers.map((layer) =>
      layer.set(key, value, ttl)
    );

    await Promise.all(promises);
  }

  async populateHigherLayers(key, value) {
    // Populate L1 and L2 caches when L3 hit occurs
    await this.cacheLayers[0].set(key, value, 300); // 5 min TTL
    await this.cacheLayers[1].set(key, value, 1800); // 30 min TTL
  }
}

class AnalyticsEngine {
  constructor() {
    this.metricsCollector = new MetricsCollector();
    this.aggregator = new MetricsAggregator();
    this.storage = new TimeSeriesStorage();
  }

  async trackMessage(message) {
    const metrics = {
      timestamp: Date.now(),
      messageId: message.id,
      senderId: message.senderId,
      recipientId: message.recipientId,
      messageType: message.type,
      size: JSON.stringify(message).length,
      shardId: this.shardManager.getShard(message.senderId),
    };

    // Collect metrics
    await this.metricsCollector.collect(metrics);

    // Aggregate for real-time dashboards
    await this.aggregator.aggregate(metrics);

    // Store for historical analysis
    await this.storage.store(metrics);
  }

  async getSystemMetrics() {
    return {
      totalMessages: await this.getTotalMessageCount(),
      activeUsers: await this.getActiveUserCount(),
      messagesPerSecond: await this.getMessagesPerSecond(),
      averageLatency: await this.getAverageLatency(),
      errorRate: await this.getErrorRate(),
      shardDistribution: await this.getShardDistribution(),
    };
  }
}

class DatabaseSharding {
  constructor() {
    this.shardConfigs = new Map();
    this.connectionPools = new Map();
  }

  async getShardConnection(shardId) {
    if (!this.connectionPools.has(shardId)) {
      const config = this.getShardConfig(shardId);
      const pool = new ConnectionPool(config);
      this.connectionPools.set(shardId, pool);
    }

    return this.connectionPools.get(shardId);
  }

  getShardConfig(shardId) {
    return {
      host: `shard-${shardId}.database.internal`,
      port: 5432,
      database: `messages_shard_${shardId}`,
      maxConnections: 100,
      minConnections: 10,
    };
  }
}

class LoadBalancer {
  constructor() {
    this.healthChecker = new HealthChecker();
    this.loadBalancingAlgorithm = "round_robin";
    this.instances = new Map();
  }

  async selectInstance(serviceType) {
    const healthyInstances = await this.healthChecker.getHealthyInstances(
      serviceType
    );

    if (healthyInstances.length === 0) {
      throw new Error("No healthy instances available");
    }

    return this.selectUsingAlgorithm(healthyInstances);
  }

  selectUsingAlgorithm(instances) {
    switch (this.loadBalancingAlgorithm) {
      case "round_robin":
        return this.roundRobin(instances);
      case "least_connections":
        return this.leastConnections(instances);
      case "weighted_round_robin":
        return this.weightedRoundRobin(instances);
      default:
        return instances[0];
    }
  }
}

// Performance optimizations for 1 trillion users
class PerformanceOptimizations {
  constructor() {
    this.connectionPooling = new ConnectionPooling();
    this.batchProcessing = new BatchProcessing();
    this.asyncProcessing = new AsyncProcessing();
    this.memoryOptimization = new MemoryOptimization();
  }

  async optimizeForScale() {
    // 1. Connection pooling
    await this.connectionPooling.optimize();

    // 2. Batch processing
    await this.batchProcessing.configure();

    // 3. Async processing
    await this.asyncProcessing.setup();

    // 4. Memory optimization
    await this.memoryOptimization.optimize();
  }
}

class ConnectionPooling {
  async optimize() {
    // Configure connection pools for each shard
    const poolConfig = {
      min: 10,
      max: 100,
      acquireTimeoutMillis: 30000,
      createTimeoutMillis: 30000,
      destroyTimeoutMillis: 5000,
      idleTimeoutMillis: 30000,
      reapIntervalMillis: 1000,
      createRetryIntervalMillis: 200,
    };

    // Apply to all shards
    for (let i = 0; i < 1000000; i++) {
      await this.configurePool(i, poolConfig);
    }
  }
}

class BatchProcessing {
  async configure() {
    // Configure batch processing for message operations
    this.batchSize = 1000;
    this.flushInterval = 100; // ms
    this.maxWaitTime = 1000; // ms

    // Start batch processors
    this.startMessageBatchProcessor();
    this.startUserBatchProcessor();
    this.startAnalyticsBatchProcessor();
  }
}

class AsyncProcessing {
  async setup() {
    // Set up async processing queues
    this.messageQueue = new Queue("messages", {
      redis: { host: "redis-cluster.internal" },
      defaultJobOptions: {
        removeOnComplete: 100,
        removeOnFail: 50,
        attempts: 3,
        backoff: {
          type: "exponential",
          delay: 2000,
        },
      },
    });

    // Process messages asynchronously
    this.messageQueue.process("send-message", 100, this.processMessage);
  }
}

class MemoryOptimization {
  async optimize() {
    // Optimize memory usage for massive scale
    this.enableMemoryCompression();
    this.configureGarbageCollection();
    this.optimizeDataStructures();
  }

  enableMemoryCompression() {
    // Enable memory compression for large data structures
    process.env.NODE_OPTIONS = "--max-old-space-size=8192 --gc-interval=100";
  }

  configureGarbageCollection() {
    // Configure aggressive garbage collection
    if (global.gc) {
      setInterval(() => {
        global.gc();
      }, 30000); // GC every 30 seconds
    }
  }

  optimizeDataStructures() {
    // Use more memory-efficient data structures
    // Replace Map with WeakMap where appropriate
    // Use typed arrays for numeric data
    // Implement object pooling for frequently created objects
  }
}
```
