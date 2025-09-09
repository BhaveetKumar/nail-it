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
    constructor(id, username, email, status = 'offline') {
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
    constructor(id, senderID, recipientID, content, messageType = 'text') {
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
const EventEmitter = require('events');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');

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
                throw new Error('User already exists');
            }
        }
        
        const user = new User(uuidv4(), username, email, 'offline');
        this.users.set(user.id, user);
        
        this.emit('userRegistered', user);
        return user;
    }
    
    // Message Sending
    async sendMessage(senderID, recipientID, content, messageType = 'text') {
        // Validate users exist
        if (!this.users.has(senderID) || !this.users.has(recipientID)) {
            throw new Error('Invalid sender or recipient');
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
        this.emit('messageSent', message);
        
        return message;
    }
    
    // Group Message Sending
    async sendGroupMessage(senderID, groupID, content, messageType = 'text') {
        const group = this.groups.get(groupID);
        if (!group) {
            throw new Error('Group not found');
        }
        
        if (!group.members.includes(senderID)) {
            throw new Error('User not a member of this group');
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
        
        this.emit('groupMessageSent', { message, group });
        return message;
    }
    
    // WebSocket Connection Management
    handleWebSocketConnection(ws, userID) {
        // Validate user
        if (!this.users.has(userID)) {
            ws.close(1008, 'Invalid user');
            return;
        }
        
        // Store connection
        this.connections.set(userID, ws);
        
        // Update user status
        const user = this.users.get(userID);
        user.status = 'online';
        user.lastSeen = new Date();
        
        // Send connection confirmation
        ws.send(JSON.stringify({
            type: 'connected',
            data: { userID, timestamp: new Date() }
        }));
        
        // Handle incoming messages
        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                this.handleWebSocketMessage(userID, message);
            } catch (error) {
                console.error('Invalid WebSocket message:', error);
            }
        });
        
        // Handle disconnection
        ws.on('close', () => {
            this.connections.delete(userID);
            user.status = 'offline';
            user.lastSeen = new Date();
            
            this.emit('userDisconnected', user);
        });
        
        // Handle errors
        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.connections.delete(userID);
        });
        
        this.emit('userConnected', user);
    }
    
    // WebSocket Message Handling
    handleWebSocketMessage(userID, message) {
        switch (message.type) {
            case 'ping':
                const ws = this.connections.get(userID);
                if (ws) {
                    ws.send(JSON.stringify({ type: 'pong', data: 'ok' }));
                }
                break;
                
            case 'send_message':
                this.sendMessage(
                    userID,
                    message.data.recipientID,
                    message.data.content,
                    message.data.messageType
                );
                break;
                
            case 'join_group':
                this.joinGroup(userID, message.data.groupID);
                break;
                
            case 'leave_group':
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
                                type: 'group_message',
                                data: message
                            });
                        }
                    }
                }
            } else {
                // Direct message
                this.sendToUser(message.recipientID, {
                    type: 'direct_message',
                    data: message
                });
            }
            
            message.delivered = true;
        } catch (error) {
            console.error('Message delivery failed:', error);
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
        
        this.emit('groupCreated', group);
        return group;
    }
    
    joinGroup(userID, groupID) {
        const group = this.groups.get(groupID);
        if (!group) {
            throw new Error('Group not found');
        }
        
        if (!group.members.includes(userID)) {
            group.members.push(userID);
            this.emit('userJoinedGroup', { userID, groupID });
        }
    }
    
    leaveGroup(userID, groupID) {
        const group = this.groups.get(groupID);
        if (!group) {
            throw new Error('Group not found');
        }
        
        const index = group.members.indexOf(userID);
        if (index > -1) {
            group.members.splice(index, 1);
            this.emit('userLeftGroup', { userID, groupID });
        }
    }
    
    // Message History
    getMessageHistory(conversationID, limit = 50, offset = 0) {
        const conversationMessages = this.messages.filter(msg => {
            if (msg.groupID) {
                return msg.groupID === conversationID;
            } else {
                return (msg.senderID === conversationID || msg.recipientID === conversationID);
            }
        });
        
        // Sort by timestamp (newest first)
        conversationMessages.sort((a, b) => b.timestamp - a.timestamp);
        
        // Apply pagination
        return conversationMessages.slice(offset, offset + limit);
    }
    
    // Get unread messages count
    getUnreadCount(userID) {
        return this.messages.filter(msg => {
            return (msg.recipientID === userID || 
                   (msg.groupID && this.groups.get(msg.groupID)?.members.includes(userID))) &&
                   !msg.read;
        }).length;
    }
    
    // Mark message as read
    markAsRead(messageID, userID) {
        const message = this.messages.find(msg => msg.id === messageID);
        if (message && (message.recipientID === userID || 
                       (message.groupID && this.groups.get(message.groupID)?.members.includes(userID)))) {
            message.read = true;
            return true;
        }
        return false;
    }
}
```

### Express.js API Implementation

```javascript
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const { MessagingService } = require('./services/MessagingService');

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
        this.app.post('/api/users/register', this.registerUser.bind(this));
        this.app.get('/api/users/:userID/status', this.getUserStatus.bind(this));
        
        // Group routes
        this.app.post('/api/groups', this.createGroup.bind(this));
        this.app.get('/api/groups/:groupID/members', this.getGroupMembers.bind(this));
        this.app.post('/api/groups/:groupID/join', this.joinGroup.bind(this));
        this.app.delete('/api/groups/:groupID/leave', this.leaveGroup.bind(this));
        
        // Message routes
        this.app.post('/api/messages/send', this.sendMessage.bind(this));
        this.app.get('/api/messages/history/:conversationID', this.getMessageHistory.bind(this));
        this.app.get('/api/messages/unread/:userID', this.getUnreadCount.bind(this));
        this.app.put('/api/messages/:messageID/read', this.markAsRead.bind(this));
        
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ 
                status: 'healthy', 
                timestamp: new Date(),
                activeConnections: this.wss.clients.size,
                totalUsers: this.messagingService.users.size
            });
        });
    }
    
    setupWebSocket() {
        this.wss.on('connection', (ws, req) => {
            const url = new URL(req.url, `http://${req.headers.host}`);
            const userID = url.searchParams.get('userID');
            
            if (!userID) {
                ws.close(1008, 'User ID required');
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
                return res.status(400).json({ error: 'Missing required fields' });
            }
            
            const user = this.messagingService.registerUser(username, email, password);
            
            res.status(201).json({
                success: true,
                data: {
                    id: user.id,
                    username: user.username,
                    email: user.email,
                    status: user.status,
                    createdAt: user.createdAt
                }
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
                return res.status(404).json({ error: 'User not found' });
            }
            
            res.json({
                success: true,
                data: {
                    id: user.id,
                    username: user.username,
                    status: user.status,
                    lastSeen: user.lastSeen
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    async createGroup(req, res) {
        try {
            const { name, description, createdBy } = req.body;
            
            if (!name || !createdBy) {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            
            const group = this.messagingService.createGroup(name, description, createdBy);
            
            res.status(201).json({
                success: true,
                data: group
            });
        } catch (error) {
            res.status(400).json({ error: error.message });
        }
    }
    
    async sendMessage(req, res) {
        try {
            const { senderID, recipientID, content, messageType } = req.body;
            
            if (!senderID || !recipientID || !content) {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            
            const message = await this.messagingService.sendMessage(
                senderID, 
                recipientID, 
                content, 
                messageType
            );
            
            res.status(201).json({
                success: true,
                data: message
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
                    total: messages.length
                }
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
                data: { unreadCount: count }
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
                res.json({ success: true, message: 'Message marked as read' });
            } else {
                res.status(404).json({ error: 'Message not found or access denied' });
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
const { MessagingAPI } = require('./MessagingAPI');
const WebSocket = require('ws');

describe('MessagingAPI', () => {
    let api;
    let server;
    
    beforeEach(() => {
        api = new MessagingAPI();
        server = api.server;
    });
    
    afterEach(() => {
        server.close();
    });
    
    describe('User Registration', () => {
        test('should register a new user successfully', async () => {
            const userData = {
                username: 'testuser',
                email: 'test@example.com',
                password: 'password123'
            };
            
            const response = await fetch('http://localhost:3000/api/users/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            });
            
            const result = await response.json();
            
            expect(response.status).toBe(201);
            expect(result.success).toBe(true);
            expect(result.data.username).toBe(userData.username);
            expect(result.data.email).toBe(userData.email);
        });
        
        test('should reject duplicate username', async () => {
            const userData = {
                username: 'testuser',
                email: 'test@example.com',
                password: 'password123'
            };
            
            // Register first user
            await fetch('http://localhost:3000/api/users/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            });
            
            // Try to register duplicate
            const response = await fetch('http://localhost:3000/api/users/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            });
            
            expect(response.status).toBe(400);
        });
    });
    
    describe('Message Sending', () => {
        test('should send message between users', async () => {
            // Register users
            const user1 = await registerUser('user1', 'user1@example.com');
            const user2 = await registerUser('user2', 'user2@example.com');
            
            const messageData = {
                senderID: user1.id,
                recipientID: user2.id,
                content: 'Hello, World!',
                messageType: 'text'
            };
            
            const response = await fetch('http://localhost:3000/api/messages/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(messageData)
            });
            
            const result = await response.json();
            
            expect(response.status).toBe(201);
            expect(result.success).toBe(true);
            expect(result.data.content).toBe(messageData.content);
            expect(result.data.senderID).toBe(user1.id);
            expect(result.data.recipientID).toBe(user2.id);
        });
    });
    
    describe('WebSocket Connection', () => {
        test('should establish WebSocket connection', (done) => {
            const ws = new WebSocket('ws://localhost:3000?userID=testuser');
            
            ws.on('open', () => {
                expect(ws.readyState).toBe(WebSocket.OPEN);
                ws.close();
                done();
            });
            
            ws.on('error', (error) => {
                done(error);
            });
        });
        
        test('should receive messages via WebSocket', (done) => {
            const ws = new WebSocket('ws://localhost:3000?userID=testuser');
            
            ws.on('message', (data) => {
                const message = JSON.parse(data);
                if (message.type === 'direct_message') {
                    expect(message.data.content).toBe('Test message');
                    ws.close();
                    done();
                }
            });
            
            ws.on('open', () => {
                // Send a message to this user
                setTimeout(() => {
                    api.messagingService.sendMessage('sender', 'testuser', 'Test message');
                }, 100);
            });
        });
    });
});

async function registerUser(username, email) {
    const response = await fetch('http://localhost:3000/api/users/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            username,
            email,
            password: 'password123'
        })
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
