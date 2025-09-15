# 13. Event Sourcing - Event-Driven Architecture

## Title & Summary
Design and implement an event sourcing system using Node.js that captures all changes as events, provides event replay capabilities, and supports CQRS pattern with event stores and projections.

## Problem Statement

Build a comprehensive event sourcing system that:

1. **Event Storage**: Store all domain events in an event store
2. **Event Replay**: Replay events to rebuild application state
3. **CQRS Pattern**: Separate command and query responsibilities
4. **Event Projections**: Create read models from events
5. **Event Versioning**: Handle event schema evolution
6. **Snapshot Management**: Optimize replay performance with snapshots

## Requirements & Constraints

### Functional Requirements
- Store and retrieve domain events
- Event replay and state reconstruction
- Command and query separation (CQRS)
- Event projections and read models
- Event versioning and migration
- Snapshot creation and restoration

### Non-Functional Requirements
- **Latency**: < 50ms for event storage
- **Throughput**: 10,000+ events per second
- **Availability**: 99.9% uptime
- **Consistency**: Eventual consistency for projections
- **Scalability**: Support millions of events
- **Durability**: Guaranteed event persistence

## API / Interfaces

### REST Endpoints

```javascript
// Event Management
POST   /api/events
GET    /api/events/{streamId}
GET    /api/events/{streamId}/{fromVersion}
POST   /api/events/{streamId}/replay

// Commands
POST   /api/commands
GET    /api/commands/{commandId}/status

// Queries
GET    /api/queries/{queryId}
GET    /api/projections
GET    /api/projections/{projectionId}

// Snapshots
POST   /api/snapshots
GET    /api/snapshots/{streamId}
```

### Request/Response Examples

```json
// Store Event
POST /api/events
{
  "streamId": "user_123",
  "eventType": "UserCreated",
  "eventData": {
    "userId": "user_123",
    "email": "john@example.com",
    "name": "John Doe"
  },
  "metadata": {
    "correlationId": "cmd_456",
    "causationId": "cmd_789"
  }
}

// Response
{
  "success": true,
  "data": {
    "eventId": "evt_001",
    "streamId": "user_123",
    "version": 1,
    "eventType": "UserCreated",
    "timestamp": "2024-01-15T10:30:00Z",
    "position": 1000
  }
}

// Get Events
GET /api/events/user_123

// Response
{
  "success": true,
  "data": {
    "streamId": "user_123",
    "events": [
      {
        "eventId": "evt_001",
        "version": 1,
        "eventType": "UserCreated",
        "eventData": {
          "userId": "user_123",
          "email": "john@example.com",
          "name": "John Doe"
        },
        "timestamp": "2024-01-15T10:30:00Z"
      }
    ],
    "currentVersion": 1
  }
}
```

## Data Model

### Core Entities

```javascript
// Event Entity
class Event {
  constructor(streamId, eventType, eventData, metadata = {}) {
    this.id = this.generateID();
    this.streamId = streamId;
    this.version = 0; // Will be set when stored
    this.eventType = eventType;
    this.eventData = eventData;
    this.metadata = metadata;
    this.timestamp = new Date();
    this.position = 0; // Global position in event store
  }
}

// Event Stream Entity
class EventStream {
  constructor(streamId) {
    this.streamId = streamId;
    this.events = [];
    this.currentVersion = 0;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Command Entity
class Command {
  constructor(commandType, commandData, metadata = {}) {
    this.id = this.generateID();
    this.commandType = commandType;
    this.commandData = commandData;
    this.metadata = metadata;
    this.status = "pending"; // 'pending', 'processing', 'completed', 'failed'
    this.createdAt = new Date();
    this.processedAt = null;
    this.error = null;
  }
}

// Projection Entity
class Projection {
  constructor(projectionId, name, eventTypes) {
    this.id = projectionId;
    this.name = name;
    this.eventTypes = eventTypes;
    this.lastProcessedPosition = 0;
    this.status = "active"; // 'active', 'paused', 'error'
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Snapshot Entity
class Snapshot {
  constructor(streamId, version, data) {
    this.id = this.generateID();
    this.streamId = streamId;
    this.version = version;
    this.data = data;
    this.createdAt = new Date();
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. Basic event storage in memory
2. Simple event replay
3. No projections or snapshots
4. Single-threaded processing

### Production-Ready Design
1. **Event Store**: Persistent event storage with versioning
2. **CQRS Implementation**: Separate command and query models
3. **Event Projections**: Asynchronous projection processing
4. **Snapshot Management**: Optimized replay with snapshots
5. **Event Versioning**: Schema evolution and migration
6. **High Availability**: Distributed event processing

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const { v4: uuidv4 } = require("uuid");

class EventSourcingService extends EventEmitter {
  constructor() {
    super();
    this.eventStore = new Map();
    this.eventStreams = new Map();
    this.commands = new Map();
    this.projections = new Map();
    this.snapshots = new Map();
    this.globalPosition = 0;
    this.projectionProcessors = new Map();
    
    // Start background tasks
    this.startProjectionProcessing();
    this.startSnapshotCreation();
    this.startCommandProcessing();
  }

  // Event Storage
  async storeEvent(eventData) {
    try {
      const { streamId, eventType, eventData: data, metadata } = eventData;
      
      // Get or create event stream
      let stream = this.eventStreams.get(streamId);
      if (!stream) {
        stream = new EventStream(streamId);
        this.eventStreams.set(streamId, stream);
      }
      
      // Create event
      const event = new Event(streamId, eventType, data, metadata);
      event.version = stream.currentVersion + 1;
      event.position = ++this.globalPosition;
      
      // Store event
      this.eventStore.set(event.id, event);
      stream.events.push(event);
      stream.currentVersion = event.version;
      stream.updatedAt = new Date();
      
      this.emit("eventStored", { event, stream });
      
      // Process projections
      this.processProjections(event);
      
      return event;
      
    } catch (error) {
      console.error("Event storage error:", error);
      throw error;
    }
  }

  // Event Retrieval
  async getEvents(streamId, fromVersion = 0) {
    try {
      const stream = this.eventStreams.get(streamId);
      if (!stream) {
        return { streamId, events: [], currentVersion: 0 };
      }
      
      const events = stream.events.filter(event => event.version > fromVersion);
      
      return {
        streamId,
        events,
        currentVersion: stream.currentVersion
      };
      
    } catch (error) {
      console.error("Event retrieval error:", error);
      throw error;
    }
  }

  // Event Replay
  async replayEvents(streamId, fromVersion = 0, toVersion = null) {
    try {
      const stream = this.eventStreams.get(streamId);
      if (!stream) {
        throw new Error("Stream not found");
      }
      
      let events = stream.events.filter(event => event.version > fromVersion);
      
      if (toVersion !== null) {
        events = events.filter(event => event.version <= toVersion);
      }
      
      // Start from snapshot if available
      const snapshot = this.getLatestSnapshot(streamId, fromVersion);
      let currentState = snapshot ? snapshot.data : {};
      
      // Replay events
      for (const event of events) {
        currentState = this.applyEvent(currentState, event);
      }
      
      this.emit("eventsReplayed", { streamId, events, finalState: currentState });
      
      return {
        streamId,
        events,
        finalState: currentState,
        fromVersion,
        toVersion: toVersion || stream.currentVersion
      };
      
    } catch (error) {
      console.error("Event replay error:", error);
      throw error;
    }
  }

  // Command Processing
  async processCommand(commandData) {
    try {
      const { commandType, commandData: data, metadata } = commandData;
      
      // Create command
      const command = new Command(commandType, data, metadata);
      this.commands.set(command.id, command);
      
      // Process command
      command.status = "processing";
      const result = await this.executeCommand(command);
      
      command.status = "completed";
      command.processedAt = new Date();
      
      this.emit("commandProcessed", { command, result });
      
      return { command, result };
      
    } catch (error) {
      console.error("Command processing error:", error);
      
      if (command) {
        command.status = "failed";
        command.error = error.message;
        command.processedAt = new Date();
      }
      
      throw error;
    }
  }

  async executeCommand(command) {
    // Command handlers would be implemented here
    switch (command.commandType) {
      case "CreateUser":
        return await this.handleCreateUser(command);
      case "UpdateUser":
        return await this.handleUpdateUser(command);
      case "DeleteUser":
        return await this.handleDeleteUser(command);
      default:
        throw new Error(`Unknown command type: ${command.commandType}`);
    }
  }

  async handleCreateUser(command) {
    const { userId, email, name } = command.commandData;
    
    // Store event
    const event = await this.storeEvent({
      streamId: userId,
      eventType: "UserCreated",
      eventData: { userId, email, name },
      metadata: { correlationId: command.id }
    });
    
    return { event, userId };
  }

  async handleUpdateUser(command) {
    const { userId, updates } = command.commandData;
    
    // Store event
    const event = await this.storeEvent({
      streamId: userId,
      eventType: "UserUpdated",
      eventData: { userId, updates },
      metadata: { correlationId: command.id }
    });
    
    return { event, userId };
  }

  async handleDeleteUser(command) {
    const { userId } = command.commandData;
    
    // Store event
    const event = await this.storeEvent({
      streamId: userId,
      eventType: "UserDeleted",
      eventData: { userId },
      metadata: { correlationId: command.id }
    });
    
    return { event, userId };
  }

  // Projection Management
  async createProjection(projectionData) {
    try {
      const { projectionId, name, eventTypes } = projectionData;
      
      const projection = new Projection(projectionId, name, eventTypes);
      this.projections.set(projectionId, projection);
      
      // Start projection processor
      this.startProjectionProcessor(projection);
      
      this.emit("projectionCreated", projection);
      
      return projection;
      
    } catch (error) {
      console.error("Projection creation error:", error);
      throw error;
    }
  }

  processProjections(event) {
    for (const [projectionId, projection] of this.projections) {
      if (projection.eventTypes.includes(event.eventType)) {
        this.emit("projectionEvent", { projection, event });
      }
    }
  }

  startProjectionProcessor(projection) {
    const processor = {
      projection,
      isProcessing: false,
      lastProcessedPosition: 0
    };
    
    this.projectionProcessors.set(projection.id, processor);
  }

  // Snapshot Management
  async createSnapshot(streamId, version, data) {
    try {
      const snapshot = new Snapshot(streamId, version, data);
      this.snapshots.set(snapshot.id, snapshot);
      
      this.emit("snapshotCreated", snapshot);
      
      return snapshot;
      
    } catch (error) {
      console.error("Snapshot creation error:", error);
      throw error;
    }
  }

  getLatestSnapshot(streamId, maxVersion = null) {
    const streamSnapshots = Array.from(this.snapshots.values())
      .filter(snapshot => snapshot.streamId === streamId)
      .sort((a, b) => b.version - a.version);
    
    if (streamSnapshots.length === 0) {
      return null;
    }
    
    if (maxVersion !== null) {
      return streamSnapshots.find(snapshot => snapshot.version <= maxVersion) || null;
    }
    
    return streamSnapshots[0];
  }

  // Background Tasks
  startProjectionProcessing() {
    setInterval(() => {
      this.processProjectionEvents();
    }, 1000); // Process every second
  }

  processProjectionEvents() {
    for (const [projectionId, processor] of this.projectionProcessors) {
      if (processor.isProcessing) continue;
      
      processor.isProcessing = true;
      
      try {
        // Process events for this projection
        const events = this.getEventsForProjection(processor.projection);
        
        for (const event of events) {
          this.processProjectionEvent(processor.projection, event);
          processor.lastProcessedPosition = event.position;
        }
        
      } catch (error) {
        console.error(`Projection processing error for ${projectionId}:`, error);
        processor.projection.status = "error";
      } finally {
        processor.isProcessing = false;
      }
    }
  }

  getEventsForProjection(projection) {
    const events = Array.from(this.eventStore.values())
      .filter(event => 
        projection.eventTypes.includes(event.eventType) &&
        event.position > this.projectionProcessors.get(projection.id)?.lastProcessedPosition
      )
      .sort((a, b) => a.position - b.position);
    
    return events;
  }

  processProjectionEvent(projection, event) {
    // Projection-specific processing would be implemented here
    this.emit("projectionProcessed", { projection, event });
  }

  startSnapshotCreation() {
    setInterval(() => {
      this.createSnapshotsForStreams();
    }, 300000); // Run every 5 minutes
  }

  createSnapshotsForStreams() {
    for (const [streamId, stream] of this.eventStreams) {
      // Create snapshot if stream has many events
      if (stream.events.length > 100) {
        const latestSnapshot = this.getLatestSnapshot(streamId);
        const snapshotVersion = latestSnapshot ? latestSnapshot.version : 0;
        
        // Create snapshot if enough events since last snapshot
        if (stream.currentVersion - snapshotVersion > 50) {
          this.createSnapshotForStream(streamId);
        }
      }
    }
  }

  async createSnapshotForStream(streamId) {
    try {
      const replayResult = await this.replayEvents(streamId, 0);
      
      await this.createSnapshot(streamId, replayResult.toVersion, replayResult.finalState);
      
    } catch (error) {
      console.error(`Snapshot creation error for stream ${streamId}:`, error);
    }
  }

  startCommandProcessing() {
    setInterval(() => {
      this.processPendingCommands();
    }, 1000); // Process every second
  }

  processPendingCommands() {
    const pendingCommands = Array.from(this.commands.values())
      .filter(command => command.status === "pending");
    
    for (const command of pendingCommands) {
      this.processCommand(command).catch(error => {
        console.error(`Command processing error:`, error);
      });
    }
  }

  // Utility Methods
  applyEvent(currentState, event) {
    // Event application logic would be implemented here
    switch (event.eventType) {
      case "UserCreated":
        return {
          ...currentState,
          userId: event.eventData.userId,
          email: event.eventData.email,
          name: event.eventData.name,
          createdAt: event.timestamp
        };
      
      case "UserUpdated":
        return {
          ...currentState,
          ...event.eventData.updates,
          updatedAt: event.timestamp
        };
      
      case "UserDeleted":
        return {
          ...currentState,
          deleted: true,
          deletedAt: event.timestamp
        };
      
      default:
        return currentState;
    }
  }

  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { EventSourcingService } = require("./services/EventSourcingService");

class EventSourcingAPI {
  constructor() {
    this.app = express();
    this.eventSourcing = new EventSourcingService();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupEventHandlers();
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
    // Event management
    this.app.post("/api/events", this.storeEvent.bind(this));
    this.app.get("/api/events/:streamId", this.getEvents.bind(this));
    this.app.get("/api/events/:streamId/:fromVersion", this.getEventsFromVersion.bind(this));
    this.app.post("/api/events/:streamId/replay", this.replayEvents.bind(this));
    
    // Commands
    this.app.post("/api/commands", this.processCommand.bind(this));
    this.app.get("/api/commands/:commandId/status", this.getCommandStatus.bind(this));
    
    // Queries
    this.app.get("/api/queries/:queryId", this.getQuery.bind(this));
    this.app.get("/api/projections", this.getProjections.bind(this));
    this.app.get("/api/projections/:projectionId", this.getProjection.bind(this));
    
    // Snapshots
    this.app.post("/api/snapshots", this.createSnapshot.bind(this));
    this.app.get("/api/snapshots/:streamId", this.getSnapshots.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalEvents: this.eventSourcing.eventStore.size,
        totalStreams: this.eventSourcing.eventStreams.size,
        totalCommands: this.eventSourcing.commands.size
      });
    });
  }

  setupEventHandlers() {
    this.eventSourcing.on("eventStored", ({ event, stream }) => {
      console.log(`Event stored: ${event.eventType} for stream ${stream.streamId}`);
    });
    
    this.eventSourcing.on("commandProcessed", ({ command, result }) => {
      console.log(`Command processed: ${command.commandType} (${command.id})`);
    });
    
    this.eventSourcing.on("projectionCreated", (projection) => {
      console.log(`Projection created: ${projection.name} (${projection.id})`);
    });
  }

  // HTTP Handlers
  async storeEvent(req, res) {
    try {
      const event = await this.eventSourcing.storeEvent(req.body);
      
      res.status(201).json({
        success: true,
        data: event
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getEvents(req, res) {
    try {
      const { streamId } = req.params;
      const { fromVersion = 0 } = req.query;
      
      const result = await this.eventSourcing.getEvents(streamId, parseInt(fromVersion));
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async replayEvents(req, res) {
    try {
      const { streamId } = req.params;
      const { fromVersion = 0, toVersion = null } = req.body;
      
      const result = await this.eventSourcing.replayEvents(
        streamId, 
        parseInt(fromVersion), 
        toVersion ? parseInt(toVersion) : null
      );
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async processCommand(req, res) {
    try {
      const result = await this.eventSourcing.processCommand(req.body);
      
      res.status(201).json({
        success: true,
        data: result
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getCommandStatus(req, res) {
    try {
      const { commandId } = req.params;
      
      const command = this.eventSourcing.commands.get(commandId);
      if (!command) {
        return res.status(404).json({ error: "Command not found" });
      }
      
      res.json({
        success: true,
        data: {
          id: command.id,
          commandType: command.commandType,
          status: command.status,
          createdAt: command.createdAt,
          processedAt: command.processedAt,
          error: command.error
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async createProjection(req, res) {
    try {
      const projection = await this.eventSourcing.createProjection(req.body);
      
      res.status(201).json({
        success: true,
        data: projection
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async createSnapshot(req, res) {
    try {
      const { streamId, version, data } = req.body;
      
      const snapshot = await this.eventSourcing.createSnapshot(streamId, version, data);
      
      res.status(201).json({
        success: true,
        data: snapshot
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Event Sourcing API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new EventSourcingAPI();
  api.start(3000);
}

module.exports = { EventSourcingAPI };
```

## Key Features

### Event Storage & Retrieval
- **Persistent Event Store**: Reliable event storage with versioning
- **Event Replay**: Rebuild application state from events
- **Stream Management**: Organize events by stream ID
- **Global Ordering**: Maintain global event position

### CQRS Implementation
- **Command Processing**: Handle write operations as commands
- **Query Separation**: Separate read models from write models
- **Event Projections**: Create read models from events
- **Asynchronous Processing**: Non-blocking command and projection processing

### Snapshot Management
- **Performance Optimization**: Reduce replay time with snapshots
- **Automatic Creation**: Create snapshots based on event count
- **Snapshot Restoration**: Start replay from latest snapshot
- **Version Management**: Track snapshot versions

## Extension Ideas

### Advanced Features
1. **Event Versioning**: Handle schema evolution and migration
2. **Event Sourcing Patterns**: Saga, Process Manager, and Event Store
3. **Event Replay Optimization**: Parallel replay and caching
4. **Event Archiving**: Long-term storage and retrieval
5. **Event Analytics**: Event patterns and insights

### Enterprise Features
1. **Multi-tenant Support**: Isolated event streams per tenant
2. **Event Encryption**: Secure event storage and transmission
3. **Audit Trail**: Complete event history and compliance
4. **Performance Monitoring**: Event processing metrics
5. **Integration APIs**: Third-party system integration
