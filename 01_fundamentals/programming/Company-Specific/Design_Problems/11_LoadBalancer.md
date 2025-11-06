---
# Auto-generated front matter
Title: 11 Loadbalancer
LastUpdated: 2025-11-06T20:45:58.773243
Tags: []
Status: draft
---

# 11. Load Balancer - Traffic Distribution System

## Title & Summary
Design and implement a load balancer using Node.js that distributes incoming requests across multiple backend servers with health checking, session persistence, and intelligent routing algorithms.

## Problem Statement

Build a comprehensive load balancer that:

1. **Request Distribution**: Route traffic across multiple backend servers
2. **Health Monitoring**: Monitor server health and remove unhealthy instances
3. **Load Balancing Algorithms**: Support multiple routing strategies
4. **Session Persistence**: Maintain user sessions across requests
5. **SSL Termination**: Handle HTTPS traffic and certificate management
6. **Metrics & Monitoring**: Track performance and server statistics

## Requirements & Constraints

### Functional Requirements
- Multiple load balancing algorithms (round-robin, least connections, weighted)
- Health checking and server monitoring
- Session persistence and sticky sessions
- SSL/TLS termination
- Request routing and forwarding
- Server pool management
- Metrics collection and reporting

### Non-Functional Requirements
- **Latency**: < 10ms overhead for request routing
- **Throughput**: 100,000+ requests per second
- **Availability**: 99.99% uptime
- **Scalability**: Support 1000+ backend servers
- **Reliability**: Graceful handling of server failures
- **Security**: Secure request forwarding and SSL handling

## API / Interfaces

### REST Endpoints

```javascript
// Load Balancer Management
GET    /api/health
GET    /api/servers
POST   /api/servers
DELETE /api/servers/{serverId}
PUT    /api/servers/{serverId}/weight
GET    /api/servers/{serverId}/health

// Configuration
GET    /api/config
PUT    /api/config
GET    /api/algorithms
PUT    /api/algorithm

// Metrics
GET    /api/metrics
GET    /api/metrics/servers
GET    /api/metrics/requests
```

### Request/Response Examples

```json
// Add Server
POST /api/servers
{
  "host": "192.168.1.100",
  "port": 8080,
  "weight": 10,
  "healthCheckPath": "/health",
  "healthCheckInterval": 30
}

// Response
{
  "success": true,
  "data": {
    "serverId": "server_123",
    "host": "192.168.1.100",
    "port": 8080,
    "weight": 10,
    "status": "healthy",
    "addedAt": "2024-01-15T10:30:00Z"
  }
}

// Load Balancer Metrics
{
  "success": true,
  "data": {
    "totalRequests": 1500000,
    "requestsPerSecond": 2500,
    "averageResponseTime": 45,
    "activeServers": 5,
    "unhealthyServers": 0,
    "algorithm": "round_robin",
    "uptime": "7d 12h 30m"
  }
}
```

## Data Model

### Core Entities

```javascript
// Server Entity
class Server {
  constructor(host, port, weight = 1) {
    this.id = this.generateID();
    this.host = host;
    this.port = port;
    this.weight = weight;
    this.status = "unknown"; // 'healthy', 'unhealthy', 'unknown'
    this.activeConnections = 0;
    this.totalRequests = 0;
    this.responseTime = 0;
    this.lastHealthCheck = null;
    this.healthCheckPath = "/health";
    this.healthCheckInterval = 30; // seconds
    this.addedAt = new Date();
    this.lastSeen = new Date();
  }
}

// Request Entity
class Request {
  constructor(method, url, headers, body) {
    this.id = this.generateID();
    this.method = method;
    this.url = url;
    this.headers = headers;
    this.body = body;
    this.timestamp = new Date();
    this.serverId = null;
    this.responseTime = null;
    this.statusCode = null;
  }
}

// Session Entity
class Session {
  constructor(sessionId, serverId) {
    this.sessionId = sessionId;
    this.serverId = serverId;
    this.createdAt = new Date();
    this.lastAccessed = new Date();
    this.expiresAt = new Date(Date.now() + 3600000); // 1 hour
  }
}

// Load Balancer Config Entity
class LoadBalancerConfig {
  constructor() {
    this.algorithm = "round_robin"; // 'round_robin', 'least_connections', 'weighted', 'ip_hash'
    this.healthCheckEnabled = true;
    this.healthCheckInterval = 30;
    this.healthCheckTimeout = 5;
    this.maxRetries = 3;
    this.sessionPersistence = false;
    this.sslTermination = false;
    this.sslCertPath = "";
    this.sslKeyPath = "";
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. Basic round-robin routing
2. Simple health checking
3. No session persistence
4. In-memory server management

### Production-Ready Design
1. **Multiple Algorithms**: Support various load balancing strategies
2. **Health Monitoring**: Comprehensive server health checking
3. **Session Persistence**: Sticky sessions and session management
4. **SSL Termination**: HTTPS handling and certificate management
5. **Metrics Collection**: Performance monitoring and analytics
6. **High Availability**: Failover and redundancy

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const http = require("http");
const https = require("https");
const { v4: uuidv4 } = require("uuid");

class LoadBalancerService extends EventEmitter {
  constructor() {
    super();
    this.servers = new Map();
    this.sessions = new Map();
    this.config = new LoadBalancerConfig();
    this.metrics = {
      totalRequests: 0,
      requestsPerSecond: 0,
      averageResponseTime: 0,
      activeConnections: 0,
      startTime: new Date()
    };
    this.healthCheckInterval = null;
    this.requestCounter = 0;
    
    // Start background tasks
    this.startHealthChecking();
    this.startMetricsCollection();
    this.startSessionCleanup();
  }

  // Server Management
  async addServer(serverData) {
    try {
      const server = new Server(
        serverData.host,
        serverData.port,
        serverData.weight || 1
      );
      
      // Set additional properties
      if (serverData.healthCheckPath) server.healthCheckPath = serverData.healthCheckPath;
      if (serverData.healthCheckInterval) server.healthCheckInterval = serverData.healthCheckInterval;
      
      // Store server
      this.servers.set(server.id, server);
      
      // Perform initial health check
      await this.performHealthCheck(server);
      
      this.emit("serverAdded", server);
      
      return server;
      
    } catch (error) {
      console.error("Add server error:", error);
      throw error;
    }
  }

  async removeServer(serverId) {
    try {
      const server = this.servers.get(serverId);
      if (!server) {
        throw new Error("Server not found");
      }
      
      // Remove server
      this.servers.delete(serverId);
      
      // Clean up sessions
      this.cleanupServerSessions(serverId);
      
      this.emit("serverRemoved", server);
      
      return true;
      
    } catch (error) {
      console.error("Remove server error:", error);
      throw error;
    }
  }

  // Load Balancing
  async routeRequest(req, res) {
    try {
      const request = new Request(req.method, req.url, req.headers, req.body);
      this.metrics.totalRequests++;
      this.requestCounter++;
      
      // Select server based on algorithm
      const server = this.selectServer(request);
      
      if (!server) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "No healthy servers available" }));
        return;
      }
      
      // Handle session persistence
      if (this.config.sessionPersistence) {
        this.handleSessionPersistence(request, server);
      }
      
      // Forward request to selected server
      await this.forwardRequest(request, server, req, res);
      
    } catch (error) {
      console.error("Request routing error:", error);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Internal server error" }));
    }
  }

  selectServer(request) {
    const healthyServers = Array.from(this.servers.values())
      .filter(server => server.status === "healthy");
    
    if (healthyServers.length === 0) {
      return null;
    }
    
    switch (this.config.algorithm) {
      case "round_robin":
        return this.selectRoundRobin(healthyServers);
      case "least_connections":
        return this.selectLeastConnections(healthyServers);
      case "weighted":
        return this.selectWeighted(healthyServers);
      case "ip_hash":
        return this.selectIPHash(healthyServers, request);
      default:
        return this.selectRoundRobin(healthyServers);
    }
  }

  selectRoundRobin(servers) {
    const index = this.requestCounter % servers.length;
    return servers[index];
  }

  selectLeastConnections(servers) {
    return servers.reduce((min, server) => 
      server.activeConnections < min.activeConnections ? server : min
    );
  }

  selectWeighted(servers) {
    const totalWeight = servers.reduce((sum, server) => sum + server.weight, 0);
    let random = Math.random() * totalWeight;
    
    for (const server of servers) {
      random -= server.weight;
      if (random <= 0) {
        return server;
      }
    }
    
    return servers[0];
  }

  selectIPHash(servers, request) {
    const clientIP = this.getClientIP(request);
    const hash = this.hashString(clientIP);
    const index = hash % servers.length;
    return servers[index];
  }

  // Request Forwarding
  async forwardRequest(request, server, originalReq, originalRes) {
    try {
      const startTime = Date.now();
      server.activeConnections++;
      server.totalRequests++;
      request.serverId = server.id;
      
      // Prepare request options
      const options = {
        hostname: server.host,
        port: server.port,
        path: request.url,
        method: request.method,
        headers: {
          ...request.headers,
          "X-Forwarded-For": this.getClientIP(request),
          "X-Forwarded-Proto": originalReq.connection.encrypted ? "https" : "http"
        }
      };
      
      // Create proxy request
      const proxyReq = http.request(options, (proxyRes) => {
        const endTime = Date.now();
        const responseTime = endTime - startTime;
        
        // Update metrics
        request.responseTime = responseTime;
        request.statusCode = proxyRes.statusCode;
        server.responseTime = responseTime;
        
        // Forward response headers
        originalRes.writeHead(proxyRes.statusCode, proxyRes.headers);
        
        // Forward response body
        proxyRes.pipe(originalRes);
        
        // Clean up
        server.activeConnections--;
        this.emit("requestCompleted", { request, server, responseTime });
      });
      
      // Handle proxy request errors
      proxyReq.on("error", (error) => {
        server.activeConnections--;
        server.status = "unhealthy";
        this.emit("requestFailed", { request, server, error });
        
        originalRes.writeHead(502, { "Content-Type": "application/json" });
        originalRes.end(JSON.stringify({ error: "Bad gateway" }));
      });
      
      // Forward request body
      if (request.body) {
        proxyReq.write(request.body);
      }
      
      proxyReq.end();
      
    } catch (error) {
      console.error("Request forwarding error:", error);
      throw error;
    }
  }

  // Health Checking
  startHealthChecking() {
    if (!this.config.healthCheckEnabled) return;
    
    this.healthCheckInterval = setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval * 1000);
  }

  async performHealthChecks() {
    const healthCheckPromises = Array.from(this.servers.values()).map(server => 
      this.performHealthCheck(server)
    );
    
    await Promise.allSettled(healthCheckPromises);
  }

  async performHealthCheck(server) {
    try {
      const startTime = Date.now();
      
      const options = {
        hostname: server.host,
        port: server.port,
        path: server.healthCheckPath,
        method: "GET",
        timeout: this.config.healthCheckTimeout * 1000
      };
      
      const healthCheckPromise = new Promise((resolve, reject) => {
        const req = http.request(options, (res) => {
          const endTime = Date.now();
          const responseTime = endTime - startTime;
          
          if (res.statusCode >= 200 && res.statusCode < 300) {
            server.status = "healthy";
            server.lastHealthCheck = new Date();
            server.lastSeen = new Date();
            resolve({ server, status: "healthy", responseTime });
          } else {
            server.status = "unhealthy";
            reject(new Error(`Health check failed with status ${res.statusCode}`));
          }
        });
        
        req.on("error", (error) => {
          server.status = "unhealthy";
          reject(error);
        });
        
        req.on("timeout", () => {
          server.status = "unhealthy";
          reject(new Error("Health check timeout"));
        });
        
        req.end();
      });
      
      await healthCheckPromise;
      
    } catch (error) {
      server.status = "unhealthy";
      this.emit("serverUnhealthy", { server, error: error.message });
    }
  }

  // Session Management
  handleSessionPersistence(request, server) {
    const sessionId = this.extractSessionId(request);
    
    if (sessionId) {
      const session = this.sessions.get(sessionId);
      if (session && session.serverId === server.id) {
        session.lastAccessed = new Date();
        return;
      }
    }
    
    // Create new session
    if (sessionId) {
      const session = new Session(sessionId, server.id);
      this.sessions.set(sessionId, session);
    }
  }

  extractSessionId(request) {
    // Extract session ID from cookies or headers
    const cookies = request.headers.cookie;
    if (cookies) {
      const sessionMatch = cookies.match(/sessionId=([^;]+)/);
      if (sessionMatch) {
        return sessionMatch[1];
      }
    }
    
    return null;
  }

  cleanupServerSessions(serverId) {
    for (const [sessionId, session] of this.sessions) {
      if (session.serverId === serverId) {
        this.sessions.delete(sessionId);
      }
    }
  }

  // Metrics Collection
  startMetricsCollection() {
    setInterval(() => {
      this.updateMetrics();
    }, 1000); // Update every second
  }

  updateMetrics() {
    const now = Date.now();
    const uptime = now - this.metrics.startTime.getTime();
    
    // Calculate requests per second
    this.metrics.requestsPerSecond = this.requestCounter;
    this.requestCounter = 0;
    
    // Calculate average response time
    const servers = Array.from(this.servers.values());
    if (servers.length > 0) {
      const totalResponseTime = servers.reduce((sum, server) => sum + server.responseTime, 0);
      this.metrics.averageResponseTime = totalResponseTime / servers.length;
    }
    
    // Update active connections
    this.metrics.activeConnections = servers.reduce((sum, server) => sum + server.activeConnections, 0);
    
    this.emit("metricsUpdated", this.metrics);
  }

  // Background Tasks
  startSessionCleanup() {
    setInterval(() => {
      this.cleanupExpiredSessions();
    }, 300000); // Run every 5 minutes
  }

  cleanupExpiredSessions() {
    const now = new Date();
    const expiredSessions = [];
    
    for (const [sessionId, session] of this.sessions) {
      if (session.expiresAt < now) {
        expiredSessions.push(sessionId);
      }
    }
    
    expiredSessions.forEach(sessionId => {
      this.sessions.delete(sessionId);
    });
    
    if (expiredSessions.length > 0) {
      this.emit("sessionsCleaned", expiredSessions.length);
    }
  }

  // Utility Methods
  getClientIP(request) {
    return request.headers["x-forwarded-for"] || 
           request.headers["x-real-ip"] || 
           request.connection.remoteAddress || 
           "unknown";
  }

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
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
const { LoadBalancerService } = require("./services/LoadBalancerService");

class LoadBalancerAPI {
  constructor() {
    this.app = express();
    this.loadBalancer = new LoadBalancerService();
    
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
    // Load balancer management
    this.app.get("/api/health", this.getHealth.bind(this));
    this.app.get("/api/servers", this.getServers.bind(this));
    this.app.post("/api/servers", this.addServer.bind(this));
    this.app.delete("/api/servers/:serverId", this.removeServer.bind(this));
    this.app.put("/api/servers/:serverId/weight", this.updateServerWeight.bind(this));
    this.app.get("/api/servers/:serverId/health", this.getServerHealth.bind(this));
    
    // Configuration
    this.app.get("/api/config", this.getConfig.bind(this));
    this.app.put("/api/config", this.updateConfig.bind(this));
    this.app.get("/api/algorithms", this.getAlgorithms.bind(this));
    this.app.put("/api/algorithm", this.updateAlgorithm.bind(this));
    
    // Metrics
    this.app.get("/api/metrics", this.getMetrics.bind(this));
    this.app.get("/api/metrics/servers", this.getServerMetrics.bind(this));
    this.app.get("/api/metrics/requests", this.getRequestMetrics.bind(this));
    
    // Proxy all other requests
    this.app.all("*", this.proxyRequest.bind(this));
  }

  setupEventHandlers() {
    this.loadBalancer.on("serverAdded", (server) => {
      console.log(`Server added: ${server.host}:${server.port}`);
    });
    
    this.loadBalancer.on("serverUnhealthy", ({ server, error }) => {
      console.log(`Server unhealthy: ${server.host}:${server.port} - ${error}`);
    });
    
    this.loadBalancer.on("requestCompleted", ({ request, server, responseTime }) => {
      console.log(`Request completed: ${request.method} ${request.url} -> ${server.host}:${server.port} (${responseTime}ms)`);
    });
  }

  // HTTP Handlers
  async getHealth(req, res) {
    try {
      const healthyServers = Array.from(this.loadBalancer.servers.values())
        .filter(server => server.status === "healthy").length;
      
      const totalServers = this.loadBalancer.servers.size;
      
      res.json({
        status: "healthy",
        servers: {
          total: totalServers,
          healthy: healthyServers,
          unhealthy: totalServers - healthyServers
        },
        uptime: Date.now() - this.loadBalancer.metrics.startTime.getTime()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getServers(req, res) {
    try {
      const servers = Array.from(this.loadBalancer.servers.values());
      
      res.json({
        success: true,
        data: servers
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async addServer(req, res) {
    try {
      const server = await this.loadBalancer.addServer(req.body);
      
      res.status(201).json({
        success: true,
        data: server
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async removeServer(req, res) {
    try {
      const { serverId } = req.params;
      
      await this.loadBalancer.removeServer(serverId);
      
      res.json({
        success: true,
        message: "Server removed successfully"
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getMetrics(req, res) {
    try {
      res.json({
        success: true,
        data: this.loadBalancer.metrics
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async proxyRequest(req, res) {
    try {
      await this.loadBalancer.routeRequest(req, res);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Load Balancer API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new LoadBalancerAPI();
  api.start(3000);
}

module.exports = { LoadBalancerAPI };
```

## Key Features

### Load Balancing Algorithms
- **Round Robin**: Equal distribution across servers
- **Least Connections**: Route to server with fewest active connections
- **Weighted**: Distribution based on server capacity
- **IP Hash**: Consistent routing based on client IP

### Health Monitoring
- **Active Health Checks**: Regular server health monitoring
- **Automatic Failover**: Remove unhealthy servers from rotation
- **Health Check Customization**: Configurable health check paths and intervals
- **Server Recovery**: Automatic re-addition of recovered servers

### Session Management
- **Session Persistence**: Sticky sessions for stateful applications
- **Session Cleanup**: Automatic cleanup of expired sessions
- **Session Affinity**: Route requests to the same server
- **Session Monitoring**: Track session usage and expiration

## Extension Ideas

### Advanced Features
1. **SSL Termination**: HTTPS handling and certificate management
2. **Content Caching**: Cache responses for improved performance
3. **Rate Limiting**: Request throttling and abuse prevention
4. **Geographic Routing**: Route based on client location
5. **A/B Testing**: Traffic splitting for testing

### Enterprise Features
1. **Multi-region Support**: Global load balancing
2. **Advanced Metrics**: Detailed performance analytics
3. **API Gateway**: Request transformation and validation
4. **Security Features**: DDoS protection and WAF integration
5. **Configuration Management**: Dynamic configuration updates
