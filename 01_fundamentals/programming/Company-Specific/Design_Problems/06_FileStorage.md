---
# Auto-generated front matter
Title: 06 Filestorage
LastUpdated: 2025-11-06T20:45:58.773841
Tags: []
Status: draft
---

# 06. File Storage - Distributed File Management System

## Title & Summary
Design and implement a distributed file storage system using Node.js that handles file uploads, downloads, versioning, and provides CDN integration with metadata management.

## Problem Statement

Build a comprehensive file storage system that:

1. **File Management**: Upload, download, delete, and organize files
2. **Versioning**: Track file versions and enable rollback
3. **Metadata**: Store and search file metadata
4. **CDN Integration**: Global file distribution
5. **Access Control**: User permissions and file sharing
6. **Backup & Recovery**: Data redundancy and disaster recovery

## Requirements & Constraints

### Functional Requirements
- File upload with progress tracking
- File download with range requests
- File versioning and history
- Metadata search and filtering
- User access control and permissions
- File sharing and collaboration
- Backup and recovery operations

### Non-Functional Requirements
- **Latency**: < 200ms for metadata operations
- **Throughput**: 1000+ concurrent uploads
- **Availability**: 99.9% uptime
- **Scalability**: Handle 1B+ files
- **Storage**: Efficient storage and retrieval
- **Security**: File encryption and access control

## API / Interfaces

### REST Endpoints

```javascript
// File Management
POST   /api/files/upload
GET    /api/files/{fileId}
PUT    /api/files/{fileId}
DELETE /api/files/{fileId}
GET    /api/files/{fileId}/download
GET    /api/files/{fileId}/versions

// Metadata
GET    /api/files
GET    /api/files/search
POST   /api/files/{fileId}/metadata
GET    /api/files/{fileId}/metadata

// Sharing
POST   /api/files/{fileId}/share
GET    /api/files/shared
DELETE /api/files/{fileId}/share/{shareId}

// Backup
POST   /api/backup/start
GET    /api/backup/status
POST   /api/backup/restore
```

### Request/Response Examples

```json
// Upload File
POST /api/files/upload
Content-Type: multipart/form-data

{
  "file": <binary_data>,
  "metadata": {
    "name": "document.pdf",
    "description": "Important document",
    "tags": ["work", "important"],
    "folder": "/documents"
  }
}

// Response
{
  "success": true,
  "data": {
    "fileId": "file_123",
    "name": "document.pdf",
    "size": 1024000,
    "mimeType": "application/pdf",
    "uploadedAt": "2024-01-15T10:30:00Z",
    "version": 1,
    "checksum": "sha256:abc123...",
    "url": "https://cdn.example.com/files/file_123"
  }
}

// File Metadata
{
  "success": true,
  "data": {
    "fileId": "file_123",
    "name": "document.pdf",
    "size": 1024000,
    "mimeType": "application/pdf",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:30:00Z",
    "version": 1,
    "checksum": "sha256:abc123...",
    "metadata": {
      "description": "Important document",
      "tags": ["work", "important"],
      "folder": "/documents",
      "owner": "user_456"
    },
    "permissions": {
      "owner": "user_456",
      "read": ["user_789"],
      "write": ["user_456"]
    }
  }
}
```

## Data Model

### Core Entities

```javascript
// File Entity
class File {
  constructor(name, mimeType, size, ownerId) {
    this.id = this.generateID();
    this.name = name;
    this.mimeType = mimeType;
    this.size = size;
    this.ownerId = ownerId;
    this.version = 1;
    this.checksum = null;
    this.storagePath = null;
    this.cdnUrl = null;
    this.isActive = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.metadata = {};
    this.permissions = {
      owner: ownerId,
      read: [],
      write: []
    };
  }
}

// File Version Entity
class FileVersion {
  constructor(fileId, version, storagePath, checksum) {
    this.id = this.generateID();
    this.fileId = fileId;
    this.version = version;
    this.storagePath = storagePath;
    this.checksum = checksum;
    this.size = 0;
    this.createdAt = new Date();
    this.uploadedBy = null;
    this.changeLog = "";
  }
}

// File Metadata Entity
class FileMetadata {
  constructor(fileId, metadata) {
    this.id = this.generateID();
    this.fileId = fileId;
    this.metadata = metadata;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// File Share Entity
class FileShare {
  constructor(fileId, sharedBy, permissions) {
    this.id = this.generateID();
    this.fileId = fileId;
    this.sharedBy = sharedBy;
    this.permissions = permissions; // 'read', 'write', 'admin'
    this.expiresAt = null;
    this.password = null;
    this.isActive = true;
    this.createdAt = new Date();
    this.accessCount = 0;
  }
}

// Storage Node Entity
class StorageNode {
  constructor(nodeId, capacity, location) {
    this.id = nodeId;
    this.capacity = capacity;
    this.usedSpace = 0;
    this.location = location;
    this.status = "active"; // 'active', 'maintenance', 'offline'
    this.lastHeartbeat = new Date();
    this.replicationFactor = 3;
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. Local file system storage
2. Basic file operations
3. Simple metadata storage
4. No versioning or sharing

### Production-Ready Design
1. **Distributed Storage**: Multiple storage nodes
2. **CDN Integration**: Global file distribution
3. **Versioning System**: Complete file history
4. **Metadata Search**: Full-text search capabilities
5. **Access Control**: Role-based permissions
6. **Backup Strategy**: Multi-region replication

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const fs = require("fs").promises;
const path = require("path");
const crypto = require("crypto");
const { v4: uuidv4 } = require("uuid");

class FileStorageService extends EventEmitter {
  constructor() {
    super();
    this.files = new Map();
    this.fileVersions = new Map();
    this.metadata = new Map();
    this.shares = new Map();
    this.storageNodes = new Map();
    this.uploadQueue = [];
    this.isProcessing = false;
    
    // Initialize storage nodes
    this.initializeStorageNodes();
    
    // Start background tasks
    this.startUploadProcessor();
    this.startCleanupTask();
    this.startBackupTask();
  }

  initializeStorageNodes() {
    // Initialize storage nodes (in production, this would be from config)
    const nodes = [
      { id: "node1", capacity: 1000000000, location: "us-east-1" },
      { id: "node2", capacity: 1000000000, location: "us-west-1" },
      { id: "node3", capacity: 1000000000, location: "eu-west-1" }
    ];
    
    nodes.forEach(node => {
      this.storageNodes.set(node.id, new StorageNode(node.id, node.capacity, node.location));
    });
  }

  // File Upload
  async uploadFile(fileData, metadata, ownerId) {
    try {
      // Validate file
      this.validateFile(fileData);
      
      // Calculate checksum
      const checksum = this.calculateChecksum(fileData.buffer);
      
      // Check for existing file
      const existingFile = this.findFileByChecksum(checksum, ownerId);
      if (existingFile) {
        return this.createFileVersion(existingFile, fileData, checksum);
      }
      
      // Create new file
      const file = new File(
        fileData.originalname,
        fileData.mimetype,
        fileData.size,
        ownerId
      );
      
      file.checksum = checksum;
      file.metadata = metadata || {};
      
      // Store file
      this.files.set(file.id, file);
      
      // Add to upload queue
      this.uploadQueue.push({ file, fileData });
      
      this.emit("fileUploaded", file);
      
      return file;
      
    } catch (error) {
      console.error("File upload error:", error);
      throw error;
    }
  }

  // File Download
  async downloadFile(fileId, userId, range = null) {
    try {
      const file = this.files.get(fileId);
      if (!file) {
        throw new Error("File not found");
      }
      
      // Check permissions
      if (!this.hasReadPermission(file, userId)) {
        throw new Error("Access denied");
      }
      
      // Get latest version
      const version = this.getLatestVersion(fileId);
      if (!version) {
        throw new Error("File version not found");
      }
      
      // Read file from storage
      const filePath = version.storagePath;
      const fileBuffer = await fs.readFile(filePath);
      
      // Handle range requests
      if (range) {
        const { start, end } = this.parseRange(range, fileBuffer.length);
        const partialBuffer = fileBuffer.slice(start, end + 1);
        
        return {
          buffer: partialBuffer,
          range: { start, end, total: fileBuffer.length },
          headers: {
            "Content-Range": `bytes ${start}-${end}/${fileBuffer.length}`,
            "Accept-Ranges": "bytes",
            "Content-Length": partialBuffer.length
          }
        };
      }
      
      return {
        buffer: fileBuffer,
        headers: {
          "Content-Length": fileBuffer.length,
          "Content-Type": file.mimeType
        }
      };
      
    } catch (error) {
      console.error("File download error:", error);
      throw error;
    }
  }

  // File Versioning
  async createFileVersion(file, fileData, checksum) {
    const newVersion = file.version + 1;
    
    const version = new FileVersion(
      file.id,
      newVersion,
      null, // Will be set after storage
      checksum
    );
    
    version.size = fileData.size;
    version.uploadedBy = file.ownerId;
    
    // Store version
    this.fileVersions.set(version.id, version);
    
    // Update file
    file.version = newVersion;
    file.updatedAt = new Date();
    file.checksum = checksum;
    
    // Add to upload queue
    this.uploadQueue.push({ file, fileData, version });
    
    this.emit("fileVersionCreated", { file, version });
    
    return file;
  }

  // Metadata Management
  async updateMetadata(fileId, metadata, userId) {
    const file = this.files.get(fileId);
    if (!file) {
      throw new Error("File not found");
    }
    
    // Check write permissions
    if (!this.hasWritePermission(file, userId)) {
      throw new Error("Access denied");
    }
    
    // Update metadata
    file.metadata = { ...file.metadata, ...metadata };
    file.updatedAt = new Date();
    
    // Store metadata
    const fileMetadata = new FileMetadata(fileId, file.metadata);
    this.metadata.set(fileMetadata.id, fileMetadata);
    
    this.emit("metadataUpdated", { file, metadata });
    
    return file;
  }

  // File Sharing
  async shareFile(fileId, sharedBy, permissions, options = {}) {
    const file = this.files.get(fileId);
    if (!file) {
      throw new Error("File not found");
    }
    
    // Check admin permissions
    if (!this.hasAdminPermission(file, sharedBy)) {
      throw new Error("Access denied");
    }
    
    const share = new FileShare(fileId, sharedBy, permissions);
    
    if (options.expiresAt) {
      share.expiresAt = new Date(options.expiresAt);
    }
    
    if (options.password) {
      share.password = this.hashPassword(options.password);
    }
    
    this.shares.set(share.id, share);
    
    this.emit("fileShared", { file, share });
    
    return share;
  }

  // File Search
  async searchFiles(query, userId, options = {}) {
    const results = [];
    
    for (const file of this.files.values()) {
      // Check permissions
      if (!this.hasReadPermission(file, userId)) {
        continue;
      }
      
      // Search in metadata
      const matches = this.searchInMetadata(file, query);
      if (matches.length > 0) {
        results.push({
          file,
          matches,
          score: this.calculateSearchScore(file, query)
        });
      }
    }
    
    // Sort by score
    results.sort((a, b) => b.score - a.score);
    
    // Apply pagination
    const { limit = 50, offset = 0 } = options;
    return results.slice(offset, offset + limit);
  }

  // Storage Management
  async storeFile(file, fileData, version = null) {
    try {
      // Select storage node
      const storageNode = this.selectStorageNode();
      if (!storageNode) {
        throw new Error("No available storage nodes");
      }
      
      // Generate storage path
      const storagePath = this.generateStoragePath(file.id, version?.version || file.version);
      
      // Create directory if not exists
      await this.ensureDirectoryExists(path.dirname(storagePath));
      
      // Write file to storage
      await fs.writeFile(storagePath, fileData.buffer);
      
      // Update file/version with storage path
      if (version) {
        version.storagePath = storagePath;
      } else {
        file.storagePath = storagePath;
      }
      
      // Update storage node usage
      storageNode.usedSpace += fileData.size;
      
      // Generate CDN URL
      const cdnUrl = this.generateCDNUrl(file.id);
      file.cdnUrl = cdnUrl;
      
      this.emit("fileStored", { file, version, storageNode });
      
      return { storagePath, cdnUrl };
      
    } catch (error) {
      console.error("File storage error:", error);
      throw error;
    }
  }

  // Background Tasks
  startUploadProcessor() {
    setInterval(() => {
      this.processUploadQueue();
    }, 1000); // Process every second
  }

  async processUploadQueue() {
    if (this.isProcessing || this.uploadQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    while (this.uploadQueue.length > 0) {
      const { file, fileData, version } = this.uploadQueue.shift();
      
      try {
        await this.storeFile(file, fileData, version);
      } catch (error) {
        console.error("Upload processing error:", error);
        // Re-queue for retry
        this.uploadQueue.push({ file, fileData, version });
      }
    }
    
    this.isProcessing = false;
  }

  startCleanupTask() {
    setInterval(() => {
      this.cleanupExpiredShares();
      this.cleanupOrphanedFiles();
    }, 3600000); // Run every hour
  }

  cleanupExpiredShares() {
    const now = new Date();
    const expiredShares = [];
    
    for (const [shareId, share] of this.shares) {
      if (share.expiresAt && share.expiresAt < now) {
        expiredShares.push(shareId);
      }
    }
    
    expiredShares.forEach(shareId => {
      this.shares.delete(shareId);
    });
    
    if (expiredShares.length > 0) {
      this.emit("sharesExpired", expiredShares.length);
    }
  }

  startBackupTask() {
    setInterval(() => {
      this.performBackup();
    }, 86400000); // Run daily
  }

  async performBackup() {
    try {
      console.log("Starting backup process...");
      
      // In production, this would backup to multiple regions
      const backupData = {
        files: Array.from(this.files.entries()),
        versions: Array.from(this.fileVersions.entries()),
        metadata: Array.from(this.metadata.entries()),
        timestamp: new Date()
      };
      
      // Store backup (simplified)
      const backupPath = `./backups/backup_${Date.now()}.json`;
      await fs.writeFile(backupPath, JSON.stringify(backupData, null, 2));
      
      this.emit("backupCompleted", { backupPath, fileCount: this.files.size });
      
    } catch (error) {
      console.error("Backup error:", error);
      this.emit("backupFailed", error);
    }
  }

  // Utility Methods
  validateFile(fileData) {
    if (!fileData || !fileData.buffer) {
      throw new Error("Invalid file data");
    }
    
    if (fileData.size > 100 * 1024 * 1024) { // 100MB limit
      throw new Error("File too large");
    }
    
    const allowedTypes = [
      "image/jpeg", "image/png", "image/gif",
      "application/pdf", "text/plain", "application/json"
    ];
    
    if (!allowedTypes.includes(fileData.mimetype)) {
      throw new Error("File type not allowed");
    }
  }

  calculateChecksum(buffer) {
    return crypto.createHash("sha256").update(buffer).digest("hex");
  }

  findFileByChecksum(checksum, ownerId) {
    for (const file of this.files.values()) {
      if (file.checksum === checksum && file.ownerId === ownerId) {
        return file;
      }
    }
    return null;
  }

  getLatestVersion(fileId) {
    let latestVersion = null;
    
    for (const version of this.fileVersions.values()) {
      if (version.fileId === fileId) {
        if (!latestVersion || version.version > latestVersion.version) {
          latestVersion = version;
        }
      }
    }
    
    return latestVersion;
  }

  hasReadPermission(file, userId) {
    return file.ownerId === userId || 
           file.permissions.read.includes(userId) ||
           file.permissions.write.includes(userId);
  }

  hasWritePermission(file, userId) {
    return file.ownerId === userId || 
           file.permissions.write.includes(userId);
  }

  hasAdminPermission(file, userId) {
    return file.ownerId === userId;
  }

  selectStorageNode() {
    // Select storage node with least usage
    let selectedNode = null;
    let minUsage = Infinity;
    
    for (const node of this.storageNodes.values()) {
      if (node.status === "active" && node.usedSpace < minUsage) {
        selectedNode = node;
        minUsage = node.usedSpace;
      }
    }
    
    return selectedNode;
  }

  generateStoragePath(fileId, version) {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    
    return `./storage/${year}/${month}/${day}/${fileId}_v${version}`;
  }

  generateCDNUrl(fileId) {
    return `https://cdn.example.com/files/${fileId}`;
  }

  parseRange(range, fileSize) {
    const match = range.match(/bytes=(\d+)-(\d*)/);
    if (!match) {
      throw new Error("Invalid range");
    }
    
    const start = parseInt(match[1]);
    const end = match[2] ? parseInt(match[2]) : fileSize - 1;
    
    return { start, end };
  }

  searchInMetadata(file, query) {
    const matches = [];
    const searchText = query.toLowerCase();
    
    // Search in file name
    if (file.name.toLowerCase().includes(searchText)) {
      matches.push({ field: "name", value: file.name });
    }
    
    // Search in metadata
    Object.entries(file.metadata).forEach(([key, value]) => {
      if (String(value).toLowerCase().includes(searchText)) {
        matches.push({ field: key, value });
      }
    });
    
    return matches;
  }

  calculateSearchScore(file, query) {
    let score = 0;
    const searchText = query.toLowerCase();
    
    // Name match gets highest score
    if (file.name.toLowerCase().includes(searchText)) {
      score += 100;
    }
    
    // Metadata matches get lower scores
    Object.values(file.metadata).forEach(value => {
      if (String(value).toLowerCase().includes(searchText)) {
        score += 10;
      }
    });
    
    return score;
  }

  async ensureDirectoryExists(dirPath) {
    try {
      await fs.access(dirPath);
    } catch (error) {
      await fs.mkdir(dirPath, { recursive: true });
    }
  }

  hashPassword(password) {
    return crypto.createHash("sha256").update(password).digest("hex");
  }

  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { FileStorageService } = require("./services/FileStorageService");

class FileStorageAPI {
  constructor() {
    this.app = express();
    this.fileService = new FileStorageService();
    
    // Configure multer for file uploads
    this.upload = multer({
      storage: multer.memoryStorage(),
      limits: {
        fileSize: 100 * 1024 * 1024 // 100MB
      }
    });
    
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
    // File management
    this.app.post("/api/files/upload", this.upload.single("file"), this.uploadFile.bind(this));
    this.app.get("/api/files/:fileId", this.getFile.bind(this));
    this.app.put("/api/files/:fileId", this.updateFile.bind(this));
    this.app.delete("/api/files/:fileId", this.deleteFile.bind(this));
    this.app.get("/api/files/:fileId/download", this.downloadFile.bind(this));
    this.app.get("/api/files/:fileId/versions", this.getFileVersions.bind(this));
    
    // Metadata
    this.app.get("/api/files", this.getFiles.bind(this));
    this.app.get("/api/files/search", this.searchFiles.bind(this));
    this.app.post("/api/files/:fileId/metadata", this.updateMetadata.bind(this));
    this.app.get("/api/files/:fileId/metadata", this.getMetadata.bind(this));
    
    // Sharing
    this.app.post("/api/files/:fileId/share", this.shareFile.bind(this));
    this.app.get("/api/files/shared", this.getSharedFiles.bind(this));
    this.app.delete("/api/files/:fileId/share/:shareId", this.unshareFile.bind(this));
    
    // Backup
    this.app.post("/api/backup/start", this.startBackup.bind(this));
    this.app.get("/api/backup/status", this.getBackupStatus.bind(this));
    this.app.post("/api/backup/restore", this.restoreBackup.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalFiles: this.fileService.files.size,
        totalVersions: this.fileService.fileVersions.size,
        activeShares: this.fileService.shares.size
      });
    });
  }

  setupEventHandlers() {
    this.fileService.on("fileUploaded", (file) => {
      console.log(`File uploaded: ${file.name} (${file.id})`);
    });
    
    this.fileService.on("fileStored", ({ file, storageNode }) => {
      console.log(`File stored: ${file.name} on ${storageNode.id}`);
    });
    
    this.fileService.on("backupCompleted", ({ backupPath, fileCount }) => {
      console.log(`Backup completed: ${fileCount} files backed up to ${backupPath}`);
    });
  }

  // HTTP Handlers
  async uploadFile(req, res) {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file provided" });
      }
      
      const { metadata } = req.body;
      const userId = req.headers["x-user-id"] || "anonymous";
      
      const file = await this.fileService.uploadFile(
        req.file,
        metadata ? JSON.parse(metadata) : {},
        userId
      );
      
      res.status(201).json({
        success: true,
        data: {
          fileId: file.id,
          name: file.name,
          size: file.size,
          mimeType: file.mimeType,
          uploadedAt: file.createdAt,
          version: file.version,
          checksum: file.checksum,
          url: file.cdnUrl
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getFile(req, res) {
    try {
      const { fileId } = req.params;
      const userId = req.headers["x-user-id"] || "anonymous";
      
      const file = this.fileService.files.get(fileId);
      if (!file) {
        return res.status(404).json({ error: "File not found" });
      }
      
      // Check permissions
      if (!this.fileService.hasReadPermission(file, userId)) {
        return res.status(403).json({ error: "Access denied" });
      }
      
      res.json({
        success: true,
        data: {
          fileId: file.id,
          name: file.name,
          size: file.size,
          mimeType: file.mimeType,
          createdAt: file.createdAt,
          updatedAt: file.updatedAt,
          version: file.version,
          checksum: file.checksum,
          metadata: file.metadata,
          permissions: file.permissions,
          url: file.cdnUrl
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async downloadFile(req, res) {
    try {
      const { fileId } = req.params;
      const userId = req.headers["x-user-id"] || "anonymous";
      const range = req.headers.range;
      
      const result = await this.fileService.downloadFile(fileId, userId, range);
      
      // Set headers
      Object.entries(result.headers).forEach(([key, value]) => {
        res.set(key, value);
      });
      
      if (range) {
        res.status(206); // Partial Content
      }
      
      res.send(result.buffer);
    } catch (error) {
      res.status(404).json({ error: error.message });
    }
  }

  async searchFiles(req, res) {
    try {
      const { q, limit = 50, offset = 0 } = req.query;
      const userId = req.headers["x-user-id"] || "anonymous";
      
      if (!q) {
        return res.status(400).json({ error: "Search query required" });
      }
      
      const results = await this.fileService.searchFiles(q, userId, { limit, offset });
      
      res.json({
        success: true,
        data: results,
        pagination: {
          limit: parseInt(limit),
          offset: parseInt(offset),
          total: results.length
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async shareFile(req, res) {
    try {
      const { fileId } = req.params;
      const { permissions, expiresAt, password } = req.body;
      const userId = req.headers["x-user-id"] || "anonymous";
      
      const share = await this.fileService.shareFile(
        fileId,
        userId,
        permissions,
        { expiresAt, password }
      );
      
      res.status(201).json({
        success: true,
        data: {
          shareId: share.id,
          fileId: share.fileId,
          permissions: share.permissions,
          expiresAt: share.expiresAt,
          createdAt: share.createdAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async startBackup(req, res) {
    try {
      // Trigger backup
      this.fileService.performBackup();
      
      res.json({
        success: true,
        message: "Backup started"
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`File Storage API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new FileStorageAPI();
  api.start(3000);
}

module.exports = { FileStorageAPI };
```

## Key Features

### File Management
- **Upload/Download**: Efficient file transfer with progress tracking
- **Versioning**: Complete file history and rollback capabilities
- **Metadata**: Rich metadata storage and search
- **Access Control**: Role-based permissions and sharing

### Storage & Distribution
- **Distributed Storage**: Multiple storage nodes for redundancy
- **CDN Integration**: Global file distribution
- **Range Requests**: Efficient partial file downloads
- **Checksum Validation**: Data integrity verification

### Security & Reliability
- **Access Control**: User permissions and file sharing
- **Encryption**: File encryption and secure storage
- **Backup & Recovery**: Multi-region backup strategy
- **Fault Tolerance**: Storage node failure handling

### Performance & Scalability
- **Efficient Storage**: Optimized file organization
- **Search Capabilities**: Fast metadata search
- **Background Processing**: Asynchronous file operations
- **Load Balancing**: Distributed storage management

## Extension Ideas

### Advanced Features
1. **File Compression**: Automatic compression for storage optimization
2. **Image Processing**: Thumbnail generation and image manipulation
3. **Virus Scanning**: File security scanning
4. **Content Delivery**: Advanced CDN integration
5. **File Synchronization**: Real-time file sync across devices

### Enterprise Features
1. **Multi-tenancy**: Isolated file environments
2. **Advanced Analytics**: File usage and access analytics
3. **Compliance**: GDPR and data retention policies
4. **Integration APIs**: Third-party service integration
5. **Audit Trails**: Complete file access logging
