---
# Auto-generated front matter
Title: 05 Taskscheduler
LastUpdated: 2025-11-06T20:45:58.778780
Tags: []
Status: draft
---

# 05. Task Scheduler - Job Management System

## Title & Summary
Design and implement a distributed task scheduler using Node.js that manages recurring jobs, one-time tasks, and provides real-time monitoring with fault tolerance.

## Problem Statement

Build a comprehensive task scheduling system that:

1. **Job Management**: Create, update, delete, and execute scheduled tasks
2. **Multiple Triggers**: Cron expressions, intervals, and one-time execution
3. **Distributed Execution**: Work across multiple worker nodes
4. **Fault Tolerance**: Handle worker failures and job retries
5. **Monitoring**: Real-time job status and performance metrics
6. **Priority Queues**: Different priority levels for job execution

## Requirements & Constraints

### Functional Requirements
- Schedule jobs with cron expressions or intervals
- Support one-time and recurring jobs
- Job priority and dependency management
- Real-time job status monitoring
- Job retry and failure handling
- Worker node management
- Job history and logging

### Non-Functional Requirements
- **Latency**: < 1 second for job scheduling
- **Throughput**: 10,000+ jobs per minute
- **Availability**: 99.9% uptime
- **Scalability**: Support 1M+ scheduled jobs
- **Reliability**: 99.95% job execution success rate
- **Memory**: < 2GB per worker node

## API / Interfaces

### REST Endpoints

```javascript
// Job Management
POST   /api/jobs
GET    /api/jobs
GET    /api/jobs/{jobId}
PUT    /api/jobs/{jobId}
DELETE /api/jobs/{jobId}
POST   /api/jobs/{jobId}/execute
POST   /api/jobs/{jobId}/pause
POST   /api/jobs/{jobId}/resume

// Worker Management
GET    /api/workers
GET    /api/workers/{workerId}
POST   /api/workers/register
DELETE /api/workers/{workerId}

// Monitoring
GET    /api/jobs/{jobId}/status
GET    /api/jobs/{jobId}/history
GET    /api/monitoring/dashboard
GET    /api/monitoring/metrics
```

### Request/Response Examples

```json
// Create Job
POST /api/jobs
{
  "name": "daily-report",
  "description": "Generate daily sales report",
  "type": "recurring",
  "schedule": "0 9 * * *",
  "priority": "high",
  "payload": {
    "reportType": "sales",
    "email": "admin@company.com"
  },
  "retryPolicy": {
    "maxRetries": 3,
    "retryDelay": 300
  },
  "timeout": 3600
}

// Response
{
  "success": true,
  "data": {
    "jobId": "job_123",
    "name": "daily-report",
    "status": "scheduled",
    "nextRun": "2024-01-16T09:00:00Z",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}

// Job Status
{
  "success": true,
  "data": {
    "jobId": "job_123",
    "status": "running",
    "currentRun": {
      "runId": "run_456",
      "startedAt": "2024-01-16T09:00:00Z",
      "workerId": "worker_789"
    },
    "nextRun": "2024-01-17T09:00:00Z",
    "lastRun": {
      "runId": "run_455",
      "status": "completed",
      "startedAt": "2024-01-15T09:00:00Z",
      "completedAt": "2024-01-15T09:05:00Z",
      "duration": 300
    }
  }
}
```

## Data Model

### Core Entities

```javascript
// Job Entity
class Job {
  constructor(name, type, schedule, payload = {}) {
    this.id = this.generateID();
    this.name = name;
    this.description = "";
    this.type = type; // 'one-time', 'recurring'
    this.schedule = schedule; // cron expression or ISO date
    this.payload = payload;
    this.priority = "normal"; // 'low', 'normal', 'high', 'critical'
    this.status = "scheduled"; // 'scheduled', 'running', 'completed', 'failed', 'paused'
    this.retryPolicy = {
      maxRetries: 3,
      retryDelay: 300 // seconds
    };
    this.timeout = 3600; // seconds
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.nextRun = null;
    this.lastRun = null;
    this.dependencies = [];
  }
}

// Job Run Entity
class JobRun {
  constructor(jobId, workerId) {
    this.id = this.generateID();
    this.jobId = jobId;
    this.workerId = workerId;
    this.status = "running"; // 'running', 'completed', 'failed', 'timeout'
    this.startedAt = new Date();
    this.completedAt = null;
    this.duration = null;
    this.error = null;
    this.result = null;
    this.retryCount = 0;
  }
}

// Worker Entity
class Worker {
  constructor(workerId, capabilities = []) {
    this.id = workerId;
    this.capabilities = capabilities;
    this.status = "active"; // 'active', 'busy', 'idle', 'offline'
    this.lastHeartbeat = new Date();
    this.currentJobs = new Set();
    this.maxConcurrentJobs = 10;
    this.registeredAt = new Date();
  }
}

// Job Queue Entity
class JobQueue {
  constructor(priority = "normal") {
    this.priority = priority;
    this.jobs = [];
    this.maxSize = 10000;
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory job storage with arrays
2. Basic cron parsing and scheduling
3. Single worker execution
4. Simple retry mechanism

### Production-Ready Design
1. **Distributed Architecture**: Multiple worker nodes
2. **Message Queue**: Redis for job distribution
3. **Cron Parser**: Robust cron expression handling
4. **Fault Tolerance**: Worker failure detection and recovery
5. **Monitoring**: Real-time metrics and alerting
6. **Persistence**: Database for job storage

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const cron = require("node-cron");
const { v4: uuidv4 } = require("uuid");

class TaskSchedulerService extends EventEmitter {
  constructor() {
    super();
    this.jobs = new Map();
    this.jobRuns = new Map();
    this.workers = new Map();
    this.queues = new Map();
    this.cronJobs = new Map();
    
    // Initialize priority queues
    this.initializeQueues();
    
    // Start background tasks
    this.startScheduler();
    this.startWorkerMonitor();
    this.startCleanupTask();
  }

  initializeQueues() {
    const priorities = ["low", "normal", "high", "critical"];
    priorities.forEach(priority => {
      this.queues.set(priority, new JobQueue(priority));
    });
  }

  // Job Management
  async createJob(jobData) {
    try {
      const job = new Job(
        jobData.name,
        jobData.type,
        jobData.schedule,
        jobData.payload
      );
      
      // Set additional properties
      if (jobData.description) job.description = jobData.description;
      if (jobData.priority) job.priority = jobData.priority;
      if (jobData.retryPolicy) job.retryPolicy = jobData.retryPolicy;
      if (jobData.timeout) job.timeout = jobData.timeout;
      if (jobData.dependencies) job.dependencies = jobData.dependencies;
      
      // Calculate next run time
      job.nextRun = this.calculateNextRun(job);
      
      // Store job
      this.jobs.set(job.id, job);
      
      // Schedule job if recurring
      if (job.type === "recurring") {
        this.scheduleRecurringJob(job);
      }
      
      this.emit("jobCreated", job);
      
      return job;
      
    } catch (error) {
      console.error("Job creation error:", error);
      throw error;
    }
  }

  async updateJob(jobId, updates) {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    
    // Update properties
    Object.keys(updates).forEach(key => {
      if (key !== "id" && job.hasOwnProperty(key)) {
        job[key] = updates[key];
      }
    });
    
    job.updatedAt = new Date();
    
    // Reschedule if schedule changed
    if (updates.schedule) {
      job.nextRun = this.calculateNextRun(job);
      this.rescheduleJob(job);
    }
    
    this.emit("jobUpdated", job);
    
    return job;
  }

  async deleteJob(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    
    // Cancel cron job if recurring
    if (job.type === "recurring" && this.cronJobs.has(jobId)) {
      this.cronJobs.get(jobId).destroy();
      this.cronJobs.delete(jobId);
    }
    
    // Remove from queues
    this.removeJobFromQueues(jobId);
    
    this.jobs.delete(jobId);
    
    this.emit("jobDeleted", job);
    
    return true;
  }

  // Job Execution
  async executeJob(jobId, workerId = null) {
    try {
      const job = this.jobs.get(jobId);
      if (!job) {
        throw new Error("Job not found");
      }
      
      if (job.status === "running") {
        throw new Error("Job is already running");
      }
      
      // Check dependencies
      if (!await this.checkDependencies(job)) {
        throw new Error("Job dependencies not met");
      }
      
      // Find available worker
      const worker = workerId ? this.workers.get(workerId) : this.findAvailableWorker();
      if (!worker) {
        throw new Error("No available workers");
      }
      
      // Create job run
      const jobRun = new JobRun(jobId, worker.id);
      this.jobRuns.set(jobRun.id, jobRun);
      
      // Update job status
      job.status = "running";
      job.lastRun = jobRun.id;
      job.updatedAt = new Date();
      
      // Update worker status
      worker.status = "busy";
      worker.currentJobs.add(jobId);
      worker.lastHeartbeat = new Date();
      
      // Execute job
      this.emit("jobStarted", { job, jobRun, worker });
      
      try {
        const result = await this.runJob(job, jobRun);
        
        // Job completed successfully
        jobRun.status = "completed";
        jobRun.completedAt = new Date();
        jobRun.duration = jobRun.completedAt - jobRun.startedAt;
        jobRun.result = result;
        
        job.status = "completed";
        job.updatedAt = new Date();
        
        // Calculate next run for recurring jobs
        if (job.type === "recurring") {
          job.nextRun = this.calculateNextRun(job);
        }
        
        this.emit("jobCompleted", { job, jobRun, worker });
        
      } catch (error) {
        // Job failed
        jobRun.status = "failed";
        jobRun.completedAt = new Date();
        jobRun.duration = jobRun.completedAt - jobRun.startedAt;
        jobRun.error = error.message;
        
        // Handle retry
        if (jobRun.retryCount < job.retryPolicy.maxRetries) {
          jobRun.retryCount++;
          job.status = "scheduled";
          job.nextRun = new Date(Date.now() + job.retryPolicy.retryDelay * 1000);
          
          this.emit("jobRetry", { job, jobRun, worker });
        } else {
          job.status = "failed";
          this.emit("jobFailed", { job, jobRun, worker });
        }
        
        job.updatedAt = new Date();
      } finally {
        // Update worker status
        worker.currentJobs.delete(jobId);
        if (worker.currentJobs.size === 0) {
          worker.status = "idle";
        }
        worker.lastHeartbeat = new Date();
      }
      
      return jobRun;
      
    } catch (error) {
      console.error("Job execution error:", error);
      throw error;
    }
  }

  // Worker Management
  registerWorker(workerId, capabilities = []) {
    const worker = new Worker(workerId, capabilities);
    this.workers.set(workerId, worker);
    
    this.emit("workerRegistered", worker);
    
    return worker;
  }

  updateWorkerHeartbeat(workerId) {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.lastHeartbeat = new Date();
      worker.status = worker.currentJobs.size > 0 ? "busy" : "idle";
    }
  }

  findAvailableWorker() {
    for (const worker of this.workers.values()) {
      if (worker.status === "idle" && worker.currentJobs.size < worker.maxConcurrentJobs) {
        return worker;
      }
    }
    return null;
  }

  // Scheduling
  scheduleRecurringJob(job) {
    if (job.type !== "recurring") return;
    
    const cronJob = cron.schedule(job.schedule, () => {
      this.executeJob(job.id);
    }, {
      scheduled: false
    });
    
    this.cronJobs.set(job.id, cronJob);
    cronJob.start();
  }

  rescheduleJob(job) {
    if (job.type !== "recurring") return;
    
    // Stop existing cron job
    if (this.cronJobs.has(job.id)) {
      this.cronJobs.get(job.id).destroy();
    }
    
    // Create new cron job
    this.scheduleRecurringJob(job);
  }

  calculateNextRun(job) {
    if (job.type === "one-time") {
      return new Date(job.schedule);
    }
    
    if (job.type === "recurring") {
      // Use cron-parser to calculate next run
      const parser = require("cron-parser");
      const interval = parser.parseExpression(job.schedule);
      return interval.next().toDate();
    }
    
    return null;
  }

  // Job Execution
  async runJob(job, jobRun) {
    // Simulate job execution
    // In production, this would execute actual job logic
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Job timeout"));
      }, job.timeout * 1000);
      
      // Simulate job work
      setTimeout(() => {
        clearTimeout(timeout);
        
        // Simulate success/failure
        if (Math.random() > 0.1) { // 90% success rate
          resolve({
            message: "Job completed successfully",
            data: job.payload
          });
        } else {
          reject(new Error("Simulated job failure"));
        }
      }, Math.random() * 5000); // Random execution time
    });
  }

  // Dependencies
  async checkDependencies(job) {
    if (!job.dependencies || job.dependencies.length === 0) {
      return true;
    }
    
    for (const depJobId of job.dependencies) {
      const depJob = this.jobs.get(depJobId);
      if (!depJob || depJob.status !== "completed") {
        return false;
      }
    }
    
    return true;
  }

  // Queue Management
  addJobToQueue(job) {
    const queue = this.queues.get(job.priority);
    if (queue && queue.jobs.length < queue.maxSize) {
      queue.jobs.push(job.id);
      this.sortQueue(queue);
    }
  }

  removeJobFromQueues(jobId) {
    for (const queue of this.queues.values()) {
      const index = queue.jobs.indexOf(jobId);
      if (index > -1) {
        queue.jobs.splice(index, 1);
      }
    }
  }

  sortQueue(queue) {
    queue.jobs.sort((a, b) => {
      const jobA = this.jobs.get(a);
      const jobB = this.jobs.get(b);
      
      if (!jobA || !jobB) return 0;
      
      // Sort by priority, then by next run time
      const priorityOrder = { critical: 4, high: 3, normal: 2, low: 1 };
      const priorityDiff = priorityOrder[jobB.priority] - priorityOrder[jobA.priority];
      
      if (priorityDiff !== 0) return priorityDiff;
      
      return new Date(jobA.nextRun) - new Date(jobB.nextRun);
    });
  }

  // Background Tasks
  startScheduler() {
    setInterval(() => {
      this.processScheduledJobs();
    }, 1000); // Check every second
  }

  processScheduledJobs() {
    const now = new Date();
    
    for (const job of this.jobs.values()) {
      if (job.status === "scheduled" && job.nextRun && job.nextRun <= now) {
        this.addJobToQueue(job);
      }
    }
    
    // Process queues
    this.processQueues();
  }

  processQueues() {
    const priorities = ["critical", "high", "normal", "low"];
    
    for (const priority of priorities) {
      const queue = this.queues.get(priority);
      if (queue.jobs.length === 0) continue;
      
      const jobId = queue.jobs.shift();
      const job = this.jobs.get(jobId);
      
      if (job && job.status === "scheduled") {
        this.executeJob(jobId);
      }
    }
  }

  startWorkerMonitor() {
    setInterval(() => {
      this.monitorWorkers();
    }, 30000); // Check every 30 seconds
  }

  monitorWorkers() {
    const now = new Date();
    const timeout = 60000; // 1 minute timeout
    
    for (const [workerId, worker] of this.workers) {
      if (now - worker.lastHeartbeat > timeout) {
        worker.status = "offline";
        
        // Reassign jobs to other workers
        for (const jobId of worker.currentJobs) {
          const job = this.jobs.get(jobId);
          if (job) {
            job.status = "scheduled";
            this.addJobToQueue(job);
          }
        }
        
        worker.currentJobs.clear();
        
        this.emit("workerOffline", worker);
      }
    }
  }

  startCleanupTask() {
    setInterval(() => {
      this.cleanupOldRuns();
    }, 3600000); // Cleanup every hour
  }

  cleanupOldRuns() {
    const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
    const oldRuns = [];
    
    for (const [runId, run] of this.jobRuns) {
      if (run.startedAt < cutoff) {
        oldRuns.push(runId);
      }
    }
    
    oldRuns.forEach(runId => {
      this.jobRuns.delete(runId);
    });
    
    if (oldRuns.length > 0) {
      this.emit("runsCleaned", oldRuns.length);
    }
  }

  // Utility Methods
  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { TaskSchedulerService } = require("./services/TaskSchedulerService");

class TaskSchedulerAPI {
  constructor() {
    this.app = express();
    this.scheduler = new TaskSchedulerService();
    
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
    // Job management
    this.app.post("/api/jobs", this.createJob.bind(this));
    this.app.get("/api/jobs", this.getJobs.bind(this));
    this.app.get("/api/jobs/:jobId", this.getJob.bind(this));
    this.app.put("/api/jobs/:jobId", this.updateJob.bind(this));
    this.app.delete("/api/jobs/:jobId", this.deleteJob.bind(this));
    this.app.post("/api/jobs/:jobId/execute", this.executeJob.bind(this));
    this.app.post("/api/jobs/:jobId/pause", this.pauseJob.bind(this));
    this.app.post("/api/jobs/:jobId/resume", this.resumeJob.bind(this));
    
    // Worker management
    this.app.get("/api/workers", this.getWorkers.bind(this));
    this.app.get("/api/workers/:workerId", this.getWorker.bind(this));
    this.app.post("/api/workers/register", this.registerWorker.bind(this));
    this.app.delete("/api/workers/:workerId", this.unregisterWorker.bind(this));
    this.app.post("/api/workers/:workerId/heartbeat", this.updateHeartbeat.bind(this));
    
    // Monitoring
    this.app.get("/api/jobs/:jobId/status", this.getJobStatus.bind(this));
    this.app.get("/api/jobs/:jobId/history", this.getJobHistory.bind(this));
    this.app.get("/api/monitoring/dashboard", this.getDashboard.bind(this));
    this.app.get("/api/monitoring/metrics", this.getMetrics.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalJobs: this.scheduler.jobs.size,
        totalWorkers: this.scheduler.workers.size,
        activeJobs: Array.from(this.scheduler.jobs.values())
          .filter(job => job.status === "running").length
      });
    });
  }

  setupEventHandlers() {
    this.scheduler.on("jobCreated", (job) => {
      console.log(`Job created: ${job.name} (${job.id})`);
    });
    
    this.scheduler.on("jobStarted", ({ job, jobRun, worker }) => {
      console.log(`Job started: ${job.name} on worker ${worker.id}`);
    });
    
    this.scheduler.on("jobCompleted", ({ job, jobRun, worker }) => {
      console.log(`Job completed: ${job.name} in ${jobRun.duration}ms`);
    });
    
    this.scheduler.on("jobFailed", ({ job, jobRun, worker }) => {
      console.log(`Job failed: ${job.name} - ${jobRun.error}`);
    });
    
    this.scheduler.on("workerOffline", (worker) => {
      console.log(`Worker offline: ${worker.id}`);
    });
  }

  // HTTP Handlers
  async createJob(req, res) {
    try {
      const job = await this.scheduler.createJob(req.body);
      
      res.status(201).json({
        success: true,
        data: {
          jobId: job.id,
          name: job.name,
          status: job.status,
          nextRun: job.nextRun,
          createdAt: job.createdAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getJobs(req, res) {
    try {
      const { status, priority, limit = 50, offset = 0 } = req.query;
      
      let jobs = Array.from(this.scheduler.jobs.values());
      
      // Filter by status
      if (status) {
        jobs = jobs.filter(job => job.status === status);
      }
      
      // Filter by priority
      if (priority) {
        jobs = jobs.filter(job => job.priority === priority);
      }
      
      // Apply pagination
      const paginatedJobs = jobs.slice(
        parseInt(offset), 
        parseInt(offset) + parseInt(limit)
      );
      
      res.json({
        success: true,
        data: paginatedJobs,
        pagination: {
          limit: parseInt(limit),
          offset: parseInt(offset),
          total: jobs.length
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getJob(req, res) {
    try {
      const { jobId } = req.params;
      const job = this.scheduler.jobs.get(jobId);
      
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      
      res.json({
        success: true,
        data: job
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async executeJob(req, res) {
    try {
      const { jobId } = req.params;
      const { workerId } = req.body;
      
      const jobRun = await this.scheduler.executeJob(jobId, workerId);
      
      res.json({
        success: true,
        data: {
          runId: jobRun.id,
          jobId: jobRun.jobId,
          workerId: jobRun.workerId,
          status: jobRun.status,
          startedAt: jobRun.startedAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getJobStatus(req, res) {
    try {
      const { jobId } = req.params;
      const job = this.scheduler.jobs.get(jobId);
      
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      
      let currentRun = null;
      if (job.lastRun) {
        currentRun = this.scheduler.jobRuns.get(job.lastRun);
      }
      
      res.json({
        success: true,
        data: {
          jobId: job.id,
          status: job.status,
          currentRun: currentRun ? {
            runId: currentRun.id,
            startedAt: currentRun.startedAt,
            workerId: currentRun.workerId
          } : null,
          nextRun: job.nextRun,
          lastRun: job.lastRun
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getJobHistory(req, res) {
    try {
      const { jobId } = req.params;
      const { limit = 50, offset = 0 } = req.query;
      
      const jobRuns = Array.from(this.scheduler.jobRuns.values())
        .filter(run => run.jobId === jobId)
        .sort((a, b) => b.startedAt - a.startedAt);
      
      const paginatedRuns = jobRuns.slice(
        parseInt(offset), 
        parseInt(offset) + parseInt(limit)
      );
      
      res.json({
        success: true,
        data: paginatedRuns,
        pagination: {
          limit: parseInt(limit),
          offset: parseInt(offset),
          total: jobRuns.length
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async registerWorker(req, res) {
    try {
      const { workerId, capabilities } = req.body;
      
      const worker = this.scheduler.registerWorker(workerId, capabilities);
      
      res.status(201).json({
        success: true,
        data: {
          workerId: worker.id,
          capabilities: worker.capabilities,
          status: worker.status,
          registeredAt: worker.registeredAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async updateHeartbeat(req, res) {
    try {
      const { workerId } = req.params;
      
      this.scheduler.updateWorkerHeartbeat(workerId);
      
      res.json({
        success: true,
        message: "Heartbeat updated"
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async getDashboard(req, res) {
    try {
      const jobs = Array.from(this.scheduler.jobs.values());
      const workers = Array.from(this.scheduler.workers.values());
      const runs = Array.from(this.scheduler.jobRuns.values());
      
      const dashboard = {
        summary: {
          totalJobs: jobs.length,
          activeJobs: jobs.filter(job => job.status === "running").length,
          completedJobs: jobs.filter(job => job.status === "completed").length,
          failedJobs: jobs.filter(job => job.status === "failed").length,
          totalWorkers: workers.length,
          activeWorkers: workers.filter(w => w.status !== "offline").length
        },
        recentRuns: runs
          .sort((a, b) => b.startedAt - a.startedAt)
          .slice(0, 10),
        jobStats: {
          byStatus: this.groupBy(jobs, "status"),
          byPriority: this.groupBy(jobs, "priority"),
          byType: this.groupBy(jobs, "type")
        },
        workerStats: {
          byStatus: this.groupBy(workers, "status")
        }
      };
      
      res.json({
        success: true,
        data: dashboard
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  // Utility Methods
  groupBy(array, key) {
    return array.reduce((groups, item) => {
      const group = item[key];
      groups[group] = groups[group] || [];
      groups[group].push(item);
      return groups;
    }, {});
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Task Scheduler API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new TaskSchedulerAPI();
  api.start(3000);
}

module.exports = { TaskSchedulerAPI };
```

## Key Features

### Job Management
- **Multiple Triggers**: Cron expressions, intervals, and one-time execution
- **Priority Queues**: Different priority levels for job execution
- **Dependencies**: Job dependency management
- **Retry Logic**: Configurable retry policies

### Distributed Execution
- **Worker Management**: Multiple worker node support
- **Load Balancing**: Automatic job distribution
- **Fault Tolerance**: Worker failure detection and recovery
- **Heartbeat Monitoring**: Real-time worker status

### Monitoring & Analytics
- **Real-time Status**: Live job and worker monitoring
- **Performance Metrics**: Execution times and success rates
- **Job History**: Complete execution history
- **Dashboard**: Comprehensive monitoring dashboard

### Reliability & Performance
- **Fault Tolerance**: Handle worker failures gracefully
- **High Availability**: Multiple worker nodes
- **Scalability**: Support for millions of jobs
- **Efficient Scheduling**: Optimized job queue management

## Extension Ideas

### Advanced Features
1. **Job Dependencies**: Complex dependency graphs
2. **Job Templates**: Reusable job configurations
3. **Bulk Operations**: Mass job management
4. **Job Chaining**: Sequential job execution
5. **Conditional Execution**: Dynamic job scheduling

### Enterprise Features
1. **Multi-tenancy**: Isolated job environments
2. **Advanced Analytics**: Custom reports and metrics
3. **Integration APIs**: Third-party service integration
4. **Compliance**: Audit trails and logging
5. **Resource Management**: CPU and memory limits
