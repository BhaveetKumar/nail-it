# 🖥️ Operating System Concepts Guide - Node.js Perspective

> **Comprehensive guide to operating system concepts with Node.js examples and implementations**

## 🎯 **Overview**

This guide covers essential operating system concepts from a Node.js developer's perspective, including processes, threads, memory management, file systems, and system calls. Each concept is explained with practical Node.js examples and real-world applications.

## 📚 **Table of Contents**

1. [Process Management](#process-management/)
2. [Threading and Concurrency](#threading-and-concurrency/)
3. [Memory Management](#memory-management/)
4. [File System Operations](#file-system-operations/)
5. [System Calls and APIs](#system-calls-and-apis/)
6. [Inter-Process Communication](#inter-process-communication/)
7. [Scheduling and Performance](#scheduling-and-performance/)
8. [Security and Permissions](#security-and-permissions/)

---

## 🔄 **Process Management**

### **Process Creation and Management**

```javascript
// Process Management in Node.js
const { spawn, exec, fork } = require('child_process');
const os = require('os');
const process = require('process');

class ProcessManager {
    constructor() {
        this.processes = new Map();
        this.processId = process.pid;
        this.parentId = process.ppid;
    }
    
    // Create a new process using spawn
    createProcess(command, args = [], options = {}) {
        const childProcess = spawn(command, args, {
            stdio: 'pipe',
            ...options
        });
        
        const processInfo = {
            pid: childProcess.pid,
            command,
            args,
            startTime: Date.now(),
            status: 'running'
        };
        
        this.processes.set(childProcess.pid, processInfo);
        
        // Handle process events
        childProcess.on('exit', (code, signal) => {
            this.handleProcessExit(childProcess.pid, code, signal);
        });
        
        childProcess.on('error', (error) => {
            this.handleProcessError(childProcess.pid, error);
        });
        
        return childProcess;
    }
    
    // Execute a command and get output
    async executeCommand(command, options = {}) {
        return new Promise((resolve, reject) => {
            exec(command, options, (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                    return;
                }
                
                resolve({
                    stdout: stdout.toString(),
                    stderr: stderr.toString(),
                    exitCode: 0
                });
            });
        });
    }
    
    // Fork a new Node.js process
    forkProcess(modulePath, args = [], options = {}) {
        const childProcess = fork(modulePath, args, {
            silent: false,
            ...options
        });
        
        const processInfo = {
            pid: childProcess.pid,
            modulePath,
            args,
            startTime: Date.now(),
            status: 'running',
            type: 'forked'
        };
        
        this.processes.set(childProcess.pid, processInfo);
        
        return childProcess;
    }
    
    // Get process information
    getProcessInfo(pid) {
        return this.processes.get(pid);
    }
    
    // List all processes
    listProcesses() {
        return Array.from(this.processes.values());
    }
    
    // Kill a process
    killProcess(pid, signal = 'SIGTERM') {
        const processInfo = this.processes.get(pid);
        if (!processInfo) {
            throw new Error(`Process ${pid} not found`);
        }
        
        try {
            process.kill(pid, signal);
            processInfo.status = 'terminated';
            processInfo.endTime = Date.now();
        } catch (error) {
            throw new Error(`Failed to kill process ${pid}: ${error.message}`);
        }
    }
    
    handleProcessExit(pid, code, signal) {
        const processInfo = this.processes.get(pid);
        if (processInfo) {
            processInfo.status = 'exited';
            processInfo.exitCode = code;
            processInfo.signal = signal;
            processInfo.endTime = Date.now();
        }
    }
    
    handleProcessError(pid, error) {
        const processInfo = this.processes.get(pid);
        if (processInfo) {
            processInfo.status = 'error';
            processInfo.error = error.message;
            processInfo.endTime = Date.now();
        }
    }
}

// Usage example
async function demonstrateProcessManagement() {
    const pm = new ProcessManager();
    
    // Create a new process
    const child = pm.createProcess('node', ['-e', 'console.log("Hello from child process")']);
    
    // Wait for process to complete
    child.on('exit', (code) => {
        console.log(`Child process exited with code ${code}`);
    });
    
    // Execute a command
    try {
        const result = await pm.executeCommand('ls -la');
        console.log('Command output:', result.stdout);
    } catch (error) {
        console.error('Command failed:', error.message);
    }
    
    // Fork a Node.js process
    const forked = pm.forkProcess('./worker.js', ['arg1', 'arg2']);
    
    forked.on('message', (message) => {
        console.log('Message from forked process:', message);
    });
}
```

### **Process Monitoring and Statistics**

```javascript
// Process Monitoring
class ProcessMonitor {
    constructor() {
        this.startTime = Date.now();
        this.memoryUsage = process.memoryUsage();
        this.cpuUsage = process.cpuUsage();
    }
    
    // Get current process statistics
    getProcessStats() {
        const currentMemory = process.memoryUsage();
        const currentCpu = process.cpuUsage(this.cpuUsage);
        
        return {
            pid: process.pid,
            ppid: process.ppid,
            uptime: process.uptime(),
            memory: {
                rss: currentMemory.rss,
                heapTotal: currentMemory.heapTotal,
                heapUsed: currentMemory.heapUsed,
                external: currentMemory.external,
                arrayBuffers: currentMemory.arrayBuffers
            },
            cpu: {
                user: currentCpu.user,
                system: currentCpu.system
            },
            platform: process.platform,
            arch: process.arch,
            nodeVersion: process.version
        };
    }
    
    // Monitor memory usage over time
    startMemoryMonitoring(intervalMs = 5000) {
        return setInterval(() => {
            const stats = this.getProcessStats();
            console.log('Memory Usage:', {
                rss: `${Math.round(stats.memory.rss / 1024 / 1024)} MB`,
                heapUsed: `${Math.round(stats.memory.heapUsed / 1024 / 1024)} MB`,
                heapTotal: `${Math.round(stats.memory.heapTotal / 1024 / 1024)} MB`
            });
        }, intervalMs);
    }
    
    // Get system information
    getSystemInfo() {
        return {
            platform: os.platform(),
            arch: os.arch(),
            release: os.release(),
            hostname: os.hostname(),
            totalMemory: os.totalmem(),
            freeMemory: os.freemem(),
            cpus: os.cpus(),
            uptime: os.uptime(),
            loadAverage: os.loadavg()
        };
    }
}
```

---

## 🧵 **Threading and Concurrency**

### **Worker Threads in Node.js**

```javascript
// Worker Threads Implementation
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');

class ThreadManager {
    constructor() {
        this.workers = new Map();
        this.maxWorkers = os.cpus().length;
    }
    
    // Create a worker thread
    createWorker(workerScript, data = {}) {
        const worker = new Worker(workerScript, {
            workerData: data
        });
        
        const workerInfo = {
            id: worker.threadId,
            script: workerScript,
            data,
            startTime: Date.now(),
            status: 'running'
        };
        
        this.workers.set(worker.threadId, workerInfo);
        
        // Handle worker events
        worker.on('message', (message) => {
            this.handleWorkerMessage(worker.threadId, message);
        });
        
        worker.on('error', (error) => {
            this.handleWorkerError(worker.threadId, error);
        });
        
        worker.on('exit', (code) => {
            this.handleWorkerExit(worker.threadId, code);
        });
        
        return worker;
    }
    
    // Execute task in worker thread
    async executeInWorker(workerScript, taskData) {
        return new Promise((resolve, reject) => {
            const worker = this.createWorker(workerScript, taskData);
            
            worker.on('message', (result) => {
                resolve(result);
                worker.terminate();
            });
            
            worker.on('error', (error) => {
                reject(error);
            });
            
            worker.on('exit', (code) => {
                if (code !== 0) {
                    reject(new Error(`Worker stopped with exit code ${code}`));
                }
            });
        });
    }
    
    // Parallel processing with multiple workers
    async parallelProcess(data, workerScript, concurrency = this.maxWorkers) {
        const chunks = this.chunkArray(data, Math.ceil(data.length / concurrency));
        const promises = chunks.map(chunk => 
            this.executeInWorker(workerScript, { data: chunk })
        );
        
        const results = await Promise.all(promises);
        return results.flat();
    }
    
    chunkArray(array, chunkSize) {
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }
    
    handleWorkerMessage(workerId, message) {
        console.log(`Worker ${workerId} sent message:`, message);
    }
    
    handleWorkerError(workerId, error) {
        console.error(`Worker ${workerId} error:`, error);
        const workerInfo = this.workers.get(workerId);
        if (workerInfo) {
            workerInfo.status = 'error';
            workerInfo.error = error.message;
        }
    }
    
    handleWorkerExit(workerId, code) {
        console.log(`Worker ${workerId} exited with code ${code}`);
        const workerInfo = this.workers.get(workerId);
        if (workerInfo) {
            workerInfo.status = 'exited';
            workerInfo.exitCode = code;
            workerInfo.endTime = Date.now();
        }
    }
}

// Worker script example
if (isMainThread) {
    // Main thread code
    const tm = new ThreadManager();
    
    // Example: Process large array in parallel
    const largeArray = Array.from({ length: 1000000 }, (_, i) => i);
    
    tm.parallelProcess(largeArray, './worker.js', 4)
        .then(results => {
            console.log('Processing completed:', results.length);
        })
        .catch(error => {
            console.error('Processing failed:', error);
        });
} else {
    // Worker thread code
    const { workerData, parentPort } = require('worker_threads');
    
    // Process the data
    const processedData = workerData.data.map(item => item * 2);
    
    // Send result back to main thread
    parentPort.postMessage(processedData);
}
```

### **Event Loop and Asynchronous Programming**

```javascript
// Event Loop Understanding
class EventLoopMonitor {
    constructor() {
        this.phases = [
            'timers',
            'pending_callbacks',
            'idle_prepare',
            'poll',
            'check',
            'close_callbacks'
        ];
        this.phaseTimes = new Map();
    }
    
    // Monitor event loop lag
    monitorEventLoop() {
        const start = process.hrtime.bigint();
        
        setImmediate(() => {
            const lag = Number(process.hrtime.bigint() - start) / 1000000; // Convert to ms
            console.log(`Event loop lag: ${lag.toFixed(2)}ms`);
        });
    }
    
    // Demonstrate different phases
    demonstrateEventLoopPhases() {
        console.log('=== Event Loop Phase Demonstration ===');
        
        // Phase 1: Timers
        setTimeout(() => {
            console.log('1. Timer callback');
        }, 0);
        
        // Phase 2: I/O callbacks
        setImmediate(() => {
            console.log('2. setImmediate callback');
        });
        
        // Phase 3: Poll phase
        process.nextTick(() => {
            console.log('3. process.nextTick callback');
        });
        
        // Phase 4: Check phase
        setImmediate(() => {
            console.log('4. Another setImmediate callback');
        });
        
        console.log('5. Synchronous code');
    }
    
    // Monitor CPU usage
    monitorCPUUsage() {
        const startUsage = process.cpuUsage();
        
        setInterval(() => {
            const currentUsage = process.cpuUsage(startUsage);
            const totalTime = currentUsage.user + currentUsage.system;
            const cpuPercent = (totalTime / 1000000) * 100; // Convert to percentage
            
            console.log(`CPU Usage: ${cpuPercent.toFixed(2)}%`);
        }, 1000);
    }
}
```

---

## 💾 **Memory Management**

### **Memory Allocation and Garbage Collection**

```javascript
// Memory Management in Node.js
class MemoryManager {
    constructor() {
        this.memorySnapshots = [];
        this.gcEnabled = false;
    }
    
    // Enable garbage collection monitoring
    enableGCMonitoring() {
        if (global.gc) {
            this.gcEnabled = true;
            console.log('Garbage collection monitoring enabled');
        } else {
            console.log('Run with --expose-gc flag to enable GC monitoring');
        }
    }
    
    // Take memory snapshot
    takeMemorySnapshot(label = '') {
        const snapshot = {
            label,
            timestamp: Date.now(),
            memory: process.memoryUsage(),
            heap: this.getHeapStatistics()
        };
        
        this.memorySnapshots.push(snapshot);
        return snapshot;
    }
    
    // Get heap statistics
    getHeapStatistics() {
        if (global.gc) {
            global.gc();
        }
        
        return {
            totalHeapSize: process.memoryUsage().heapTotal,
            usedHeapSize: process.memoryUsage().heapUsed,
            heapSizeLimit: process.memoryUsage().heapTotal * 2, // Approximate
            mallocedMemory: process.memoryUsage().external
        };
    }
    
    // Monitor memory leaks
    monitorMemoryLeaks(intervalMs = 5000) {
        return setInterval(() => {
            const snapshot = this.takeMemorySnapshot('leak_check');
            
            if (this.memorySnapshots.length > 1) {
                const previous = this.memorySnapshots[this.memorySnapshots.length - 2];
                const current = snapshot;
                
                const memoryIncrease = current.memory.heapUsed - previous.memory.heapUsed;
                const timeIncrease = current.timestamp - previous.timestamp;
                
                if (memoryIncrease > 0 && timeIncrease > 0) {
                    const leakRate = memoryIncrease / timeIncrease; // bytes per ms
                    console.log(`Memory leak rate: ${(leakRate * 1000).toFixed(2)} bytes/second`);
                }
            }
        }, intervalMs);
    }
    
    // Force garbage collection
    forceGC() {
        if (global.gc) {
            global.gc();
            console.log('Garbage collection forced');
        } else {
            console.log('GC not available. Run with --expose-gc flag');
        }
    }
    
    // Memory optimization techniques
    optimizeMemory() {
        // Clear unused references
        this.memorySnapshots = this.memorySnapshots.slice(-10); // Keep only last 10
        
        // Force garbage collection
        this.forceGC();
        
        console.log('Memory optimization completed');
    }
}

// Memory leak detection
class MemoryLeakDetector {
    constructor() {
        this.allocations = new Map();
        this.interval = null;
    }
    
    startTracking() {
        this.interval = setInterval(() => {
            const memory = process.memoryUsage();
            const timestamp = Date.now();
            
            this.allocations.set(timestamp, {
                rss: memory.rss,
                heapUsed: memory.heapUsed,
                heapTotal: memory.heapTotal
            });
            
            // Keep only last 100 measurements
            if (this.allocations.size > 100) {
                const firstKey = this.allocations.keys().next().value;
                this.allocations.delete(firstKey);
            }
            
            this.analyzeTrend();
        }, 1000);
    }
    
    analyzeTrend() {
        if (this.allocations.size < 10) return;
        
        const values = Array.from(this.allocations.values());
        const recent = values.slice(-5);
        const older = values.slice(-10, -5);
        
        const recentAvg = recent.reduce((sum, v) => sum + v.heapUsed, 0) / recent.length;
        const olderAvg = older.reduce((sum, v) => sum + v.heapUsed, 0) / older.length;
        
        const increase = recentAvg - olderAvg;
        const increasePercent = (increase / olderAvg) * 100;
        
        if (increasePercent > 10) {
            console.warn(`Potential memory leak detected: ${increasePercent.toFixed(2)}% increase`);
        }
    }
    
    stopTracking() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }
}
```

---

## 📁 **File System Operations**

### **File System Management**

```javascript
// File System Operations
const fs = require('fs').promises;
const path = require('path');
const { watch } = require('fs');

class FileSystemManager {
    constructor() {
        this.watchers = new Map();
        this.fileHandles = new Map();
    }
    
    // Read file with error handling
    async readFile(filePath, options = {}) {
        try {
            const data = await fs.readFile(filePath, options);
            return {
                success: true,
                data,
                size: data.length,
                path: filePath
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                path: filePath
            };
        }
    }
    
    // Write file with atomic operation
    async writeFile(filePath, data, options = {}) {
        const tempPath = filePath + '.tmp';
        
        try {
            // Write to temporary file first
            await fs.writeFile(tempPath, data, options);
            
            // Atomic rename
            await fs.rename(tempPath, filePath);
            
            return {
                success: true,
                path: filePath,
                size: data.length
            };
        } catch (error) {
            // Clean up temp file if it exists
            try {
                await fs.unlink(tempPath);
            } catch (cleanupError) {
                // Ignore cleanup errors
            }
            
            return {
                success: false,
                error: error.message,
                path: filePath
            };
        }
    }
    
    // Watch file for changes
    watchFile(filePath, callback) {
        const watcher = watch(filePath, (eventType, filename) => {
            callback({
                eventType,
                filename,
                path: filePath,
                timestamp: Date.now()
            });
        });
        
        this.watchers.set(filePath, watcher);
        return watcher;
    }
    
    // Stop watching file
    unwatchFile(filePath) {
        const watcher = this.watchers.get(filePath);
        if (watcher) {
            watcher.close();
            this.watchers.delete(filePath);
        }
    }
    
    // Get file statistics
    async getFileStats(filePath) {
        try {
            const stats = await fs.stat(filePath);
            return {
                success: true,
                stats: {
                    size: stats.size,
                    isFile: stats.isFile(),
                    isDirectory: stats.isDirectory(),
                    isSymbolicLink: stats.isSymbolicLink(),
                    atime: stats.atime,
                    mtime: stats.mtime,
                    ctime: stats.ctime,
                    birthtime: stats.birthtime,
                    mode: stats.mode,
                    uid: stats.uid,
                    gid: stats.gid
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    // Directory operations
    async createDirectory(dirPath, recursive = true) {
        try {
            await fs.mkdir(dirPath, { recursive });
            return { success: true, path: dirPath };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async listDirectory(dirPath) {
        try {
            const entries = await fs.readdir(dirPath, { withFileTypes: true });
            return {
                success: true,
                entries: entries.map(entry => ({
                    name: entry.name,
                    isFile: entry.isFile(),
                    isDirectory: entry.isDirectory(),
                    isSymbolicLink: entry.isSymbolicLink()
                }))
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // File streaming
    async streamFile(inputPath, outputPath) {
        const { createReadStream, createWriteStream } = require('fs');
        const { pipeline } = require('stream/promises');
        
        try {
            const readStream = createReadStream(inputPath);
            const writeStream = createWriteStream(outputPath);
            
            await pipeline(readStream, writeStream);
            
            return { success: true, inputPath, outputPath };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
}
```

---

## 🔧 **System Calls and APIs**

### **System Information and Monitoring**

```javascript
// System Information and Monitoring
const os = require('os');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class SystemMonitor {
    constructor() {
        this.startTime = Date.now();
        this.metrics = {
            cpu: [],
            memory: [],
            network: [],
            disk: []
        };
    }
    
    // Get comprehensive system information
    async getSystemInfo() {
        return {
            platform: os.platform(),
            arch: os.arch(),
            release: os.release(),
            hostname: os.hostname(),
            uptime: os.uptime(),
            loadAverage: os.loadavg(),
            totalMemory: os.totalmem(),
            freeMemory: os.freemem(),
            cpus: os.cpus(),
            networkInterfaces: os.networkInterfaces(),
            userInfo: os.userInfo(),
            tmpdir: os.tmpdir(),
            homedir: os.homedir()
        };
    }
    
    // Monitor CPU usage
    async getCPUUsage() {
        const cpus = os.cpus();
        const totalIdle = cpus.reduce((acc, cpu) => acc + cpu.times.idle, 0);
        const totalTick = cpus.reduce((acc, cpu) => {
            return acc + Object.values(cpu.times).reduce((a, b) => a + b, 0);
        }, 0);
        
        const idle = totalIdle / cpus.length;
        const total = totalTick / cpus.length;
        const usage = 100 - ~~(100 * idle / total);
        
        return {
            usage: usage,
            cores: cpus.length,
            loadAverage: os.loadavg(),
            cpus: cpus.map(cpu => ({
                model: cpu.model,
                speed: cpu.speed,
                times: cpu.times
            }))
        };
    }
    
    // Monitor memory usage
    getMemoryUsage() {
        const total = os.totalmem();
        const free = os.freemem();
        const used = total - free;
        
        return {
            total: this.formatBytes(total),
            used: this.formatBytes(used),
            free: this.formatBytes(free),
            usagePercent: ((used / total) * 100).toFixed(2)
        };
    }
    
    // Monitor disk usage
    async getDiskUsage() {
        try {
            const { stdout } = await execAsync('df -h');
            const lines = stdout.trim().split('\n').slice(1);
            
            return lines.map(line => {
                const parts = line.split(/\s+/);
                return {
                    filesystem: parts[0],
                    size: parts[1],
                    used: parts[2],
                    available: parts[3],
                    usagePercent: parts[4],
                    mounted: parts[5]
                };
            });
        } catch (error) {
            console.error('Error getting disk usage:', error.message);
            return [];
        }
    }
    
    // Monitor network interfaces
    getNetworkInfo() {
        const interfaces = os.networkInterfaces();
        const networkInfo = {};
        
        Object.keys(interfaces).forEach(name => {
            networkInfo[name] = interfaces[name].map(iface => ({
                address: iface.address,
                netmask: iface.netmask,
                family: iface.family,
                mac: iface.mac,
                internal: iface.internal
            }));
        });
        
        return networkInfo;
    }
    
    // Start continuous monitoring
    startMonitoring(intervalMs = 5000) {
        return setInterval(async () => {
            const timestamp = Date.now();
            
            // CPU monitoring
            const cpuUsage = await this.getCPUUsage();
            this.metrics.cpu.push({ timestamp, ...cpuUsage });
            
            // Memory monitoring
            const memoryUsage = this.getMemoryUsage();
            this.metrics.memory.push({ timestamp, ...memoryUsage });
            
            // Keep only last 100 measurements
            Object.keys(this.metrics).forEach(key => {
                if (this.metrics[key].length > 100) {
                    this.metrics[key] = this.metrics[key].slice(-100);
                }
            });
            
            console.log('System Metrics:', {
                cpu: `${cpuUsage.usage}%`,
                memory: memoryUsage.usagePercent + '%',
                load: cpuUsage.loadAverage[0].toFixed(2)
            });
        }, intervalMs);
    }
    
    // Get performance metrics
    getPerformanceMetrics() {
        const processMemory = process.memoryUsage();
        const cpuUsage = process.cpuUsage();
        
        return {
            process: {
                pid: process.pid,
                uptime: process.uptime(),
                memory: {
                    rss: this.formatBytes(processMemory.rss),
                    heapTotal: this.formatBytes(processMemory.heapTotal),
                    heapUsed: this.formatBytes(processMemory.heapUsed),
                    external: this.formatBytes(processMemory.external)
                },
                cpu: {
                    user: cpuUsage.user,
                    system: cpuUsage.system
                }
            },
            system: {
                memory: this.getMemoryUsage(),
                loadAverage: os.loadavg()
            }
        };
    }
    
    formatBytes(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }
}
```

---

## 🔗 **Inter-Process Communication**

### **IPC Mechanisms**

```javascript
// Inter-Process Communication
const { IPCChannel } = require('child_process');
const net = require('net');
const dgram = require('dgram');

class IPCManager {
    constructor() {
        this.servers = new Map();
        this.clients = new Map();
    }
    
    // TCP Server for IPC
    createTCPServer(port, onConnection) {
        const server = net.createServer((socket) => {
            console.log('Client connected');
            
            socket.on('data', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    onConnection(message, socket);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            });
            
            socket.on('end', () => {
                console.log('Client disconnected');
            });
            
            socket.on('error', (error) => {
                console.error('Socket error:', error);
            });
        });
        
        server.listen(port, () => {
            console.log(`TCP server listening on port ${port}`);
        });
        
        this.servers.set(port, server);
        return server;
    }
    
    // TCP Client for IPC
    createTCPClient(port, host = 'localhost') {
        const client = net.createConnection(port, host, () => {
            console.log('Connected to server');
        });
        
        client.on('data', (data) => {
            try {
                const message = JSON.parse(data.toString());
                console.log('Received:', message);
            } catch (error) {
                console.error('Error parsing response:', error);
            }
        });
        
        client.on('end', () => {
            console.log('Disconnected from server');
        });
        
        client.on('error', (error) => {
            console.error('Client error:', error);
        });
        
        this.clients.set(`${host}:${port}`, client);
        return client;
    }
    
    // Send message via TCP
    sendTCPMessage(client, message) {
        const data = JSON.stringify(message);
        client.write(data);
    }
    
    // UDP Server for IPC
    createUDPServer(port, onMessage) {
        const server = dgram.createSocket('udp4');
        
        server.on('message', (msg, rinfo) => {
            try {
                const message = JSON.parse(msg.toString());
                onMessage(message, rinfo);
            } catch (error) {
                console.error('Error parsing UDP message:', error);
            }
        });
        
        server.on('listening', () => {
            const address = server.address();
            console.log(`UDP server listening on ${address.address}:${address.port}`);
        });
        
        server.bind(port);
        this.servers.set(`udp:${port}`, server);
        return server;
    }
    
    // UDP Client for IPC
    createUDPClient() {
        const client = dgram.createSocket('udp4');
        this.clients.set('udp', client);
        return client;
    }
    
    // Send UDP message
    sendUDPMessage(client, message, port, host = 'localhost') {
        const data = JSON.stringify(message);
        client.send(data, port, host, (error) => {
            if (error) {
                console.error('UDP send error:', error);
            }
        });
    }
    
    // Named Pipes (Windows) / Unix Domain Sockets
    createNamedPipeServer(pipeName, onConnection) {
        const server = net.createServer((socket) => {
            onConnection(socket);
        });
        
        server.listen(pipeName, () => {
            console.log(`Named pipe server listening on ${pipeName}`);
        });
        
        this.servers.set(pipeName, server);
        return server;
    }
    
    // Cleanup resources
    cleanup() {
        this.servers.forEach(server => server.close());
        this.clients.forEach(client => client.close());
        this.servers.clear();
        this.clients.clear();
    }
}
```

---

## ⚡ **Scheduling and Performance**

### **Process Scheduling**

```javascript
// Process Scheduling and Performance
class ProcessScheduler {
    constructor() {
        this.tasks = [];
        this.running = false;
        this.quantum = 100; // Time quantum in ms
    }
    
    // Add task to scheduler
    addTask(task) {
        const taskInfo = {
            id: task.id || this.generateTaskId(),
            priority: task.priority || 0,
            burstTime: task.burstTime || 1000,
            arrivalTime: Date.now(),
            remainingTime: task.burstTime || 1000,
            status: 'ready',
            startTime: null,
            endTime: null,
            waitTime: 0,
            turnaroundTime: 0,
            execute: task.execute
        };
        
        this.tasks.push(taskInfo);
        this.sortTasksByPriority();
        
        if (!this.running) {
            this.startScheduler();
        }
    }
    
    // Round Robin Scheduling
    roundRobin() {
        if (this.tasks.length === 0) return;
        
        const readyTasks = this.tasks.filter(task => task.status === 'ready');
        if (readyTasks.length === 0) return;
        
        const currentTask = readyTasks[0];
        
        if (currentTask.status === 'ready') {
            currentTask.status = 'running';
            currentTask.startTime = currentTask.startTime || Date.now();
        }
        
        // Execute task for quantum time
        const executionTime = Math.min(this.quantum, currentTask.remainingTime);
        
        setTimeout(() => {
            currentTask.remainingTime -= executionTime;
            
            if (currentTask.remainingTime <= 0) {
                // Task completed
                currentTask.status = 'completed';
                currentTask.endTime = Date.now();
                currentTask.turnaroundTime = currentTask.endTime - currentTask.arrivalTime;
                currentTask.waitTime = currentTask.turnaroundTime - currentTask.burstTime;
                
                console.log(`Task ${currentTask.id} completed`);
            } else {
                // Task preempted
                currentTask.status = 'ready';
                console.log(`Task ${currentTask.id} preempted`);
            }
            
            // Continue scheduling
            this.roundRobin();
        }, executionTime);
    }
    
    // Priority Scheduling
    priorityScheduling() {
        const readyTasks = this.tasks.filter(task => task.status === 'ready');
        if (readyTasks.length === 0) return;
        
        // Sort by priority (higher number = higher priority)
        readyTasks.sort((a, b) => b.priority - a.priority);
        
        const currentTask = readyTasks[0];
        currentTask.status = 'running';
        currentTask.startTime = currentTask.startTime || Date.now();
        
        // Execute until completion
        setTimeout(() => {
            currentTask.status = 'completed';
            currentTask.endTime = Date.now();
            currentTask.turnaroundTime = currentTask.endTime - currentTask.arrivalTime;
            currentTask.waitTime = currentTask.turnaroundTime - currentTask.burstTime;
            
            console.log(`Task ${currentTask.id} completed`);
            this.priorityScheduling();
        }, currentTask.remainingTime);
    }
    
    // First Come First Served
    fcfs() {
        const readyTasks = this.tasks.filter(task => task.status === 'ready');
        if (readyTasks.length === 0) return;
        
        // Sort by arrival time
        readyTasks.sort((a, b) => a.arrivalTime - b.arrivalTime);
        
        const currentTask = readyTasks[0];
        currentTask.status = 'running';
        currentTask.startTime = currentTask.startTime || Date.now();
        
        setTimeout(() => {
            currentTask.status = 'completed';
            currentTask.endTime = Date.now();
            currentTask.turnaroundTime = currentTask.endTime - currentTask.arrivalTime;
            currentTask.waitTime = currentTask.turnaroundTime - currentTask.burstTime;
            
            console.log(`Task ${currentTask.id} completed`);
            this.fcfs();
        }, currentTask.remainingTime);
    }
    
    startScheduler(algorithm = 'roundRobin') {
        this.running = true;
        
        switch (algorithm) {
            case 'roundRobin':
                this.roundRobin();
                break;
            case 'priority':
                this.priorityScheduling();
                break;
            case 'fcfs':
                this.fcfs();
                break;
        }
    }
    
    sortTasksByPriority() {
        this.tasks.sort((a, b) => b.priority - a.priority);
    }
    
    generateTaskId() {
        return 'task_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    getStatistics() {
        const completedTasks = this.tasks.filter(task => task.status === 'completed');
        
        if (completedTasks.length === 0) {
            return { message: 'No completed tasks' };
        }
        
        const totalWaitTime = completedTasks.reduce((sum, task) => sum + task.waitTime, 0);
        const totalTurnaroundTime = completedTasks.reduce((sum, task) => sum + task.turnaroundTime, 0);
        
        return {
            totalTasks: completedTasks.length,
            averageWaitTime: totalWaitTime / completedTasks.length,
            averageTurnaroundTime: totalTurnaroundTime / completedTasks.length,
            tasks: completedTasks.map(task => ({
                id: task.id,
                waitTime: task.waitTime,
                turnaroundTime: task.turnaroundTime,
                burstTime: task.burstTime
            }))
        };
    }
}
```

---

## 🔒 **Security and Permissions**

### **File Permissions and Security**

```javascript
// Security and Permissions
const fs = require('fs').promises;
const path = require('path');

class SecurityManager {
    constructor() {
        this.permissions = new Map();
    }
    
    // Check file permissions
    async checkFilePermissions(filePath) {
        try {
            const stats = await fs.stat(filePath);
            const mode = stats.mode;
            
            return {
                readable: !!(mode & 0o400),
                writable: !!(mode & 0o200),
                executable: !!(mode & 0o100),
                owner: {
                    read: !!(mode & 0o400),
                    write: !!(mode & 0o200),
                    execute: !!(mode & 0o100)
                },
                group: {
                    read: !!(mode & 0o040),
                    write: !!(mode & 0o020),
                    execute: !!(mode & 0o010)
                },
                others: {
                    read: !!(mode & 0o004),
                    write: !!(mode & 0o002),
                    execute: !!(mode & 0o001)
                },
                mode: mode.toString(8)
            };
        } catch (error) {
            return { error: error.message };
        }
    }
    
    // Set file permissions
    async setFilePermissions(filePath, mode) {
        try {
            await fs.chmod(filePath, mode);
            return { success: true, mode: mode.toString(8) };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Validate file path for security
    validateFilePath(filePath, basePath = process.cwd()) {
        const resolvedPath = path.resolve(filePath);
        const resolvedBase = path.resolve(basePath);
        
        if (!resolvedPath.startsWith(resolvedBase)) {
            return {
                valid: false,
                error: 'Path traversal detected'
            };
        }
        
        return { valid: true, path: resolvedPath };
    }
    
    // Sanitize filename
    sanitizeFilename(filename) {
        // Remove dangerous characters
        const sanitized = filename
            .replace(/[^a-zA-Z0-9._-]/g, '_')
            .replace(/\.{2,}/g, '.')
            .replace(/^\.+|\.+$/g, '');
        
        return sanitized || 'unnamed';
    }
    
    // Check if file is safe to process
    async isFileSafe(filePath) {
        try {
            const stats = await fs.stat(filePath);
            
            // Check file size (max 100MB)
            if (stats.size > 100 * 1024 * 1024) {
                return { safe: false, reason: 'File too large' };
            }
            
            // Check if it's a regular file
            if (!stats.isFile()) {
                return { safe: false, reason: 'Not a regular file' };
            }
            
            // Check file extension
            const ext = path.extname(filePath).toLowerCase();
            const dangerousExts = ['.exe', '.bat', '.cmd', '.sh', '.ps1'];
            
            if (dangerousExts.includes(ext)) {
                return { safe: false, reason: 'Potentially dangerous file type' };
            }
            
            return { safe: true };
        } catch (error) {
            return { safe: false, reason: error.message };
        }
    }
    
    // Create secure temporary file
    async createSecureTempFile(data, options = {}) {
        const crypto = require('crypto');
        const os = require('os');
        
        const tempDir = options.tempDir || os.tmpdir();
        const prefix = options.prefix || 'secure_';
        const suffix = options.suffix || '.tmp';
        
        const randomName = crypto.randomBytes(16).toString('hex');
        const filename = this.sanitizeFilename(prefix + randomName + suffix);
        const filePath = path.join(tempDir, filename);
        
        try {
            await fs.writeFile(filePath, data, { mode: 0o600 }); // Read/write for owner only
            return { success: true, path: filePath };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    // Clean up secure temporary files
    async cleanupTempFiles(filePaths) {
        const results = [];
        
        for (const filePath of filePaths) {
            try {
                await fs.unlink(filePath);
                results.push({ path: filePath, success: true });
            } catch (error) {
                results.push({ path: filePath, success: false, error: error.message });
            }
        }
        
        return results;
    }
}

// Usage examples
async function demonstrateOSConcepts() {
    console.log('=== Operating System Concepts Demo ===');
    
    // Process Management
    const pm = new ProcessManager();
    const child = pm.createProcess('node', ['-e', 'console.log("Hello from child")']);
    
    // Memory Management
    const mm = new MemoryManager();
    mm.enableGCMonitoring();
    const snapshot = mm.takeMemorySnapshot('demo');
    console.log('Memory snapshot:', snapshot);
    
    // File System Operations
    const fsm = new FileSystemManager();
    const result = await fsm.readFile('./package.json');
    console.log('File read result:', result.success);
    
    // System Monitoring
    const sm = new SystemMonitor();
    const systemInfo = await sm.getSystemInfo();
    console.log('System info:', {
        platform: systemInfo.platform,
        arch: systemInfo.arch,
        totalMemory: sm.formatBytes(systemInfo.totalMemory)
    });
    
    // Security
    const sec = new SecurityManager();
    const permissions = await sec.checkFilePermissions('./package.json');
    console.log('File permissions:', permissions);
}

// Export for use in other modules
module.exports = {
    ProcessManager,
    ProcessMonitor,
    ThreadManager,
    MemoryManager,
    FileSystemManager,
    SystemMonitor,
    IPCManager,
    ProcessScheduler,
    SecurityManager
};

// Run demo if this file is executed directly
if (require.main === module) {
    demonstrateOSConcepts().catch(console.error);
}
```

---

## 🎯 **Key Takeaways**

### **Process Management**
- Use `spawn`, `exec`, and `fork` for different process creation needs
- Monitor process lifecycle with event handlers
- Implement proper error handling and cleanup

### **Memory Management**
- Monitor memory usage with `process.memoryUsage()`
- Use garbage collection monitoring for leak detection
- Implement memory optimization strategies

### **File System Operations**
- Use atomic operations for file writes
- Implement proper error handling
- Monitor file changes with watchers

### **System Monitoring**
- Track CPU, memory, and disk usage
- Implement performance metrics collection
- Use system information for optimization

### **Security Best Practices**
- Validate file paths to prevent traversal attacks
- Sanitize filenames and user input
- Implement proper file permissions
- Use secure temporary file handling

---

**🎉 This comprehensive guide covers all essential operating system concepts from a Node.js perspective!**
