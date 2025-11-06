---
# Auto-generated front matter
Title: Database Replication
LastUpdated: 2025-11-06T20:45:58.288939
Tags: []
Status: draft
---

# Database Replication - High Availability and Scalability

## Overview

Database replication is the process of creating and maintaining multiple copies of a database to improve availability, performance, and fault tolerance. It's essential for building scalable and reliable systems.

## Key Concepts

- **Master-Slave Replication**: One master database with multiple read-only slaves
- **Master-Master Replication**: Multiple master databases that can accept writes
- **Synchronous Replication**: Writes are confirmed only after all replicas are updated
- **Asynchronous Replication**: Writes are confirmed before all replicas are updated
- **Read Replicas**: Read-only copies used for scaling read operations
- **Failover**: Automatic switching to a backup when primary fails

## Replication Types

### 1. Master-Slave Replication
- Single master accepts writes
- Multiple slaves handle reads
- Simple to implement
- Single point of failure for writes

### 2. Master-Master Replication
- Multiple masters accept writes
- More complex to implement
- Better write availability
- Risk of conflicts

### 3. Chain Replication
- Data flows through a chain of replicas
- Good for wide-area networks
- Higher latency
- More fault-tolerant

## Database Replication Architecture

```mermaid
graph TD
    A[Master Database] --> B[Slave 1]
    A --> C[Slave 2]
    A --> D[Slave 3]
    
    E[Application] --> A
    E --> B
    E --> C
    E --> D
    
    F[Load Balancer] --> E
    
    G[Replication Log] --> A
    G --> B
    G --> C
    G --> D
```

## Go Implementation

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "sync"
    "time"
    
    _ "github.com/lib/pq"
)

// ReplicationConfig represents replication configuration
type ReplicationConfig struct {
    MasterDSN     string
    SlaveDSNs     []string
    SyncMode      bool
    MaxRetries    int
    RetryInterval time.Duration
}

// DatabaseReplication manages database replication
type DatabaseReplication struct {
    config     *ReplicationConfig
    master     *sql.DB
    slaves     []*sql.DB
    mutex      sync.RWMutex
    isHealthy  bool
    lastSync   time.Time
}

// NewDatabaseReplication creates a new database replication manager
func NewDatabaseReplication(config *ReplicationConfig) (*DatabaseReplication, error) {
    // Connect to master
    master, err := sql.Open("postgres", config.MasterDSN)
    if err != nil {
        return nil, fmt.Errorf("failed to connect to master: %v", err)
    }
    
    // Test master connection
    if err := master.Ping(); err != nil {
        return nil, fmt.Errorf("failed to ping master: %v", err)
    }
    
    // Connect to slaves
    var slaves []*sql.DB
    for _, dsn := range config.SlaveDSNs {
        slave, err := sql.Open("postgres", dsn)
        if err != nil {
            log.Printf("Warning: failed to connect to slave %s: %v", dsn, err)
            continue
        }
        
        if err := slave.Ping(); err != nil {
            log.Printf("Warning: failed to ping slave %s: %v", dsn, err)
            continue
        }
        
        slaves = append(slaves, slave)
    }
    
    if len(slaves) == 0 {
        return nil, fmt.Errorf("no healthy slaves available")
    }
    
    return &DatabaseReplication{
        config:    config,
        master:    master,
        slaves:    slaves,
        isHealthy: true,
    }, nil
}

// Write writes data to the master database
func (dr *DatabaseReplication) Write(ctx context.Context, query string, args ...interface{}) error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    if !dr.isHealthy {
        return fmt.Errorf("replication is not healthy")
    }
    
    // Execute on master
    result, err := dr.master.ExecContext(ctx, query, args...)
    if err != nil {
        return fmt.Errorf("failed to execute on master: %v", err)
    }
    
    // Get affected rows
    affected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to get affected rows: %v", err)
    }
    
    log.Printf("Write executed on master, affected rows: %d", affected)
    
    // If synchronous mode, wait for slaves
    if dr.config.SyncMode {
        return dr.syncToSlaves(ctx, query, args...)
    }
    
    // If asynchronous mode, sync in background
    go func() {
        if err := dr.syncToSlaves(context.Background(), query, args...); err != nil {
            log.Printf("Failed to sync to slaves: %v", err)
        }
    }()
    
    return nil
}

// Read reads data from a slave database
func (dr *DatabaseReplication) Read(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    
    if !dr.isHealthy {
        return nil, fmt.Errorf("replication is not healthy")
    }
    
    // Select a random slave for read
    if len(dr.slaves) == 0 {
        return nil, fmt.Errorf("no slaves available")
    }
    
    slave := dr.slaves[0] // In production, use load balancing
    
    rows, err := slave.QueryContext(ctx, query, args...)
    if err != nil {
        return nil, fmt.Errorf("failed to query slave: %v", err)
    }
    
    return rows, nil
}

// syncToSlaves syncs data to all slave databases
func (dr *DatabaseReplication) syncToSlaves(ctx context.Context, query string, args ...interface{}) error {
    var wg sync.WaitGroup
    errors := make(chan error, len(dr.slaves))
    
    for i, slave := range dr.slaves {
        wg.Add(1)
        go func(index int, s *sql.DB) {
            defer wg.Done()
            
            if err := dr.syncToSlave(ctx, s, query, args...); err != nil {
                errors <- fmt.Errorf("slave %d sync failed: %v", index, err)
            }
        }(i, slave)
    }
    
    wg.Wait()
    close(errors)
    
    // Check for errors
    var syncErrors []error
    for err := range errors {
        syncErrors = append(syncErrors, err)
    }
    
    if len(syncErrors) > 0 {
        return fmt.Errorf("sync errors: %v", syncErrors)
    }
    
    dr.lastSync = time.Now()
    return nil
}

// syncToSlave syncs data to a specific slave
func (dr *DatabaseReplication) syncToSlave(ctx context.Context, slave *sql.DB, query string, args ...interface{}) error {
    for attempt := 0; attempt < dr.config.MaxRetries; attempt++ {
        if err := slave.Ping(); err != nil {
            log.Printf("Slave ping failed, attempt %d: %v", attempt+1, err)
            time.Sleep(dr.config.RetryInterval)
            continue
        }
        
        result, err := slave.ExecContext(ctx, query, args...)
        if err != nil {
            log.Printf("Slave exec failed, attempt %d: %v", attempt+1, err)
            time.Sleep(dr.config.RetryInterval)
            continue
        }
        
        affected, err := result.RowsAffected()
        if err != nil {
            log.Printf("Slave rows affected failed, attempt %d: %v", attempt+1, err)
            time.Sleep(dr.config.RetryInterval)
            continue
        }
        
        log.Printf("Slave sync successful, affected rows: %d", affected)
        return nil
    }
    
    return fmt.Errorf("failed to sync to slave after %d attempts", dr.config.MaxRetries)
}

// HealthCheck checks the health of all databases
func (dr *DatabaseReplication) HealthCheck() error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    // Check master
    if err := dr.master.Ping(); err != nil {
        dr.isHealthy = false
        return fmt.Errorf("master is unhealthy: %v", err)
    }
    
    // Check slaves
    healthySlaves := 0
    for i, slave := range dr.slaves {
        if err := slave.Ping(); err != nil {
            log.Printf("Slave %d is unhealthy: %v", i, err)
            continue
        }
        healthySlaves++
    }
    
    if healthySlaves == 0 {
        dr.isHealthy = false
        return fmt.Errorf("no healthy slaves available")
    }
    
    dr.isHealthy = true
    return nil
}

// GetReplicationLag returns the replication lag
func (dr *DatabaseReplication) GetReplicationLag() (time.Duration, error) {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    
    if !dr.isHealthy {
        return 0, fmt.Errorf("replication is not healthy")
    }
    
    // Get master timestamp
    var masterTime time.Time
    err := dr.master.QueryRow("SELECT NOW()").Scan(&masterTime)
    if err != nil {
        return 0, fmt.Errorf("failed to get master time: %v", err)
    }
    
    // Get slave timestamp
    var slaveTime time.Time
    err = dr.slaves[0].QueryRow("SELECT NOW()").Scan(&slaveTime)
    if err != nil {
        return 0, fmt.Errorf("failed to get slave time: %v", err)
    }
    
    return masterTime.Sub(slaveTime), nil
}

// PromoteSlave promotes a slave to master
func (dr *DatabaseReplication) PromoteSlave(slaveIndex int) error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    if slaveIndex >= len(dr.slaves) {
        return fmt.Errorf("invalid slave index: %d", slaveIndex)
    }
    
    // Promote slave to master
    oldMaster := dr.master
    newMaster := dr.slaves[slaveIndex]
    
    // Update master
    dr.master = newMaster
    
    // Update slaves list
    dr.slaves = append(dr.slaves[:slaveIndex], dr.slaves[slaveIndex+1:]...)
    dr.slaves = append(dr.slaves, oldMaster)
    
    log.Printf("Slave %d promoted to master", slaveIndex)
    return nil
}

// AddSlave adds a new slave to the replication
func (dr *DatabaseReplication) AddSlave(dsn string) error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    slave, err := sql.Open("postgres", dsn)
    if err != nil {
        return fmt.Errorf("failed to connect to new slave: %v", err)
    }
    
    if err := slave.Ping(); err != nil {
        return fmt.Errorf("failed to ping new slave: %v", err)
    }
    
    dr.slaves = append(dr.slaves, slave)
    log.Printf("New slave added: %s", dsn)
    
    return nil
}

// RemoveSlave removes a slave from the replication
func (dr *DatabaseReplication) RemoveSlave(slaveIndex int) error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    if slaveIndex >= len(dr.slaves) {
        return fmt.Errorf("invalid slave index: %d", slaveIndex)
    }
    
    // Close the slave connection
    if err := dr.slaves[slaveIndex].Close(); err != nil {
        log.Printf("Warning: failed to close slave %d: %v", slaveIndex, err)
    }
    
    // Remove from slaves list
    dr.slaves = append(dr.slaves[:slaveIndex], dr.slaves[slaveIndex+1:]...)
    
    log.Printf("Slave %d removed", slaveIndex)
    return nil
}

// GetReplicationStatus returns the current replication status
func (dr *DatabaseReplication) GetReplicationStatus() map[string]interface{} {
    dr.mutex.RLock()
    defer dr.mutex.RUnlock()
    
    status := map[string]interface{}{
        "is_healthy":    dr.isHealthy,
        "last_sync":     dr.lastSync,
        "slave_count":   len(dr.slaves),
        "sync_mode":     dr.config.SyncMode,
    }
    
    // Get replication lag
    if lag, err := dr.GetReplicationLag(); err == nil {
        status["replication_lag"] = lag.String()
    }
    
    return status
}

// Close closes all database connections
func (dr *DatabaseReplication) Close() error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()
    
    // Close master
    if err := dr.master.Close(); err != nil {
        log.Printf("Warning: failed to close master: %v", err)
    }
    
    // Close slaves
    for i, slave := range dr.slaves {
        if err := slave.Close(); err != nil {
            log.Printf("Warning: failed to close slave %d: %v", i, err)
        }
    }
    
    return nil
}

// Example usage
func main() {
    // Create replication config
    config := &ReplicationConfig{
        MasterDSN:     "postgres://user:password@localhost:5432/master_db?sslmode=disable",
        SlaveDSNs:     []string{
            "postgres://user:password@localhost:5433/slave1_db?sslmode=disable",
            "postgres://user:password@localhost:5434/slave2_db?sslmode=disable",
        },
        SyncMode:      false,
        MaxRetries:    3,
        RetryInterval: 1 * time.Second,
    }
    
    // Create replication manager
    replication, err := NewDatabaseReplication(config)
    if err != nil {
        log.Fatalf("Failed to create replication: %v", err)
    }
    defer replication.Close()
    
    // Health check
    if err := replication.HealthCheck(); err != nil {
        log.Fatalf("Health check failed: %v", err)
    }
    
    // Write data
    ctx := context.Background()
    if err := replication.Write(ctx, "INSERT INTO users (name, email) VALUES ($1, $2)", "John Doe", "john@example.com"); err != nil {
        log.Printf("Write failed: %v", err)
    }
    
    // Read data
    rows, err := replication.Read(ctx, "SELECT * FROM users WHERE name = $1", "John Doe")
    if err != nil {
        log.Printf("Read failed: %v", err)
    } else {
        defer rows.Close()
        
        for rows.Next() {
            var id int
            var name, email string
            if err := rows.Scan(&id, &name, &email); err != nil {
                log.Printf("Scan failed: %v", err)
                continue
            }
            log.Printf("User: ID=%d, Name=%s, Email=%s", id, name, email)
        }
    }
    
    // Get replication status
    status := replication.GetReplicationStatus()
    log.Printf("Replication status: %+v", status)
    
    // Get replication lag
    if lag, err := replication.GetReplicationLag(); err == nil {
        log.Printf("Replication lag: %v", lag)
    }
}
```

## Node.js Implementation

```javascript
const { Pool } = require('pg');

class DatabaseReplication {
  constructor(config) {
    this.config = config;
    this.master = null;
    this.slaves = [];
    this.isHealthy = false;
    this.lastSync = null;
  }

  async initialize() {
    // Connect to master
    this.master = new Pool({
      connectionString: this.config.masterDSN,
    });

    try {
      await this.master.query('SELECT 1');
    } catch (error) {
      throw new Error(`Failed to connect to master: ${error.message}`);
    }

    // Connect to slaves
    for (const dsn of this.config.slaveDSNs) {
      try {
        const slave = new Pool({
          connectionString: dsn,
        });
        await slave.query('SELECT 1');
        this.slaves.push(slave);
      } catch (error) {
        console.warn(`Warning: failed to connect to slave ${dsn}: ${error.message}`);
      }
    }

    if (this.slaves.length === 0) {
      throw new Error('No healthy slaves available');
    }

    this.isHealthy = true;
  }

  async write(query, params = []) {
    if (!this.isHealthy) {
      throw new Error('Replication is not healthy');
    }

    // Execute on master
    const result = await this.master.query(query, params);
    console.log(`Write executed on master, affected rows: ${result.rowCount}`);

    // If synchronous mode, wait for slaves
    if (this.config.syncMode) {
      await this.syncToSlaves(query, params);
    } else {
      // If asynchronous mode, sync in background
      this.syncToSlaves(query, params).catch(error => {
        console.error('Failed to sync to slaves:', error);
      });
    }

    return result;
  }

  async read(query, params = []) {
    if (!this.isHealthy) {
      throw new Error('Replication is not healthy');
    }

    if (this.slaves.length === 0) {
      throw new Error('No slaves available');
    }

    // Select a random slave for read
    const slave = this.slaves[0]; // In production, use load balancing

    const result = await slave.query(query, params);
    return result;
  }

  async syncToSlaves(query, params) {
    const promises = this.slaves.map(async (slave, index) => {
      try {
        await this.syncToSlave(slave, query, params);
      } catch (error) {
        throw new Error(`Slave ${index} sync failed: ${error.message}`);
      }
    });

    await Promise.all(promises);
    this.lastSync = new Date();
  }

  async syncToSlave(slave, query, params) {
    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        await slave.query('SELECT 1'); // Ping
        const result = await slave.query(query, params);
        console.log(`Slave sync successful, affected rows: ${result.rowCount}`);
        return;
      } catch (error) {
        console.log(`Slave exec failed, attempt ${attempt + 1}: ${error.message}`);
        if (attempt < this.config.maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, this.config.retryInterval));
        }
      }
    }

    throw new Error(`Failed to sync to slave after ${this.config.maxRetries} attempts`);
  }

  async healthCheck() {
    try {
      // Check master
      await this.master.query('SELECT 1');
    } catch (error) {
      this.isHealthy = false;
      throw new Error(`Master is unhealthy: ${error.message}`);
    }

    // Check slaves
    let healthySlaves = 0;
    for (let i = 0; i < this.slaves.length; i++) {
      try {
        await this.slaves[i].query('SELECT 1');
        healthySlaves++;
      } catch (error) {
        console.log(`Slave ${i} is unhealthy: ${error.message}`);
      }
    }

    if (healthySlaves === 0) {
      this.isHealthy = false;
      throw new Error('No healthy slaves available');
    }

    this.isHealthy = true;
  }

  async getReplicationLag() {
    if (!this.isHealthy) {
      throw new Error('Replication is not healthy');
    }

    // Get master timestamp
    const masterResult = await this.master.query('SELECT NOW()');
    const masterTime = new Date(masterResult.rows[0].now);

    // Get slave timestamp
    const slaveResult = await this.slaves[0].query('SELECT NOW()');
    const slaveTime = new Date(slaveResult.rows[0].now);

    return masterTime - slaveTime;
  }

  async promoteSlave(slaveIndex) {
    if (slaveIndex >= this.slaves.length) {
      throw new Error(`Invalid slave index: ${slaveIndex}`);
    }

    // Promote slave to master
    const oldMaster = this.master;
    const newMaster = this.slaves[slaveIndex];

    // Update master
    this.master = newMaster;

    // Update slaves list
    this.slaves.splice(slaveIndex, 1);
    this.slaves.push(oldMaster);

    console.log(`Slave ${slaveIndex} promoted to master`);
  }

  async addSlave(dsn) {
    try {
      const slave = new Pool({
        connectionString: dsn,
      });
      await slave.query('SELECT 1');
      this.slaves.push(slave);
      console.log(`New slave added: ${dsn}`);
    } catch (error) {
      throw new Error(`Failed to add slave: ${error.message}`);
    }
  }

  async removeSlave(slaveIndex) {
    if (slaveIndex >= this.slaves.length) {
      throw new Error(`Invalid slave index: ${slaveIndex}`);
    }

    // Close the slave connection
    try {
      await this.slaves[slaveIndex].end();
    } catch (error) {
      console.warn(`Warning: failed to close slave ${slaveIndex}: ${error.message}`);
    }

    // Remove from slaves list
    this.slaves.splice(slaveIndex, 1);

    console.log(`Slave ${slaveIndex} removed`);
  }

  getReplicationStatus() {
    const status = {
      isHealthy: this.isHealthy,
      lastSync: this.lastSync,
      slaveCount: this.slaves.length,
      syncMode: this.config.syncMode,
    };

    return status;
  }

  async close() {
    // Close master
    try {
      await this.master.end();
    } catch (error) {
      console.warn(`Warning: failed to close master: ${error.message}`);
    }

    // Close slaves
    for (let i = 0; i < this.slaves.length; i++) {
      try {
        await this.slaves[i].end();
      } catch (error) {
        console.warn(`Warning: failed to close slave ${i}: ${error.message}`);
      }
    }
  }
}

// Example usage
async function main() {
  // Create replication config
  const config = {
    masterDSN: 'postgres://user:password@localhost:5432/master_db?sslmode=disable',
    slaveDSNs: [
      'postgres://user:password@localhost:5433/slave1_db?sslmode=disable',
      'postgres://user:password@localhost:5434/slave2_db?sslmode=disable',
    ],
    syncMode: false,
    maxRetries: 3,
    retryInterval: 1000, // 1 second
  };

  // Create replication manager
  const replication = new DatabaseReplication(config);
  await replication.initialize();

  try {
    // Health check
    await replication.healthCheck();

    // Write data
    await replication.write(
      'INSERT INTO users (name, email) VALUES ($1, $2)',
      ['John Doe', 'john@example.com']
    );

    // Read data
    const result = await replication.read(
      'SELECT * FROM users WHERE name = $1',
      ['John Doe']
    );

    console.log('Users:', result.rows);

    // Get replication status
    const status = replication.getReplicationStatus();
    console.log('Replication status:', status);

    // Get replication lag
    try {
      const lag = await replication.getReplicationLag();
      console.log('Replication lag:', lag, 'ms');
    } catch (error) {
      console.log('Failed to get replication lag:', error.message);
    }
  } finally {
    await replication.close();
  }
}

if (require.main === module) {
  main().catch(console.error);
}
```

## Benefits

1. **High Availability**: Multiple copies prevent single point of failure
2. **Read Scaling**: Distribute read operations across multiple replicas
3. **Geographic Distribution**: Place replicas in different locations
4. **Backup**: Replicas serve as live backups
5. **Load Distribution**: Balance load across multiple databases

## Trade-offs

1. **Consistency**: Eventual consistency in asynchronous replication
2. **Complexity**: More complex than single database
3. **Latency**: Synchronous replication adds latency
4. **Storage**: Multiple copies require more storage

## Use Cases

- **High-Traffic Applications**: Scale read operations
- **Global Applications**: Reduce latency with geographic distribution
- **Critical Systems**: Ensure high availability
- **Analytics**: Use read replicas for reporting

## Best Practices

1. **Monitor Lag**: Monitor replication lag
2. **Health Checks**: Regular health checks
3. **Failover Planning**: Plan for automatic failover
4. **Data Consistency**: Consider consistency requirements
5. **Testing**: Test failover scenarios

## Common Pitfalls

1. **Split-Brain**: Multiple masters accepting writes
2. **Data Loss**: Asynchronous replication can lose data
3. **Performance**: Replication can impact performance
4. **Complexity**: Managing multiple databases

## Interview Questions

1. **What's the difference between synchronous and asynchronous replication?**
   - Synchronous waits for all replicas, asynchronous doesn't

2. **How do you handle replication lag?**
   - Monitor lag, use read-after-write consistency, or synchronous replication

3. **What happens if a replica fails?**
   - Remove from rotation, promote another replica, or add new replica

4. **How do you ensure data consistency?**
   - Use appropriate consistency models, monitor replication, handle conflicts

## Time Complexity

- **Write Operations**: O(1) for basic writes, O(n) for synchronous replication
- **Read Operations**: O(1) for basic reads
- **Health Checks**: O(n) where n is number of replicas

## Space Complexity

- **Storage**: O(n) where n is number of replicas
- **Memory**: O(1) for basic operations
- **Network**: O(n) for replication traffic

The optimal solution uses:
1. **Appropriate Replication Mode**: Choose sync/async based on requirements
2. **Health Monitoring**: Monitor all replicas continuously
3. **Failover Strategy**: Implement automatic failover
4. **Load Balancing**: Distribute reads across replicas
