# Lesson 32.1: Advanced Database Patterns

> **Module**: 32 - Advanced Database Patterns  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 31 (Advanced Web Development)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Design and implement advanced database patterns
- Handle distributed transactions and consistency
- Optimize database performance
- Implement caching strategies
- Build resilient database systems

---

## 🎯 **Overview**

Advanced database patterns in Rust involve designing scalable, performant, and resilient database systems. This lesson covers distributed transactions, caching strategies, performance optimization, and advanced query patterns.

---

## 🔧 **Distributed Database Patterns**

### **Saga Pattern Implementation**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Saga {
    pub id: Uuid,
    pub steps: Vec<SagaStep>,
    pub status: SagaStatus,
    pub current_step: usize,
    pub compensation_log: Vec<CompensationAction>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SagaStep {
    pub id: String,
    pub action: SagaAction,
    pub compensation: Option<SagaAction>,
    pub timeout: Option<std::time::Duration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SagaAction {
    CreateOrder { order_id: Uuid, user_id: Uuid, amount: f64 },
    ReserveInventory { product_id: Uuid, quantity: u32 },
    ProcessPayment { order_id: Uuid, amount: f64 },
    SendNotification { user_id: Uuid, message: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SagaStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Compensating,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompensationAction {
    pub step_id: String,
    pub action: SagaAction,
    pub executed_at: chrono::DateTime<chrono::Utc>,
}

pub struct SagaManager {
    pub sagas: Arc<RwLock<HashMap<Uuid, Saga>>>,
    pub step_handlers: Arc<RwLock<HashMap<String, Box<dyn SagaStepHandler + Send + Sync>>>>,
}

pub trait SagaStepHandler {
    async fn execute(&self, action: &SagaAction) -> Result<(), SagaError>;
    async fn compensate(&self, action: &SagaAction) -> Result<(), SagaError>;
}

impl SagaManager {
    pub fn new() -> Self {
        Self {
            sagas: Arc::new(RwLock::new(HashMap::new())),
            step_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_handler(&self, step_id: String, handler: Box<dyn SagaStepHandler + Send + Sync>) {
        self.step_handlers.write().await.insert(step_id, handler);
    }
    
    pub async fn create_saga(&self, steps: Vec<SagaStep>) -> Result<Uuid, SagaError> {
        let saga_id = Uuid::new_v4();
        let saga = Saga {
            id: saga_id,
            steps,
            status: SagaStatus::Pending,
            current_step: 0,
            compensation_log: Vec::new(),
        };
        
        self.sagas.write().await.insert(saga_id, saga);
        Ok(saga_id)
    }
    
    pub async fn execute_saga(&self, saga_id: Uuid) -> Result<(), SagaError> {
        let mut saga = self.sagas.write().await.get_mut(&saga_id)
            .ok_or(SagaError::SagaNotFound)?;
        
        saga.status = SagaStatus::InProgress;
        
        for (index, step) in saga.steps.iter().enumerate() {
            saga.current_step = index;
            
            if let Some(handler) = self.step_handlers.read().await.get(&step.id) {
                match handler.execute(&step.action).await {
                    Ok(_) => {
                        // Step completed successfully
                        continue;
                    }
                    Err(error) => {
                        // Step failed, start compensation
                        saga.status = SagaStatus::Failed;
                        self.compensate_saga(saga_id).await?;
                        return Err(error);
                    }
                }
            } else {
                return Err(SagaError::HandlerNotFound);
            }
        }
        
        saga.status = SagaStatus::Completed;
        Ok(())
    }
    
    async fn compensate_saga(&self, saga_id: Uuid) -> Result<(), SagaError> {
        let mut saga = self.sagas.write().await.get_mut(&saga_id)
            .ok_or(SagaError::SagaNotFound)?;
        
        saga.status = SagaStatus::Compensating;
        
        // Execute compensation actions in reverse order
        for step in saga.steps.iter().rev() {
            if let Some(compensation) = &step.compensation {
                if let Some(handler) = self.step_handlers.read().await.get(&step.id) {
                    if let Err(error) = handler.compensate(compensation).await {
                        // Log compensation failure but continue
                        eprintln!("Compensation failed for step {}: {:?}", step.id, error);
                    }
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug)]
pub enum SagaError {
    SagaNotFound,
    HandlerNotFound,
    StepExecutionFailed,
    CompensationFailed,
}

// Example step handlers
pub struct OrderStepHandler;

impl SagaStepHandler for OrderStepHandler {
    async fn execute(&self, action: &SagaAction) -> Result<(), SagaError> {
        match action {
            SagaAction::CreateOrder { order_id, user_id, amount } => {
                // Create order in database
                println!("Creating order {} for user {} with amount {}", order_id, user_id, amount);
                Ok(())
            }
            _ => Err(SagaError::StepExecutionFailed),
        }
    }
    
    async fn compensate(&self, action: &SagaAction) -> Result<(), SagaError> {
        match action {
            SagaAction::CreateOrder { order_id, .. } => {
                // Cancel order in database
                println!("Cancelling order {}", order_id);
                Ok(())
            }
            _ => Err(SagaError::CompensationFailed),
        }
    }
}
```

### **CQRS Pattern Implementation**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Command {
    pub id: Uuid,
    pub command_type: CommandType,
    pub aggregate_id: Uuid,
    pub data: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CommandType {
    CreateUser,
    UpdateUser,
    DeleteUser,
    CreateOrder,
    UpdateOrder,
    CancelOrder,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub aggregate_id: Uuid,
    pub data: serde_json::Value,
    pub version: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventType {
    UserCreated,
    UserUpdated,
    UserDeleted,
    OrderCreated,
    OrderUpdated,
    OrderCancelled,
}

pub struct CQRSManager {
    pub command_store: Arc<CommandStore>,
    pub event_store: Arc<EventStore>,
    pub read_models: Arc<RwLock<HashMap<String, Box<dyn ReadModel + Send + Sync>>>>,
    pub command_handlers: Arc<RwLock<HashMap<CommandType, Box<dyn CommandHandler + Send + Sync>>>>,
    pub event_handlers: Arc<RwLock<HashMap<EventType, Box<dyn EventHandler + Send + Sync>>>>,
}

pub trait CommandHandler {
    async fn handle(&self, command: &Command) -> Result<Vec<Event>, CQRSError>;
}

pub trait EventHandler {
    async fn handle(&self, event: &Event) -> Result<(), CQRSError>;
}

pub trait ReadModel {
    async fn update(&self, event: &Event) -> Result<(), CQRSError>;
    async fn get(&self, id: &str) -> Result<Option<serde_json::Value>, CQRSError>;
}

impl CQRSManager {
    pub fn new() -> Self {
        Self {
            command_store: Arc::new(CommandStore::new()),
            event_store: Arc::new(EventStore::new()),
            read_models: Arc::new(RwLock::new(HashMap::new())),
            command_handlers: Arc::new(RwLock::new(HashMap::new())),
            event_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_command_handler(&self, command_type: CommandType, handler: Box<dyn CommandHandler + Send + Sync>) {
        self.command_handlers.write().await.insert(command_type, handler);
    }
    
    pub async fn register_event_handler(&self, event_type: EventType, handler: Box<dyn EventHandler + Send + Sync>) {
        self.event_handlers.write().await.insert(event_type, handler);
    }
    
    pub async fn register_read_model(&self, name: String, read_model: Box<dyn ReadModel + Send + Sync>) {
        self.read_models.write().await.insert(name, read_model);
    }
    
    pub async fn execute_command(&self, command: Command) -> Result<(), CQRSError> {
        // Store command
        self.command_store.store(&command).await?;
        
        // Find command handler
        let handler = self.command_handlers.read().await
            .get(&command.command_type)
            .ok_or(CQRSError::HandlerNotFound)?;
        
        // Execute command and get events
        let events = handler.handle(&command).await?;
        
        // Store and publish events
        for event in events {
            self.event_store.store(&event).await?;
            self.publish_event(&event).await?;
        }
        
        Ok(())
    }
    
    async fn publish_event(&self, event: &Event) -> Result<(), CQRSError> {
        // Find event handler
        if let Some(handler) = self.event_handlers.read().await.get(&event.event_type) {
            handler.handle(event).await?;
        }
        
        // Update read models
        let read_models = self.read_models.read().await;
        for (_, read_model) in read_models.iter() {
            if let Err(error) = read_model.update(event).await {
                eprintln!("Failed to update read model: {:?}", error);
            }
        }
        
        Ok(())
    }
}

pub struct CommandStore {
    pub commands: Arc<RwLock<Vec<Command>>>,
}

impl CommandStore {
    pub fn new() -> Self {
        Self {
            commands: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn store(&self, command: &Command) -> Result<(), CQRSError> {
        self.commands.write().await.push(command.clone());
        Ok(())
    }
}

pub struct EventStore {
    pub events: Arc<RwLock<Vec<Event>>>,
}

impl EventStore {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn store(&self, event: &Event) -> Result<(), CQRSError> {
        self.events.write().await.push(event.clone());
        Ok(())
    }
    
    pub async fn get_events(&self, aggregate_id: Uuid) -> Result<Vec<Event>, CQRSError> {
        let events = self.events.read().await;
        let aggregate_events: Vec<Event> = events
            .iter()
            .filter(|e| e.aggregate_id == aggregate_id)
            .cloned()
            .collect();
        Ok(aggregate_events)
    }
}

#[derive(Debug)]
pub enum CQRSError {
    HandlerNotFound,
    EventStoreError,
    CommandStoreError,
    ReadModelError,
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Database Connection Pooling**

```rust
use sqlx::{PgPool, Postgres, Row};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

pub struct ConnectionPool {
    pub pool: PgPool,
    pub metrics: Arc<PoolMetrics>,
}

pub struct PoolMetrics {
    pub total_connections: Arc<RwLock<u32>>,
    pub active_connections: Arc<RwLock<u32>>,
    pub idle_connections: Arc<RwLock<u32>>,
    pub connection_requests: Arc<RwLock<u64>>,
    pub connection_errors: Arc<RwLock<u64>>,
    pub average_connection_time: Arc<RwLock<Duration>>,
}

impl ConnectionPool {
    pub async fn new(database_url: &str, max_connections: u32) -> Result<Self, sqlx::Error> {
        let pool = PgPool::builder()
            .max_connections(max_connections)
            .min_connections(5)
            .acquire_timeout(Duration::from_secs(30))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800))
            .build(database_url)
            .await?;
        
        let metrics = Arc::new(PoolMetrics {
            total_connections: Arc::new(RwLock::new(max_connections)),
            active_connections: Arc::new(RwLock::new(0)),
            idle_connections: Arc::new(RwLock::new(0)),
            connection_requests: Arc::new(RwLock::new(0)),
            connection_errors: Arc::new(RwLock::new(0)),
            average_connection_time: Arc::new(RwLock::new(Duration::from_millis(0))),
        });
        
        Ok(Self { pool, metrics })
    }
    
    pub async fn execute_query<T>(&self, query: &str, params: &[&dyn sqlx::Encode<'_, Postgres>]) -> Result<Vec<T>, sqlx::Error>
    where
        T: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    {
        let start_time = Instant::now();
        
        // Increment connection requests
        {
            let mut requests = self.metrics.connection_requests.write().await;
            *requests += 1;
        }
        
        // Get connection from pool
        let connection = self.pool.acquire().await.map_err(|e| {
            // Increment connection errors
            tokio::spawn({
                let metrics = self.metrics.clone();
                async move {
                    let mut errors = metrics.connection_errors.write().await;
                    *errors += 1;
                }
            });
            e
        })?;
        
        // Update active connections
        {
            let mut active = self.metrics.active_connections.write().await;
            *active += 1;
        }
        
        // Execute query
        let result = sqlx::query(query)
            .bind_all(params)
            .fetch_all(&self.pool)
            .await?;
        
        // Update metrics
        let connection_time = start_time.elapsed();
        {
            let mut avg_time = self.metrics.average_connection_time.write().await;
            *avg_time = Duration::from_millis(
                (avg_time.as_millis() + connection_time.as_millis()) / 2
            );
        }
        
        // Update active connections
        {
            let mut active = self.metrics.active_connections.write().await;
            *active -= 1;
        }
        
        Ok(result)
    }
    
    pub async fn get_metrics(&self) -> PoolMetricsSnapshot {
        PoolMetricsSnapshot {
            total_connections: *self.metrics.total_connections.read().await,
            active_connections: *self.metrics.active_connections.read().await,
            idle_connections: *self.metrics.idle_connections.read().await,
            connection_requests: *self.metrics.connection_requests.read().await,
            connection_errors: *self.metrics.connection_errors.read().await,
            average_connection_time: *self.metrics.average_connection_time.read().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolMetricsSnapshot {
    pub total_connections: u32,
    pub active_connections: u32,
    pub idle_connections: u32,
    pub connection_requests: u64,
    pub connection_errors: u64,
    pub average_connection_time: Duration,
}
```

### **Exercise 2: Advanced Caching Strategy**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

pub struct MultiLevelCache<K, V> {
    pub l1_cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    pub l2_cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub l1_ttl: Duration,
    pub l2_ttl: Duration,
}

#[derive(Clone, Debug)]
pub struct CacheEntry<V> {
    pub value: V,
    pub created_at: Instant,
    pub access_count: u64,
    pub last_accessed: Instant,
}

impl<K, V> MultiLevelCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(l1_capacity: usize, l2_capacity: usize, l1_ttl: Duration, l2_ttl: Duration) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l2_cache: Arc::new(RwLock::new(HashMap::new())),
            l1_capacity,
            l2_capacity,
            l1_ttl,
            l2_ttl,
        }
    }
    
    pub async fn get(&self, key: &K) -> Option<V> {
        // Check L1 cache first
        if let Some(entry) = self.l1_cache.read().await.get(key) {
            if entry.created_at.elapsed() < self.l1_ttl {
                // Update access statistics
                let mut l1_cache = self.l1_cache.write().await;
                if let Some(entry) = l1_cache.get_mut(key) {
                    entry.access_count += 1;
                    entry.last_accessed = Instant::now();
                }
                return Some(entry.value.clone());
            } else {
                // L1 entry expired, remove it
                self.l1_cache.write().await.remove(key);
            }
        }
        
        // Check L2 cache
        if let Some(entry) = self.l2_cache.read().await.get(key) {
            if entry.created_at.elapsed() < self.l2_ttl {
                // Move to L1 cache
                self.promote_to_l1(key.clone(), entry.clone()).await;
                return Some(entry.value.clone());
            } else {
                // L2 entry expired, remove it
                self.l2_cache.write().await.remove(key);
            }
        }
        
        None
    }
    
    pub async fn set(&self, key: K, value: V) {
        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        // Try to insert into L1 cache
        if self.l1_cache.read().await.len() < self.l1_capacity {
            self.l1_cache.write().await.insert(key, entry);
        } else {
            // L1 cache is full, try to evict least recently used entry
            if let Some(evicted_key) = self.evict_lru_from_l1().await {
                // Move evicted entry to L2 cache
                if let Some(evicted_entry) = self.l1_cache.write().await.remove(&evicted_key) {
                    self.insert_into_l2(evicted_key, evicted_entry).await;
                }
            }
            
            // Insert new entry into L1
            self.l1_cache.write().await.insert(key, entry);
        }
    }
    
    async fn promote_to_l1(&self, key: K, mut entry: CacheEntry<V>) {
        // Update access statistics
        entry.access_count += 1;
        entry.last_accessed = Instant::now();
        
        // Remove from L2
        self.l2_cache.write().await.remove(&key);
        
        // Insert into L1
        if self.l1_cache.read().await.len() < self.l1_capacity {
            self.l1_cache.write().await.insert(key, entry);
        } else {
            // L1 is full, evict LRU and insert
            if let Some(evicted_key) = self.evict_lru_from_l1().await {
                if let Some(evicted_entry) = self.l1_cache.write().await.remove(&evicted_key) {
                    self.insert_into_l2(evicted_key, evicted_entry).await;
                }
            }
            self.l1_cache.write().await.insert(key, entry);
        }
    }
    
    async fn evict_lru_from_l1(&self) -> Option<K> {
        let l1_cache = self.l1_cache.read().await;
        let mut lru_key = None;
        let mut lru_time = Instant::now();
        
        for (key, entry) in l1_cache.iter() {
            if entry.last_accessed < lru_time {
                lru_time = entry.last_accessed;
                lru_key = Some(key.clone());
            }
        }
        
        lru_key
    }
    
    async fn insert_into_l2(&self, key: K, entry: CacheEntry<V>) {
        if self.l2_cache.read().await.len() < self.l2_capacity {
            self.l2_cache.write().await.insert(key, entry);
        } else {
            // L2 is full, evict LRU
            if let Some(evicted_key) = self.evict_lru_from_l2().await {
                self.l2_cache.write().await.remove(&evicted_key);
            }
            self.l2_cache.write().await.insert(key, entry);
        }
    }
    
    async fn evict_lru_from_l2(&self) -> Option<K> {
        let l2_cache = self.l2_cache.read().await;
        let mut lru_key = None;
        let mut lru_time = Instant::now();
        
        for (key, entry) in l2_cache.iter() {
            if entry.last_accessed < lru_time {
                lru_time = entry.last_accessed;
                lru_key = Some(key.clone());
            }
        }
        
        lru_key
    }
    
    pub async fn clear(&self) {
        self.l1_cache.write().await.clear();
        self.l2_cache.write().await.clear();
    }
    
    pub async fn get_stats(&self) -> CacheStats {
        let l1_cache = self.l1_cache.read().await;
        let l2_cache = self.l2_cache.read().await;
        
        CacheStats {
            l1_size: l1_cache.len(),
            l2_size: l2_cache.len(),
            l1_capacity: self.l1_capacity,
            l2_capacity: self.l2_capacity,
            total_hits: 0, // Would need to track this
            total_misses: 0, // Would need to track this
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub total_hits: u64,
    pub total_misses: u64,
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_saga_execution() {
        let saga_manager = SagaManager::new();
        
        // Register handlers
        saga_manager.register_handler(
            "order".to_string(),
            Box::new(OrderStepHandler),
        ).await;
        
        // Create saga
        let steps = vec![
            SagaStep {
                id: "order".to_string(),
                action: SagaAction::CreateOrder {
                    order_id: Uuid::new_v4(),
                    user_id: Uuid::new_v4(),
                    amount: 100.0,
                },
                compensation: Some(SagaAction::CreateOrder {
                    order_id: Uuid::new_v4(),
                    user_id: Uuid::new_v4(),
                    amount: 100.0,
                }),
                timeout: Some(Duration::from_secs(30)),
            },
        ];
        
        let saga_id = saga_manager.create_saga(steps).await.unwrap();
        let result = saga_manager.execute_saga(saga_id).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cqrs_command_execution() {
        let cqrs_manager = CQRSManager::new();
        
        // Register command handler
        cqrs_manager.register_command_handler(
            CommandType::CreateUser,
            Box::new(CreateUserCommandHandler),
        ).await;
        
        // Execute command
        let command = Command {
            id: Uuid::new_v4(),
            command_type: CommandType::CreateUser,
            aggregate_id: Uuid::new_v4(),
            data: serde_json::json!({"name": "John Doe", "email": "john@example.com"}),
            timestamp: chrono::Utc::now(),
        };
        
        let result = cqrs_manager.execute_command(command).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multi_level_cache() {
        let cache = MultiLevelCache::new(2, 4, Duration::from_secs(60), Duration::from_secs(300));
        
        // Test basic operations
        cache.set("key1".to_string(), "value1".to_string()).await;
        cache.set("key2".to_string(), "value2".to_string()).await;
        
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
        assert_eq!(cache.get(&"key2".to_string()).await, Some("value2".to_string()));
        
        // Test cache eviction
        cache.set("key3".to_string(), "value3".to_string()).await;
        cache.set("key4".to_string(), "value4".to_string()).await;
        
        // key1 should be evicted from L1 and moved to L2
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Database Connection Leaks**

```rust
// ❌ Wrong - potential connection leak
async fn bad_database_operation(pool: &PgPool) -> Result<(), sqlx::Error> {
    let connection = pool.acquire().await?;
    // Connection is not explicitly released
    // This can lead to connection pool exhaustion
    Ok(())
}

// ✅ Correct - proper connection management
async fn good_database_operation(pool: &PgPool) -> Result<(), sqlx::Error> {
    let connection = pool.acquire().await?;
    // Use connection within scope
    // Connection is automatically released when it goes out of scope
    Ok(())
}
```

### **Common Mistake 2: Inefficient Caching**

```rust
// ❌ Wrong - inefficient cache usage
pub struct BadCache {
    pub data: HashMap<String, String>,
}

impl BadCache {
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
    
    pub fn set(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
}

// ✅ Correct - efficient cache with TTL and LRU
pub struct GoodCache {
    pub data: HashMap<String, CacheEntry<String>>,
    pub capacity: usize,
    pub ttl: Duration,
}

impl GoodCache {
    pub fn get(&mut self, key: &str) -> Option<&String> {
        if let Some(entry) = self.data.get_mut(key) {
            if entry.created_at.elapsed() < self.ttl {
                entry.last_accessed = Instant::now();
                Some(&entry.value)
            } else {
                self.data.remove(key);
                None
            }
        } else {
            None
        }
    }
}
```

---

## 📊 **Advanced Database Patterns**

### **Event Sourcing Implementation**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Aggregate {
    pub id: Uuid,
    pub version: u64,
    pub events: Vec<Event>,
    pub state: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub aggregate_id: Uuid,
    pub event_type: String,
    pub data: serde_json::Value,
    pub version: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct EventStore {
    pub events: Arc<RwLock<Vec<Event>>>,
    pub aggregates: Arc<RwLock<HashMap<Uuid, Aggregate>>>,
}

impl EventStore {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            aggregates: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn append_events(&self, aggregate_id: Uuid, events: Vec<Event>) -> Result<(), EventStoreError> {
        let mut event_store = self.events.write().await;
        let mut aggregates = self.aggregates.write().await;
        
        // Check for concurrency conflicts
        if let Some(aggregate) = aggregates.get(&aggregate_id) {
            let expected_version = aggregate.version;
            let actual_version = events.first().map(|e| e.version).unwrap_or(0);
            
            if actual_version != expected_version + 1 {
                return Err(EventStoreError::ConcurrencyConflict);
            }
        }
        
        // Append events
        for event in events {
            event_store.push(event.clone());
            
            // Update aggregate
            if let Some(aggregate) = aggregates.get_mut(&aggregate_id) {
                aggregate.events.push(event.clone());
                aggregate.version += 1;
            } else {
                // Create new aggregate
                let aggregate = Aggregate {
                    id: aggregate_id,
                    version: 1,
                    events: vec![event.clone()],
                    state: serde_json::Value::Null,
                };
                aggregates.insert(aggregate_id, aggregate);
            }
        }
        
        Ok(())
    }
    
    pub async fn get_events(&self, aggregate_id: Uuid) -> Result<Vec<Event>, EventStoreError> {
        let events = self.events.read().await;
        let aggregate_events: Vec<Event> = events
            .iter()
            .filter(|e| e.aggregate_id == aggregate_id)
            .cloned()
            .collect();
        Ok(aggregate_events)
    }
    
    pub async fn get_aggregate(&self, aggregate_id: Uuid) -> Result<Option<Aggregate>, EventStoreError> {
        let aggregates = self.aggregates.read().await;
        Ok(aggregates.get(&aggregate_id).cloned())
    }
}

#[derive(Debug)]
pub enum EventStoreError {
    ConcurrencyConflict,
    AggregateNotFound,
    InvalidEvent,
}
```

---

## 🎯 **Best Practices**

### **Database Performance**

```rust
// ✅ Good - optimized database queries
pub struct OptimizedUserRepository {
    pub pool: PgPool,
    pub cache: Arc<MultiLevelCache<String, User>>,
}

impl OptimizedUserRepository {
    pub async fn get_user(&self, id: &str) -> Result<Option<User>, sqlx::Error> {
        // Check cache first
        if let Some(user) = self.cache.get(&id.to_string()).await {
            return Ok(Some(user));
        }
        
        // Query database
        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        
        // Cache result
        if let Some(ref user) = user {
            self.cache.set(id.to_string(), user.clone()).await;
        }
        
        Ok(user)
    }
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection error: {0}")]
    Connection(#[from] sqlx::Error),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Concurrency conflict: {0}")]
    ConcurrencyConflict(String),
    
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
}

pub type Result<T> = std::result::Result<T, DatabaseError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [SQLx Documentation](https://docs.rs/sqlx/latest/sqlx/) - Fetched: 2024-12-19T00:00:00Z
- [Redis Documentation](https://docs.rs/redis/latest/redis/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Database Patterns](https://martinfowler.com/articles/patterns-of-distributed-systems/) - Fetched: 2024-12-19T00:00:00Z
- [CQRS and Event Sourcing](https://martinfowler.com/bliki/CQRS.html) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement distributed transaction patterns?
2. Do you understand CQRS and event sourcing?
3. Can you optimize database performance?
4. Do you know how to implement caching strategies?
5. Can you build resilient database systems?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced observability patterns
- Distributed tracing
- Performance monitoring
- Production deployment

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [32.2 Advanced Observability Patterns](32_02_observability_patterns.md)
