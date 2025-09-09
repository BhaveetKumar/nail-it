package memento

import (
	"time"
)

// Originator defines the interface for objects that can create and restore mementos
type Originator interface {
	CreateMemento() Memento
	RestoreMemento(memento Memento) error
	GetState() interface{}
	SetState(state interface{}) error
	GetID() string
	GetName() string
	GetType() string
	GetVersion() int
	SetVersion(version int)
	GetLastModified() time.Time
	SetLastModified(timestamp time.Time)
	IsDirty() bool
	SetDirty(dirty bool)
}

// Memento defines the interface for mementos
type Memento interface {
	GetID() string
	GetOriginatorID() string
	GetState() interface{}
	GetTimestamp() time.Time
	GetVersion() int
	GetType() string
	GetDescription() string
	SetDescription(description string)
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
	IsValid() bool
	SetValid(valid bool)
	GetSize() int64
	GetChecksum() string
	SetChecksum(checksum string)
}

// Caretaker defines the interface for managing mementos
type Caretaker interface {
	SaveMemento(memento Memento) error
	GetMemento(id string) (Memento, error)
	GetMementosByOriginator(originatorID string) ([]Memento, error)
	GetMementosByType(mementoType string) ([]Memento, error)
	GetMementosByDateRange(start, end time.Time) ([]Memento, error)
	DeleteMemento(id string) error
	DeleteMementosByOriginator(originatorID string) error
	DeleteMementosByType(mementoType string) error
	DeleteMementosByDateRange(start, end time.Time) error
	GetMementoCount() int
	GetMementoCountByOriginator(originatorID string) int
	GetMementoCountByType(mementoType string) int
	GetMementoCountByDateRange(start, end time.Time) int
	GetMementoSize() int64
	GetMementoSizeByOriginator(originatorID string) int64
	GetMementoSizeByType(mementoType string) int64
	GetMementoSizeByDateRange(start, end time.Time) int64
	Cleanup() error
	GetStats() map[string]interface{}
}

// MementoManager defines the interface for managing multiple caretakers
type MementoManager interface {
	CreateCaretaker(name string) (Caretaker, error)
	GetCaretaker(name string) (Caretaker, error)
	RemoveCaretaker(name string) error
	ListCaretakers() []string
	GetCaretakerCount() int
	GetCaretakerStats() map[string]interface{}
	Cleanup() error
}

// MementoStore defines the interface for persistent storage of mementos
type MementoStore interface {
	Save(memento Memento) error
	Load(id string) (Memento, error)
	LoadByOriginator(originatorID string) ([]Memento, error)
	LoadByType(mementoType string) ([]Memento, error)
	LoadByDateRange(start, end time.Time) ([]Memento, error)
	Delete(id string) error
	DeleteByOriginator(originatorID string) error
	DeleteByType(mementoType string) error
	DeleteByDateRange(start, end time.Time) error
	Count() (int, error)
	CountByOriginator(originatorID string) (int, error)
	CountByType(mementoType string) (int, error)
	CountByDateRange(start, end time.Time) (int, error)
	Size() (int64, error)
	SizeByOriginator(originatorID string) (int64, error)
	SizeByType(mementoType string) (int64, error)
	SizeByDateRange(start, end time.Time) (int64, error)
	Cleanup() error
	GetStats() (map[string]interface{}, error)
}

// MementoSerializer defines the interface for serializing mementos
type MementoSerializer interface {
	Serialize(memento Memento) ([]byte, error)
	Deserialize(data []byte) (Memento, error)
	SerializeToJSON(memento Memento) ([]byte, error)
	DeserializeFromJSON(data []byte) (Memento, error)
	SerializeToXML(memento Memento) ([]byte, error)
	DeserializeFromXML(data []byte) (Memento, error)
	SerializeToYAML(memento Memento) ([]byte, error)
	DeserializeFromYAML(data []byte) (Memento, error)
	GetSupportedFormats() []string
	GetDefaultFormat() string
}

// MementoCompressor defines the interface for compressing mementos
type MementoCompressor interface {
	Compress(data []byte) ([]byte, error)
	Decompress(data []byte) ([]byte, error)
	CompressString(data string) (string, error)
	DecompressString(data string) (string, error)
	GetCompressionRatio() float64
	GetCompressionTime() time.Duration
	GetSupportedAlgorithms() []string
	GetDefaultAlgorithm() string
}

// MementoEncryptor defines the interface for encrypting mementos
type MementoEncryptor interface {
	Encrypt(data []byte, key []byte) ([]byte, error)
	Decrypt(data []byte, key []byte) ([]byte, error)
	EncryptString(data string, key string) (string, error)
	DecryptString(data string, key string) (string, error)
	GenerateKey() ([]byte, error)
	Hash(data []byte) ([]byte, error)
	HashString(data string) (string, error)
	GetSupportedAlgorithms() []string
	GetDefaultAlgorithm() string
}

// MementoValidator defines the interface for validating mementos
type MementoValidator interface {
	Validate(memento Memento) error
	ValidateState(state interface{}) error
	ValidateChecksum(memento Memento) error
	ValidateSize(memento Memento) error
	ValidateVersion(memento Memento) error
	ValidateTimestamp(memento Memento) error
	ValidateMetadata(memento Memento) error
	GetValidationRules() map[string]interface{}
	SetValidationRules(rules map[string]interface{})
	GetValidationErrors() []string
	ClearValidationErrors()
}

// MementoCache defines the interface for caching mementos
type MementoCache interface {
	Get(key string) (Memento, bool)
	Set(key string, memento Memento, ttl time.Duration) error
	Delete(key string) error
	Clear() error
	Size() int
	Keys() []string
	GetStats() map[string]interface{}
	GetHitRate() float64
	GetMissRate() float64
	GetEvictionCount() int64
	GetExpirationCount() int64
}

// MementoIndex defines the interface for indexing mementos
type MementoIndex interface {
	Index(memento Memento) error
	Search(query string) ([]Memento, error)
	SearchByOriginator(originatorID string) ([]Memento, error)
	SearchByType(mementoType string) ([]Memento, error)
	SearchByDateRange(start, end time.Time) ([]Memento, error)
	SearchByMetadata(metadata map[string]interface{}) ([]Memento, error)
	Remove(id string) error
	Clear() error
	GetStats() map[string]interface{}
	GetIndexSize() int
	GetIndexMemoryUsage() int64
}

// MementoBackup defines the interface for backing up mementos
type MementoBackup interface {
	Backup(mementos []Memento) error
	Restore(backupID string) ([]Memento, error)
	ListBackups() ([]BackupInfo, error)
	DeleteBackup(backupID string) error
	GetBackupStats() map[string]interface{}
	GetBackupSize(backupID string) (int64, error)
	GetBackupCount() int
	GetBackupCountByDateRange(start, end time.Time) int
	GetBackupSizeByDateRange(start, end time.Time) int64
}

// BackupInfo defines the interface for backup information
type BackupInfo interface {
	GetID() string
	GetName() string
	GetDescription() string
	GetTimestamp() time.Time
	GetSize() int64
	GetMementoCount() int
	GetStatus() string
	GetMetadata() map[string]interface{}
	SetStatus(status string)
	SetMetadata(metadata map[string]interface{})
	IsValid() bool
	SetValid(valid bool)
	GetChecksum() string
	SetChecksum(checksum string)
}

// MementoReplicator defines the interface for replicating mementos
type MementoReplicator interface {
	Replicate(memento Memento, targets []string) error
	ReplicateBatch(mementos []Memento, targets []string) error
	GetReplicationStatus(mementoID string) (map[string]string, error)
	GetReplicationStats() map[string]interface{}
	GetReplicationTargets() []string
	AddReplicationTarget(target string) error
	RemoveReplicationTarget(target string) error
	GetReplicationLatency() time.Duration
	GetReplicationThroughput() float64
	GetReplicationErrors() []string
	ClearReplicationErrors()
}

// MementoMonitor defines the interface for monitoring mementos
type MementoMonitor interface {
	RecordMementoCreated(memento Memento) error
	RecordMementoRestored(memento Memento) error
	RecordMementoDeleted(memento Memento) error
	RecordMementoBackedUp(memento Memento) error
	RecordMementoReplicated(memento Memento) error
	GetMementoMetrics() map[string]interface{}
	GetMementoCounts() map[string]int64
	GetMementoSizes() map[string]int64
	GetMementoRates() map[string]float64
	GetMementoLatencies() map[string]time.Duration
	GetMementoErrors() []string
	ClearMementoErrors()
	GetMementoHealth() map[string]interface{}
	GetMementoAlerts() []Alert
	GetMementoTrends() map[string]interface{}
}

// Alert defines the interface for alerts
type Alert interface {
	GetID() string
	GetType() string
	GetSeverity() string
	GetMessage() string
	GetTimestamp() time.Time
	GetStatus() string
	GetSource() string
	GetData() map[string]interface{}
	SetStatus(status string)
	Resolve() error
	IsActive() bool
	IsResolved() bool
}

// MementoScheduler defines the interface for scheduling memento operations
type MementoScheduler interface {
	ScheduleBackup(schedule string, mementos []Memento) error
	ScheduleCleanup(schedule string, criteria map[string]interface{}) error
	ScheduleReplication(schedule string, mementos []Memento, targets []string) error
	ScheduleValidation(schedule string, mementos []Memento) error
	GetScheduledTasks() ([]ScheduledTask, error)
	CancelTask(taskID string) error
	GetTaskStatus(taskID string) (string, error)
	GetTaskStats() map[string]interface{}
	GetTaskHistory() ([]TaskHistory, error)
}

// ScheduledTask defines the interface for scheduled tasks
type ScheduledTask interface {
	GetID() string
	GetName() string
	GetType() string
	GetSchedule() string
	GetStatus() string
	GetNextRun() time.Time
	GetLastRun() time.Time
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetStatus(status string)
	SetNextRun(nextRun time.Time)
	SetLastRun(lastRun time.Time)
	SetUpdatedAt(updatedAt time.Time)
	SetMetadata(metadata map[string]interface{})
	IsActive() bool
	SetActive(active bool)
	GetRetryCount() int
	SetRetryCount(retryCount int)
	GetMaxRetries() int
	SetMaxRetries(maxRetries int)
}

// TaskHistory defines the interface for task history
type TaskHistory interface {
	GetID() string
	GetTaskID() string
	GetStatus() string
	GetStartTime() time.Time
	GetEndTime() time.Time
	GetDuration() time.Duration
	GetError() string
	GetResult() map[string]interface{}
	GetMetadata() map[string]interface{}
	SetStatus(status string)
	SetEndTime(endTime time.Time)
	SetError(error string)
	SetResult(result map[string]interface{})
	SetMetadata(metadata map[string]interface{})
	IsSuccessful() bool
	IsFailed() bool
	IsRunning() bool
	IsCompleted() bool
}

// MementoAuditor defines the interface for auditing memento operations
type MementoAuditor interface {
	AuditMementoCreated(memento Memento, userID string) error
	AuditMementoRestored(memento Memento, userID string) error
	AuditMementoDeleted(memento Memento, userID string) error
	AuditMementoBackedUp(memento Memento, userID string) error
	AuditMementoReplicated(memento Memento, userID string) error
	GetAuditLogs() ([]AuditLog, error)
	GetAuditLogsByUser(userID string) ([]AuditLog, error)
	GetAuditLogsByMemento(mementoID string) ([]AuditLog, error)
	GetAuditLogsByDateRange(start, end time.Time) ([]AuditLog, error)
	GetAuditLogsByAction(action string) ([]AuditLog, error)
	GetAuditStats() map[string]interface{}
	ClearAuditLogs() error
	ExportAuditLogs(format string) ([]byte, error)
}

// AuditLog defines the interface for audit logs
type AuditLog interface {
	GetID() string
	GetUserID() string
	GetAction() string
	GetResource() string
	GetDetails() map[string]interface{}
	GetIP() string
	GetUserAgent() string
	GetTimestamp() time.Time
	GetSessionID() string
	GetRequestID() string
	GetResponseCode() int
	GetResponseTime() time.Duration
	SetResponseCode(code int)
	SetResponseTime(responseTime time.Duration)
	IsSuccessful() bool
	SetSuccessful(successful bool)
	GetError() string
	SetError(error string)
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
}

// MementoConfig defines the interface for memento configuration
type MementoConfig interface {
	GetName() string
	GetVersion() string
	GetDescription() string
	GetMaxMementos() int
	GetMaxMementoSize() int64
	GetMaxMementoAge() time.Duration
	GetCleanupInterval() time.Duration
	GetBackupInterval() time.Duration
	GetReplicationInterval() time.Duration
	GetValidationInterval() time.Duration
	GetCompressionEnabled() bool
	GetEncryptionEnabled() bool
	GetCachingEnabled() bool
	GetIndexingEnabled() bool
	GetMonitoringEnabled() bool
	GetAuditingEnabled() bool
	GetSchedulingEnabled() bool
	GetBackupEnabled() bool
	GetReplicationEnabled() bool
	GetValidationEnabled() bool
	GetSupportedFormats() []string
	GetDefaultFormat() string
	GetSupportedAlgorithms() []string
	GetDefaultAlgorithm() string
	GetValidationRules() map[string]interface{}
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
	GetDatabase() DatabaseConfig
	GetCache() CacheConfig
	GetMessageQueue() MessageQueueConfig
	GetWebSocket() WebSocketConfig
	GetSecurity() SecurityConfig
	GetMonitoring() MonitoringConfig
	GetLogging() LoggingConfig
}

// DatabaseConfig defines the interface for database configuration
type DatabaseConfig interface {
	GetMySQL() MySQLConfig
	GetMongoDB() MongoDBConfig
	GetRedis() RedisConfig
}

// MySQLConfig defines the interface for MySQL configuration
type MySQLConfig interface {
	GetHost() string
	GetPort() int
	GetUsername() string
	GetPassword() string
	GetDatabase() string
	GetMaxConnections() int
	GetMaxIdleConnections() int
	GetConnectionMaxLifetime() time.Duration
}

// MongoDBConfig defines the interface for MongoDB configuration
type MongoDBConfig interface {
	GetURI() string
	GetDatabase() string
	GetMaxPoolSize() int
	GetMinPoolSize() int
	GetMaxIdleTime() time.Duration
}

// RedisConfig defines the interface for Redis configuration
type RedisConfig interface {
	GetHost() string
	GetPort() int
	GetPassword() string
	GetDB() int
	GetMaxRetries() int
	GetPoolSize() int
	GetMinIdleConnections() int
}

// CacheConfig defines the interface for cache configuration
type CacheConfig interface {
	GetEnabled() bool
	GetType() string
	GetTTL() time.Duration
	GetMaxSize() int64
	GetCleanupInterval() time.Duration
	GetEvictionPolicy() string
}

// MessageQueueConfig defines the interface for message queue configuration
type MessageQueueConfig interface {
	GetEnabled() bool
	GetBrokers() []string
	GetTopics() []string
	GetConsumerGroup() string
	GetAutoCommit() bool
	GetCommitInterval() time.Duration
}

// WebSocketConfig defines the interface for WebSocket configuration
type WebSocketConfig interface {
	GetEnabled() bool
	GetPort() int
	GetReadBufferSize() int
	GetWriteBufferSize() int
	GetHandshakeTimeout() time.Duration
	GetPingPeriod() time.Duration
	GetPongWait() time.Duration
	GetWriteWait() time.Duration
	GetMaxMessageSize() int
	GetMaxConnections() int
}

// SecurityConfig defines the interface for security configuration
type SecurityConfig interface {
	GetEnabled() bool
	GetJWTSecret() string
	GetTokenExpiry() time.Duration
	GetAllowedOrigins() []string
	GetRateLimitEnabled() bool
	GetRateLimitRequests() int
	GetRateLimitWindow() time.Duration
	GetCORSEnabled() bool
	GetCORSOrigins() []string
}

// MonitoringConfig defines the interface for monitoring configuration
type MonitoringConfig interface {
	GetEnabled() bool
	GetPort() int
	GetPath() string
	GetCollectInterval() time.Duration
	GetHealthCheckInterval() time.Duration
	GetMetricsRetention() time.Duration
	GetAlertThresholds() map[string]interface{}
}

// LoggingConfig defines the interface for logging configuration
type LoggingConfig interface {
	GetLevel() string
	GetFormat() string
	GetOutput() string
	GetFilePath() string
	GetMaxSize() int
	GetMaxBackups() int
	GetMaxAge() int
	GetCompress() bool
}
