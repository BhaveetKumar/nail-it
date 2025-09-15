package iterator

import (
	"fmt"
	"time"
)

// BaseIterator provides common functionality for all iterators
type BaseIterator struct {
	Index     int           `json:"index"`
	Size      int           `json:"size"`
	Type      string        `json:"type"`
	Valid     bool          `json:"valid"`
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
}

// GetIndex returns the current index
func (bi *BaseIterator) GetIndex() int {
	return bi.Index
}

// GetSize returns the total size
func (bi *BaseIterator) GetSize() int {
	return bi.Size
}

// GetType returns the iterator type
func (bi *BaseIterator) GetType() string {
	return bi.Type
}

// IsValid returns whether the iterator is valid
func (bi *BaseIterator) IsValid() bool {
	return bi.Valid
}

// Reset resets the iterator to the beginning
func (bi *BaseIterator) Reset() {
	bi.Index = 0
	bi.UpdatedAt = time.Now()
}

// Close closes the iterator
func (bi *BaseIterator) Close() {
	bi.Valid = false
	bi.UpdatedAt = time.Now()
}

// SliceIterator iterates over a slice
type SliceIterator struct {
	BaseIterator
	Items []interface{} `json:"items"`
}

// NewSliceIterator creates a new slice iterator
func NewSliceIterator(items []interface{}) *SliceIterator {
	return &SliceIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      len(items),
			Type:      "slice",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Items: items,
	}
}

// HasNext returns true if there are more items
func (si *SliceIterator) HasNext() bool {
	return si.Index < si.Size
}

// Next returns the next item
func (si *SliceIterator) Next() interface{} {
	if !si.HasNext() {
		return nil
	}
	
	item := si.Items[si.Index]
	si.Index++
	si.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (si *SliceIterator) GetCurrent() interface{} {
	if si.Index == 0 || si.Index > si.Size {
		return nil
	}
	
	return si.Items[si.Index-1]
}

// MapIterator iterates over a map
type MapIterator struct {
	BaseIterator
	Items map[string]interface{} `json:"items"`
	Keys  []string               `json:"keys"`
}

// NewMapIterator creates a new map iterator
func NewMapIterator(items map[string]interface{}) *MapIterator {
	keys := make([]string, 0, len(items))
	for key := range items {
		keys = append(keys, key)
	}
	
	return &MapIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      len(items),
			Type:      "map",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Items: items,
		Keys:  keys,
	}
}

// HasNext returns true if there are more items
func (mi *MapIterator) HasNext() bool {
	return mi.Index < mi.Size
}

// Next returns the next item
func (mi *MapIterator) Next() interface{} {
	if !mi.HasNext() {
		return nil
	}
	
	key := mi.Keys[mi.Index]
	item := map[string]interface{}{
		"key":   key,
		"value": mi.Items[key],
	}
	
	mi.Index++
	mi.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (mi *MapIterator) GetCurrent() interface{} {
	if mi.Index == 0 || mi.Index > mi.Size {
		return nil
	}
	
	key := mi.Keys[mi.Index-1]
	return map[string]interface{}{
		"key":   key,
		"value": mi.Items[key],
	}
}

// ChannelIterator iterates over a channel
type ChannelIterator struct {
	BaseIterator
	Channel <-chan interface{} `json:"-"`
	Current interface{}        `json:"current"`
	Closed  bool               `json:"closed"`
}

// NewChannelIterator creates a new channel iterator
func NewChannelIterator(channel <-chan interface{}) *ChannelIterator {
	return &ChannelIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      -1, // Unknown size for channels
			Type:      "channel",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Channel: channel,
		Closed:  false,
	}
}

// HasNext returns true if there are more items
func (ci *ChannelIterator) HasNext() bool {
	if ci.Closed {
		return false
	}
	
	select {
	case item, ok := <-ci.Channel:
		if !ok {
			ci.Closed = true
			return false
		}
		ci.Current = item
		return true
	default:
		return false
	}
}

// Next returns the next item
func (ci *ChannelIterator) Next() interface{} {
	if !ci.HasNext() {
		return nil
	}
	
	item := ci.Current
	ci.Index++
	ci.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (ci *ChannelIterator) GetCurrent() interface{} {
	return ci.Current
}

// GetSize returns the size (unknown for channels)
func (ci *ChannelIterator) GetSize() int {
	return -1
}

// DatabaseIterator iterates over database results
type DatabaseIterator struct {
	BaseIterator
	Query    interface{} `json:"query"`
	Results  []interface{} `json:"results"`
	Position int         `json:"position"`
}

// NewDatabaseIterator creates a new database iterator
func NewDatabaseIterator(query interface{}, results []interface{}) *DatabaseIterator {
	return &DatabaseIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      len(results),
			Type:      "database",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Query:    query,
		Results:  results,
		Position: 0,
	}
}

// HasNext returns true if there are more items
func (di *DatabaseIterator) HasNext() bool {
	return di.Position < len(di.Results)
}

// Next returns the next item
func (di *DatabaseIterator) Next() interface{} {
	if !di.HasNext() {
		return nil
	}
	
	item := di.Results[di.Position]
	di.Position++
	di.Index++
	di.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (di *DatabaseIterator) GetCurrent() interface{} {
	if di.Position == 0 || di.Position > len(di.Results) {
		return nil
	}
	
	return di.Results[di.Position-1]
}

// FileIterator iterates over file lines
type FileIterator struct {
	BaseIterator
	FilePath string   `json:"file_path"`
	Lines    []string `json:"lines"`
	Position int      `json:"position"`
}

// NewFileIterator creates a new file iterator
func NewFileIterator(filePath string, lines []string) *FileIterator {
	return &FileIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      len(lines),
			Type:      "file",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		FilePath: filePath,
		Lines:    lines,
		Position: 0,
	}
}

// HasNext returns true if there are more items
func (fi *FileIterator) HasNext() bool {
	return fi.Position < len(fi.Lines)
}

// Next returns the next item
func (fi *FileIterator) Next() interface{} {
	if !fi.HasNext() {
		return nil
	}
	
	line := fi.Lines[fi.Position]
	fi.Position++
	fi.Index++
	fi.UpdatedAt = time.Now()
	
	return map[string]interface{}{
		"line_number": fi.Position,
		"content":     line,
		"file_path":   fi.FilePath,
	}
}

// GetCurrent returns the current item
func (fi *FileIterator) GetCurrent() interface{} {
	if fi.Position == 0 || fi.Position > len(fi.Lines) {
		return nil
	}
	
	return map[string]interface{}{
		"line_number": fi.Position,
		"content":     fi.Lines[fi.Position-1],
		"file_path":   fi.FilePath,
	}
}

// FilteredIterator wraps another iterator with filtering
type FilteredIterator struct {
	BaseIterator
	Iterator Iterator `json:"-"`
	Filter   Filter   `json:"-"`
}

// NewFilteredIterator creates a new filtered iterator
func NewFilteredIterator(iterator Iterator, filter Filter) *FilteredIterator {
	return &FilteredIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      iterator.GetSize(),
			Type:      "filtered",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Iterator: iterator,
		Filter:   filter,
	}
}

// HasNext returns true if there are more items
func (fi *FilteredIterator) HasNext() bool {
	for fi.Iterator.HasNext() {
		item := fi.Iterator.Next()
		if fi.Filter.Filter(item) {
			return true
		}
	}
	return false
}

// Next returns the next item
func (fi *FilteredIterator) Next() interface{} {
	if !fi.HasNext() {
		return nil
	}
	
	item := fi.Iterator.GetCurrent()
	fi.Index++
	fi.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (fi *FilteredIterator) GetCurrent() interface{} {
	return fi.Iterator.GetCurrent()
}

// SortedIterator wraps another iterator with sorting
type SortedIterator struct {
	BaseIterator
	Iterator Iterator `json:"-"`
	Sorter   Sorter   `json:"-"`
	Items    []interface{} `json:"items"`
	Position int       `json:"position"`
}

// NewSortedIterator creates a new sorted iterator
func NewSortedIterator(iterator Iterator, sorter Sorter) *SortedIterator {
	items := make([]interface{}, 0)
	for iterator.HasNext() {
		items = append(items, iterator.Next())
	}
	
	sortedItems := sorter.Sort(items)
	
	return &SortedIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      len(sortedItems),
			Type:      "sorted",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Iterator: iterator,
		Sorter:   sorter,
		Items:    sortedItems,
		Position: 0,
	}
}

// HasNext returns true if there are more items
func (si *SortedIterator) HasNext() bool {
	return si.Position < len(si.Items)
}

// Next returns the next item
func (si *SortedIterator) Next() interface{} {
	if !si.HasNext() {
		return nil
	}
	
	item := si.Items[si.Position]
	si.Position++
	si.Index++
	si.UpdatedAt = time.Now()
	
	return item
}

// GetCurrent returns the current item
func (si *SortedIterator) GetCurrent() interface{} {
	if si.Position == 0 || si.Position > len(si.Items) {
		return nil
	}
	
	return si.Items[si.Position-1]
}

// TransformedIterator wraps another iterator with transformation
type TransformedIterator struct {
	BaseIterator
	Iterator     Iterator     `json:"-"`
	Transformer  Transformer  `json:"-"`
}

// NewTransformedIterator creates a new transformed iterator
func NewTransformedIterator(iterator Iterator, transformer Transformer) *TransformedIterator {
	return &TransformedIterator{
		BaseIterator: BaseIterator{
			Index:     0,
			Size:      iterator.GetSize(),
			Type:      "transformed",
			Valid:     true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Iterator:    iterator,
		Transformer: transformer,
	}
}

// HasNext returns true if there are more items
func (ti *TransformedIterator) HasNext() bool {
	return ti.Iterator.HasNext()
}

// Next returns the next item
func (ti *TransformedIterator) Next() interface{} {
	if !ti.HasNext() {
		return nil
	}
	
	item := ti.Iterator.Next()
	transformed := ti.Transformer.Transform(item)
	ti.Index++
	ti.UpdatedAt = time.Now()
	
	return transformed
}

// GetCurrent returns the current item
func (ti *TransformedIterator) GetCurrent() interface{} {
	if ti.Iterator.GetCurrent() == nil {
		return nil
	}
	
	return ti.Transformer.Transform(ti.Iterator.GetCurrent())
}

// Collection implementations

// SliceCollection represents a slice-based collection
type SliceCollection struct {
	Items []interface{} `json:"items"`
	Type  string        `json:"type"`
}

// NewSliceCollection creates a new slice collection
func NewSliceCollection(items []interface{}) *SliceCollection {
	return &SliceCollection{
		Items: items,
		Type:  "slice",
	}
}

// CreateIterator creates an iterator for the collection
func (sc *SliceCollection) CreateIterator() Iterator {
	return NewSliceIterator(sc.Items)
}

// GetSize returns the size of the collection
func (sc *SliceCollection) GetSize() int {
	return len(sc.Items)
}

// GetType returns the collection type
func (sc *SliceCollection) GetType() string {
	return sc.Type
}

// IsEmpty returns true if the collection is empty
func (sc *SliceCollection) IsEmpty() bool {
	return len(sc.Items) == 0
}

// Clear clears the collection
func (sc *SliceCollection) Clear() {
	sc.Items = make([]interface{}, 0)
}

// Add adds an item to the collection
func (sc *SliceCollection) Add(item interface{}) error {
	sc.Items = append(sc.Items, item)
	return nil
}

// Remove removes an item from the collection
func (sc *SliceCollection) Remove(item interface{}) error {
	for i, existingItem := range sc.Items {
		if existingItem == item {
			sc.Items = append(sc.Items[:i], sc.Items[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("item not found")
}

// Contains returns true if the collection contains the item
func (sc *SliceCollection) Contains(item interface{}) bool {
	for _, existingItem := range sc.Items {
		if existingItem == item {
			return true
		}
	}
	return false
}

// ToSlice returns the collection as a slice
func (sc *SliceCollection) ToSlice() []interface{} {
	return sc.Items
}

// MapCollection represents a map-based collection
type MapCollection struct {
	Items map[string]interface{} `json:"items"`
	Type  string                 `json:"type"`
}

// NewMapCollection creates a new map collection
func NewMapCollection(items map[string]interface{}) *MapCollection {
	return &MapCollection{
		Items: items,
		Type:  "map",
	}
}

// CreateIterator creates an iterator for the collection
func (mc *MapCollection) CreateIterator() Iterator {
	return NewMapIterator(mc.Items)
}

// GetSize returns the size of the collection
func (mc *MapCollection) GetSize() int {
	return len(mc.Items)
}

// GetType returns the collection type
func (mc *MapCollection) GetType() string {
	return mc.Type
}

// IsEmpty returns true if the collection is empty
func (mc *MapCollection) IsEmpty() bool {
	return len(mc.Items) == 0
}

// Clear clears the collection
func (mc *MapCollection) Clear() {
	mc.Items = make(map[string]interface{})
}

// Add adds an item to the collection
func (mc *MapCollection) Add(item interface{}) error {
	if keyValue, ok := item.(map[string]interface{}); ok {
		if key, exists := keyValue["key"]; exists {
			if value, exists := keyValue["value"]; exists {
				mc.Items[fmt.Sprintf("%v", key)] = value
				return nil
			}
		}
	}
	return fmt.Errorf("invalid item format")
}

// Remove removes an item from the collection
func (mc *MapCollection) Remove(item interface{}) error {
	if key, ok := item.(string); ok {
		if _, exists := mc.Items[key]; exists {
			delete(mc.Items, key)
			return nil
		}
	}
	return fmt.Errorf("item not found")
}

// Contains returns true if the collection contains the item
func (mc *MapCollection) Contains(item interface{}) bool {
	if key, ok := item.(string); ok {
		_, exists := mc.Items[key]
		return exists
	}
	return false
}

// ToSlice returns the collection as a slice
func (mc *MapCollection) ToSlice() []interface{} {
	slice := make([]interface{}, 0, len(mc.Items))
	for key, value := range mc.Items {
		slice = append(slice, map[string]interface{}{
			"key":   key,
			"value": value,
		})
	}
	return slice
}

// IteratorStatistics represents statistics for an iterator
type IteratorStatistics struct {
	TotalItems     int64     `json:"total_items"`
	ProcessedItems int64     `json:"processed_items"`
	SkippedItems   int64     `json:"skipped_items"`
	AverageTime    float64   `json:"average_time"`
	MaxTime        float64   `json:"max_time"`
	MinTime        float64   `json:"min_time"`
	LastAccess     time.Time `json:"last_access"`
	CreatedAt      time.Time `json:"created_at"`
}

// CacheStats represents cache statistics
type CacheStats struct {
	Hits       int64     `json:"hits"`
	Misses     int64     `json:"misses"`
	HitRate    float64   `json:"hit_rate"`
	Size       int64     `json:"size"`
	MaxSize    int64     `json:"max_size"`
	Evictions  int64     `json:"evictions"`
	LastAccess time.Time `json:"last_access"`
}

// MessageQueueStats represents message queue statistics
type MessageQueueStats struct {
	TotalMessages     int64     `json:"total_messages"`
	PublishedMessages int64     `json:"published_messages"`
	ConsumedMessages  int64     `json:"consumed_messages"`
	FailedMessages    int64     `json:"failed_messages"`
	QueueSize         int64     `json:"queue_size"`
	LastMessage       time.Time `json:"last_message"`
}

// WebSocketStats represents WebSocket statistics
type WebSocketStats struct {
	TotalConnections    int64     `json:"total_connections"`
	ActiveConnections   int64     `json:"active_connections"`
	TotalMessages       int64     `json:"total_messages"`
	MessagesPerSecond   float64   `json:"messages_per_second"`
	LastConnection      time.Time `json:"last_connection"`
	LastDisconnection   time.Time `json:"last_disconnection"`
}

// HealthStatus represents overall health status
type HealthStatus struct {
	Status    string          `json:"status"`
	Timestamp time.Time       `json:"timestamp"`
	Services  []ServiceHealth `json:"services"`
}

// ServiceHealth represents service health
type ServiceHealth struct {
	Name      string        `json:"name"`
	Status    string        `json:"status"`
	Healthy   bool          `json:"healthy"`
	Message   string        `json:"message"`
	Latency   time.Duration `json:"latency"`
	Timestamp time.Time     `json:"timestamp"`
}

// ServiceMetrics represents service metrics
type ServiceMetrics struct {
	ServiceName        string    `json:"service_name"`
	TotalRequests      int64     `json:"total_requests"`
	SuccessfulRequests int64     `json:"successful_requests"`
	FailedRequests     int64     `json:"failed_requests"`
	AverageLatency     float64   `json:"average_latency"`
	MaxLatency         float64   `json:"max_latency"`
	MinLatency         float64   `json:"min_latency"`
	SuccessRate        float64   `json:"success_rate"`
	LastRequest        time.Time `json:"last_request"`
	LastError          time.Time `json:"last_error"`
}

// SystemMetrics represents system metrics
type SystemMetrics struct {
	CPUUsage    float64   `json:"cpu_usage"`
	MemoryUsage float64   `json:"memory_usage"`
	DiskUsage   float64   `json:"disk_usage"`
	NetworkIO   float64   `json:"network_io"`
	Timestamp   time.Time `json:"timestamp"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Details   map[string]interface{} `json:"details"`
	IP        string                 `json:"ip"`
	UserAgent string                 `json:"user_agent"`
	Timestamp time.Time              `json:"timestamp"`
}

// IteratorConfig represents iterator configuration
type IteratorConfig struct {
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	MaxIterators int              `json:"max_iterators"`
	Timeout     time.Duration     `json:"timeout"`
	RetryCount  int               `json:"retry_count"`
	Types       []string          `json:"types"`
	Database    DatabaseConfig    `json:"database"`
	Cache       CacheConfig       `json:"cache"`
	MessageQueue MessageQueueConfig `json:"message_queue"`
	WebSocket   WebSocketConfig   `json:"websocket"`
	Security    SecurityConfig    `json:"security"`
	Monitoring  MonitoringConfig  `json:"monitoring"`
	Logging     LoggingConfig     `json:"logging"`
}

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	MySQL    MySQLConfig    `json:"mysql"`
	MongoDB  MongoDBConfig  `json:"mongodb"`
	Redis    RedisConfig    `json:"redis"`
}

// MySQLConfig represents MySQL configuration
type MySQLConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Username string `json:"username"`
	Password string `json:"password"`
	Database string `json:"database"`
}

// MongoDBConfig represents MongoDB configuration
type MongoDBConfig struct {
	URI      string `json:"uri"`
	Database string `json:"database"`
}

// RedisConfig represents Redis configuration
type RedisConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Password string `json:"password"`
	DB       int    `json:"db"`
}

// CacheConfig represents cache configuration
type CacheConfig struct {
	Enabled         bool          `json:"enabled"`
	Type            string        `json:"type"`
	TTL             time.Duration `json:"ttl"`
	MaxSize         int64         `json:"max_size"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// MessageQueueConfig represents message queue configuration
type MessageQueueConfig struct {
	Enabled bool     `json:"enabled"`
	Brokers []string `json:"brokers"`
	Topics  []string `json:"topics"`
}

// WebSocketConfig represents WebSocket configuration
type WebSocketConfig struct {
	Enabled           bool          `json:"enabled"`
	Port              int           `json:"port"`
	ReadBufferSize    int           `json:"read_buffer_size"`
	WriteBufferSize   int           `json:"write_buffer_size"`
	HandshakeTimeout  time.Duration `json:"handshake_timeout"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	Enabled           bool     `json:"enabled"`
	JWTSecret         string   `json:"jwt_secret"`
	TokenExpiry       time.Duration `json:"token_expiry"`
	AllowedOrigins    []string `json:"allowed_origins"`
	RateLimitEnabled  bool     `json:"rate_limit_enabled"`
	RateLimitRequests int      `json:"rate_limit_requests"`
	RateLimitWindow   time.Duration `json:"rate_limit_window"`
}

// MonitoringConfig represents monitoring configuration
type MonitoringConfig struct {
	Enabled         bool          `json:"enabled"`
	Port            int           `json:"port"`
	Path            string        `json:"path"`
	CollectInterval time.Duration `json:"collect_interval"`
}

// LoggingConfig represents logging configuration
type LoggingConfig struct {
	Level  string `json:"level"`
	Format string `json:"format"`
	Output string `json:"output"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Timestamp time.Time `json:"timestamp"`
}
