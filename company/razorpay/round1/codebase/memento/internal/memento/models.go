package memento

import (
	"time"
)

// BaseOriginator provides common functionality for all originators
type BaseOriginator struct {
	ID           string      `json:"id"`
	Name         string      `json:"name"`
	Type         string      `json:"type"`
	Version      int         `json:"version"`
	State        interface{} `json:"state"`
	LastModified time.Time   `json:"last_modified"`
	Dirty        bool        `json:"dirty"`
}

// GetID returns the originator ID
func (bo *BaseOriginator) GetID() string {
	return bo.ID
}

// GetName returns the originator name
func (bo *BaseOriginator) GetName() string {
	return bo.Name
}

// GetType returns the originator type
func (bo *BaseOriginator) GetType() string {
	return bo.Type
}

// GetVersion returns the originator version
func (bo *BaseOriginator) GetVersion() int {
	return bo.Version
}

// SetVersion sets the originator version
func (bo *BaseOriginator) SetVersion(version int) {
	bo.Version = version
}

// GetState returns the originator state
func (bo *BaseOriginator) GetState() interface{} {
	return bo.State
}

// SetState sets the originator state
func (bo *BaseOriginator) SetState(state interface{}) error {
	bo.State = state
	bo.LastModified = time.Now()
	bo.Dirty = true
	return nil
}

// GetLastModified returns the last modified timestamp
func (bo *BaseOriginator) GetLastModified() time.Time {
	return bo.LastModified
}

// SetLastModified sets the last modified timestamp
func (bo *BaseOriginator) SetLastModified(timestamp time.Time) {
	bo.LastModified = timestamp
}

// IsDirty returns whether the originator is dirty
func (bo *BaseOriginator) IsDirty() bool {
	return bo.Dirty
}

// SetDirty sets the dirty flag
func (bo *BaseOriginator) SetDirty(dirty bool) {
	bo.Dirty = dirty
}

// CreateMemento creates a memento from the current state
func (bo *BaseOriginator) CreateMemento() Memento {
	return &BaseMemento{
		ID:           generateID(),
		OriginatorID: bo.ID,
		State:        bo.State,
		Timestamp:    time.Now(),
		Version:      bo.Version,
		Type:         bo.Type,
		Description:  "Memento for " + bo.Name,
		Metadata:     make(map[string]interface{}),
		Valid:        true,
		Size:         calculateSize(bo.State),
		Checksum:     calculateChecksum(bo.State),
	}
}

// RestoreMemento restores the originator state from a memento
func (bo *BaseOriginator) RestoreMemento(memento Memento) error {
	if memento.GetOriginatorID() != bo.ID {
		return ErrInvalidOriginator
	}

	if !memento.IsValid() {
		return ErrInvalidMemento
	}

	bo.State = memento.GetState()
	bo.Version = memento.GetVersion()
	bo.LastModified = memento.GetTimestamp()
	bo.Dirty = false

	return nil
}

// BaseMemento provides common functionality for all mementos
type BaseMemento struct {
	ID           string                 `json:"id"`
	OriginatorID string                 `json:"originator_id"`
	State        interface{}            `json:"state"`
	Timestamp    time.Time              `json:"timestamp"`
	Version      int                    `json:"version"`
	Type         string                 `json:"type"`
	Description  string                 `json:"description"`
	Metadata     map[string]interface{} `json:"metadata"`
	Valid        bool                   `json:"valid"`
	Size         int64                  `json:"size"`
	Checksum     string                 `json:"checksum"`
}

// GetID returns the memento ID
func (bm *BaseMemento) GetID() string {
	return bm.ID
}

// GetOriginatorID returns the originator ID
func (bm *BaseMemento) GetOriginatorID() string {
	return bm.OriginatorID
}

// GetState returns the memento state
func (bm *BaseMemento) GetState() interface{} {
	return bm.State
}

// GetTimestamp returns the memento timestamp
func (bm *BaseMemento) GetTimestamp() time.Time {
	return bm.Timestamp
}

// GetVersion returns the memento version
func (bm *BaseMemento) GetVersion() int {
	return bm.Version
}

// GetType returns the memento type
func (bm *BaseMemento) GetType() string {
	return bm.Type
}

// GetDescription returns the memento description
func (bm *BaseMemento) GetDescription() string {
	return bm.Description
}

// SetDescription sets the memento description
func (bm *BaseMemento) SetDescription(description string) {
	bm.Description = description
}

// GetMetadata returns the memento metadata
func (bm *BaseMemento) GetMetadata() map[string]interface{} {
	return bm.Metadata
}

// SetMetadata sets the memento metadata
func (bm *BaseMemento) SetMetadata(metadata map[string]interface{}) {
	bm.Metadata = metadata
}

// IsValid returns whether the memento is valid
func (bm *BaseMemento) IsValid() bool {
	return bm.Valid
}

// SetValid sets the valid flag
func (bm *BaseMemento) SetValid(valid bool) {
	bm.Valid = valid
}

// GetSize returns the memento size
func (bm *BaseMemento) GetSize() int64 {
	return bm.Size
}

// GetChecksum returns the memento checksum
func (bm *BaseMemento) GetChecksum() string {
	return bm.Checksum
}

// SetChecksum sets the memento checksum
func (bm *BaseMemento) SetChecksum(checksum string) {
	bm.Checksum = checksum
}

// ConcreteOriginator implements a concrete originator
type ConcreteOriginator struct {
	BaseOriginator
	Data map[string]interface{} `json:"data"`
}

// NewConcreteOriginator creates a new concrete originator
func NewConcreteOriginator(id, name, originatorType string) *ConcreteOriginator {
	return &ConcreteOriginator{
		BaseOriginator: BaseOriginator{
			ID:           id,
			Name:         name,
			Type:         originatorType,
			Version:      1,
			State:        make(map[string]interface{}),
			LastModified: time.Now(),
			Dirty:        false,
		},
		Data: make(map[string]interface{}),
	}
}

// SetData sets the originator data
func (co *ConcreteOriginator) SetData(data map[string]interface{}) error {
	co.Data = data
	co.State = data
	co.LastModified = time.Now()
	co.Dirty = true
	return nil
}

// GetData returns the originator data
func (co *ConcreteOriginator) GetData() map[string]interface{} {
	return co.Data
}

// UpdateData updates the originator data
func (co *ConcreteOriginator) UpdateData(key string, value interface{}) error {
	co.Data[key] = value
	co.State = co.Data
	co.LastModified = time.Now()
	co.Dirty = true
	return nil
}

// DeleteData deletes data from the originator
func (co *ConcreteOriginator) DeleteData(key string) error {
	delete(co.Data, key)
	co.State = co.Data
	co.LastModified = time.Now()
	co.Dirty = true
	return nil
}

// ClearData clears all data from the originator
func (co *ConcreteOriginator) ClearData() error {
	co.Data = make(map[string]interface{})
	co.State = co.Data
	co.LastModified = time.Now()
	co.Dirty = true
	return nil
}

// DocumentOriginator implements a document originator
type DocumentOriginator struct {
	BaseOriginator
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
	Format   string                 `json:"format"`
	Encoding string                 `json:"encoding"`
}

// NewDocumentOriginator creates a new document originator
func NewDocumentOriginator(id, name string) *DocumentOriginator {
	return &DocumentOriginator{
		BaseOriginator: BaseOriginator{
			ID:           id,
			Name:         name,
			Type:         "document",
			Version:      1,
			State:        make(map[string]interface{}),
			LastModified: time.Now(),
			Dirty:        false,
		},
		Content:  "",
		Metadata: make(map[string]interface{}),
		Format:   "text",
		Encoding: "utf-8",
	}
}

// SetContent sets the document content
func (do *DocumentOriginator) SetContent(content string) error {
	do.Content = content
	do.State = map[string]interface{}{
		"content":  content,
		"metadata": do.Metadata,
		"format":   do.Format,
		"encoding": do.Encoding,
	}
	do.LastModified = time.Now()
	do.Dirty = true
	return nil
}

// GetContent returns the document content
func (do *DocumentOriginator) GetContent() string {
	return do.Content
}

// SetMetadata sets the document metadata
func (do *DocumentOriginator) SetMetadata(metadata map[string]interface{}) error {
	do.Metadata = metadata
	do.State = map[string]interface{}{
		"content":  do.Content,
		"metadata": metadata,
		"format":   do.Format,
		"encoding": do.Encoding,
	}
	do.LastModified = time.Now()
	do.Dirty = true
	return nil
}

// GetMetadata returns the document metadata
func (do *DocumentOriginator) GetMetadata() map[string]interface{} {
	return do.Metadata
}

// SetFormat sets the document format
func (do *DocumentOriginator) SetFormat(format string) error {
	do.Format = format
	do.State = map[string]interface{}{
		"content":  do.Content,
		"metadata": do.Metadata,
		"format":   format,
		"encoding": do.Encoding,
	}
	do.LastModified = time.Now()
	do.Dirty = true
	return nil
}

// GetFormat returns the document format
func (do *DocumentOriginator) GetFormat() string {
	return do.Format
}

// SetEncoding sets the document encoding
func (do *DocumentOriginator) SetEncoding(encoding string) error {
	do.Encoding = encoding
	do.State = map[string]interface{}{
		"content":  do.Content,
		"metadata": do.Metadata,
		"format":   do.Format,
		"encoding": encoding,
	}
	do.LastModified = time.Now()
	do.Dirty = true
	return nil
}

// GetEncoding returns the document encoding
func (do *DocumentOriginator) GetEncoding() string {
	return do.Encoding
}

// DatabaseOriginator implements a database originator
type DatabaseOriginator struct {
	BaseOriginator
	Table       string                   `json:"table"`
	Schema      string                   `json:"schema"`
	Records     []map[string]interface{} `json:"records"`
	Indexes     []string                 `json:"indexes"`
	Constraints []string                 `json:"constraints"`
}

// NewDatabaseOriginator creates a new database originator
func NewDatabaseOriginator(id, name, table, schema string) *DatabaseOriginator {
	return &DatabaseOriginator{
		BaseOriginator: BaseOriginator{
			ID:           id,
			Name:         name,
			Type:         "database",
			Version:      1,
			State:        make(map[string]interface{}),
			LastModified: time.Now(),
			Dirty:        false,
		},
		Table:       table,
		Schema:      schema,
		Records:     make([]map[string]interface{}, 0),
		Indexes:     make([]string, 0),
		Constraints: make([]string, 0),
	}
}

// SetTable sets the database table
func (dbo *DatabaseOriginator) SetTable(table string) error {
	dbo.Table = table
	dbo.updateState()
	return nil
}

// GetTable returns the database table
func (dbo *DatabaseOriginator) GetTable() string {
	return dbo.Table
}

// SetSchema sets the database schema
func (dbo *DatabaseOriginator) SetSchema(schema string) error {
	dbo.Schema = schema
	dbo.updateState()
	return nil
}

// GetSchema returns the database schema
func (dbo *DatabaseOriginator) GetSchema() string {
	return dbo.Schema
}

// AddRecord adds a record to the database
func (dbo *DatabaseOriginator) AddRecord(record map[string]interface{}) error {
	dbo.Records = append(dbo.Records, record)
	dbo.updateState()
	return nil
}

// GetRecords returns the database records
func (dbo *DatabaseOriginator) GetRecords() []map[string]interface{} {
	return dbo.Records
}

// UpdateRecord updates a record in the database
func (dbo *DatabaseOriginator) UpdateRecord(index int, record map[string]interface{}) error {
	if index < 0 || index >= len(dbo.Records) {
		return ErrInvalidIndex
	}
	dbo.Records[index] = record
	dbo.updateState()
	return nil
}

// DeleteRecord deletes a record from the database
func (dbo *DatabaseOriginator) DeleteRecord(index int) error {
	if index < 0 || index >= len(dbo.Records) {
		return ErrInvalidIndex
	}
	dbo.Records = append(dbo.Records[:index], dbo.Records[index+1:]...)
	dbo.updateState()
	return nil
}

// ClearRecords clears all records from the database
func (dbo *DatabaseOriginator) ClearRecords() error {
	dbo.Records = make([]map[string]interface{}, 0)
	dbo.updateState()
	return nil
}

// AddIndex adds an index to the database
func (dbo *DatabaseOriginator) AddIndex(index string) error {
	dbo.Indexes = append(dbo.Indexes, index)
	dbo.updateState()
	return nil
}

// GetIndexes returns the database indexes
func (dbo *DatabaseOriginator) GetIndexes() []string {
	return dbo.Indexes
}

// AddConstraint adds a constraint to the database
func (dbo *DatabaseOriginator) AddConstraint(constraint string) error {
	dbo.Constraints = append(dbo.Constraints, constraint)
	dbo.updateState()
	return nil
}

// GetConstraints returns the database constraints
func (dbo *DatabaseOriginator) GetConstraints() []string {
	return dbo.Constraints
}

// updateState updates the originator state
func (dbo *DatabaseOriginator) updateState() {
	dbo.State = map[string]interface{}{
		"table":       dbo.Table,
		"schema":      dbo.Schema,
		"records":     dbo.Records,
		"indexes":     dbo.Indexes,
		"constraints": dbo.Constraints,
	}
	dbo.LastModified = time.Now()
	dbo.Dirty = true
}

// FileOriginator implements a file originator
type FileOriginator struct {
	BaseOriginator
	Path     string                 `json:"path"`
	Content  []byte                 `json:"content"`
	Size     int64                  `json:"size"`
	Modified time.Time              `json:"modified"`
	Metadata map[string]interface{} `json:"metadata"`
}

// NewFileOriginator creates a new file originator
func NewFileOriginator(id, name, path string) *FileOriginator {
	return &FileOriginator{
		BaseOriginator: BaseOriginator{
			ID:           id,
			Name:         name,
			Type:         "file",
			Version:      1,
			State:        make(map[string]interface{}),
			LastModified: time.Now(),
			Dirty:        false,
		},
		Path:     path,
		Content:  make([]byte, 0),
		Size:     0,
		Modified: time.Now(),
		Metadata: make(map[string]interface{}),
	}
}

// SetPath sets the file path
func (fo *FileOriginator) SetPath(path string) error {
	fo.Path = path
	fo.updateState()
	return nil
}

// GetPath returns the file path
func (fo *FileOriginator) GetPath() string {
	return fo.Path
}

// SetContent sets the file content
func (fo *FileOriginator) SetContent(content []byte) error {
	fo.Content = content
	fo.Size = int64(len(content))
	fo.updateState()
	return nil
}

// GetContent returns the file content
func (fo *FileOriginator) GetContent() []byte {
	return fo.Content
}

// GetSize returns the file size
func (fo *FileOriginator) GetSize() int64 {
	return fo.Size
}

// SetModified sets the file modified time
func (fo *FileOriginator) SetModified(modified time.Time) error {
	fo.Modified = modified
	fo.updateState()
	return nil
}

// GetModified returns the file modified time
func (fo *FileOriginator) GetModified() time.Time {
	return fo.Modified
}

// SetMetadata sets the file metadata
func (fo *FileOriginator) SetMetadata(metadata map[string]interface{}) error {
	fo.Metadata = metadata
	fo.updateState()
	return nil
}

// GetMetadata returns the file metadata
func (fo *FileOriginator) GetMetadata() map[string]interface{} {
	return fo.Metadata
}

// updateState updates the originator state
func (fo *FileOriginator) updateState() {
	fo.State = map[string]interface{}{
		"path":     fo.Path,
		"content":  fo.Content,
		"size":     fo.Size,
		"modified": fo.Modified,
		"metadata": fo.Metadata,
	}
	fo.LastModified = time.Now()
	fo.Dirty = true
}

// ConfigurationOriginator implements a configuration originator
type ConfigurationOriginator struct {
	BaseOriginator
	Config      map[string]interface{} `json:"config"`
	Defaults    map[string]interface{} `json:"defaults"`
	Overrides   map[string]interface{} `json:"overrides"`
	Version     string                 `json:"version"`
	Environment string                 `json:"environment"`
}

// NewConfigurationOriginator creates a new configuration originator
func NewConfigurationOriginator(id, name string) *ConfigurationOriginator {
	return &ConfigurationOriginator{
		BaseOriginator: BaseOriginator{
			ID:           id,
			Name:         name,
			Type:         "configuration",
			Version:      1,
			State:        make(map[string]interface{}),
			LastModified: time.Now(),
			Dirty:        false,
		},
		Config:      make(map[string]interface{}),
		Defaults:    make(map[string]interface{}),
		Overrides:   make(map[string]interface{}),
		Version:     "1.0.0",
		Environment: "development",
	}
}

// SetConfig sets the configuration
func (co *ConfigurationOriginator) SetConfig(config map[string]interface{}) error {
	co.Config = config
	co.updateState()
	return nil
}

// GetConfig returns the configuration
func (co *ConfigurationOriginator) GetConfig() map[string]interface{} {
	return co.Config
}

// SetDefaults sets the default configuration
func (co *ConfigurationOriginator) SetDefaults(defaults map[string]interface{}) error {
	co.Defaults = defaults
	co.updateState()
	return nil
}

// GetDefaults returns the default configuration
func (co *ConfigurationOriginator) GetDefaults() map[string]interface{} {
	return co.Defaults
}

// SetOverrides sets the configuration overrides
func (co *ConfigurationOriginator) SetOverrides(overrides map[string]interface{}) error {
	co.Overrides = overrides
	co.updateState()
	return nil
}

// GetOverrides returns the configuration overrides
func (co *ConfigurationOriginator) GetOverrides() map[string]interface{} {
	return co.Overrides
}

// SetVersion sets the configuration version
func (co *ConfigurationOriginator) SetVersion(version string) error {
	co.Version = version
	co.updateState()
	return nil
}

// GetVersion returns the configuration version
func (co *ConfigurationOriginator) GetVersion() string {
	return co.Version
}

// SetEnvironment sets the configuration environment
func (co *ConfigurationOriginator) SetEnvironment(environment string) error {
	co.Environment = environment
	co.updateState()
	return nil
}

// GetEnvironment returns the configuration environment
func (co *ConfigurationOriginator) GetEnvironment() string {
	return co.Environment
}

// updateState updates the originator state
func (co *ConfigurationOriginator) updateState() {
	co.State = map[string]interface{}{
		"config":      co.Config,
		"defaults":    co.Defaults,
		"overrides":   co.Overrides,
		"version":     co.Version,
		"environment": co.Environment,
	}
	co.LastModified = time.Now()
	co.Dirty = true
}

// Helper functions

// generateID generates a unique ID
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

// randomString generates a random string
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}

// calculateSize calculates the size of a state
func calculateSize(state interface{}) int64 {
	// Simple size calculation - in real implementation, use proper serialization
	return int64(len(stringify(state)))
}

// calculateChecksum calculates the checksum of a state
func calculateChecksum(state interface{}) string {
	// Simple checksum calculation - in real implementation, use proper hashing
	return stringify(state)
}

// stringify converts a state to string
func stringify(state interface{}) string {
	// Simple string conversion - in real implementation, use proper serialization
	return "state"
}
