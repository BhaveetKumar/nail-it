package strategy

import (
	"context"
	"fmt"
	"time"
)

// FileLoggingStrategy implements LoggingStrategy for file logging
type FileLoggingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewFileLoggingStrategy creates a new file logging strategy
func NewFileLoggingStrategy() *FileLoggingStrategy {
	return &FileLoggingStrategy{
		timeout:   20 * time.Millisecond,
		available: true,
	}
}

// Log logs message to file
func (f *FileLoggingStrategy) Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error {
	// Simulate file logging
	time.Sleep(f.timeout)

	// Mock file logging
	fmt.Printf("[FILE] %s: %s %v\n", level.String(), message, fields)
	return nil
}

// GetStrategyName returns the strategy name
func (f *FileLoggingStrategy) GetStrategyName() string {
	return "file"
}

// GetSupportedLevels returns supported levels
func (f *FileLoggingStrategy) GetSupportedLevels() []LogLevel {
	return []LogLevel{LogLevelDebug, LogLevelInfo, LogLevelWarn, LogLevelError, LogLevelFatal}
}

// GetLogTime returns log time
func (f *FileLoggingStrategy) GetLogTime() time.Duration {
	return f.timeout
}

// IsAvailable returns availability status
func (f *FileLoggingStrategy) IsAvailable() bool {
	return f.available
}

// ConsoleLoggingStrategy implements LoggingStrategy for console logging
type ConsoleLoggingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewConsoleLoggingStrategy creates a new console logging strategy
func NewConsoleLoggingStrategy() *ConsoleLoggingStrategy {
	return &ConsoleLoggingStrategy{
		timeout:   5 * time.Millisecond,
		available: true,
	}
}

// Log logs message to console
func (c *ConsoleLoggingStrategy) Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error {
	// Simulate console logging
	time.Sleep(c.timeout)

	// Mock console logging
	fmt.Printf("[CONSOLE] %s: %s %v\n", level.String(), message, fields)
	return nil
}

// GetStrategyName returns the strategy name
func (c *ConsoleLoggingStrategy) GetStrategyName() string {
	return "console"
}

// GetSupportedLevels returns supported levels
func (c *ConsoleLoggingStrategy) GetSupportedLevels() []LogLevel {
	return []LogLevel{LogLevelDebug, LogLevelInfo, LogLevelWarn, LogLevelError, LogLevelFatal}
}

// GetLogTime returns log time
func (c *ConsoleLoggingStrategy) GetLogTime() time.Duration {
	return c.timeout
}

// IsAvailable returns availability status
func (c *ConsoleLoggingStrategy) IsAvailable() bool {
	return c.available
}

// DatabaseLoggingStrategy implements LoggingStrategy for database logging
type DatabaseLoggingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewDatabaseLoggingStrategy creates a new database logging strategy
func NewDatabaseLoggingStrategy() *DatabaseLoggingStrategy {
	return &DatabaseLoggingStrategy{
		timeout:   100 * time.Millisecond,
		available: true,
	}
}

// Log logs message to database
func (d *DatabaseLoggingStrategy) Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error {
	// Simulate database logging
	time.Sleep(d.timeout)

	// Mock database logging
	fmt.Printf("[DATABASE] %s: %s %v\n", level.String(), message, fields)
	return nil
}

// GetStrategyName returns the strategy name
func (d *DatabaseLoggingStrategy) GetStrategyName() string {
	return "database"
}

// GetSupportedLevels returns supported levels
func (d *DatabaseLoggingStrategy) GetSupportedLevels() []LogLevel {
	return []LogLevel{LogLevelInfo, LogLevelWarn, LogLevelError, LogLevelFatal}
}

// GetLogTime returns log time
func (d *DatabaseLoggingStrategy) GetLogTime() time.Duration {
	return d.timeout
}

// IsAvailable returns availability status
func (d *DatabaseLoggingStrategy) IsAvailable() bool {
	return d.available
}

// RemoteLoggingStrategy implements LoggingStrategy for remote logging
type RemoteLoggingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewRemoteLoggingStrategy creates a new remote logging strategy
func NewRemoteLoggingStrategy() *RemoteLoggingStrategy {
	return &RemoteLoggingStrategy{
		timeout:   150 * time.Millisecond,
		available: true,
	}
}

// Log logs message to remote service
func (r *RemoteLoggingStrategy) Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error {
	// Simulate remote logging
	time.Sleep(r.timeout)

	// Mock remote logging
	fmt.Printf("[REMOTE] %s: %s %v\n", level.String(), message, fields)
	return nil
}

// GetStrategyName returns the strategy name
func (r *RemoteLoggingStrategy) GetStrategyName() string {
	return "remote"
}

// GetSupportedLevels returns supported levels
func (r *RemoteLoggingStrategy) GetSupportedLevels() []LogLevel {
	return []LogLevel{LogLevelWarn, LogLevelError, LogLevelFatal}
}

// GetLogTime returns log time
func (r *RemoteLoggingStrategy) GetLogTime() time.Duration {
	return r.timeout
}

// IsAvailable returns availability status
func (r *RemoteLoggingStrategy) IsAvailable() bool {
	return r.available
}
