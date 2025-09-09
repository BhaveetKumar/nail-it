package logger

import (
	"os"
	"sync"

	"github.com/sirupsen/logrus"
)

// Logger implements Singleton pattern for logging
type Logger struct {
	logger *logrus.Logger
	mutex  sync.RWMutex
}

var (
	loggerInstance *Logger
	loggerOnce     sync.Once
)

// GetLogger returns the singleton instance of Logger
func GetLogger() *Logger {
	loggerOnce.Do(func() {
		loggerInstance = &Logger{
			logger: logrus.New(),
		}
		loggerInstance.setupLogger()
	})
	return loggerInstance
}

// setupLogger configures the logger
func (l *Logger) setupLogger() {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	// Set log level
	level := os.Getenv("LOG_LEVEL")
	if level == "" {
		level = "info"
	}

	logLevel, err := logrus.ParseLevel(level)
	if err != nil {
		logLevel = logrus.InfoLevel
	}

	l.logger.SetLevel(logLevel)

	// Set formatter
	l.logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: "2006-01-02 15:04:05",
	})

	// Set output
	l.logger.SetOutput(os.Stdout)
}

// Info logs an info message
func (l *Logger) Info(message string, fields ...interface{}) {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	entry := l.logger.WithFields(l.convertToFields(fields...))
	entry.Info(message)
}

// Error logs an error message
func (l *Logger) Error(message string, fields ...interface{}) {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	entry := l.logger.WithFields(l.convertToFields(fields...))
	entry.Error(message)
}

// Warn logs a warning message
func (l *Logger) Warn(message string, fields ...interface{}) {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	entry := l.logger.WithFields(l.convertToFields(fields...))
	entry.Warn(message)
}

// Debug logs a debug message
func (l *Logger) Debug(message string, fields ...interface{}) {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	entry := l.logger.WithFields(l.convertToFields(fields...))
	entry.Debug(message)
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(message string, fields ...interface{}) {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	entry := l.logger.WithFields(l.convertToFields(fields...))
	entry.Fatal(message)
}

// WithField creates a new logger entry with a field
func (l *Logger) WithField(key string, value interface{}) *logrus.Entry {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	return l.logger.WithField(key, value)
}

// WithFields creates a new logger entry with fields
func (l *Logger) WithFields(fields logrus.Fields) *logrus.Entry {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	
	return l.logger.WithFields(fields)
}

// convertToFields converts variadic interface{} to logrus.Fields
func (l *Logger) convertToFields(fields ...interface{}) logrus.Fields {
	result := make(logrus.Fields)
	
	for i := 0; i < len(fields); i += 2 {
		if i+1 < len(fields) {
			if key, ok := fields[i].(string); ok {
				result[key] = fields[i+1]
			}
		}
	}
	
	return result
}

// SetLevel sets the log level
func (l *Logger) SetLevel(level logrus.Level) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.logger.SetLevel(level)
}

// GetLevel returns the current log level
func (l *Logger) GetLevel() logrus.Level {
	l.mutex.RLock()
	defer l.mutex.RUnlock()
	return l.logger.GetLevel()
}
