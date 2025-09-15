package database

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
	"singleton-service/internal/config"
	"singleton-service/internal/logger"
)

// MySQLManager implements Singleton pattern for MySQL database connection
type MySQLManager struct {
	db    *sql.DB
	mutex sync.RWMutex
}

var (
	mysqlManager *MySQLManager
	mysqlOnce    sync.Once
)

// GetMySQLManager returns the singleton instance of MySQLManager
func GetMySQLManager() *MySQLManager {
	mysqlOnce.Do(func() {
		mysqlManager = &MySQLManager{}
		mysqlManager.connect()
	})
	return mysqlManager
}

// connect establishes connection to MySQL database
func (mm *MySQLManager) connect() {
	cfg := config.GetConfigManager()
	dbConfig := cfg.GetDatabaseConfig()
	log := logger.GetLogger()

	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		dbConfig.User,
		dbConfig.Password,
		dbConfig.Host,
		dbConfig.Port,
		dbConfig.DBName,
	)

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal("Failed to open MySQL connection", "error", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(dbConfig.MaxConns)
	db.SetMaxIdleConns(dbConfig.MaxIdle)
	db.SetConnMaxLifetime(time.Hour)

	// Test connection
	if err := db.Ping(); err != nil {
		log.Fatal("Failed to ping MySQL database", "error", err)
	}

	mm.mutex.Lock()
	mm.db = db
	mm.mutex.Unlock()

	log.Info("MySQL connection established successfully")
}

// GetDB returns the database connection
func (mm *MySQLManager) GetDB() *sql.DB {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.db
}

// Close closes the database connection
func (mm *MySQLManager) Close() error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	if mm.db != nil {
		return mm.db.Close()
	}
	return nil
}

// Ping tests the database connection
func (mm *MySQLManager) Ping() error {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	
	if mm.db == nil {
		return fmt.Errorf("database connection is nil")
	}
	return mm.db.Ping()
}

// Health check for the database
func (mm *MySQLManager) HealthCheck() error {
	return mm.Ping()
}

// CreateTables creates necessary tables
func (mm *MySQLManager) CreateTables() error {
	mm.mutex.RLock()
	db := mm.db
	mm.mutex.RUnlock()

	queries := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id VARCHAR(36) PRIMARY KEY,
			email VARCHAR(255) UNIQUE NOT NULL,
			name VARCHAR(255) NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS payments (
			id VARCHAR(36) PRIMARY KEY,
			user_id VARCHAR(36) NOT NULL,
			amount DECIMAL(10,2) NOT NULL,
			currency VARCHAR(3) NOT NULL,
			status VARCHAR(50) NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
			FOREIGN KEY (user_id) REFERENCES users(id)
		)`,
		`CREATE TABLE IF NOT EXISTS audit_logs (
			id VARCHAR(36) PRIMARY KEY,
			action VARCHAR(100) NOT NULL,
			entity_type VARCHAR(50) NOT NULL,
			entity_id VARCHAR(36) NOT NULL,
			user_id VARCHAR(36),
			details JSON,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
	}

	for _, query := range queries {
		if _, err := db.Exec(query); err != nil {
			return fmt.Errorf("failed to create table: %w", err)
		}
	}

	log := logger.GetLogger()
	log.Info("Database tables created successfully")
	return nil
}
