package factory

import (
	"context"
	"database/sql"
	"fmt"
	"sync"

	"factory-service/internal/config"
	"factory-service/internal/logger"
	"go.mongodb.org/mongo-driver/mongo"
)

// DatabaseConnection interface defines the contract for database connections
type DatabaseConnection interface {
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error
	Ping(ctx context.Context) error
	GetConnection() interface{}
	GetDatabaseType() string
	ExecuteQuery(ctx context.Context, query string, args ...interface{}) (interface{}, error)
}

// DatabaseFactory implements the Factory pattern for creating database connections
type DatabaseFactory struct {
	databases map[string]func() DatabaseConnection
	mutex     sync.RWMutex
}

var (
	databaseFactory *DatabaseFactory
	databaseOnce    sync.Once
)

// GetDatabaseFactory returns the singleton instance of DatabaseFactory
func GetDatabaseFactory() *DatabaseFactory {
	databaseOnce.Do(func() {
		databaseFactory = &DatabaseFactory{
			databases: make(map[string]func() DatabaseConnection),
		}
		databaseFactory.registerDefaultDatabases()
	})
	return databaseFactory
}

// registerDefaultDatabases registers the default database connections
func (df *DatabaseFactory) registerDefaultDatabases() {
	df.mutex.Lock()
	defer df.mutex.Unlock()

	// Register MySQL database
	df.databases["mysql"] = func() DatabaseConnection {
		return NewMySQLConnection()
	}

	// Register PostgreSQL database
	df.databases["postgresql"] = func() DatabaseConnection {
		return NewPostgreSQLConnection()
	}

	// Register MongoDB database
	df.databases["mongodb"] = func() DatabaseConnection {
		return NewMongoDBConnection()
	}

	// Register SQLite database
	df.databases["sqlite"] = func() DatabaseConnection {
		return NewSQLiteConnection()
	}
}

// RegisterDatabase registers a new database connection
func (df *DatabaseFactory) RegisterDatabase(name string, creator func() DatabaseConnection) {
	df.mutex.Lock()
	defer df.mutex.Unlock()
	df.databases[name] = creator
}

// CreateDatabase creates a database connection instance
func (df *DatabaseFactory) CreateDatabase(dbType string) (DatabaseConnection, error) {
	df.mutex.RLock()
	creator, exists := df.databases[dbType]
	df.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("database type '%s' not supported", dbType)
	}

	return creator(), nil
}

// GetAvailableDatabases returns the list of available database types
func (df *DatabaseFactory) GetAvailableDatabases() []string {
	df.mutex.RLock()
	defer df.mutex.RUnlock()

	databases := make([]string, 0, len(df.databases))
	for name := range df.databases {
		databases = append(databases, name)
	}
	return databases
}

// MySQLConnection implements DatabaseConnection for MySQL
type MySQLConnection struct {
	db       *sql.DB
	host     string
	port     int
	database string
	username string
	password string
}

// NewMySQLConnection creates a new MySQL connection instance
func NewMySQLConnection() *MySQLConnection {
	cfg := config.GetConfigManager()
	dbConfig := cfg.GetDatabaseConfig()
	return &MySQLConnection{
		host:     dbConfig.Host,
		port:     dbConfig.Port,
		database: dbConfig.DBName,
		username: dbConfig.User,
		password: dbConfig.Password,
	}
}

func (mc *MySQLConnection) Connect(ctx context.Context) error {
	log := logger.GetLogger()
	log.Info("Connecting to MySQL database", "host", mc.host, "port", mc.port, "database", mc.database)

	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		mc.username, mc.password, mc.host, mc.port, mc.database)

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return fmt.Errorf("failed to open MySQL connection: %w", err)
	}

	mc.db = db

	// Test connection
	if err := mc.Ping(ctx); err != nil {
		return fmt.Errorf("failed to ping MySQL database: %w", err)
	}

	log.Info("MySQL connection established successfully")
	return nil
}

func (mc *MySQLConnection) Disconnect(ctx context.Context) error {
	if mc.db != nil {
		return mc.db.Close()
	}
	return nil
}

func (mc *MySQLConnection) Ping(ctx context.Context) error {
	if mc.db == nil {
		return fmt.Errorf("database connection is nil")
	}
	return mc.db.PingContext(ctx)
}

func (mc *MySQLConnection) GetConnection() interface{} {
	return mc.db
}

func (mc *MySQLConnection) GetDatabaseType() string {
	return "mysql"
}

func (mc *MySQLConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	if mc.db == nil {
		return nil, fmt.Errorf("database connection is nil")
	}

	rows, err := mc.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range columns {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	return results, nil
}

// PostgreSQLConnection implements DatabaseConnection for PostgreSQL
type PostgreSQLConnection struct {
	db       *sql.DB
	host     string
	port     int
	database string
	username string
	password string
}

// NewPostgreSQLConnection creates a new PostgreSQL connection instance
func NewPostgreSQLConnection() *PostgreSQLConnection {
	cfg := config.GetConfigManager()
	dbConfig := cfg.GetDatabaseConfig()
	return &PostgreSQLConnection{
		host:     dbConfig.Host,
		port:     dbConfig.Port,
		database: dbConfig.DBName,
		username: dbConfig.User,
		password: dbConfig.Password,
	}
}

func (pc *PostgreSQLConnection) Connect(ctx context.Context) error {
	log := logger.GetLogger()
	log.Info("Connecting to PostgreSQL database", "host", pc.host, "port", pc.port, "database", pc.database)

	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		pc.host, pc.port, pc.username, pc.password, pc.database)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return fmt.Errorf("failed to open PostgreSQL connection: %w", err)
	}

	pc.db = db

	// Test connection
	if err := pc.Ping(ctx); err != nil {
		return fmt.Errorf("failed to ping PostgreSQL database: %w", err)
	}

	log.Info("PostgreSQL connection established successfully")
	return nil
}

func (pc *PostgreSQLConnection) Disconnect(ctx context.Context) error {
	if pc.db != nil {
		return pc.db.Close()
	}
	return nil
}

func (pc *PostgreSQLConnection) Ping(ctx context.Context) error {
	if pc.db == nil {
		return fmt.Errorf("database connection is nil")
	}
	return pc.db.PingContext(ctx)
}

func (pc *PostgreSQLConnection) GetConnection() interface{} {
	return pc.db
}

func (pc *PostgreSQLConnection) GetDatabaseType() string {
	return "postgresql"
}

func (pc *PostgreSQLConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	if pc.db == nil {
		return nil, fmt.Errorf("database connection is nil")
	}

	rows, err := pc.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range columns {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	return results, nil
}

// MongoDBConnection implements DatabaseConnection for MongoDB
type MongoDBConnection struct {
	client   *mongo.Client
	database *mongo.Database
	uri      string
	dbName   string
}

// NewMongoDBConnection creates a new MongoDB connection instance
func NewMongoDBConnection() *MongoDBConnection {
	cfg := config.GetConfigManager()
	mongoConfig := cfg.GetMongoDBConfig()
	return &MongoDBConnection{
		uri:    mongoConfig.URI,
		dbName: mongoConfig.Database,
	}
}

func (mdc *MongoDBConnection) Connect(ctx context.Context) error {
	log := logger.GetLogger()
	log.Info("Connecting to MongoDB database", "uri", mdc.uri, "database", mdc.dbName)

	client, err := mongo.Connect(ctx, mongo.Client().ApplyURI(mdc.uri))
	if err != nil {
		return fmt.Errorf("failed to connect to MongoDB: %w", err)
	}

	mdc.client = client
	mdc.database = client.Database(mdc.dbName)

	// Test connection
	if err := mdc.Ping(ctx); err != nil {
		return fmt.Errorf("failed to ping MongoDB database: %w", err)
	}

	log.Info("MongoDB connection established successfully")
	return nil
}

func (mdc *MongoDBConnection) Disconnect(ctx context.Context) error {
	if mdc.client != nil {
		return mdc.client.Disconnect(ctx)
	}
	return nil
}

func (mdc *MongoDBConnection) Ping(ctx context.Context) error {
	if mdc.client == nil {
		return fmt.Errorf("MongoDB client is nil")
	}
	return mdc.client.Ping(ctx, nil)
}

func (mdc *MongoDBConnection) GetConnection() interface{} {
	return mdc.database
}

func (mdc *MongoDBConnection) GetDatabaseType() string {
	return "mongodb"
}

func (mdc *MongoDBConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	if mdc.database == nil {
		return nil, fmt.Errorf("MongoDB database is nil")
	}

	// For MongoDB, we'll implement a simple collection query
	// In a real implementation, you'd parse the query and execute accordingly
	collection := mdc.database.Collection("test")
	
	cursor, err := collection.Find(ctx, map[string]interface{}{})
	if err != nil {
		return nil, fmt.Errorf("failed to execute MongoDB query: %w", err)
	}
	defer cursor.Close(ctx)

	var results []map[string]interface{}
	if err := cursor.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("failed to decode MongoDB results: %w", err)
	}

	return results, nil
}

// SQLiteConnection implements DatabaseConnection for SQLite
type SQLiteConnection struct {
	db       *sql.DB
	database string
}

// NewSQLiteConnection creates a new SQLite connection instance
func NewSQLiteConnection() *SQLiteConnection {
	cfg := config.GetConfigManager()
	dbConfig := cfg.GetDatabaseConfig()
	return &SQLiteConnection{
		database: dbConfig.DBName,
	}
}

func (sc *SQLiteConnection) Connect(ctx context.Context) error {
	log := logger.GetLogger()
	log.Info("Connecting to SQLite database", "database", sc.database)

	db, err := sql.Open("sqlite3", sc.database)
	if err != nil {
		return fmt.Errorf("failed to open SQLite connection: %w", err)
	}

	sc.db = db

	// Test connection
	if err := sc.Ping(ctx); err != nil {
		return fmt.Errorf("failed to ping SQLite database: %w", err)
	}

	log.Info("SQLite connection established successfully")
	return nil
}

func (sc *SQLiteConnection) Disconnect(ctx context.Context) error {
	if sc.db != nil {
		return sc.db.Close()
	}
	return nil
}

func (sc *SQLiteConnection) Ping(ctx context.Context) error {
	if sc.db == nil {
		return fmt.Errorf("database connection is nil")
	}
	return sc.db.PingContext(ctx)
}

func (sc *SQLiteConnection) GetConnection() interface{} {
	return sc.db
}

func (sc *SQLiteConnection) GetDatabaseType() string {
	return "sqlite"
}

func (sc *SQLiteConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	if sc.db == nil {
		return nil, fmt.Errorf("database connection is nil")
	}

	rows, err := sc.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range columns {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	return results, nil
}
