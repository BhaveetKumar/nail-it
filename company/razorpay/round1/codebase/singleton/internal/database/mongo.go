package database

import (
	"context"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"singleton-service/internal/config"
	"singleton-service/internal/logger"
)

// MongoManager implements Singleton pattern for MongoDB connection
type MongoManager struct {
	client   *mongo.Client
	database *mongo.Database
	mutex    sync.RWMutex
}

var (
	mongoManager *MongoManager
	mongoOnce    sync.Once
)

// GetMongoManager returns the singleton instance of MongoManager
func GetMongoManager() *MongoManager {
	mongoOnce.Do(func() {
		mongoManager = &MongoManager{}
		mongoManager.connect()
	})
	return mongoManager
}

// connect establishes connection to MongoDB
func (mm *MongoManager) connect() {
	cfg := config.GetConfigManager()
	mongoConfig := cfg.GetMongoDBConfig()
	log := logger.GetLogger()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientOptions := options.Client().ApplyURI(mongoConfig.URI)
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatal("Failed to connect to MongoDB", "error", err)
	}

	// Test connection
	if err := client.Ping(ctx, nil); err != nil {
		log.Fatal("Failed to ping MongoDB", "error", err)
	}

	database := client.Database(mongoConfig.Database)

	mm.mutex.Lock()
	mm.client = client
	mm.database = database
	mm.mutex.Unlock()

	log.Info("MongoDB connection established successfully")
}

// GetClient returns the MongoDB client
func (mm *MongoManager) GetClient() *mongo.Client {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.client
}

// GetDatabase returns the MongoDB database
func (mm *MongoManager) GetDatabase() *mongo.Database {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.database
}

// GetCollection returns a MongoDB collection
func (mm *MongoManager) GetCollection(name string) *mongo.Collection {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.database.Collection(name)
}

// Close closes the MongoDB connection
func (mm *MongoManager) Close() error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	if mm.client != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return mm.client.Disconnect(ctx)
	}
	return nil
}

// Ping tests the MongoDB connection
func (mm *MongoManager) Ping() error {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()

	if mm.client == nil {
		return fmt.Errorf("MongoDB client is nil")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return mm.client.Ping(ctx, nil)
}

// Health check for MongoDB
func (mm *MongoManager) HealthCheck() error {
	return mm.Ping()
}

// CreateIndexes creates necessary indexes
func (mm *MongoManager) CreateIndexes() error {
	mm.mutex.RLock()
	database := mm.database
	mm.mutex.RUnlock()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create indexes for users collection
	usersCollection := database.Collection("users")
	userIndexes := []mongo.IndexModel{
		{
			Keys: map[string]interface{}{"email": 1},
			Options: options.Index().SetUnique(true),
		},
		{
			Keys: map[string]interface{}{"created_at": -1},
		},
	}

	if _, err := usersCollection.Indexes().CreateMany(ctx, userIndexes); err != nil {
		return fmt.Errorf("failed to create user indexes: %w", err)
	}

	// Create indexes for payments collection
	paymentsCollection := database.Collection("payments")
	paymentIndexes := []mongo.IndexModel{
		{
			Keys: map[string]interface{}{"user_id": 1},
		},
		{
			Keys: map[string]interface{}{"status": 1},
		},
		{
			Keys: map[string]interface{}{"created_at": -1},
		},
	}

	if _, err := paymentsCollection.Indexes().CreateMany(ctx, paymentIndexes); err != nil {
		return fmt.Errorf("failed to create payment indexes: %w", err)
	}

	// Create indexes for audit_logs collection
	auditCollection := database.Collection("audit_logs")
	auditIndexes := []mongo.IndexModel{
		{
			Keys: map[string]interface{}{"entity_type": 1, "entity_id": 1},
		},
		{
			Keys: map[string]interface{}{"user_id": 1},
		},
		{
			Keys: map[string]interface{}{"created_at": -1},
		},
	}

	if _, err := auditCollection.Indexes().CreateMany(ctx, auditIndexes); err != nil {
		return fmt.Errorf("failed to create audit log indexes: %w", err)
	}

	log := logger.GetLogger()
	log.Info("MongoDB indexes created successfully")
	return nil
}
