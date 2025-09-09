package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/gorilla/websocket"
	"github.com/patrickmn/go-cache"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"unit_of_work/internal/unit_of_work"
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize services
	unitOfWorkManager := initUnitOfWorkManager(logger)

	// Initialize databases
	mysqlDB := initMySQL()
	mongoDB := initMongoDB()
	redisClient := initRedis()

	// Initialize cache
	cacheClient := cache.New(5*time.Minute, 10*time.Minute)

	// Initialize WebSocket hub
	hub := initWebSocketHub()

	// Initialize Kafka producer
	kafkaProducer := initKafkaProducer()

	// Initialize configuration
	config := &unit_of_work.UnitOfWorkConfig{
		Name:                    "Unit of Work Service",
		Version:                 "1.0.0",
		Description:             "Unit of Work pattern implementation with microservice architecture",
		MaxEntities:             10000,
		MaxRepositories:         100,
		TransactionTimeout:      30 * time.Second,
		CleanupInterval:         1 * time.Hour,
		ValidationEnabled:       true,
		CachingEnabled:          true,
		MonitoringEnabled:       true,
		AuditingEnabled:         true,
		SupportedEntityTypes:    []string{"user", "order", "product", "payment", "custom"},
		DefaultEntityType:       "custom",
		ValidationRules: map[string]interface{}{
			"max_name_length": 100,
			"max_description_length": 500,
		},
		Metadata: map[string]interface{}{
			"environment": "production",
			"region":      "us-east-1",
		},
		Database: unit_of_work.DatabaseConfig{
			MySQL: unit_of_work.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "unit_of_work_db",
			},
			MongoDB: unit_of_work.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "unit_of_work_db",
			},
			Redis: unit_of_work.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: unit_of_work.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TTL:             5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: unit_of_work.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"unit-of-work-events"},
		},
		WebSocket: unit_of_work.WebSocketConfig{
			Enabled:           true,
			Port:              8080,
			ReadBufferSize:    1024,
			WriteBufferSize:   1024,
			HandshakeTimeout:  10 * time.Second,
		},
		Security: unit_of_work.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: unit_of_work.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: unit_of_work.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Initialize unit of work service
	unitOfWorkService := unit_of_work.NewUnitOfWorkServiceManager(config)

	// Initialize router
	router := gin.Default()

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		healthChecks := map[string]interface{}{
			"mysql":    checkMySQLHealth(mysqlDB),
			"mongodb":  checkMongoDBHealth(mongoDB),
			"redis":    checkRedisHealth(redisClient),
			"cache":    checkCacheHealth(cacheClient),
			"websocket": checkWebSocketHealth(hub),
			"kafka":    checkKafkaHealth(kafkaProducer),
		}

		status := http.StatusOK
		for _, check := range healthChecks {
			if !check.(bool) {
				status = http.StatusServiceUnavailable
				break
			}
		}

		c.JSON(status, gin.H{
			"status":     "healthy",
			"components": healthChecks,
			"timestamp":  time.Now(),
		})
	})

	// Unit of Work endpoints
	unitOfWorkGroup := router.Group("/api/v1/unit-of-work")
	{
		unitOfWorkGroup.POST("/entities/new", func(c *gin.Context) {
			var req struct {
				Type        string `json:"type"`
				Name        string `json:"name"`
				Description string `json:"description"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create entity based on type
			var entity unit_of_work.Entity
			switch req.Type {
			case "user":
				entity = unit_of_work.NewUserEntity(req.Name, req.Description, "", "", "", "user")
			case "order":
				entity = unit_of_work.NewOrderEntity(req.Name, req.Description, "", "", 0, "USD")
			case "product":
				entity = unit_of_work.NewProductEntity(req.Name, req.Description, "", 0, "USD", "", "")
			case "payment":
				entity = unit_of_work.NewPaymentEntity(req.Name, req.Description, "", "", 0, "USD", "card")
			default:
				entity = &unit_of_work.ConcreteEntity{
					ID:          unit_of_work.GenerateID(),
					Type:        req.Type,
					Name:        req.Name,
					Description: req.Description,
					Metadata:    make(map[string]interface{}),
					CreatedAt:   time.Now(),
					UpdatedAt:   time.Now(),
					Active:      true,
					Dirty:       false,
					New:         true,
					Deleted:     false,
				}
			}

			// Register new entity
			err := unitOfWorkService.RegisterNew(entity)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, gin.H{
				"message": "Entity registered as new",
				"entity":  entity.GetID(),
			})
		})

		unitOfWorkGroup.POST("/entities/dirty", func(c *gin.Context) {
			var req struct {
				EntityID string `json:"entity_id"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create a mock entity for demonstration
			entity := &unit_of_work.ConcreteEntity{
				ID:          req.EntityID,
				Type:        "custom",
				Name:        "Mock Entity",
				Description: "Mock entity for demonstration",
				Metadata:    make(map[string]interface{}),
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
				Active:      true,
				Dirty:       true,
				New:         false,
				Deleted:     false,
			}

			// Register dirty entity
			err := unitOfWorkService.RegisterDirty(entity)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Entity registered as dirty"})
		})

		unitOfWorkGroup.POST("/entities/deleted", func(c *gin.Context) {
			var req struct {
				EntityID string `json:"entity_id"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create a mock entity for demonstration
			entity := &unit_of_work.ConcreteEntity{
				ID:          req.EntityID,
				Type:        "custom",
				Name:        "Mock Entity",
				Description: "Mock entity for demonstration",
				Metadata:    make(map[string]interface{}),
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
				Active:      true,
				Dirty:       false,
				New:         false,
				Deleted:     true,
			}

			// Register deleted entity
			err := unitOfWorkService.RegisterDeleted(entity)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Entity registered as deleted"})
		})

		unitOfWorkGroup.POST("/entities/clean", func(c *gin.Context) {
			var req struct {
				EntityID string `json:"entity_id"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create a mock entity for demonstration
			entity := &unit_of_work.ConcreteEntity{
				ID:          req.EntityID,
				Type:        "custom",
				Name:        "Mock Entity",
				Description: "Mock entity for demonstration",
				Metadata:    make(map[string]interface{}),
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
				Active:      true,
				Dirty:       false,
				New:         false,
				Deleted:     false,
			}

			// Register clean entity
			err := unitOfWorkService.RegisterClean(entity)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Entity registered as clean"})
		})

		unitOfWorkGroup.POST("/commit", func(c *gin.Context) {
			// Commit unit of work
			err := unitOfWorkService.Commit()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Unit of work committed successfully"})
		})

		unitOfWorkGroup.POST("/rollback", func(c *gin.Context) {
			// Rollback unit of work
			err := unitOfWorkService.Rollback()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Unit of work rolled back successfully"})
		})

		unitOfWorkGroup.GET("/entities", func(c *gin.Context) {
			entities := unitOfWorkService.GetEntities()
			c.JSON(http.StatusOK, gin.H{"entities": entities})
		})

		unitOfWorkGroup.GET("/entities/new", func(c *gin.Context) {
			entities := unitOfWorkService.GetNewEntities()
			c.JSON(http.StatusOK, gin.H{"entities": entities})
		})

		unitOfWorkGroup.GET("/entities/dirty", func(c *gin.Context) {
			entities := unitOfWorkService.GetDirtyEntities()
			c.JSON(http.StatusOK, gin.H{"entities": entities})
		})

		unitOfWorkGroup.GET("/entities/deleted", func(c *gin.Context) {
			entities := unitOfWorkService.GetDeletedEntities()
			c.JSON(http.StatusOK, gin.H{"entities": entities})
		})

		unitOfWorkGroup.GET("/entities/clean", func(c *gin.Context) {
			entities := unitOfWorkService.GetCleanEntities()
			c.JSON(http.StatusOK, gin.H{"entities": entities})
		})

		unitOfWorkGroup.DELETE("/entities", func(c *gin.Context) {
			err := unitOfWorkService.Clear()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "All entities cleared successfully"})
		})
	}

	// Repository endpoints
	repositoryGroup := router.Group("/api/v1/repositories")
	{
		repositoryGroup.POST("/", func(c *gin.Context) {
			var req struct {
				EntityType  string `json:"entity_type"`
				Name        string `json:"name"`
				Description string `json:"description"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create repository based on entity type
			var repository unit_of_work.Repository
			switch req.EntityType {
			case "user":
				repository = unit_of_work.NewUserRepository(req.Name, req.Description)
			case "order":
				repository = unit_of_work.NewOrderRepository(req.Name, req.Description)
			case "product":
				repository = unit_of_work.NewProductRepository(req.Name, req.Description)
			case "payment":
				repository = unit_of_work.NewPaymentRepository(req.Name, req.Description)
			default:
				repository = &unit_of_work.ConcreteRepository{
					ID:          unit_of_work.GenerateID(),
					Type:        req.EntityType,
					Name:        req.Name,
					Description: req.Description,
					Metadata:    make(map[string]interface{}),
					CreatedAt:   time.Now(),
					UpdatedAt:   time.Now(),
					Active:      true,
					Entities:    make(map[string]unit_of_work.Entity),
				}
			}

			// Register repository
			err := unitOfWorkService.RegisterRepository(req.EntityType, repository)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, gin.H{
				"message":    "Repository registered successfully",
				"repository": repository.GetID(),
			})
		})

		repositoryGroup.GET("/:entity_type", func(c *gin.Context) {
			entityType := c.Param("entity_type")
			repository, err := unitOfWorkService.GetRepository(entityType)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"id":          repository.GetID(),
				"type":        repository.GetType(),
				"name":        repository.GetName(),
				"description": repository.GetDescription(),
				"active":      repository.IsActive(),
				"created_at":  repository.GetCreatedAt(),
				"updated_at":  repository.GetUpdatedAt(),
				"metadata":    repository.GetMetadata(),
			})
		})

		repositoryGroup.GET("/", func(c *gin.Context) {
			repositoryTypes := unitOfWorkService.GetRepositoryTypes()
			c.JSON(http.StatusOK, gin.H{"repository_types": repositoryTypes})
		})
	}

	// Stats endpoint
	router.GET("/api/v1/stats", func(c *gin.Context) {
		stats := unitOfWorkService.GetStats()
		c.JSON(http.StatusOK, stats)
	})

	// Service info endpoint
	router.GET("/api/v1/info", func(c *gin.Context) {
		info := unitOfWorkService.GetServiceInfo()
		c.JSON(http.StatusOK, info)
	})

	// WebSocket endpoint
	router.GET("/ws", func(c *gin.Context) {
		handleWebSocket(c, hub)
	})

	// Start server
	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Graceful shutdown
	logger.Info("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Server exited")
}

// Mock implementations for demonstration
type MockUnitOfWorkManager struct{}

func (muowm *MockUnitOfWorkManager) RegisterNew(entity unit_of_work.Entity) error {
	return nil
}

func (muowm *MockUnitOfWorkManager) RegisterDirty(entity unit_of_work.Entity) error {
	return nil
}

func (muowm *MockUnitOfWorkManager) RegisterDeleted(entity unit_of_work.Entity) error {
	return nil
}

func (muowm *MockUnitOfWorkManager) RegisterClean(entity unit_of_work.Entity) error {
	return nil
}

func (muowm *MockUnitOfWorkManager) Commit() error {
	return nil
}

func (muowm *MockUnitOfWorkManager) Rollback() error {
	return nil
}

func (muowm *MockUnitOfWorkManager) GetRepository(entityType string) (unit_of_work.Repository, error) {
	return nil, nil
}

func (muowm *MockUnitOfWorkManager) RegisterRepository(entityType string, repository unit_of_work.Repository) error {
	return nil
}

func (muowm *MockUnitOfWorkManager) GetEntities() map[string][]unit_of_work.Entity {
	return make(map[string][]unit_of_work.Entity)
}

func (muowm *MockUnitOfWorkManager) GetNewEntities() []unit_of_work.Entity {
	return []unit_of_work.Entity{}
}

func (muowm *MockUnitOfWorkManager) GetDirtyEntities() []unit_of_work.Entity {
	return []unit_of_work.Entity{}
}

func (muowm *MockUnitOfWorkManager) GetDeletedEntities() []unit_of_work.Entity {
	return []unit_of_work.Entity{}
}

func (muowm *MockUnitOfWorkManager) GetCleanEntities() []unit_of_work.Entity {
	return []unit_of_work.Entity{}
}

func (muowm *MockUnitOfWorkManager) Clear() error {
	return nil
}

func (muowm *MockUnitOfWorkManager) GetStats() map[string]interface{} {
	return map[string]interface{}{}
}

func (muowm *MockUnitOfWorkManager) GetServiceInfo() map[string]interface{} {
	return map[string]interface{}{}
}

func (muowm *MockUnitOfWorkManager) GetHealthStatus() map[string]interface{} {
	return map[string]interface{}{}
}

type MockWebSocketHub struct {
	broadcast chan []byte
}

func (mwh *MockWebSocketHub) Run() {}
func (mwh *MockWebSocketHub) Register(client *websocket.Conn) {}
func (mwh *MockWebSocketHub) Unregister(client *websocket.Conn) {}
func (mwh *MockWebSocketHub) Broadcast(message []byte) {
	mwh.broadcast <- message
}

type MockKafkaProducer struct{}

func (mkp *MockKafkaProducer) SendMessage(topic string, message []byte) error {
	return nil
}

func (mkp *MockKafkaProducer) Close() error {
	return nil
}

// Initialize mock services
func initUnitOfWorkManager(logger *zap.Logger) *MockUnitOfWorkManager {
	return &MockUnitOfWorkManager{}
}

func initMySQL() *gorm.DB {
	// Mock MySQL connection
	return nil
}

func initMongoDB() *mongo.Client {
	// Mock MongoDB connection
	return nil
}

func initRedis() *redis.Client {
	// Mock Redis connection
	return nil
}

func initWebSocketHub() *MockWebSocketHub {
	return &MockWebSocketHub{
		broadcast: make(chan []byte),
	}
}

func initKafkaProducer() *MockKafkaProducer {
	return &MockKafkaProducer{}
}

// Health check functions
func checkMySQLHealth(db *gorm.DB) bool {
	// Mock health check
	return true
}

func checkMongoDBHealth(client *mongo.Client) bool {
	// Mock health check
	return true
}

func checkRedisHealth(client *redis.Client) bool {
	// Mock health check
	return true
}

func checkCacheHealth(cache *cache.Cache) bool {
	// Mock health check
	return true
}

func checkWebSocketHealth(hub *MockWebSocketHub) bool {
	// Mock health check
	return true
}

func checkKafkaHealth(producer *MockKafkaProducer) bool {
	// Mock health check
	return true
}

// WebSocket handler
func handleWebSocket(c *gin.Context, hub *MockWebSocketHub) {
	// Mock WebSocket handling
	c.JSON(http.StatusOK, gin.H{"message": "WebSocket endpoint"})
}
