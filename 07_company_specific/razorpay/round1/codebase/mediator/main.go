package main

import (
	"context"
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
	"go.uber.org/zap"
	"gorm.io/gorm"

	"mediator/internal/mediator"
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize services
	mediatorManager := initMediatorManager(logger)

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
	config := &mediator.MediatorConfig{
		Name:          "Mediator Service",
		Version:       "1.0.0",
		Description:   "Mediator pattern implementation with microservice architecture",
		MaxMediators:  100,
		MaxColleagues: 1000,
		Timeout:       30 * time.Minute,
		RetryCount:    3,
		Types:         []string{"message", "event", "command", "query", "notification", "workflow", "service", "resource", "task", "job"},
		Database: mediator.DatabaseConfig{
			MySQL: mediator.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "mediator_db",
			},
			MongoDB: mediator.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "mediator_db",
			},
			Redis: mediator.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: mediator.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TTL:             5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: mediator.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"mediator-events"},
		},
		WebSocket: mediator.WebSocketConfig{
			Enabled:          true,
			Port:             8080,
			ReadBufferSize:   1024,
			WriteBufferSize:  1024,
			HandshakeTimeout: 10 * time.Second,
		},
		Security: mediator.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: mediator.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: mediator.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Initialize mediator service
	mediatorService := mediator.NewMediatorService(config)

	// Initialize router
	router := gin.Default()

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		healthChecks := map[string]interface{}{
			"mysql":     checkMySQLHealth(mysqlDB),
			"mongodb":   checkMongoDBHealth(mongoDB),
			"redis":     checkRedisHealth(redisClient),
			"cache":     checkCacheHealth(cacheClient),
			"websocket": checkWebSocketHealth(hub),
			"kafka":     checkKafkaHealth(kafkaProducer),
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

	// Mediator endpoints
	mediatorGroup := router.Group("/api/v1/mediators")
	{
		mediatorGroup.POST("/", func(c *gin.Context) {
			var req struct {
				Name string `json:"name"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create mediator
			mediator, err := mediatorService.GetMediator().CreateMediator(req.Name)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, gin.H{
				"message":  "Mediator created successfully",
				"mediator": req.Name,
			})
		})

		mediatorGroup.GET("/:name", func(c *gin.Context) {
			name := c.Param("name")
			mediator, err := mediatorService.GetMediator().GetMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			colleagues := mediator.GetColleagues()
			colleagueInfo := make([]map[string]interface{}, 0, len(colleagues))
			for _, colleague := range colleagues {
				colleagueInfo = append(colleagueInfo, map[string]interface{}{
					"id":     colleague.GetID(),
					"name":   colleague.GetName(),
					"type":   colleague.GetType(),
					"active": colleague.IsActive(),
				})
			}

			c.JSON(http.StatusOK, gin.H{
				"name":       name,
				"colleagues": colleagueInfo,
				"count":      len(colleagues),
			})
		})

		mediatorGroup.DELETE("/:name", func(c *gin.Context) {
			name := c.Param("name")
			err := mediatorService.GetMediator().RemoveMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Mediator removed successfully"})
		})

		mediatorGroup.GET("/", func(c *gin.Context) {
			mediators := mediatorService.GetMediator().ListMediators()
			c.JSON(http.StatusOK, gin.H{"mediators": mediators})
		})

		mediatorGroup.POST("/:name/colleagues", func(c *gin.Context) {
			name := c.Param("name")
			var req struct {
				ID   string `json:"id"`
				Name string `json:"name"`
				Type string `json:"type"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create colleague
			colleague := &mediator.BaseColleague{
				ID:           req.ID,
				Name:         req.Name,
				Type:         req.Type,
				Active:       true,
				LastActivity: time.Now(),
			}

			// Get mediator
			mediator, err := mediatorService.GetMediator().GetMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Register colleague
			err = mediator.RegisterColleague(colleague)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, gin.H{"message": "Colleague registered successfully"})
		})

		mediatorGroup.DELETE("/:name/colleagues/:colleagueID", func(c *gin.Context) {
			name := c.Param("name")
			colleagueID := c.Param("colleagueID")

			// Get mediator
			mediator, err := mediatorService.GetMediator().GetMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Unregister colleague
			err = mediator.UnregisterColleague(colleagueID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Colleague unregistered successfully"})
		})

		mediatorGroup.POST("/:name/messages", func(c *gin.Context) {
			name := c.Param("name")
			var req struct {
				SenderID    string      `json:"sender_id"`
				RecipientID string      `json:"recipient_id"`
				Message     interface{} `json:"message"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Get mediator
			mediator, err := mediatorService.GetMediator().GetMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Send message
			err = mediator.SendMessage(req.SenderID, req.RecipientID, req.Message)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Message sent successfully"})
		})

		mediatorGroup.POST("/:name/broadcast", func(c *gin.Context) {
			name := c.Param("name")
			var req struct {
				SenderID string      `json:"sender_id"`
				Message  interface{} `json:"message"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Get mediator
			mediator, err := mediatorService.GetMediator().GetMediator(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Broadcast message
			err = mediator.BroadcastMessage(req.SenderID, req.Message)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Message broadcasted successfully"})
		})
	}

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
type MockMediatorManager struct{}

func (mmm *MockMediatorManager) CreateMediator(name string) (mediator.Mediator, error) {
	return nil, nil
}

func (mmm *MockMediatorManager) GetMediator(name string) (mediator.Mediator, error) {
	return nil, nil
}

func (mmm *MockMediatorManager) RemoveMediator(name string) error {
	return nil
}

func (mmm *MockMediatorManager) ListMediators() []string {
	return []string{"test-mediator"}
}

type MockWebSocketHub struct {
	broadcast chan []byte
}

func (mwh *MockWebSocketHub) Run()                              {}
func (mwh *MockWebSocketHub) Register(client *websocket.Conn)   {}
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
func initMediatorManager(logger *zap.Logger) *MockMediatorManager {
	return &MockMediatorManager{}
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
