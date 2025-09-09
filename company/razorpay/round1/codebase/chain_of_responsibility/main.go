package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Shopify/sarama"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/gorilla/websocket"
	"github.com/patrickmn/go-cache"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"chain_of_responsibility/internal/chain"
)

var (
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize services
	chainService := initChainService(logger)
	
	// Initialize databases
	mysqlDB := initMySQL()
	mongoDB := initMongoDB()
	redisClient := initRedis()

	// Initialize Kafka
	kafkaProducer := initKafkaProducer()
	kafkaConsumer := initKafkaConsumer()

	// Initialize WebSocket hub
	wsHub := initWebSocketHub()

	// Setup routes
	router := setupRoutes(chainService, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub, logger)

	// Start server
	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	// Start WebSocket hub
	go wsHub.Run()

	// Start Kafka consumer
	go kafkaConsumer.Start()

	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	logger.Info("Chain of Responsibility service started on :8080")

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	logger.Info("Server exited")
}

func initChainService(logger *zap.Logger) *chain.ChainService {
	// Create mock implementations
	cache := cache.New(5*time.Minute, 10*time.Minute)
	metrics := &MockMetrics{}
	database := &MockDatabase{}

	// Create chain configuration
	config := chain.ChainConfig{
		Name:        "chain-of-responsibility-service",
		Version:     "1.0.0",
		Description: "Chain of Responsibility Pattern Implementation Service",
		MaxHandlers: 10,
		Timeout:     30 * time.Second,
		RetryCount:  3,
		Handlers: []chain.HandlerConfig{
			{
				Name:     "authentication",
				Type:     "authentication",
				Priority: 1,
				Enabled:  true,
				Config: map[string]interface{}{
					"jwt_secret":   "your-secret-key",
					"token_expiry": "24h",
				},
			},
			{
				Name:     "authorization",
				Type:     "authorization",
				Priority: 2,
				Enabled:  true,
				Config: map[string]interface{}{
					"permissions": map[string][]string{
						"user":    {"read", "write"},
						"admin":   {"read", "write", "delete", "admin"},
						"moderator": {"read", "write", "moderate"},
					},
				},
			},
			{
				Name:     "validation",
				Type:     "validation",
				Priority: 3,
				Enabled:  true,
				Config: map[string]interface{}{
					"rules": map[string]interface{}{
						"email":    "required|email",
						"password": "required|min:8",
						"username": "required|min:3|max:20",
					},
				},
			},
			{
				Name:     "rate_limit",
				Type:     "rate_limit",
				Priority: 4,
				Enabled:  true,
				Config: map[string]interface{}{
					"requests_per_minute": 100,
					"burst_size":          20,
				},
			},
			{
				Name:     "logging",
				Type:     "logging",
				Priority: 5,
				Enabled:  true,
				Config: map[string]interface{}{
					"log_level":  "info",
					"log_format": "json",
				},
			},
		},
		Database: chain.DatabaseConfig{
			MySQL: chain.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "chain_db",
			},
			MongoDB: chain.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "chain_db",
			},
			Redis: chain.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: chain.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TT:              5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: chain.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"requests", "responses", "logs"},
		},
		WebSocket: chain.WebSocketConfig{
			Enabled:          true,
			Port:             8080,
			ReadBufferSize:   1024,
			WriteBufferSize:  1024,
			HandshakeTimeout: 10 * time.Second,
		},
		Security: chain.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: chain.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: chain.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Create chain manager
	chainManager := chain.NewChainManager(cache, &MockLogger{logger: logger}, metrics, config)

	// Create chain service
	chainService := chain.NewChainService(
		chainManager,
		cache,
		database,
		&MockLogger{logger: logger},
		metrics,
		config,
	)

	return chainService
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/chain_db?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("Failed to connect to MySQL: %v", err)
	}
	return db
}

func initMongoDB() *mongo.Database {
	client, err := mongo.Connect(context.Background(), options.Client().ApplyURI("mongodb://localhost:27017"))
	if err != nil {
		log.Fatalf("Failed to connect to MongoDB: %v", err)
	}
	return client.Database("chain_db")
}

func initRedis() *redis.Client {
	client := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})
	return client
}

func initKafkaProducer() sarama.SyncProducer {
	config := sarama.NewConfig()
	config.Producer.RequiredAcks = sarama.WaitForAll
	config.Producer.Retry.Max = 3
	config.Producer.Return.Successes = true

	producer, err := sarama.NewSyncProducer([]string{"localhost:9092"}, config)
	if err != nil {
		log.Fatalf("Failed to create Kafka producer: %v", err)
	}
	return producer
}

func initKafkaConsumer() sarama.Consumer {
	config := sarama.NewConfig()
	config.Consumer.Return.Errors = true

	consumer, err := sarama.NewConsumer([]string{"localhost:9092"}, config)
	if err != nil {
		log.Fatalf("Failed to create Kafka consumer: %v", err)
	}
	return consumer
}

func initWebSocketHub() *WebSocketHub {
	return NewWebSocketHub()
}

func setupRoutes(
	chainService *chain.ChainService,
	mysqlDB *gorm.DB,
	mongoDB *mongo.Database,
	redisClient *redis.Client,
	kafkaProducer sarama.SyncProducer,
	kafkaConsumer sarama.Consumer,
	wsHub *WebSocketHub,
	logger *zap.Logger,
) *gin.Engine {
	router := gin.Default()

	// Health check
	router.GET("/health", func(c *gin.Context) {
		stats := chainService.GetChainStatistics()
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"stats":  stats,
			"timestamp": time.Now(),
		})
	})

	// Authentication endpoints
	authGroup := router.Group("/api/v1/auth")
	{
		authGroup.POST("/authenticate", func(c *gin.Context) {
			var request chain.AuthenticationRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := chainService.ProcessAuthenticationRequest(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})

		authGroup.POST("/authorize", func(c *gin.Context) {
			var request chain.AuthorizationRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := chainService.ProcessAuthorizationRequest(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// Validation endpoints
	validationGroup := router.Group("/api/v1/validation")
	{
		validationGroup.POST("/validate", func(c *gin.Context) {
			var request chain.ValidationRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := chainService.ProcessValidationRequest(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// Rate limiting endpoints
	rateLimitGroup := router.Group("/api/v1/rate-limit")
	{
		rateLimitGroup.POST("/check", func(c *gin.Context) {
			var request chain.RateLimitRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := chainService.ProcessRateLimitRequest(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// Logging endpoints
	loggingGroup := router.Group("/api/v1/logging")
	{
		loggingGroup.POST("/log", func(c *gin.Context) {
			var request chain.LoggingRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := chainService.ProcessLoggingRequest(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// Chain management endpoints
	chainGroup := router.Group("/api/v1/chain")
	{
		chainGroup.GET("/statistics", func(c *gin.Context) {
			stats := chainService.GetChainStatistics()
			c.JSON(http.StatusOK, stats)
		})

		chainGroup.GET("/handlers", func(c *gin.Context) {
			handlers := chainService.GetAllHandlers()
			c.JSON(http.StatusOK, handlers)
		})

		chainGroup.GET("/handlers/:name", func(c *gin.Context) {
			name := c.Param("name")
			handler, err := chainService.GetHandler(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, handler)
		})

		chainGroup.GET("/handlers/:name/statistics", func(c *gin.Context) {
			name := c.Param("name")
			stats, err := chainService.GetHandlerStatistics(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, stats)
		})

		chainGroup.POST("/optimize", func(c *gin.Context) {
			if err := chainService.OptimizeChain(); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Chain optimized successfully"})
		})

		chainGroup.POST("/validate", func(c *gin.Context) {
			if err := chainService.ValidateChain(); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Chain validation passed"})
		})
	}

	// WebSocket route
	router.GET("/ws", func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			logger.Error("WebSocket upgrade failed", zap.Error(err))
			return
		}
		wsHub.Register(conn)
	})

	return router
}

// Mock implementations for testing
type MockLogger struct {
	logger *zap.Logger
}

func (ml *MockLogger) Info(msg string, fields ...interface{}) {
	ml.logger.Info(msg, zap.Any("fields", fields))
}

func (ml *MockLogger) Error(msg string, fields ...interface{}) {
	ml.logger.Error(msg, zap.Any("fields", fields))
}

func (ml *MockLogger) Debug(msg string, fields ...interface{}) {
	ml.logger.Debug(msg, zap.Any("fields", fields))
}

func (ml *MockLogger) Warn(msg string, fields ...interface{}) {
	ml.logger.Warn(msg, zap.Any("fields", fields))
}

type MockMetrics struct{}

func (mm *MockMetrics) IncrementCounter(name string, labels map[string]string)                     {}
func (mm *MockMetrics) RecordHistogram(name string, value float64, labels map[string]string)       {}
func (mm *MockMetrics) RecordGauge(name string, value float64, labels map[string]string)           {}
func (mm *MockMetrics) RecordTiming(name string, duration time.Duration, labels map[string]string) {}

type MockDatabase struct{}

func (md *MockDatabase) Save(ctx context.Context, collection string, data interface{}) error {
	return nil
}

func (md *MockDatabase) Find(ctx context.Context, collection string, query interface{}) (interface{}, error) {
	return nil, nil
}

func (md *MockDatabase) Update(ctx context.Context, collection string, id string, data interface{}) error {
	return nil
}

func (md *MockDatabase) Delete(ctx context.Context, collection string, id string) error {
	return nil
}

func (md *MockDatabase) Transaction(ctx context.Context, fn func(ctx context.Context) error) error {
	return fn(ctx)
}

// WebSocketHub manages WebSocket connections
type WebSocketHub struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
}

func NewWebSocketHub() *WebSocketHub {
	return &WebSocketHub{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan []byte),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
	}
}

func (h *WebSocketHub) Register(conn *websocket.Conn) {
	h.register <- conn
}

func (h *WebSocketHub) Run() {
	for {
		select {
		case conn := <-h.register:
			h.clients[conn] = true
			log.Printf("WebSocket client connected. Total clients: %d", len(h.clients))

		case conn := <-h.unregister:
			if _, ok := h.clients[conn]; ok {
				delete(h.clients, conn)
				conn.Close()
				log.Printf("WebSocket client disconnected. Total clients: %d", len(h.clients))
			}

		case message := <-h.broadcast:
			for conn := range h.clients {
				if err := conn.WriteMessage(websocket.TextMessage, message); err != nil {
					delete(h.clients, conn)
					conn.Close()
				}
			}
		}
	}
}

func (h *WebSocketHub) Broadcast(message []byte) {
	h.broadcast <- message
}
