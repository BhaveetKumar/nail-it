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

	"flyweight/internal/flyweight"
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
	flyweightService := initFlyweightService(logger)
	
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
	router := setupRoutes(flyweightService, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	logger.Info("Flyweight service started on :8080")

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

func initFlyweightService(logger *zap.Logger) *flyweight.FlyweightService {
	// Create mock implementations
	cache := cache.New(5*time.Minute, 10*time.Minute)
	metrics := &MockMetrics{}
	database := &MockDatabase{}

	// Create flyweight configuration
	config := flyweight.FlyweightConfig{
		Name:        "flyweight-service",
		Version:     "1.0.0",
		Description: "Flyweight pattern implementation service",
		MaxSize:     1000,
		TT:          5 * time.Minute,
		CleanupInterval: 10 * time.Minute,
		Types:       []string{"product", "user", "order", "notification", "configuration"},
		Database: flyweight.DatabaseConfig{
			MySQL: flyweight.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "flyweight_db",
			},
			MongoDB: flyweight.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "flyweight_db",
			},
			Redis: flyweight.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: flyweight.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TT:              5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: flyweight.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"flyweights", "cache", "notifications"},
		},
		WebSocket: flyweight.WebSocketConfig{
			Enabled:          true,
			Port:             8080,
			ReadBufferSize:   1024,
			WriteBufferSize:  1024,
			HandshakeTimeout: 10 * time.Second,
		},
		Security: flyweight.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: flyweight.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: flyweight.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Create flyweight factory
	factory := flyweight.NewFlyweightFactory(cache, &MockLogger{logger: logger}, metrics, config)

	// Create flyweight service
	flyweightService := flyweight.NewFlyweightService(
		factory,
		cache,
		database,
		&MockLogger{logger: logger},
		metrics,
		config,
	)

	return flyweightService
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/flyweight_db?charset=utf8mb4&parseTime=True&loc=Local"
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
	return client.Database("flyweight_db")
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
	flyweightService *flyweight.FlyweightService,
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
		stats := flyweightService.GetFactoryStats()
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"stats":  stats,
			"timestamp": time.Now(),
		})
	})

	// Product endpoints
	productGroup := router.Group("/api/v1/products")
	{
		productGroup.GET("/:id", func(c *gin.Context) {
			productID := c.Param("id")
			product, err := flyweightService.GetProduct(context.Background(), productID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, product)
		})
	}

	// User endpoints
	userGroup := router.Group("/api/v1/users")
	{
		userGroup.GET("/:id", func(c *gin.Context) {
			userID := c.Param("id")
			user, err := flyweightService.GetUser(context.Background(), userID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, user)
		})
	}

	// Order endpoints
	orderGroup := router.Group("/api/v1/orders")
	{
		orderGroup.GET("/:id", func(c *gin.Context) {
			orderID := c.Param("id")
			order, err := flyweightService.GetOrder(context.Background(), orderID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, order)
		})
	}

	// Notification template endpoints
	notificationGroup := router.Group("/api/v1/notifications")
	{
		notificationGroup.GET("/templates/:id", func(c *gin.Context) {
			templateID := c.Param("id")
			template, err := flyweightService.GetNotificationTemplate(context.Background(), templateID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, template)
		})
	}

	// Configuration endpoints
	configGroup := router.Group("/api/v1/configurations")
	{
		configGroup.GET("/:key", func(c *gin.Context) {
			configKey := c.Param("key")
			config, err := flyweightService.GetConfiguration(context.Background(), configKey)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, config)
		})
	}

	// Factory management endpoints
	factoryGroup := router.Group("/api/v1/factory")
	{
		factoryGroup.GET("/stats", func(c *gin.Context) {
			stats := flyweightService.GetFactoryStats()
			c.JSON(http.StatusOK, stats)
		})

		factoryGroup.POST("/cleanup", func(c *gin.Context) {
			flyweightService.ClearUnusedFlyweights()
			c.JSON(http.StatusOK, gin.H{"message": "Cleanup completed"})
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
