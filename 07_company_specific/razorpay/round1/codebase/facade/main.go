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

	"facade/internal/facade"
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
	facadeService := initFacadeService(logger)
	
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
	router := setupRoutes(facadeService, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	logger.Info("Facade service started on :8080")

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

func initFacadeService(logger *zap.Logger) *facade.ECommerceFacade {
	// Create mock implementations
	paymentService := facade.NewMockPaymentService(&MockLogger{logger: logger})
	notificationService := facade.NewMockNotificationService(&MockLogger{logger: logger})
	userService := facade.NewMockUserService(&MockLogger{logger: logger})
	orderService := facade.NewMockOrderService(&MockLogger{logger: logger})
	inventoryService := facade.NewMockInventoryService(&MockLogger{logger: logger})
	analyticsService := facade.NewMockAnalyticsService(&MockLogger{logger: logger})
	auditService := facade.NewMockAuditService(&MockLogger{logger: logger})
	cacheService := &MockCacheService{}
	databaseService := &MockDatabaseService{}
	messageQueueService := &MockMessageQueueService{}
	websocketService := &MockWebSocketService{}
	securityService := &MockSecurityService{}
	configService := &MockConfigurationService{}
	healthService := &MockHealthService{}
	monitoringService := &MockMonitoringService{}

	// Create facade configuration
	config := facade.FacadeConfig{
		Name:        "ecommerce-facade",
		Version:     "1.0.0",
		Description: "E-commerce facade service",
		Services: map[string]facade.ServiceConfig{
			"payment": {
				Name:       "payment-service",
				Enabled:    true,
				URL:        "http://localhost:8081",
				Timeout:    30 * time.Second,
				RetryCount: 3,
				Headers:    map[string]string{"Content-Type": "application/json"},
			},
			"notification": {
				Name:       "notification-service",
				Enabled:    true,
				URL:        "http://localhost:8082",
				Timeout:    30 * time.Second,
				RetryCount: 3,
				Headers:    map[string]string{"Content-Type": "application/json"},
			},
			"user": {
				Name:       "user-service",
				Enabled:    true,
				URL:        "http://localhost:8083",
				Timeout:    30 * time.Second,
				RetryCount: 3,
				Headers:    map[string]string{"Content-Type": "application/json"},
			},
			"order": {
				Name:       "order-service",
				Enabled:    true,
				URL:        "http://localhost:8084",
				Timeout:    30 * time.Second,
				RetryCount: 3,
				Headers:    map[string]string{"Content-Type": "application/json"},
			},
			"inventory": {
				Name:       "inventory-service",
				Enabled:    true,
				URL:        "http://localhost:8085",
				Timeout:    30 * time.Second,
				RetryCount: 3,
				Headers:    map[string]string{"Content-Type": "application/json"},
			},
		},
		Database: facade.DatabaseConfig{
			MySQL: facade.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "facade_db",
			},
			MongoDB: facade.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "facade_db",
			},
			Redis: facade.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: facade.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TT:              5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: facade.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"orders", "payments", "notifications"},
		},
		WebSocket: facade.WebSocketConfig{
			Enabled:          true,
			Port:             8080,
			ReadBufferSize:   1024,
			WriteBufferSize:  1024,
			HandshakeTimeout: 10 * time.Second,
		},
		Security: facade.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: facade.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: facade.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Create facade service
	facadeService := facade.NewECommerceFacade(
		paymentService,
		notificationService,
		userService,
		orderService,
		inventoryService,
		analyticsService,
		auditService,
		cacheService,
		databaseService,
		messageQueueService,
		websocketService,
		securityService,
		configService,
		healthService,
		monitoringService,
		&MockLogger{logger: logger},
		&MockMetrics{},
		config,
	)

	return facadeService
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/facade_db?charset=utf8mb4&parseTime=True&loc=Local"
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
	return client.Database("facade_db")
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
	facadeService *facade.ECommerceFacade,
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
		health, err := facadeService.GetSystemHealth(context.Background())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		status := http.StatusOK
		if health.Overall.Status != "healthy" {
			status = http.StatusServiceUnavailable
		}

		c.JSON(status, health)
	})

	// Order processing
	orderGroup := router.Group("/api/v1/orders")
	{
		orderGroup.POST("/process", func(c *gin.Context) {
			var request facade.ProcessOrderRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := facadeService.ProcessOrder(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// User dashboard
	userGroup := router.Group("/api/v1/users")
	{
		userGroup.GET("/:user_id/dashboard", func(c *gin.Context) {
			userID := c.Param("user_id")
			dashboard, err := facadeService.GetUserDashboard(context.Background(), userID)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, dashboard)
		})
	}

	// Notifications
	notificationGroup := router.Group("/api/v1/notifications")
	{
		notificationGroup.POST("/send", func(c *gin.Context) {
			var request facade.NotificationRequest
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := facadeService.SendNotification(context.Background(), request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})
	}

	// System health
	systemGroup := router.Group("/api/v1/system")
	{
		systemGroup.GET("/health", func(c *gin.Context) {
			health, err := facadeService.GetSystemHealth(context.Background())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, health)
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

type MockCacheService struct{}

func (mcs *MockCacheService) Get(ctx context.Context, key string) (interface{}, error) {
	return nil, nil
}

func (mcs *MockCacheService) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	return nil
}

func (mcs *MockCacheService) Delete(ctx context.Context, key string) error {
	return nil
}

func (mcs *MockCacheService) Clear(ctx context.Context) error {
	return nil
}

func (mcs *MockCacheService) GetStats() facade.CacheStats {
	return facade.CacheStats{}
}

type MockDatabaseService struct{}

func (mds *MockDatabaseService) Save(ctx context.Context, collection string, data interface{}) error {
	return nil
}

func (mds *MockDatabaseService) Find(ctx context.Context, collection string, query interface{}) (interface{}, error) {
	return nil, nil
}

func (mds *MockDatabaseService) Update(ctx context.Context, collection string, id string, data interface{}) error {
	return nil
}

func (mds *MockDatabaseService) Delete(ctx context.Context, collection string, id string) error {
	return nil
}

func (mds *MockDatabaseService) Transaction(ctx context.Context, fn func(ctx context.Context) error) error {
	return fn(ctx)
}

type MockMessageQueueService struct{}

func (mmqs *MockMessageQueueService) Publish(ctx context.Context, topic string, message interface{}) error {
	return nil
}

func (mmqs *MockMessageQueueService) Subscribe(ctx context.Context, topic string, handler func(interface{}) error) error {
	return nil
}

func (mmqs *MockMessageQueueService) Close() error {
	return nil
}

func (mmqs *MockMessageQueueService) GetStats() facade.MessageQueueStats {
	return facade.MessageQueueStats{}
}

type MockWebSocketService struct{}

func (mws *MockWebSocketService) Send(ctx context.Context, clientID string, message interface{}) error {
	return nil
}

func (mws *MockWebSocketService) Broadcast(ctx context.Context, message interface{}) error {
	return nil
}

func (mws *MockWebSocketService) Register(ctx context.Context, clientID string, conn interface{}) error {
	return nil
}

func (mws *MockWebSocketService) Unregister(ctx context.Context, clientID string) error {
	return nil
}

func (mws *MockWebSocketService) GetStats() facade.WebSocketStats {
	return facade.WebSocketStats{}
}

type MockSecurityService struct{}

func (mss *MockSecurityService) ValidateToken(ctx context.Context, token string) (*facade.TokenClaims, error) {
	return &facade.TokenClaims{
		UserID:    "user_123",
		Username:  "testuser",
		Email:     "test@example.com",
		Roles:     []string{"user"},
		ExpiresAt: time.Now().Add(24 * time.Hour),
	}, nil
}

func (mss *MockSecurityService) GenerateToken(ctx context.Context, userID string) (string, error) {
	return "mock_token", nil
}

func (mss *MockSecurityService) CheckPermission(ctx context.Context, userID string, resource string, action string) bool {
	return true
}

func (mss *MockSecurityService) EncryptData(ctx context.Context, data []byte) ([]byte, error) {
	return data, nil
}

func (mss *MockSecurityService) DecryptData(ctx context.Context, encryptedData []byte) ([]byte, error) {
	return encryptedData, nil
}

type MockConfigurationService struct{}

func (mcs *MockConfigurationService) GetConfig(ctx context.Context, key string) (interface{}, error) {
	return "mock_value", nil
}

func (mcs *MockConfigurationService) SetConfig(ctx context.Context, key string, value interface{}) error {
	return nil
}

func (mcs *MockConfigurationService) GetAllConfigs(ctx context.Context) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

func (mcs *MockConfigurationService) ReloadConfig(ctx context.Context) error {
	return nil
}

type MockHealthService struct{}

func (mhs *MockHealthService) CheckHealth(ctx context.Context) (*facade.HealthStatus, error) {
	return &facade.HealthStatus{
		Status:    "healthy",
		Timestamp: time.Now(),
		Services:  []facade.ServiceHealth{},
	}, nil
}

func (mhs *MockHealthService) CheckServiceHealth(ctx context.Context, serviceName string) (*facade.ServiceHealth, error) {
	return &facade.ServiceHealth{
		Name:      serviceName,
		Status:    "healthy",
		Healthy:   true,
		Message:   "Service is healthy",
		Latency:   10 * time.Millisecond,
		Timestamp: time.Now(),
	}, nil
}

func (mhs *MockHealthService) GetAllServicesHealth(ctx context.Context) ([]*facade.ServiceHealth, error) {
	return []*facade.ServiceHealth{}, nil
}

type MockMonitoringService struct{}

func (mms *MockMonitoringService) RecordRequest(ctx context.Context, service string, duration time.Duration, success bool) {}
func (mms *MockMonitoringService) RecordError(ctx context.Context, service string, err error)                        {}
func (mms *MockMonitoringService) GetServiceMetrics(ctx context.Context, service string) (*facade.ServiceMetrics, error) {
	return &facade.ServiceMetrics{
		ServiceName:        service,
		TotalRequests:      1000,
		SuccessfulRequests: 950,
		FailedRequests:     50,
		AverageLatency:     150.5,
		MaxLatency:         500.0,
		MinLatency:         50.0,
		SuccessRate:        95.0,
		LastRequest:        time.Now(),
		LastError:          time.Now().Add(-time.Hour),
	}, nil
}

func (mms *MockMonitoringService) GetSystemMetrics(ctx context.Context) (*facade.SystemMetrics, error) {
	return &facade.SystemMetrics{
		CPUUsage:    25.5,
		MemoryUsage: 60.2,
		DiskUsage:   45.8,
		NetworkIO:   100.0,
		Timestamp:   time.Now(),
	}, nil
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
