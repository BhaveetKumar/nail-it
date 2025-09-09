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

	"decorator/internal/decorator"
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
	decoratorManager := initDecoratorManager(logger)

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
	router := setupRoutes(decoratorManager, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	logger.Info("Decorator service started on :8080")

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

func initDecoratorManager(logger *zap.Logger) *decorator.DecoratorManager {
	// Create mock implementations
	metrics := &MockMetrics{}
	cache := cache.New(5*time.Minute, 10*time.Minute)

	// Create decorator configuration
	config := decorator.DecoratorConfig{
		Logging: decorator.LoggingConfig{
			Enabled:     true,
			Level:       "info",
			Format:      "json",
			Output:      "stdout",
			Fields:      []string{"component", "request", "response", "duration"},
			IncludeData: true,
		},
		Metrics: decorator.MetricsConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
			Labels:          []string{"component", "decorator", "status"},
		},
		Cache: decorator.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TT:              5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
			Compression:     true,
		},
		Security: decorator.SecurityConfig{
			Enabled:          true,
			ValidateInput:    true,
			SanitizeInput:    true,
			CheckPermissions: true,
			AuditLogging:     true,
			AllowedOrigins:   []string{"*"},
			MaxRequestSize:   1024 * 1024, // 1MB
		},
		RateLimit: decorator.RateLimitConfig{
			Enabled:           true,
			RequestsPerMinute: 100,
			BurstSize:         20,
			WindowSize:        time.Minute,
			KeyFunc:           "user_id",
		},
		CircuitBreaker: decorator.CircuitBreakerConfig{
			Enabled:          true,
			FailureThreshold: 5,
			SuccessThreshold: 3,
			Timeout:          30 * time.Second,
			MaxRequests:      10,
		},
		Retry: decorator.RetryConfig{
			Enabled:       true,
			MaxAttempts:   3,
			InitialDelay:  100 * time.Millisecond,
			MaxDelay:      5 * time.Second,
			BackoffFactor: 2.0,
		},
		Monitoring: decorator.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			LogLevel:        "info",
			CollectInterval: 30 * time.Second,
			CustomMetrics:   []string{"request_duration", "cache_hits", "rate_limit_hits"},
		},
		Validation: decorator.ValidationConfig{
			Enabled:    true,
			Rules:      map[string]interface{}{},
			Schemas:    map[string]interface{}{},
			StrictMode: false,
		},
		Encryption: decorator.EncryptionConfig{
			Enabled:    true,
			Algorithm:  "AES-256-GCM",
			KeySize:    32,
			SaltLength: 16,
			Iterations: 10000,
		},
		Compression: decorator.CompressionConfig{
			Enabled:   true,
			Algorithm: "gzip",
			Level:     6,
			MinSize:   1024,
		},
		Serialization: decorator.SerializationConfig{
			Enabled:     true,
			Format:      "json",
			Compression: true,
			Encryption:  false,
		},
		Notification: decorator.NotificationConfig{
			Enabled:    true,
			Channels:   []string{"email", "sms", "push"},
			RetryCount: 3,
			RetryDelay: time.Second,
			BatchSize:  100,
		},
		Analytics: decorator.AnalyticsConfig{
			Enabled:       true,
			Events:        []string{"request", "response", "error"},
			BatchSize:     1000,
			FlushInterval: 30 * time.Second,
		},
		Audit: decorator.AuditConfig{
			Enabled:     true,
			Events:      []string{"create", "update", "delete", "read"},
			Retention:   90 * 24 * time.Hour, // 90 days
			Compression: true,
		},
	}

	// Create decorator manager
	decoratorManager := decorator.NewDecoratorManager(config, &MockLogger{logger: logger}, metrics)

	// Register components
	registerComponents(decoratorManager)

	// Register decorators
	registerDecorators(decoratorManager, config, cache, metrics)

	return decoratorManager
}

func registerComponents(decoratorManager *decorator.DecoratorManager) {
	// Register payment component
	decoratorManager.RegisterComponent(decorator.NewPaymentComponent())

	// Register notification component
	decoratorManager.RegisterComponent(decorator.NewNotificationComponent())

	// Register user component
	decoratorManager.RegisterComponent(decorator.NewUserComponent())

	// Register order component
	decoratorManager.RegisterComponent(decorator.NewOrderComponent())

	// Register inventory component
	decoratorManager.RegisterComponent(decorator.NewInventoryComponent())

	// Register analytics component
	decoratorManager.RegisterComponent(decorator.NewAnalyticsComponent())

	// Register audit component
	decoratorManager.RegisterComponent(decorator.NewAuditComponent())
}

func registerDecorators(
	decoratorManager *decorator.DecoratorManager,
	config decorator.DecoratorConfig,
	cache decorator.Cache,
	metrics decorator.Metrics,
) {
	// Register logging decorator
	decoratorManager.RegisterDecorator(decorator.NewLoggingDecorator(&MockLogger{}, config.Logging))

	// Register metrics decorator
	decoratorManager.RegisterDecorator(decorator.NewMetricsDecorator(metrics, config.Metrics))

	// Register cache decorator
	decoratorManager.RegisterDecorator(decorator.NewCacheDecorator(cache, config.Cache))

	// Register security decorator
	decoratorManager.RegisterDecorator(decorator.NewSecurityDecorator(&MockSecurity{}, config.Security))

	// Register rate limit decorator
	decoratorManager.RegisterDecorator(decorator.NewRateLimitDecorator(&MockRateLimiter{}, config.RateLimit))

	// Register circuit breaker decorator
	decoratorManager.RegisterDecorator(decorator.NewCircuitBreakerDecorator(&MockCircuitBreaker{}, config.CircuitBreaker))

	// Register retry decorator
	decoratorManager.RegisterDecorator(decorator.NewRetryDecorator(&MockRetry{}, config.Retry))

	// Register monitoring decorator
	decoratorManager.RegisterDecorator(decorator.NewMonitoringDecorator(&MockMonitoring{}, config.Monitoring))

	// Register validation decorator
	decoratorManager.RegisterDecorator(decorator.NewValidationDecorator(&MockValidation{}, config.Validation))

	// Register encryption decorator
	decoratorManager.RegisterDecorator(decorator.NewEncryptionDecorator(&MockEncryption{}, config.Encryption))

	// Register compression decorator
	decoratorManager.RegisterDecorator(decorator.NewCompressionDecorator(&MockCompression{}, config.Compression))
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/decorator_db?charset=utf8mb4&parseTime=True&loc=Local"
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
	return client.Database("decorator_db")
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
	decoratorManager *decorator.DecoratorManager,
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
		healthChecks, err := decoratorManager.GetAllComponentsHealth(context.Background())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		allHealthy := true
		for _, health := range healthChecks {
			if !health.Healthy {
				allHealthy = false
				break
			}
		}

		status := http.StatusOK
		if !allHealthy {
			status = http.StatusServiceUnavailable
		}

		c.JSON(status, gin.H{
			"status":     "healthy",
			"components": healthChecks,
			"timestamp":  time.Now(),
		})
	})

	// Component execution
	componentGroup := router.Group("/api/v1/components")
	{
		componentGroup.POST("/:component/execute", func(c *gin.Context) {
			componentName := c.Param("component")
			var request struct {
				Data       interface{} `json:"data"`
				Decorators []string    `json:"decorators"`
			}
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := decoratorManager.ExecuteComponent(
				context.Background(),
				componentName,
				request.Decorators,
				request.Data,
			)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})

		componentGroup.GET("/:component/health", func(c *gin.Context) {
			componentName := c.Param("component")
			health, err := decoratorManager.GetComponentHealth(context.Background(), componentName)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			status := http.StatusOK
			if !health.Healthy {
				status = http.StatusServiceUnavailable
			}

			c.JSON(status, health)
		})

		componentGroup.GET("/:component/metrics", func(c *gin.Context) {
			componentName := c.Param("component")
			metrics, err := decoratorManager.GetComponentMetrics(componentName)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, metrics)
		})
	}

	// Decorator management
	decoratorGroup := router.Group("/api/v1/decorators")
	{
		decoratorGroup.GET("/", func(c *gin.Context) {
			decorators := decoratorManager.ListDecorators()
			c.JSON(http.StatusOK, gin.H{"decorators": decorators})
		})

		decoratorGroup.GET("/:component/chain", func(c *gin.Context) {
			componentName := c.Param("component")
			decorators := c.QueryArray("decorators")

			chain, err := decoratorManager.GetDecoratorChain(componentName, decorators)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"chain": chain})
		})
	}

	// Component management
	managementGroup := router.Group("/api/v1/management")
	{
		managementGroup.GET("/components", func(c *gin.Context) {
			components := decoratorManager.ListComponents()
			c.JSON(http.StatusOK, gin.H{"components": components})
		})

		managementGroup.DELETE("/components/:component", func(c *gin.Context) {
			componentName := c.Param("component")
			err := decoratorManager.RemoveComponent(componentName)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Component removed successfully"})
		})

		managementGroup.DELETE("/decorators/:decorator", func(c *gin.Context) {
			decoratorName := c.Param("decorator")
			err := decoratorManager.RemoveDecorator(decoratorName)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Decorator removed successfully"})
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

type MockSecurity struct{}

func (ms *MockSecurity) ValidateInput(input interface{}) error {
	return nil
}

func (ms *MockSecurity) SanitizeInput(input interface{}) interface{} {
	return input
}

func (ms *MockSecurity) CheckPermission(ctx context.Context, userID string, resource string, action string) bool {
	return true
}

func (ms *MockSecurity) AuditLog(ctx context.Context, action string, userID string, details map[string]interface{}) {
}

type MockRateLimiter struct{}

func (mrl *MockRateLimiter) Allow(key string) bool {
	return true
}

func (mrl *MockRateLimiter) Wait(ctx context.Context, key string) error {
	return nil
}

func (mrl *MockRateLimiter) GetLimit() int {
	return 100
}

func (mrl *MockRateLimiter) GetRemaining(key string) int {
	return 100
}

func (mrl *MockRateLimiter) Reset(key string) {}

type MockCircuitBreaker struct{}

func (mcb *MockCircuitBreaker) Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error) {
	return operation()
}

func (mcb *MockCircuitBreaker) GetState() string {
	return "closed"
}

func (mcb *MockCircuitBreaker) Reset() {}

func (mcb *MockCircuitBreaker) GetStats() decorator.CircuitBreakerStats {
	return decorator.CircuitBreakerStats{
		State:              "closed",
		TotalRequests:      1000,
		SuccessfulRequests: 950,
		FailedRequests:     50,
		FailureRate:        5.0,
	}
}

type MockRetry struct{}

func (mr *MockRetry) Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error) {
	return operation()
}

func (mr *MockRetry) GetMaxAttempts() int {
	return 3
}

func (mr *MockRetry) GetDelay(attempt int) time.Duration {
	return time.Duration(attempt) * 100 * time.Millisecond
}

func (mr *MockRetry) ShouldRetry(attempt int, err error) bool {
	return attempt < 3
}

type MockMonitoring struct{}

func (mm *MockMonitoring) RecordRequest(ctx context.Context, component string, duration time.Duration, success bool) {
}
func (mm *MockMonitoring) RecordError(ctx context.Context, component string, err error)            {}
func (mm *MockMonitoring) RecordCustomMetric(name string, value float64, labels map[string]string) {}
func (mm *MockMonitoring) GetComponentMetrics(component string) (*decorator.ComponentMetrics, error) {
	return &decorator.ComponentMetrics{
		ComponentName:       component,
		TotalRequests:       1000,
		SuccessfulRequests:  950,
		FailedRequests:      50,
		AverageLatency:      150.5,
		MaxLatency:          500.0,
		MinLatency:          50.0,
		SuccessRate:         95.0,
		LastRequest:         time.Now(),
		LastError:           time.Now().Add(-time.Hour),
		CacheHits:           200,
		CacheMisses:         800,
		RateLimitHits:       10,
		CircuitBreakerTrips: 5,
	}, nil
}

type MockValidation struct{}

func (mv *MockValidation) Validate(ctx context.Context, data interface{}) error {
	return nil
}

func (mv *MockValidation) ValidateSchema(ctx context.Context, data interface{}, schema interface{}) error {
	return nil
}

func (mv *MockValidation) GetValidationRules() map[string]interface{} {
	return map[string]interface{}{}
}

type MockEncryption struct{}

func (me *MockEncryption) Encrypt(data []byte) ([]byte, error) {
	return data, nil
}

func (me *MockEncryption) Decrypt(encryptedData []byte) ([]byte, error) {
	return encryptedData, nil
}

func (me *MockEncryption) Hash(data []byte) ([]byte, error) {
	return data, nil
}

func (me *MockEncryption) VerifyHash(data []byte, hash []byte) bool {
	return true
}

type MockCompression struct{}

func (mc *MockCompression) Compress(data []byte) ([]byte, error) {
	return data, nil
}

func (mc *MockCompression) Decompress(compressedData []byte) ([]byte, error) {
	return compressedData, nil
}

func (mc *MockCompression) GetCompressionRatio(originalSize, compressedSize int) float64 {
	return 1.0
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
