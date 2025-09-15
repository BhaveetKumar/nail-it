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

	"proxy/internal/proxy"
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
	proxyManager := initProxyManager(logger)

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
	router := setupRoutes(proxyManager, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	logger.Info("Proxy service started on :8080")

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

func initProxyManager(logger *zap.Logger) *proxy.ProxyManager {
	// Create mock implementations
	cache := cache.New(5*time.Minute, 10*time.Minute)
	metrics := &MockMetrics{}
	circuitBreaker := &MockCircuitBreaker{}
	rateLimiter := &MockRateLimiter{}
	security := &MockSecurity{}
	monitoring := &MockMonitoring{}

	// Create proxy configuration
	config := proxy.ProxyConfig{
		Name: "proxy-service",
		Port: 8080,
		Host: "0.0.0.0",
		Services: []proxy.ServiceConfig{
			{
				Name:        "payment-service",
				URL:         "http://localhost:8081",
				HealthCheck: "/health",
				Timeout:     30 * time.Second,
				RetryCount:  3,
				Weight:      1,
				Enabled:     true,
			},
			{
				Name:        "notification-service",
				URL:         "http://localhost:8082",
				HealthCheck: "/health",
				Timeout:     30 * time.Second,
				RetryCount:  3,
				Weight:      1,
				Enabled:     true,
			},
			{
				Name:        "user-service",
				URL:         "http://localhost:8083",
				HealthCheck: "/health",
				Timeout:     30 * time.Second,
				RetryCount:  3,
				Weight:      1,
				Enabled:     true,
			},
		},
		Cache: proxy.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TTL:             5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		RateLimit: proxy.RateLimitConfig{
			Enabled:           true,
			RequestsPerMinute: 100,
			BurstSize:         20,
			WindowSize:        time.Minute,
		},
		CircuitBreaker: proxy.CircuitBreakerConfig{
			Enabled:          true,
			FailureThreshold: 5,
			SuccessThreshold: 3,
			Timeout:          30 * time.Second,
			MaxRequests:      10,
		},
		Security: proxy.SecurityConfig{
			Enabled:        true,
			RequireAuth:    false,
			AllowedOrigins: []string{"*"},
			AllowedMethods: []string{"GET", "POST", "PUT", "DELETE"},
			AllowedHeaders: []string{"*"},
			MaxRequestSize: 1024 * 1024, // 1MB
			ValidateInput:  true,
			SanitizeInput:  true,
		},
		Monitoring: proxy.MonitoringConfig{
			Enabled:         true,
			MetricsPort:     9090,
			LogLevel:        "info",
			LogFormat:       "json",
			CollectInterval: 30 * time.Second,
		},
		LoadBalancing: proxy.LoadBalancingConfig{
			Enabled:     true,
			Algorithm:   "round_robin",
			HealthCheck: true,
			Interval:    30 * time.Second,
		},
		Retry: proxy.RetryConfig{
			Enabled:       true,
			MaxAttempts:   3,
			InitialDelay:  100 * time.Millisecond,
			MaxDelay:      5 * time.Second,
			BackoffFactor: 2.0,
		},
	}

	// Create proxy manager
	proxyManager := proxy.NewProxyManager(config, &MockLogger{logger: logger}, metrics)

	// Register services
	registerServices(proxyManager, config, cache, metrics, circuitBreaker, rateLimiter, security, monitoring)

	return proxyManager
}

func registerServices(
	proxyManager *proxy.ProxyManager,
	config proxy.ProxyConfig,
	cache proxy.Cache,
	metrics proxy.Metrics,
	circuitBreaker proxy.CircuitBreaker,
	rateLimiter proxy.RateLimiter,
	security proxy.Security,
	monitoring proxy.Monitoring,
) {
	// Register payment service
	paymentService := proxy.NewPaymentService(config.Services[0])
	paymentProxy := proxy.NewServiceProxy(
		paymentService,
		cache,
		&MockLogger{},
		metrics,
		circuitBreaker,
		rateLimiter,
		security,
		monitoring,
		config.Services[0],
	)
	proxyManager.RegisterService(paymentService, paymentProxy)

	// Register notification service
	notificationService := proxy.NewNotificationService(config.Services[1])
	notificationProxy := proxy.NewServiceProxy(
		notificationService,
		cache,
		&MockLogger{},
		metrics,
		circuitBreaker,
		rateLimiter,
		security,
		monitoring,
		config.Services[1],
	)
	proxyManager.RegisterService(notificationService, notificationProxy)

	// Register user service
	userService := proxy.NewUserService(config.Services[2])
	userProxy := proxy.NewServiceProxy(
		userService,
		cache,
		&MockLogger{},
		metrics,
		circuitBreaker,
		rateLimiter,
		security,
		monitoring,
		config.Services[2],
	)
	proxyManager.RegisterService(userService, userProxy)
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/proxy_db?charset=utf8mb4&parseTime=True&loc=Local"
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
	return client.Database("proxy_db")
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
	proxyManager *proxy.ProxyManager,
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
		healthChecks, err := proxyManager.GetAllServicesHealth(context.Background())
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
			"status":    "healthy",
			"services":  healthChecks,
			"timestamp": time.Now(),
		})
	})

	// Service routes
	serviceGroup := router.Group("/api/v1/services")
	{
		serviceGroup.POST("/:service/process", func(c *gin.Context) {
			serviceName := c.Param("service")
			var request interface{}
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			response, err := proxyManager.ProcessRequest(context.Background(), serviceName, request)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, response)
		})

		serviceGroup.GET("/:service/health", func(c *gin.Context) {
			serviceName := c.Param("service")
			health, err := proxyManager.GetServiceHealth(context.Background(), serviceName)
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

		serviceGroup.GET("/:service/metrics", func(c *gin.Context) {
			serviceName := c.Param("service")
			metrics, err := proxyManager.GetServiceMetrics(serviceName)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, metrics)
		})
	}

	// Proxy stats
	router.GET("/api/v1/stats", func(c *gin.Context) {
		stats := proxyManager.GetStats()
		c.JSON(http.StatusOK, stats)
	})

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

func (mm *MockMetrics) IncrementCounter(name string, labels map[string]string)               {}
func (mm *MockMetrics) RecordHistogram(name string, value float64, labels map[string]string) {}
func (mm *MockMetrics) RecordGauge(name string, value float64, labels map[string]string)     {}

type MockCircuitBreaker struct{}

func (mcb *MockCircuitBreaker) Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error) {
	return operation()
}

func (mcb *MockCircuitBreaker) GetState() string {
	return "closed"
}

func (mcb *MockCircuitBreaker) Reset() {}

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

type MockSecurity struct{}

func (ms *MockSecurity) ValidateInput(input interface{}) error {
	return nil
}

func (ms *MockSecurity) SanitizeInput(input interface{}) interface{} {
	return input
}

func (ms *MockSecurity) CheckRateLimit(ctx context.Context, key string) bool {
	return true
}

func (ms *MockSecurity) AuditLog(ctx context.Context, action string, userID string, details map[string]interface{}) {
}

type MockMonitoring struct{}

func (mm *MockMonitoring) RecordRequest(ctx context.Context, service string, duration time.Duration, success bool) {
}
func (mm *MockMonitoring) RecordError(ctx context.Context, service string, err error) {}
func (mm *MockMonitoring) GetServiceMetrics(service string) (*proxy.ServiceMetrics, error) {
	return &proxy.ServiceMetrics{
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
