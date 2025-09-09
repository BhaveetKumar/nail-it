package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/segmentio/kafka-go"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"github.com/razorpay/round1/codebase/circuit_breaker/configs"
	"github.com/razorpay/round1/codebase/circuit_breaker/internal/circuit_breaker"
	"github.com/razorpay/round1/codebase/circuit_breaker/internal/handlers"
	"github.com/razorpay/round1/codebase/circuit_breaker/internal/kafka"
	"github.com/razorpay/round1/codebase/circuit_breaker/internal/websocket"
)

func main() {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatal("Failed to initialize logger:", err)
	}
	defer logger.Sync()

	// Load configuration
	config, err := configs.LoadConfig()
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	// Initialize database connections
	mysqlDB, err := initMySQL(config.Database.MySQL)
	if err != nil {
		logger.Fatal("Failed to initialize MySQL", zap.Error(err))
	}

	mongoDB, err := initMongoDB(config.Database.MongoDB)
	if err != nil {
		logger.Fatal("Failed to initialize MongoDB", zap.Error(err))
	}

	// Initialize Kafka
	kafkaProducer, kafkaConsumer, err := initKafka(config.Kafka)
	if err != nil {
		logger.Fatal("Failed to initialize Kafka", zap.Error(err))
	}

	// Initialize WebSocket hub
	wsHub := websocket.NewHub()
	go wsHub.Run()

	// Initialize Circuit Breaker service
	circuitBreakerConfig := &circuit_breaker.ServiceConfig{
		Name:                    config.CircuitBreaker.Name,
		Version:                 config.CircuitBreaker.Version,
		Description:             config.CircuitBreaker.Description,
		MaxCircuitBreakers:      config.CircuitBreaker.MaxCircuitBreakers,
		CleanupInterval:         config.CircuitBreaker.CleanupInterval,
		ValidationEnabled:       config.CircuitBreaker.ValidationEnabled,
		CachingEnabled:          config.CircuitBreaker.CachingEnabled,
		MonitoringEnabled:       config.CircuitBreaker.MonitoringEnabled,
		AuditingEnabled:         config.CircuitBreaker.AuditingEnabled,
		DefaultFailureThreshold: config.CircuitBreaker.DefaultFailureThreshold,
		DefaultSuccessThreshold: config.CircuitBreaker.DefaultSuccessThreshold,
		DefaultTimeout:          config.CircuitBreaker.DefaultTimeout,
		DefaultResetTimeout:     config.CircuitBreaker.DefaultResetTimeout,
		SupportedTypes:          config.CircuitBreaker.SupportedTypes,
		ValidationRules:         config.CircuitBreaker.ValidationRules,
		Metadata:                config.CircuitBreaker.Metadata,
	}

	circuitBreakerService := circuit_breaker.NewCircuitBreakerService(circuitBreakerConfig)

	// Initialize handlers
	handlers := handlers.NewHandlers(circuitBreakerService, mysqlDB, mongoDB, kafkaProducer, kafkaConsumer, wsHub, logger)

	// Initialize Gin router
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Setup routes
	setupRoutes(router, handlers)

	// Start server
	server := &http.Server{
		Addr:    config.Server.Port,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		logger.Info("Starting server", zap.String("port", config.Server.Port))
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Give outstanding requests a deadline for completion
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Attempt graceful shutdown
	if err := server.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Server exited")
}

func initMySQL(config configs.MySQLConfig) (*gorm.DB, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		config.Username, config.Password, config.Host, config.Port, config.Database)

	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, err
	}

	// Configure connection pool
	sqlDB, err := db.DB()
	if err != nil {
		return nil, err
	}

	sqlDB.SetMaxIdleConns(config.MaxIdleConns)
	sqlDB.SetMaxOpenConns(config.MaxOpenConns)
	sqlDB.SetConnMaxLifetime(time.Duration(config.ConnMaxLifetime) * time.Second)

	return db, nil
}

func initMongoDB(config configs.MongoDBConfig) (*mongo.Database, error) {
	clientOptions := options.Client().ApplyURI(config.URI)
	client, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		return nil, err
	}

	// Test the connection
	err = client.Ping(context.Background(), nil)
	if err != nil {
		return nil, err
	}

	return client.Database(config.Database), nil
}

func initKafka(config configs.KafkaConfig) (*kafka.Writer, *kafka.Reader, error) {
	// Initialize Kafka producer
	producer := &kafka.Writer{
		Addr:     kafka.TCP(config.Brokers...),
		Balancer: &kafka.LeastBytes{},
	}

	// Initialize Kafka consumer
	consumer := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  config.Brokers,
		GroupID:  config.GroupID,
		Topic:    config.Topic,
		MinBytes: 10e3, // 10KB
		MaxBytes: 10e6, // 10MB
	})

	return producer, consumer, nil
}

func setupRoutes(router *gin.Engine, handlers *handlers.Handlers) {
	// Health check
	router.GET("/health", handlers.HealthCheck)

	// Circuit Breaker endpoints
	circuitBreakerGroup := router.Group("/api/v1/circuit-breaker")
	{
		// Circuit Breaker management endpoints
		circuitBreakerGroup.POST("/circuit-breakers", handlers.CreateCircuitBreaker)
		circuitBreakerGroup.DELETE("/circuit-breakers/:id", handlers.DestroyCircuitBreaker)
		circuitBreakerGroup.GET("/circuit-breakers/:id", handlers.GetCircuitBreaker)
		circuitBreakerGroup.GET("/circuit-breakers", handlers.ListCircuitBreakers)
		circuitBreakerGroup.GET("/circuit-breakers/:id/stats", handlers.GetCircuitBreakerStats)
		circuitBreakerGroup.GET("/circuit-breakers/stats", handlers.GetAllCircuitBreakerStats)
		circuitBreakerGroup.PUT("/circuit-breakers/:id/active", handlers.SetCircuitBreakerActive)

		// Circuit Breaker execution endpoints
		circuitBreakerGroup.POST("/circuit-breakers/:id/execute", handlers.ExecuteWithCircuitBreaker)
		circuitBreakerGroup.POST("/circuit-breakers/:id/execute-async", handlers.ExecuteWithCircuitBreakerAsync)

		// Service management endpoints
		circuitBreakerGroup.GET("/stats", handlers.GetServiceStats)
		circuitBreakerGroup.POST("/cleanup", handlers.Cleanup)
		circuitBreakerGroup.GET("/health", handlers.GetHealthStatus)
	}

	// WebSocket endpoint
	router.GET("/ws", handlers.WebSocketHandler)
}
