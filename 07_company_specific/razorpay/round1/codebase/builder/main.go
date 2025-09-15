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

	"github.com/razorpay/round1/codebase/builder/configs"
	"github.com/razorpay/round1/codebase/builder/internal/builder"
	"github.com/razorpay/round1/codebase/builder/internal/handlers"
	"github.com/razorpay/round1/codebase/builder/internal/kafka"
	"github.com/razorpay/round1/codebase/builder/internal/websocket"
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

	// Initialize Builder service
	builderConfig := &builder.ServiceConfig{
		Name:                  config.Builder.Name,
		Version:               config.Builder.Version,
		Description:           config.Builder.Description,
		MaxBuilders:           config.Builder.MaxBuilders,
		MaxProducts:           config.Builder.MaxProducts,
		MaxDirectors:          config.Builder.MaxDirectors,
		CleanupInterval:       config.Builder.CleanupInterval,
		ValidationEnabled:     config.Builder.ValidationEnabled,
		CachingEnabled:        config.Builder.CachingEnabled,
		MonitoringEnabled:     config.Builder.MonitoringEnabled,
		AuditingEnabled:       config.Builder.AuditingEnabled,
		SupportedBuilderTypes: config.Builder.SupportedBuilderTypes,
		SupportedProductTypes: config.Builder.SupportedProductTypes,
		ValidationRules:       config.Builder.ValidationRules,
		Metadata:              config.Builder.Metadata,
	}

	builderService := builder.NewBuilderService(builderConfig)
	productService := builder.NewProductService(builderConfig)
	directorService := builder.NewDirectorService(builderConfig)

	// Initialize handlers
	handlers := handlers.NewHandlers(builderService, productService, directorService, mysqlDB, mongoDB, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	// Builder endpoints
	builderGroup := router.Group("/api/v1/builder")
	{
		// Builder management endpoints
		builderGroup.POST("/builders", handlers.CreateBuilder)
		builderGroup.DELETE("/builders/:type", handlers.DestroyBuilder)
		builderGroup.GET("/builders/:type", handlers.GetBuilder)
		builderGroup.GET("/builders", handlers.ListBuilders)
		builderGroup.GET("/builders/:type/stats", handlers.GetBuilderStats)
		builderGroup.GET("/builders/stats", handlers.GetAllBuilderStats)
		builderGroup.PUT("/builders/:type/active", handlers.SetBuilderActive)

		// Product management endpoints
		builderGroup.POST("/products", handlers.CreateProduct)
		builderGroup.DELETE("/products/:id", handlers.DestroyProduct)
		builderGroup.GET("/products/:id", handlers.GetProduct)
		builderGroup.GET("/products", handlers.ListProducts)
		builderGroup.GET("/products/:id/stats", handlers.GetProductStats)
		builderGroup.GET("/products/stats", handlers.GetAllProductStats)
		builderGroup.PUT("/products/:id/active", handlers.SetProductActive)

		// Director management endpoints
		builderGroup.POST("/directors", handlers.CreateDirector)
		builderGroup.DELETE("/directors/:id", handlers.DestroyDirector)
		builderGroup.GET("/directors/:id", handlers.GetDirector)
		builderGroup.GET("/directors", handlers.ListDirectors)
		builderGroup.GET("/directors/:id/stats", handlers.GetDirectorStats)
		builderGroup.GET("/directors/stats", handlers.GetAllDirectorStats)
		builderGroup.PUT("/directors/:id/active", handlers.SetDirectorActive)

		// Service management endpoints
		builderGroup.GET("/stats", handlers.GetServiceStats)
		builderGroup.POST("/cleanup", handlers.Cleanup)
		builderGroup.GET("/health", handlers.GetHealthStatus)
	}

	// WebSocket endpoint
	router.GET("/ws", handlers.WebSocketHandler)
}
