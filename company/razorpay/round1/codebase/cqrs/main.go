package main

import (
	"context"
	"log"
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

	"github.com/razorpay/round1/codebase/cqrs/internal/cqrs"
	"github.com/razorpay/round1/codebase/cqrs/internal/handlers"
	"github.com/razorpay/round1/codebase/cqrs/internal/kafka"
	"github.com/razorpay/round1/codebase/cqrs/internal/websocket"
	"github.com/razorpay/round1/codebase/cqrs/configs"
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

	// Initialize CQRS service
	cqrsConfig := &cqrs.CQRSConfig{
		Name:                    config.CQRS.Name,
		Version:                 config.CQRS.Version,
		Description:             config.CQRS.Description,
		MaxCommands:             config.CQRS.MaxCommands,
		MaxQueries:              config.CQRS.MaxQueries,
		MaxEvents:               config.CQRS.MaxEvents,
		MaxReadModels:           config.CQRS.MaxReadModels,
		CleanupInterval:         config.CQRS.CleanupInterval,
		ValidationEnabled:       config.CQRS.ValidationEnabled,
		CachingEnabled:          config.CQRS.CachingEnabled,
		MonitoringEnabled:       config.CQRS.MonitoringEnabled,
		AuditingEnabled:         config.CQRS.AuditingEnabled,
		SupportedCommandTypes:   config.CQRS.SupportedCommandTypes,
		SupportedQueryTypes:     config.CQRS.SupportedQueryTypes,
		SupportedEventTypes:     config.CQRS.SupportedEventTypes,
		SupportedReadModelTypes: config.CQRS.SupportedReadModelTypes,
		ValidationRules:         config.CQRS.ValidationRules,
		Metadata:                config.CQRS.Metadata,
	}

	cqrsService := cqrs.NewService(cqrsConfig)

	// Initialize handlers
	handlers := handlers.NewHandlers(cqrsService, mysqlDB, mongoDB, kafkaProducer, kafkaConsumer, wsHub, logger)

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

	// CQRS endpoints
	cqrsGroup := router.Group("/api/v1/cqrs")
	{
		// Command endpoints
		cqrsGroup.POST("/commands", handlers.SendCommand)
		cqrsGroup.GET("/commands/:id", handlers.GetCommand)
		cqrsGroup.GET("/commands", handlers.ListCommands)

		// Query endpoints
		cqrsGroup.POST("/queries", handlers.SendQuery)
		cqrsGroup.GET("/queries/:id", handlers.GetQuery)
		cqrsGroup.GET("/queries", handlers.ListQueries)

		// Event endpoints
		cqrsGroup.POST("/events", handlers.PublishEvent)
		cqrsGroup.GET("/events/:id", handlers.GetEvent)
		cqrsGroup.GET("/events", handlers.ListEvents)

		// Read model endpoints
		cqrsGroup.POST("/read-models", handlers.SaveReadModel)
		cqrsGroup.GET("/read-models/:id", handlers.GetReadModel)
		cqrsGroup.GET("/read-models", handlers.ListReadModels)
		cqrsGroup.DELETE("/read-models/:id", handlers.DeleteReadModel)

		// Handler registration endpoints
		cqrsGroup.POST("/handlers/commands", handlers.RegisterCommandHandler)
		cqrsGroup.POST("/handlers/queries", handlers.RegisterQueryHandler)
		cqrsGroup.POST("/handlers/events", handlers.RegisterEventHandler)
		cqrsGroup.DELETE("/handlers/commands/:type", handlers.UnregisterCommandHandler)
		cqrsGroup.DELETE("/handlers/queries/:type", handlers.UnregisterQueryHandler)
		cqrsGroup.DELETE("/handlers/events/:type", handlers.UnregisterEventHandler)

		// Service management endpoints
		cqrsGroup.GET("/stats", handlers.GetServiceStats)
		cqrsGroup.POST("/cleanup", handlers.Cleanup)
		cqrsGroup.GET("/health", handlers.GetHealthStatus)
	}

	// WebSocket endpoint
	router.GET("/ws", handlers.WebSocketHandler)
}
