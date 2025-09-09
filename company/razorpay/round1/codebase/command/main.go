package main

import (
	"context"
	"log"
	"command/internal/config"
	"command/internal/database"
	"command/internal/handlers"
	"command/internal/kafka"
	"command/internal/logger"
	"command/internal/redis"
	"command/internal/server"
	"command/internal/command"
	"command/internal/websocket"
	"time"
)

func main() {
	// Initialize configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize logger
	logger, err := logger.NewLogger(cfg.LogLevel)
	if err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logger.Sync()

	// Initialize database connections
	mysqlDB, err := database.NewMySQLConnection(cfg.Database)
	if err != nil {
		logger.Fatal("Failed to connect to MySQL", "error", err)
	}
	defer mysqlDB.Close()

	mongoDB, err := database.NewMongoConnection(cfg.MongoDB)
	if err != nil {
		logger.Fatal("Failed to connect to MongoDB", "error", err)
	}
	defer mongoDB.Close()

	// Initialize Redis
	redisClient, err := redis.NewRedisClient(cfg.Redis)
	if err != nil {
		logger.Fatal("Failed to connect to Redis", "error", err)
	}
	defer redisClient.Close()

	// Initialize Kafka
	kafkaProducer, err := kafka.NewKafkaProducer(cfg.Kafka)
	if err != nil {
		logger.Fatal("Failed to create Kafka producer", "error", err)
	}
	defer kafkaProducer.Close()

	kafkaConsumer, err := kafka.NewKafkaConsumer(cfg.Kafka)
	if err != nil {
		logger.Fatal("Failed to create Kafka consumer", "error", err)
	}
	defer kafkaConsumer.Close()

	// Initialize WebSocket hub
	wsHub := websocket.NewHub()
	go wsHub.Run()

	// Initialize command components
	auditor := command.NewCommandAuditor()
	metrics := command.NewCommandMetrics()
	
	// Initialize command invoker
	invoker := command.NewCommandInvoker(auditor, nil, metrics, cfg.Command)

	// Register command handlers
	paymentHandler := command.NewPaymentCommandHandler()
	invoker.RegisterHandler(paymentHandler)

	userHandler := command.NewUserCommandHandler()
	invoker.RegisterHandler(userHandler)

	orderHandler := command.NewOrderCommandHandler()
	invoker.RegisterHandler(orderHandler)

	notificationHandler := command.NewNotificationCommandHandler()
	invoker.RegisterHandler(notificationHandler)

	inventoryHandler := command.NewInventoryCommandHandler()
	invoker.RegisterHandler(inventoryHandler)

	refundHandler := command.NewRefundCommandHandler()
	invoker.RegisterHandler(refundHandler)

	auditHandler := command.NewAuditCommandHandler()
	invoker.RegisterHandler(auditHandler)

	systemHandler := command.NewSystemCommandHandler()
	invoker.RegisterHandler(systemHandler)

	// Initialize handlers
	handlers := handlers.NewHandlers(
		invoker,
		auditor,
		metrics,
		wsHub,
		kafkaProducer,
		mysqlDB,
		mongoDB,
		redisClient,
		logger,
	)

	// Initialize server
	srv := server.NewServer(cfg.Server, handlers, logger)

	// Start Kafka consumer
	go func() {
		if err := kafkaConsumer.Start(context.Background()); err != nil {
			logger.Error("Kafka consumer error", "error", err)
		}
	}()

	// Start server
	logger.Info("Starting Command Pattern service", "port", cfg.Server.Port)
	if err := srv.Start(); err != nil {
		logger.Fatal("Failed to start server", "error", err)
	}
}
