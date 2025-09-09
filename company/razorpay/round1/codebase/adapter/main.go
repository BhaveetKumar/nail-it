package main

import (
	"adapter/internal/adapter"
	"adapter/internal/config"
	"adapter/internal/database"
	"adapter/internal/handlers"
	"adapter/internal/kafka"
	"adapter/internal/logger"
	"adapter/internal/redis"
	"adapter/internal/server"
	"adapter/internal/websocket"
	"context"
	"log"
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

	// Initialize adapter components
	adapterManager := adapter.NewAdapterManager()
	metrics := adapter.NewAdapterMetrics()

	// Initialize adapter factory
	adapterFactory := adapter.NewAdapterFactory(cfg.Adapter)

	// Create and register payment gateway adapters
	stripeGateway, _ := adapterFactory.CreatePaymentGateway("stripe")
	adapterManager.RegisterAdapter("payment_gateway", stripeGateway)

	razorpayGateway, _ := adapterFactory.CreatePaymentGateway("razorpay")
	adapterManager.RegisterAdapter("payment_gateway", razorpayGateway)

	paypalGateway, _ := adapterFactory.CreatePaymentGateway("paypal")
	adapterManager.RegisterAdapter("payment_gateway", paypalGateway)

	bankTransferGateway, _ := adapterFactory.CreatePaymentGateway("bank_transfer")
	adapterManager.RegisterAdapter("payment_gateway", bankTransferGateway)

	// Create and register notification service adapters
	emailService, _ := adapterFactory.CreateNotificationService("email")
	adapterManager.RegisterAdapter("notification_service", emailService)

	smsService, _ := adapterFactory.CreateNotificationService("sms")
	adapterManager.RegisterAdapter("notification_service", smsService)

	pushService, _ := adapterFactory.CreateNotificationService("push")
	adapterManager.RegisterAdapter("notification_service", pushService)

	webhookService, _ := adapterFactory.CreateNotificationService("webhook")
	adapterManager.RegisterAdapter("notification_service", webhookService)

	slackService, _ := adapterFactory.CreateNotificationService("slack")
	adapterManager.RegisterAdapter("notification_service", slackService)

	// Create and register database adapters
	mysqlAdapter, _ := adapterFactory.CreateDatabaseAdapter("mysql")
	adapterManager.RegisterAdapter("database", mysqlAdapter)

	postgresqlAdapter, _ := adapterFactory.CreateDatabaseAdapter("postgresql")
	adapterManager.RegisterAdapter("database", postgresqlAdapter)

	mongodbAdapter, _ := adapterFactory.CreateDatabaseAdapter("mongodb")
	adapterManager.RegisterAdapter("database", mongodbAdapter)

	// Initialize handlers
	handlers := handlers.NewHandlers(
		adapterManager,
		adapterFactory,
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
	logger.Info("Starting Adapter Pattern service", "port", cfg.Server.Port)
	if err := srv.Start(); err != nil {
		logger.Fatal("Failed to start server", "error", err)
	}
}
