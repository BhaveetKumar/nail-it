package main

import (
	"context"
	"log"
	"strategy/internal/config"
	"strategy/internal/database"
	"strategy/internal/handlers"
	"strategy/internal/kafka"
	"strategy/internal/logger"
	"strategy/internal/redis"
	"strategy/internal/server"
	"strategy/internal/strategy"
	"strategy/internal/websocket"
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

	// Initialize strategy metrics
	metrics := strategy.NewStrategyMetrics()

	// Initialize strategy managers
	paymentManager := strategy.NewStrategyManager("stripe", "bank_transfer", 5*time.Second, 3, metrics)
	notificationManager := strategy.NewStrategyManager("email", "sms", 3*time.Second, 3, metrics)
	pricingManager := strategy.NewStrategyManager("standard", "discount", 2*time.Second, 3, metrics)
	authManager := strategy.NewStrategyManager("jwt", "basic", 1*time.Second, 3, metrics)
	cachingManager := strategy.NewStrategyManager("redis", "memory", 1*time.Second, 3, metrics)
	loggingManager := strategy.NewStrategyManager("console", "file", 1*time.Second, 3, metrics)
	dataProcessingManager := strategy.NewStrategyManager("json", "xml", 2*time.Second, 3, metrics)

	// Initialize strategy factory
	strategyFactory := strategy.NewStrategyFactory(cfg.Strategy)

	// Register payment strategies
	stripeStrategy, _ := strategyFactory.CreatePaymentStrategy("stripe")
	paymentManager.RegisterStrategy("stripe", stripeStrategy)

	razorpayStrategy, _ := strategyFactory.CreatePaymentStrategy("razorpay")
	paymentManager.RegisterStrategy("razorpay", razorpayStrategy)

	paypalStrategy, _ := strategyFactory.CreatePaymentStrategy("paypal")
	paymentManager.RegisterStrategy("paypal", paypalStrategy)

	bankTransferStrategy, _ := strategyFactory.CreatePaymentStrategy("bank_transfer")
	paymentManager.RegisterStrategy("bank_transfer", bankTransferStrategy)

	// Register notification strategies
	emailStrategy, _ := strategyFactory.CreateNotificationStrategy("email")
	notificationManager.RegisterStrategy("email", emailStrategy)

	smsStrategy, _ := strategyFactory.CreateNotificationStrategy("sms")
	notificationManager.RegisterStrategy("sms", smsStrategy)

	pushStrategy, _ := strategyFactory.CreateNotificationStrategy("push")
	notificationManager.RegisterStrategy("push", pushStrategy)

	webhookStrategy, _ := strategyFactory.CreateNotificationStrategy("webhook")
	notificationManager.RegisterStrategy("webhook", webhookStrategy)

	slackStrategy, _ := strategyFactory.CreateNotificationStrategy("slack")
	notificationManager.RegisterStrategy("slack", slackStrategy)

	// Register pricing strategies
	standardStrategy, _ := strategyFactory.CreatePricingStrategy("standard")
	pricingManager.RegisterStrategy("standard", standardStrategy)

	discountStrategy, _ := strategyFactory.CreatePricingStrategy("discount")
	pricingManager.RegisterStrategy("discount", discountStrategy)

	dynamicStrategy, _ := strategyFactory.CreatePricingStrategy("dynamic")
	pricingManager.RegisterStrategy("dynamic", dynamicStrategy)

	tieredStrategy, _ := strategyFactory.CreatePricingStrategy("tiered")
	pricingManager.RegisterStrategy("tiered", tieredStrategy)

	// Register authentication strategies
	jwtStrategy, _ := strategyFactory.CreateAuthenticationStrategy("jwt")
	authManager.RegisterStrategy("jwt", jwtStrategy)

	oauthStrategy, _ := strategyFactory.CreateAuthenticationStrategy("oauth")
	authManager.RegisterStrategy("oauth", oauthStrategy)

	basicStrategy, _ := strategyFactory.CreateAuthenticationStrategy("basic")
	authManager.RegisterStrategy("basic", basicStrategy)

	apiKeyStrategy, _ := strategyFactory.CreateAuthenticationStrategy("api_key")
	authManager.RegisterStrategy("api_key", apiKeyStrategy)

	// Register caching strategies
	redisStrategy, _ := strategyFactory.CreateCachingStrategy("redis")
	cachingManager.RegisterStrategy("redis", redisStrategy)

	memoryStrategy, _ := strategyFactory.CreateCachingStrategy("memory")
	cachingManager.RegisterStrategy("memory", memoryStrategy)

	databaseStrategy, _ := strategyFactory.CreateCachingStrategy("database")
	cachingManager.RegisterStrategy("database", databaseStrategy)

	hybridStrategy, _ := strategyFactory.CreateCachingStrategy("hybrid")
	cachingManager.RegisterStrategy("hybrid", hybridStrategy)

	// Register logging strategies
	fileStrategy, _ := strategyFactory.CreateLoggingStrategy("file")
	loggingManager.RegisterStrategy("file", fileStrategy)

	consoleStrategy, _ := strategyFactory.CreateLoggingStrategy("console")
	loggingManager.RegisterStrategy("console", consoleStrategy)

	databaseLogStrategy, _ := strategyFactory.CreateLoggingStrategy("database")
	loggingManager.RegisterStrategy("database", databaseLogStrategy)

	remoteStrategy, _ := strategyFactory.CreateLoggingStrategy("remote")
	loggingManager.RegisterStrategy("remote", remoteStrategy)

	// Register data processing strategies
	jsonStrategy, _ := strategyFactory.CreateDataProcessingStrategy("json")
	dataProcessingManager.RegisterStrategy("json", jsonStrategy)

	xmlStrategy, _ := strategyFactory.CreateDataProcessingStrategy("xml")
	dataProcessingManager.RegisterStrategy("xml", xmlStrategy)

	csvStrategy, _ := strategyFactory.CreateDataProcessingStrategy("csv")
	dataProcessingManager.RegisterStrategy("csv", csvStrategy)

	binaryStrategy, _ := strategyFactory.CreateDataProcessingStrategy("binary")
	dataProcessingManager.RegisterStrategy("binary", binaryStrategy)

	// Initialize strategy selector
	strategySelector := strategy.NewStrategySelector(
		paymentManager,
		notificationManager,
		pricingManager,
		authManager,
		cachingManager,
		loggingManager,
		dataProcessingManager,
	)

	// Initialize handlers
	handlers := handlers.NewHandlers(
		paymentManager,
		notificationManager,
		pricingManager,
		authManager,
		cachingManager,
		loggingManager,
		dataProcessingManager,
		strategySelector,
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
	logger.Info("Starting Strategy Pattern service", "port", cfg.Server.Port)
	if err := srv.Start(); err != nil {
		logger.Fatal("Failed to start server", "error", err)
	}
}


continue

continue