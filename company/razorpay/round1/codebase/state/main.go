package main

import (
	"context"
	"log"
	"state/internal/config"
	"state/internal/database"
	"state/internal/handlers"
	"state/internal/kafka"
	"state/internal/logger"
	"state/internal/redis"
	"state/internal/server"
	"state/internal/state"
	"state/internal/websocket"
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

	// Initialize state components
	stateManager := state.NewStateManager()
	metrics := state.NewStateMetrics()
	
	// Initialize state machine
	stateMachine := state.NewStateMachine("payment_state_machine", "pending", []string{"completed", "failed", "cancelled", "refunded"})

	// Add payment states
	pendingState := state.NewPaymentPendingState()
	stateMachine.AddState(pendingState)

	completedState := state.NewPaymentCompletedState()
	stateMachine.AddState(completedState)

	failedState := state.NewPaymentFailedState()
	stateMachine.AddState(failedState)

	cancelledState := state.NewPaymentCancelledState()
	stateMachine.AddState(cancelledState)

	refundedState := state.NewPaymentRefundedState()
	stateMachine.AddState(refundedState)

	// Initialize handlers
	handlers := handlers.NewHandlers(
		stateMachine,
		stateManager,
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
	logger.Info("Starting State Pattern service", "port", cfg.Server.Port)
	if err := srv.Start(); err != nil {
		logger.Fatal("Failed to start server", "error", err)
	}
}
