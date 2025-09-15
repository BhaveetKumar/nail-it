package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"singleton-service/internal/config"
	"singleton-service/internal/database"
	"singleton-service/internal/handlers"
	"singleton-service/internal/kafka"
	"singleton-service/internal/logger"
	"singleton-service/internal/redis"
	"singleton-service/internal/server"
	"singleton-service/internal/websocket"
)

func main() {
	// Initialize configuration
	cfg := config.GetConfigManager()

	// Initialize logger singleton
	logger := logger.GetLogger()
	logger.Info("Starting Singleton Service", "version", "1.0.0")

	// Initialize database connections
	mysqlDB := database.GetMySQLManager()
	mongoDB := database.GetMongoManager()

	// Initialize Redis
	redisClient := redis.GetRedisManager()

	// Initialize Kafka
	kafkaProducer := kafka.GetKafkaProducer()
	kafkaConsumer := kafka.GetKafkaConsumer()

	// Initialize WebSocket hub
	wsHub := websocket.GetWebSocketHub()

	// Initialize handlers
	handlers := handlers.NewHandlers(mysqlDB, mongoDB, redisClient, kafkaProducer, wsHub)

	// Initialize HTTP server
	srv := server.NewServer(cfg, handlers, wsHub)

	// Start Kafka consumer
	go kafkaConsumer.Start(context.Background())

	// Start WebSocket hub
	go wsHub.Run()

	// Start HTTP server
	go func() {
		if err := srv.Start(); err != nil {
			logger.Fatal("Failed to start server", "error", err)
		}
	}()

	logger.Info("Singleton Service started successfully")

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down Singleton Service...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", "error", err)
	}

	// Close database connections
	mysqlDB.Close()
	mongoDB.Close()
	redisClient.Close()
	kafkaProducer.Close()
	kafkaConsumer.Close()

	logger.Info("Singleton Service stopped")
}
