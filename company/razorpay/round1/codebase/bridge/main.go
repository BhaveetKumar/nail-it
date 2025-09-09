package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"bridge/internal/bridge"
)

var (
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
)

func main() {
	// Initialize services
	paymentManager := bridge.NewPaymentManager()
	notificationManager := bridge.NewNotificationManager()
	bridgeService := bridge.NewBridgeService()
	metricsService := bridge.NewMetricsService()

	// Initialize databases
	mysqlDB := initMySQL()
	mongoDB := initMongoDB()

	// Initialize Redis
	redisClient := initRedis()

	// Initialize Kafka
	kafkaProducer := initKafkaProducer()
	kafkaConsumer := initKafkaConsumer()

	// Initialize WebSocket hub
	wsHub := initWebSocketHub()

	// Register payment gateways
	registerPaymentGateways(paymentManager)

	// Register notification channels
	registerNotificationChannels(notificationManager)

	// Setup routes
	router := setupRoutes(paymentManager, notificationManager, bridgeService, metricsService, mysqlDB, mongoDB, redisClient, kafkaProducer, kafkaConsumer, wsHub)

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

	log.Println("Bridge service started on :8080")

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exited")
}

func initMySQL() *gorm.DB {
	dsn := "root:password@tcp(localhost:3306)/bridge_db?charset=utf8mb4&parseTime=True&loc=Local"
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
	return client.Database("bridge_db")
}

func initRedis() *redis.Client {
	client := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})
	return client
}

func initKafkaProducer() *kafka.Producer {
	producer, err := kafka.NewProducer(&kafka.ConfigMap{
		"bootstrap.servers": "localhost:9092",
	})
	if err != nil {
		log.Fatalf("Failed to create Kafka producer: %v", err)
	}
	return producer
}

func initKafkaConsumer() *kafka.Consumer {
	consumer, err := kafka.NewConsumer(&kafka.ConfigMap{
		"bootstrap.servers": "localhost:9092",
		"group.id":          "bridge-service",
		"auto.offset.reset": "earliest",
	})
	if err != nil {
		log.Fatalf("Failed to create Kafka consumer: %v", err)
	}
	return consumer
}

func initWebSocketHub() *WebSocketHub {
	return NewWebSocketHub()
}

func registerPaymentGateways(paymentManager *bridge.PaymentManager) {
	// Register Razorpay
	razorpayConfig := bridge.PaymentGatewayConfig{
		Name:        "razorpay",
		APIKey:      "rzp_test_1234567890",
		SecretKey:   "secret_1234567890",
		BaseURL:     "https://api.razorpay.com",
		Environment: "test",
	}
	paymentManager.RegisterGateway("razorpay", bridge.NewRazorpayPaymentGateway(razorpayConfig))

	// Register Stripe
	stripeConfig := bridge.PaymentGatewayConfig{
		Name:        "stripe",
		APIKey:      "sk_test_1234567890",
		SecretKey:   "secret_1234567890",
		BaseURL:     "https://api.stripe.com",
		Environment: "test",
	}
	paymentManager.RegisterGateway("stripe", bridge.NewStripePaymentGateway(stripeConfig))

	// Register PayUMoney
	payuConfig := bridge.PaymentGatewayConfig{
		Name:        "payumoney",
		APIKey:      "payu_1234567890",
		SecretKey:   "secret_1234567890",
		BaseURL:     "https://test.payu.in",
		Environment: "test",
	}
	paymentManager.RegisterGateway("payumoney", bridge.NewPayUMPaymentGateway(payuConfig))
}

func registerNotificationChannels(notificationManager *bridge.NotificationManager) {
	// Register Email
	emailConfig := bridge.NotificationChannelConfig{
		Name:        "email",
		APIKey:      "email_api_key",
		SecretKey:   "email_secret",
		BaseURL:     "https://api.sendgrid.com",
		Environment: "test",
	}
	notificationManager.RegisterChannel("email", bridge.NewEmailNotificationChannel(emailConfig))

	// Register SMS
	smsConfig := bridge.NotificationChannelConfig{
		Name:        "sms",
		APIKey:      "sms_api_key",
		SecretKey:   "sms_secret",
		BaseURL:     "https://api.twilio.com",
		Environment: "test",
	}
	notificationManager.RegisterChannel("sms", bridge.NewSMSNotificationChannel(smsConfig))

	// Register Push
	pushConfig := bridge.NotificationChannelConfig{
		Name:        "push",
		APIKey:      "push_api_key",
		SecretKey:   "push_secret",
		BaseURL:     "https://fcm.googleapis.com",
		Environment: "test",
	}
	notificationManager.RegisterChannel("push", bridge.NewPushNotificationChannel(pushConfig))

	// Register WhatsApp
	whatsappConfig := bridge.NotificationChannelConfig{
		Name:        "whatsapp",
		APIKey:      "whatsapp_api_key",
		SecretKey:   "whatsapp_secret",
		BaseURL:     "https://api.whatsapp.com",
		Environment: "test",
	}
	notificationManager.RegisterChannel("whatsapp", bridge.NewWhatsAppNotificationChannel(whatsappConfig))
}

func setupRoutes(
	paymentManager *bridge.PaymentManager,
	notificationManager *bridge.NotificationManager,
	bridgeService *bridge.BridgeService,
	metricsService *bridge.MetricsService,
	mysqlDB *gorm.DB,
	mongoDB *mongo.Database,
	redisClient *redis.Client,
	kafkaProducer *kafka.Producer,
	kafkaConsumer *kafka.Consumer,
	wsHub *WebSocketHub,
) *gin.Engine {
	router := gin.Default()

	// Health check
	router.GET("/health", func(c *gin.Context) {
		status := metricsService.GetHealthStatus(context.Background())
		c.JSON(http.StatusOK, status)
	})

	// Payment routes
	paymentGroup := router.Group("/api/v1/payments")
	{
		paymentGroup.POST("/:gateway", func(c *gin.Context) {
			gateway := c.Param("gateway")
			var req bridge.PaymentRequest
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			req.CreatedAt = time.Now()
			response, err := paymentManager.ProcessPayment(context.Background(), gateway, req)
			if err != nil {
				metricsService.RecordPaymentFailure()
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			metricsService.RecordPaymentSuccess(req.Amount)
			c.JSON(http.StatusOK, response)
		})

		paymentGroup.POST("/:gateway/refund", func(c *gin.Context) {
			gateway := c.Param("gateway")
			var req struct {
				TransactionID string  `json:"transaction_id"`
				Amount        float64 `json:"amount"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			err := paymentManager.RefundPayment(context.Background(), gateway, req.TransactionID, req.Amount)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Refund processed successfully"})
		})
	}

	// Notification routes
	notificationGroup := router.Group("/api/v1/notifications")
	{
		notificationGroup.POST("/:channel", func(c *gin.Context) {
			channel := c.Param("channel")
			var req bridge.NotificationRequest
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			req.CreatedAt = time.Now()
			response, err := notificationManager.SendNotification(context.Background(), channel, req)
			if err != nil {
				metricsService.RecordNotificationFailure()
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			metricsService.RecordNotificationSuccess()
			c.JSON(http.StatusOK, response)
		})
	}

	// Bridge service routes
	bridgeGroup := router.Group("/api/v1/bridge")
	{
		bridgeGroup.POST("/payment-with-notification", func(c *gin.Context) {
			var req struct {
				Gateway       string                        `json:"gateway"`
				Channel       string                        `json:"channel"`
				Payment       bridge.PaymentRequest         `json:"payment"`
				Notification  bridge.NotificationRequest    `json:"notification"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			req.Payment.CreatedAt = time.Now()
			req.Notification.CreatedAt = time.Now()

			paymentResp, notificationResp, err := bridgeService.ProcessPaymentWithNotification(
				context.Background(),
				req.Gateway,
				req.Channel,
				req.Payment,
				req.Notification,
			)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"payment":      paymentResp,
				"notification": notificationResp,
			})
		})
	}

	// Metrics routes
	metricsGroup := router.Group("/api/v1/metrics")
	{
		metricsGroup.GET("/payments", func(c *gin.Context) {
			metrics, err := metricsService.GetPaymentMetrics(context.Background())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, metrics)
		})

		metricsGroup.GET("/notifications", func(c *gin.Context) {
			metrics, err := metricsService.GetNotificationMetrics(context.Background())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, metrics)
		})
	}

	// WebSocket route
	router.GET("/ws", func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Printf("WebSocket upgrade failed: %v", err)
			return
		}
		wsHub.Register(conn)
	})

	return router
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
