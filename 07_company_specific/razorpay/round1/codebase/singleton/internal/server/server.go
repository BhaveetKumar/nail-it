package server

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"singleton-service/internal/config"
	"singleton-service/internal/handlers"
	"singleton-service/internal/logger"
	"singleton-service/internal/websocket"
)

// Server represents the HTTP server
type Server struct {
	httpServer *http.Server
	handlers   *handlers.Handlers
	wsHub      *websocket.Hub
}

// NewServer creates a new Server instance
func NewServer(cfg *config.ConfigManager, handlers *handlers.Handlers, wsHub *websocket.Hub) *Server {
	serverConfig := cfg.GetServerConfig()
	
	// Set Gin mode
	gin.SetMode(gin.ReleaseMode)
	
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	
	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	// Setup routes
	setupRoutes(router, handlers, wsHub)
	
	httpServer := &http.Server{
		Addr:         ":" + string(rune(serverConfig.Port)),
		Handler:      router,
		ReadTimeout:  serverConfig.ReadTimeout,
		WriteTimeout: serverConfig.WriteTimeout,
		IdleTimeout:  serverConfig.IdleTimeout,
	}
	
	return &Server{
		httpServer: httpServer,
		handlers:   handlers,
		wsHub:      wsHub,
	}
}

// setupRoutes configures all the routes
func setupRoutes(router *gin.Engine, handlers *handlers.Handlers, wsHub *websocket.Hub) {
	// Health check
	router.GET("/health", handlers.HealthCheck)
	
	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// User routes
		users := v1.Group("/users")
		{
			users.POST("", handlers.CreateUser)
			users.GET("/:id", handlers.GetUser)
			users.PUT("/:id", handlers.UpdateUser)
			users.GET("/:user_id/payments", handlers.GetUserPayments)
		}
		
		// Payment routes
		payments := v1.Group("/payments")
		{
			payments.POST("", handlers.CreatePayment)
			payments.GET("/:id", handlers.GetPayment)
			payments.PUT("/:id/status", handlers.UpdatePaymentStatus)
		}
	}
	
	// WebSocket route
	router.GET("/ws", func(c *gin.Context) {
		handleWebSocket(c, wsHub)
	})
}

// handleWebSocket handles WebSocket connections
func handleWebSocket(c *gin.Context, wsHub *websocket.Hub) {
	// Get user ID and client ID from query parameters
	userID := c.Query("user_id")
	clientID := c.Query("client_id")
	
	if userID == "" || clientID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user_id and client_id are required"})
		return
	}
	
	// Upgrade HTTP connection to WebSocket
	conn, err := websocket.Upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.GetLogger().Error("Failed to upgrade connection to WebSocket", "error", err)
		return
	}
	
	// Register client
	client := wsHub.RegisterClient(conn, userID, clientID)
	
	// Start reading and writing pumps
	go client.WritePump()
	go client.ReadPump()
}

// Start starts the HTTP server
func (s *Server) Start() error {
	log := logger.GetLogger()
	log.Info("Starting HTTP server", "addr", s.httpServer.Addr)
	
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the HTTP server
func (s *Server) Shutdown(ctx context.Context) error {
	log := logger.GetLogger()
	log.Info("Shutting down HTTP server")
	
	return s.httpServer.Shutdown(ctx)
}
