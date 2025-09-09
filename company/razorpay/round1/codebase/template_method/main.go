package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/gorilla/websocket"
	"github.com/patrickmn/go-cache"
	"go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"template_method/internal/template_method"
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize services
	templateMethodManager := initTemplateMethodManager(logger)

	// Initialize databases
	mysqlDB := initMySQL()
	mongoDB := initMongoDB()
	redisClient := initRedis()

	// Initialize cache
	cacheClient := cache.New(5*time.Minute, 10*time.Minute)

	// Initialize WebSocket hub
	hub := initWebSocketHub()

	// Initialize Kafka producer
	kafkaProducer := initKafkaProducer()

	// Initialize configuration
	config := &template_method.TemplateMethodConfig{
		Name:               "Template Method Service",
		Version:            "1.0.0",
		Description:        "Template Method pattern implementation with microservice architecture",
		MaxTemplateMethods: 1000,
		MaxSteps:           100,
		MaxExecutionTime:   30 * time.Minute,
		MaxRetries:         3,
		RetryDelay:         1 * time.Second,
		RetryBackoff:       2.0,
		ValidationEnabled:  true,
		CachingEnabled:     true,
		MonitoringEnabled:  true,
		AuditingEnabled:    true,
		SchedulingEnabled:  true,
		SupportedTypes:     []string{"document_processing", "data_validation", "workflow", "api_request", "database_operation"},
		DefaultType:        "workflow",
		ValidationRules: map[string]interface{}{
			"max_steps":          100,
			"max_execution_time": 30 * time.Minute,
			"max_retries":        3,
		},
		Metadata: map[string]interface{}{
			"environment": "production",
			"region":      "us-east-1",
		},
		Database: template_method.DatabaseConfig{
			MySQL: template_method.MySQLConfig{
				Host:     "localhost",
				Port:     3306,
				Username: "root",
				Password: "password",
				Database: "template_method_db",
			},
			MongoDB: template_method.MongoDBConfig{
				URI:      "mongodb://localhost:27017",
				Database: "template_method_db",
			},
			Redis: template_method.RedisConfig{
				Host:     "localhost",
				Port:     6379,
				Password: "",
				DB:       0,
			},
		},
		Cache: template_method.CacheConfig{
			Enabled:         true,
			Type:            "memory",
			TTL:             5 * time.Minute,
			MaxSize:         1000,
			CleanupInterval: 10 * time.Minute,
		},
		MessageQueue: template_method.MessageQueueConfig{
			Enabled: true,
			Brokers: []string{"localhost:9092"},
			Topics:  []string{"template-method-events"},
		},
		WebSocket: template_method.WebSocketConfig{
			Enabled:          true,
			Port:             8080,
			ReadBufferSize:   1024,
			WriteBufferSize:  1024,
			HandshakeTimeout: 10 * time.Second,
		},
		Security: template_method.SecurityConfig{
			Enabled:           true,
			JWTSecret:         "your-secret-key",
			TokenExpiry:       24 * time.Hour,
			AllowedOrigins:    []string{"*"},
			RateLimitEnabled:  true,
			RateLimitRequests: 100,
			RateLimitWindow:   time.Minute,
		},
		Monitoring: template_method.MonitoringConfig{
			Enabled:         true,
			Port:            9090,
			Path:            "/metrics",
			CollectInterval: 30 * time.Second,
		},
		Logging: template_method.LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}

	// Initialize template method service
	templateMethodService := template_method.NewTemplateMethodService(config)

	// Initialize router
	router := gin.Default()

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		healthChecks := map[string]interface{}{
			"mysql":     checkMySQLHealth(mysqlDB),
			"mongodb":   checkMongoDBHealth(mongoDB),
			"redis":     checkRedisHealth(redisClient),
			"cache":     checkCacheHealth(cacheClient),
			"websocket": checkWebSocketHealth(hub),
			"kafka":     checkKafkaHealth(kafkaProducer),
		}

		status := http.StatusOK
		for _, check := range healthChecks {
			if !check.(bool) {
				status = http.StatusServiceUnavailable
				break
			}
		}

		c.JSON(status, gin.H{
			"status":     "healthy",
			"components": healthChecks,
			"timestamp":  time.Now(),
		})
	})

	// Template method endpoints
	templateMethodGroup := router.Group("/api/v1/template-methods")
	{
		templateMethodGroup.POST("/", func(c *gin.Context) {
			var req struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Type        string `json:"type"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Create template method based on type
			var template template_method.TemplateMethod
			switch req.Type {
			case "document_processing":
				template = template_method.NewDocumentProcessingTemplateMethod(req.Name, req.Description, "text")
			case "data_validation":
				template = template_method.NewDataValidationTemplateMethod(req.Name, req.Description, "strict")
			case "workflow":
				template = template_method.NewWorkflowTemplateMethod(req.Name, req.Description, "sequential")
			case "api_request":
				template = template_method.NewAPIRequestTemplateMethod(req.Name, req.Description, "rest")
			case "database_operation":
				template = template_method.NewDatabaseOperationTemplateMethod(req.Name, req.Description, "select", "users")
			default:
				template = template_method.NewConcreteTemplateMethod(req.Name, req.Description, req.Type)
			}

			// Create template method
			err := templateMethodService.CreateTemplateMethod(req.Name, template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, gin.H{
				"message":  "Template method created successfully",
				"template": req.Name,
			})
		})

		templateMethodGroup.GET("/:name", func(c *gin.Context) {
			name := c.Param("name")
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			steps := template.GetSteps()
			stepInfo := make([]map[string]interface{}, 0, len(steps))
			for _, step := range steps {
				stepInfo = append(stepInfo, map[string]interface{}{
					"name":         step.GetName(),
					"description":  step.GetDescription(),
					"type":         step.GetType(),
					"status":       step.GetStatus(),
					"dependencies": step.GetDependencies(),
				})
			}

			c.JSON(http.StatusOK, gin.H{
				"name":         template.GetName(),
				"description":  template.GetDescription(),
				"status":       template.GetStatus(),
				"steps":        stepInfo,
				"count":        len(steps),
				"current_step": template.GetCurrentStep(),
				"start_time":   template.GetStartTime(),
				"end_time":     template.GetEndTime(),
				"duration":     template.GetDuration(),
				"completed":    template.IsCompleted(),
				"failed":       template.IsFailed(),
				"running":      template.IsRunning(),
			})
		})

		templateMethodGroup.DELETE("/:name", func(c *gin.Context) {
			name := c.Param("name")
			err := templateMethodService.RemoveTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Template method removed successfully"})
		})

		templateMethodGroup.GET("/", func(c *gin.Context) {
			templates := templateMethodService.ListTemplateMethods()
			c.JSON(http.StatusOK, gin.H{"templates": templates})
		})

		templateMethodGroup.POST("/:name/execute", func(c *gin.Context) {
			name := c.Param("name")

			// Get template method
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Execute template method
			err = templateMethodService.ExecuteTemplateMethod(template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Template method executed successfully"})
		})

		templateMethodGroup.POST("/:name/steps", func(c *gin.Context) {
			name := c.Param("name")
			var req struct {
				StepName     string   `json:"step_name"`
				Description  string   `json:"description"`
				Type         string   `json:"type"`
				Dependencies []string `json:"dependencies"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Get template method
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Create step
			step := template_method.NewConcreteStep(req.StepName, req.Description, req.Type)
			step.SetDependencies(req.Dependencies)

			// Add step to template method
			if concreteTemplate, ok := template.(*template_method.ConcreteTemplateMethod); ok {
				err = concreteTemplate.AddStep(step)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
					return
				}
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid template method type"})
				return
			}

			c.JSON(http.StatusCreated, gin.H{"message": "Step added successfully"})
		})

		templateMethodGroup.GET("/:name/steps/:stepName", func(c *gin.Context) {
			name := c.Param("name")
			stepName := c.Param("stepName")

			// Get template method
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Find step
			var step template_method.Step
			for _, s := range template.GetSteps() {
				if s.GetName() == stepName {
					step = s
					break
				}
			}

			if step == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Step not found"})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"name":         step.GetName(),
				"description":  step.GetDescription(),
				"type":         step.GetType(),
				"status":       step.GetStatus(),
				"dependencies": step.GetDependencies(),
				"start_time":   step.GetStartTime(),
				"end_time":     step.GetEndTime(),
				"duration":     step.GetDuration(),
				"completed":    step.IsCompleted(),
				"failed":       step.IsFailed(),
				"running":      step.IsRunning(),
			})
		})

		templateMethodGroup.POST("/:name/steps/:stepName/execute", func(c *gin.Context) {
			name := c.Param("name")
			stepName := c.Param("stepName")

			// Get template method
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Find step
			var step template_method.Step
			for _, s := range template.GetSteps() {
				if s.GetName() == stepName {
					step = s
					break
				}
			}

			if step == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Step not found"})
				return
			}

			// Execute step
			err = templateMethodService.ExecuteStep(step)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Step executed successfully"})
		})

		templateMethodGroup.GET("/:name/stats", func(c *gin.Context) {
			name := c.Param("name")

			// Get template method
			template, err := templateMethodService.GetTemplateMethod(name)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			stats := map[string]interface{}{
				"name":         template.GetName(),
				"description":  template.GetDescription(),
				"status":       template.GetStatus(),
				"steps":        len(template.GetSteps()),
				"current_step": template.GetCurrentStep(),
				"start_time":   template.GetStartTime(),
				"end_time":     template.GetEndTime(),
				"duration":     template.GetDuration(),
				"completed":    template.IsCompleted(),
				"failed":       template.IsFailed(),
				"running":      template.IsRunning(),
				"error":        template.GetError(),
				"metadata":     template.GetMetadata(),
			}

			c.JSON(http.StatusOK, stats)
		})

		templateMethodGroup.GET("/stats", func(c *gin.Context) {
			stats := templateMethodService.GetExecutionStats()
			c.JSON(http.StatusOK, stats)
		})

		templateMethodGroup.GET("/history", func(c *gin.Context) {
			history := templateMethodService.GetExecutionHistory()
			c.JSON(http.StatusOK, gin.H{"history": history})
		})

		templateMethodGroup.DELETE("/history", func(c *gin.Context) {
			err := templateMethodService.ClearExecutionHistory()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"message": "Execution history cleared successfully"})
		})
	}

	// WebSocket endpoint
	router.GET("/ws", func(c *gin.Context) {
		handleWebSocket(c, hub)
	})

	// Start server
	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Graceful shutdown
	logger.Info("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", zap.Error(err))
	}

	// Cleanup template method service
	templateMethodService.Cleanup()

	logger.Info("Server exited")
}

// Mock implementations for demonstration
type MockTemplateMethodManager struct{}

func (mtmm *MockTemplateMethodManager) CreateTemplateMethod(name string, template template_method.TemplateMethod) error {
	return nil
}

func (mtmm *MockTemplateMethodManager) GetTemplateMethod(name string) (template_method.TemplateMethod, error) {
	return nil, nil
}

func (mtmm *MockTemplateMethodManager) RemoveTemplateMethod(name string) error {
	return nil
}

func (mtmm *MockTemplateMethodManager) ListTemplateMethods() []string {
	return []string{"test-template"}
}

func (mtmm *MockTemplateMethodManager) GetTemplateMethodCount() int {
	return 1
}

func (mtmm *MockTemplateMethodManager) GetTemplateMethodStats() map[string]interface{} {
	return map[string]interface{}{
		"total_templates": 1,
		"templates": map[string]interface{}{
			"test-template": map[string]interface{}{
				"name":         "test-template",
				"description":  "Test template method",
				"status":       "pending",
				"steps":        0,
				"current_step": 0,
			},
		},
	}
}

func (mtmm *MockTemplateMethodManager) Cleanup() error {
	return nil
}

type MockWebSocketHub struct {
	broadcast chan []byte
}

func (mwh *MockWebSocketHub) Run()                              {}
func (mwh *MockWebSocketHub) Register(client *websocket.Conn)   {}
func (mwh *MockWebSocketHub) Unregister(client *websocket.Conn) {}
func (mwh *MockWebSocketHub) Broadcast(message []byte) {
	mwh.broadcast <- message
}

type MockKafkaProducer struct{}

func (mkp *MockKafkaProducer) SendMessage(topic string, message []byte) error {
	return nil
}

func (mkp *MockKafkaProducer) Close() error {
	return nil
}

// Initialize mock services
func initTemplateMethodManager(logger *zap.Logger) *MockTemplateMethodManager {
	return &MockTemplateMethodManager{}
}

func initMySQL() *gorm.DB {
	// Mock MySQL connection
	return nil
}

func initMongoDB() *mongo.Client {
	// Mock MongoDB connection
	return nil
}

func initRedis() *redis.Client {
	// Mock Redis connection
	return nil
}

func initWebSocketHub() *MockWebSocketHub {
	return &MockWebSocketHub{
		broadcast: make(chan []byte),
	}
}

func initKafkaProducer() *MockKafkaProducer {
	return &MockKafkaProducer{}
}

// Health check functions
func checkMySQLHealth(db *gorm.DB) bool {
	// Mock health check
	return true
}

func checkMongoDBHealth(client *mongo.Client) bool {
	// Mock health check
	return true
}

func checkRedisHealth(client *redis.Client) bool {
	// Mock health check
	return true
}

func checkCacheHealth(cache *cache.Cache) bool {
	// Mock health check
	return true
}

func checkWebSocketHealth(hub *MockWebSocketHub) bool {
	// Mock health check
	return true
}

func checkKafkaHealth(producer *MockKafkaProducer) bool {
	// Mock health check
	return true
}

// WebSocket handler
func handleWebSocket(c *gin.Context, hub *MockWebSocketHub) {
	// Mock WebSocket handling
	c.JSON(http.StatusOK, gin.H{"message": "WebSocket endpoint"})
}
