package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"singleton-service/internal/kafka"
	"singleton-service/internal/logger"
	"singleton-service/internal/models"
	"singleton-service/internal/redis"
	"singleton-service/internal/websocket"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

// Handlers contains all HTTP handlers
type Handlers struct {
	mysqlDB       *sql.DB
	mongoDB       *mongo.Database
	redisClient   *redis.RedisManager
	kafkaProducer *kafka.KafkaProducer
	wsHub         *websocket.Hub
}

// NewHandlers creates a new Handlers instance
func NewHandlers(mysqlDB *sql.DB, mongoDB *mongo.Database, redisClient *redis.RedisManager, kafkaProducer *kafka.KafkaProducer, wsHub *websocket.Hub) *Handlers {
	return &Handlers{
		mysqlDB:       mysqlDB,
		mongoDB:       mongoDB,
		redisClient:   redisClient,
		kafkaProducer: kafkaProducer,
		wsHub:         wsHub,
	}
}

// CreateUser creates a new user
func (h *Handlers) CreateUser(c *gin.Context) {
	var req models.CreateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate user ID
	userID := uuid.New().String()

	// Create user in MySQL
	query := `INSERT INTO users (id, email, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`
	_, err := h.mysqlDB.Exec(query, userID, req.Email, req.Name, time.Now(), time.Now())
	if err != nil {
		logger.GetLogger().Error("Failed to create user in MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
		return
	}

	// Create user in MongoDB
	user := models.User{
		ID:        userID,
		Email:     req.Email,
		Name:      req.Name,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	_, err = h.mongoDB.Collection("users").InsertOne(context.Background(), user)
	if err != nil {
		logger.GetLogger().Error("Failed to create user in MongoDB", "error", err)
		// Continue even if MongoDB fails
	}

	// Cache user data in Redis
	userKey := "user:" + userID
	userData, _ := json.Marshal(user)
	h.redisClient.Set(context.Background(), userKey, userData, 24*time.Hour)

	// Publish event to Kafka
	eventData := map[string]interface{}{
		"user_id": userID,
		"email":   req.Email,
		"name":    req.Name,
	}
	h.kafkaProducer.PublishUserEvent(context.Background(), userID, "user_created", eventData)

	// Send WebSocket notification
	message := &websocket.Message{
		Type:      "user_created",
		Data:      user,
		Timestamp: time.Now().Unix(),
		UserID:    userID,
	}
	h.wsHub.BroadcastMessage(message)

	c.JSON(http.StatusCreated, gin.H{
		"message": "User created successfully",
		"user_id": userID,
		"user":    user,
	})
}

// GetUser retrieves a user by ID
func (h *Handlers) GetUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "User ID is required"})
		return
	}

	// Try to get from Redis cache first
	userKey := "user:" + userID
	cachedUser, err := h.redisClient.Get(context.Background(), userKey)
	if err == nil {
		var user models.User
		if err := json.Unmarshal([]byte(cachedUser), &user); err == nil {
			c.JSON(http.StatusOK, gin.H{"user": user})
			return
		}
	}

	// Get from MySQL
	query := `SELECT id, email, name, created_at, updated_at FROM users WHERE id = ?`
	row := h.mysqlDB.QueryRow(query, userID)

	var user models.User
	err = row.Scan(&user.ID, &user.Email, &user.Name, &user.CreatedAt, &user.UpdatedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
			return
		}
		logger.GetLogger().Error("Failed to get user from MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user"})
		return
	}

	// Cache the result
	userData, _ := json.Marshal(user)
	h.redisClient.Set(context.Background(), userKey, userData, 24*time.Hour)

	c.JSON(http.StatusOK, gin.H{"user": user})
}

// UpdateUser updates a user
func (h *Handlers) UpdateUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "User ID is required"})
		return
	}

	var req models.UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update user in MySQL
	query := `UPDATE users SET name = ?, updated_at = ? WHERE id = ?`
	result, err := h.mysqlDB.Exec(query, req.Name, time.Now(), userID)
	if err != nil {
		logger.GetLogger().Error("Failed to update user in MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update user"})
		return
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil || rowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}

	// Update user in MongoDB
	filter := bson.M{"id": userID}
	update := bson.M{
		"$set": bson.M{
			"name":       req.Name,
			"updated_at": time.Now(),
		},
	}
	h.mongoDB.Collection("users").UpdateOne(context.Background(), filter, update)

	// Invalidate cache
	h.redisClient.Del(context.Background(), "user:"+userID)

	// Publish event to Kafka
	eventData := map[string]interface{}{
		"user_id": userID,
		"name":    req.Name,
	}
	h.kafkaProducer.PublishUserEvent(context.Background(), userID, "user_updated", eventData)

	// Send WebSocket notification
	message := &websocket.Message{
		Type:      "user_updated",
		Data:      eventData,
		Timestamp: time.Now().Unix(),
		UserID:    userID,
	}
	h.wsHub.BroadcastMessage(message)

	c.JSON(http.StatusOK, gin.H{"message": "User updated successfully"})
}

// CreatePayment creates a new payment
func (h *Handlers) CreatePayment(c *gin.Context) {
	var req models.CreatePaymentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate payment ID
	paymentID := uuid.New().String()

	// Create payment in MySQL
	query := `INSERT INTO payments (id, user_id, amount, currency, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)`
	_, err := h.mysqlDB.Exec(query, paymentID, req.UserID, req.Amount, req.Currency, "pending", time.Now(), time.Now())
	if err != nil {
		logger.GetLogger().Error("Failed to create payment in MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create payment"})
		return
	}

	// Create payment in MongoDB
	payment := models.Payment{
		ID:        paymentID,
		UserID:    req.UserID,
		Amount:    req.Amount,
		Currency:  req.Currency,
		Status:    "pending",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	_, err = h.mongoDB.Collection("payments").InsertOne(context.Background(), payment)
	if err != nil {
		logger.GetLogger().Error("Failed to create payment in MongoDB", "error", err)
		// Continue even if MongoDB fails
	}

	// Cache payment data in Redis
	paymentKey := "payment:" + paymentID
	paymentData, _ := json.Marshal(payment)
	h.redisClient.Set(context.Background(), paymentKey, paymentData, 24*time.Hour)

	// Publish event to Kafka
	eventData := map[string]interface{}{
		"payment_id": paymentID,
		"user_id":    req.UserID,
		"amount":     req.Amount,
		"currency":   req.Currency,
		"status":     "pending",
	}
	h.kafkaProducer.PublishPaymentEvent(context.Background(), paymentID, "payment_created", eventData)

	// Send WebSocket notification
	message := &websocket.Message{
		Type:      "payment_created",
		Data:      payment,
		Timestamp: time.Now().Unix(),
		UserID:    req.UserID,
	}
	h.wsHub.SendToUser(req.UserID, message)

	c.JSON(http.StatusCreated, gin.H{
		"message":    "Payment created successfully",
		"payment_id": paymentID,
		"payment":    payment,
	})
}

// GetPayment retrieves a payment by ID
func (h *Handlers) GetPayment(c *gin.Context) {
	paymentID := c.Param("id")
	if paymentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Payment ID is required"})
		return
	}

	// Try to get from Redis cache first
	paymentKey := "payment:" + paymentID
	cachedPayment, err := h.redisClient.Get(context.Background(), paymentKey)
	if err == nil {
		var payment models.Payment
		if err := json.Unmarshal([]byte(cachedPayment), &payment); err == nil {
			c.JSON(http.StatusOK, gin.H{"payment": payment})
			return
		}
	}

	// Get from MySQL
	query := `SELECT id, user_id, amount, currency, status, created_at, updated_at FROM payments WHERE id = ?`
	row := h.mysqlDB.QueryRow(query, paymentID)

	var payment models.Payment
	err = row.Scan(&payment.ID, &payment.UserID, &payment.Amount, &payment.Currency, &payment.Status, &payment.CreatedAt, &payment.UpdatedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Payment not found"})
			return
		}
		logger.GetLogger().Error("Failed to get payment from MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get payment"})
		return
	}

	// Cache the result
	paymentData, _ := json.Marshal(payment)
	h.redisClient.Set(context.Background(), paymentKey, paymentData, 24*time.Hour)

	c.JSON(http.StatusOK, gin.H{"payment": payment})
}

// UpdatePaymentStatus updates a payment status
func (h *Handlers) UpdatePaymentStatus(c *gin.Context) {
	paymentID := c.Param("id")
	if paymentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Payment ID is required"})
		return
	}

	var req models.UpdatePaymentStatusRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update payment status in MySQL
	query := `UPDATE payments SET status = ?, updated_at = ? WHERE id = ?`
	result, err := h.mysqlDB.Exec(query, req.Status, time.Now(), paymentID)
	if err != nil {
		logger.GetLogger().Error("Failed to update payment in MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update payment"})
		return
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil || rowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "Payment not found"})
		return
	}

	// Update payment in MongoDB
	filter := bson.M{"id": paymentID}
	update := bson.M{
		"$set": bson.M{
			"status":     req.Status,
			"updated_at": time.Now(),
		},
	}
	h.mongoDB.Collection("payments").UpdateOne(context.Background(), filter, update)

	// Invalidate cache
	h.redisClient.Del(context.Background(), "payment:"+paymentID)

	// Publish event to Kafka
	eventData := map[string]interface{}{
		"payment_id": paymentID,
		"status":     req.Status,
	}
	h.kafkaProducer.PublishPaymentEvent(context.Background(), paymentID, "payment_updated", eventData)

	// Send WebSocket notification
	message := &websocket.Message{
		Type:      "payment_updated",
		Data:      eventData,
		Timestamp: time.Now().Unix(),
	}
	h.wsHub.BroadcastMessage(message)

	c.JSON(http.StatusOK, gin.H{"message": "Payment status updated successfully"})
}

// GetUserPayments retrieves payments for a user
func (h *Handlers) GetUserPayments(c *gin.Context) {
	userID := c.Param("user_id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "User ID is required"})
		return
	}

	// Parse pagination parameters
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	offset := (page - 1) * limit

	// Get payments from MySQL
	query := `SELECT id, user_id, amount, currency, status, created_at, updated_at FROM payments WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?`
	rows, err := h.mysqlDB.Query(query, userID, limit, offset)
	if err != nil {
		logger.GetLogger().Error("Failed to get user payments from MySQL", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get payments"})
		return
	}
	defer rows.Close()

	var payments []models.Payment
	for rows.Next() {
		var payment models.Payment
		err := rows.Scan(&payment.ID, &payment.UserID, &payment.Amount, &payment.Currency, &payment.Status, &payment.CreatedAt, &payment.UpdatedAt)
		if err != nil {
			logger.GetLogger().Error("Failed to scan payment", "error", err)
			continue
		}
		payments = append(payments, payment)
	}

	c.JSON(http.StatusOK, gin.H{
		"payments": payments,
		"page":     page,
		"limit":    limit,
		"count":    len(payments),
	})
}

// HealthCheck returns the health status of the service
func (h *Handlers) HealthCheck(c *gin.Context) {
	// Check MySQL connection
	if err := h.mysqlDB.Ping(); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "MySQL connection failed",
		})
		return
	}

	// Check MongoDB connection
	if err := h.mongoDB.Client().Ping(context.Background(), nil); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "MongoDB connection failed",
		})
		return
	}

	// Check Redis connection
	if err := h.redisClient.Ping(context.Background()); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "Redis connection failed",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":            "healthy",
		"connected_clients": h.wsHub.GetConnectedClients(),
		"connected_users":   h.wsHub.GetConnectedUsers(),
		"timestamp":         time.Now().Unix(),
	})
}
