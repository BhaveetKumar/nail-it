package factory

import (
	"context"
	"fmt"
	"sync"

	"factory-service/internal/logger"
	"factory-service/internal/models"
)

// PaymentSystemFactory interface defines the contract for payment system factories
type PaymentSystemFactory interface {
	CreatePaymentGateway() PaymentGateway
	CreateNotificationChannel() NotificationChannel
	CreateDatabaseConnection() DatabaseConnection
	GetSystemName() string
}

// NotificationSystemFactory interface defines the contract for notification system factories
type NotificationSystemFactory interface {
	CreateEmailChannel() NotificationChannel
	CreateSMSChannel() NotificationChannel
	CreatePushChannel() NotificationChannel
	GetSystemName() string
}

// DatabaseSystemFactory interface defines the contract for database system factories
type DatabaseSystemFactory interface {
	CreateMySQLConnection() DatabaseConnection
	CreatePostgreSQLConnection() DatabaseConnection
	CreateMongoDBConnection() DatabaseConnection
	GetSystemName() string
}

// AbstractFactory implements the Abstract Factory pattern
type AbstractFactory struct {
	paymentSystemFactory      PaymentSystemFactory
	notificationSystemFactory NotificationSystemFactory
	databaseSystemFactory     DatabaseSystemFactory
	mutex                     sync.RWMutex
}

var (
	abstractFactory *AbstractFactory
	abstractOnce    sync.Once
)

// GetAbstractFactory returns the singleton instance of AbstractFactory
func GetAbstractFactory() *AbstractFactory {
	abstractOnce.Do(func() {
		abstractFactory = &AbstractFactory{}
		abstractFactory.initializeFactories()
	})
	return abstractFactory
}

// initializeFactories initializes all the system factories
func (af *AbstractFactory) initializeFactories() {
	af.mutex.Lock()
	defer af.mutex.Unlock()

	af.paymentSystemFactory = NewPaymentSystemFactory()
	af.notificationSystemFactory = NewNotificationSystemFactory()
	af.databaseSystemFactory = NewDatabaseSystemFactory()
}

// GetPaymentSystemFactory returns the payment system factory
func (af *AbstractFactory) GetPaymentSystemFactory() PaymentSystemFactory {
	af.mutex.RLock()
	defer af.mutex.RUnlock()
	return af.paymentSystemFactory
}

// GetNotificationSystemFactory returns the notification system factory
func (af *AbstractFactory) GetNotificationSystemFactory() NotificationSystemFactory {
	af.mutex.RLock()
	defer af.mutex.RUnlock()
	return af.notificationSystemFactory
}

// GetDatabaseSystemFactory returns the database system factory
func (af *AbstractFactory) GetDatabaseSystemFactory() DatabaseSystemFactory {
	af.mutex.RLock()
	defer af.mutex.RUnlock()
	return af.databaseSystemFactory
}

// PaymentSystemFactoryImpl implements PaymentSystemFactory
type PaymentSystemFactoryImpl struct {
	gatewayFactory *PaymentGatewayFactory
}

// NewPaymentSystemFactory creates a new payment system factory
func NewPaymentSystemFactory() *PaymentSystemFactoryImpl {
	return &PaymentSystemFactoryImpl{
		gatewayFactory: GetPaymentGatewayFactory(),
	}
}

func (psf *PaymentSystemFactoryImpl) CreatePaymentGateway() PaymentGateway {
	// Create a default payment gateway (Stripe)
	gateway, _ := psf.gatewayFactory.CreateGateway("stripe")
	return gateway
}

func (psf *PaymentSystemFactoryImpl) CreateNotificationChannel() NotificationChannel {
	// Create a default notification channel (Email)
	channelFactory := GetNotificationChannelFactory()
	channel, _ := channelFactory.CreateChannel("email")
	return channel
}

func (psf *PaymentSystemFactoryImpl) CreateDatabaseConnection() DatabaseConnection {
	// Create a default database connection (MySQL)
	dbFactory := GetDatabaseFactory()
	connection, _ := dbFactory.CreateDatabase("mysql")
	return connection
}

func (psf *PaymentSystemFactoryImpl) GetSystemName() string {
	return "Payment System"
}

// NotificationSystemFactoryImpl implements NotificationSystemFactory
type NotificationSystemFactoryImpl struct {
	channelFactory *NotificationChannelFactory
}

// NewNotificationSystemFactory creates a new notification system factory
func NewNotificationSystemFactory() *NotificationSystemFactoryImpl {
	return &NotificationSystemFactoryImpl{
		channelFactory: GetNotificationChannelFactory(),
	}
}

func (nsf *NotificationSystemFactoryImpl) CreateEmailChannel() NotificationChannel {
	channel, _ := nsf.channelFactory.CreateChannel("email")
	return channel
}

func (nsf *NotificationSystemFactoryImpl) CreateSMSChannel() NotificationChannel {
	channel, _ := nsf.channelFactory.CreateChannel("sms")
	return channel
}

func (nsf *NotificationSystemFactoryImpl) CreatePushChannel() NotificationChannel {
	channel, _ := nsf.channelFactory.CreateChannel("push")
	return channel
}

func (nsf *NotificationSystemFactoryImpl) GetSystemName() string {
	return "Notification System"
}

// DatabaseSystemFactoryImpl implements DatabaseSystemFactory
type DatabaseSystemFactoryImpl struct {
	databaseFactory *DatabaseFactory
}

// NewDatabaseSystemFactory creates a new database system factory
func NewDatabaseSystemFactory() *DatabaseSystemFactoryImpl {
	return &DatabaseSystemFactoryImpl{
		databaseFactory: GetDatabaseFactory(),
	}
}

func (dsf *DatabaseSystemFactoryImpl) CreateMySQLConnection() DatabaseConnection {
	connection, _ := dsf.databaseFactory.CreateDatabase("mysql")
	return connection
}

func (dsf *DatabaseSystemFactoryImpl) CreatePostgreSQLConnection() DatabaseConnection {
	connection, _ := dsf.databaseFactory.CreateDatabase("postgresql")
	return connection
}

func (dsf *DatabaseSystemFactoryImpl) CreateMongoDBConnection() DatabaseConnection {
	connection, _ := dsf.databaseFactory.CreateDatabase("mongodb")
	return connection
}

func (dsf *DatabaseSystemFactoryImpl) GetSystemName() string {
	return "Database System"
}

// PaymentService uses the Abstract Factory pattern
type PaymentService struct {
	paymentGateway      PaymentGateway
	notificationChannel NotificationChannel
	databaseConnection  DatabaseConnection
}

// NewPaymentService creates a new payment service using the abstract factory
func NewPaymentService() *PaymentService {
	abstractFactory := GetAbstractFactory()
	paymentSystemFactory := abstractFactory.GetPaymentSystemFactory()

	return &PaymentService{
		paymentGateway:      paymentSystemFactory.CreatePaymentGateway(),
		notificationChannel: paymentSystemFactory.CreateNotificationChannel(),
		databaseConnection:  paymentSystemFactory.CreateDatabaseConnection(),
	}
}

// ProcessPayment processes a payment using the abstract factory components
func (ps *PaymentService) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment using abstract factory components")

	// Validate payment
	if err := ps.paymentGateway.ValidatePayment(request); err != nil {
		return nil, fmt.Errorf("payment validation failed: %w", err)
	}

	// Process payment
	response, err := ps.paymentGateway.ProcessPayment(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("payment processing failed: %w", err)
	}

	// Send notification
	notificationRequest := &models.NotificationRequest{
		ID:        request.ID,
		Recipient: request.UserID,
		Subject:   "Payment Processed",
		Message:   fmt.Sprintf("Your payment of %.2f %s has been processed successfully", request.Amount, request.Currency),
		Title:     "Payment Success",
	}

	if err := ps.notificationChannel.ValidateNotification(notificationRequest); err == nil {
		_, err := ps.notificationChannel.SendNotification(ctx, notificationRequest)
		if err != nil {
			log.Error("Failed to send payment notification", "error", err)
		}
	}

	// Store payment in database
	if err := ps.databaseConnection.Connect(ctx); err == nil {
		defer ps.databaseConnection.Disconnect(ctx)

		query := "INSERT INTO payments (id, user_id, amount, currency, status) VALUES (?, ?, ?, ?, ?)"
		_, err := ps.databaseConnection.ExecuteQuery(ctx, query, request.ID, request.UserID, request.Amount, request.Currency, response.Status)
		if err != nil {
			log.Error("Failed to store payment in database", "error", err)
		}
	}

	return response, nil
}

// GetSystemInfo returns information about the system components
func (ps *PaymentService) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"payment_gateway":      ps.paymentGateway.GetGatewayName(),
		"notification_channel": ps.notificationChannel.GetChannelName(),
		"database_type":        ps.databaseConnection.GetDatabaseType(),
	}
}

// NotificationService uses the Abstract Factory pattern
type NotificationService struct {
	emailChannel NotificationChannel
	smsChannel   NotificationChannel
	pushChannel  NotificationChannel
}

// NewNotificationService creates a new notification service using the abstract factory
func NewNotificationService() *NotificationService {
	abstractFactory := GetAbstractFactory()
	notificationSystemFactory := abstractFactory.GetNotificationSystemFactory()

	return &NotificationService{
		emailChannel: notificationSystemFactory.CreateEmailChannel(),
		smsChannel:   notificationSystemFactory.CreateSMSChannel(),
		pushChannel:  notificationSystemFactory.CreatePushChannel(),
	}
}

// SendMultiChannelNotification sends notifications through multiple channels
func (ns *NotificationService) SendMultiChannelNotification(ctx context.Context, request *models.NotificationRequest) ([]*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending multi-channel notification")

	var responses []*models.NotificationResponse

	// Send email notification
	if err := ns.emailChannel.ValidateNotification(request); err == nil {
		response, err := ns.emailChannel.SendNotification(ctx, request)
		if err != nil {
			log.Error("Failed to send email notification", "error", err)
		} else {
			responses = append(responses, response)
		}
	}

	// Send SMS notification
	if err := ns.smsChannel.ValidateNotification(request); err == nil {
		response, err := ns.smsChannel.SendNotification(ctx, request)
		if err != nil {
			log.Error("Failed to send SMS notification", "error", err)
		} else {
			responses = append(responses, response)
		}
	}

	// Send push notification
	if err := ns.pushChannel.ValidateNotification(request); err == nil {
		response, err := ns.pushChannel.SendNotification(ctx, request)
		if err != nil {
			log.Error("Failed to send push notification", "error", err)
		} else {
			responses = append(responses, response)
		}
	}

	return responses, nil
}

// GetSystemInfo returns information about the notification system components
func (ns *NotificationService) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"email_channel": ns.emailChannel.GetChannelName(),
		"sms_channel":   ns.smsChannel.GetChannelName(),
		"push_channel":  ns.pushChannel.GetChannelName(),
	}
}

// DatabaseService uses the Abstract Factory pattern
type DatabaseService struct {
	mysqlConnection      DatabaseConnection
	postgresqlConnection DatabaseConnection
	mongodbConnection    DatabaseConnection
}

// NewDatabaseService creates a new database service using the abstract factory
func NewDatabaseService() *DatabaseService {
	abstractFactory := GetAbstractFactory()
	databaseSystemFactory := abstractFactory.GetDatabaseSystemFactory()

	return &DatabaseService{
		mysqlConnection:      databaseSystemFactory.CreateMySQLConnection(),
		postgresqlConnection: databaseSystemFactory.CreatePostgreSQLConnection(),
		mongodbConnection:    databaseSystemFactory.CreateMongoDBConnection(),
	}
}

// ExecuteQueryOnAllDatabases executes a query on all available databases
func (ds *DatabaseService) ExecuteQueryOnAllDatabases(ctx context.Context, query string, args ...interface{}) (map[string]interface{}, error) {
	log := logger.GetLogger()
	log.Info("Executing query on all databases")

	results := make(map[string]interface{})

	// Execute on MySQL
	if err := ds.mysqlConnection.Connect(ctx); err == nil {
		defer ds.mysqlConnection.Disconnect(ctx)

		result, err := ds.mysqlConnection.ExecuteQuery(ctx, query, args...)
		if err != nil {
			log.Error("Failed to execute query on MySQL", "error", err)
		} else {
			results["mysql"] = result
		}
	}

	// Execute on PostgreSQL
	if err := ds.postgresqlConnection.Connect(ctx); err == nil {
		defer ds.postgresqlConnection.Disconnect(ctx)

		result, err := ds.postgresqlConnection.ExecuteQuery(ctx, query, args...)
		if err != nil {
			log.Error("Failed to execute query on PostgreSQL", "error", err)
		} else {
			results["postgresql"] = result
		}
	}

	// Execute on MongoDB
	if err := ds.mongodbConnection.Connect(ctx); err == nil {
		defer ds.mongodbConnection.Disconnect(ctx)

		result, err := ds.mongodbConnection.ExecuteQuery(ctx, query, args...)
		if err != nil {
			log.Error("Failed to execute query on MongoDB", "error", err)
		} else {
			results["mongodb"] = result
		}
	}

	return results, nil
}

// GetSystemInfo returns information about the database system components
func (ds *DatabaseService) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"mysql_connection":      ds.mysqlConnection.GetDatabaseType(),
		"postgresql_connection": ds.postgresqlConnection.GetDatabaseType(),
		"mongodb_connection":    ds.mongodbConnection.GetDatabaseType(),
	}
}
