package config

import (
	"sync"
	"time"

	"github.com/spf13/viper"
)

// Config represents application configuration
type Config struct {
	Server        ServerConfig        `mapstructure:"server"`
	Database      DatabaseConfig      `mapstructure:"database"`
	Redis         RedisConfig         `mapstructure:"redis"`
	Kafka         KafkaConfig         `mapstructure:"kafka"`
	MongoDB       MongoDBConfig       `mapstructure:"mongodb"`
	Stripe        StripeConfig        `mapstructure:"stripe"`
	PayPal        PayPalConfig        `mapstructure:"paypal"`
	Razorpay      RazorpayConfig      `mapstructure:"razorpay"`
	BankTransfer  BankTransferConfig  `mapstructure:"bank_transfer"`
	DigitalWallet DigitalWalletConfig `mapstructure:"digital_wallet"`
	Email         EmailConfig         `mapstructure:"email"`
	SMS           SMSConfig           `mapstructure:"sms"`
	Push          PushConfig          `mapstructure:"push"`
	WhatsApp      WhatsAppConfig      `mapstructure:"whatsapp"`
	Slack         SlackConfig         `mapstructure:"slack"`
}

type ServerConfig struct {
	Port         int           `mapstructure:"port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
}

type DatabaseConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	User     string `mapstructure:"user"`
	Password string `mapstructure:"password"`
	DBName   string `mapstructure:"dbname"`
	SSLMode  string `mapstructure:"sslmode"`
	MaxConns int    `mapstructure:"max_conns"`
	MaxIdle  int    `mapstructure:"max_idle"`
}

type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
}

type KafkaConfig struct {
	Brokers []string `mapstructure:"brokers"`
	Topic   string   `mapstructure:"topic"`
	GroupID string   `mapstructure:"group_id"`
}

type MongoDBConfig struct {
	URI      string `mapstructure:"uri"`
	Database string `mapstructure:"database"`
}

type StripeConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type PayPalConfig struct {
	ClientID     string `mapstructure:"client_id"`
	ClientSecret string `mapstructure:"client_secret"`
}

type RazorpayConfig struct {
	KeyID     string `mapstructure:"key_id"`
	KeySecret string `mapstructure:"key_secret"`
}

type BankTransferConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type DigitalWalletConfig struct {
	Provider string `mapstructure:"provider"`
	APIKey   string `mapstructure:"api_key"`
}

type EmailConfig struct {
	SMTPHost string `mapstructure:"smtp_host"`
	SMTPPort int    `mapstructure:"smtp_port"`
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
}

type SMSConfig struct {
	APIKey    string `mapstructure:"api_key"`
	APISecret string `mapstructure:"api_secret"`
}

type PushConfig struct {
	ServerKey string `mapstructure:"server_key"`
}

type WhatsAppConfig struct {
	APIKey    string `mapstructure:"api_key"`
	APISecret string `mapstructure:"api_secret"`
}

type SlackConfig struct {
	WebhookURL string `mapstructure:"webhook_url"`
	Channel    string `mapstructure:"channel"`
}

// ConfigManager implements Singleton pattern for configuration
type ConfigManager struct {
	config *Config
	mutex  sync.RWMutex
}

var (
	configManager *ConfigManager
	once          sync.Once
)

// GetConfigManager returns the singleton instance of ConfigManager
func GetConfigManager() *ConfigManager {
	once.Do(func() {
		configManager = &ConfigManager{}
		configManager.loadConfig()
	})
	return configManager
}

// loadConfig loads configuration from environment variables and config files
func (cm *ConfigManager) loadConfig() {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./configs")
	viper.AddConfigPath("/etc/factory-service")

	// Set default values
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", "30s")
	viper.SetDefault("server.write_timeout", "30s")
	viper.SetDefault("server.idle_timeout", "60s")

	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 3306)
	viper.SetDefault("database.user", "root")
	viper.SetDefault("database.password", "")
	viper.SetDefault("database.dbname", "factory_db")
	viper.SetDefault("database.sslmode", "disable")
	viper.SetDefault("database.max_conns", 100)
	viper.SetDefault("database.max_idle", 10)

	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.password", "")
	viper.SetDefault("redis.db", 0)

	viper.SetDefault("kafka.brokers", []string{"localhost:9092"})
	viper.SetDefault("kafka.topic", "factory-events")
	viper.SetDefault("kafka.group_id", "factory-service")

	viper.SetDefault("mongodb.uri", "mongodb://localhost:27017")
	viper.SetDefault("mongodb.database", "factory_db")

	// Payment gateway defaults
	viper.SetDefault("stripe.api_key", "sk_test_...")
	viper.SetDefault("paypal.client_id", "paypal_client_id")
	viper.SetDefault("paypal.client_secret", "paypal_client_secret")
	viper.SetDefault("razorpay.key_id", "rzp_test_...")
	viper.SetDefault("razorpay.key_secret", "razorpay_key_secret")
	viper.SetDefault("bank_transfer.api_key", "bank_api_key")
	viper.SetDefault("digital_wallet.provider", "default")
	viper.SetDefault("digital_wallet.api_key", "wallet_api_key")

	// Notification channel defaults
	viper.SetDefault("email.smtp_host", "smtp.gmail.com")
	viper.SetDefault("email.smtp_port", 587)
	viper.SetDefault("email.username", "user@gmail.com")
	viper.SetDefault("email.password", "password")
	viper.SetDefault("sms.api_key", "sms_api_key")
	viper.SetDefault("sms.api_secret", "sms_api_secret")
	viper.SetDefault("push.server_key", "push_server_key")
	viper.SetDefault("whatsapp.api_key", "whatsapp_api_key")
	viper.SetDefault("whatsapp.api_secret", "whatsapp_api_secret")
	viper.SetDefault("slack.webhook_url", "https://hooks.slack.com/...")
	viper.SetDefault("slack.channel", "#general")

	// Enable reading from environment variables
	viper.AutomaticEnv()

	// Read config file
	if err := viper.ReadInConfig(); err != nil {
		// Config file not found, use defaults
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		panic("Failed to unmarshal config: " + err.Error())
	}

	cm.mutex.Lock()
	cm.config = &config
	cm.mutex.Unlock()
}

// GetConfig returns the current configuration
func (cm *ConfigManager) GetConfig() *Config {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config
}

// GetServerConfig returns server configuration
func (cm *ConfigManager) GetServerConfig() ServerConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Server
}

// GetDatabaseConfig returns database configuration
func (cm *ConfigManager) GetDatabaseConfig() DatabaseConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Database
}

// GetRedisConfig returns Redis configuration
func (cm *ConfigManager) GetRedisConfig() RedisConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Redis
}

// GetKafkaConfig returns Kafka configuration
func (cm *ConfigManager) GetKafkaConfig() KafkaConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Kafka
}

// GetMongoDBConfig returns MongoDB configuration
func (cm *ConfigManager) GetMongoDBConfig() MongoDBConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.MongoDB
}

// GetStripeConfig returns Stripe configuration
func (cm *ConfigManager) GetStripeConfig() StripeConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Stripe
}

// GetPayPalConfig returns PayPal configuration
func (cm *ConfigManager) GetPayPalConfig() PayPalConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.PayPal
}

// GetRazorpayConfig returns Razorpay configuration
func (cm *ConfigManager) GetRazorpayConfig() RazorpayConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Razorpay
}

// GetBankTransferConfig returns Bank Transfer configuration
func (cm *ConfigManager) GetBankTransferConfig() BankTransferConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.BankTransfer
}

// GetDigitalWalletConfig returns Digital Wallet configuration
func (cm *ConfigManager) GetDigitalWalletConfig() DigitalWalletConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.DigitalWallet
}

// GetEmailConfig returns Email configuration
func (cm *ConfigManager) GetEmailConfig() EmailConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Email
}

// GetSMSConfig returns SMS configuration
func (cm *ConfigManager) GetSMSConfig() SMSConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.SMS
}

// GetPushConfig returns Push configuration
func (cm *ConfigManager) GetPushConfig() PushConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Push
}

// GetWhatsAppConfig returns WhatsApp configuration
func (cm *ConfigManager) GetWhatsAppConfig() WhatsAppConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.WhatsApp
}

// GetSlackConfig returns Slack configuration
func (cm *ConfigManager) GetSlackConfig() SlackConfig {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	return cm.config.Slack
}

// ReloadConfig reloads configuration from files and environment
func (cm *ConfigManager) ReloadConfig() {
	cm.loadConfig()
}
