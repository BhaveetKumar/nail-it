package config

import (
	"sync"
	"time"

	"github.com/spf13/viper"
)

// Config represents application configuration
type Config struct {
	Server   ServerConfig   `mapstructure:"server"`
	Database DatabaseConfig `mapstructure:"database"`
	Redis    RedisConfig    `mapstructure:"redis"`
	Kafka    KafkaConfig    `mapstructure:"kafka"`
	MongoDB  MongoDBConfig  `mapstructure:"mongodb"`
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
	viper.AddConfigPath("/etc/singleton-service")

	// Set default values
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", "30s")
	viper.SetDefault("server.write_timeout", "30s")
	viper.SetDefault("server.idle_timeout", "60s")

	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 3306)
	viper.SetDefault("database.user", "root")
	viper.SetDefault("database.password", "")
	viper.SetDefault("database.dbname", "singleton_db")
	viper.SetDefault("database.sslmode", "disable")
	viper.SetDefault("database.max_conns", 100)
	viper.SetDefault("database.max_idle", 10)

	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.password", "")
	viper.SetDefault("redis.db", 0)

	viper.SetDefault("kafka.brokers", []string{"localhost:9092"})
	viper.SetDefault("kafka.topic", "singleton-events")
	viper.SetDefault("kafka.group_id", "singleton-service")

	viper.SetDefault("mongodb.uri", "mongodb://localhost:27017")
	viper.SetDefault("mongodb.database", "singleton_db")

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

// ReloadConfig reloads configuration from files and environment
func (cm *ConfigManager) ReloadConfig() {
	cm.loadConfig()
}
