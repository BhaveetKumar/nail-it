package redis

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"singleton-service/internal/config"
	"singleton-service/internal/logger"
)

// RedisManager implements Singleton pattern for Redis connection
type RedisManager struct {
	client *redis.Client
	mutex  sync.RWMutex
}

var (
	redisManager *RedisManager
	redisOnce    sync.Once
)

// GetRedisManager returns the singleton instance of RedisManager
func GetRedisManager() *RedisManager {
	redisOnce.Do(func() {
		redisManager = &RedisManager{}
		redisManager.connect()
	})
	return redisManager
}

// connect establishes connection to Redis
func (rm *RedisManager) connect() {
	cfg := config.GetConfigManager()
	redisConfig := cfg.GetRedisConfig()
	log := logger.GetLogger()

	rdb := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", redisConfig.Host, redisConfig.Port),
		Password: redisConfig.Password,
		DB:       redisConfig.DB,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatal("Failed to connect to Redis", "error", err)
	}

	rm.mutex.Lock()
	rm.client = rdb
	rm.mutex.Unlock()

	log.Info("Redis connection established successfully")
}

// GetClient returns the Redis client
func (rm *RedisManager) GetClient() *redis.Client {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client
}

// Set sets a key-value pair with expiration
func (rm *RedisManager) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Set(ctx, key, value, expiration).Err()
}

// Get gets a value by key
func (rm *RedisManager) Get(ctx context.Context, key string) (string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Get(ctx, key).Result()
}

// Del deletes a key
func (rm *RedisManager) Del(ctx context.Context, keys ...string) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Del(ctx, keys...).Err()
}

// Exists checks if a key exists
func (rm *RedisManager) Exists(ctx context.Context, keys ...string) (int64, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Exists(ctx, keys...).Result()
}

// Expire sets expiration for a key
func (rm *RedisManager) Expire(ctx context.Context, key string, expiration time.Duration) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Expire(ctx, key, expiration).Err()
}

// HSet sets a field in a hash
func (rm *RedisManager) HSet(ctx context.Context, key string, values ...interface{}) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.HSet(ctx, key, values...).Err()
}

// HGet gets a field from a hash
func (rm *RedisManager) HGet(ctx context.Context, key, field string) (string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.HGet(ctx, key, field).Result()
}

// HGetAll gets all fields from a hash
func (rm *RedisManager) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.HGetAll(ctx, key).Result()
}

// LPush pushes values to the left of a list
func (rm *RedisManager) LPush(ctx context.Context, key string, values ...interface{}) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.LPush(ctx, key, values...).Err()
}

// RPush pushes values to the right of a list
func (rm *RedisManager) RPush(ctx context.Context, key string, values ...interface{}) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.RPush(ctx, key, values...).Err()
}

// LPop pops a value from the left of a list
func (rm *RedisManager) LPop(ctx context.Context, key string) (string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.LPop(ctx, key).Result()
}

// RPop pops a value from the right of a list
func (rm *RedisManager) RPop(ctx context.Context, key string) (string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.RPop(ctx, key).Result()
}

// SAdd adds members to a set
func (rm *RedisManager) SAdd(ctx context.Context, key string, members ...interface{}) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.SAdd(ctx, key, members...).Err()
}

// SMembers gets all members of a set
func (rm *RedisManager) SMembers(ctx context.Context, key string) ([]string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.SMembers(ctx, key).Result()
}

// SIsMember checks if a member exists in a set
func (rm *RedisManager) SIsMember(ctx context.Context, key string, member interface{}) (bool, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.SIsMember(ctx, key, member).Result()
}

// ZAdd adds members to a sorted set
func (rm *RedisManager) ZAdd(ctx context.Context, key string, members ...redis.Z) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.ZAdd(ctx, key, members...).Err()
}

// ZRange gets members from a sorted set by rank
func (rm *RedisManager) ZRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.ZRange(ctx, key, start, stop).Result()
}

// ZRangeByScore gets members from a sorted set by score
func (rm *RedisManager) ZRangeByScore(ctx context.Context, key string, opt *redis.ZRangeBy) ([]string, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.ZRangeByScore(ctx, key, opt).Result()
}

// Publish publishes a message to a channel
func (rm *RedisManager) Publish(ctx context.Context, channel string, message interface{}) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Publish(ctx, channel, message).Err()
}

// Subscribe subscribes to channels
func (rm *RedisManager) Subscribe(ctx context.Context, channels ...string) *redis.PubSub {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	return rm.client.Subscribe(ctx, channels...)
}

// Close closes the Redis connection
func (rm *RedisManager) Close() error {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	if rm.client != nil {
		return rm.client.Close()
	}
	return nil
}

// Ping tests the Redis connection
func (rm *RedisManager) Ping(ctx context.Context) error {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()

	if rm.client == nil {
		return fmt.Errorf("Redis client is nil")
	}

	return rm.client.Ping(ctx).Err()
}

// Health check for Redis
func (rm *RedisManager) HealthCheck(ctx context.Context) error {
	return rm.Ping(ctx)
}
