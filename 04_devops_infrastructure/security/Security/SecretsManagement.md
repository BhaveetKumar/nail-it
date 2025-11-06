---
# Auto-generated front matter
Title: Secretsmanagement
LastUpdated: 2025-11-06T20:45:59.137366
Tags: []
Status: draft
---

# 🔐 Secrets Management: Secure Storage and Access Control

> **Master secrets management for secure storage, rotation, and access control of sensitive data**

## 📚 Concept

**Detailed Explanation:**
Secrets management is a critical security practice that involves the secure storage, access control, rotation, and lifecycle management of sensitive information such as passwords, API keys, certificates, tokens, and other credentials. It provides a centralized, secure way to manage sensitive data while ensuring that only authorized applications and users can access it when needed.

**Core Philosophy:**

- **Security First**: Protect sensitive data from unauthorized access and exposure
- **Centralized Management**: Single source of truth for all secrets and credentials
- **Least Privilege**: Grant minimum necessary access to secrets
- **Automated Lifecycle**: Automate secret rotation, expiration, and renewal
- **Audit and Compliance**: Maintain comprehensive audit trails and meet regulatory requirements
- **Integration**: Seamless integration with applications and infrastructure

**Why Secrets Management Matters:**

- **Security**: Prevent credential theft, data breaches, and unauthorized access
- **Compliance**: Meet regulatory requirements for data protection and audit trails
- **Operational Efficiency**: Automate secret lifecycle management and reduce manual errors
- **Risk Mitigation**: Reduce the risk of credential exposure and compromise
- **Scalability**: Manage secrets across large-scale, distributed applications
- **Cost Reduction**: Prevent security incidents that can be expensive to resolve
- **Business Continuity**: Ensure applications can access required credentials reliably
- **Developer Experience**: Simplify secret management for development teams

**Key Features:**

**1. Secure Storage:**

- **Encryption at Rest**: All secrets encrypted using strong encryption algorithms
- **Encryption in Transit**: Secure communication channels for secret access
- **Key Management**: Proper key management and rotation for encryption keys
- **Secure Backend**: Use secure storage backends (Consul, etcd, S3, etc.)
- **Benefits**: Protection against data breaches, compliance with security standards
- **Use Cases**: Database credentials, API keys, certificates, tokens

**2. Access Control:**

- **Role-Based Access Control (RBAC)**: Grant access based on user roles and responsibilities
- **Policy-Based Permissions**: Define fine-grained access policies for secrets
- **Authentication**: Strong authentication mechanisms for secret access
- **Authorization**: Verify permissions before granting access to secrets
- **Benefits**: Prevent unauthorized access, implement least privilege principle
- **Use Cases**: Multi-tenant applications, team-based access, service-to-service authentication

**3. Rotation:**

- **Automatic Rotation**: Automatically rotate secrets at regular intervals
- **Zero-Downtime Rotation**: Rotate secrets without service interruption
- **Rollback Capability**: Rollback to previous secret versions if needed
- **Rotation Policies**: Define rotation schedules and policies for different secret types
- **Benefits**: Reduce exposure window, maintain security compliance
- **Use Cases**: Database passwords, API keys, SSL certificates, service tokens

**4. Audit Logging:**

- **Access Logging**: Log all secret access attempts and operations
- **Change Tracking**: Track all changes to secrets and access policies
- **Compliance Reporting**: Generate reports for regulatory compliance
- **Real-time Monitoring**: Monitor secret access patterns and detect anomalies
- **Benefits**: Security monitoring, compliance reporting, forensic analysis
- **Use Cases**: Security audits, compliance requirements, incident investigation

**5. Integration:**

- **Application Integration**: Easy integration with applications and services
- **API Access**: RESTful APIs for programmatic secret access
- **SDK Support**: Software development kits for various programming languages
- **Service Mesh Integration**: Integration with service mesh for automatic secret injection
- **Benefits**: Simplified implementation, reduced development overhead
- **Use Cases**: Microservices, cloud-native applications, legacy system integration

**6. Compliance:**

- **Regulatory Compliance**: Meet requirements for GDPR, HIPAA, SOX, PCI DSS
- **Data Retention**: Implement proper data retention and deletion policies
- **Audit Trails**: Maintain comprehensive audit trails for compliance
- **Reporting**: Generate compliance reports and documentation
- **Benefits**: Meet regulatory requirements, avoid compliance violations
- **Use Cases**: Financial services, healthcare, government, e-commerce

**Advanced Secrets Management Concepts:**

- **Secret Versioning**: Maintain multiple versions of secrets for rollback
- **Secret Sharing**: Secure sharing of secrets between teams and services
- **Dynamic Secrets**: Generate secrets on-demand with automatic expiration
- **Secret Injection**: Automatically inject secrets into applications and containers
- **Cross-Platform Support**: Support for multiple cloud providers and platforms
- **High Availability**: Ensure secrets are available when needed
- **Disaster Recovery**: Backup and recovery procedures for secrets

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive secrets management strategy for a large-scale, multi-cloud enterprise with strict compliance requirements?**

**Answer:** Comprehensive enterprise secrets management strategy:

- **Centralized Architecture**: Implement a centralized secrets management platform (Vault, AWS Secrets Manager, Azure Key Vault)
- **Multi-Cloud Support**: Use cloud-native secrets services with cross-cloud synchronization
- **Access Control**: Implement fine-grained RBAC with policy-based permissions
- **Automated Rotation**: Set up automated secret rotation with zero-downtime deployment
- **Audit and Compliance**: Implement comprehensive audit logging and compliance reporting
- **Integration**: Integrate with CI/CD pipelines, service mesh, and application frameworks
- **Monitoring**: Deploy monitoring and alerting for secret access patterns and anomalies
- **Disaster Recovery**: Implement backup and recovery procedures for secrets
- **Training**: Provide training for teams on secrets management best practices
- **Governance**: Establish governance processes for secret lifecycle management
- **Testing**: Implement testing procedures for secret rotation and access
- **Documentation**: Maintain comprehensive documentation and runbooks

**Q2: What are the key considerations when implementing secrets management for a microservices architecture with service-to-service communication?**

**Answer:** Microservices secrets management implementation:

- **Service Identity**: Implement service identity and authentication for each microservice
- **Secret Injection**: Use service mesh or sidecar patterns for automatic secret injection
- **Dynamic Secrets**: Generate short-lived secrets for service-to-service communication
- **Secret Sharing**: Implement secure secret sharing between related services
- **Access Control**: Use fine-grained access control for service-specific secrets
- **Monitoring**: Monitor secret access patterns across all microservices
- **Rotation**: Implement coordinated secret rotation across dependent services
- **Compliance**: Ensure all services meet compliance requirements for secret handling
- **Documentation**: Maintain service-specific documentation for secret usage
- **Testing**: Implement testing procedures for secret access and rotation
- **Incident Response**: Have clear procedures for secret-related incidents
- **Performance**: Optimize secret access for high-performance microservices

**Q3: How do you optimize secrets management for performance, security, and cost in production environments?**

**Answer:** Production secrets management optimization:

- **Caching Strategy**: Implement secure caching for frequently accessed secrets
- **Connection Pooling**: Use connection pooling for secret management services
- **Performance Monitoring**: Monitor secret access performance and optimize bottlenecks
- **Cost Optimization**: Use appropriate secret management services based on usage patterns
- **Security Hardening**: Implement additional security measures for production environments
- **High Availability**: Ensure secrets management services are highly available
- **Load Balancing**: Implement load balancing for secret management services
- **Error Handling**: Implement robust error handling and retry mechanisms
- **Performance Testing**: Conduct performance testing of secret access patterns
- **Cost Monitoring**: Monitor costs and optimize resource usage
- **Documentation**: Maintain performance baselines and optimization procedures
- **Regular Reviews**: Conduct regular reviews of performance and cost optimization

## 🏗️ Secrets Management Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Secrets Management Stack                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Application │  │   Service   │  │   User      │     │
│  │     A       │  │     B       │  │   Access    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Secrets Manager                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Vault     │  │   AWS KMS   │  │   Azure     │ │ │
│  │  │   (HashiCorp)│  │   Secrets   │  │   Key Vault│ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Storage Backend                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Consul    │  │   etcd      │  │   S3        │ │ │
│  │  │   Storage   │  │   Storage   │  │   Storage   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Audit     │  │   Rotation  │  │   Backup    │     │
│  │   Logs      │  │   Engine    │  │   System    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### HashiCorp Vault Configuration

```hcl
# vault.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
  service = "vault"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true

disable_mlock = true
```

### Vault Policies

```hcl
# policies/app-policy.hcl
path "secret/data/my-app/*" {
  capabilities = ["read"]
}

path "secret/data/my-app/database" {
  capabilities = ["read", "update"]
}

path "secret/data/my-app/api-keys" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
```

### Vault Authentication

```go
// vault-client.go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/hashicorp/vault/api"
    "github.com/hashicorp/vault/api/auth/approle"
)

type VaultClient struct {
    client *api.Client
    token  string
}

func NewVaultClient(vaultAddr, roleID, secretID string) (*VaultClient, error) {
    config := api.DefaultConfig()
    config.Address = vaultAddr

    client, err := api.NewClient(config)
    if err != nil {
        return nil, fmt.Errorf("failed to create vault client: %w", err)
    }

    // Authenticate using AppRole
    authMethod, err := approle.NewAppRoleAuth(roleID, &approle.SecretID{
        FromString: secretID,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create auth method: %w", err)
    }

    authInfo, err := client.Auth().Login(context.Background(), authMethod)
    if err != nil {
        return nil, fmt.Errorf("failed to authenticate: %w", err)
    }

    client.SetToken(authInfo.Auth.ClientToken)

    return &VaultClient{
        client: client,
        token:  authInfo.Auth.ClientToken,
    }, nil
}

func (vc *VaultClient) GetSecret(path string) (map[string]interface{}, error) {
    secret, err := vc.client.Logical().Read(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read secret: %w", err)
    }

    if secret == nil || secret.Data == nil {
        return nil, fmt.Errorf("secret not found at path: %s", path)
    }

    return secret.Data, nil
}

func (vc *VaultClient) PutSecret(path string, data map[string]interface{}) error {
    secretData := map[string]interface{}{
        "data": data,
    }

    _, err := vc.client.Logical().Write(path, secretData)
    if err != nil {
        return fmt.Errorf("failed to write secret: %w", err)
    }

    return nil
}

func (vc *VaultClient) RenewToken() error {
    secret, err := vc.client.Auth().Token().RenewSelf(0)
    if err != nil {
        return fmt.Errorf("failed to renew token: %w", err)
    }

    vc.token = secret.Auth.ClientToken
    vc.client.SetToken(vc.token)

    return nil
}

func (vc *VaultClient) StartTokenRenewal() {
    ticker := time.NewTicker(30 * time.Minute)
    go func() {
        for range ticker.C {
            if err := vc.RenewToken(); err != nil {
                log.Printf("Failed to renew token: %v", err)
            }
        }
    }()
}
```

### Application Integration

```go
// main.go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
)

type Config struct {
    DatabaseURL      string
    DatabasePassword string
    APIKey           string
    JWTSecret        string
    RedisPassword    string
}

type ConfigManager struct {
    vaultClient *VaultClient
    config      *Config
    logger      *zap.Logger
}

func NewConfigManager(vaultClient *VaultClient, logger *zap.Logger) *ConfigManager {
    return &ConfigManager{
        vaultClient: vaultClient,
        logger:      logger,
    }
}

func (cm *ConfigManager) LoadConfig() error {
    // Load database credentials
    dbSecret, err := cm.vaultClient.GetSecret("secret/data/my-app/database")
    if err != nil {
        return fmt.Errorf("failed to load database secret: %w", err)
    }

    // Load API keys
    apiSecret, err := cm.vaultClient.GetSecret("secret/data/my-app/api-keys")
    if err != nil {
        return fmt.Errorf("failed to load API secret: %w", err)
    }

    // Load JWT secret
    jwtSecret, err := cm.vaultClient.GetSecret("secret/data/my-app/jwt")
    if err != nil {
        return fmt.Errorf("failed to load JWT secret: %w", err)
    }

    // Load Redis credentials
    redisSecret, err := cm.vaultClient.GetSecret("secret/data/my-app/redis")
    if err != nil {
        return fmt.Errorf("failed to load Redis secret: %w", err)
    }

    cm.config = &Config{
        DatabaseURL:      dbSecret["url"].(string),
        DatabasePassword: dbSecret["password"].(string),
        APIKey:           apiSecret["key"].(string),
        JWTSecret:        jwtSecret["secret"].(string),
        RedisPassword:    redisSecret["password"].(string),
    }

    cm.logger.Info("Configuration loaded successfully")
    return nil
}

func (cm *ConfigManager) GetConfig() *Config {
    return cm.config
}

func (cm *ConfigManager) RefreshConfig() error {
    return cm.LoadConfig()
}

// Database service with secrets
type DatabaseService struct {
    config *Config
    logger *zap.Logger
}

func NewDatabaseService(config *Config, logger *zap.Logger) *DatabaseService {
    return &DatabaseService{
        config: config,
        logger: logger,
    }
}

func (ds *DatabaseService) Connect() error {
    // Use database credentials from vault
    connectionString := fmt.Sprintf("%s?password=%s",
        ds.config.DatabaseURL,
        ds.config.DatabasePassword,
    )

    ds.logger.Info("Connecting to database",
        zap.String("url", ds.config.DatabaseURL),
    )

    // Simulate database connection
    time.Sleep(100 * time.Millisecond)

    ds.logger.Info("Database connected successfully")
    return nil
}

// API service with secrets
type APIService struct {
    config *Config
    logger *zap.Logger
}

func NewAPIService(config *Config, logger *zap.Logger) *APIService {
    return &APIService{
        config: config,
        logger: logger,
    }
}

func (as *APIService) MakeAPICall(endpoint string) error {
    // Use API key from vault
    as.logger.Info("Making API call",
        zap.String("endpoint", endpoint),
        zap.String("api_key", as.config.APIKey[:8]+"..."),
    )

    // Simulate API call
    time.Sleep(50 * time.Millisecond)

    as.logger.Info("API call completed successfully")
    return nil
}

// JWT service with secrets
type JWTService struct {
    config *Config
    logger *zap.Logger
}

func NewJWTService(config *Config, logger *zap.Logger) *JWTService {
    return &JWTService{
        config: config,
        logger: logger,
    }
}

func (js *JWTService) GenerateToken(userID string) (string, error) {
    // Use JWT secret from vault
    js.logger.Info("Generating JWT token",
        zap.String("user_id", userID),
        zap.String("secret", js.config.JWTSecret[:8]+"..."),
    )

    // Simulate token generation
    token := fmt.Sprintf("jwt_token_for_%s", userID)

    js.logger.Info("JWT token generated successfully")
    return token, nil
}

// Redis service with secrets
type RedisService struct {
    config *Config
    logger *zap.Logger
}

func NewRedisService(config *Config, logger *zap.Logger) *RedisService {
    return &RedisService{
        config: config,
        logger: logger,
    }
}

func (rs *RedisService) Connect() error {
    // Use Redis password from vault
    rs.logger.Info("Connecting to Redis",
        zap.String("password", rs.config.RedisPassword[:8]+"..."),
    )

    // Simulate Redis connection
    time.Sleep(50 * time.Millisecond)

    rs.logger.Info("Redis connected successfully")
    return nil
}

// HTTP handlers
func setupRoutes(configManager *ConfigManager, logger *zap.Logger) *gin.Engine {
    r := gin.New()
    r.Use(gin.Recovery())

    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "timestamp": time.Now().UTC(),
        })
    })

    // Config refresh endpoint
    r.POST("/config/refresh", func(c *gin.Context) {
        if err := configManager.RefreshConfig(); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "Failed to refresh configuration",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "message": "Configuration refreshed successfully",
        })
    })

    // Database endpoint
    r.GET("/database/status", func(c *gin.Context) {
        config := configManager.GetConfig()
        dbService := NewDatabaseService(config, logger)

        if err := dbService.Connect(); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "Database connection failed",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "status": "connected",
        })
    })

    // API endpoint
    r.GET("/api/status", func(c *gin.Context) {
        config := configManager.GetConfig()
        apiService := NewAPIService(config, logger)

        if err := apiService.MakeAPICall("/status"); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "API call failed",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "status": "success",
        })
    })

    // JWT endpoint
    r.POST("/jwt/generate", func(c *gin.Context) {
        var request struct {
            UserID string `json:"user_id" binding:"required"`
        }

        if err := c.ShouldBindJSON(&request); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{
                "error": "Invalid request",
            })
            return
        }

        config := configManager.GetConfig()
        jwtService := NewJWTService(config, logger)

        token, err := jwtService.GenerateToken(request.UserID)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "Token generation failed",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "token": token,
        })
    })

    // Redis endpoint
    r.GET("/redis/status", func(c *gin.Context) {
        config := configManager.GetConfig()
        redisService := NewRedisService(config, logger)

        if err := redisService.Connect(); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error": "Redis connection failed",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "status": "connected",
        })
    })

    return r
}

func main() {
    // Setup logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Initialize Vault client
    vaultAddr := os.Getenv("VAULT_ADDR")
    if vaultAddr == "" {
        vaultAddr = "http://localhost:8200"
    }

    roleID := os.Getenv("VAULT_ROLE_ID")
    secretID := os.Getenv("VAULT_SECRET_ID")

    if roleID == "" || secretID == "" {
        logger.Fatal("VAULT_ROLE_ID and VAULT_SECRET_ID must be set")
    }

    vaultClient, err := NewVaultClient(vaultAddr, roleID, secretID)
    if err != nil {
        logger.Fatal("Failed to create Vault client", zap.Error(err))
    }

    // Start token renewal
    vaultClient.StartTokenRenewal()

    // Initialize config manager
    configManager := NewConfigManager(vaultClient, logger)

    // Load configuration
    if err := configManager.LoadConfig(); err != nil {
        logger.Fatal("Failed to load configuration", zap.Error(err))
    }

    // Setup routes
    router := setupRoutes(configManager, logger)

    // Start server
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    logger.Info("Starting server",
        zap.String("port", port),
        zap.String("vault_addr", vaultAddr),
    )

    if err := router.Run(":" + port); err != nil {
        logger.Fatal("Failed to start server", zap.Error(err))
    }
}
```

### AWS Secrets Manager Integration

```go
// aws-secrets.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/secretsmanager"
)

type AWSSecretsManager struct {
    client *secretsmanager.SecretsManager
}

func NewAWSSecretsManager(region string) (*AWSSecretsManager, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String(region),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create AWS session: %w", err)
    }

    client := secretsmanager.New(sess)

    return &AWSSecretsManager{
        client: client,
    }, nil
}

func (asm *AWSSecretsManager) GetSecret(secretName string) (map[string]interface{}, error) {
    input := &secretsmanager.GetSecretValueInput{
        SecretId: aws.String(secretName),
    }

    result, err := asm.client.GetSecretValue(input)
    if err != nil {
        return nil, fmt.Errorf("failed to get secret: %w", err)
    }

    var secretData map[string]interface{}
    if err := json.Unmarshal([]byte(*result.SecretString), &secretData); err != nil {
        return nil, fmt.Errorf("failed to unmarshal secret: %w", err)
    }

    return secretData, nil
}

func (asm *AWSSecretsManager) CreateSecret(secretName string, secretData map[string]interface{}) error {
    secretJSON, err := json.Marshal(secretData)
    if err != nil {
        return fmt.Errorf("failed to marshal secret: %w", err)
    }

    input := &secretsmanager.CreateSecretInput{
        Name:         aws.String(secretName),
        SecretString: aws.String(string(secretJSON)),
    }

    _, err = asm.client.CreateSecret(input)
    if err != nil {
        return fmt.Errorf("failed to create secret: %w", err)
    }

    return nil
}

func (asm *AWSSecretsManager) UpdateSecret(secretName string, secretData map[string]interface{}) error {
    secretJSON, err := json.Marshal(secretData)
    if err != nil {
        return fmt.Errorf("failed to marshal secret: %w", err)
    }

    input := &secretsmanager.UpdateSecretInput{
        SecretId:     aws.String(secretName),
        SecretString: aws.String(string(secretJSON)),
    }

    _, err = asm.client.UpdateSecret(input)
    if err != nil {
        return fmt.Errorf("failed to update secret: %w", err)
    }

    return nil
}

func (asm *AWSSecretsManager) DeleteSecret(secretName string) error {
    input := &secretsmanager.DeleteSecretInput{
        SecretId: aws.String(secretName),
    }

    _, err := asm.client.DeleteSecret(input)
    if err != nil {
        return fmt.Errorf("failed to delete secret: %w", err)
    }

    return nil
}

func (asm *AWSSecretsManager) RotateSecret(secretName string) error {
    input := &secretsmanager.RotateSecretInput{
        SecretId: aws.String(secretName),
    }

    _, err := asm.client.RotateSecret(input)
    if err != nil {
        return fmt.Errorf("failed to rotate secret: %w", err)
    }

    return nil
}
```

### Kubernetes Secrets Integration

```yaml
# kubernetes-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-app-secrets
  namespace: default
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  database-password: <base64-encoded-database-password>
  api-key: <base64-encoded-api-key>
  jwt-secret: <base64-encoded-jwt-secret>
  redis-password: <base64-encoded-redis-password>
---
apiVersion: v1
kind: Secret
metadata:
  name: vault-token
  namespace: default
type: Opaque
data:
  token: <base64-encoded-vault-token>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: my-app
          image: my-app:latest
          ports:
            - containerPort: 8080
          env:
            - name: VAULT_ADDR
              value: "http://vault:8200"
            - name: VAULT_ROLE_ID
              valueFrom:
                secretKeyRef:
                  name: vault-token
                  key: role-id
            - name: VAULT_SECRET_ID
              valueFrom:
                secretKeyRef:
                  name: vault-token
                  key: secret-id
          volumeMounts:
            - name: secrets
              mountPath: /etc/secrets
              readOnly: true
      volumes:
        - name: secrets
          secret:
            secretName: my-app-secrets
```

### Docker Compose for Secrets Management

```yaml
# docker-compose.yml
version: "3.8"

services:
  vault:
    image: vault:latest
    ports:
      - "8200:8200"
    environment:
      - VAULT_DEV_ROOT_TOKEN_ID=root
      - VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200
    volumes:
      - vault_data:/vault/data
    command: vault server -dev

  consul:
    image: consul:latest
    ports:
      - "8500:8500"
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    volumes:
      - consul_data:/consul/data
    command: consul agent -server -bootstrap-expect=1 -data-dir=/consul/data -ui -client=0.0.0.0

  my-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - VAULT_ADDR=http://vault:8200
      - VAULT_ROLE_ID=my-app-role-id
      - VAULT_SECRET_ID=my-app-secret-id
      - LOG_LEVEL=info
    depends_on:
      - vault

volumes:
  vault_data:
  consul_data:
```

## 🚀 Best Practices

### 1. Secret Rotation

```go
// Implement automatic secret rotation
func (cm *ConfigManager) RotateSecrets() error {
    // Rotate database password
    if err := cm.rotateDatabasePassword(); err != nil {
        return err
    }

    // Rotate API keys
    if err := cm.rotateAPIKeys(); err != nil {
        return err
    }

    // Reload configuration
    return cm.LoadConfig()
}
```

### 2. Access Control

```hcl
# Use least privilege principle
path "secret/data/my-app/*" {
  capabilities = ["read"]
}
```

### 3. Audit Logging

```go
// Log all secret access
func (cm *ConfigManager) logSecretAccess(secretPath string) {
    cm.logger.Info("Secret accessed",
        zap.String("path", secretPath),
        zap.Time("timestamp", time.Now()),
    )
}
```

## 🏢 Industry Insights

### Secrets Management Usage Patterns

- **Application Secrets**: Database credentials, API keys
- **Infrastructure Secrets**: SSL certificates, SSH keys
- **User Secrets**: Passwords, tokens
- **Service Secrets**: Inter-service authentication

### Enterprise Secrets Strategy

- **Centralized Management**: Single source of truth
- **Automated Rotation**: Regular secret updates
- **Access Control**: Role-based permissions
- **Compliance**: Audit trails and reporting

## 🎯 Interview Questions

### Basic Level

1. **What is secrets management?**

   - Secure storage of sensitive data
   - Access control and permissions
   - Secret rotation and lifecycle
   - Audit logging and compliance

2. **What are common types of secrets?**

   - Passwords and credentials
   - API keys and tokens
   - SSL certificates
   - Database connection strings

3. **What is HashiCorp Vault?**
   - Secrets management platform
   - Encryption and access control
   - Secret rotation
   - Audit logging

### Intermediate Level

4. **How do you implement secret rotation?**

   ```go
   func (cm *ConfigManager) RotateSecrets() error {
       // Generate new secrets
       // Update in vault
       // Reload configuration
       return cm.LoadConfig()
   }
   ```

5. **How do you handle secret access control?**

   - Role-based access control
   - Policy-based permissions
   - Least privilege principle
   - Audit logging

6. **How do you integrate secrets with applications?**
   - Vault client libraries
   - Environment variables
   - Configuration management
   - Service mesh integration

### Advanced Level

7. **How do you implement secret rotation?**

   - Automated rotation policies
   - Zero-downtime rotation
   - Rollback mechanisms
   - Monitoring and alerting

8. **How do you handle secret compliance?**

   - Audit trails
   - Access logging
   - Compliance reporting
   - Data retention policies

9. **How do you implement secret security?**
   - Encryption at rest
   - Encryption in transit
   - Access control
   - Threat detection

---

**Next**: [Zero Trust Architecture](ZeroTrustArchitecture.md) - Network security, identity verification, access control
