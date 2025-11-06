---
# Auto-generated front matter
Title: Goprojectlayout
LastUpdated: 2025-11-06T20:45:58.612572
Tags: []
Status: draft
---

# Go Project Layout Guide

This guide provides recommended project layouts for Go applications, with a focus on backend services and fintech applications.

## Standard Go Project Layout

### Basic Structure

```
project/
├── cmd/                    # Main applications
│   ├── server/            # Server application
│   │   └── main.go
│   ├── worker/            # Background worker
│   │   └── main.go
│   └── cli/               # CLI tool
│       └── main.go
├── internal/              # Private application code
│   ├── config/           # Configuration
│   ├── handlers/         # HTTP handlers
│   ├── services/         # Business logic
│   ├── repositories/     # Data access
│   ├── models/           # Data models
│   └── middleware/       # HTTP middleware
├── pkg/                   # Library code (importable)
│   ├── auth/             # Authentication utilities
│   ├── database/         # Database utilities
│   └── utils/            # General utilities
├── api/                   # API definitions
│   ├── openapi/          # OpenAPI specs
│   └── proto/            # Protocol buffers
├── web/                   # Web assets
│   ├── static/           # Static files
│   └── templates/        # HTML templates
├── scripts/               # Build and deployment scripts
├── deployments/           # Deployment configs
│   ├── docker/           # Docker files
│   └── k8s/              # Kubernetes manifests
├── docs/                  # Documentation
├── test/                  # Test data and scripts
├── go.mod                 # Go module file
├── go.sum                 # Go module checksums
├── Makefile              # Build automation
├── Dockerfile            # Container definition
└── README.md             # Project documentation
```

## Backend Service Layout

### Microservice Structure

```
payment-service/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── config/
│   │   ├── config.go
│   │   └── env.go
│   ├── handlers/
│   │   ├── payment.go
│   │   ├── health.go
│   │   └── middleware.go
│   ├── services/
│   │   ├── payment/
│   │   │   ├── service.go
│   │   │   ├── validator.go
│   │   │   └── processor.go
│   │   └── notification/
│   │       └── service.go
│   ├── repositories/
│   │   ├── payment/
│   │   │   ├── repository.go
│   │   │   └── postgres.go
│   │   └── user/
│   │       └── repository.go
│   ├── models/
│   │   ├── payment.go
│   │   ├── user.go
│   │   └── errors.go
│   ├── middleware/
│   │   ├── auth.go
│   │   ├── logging.go
│   │   └── rate_limit.go
│   └── database/
│       ├── connection.go
│       └── migrations/
├── pkg/
│   ├── logger/
│   │   └── logger.go
│   ├── validator/
│   │   └── validator.go
│   └── crypto/
│       └── encryption.go
├── api/
│   └── openapi/
│       └── payment-api.yaml
├── deployments/
│   ├── docker/
│   │   └── Dockerfile
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   └── migrate.sh
├── test/
│   ├── fixtures/
│   └── integration/
├── go.mod
├── go.sum
├── Makefile
├── Dockerfile
└── README.md
```

## Domain-Driven Design Layout

### DDD Structure

```
payment-domain/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── domain/
│   │   ├── payment/
│   │   │   ├── entity.go
│   │   │   ├── valueobject.go
│   │   │   ├── repository.go
│   │   │   └── service.go
│   │   ├── user/
│   │   │   ├── entity.go
│   │   │   ├── repository.go
│   │   │   └── service.go
│   │   └── shared/
│   │       ├── events.go
│   │       └── errors.go
│   ├── application/
│   │   ├── commands/
│   │   │   ├── create_payment.go
│   │   │   └── process_payment.go
│   │   ├── queries/
│   │   │   ├── get_payment.go
│   │   │   └── list_payments.go
│   │   └── handlers/
│   │       ├── payment_handler.go
│   │       └── user_handler.go
│   ├── infrastructure/
│   │   ├── persistence/
│   │   │   ├── postgres/
│   │   │   │   ├── payment_repository.go
│   │   │   │   └── user_repository.go
│   │   │   └── redis/
│   │   │       └── cache_repository.go
│   │   ├── messaging/
│   │   │   ├── kafka/
│   │   │   └── rabbitmq/
│   │   └── external/
│   │       ├── payment_gateway/
│   │       └── notification_service/
│   └── interfaces/
│       ├── http/
│       │   ├── handlers/
│       │   ├── middleware/
│       │   └── routes.go
│       └── grpc/
│           ├── handlers/
│           └── server.go
├── pkg/
│   ├── events/
│   │   ├── publisher.go
│   │   └── subscriber.go
│   └── utils/
│       ├── uuid.go
│       └── time.go
├── api/
│   ├── openapi/
│   └── proto/
├── deployments/
├── scripts/
├── test/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── go.mod
├── go.sum
└── README.md
```

## Clean Architecture Layout

### Clean Architecture Structure

```
clean-payment-service/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── entities/          # Business entities
│   │   ├── payment.go
│   │   ├── user.go
│   │   └── transaction.go
│   ├── usecases/          # Business logic
│   │   ├── payment/
│   │   │   ├── create_payment.go
│   │   │   ├── process_payment.go
│   │   │   └── get_payment.go
│   │   └── user/
│   │       └── get_user.go
│   ├── interfaces/        # External interfaces
│   │   ├── repositories/
│   │   │   ├── payment_repository.go
│   │   │   └── user_repository.go
│   │   ├── handlers/
│   │   │   ├── payment_handler.go
│   │   │   └── user_handler.go
│   │   └── services/
│   │       ├── payment_gateway.go
│   │       └── notification_service.go
│   ├── frameworks/        # External frameworks
│   │   ├── database/
│   │   │   ├── postgres/
│   │   │   └── redis/
│   │   ├── web/
│   │   │   ├── gin/
│   │   │   └── echo/
│   │   └── messaging/
│   │       └── kafka/
│   └── config/
│       ├── config.go
│       └── env.go
├── pkg/
│   ├── errors/
│   │   └── errors.go
│   └── utils/
│       └── utils.go
├── api/
├── deployments/
├── scripts/
├── test/
├── go.mod
├── go.sum
└── README.md
```

## Monorepo Layout

### Monorepo Structure

```
payment-platform/
├── services/
│   ├── payment-service/
│   │   ├── cmd/
│   │   ├── internal/
│   │   ├── pkg/
│   │   ├── go.mod
│   │   └── Dockerfile
│   ├── user-service/
│   │   ├── cmd/
│   │   ├── internal/
│   │   ├── pkg/
│   │   ├── go.mod
│   │   └── Dockerfile
│   └── notification-service/
│       ├── cmd/
│       ├── internal/
│       ├── pkg/
│       ├── go.mod
│       └── Dockerfile
├── shared/
│   ├── pkg/
│   │   ├── auth/
│   │   ├── database/
│   │   ├── events/
│   │   └── utils/
│   ├── api/
│   │   ├── openapi/
│   │   └── proto/
│   └── go.mod
├── tools/
│   ├── migration-tool/
│   ├── test-runner/
│   └── deployment-script/
├── deployments/
│   ├── docker-compose.yml
│   ├── k8s/
│   └── terraform/
├── docs/
├── scripts/
├── go.work              # Go workspace file
└── README.md
```

## Key Directories Explained

### cmd/

- **Purpose**: Main applications for the project
- **Structure**: One subdirectory per main application
- **Example**: `cmd/server/main.go`, `cmd/worker/main.go`
- **Best Practice**: Keep main.go files small, delegate to internal packages

### internal/

- **Purpose**: Private application code that cannot be imported by other projects
- **Structure**: Organized by feature or layer
- **Example**: `internal/handlers/`, `internal/services/`
- **Best Practice**: Use this for all application-specific code

### pkg/

- **Purpose**: Library code that can be imported by other projects
- **Structure**: Organized by functionality
- **Example**: `pkg/logger/`, `pkg/validator/`
- **Best Practice**: Only put code here that others might want to import

### api/

- **Purpose**: API definitions and contracts
- **Structure**: Organized by API type
- **Example**: `api/openapi/`, `api/proto/`
- **Best Practice**: Keep API definitions separate from implementation

### deployments/

- **Purpose**: Deployment configurations
- **Structure**: Organized by deployment target
- **Example**: `deployments/docker/`, `deployments/k8s/`
- **Best Practice**: Include all deployment artifacts

## File Naming Conventions

### Go Files

- **Packages**: Use lowercase, single word
- **Files**: Use snake_case for multi-word names
- **Examples**: `payment_service.go`, `user_repository.go`

### Configuration Files

- **Environment**: `.env`, `.env.local`, `.env.production`
- **Docker**: `Dockerfile`, `docker-compose.yml`
- **Kubernetes**: `deployment.yaml`, `service.yaml`

### Documentation

- **Main**: `README.md`
- **API**: `API.md`
- **Deployment**: `DEPLOYMENT.md`
- **Development**: `DEVELOPMENT.md`

## Go Module Best Practices

### go.mod Structure

```go
module github.com/company/payment-service

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    github.com/redis/go-redis/v9 v9.0.5
)

require (
    // Indirect dependencies
)
```

### Version Management

- Use semantic versioning
- Pin major versions for stability
- Use `go mod tidy` regularly
- Consider using `go.work` for monorepos

## Build and Deployment

### Makefile Example

```makefile
.PHONY: build test clean docker

# Build the application
build:
	go build -o bin/server cmd/server/main.go

# Run tests
test:
	go test -v ./...

# Run tests with coverage
test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f coverage.out

# Build Docker image
docker:
	docker build -t payment-service:latest .

# Run with Docker Compose
docker-compose:
	docker-compose up -d

# Database migrations
migrate-up:
	migrate -path migrations -database "postgres://user:pass@localhost/db?sslmode=disable" up

migrate-down:
	migrate -path migrations -database "postgres://user:pass@localhost/db?sslmode=disable" down
```

### Dockerfile Example

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main cmd/server/main.go

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .
COPY --from=builder /app/config ./config

EXPOSE 8080
CMD ["./main"]
```

## Testing Structure

### Test Organization

```
test/
├── unit/                 # Unit tests
│   ├── services/
│   ├── repositories/
│   └── handlers/
├── integration/          # Integration tests
│   ├── database/
│   ├── api/
│   └── external/
├── e2e/                  # End-to-end tests
│   ├── scenarios/
│   └── fixtures/
├── fixtures/             # Test data
│   ├── users.json
│   └── payments.json
└── mocks/                # Mock implementations
    ├── payment_gateway.go
    └── notification_service.go
```

### Test File Naming

- **Unit tests**: `*_test.go` in the same package
- **Integration tests**: `*_integration_test.go`
- **Benchmark tests**: `*_benchmark_test.go`

## Environment Configuration

### Configuration Structure

```
config/
├── config.go            # Configuration struct
├── env.go               # Environment variable loading
├── development.yaml     # Development config
├── staging.yaml         # Staging config
├── production.yaml      # Production config
└── test.yaml           # Test config
```

### Configuration Example

```go
type Config struct {
    Server   ServerConfig   `yaml:"server"`
    Database DatabaseConfig `yaml:"database"`
    Redis    RedisConfig    `yaml:"redis"`
    Payment  PaymentConfig  `yaml:"payment"`
}

type ServerConfig struct {
    Port         int           `yaml:"port" env:"SERVER_PORT" env-default:"8080"`
    ReadTimeout  time.Duration `yaml:"read_timeout" env:"SERVER_READ_TIMEOUT" env-default:"30s"`
    WriteTimeout time.Duration `yaml:"write_timeout" env:"SERVER_WRITE_TIMEOUT" env-default:"30s"`
}

type DatabaseConfig struct {
    Host     string `yaml:"host" env:"DB_HOST" env-default:"localhost"`
    Port     int    `yaml:"port" env:"DB_PORT" env-default:"5432"`
    Name     string `yaml:"name" env:"DB_NAME" env-default:"payments"`
    User     string `yaml:"user" env:"DB_USER" env-default:"postgres"`
    Password string `yaml:"password" env:"DB_PASSWORD"`
    SSLMode  string `yaml:"ssl_mode" env:"DB_SSL_MODE" env-default:"disable"`
}
```

## Best Practices

### 1. Package Organization

- Keep packages focused and cohesive
- Use meaningful package names
- Avoid circular dependencies
- Group related functionality

### 2. Import Management

- Use absolute imports
- Group imports (standard, third-party, local)
- Use `goimports` for formatting
- Remove unused imports

### 3. Error Handling

- Use custom error types
- Wrap errors with context
- Handle errors at appropriate levels
- Log errors with sufficient context

### 4. Testing

- Write tests for all public functions
- Use table-driven tests
- Mock external dependencies
- Test error conditions

### 5. Documentation

- Document all public APIs
- Use meaningful variable names
- Write clear commit messages
- Maintain up-to-date README

This project layout guide provides a solid foundation for organizing Go applications, with particular emphasis on backend services and fintech applications. Choose the structure that best fits your project's needs and team preferences.
