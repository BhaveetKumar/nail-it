# 🐳 Docker Basics: Containerization and Image Management

> **Master Docker for containerization, image management, and application packaging**

## 📚 Concept

Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers. It provides consistency across development, testing, and production environments.

### Key Features
- **Containerization**: Package applications with dependencies
- **Image Management**: Build, store, and distribute images
- **Portability**: Run anywhere Docker is installed
- **Isolation**: Process and resource isolation
- **Efficiency**: Shared kernel, minimal overhead
- **Versioning**: Image versioning and tagging

## 🏗️ Docker Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Host                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Docker    │  │   Docker    │  │   Docker    │     │
│  │  Container  │  │  Container  │  │  Container  │     │
│  │   (App 1)   │  │   (App 2)   │  │   (App 3)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Docker Engine                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Docker    │  │   Docker    │  │   Docker    │ │ │
│  │  │    API      │  │   Daemon    │  │   CLI       │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Docker    │  │   Docker    │  │   Docker    │     │
│  │   Images    │  │   Volumes   │  │  Networks   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Dockerfile for Go Application

```dockerfile
# Dockerfile
# Multi-stage build for Go application
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/main .

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Change ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./main"]
```

### Dockerfile for Node.js Application

```dockerfile
# Dockerfile for Node.js
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Change ownership
RUN chown -R nextjs:nodejs /app

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start the application
CMD ["npm", "start"]
```

### Dockerfile for Python Application

```dockerfile
# Dockerfile for Python
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "app.py"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=info
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  app-network:
    driver: bridge
```

### Docker Commands

```bash
# Build Docker image
docker build -t my-app:latest .

# Build with build args
docker build --build-arg VERSION=1.0.0 -t my-app:1.0.0 .

# Run container
docker run -d -p 8080:8080 --name my-app my-app:latest

# Run with environment variables
docker run -d -p 8080:8080 -e DATABASE_URL=postgresql://localhost:5432/mydb my-app:latest

# Run with volumes
docker run -d -p 8080:8080 -v /host/path:/container/path my-app:latest

# Run with network
docker run -d -p 8080:8080 --network my-network my-app:latest

# List containers
docker ps
docker ps -a

# List images
docker images

# Remove container
docker rm my-app

# Remove image
docker rmi my-app:latest

# View logs
docker logs my-app
docker logs -f my-app

# Execute command in container
docker exec -it my-app /bin/bash

# Copy files
docker cp my-app:/app/logs/app.log ./logs/

# Inspect container
docker inspect my-app

# Save image
docker save -o my-app.tar my-app:latest

# Load image
docker load -i my-app.tar

# Tag image
docker tag my-app:latest my-registry.com/my-app:1.0.0

# Push image
docker push my-registry.com/my-app:1.0.0

# Pull image
docker pull my-registry.com/my-app:1.0.0
```

## 🚀 Best Practices

### 1. Security Best Practices
```dockerfile
# Use non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Switch to non-root user
USER appuser

# Use specific base image versions
FROM node:18-alpine

# Remove package manager cache
RUN npm ci --only=production && npm cache clean --force
```

### 2. Performance Optimization
```dockerfile
# Multi-stage build
FROM golang:1.21-alpine AS builder
# ... build steps

FROM alpine:latest
COPY --from=builder /app/main .

# Use .dockerignore
# node_modules
# .git
# *.log
```

### 3. Health Checks
```dockerfile
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

## 🏢 Industry Insights

### Docker Usage Patterns
- **Microservices**: Containerized services
- **CI/CD**: Build and deployment
- **Development**: Consistent environments
- **Production**: Scalable deployments

### Enterprise Docker Strategy
- **Security**: Image scanning and policies
- **Registry**: Private image registry
- **Orchestration**: Kubernetes integration
- **Monitoring**: Container monitoring

## 🎯 Interview Questions

### Basic Level
1. **What is Docker?**
   - Containerization platform
   - Application packaging
   - Environment consistency
   - Resource isolation

2. **What is a Docker image?**
   - Read-only template
   - Application snapshot
   - Layered filesystem
   - Versioned artifact

3. **What is a Docker container?**
   - Running instance
   - Isolated process
   - Resource limits
   - Ephemeral state

### Intermediate Level
4. **How do you optimize Docker images?**
   ```dockerfile
   # Multi-stage build
   FROM golang:1.21-alpine AS builder
   # ... build steps
   
   FROM alpine:latest
   COPY --from=builder /app/main .
   ```

5. **How do you handle Docker security?**
   - Use non-root users
   - Scan images for vulnerabilities
   - Use specific base image versions
   - Implement least privilege

6. **How do you manage Docker volumes?**
   - Named volumes
   - Bind mounts
   - Volume drivers
   - Backup strategies

### Advanced Level
7. **How do you implement Docker patterns?**
   - Multi-stage builds
   - Health checks
   - Init containers
   - Sidecar patterns

8. **How do you handle Docker networking?**
   - Bridge networks
   - Overlay networks
   - Service discovery
   - Load balancing

9. **How do you implement Docker monitoring?**
   - Container metrics
   - Log aggregation
   - Health monitoring
   - Performance tracking

---

**Next**: [Docker Compose](./DockerCompose.md) - Multi-container applications, orchestration
