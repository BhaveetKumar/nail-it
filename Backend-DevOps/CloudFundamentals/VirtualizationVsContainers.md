# 🐳 Virtualization vs Containers: VMs vs Containers, Docker Basics

> **Master containerization and virtualization for modern backend development**

## 📚 Concept

### Virtualization

Virtualization creates virtual machines (VMs) that run complete operating systems on top of a hypervisor, providing hardware abstraction.

### Containers

Containers package applications with their dependencies, sharing the host OS kernel while providing process isolation.

## 🏗️ Architecture Comparison

### Virtualization Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Host OS                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Hypervisor  │  │ Hypervisor  │  │ Hypervisor  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Guest OS    │  │ Guest OS    │  │ Guest OS    │     │
│  │ App 1       │  │ App 2       │  │ App 3       │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### Container Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Host OS                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Container   │  │ Container   │  │ Container   │     │
│  │ App 1       │  │ App 2       │  │ App 3       │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Container Runtime                      │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Docker Implementation

```dockerfile
# Dockerfile for Go application
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

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

### Multi-stage Dockerfile

```dockerfile
# Multi-stage build for Node.js application
FROM node:18-alpine AS dependencies

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:18-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runtime

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy dependencies
COPY --from=dependencies --chown=nextjs:nodejs /app/node_modules ./node_modules

# Copy built application
COPY --from=build --chown=nextjs:nodejs /app/.next ./.next
COPY --from=build --chown=nextjs:nodejs /app/public ./public
COPY --from=build --chown=nextjs:nodejs /app/package*.json ./

USER nextjs

EXPOSE 3000

ENV NODE_ENV=production
ENV PORT=3000

CMD ["npm", "start"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  web:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
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
    command: redis-server --appendonly yes
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
      - web
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

### Container Orchestration with Docker Swarm

```yaml
# docker-stack.yml
version: "3.8"

services:
  web:
    image: myapp:latest
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://user:password@db:5432/mydb
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M
      placement:
        constraints:
          - node.role == worker
    networks:
      - app-network

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        delay: 10s
      resources:
        limits:
          cpus: "0.25"
          memory: 256M
        reservations:
          cpus: "0.1"
          memory: 128M
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:

networks:
  app-network:
    driver: overlay
    attachable: true
```

## 🚀 Best Practices

### 1. Container Security

```dockerfile
# Security best practices
FROM alpine:latest

# Install security updates
RUN apk update && apk upgrade

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set proper permissions
RUN chmod 755 /app

# Use specific versions
FROM node:18.19.0-alpine3.18

# Remove unnecessary packages
RUN apk del build-dependencies

# Use multi-stage builds
FROM node:18-alpine AS builder
# ... build steps
FROM node:18-alpine AS runtime
COPY --from=builder /app/dist ./dist
```

### 2. Container Optimization

```dockerfile
# Optimize image size
FROM node:18-alpine

# Use .dockerignore
COPY . .

# Use specific COPY commands
COPY package*.json ./
RUN npm ci --only=production

# Remove cache
RUN npm cache clean --force

# Use distroless images
FROM gcr.io/distroless/nodejs18-debian11
COPY --from=builder /app/dist ./dist
```

### 3. Container Monitoring

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro

volumes:
  grafana_data:
```

## 🏢 Industry Insights

### Docker Adoption

- **Netflix**: Containerized microservices
- **Uber**: Container orchestration
- **Spotify**: Development environment standardization
- **PayPal**: Application deployment

### Container Orchestration

- **Kubernetes**: Industry standard
- **Docker Swarm**: Simple orchestration
- **Amazon ECS**: Managed container service
- **Google GKE**: Managed Kubernetes

### Container Security

- **Image scanning**: Vulnerability detection
- **Runtime security**: Process monitoring
- **Network policies**: Traffic control
- **Secrets management**: Secure configuration

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between VMs and containers?**

   - VMs: Full OS, hypervisor, hardware abstraction
   - Containers: Shared OS kernel, lightweight, faster startup

2. **What are the benefits of containers?**

   - Portability
   - Consistency
   - Resource efficiency
   - Faster deployment
   - Scalability

3. **What is Docker?**
   - Containerization platform
   - Image-based deployment
   - Container runtime
   - Development tooling

### Intermediate Level

4. **How do you optimize Docker images?**

   ```dockerfile
   # Use multi-stage builds
   FROM node:18-alpine AS builder
   # ... build steps
   FROM node:18-alpine AS runtime
   COPY --from=builder /app/dist ./dist
   ```

5. **How do you handle container networking?**

   - Bridge networks
   - Overlay networks
   - Service discovery
   - Load balancing

6. **How do you manage container secrets?**
   - Docker secrets
   - Environment variables
   - Volume mounts
   - External secret managers

### Advanced Level

7. **How do you implement container orchestration?**

   - Service discovery
   - Load balancing
   - Health checks
   - Rolling updates
   - Resource management

8. **How do you handle container security?**

   - Image scanning
   - Runtime security
   - Network policies
   - Access control
   - Compliance

9. **How do you implement container monitoring?**
   - Metrics collection
   - Log aggregation
   - Health checks
   - Alerting
   - Tracing

---

**Next**: [Networking in Cloud](./NetworkingInCloud.md) - VPC, subnets, security groups, load balancers
