# 🐳 Docker Compose: Multi-Container Applications and Orchestration

> **Master Docker Compose for multi-container applications and local development**

## 📚 Concept

Docker Compose is a tool for defining and running multi-container Docker applications. It uses YAML files to configure application services, networks, and volumes, making it easy to manage complex applications with multiple components.

### Key Features
- **Multi-Container**: Define multiple services in one file
- **Service Orchestration**: Start, stop, and manage services
- **Networking**: Automatic service discovery
- **Volumes**: Persistent data storage
- **Environment**: Development and production configurations
- **Scaling**: Scale services up and down

## 🏗️ Docker Compose Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Service   │  │   Service   │  │   Service   │     │
│  │   (Web)     │  │   (DB)      │  │   (Cache)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Docker Network                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Service   │  │   Service   │  │   Service   │ │ │
│  │  │ Discovery   │  │ Discovery   │  │ Discovery   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Docker    │  │   Docker    │  │   Docker    │     │
│  │   Volumes   │  │   Volumes   │  │   Volumes   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Complete Application Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Web Application
  web:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - VERSION=1.0.0
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=info
      - ENV=development
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

  # Database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
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
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass password
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 256M
        reservations:
          cpus: '0.1'
          memory: 128M

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./static:/usr/share/nginx/html/static
    depends_on:
      - web
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 256M
        reservations:
          cpus: '0.1'
          memory: 128M

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - app-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - app-network
    restart: unless-stopped

  # Log Aggregation
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - app-network
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  elasticsearch_data:
    driver: local

networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Development Configuration

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - DEBUG=true
    ports:
      - "3000:3000"
      - "9229:9229"  # Debug port
    command: npm run dev

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb_dev
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"

volumes:
  postgres_dev_data:
```

### Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    image: my-registry.com/my-app:latest
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:${DB_PASSWORD}@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_prod_data:/data
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

volumes:
  postgres_prod_data:
  redis_prod_data:
```

### Docker Compose Commands

```bash
# Start services
docker-compose up
docker-compose up -d  # Detached mode
docker-compose up --build  # Rebuild images

# Start specific services
docker-compose up web db

# Stop services
docker-compose down
docker-compose down -v  # Remove volumes

# Scale services
docker-compose up --scale web=3

# View logs
docker-compose logs
docker-compose logs -f web
docker-compose logs --tail=100 web

# Execute commands
docker-compose exec web bash
docker-compose exec db psql -U user -d mydb

# Build services
docker-compose build
docker-compose build --no-cache

# Pull images
docker-compose pull

# View service status
docker-compose ps

# View service configuration
docker-compose config

# Run one-time commands
docker-compose run web npm test
docker-compose run db psql -U user -d mydb

# Restart services
docker-compose restart
docker-compose restart web

# Pause/Unpause services
docker-compose pause
docker-compose unpause

# Remove services
docker-compose rm
docker-compose rm -f  # Force removal

# Use override files
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## 🚀 Best Practices

### 1. Service Dependencies
```yaml
# Use health checks for dependencies
services:
  web:
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy

  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 2. Environment Configuration
```yaml
# Use environment files
services:
  web:
    env_file:
      - .env
      - .env.local
    environment:
      - NODE_ENV=${NODE_ENV}
      - DATABASE_URL=${DATABASE_URL}
```

### 3. Resource Management
```yaml
# Set resource limits
services:
  web:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

## 🏢 Industry Insights

### Docker Compose Usage Patterns
- **Development**: Local development environments
- **Testing**: Integration testing
- **CI/CD**: Build and test pipelines
- **Production**: Simple deployments

### Enterprise Docker Compose Strategy
- **Multi-Environment**: Dev, staging, production
- **Security**: Secrets management
- **Monitoring**: Health checks and logging
- **Scaling**: Resource limits and scaling

## 🎯 Interview Questions

### Basic Level
1. **What is Docker Compose?**
   - Multi-container orchestration
   - YAML configuration
   - Service management
   - Local development

2. **What are Docker Compose services?**
   - Container definitions
   - Service configuration
   - Dependencies
   - Networking

3. **What are Docker Compose volumes?**
   - Persistent storage
   - Data sharing
   - Volume drivers
   - Backup strategies

### Intermediate Level
4. **How do you handle service dependencies?**
   ```yaml
   services:
     web:
       depends_on:
         db:
           condition: service_healthy
   ```

5. **How do you manage environment configurations?**
   - Environment files
   - Variable substitution
   - Override files
   - Secrets management

6. **How do you implement health checks?**
   ```yaml
   services:
     web:
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

### Advanced Level
7. **How do you implement Docker Compose patterns?**
   - Multi-stage builds
   - Service discovery
   - Load balancing
   - Monitoring

8. **How do you handle Docker Compose scaling?**
   - Service scaling
   - Resource limits
   - Load balancing
   - Performance optimization

9. **How do you implement Docker Compose monitoring?**
   - Health checks
   - Log aggregation
   - Metrics collection
   - Alerting

---

**Next**: [Kubernetes Basics](./KubernetesBasics.md) - Container orchestration, scaling, management
