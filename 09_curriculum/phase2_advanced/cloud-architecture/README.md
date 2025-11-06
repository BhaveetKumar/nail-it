---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.426405
Tags: []
Status: draft
---

# Cloud Architecture

## Table of Contents

1. [Overview](#overview)
2. [Cloud Computing Models](#cloud-computing-models)
3. [Microservices and Containerization](#microservices-and-containerization)
4. [Kubernetes Orchestration](#kubernetes-orchestration)
5. [Serverless Architecture](#serverless-architecture)
6. [Cloud Security](#cloud-security)
7. [Cloud Monitoring and Observability](#cloud-monitoring-and-observability)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master cloud computing models (IaaS, PaaS, SaaS)
- Design and implement microservices architectures
- Orchestrate containers with Kubernetes
- Build serverless applications
- Implement cloud security best practices
- Monitor and observe cloud systems

### What is Cloud Architecture?

Cloud Architecture involves designing, implementing, and managing scalable, reliable, and secure applications using cloud computing services and technologies.

## Cloud Computing Models

### 1. Infrastructure as a Service (IaaS)

#### IaaS Implementation with Terraform
```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "main-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "main-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "public-subnet-${count.index + 1}"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "private-subnet-${count.index + 1}"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = 2
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "web" {
  name_prefix = "web-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "web-sg"
  }
}

resource "aws_security_group" "database" {
  name_prefix = "database-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "database-sg"
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "main-db"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "maindb"
  username = "admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Name = "main-db"
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "main-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "main-db-subnet-group"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "main-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.web.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name = "main-alb"
  }
}

resource "aws_lb_target_group" "web" {
  name     = "web-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "web" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.web.arn
  }
}

# Auto Scaling Group
resource "aws_launch_template" "web" {
  name_prefix   = "web-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = "t3.micro"

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = base64encode(templatefile("user_data.sh", {
    db_host = aws_db_instance.main.endpoint
    db_name = aws_db_instance.main.db_name
    db_user = aws_db_instance.main.username
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "web-instance"
    }
  }
}

resource "aws_autoscaling_group" "web" {
  name                = "web-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.web.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 2
  max_size         = 10
  desired_capacity = 2

  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "web-instance"
    propagate_at_launch = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.web.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.web.name
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "70"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.web.name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "20"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.web.name
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Outputs
output "vpc_id" {
  value = aws_vpc.main.id
}

output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "rds_endpoint" {
  value = aws_db_instance.main.endpoint
}
```

### 2. Platform as a Service (PaaS)

#### PaaS with Docker and Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: ./web
    ports:
      - "80:8080"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    networks:
      - app-network
    restart: unless-stopped

  api:
    build: ./api
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    networks:
      - app-network
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - app-network
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    networks:
      - app-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
      - api
    networks:
      - app-network
    restart: unless-stopped

  worker:
    build: ./worker
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
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

## Microservices and Containerization

### 1. Docker Containerization

#### Multi-stage Dockerfile
```dockerfile
# Dockerfile
# Build stage
FROM node:16-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:16-alpine AS production

# Create app user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package*.json ./

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
```

#### Go Microservice Dockerfile
```dockerfile
# Dockerfile
# Build stage
FROM golang:1.19-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Production stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy binary
COPY --from=builder /app/main .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Start application
CMD ["./main"]
```

### 2. Microservices Architecture

#### Service Discovery
```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/gorilla/mux"
)

type Service struct {
    Name     string `json:"name"`
    Host     string `json:"host"`
    Port     int    `json:"port"`
    Health   bool   `json:"health"`
    LastSeen time.Time `json:"last_seen"`
}

type ServiceRegistry struct {
    services map[string]*Service
    mutex    sync.RWMutex
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string]*Service),
    }
}

func (sr *ServiceRegistry) Register(service *Service) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    service.LastSeen = time.Now()
    sr.services[service.Name] = service
    log.Printf("Registered service: %s at %s:%d", service.Name, service.Host, service.Port)
}

func (sr *ServiceRegistry) Deregister(serviceName string) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    delete(sr.services, serviceName)
    log.Printf("Deregistered service: %s", serviceName)
}

func (sr *ServiceRegistry) GetService(serviceName string) (*Service, bool) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    service, exists := sr.services[serviceName]
    if !exists {
        return nil, false
    }
    
    // Check if service is still healthy
    if time.Since(service.LastSeen) > 30*time.Second {
        service.Health = false
    }
    
    return service, service.Health
}

func (sr *ServiceRegistry) ListServices() []*Service {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    services := make([]*Service, 0, len(sr.services))
    for _, service := range sr.services {
        services = append(services, service)
    }
    return services
}

func (sr *ServiceRegistry) HealthCheck() {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    for name, service := range sr.services {
        if time.Since(service.LastSeen) > 30*time.Second {
            service.Health = false
            log.Printf("Service %s marked as unhealthy", name)
        }
    }
}

type ServiceDiscovery struct {
    registry *ServiceRegistry
    router   *mux.Router
}

func NewServiceDiscovery() *ServiceDiscovery {
    registry := NewServiceRegistry()
    router := mux.NewRouter()
    
    sd := &ServiceDiscovery{
        registry: registry,
        router:   router,
    }
    
    sd.setupRoutes()
    return sd
}

func (sd *ServiceDiscovery) setupRoutes() {
    sd.router.HandleFunc("/register", sd.registerHandler).Methods("POST")
    sd.router.HandleFunc("/deregister/{name}", sd.deregisterHandler).Methods("DELETE")
    sd.router.HandleFunc("/services", sd.listServicesHandler).Methods("GET")
    sd.router.HandleFunc("/services/{name}", sd.getServiceHandler).Methods("GET")
    sd.router.HandleFunc("/health", sd.healthHandler).Methods("GET")
}

func (sd *ServiceDiscovery) registerHandler(w http.ResponseWriter, r *http.Request) {
    var service Service
    if err := json.NewDecoder(r.Body).Decode(&service); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    sd.registry.Register(&service)
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(map[string]string{"status": "registered"})
}

func (sd *ServiceDiscovery) deregisterHandler(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    serviceName := vars["name"]
    
    sd.registry.Deregister(serviceName)
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "deregistered"})
}

func (sd *ServiceDiscovery) listServicesHandler(w http.ResponseWriter, r *http.Request) {
    services := sd.registry.ListServices()
    json.NewEncoder(w).Encode(services)
}

func (sd *ServiceDiscovery) getServiceHandler(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    serviceName := vars["name"]
    
    service, exists := sd.registry.GetService(serviceName)
    if !exists {
        http.Error(w, "Service not found", http.StatusNotFound)
        return
    }
    
    json.NewEncoder(w).Encode(service)
}

func (sd *ServiceDiscovery) healthHandler(w http.ResponseWriter, r *http.Request) {
    json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func (sd *ServiceDiscovery) StartHealthCheck() {
    ticker := time.NewTicker(10 * time.Second)
    go func() {
        for range ticker.C {
            sd.registry.HealthCheck()
        }
    }()
}

func main() {
    sd := NewServiceDiscovery()
    sd.StartHealthCheck()
    
    log.Println("Service Discovery starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", sd.router))
}
```

## Kubernetes Orchestration

### 1. Kubernetes Manifests

#### Deployment and Service
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: myapp:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis-url
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  selector:
    app: web-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: myapp-tls
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-app-service
            port:
              number: 80
```

#### ConfigMap and Secret
```yaml
# config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  redis-url: "redis://redis-service:6379"
  log-level: "info"
  max-connections: "100"
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAZGI6NTQzMi9teWFwcA==
  api-key: eW91ci1hcGkta2V5LWhlcmU=
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: myapp
        - name: POSTGRES_USER
          value: user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### 2. Horizontal Pod Autoscaler

#### HPA Configuration
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

## Serverless Architecture

### 1. AWS Lambda Functions

#### Lambda Function with Go
```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
)

type Request struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

type Response struct {
    Message string `json:"message"`
    Status  int    `json:"status"`
}

func HandleRequest(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // Parse request body
    var req Request
    if err := json.Unmarshal([]byte(request.Body), &req); err != nil {
        return events.APIGatewayProxyResponse{
            StatusCode: 400,
            Body:       `{"error": "Invalid JSON"}`,
        }, nil
    }

    // Process request
    message := fmt.Sprintf("Hello %s, you are %d years old", req.Name, req.Age)
    
    response := Response{
        Message: message,
        Status:  200,
    }

    // Marshal response
    responseBody, err := json.Marshal(response)
    if err != nil {
        return events.APIGatewayProxyResponse{
            StatusCode: 500,
            Body:       `{"error": "Internal server error"}`,
        }, nil
    }

    return events.APIGatewayProxyResponse{
        StatusCode: 200,
        Headers: map[string]string{
            "Content-Type": "application/json",
        },
        Body: string(responseBody),
    }, nil
}

func main() {
    lambda.Start(HandleRequest)
}
```

#### Serverless Framework Configuration
```yaml
# serverless.yml
service: my-serverless-app

provider:
  name: aws
  runtime: go1.x
  region: us-west-2
  stage: ${opt:stage, 'dev'}
  environment:
    STAGE: ${self:provider.stage}
    DB_TABLE: ${self:service}-${self:provider.stage}-users
  iamRoleStatements:
    - Effect: Allow
      Action:
        - dynamodb:Query
        - dynamodb:Scan
        - dynamodb:GetItem
        - dynamodb:PutItem
        - dynamodb:UpdateItem
        - dynamodb:DeleteItem
      Resource: "arn:aws:dynamodb:${self:provider.region}:*:table/${self:provider.environment.DB_TABLE}"

functions:
  createUser:
    handler: bin/createUser
    events:
      - http:
          path: users
          method: post
          cors: true
    environment:
      FUNCTION_NAME: createUser
  
  getUser:
    handler: bin/getUser
    events:
      - http:
          path: users/{id}
          method: get
          cors: true
    environment:
      FUNCTION_NAME: getUser
  
  updateUser:
    handler: bin/updateUser
    events:
      - http:
          path: users/{id}
          method: put
          cors: true
    environment:
      FUNCTION_NAME: updateUser
  
  deleteUser:
    handler: bin/deleteUser
    events:
      - http:
          path: users/{id}
          method: delete
          cors: true
    environment:
      FUNCTION_NAME: deleteUser

resources:
  Resources:
    UsersTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: ${self:provider.environment.DB_TABLE}
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
        StreamSpecification:
          StreamViewType: NEW_AND_OLD_IMAGES

plugins:
  - serverless-offline
  - serverless-dynamodb-local

custom:
  serverless-offline:
    httpPort: 3000
  dynamodb:
    start:
      port: 8000
      inMemory: true
      migrate: true
```

### 2. Azure Functions

#### Azure Function with Node.js
```javascript
// index.js
const { app } = require('@azure/functions');

app.http('httpTrigger', {
    methods: ['GET', 'POST'],
    authLevel: 'anonymous',
    handler: async (request, context) => {
        context.log('HTTP trigger function processed a request.');

        const name = request.query.get('name') || await request.text() || 'world';

        return {
            body: JSON.stringify({
                message: `Hello, ${name}!`,
                timestamp: new Date().toISOString()
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        };
    }
});

// function.json
{
    "bindings": [
        {
            "authLevel": "anonymous",
            "type": "httpTrigger",
            "direction": "in",
            "name": "req",
            "methods": [
                "get",
                "post"
            ]
        },
        {
            "type": "http",
            "direction": "out",
            "name": "res"
        }
    ]
}
```

## Cloud Security

### 1. Identity and Access Management

#### IAM Policy Example
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Access",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::my-bucket/*"
        },
        {
            "Sid": "AllowDynamoDBAccess",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": "arn:aws:dynamodb:us-west-2:123456789012:table/MyTable"
        },
        {
            "Sid": "DenyS3DeleteBucket",
            "Effect": "Deny",
            "Action": "s3:DeleteBucket",
            "Resource": "*"
        }
    ]
}
```

### 2. Network Security

#### Security Group Configuration
```yaml
# security-groups.yaml
apiVersion: v1
kind: NetworkPolicy
metadata:
  name: web-app-network-policy
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

## Cloud Monitoring and Observability

### 1. Prometheus and Grafana

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "Kubernetes Cluster Monitoring",
    "tags": ["kubernetes"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m])",
            "legendFormat": "{{pod}}"
          }
        ],
        "yAxes": [
          {
            "label": "CPU Usage (cores)"
          }
        ]
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes",
            "legendFormat": "{{pod}}"
          }
        ],
        "yAxes": [
          {
            "label": "Memory Usage (bytes)"
          }
        ]
      },
      {
        "id": 3,
        "title": "Pod Status",
        "type": "stat",
        "targets": [
          {
            "expr": "kube_pod_status_phase",
            "legendFormat": "{{phase}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Follow-up Questions

### 1. Cloud Computing Models
**Q: What's the difference between IaaS, PaaS, and SaaS?**
A: IaaS provides virtualized computing resources, PaaS provides a platform for application development, and SaaS provides complete software applications.

### 2. Containerization
**Q: What are the benefits of using containers over virtual machines?**
A: Containers are more lightweight, start faster, use fewer resources, and provide better portability across different environments.

### 3. Kubernetes
**Q: What's the difference between a Deployment and a StatefulSet?**
A: Deployments are for stateless applications with random pod names, while StatefulSets are for stateful applications with stable, ordered pod names and persistent storage.

## Sources

### Books
- **Kubernetes: Up and Running** by Kelsey Hightower
- **Designing Distributed Systems** by Brendan Burns
- **Cloud Native Patterns** by Cornelia Davis

### Online Resources
- **Kubernetes Documentation** - Official Kubernetes guides
- **AWS Well-Architected Framework** - Cloud architecture best practices
- **Docker Documentation** - Containerization guides

## Projects

### 1. Microservices Platform
**Objective**: Build a complete microservices platform
**Requirements**: Service discovery, API gateway, monitoring
**Deliverables**: Production-ready microservices architecture

### 2. Kubernetes Cluster
**Objective**: Deploy and manage a Kubernetes cluster
**Requirements**: Multi-node cluster, monitoring, CI/CD
**Deliverables**: Complete Kubernetes deployment

### 3. Serverless Application
**Objective**: Create a serverless application
**Requirements**: Multiple functions, database, monitoring
**Deliverables**: End-to-end serverless solution

---

**Next**: [Advanced Algorithms](../../../README.md) | **Previous**: [Machine Learning](../../../README.md) | **Up**: [Phase 2](README.md)



## Implementations

<!-- AUTO-GENERATED ANCHOR: originally referenced as #implementations -->

Placeholder content. Please replace with proper section.
