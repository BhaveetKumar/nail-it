# 🚀 DevOps Tools - Complete Guide

> **Comprehensive DevOps guide with tools, best practices, and real-world examples**

## 📋 Table of Contents

1. [Containerization with Docker](#containerization-with-docker)
2. [Orchestration with Kubernetes](#orchestration-with-kubernetes)
3. [CI/CD Pipelines](#cicd-pipelines)
4. [Infrastructure as Code](#infrastructure-as-code)
5. [Monitoring & Logging](#monitoring--logging)
6. [Security & Compliance](#security--compliance)
7. [Cloud Platforms](#cloud-platforms)
8. [Best Practices](#best-practices)

---

## 🐳 Containerization with Docker

### **1. Docker Fundamentals**

#### **Dockerfile Best Practices**
```dockerfile
# Multi-stage build for Go application
# Stage 1: Build stage - contains build tools and dependencies
FROM golang:1.21-alpine AS builder

# Set working directory inside container
WORKDIR /app

# Copy go mod files first for better layer caching
# This allows Docker to cache the dependency download layer
COPY go.mod go.sum ./

# Download dependencies (cached if go.mod/go.sum unchanged)
RUN go mod download

# Copy source code
COPY . .

# Build the application with optimizations
# CGO_ENABLED=0: Disable CGO for static binary
# GOOS=linux: Target Linux OS
# -a -installsuffix cgo: Force rebuild of all packages
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Stage 2: Runtime stage - minimal image for production
FROM alpine:latest

# Install ca-certificates for HTTPS connections
RUN apk --no-cache add ca-certificates

# Create non-root user for security
RUN adduser -D -s /bin/sh appuser

# Set working directory
WORKDIR /root/

# Copy binary from builder stage
COPY --from=builder /app/main .

# Change ownership to non-root user
RUN chown appuser:appuser main

# Switch to non-root user for security
USER appuser

# Expose port 8080
EXPOSE 8080

# Health check to monitor application status
# --interval=30s: Check every 30 seconds
# --timeout=3s: Wait 3 seconds for response
# --start-period=5s: Wait 5 seconds before first check
# --retries=3: Retry 3 times before marking unhealthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./main"]
```

**Key Concepts Explained:**
- **Multi-stage Build**: Separate build and runtime stages for smaller images
- **Layer Caching**: Copy dependencies first to leverage Docker layer caching
- **Security**: Non-root user, minimal base image, static binary
- **Health Checks**: Built-in monitoring for container health
- **Optimization**: Static binary, minimal runtime dependencies

#### **Docker Compose for Development**
```yaml
# Docker Compose version
version: '3.8'

services:
  # Main application service
  app:
    build: .                    # Build from current directory Dockerfile
    ports:
      - "8080:8080"            # Map host port 8080 to container port 8080
    environment:
      - DB_HOST=postgres       # Database host (service name)
      - DB_PORT=5432           # Database port
      - DB_NAME=myapp          # Database name
      - DB_USER=postgres       # Database user
      - DB_PASSWORD=password   # Database password
    depends_on:
      - postgres               # Wait for postgres to start
      - redis                  # Wait for redis to start
    volumes:
      - ./logs:/app/logs       # Mount host logs directory to container
    networks:
      - app-network            # Connect to custom network

  # PostgreSQL database service
  postgres:
    image: postgres:15-alpine  # Use PostgreSQL 15 Alpine image
    environment:
      - POSTGRES_DB=myapp      # Create database named 'myapp'
      - POSTGRES_USER=postgres # Create user 'postgres'
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - app-network

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

volumes:
  postgres_data:
  redis_data:

networks:
  app-network:
    driver: bridge
```

### **2. Docker Security Best Practices**

```bash
#!/bin/bash
# Docker security scanning script

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image myapp:latest

# Check for secrets
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    trufflesecurity/trufflehog docker --image myapp:latest

# Analyze image layers
docker history myapp:latest

# Check running containers
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

# Monitor resource usage
docker stats --no-stream
```

---

## ☸️ Orchestration with Kubernetes

### **3. Kubernetes Manifests**

#### **Deployment with Health Checks**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-deployment
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
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
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  host: cG9zdGdyZXM=  # base64 encoded
  password: cGFzc3dvcmQ=  # base64 encoded
```

#### **Horizontal Pod Autoscaler**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp-deployment
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
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### **4. Kubernetes Operators**

#### **Custom Resource Definition**
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              databaseName:
                type: string
              replicas:
                type: integer
                minimum: 1
                maximum: 5
              storage:
                type: object
                properties:
                  size:
                    type: string
                  class:
                    type: string
          status:
            type: object
            properties:
              phase:
                type: string
              readyReplicas:
                type: integer
    subresources:
      status: {}
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database
```

---

## 🔄 CI/CD Pipelines

### **5. GitHub Actions Workflow**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
        cache: true
    
    - name: Run tests
      run: |
        go test -v ./...
        go test -race ./...
        go test -coverprofile=coverage.out ./...
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'security-scan-results.sarif'

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/myapp-deployment \
          myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        kubectl rollout status deployment/myapp-deployment
    
    - name: Run smoke tests
      run: |
        kubectl get pods -l app=myapp
        kubectl get services
```

### **6. Jenkins Pipeline**

```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_NAME = 'myapp'
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'go test -v ./...'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'go test -tags=integration -v ./...'
                    }
                }
                stage('Security Scan') {
                    steps {
                        sh 'gosec ./...'
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh """
                    kubectl config use-context staging
                    kubectl set image deployment/myapp-deployment \
                        myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}
                    kubectl rollout status deployment/myapp-deployment
                """
            }
        }
        
        stage('E2E Tests') {
            steps {
                sh 'npm test -- --config=e2e.config.js'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    kubectl config use-context production
                    kubectl set image deployment/myapp-deployment \
                        myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}
                    kubectl rollout status deployment/myapp-deployment
                """
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            slackSend channel: '#deployments',
                      color: 'good',
                      message: "✅ Deployment successful: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
        failure {
            slackSend channel: '#deployments',
                      color: 'danger',
                      message: "❌ Deployment failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
    }
}
```

---

## 🏗️ Infrastructure as Code

### **7. Terraform Configuration**

#### **AWS Infrastructure**
```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "Terraform"
    }
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

# Route Table for Public Subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

# Route Table Association for Public Subnets
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.allowed_cidr_blocks
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
  
  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.private[*].id
  
  scaling_config {
    desired_size = var.node_group_desired_size
    max_size     = var.node_group_max_size
    min_size     = var.node_group_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
  
  tags = {
    Name = "${var.project_name}-node-group"
  }
}
```

#### **Variables**
```hcl
# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "myapp"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_group_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "node_group_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "node_group_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access EKS API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
```

---

## 📊 Monitoring & Logging

### **8. Prometheus & Grafana Setup**

#### **Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

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

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
```

#### **Grafana Dashboard**
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
            "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "max": 100,
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "max": 100,
            "min": 0
          }
        ]
      },
      {
        "id": 3,
        "title": "Pod Status",
        "type": "table",
        "targets": [
          {
            "expr": "kube_pod_status_phase",
            "format": "table"
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

### **9. ELK Stack for Logging**

#### **Elasticsearch Configuration**
```yaml
# elasticsearch.yml
cluster.name: "myapp-cluster"
node.name: "elasticsearch-node-1"
network.host: 0.0.0.0
discovery.type: single-node

# Security settings
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.key: /usr/share/elasticsearch/config/certs/elasticsearch.key
xpack.security.transport.ssl.certificate: /usr/share/elasticsearch/config/certs/elasticsearch.crt
xpack.security.transport.ssl.certificate_authorities: /usr/share/elasticsearch/config/certs/ca.crt

# Index settings
action.auto_create_index: false
action.destructive_requires_name: true

# Memory settings
bootstrap.memory_lock: true
```

#### **Logstash Configuration**
```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "myapp" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      add_field => { "service" => "myapp" }
    }
  }
  
  if [fields][service] == "nginx" {
    grok {
      match => { "message" => "%{NGINXACCESS}" }
    }
    
    date {
      match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{+YYYY.MM.dd}"
    user => "elastic"
    password => "${ELASTIC_PASSWORD}"
  }
}
```

---

## 🔒 Security & Compliance

### **10. Security Scanning**

#### **Container Security**
```bash
#!/bin/bash
# Security scanning script

# Trivy vulnerability scan
trivy image --severity HIGH,CRITICAL myapp:latest

# Docker Bench Security
docker run --rm --net host --pid host --userns host --cap-add audit_control \
    -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
    -v /etc:/etc:ro \
    -v /usr/bin/containerd:/usr/bin/containerd:ro \
    -v /usr/bin/runc:/usr/bin/runc:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    --label docker_bench_security \
    docker/docker-bench-security

# Falco runtime security
falco -c /etc/falco/falco.yaml

# OPA Gatekeeper policies
kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml
```

#### **Kubernetes Security Policies**
```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
---
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
spec:
  podSelector: {}
  policyTypes:
  - Ingress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-myapp-ingress
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

---

## ☁️ Cloud Platforms

### **11. AWS Best Practices**

#### **CloudFormation Template**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'EKS Cluster with managed node groups'

Parameters:
  ClusterName:
    Type: String
    Default: myapp-cluster
    Description: Name of the EKS cluster
  
  NodeGroupName:
    Type: String
    Default: myapp-nodes
    Description: Name of the node group
  
  NodeInstanceType:
    Type: String
    Default: t3.medium
    AllowedValues:
      - t3.small
      - t3.medium
      - t3.large
    Description: EC2 instance type for the node group

Resources:
  EKSCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: !Ref ClusterName
      Version: '1.28'
      RoleArn: !GetAtt EKSClusterRole.Arn
      ResourcesVpcConfig:
        SecurityGroupIds:
          - !Ref ClusterSecurityGroup
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
        EndpointConfig:
          PrivateAccess: true
          PublicAccess: true
          PublicAccessCidrs:
            - 0.0.0.0/0

  EKSClusterRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: eks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

  NodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSCluster
      NodeGroupName: !Ref NodeGroupName
      NodeRole: !GetAtt NodeGroupRole.Arn
      InstanceTypes:
        - !Ref NodeInstanceType
      ScalingConfig:
        MinSize: 1
        MaxSize: 10
        DesiredSize: 3
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      UpdateConfig:
        MaxUnavailable: 1

  NodeGroupRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

Outputs:
  ClusterName:
    Description: Name of the EKS cluster
    Value: !Ref EKSCluster
    Export:
      Name: !Sub '${AWS::StackName}-ClusterName'
  
  ClusterEndpoint:
    Description: Endpoint for the EKS cluster
    Value: !GetAtt EKSCluster.Endpoint
    Export:
      Name: !Sub '${AWS::StackName}-ClusterEndpoint'
```

---

## 📚 Additional Resources

### **Books**
- [The Phoenix Project](https://www.oreilly.com/library/view/the-phoenix-project/9781457191350/) - Gene Kim
- [The DevOps Handbook](https://www.oreilly.com/library/view/the-devops-handbook/9781457191381/) - Gene Kim
- [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) - Google

### **Online Resources**
- [Kubernetes Documentation](https://kubernetes.io/docs/) - Official K8s docs
- [Docker Documentation](https://docs.docker.com/) - Official Docker docs
- [Terraform Documentation](https://www.terraform.io/docs/) - Infrastructure as Code

### **Video Resources**
- [TechWorld with Nana](https://www.youtube.com/c/TechWorldwithNana) - DevOps tutorials
- [KodeKloud](https://www.youtube.com/c/KodeKloud) - Kubernetes and DevOps
- [Cloud Native Computing Foundation](https://www.youtube.com/c/CloudNativeComputingFoundation) - CNCF projects

---

*This comprehensive guide covers essential DevOps tools and practices with real-world examples and best practices for modern software development.*
