# Deployment & DevOps Guide

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Setup](#infrastructure-setup)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Container Orchestration](#container-orchestration)
5. [Monitoring & Logging](#monitoring--logging)
6. [Security & Compliance](#security--compliance)
7. [Backup & Recovery](#backup--recovery)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)

## Overview

### Learning Objectives

- Deploy the Master Engineer Curriculum to production
- Implement CI/CD pipelines for automated deployment
- Set up monitoring and logging systems
- Ensure security and compliance
- Implement backup and disaster recovery

### What is Deployment & DevOps?

Deployment and DevOps involve the processes, tools, and practices used to deploy, monitor, and maintain the Master Engineer Curriculum in production environments.

## Infrastructure Setup

### 1. Cloud Infrastructure

#### AWS Infrastructure as Code
```yaml
# infrastructure/aws/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
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
    Name = "master-engineer-curriculum-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "master-engineer-curriculum-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count  = 2
  vpc_id = aws_vpc.main.id

  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "master-engineer-curriculum-public-${count.index + 1}"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count  = 2
  vpc_id = aws_vpc.main.id

  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "master-engineer-curriculum-private-${count.index + 1}"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "master-engineer-curriculum"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy,
  ]
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "master-engineer-curriculum-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "curriculum"
  username = "curriculum_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "master-engineer-curriculum-final-snapshot"
  
  tags = {
    Name = "master-engineer-curriculum-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "master-engineer-curriculum-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "master-engineer-curriculum-cache"
  description                = "Redis cluster for Master Engineer Curriculum"
  
  node_type            = "cache.t3.micro"
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

# S3 Bucket for Static Content
resource "aws_s3_bucket" "static_content" {
  bucket = "master-engineer-curriculum-static-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_versioning" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_s3_bucket.static_content.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.static_content.bucket}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main.cloudfront_access_identity_path
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.static_content.bucket}"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
```

### 2. Kubernetes Manifests

#### Application Deployment
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: master-engineer-curriculum
  labels:
    name: master-engineer-curriculum

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: curriculum-config
  namespace: master-engineer-curriculum
data:
  DATABASE_URL: "postgresql://curriculum_user:password@db-service:5432/curriculum"
  REDIS_URL: "redis://redis-service:6379"
  JWT_SECRET: "your-jwt-secret"
  API_BASE_URL: "https://api.masterengineer.com"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: curriculum-secrets
  namespace: master-engineer-curriculum
type: Opaque
data:
  db-password: cGFzc3dvcmQ=  # base64 encoded password
  jwt-secret: eW91ci1qd3Qtc2VjcmV0  # base64 encoded JWT secret

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: curriculum-api
  namespace: master-engineer-curriculum
  labels:
    app: curriculum-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: curriculum-api
  template:
    metadata:
      labels:
        app: curriculum-api
    spec:
      containers:
      - name: curriculum-api
        image: master-engineer-curriculum/api:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: curriculum-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: curriculum-config
              key: REDIS_URL
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: curriculum-secrets
              key: db-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: curriculum-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
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
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: curriculum-api-service
  namespace: master-engineer-curriculum
spec:
  selector:
    app: curriculum-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: curriculum-ingress
  namespace: master-engineer-curriculum
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.masterengineer.com
    secretName: curriculum-tls
  rules:
  - host: api.masterengineer.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: curriculum-api-service
            port:
              number: 80
```

## CI/CD Pipeline

### 1. GitHub Actions Workflow

#### Complete CI/CD Pipeline
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-west-2
  EKS_CLUSTER_NAME: master-engineer-curriculum
  ECR_REPOSITORY: master-engineer-curriculum

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: |
        go mod download
        npm ci
    
    - name: Run Go tests
      run: |
        go test -v ./...
        go test -race -coverprofile=coverage.out ./...
    
    - name: Run Node.js tests
      run: npm test
    
    - name: Run security scan
      run: |
        go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
        gosec ./...
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
    
    - name: Build and push API image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY/api:$IMAGE_TAG -f Dockerfile.api .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY/api:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY/api:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY/api:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY/api:latest
    
    - name: Build and push Frontend image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY/frontend:$IMAGE_TAG -f Dockerfile.frontend .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY/frontend:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY/frontend:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY/frontend:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY/frontend:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name ${{ env.EKS_CLUSTER_NAME }}
    
    - name: Deploy to Kubernetes
      run: |
        envsubst < k8s/deployment.yaml | kubectl apply -f -
        kubectl rollout status deployment/curriculum-api -n master-engineer-curriculum
    
    - name: Run database migrations
      run: |
        kubectl run migration-job --image=$ECR_REGISTRY/$ECR_REPOSITORY/api:${{ github.sha }} --rm -i --restart=Never --command -- /app/migrate up
    
    - name: Run smoke tests
      run: |
        kubectl get pods -n master-engineer-curriculum
        kubectl get services -n master-engineer-curriculum
```

### 2. Docker Configuration

#### Multi-stage Dockerfile
```dockerfile
# Dockerfile.api
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/api

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .
COPY --from=builder /app/migrations ./migrations

EXPOSE 8080
CMD ["./main"]

---
# Dockerfile.frontend
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Container Orchestration

### 1. Kubernetes Deployment

#### Helm Chart
```yaml
# helm/curriculum/Chart.yaml
apiVersion: v2
name: master-engineer-curriculum
description: Master Engineer Curriculum Application
type: application
version: 0.1.0
appVersion: "1.0.0"

---
# helm/curriculum/values.yaml
replicaCount: 3

image:
  repository: master-engineer-curriculum/api
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.masterengineer.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: curriculum-tls
      hosts:
        - api.masterengineer.com

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

---
# helm/curriculum/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "curriculum.fullname" . }}
  labels:
    {{- include "curriculum.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "curriculum.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "curriculum.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "curriculum.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: DATABASE_URL
              valueFrom:
                configMapKeyRef:
                  name: curriculum-config
                  key: DATABASE_URL
            - name: REDIS_URL
              valueFrom:
                configMapKeyRef:
                  name: curriculum-config
                  key: REDIS_URL
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

## Monitoring & Logging

### 1. Prometheus & Grafana

#### Monitoring Configuration
```yaml
# monitoring/prometheus-config.yaml
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

  - job_name: 'curriculum-api'
    static_configs:
      - targets: ['curriculum-api-service:80']
    metrics_path: /metrics
    scrape_interval: 5s

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

---
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "Master Engineer Curriculum Dashboard",
    "tags": ["curriculum", "education"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### 2. ELK Stack

#### Logging Configuration
```yaml
# logging/elasticsearch.yml
cluster.name: master-engineer-curriculum
node.name: elasticsearch-1
network.host: 0.0.0.0
discovery.type: single-node
xpack.security.enabled: false

---
# logging/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "curriculum-api" {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
    date {
      match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "curriculum-logs-%{+YYYY.MM.dd}"
  }
}

---
# logging/filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/log/containers/*curriculum*.log'
  processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
      matchers:
      - logs_path:
          logs_path: "/var/log/containers/"

output.logstash:
  hosts: ["logstash:5044"]
```

## Security & Compliance

### 1. Security Scanning

#### Security Configuration
```yaml
# security/trivy-scan.yml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

---
# security/opa-policy.rego
package curriculum

import rego.v1

# Deny requests that don't have proper authentication
deny[msg] {
    input.request.method == "POST"
    not input.request.headers["Authorization"]
    msg := "Authentication required for POST requests"
}

# Deny requests to admin endpoints without admin role
deny[msg] {
    input.request.path = ["admin", _]
    not input.user.roles[_] == "admin"
    msg := "Admin role required for admin endpoints"
}

# Allow read operations for authenticated users
allow {
    input.request.method in ["GET", "HEAD"]
    input.user.authenticated
}

# Allow write operations for authenticated users with appropriate role
allow {
    input.request.method in ["POST", "PUT", "DELETE"]
    input.user.authenticated
    input.user.roles[_] in ["admin", "instructor"]
}
```

### 2. Compliance

#### SOC 2 Compliance
```yaml
# compliance/soc2-controls.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: soc2-controls
  namespace: master-engineer-curriculum
data:
  # CC6.1 - Logical and Physical Access Security
  access_control_policy: |
    - All access must be authenticated
    - Multi-factor authentication required for admin access
    - Regular access reviews and deprovisioning
  
  # CC6.2 - System Access Controls
  system_access_policy: |
    - Role-based access control (RBAC) implemented
    - Principle of least privilege enforced
    - Regular access audits
  
  # CC6.3 - Data Protection
  data_protection_policy: |
    - Encryption at rest and in transit
    - Data classification and handling procedures
    - Regular security assessments
  
  # CC6.4 - Monitoring and Logging
  monitoring_policy: |
    - Comprehensive logging of all system activities
    - Real-time monitoring and alerting
    - Regular log review and analysis
  
  # CC6.5 - Incident Response
  incident_response_policy: |
    - Documented incident response procedures
    - 24/7 monitoring and response capability
    - Regular incident response testing
```

## Backup & Recovery

### 1. Database Backup

#### Backup Strategy
```bash
#!/bin/bash
# scripts/backup-database.sh

set -e

# Configuration
DB_HOST="curriculum-db.cluster-xyz.us-west-2.rds.amazonaws.com"
DB_NAME="curriculum"
DB_USER="curriculum_user"
S3_BUCKET="master-engineer-curriculum-backups"
BACKUP_RETENTION_DAYS=30

# Create backup
BACKUP_FILE="curriculum-backup-$(date +%Y%m%d-%H%M%S).sql"
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE
BACKUP_FILE="${BACKUP_FILE}.gz"

# Upload to S3
aws s3 cp $BACKUP_FILE s3://$S3_BUCKET/database/

# Clean up local file
rm $BACKUP_FILE

# Clean up old backups
aws s3 ls s3://$S3_BUCKET/database/ --recursive | \
  awk '$1 < "'$(date -d "$BACKUP_RETENTION_DAYS days ago" +%Y-%m-%d)'" {print $4}' | \
  xargs -I {} aws s3 rm s3://$S3_BUCKET/{}

echo "Backup completed: $BACKUP_FILE"
```

### 2. Disaster Recovery

#### Recovery Procedures
```yaml
# disaster-recovery/recovery-plan.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
  namespace: master-engineer-curriculum
data:
  recovery_procedures: |
    # RTO: 4 hours, RPO: 1 hour
    
    ## Database Recovery
    1. Identify the most recent backup
    2. Restore database from backup
    3. Apply any transaction logs if available
    4. Verify data integrity
    
    ## Application Recovery
    1. Deploy application to new infrastructure
    2. Update DNS records to point to new infrastructure
    3. Verify application functionality
    4. Monitor system health
    
    ## Data Recovery
    1. Restore static content from S3
    2. Restore user uploads from S3
    3. Verify all data is accessible
    4. Update CDN cache if necessary
  
  contact_information: |
    Primary: on-call@masterengineer.com
    Secondary: engineering@masterengineer.com
    Escalation: cto@masterengineer.com
  
  recovery_checklist: |
    - [ ] Database restored and verified
    - [ ] Application deployed and healthy
    - [ ] DNS updated and propagated
    - [ ] Monitoring alerts configured
    - [ ] User notifications sent
    - [ ] Post-incident review scheduled
```

## Follow-up Questions

### 1. Deployment Strategy
**Q: What's the best deployment strategy for the curriculum?**
A: Use blue-green or rolling deployments with proper testing, monitoring, and rollback capabilities.

### 2. Monitoring
**Q: How do you implement effective monitoring?**
A: Use comprehensive metrics, logging, alerting, and dashboards to monitor application health, performance, and user experience.

### 3. Security
**Q: What security measures are essential for production deployment?**
A: Implement authentication, authorization, encryption, security scanning, compliance controls, and regular security audits.

## Sources

### Infrastructure
- **Terraform**: [Infrastructure as Code](https://www.terraform.io/)
- **AWS**: [Cloud Services](https://aws.amazon.com/)
- **Kubernetes**: [Container Orchestration](https://kubernetes.io/)

### CI/CD
- **GitHub Actions**: [CI/CD Platform](https://github.com/features/actions)
- **Jenkins**: [Automation Server](https://www.jenkins.io/)
- **GitLab CI**: [DevOps Platform](https://about.gitlab.com/stages-devops-lifecycle/continuous-integration/)

### Monitoring
- **Prometheus**: [Monitoring System](https://prometheus.io/)
- **Grafana**: [Visualization Platform](https://grafana.com/)
- **ELK Stack**: [Logging Platform](https://www.elastic.co/elastic-stack/)

---

**Next**: [Mobile App](../mobile_app/README.md) | **Previous**: [Study Tracker](../study_tracker/README.md) | **Up**: [Deployment DevOps](../README.md)
