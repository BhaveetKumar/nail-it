# ☁️ Multi-Cloud: Cross-Cloud Architecture and Management

> **Master multi-cloud strategies for high availability, cost optimization, and vendor independence**

## 📚 Concept

Multi-cloud is the use of multiple cloud computing services from different providers to avoid vendor lock-in, improve reliability, and optimize costs. It involves distributing workloads across AWS, GCP, Azure, and other cloud platforms while maintaining consistent management and security.

### Key Features
- **Vendor Independence**: Avoid single cloud provider dependency
- **High Availability**: Distribute workloads across multiple clouds
- **Cost Optimization**: Leverage best pricing from different providers
- **Risk Mitigation**: Reduce impact of cloud provider outages
- **Compliance**: Meet regulatory requirements across regions
- **Performance**: Optimize for different geographic regions

## 🏗️ Multi-Cloud Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Multi-Cloud Architecture                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │     AWS     │  │     GCP     │  │    Azure    │     │
│  │   Region    │  │   Region    │  │   Region    │     │
│  │   us-east-1 │  │  us-central1│  │  eastus     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Cloud Management Layer               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Terraform │  │   Ansible   │  │   Pulumi    │ │ │
│  │  │   (IaC)     │  │   (Config)  │  │   (IaC)     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Application Layer                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Load      │  │   Service   │  │   Data      │ │ │
│  │  │   Balancer  │  │   Mesh      │  │   Sync      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Monitoring│  │   Security  │  │   Backup    │     │
│  │   & Logging │  │   & IAM     │  │   & DR      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Multi-Cloud Infrastructure with Terraform

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
      Cloud       = "aws"
    }
  }
}

# GCP Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  
  default_labels = {
    environment = var.environment
    project     = var.project_name
    managed_by  = "terraform"
    cloud       = "gcp"
  }
}

# Azure Provider
provider "azurerm" {
  features {}
  
  default_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
    Cloud       = "azure"
  }
}

# AWS VPC
resource "aws_vpc" "main" {
  cidr_block           = var.aws_vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-aws-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-aws-igw"
  }
}

resource "aws_subnet" "public" {
  count = length(var.aws_availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.aws_public_subnet_cidrs[count.index]
  availability_zone       = var.aws_availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-aws-public-subnet-${count.index + 1}"
  }
}

# GCP VPC
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-gcp-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "public" {
  count = length(var.gcp_zones)

  name          = "${var.project_name}-gcp-public-subnet-${count.index + 1}"
  ip_cidr_range = var.gcp_public_subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.main.id
}

# Azure VNet
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-azure-vnet"
  address_space       = [var.azure_vnet_cidr]
  location            = var.azure_location
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_subnet" "public" {
  count = length(var.azure_zones)

  name                 = "${var.project_name}-azure-public-subnet-${count.index + 1}"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.azure_public_subnet_cidrs[count.index]]
}

resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-azure-rg"
  location = var.azure_location
}

# AWS Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-aws-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.web.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name = "${var.project_name}-aws-alb"
  }
}

# GCP Load Balancer
resource "google_compute_global_forwarding_rule" "main" {
  name       = "${var.project_name}-gcp-lb"
  target     = google_compute_target_http_proxy.main.id
  port_range = "80"
}

resource "google_compute_target_http_proxy" "main" {
  name    = "${var.project_name}-gcp-http-proxy"
  url_map = google_compute_url_map.main.id
}

resource "google_compute_url_map" "main" {
  name            = "${var.project_name}-gcp-url-map"
  default_service = google_compute_backend_service.main.id
}

resource "google_compute_backend_service" "main" {
  name        = "${var.project_name}-gcp-backend"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 10

  backend {
    group = google_compute_instance_group.main.id
  }

  health_checks = [google_compute_health_check.main.id]
}

# Azure Load Balancer
resource "azurerm_lb" "main" {
  name                = "${var.project_name}-azure-lb"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  frontend_ip_configuration {
    name                 = "PublicIPAddress"
    public_ip_address_id = azurerm_public_ip.main.id
  }
}

resource "azurerm_public_ip" "main" {
  name                = "${var.project_name}-azure-pip"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
}

# AWS EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-aws-eks"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.public[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy,
  ]

  tags = {
    Name = "${var.project_name}-aws-eks"
  }
}

resource "aws_iam_role" "eks_cluster" {
  name = "${var.project_name}-aws-eks-cluster-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# GCP GKE Cluster
resource "google_container_cluster" "main" {
  name     = "${var.project_name}-gcp-gke"
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.public[0].id

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "main" {
  name       = "${var.project_name}-gcp-node-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 2

  node_config {
    preemptible  = true
    machine_type = "e2-medium"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# Azure AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-azure-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}-aks"

  default_node_pool {
    name       = "default"
    node_count = 2
    vm_size    = "Standard_D2s_v3"
  }

  identity {
    type = "SystemAssigned"
  }
}

# AWS RDS Database
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-aws-rds"

  engine         = "postgres"
  engine_version = "14.7"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "myapp"
  username = "postgres"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  tags = {
    Name = "${var.project_name}-aws-rds"
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-aws-db-subnet-group"
  subnet_ids = aws_subnet.public[*].id

  tags = {
    Name = "${var.project_name}-aws-db-subnet-group"
  }
}

# GCP Cloud SQL Database
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-gcp-sql"
  database_version = "POSTGRES_14"
  region           = var.gcp_region

  settings {
    tier = "db-f1-micro"

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }

    ip_configuration {
      ipv4_enabled = true
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "main" {
  name     = "myapp"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "main" {
  name     = "postgres"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Azure Database for PostgreSQL
resource "azurerm_postgresql_server" "main" {
  name                = "${var.project_name}-azure-postgres"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  administrator_login          = "postgres"
  administrator_login_password = var.db_password

  sku_name   = "B_Gen5_1"
  version    = "11"
  storage_mb = 5120

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false
  auto_grow_enabled            = true

  ssl_enforcement_enabled = true
}

resource "azurerm_postgresql_database" "main" {
  name                = "myapp"
  resource_group_name = azurerm_resource_group.main.name
  server_name         = azurerm_postgresql_server.main.name
  charset             = "UTF8"
  collation           = "English_United States.1252"
}

# Cross-Cloud Networking
resource "aws_vpc_peering_connection" "aws_gcp" {
  vpc_id      = aws_vpc.main.id
  peer_vpc_id = google_compute_network.main.id
  peer_region = var.gcp_region

  tags = {
    Name = "${var.project_name}-aws-gcp-peering"
  }
}

resource "aws_vpc_peering_connection" "aws_azure" {
  vpc_id      = aws_vpc.main.id
  peer_vpc_id = azurerm_virtual_network.main.id
  peer_region = var.azure_location

  tags = {
    Name = "${var.project_name}-aws-azure-peering"
  }
}

# Security Groups
resource "aws_security_group" "web" {
  name_prefix = "${var.project_name}-aws-web-"
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
    Name = "${var.project_name}-aws-web-sg"
  }
}

resource "aws_security_group" "database" {
  name_prefix = "${var.project_name}-aws-db-"
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
    Name = "${var.project_name}-aws-db-sg"
  }
}

# Outputs
output "aws_vpc_id" {
  description = "ID of the AWS VPC"
  value       = aws_vpc.main.id
}

output "gcp_vpc_id" {
  description = "ID of the GCP VPC"
  value       = google_compute_network.main.id
}

output "azure_vnet_id" {
  description = "ID of the Azure VNet"
  value       = azurerm_virtual_network.main.id
}

output "aws_alb_dns" {
  description = "DNS name of the AWS ALB"
  value       = aws_lb.main.dns_name
}

output "gcp_lb_ip" {
  description = "IP address of the GCP Load Balancer"
  value       = google_compute_global_forwarding_rule.main.ip_address
}

output "azure_lb_ip" {
  description = "IP address of the Azure Load Balancer"
  value       = azurerm_public_ip.main.ip_address
}

output "aws_eks_endpoint" {
  description = "Endpoint of the AWS EKS cluster"
  value       = aws_eks_cluster.main.endpoint
}

output "gcp_gke_endpoint" {
  description = "Endpoint of the GCP GKE cluster"
  value       = google_container_cluster.main.endpoint
}

output "azure_aks_endpoint" {
  description = "Endpoint of the Azure AKS cluster"
  value       = azurerm_kubernetes_cluster.main.fqdn
}
```

### Multi-Cloud Application Deployment

```yaml
# kubernetes/multi-cloud-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-aws
  namespace: default
  labels:
    app: my-app
    cloud: aws
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      cloud: aws
  template:
    metadata:
      labels:
        app: my-app
        cloud: aws
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: CLOUD_PROVIDER
          value: "aws"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aws-db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: aws-redis-secret
              key: url
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
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-gcp
  namespace: default
  labels:
    app: my-app
    cloud: gcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      cloud: gcp
  template:
    metadata:
      labels:
        app: my-app
        cloud: gcp
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: CLOUD_PROVIDER
          value: "gcp"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gcp-db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: gcp-redis-secret
              key: url
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
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-azure
  namespace: default
  labels:
    app: my-app
    cloud: azure
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      cloud: azure
  template:
    metadata:
      labels:
        app: my-app
        cloud: azure
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: CLOUD_PROVIDER
          value: "azure"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: azure-db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: azure-redis-secret
              key: url
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
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
  namespace: default
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: default
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - my-app.example.com
    secretName: my-app-tls
  rules:
  - host: my-app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-app-service
            port:
              number: 80
```

### Multi-Cloud Monitoring

```yaml
# monitoring/multi-cloud-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'multi-cloud'
        environment: 'production'

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

      - job_name: 'kubernetes-aws'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - default
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_label_cloud]
            action: keep
            regex: aws

      - job_name: 'kubernetes-gcp'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - default
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_label_cloud]
            action: keep
            regex: gcp

      - job_name: 'kubernetes-azure'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - default
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_label_cloud]
            action: keep
            regex: azure

      - job_name: 'aws-cloudwatch'
        static_configs:
          - targets: ['aws-cloudwatch-exporter:9106']

      - job_name: 'gcp-stackdriver'
        static_configs:
          - targets: ['gcp-stackdriver-exporter:9107']

      - job_name: 'azure-monitor'
        static_configs:
          - targets: ['azure-monitor-exporter:9108']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aws-cloudwatch-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aws-cloudwatch-exporter
  template:
    metadata:
      labels:
        app: aws-cloudwatch-exporter
    spec:
      containers:
      - name: aws-cloudwatch-exporter
        image: prom/cloudwatch-exporter:latest
        ports:
        - containerPort: 9106
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-access-key
        - name: AWS_REGION
          value: "us-east-1"
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: aws-cloudwatch-config
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gcp-stackdriver-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gcp-stackdriver-exporter
  template:
    metadata:
      labels:
        app: gcp-stackdriver-exporter
    spec:
      containers:
      - name: gcp-stackdriver-exporter
        image: prom/stackdriver-exporter:latest
        ports:
        - containerPort: 9107
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/config/service-account.json"
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        secret:
          secretName: gcp-service-account
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: azure-monitor-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: azure-monitor-exporter
  template:
    metadata:
      labels:
        app: azure-monitor-exporter
    spec:
      containers:
      - name: azure-monitor-exporter
        image: prom/azure-exporter:latest
        ports:
        - containerPort: 9108
        env:
        - name: AZURE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: client-id
        - name: AZURE_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: client-secret
        - name: AZURE_TENANT_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: tenant-id
        - name: AZURE_SUBSCRIPTION_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: subscription-id
```

### Multi-Cloud Data Synchronization

```go
// data-sync.go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
    "cloud.google.com/go/storage"
    "github.com/Azure/azure-storage-blob-go/azblob"
)

type MultiCloudDataSync struct {
    awsS3Client   *s3.S3
    gcpClient     *storage.Client
    azureClient   azblob.ServiceURL
    logger        *log.Logger
}

func NewMultiCloudDataSync(awsRegion, gcpProjectID, azureAccountName, azureAccountKey string) (*MultiCloudDataSync, error) {
    // Initialize AWS S3 client
    awsSession, err := session.NewSession(&aws.Config{
        Region: aws.String(awsRegion),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create AWS session: %w", err)
    }
    awsS3Client := s3.New(awsSession)

    // Initialize GCP Storage client
    gcpClient, err := storage.NewClient(context.Background())
    if err != nil {
        return nil, fmt.Errorf("failed to create GCP client: %w", err)
    }

    // Initialize Azure Blob Storage client
    credential, err := azblob.NewSharedKeyCredential(azureAccountName, azureAccountKey)
    if err != nil {
        return nil, fmt.Errorf("failed to create Azure credential: %w", err)
    }
    pipeline := azblob.NewPipeline(credential, azblob.PipelineOptions{})
    azureClient := azblob.NewServiceURL(
        azblob.NewServiceURL(fmt.Sprintf("https://%s.blob.core.windows.net/", azureAccountName), pipeline),
        pipeline,
    )

    return &MultiCloudDataSync{
        awsS3Client: awsS3Client,
        gcpClient:   gcpClient,
        azureClient: azureClient,
        logger:      log.New(log.Writer(), "[MultiCloudDataSync] ", log.LstdFlags),
    }, nil
}

func (mcds *MultiCloudDataSync) SyncData(ctx context.Context, sourceCloud, targetCloud, bucket, key string) error {
    mcds.logger.Printf("Starting data sync from %s to %s for bucket: %s, key: %s", sourceCloud, targetCloud, bucket, key)

    // Download data from source cloud
    data, err := mcds.downloadFromSource(ctx, sourceCloud, bucket, key)
    if err != nil {
        return fmt.Errorf("failed to download from source: %w", err)
    }

    // Upload data to target cloud
    err = mcds.uploadToTarget(ctx, targetCloud, bucket, key, data)
    if err != nil {
        return fmt.Errorf("failed to upload to target: %w", err)
    }

    mcds.logger.Printf("Successfully synced data from %s to %s", sourceCloud, targetCloud)
    return nil
}

func (mcds *MultiCloudDataSync) downloadFromSource(ctx context.Context, cloud, bucket, key string) ([]byte, error) {
    switch cloud {
    case "aws":
        return mcds.downloadFromAWS(ctx, bucket, key)
    case "gcp":
        return mcds.downloadFromGCP(ctx, bucket, key)
    case "azure":
        return mcds.downloadFromAzure(ctx, bucket, key)
    default:
        return nil, fmt.Errorf("unsupported source cloud: %s", cloud)
    }
}

func (mcds *MultiCloudDataSync) uploadToTarget(ctx context.Context, cloud, bucket, key string, data []byte) error {
    switch cloud {
    case "aws":
        return mcds.uploadToAWS(ctx, bucket, key, data)
    case "gcp":
        return mcds.uploadToGCP(ctx, bucket, key, data)
    case "azure":
        return mcds.uploadToAzure(ctx, bucket, key, data)
    default:
        return fmt.Errorf("unsupported target cloud: %s", cloud)
    }
}

func (mcds *MultiCloudDataSync) downloadFromAWS(ctx context.Context, bucket, key string) ([]byte, error) {
    result, err := mcds.awsS3Client.GetObjectWithContext(ctx, &s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to get object from AWS S3: %w", err)
    }
    defer result.Body.Close()

    // Read the entire object
    data := make([]byte, 0)
    buffer := make([]byte, 1024)
    for {
        n, err := result.Body.Read(buffer)
        if n > 0 {
            data = append(data, buffer[:n]...)
        }
        if err != nil {
            break
        }
    }

    return data, nil
}

func (mcds *MultiCloudDataSync) uploadToAWS(ctx context.Context, bucket, key string, data []byte) error {
    _, err := mcds.awsS3Client.PutObjectWithContext(ctx, &s3.PutObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   aws.ReadSeekCloser(strings.NewReader(string(data))),
    })
    if err != nil {
        return fmt.Errorf("failed to put object to AWS S3: %w", err)
    }

    return nil
}

func (mcds *MultiCloudDataSync) downloadFromGCP(ctx context.Context, bucket, key string) ([]byte, error) {
    bucketHandle := mcds.gcpClient.Bucket(bucket)
    objectHandle := bucketHandle.Object(key)

    reader, err := objectHandle.NewReader(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create reader for GCP object: %w", err)
    }
    defer reader.Close()

    data, err := io.ReadAll(reader)
    if err != nil {
        return nil, fmt.Errorf("failed to read GCP object: %w", err)
    }

    return data, nil
}

func (mcds *MultiCloudDataSync) uploadToGCP(ctx context.Context, bucket, key string, data []byte) error {
    bucketHandle := mcds.gcpClient.Bucket(bucket)
    objectHandle := bucketHandle.Object(key)

    writer := objectHandle.NewWriter(ctx)
    writer.ContentType = "application/octet-stream"

    _, err := writer.Write(data)
    if err != nil {
        return fmt.Errorf("failed to write to GCP object: %w", err)
    }

    err = writer.Close()
    if err != nil {
        return fmt.Errorf("failed to close GCP object writer: %w", err)
    }

    return nil
}

func (mcds *MultiCloudDataSync) downloadFromAzure(ctx context.Context, bucket, key string) ([]byte, error) {
    containerURL := mcds.azureClient.NewContainerURL(bucket)
    blobURL := containerURL.NewBlobURL(key)

    downloadResponse, err := blobURL.Download(ctx, 0, azblob.CountToEnd, azblob.BlobAccessConditions{}, false)
    if err != nil {
        return nil, fmt.Errorf("failed to download from Azure blob: %w", err)
    }

    data, err := io.ReadAll(downloadResponse.Body(azblob.RetryReaderOptions{}))
    if err != nil {
        return nil, fmt.Errorf("failed to read Azure blob data: %w", err)
    }

    return data, nil
}

func (mcds *MultiCloudDataSync) uploadToAzure(ctx context.Context, bucket, key string, data []byte) error {
    containerURL := mcds.azureClient.NewContainerURL(bucket)
    blobURL := containerURL.NewBlockBlobURL(key)

    _, err := azblob.UploadBufferToBlockBlob(ctx, data, blobURL, azblob.UploadToBlockBlobOptions{
        BlockSize:   4 * 1024 * 1024, // 4MB blocks
        Parallelism: 16,
    })
    if err != nil {
        return fmt.Errorf("failed to upload to Azure blob: %w", err)
    }

    return nil
}

func (mcds *MultiCloudDataSync) StartSyncJob(ctx context.Context, config SyncConfig) error {
    ticker := time.NewTicker(config.Interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            for _, sync := range config.Syncs {
                err := mcds.SyncData(ctx, sync.SourceCloud, sync.TargetCloud, sync.Bucket, sync.Key)
                if err != nil {
                    mcds.logger.Printf("Sync failed: %v", err)
                    // Continue with other syncs
                }
            }
        }
    }
}

type SyncConfig struct {
    Interval time.Duration `json:"interval"`
    Syncs    []Sync        `json:"syncs"`
}

type Sync struct {
    SourceCloud string `json:"source_cloud"`
    TargetCloud string `json:"target_cloud"`
    Bucket      string `json:"bucket"`
    Key         string `json:"key"`
}
```

## 🚀 Best Practices

### 1. Cloud Provider Selection
```hcl
# Choose providers based on requirements
provider "aws" {
  region = "us-east-1"  # Best for global reach
}

provider "google" {
  project = "my-project"  # Best for AI/ML workloads
}

provider "azurerm" {
  features {}  # Best for enterprise integration
}
```

### 2. Cost Optimization
```yaml
# Use spot instances and preemptible VMs
spec:
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-preemptible: "true"
      tolerations:
      - key: cloud.google.com/gke-preemptible
        operator: Equal
        value: "true"
        effect: NoSchedule
```

### 3. Data Consistency
```go
// Implement eventual consistency
func (mcds *MultiCloudDataSync) SyncDataWithRetry(ctx context.Context, sourceCloud, targetCloud, bucket, key string) error {
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        err := mcds.SyncData(ctx, sourceCloud, targetCloud, bucket, key)
        if err == nil {
            return nil
        }
        
        if i < maxRetries-1 {
            time.Sleep(time.Duration(i+1) * time.Second)
        }
    }
    return fmt.Errorf("sync failed after %d retries", maxRetries)
}
```

## 🏢 Industry Insights

### Multi-Cloud Usage Patterns
- **Disaster Recovery**: Cross-cloud backup and failover
- **Cost Optimization**: Leverage best pricing from different providers
- **Compliance**: Meet regulatory requirements across regions
- **Performance**: Optimize for different geographic regions

### Enterprise Multi-Cloud Strategy
- **Vendor Independence**: Avoid single cloud provider dependency
- **Risk Mitigation**: Reduce impact of cloud provider outages
- **Global Reach**: Serve customers in different regions
- **Innovation**: Leverage unique capabilities of each provider

## 🎯 Interview Questions

### Basic Level
1. **What is multi-cloud?**
   - Use of multiple cloud providers
   - Vendor independence
   - Risk mitigation
   - Cost optimization

2. **What are the benefits of multi-cloud?**
   - Avoid vendor lock-in
   - Improve reliability
   - Optimize costs
   - Meet compliance requirements

3. **What are the challenges of multi-cloud?**
   - Increased complexity
   - Data consistency
   - Security management
   - Cost management

### Intermediate Level
4. **How do you implement multi-cloud?**
   - Infrastructure as Code
   - Cross-cloud networking
   - Data synchronization
   - Monitoring and logging

5. **How do you handle data consistency in multi-cloud?**
   - Eventual consistency
   - Conflict resolution
   - Data synchronization
   - Backup and recovery

6. **How do you manage costs in multi-cloud?**
   - Cost monitoring
   - Resource optimization
   - Spot instances
   - Reserved instances

### Advanced Level
7. **How do you implement multi-cloud security?**
   - Identity and access management
   - Network security
   - Data encryption
   - Compliance monitoring

8. **How do you handle multi-cloud networking?**
   - VPN connections
   - Peering connections
   - Load balancing
   - Traffic routing

9. **How do you implement multi-cloud monitoring?**
   - Centralized monitoring
   - Cross-cloud metrics
   - Alerting
   - Log aggregation

---

**Next**: [Hybrid Cloud](./HybridCloud.md) - On-premises and cloud integration, edge computing, data sovereignty
