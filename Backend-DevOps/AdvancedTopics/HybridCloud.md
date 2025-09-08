# 🌐 Hybrid Cloud: On-Premises and Cloud Integration

> **Master hybrid cloud strategies for seamless integration between on-premises and cloud environments**

## 📚 Concept

Hybrid cloud is a computing environment that combines on-premises infrastructure with public cloud services, allowing organizations to leverage the benefits of both environments. It provides flexibility, security, and cost optimization while maintaining control over sensitive data and applications.

### Key Features

- **Flexibility**: Choose the best environment for each workload
- **Security**: Keep sensitive data on-premises while using cloud for scalability
- **Cost Optimization**: Balance between capital and operational expenses
- **Compliance**: Meet regulatory requirements for data sovereignty
- **Disaster Recovery**: Leverage cloud for backup and failover
- **Gradual Migration**: Move workloads at your own pace

## 🏗️ Hybrid Cloud Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Hybrid Cloud Architecture               │
│  ┌─────────────────┐    ┌─────────────────┐           │
│  │  On-Premises    │    │   Public Cloud  │           │
│  │   Data Center   │    │   (AWS/GCP/Azure)│          │
│  │                 │    │                 │           │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │           │
│  │ │   Servers   │ │    │ │   EC2/GCE   │ │           │
│  │ │   Storage   │ │    │ │   S3/GCS    │ │           │
│  │ │   Network   │ │    │ │   VPC/VNet  │ │           │
│  │ └─────────────┘ │    │ └─────────────┘ │           │
│  └─────────────────┘    └─────────────────┘           │
│         │                       │                     │
│         ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Connectivity Layer                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   VPN       │  │   Direct    │  │   SD-WAN    │ │ │
│  │  │   Gateway   │  │   Connect   │  │   Solution  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │                       │                     │
│         ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Management Layer                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Hybrid    │  │   Cloud     │  │   Data      │ │ │
│  │  │   Manager   │  │   Broker    │  │   Sync      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │                       │                     │
│         ▼                       ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Security  │  │   Backup    │  │   Disaster  │     │
│  │   & IAM     │  │   & DR      │  │   Recovery  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Hybrid Cloud Infrastructure with Terraform

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
    vsphere = {
      source  = "hashicorp/vsphere"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "on_premises_cidr" {
  description = "CIDR block for on-premises network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "cloud_cidr" {
  description = "CIDR block for cloud network"
  type        = string
  default     = "172.16.0.0/16"
}

variable "vpn_shared_secret" {
  description = "Shared secret for VPN connection"
  type        = string
  sensitive   = true
}

# AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
      Type        = "hybrid-cloud"
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
    type        = "hybrid-cloud"
  }
}

# Azure Provider
provider "azurerm" {
  features {}

  default_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
    Type        = "hybrid-cloud"
  }
}

# vSphere Provider
provider "vsphere" {
  user                 = var.vsphere_user
  password             = var.vsphere_password
  vsphere_server       = var.vsphere_server
  allow_unverified_ssl = true
}

# AWS VPC for hybrid cloud
resource "aws_vpc" "hybrid" {
  cidr_block           = var.cloud_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-hybrid-vpc"
  }
}

resource "aws_internet_gateway" "hybrid" {
  vpc_id = aws_vpc.hybrid.id

  tags = {
    Name = "${var.project_name}-hybrid-igw"
  }
}

resource "aws_subnet" "hybrid_public" {
  count = length(var.aws_availability_zones)

  vpc_id                  = aws_vpc.hybrid.id
  cidr_block              = var.aws_public_subnet_cidrs[count.index]
  availability_zone       = var.aws_availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-hybrid-public-subnet-${count.index + 1}"
  }
}

resource "aws_subnet" "hybrid_private" {
  count = length(var.aws_availability_zones)

  vpc_id            = aws_vpc.hybrid.id
  cidr_block        = var.aws_private_subnet_cidrs[count.index]
  availability_zone = var.aws_availability_zones[count.index]

  tags = {
    Name = "${var.project_name}-hybrid-private-subnet-${count.index + 1}"
  }
}

# AWS VPN Gateway
resource "aws_vpn_gateway" "hybrid" {
  vpc_id = aws_vpc.hybrid.id

  tags = {
    Name = "${var.project_name}-hybrid-vpn-gateway"
  }
}

resource "aws_customer_gateway" "on_premises" {
  bgp_asn    = 65000
  ip_address = var.on_premises_vpn_ip
  type       = "ipsec.1"

  tags = {
    Name = "${var.project_name}-on-premises-gateway"
  }
}

resource "aws_vpn_connection" "on_premises" {
  vpn_gateway_id      = aws_vpn_gateway.hybrid.id
  customer_gateway_id = aws_customer_gateway.on_premises.id
  type                = "ipsec.1"
  static_routes_only  = true

  tags = {
    Name = "${var.project_name}-on-premises-vpn"
  }
}

# AWS Direct Connect (for production)
resource "aws_dx_connection" "on_premises" {
  count = var.enable_direct_connect ? 1 : 0

  name      = "${var.project_name}-direct-connect"
  bandwidth = "1Gbps"
  location  = var.direct_connect_location

  tags = {
    Name = "${var.project_name}-direct-connect"
  }
}

resource "aws_dx_private_virtual_interface" "on_premises" {
  count = var.enable_direct_connect ? 1 : 0

  connection_id    = aws_dx_connection.on_premises[0].id
  name             = "${var.project_name}-private-vif"
  vlan             = 100
  address_family   = "ipv4"
  bgp_asn          = 65000
  customer_address = "172.16.0.1/30"
  amazon_address   = "172.16.0.2/30"
  bgp_auth_key     = var.bgp_auth_key

  tags = {
    Name = "${var.project_name}-private-vif"
  }
}

# GCP VPC for hybrid cloud
resource "google_compute_network" "hybrid" {
  name                    = "${var.project_name}-hybrid-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "hybrid_public" {
  count = length(var.gcp_zones)

  name          = "${var.project_name}-hybrid-public-subnet-${count.index + 1}"
  ip_cidr_range = var.gcp_public_subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.hybrid.id
}

resource "google_compute_subnetwork" "hybrid_private" {
  count = length(var.gcp_zones)

  name          = "${var.project_name}-hybrid-private-subnet-${count.index + 1}"
  ip_cidr_range = var.gcp_private_subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.hybrid.id
}

# GCP VPN Gateway
resource "google_compute_vpn_gateway" "hybrid" {
  name    = "${var.project_name}-hybrid-vpn-gateway"
  network = google_compute_network.hybrid.id
  region  = var.gcp_region
}

resource "google_compute_address" "hybrid_vpn" {
  name   = "${var.project_name}-hybrid-vpn-ip"
  region = var.gcp_region
}

resource "google_compute_forwarding_rule" "hybrid_vpn" {
  name        = "${var.project_name}-hybrid-vpn-forwarding-rule"
  ip_protocol = "ESP"
  ip_address  = google_compute_address.hybrid_vpn.address
  target      = google_compute_vpn_gateway.hybrid.id
}

resource "google_compute_vpn_tunnel" "on_premises" {
  name          = "${var.project_name}-on-premises-tunnel"
  peer_ip       = var.on_premises_vpn_ip
  shared_secret = var.vpn_shared_secret

  target_vpn_gateway = google_compute_vpn_gateway.hybrid.id

  depends_on = [
    google_compute_forwarding_rule.hybrid_vpn,
  ]
}

# Azure VNet for hybrid cloud
resource "azurerm_virtual_network" "hybrid" {
  name                = "${var.project_name}-hybrid-vnet"
  address_space       = [var.cloud_cidr]
  location            = var.azure_location
  resource_group_name = azurerm_resource_group.hybrid.name
}

resource "azurerm_subnet" "hybrid_public" {
  count = length(var.azure_zones)

  name                 = "${var.project_name}-hybrid-public-subnet-${count.index + 1}"
  resource_group_name  = azurerm_resource_group.hybrid.name
  virtual_network_name = azurerm_virtual_network.hybrid.name
  address_prefixes     = [var.azure_public_subnet_cidrs[count.index]]
}

resource "azurerm_subnet" "hybrid_private" {
  count = length(var.azure_zones)

  name                 = "${var.project_name}-hybrid-private-subnet-${count.index + 1}"
  resource_group_name  = azurerm_resource_group.hybrid.name
  virtual_network_name = azurerm_virtual_network.hybrid.name
  address_prefixes     = [var.azure_private_subnet_cidrs[count.index]]
}

resource "azurerm_resource_group" "hybrid" {
  name     = "${var.project_name}-hybrid-rg"
  location = var.azure_location
}

# Azure VPN Gateway
resource "azurerm_public_ip" "hybrid_vpn" {
  name                = "${var.project_name}-hybrid-vpn-pip"
  location            = azurerm_resource_group.hybrid.location
  resource_group_name = azurerm_resource_group.hybrid.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_virtual_network_gateway" "hybrid" {
  name                = "${var.project_name}-hybrid-vpn-gateway"
  location            = azurerm_resource_group.hybrid.location
  resource_group_name = azurerm_resource_group.hybrid.name

  type     = "Vpn"
  vpn_type = "RouteBased"

  active_active = false
  enable_bgp    = false
  sku           = "VpnGw1"

  ip_configuration {
    name                          = "vnetGatewayConfig"
    public_ip_address_id          = azurerm_public_ip.hybrid_vpn.id
    private_ip_address_allocation = "Dynamic"
    subnet_id                     = azurerm_subnet.hybrid_public[0].id
  }
}

# On-premises vSphere Infrastructure
resource "vsphere_datacenter" "on_premises" {
  name = "${var.project_name}-on-premises-dc"
}

resource "vsphere_compute_cluster" "on_premises" {
  name            = "${var.project_name}-on-premises-cluster"
  datacenter_id   = vsphere_datacenter.on_premises.id
  host_system_ids = var.vsphere_host_ids

  ha_enabled = true
  ha_admission_control_policy = "failoverHosts"
  ha_admission_control_failover_host_system_ids = var.vsphere_failover_host_ids

  drs_enabled = true
  drs_automation_level = "fullyAutomated"
}

resource "vsphere_datastore_cluster" "on_premises" {
  name          = "${var.project_name}-on-premises-datastore-cluster"
  datacenter_id = vsphere_datacenter.on_premises.id

  sdrs_enabled = true
  sdrs_automation_level = "fullyAutomated"
}

resource "vsphere_network" "on_premises" {
  name          = "${var.project_name}-on-premises-network"
  datacenter_id = vsphere_datacenter.on_premises.id
  type          = "DistributedVirtualPortgroup"
  distributed_virtual_switch_uuid = var.vsphere_dvs_uuid
}

resource "vsphere_virtual_machine" "on_premises" {
  count = var.on_premises_vm_count

  name             = "${var.project_name}-on-premises-vm-${count.index + 1}"
  resource_pool_id = vsphere_compute_cluster.on_premises.resource_pool_id
  datastore_id     = vsphere_datastore_cluster.on_premises.id

  num_cpus = 2
  memory   = 4096
  guest_id = "ubuntu64Guest"

  network_interface {
    network_id = vsphere_network.on_premises.id
  }

  disk {
    label = "disk0"
    size  = 50
  }

  clone {
    template_uuid = var.vsphere_template_uuid
  }

  vapp {
    properties = {
      "hostname" = "${var.project_name}-on-premises-vm-${count.index + 1}"
      "ip-address" = var.on_premises_vm_ips[count.index]
    }
  }
}

# Hybrid Cloud Load Balancer
resource "aws_lb" "hybrid" {
  name               = "${var.project_name}-hybrid-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.hybrid_web.id]
  subnets            = aws_subnet.hybrid_public[*].id

  enable_deletion_protection = false

  tags = {
    Name = "${var.project_name}-hybrid-alb"
  }
}

resource "aws_lb_target_group" "hybrid" {
  name     = "${var.project_name}-hybrid-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.hybrid.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = {
    Name = "${var.project_name}-hybrid-tg"
  }
}

resource "aws_lb_listener" "hybrid" {
  load_balancer_arn = aws_lb.hybrid.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.hybrid.arn
  }
}

# Hybrid Cloud Security Groups
resource "aws_security_group" "hybrid_web" {
  name_prefix = "${var.project_name}-hybrid-web-"
  vpc_id      = aws_vpc.hybrid.id

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

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.on_premises_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-hybrid-web-sg"
  }
}

resource "aws_security_group" "hybrid_database" {
  name_prefix = "${var.project_name}-hybrid-db-"
  vpc_id      = aws_vpc.hybrid.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.hybrid_web.id]
  }

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.on_premises_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-hybrid-db-sg"
  }
}

# Hybrid Cloud Database
resource "aws_db_instance" "hybrid" {
  identifier = "${var.project_name}-hybrid-rds"

  engine         = "postgres"
  engine_version = "14.7"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "hybridapp"
  username = "postgres"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.hybrid_database.id]
  db_subnet_group_name   = aws_db_subnet_group.hybrid.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  tags = {
    Name = "${var.project_name}-hybrid-rds"
  }
}

resource "aws_db_subnet_group" "hybrid" {
  name       = "${var.project_name}-hybrid-db-subnet-group"
  subnet_ids = aws_subnet.hybrid_private[*].id

  tags = {
    Name = "${var.project_name}-hybrid-db-subnet-group"
  }
}

# Hybrid Cloud Monitoring
resource "aws_cloudwatch_log_group" "hybrid" {
  name              = "/aws/hybrid-cloud/${var.project_name}"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-hybrid-logs"
  }
}

resource "aws_cloudwatch_dashboard" "hybrid" {
  dashboard_name = "${var.project_name}-hybrid-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.hybrid.arn_suffix],
            [".", "TargetResponseTime", ".", "."],
            [".", "HTTPCode_Target_2XX_Count", ".", "."],
            [".", "HTTPCode_Target_4XX_Count", ".", "."],
            [".", "HTTPCode_Target_5XX_Count", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Hybrid Cloud Load Balancer Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.hybrid.id],
            [".", "DatabaseConnections", ".", "."],
            [".", "FreeableMemory", ".", "."],
            [".", "FreeStorageSpace", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Hybrid Cloud Database Metrics"
          period  = 300
        }
      }
    ]
  })
}

# Outputs
output "aws_vpc_id" {
  description = "ID of the AWS VPC"
  value       = aws_vpc.hybrid.id
}

output "gcp_vpc_id" {
  description = "ID of the GCP VPC"
  value       = google_compute_network.hybrid.id
}

output "azure_vnet_id" {
  description = "ID of the Azure VNet"
  value       = azurerm_virtual_network.hybrid.id
}

output "aws_vpn_connection_id" {
  description = "ID of the AWS VPN connection"
  value       = aws_vpn_connection.on_premises.id
}

output "gcp_vpn_tunnel_id" {
  description = "ID of the GCP VPN tunnel"
  value       = google_compute_vpn_tunnel.on_premises.id
}

output "azure_vpn_gateway_id" {
  description = "ID of the Azure VPN gateway"
  value       = azurerm_virtual_network_gateway.hybrid.id
}

output "aws_alb_dns" {
  description = "DNS name of the AWS ALB"
  value       = aws_lb.hybrid.dns_name
}

output "on_premises_vm_ips" {
  description = "IP addresses of on-premises VMs"
  value       = vsphere_virtual_machine.on_premises[*].default_ip_address
}
```

### Hybrid Cloud Application Deployment

```yaml
# kubernetes/hybrid-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-app-cloud
  namespace: default
  labels:
    app: hybrid-app
    location: cloud
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hybrid-app
      location: cloud
  template:
    metadata:
      labels:
        app: hybrid-app
        location: cloud
    spec:
      containers:
        - name: hybrid-app
          image: hybrid-app:latest
          ports:
            - containerPort: 8080
          env:
            - name: LOCATION
              value: "cloud"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: hybrid-db-secret
                  key: url
            - name: ON_PREMISES_API_URL
              value: "http://on-premises-api.internal:8080"
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
  name: hybrid-app-on-premises
  namespace: default
  labels:
    app: hybrid-app
    location: on-premises
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hybrid-app
      location: on-premises
  template:
    metadata:
      labels:
        app: hybrid-app
        location: on-premises
    spec:
      nodeSelector:
        kubernetes.io/hostname: on-premises-node-1
      containers:
        - name: hybrid-app
          image: hybrid-app:latest
          ports:
            - containerPort: 8080
          env:
            - name: LOCATION
              value: "on-premises"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: on-premises-db-secret
                  key: url
            - name: CLOUD_API_URL
              value: "http://cloud-api.external:8080"
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
  name: hybrid-app-service
  namespace: default
spec:
  selector:
    app: hybrid-app
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hybrid-app-ingress
  namespace: default
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$binary_remote_addr"
spec:
  tls:
    - hosts:
        - hybrid-app.example.com
      secretName: hybrid-app-tls
  rules:
    - host: hybrid-app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: hybrid-app-service
                port:
                  number: 80
```

### Hybrid Cloud Data Synchronization

```go
// hybrid-sync.go
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

type HybridCloudSync struct {
    cloudStorage    *CloudStorage
    onPremisesStorage *OnPremisesStorage
    logger          *log.Logger
}

type CloudStorage struct {
    awsS3Client   *s3.S3
    gcpClient     *storage.Client
    azureClient   azblob.ServiceURL
}

type OnPremisesStorage struct {
    nfsPath       string
    localPath     string
    sftpClient    *SFTPClient
}

func NewHybridCloudSync(cloudStorage *CloudStorage, onPremisesStorage *OnPremisesStorage, logger *log.Logger) *HybridCloudSync {
    return &HybridCloudSync{
        cloudStorage:      cloudStorage,
        onPremisesStorage: onPremisesStorage,
        logger:            logger,
    }
}

func (hcs *HybridCloudSync) SyncData(ctx context.Context, source, target, path string) error {
    hcs.logger.Printf("Starting hybrid sync from %s to %s for path: %s", source, target, path)

    // Download data from source
    data, err := hcs.downloadFromSource(ctx, source, path)
    if err != nil {
        return fmt.Errorf("failed to download from source: %w", err)
    }

    // Upload data to target
    err = hcs.uploadToTarget(ctx, target, path, data)
    if err != nil {
        return fmt.Errorf("failed to upload to target: %w", err)
    }

    hcs.logger.Printf("Successfully synced data from %s to %s", source, target)
    return nil
}

func (hcs *HybridCloudSync) downloadFromSource(ctx context.Context, source, path string) ([]byte, error) {
    switch source {
    case "aws":
        return hcs.downloadFromAWS(ctx, path)
    case "gcp":
        return hcs.downloadFromGCP(ctx, path)
    case "azure":
        return hcs.downloadFromAzure(ctx, path)
    case "on-premises":
        return hcs.downloadFromOnPremises(ctx, path)
    default:
        return nil, fmt.Errorf("unsupported source: %s", source)
    }
}

func (hcs *HybridCloudSync) uploadToTarget(ctx context.Context, target, path string, data []byte) error {
    switch target {
    case "aws":
        return hcs.uploadToAWS(ctx, path, data)
    case "gcp":
        return hcs.uploadToGCP(ctx, path, data)
    case "azure":
        return hcs.uploadToAzure(ctx, path, data)
    case "on-premises":
        return hcs.uploadToOnPremises(ctx, path, data)
    default:
        return fmt.Errorf("unsupported target: %s", target)
    }
}

func (hcs *HybridCloudSync) downloadFromAWS(ctx context.Context, path string) ([]byte, error) {
    result, err := hcs.cloudStorage.awsS3Client.GetObjectWithContext(ctx, &s3.GetObjectInput{
        Bucket: aws.String("hybrid-bucket"),
        Key:    aws.String(path),
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

func (hcs *HybridCloudSync) uploadToAWS(ctx context.Context, path string, data []byte) error {
    _, err := hcs.cloudStorage.awsS3Client.PutObjectWithContext(ctx, &s3.PutObjectInput{
        Bucket: aws.String("hybrid-bucket"),
        Key:    aws.String(path),
        Body:   aws.ReadSeekCloser(strings.NewReader(string(data))),
    })
    if err != nil {
        return fmt.Errorf("failed to put object to AWS S3: %w", err)
    }

    return nil
}

func (hcs *HybridCloudSync) downloadFromGCP(ctx context.Context, path string) ([]byte, error) {
    bucketHandle := hcs.cloudStorage.gcpClient.Bucket("hybrid-bucket")
    objectHandle := bucketHandle.Object(path)

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

func (hcs *HybridCloudSync) uploadToGCP(ctx context.Context, path string, data []byte) error {
    bucketHandle := hcs.cloudStorage.gcpClient.Bucket("hybrid-bucket")
    objectHandle := bucketHandle.Object(path)

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

func (hcs *HybridCloudSync) downloadFromAzure(ctx context.Context, path string) ([]byte, error) {
    containerURL := hcs.cloudStorage.azureClient.NewContainerURL("hybrid-container")
    blobURL := containerURL.NewBlobURL(path)

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

func (hcs *HybridCloudSync) uploadToAzure(ctx context.Context, path string, data []byte) error {
    containerURL := hcs.cloudStorage.azureClient.NewContainerURL("hybrid-container")
    blobURL := containerURL.NewBlockBlobURL(path)

    _, err := azblob.UploadBufferToBlockBlob(ctx, data, blobURL, azblob.UploadToBlockBlobOptions{
        BlockSize:   4 * 1024 * 1024, // 4MB blocks
        Parallelism: 16,
    })
    if err != nil {
        return fmt.Errorf("failed to upload to Azure blob: %w", err)
    }

    return nil
}

func (hcs *HybridCloudSync) downloadFromOnPremises(ctx context.Context, path string) ([]byte, error) {
    // Download from on-premises NFS or SFTP
    fullPath := filepath.Join(hcs.onPremisesStorage.nfsPath, path)

    data, err := os.ReadFile(fullPath)
    if err != nil {
        return nil, fmt.Errorf("failed to read from on-premises storage: %w", err)
    }

    return data, nil
}

func (hcs *HybridCloudSync) uploadToOnPremises(ctx context.Context, path string, data []byte) error {
    // Upload to on-premises NFS or SFTP
    fullPath := filepath.Join(hcs.onPremisesStorage.nfsPath, path)

    // Create directory if it doesn't exist
    dir := filepath.Dir(fullPath)
    if err := os.MkdirAll(dir, 0755); err != nil {
        return fmt.Errorf("failed to create directory: %w", err)
    }

    err := os.WriteFile(fullPath, data, 0644)
    if err != nil {
        return fmt.Errorf("failed to write to on-premises storage: %w", err)
    }

    return nil
}

func (hcs *HybridCloudSync) StartSyncJob(ctx context.Context, config HybridSyncConfig) error {
    ticker := time.NewTicker(config.Interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            for _, sync := range config.Syncs {
                err := hcs.SyncData(ctx, sync.Source, sync.Target, sync.Path)
                if err != nil {
                    hcs.logger.Printf("Sync failed: %v", err)
                    // Continue with other syncs
                }
            }
        }
    }
}

type HybridSyncConfig struct {
    Interval time.Duration `json:"interval"`
    Syncs    []HybridSync  `json:"syncs"`
}

type HybridSync struct {
    Source string `json:"source"`
    Target string `json:"target"`
    Path   string `json:"path"`
}
```

## 🚀 Best Practices

### 1. Network Connectivity

```hcl
# Use multiple connectivity options
resource "aws_vpn_connection" "on_premises" {
  vpn_gateway_id      = aws_vpn_gateway.hybrid.id
  customer_gateway_id = aws_customer_gateway.on_premises.id
  type                = "ipsec.1"
  static_routes_only  = true
}

resource "aws_dx_connection" "on_premises" {
  name      = "${var.project_name}-direct-connect"
  bandwidth = "1Gbps"
  location  = var.direct_connect_location
}
```

### 2. Data Sovereignty

```yaml
# Keep sensitive data on-premises
spec:
  template:
    spec:
      nodeSelector:
        location: on-premises
      containers:
        - name: sensitive-app
          env:
            - name: DATA_LOCATION
              value: "on-premises"
```

### 3. Cost Optimization

```go
// Use cloud for burst capacity
func (hcs *HybridCloudSync) ScaleBasedOnDemand(ctx context.Context) error {
    // Monitor on-premises capacity
    // Scale to cloud when needed
    // Scale down when demand decreases
    return nil
}
```

## 🏢 Industry Insights

### Hybrid Cloud Usage Patterns

- **Data Sovereignty**: Keep sensitive data on-premises
- **Compliance**: Meet regulatory requirements
- **Cost Optimization**: Balance capital and operational expenses
- **Disaster Recovery**: Use cloud for backup and failover

### Enterprise Hybrid Cloud Strategy

- **Gradual Migration**: Move workloads at your own pace
- **Workload Placement**: Choose the best environment for each workload
- **Unified Management**: Single pane of glass for all environments
- **Security**: Consistent security policies across environments

## 🎯 Interview Questions

### Basic Level

1. **What is hybrid cloud?**

   - Combination of on-premises and cloud
   - Flexibility and control
   - Cost optimization
   - Compliance and security

2. **What are the benefits of hybrid cloud?**

   - Flexibility
   - Security
   - Cost optimization
   - Compliance

3. **What are the challenges of hybrid cloud?**
   - Complexity
   - Network connectivity
   - Data consistency
   - Management overhead

### Intermediate Level

4. **How do you implement hybrid cloud?**

   - Network connectivity
   - Data synchronization
   - Workload placement
   - Management tools

5. **How do you handle data consistency in hybrid cloud?**

   - Data synchronization
   - Conflict resolution
   - Backup and recovery
   - Monitoring

6. **How do you manage costs in hybrid cloud?**
   - Workload placement
   - Resource optimization
   - Cost monitoring
   - Right-sizing

### Advanced Level

7. **How do you implement hybrid cloud security?**

   - Network security
   - Data encryption
   - Identity management
   - Compliance

8. **How do you handle hybrid cloud networking?**

   - VPN connections
   - Direct connect
   - Load balancing
   - Traffic routing

9. **How do you implement hybrid cloud monitoring?**
   - Unified monitoring
   - Cross-environment metrics
   - Alerting
   - Log aggregation

---

**Next**: [Edge Computing](./EdgeComputing.md) - Edge infrastructure, IoT, real-time processing, latency optimization
