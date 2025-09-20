# 🖥️ GCP Compute Engine: Virtual Machines, Instance Groups, and Auto-Scaling

> **Master Google Cloud Compute Engine for scalable virtual machine deployments**

## 📚 Concept

**Detailed Explanation:**
Google Cloud Compute Engine is a robust Infrastructure as a Service (IaaS) platform that provides scalable, high-performance virtual machines running in Google's global infrastructure. It offers enterprise-grade compute resources with advanced features for building, deploying, and managing applications at scale.

**Core Philosophy:**

- **Scalability**: Automatically scale resources based on demand
- **Global Infrastructure**: Leverage Google's worldwide network and data centers
- **Cost Optimization**: Pay only for what you use with flexible pricing models
- **Security**: Built-in security features and compliance capabilities
- **Performance**: High-performance compute with custom machine types
- **Reliability**: 99.95% uptime SLA with automatic failover

**Why Compute Engine Matters:**

- **Enterprise Scale**: Handle workloads from small applications to large-scale enterprise systems
- **Global Reach**: Deploy applications across multiple regions and zones
- **Cost Efficiency**: Optimize costs with preemptible instances and committed use discounts
- **Integration**: Seamless integration with other Google Cloud services
- **Flexibility**: Choose from predefined or custom machine types
- **Performance**: High-performance networking and storage options

**Key Features:**

**1. Virtual Machines:**

- **On-Demand Capacity**: Provision virtual machines instantly when needed
- **Custom Machine Types**: Create custom machine configurations for specific workloads
- **Persistent Disks**: High-performance, durable block storage
- **Live Migration**: Automatic migration without downtime during maintenance
- **Sustained Use Discounts**: Automatic discounts for long-running instances

**2. Machine Types:**

- **General Purpose**: Balanced CPU and memory for most workloads
- **Compute Optimized**: High CPU-to-memory ratio for compute-intensive tasks
- **Memory Optimized**: High memory-to-CPU ratio for memory-intensive applications
- **Storage Optimized**: High disk throughput for I/O-intensive workloads
- **GPU Accelerated**: Access to NVIDIA GPUs for machine learning and graphics

**3. Instance Groups:**

- **Managed Instance Groups**: Automatically managed groups of identical instances
- **Unmanaged Instance Groups**: Manually managed groups of diverse instances
- **Auto Healing**: Automatic replacement of unhealthy instances
- **Rolling Updates**: Zero-downtime updates across instance groups
- **Load Balancing**: Built-in load balancing across group members

**4. Auto Scaling:**

- **Automatic Scaling**: Scale based on CPU utilization, load balancing capacity, or custom metrics
- **Predictive Scaling**: Use machine learning to predict scaling needs
- **Cooldown Periods**: Prevent rapid scaling oscillations
- **Scaling Policies**: Fine-tune scaling behavior for different scenarios
- **Cost Optimization**: Scale down during low-demand periods

**5. Global Load Balancing:**

- **Global Anycast IP**: Single IP address that routes to the nearest healthy instance
- **Health Checks**: Continuous monitoring of instance health
- **Traffic Distribution**: Intelligent routing based on proximity and capacity
- **SSL Termination**: Built-in SSL certificate management
- **DDoS Protection**: Protection against distributed denial-of-service attacks

**6. Preemptible Instances:**

- **Cost Savings**: Up to 80% discount compared to regular instances
- **Interruptible**: Can be terminated with 30 seconds notice
- **Batch Processing**: Ideal for fault-tolerant, batch-oriented workloads
- **Spot Pricing**: Dynamic pricing based on supply and demand
- **Workload Distribution**: Distribute work across multiple preemptible instances

**Discussion Questions & Answers:**

**Q1: How do you design a highly available and scalable architecture using Compute Engine?**

**Answer:** High availability and scalability design:

- **Multi-Zone Deployment**: Deploy instances across multiple availability zones
- **Load Balancing**: Use global load balancer for traffic distribution
- **Auto Scaling**: Implement horizontal auto scaling based on metrics
- **Health Checks**: Configure comprehensive health monitoring
- **Database Replication**: Use managed database services with replication
- **CDN Integration**: Use Cloud CDN for static content delivery
- **Monitoring**: Implement comprehensive monitoring and alerting
- **Disaster Recovery**: Design cross-region backup and failover strategies

**Q2: What are the key considerations for optimizing costs in Compute Engine?**

**Answer:** Cost optimization strategies:

- **Right-Sizing**: Choose appropriate machine types for workloads
- **Preemptible Instances**: Use for fault-tolerant, batch workloads
- **Committed Use Discounts**: Commit to 1-3 year terms for predictable workloads
- **Sustained Use Discounts**: Automatic discounts for long-running instances
- **Auto Scaling**: Scale down during low-demand periods
- **Reserved Instances**: Reserve capacity for predictable workloads
- **Spot Instances**: Use spot pricing for flexible workloads
- **Resource Monitoring**: Monitor and optimize resource utilization

**Q3: How do you implement security best practices for Compute Engine instances?**

**Answer:** Security implementation:

- **Network Security**: Use VPC with proper firewall rules and network segmentation
- **Identity and Access Management**: Implement proper IAM roles and permissions
- **OS Security**: Keep operating systems updated and use security-hardened images
- **Encryption**: Enable encryption at rest and in transit
- **Access Control**: Use OS Login and SSH key management
- **Monitoring**: Implement security monitoring and logging
- **Compliance**: Follow security compliance frameworks and best practices
- **Incident Response**: Have security incident response procedures in place

## 🏗️ Compute Engine Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud Region                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Availability │  │ Availability │  │ Availability │     │
│  │    Zone A    │  │    Zone B    │  │    Zone C    │     │
│  │             │  │             │  │             │     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │
│  │ │ Compute │ │  │ │ Compute │ │  │ │ Compute │ │     │
│  │ │Instance │ │  │ │Instance │ │  │ │Instance │ │     │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Global Load Balancer                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Health    │  │   Traffic   │  │   SSL       │ │ │
│  │  │   Checks    │  │   Policy    │  │   Term.     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Compute Engine with Deployment Manager

```yaml
# compute-engine.yaml
imports:
  - path: compute-instance.jinja
  - path: network.jinja

resources:
  # VPC Network
  - name: production-vpc
    type: network.jinja
    properties:
      name: production-vpc
      autoCreateSubnetworks: false
      routingConfig:
        routingMode: REGIONAL

  # Subnets
  - name: production-subnet-1
    type: compute.v1.subnetwork
    properties:
      name: production-subnet-1
      region: us-central1
      network: $(ref.production-vpc.selfLink)
      ipCidrRange: 10.0.1.0/24
      privateIpGoogleAccess: false

  - name: production-subnet-2
    type: compute.v1.subnetwork
    properties:
      name: production-subnet-2
      region: us-central1
      network: $(ref.production-vpc.selfLink)
      ipCidrRange: 10.0.2.0/24
      privateIpGoogleAccess: true

  # Firewall Rules
  - name: production-firewall-rule
    type: compute.v1.firewall
    properties:
      name: production-firewall-rule
      network: $(ref.production-vpc.selfLink)
      sourceRanges: ["0.0.0.0/0"]
      allowed:
        - IPProtocol: TCP
          ports: ["80", "443", "22"]
      targetTags: ["web-server"]

  # Instance Template
  - name: production-instance-template
    type: compute.v1.instanceTemplate
    properties:
      name: production-instance-template
      properties:
        machineType: f1-micro
        disks:
          - boot: true
            autoDelete: true
            initializeParams:
              sourceImage: projects/debian-cloud/global/images/family/debian-11
              diskSizeGb: 10
        networkInterfaces:
          - network: $(ref.production-vpc.selfLink)
            subnetwork: $(ref.production-subnet-1.selfLink)
            accessConfigs:
              - name: External NAT
                type: ONE_TO_ONE_NAT
        tags:
          items: ["web-server"]
        metadata:
          items:
            - key: startup-script
              value: |
                #!/bin/bash
                apt-get update
                apt-get install -y apache2
                systemctl start apache2
                systemctl enable apache2
                echo "<h1>Hello from GCP!</h1>" > /var/www/html/index.html
                echo "OK" > /var/www/html/health

  # Instance Group Manager
  - name: production-instance-group-manager
    type: compute.v1.instanceGroupManager
    properties:
      name: production-instance-group-manager
      zone: us-central1-a
      instanceTemplate: $(ref.production-instance-template.selfLink)
      targetSize: 2
      baseInstanceName: production-instance
      autoHealingPolicies:
        - healthCheck: $(ref.production-health-check.selfLink)
          initialDelaySec: 300

  # Health Check
  - name: production-health-check
    type: compute.v1.httpHealthCheck
    properties:
      name: production-health-check
      requestPath: /health
      port: 80
      checkIntervalSec: 30
      timeoutSec: 5
      healthyThreshold: 2
      unhealthyThreshold: 3

  # Backend Service
  - name: production-backend-service
    type: compute.v1.backendService
    properties:
      name: production-backend-service
      protocol: HTTP
      port: 80
      timeoutSec: 30
      healthChecks:
        - $(ref.production-health-check.selfLink)
      backends:
        - group: $(ref.production-instance-group-manager.selfLink)
          balancingMode: UTILIZATION
          maxUtilization: 0.8

  # URL Map (Load Balancer)
  - name: production-url-map
    type: compute.v1.urlMap
    properties:
      name: production-url-map
      defaultService: $(ref.production-backend-service.selfLink)

  # Target HTTP Proxy
  - name: production-target-http-proxy
    type: compute.v1.targetHttpProxy
    properties:
      name: production-target-http-proxy
      urlMap: $(ref.production-url-map.selfLink)

  # Global Forwarding Rule
  - name: production-global-forwarding-rule
    type: compute.v1.globalForwardingRule
    properties:
      name: production-global-forwarding-rule
      target: $(ref.production-target-http-proxy.selfLink)
      portRange: 80
      IPProtocol: TCP

  # Auto Scaling Policy
  - name: production-autoscaler
    type: compute.v1.autoscaler
    properties:
      name: production-autoscaler
      zone: us-central1-a
      target: $(ref.production-instance-group-manager.selfLink)
      autoscalingPolicy:
        minNumReplicas: 1
        maxNumReplicas: 10
        coolDownPeriodSec: 60
        cpuUtilization:
          utilizationTarget: 0.6
        loadBalancingUtilization:
          utilizationTarget: 0.8

  # Preemptible Instance Template
  - name: production-preemptible-template
    type: compute.v1.instanceTemplate
    properties:
      name: production-preemptible-template
      properties:
        machineType: f1-micro
        scheduling:
          preemptible: true
        disks:
          - boot: true
            autoDelete: true
            initializeParams:
              sourceImage: projects/debian-cloud/global/images/family/debian-11
              diskSizeGb: 10
        networkInterfaces:
          - network: $(ref.production-vpc.selfLink)
            subnetwork: $(ref.production-subnet-1.selfLink)
            accessConfigs:
              - name: External NAT
                type: ONE_TO_ONE_NAT
        tags:
          items: ["batch-worker"]
        metadata:
          items:
            - key: startup-script
              value: |
                #!/bin/bash
                apt-get update
                apt-get install -y python3 python3-pip
                pip3 install requests
                # Add your batch processing script here

  # Preemptible Instance Group
  - name: production-preemptible-group
    type: compute.v1.instanceGroupManager
    properties:
      name: production-preemptible-group
      zone: us-central1-a
      instanceTemplate: $(ref.production-preemptible-template.selfLink)
      targetSize: 5
      baseInstanceName: production-preemptible
```

### Compute Engine with Terraform

```hcl
# compute-engine.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.environment}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

# Subnets
resource "google_compute_subnetwork" "public" {
  count = 2

  name          = "${var.environment}-public-subnet-${count.index + 1}"
  ip_cidr_range = "10.0.${count.index + 1}.0/24"
  region        = var.region
  network       = google_compute_network.main.id

  private_ip_google_access = false
}

resource "google_compute_subnetwork" "private" {
  count = 2

  name          = "${var.environment}-private-subnet-${count.index + 1}"
  ip_cidr_range = "10.0.${count.index + 10}.0/24"
  region        = var.region
  network       = google_compute_network.main.id

  private_ip_google_access = true
}

# Firewall Rules
resource "google_compute_firewall" "web" {
  name    = "${var.environment}-web-firewall"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["web-server"]
}

resource "google_compute_firewall" "ssh" {
  name    = "${var.environment}-ssh-firewall"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

# Instance Template
resource "google_compute_instance_template" "web" {
  name_prefix  = "${var.environment}-web-template-"
  description  = "Template for web servers"
  machine_type = "f1-micro"

  disk {
    source_image = "debian-cloud/debian-11"
    auto_delete  = true
    boot         = true
    disk_size_gb = 10
  }

  network_interface {
    network    = google_compute_network.main.id
    subnetwork = google_compute_subnetwork.public[0].id

    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    startup-script = templatefile("${path.module}/startup-script.sh", {
      environment = var.environment
    })
  }

  tags = ["web-server", "ssh"]

  lifecycle {
    create_before_destroy = true
  }
}

# Instance Group Manager
resource "google_compute_instance_group_manager" "web" {
  name = "${var.environment}-web-igm"

  base_instance_name = "${var.environment}-web"
  zone               = "${var.region}-a"

  version {
    instance_template = google_compute_instance_template.web.id
  }

  target_size = 2

  auto_healing_policies {
    health_check      = google_compute_health_check.web.id
    initial_delay_sec = 300
  }
}

# Health Check
resource "google_compute_health_check" "web" {
  name = "${var.environment}-web-health-check"

  http_health_check {
    request_path = "/health"
    port         = 80
  }

  check_interval_sec  = 30
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

# Backend Service
resource "google_compute_backend_service" "web" {
  name        = "${var.environment}-web-backend"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 30

  backend {
    group = google_compute_instance_group_manager.web.instance_group
  }

  health_checks = [google_compute_health_check.web.id]
}

# URL Map
resource "google_compute_url_map" "web" {
  name            = "${var.environment}-web-url-map"
  default_service = google_compute_backend_service.web.id
}

# Target HTTP Proxy
resource "google_compute_target_http_proxy" "web" {
  name    = "${var.environment}-web-proxy"
  url_map = google_compute_url_map.web.id
}

# Global Forwarding Rule
resource "google_compute_global_forwarding_rule" "web" {
  name       = "${var.environment}-web-forwarding-rule"
  target     = google_compute_target_http_proxy.web.id
  port_range = "80"
  ip_protocol = "TCP"
}

# Auto Scaler
resource "google_compute_autoscaler" "web" {
  name   = "${var.environment}-web-autoscaler"
  zone   = "${var.region}-a"
  target = google_compute_instance_group_manager.web.id

  autoscaling_policy {
    max_replicas    = 10
    min_replicas    = 1
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }

    load_balancing_utilization {
      target = 0.8
    }
  }
}

# Preemptible Instance Template
resource "google_compute_instance_template" "batch" {
  name_prefix  = "${var.environment}-batch-template-"
  description  = "Template for batch processing"
  machine_type = "f1-micro"

  disk {
    source_image = "debian-cloud/debian-11"
    auto_delete  = true
    boot         = true
    disk_size_gb = 10
  }

  network_interface {
    network    = google_compute_network.main.id
    subnetwork = google_compute_subnetwork.private[0].id
  }

  scheduling {
    preemptible = true
  }

  metadata = {
    startup-script = templatefile("${path.module}/batch-script.sh", {
      environment = var.environment
    })
  }

  tags = ["batch-worker"]

  lifecycle {
    create_before_destroy = true
  }
}

# Preemptible Instance Group
resource "google_compute_instance_group_manager" "batch" {
  name = "${var.environment}-batch-igm"

  base_instance_name = "${var.environment}-batch"
  zone               = "${var.region}-a"

  version {
    instance_template = google_compute_instance_template.batch.id
  }

  target_size = 5
}

# Startup Script
resource "local_file" "startup_script" {
  content = templatefile("${path.module}/startup-script.sh", {
    environment = var.environment
  })
  filename = "${path.module}/startup-script.sh"
}

# Batch Script
resource "local_file" "batch_script" {
  content = templatefile("${path.module}/batch-script.sh", {
    environment = var.environment
  })
  filename = "${path.module}/batch-script.sh"
}

# Outputs
output "load_balancer_ip" {
  description = "Load Balancer IP Address"
  value       = google_compute_global_forwarding_rule.web.ip_address
}

output "instance_group_name" {
  description = "Instance Group Name"
  value       = google_compute_instance_group_manager.web.name
}
```

### Startup Script

```bash
#!/bin/bash
# startup-script.sh

# Update system
apt-get update

# Install Apache
apt-get install -y apache2

# Start and enable Apache
systemctl start apache2
systemctl enable apache2

# Create health check endpoint
echo "OK" > /var/www/html/health

# Create index page
cat > /var/www/html/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>${environment} Environment</title>
</head>
<body>
    <h1>Hello from GCP ${environment} environment!</h1>
    <p>Instance ID: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/id -H "Metadata-Flavor: Google")</p>
    <p>Zone: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H "Metadata-Flavor: Google")</p>
</body>
</html>
EOF

# Install Cloud Logging agent
curl -sSO https://dl.google.com/cloudagents/add-logging-agent-repo.sh
bash add-logging-agent-repo.sh --also-install

# Install Cloud Monitoring agent
curl -sSO https://dl.google.com/cloudagents/add-monitoring-agent-repo.sh
bash add-monitoring-agent-repo.sh --also-install

# Configure monitoring
cat > /etc/google-fluentd/config.d/application.conf << EOF
<source>
  @type tail
  path /var/log/apache2/access.log
  pos_file /var/lib/google-fluentd/pos/apache2-access.log.pos
  tag apache2.access
  format apache2
</source>

<source>
  @type tail
  path /var/log/apache2/error.log
  pos_file /var/lib/google-fluentd/pos/apache2-error.log.pos
  tag apache2.error
  format /^(?<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?<level>\w+)\] \[pid (?<pid>\d+)\] \[client (?<client>\S+)\] (?<message>.*)$/
</source>
EOF

# Restart logging agent
systemctl restart google-fluentd
```

### Batch Script

```bash
#!/bin/bash
# batch-script.sh

# Update system
apt-get update

# Install Python and dependencies
apt-get install -y python3 python3-pip
pip3 install requests google-cloud-storage

# Create batch processing script
cat > /opt/batch_processor.py << 'EOF'
import os
import time
import requests
from google.cloud import storage

def process_batch():
    """Process batch jobs"""
    print("Starting batch processing...")

    # Get instance metadata
    try:
        response = requests.get(
            'http://metadata.google.internal/computeMetadata/v1/instance/id',
            headers={'Metadata-Flavor': 'Google'}
        )
        instance_id = response.text
        print(f"Running on instance: {instance_id}")
    except Exception as e:
        print(f"Error getting instance metadata: {e}")

    # Simulate batch processing
    for i in range(100):
        print(f"Processing item {i+1}/100")
        time.sleep(1)

    print("Batch processing completed!")

if __name__ == "__main__":
    process_batch()
EOF

# Make script executable
chmod +x /opt/batch_processor.py

# Create systemd service
cat > /etc/systemd/system/batch-processor.service << EOF
[Unit]
Description=Batch Processor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /opt/batch_processor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable batch-processor
systemctl start batch-processor
```

## 🚀 Best Practices

### 1. Machine Type Selection

```yaml
# Choose machine types based on workload
General Purpose:
  - f1-micro: Development, testing
  - g1-small: Small applications
  - n1-standard-1: Medium applications
  - n1-standard-2: Large applications

Compute Optimized:
  - c2-standard-4: CPU-intensive workloads
  - c2-standard-8: High-performance computing
  - c2-standard-16: Batch processing

Memory Optimized:
  - m1-megamem-96: Memory-intensive applications
  - m1-ultramem-40: In-memory databases
  - m1-ultramem-80: Real-time analytics

Storage Optimized:
  - n2-highmem-2: High I/O databases
  - n2-highmem-4: NoSQL databases
  - n2-highmem-8: Data warehousing
```

### 2. Security Best Practices

```yaml
# Secure Compute Engine configuration
FirewallRule:
  type: compute.v1.firewall
  properties:
    name: secure-firewall-rule
    network: $(ref.VPC.selfLink)
    sourceRanges: ["10.0.0.0/8"]
    allowed:
      - IPProtocol: tcp
        ports: ["80", "443"]
    targetTags: ["web-server"]
    direction: INGRESS
```

### 3. Cost Optimization

```yaml
# Use preemptible instances for non-critical workloads
PreemptibleInstanceTemplate:
  type: compute.v1.instanceTemplate
  properties:
    properties:
      scheduling:
        preemptible: true
      machineType: f1-micro
      disks:
        - boot: true
          autoDelete: true
          initializeParams:
            sourceImage: projects/debian-cloud/global/images/family/debian-11
```

## 🏢 Industry Insights

### Google's Compute Engine Usage

- **Global Infrastructure**: Multi-region deployment
- **Auto Scaling**: Traffic-based scaling
- **Preemptible Instances**: Cost optimization for batch processing
- **Custom Images**: Pre-configured application images

### Netflix's GCP Strategy

- **Content Delivery**: Video content processing
- **Data Analytics**: User behavior analysis
- **Machine Learning**: Recommendation systems
- **Global Deployment**: Multi-region architecture

### Spotify's Compute Engine Approach

- **Music Processing**: Audio file processing
- **Data Pipeline**: ETL operations
- **Real-time Analytics**: User listening data
- **Cost Optimization**: Preemptible instances

## 🎯 Interview Questions

### Basic Level

1. **What is Compute Engine?**

   - Virtual machines in Google Cloud
   - On-demand compute capacity
   - Global infrastructure
   - Pay-per-use pricing

2. **What are the different machine types?**

   - General Purpose: f1, g1, n1
   - Compute Optimized: c2, c3
   - Memory Optimized: m1, m2
   - Storage Optimized: n2-highmem

3. **What is auto scaling?**
   - Automatic capacity adjustment
   - Scale based on metrics
   - Cost optimization
   - High availability

### Intermediate Level

4. **How do you implement high availability?**

   ```yaml
   # Multi-zone deployment
   InstanceGroupManager:
     type: compute.v1.instanceGroupManager
     properties:
       zone: us-central1-a
       targetSize: 2
       autoHealingPolicies:
         - healthCheck: $(ref.HealthCheck.selfLink)
           initialDelaySec: 300
   ```

5. **How do you optimize Compute Engine costs?**

   - Preemptible instances
   - Committed use discounts
   - Auto scaling
   - Right-sizing instances

6. **How do you secure Compute Engine instances?**
   - Firewall rules
   - VPC networks
   - IAM roles
   - OS login

### Advanced Level

7. **How do you implement disaster recovery?**

   - Multi-region deployment
   - Cross-region replication
   - Backup strategies
   - Failover mechanisms

8. **How do you handle preemptible instance interruptions?**

   - Checkpointing
   - Graceful shutdown
   - State management
   - Workload distribution

9. **How do you implement blue-green deployments?**
   - Two identical environments
   - Traffic switching
   - Zero-downtime deployment
   - Rollback capabilities

---

**Next**: [GCP Cloud Storage](GCP_CloudStorage.md) - Object storage, versioning, lifecycle policies
