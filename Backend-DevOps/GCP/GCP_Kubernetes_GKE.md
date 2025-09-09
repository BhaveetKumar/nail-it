# ☸️ GCP GKE: Google Kubernetes Engine

> **Master GCP GKE for managed Kubernetes clusters in the cloud**

## 📚 Concept

Google Kubernetes Engine (GKE) is a managed environment for deploying, managing, and scaling containerized applications using Google infrastructure. GKE provides a fully managed Kubernetes service with automatic scaling, updates, and security features.

### Key Features
- **Managed Kubernetes**: Google manages the control plane
- **Auto-scaling**: Automatic node and pod scaling
- **Security**: Built-in security features and compliance
- **Integration**: Seamless integration with GCP services
- **Multi-cluster**: Support for multi-cluster deployments
- **Serverless**: GKE Autopilot for serverless containers

## 🏗️ GKE Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GKE Architecture                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   GKE       │  │   GKE       │  │   GKE       │     │
│  │   Control   │  │   Control   │  │   Control   │     │
│  │   Plane     │  │   Plane     │  │   Plane     │     │
│  │   (Zone-1)  │  │   (Zone-2)  │  │   (Zone-3)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              VPC Network                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Node      │  │   Node      │  │   Node      │ │ │
│  │  │   Pool 1    │  │   Pool 2    │  │   Pool 3    │ │ │
│  │  │   (Zone-1)  │  │   (Zone-2)  │  │   (Zone-3)  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Workloads │  │   Services  │  │   Ingress   │     │
│  │   (Pods)    │  │   (Load     │  │   (Traffic  │     │
│  │             │  │   Balancer) │  │   Routing)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### GKE with Terraform

```hcl
# gke.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "my-gke-cluster"
}

variable "node_count" {
  description = "Number of nodes per zone"
  type        = number
  default     = 3
}

variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "e2-medium"
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.environment}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

# Subnet
resource "google_compute_subnetwork" "main" {
  name          = "${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.main.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.environment}-allow-internal"
  network = google_compute_network.main.name
  project = var.project_id

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/16", "10.1.0.0/16", "10.2.0.0/20"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.environment}-allow-ssh"
  network = google_compute_network.main.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gke-node"]
}

# Service Account for GKE
resource "google_service_account" "gke_service" {
  account_id   = "${var.environment}-gke-service"
  display_name = "${var.environment} GKE Service Account"
  description  = "Service account for GKE cluster"
  project      = var.project_id
}

# IAM Bindings for GKE Service Account
resource "google_project_iam_binding" "gke_service_compute" {
  project = var.project_id
  role    = "roles/compute.viewer"

  members = [
    "serviceAccount:${google_service_account.gke_service.email}",
  ]
}

resource "google_project_iam_binding" "gke_service_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"

  members = [
    "serviceAccount:${google_service_account.gke_service.email}",
  ]
}

resource "google_project_iam_binding" "gke_service_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"

  members = [
    "serviceAccount:${google_service_account.gke_service.email}",
  ]
}

resource "google_project_iam_binding" "gke_service_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"

  members = [
    "serviceAccount:${google_service_account.gke_service.email}",
  ]
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.main.name

  # Enable network policy
  network_policy {
    enabled = true
  }

  # Enable IP aliasing
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Enable private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Enable master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Enable cluster autoscaling
  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = 1
      maximum       = 10
    }
    resource_limits {
      resource_type = "memory"
      minimum       = 1
      maximum       = 100
    }
  }

  # Enable vertical pod autoscaling
  vertical_pod_autoscaling {
    enabled = true
  }

  # Enable horizontal pod autoscaling
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }

  # Enable maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Enable node auto-repair and auto-upgrade
  node_config {
    service_account = google_service_account.gke_service.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
    }

    tags = ["gke-node"]

    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  # Enable release channel
  release_channel {
    channel = "REGULAR"
  }

  # Enable monitoring
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "APISERVER", "CONTROLLER_MANAGER", "SCHEDULER"]
  }

  # Enable logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  depends_on = [
    google_project_iam_binding.gke_service_compute,
    google_project_iam_binding.gke_service_storage,
    google_project_iam_binding.gke_service_monitoring,
    google_project_iam_binding.gke_service_logging,
  ]
}

# Node Pool
resource "google_container_node_pool" "main" {
  name       = "${var.environment}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.main.name
  project    = var.project_id
  node_count = var.node_count

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = false
    machine_type = var.machine_type

    service_account = google_service_account.gke_service.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
    }

    tags = ["gke-node"]

    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Enable shielded nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Enable guest OS features
    guest_accelerator {
      type  = "nvidia-tesla-k80"
      count = 1
    }
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# Additional Node Pool for Spot Instances
resource "google_container_node_pool" "spot" {
  name       = "${var.environment}-spot-node-pool"
  location   = var.region
  cluster    = google_container_cluster.main.name
  project    = var.project_id
  node_count = 2

  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = var.machine_type

    service_account = google_service_account.gke_service.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-type   = "spot"
    }

    taints {
      key    = "cloud.google.com/gke-preemptible"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    tags = ["gke-node", "spot"]
  }
}

# Kubernetes Provider
provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

# Helm Provider
provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.provider.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
  }
}

# Data source for client config
data "google_client_config" "provider" {}

# Kubernetes Namespaces
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
  }
}

resource "kubernetes_namespace" "ingress" {
  metadata {
    name = "ingress-nginx"
  }
}

resource "kubernetes_namespace" "cert_manager" {
  metadata {
    name = "cert-manager"
  }
}

# Nginx Ingress Controller
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  version    = "4.8.3"

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }

  set {
    name  = "controller.service.annotations.cloud\\.google\\.com/load-balancer-type"
    value = "External"
  }

  set {
    name  = "controller.admissionWebhooks.enabled"
    value = "false"
  }

  depends_on = [google_container_node_pool.main]
}

# Cert Manager
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  version    = "v1.13.2"

  set {
    name  = "installCRDs"
    value = "true"
  }

  depends_on = [google_container_node_pool.main]
}

# Prometheus and Grafana
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "55.5.0"

  set {
    name  = "grafana.service.type"
    value = "LoadBalancer"
  }

  set {
    name  = "grafana.adminPassword"
    value = "admin123"
  }

  depends_on = [google_container_node_pool.main]
}

# Application Deployment
resource "kubernetes_deployment" "web_app" {
  metadata {
    name      = "web-app"
    namespace = "default"
    labels = {
      app = "web-app"
    }
  }

  spec {
    replicas = 3

    selector {
      match_labels = {
        app = "web-app"
      }
    }

    template {
      metadata {
        labels = {
          app = "web-app"
        }
      }

      spec {
        container {
          image = "nginx:alpine"
          name  = "web-app"

          port {
            container_port = 80
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "200m"
              memory = "256Mi"
            }
          }

          liveness_probe {
            http_get {
              path = "/"
              port = 80
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/"
              port = 80
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
      }
    }
  }

  depends_on = [google_container_node_pool.main]
}

# Service
resource "kubernetes_service" "web_app" {
  metadata {
    name      = "web-app-service"
    namespace = "default"
  }

  spec {
    selector = {
      app = "web-app"
    }

    port {
      port        = 80
      target_port = 80
    }

    type = "ClusterIP"
  }
}

# Ingress
resource "kubernetes_ingress_v1" "web_app" {
  metadata {
    name      = "web-app-ingress"
    namespace = "default"
    annotations = {
      "kubernetes.io/ingress.class"                = "nginx"
      "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect" = "true"
    }
  }

  spec {
    tls {
      hosts       = ["web-app.example.com"]
      secret_name = "web-app-tls"
    }

    rule {
      host = "web-app.example.com"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "web-app-service"
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}

# Cluster Issuer for Let's Encrypt
resource "kubernetes_manifest" "cluster_issuer" {
  manifest = {
    apiVersion = "cert-manager.io/v1"
    kind       = "ClusterIssuer"
    metadata = {
      name = "letsencrypt-prod"
    }
    spec = {
      acme = {
        server = "https://acme-v02.api.letsencrypt.org/directory"
        email  = "admin@example.com"
        privateKeySecretRef = {
          name = "letsencrypt-prod"
        }
        solvers = [
          {
            http01 = {
              ingress = {
                class = "nginx"
              }
            }
          }
        ]
      }
    }
  }

  depends_on = [helm_release.cert_manager]
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler_v2" "web_app" {
  metadata {
    name      = "web-app-hpa"
    namespace = "default"
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "web-app"
    }

    min_replicas = 2
    max_replicas = 10

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }
}

# Pod Disruption Budget
resource "kubernetes_pod_disruption_budget_v1" "web_app" {
  metadata {
    name      = "web-app-pdb"
    namespace = "default"
  }

  spec {
    min_available = 1

    selector {
      match_labels = {
        app = "web-app"
      }
    }
  }
}

# Network Policy
resource "kubernetes_network_policy" "web_app" {
  metadata {
    name      = "web-app-netpol"
    namespace = "default"
  }

  spec {
    pod_selector {
      match_labels = {
        app = "web-app"
      }
    }

    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "ingress-nginx"
          }
        }
      }
      ports {
        port     = "80"
        protocol = "TCP"
      }
    }

    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "kube-system"
          }
        }
      }
      ports {
        port     = "53"
        protocol = "UDP"
      }
    }
  }
}

# Outputs
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.main.endpoint
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

output "cluster_network" {
  description = "GKE cluster network"
  value       = google_container_cluster.main.network
}

output "cluster_subnetwork" {
  description = "GKE cluster subnetwork"
  value       = google_container_cluster.main.subnetwork
}
```

### Kubernetes Application Deployment

```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: default
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
        image: nginx:alpine
        ports:
        - containerPort: 80
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
  namespace: default
spec:
  selector:
    app: web-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app-ingress
  namespace: default
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - web-app.example.com
    secretName: web-app-tls
  rules:
  - host: web-app.example.com
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

## 🚀 Best Practices

### 1. Cluster Security
```hcl
# Enable private cluster
private_cluster_config {
  enable_private_nodes    = true
  enable_private_endpoint = false
}

# Enable network policy
network_policy {
  enabled = true
}
```

### 2. Node Pool Configuration
```hcl
# Use multiple node pools
resource "google_container_node_pool" "spot" {
  node_config {
    preemptible = true
    taints {
      key    = "cloud.google.com/gke-preemptible"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}
```

### 3. Monitoring and Logging
```hcl
# Enable monitoring
monitoring_config {
  enable_components = ["SYSTEM_COMPONENTS", "APISERVER"]
}

# Enable logging
logging_config {
  enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
}
```

## 🏢 Industry Insights

### GKE Usage Patterns
- **Managed Kubernetes**: Google manages control plane
- **Auto-scaling**: Cluster and pod autoscaling
- **Security**: Workload identity and network policies
- **Integration**: Seamless GCP service integration

### Enterprise GKE Strategy
- **Multi-cluster**: Multiple clusters for different environments
- **Security**: Private clusters and network policies
- **Monitoring**: Prometheus and Grafana integration
- **CI/CD**: GitOps with ArgoCD

## 🎯 Interview Questions

### Basic Level
1. **What is GKE?**
   - Managed Kubernetes service
   - Google manages control plane
   - Auto-scaling and security features

2. **What are the benefits of GKE?**
   - Managed service
   - Auto-scaling
   - Security features
   - GCP integration

3. **What are GKE node pools?**
   - Groups of nodes with same configuration
   - Different machine types and sizes
   - Separate scaling policies

### Intermediate Level
4. **How do you secure a GKE cluster?**
   - Private clusters
   - Network policies
   - Workload identity
   - Binary authorization

5. **How do you implement auto-scaling with GKE?**
   - Cluster autoscaling
   - Horizontal pod autoscaling
   - Vertical pod autoscaling
   - Node pool autoscaling

6. **How do you monitor a GKE cluster?**
   - Cloud Monitoring integration
   - Prometheus and Grafana
   - Logging configuration
   - Custom metrics

### Advanced Level
7. **How do you implement multi-cluster GKE?**
   - Multiple clusters
   - Cross-cluster networking
   - Centralized management
   - Workload distribution

8. **How do you handle GKE upgrades?**
   - Release channels
   - Node pool upgrades
   - Cluster upgrades
   - Rolling updates

9. **How do you implement disaster recovery with GKE?**
   - Multi-region deployment
   - Backup strategies
   - Cross-region replication
   - Failover procedures
