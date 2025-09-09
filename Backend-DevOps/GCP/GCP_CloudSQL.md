# 🗄️ GCP Cloud SQL: Managed Database Service

> **Master GCP Cloud SQL for scalable, managed relational databases**

## 📚 Concept

Cloud SQL is a fully managed relational database service that makes it easy to set up, maintain, manage, and administer your relational databases on Google Cloud Platform. It supports MySQL, PostgreSQL, and SQL Server databases with automatic backups, replication, and scaling.

### Key Features
- **Fully Managed**: Automated backups, updates, and maintenance
- **High Availability**: Automatic failover and replication
- **Scalability**: Vertical and horizontal scaling
- **Security**: Encryption at rest and in transit
- **Multiple Engines**: MySQL, PostgreSQL, SQL Server
- **Integration**: Seamless integration with other GCP services

## 🏗️ Cloud SQL Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Cloud SQL Architecture                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Primary   │  │   Read      │  │   Read      │     │
│  │   Instance  │  │   Replica   │  │   Replica   │     │
│  │   (Master)  │  │   (Slave)   │  │   (Slave)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              VPC Network                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Private   │  │   Private   │  │   Private   │ │ │
│  │  │   IP        │  │   IP        │  │   IP        │ │ │
│  │  │   (Primary) │  │   (Replica) │  │   (Replica) │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Cloud     │  │   Cloud     │  │   Cloud     │     │
│  │   Storage   │  │   Storage   │  │   Storage   │     │
│  │   (Backup)  │  │   (Backup)  │  │   (Backup)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Cloud SQL with Terraform

```hcl
# cloudsql.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
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

variable "database_version" {
  description = "Database version"
  type        = string
  default     = "POSTGRES_14"
}

variable "machine_type" {
  description = "Machine type for Cloud SQL instance"
  type        = string
  default     = "db-f1-micro"
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
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.main.id
  project       = var.project_id
}

# Private IP allocation
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.environment}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
  project       = var.project_id
}

# Private connection
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
  project                 = var.project_id
}

# Cloud SQL Instance
resource "google_sql_database_instance" "main" {
  name             = "${var.environment}-cloudsql-instance"
  database_version = var.database_version
  region           = var.region
  project          = var.project_id

  settings {
    tier                        = var.machine_type
    availability_type           = "REGIONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = 100
    disk_autoresize             = true
    disk_autoresize_limit       = 200
    deletion_protection_enabled = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.main.id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
    }

    database_flags {
      name  = "log_statement"
      value = "all"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }

    maintenance_window {
      day          = 7
      hour         = 3
      update_track = "stable"
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]

  deletion_protection = true
}

# Cloud SQL Database
resource "google_sql_database" "main" {
  name     = "myapp"
  instance = google_sql_database_instance.main.name
  project  = var.project_id
}

# Cloud SQL User
resource "google_sql_user" "main" {
  name     = "app_user"
  instance = google_sql_database_instance.main.name
  password = var.db_password
  project  = var.project_id
}

# Read Replica
resource "google_sql_database_instance" "read_replica" {
  name                 = "${var.environment}-cloudsql-read-replica"
  master_instance_name = google_sql_database_instance.main.name
  region               = var.region
  project              = var.project_id

  replica_configuration {
    failover_target = false
  }

  settings {
    tier                        = var.machine_type
    availability_type           = "ZONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = 100
    disk_autoresize             = true
    disk_autoresize_limit       = 200

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.main.id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Cloud SQL SSL Certificate
resource "google_sql_ssl_cert" "main" {
  common_name = "${var.environment}-ssl-cert"
  instance    = google_sql_database_instance.main.name
  project     = var.project_id
}

# IAM Binding for Cloud SQL
resource "google_project_iam_binding" "cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"

  members = [
    "serviceAccount:app-service@${var.project_id}.iam.gserviceaccount.com",
  ]
}

# Cloud SQL IAM Database User
resource "google_sql_user" "iam_user" {
  name     = "app-service@${var.project_id}.iam.gserviceaccount.com"
  instance = google_sql_database_instance.main.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
  project  = var.project_id
}

# Secret Manager Secret for Database Password
resource "google_secret_manager_secret" "db_password" {
  secret_id = "${var.environment}-db-password"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# IAM Binding for Secret Manager
resource "google_secret_manager_secret_iam_binding" "db_password" {
  secret_id = google_secret_manager_secret.db_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  members = [
    "serviceAccount:app-service@${var.project_id}.iam.gserviceaccount.com",
  ]
}

# Monitoring and Alerting
resource "google_monitoring_alert_policy" "cloudsql_cpu" {
  display_name = "Cloud SQL High CPU Usage"
  project      = var.project_id

  conditions {
    display_name = "Cloud SQL CPU usage is high"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND resource.label.instance_id=\"${google_sql_database_instance.main.name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "cloudsql_connections" {
  display_name = "Cloud SQL High Connection Count"
  project      = var.project_id

  conditions {
    display_name = "Cloud SQL connection count is high"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND resource.label.instance_id=\"${google_sql_database_instance.main.name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 100

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification Channel"
  type         = "email"
  project      = var.project_id

  labels = {
    email_address = "admin@example.com"
  }
}

# Outputs
output "cloudsql_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.main.name
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "cloudsql_private_ip" {
  description = "Cloud SQL private IP address"
  value       = google_sql_database_instance.main.private_ip_address
}

output "read_replica_name" {
  description = "Read replica instance name"
  value       = google_sql_database_instance.read_replica.name
}
```

### Go Application with Cloud SQL Integration

```go
// main.go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    "cloud.google.com/go/secretmanager/apiv1"
    "cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
    "github.com/gin-gonic/gin"
    _ "github.com/lib/pq"
)

type Database struct {
    db *sql.DB
}

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

type Product struct {
    ID          int     `json:"id"`
    Name        string  `json:"name"`
    Description string  `json:"description"`
    Price       float64 `json:"price"`
    Category    string  `json:"category"`
    Stock       int     `json:"stock"`
}

type Order struct {
    ID        int       `json:"id"`
    UserID    int       `json:"user_id"`
    ProductID int       `json:"product_id"`
    Quantity  int       `json:"quantity"`
    Total     float64   `json:"total"`
    Status    string    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
}

func NewDatabase() (*Database, error) {
    // Get database connection details from environment
    projectID := os.Getenv("GCP_PROJECT_ID")
    instanceName := os.Getenv("CLOUDSQL_INSTANCE_NAME")
    databaseName := os.Getenv("CLOUDSQL_DATABASE_NAME")
    username := os.Getenv("CLOUDSQL_USERNAME")

    // Get password from Secret Manager
    password, err := getSecret(projectID, "production-db-password")
    if err != nil {
        return nil, fmt.Errorf("failed to get database password: %w", err)
    }

    // Build connection string
    connStr := fmt.Sprintf("host=/cloudsql/%s user=%s password=%s dbname=%s sslmode=require",
        instanceName, username, password, databaseName)

    db, err := sql.Open("postgres", connStr)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }

    // Test connection
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }

    // Configure connection pool
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)

    return &Database{db: db}, nil
}

func getSecret(projectID, secretID string) (string, error) {
    ctx := context.Background()
    client, err := secretmanager.NewClient(ctx)
    if err != nil {
        return "", fmt.Errorf("failed to create secret manager client: %w", err)
    }
    defer client.Close()

    name := fmt.Sprintf("projects/%s/secrets/%s/versions/latest", projectID, secretID)
    req := &secretmanagerpb.AccessSecretVersionRequest{
        Name: name,
    }

    result, err := client.AccessSecretVersion(ctx, req)
    if err != nil {
        return "", fmt.Errorf("failed to access secret: %w", err)
    }

    return string(result.Payload.Data), nil
}

func (d *Database) CreateUser(ctx context.Context, name, email string) (*User, error) {
    query := `
        INSERT INTO users (name, email, created_at)
        VALUES ($1, $2, $3)
        RETURNING id, name, email, created_at
    `

    var user User
    err := d.db.QueryRowContext(ctx, query, name, email, time.Now()).Scan(
        &user.ID, &user.Name, &user.Email, &user.CreatedAt,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create user: %w", err)
    }

    return &user, nil
}

func (d *Database) GetUser(ctx context.Context, id int) (*User, error) {
    query := `
        SELECT id, name, email, created_at
        FROM users
        WHERE id = $1
    `

    var user User
    err := d.db.QueryRowContext(ctx, query, id).Scan(
        &user.ID, &user.Name, &user.Email, &user.CreatedAt,
    )
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user not found")
        }
        return nil, fmt.Errorf("failed to get user: %w", err)
    }

    return &user, nil
}

func (d *Database) ListUsers(ctx context.Context, limit, offset int) ([]*User, error) {
    query := `
        SELECT id, name, email, created_at
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    `

    rows, err := d.db.QueryContext(ctx, query, limit, offset)
    if err != nil {
        return nil, fmt.Errorf("failed to list users: %w", err)
    }
    defer rows.Close()

    var users []*User
    for rows.Next() {
        var user User
        err := rows.Scan(&user.ID, &user.Name, &user.Email, &user.CreatedAt)
        if err != nil {
            return nil, fmt.Errorf("failed to scan user: %w", err)
        }
        users = append(users, &user)
    }

    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("error iterating users: %w", err)
    }

    return users, nil
}

func (d *Database) CreateProduct(ctx context.Context, name, description string, price float64, category string, stock int) (*Product, error) {
    query := `
        INSERT INTO products (name, description, price, category, stock)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, name, description, price, category, stock
    `

    var product Product
    err := d.db.QueryRowContext(ctx, query, name, description, price, category, stock).Scan(
        &product.ID, &product.Name, &product.Description, &product.Price, &product.Category, &product.Stock,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create product: %w", err)
    }

    return &product, nil
}

func (d *Database) GetProduct(ctx context.Context, id int) (*Product, error) {
    query := `
        SELECT id, name, description, price, category, stock
        FROM products
        WHERE id = $1
    `

    var product Product
    err := d.db.QueryRowContext(ctx, query, id).Scan(
        &product.ID, &product.Name, &product.Description, &product.Price, &product.Category, &product.Stock,
    )
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("product not found")
        }
        return nil, fmt.Errorf("failed to get product: %w", err)
    }

    return &product, nil
}

func (d *Database) ListProducts(ctx context.Context, limit, offset int) ([]*Product, error) {
    query := `
        SELECT id, name, description, price, category, stock
        FROM products
        ORDER BY name
        LIMIT $1 OFFSET $2
    `

    rows, err := d.db.QueryContext(ctx, query, limit, offset)
    if err != nil {
        return nil, fmt.Errorf("failed to list products: %w", err)
    }
    defer rows.Close()

    var products []*Product
    for rows.Next() {
        var product Product
        err := rows.Scan(&product.ID, &product.Name, &product.Description, &product.Price, &product.Category, &product.Stock)
        if err != nil {
            return nil, fmt.Errorf("failed to scan product: %w", err)
        }
        products = append(products, &product)
    }

    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("error iterating products: %w", err)
    }

    return products, nil
}

func (d *Database) CreateOrder(ctx context.Context, userID, productID, quantity int) (*Order, error) {
    // Start transaction
    tx, err := d.db.BeginTx(ctx, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to begin transaction: %w", err)
    }
    defer tx.Rollback()

    // Get product details
    var product Product
    err = tx.QueryRowContext(ctx, "SELECT id, name, price, stock FROM products WHERE id = $1", productID).Scan(
        &product.ID, &product.Name, &product.Price, &product.Stock,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to get product: %w", err)
    }

    // Check stock availability
    if product.Stock < quantity {
        return nil, fmt.Errorf("insufficient stock: requested %d, available %d", quantity, product.Stock)
    }

    // Calculate total
    total := product.Price * float64(quantity)

    // Create order
    var order Order
    err = tx.QueryRowContext(ctx, `
        INSERT INTO orders (user_id, product_id, quantity, total, status, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, user_id, product_id, quantity, total, status, created_at
    `, userID, productID, quantity, total, "pending", time.Now()).Scan(
        &order.ID, &order.UserID, &order.ProductID, &order.Quantity, &order.Total, &order.Status, &order.CreatedAt,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create order: %w", err)
    }

    // Update stock
    _, err = tx.ExecContext(ctx, "UPDATE products SET stock = stock - $1 WHERE id = $2", quantity, productID)
    if err != nil {
        return nil, fmt.Errorf("failed to update stock: %w", err)
    }

    // Commit transaction
    if err := tx.Commit(); err != nil {
        return nil, fmt.Errorf("failed to commit transaction: %w", err)
    }

    return &order, nil
}

func (d *Database) GetOrder(ctx context.Context, id int) (*Order, error) {
    query := `
        SELECT id, user_id, product_id, quantity, total, status, created_at
        FROM orders
        WHERE id = $1
    `

    var order Order
    err := d.db.QueryRowContext(ctx, query, id).Scan(
        &order.ID, &order.UserID, &order.ProductID, &order.Quantity, &order.Total, &order.Status, &order.CreatedAt,
    )
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("order not found")
        }
        return nil, fmt.Errorf("failed to get order: %w", err)
    }

    return &order, nil
}

func (d *Database) ListOrders(ctx context.Context, limit, offset int) ([]*Order, error) {
    query := `
        SELECT id, user_id, product_id, quantity, total, status, created_at
        FROM orders
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    `

    rows, err := d.db.QueryContext(ctx, query, limit, offset)
    if err != nil {
        return nil, fmt.Errorf("failed to list orders: %w", err)
    }
    defer rows.Close()

    var orders []*Order
    for rows.Next() {
        var order Order
        err := rows.Scan(&order.ID, &order.UserID, &order.ProductID, &order.Quantity, &order.Total, &order.Status, &order.CreatedAt)
        if err != nil {
            return nil, fmt.Errorf("failed to scan order: %w", err)
        }
        orders = append(orders, &order)
    }

    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("error iterating orders: %w", err)
    }

    return orders, nil
}

func (d *Database) Close() error {
    return d.db.Close()
}

// HTTP handlers
func setupRoutes(db *Database) *gin.Engine {
    r := gin.Default()

    // Health check
    r.GET("/health", func(c *gin.Context) {
        ctx := c.Request.Context()
        if err := db.db.PingContext(ctx); err != nil {
            c.JSON(http.StatusServiceUnavailable, gin.H{
                "status": "unhealthy",
                "error":  err.Error(),
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
        })
    })

    // API routes
    api := r.Group("/api/v1")
    {
        // User routes
        api.POST("/users", func(c *gin.Context) {
            var req struct {
                Name  string `json:"name" binding:"required"`
                Email string `json:"email" binding:"required,email"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            user, err := db.CreateUser(c.Request.Context(), req.Name, req.Email)
            if err != nil {
                log.Printf("Error creating user: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
                return
            }

            c.JSON(http.StatusCreated, user)
        })

        api.GET("/users/:id", func(c *gin.Context) {
            var req struct {
                ID int `uri:"id" binding:"required"`
            }

            if err := c.ShouldBindUri(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            user, err := db.GetUser(c.Request.Context(), req.ID)
            if err != nil {
                if err.Error() == "user not found" {
                    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
                    return
                }
                log.Printf("Error getting user: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user"})
                return
            }

            c.JSON(http.StatusOK, user)
        })

        api.GET("/users", func(c *gin.Context) {
            limit := 10
            offset := 0

            if l := c.Query("limit"); l != "" {
                if parsed, err := fmt.Sscanf(l, "%d", &limit); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
                    return
                }
            }

            if o := c.Query("offset"); o != "" {
                if parsed, err := fmt.Sscanf(o, "%d", &offset); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid offset parameter"})
                    return
                }
            }

            users, err := db.ListUsers(c.Request.Context(), limit, offset)
            if err != nil {
                log.Printf("Error listing users: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list users"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "users": users,
                "limit": limit,
                "offset": offset,
            })
        })

        // Product routes
        api.POST("/products", func(c *gin.Context) {
            var req struct {
                Name        string  `json:"name" binding:"required"`
                Description string  `json:"description" binding:"required"`
                Price       float64 `json:"price" binding:"required,min=0"`
                Category    string  `json:"category" binding:"required"`
                Stock       int     `json:"stock" binding:"required,min=0"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            product, err := db.CreateProduct(c.Request.Context(), req.Name, req.Description, req.Price, req.Category, req.Stock)
            if err != nil {
                log.Printf("Error creating product: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create product"})
                return
            }

            c.JSON(http.StatusCreated, product)
        })

        api.GET("/products/:id", func(c *gin.Context) {
            var req struct {
                ID int `uri:"id" binding:"required"`
            }

            if err := c.ShouldBindUri(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            product, err := db.GetProduct(c.Request.Context(), req.ID)
            if err != nil {
                if err.Error() == "product not found" {
                    c.JSON(http.StatusNotFound, gin.H{"error": "Product not found"})
                    return
                }
                log.Printf("Error getting product: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get product"})
                return
            }

            c.JSON(http.StatusOK, product)
        })

        api.GET("/products", func(c *gin.Context) {
            limit := 10
            offset := 0

            if l := c.Query("limit"); l != "" {
                if parsed, err := fmt.Sscanf(l, "%d", &limit); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
                    return
                }
            }

            if o := c.Query("offset"); o != "" {
                if parsed, err := fmt.Sscanf(o, "%d", &offset); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid offset parameter"})
                    return
                }
            }

            products, err := db.ListProducts(c.Request.Context(), limit, offset)
            if err != nil {
                log.Printf("Error listing products: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list products"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "products": products,
                "limit": limit,
                "offset": offset,
            })
        })

        // Order routes
        api.POST("/orders", func(c *gin.Context) {
            var req struct {
                UserID    int `json:"user_id" binding:"required"`
                ProductID int `json:"product_id" binding:"required"`
                Quantity  int `json:"quantity" binding:"required,min=1"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            order, err := db.CreateOrder(c.Request.Context(), req.UserID, req.ProductID, req.Quantity)
            if err != nil {
                log.Printf("Error creating order: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
                return
            }

            c.JSON(http.StatusCreated, order)
        })

        api.GET("/orders/:id", func(c *gin.Context) {
            var req struct {
                ID int `uri:"id" binding:"required"`
            }

            if err := c.ShouldBindUri(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            order, err := db.GetOrder(c.Request.Context(), req.ID)
            if err != nil {
                if err.Error() == "order not found" {
                    c.JSON(http.StatusNotFound, gin.H{"error": "Order not found"})
                    return
                }
                log.Printf("Error getting order: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get order"})
                return
            }

            c.JSON(http.StatusOK, order)
        })

        api.GET("/orders", func(c *gin.Context) {
            limit := 10
            offset := 0

            if l := c.Query("limit"); l != "" {
                if parsed, err := fmt.Sscanf(l, "%d", &limit); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
                    return
                }
            }

            if o := c.Query("offset"); o != "" {
                if parsed, err := fmt.Sscanf(o, "%d", &offset); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid offset parameter"})
                    return
                }
            }

            orders, err := db.ListOrders(c.Request.Context(), limit, offset)
            if err != nil {
                log.Printf("Error listing orders: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list orders"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "orders": orders,
                "limit": limit,
                "offset": offset,
            })
        })
    }

    return r
}

func main() {
    // Initialize database
    db, err := NewDatabase()
    if err != nil {
        log.Fatalf("Failed to initialize database: %v", err)
    }
    defer db.Close()

    // Setup routes
    r := setupRoutes(db)

    // Start server
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    log.Printf("Server starting on port %s", port)
    if err := r.Run(":" + port); err != nil {
        log.Fatalf("Failed to start server: %v", err)
    }
}
```

## 🚀 Best Practices

### 1. Connection Management
```go
// Configure connection pool
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)
```

### 2. Security
```hcl
# Use private IP
ip_configuration {
  ipv4_enabled = false
  private_network = google_compute_network.main.id
  require_ssl = true
}
```

### 3. Monitoring
```hcl
# Enable query insights
insights_config {
  query_insights_enabled = true
  query_string_length = 1024
  record_application_tags = true
}
```

## 🏢 Industry Insights

### Cloud SQL Usage Patterns
- **Managed Service**: Automated backups and maintenance
- **High Availability**: Regional instances with automatic failover
- **Read Replicas**: Scale read operations
- **Security**: Encryption and IAM integration

### Enterprise Cloud SQL Strategy
- **Multi-Region**: Cross-region replication for disaster recovery
- **Performance**: Query optimization and connection pooling
- **Cost Management**: Right-sizing and reserved instances
- **Compliance**: Audit logging and data encryption

## 🎯 Interview Questions

### Basic Level
1. **What is Cloud SQL?**
   - Managed relational database service
   - Supports MySQL, PostgreSQL, SQL Server
   - Automated backups and maintenance

2. **What are the benefits of Cloud SQL?**
   - Fully managed service
   - High availability
   - Automatic scaling
   - Security features

3. **What database engines does Cloud SQL support?**
   - MySQL
   - PostgreSQL
   - SQL Server

### Intermediate Level
4. **How do you implement high availability with Cloud SQL?**
   - Regional instances
   - Automatic failover
   - Read replicas
   - Cross-region replication

5. **How do you secure Cloud SQL instances?**
   - Private IP addresses
   - SSL/TLS encryption
   - IAM integration
   - VPC networking

6. **How do you optimize Cloud SQL performance?**
   - Connection pooling
   - Query optimization
   - Read replicas
   - Proper indexing

### Advanced Level
7. **How do you implement disaster recovery with Cloud SQL?**
   - Cross-region replication
   - Automated backups
   - Point-in-time recovery
   - Failover procedures

8. **How do you handle Cloud SQL scaling?**
   - Vertical scaling
   - Read replicas
   - Connection pooling
   - Query optimization

9. **How do you implement Cloud SQL monitoring?**
   - Cloud Monitoring integration
   - Query insights
   - Performance metrics
   - Alerting policies
