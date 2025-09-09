# 🗄️ AWS RDS: Relational Database Service

> **Master AWS RDS for scalable, managed relational databases**

## 📚 Concept

Amazon RDS (Relational Database Service) is a managed database service that makes it easy to set up, operate, and scale relational databases in the cloud. It provides cost-efficient, resizable capacity while automating time-consuming administration tasks.

### Key Features

- **Managed Service**: Automated backups, patching, and monitoring
- **Multi-AZ Deployment**: High availability and fault tolerance
- **Read Replicas**: Improved read performance and scalability
- **Automated Scaling**: Storage and compute scaling
- **Security**: Encryption at rest and in transit
- **Multiple Engines**: MySQL, PostgreSQL, MariaDB, Oracle, SQL Server

## 🏗️ RDS Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS RDS Architecture                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Primary   │  │   Standby   │  │   Read      │     │
│  │   Database  │  │   Database  │  │   Replicas  │     │
│  │   (Multi-AZ)│  │   (Multi-AZ)│  │   (Cross-AZ)│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Application Layer                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Web       │  │   API       │  │   Batch     │ │ │
│  │  │   Servers   │  │   Servers   │  │   Jobs      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              AWS Services                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   CloudWatch│  │   CloudTrail│  │   IAM       │ │ │
│  │  │   Monitoring│  │   Logging   │  │   Security  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### RDS with Terraform

```hcl
# rds.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "myapp"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "admin"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Data sources
data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["${var.environment}-vpc"]
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }

  filter {
    name   = "tag:Type"
    values = ["private"]
  }
}

# DB subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${var.environment}-db-subnet-group"
  subnet_ids = data.aws_subnets.private.ids

  tags = {
    Name        = "${var.environment}-db-subnet-group"
    Environment = var.environment
  }
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${var.environment}-rds-"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.environment}-rds-sg"
    Environment = var.environment
  }
}

# RDS instance
resource "aws_db_instance" "main" {
  identifier = "${var.environment}-${var.db_name}"

  # Engine configuration
  engine         = "postgres"
  engine_version = "14.7"
  instance_class = var.db_instance_class

  # Storage configuration
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  # Database configuration
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  # Network configuration
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  publicly_accessible    = false

  # Backup configuration
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  copy_tags_to_snapshot  = true

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  # Multi-AZ for high availability
  multi_az = true

  # Deletion protection
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.environment}-${var.db_name}-final-snapshot"

  tags = {
    Name        = "${var.environment}-${var.db_name}"
    Environment = var.environment
  }
}

# Read replicas
resource "aws_db_instance" "read_replica" {
  count = 2

  identifier = "${var.environment}-${var.db_name}-replica-${count.index + 1}"

  # Replica configuration
  replicate_source_db = aws_db_instance.main.identifier
  instance_class      = var.db_instance_class

  # Storage configuration
  storage_encrypted = true

  # Network configuration
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  publicly_accessible    = false

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  tags = {
    Name        = "${var.environment}-${var.db_name}-replica-${count.index + 1}"
    Environment = var.environment
  }
}

# IAM role for enhanced monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${var.environment}-rds-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.environment}-rds-enhanced-monitoring"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${var.environment}-rds-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = {
    Name        = "${var.environment}-rds-cpu-utilization"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "database_connections" {
  alarm_name          = "${var.environment}-rds-database-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS database connections"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = {
    Name        = "${var.environment}-rds-database-connections"
    Environment = var.environment
  }
}

# SNS topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.environment}-rds-alerts"

  tags = {
    Name        = "${var.environment}-rds-alerts"
    Environment = var.environment
  }
}

# Outputs
output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "read_replica_endpoints" {
  description = "Read replica endpoints"
  value       = aws_db_instance.read_replica[*].endpoint
}
```

### Go Application with RDS

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

    "github.com/gin-gonic/gin"
    "github.com/lib/pq"
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

func NewDatabase() (*Database, error) {
    // Get database connection string from environment
    dbHost := os.Getenv("DB_HOST")
    dbPort := os.Getenv("DB_PORT")
    dbUser := os.Getenv("DB_USER")
    dbPassword := os.Getenv("DB_PASSWORD")
    dbName := os.Getenv("DB_NAME")

    if dbHost == "" {
        dbHost = "localhost"
    }
    if dbPort == "" {
        dbPort = "5432"
    }

    connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=require",
        dbHost, dbPort, dbUser, dbPassword, dbName)

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

func (d *Database) UpdateUser(ctx context.Context, id int, name, email string) (*User, error) {
    query := `
        UPDATE users
        SET name = $1, email = $2
        WHERE id = $3
        RETURNING id, name, email, created_at
    `

    var user User
    err := d.db.QueryRowContext(ctx, query, name, email, id).Scan(
        &user.ID, &user.Name, &user.Email, &user.CreatedAt,
    )
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user not found")
        }
        return nil, fmt.Errorf("failed to update user: %w", err)
    }

    return &user, nil
}

func (d *Database) DeleteUser(ctx context.Context, id int) error {
    query := `DELETE FROM users WHERE id = $1`

    result, err := d.db.ExecContext(ctx, query, id)
    if err != nil {
        return fmt.Errorf("failed to delete user: %w", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to get rows affected: %w", err)
    }

    if rowsAffected == 0 {
        return fmt.Errorf("user not found")
    }

    return nil
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

        api.PUT("/users/:id", func(c *gin.Context) {
            var req struct {
                ID int `uri:"id" binding:"required"`
            }

            if err := c.ShouldBindUri(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            var updateReq struct {
                Name  string `json:"name" binding:"required"`
                Email string `json:"email" binding:"required,email"`
            }

            if err := c.ShouldBindJSON(&updateReq); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            user, err := db.UpdateUser(c.Request.Context(), req.ID, updateReq.Name, updateReq.Email)
            if err != nil {
                if err.Error() == "user not found" {
                    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
                    return
                }
                log.Printf("Error updating user: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update user"})
                return
            }

            c.JSON(http.StatusOK, user)
        })

        api.DELETE("/users/:id", func(c *gin.Context) {
            var req struct {
                ID int `uri:"id" binding:"required"`
            }

            if err := c.ShouldBindUri(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            err := db.DeleteUser(c.Request.Context(), req.ID)
            if err != nil {
                if err.Error() == "user not found" {
                    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
                    return
                }
                log.Printf("Error deleting user: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete user"})
                return
            }

            c.JSON(http.StatusNoContent, nil)
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

### 1. Database Design

```sql
-- Create users table with proper indexing
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);
```

### 2. Connection Pooling

```go
// Configure connection pool
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)
```

### 3. Monitoring

```hcl
# CloudWatch alarms for RDS
resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "rds-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  threshold           = "80"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
}
```

## 🏢 Industry Insights

### RDS Usage Patterns

- **Multi-AZ Deployment**: High availability for production workloads
- **Read Replicas**: Scale read operations across multiple instances
- **Automated Backups**: Point-in-time recovery and automated snapshots
- **Performance Insights**: Monitor and optimize database performance

### Enterprise RDS Strategy

- **Database Migration**: Use AWS DMS for database migration
- **Cross-Region Replication**: Global read replicas for disaster recovery
- **Security**: Encryption at rest and in transit, VPC security groups
- **Cost Optimization**: Reserved instances, right-sizing, storage optimization

## 🎯 Interview Questions

### Basic Level

1. **What is AWS RDS?**

   - Managed relational database service
   - Automated backups and patching
   - Multi-AZ deployment for high availability

2. **What database engines does RDS support?**

   - MySQL, PostgreSQL, MariaDB
   - Oracle, SQL Server
   - Aurora (MySQL and PostgreSQL compatible)

3. **What are the benefits of using RDS?**
   - Managed service
   - Automated backups
   - High availability
   - Scalability

### Intermediate Level

4. **How do you implement high availability with RDS?**

   - Multi-AZ deployment
   - Read replicas
   - Automated failover
   - Cross-region replication

5. **How do you optimize RDS performance?**

   - Read replicas for read scaling
   - Performance Insights
   - Proper indexing
   - Connection pooling

6. **How do you secure RDS instances?**
   - VPC security groups
   - Encryption at rest and in transit
   - IAM database authentication
   - Network isolation

### Advanced Level

7. **How do you implement database migration to RDS?**

   - AWS DMS for data migration
   - Schema conversion tools
   - Blue-green deployment
   - Rollback strategies

8. **How do you handle RDS scaling?**

   - Vertical scaling (instance class)
   - Horizontal scaling (read replicas)
   - Storage scaling
   - Aurora Serverless

9. **How do you implement disaster recovery with RDS?**
   - Multi-AZ deployment
   - Cross-region read replicas
   - Automated backups
   - Point-in-time recovery
