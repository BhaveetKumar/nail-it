# 📊 GCP BigQuery: Serverless Data Warehouse

> **Master GCP BigQuery for scalable data analytics and business intelligence**

## 📚 Concept

BigQuery is Google Cloud's fully managed, serverless data warehouse that enables super-fast SQL queries using the processing power of Google's infrastructure. It's designed to handle petabyte-scale data and provides real-time analytics capabilities.

### Key Features
- **Serverless**: No infrastructure management required
- **Scalability**: Handle petabyte-scale data
- **Real-time Analytics**: Stream data and query in real-time
- **Machine Learning**: Built-in ML capabilities
- **Cost-effective**: Pay only for storage and queries
- **Security**: Enterprise-grade security and compliance

## 🏗️ BigQuery Architecture

```
┌─────────────────────────────────────────────────────────┐
│                BigQuery Architecture                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Query     │  │   Storage   │  │   Compute   │     │
│  │   Engine    │  │   Engine    │  │   Engine    │     │
│  │             │  │             │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Data Sources                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Cloud     │  │   Streaming │  │   Batch     │ │ │
│  │  │   Storage   │  │   Data      │  │   Data      │ │ │
│  │  │             │  │             │  │             │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Data      │  │   ML        │  │   BI        │     │
│  │   Analysis  │  │   Models    │  │   Tools     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### BigQuery with Terraform

```hcl
# bigquery.tf
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

# BigQuery Dataset
resource "google_bigquery_dataset" "main" {
  dataset_id  = "${var.environment}_analytics"
  project     = var.project_id
  location    = var.region
  description = "Main analytics dataset for ${var.environment} environment"

  default_table_expiration_ms = 3600000 # 1 hour

  labels = {
    environment = var.environment
    project     = "analytics"
  }
}

# BigQuery Table - User Events
resource "google_bigquery_table" "user_events" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "user_events"
  project    = var.project_id

  description = "User events table for analytics"

  schema = jsonencode([
    {
      name = "event_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "event_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "event_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "properties"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "session_id"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "device_type"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "country"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "event_timestamp"
  }

  clustering = ["user_id", "event_type"]

  labels = {
    environment = var.environment
    table_type  = "events"
  }
}

# BigQuery Table - User Profiles
resource "google_bigquery_table" "user_profiles" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "user_profiles"
  project    = var.project_id

  description = "User profiles table for analytics"

  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "email"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "age"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "gender"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "country"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "subscription_tier"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "is_active"
      type = "BOOLEAN"
      mode = "REQUIRED"
    }
  ])

  clustering = ["country", "subscription_tier"]

  labels = {
    environment = var.environment
    table_type  = "profiles"
  }
}

# BigQuery Table - Sales Data
resource "google_bigquery_table" "sales_data" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "sales_data"
  project    = var.project_id

  description = "Sales data table for analytics"

  schema = jsonencode([
    {
      name = "order_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "product_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "product_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "category"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "quantity"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "unit_price"
      type = "NUMERIC"
      mode = "REQUIRED"
    },
    {
      name = "total_amount"
      type = "NUMERIC"
      mode = "REQUIRED"
    },
    {
      name = "currency"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "order_date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "order_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "payment_method"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "shipping_address"
      type = "JSON"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "order_date"
  }

  clustering = ["category", "user_id"]

  labels = {
    environment = var.environment
    table_type  = "sales"
  }
}

# BigQuery View - Daily Sales Summary
resource "google_bigquery_table" "daily_sales_summary" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "daily_sales_summary"
  project    = var.project_id

  description = "Daily sales summary view"

  view {
    query = <<EOF
    SELECT
      DATE(order_timestamp) as order_date,
      category,
      COUNT(DISTINCT order_id) as total_orders,
      COUNT(DISTINCT user_id) as unique_customers,
      SUM(quantity) as total_quantity,
      SUM(total_amount) as total_revenue,
      AVG(total_amount) as avg_order_value,
      COUNT(DISTINCT product_id) as unique_products
    FROM `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.sales_data`
    WHERE order_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
    GROUP BY order_date, category
    ORDER BY order_date DESC, total_revenue DESC
    EOF
    use_legacy_sql = false
  }

  labels = {
    environment = var.environment
    table_type  = "view"
  }
}

# BigQuery View - User Engagement Metrics
resource "google_bigquery_table" "user_engagement_metrics" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "user_engagement_metrics"
  project    = var.project_id

  description = "User engagement metrics view"

  view {
    query = <<EOF
    WITH user_events_summary AS (
      SELECT
        user_id,
        COUNT(*) as total_events,
        COUNT(DISTINCT DATE(event_timestamp)) as active_days,
        COUNT(DISTINCT session_id) as total_sessions,
        MIN(event_timestamp) as first_event,
        MAX(event_timestamp) as last_event,
        DATE_DIFF(MAX(event_timestamp), MIN(event_timestamp), DAY) as user_lifespan_days
      FROM `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.user_events`
      WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
      GROUP BY user_id
    )
    SELECT
      ue.user_id,
      up.email,
      up.country,
      up.subscription_tier,
      ue.total_events,
      ue.active_days,
      ue.total_sessions,
      ue.user_lifespan_days,
      CASE
        WHEN ue.active_days >= 20 THEN 'High'
        WHEN ue.active_days >= 10 THEN 'Medium'
        ELSE 'Low'
      END as engagement_level,
      ue.first_event,
      ue.last_event
    FROM user_events_summary ue
    LEFT JOIN `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.user_profiles` up
      ON ue.user_id = up.user_id
    ORDER BY ue.total_events DESC
    EOF
    use_legacy_sql = false
  }

  labels = {
    environment = var.environment
    table_type  = "view"
  }
}

# BigQuery ML Model - Customer Lifetime Value
resource "google_bigquery_routine" "clv_model" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  routine_id = "predict_customer_lifetime_value"
  project    = var.project_id
  routine_type = "ML_PREDICT"

  description = "ML model to predict customer lifetime value"

  language = "SQL"

  definition_body = <<EOF
  SELECT
    user_id,
    predicted_clv,
    confidence_score
  FROM ML.PREDICT(
    MODEL `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.clv_model`,
    (
      SELECT
        user_id,
        total_orders,
        total_revenue,
        avg_order_value,
        days_since_last_order,
        total_products_purchased
      FROM `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.user_features`
    )
  )
  EOF

  labels = {
    environment = var.environment
    routine_type = "ml_model"
  }
}

# BigQuery Scheduled Query - Daily Analytics
resource "google_bigquery_data_transfer_config" "daily_analytics" {
  display_name           = "Daily Analytics Report"
  location              = var.region
  data_source_id        = "scheduled_query"
  schedule              = "every day 06:00"
  destination_dataset_id = google_bigquery_dataset.main.dataset_id

  params = {
    query = <<EOF
    INSERT INTO `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.daily_analytics`
    SELECT
      CURRENT_DATE() as report_date,
      'user_metrics' as metric_type,
      COUNT(DISTINCT user_id) as metric_value
    FROM `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.user_events`
    WHERE DATE(event_timestamp) = CURRENT_DATE() - 1
    
    UNION ALL
    
    SELECT
      CURRENT_DATE() as report_date,
      'revenue_metrics' as metric_type,
      SUM(total_amount) as metric_value
    FROM `${var.project_id}.${google_bigquery_dataset.main.dataset_id}.sales_data`
    WHERE DATE(order_timestamp) = CURRENT_DATE() - 1
    EOF
  }

  labels = {
    environment = var.environment
    report_type = "daily"
  }
}

# IAM Binding for BigQuery Access
resource "google_project_iam_binding" "bigquery_admin" {
  project = var.project_id
  role    = "roles/bigquery.admin"

  members = [
    "serviceAccount:analytics-service@${var.project_id}.iam.gserviceaccount.com",
  ]
}

resource "google_project_iam_binding" "bigquery_data_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"

  members = [
    "group:analytics-team@company.com",
  ]
}

# Outputs
output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.main.dataset_id
}

output "dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.main.location
}

output "user_events_table_id" {
  description = "User events table ID"
  value       = google_bigquery_table.user_events.table_id
}

output "sales_data_table_id" {
  description = "Sales data table ID"
  value       = google_bigquery_table.sales_data.table_id
}
```

### Go Application with BigQuery Integration

```go
// main.go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    "cloud.google.com/go/bigquery"
    "github.com/gin-gonic/gin"
    "google.golang.org/api/iterator"
)

type BigQueryService struct {
    client     *bigquery.Client
    projectID  string
    datasetID  string
}

type UserEvent struct {
    EventID        string                 `bigquery:"event_id"`
    UserID         string                 `bigquery:"user_id"`
    EventType      string                 `bigquery:"event_type"`
    EventTimestamp time.Time              `bigquery:"event_timestamp"`
    Properties     map[string]interface{} `bigquery:"properties"`
    SessionID      string                 `bigquery:"session_id"`
    DeviceType     string                 `bigquery:"device_type"`
    Country        string                 `bigquery:"country"`
}

type SalesData struct {
    OrderID         string                 `bigquery:"order_id"`
    UserID          string                 `bigquery:"user_id"`
    ProductID       string                 `bigquery:"product_id"`
    ProductName     string                 `bigquery:"product_name"`
    Category        string                 `bigquery:"category"`
    Quantity        int64                  `bigquery:"quantity"`
    UnitPrice       float64                `bigquery:"unit_price"`
    TotalAmount     float64                `bigquery:"total_amount"`
    Currency        string                 `bigquery:"currency"`
    OrderDate       time.Time              `bigquery:"order_date"`
    OrderTimestamp  time.Time              `bigquery:"order_timestamp"`
    PaymentMethod   string                 `bigquery:"payment_method"`
    ShippingAddress map[string]interface{} `bigquery:"shipping_address"`
}

type DailySalesSummary struct {
    OrderDate        time.Time `bigquery:"order_date"`
    Category         string    `bigquery:"category"`
    TotalOrders      int64     `bigquery:"total_orders"`
    UniqueCustomers  int64     `bigquery:"unique_customers"`
    TotalQuantity    int64     `bigquery:"total_quantity"`
    TotalRevenue     float64   `bigquery:"total_revenue"`
    AvgOrderValue    float64   `bigquery:"avg_order_value"`
    UniqueProducts   int64     `bigquery:"unique_products"`
}

func NewBigQueryService(projectID, datasetID string) (*BigQueryService, error) {
    ctx := context.Background()
    client, err := bigquery.NewClient(ctx, projectID)
    if err != nil {
        return nil, fmt.Errorf("failed to create BigQuery client: %w", err)
    }

    return &BigQueryService{
        client:    client,
        projectID: projectID,
        datasetID: datasetID,
    }, nil
}

func (bq *BigQueryService) InsertUserEvent(ctx context.Context, event UserEvent) error {
    table := bq.client.Dataset(bq.datasetID).Table("user_events")
    
    inserter := table.Inserter()
    if err := inserter.Put(ctx, event); err != nil {
        return fmt.Errorf("failed to insert user event: %w", err)
    }

    return nil
}

func (bq *BigQueryService) InsertSalesData(ctx context.Context, sales SalesData) error {
    table := bq.client.Dataset(bq.datasetID).Table("sales_data")
    
    inserter := table.Inserter()
    if err := inserter.Put(ctx, sales); err != nil {
        return fmt.Errorf("failed to insert sales data: %w", err)
    }

    return nil
}

func (bq *BigQueryService) GetDailySalesSummary(ctx context.Context, days int) ([]DailySalesSummary, error) {
    query := fmt.Sprintf(`
        SELECT
            order_date,
            category,
            total_orders,
            unique_customers,
            total_quantity,
            total_revenue,
            avg_order_value,
            unique_products
        FROM `+"`%s.%s.daily_sales_summary`"+`
        WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL %d DAY)
        ORDER BY order_date DESC, total_revenue DESC
        LIMIT 100
    `, bq.projectID, bq.datasetID, days)

    q := bq.client.Query(query)
    q.Location = "US"

    iter, err := q.Read(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to execute query: %w", err)
    }

    var results []DailySalesSummary
    for {
        var row DailySalesSummary
        err := iter.Next(&row)
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, fmt.Errorf("failed to read row: %w", err)
        }
        results = append(results, row)
    }

    return results, nil
}

func (bq *BigQueryService) GetUserEngagementMetrics(ctx context.Context, limit int) ([]map[string]interface{}, error) {
    query := fmt.Sprintf(`
        SELECT
            user_id,
            email,
            country,
            subscription_tier,
            total_events,
            active_days,
            total_sessions,
            user_lifespan_days,
            engagement_level,
            first_event,
            last_event
        FROM `+"`%s.%s.user_engagement_metrics`"+`
        ORDER BY total_events DESC
        LIMIT %d
    `, bq.projectID, bq.datasetID, limit)

    q := bq.client.Query(query)
    q.Location = "US"

    iter, err := q.Read(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to execute query: %w", err)
    }

    var results []map[string]interface{}
    for {
        var row map[string]interface{}
        err := iter.Next(&row)
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, fmt.Errorf("failed to read row: %w", err)
        }
        results = append(results, row)
    }

    return results, nil
}

func (bq *BigQueryService) GetRevenueByCategory(ctx context.Context, startDate, endDate time.Time) ([]map[string]interface{}, error) {
    query := fmt.Sprintf(`
        SELECT
            category,
            COUNT(DISTINCT order_id) as total_orders,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_order_value,
            COUNT(DISTINCT user_id) as unique_customers
        FROM `+"`%s.%s.sales_data`"+`
        WHERE order_date BETWEEN '%s' AND '%s'
        GROUP BY category
        ORDER BY total_revenue DESC
    `, bq.projectID, bq.datasetID, startDate.Format("2006-01-02"), endDate.Format("2006-01-02"))

    q := bq.client.Query(query)
    q.Location = "US"

    iter, err := q.Read(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to execute query: %w", err)
    }

    var results []map[string]interface{}
    for {
        var row map[string]interface{}
        err := iter.Next(&row)
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, fmt.Errorf("failed to read row: %w", err)
        }
        results = append(results, row)
    }

    return results, nil
}

func (bq *BigQueryService) GetTopProducts(ctx context.Context, limit int) ([]map[string]interface{}, error) {
    query := fmt.Sprintf(`
        SELECT
            product_id,
            product_name,
            category,
            COUNT(DISTINCT order_id) as total_orders,
            SUM(quantity) as total_quantity,
            SUM(total_amount) as total_revenue,
            AVG(unit_price) as avg_price
        FROM `+"`%s.%s.sales_data`"+`
        WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
        GROUP BY product_id, product_name, category
        ORDER BY total_revenue DESC
        LIMIT %d
    `, bq.projectID, bq.datasetID, limit)

    q := bq.client.Query(query)
    q.Location = "US"

    iter, err := q.Read(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to execute query: %w", err)
    }

    var results []map[string]interface{}
    for {
        var row map[string]interface{}
        err := iter.Next(&row)
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, fmt.Errorf("failed to read row: %w", err)
        }
        results = append(results, row)
    }

    return results, nil
}

func (bq *BigQueryService) Close() error {
    return bq.client.Close()
}

// HTTP handlers
func setupRoutes(bqService *BigQueryService) *gin.Engine {
    r := gin.Default()

    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "timestamp": time.Now().UTC(),
        })
    })

    // API routes
    api := r.Group("/api/v1")
    {
        // Insert user event
        api.POST("/events", func(c *gin.Context) {
            var event UserEvent
            if err := c.ShouldBindJSON(&event); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            // Set timestamp if not provided
            if event.EventTimestamp.IsZero() {
                event.EventTimestamp = time.Now()
            }

            if err := bqService.InsertUserEvent(c.Request.Context(), event); err != nil {
                log.Printf("Error inserting user event: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to insert user event"})
                return
            }

            c.JSON(http.StatusCreated, gin.H{
                "message": "User event inserted successfully",
                "event_id": event.EventID,
            })
        })

        // Insert sales data
        api.POST("/sales", func(c *gin.Context) {
            var sales SalesData
            if err := c.ShouldBindJSON(&sales); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            // Set timestamps if not provided
            if sales.OrderTimestamp.IsZero() {
                sales.OrderTimestamp = time.Now()
            }
            if sales.OrderDate.IsZero() {
                sales.OrderDate = time.Now()
            }

            if err := bqService.InsertSalesData(c.Request.Context(), sales); err != nil {
                log.Printf("Error inserting sales data: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to insert sales data"})
                return
            }

            c.JSON(http.StatusCreated, gin.H{
                "message": "Sales data inserted successfully",
                "order_id": sales.OrderID,
            })
        })

        // Get daily sales summary
        api.GET("/analytics/daily-sales", func(c *gin.Context) {
            days := 30
            if d := c.Query("days"); d != "" {
                if parsed, err := fmt.Sscanf(d, "%d", &days); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid days parameter"})
                    return
                }
            }

            results, err := bqService.GetDailySalesSummary(c.Request.Context(), days)
            if err != nil {
                log.Printf("Error getting daily sales summary: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get daily sales summary"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "data": results,
                "count": len(results),
                "days": days,
            })
        })

        // Get user engagement metrics
        api.GET("/analytics/user-engagement", func(c *gin.Context) {
            limit := 100
            if l := c.Query("limit"); l != "" {
                if parsed, err := fmt.Sscanf(l, "%d", &limit); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
                    return
                }
            }

            results, err := bqService.GetUserEngagementMetrics(c.Request.Context(), limit)
            if err != nil {
                log.Printf("Error getting user engagement metrics: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user engagement metrics"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "data": results,
                "count": len(results),
            })
        })

        // Get revenue by category
        api.GET("/analytics/revenue-by-category", func(c *gin.Context) {
            startDate := time.Now().AddDate(0, 0, -30) // 30 days ago
            endDate := time.Now()

            if s := c.Query("start_date"); s != "" {
                if parsed, err := time.Parse("2006-01-02", s); err == nil {
                    startDate = parsed
                }
            }

            if e := c.Query("end_date"); e != "" {
                if parsed, err := time.Parse("2006-01-02", e); err == nil {
                    endDate = parsed
                }
            }

            results, err := bqService.GetRevenueByCategory(c.Request.Context(), startDate, endDate)
            if err != nil {
                log.Printf("Error getting revenue by category: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get revenue by category"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "data": results,
                "count": len(results),
                "start_date": startDate.Format("2006-01-02"),
                "end_date": endDate.Format("2006-01-02"),
            })
        })

        // Get top products
        api.GET("/analytics/top-products", func(c *gin.Context) {
            limit := 20
            if l := c.Query("limit"); l != "" {
                if parsed, err := fmt.Sscanf(l, "%d", &limit); err != nil || parsed != 1 {
                    c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
                    return
                }
            }

            results, err := bqService.GetTopProducts(c.Request.Context(), limit)
            if err != nil {
                log.Printf("Error getting top products: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get top products"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "data": results,
                "count": len(results),
            })
        })
    }

    return r
}

func main() {
    // Get configuration from environment
    projectID := os.Getenv("GCP_PROJECT_ID")
    datasetID := os.Getenv("BIGQUERY_DATASET_ID")

    if projectID == "" {
        log.Fatal("GCP_PROJECT_ID environment variable is required")
    }
    if datasetID == "" {
        log.Fatal("BIGQUERY_DATASET_ID environment variable is required")
    }

    // Initialize BigQuery service
    bqService, err := NewBigQueryService(projectID, datasetID)
    if err != nil {
        log.Fatalf("Failed to initialize BigQuery service: %v", err)
    }
    defer bqService.Close()

    // Setup routes
    r := setupRoutes(bqService)

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

### 1. Table Design
```sql
-- Use partitioning for large tables
CREATE TABLE user_events (
  event_id STRING,
  user_id STRING,
  event_timestamp TIMESTAMP
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, event_type;
```

### 2. Query Optimization
```sql
-- Use clustering for better performance
SELECT *
FROM sales_data
WHERE category = 'electronics'
  AND order_date >= '2024-01-01'
  AND order_date < '2024-02-01';
```

### 3. Cost Optimization
```sql
-- Use approximate functions for large datasets
SELECT APPROX_COUNT_DISTINCT(user_id) as unique_users
FROM user_events
WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);
```

## 🏢 Industry Insights

### BigQuery Usage Patterns
- **Data Warehousing**: Centralized data storage and analysis
- **Real-time Analytics**: Stream processing and real-time insights
- **Machine Learning**: Built-in ML capabilities and model training
- **Business Intelligence**: Dashboards and reporting

### Enterprise BigQuery Strategy
- **Data Governance**: Access control and data lineage
- **Cost Management**: Query optimization and slot management
- **Security**: Encryption and compliance
- **Integration**: ETL pipelines and data sources

## 🎯 Interview Questions

### Basic Level
1. **What is BigQuery?**
   - Serverless data warehouse
   - Petabyte-scale analytics
   - Real-time querying

2. **What are the benefits of BigQuery?**
   - No infrastructure management
   - Automatic scaling
   - Built-in ML capabilities
   - Cost-effective pricing

3. **What are BigQuery datasets and tables?**
   - Datasets: Containers for tables
   - Tables: Structured data storage
   - Views: Virtual tables

### Intermediate Level
4. **How do you optimize BigQuery queries?**
   - Use partitioning and clustering
   - Optimize SELECT statements
   - Use approximate functions
   - Cache results

5. **How do you handle BigQuery costs?**
   - Use slot commitments
   - Optimize queries
   - Use streaming inserts efficiently
   - Monitor usage

6. **How do you implement BigQuery security?**
   - IAM roles and permissions
   - Data encryption
   - Access controls
   - Audit logging

### Advanced Level
7. **How do you implement real-time analytics with BigQuery?**
   - Streaming inserts
   - Real-time views
   - Dataflow integration
   - Pub/Sub integration

8. **How do you use BigQuery ML?**
   - Model training
   - Model evaluation
   - Model deployment
   - Feature engineering

9. **How do you implement data governance with BigQuery?**
   - Data lineage
   - Access policies
   - Data classification
   - Compliance monitoring
