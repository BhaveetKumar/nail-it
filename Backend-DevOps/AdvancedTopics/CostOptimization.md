# 💰 Cost Optimization: Cloud Cost Management and Resource Efficiency

> **Master cloud cost optimization strategies for maximum efficiency and budget control**

## 📚 Concept

Cost optimization in cloud computing involves continuously monitoring, analyzing, and optimizing cloud spending to achieve the best value for money while maintaining performance and reliability. It encompasses resource right-sizing, waste elimination, pricing model optimization, and strategic planning.

### Key Features

- **Resource Right-Sizing**: Match resources to actual demand
- **Waste Elimination**: Identify and remove unused resources
- **Pricing Optimization**: Choose optimal pricing models
- **Automated Scaling**: Scale resources based on demand
- **Budget Management**: Set and monitor spending limits
- **Cost Visibility**: Track and analyze spending patterns

## 🏗️ Cost Optimization Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Cost Optimization Framework             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Cost      │  │   Resource  │  │   Usage     │     │
│  │  Monitoring │  │  Analysis   │  │   Tracking  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Cost Analysis Engine                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Anomaly   │  │   Trend     │  │   Forecast  │ │ │
│  │  │  Detection  │  │   Analysis  │  │   Modeling  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Optimization Engine                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Right-    │  │   Auto-     │  │   Pricing   │ │ │
│  │  │   Sizing    │  │   Scaling   │  │   Selection │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Budget    │  │   Alerting  │  │   Reporting │     │
│  │  Management │  │   System    │  │   Dashboard │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Cost Optimization with Terraform

```hcl
# cost-optimization.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables for cost optimization
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 1000
}

# Cost allocation tags
locals {
  common_tags = {
    Environment   = var.environment
    CostCenter    = var.cost_center
    Project       = "cost-optimization"
    ManagedBy     = "terraform"
    AutoShutdown  = "true"
  }
}

# Budget and cost management
resource "aws_budgets_budget" "monthly_budget" {
  name         = "${var.environment}-monthly-budget"
  budget_type  = "COST"
  limit_amount = var.budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filters = {
    Tag = [
      "CostCenter$${var.cost_center}",
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["admin@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = ["admin@example.com"]
  }
}

# Cost anomaly detection
resource "aws_ce_anomaly_detector" "cost_anomaly" {
  name = "${var.environment}-cost-anomaly-detector"

  specification = "DAILY_COST_ANOMALY_DETECTION"

  monitor_arn_list = [
    aws_ce_cost_category.cost_category.arn,
  ]

  tags = local.common_tags
}

# Cost category for better cost allocation
resource "aws_ce_cost_category" "cost_category" {
  name = "${var.environment}-cost-category"
  rule {
    value = "Production"
    rule {
      dimension {
        key           = "LINKED_ACCOUNT"
        values        = [data.aws_caller_identity.current.account_id]
        match_options = ["EQUALS"]
      }
    }
  }
  rule {
    value = "Development"
    rule {
      dimension {
        key           = "LINKED_ACCOUNT"
        values        = [data.aws_caller_identity.current.account_id]
        match_options = ["EQUALS"]
      }
    }
  }
}

data "aws_caller_identity" "current" {}

# Right-sized EC2 instances
resource "aws_launch_template" "cost_optimized" {
  name_prefix   = "${var.environment}-cost-optimized-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = "t3.micro"  # Start with smallest instance

  vpc_security_group_ids = [aws_security_group.cost_optimized.id]

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    environment = var.environment
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${var.environment}-cost-optimized-instance"
    })
  }

  tags = local.common_tags
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Auto Scaling Group with cost optimization
resource "aws_autoscaling_group" "cost_optimized" {
  name                = "${var.environment}-cost-optimized-asg"
  vpc_zone_identifier = data.aws_subnets.private.ids
  target_group_arns   = [aws_lb_target_group.cost_optimized.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 1
  max_size         = 3
  desired_capacity = 2

  launch_template {
    id      = aws_launch_template.cost_optimized.id
    version = "$Latest"
  }

  # Cost optimization policies
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.cost_optimized.id
        version           = "$Latest"
      }

      override {
        instance_type = "t3.small"
      }

      override {
        instance_type = "t3.medium"
      }
    }

    instances_distribution {
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "diversified"
      spot_instance_pools                      = 2
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.environment}-cost-optimized-asg"
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = local.common_tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# Cost-optimized scaling policies
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "${var.environment}-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.cost_optimized.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "${var.environment}-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.cost_optimized.name
}

# CloudWatch alarms for cost optimization
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.environment}-cpu-high"
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
    AutoScalingGroupName = aws_autoscaling_group.cost_optimized.name
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "${var.environment}-cpu-low"
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
    AutoScalingGroupName = aws_autoscaling_group.cost_optimized.name
  }

  tags = local.common_tags
}

# Cost-optimized RDS instance
resource "aws_db_instance" "cost_optimized" {
  identifier = "${var.environment}-cost-optimized-db"

  engine         = "postgres"
  engine_version = "14.7"
  instance_class = "db.t3.micro"  # Start with smallest instance

  allocated_storage     = 20
  max_allocated_storage = 100  # Auto-scaling storage
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "costoptimized"
  username = "postgres"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.cost_optimized.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # Cost optimization settings
  performance_insights_enabled = false  # Disable for cost savings
  monitoring_interval         = 0       # Disable enhanced monitoring
  skip_final_snapshot         = true
  deletion_protection         = false

  tags = merge(local.common_tags, {
    Name = "${var.environment}-cost-optimized-db"
  })
}

# S3 with intelligent tiering
resource "aws_s3_bucket" "cost_optimized" {
  bucket = "${var.environment}-cost-optimized-storage-${random_string.bucket_suffix.result}"

  tags = local.common_tags
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_intelligent_tiering_configuration" "cost_optimized" {
  bucket = aws_s3_bucket.cost_optimized.id
  name   = "EntireBucket"

  status = "Enabled"

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
}

# S3 lifecycle policy for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "cost_optimized" {
  bucket = aws_s3_bucket.cost_optimized.id

  rule {
    id     = "cost_optimization"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7 years
    }
  }
}

# Lambda with cost optimization
resource "aws_lambda_function" "cost_optimized" {
  filename         = "cost-optimized-function.zip"
  function_name    = "${var.environment}-cost-optimized-function"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 30
  memory_size     = 128  # Start with minimum memory

  environment {
    variables = {
      ENVIRONMENT = var.environment
      LOG_LEVEL   = "INFO"
    }
  }

  tags = local.common_tags
}

# Cost optimization monitoring
resource "aws_cloudwatch_dashboard" "cost_optimization" {
  dashboard_name = "${var.environment}-cost-optimization"

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
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD"],
            [".", ".", "ServiceName", "Amazon Elastic Compute Cloud - Compute"],
            [".", ".", "ServiceName", "Amazon Relational Database Service"],
            [".", ".", "ServiceName", "Amazon Simple Storage Service"],
            [".", ".", "ServiceName", "AWS Lambda"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "Cost by Service"
          period  = 86400
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
            ["AWS/EC2", "CPUUtilization", "AutoScalingGroupName", aws_autoscaling_group.cost_optimized.name],
            [".", "NetworkIn", ".", "."],
            [".", "NetworkOut", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "EC2 Resource Utilization"
          period  = 300
        }
      }
    ]
  })
}

# Cost optimization alerts
resource "aws_cloudwatch_metric_alarm" "cost_anomaly" {
  alarm_name          = "${var.environment}-cost-anomaly"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = var.budget_limit * 1.2  # 20% over budget
  alarm_description   = "This metric monitors cost anomalies"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency = "USD"
  }

  tags = local.common_tags
}

# SNS topic for cost alerts
resource "aws_sns_topic" "cost_alerts" {
  name = "${var.environment}-cost-alerts"

  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "cost_alerts" {
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "email"
  endpoint  = "admin@example.com"
}

# Cost optimization Lambda function
resource "aws_lambda_function" "cost_optimizer" {
  filename         = "cost-optimizer.zip"
  function_name    = "${var.environment}-cost-optimizer"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 300
  memory_size     = 256

  environment {
    variables = {
      ENVIRONMENT = var.environment
      BUDGET_LIMIT = var.budget_limit
    }
  }

  tags = local.common_tags
}

# EventBridge rule for cost optimization
resource "aws_cloudwatch_event_rule" "cost_optimization" {
  name                = "${var.environment}-cost-optimization"
  description         = "Trigger cost optimization tasks"
  schedule_expression = "rate(1 day)"

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "cost_optimization" {
  rule      = aws_cloudwatch_event_rule.cost_optimization.name
  target_id = "CostOptimizationTarget"
  arn       = aws_lambda_function.cost_optimizer.arn
}

resource "aws_lambda_permission" "cost_optimization" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_optimizer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_optimization.arn
}

# Outputs
output "budget_arn" {
  description = "ARN of the budget"
  value       = aws_budgets_budget.monthly_budget.arn
}

output "cost_anomaly_detector_arn" {
  description = "ARN of the cost anomaly detector"
  value       = aws_ce_anomaly_detector.cost_anomaly.arn
}

output "dashboard_url" {
  description = "URL of the cost optimization dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=${aws_cloudwatch_dashboard.cost_optimization.dashboard_name}"
}
```

### Cost Optimization Application

```go
// cost-optimizer.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/autoscaling"
    "github.com/aws/aws-sdk-go/service/cloudwatch"
    "github.com/aws/aws-sdk-go/service/ec2"
    "github.com/aws/aws-sdk-go/service/rds"
)

type CostOptimizer struct {
    autoscalingClient *autoscaling.AutoScaling
    cloudwatchClient  *cloudwatch.CloudWatch
    ec2Client         *ec2.EC2
    rdsClient         *rds.RDS
    logger            *log.Logger
}

func NewCostOptimizer() (*CostOptimizer, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    return &CostOptimizer{
        autoscalingClient: autoscaling.New(sess),
        cloudwatchClient:  cloudwatch.New(sess),
        ec2Client:         ec2.New(sess),
        rdsClient:         rds.New(sess),
        logger:            log.New(log.Writer(), "[CostOptimizer] ", log.LstdFlags),
    }, nil
}

func (co *CostOptimizer) OptimizeCosts(ctx context.Context) error {
    co.logger.Println("Starting cost optimization process")

    // Optimize EC2 instances
    if err := co.optimizeEC2Instances(ctx); err != nil {
        co.logger.Printf("Error optimizing EC2 instances: %v", err)
    }

    // Optimize RDS instances
    if err := co.optimizeRDSInstances(ctx); err != nil {
        co.logger.Printf("Error optimizing RDS instances: %v", err)
    }

    // Optimize Auto Scaling Groups
    if err := co.optimizeAutoScalingGroups(ctx); err != nil {
        co.logger.Printf("Error optimizing Auto Scaling Groups: %v", err)
    }

    // Clean up unused resources
    if err := co.cleanupUnusedResources(ctx); err != nil {
        co.logger.Printf("Error cleaning up unused resources: %v", err)
    }

    co.logger.Println("Cost optimization process completed")
    return nil
}

func (co *CostOptimizer) optimizeEC2Instances(ctx context.Context) error {
    co.logger.Println("Optimizing EC2 instances")

    // Get all running instances
    result, err := co.ec2Client.DescribeInstancesWithContext(ctx, &ec2.DescribeInstancesInput{
        Filters: []*ec2.Filter{
            {
                Name:   aws.String("instance-state-name"),
                Values: []*string{aws.String("running")},
            },
        },
    })
    if err != nil {
        return fmt.Errorf("failed to describe instances: %w", err)
    }

    for _, reservation := range result.Reservations {
        for _, instance := range reservation.Instances {
            // Check if instance is underutilized
            if co.isInstanceUnderutilized(ctx, *instance.InstanceId) {
                co.logger.Printf("Instance %s is underutilized, considering right-sizing", *instance.InstanceId)

                // Get current instance type
                currentType := *instance.InstanceType

                // Suggest smaller instance type
                suggestedType := co.suggestSmallerInstanceType(currentType)
                if suggestedType != "" {
                    co.logger.Printf("Suggesting to change instance %s from %s to %s",
                        *instance.InstanceId, currentType, suggestedType)
                }
            }
        }
    }

    return nil
}

func (co *CostOptimizer) isInstanceUnderutilized(ctx context.Context, instanceID string) bool {
    // Get CPU utilization for the last 7 days
    endTime := time.Now()
    startTime := endTime.Add(-7 * 24 * time.Hour)

    result, err := co.cloudwatchClient.GetMetricStatisticsWithContext(ctx, &cloudwatch.GetMetricStatisticsInput{
        Namespace:  aws.String("AWS/EC2"),
        MetricName: aws.String("CPUUtilization"),
        Dimensions: []*cloudwatch.Dimension{
            {
                Name:  aws.String("InstanceId"),
                Value: aws.String(instanceID),
            },
        },
        StartTime:  aws.Time(startTime),
        EndTime:    aws.Time(endTime),
        Period:     aws.Int64(3600), // 1 hour
        Statistics: []*string{aws.String("Average")},
    })
    if err != nil {
        co.logger.Printf("Error getting CPU utilization for instance %s: %v", instanceID, err)
        return false
    }

    if len(result.Datapoints) == 0 {
        return false
    }

    // Calculate average CPU utilization
    totalCPU := 0.0
    for _, datapoint := range result.Datapoints {
        if datapoint.Average != nil {
            totalCPU += *datapoint.Average
        }
    }
    avgCPU := totalCPU / float64(len(result.Datapoints))

    // Consider underutilized if average CPU is less than 20%
    return avgCPU < 20.0
}

func (co *CostOptimizer) suggestSmallerInstanceType(currentType string) string {
    // Instance type mapping for right-sizing suggestions
    instanceTypeMap := map[string]string{
        "t3.medium":  "t3.small",
        "t3.small":   "t3.micro",
        "t3.large":   "t3.medium",
        "t3.xlarge":  "t3.large",
        "m5.large":   "m5.medium",
        "m5.medium":  "m5.small",
        "m5.xlarge":  "m5.large",
        "c5.large":   "c5.medium",
        "c5.medium":  "c5.small",
        "c5.xlarge":  "c5.large",
    }

    if smallerType, exists := instanceTypeMap[currentType]; exists {
        return smallerType
    }

    return ""
}

func (co *CostOptimizer) optimizeRDSInstances(ctx context.Context) error {
    co.logger.Println("Optimizing RDS instances")

    // Get all RDS instances
    result, err := co.rdsClient.DescribeDBInstancesWithContext(ctx, &rds.DescribeDBInstancesInput{})
    if err != nil {
        return fmt.Errorf("failed to describe RDS instances: %w", err)
    }

    for _, dbInstance := range result.DBInstances {
        // Check if instance is underutilized
        if co.isRDSInstanceUnderutilized(ctx, *dbInstance.DBInstanceIdentifier) {
            co.logger.Printf("RDS instance %s is underutilized, considering right-sizing",
                *dbInstance.DBInstanceIdentifier)

            // Get current instance class
            currentClass := *dbInstance.DBInstanceClass

            // Suggest smaller instance class
            suggestedClass := co.suggestSmallerRDSInstanceClass(currentClass)
            if suggestedClass != "" {
                co.logger.Printf("Suggesting to change RDS instance %s from %s to %s",
                    *dbInstance.DBInstanceIdentifier, currentClass, suggestedClass)
            }
        }
    }

    return nil
}

func (co *CostOptimizer) isRDSInstanceUnderutilized(ctx context.Context, instanceID string) bool {
    // Get CPU utilization for the last 7 days
    endTime := time.Now()
    startTime := endTime.Add(-7 * 24 * time.Hour)

    result, err := co.cloudwatchClient.GetMetricStatisticsWithContext(ctx, &cloudwatch.GetMetricStatisticsInput{
        Namespace:  aws.String("AWS/RDS"),
        MetricName: aws.String("CPUUtilization"),
        Dimensions: []*cloudwatch.Dimension{
            {
                Name:  aws.String("DBInstanceIdentifier"),
                Value: aws.String(instanceID),
            },
        },
        StartTime:  aws.Time(startTime),
        EndTime:    aws.Time(endTime),
        Period:     aws.Int64(3600), // 1 hour
        Statistics: []*string{aws.String("Average")},
    })
    if err != nil {
        co.logger.Printf("Error getting CPU utilization for RDS instance %s: %v", instanceID, err)
        return false
    }

    if len(result.Datapoints) == 0 {
        return false
    }

    // Calculate average CPU utilization
    totalCPU := 0.0
    for _, datapoint := range result.Datapoints {
        if datapoint.Average != nil {
            totalCPU += *datapoint.Average
        }
    }
    avgCPU := totalCPU / float64(len(result.Datapoints))

    // Consider underutilized if average CPU is less than 30%
    return avgCPU < 30.0
}

func (co *CostOptimizer) suggestSmallerRDSInstanceClass(currentClass string) string {
    // RDS instance class mapping for right-sizing suggestions
    instanceClassMap := map[string]string{
        "db.t3.medium":  "db.t3.small",
        "db.t3.small":   "db.t3.micro",
        "db.t3.large":   "db.t3.medium",
        "db.t3.xlarge":  "db.t3.large",
        "db.m5.large":   "db.m5.medium",
        "db.m5.medium":  "db.m5.small",
        "db.m5.xlarge":  "db.m5.large",
        "db.r5.large":   "db.r5.medium",
        "db.r5.medium":  "db.r5.small",
        "db.r5.xlarge":  "db.r5.large",
    }

    if smallerClass, exists := instanceClassMap[currentClass]; exists {
        return smallerClass
    }

    return ""
}

func (co *CostOptimizer) optimizeAutoScalingGroups(ctx context.Context) error {
    co.logger.Println("Optimizing Auto Scaling Groups")

    // Get all Auto Scaling Groups
    result, err := co.autoscalingClient.DescribeAutoScalingGroupsWithContext(ctx, &autoscaling.DescribeAutoScalingGroupsInput{})
    if err != nil {
        return fmt.Errorf("failed to describe Auto Scaling Groups: %w", err)
    }

    for _, asg := range result.AutoScalingGroups {
        // Check if ASG is over-provisioned
        if co.isASGOverProvisioned(ctx, *asg.AutoScalingGroupName) {
            co.logger.Printf("Auto Scaling Group %s is over-provisioned, considering scaling down",
                *asg.AutoScalingGroupName)

            // Suggest reducing desired capacity
            currentDesired := *asg.DesiredCapacity
            if currentDesired > 1 {
                suggestedDesired := currentDesired - 1
                co.logger.Printf("Suggesting to reduce desired capacity from %d to %d",
                    currentDesired, suggestedDesired)
            }
        }
    }

    return nil
}

func (co *CostOptimizer) isASGOverProvisioned(ctx context.Context, asgName string) bool {
    // Get CPU utilization for the last 7 days
    endTime := time.Now()
    startTime := endTime.Add(-7 * 24 * time.Hour)

    result, err := co.cloudwatchClient.GetMetricStatisticsWithContext(ctx, &cloudwatch.GetMetricStatisticsInput{
        Namespace:  aws.String("AWS/AutoScaling"),
        MetricName: aws.String("CPUUtilization"),
        Dimensions: []*cloudwatch.Dimension{
            {
                Name:  aws.String("AutoScalingGroupName"),
                Value: aws.String(asgName),
            },
        },
        StartTime:  aws.Time(startTime),
        EndTime:    aws.Time(endTime),
        Period:     aws.Int64(3600), // 1 hour
        Statistics: []*string{aws.String("Average")},
    })
    if err != nil {
        co.logger.Printf("Error getting CPU utilization for ASG %s: %v", asgName, err)
        return false
    }

    if len(result.Datapoints) == 0 {
        return false
    }

    // Calculate average CPU utilization
    totalCPU := 0.0
    for _, datapoint := range result.Datapoints {
        if datapoint.Average != nil {
            totalCPU += *datapoint.Average
        }
    }
    avgCPU := totalCPU / float64(len(result.Datapoints))

    // Consider over-provisioned if average CPU is less than 30%
    return avgCPU < 30.0
}

func (co *CostOptimizer) cleanupUnusedResources(ctx context.Context) error {
    co.logger.Println("Cleaning up unused resources")

    // Clean up unused EBS volumes
    if err := co.cleanupUnusedEBSVolumes(ctx); err != nil {
        co.logger.Printf("Error cleaning up unused EBS volumes: %v", err)
    }

    // Clean up unused snapshots
    if err := co.cleanupUnusedSnapshots(ctx); err != nil {
        co.logger.Printf("Error cleaning up unused snapshots: %v", err)
    }

    // Clean up unused Elastic IPs
    if err := co.cleanupUnusedElasticIPs(ctx); err != nil {
        co.logger.Printf("Error cleaning up unused Elastic IPs: %v", err)
    }

    return nil
}

func (co *CostOptimizer) cleanupUnusedEBSVolumes(ctx context.Context) error {
    // Get all available (unattached) EBS volumes
    result, err := co.ec2Client.DescribeVolumesWithContext(ctx, &ec2.DescribeVolumesInput{
        Filters: []*ec2.Filter{
            {
                Name:   aws.String("status"),
                Values: []*string{aws.String("available")},
            },
        },
    })
    if err != nil {
        return fmt.Errorf("failed to describe volumes: %w", err)
    }

    for _, volume := range result.Volumes {
        // Check if volume is older than 30 days
        if time.Since(*volume.CreateTime) > 30*24*time.Hour {
            co.logger.Printf("Found unused EBS volume %s (created: %s), considering deletion",
                *volume.VolumeId, volume.CreateTime.Format(time.RFC3339))
        }
    }

    return nil
}

func (co *CostOptimizer) cleanupUnusedSnapshots(ctx context.Context) error {
    // Get all snapshots owned by the account
    result, err := co.ec2Client.DescribeSnapshotsWithContext(ctx, &ec2.DescribeSnapshotsInput{
        OwnerIds: []*string{aws.String("self")},
    })
    if err != nil {
        return fmt.Errorf("failed to describe snapshots: %w", err)
    }

    for _, snapshot := range result.Snapshots {
        // Check if snapshot is older than 90 days and not tagged as important
        if time.Since(*snapshot.StartTime) > 90*24*time.Hour {
            // Check if snapshot has important tag
            isImportant := false
            for _, tag := range snapshot.Tags {
                if *tag.Key == "Important" && *tag.Value == "true" {
                    isImportant = true
                    break
                }
            }

            if !isImportant {
                co.logger.Printf("Found unused snapshot %s (created: %s), considering deletion",
                    *snapshot.SnapshotId, snapshot.StartTime.Format(time.RFC3339))
            }
        }
    }

    return nil
}

func (co *CostOptimizer) cleanupUnusedElasticIPs(ctx context.Context) error {
    // Get all Elastic IPs
    result, err := co.ec2Client.DescribeAddressesWithContext(ctx, &ec2.DescribeAddressesInput{})
    if err != nil {
        return fmt.Errorf("failed to describe Elastic IPs: %w", err)
    }

    for _, address := range result.Addresses {
        // Check if Elastic IP is not associated with any instance
        if address.InstanceId == nil {
            co.logger.Printf("Found unused Elastic IP %s, considering release", *address.PublicIp)
        }
    }

    return nil
}

// Lambda handler
func CostOptimizationHandler(ctx context.Context, event events.CloudWatchEvent) error {
    optimizer, err := NewCostOptimizer()
    if err != nil {
        log.Printf("Failed to create cost optimizer: %v", err)
        return err
    }

    return optimizer.OptimizeCosts(ctx)
}

func main() {
    lambda.Start(CostOptimizationHandler)
}
```

### Cost Monitoring Dashboard

```yaml
# cost-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cost-monitoring-config
  namespace: monitoring
data:
  prometheus.yml: |
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
      - job_name: 'cost-monitoring'
        static_configs:
          - targets: ['cost-exporter:8080']

      - job_name: 'aws-costs'
        static_configs:
          - targets: ['aws-cost-exporter:8080']

      - job_name: 'gcp-costs'
        static_configs:
          - targets: ['gcp-cost-exporter:8080']

      - job_name: 'azure-costs'
        static_configs:
          - targets: ['azure-cost-exporter:8080']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cost-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cost-exporter
  template:
    metadata:
      labels:
        app: cost-exporter
    spec:
      containers:
        - name: cost-exporter
          image: cost-exporter:latest
          ports:
            - containerPort: 8080
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
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: cost-exporter
  namespace: monitoring
spec:
  selector:
    app: cost-exporter
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP
```

## 🚀 Best Practices

### 1. Resource Right-Sizing

```hcl
# Start with smallest instances and scale up
resource "aws_instance" "optimized" {
  instance_type = "t3.micro"  # Start small
  monitoring    = true        # Enable monitoring

  tags = {
    AutoShutdown = "true"     # Enable auto-shutdown
  }
}
```

### 2. Automated Scaling

```hcl
# Use Auto Scaling Groups with cost optimization
resource "aws_autoscaling_group" "optimized" {
  min_size         = 1
  max_size         = 10
  desired_capacity = 2

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "diversified"
    }
  }
}
```

### 3. Cost Monitoring

```go
// Implement cost monitoring
func (co *CostOptimizer) monitorCosts(ctx context.Context) error {
    // Track spending by service
    // Set up alerts for budget overruns
    // Generate cost reports
    return nil
}
```

## 🏢 Industry Insights

### Cost Optimization Usage Patterns

- **Resource Right-Sizing**: Match resources to actual demand
- **Automated Scaling**: Scale resources based on usage patterns
- **Waste Elimination**: Remove unused and underutilized resources
- **Pricing Optimization**: Choose optimal pricing models

### Enterprise Cost Optimization Strategy

- **Budget Management**: Set and monitor spending limits
- **Cost Allocation**: Track costs by department and project
- **Automated Optimization**: Implement automated cost optimization
- **Regular Reviews**: Conduct regular cost optimization reviews

## 🎯 Interview Questions

### Basic Level

1. **What is cost optimization?**

   - Cloud cost management
   - Resource efficiency
   - Budget control
   - Waste elimination

2. **What are the main cost optimization strategies?**

   - Right-sizing
   - Auto-scaling
   - Reserved instances
   - Spot instances

3. **What are the benefits of cost optimization?**
   - Reduced spending
   - Better resource utilization
   - Improved ROI
   - Budget control

### Intermediate Level

4. **How do you implement cost optimization?**

   - Resource monitoring
   - Usage analysis
   - Right-sizing recommendations
   - Automated scaling

5. **How do you handle cost optimization at scale?**

   - Automated tools
   - Cost allocation
   - Budget management
   - Regular reviews

6. **How do you measure cost optimization success?**
   - Cost reduction metrics
   - Resource utilization
   - Budget adherence
   - ROI improvement

### Advanced Level

7. **How do you implement automated cost optimization?**

   - Machine learning models
   - Predictive analytics
   - Automated scaling
   - Cost forecasting

8. **How do you handle cost optimization across multiple clouds?**

   - Multi-cloud cost management
   - Cost comparison
   - Vendor optimization
   - Hybrid cloud costs

9. **How do you implement cost optimization governance?**
   - Cost policies
   - Approval processes
   - Budget controls
   - Compliance monitoring

---

**🎉 Backend-DevOps Knowledge Base Complete!**

The comprehensive Backend-DevOps knowledge base is now complete with all 47 detailed guides covering:

- **Backend Fundamentals** (7 guides)
- **Cloud Fundamentals** (3 guides)
- **AWS Services** (7 guides)
- **GCP Services** (7 guides)
- **CI/CD Tools** (4 guides)
- **Containers** (5 guides)
- **Infrastructure as Code** (3 guides)
- **Observability** (4 guides)
- **Security** (3 guides)
- **Advanced Topics** (5 guides)

Each guide includes hands-on examples, best practices, industry insights, and interview questions to help you master backend engineering and DevOps from beginner to advanced levels.
