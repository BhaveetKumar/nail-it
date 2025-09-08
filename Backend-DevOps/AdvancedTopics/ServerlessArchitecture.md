# ⚡ Serverless Architecture: Function as a Service and Event-Driven Computing

> **Master serverless computing for auto-scaling, event-driven applications, and cost optimization**

## 📚 Concept

Serverless architecture is a cloud computing execution model where the cloud provider automatically manages the infrastructure and automatically allocates resources to run code. Developers focus on writing business logic while the platform handles scaling, provisioning, and maintenance of servers.

### Key Features
- **Auto-Scaling**: Automatically scale based on demand
- **Event-Driven**: Triggered by events and requests
- **Pay-per-Use**: Only pay for actual execution time
- **No Server Management**: Cloud provider handles infrastructure
- **Stateless**: Functions are stateless and ephemeral
- **Microservices**: Natural fit for microservices architecture

## 🏗️ Serverless Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Serverless Architecture                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   HTTP      │  │   Events    │  │   Schedules │     │
│  │  Requests   │  │   (S3, SQS) │  │   (Cron)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              API Gateway                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Routing   │  │   Auth      │  │   Rate      │ │ │
│  │  │   & Load    │  │   & Authz   │  │   Limiting  │ │ │
│  │  │   Balancing │  │             │  │             │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Function Runtime                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   AWS       │  │   GCP       │  │   Azure     │ │ │
│  │  │   Lambda    │  │   Functions │  │   Functions │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Database  │  │   Storage   │  │   External  │     │
│  │   Services  │  │   Services  │  │   APIs      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### AWS Lambda Functions

```go
// main.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/dynamodb"
    "github.com/aws/aws-sdk-go/service/s3"
    "github.com/aws/aws-sdk-go/service/sqs"
)

type User struct {
    ID       string `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    Created  string `json:"created"`
}

type CreateUserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

type CreateUserResponse struct {
    User    *User  `json:"user,omitempty"`
    Error   string `json:"error,omitempty"`
    Message string `json:"message,omitempty"`
}

// Create User Lambda Function
func CreateUserHandler(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // Parse request body
    var createReq CreateUserRequest
    if err := json.Unmarshal([]byte(request.Body), &createReq); err != nil {
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusBadRequest,
            Body:       `{"error": "Invalid request body"}`,
        }, nil
    }

    // Validate input
    if createReq.Name == "" || createReq.Email == "" {
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusBadRequest,
            Body:       `{"error": "Name and email are required"}`,
        }, nil
    }

    // Create user
    user, err := createUser(ctx, createReq.Name, createReq.Email)
    if err != nil {
        log.Printf("Error creating user: %v", err)
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusInternalServerError,
            Body:       `{"error": "Failed to create user"}`,
        }, nil
    }

    // Send welcome email
    if err := sendWelcomeEmail(ctx, user); err != nil {
        log.Printf("Error sending welcome email: %v", err)
        // Don't fail the request, just log the error
    }

    // Return response
    response := CreateUserResponse{
        User:    user,
        Message: "User created successfully",
    }

    responseBody, _ := json.Marshal(response)
    return events.APIGatewayProxyResponse{
        StatusCode: http.StatusCreated,
        Headers: map[string]string{
            "Content-Type": "application/json",
        },
        Body: string(responseBody),
    }, nil
}

func createUser(ctx context.Context, name, email string) (*User, error) {
    // Create DynamoDB session
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    dynamoDB := dynamodb.New(sess)

    // Generate user ID
    userID := generateUserID()

    // Create user item
    user := &User{
        ID:      userID,
        Name:    name,
        Email:   email,
        Created: time.Now().UTC().Format(time.RFC3339),
    }

    // Save to DynamoDB
    item := map[string]*dynamodb.AttributeValue{
        "id": {
            S: aws.String(user.ID),
        },
        "name": {
            S: aws.String(user.Name),
        },
        "email": {
            S: aws.String(user.Email),
        },
        "created": {
            S: aws.String(user.Created),
        },
    }

    _, err = dynamoDB.PutItemWithContext(ctx, &dynamodb.PutItemInput{
        TableName: aws.String("users"),
        Item:      item,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to save user to DynamoDB: %w", err)
    }

    return user, nil
}

func sendWelcomeEmail(ctx context.Context, user *User) error {
    // Create SQS session
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return fmt.Errorf("failed to create session: %w", err)
    }

    sqs := sqs.New(sess)

    // Create email message
    emailMessage := map[string]interface{}{
        "to":      user.Email,
        "subject": "Welcome to our platform!",
        "body":    fmt.Sprintf("Hello %s, welcome to our platform!", user.Name),
        "type":    "welcome",
    }

    messageBody, err := json.Marshal(emailMessage)
    if err != nil {
        return fmt.Errorf("failed to marshal email message: %w", err)
    }

    // Send to SQS queue
    _, err = sqs.SendMessageWithContext(ctx, &sqs.SendMessageInput{
        QueueUrl:    aws.String("https://sqs.us-east-1.amazonaws.com/123456789012/email-queue"),
        MessageBody: aws.String(string(messageBody)),
    })
    if err != nil {
        return fmt.Errorf("failed to send message to SQS: %w", err)
    }

    return nil
}

func generateUserID() string {
    return fmt.Sprintf("user_%d", time.Now().UnixNano())
}

// Get User Lambda Function
func GetUserHandler(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // Get user ID from path parameters
    userID := request.PathParameters["id"]
    if userID == "" {
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusBadRequest,
            Body:       `{"error": "User ID is required"}`,
        }, nil
    }

    // Get user from DynamoDB
    user, err := getUser(ctx, userID)
    if err != nil {
        log.Printf("Error getting user: %v", err)
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusInternalServerError,
            Body:       `{"error": "Failed to get user"}`,
        }, nil
    }

    if user == nil {
        return events.APIGatewayProxyResponse{
            StatusCode: http.StatusNotFound,
            Body:       `{"error": "User not found"}`,
        }, nil
    }

    // Return response
    responseBody, _ := json.Marshal(user)
    return events.APIGatewayProxyResponse{
        StatusCode: http.StatusOK,
        Headers: map[string]string{
            "Content-Type": "application/json",
        },
        Body: string(responseBody),
    }, nil
}

func getUser(ctx context.Context, userID string) (*User, error) {
    // Create DynamoDB session
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    dynamoDB := dynamodb.New(sess)

    // Get user from DynamoDB
    result, err := dynamoDB.GetItemWithContext(ctx, &dynamodb.GetItemInput{
        TableName: aws.String("users"),
        Key: map[string]*dynamodb.AttributeValue{
            "id": {
                S: aws.String(userID),
            },
        },
    })
    if err != nil {
        return nil, fmt.Errorf("failed to get user from DynamoDB: %w", err)
    }

    if result.Item == nil {
        return nil, nil // User not found
    }

    // Parse user from DynamoDB item
    user := &User{
        ID:      *result.Item["id"].S,
        Name:    *result.Item["name"].S,
        Email:   *result.Item["email"].S,
        Created: *result.Item["created"].S,
    }

    return user, nil
}

// S3 Event Handler
func S3EventHandler(ctx context.Context, event events.S3Event) error {
    for _, record := range event.Records {
        bucket := record.S3.Bucket.Name
        key := record.S3.Object.Key

        log.Printf("Processing S3 event: bucket=%s, key=%s", bucket, key)

        // Process the uploaded file
        if err := processUploadedFile(ctx, bucket, key); err != nil {
            log.Printf("Error processing file %s: %v", key, err)
            return err
        }
    }

    return nil
}

func processUploadedFile(ctx context.Context, bucket, key string) error {
    // Create S3 session
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return fmt.Errorf("failed to create session: %w", err)
    }

    s3Client := s3.New(sess)

    // Get object from S3
    result, err := s3Client.GetObjectWithContext(ctx, &s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return fmt.Errorf("failed to get object from S3: %w", err)
    }
    defer result.Body.Close()

    // Process the file content
    // For example, resize images, extract text, etc.
    log.Printf("Processing file: %s", key)

    // Create processed file
    processedKey := fmt.Sprintf("processed/%s", key)
    _, err = s3Client.PutObjectWithContext(ctx, &s3.PutObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(processedKey),
        Body:   result.Body,
    })
    if err != nil {
        return fmt.Errorf("failed to upload processed file: %w", err)
    }

    log.Printf("File processed successfully: %s", processedKey)
    return nil
}

// SQS Event Handler
func SQSEventHandler(ctx context.Context, event events.SQSEvent) error {
    for _, record := range event.Records {
        log.Printf("Processing SQS message: %s", record.Body)

        // Parse message
        var message map[string]interface{}
        if err := json.Unmarshal([]byte(record.Body), &message); err != nil {
            log.Printf("Error parsing SQS message: %v", err)
            continue
        }

        // Process message based on type
        messageType, ok := message["type"].(string)
        if !ok {
            log.Printf("Invalid message type")
            continue
        }

        switch messageType {
        case "welcome":
            if err := processWelcomeEmail(ctx, message); err != nil {
                log.Printf("Error processing welcome email: %v", err)
                return err
            }
        case "notification":
            if err := processNotification(ctx, message); err != nil {
                log.Printf("Error processing notification: %v", err)
                return err
            }
        default:
            log.Printf("Unknown message type: %s", messageType)
        }
    }

    return nil
}

func processWelcomeEmail(ctx context.Context, message map[string]interface{}) error {
    to, _ := message["to"].(string)
    subject, _ := message["subject"].(string)
    body, _ := message["body"].(string)

    log.Printf("Sending welcome email to: %s", to)
    // Implement email sending logic here
    // For example, using SES, SendGrid, etc.

    return nil
}

func processNotification(ctx context.Context, message map[string]interface{}) error {
    // Process notification logic
    log.Printf("Processing notification: %v", message)
    return nil
}

// Scheduled Function
func ScheduledFunction(ctx context.Context, event events.CloudWatchEvent) error {
    log.Printf("Scheduled function executed at: %s", time.Now().UTC())

    // Perform scheduled tasks
    if err := cleanupExpiredData(ctx); err != nil {
        log.Printf("Error cleaning up expired data: %v", err)
        return err
    }

    if err := generateReports(ctx); err != nil {
        log.Printf("Error generating reports: %v", err)
        return err
    }

    return nil
}

func cleanupExpiredData(ctx context.Context) error {
    // Clean up expired data from DynamoDB
    log.Printf("Cleaning up expired data...")
    // Implement cleanup logic
    return nil
}

func generateReports(ctx context.Context) error {
    // Generate daily reports
    log.Printf("Generating reports...")
    // Implement report generation logic
    return nil
}

// Main function for Lambda
func main() {
    // Determine which function to run based on environment variable
    functionName := os.Getenv("FUNCTION_NAME")
    
    switch functionName {
    case "create-user":
        lambda.Start(CreateUserHandler)
    case "get-user":
        lambda.Start(GetUserHandler)
    case "s3-event":
        lambda.Start(S3EventHandler)
    case "sqs-event":
        lambda.Start(SQSEventHandler)
    case "scheduled":
        lambda.Start(ScheduledFunction)
    default:
        log.Fatal("Unknown function name:", functionName)
    }
}
```

### Serverless Infrastructure with Terraform

```hcl
# serverless.tf
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
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "serverless-app"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# DynamoDB Table
resource "aws_dynamodb_table" "users" {
  name           = "${var.project_name}-users"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"

  attribute {
    name = "id"
    type = "S"
  }

  tags = {
    Name        = "${var.project_name}-users"
    Environment = var.environment
  }
}

# S3 Bucket for file storage
resource "aws_s3_bucket" "app_storage" {
  bucket = "${var.project_name}-storage-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-storage"
    Environment = var.environment
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_notification" "app_storage_notification" {
  bucket = aws_s3_bucket.app_storage.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.s3_processor.arn
    events              = ["s3:ObjectCreated:*"]
  }
}

# SQS Queue for email processing
resource "aws_sqs_queue" "email_queue" {
  name = "${var.project_name}-email-queue"

  tags = {
    Name        = "${var.project_name}-email-queue"
    Environment = var.environment
  }
}

# IAM Role for Lambda functions
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_role.name
}

resource "aws_iam_role_policy" "lambda_dynamodb_policy" {
  name = "${var.project_name}-lambda-dynamodb-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.users.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_s3_policy" {
  name = "${var.project_name}-lambda-s3-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.app_storage.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_sqs_policy" {
  name = "${var.project_name}-lambda-sqs-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.email_queue.arn
      }
    ]
  })
}

# Lambda function for user management
resource "aws_lambda_function" "user_manager" {
  filename         = "user-manager.zip"
  function_name    = "${var.project_name}-user-manager"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 30

  environment {
    variables = {
      FUNCTION_NAME = "user-manager"
      DYNAMODB_TABLE = aws_dynamodb_table.users.name
      SQS_QUEUE_URL = aws_sqs_queue.email_queue.url
    }
  }

  tags = {
    Name        = "${var.project_name}-user-manager"
    Environment = var.environment
  }
}

# Lambda function for S3 processing
resource "aws_lambda_function" "s3_processor" {
  filename         = "s3-processor.zip"
  function_name    = "${var.project_name}-s3-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 60

  environment {
    variables = {
      FUNCTION_NAME = "s3-processor"
      S3_BUCKET = aws_s3_bucket.app_storage.bucket
    }
  }

  tags = {
    Name        = "${var.project_name}-s3-processor"
    Environment = var.environment
  }
}

# Lambda function for SQS processing
resource "aws_lambda_function" "sqs_processor" {
  filename         = "sqs-processor.zip"
  function_name    = "${var.project_name}-sqs-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 30

  environment {
    variables = {
      FUNCTION_NAME = "sqs-processor"
    }
  }

  tags = {
    Name        = "${var.project_name}-sqs-processor"
    Environment = var.environment
  }
}

# Lambda function for scheduled tasks
resource "aws_lambda_function" "scheduled_tasks" {
  filename         = "scheduled-tasks.zip"
  function_name    = "${var.project_name}-scheduled-tasks"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main"
  runtime         = "go1.x"
  timeout         = 300

  environment {
    variables = {
      FUNCTION_NAME = "scheduled-tasks"
      DYNAMODB_TABLE = aws_dynamodb_table.users.name
    }
  }

  tags = {
    Name        = "${var.project_name}-scheduled-tasks"
    Environment = var.environment
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "serverless_api" {
  name        = "${var.project_name}-api"
  description = "Serverless API for ${var.project_name}"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = {
    Name        = "${var.project_name}-api"
    Environment = var.environment
  }
}

resource "aws_api_gateway_resource" "users" {
  rest_api_id = aws_api_gateway_rest_api.serverless_api.id
  parent_id   = aws_api_gateway_rest_api.serverless_api.root_resource_id
  path_part   = "users"
}

resource "aws_api_gateway_resource" "user" {
  rest_api_id = aws_api_gateway_rest_api.serverless_api.id
  parent_id   = aws_api_gateway_resource.users.id
  path_part   = "{id}"
}

# POST /users
resource "aws_api_gateway_method" "create_user" {
  rest_api_id   = aws_api_gateway_rest_api.serverless_api.id
  resource_id   = aws_api_gateway_resource.users.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "create_user" {
  rest_api_id = aws_api_gateway_rest_api.serverless_api.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.user_manager.invoke_arn
}

# GET /users/{id}
resource "aws_api_gateway_method" "get_user" {
  rest_api_id   = aws_api_gateway_rest_api.serverless_api.id
  resource_id   = aws_api_gateway_resource.user.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "get_user" {
  rest_api_id = aws_api_gateway_rest_api.serverless_api.id
  resource_id = aws_api_gateway_resource.user.id
  http_method = aws_api_gateway_method.get_user.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.user_manager.invoke_arn
}

# Lambda permissions for API Gateway
resource "aws_lambda_permission" "api_gateway_create_user" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.user_manager.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.serverless_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "api_gateway_get_user" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.user_manager.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.serverless_api.execution_arn}/*/*"
}

# Lambda permissions for S3
resource "aws_lambda_permission" "s3_processor" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.s3_processor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.app_storage.arn
}

# Lambda permissions for SQS
resource "aws_lambda_permission" "sqs_processor" {
  statement_id  = "AllowExecutionFromSQS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sqs_processor.function_name
  principal     = "sqs.amazonaws.com"
  source_arn    = aws_sqs_queue.email_queue.arn
}

# EventBridge rule for scheduled tasks
resource "aws_cloudwatch_event_rule" "scheduled_tasks" {
  name                = "${var.project_name}-scheduled-tasks"
  description         = "Trigger scheduled tasks"
  schedule_expression = "rate(1 hour)"

  tags = {
    Name        = "${var.project_name}-scheduled-tasks"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_target" "scheduled_tasks" {
  rule      = aws_cloudwatch_event_rule.scheduled_tasks.name
  target_id = "ScheduledTasksTarget"
  arn       = aws_lambda_function.scheduled_tasks.arn
}

resource "aws_lambda_permission" "scheduled_tasks" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduled_tasks.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scheduled_tasks.arn
}

# SQS Event Source Mapping
resource "aws_lambda_event_source_mapping" "sqs_processor" {
  event_source_arn = aws_sqs_queue.email_queue.arn
  function_name    = aws_lambda_function.sqs_processor.function_name
  batch_size       = 10
  maximum_batching_window_in_seconds = 5
}

# API Gateway Deployment
resource "aws_api_gateway_deployment" "serverless_api" {
  depends_on = [
    aws_api_gateway_integration.create_user,
    aws_api_gateway_integration.get_user,
  ]

  rest_api_id = aws_api_gateway_rest_api.serverless_api.id
  stage_name  = var.environment

  lifecycle {
    create_before_destroy = true
  }
}

# Outputs
output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = aws_api_gateway_deployment.serverless_api.invoke_url
}

output "dynamodb_table_name" {
  description = "Name of the DynamoDB table"
  value       = aws_dynamodb_table.users.name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.app_storage.bucket
}

output "sqs_queue_url" {
  description = "URL of the SQS queue"
  value       = aws_sqs_queue.email_queue.url
}
```

### Serverless with Docker

```dockerfile
# Dockerfile
FROM public.ecr.aws/lambda/go:1

# Copy function code
COPY main ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["main"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=lambda,apigateway,dynamodb,s3,sqs,events
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock"
      - "./localstack-data:/tmp/localstack"

  serverless-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - AWS_ENDPOINT_URL=http://localstack:4566
      - AWS_ACCESS_KEY_ID=test
      - AWS_SECRET_ACCESS_KEY=test
      - AWS_DEFAULT_REGION=us-east-1
    depends_on:
      - localstack
```

## 🚀 Best Practices

### 1. Function Design
```go
// Keep functions small and focused
func CreateUserHandler(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // Single responsibility
    // Stateless
    // Fast execution
    return response, nil
}
```

### 2. Error Handling
```go
// Implement proper error handling
func processData(ctx context.Context, data interface{}) error {
    if err := validateData(data); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    if err := saveData(ctx, data); err != nil {
        return fmt.Errorf("save failed: %w", err)
    }
    
    return nil
}
```

### 3. Resource Optimization
```hcl
# Optimize Lambda configuration
resource "aws_lambda_function" "optimized" {
  timeout     = 30
  memory_size = 256
  
  environment {
    variables = {
      LOG_LEVEL = "INFO"
    }
  }
}
```

## 🏢 Industry Insights

### Serverless Usage Patterns
- **API Backends**: RESTful APIs and GraphQL endpoints
- **Event Processing**: Real-time data processing and ETL
- **Scheduled Tasks**: Cron jobs and batch processing
- **File Processing**: Image resizing, document conversion

### Enterprise Serverless Strategy
- **Microservices**: Break down monolithic applications
- **Cost Optimization**: Pay only for actual usage
- **Scalability**: Automatic scaling based on demand
- **Development Speed**: Faster time to market

## 🎯 Interview Questions

### Basic Level
1. **What is serverless computing?**
   - Cloud execution model
   - No server management
   - Auto-scaling
   - Pay-per-use

2. **What are the benefits of serverless?**
   - Cost optimization
   - Auto-scaling
   - No infrastructure management
   - Faster development

3. **What are the challenges of serverless?**
   - Cold starts
   - Vendor lock-in
   - Debugging complexity
   - Resource limits

### Intermediate Level
4. **How do you implement serverless functions?**
   - Function design
   - Event handling
   - Error handling
   - Resource optimization

5. **How do you handle serverless state?**
   - External databases
   - Stateless design
   - Session management
   - Data persistence

6. **How do you optimize serverless performance?**
   - Cold start reduction
   - Memory optimization
   - Connection pooling
   - Caching strategies

### Advanced Level
7. **How do you implement serverless at scale?**
   - Function orchestration
   - Event-driven architecture
   - Monitoring and observability
   - Cost optimization

8. **How do you handle serverless security?**
   - IAM policies
   - VPC configuration
   - Data encryption
   - Access control

9. **How do you implement serverless monitoring?**
   - CloudWatch metrics
   - Distributed tracing
   - Error tracking
   - Performance monitoring

---

**Next**: [Cost Optimization](./CostOptimization.md) - Cloud cost management, resource optimization, budgeting strategies
