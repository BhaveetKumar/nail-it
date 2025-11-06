---
# Auto-generated front matter
Title: Aws Lambda
LastUpdated: 2025-11-06T20:45:59.150337
Tags: []
Status: draft
---

# ⚡ AWS Lambda: Serverless Functions and Event-Driven Architecture

> **Master AWS Lambda for serverless computing and event-driven applications**

## 📚 Concept

**Detailed Explanation:**
AWS Lambda is a revolutionary serverless compute service that fundamentally changes how we think about application deployment and scaling. It allows developers to run code without provisioning or managing servers, automatically handling the underlying infrastructure, scaling, and resource management. Lambda executes code in response to events and automatically manages the compute resources, making it ideal for event-driven architectures and microservices.

**Core Philosophy:**

- **Serverless First**: Eliminate server management overhead and focus on business logic
- **Event-Driven Architecture**: Respond to events from various sources automatically
- **Auto-Scaling**: Scale from zero to thousands of concurrent executions seamlessly
- **Pay-per-Use**: Only pay for the compute time you actually consume
- **Managed Infrastructure**: AWS handles all infrastructure concerns
- **Developer Productivity**: Focus on code, not infrastructure management

**Why AWS Lambda Matters:**

- **Cost Efficiency**: Pay only for actual execution time, not idle server time
- **Automatic Scaling**: Handle traffic spikes without manual intervention
- **Reduced Complexity**: No server management, patching, or capacity planning
- **Faster Time to Market**: Deploy code without infrastructure setup
- **Event-Driven**: Perfect for modern, reactive application architectures
- **Microservices**: Ideal for building loosely coupled, event-driven microservices
- **Integration**: Seamless integration with 200+ AWS services
- **Global Availability**: Deploy functions across multiple AWS regions

**Key Features:**

**1. Serverless:**

- **No Server Management**: AWS handles all server provisioning, patching, and maintenance
- **Automatic Infrastructure**: Infrastructure is completely managed by AWS
- **Zero Administration**: No need to manage operating systems, containers, or virtual machines
- **Benefits**: Reduced operational overhead, faster development cycles, lower total cost of ownership
- **Use Cases**: Microservices, event processing, data transformation, API backends

**2. Event-Driven:**

- **Event Sources**: Respond to events from S3, DynamoDB, API Gateway, SNS, SQS, and more
- **Automatic Triggers**: Functions are automatically invoked when events occur
- **Event Processing**: Process events in real-time or batch mode
- **Benefits**: Reactive architecture, real-time processing, loose coupling
- **Use Cases**: File processing, data streaming, IoT applications, webhooks

**3. Auto-Scaling:**

- **Automatic Scaling**: Scale from zero to thousands of concurrent executions
- **No Capacity Planning**: No need to predict or provision capacity
- **Instant Scaling**: Scale up or down in milliseconds based on demand
- **Concurrency Limits**: Set limits to control costs and downstream service impact
- **Benefits**: Handle traffic spikes, cost optimization, no over-provisioning
- **Use Cases**: Web applications, data processing, real-time analytics

**4. Pay-per-Use:**

- **Execution-Based Pricing**: Pay only for the compute time consumed
- **No Idle Costs**: No charges when functions are not executing
- **Granular Billing**: Billed per 100ms of execution time
- **Free Tier**: 1M free requests and 400,000 GB-seconds per month
- **Benefits**: Cost optimization, predictable pricing, no upfront costs
- **Use Cases**: Variable workloads, cost-sensitive applications, proof of concepts

**5. Multiple Runtimes:**

- **Supported Languages**: Python, Node.js, Java, C#, Go, Ruby, PHP, Rust
- **Custom Runtimes**: Bring your own runtime for any language
- **Runtime Updates**: AWS manages runtime updates and security patches
- **Benefits**: Language flexibility, reduced maintenance, security updates
- **Use Cases**: Polyglot applications, legacy system integration, specialized workloads

**6. Integration:**

- **AWS Services**: Native integration with 200+ AWS services
- **Event Sources**: S3, DynamoDB, Kinesis, SNS, SQS, API Gateway, CloudWatch
- **Destinations**: Send results to other AWS services automatically
- **Benefits**: Seamless workflows, reduced complexity, event-driven architecture
- **Use Cases**: Data pipelines, serverless applications, IoT processing

**Advanced Lambda Concepts:**

- **Cold Starts**: Initial latency when functions haven't been used recently
- **Warm Starts**: Faster execution when functions are already loaded
- **Provisioned Concurrency**: Keep functions warm to eliminate cold starts
- **Dead Letter Queues**: Handle failed function executions
- **Lambda Layers**: Share code and dependencies across functions
- **Lambda Extensions**: Add monitoring, security, and other capabilities
- **Container Images**: Deploy functions as container images for larger workloads
- **Function URLs**: Direct HTTP endpoints for Lambda functions

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive serverless architecture using AWS Lambda for a large-scale, event-driven application with complex data processing requirements?**

**Answer:** Comprehensive serverless architecture design:

- **Event-Driven Design**: Use S3, DynamoDB Streams, Kinesis, and SNS for event sources
- **Function Decomposition**: Break down complex logic into smaller, focused functions
- **Data Processing Pipeline**: Use Lambda for ETL operations with S3, DynamoDB, and RDS
- **API Gateway Integration**: Create RESTful APIs with Lambda backends
- **State Management**: Use DynamoDB for stateful operations and Step Functions for workflows
- **Error Handling**: Implement dead letter queues and retry mechanisms
- **Monitoring**: Use CloudWatch, X-Ray, and custom metrics for observability
- **Security**: Implement IAM roles, VPC configuration, and encryption
- **Performance**: Use provisioned concurrency for critical functions
- **Cost Optimization**: Implement proper timeout and memory configurations
- **Testing**: Use local testing frameworks and AWS SAM for deployment
- **Documentation**: Maintain comprehensive documentation for function purposes and dependencies

**Q2: What are the key considerations when implementing AWS Lambda for a production environment with strict performance, security, and compliance requirements?**

**Answer:** Production Lambda implementation considerations:

- **Performance Optimization**: Optimize cold starts, memory allocation, and timeout settings
- **Security Hardening**: Implement least privilege IAM roles, VPC configuration, and encryption
- **Compliance**: Ensure functions meet regulatory requirements (GDPR, HIPAA, SOX)
- **Monitoring**: Implement comprehensive logging, metrics, and alerting
- **Error Handling**: Use dead letter queues, retry mechanisms, and circuit breakers
- **Cost Management**: Monitor costs, implement proper resource allocation, and use reserved capacity
- **Disaster Recovery**: Implement backup and recovery procedures for function code and configuration
- **Testing**: Implement comprehensive testing including unit, integration, and load testing
- **Documentation**: Maintain detailed documentation for operations and troubleshooting
- **Governance**: Establish policies for function deployment, monitoring, and lifecycle management
- **Incident Response**: Have clear procedures for Lambda-related incidents
- **Regular Reviews**: Conduct regular performance and security reviews

**Q3: How do you optimize AWS Lambda for performance, cost, and reliability in enterprise environments?**

**Answer:** Enterprise Lambda optimization strategies:

- **Performance Optimization**: Use provisioned concurrency, optimize memory allocation, and implement connection pooling
- **Cost Optimization**: Right-size memory allocation, implement proper timeout settings, and use reserved capacity
- **Reliability**: Implement retry mechanisms, dead letter queues, and circuit breakers
- **Monitoring**: Use CloudWatch, X-Ray, and custom metrics for comprehensive observability
- **Security**: Implement proper IAM roles, VPC configuration, and encryption
- **Error Handling**: Use structured error handling and comprehensive logging
- **Testing**: Implement automated testing and deployment pipelines
- **Documentation**: Maintain comprehensive documentation and runbooks
- **Governance**: Establish policies for function lifecycle management
- **Training**: Provide training for teams on Lambda best practices
- **Regular Reviews**: Conduct regular performance and cost reviews
- **Incident Response**: Have clear procedures for Lambda-related incidents

## 🏗️ Lambda Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Event Sources                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    S3       │  │    API      │  │   DynamoDB  │     │
│  │   Events    │  │  Gateway    │  │   Streams   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              AWS Lambda Function                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Runtime   │  │   Handler   │  │   Memory    │ │ │
│  │  │   Python    │  │   Function  │  │   Config    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    S3       │  │   DynamoDB  │  │   SNS/SQS   │     │
│  │   Storage   │  │   Database  │  │  Messaging  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Lambda Function with CloudFormation

```yaml
# lambda-function.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "Lambda function with API Gateway and DynamoDB"

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

  FunctionName:
    Type: String
    Default: my-function
    Description: Name for the Lambda function

Resources:
  # Lambda Function
  LambdaFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub "${FunctionName}-${Environment}"
      Runtime: python3.9
      Handler: index.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        ZipFile: |
          import json
          import boto3
          import os

          dynamodb = boto3.resource('dynamodb')
          table = dynamodb.Table(os.environ['TABLE_NAME'])

          def lambda_handler(event, context):
              try:
                  # Parse the request body
                  body = json.loads(event['body']) if event.get('body') else {}

                  # Extract data from the request
                  user_id = body.get('user_id')
                  name = body.get('name')
                  email = body.get('email')

                  if not user_id or not name or not email:
                      return {
                          'statusCode': 400,
                          'headers': {
                              'Content-Type': 'application/json',
                              'Access-Control-Allow-Origin': '*'
                          },
                          'body': json.dumps({
                              'error': 'Missing required fields: user_id, name, email'
                          })
                      }

                  # Save to DynamoDB
                  table.put_item(Item={
                      'user_id': user_id,
                      'name': name,
                      'email': email,
                      'created_at': context.aws_request_id
                  })

                  return {
                      'statusCode': 200,
                      'headers': {
                          'Content-Type': 'application/json',
                          'Access-Control-Allow-Origin': '*'
                      },
                      'body': json.dumps({
                          'message': 'User created successfully',
                          'user_id': user_id
                      })
                  }

              except Exception as e:
                  return {
                      'statusCode': 500,
                      'headers': {
                          'Content-Type': 'application/json',
                          'Access-Control-Allow-Origin': '*'
                      },
                      'body': json.dumps({
                          'error': str(e)
                      })
                  }
      Environment:
        Variables:
          TABLE_NAME: !Ref DynamoDBTable
      Timeout: 30
      MemorySize: 256
      Tags:
        - Key: Name
          Value: !Sub "${FunctionName}-${Environment}"
        - Key: Environment
          Value: !Ref Environment

  # Lambda Execution Role
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: DynamoDBAccess
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:PutItem
                  - dynamodb:UpdateItem
                  - dynamodb:DeleteItem
                  - dynamodb:Query
                  - dynamodb:Scan
                Resource: !GetAtt DynamoDBTable.Arn

  # DynamoDB Table
  DynamoDBTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${FunctionName}-${Environment}-users"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: user_id
          AttributeType: S
      KeySchema:
        - AttributeName: user_id
          KeyType: HASH
      StreamSpecification:
        StreamViewType: NEW_AND_OLD_IMAGES
      Tags:
        - Key: Name
          Value: !Sub "${FunctionName}-${Environment}-users"
        - Key: Environment
          Value: !Ref Environment

  # API Gateway
  ApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: !Sub "${FunctionName}-${Environment}-api"
      Description: API Gateway for Lambda function
      EndpointConfiguration:
        Types:
          - REGIONAL

  # API Gateway Resource
  ApiGatewayResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      RestApiId: !Ref ApiGateway
      ParentId: !GetAtt ApiGateway.RootResourceId
      PathPart: users

  # API Gateway Method
  ApiGatewayMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      RestApiId: !Ref ApiGateway
      ResourceId: !Ref ApiGatewayResource
      HttpMethod: POST
      AuthorizationType: NONE
      Integration:
        Type: AWS_PROXY
        IntegrationHttpMethod: POST
        Uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${LambdaFunction.Arn}/invocations"

  # Lambda Permission for API Gateway
  LambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref LambdaFunction
      Action: lambda:InvokeFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub "arn:aws:execute-api:${AWS::Region}:${AWS::AccountId}:${ApiGateway}/*/*"

  # API Gateway Deployment
  ApiGatewayDeployment:
    Type: AWS::ApiGateway::Deployment
    DependsOn: ApiGatewayMethod
    Properties:
      RestApiId: !Ref ApiGateway
      StageName: !Ref Environment

  # CloudWatch Log Group
  CloudWatchLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/${FunctionName}-${Environment}"
      RetentionInDays: 14

  # S3 Event Trigger
  S3EventTrigger:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref LambdaFunction
      Action: lambda:InvokeFunction
      Principal: s3.amazonaws.com
      SourceArn: !Sub "arn:aws:s3:::${S3Bucket}"

  # S3 Bucket for testing
  S3Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${FunctionName}-${Environment}-${AWS::AccountId}"
      NotificationConfiguration:
        LambdaConfigurations:
          - Event: s3:ObjectCreated:*
            Function: !GetAtt LambdaFunction.Arn
            Filter:
              S3Key:
                Rules:
                  - Name: prefix
                    Value: uploads/

  # DynamoDB Stream Trigger
  DynamoDBStreamTrigger:
    Type: AWS::Lambda::EventSourceMapping
    Properties:
      EventSourceArn: !GetAtt DynamoDBTable.StreamArn
      FunctionName: !Ref LambdaFunction
      StartingPosition: LATEST
      BatchSize: 10

  # SNS Topic
  SNSTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub "${FunctionName}-${Environment}-notifications"
      DisplayName: User Notifications

  # SNS Subscription
  SNSSubscription:
    Type: AWS::SNS::Subscription
    Properties:
      Protocol: email
      TopicArn: !Ref SNSTopic
      Endpoint: !Ref NotificationEmail

Parameters:
  NotificationEmail:
    Type: String
    Description: Email address for notifications
    Default: admin@example.com

Outputs:
  ApiGatewayUrl:
    Description: API Gateway URL
    Value: !Sub "https://${ApiGateway}.execute-api.${AWS::Region}.amazonaws.com/${Environment}/users"
    Export:
      Name: !Sub "${Environment}-API-Gateway-URL"

  LambdaFunctionArn:
    Description: Lambda Function ARN
    Value: !GetAtt LambdaFunction.Arn
    Export:
      Name: !Sub "${Environment}-Lambda-Function-ARN"

  DynamoDBTableName:
    Description: DynamoDB Table Name
    Value: !Ref DynamoDBTable
    Export:
      Name: !Sub "${Environment}-DynamoDB-Table-Name"
```

### Lambda with Terraform

```hcl
# lambda.tf
terraform {
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

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "function_name" {
  description = "Lambda function name"
  type        = string
  default     = "my-function"
}

# Lambda Function
resource "aws_lambda_function" "main" {
  function_name = "${var.function_name}-${var.environment}"
  runtime       = "python3.9"
  handler       = "index.lambda_handler"
  role          = aws_iam_role.lambda_execution.arn

  filename         = "lambda_function.zip"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.main.name
    }
  }

  timeout     = 30
  memory_size = 256

  tags = {
    Name        = "${var.function_name}-${var.environment}"
    Environment = var.environment
  }
}

# Lambda Function Code
data "archive_file" "lambda_zip" {
  type        = "zip"
  output_path = "lambda_function.zip"
  source {
    content = <<EOF
import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

def lambda_handler(event, context):
    try:
        # Parse the request body
        body = json.loads(event['body']) if event.get('body') else {}

        # Extract data from the request
        user_id = body.get('user_id')
        name = body.get('name')
        email = body.get('email')

        if not user_id or not name or not email:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required fields: user_id, name, email'
                })
            }

        # Save to DynamoDB
        table.put_item(Item={
            'user_id': user_id,
            'name': name,
            'email': email,
            'created_at': context.aws_request_id
        })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'User created successfully',
                'user_id': user_id
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
EOF
    filename = "index.py"
  }
}

# Lambda Execution Role
resource "aws_iam_role" "lambda_execution" {
  name = "${var.function_name}-${var.environment}-lambda-execution"

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

# Lambda Execution Role Policy
resource "aws_iam_role_policy" "lambda_execution" {
  name = "${var.function_name}-${var.environment}-lambda-execution"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
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
        Resource = aws_dynamodb_table.main.arn
      }
    ]
  })
}

# DynamoDB Table
resource "aws_dynamodb_table" "main" {
  name           = "${var.function_name}-${var.environment}-users"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  tags = {
    Name        = "${var.function_name}-${var.environment}-users"
    Environment = var.environment
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.function_name}-${var.environment}-api"
  description = "API Gateway for Lambda function"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# API Gateway Resource
resource "aws_api_gateway_resource" "users" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "users"
}

# API Gateway Method
resource "aws_api_gateway_method" "post" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.users.id
  http_method   = "POST"
  authorization = "NONE"
}

# API Gateway Integration
resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.main.invoke_arn
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.main.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}

# API Gateway Deployment
resource "aws_api_gateway_deployment" "main" {
  depends_on = [aws_api_gateway_integration.lambda]

  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = var.environment
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/aws/lambda/${var.function_name}-${var.environment}"
  retention_in_days = 14
}

# S3 Bucket for testing
resource "aws_s3_bucket" "main" {
  bucket = "${var.function_name}-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.function_name}-${var.environment}"
    Environment = var.environment
  }
}

# Random ID for bucket suffix
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Bucket Notification
resource "aws_s3_bucket_notification" "main" {
  bucket = aws_s3_bucket.main.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.main.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }
}

# Lambda Permission for S3
resource "aws_lambda_permission" "s3" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.main.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.main.arn
}

# DynamoDB Stream Event Source Mapping
resource "aws_lambda_event_source_mapping" "dynamodb" {
  event_source_arn  = aws_dynamodb_table.main.stream_arn
  function_name     = aws_lambda_function.main.arn
  starting_position = "LATEST"
  batch_size        = 10
}

# SNS Topic
resource "aws_sns_topic" "main" {
  name = "${var.function_name}-${var.environment}-notifications"

  tags = {
    Name        = "${var.function_name}-${var.environment}-notifications"
    Environment = var.environment
  }
}

# SNS Subscription
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.main.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

variable "notification_email" {
  description = "Email address for notifications"
  type        = string
  default     = "admin@example.com"
}

# Outputs
output "api_gateway_url" {
  description = "API Gateway URL"
  value       = "${aws_api_gateway_deployment.main.invoke_url}/users"
}

output "lambda_function_arn" {
  description = "Lambda Function ARN"
  value       = aws_lambda_function.main.arn
}

output "dynamodb_table_name" {
  description = "DynamoDB Table Name"
  value       = aws_dynamodb_table.main.name
}
```

### Lambda Function with Go

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "os"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/dynamodb"
    "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

type User struct {
    UserID    string `json:"user_id"`
    Name      string `json:"name"`
    Email     string `json:"email"`
    CreatedAt string `json:"created_at"`
}

type Response struct {
    StatusCode int               `json:"statusCode"`
    Headers    map[string]string `json:"headers"`
    Body       string            `json:"body"`
}

type ErrorResponse struct {
    Error string `json:"error"`
}

type SuccessResponse struct {
    Message string `json:"message"`
    UserID  string `json:"user_id"`
}

type LambdaHandler struct {
    dynamodbClient *dynamodb.Client
    tableName      string
}

func NewLambdaHandler() (*LambdaHandler, error) {
    cfg, err := config.LoadDefaultConfig(context.TODO())
    if err != nil {
        return nil, err
    }

    dynamodbClient := dynamodb.NewFromConfig(cfg)
    tableName := os.Getenv("TABLE_NAME")

    return &LambdaHandler{
        dynamodbClient: dynamodbClient,
        tableName:      tableName,
    }, nil
}

func (h *LambdaHandler) HandleRequest(ctx context.Context, event events.APIGatewayProxyRequest) (Response, error) {
    // Parse the request body
    var user User
    if err := json.Unmarshal([]byte(event.Body), &user); err != nil {
        return createErrorResponse(400, "Invalid JSON format"), nil
    }

    // Validate required fields
    if user.UserID == "" || user.Name == "" || user.Email == "" {
        return createErrorResponse(400, "Missing required fields: user_id, name, email"), nil
    }

    // Save to DynamoDB
    if err := h.saveUser(ctx, user); err != nil {
        log.Printf("Error saving user: %v", err)
        return createErrorResponse(500, "Internal server error"), nil
    }

    // Return success response
    successResponse := SuccessResponse{
        Message: "User created successfully",
        UserID:  user.UserID,
    }

    body, _ := json.Marshal(successResponse)
    return Response{
        StatusCode: 200,
        Headers: map[string]string{
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        Body: string(body),
    }, nil
}

func (h *LambdaHandler) saveUser(ctx context.Context, user User) error {
    _, err := h.dynamodbClient.PutItem(ctx, &dynamodb.PutItemInput{
        TableName: aws.String(h.tableName),
        Item: map[string]types.AttributeValue{
            "user_id": &types.AttributeValueMemberS{
                Value: user.UserID,
            },
            "name": &types.AttributeValueMemberS{
                Value: user.Name,
            },
            "email": &types.AttributeValueMemberS{
                Value: user.Email,
            },
            "created_at": &types.AttributeValueMemberS{
                Value: user.CreatedAt,
            },
        },
    })

    return err
}

func createErrorResponse(statusCode int, message string) Response {
    errorResponse := ErrorResponse{
        Error: message,
    }

    body, _ := json.Marshal(errorResponse)
    return Response{
        StatusCode: statusCode,
        Headers: map[string]string{
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        Body: string(body),
    }
}

func main() {
    handler, err := NewLambdaHandler()
    if err != nil {
        log.Fatal(err)
    }

    lambda.Start(handler.HandleRequest)
}
```

### Lambda with S3 Event Trigger

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3Handler struct {
    s3Client *s3.Client
}

func NewS3Handler() (*S3Handler, error) {
    cfg, err := config.LoadDefaultConfig(context.TODO())
    if err != nil {
        return nil, err
    }

    s3Client := s3.NewFromConfig(cfg)

    return &S3Handler{
        s3Client: s3Client,
    }, nil
}

func (h *S3Handler) HandleS3Event(ctx context.Context, event events.S3Event) error {
    for _, record := range event.Records {
        bucket := record.S3.Bucket.Name
        key := record.S3.Object.Key

        log.Printf("Processing S3 event: s3://%s/%s", bucket, key)

        // Get object metadata
        result, err := h.s3Client.HeadObject(ctx, &s3.HeadObjectInput{
            Bucket: aws.String(bucket),
            Key:    aws.String(key),
        })
        if err != nil {
            log.Printf("Error getting object metadata: %v", err)
            continue
        }

        log.Printf("Object size: %d bytes", *result.ContentLength)
        log.Printf("Content type: %s", *result.ContentType)

        // Process the object based on its type
        if err := h.processObject(ctx, bucket, key, *result.ContentType); err != nil {
            log.Printf("Error processing object: %v", err)
            continue
        }

        log.Printf("Successfully processed: s3://%s/%s", bucket, key)
    }

    return nil
}

func (h *S3Handler) processObject(ctx context.Context, bucket, key, contentType string) error {
    switch contentType {
    case "image/jpeg", "image/png", "image/gif":
        return h.processImage(ctx, bucket, key)
    case "text/plain", "text/csv":
        return h.processText(ctx, bucket, key)
    case "application/json":
        return h.processJSON(ctx, bucket, key)
    default:
        log.Printf("Unsupported content type: %s", contentType)
        return nil
    }
}

func (h *S3Handler) processImage(ctx context.Context, bucket, key string) error {
    log.Printf("Processing image: s3://%s/%s", bucket, key)

    // Add image processing logic here
    // e.g., resize, watermark, extract metadata

    return nil
}

func (h *S3Handler) processText(ctx context.Context, bucket, key string) error {
    log.Printf("Processing text file: s3://%s/%s", bucket, key)

    // Add text processing logic here
    // e.g., extract keywords, sentiment analysis

    return nil
}

func (h *S3Handler) processJSON(ctx context.Context, bucket, key string) error {
    log.Printf("Processing JSON file: s3://%s/%s", bucket, key)

    // Add JSON processing logic here
    // e.g., validate schema, transform data

    return nil
}

func main() {
    handler, err := NewS3Handler()
    if err != nil {
        log.Fatal(err)
    }

    lambda.Start(handler.HandleS3Event)
}
```

## 🚀 Best Practices

### 1. Performance Optimization

```yaml
# Optimize Lambda function configuration
LambdaFunction:
  Type: AWS::Lambda::Function
  Properties:
    Runtime: python3.9
    Handler: index.lambda_handler
    MemorySize: 512 # Adjust based on needs
    Timeout: 30 # Set appropriate timeout
    Environment:
      Variables:
        LOG_LEVEL: INFO
    ReservedConcurrencyLimit: 10 # Prevent overwhelming downstream services
```

### 2. Error Handling

```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Your function logic here
        result = process_request(event)

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid input'})
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        }
```

### 3. Security Best Practices

```yaml
# Secure Lambda function
LambdaFunction:
  Type: AWS::Lambda::Function
  Properties:
    Role: !GetAtt LambdaExecutionRole.Arn
    VpcConfig:
      SecurityGroupIds:
        - !Ref LambdaSecurityGroup
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
    Environment:
      Variables:
        ENCRYPTION_KEY: !Ref EncryptionKey
    KmsKeyArn: !Ref LambdaKmsKey

LambdaExecutionRole:
  Type: AWS::IAM::Role
  Properties:
    AssumeRolePolicyDocument:
      Version: "2012-10-17"
      Statement:
        - Effect: Allow
          Principal:
            Service: lambda.amazonaws.com
          Action: sts:AssumeRole
    Policies:
      - PolicyName: LeastPrivilegeAccess
        PolicyDocument:
          Version: "2012-10-17"
          Statement:
            - Effect: Allow
              Action:
                - dynamodb:GetItem
                - dynamodb:PutItem
              Resource: !Sub "${DynamoDBTable.Arn}"
```

## 🏢 Industry Insights

### Netflix's Lambda Usage

- **Event Processing**: Real-time data processing
- **Microservices**: Serverless microservices
- **Data Pipeline**: ETL operations
- **Cost Optimization**: Pay-per-use model

### Airbnb's Lambda Strategy

- **Image Processing**: Photo resizing and optimization
- **Data Analytics**: Real-time analytics
- **API Gateway**: Serverless APIs
- **Event-Driven**: Event-driven architecture

### Spotify's Lambda Approach

- **Music Processing**: Audio file processing
- **Recommendations**: Real-time recommendations
- **Data Pipeline**: ETL operations
- **Cost Efficiency**: Serverless computing

## 🎯 Interview Questions

### Basic Level

1. **What is AWS Lambda?**

   - Serverless compute service
   - Event-driven execution
   - Auto-scaling
   - Pay-per-use pricing

2. **What are Lambda triggers?**

   - API Gateway
   - S3 events
   - DynamoDB streams
   - SNS/SQS messages

3. **What are Lambda limitations?**
   - 15-minute execution timeout
   - 10GB memory limit
   - 512MB temporary storage
   - Cold start latency

### Intermediate Level

4. **How do you optimize Lambda performance?**

   ```yaml
   # Performance optimization
   LambdaFunction:
     Type: AWS::Lambda::Function
     Properties:
       MemorySize: 512
       Timeout: 30
       ReservedConcurrencyLimit: 10
       Environment:
         Variables:
           LOG_LEVEL: INFO
   ```

5. **How do you handle Lambda errors?**

   - Try-catch blocks
   - Dead letter queues
   - Retry mechanisms
   - Error logging

6. **How do you secure Lambda functions?**
   - IAM roles with least privilege
   - VPC configuration
   - Environment variables encryption
   - Input validation

### Advanced Level

7. **How do you implement Lambda patterns?**

   - Fan-out pattern
   - SAGA pattern
   - Event sourcing
   - CQRS pattern

8. **How do you handle Lambda cold starts?**

   - Provisioned concurrency
   - Connection pooling
   - Keep-alive strategies
   - Warming functions

9. **How do you implement Lambda testing?**
   - Unit testing
   - Integration testing
   - Local testing
   - Mock AWS services

---

**Next**: [AWS RDS](AWS_RDS.md) - Managed databases, read replicas, backups
