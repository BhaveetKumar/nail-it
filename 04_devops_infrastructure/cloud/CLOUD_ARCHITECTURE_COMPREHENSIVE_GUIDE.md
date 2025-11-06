---
# Auto-generated front matter
Title: Cloud Architecture Comprehensive Guide
LastUpdated: 2025-11-06T20:45:59.143930
Tags: []
Status: draft
---

# ☁️ Cloud Architecture Comprehensive Guide

> **Complete guide to cloud architecture patterns, best practices, and implementation strategies**

## 📚 Table of Contents

1. [Cloud Architecture Fundamentals](#-cloud-architecture-fundamentals)
2. [AWS Architecture Patterns](#-aws-architecture-patterns)
3. [Google Cloud Architecture](#-google-cloud-architecture)
4. [Azure Architecture](#-azure-architecture)
5. [Multi-Cloud Strategies](#-multi-cloud-strategies)
6. [Serverless Architecture](#-serverless-architecture)
7. [Microservices on Cloud](#-microservices-on-cloud)
8. [Data Architecture](#-data-architecture)
9. [Security Architecture](#-security-architecture)
10. [Cost Optimization](#-cost-optimization)

---

## 🏗️ Cloud Architecture Fundamentals

### Cloud Architecture Principles

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scalability   │    │   Reliability   │    │   Performance   │
│   - Auto-scaling│    │   - Multi-AZ    │    │   - CDN         │
│   - Load        │    │   - Backup      │    │   - Caching     │
│     Balancing   │    │   - Monitoring  │    │   - Optimization│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │   Security & Compliance │
                    │   - IAM                 │
                    │   - Encryption          │
                    │   - Compliance          │
                    └─────────────────────────┘
```

### Well-Architected Framework

#### 1. Operational Excellence
- **Infrastructure as Code**: Terraform, CloudFormation
- **Automated Deployments**: CI/CD pipelines
- **Monitoring & Logging**: CloudWatch, Stackdriver
- **Incident Response**: Runbooks, escalation procedures

#### 2. Security
- **Identity & Access Management**: IAM, RBAC
- **Data Protection**: Encryption at rest and in transit
- **Network Security**: VPC, Security Groups, NACLs
- **Compliance**: SOC, PCI, GDPR compliance

#### 3. Reliability
- **Fault Tolerance**: Multi-AZ deployment
- **Disaster Recovery**: Backup and restore procedures
- **Auto-scaling**: Handle traffic spikes
- **Health Checks**: Application and infrastructure monitoring

#### 4. Performance Efficiency
- **Right-sizing**: Choose appropriate instance types
- **Caching**: Redis, ElastiCache, CloudFront
- **CDN**: Global content delivery
- **Database Optimization**: Read replicas, sharding

#### 5. Cost Optimization
- **Resource Tagging**: Track and manage costs
- **Reserved Instances**: Long-term cost savings
- **Spot Instances**: Use for fault-tolerant workloads
- **Auto-scaling**: Scale down during low usage

---

## 🚀 AWS Architecture Patterns

### 1. Three-Tier Architecture

```yaml
# Terraform configuration for three-tier architecture
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
}

# Public Subnets (Web Tier)
resource "aws_subnet" "public" {
  count  = 2
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
}

# Private Subnets (Application Tier)
resource "aws_subnet" "private" {
  count  = 2
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
}

# Database Subnets (Data Tier)
resource "aws_subnet" "database" {
  count  = 2
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.${count.index + 20}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "main-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {
  name                = "app-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.app.arn]
  health_check_type   = "ELB"
  min_size            = 2
  max_size            = 10
  desired_capacity    = 3

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "main-db"
  engine     = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  allocated_storage = 20
  storage_encrypted = true
  
  db_name  = "mydb"
  username = "admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}
```

### 2. Microservices Architecture

```yaml
# ECS Cluster for Microservices
resource "aws_ecs_cluster" "main" {
  name = "microservices-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Service for User Service
resource "aws_ecs_service" "user_service" {
  name            = "user-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.user_service.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.user_service.arn
    container_name   = "user-service"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.main]
}

# ECS Task Definition
resource "aws_ecs_task_definition" "user_service" {
  family                   = "user-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "user-service"
      image = "myregistry/user-service:latest"
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "DATABASE_URL"
          value = "postgresql://admin:password@${aws_db_instance.main.endpoint}/mydb"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/user-service"
          "awslogs-region"        = "us-west-2"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# API Gateway
resource "aws_api_gateway_rest_api" "main" {
  name        = "microservices-api"
  description = "API Gateway for microservices"
}

resource "aws_api_gateway_resource" "users" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "users"
}

resource "aws_api_gateway_method" "users_get" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.users.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "users_get" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_get.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lb.main.dns_name
}
```

### 3. Event-Driven Architecture

```yaml
# EventBridge for Event-Driven Architecture
resource "aws_cloudwatch_event_rule" "order_created" {
  name        = "order-created"
  description = "Capture order creation events"

  event_pattern = jsonencode({
    source      = ["myapp.orders"]
    detail-type = ["Order Created"]
  })
}

resource "aws_cloudwatch_event_target" "inventory_update" {
  rule      = aws_cloudwatch_event_rule.order_created.name
  target_id = "InventoryUpdateTarget"
  arn       = aws_lambda_function.inventory_update.arn
}

# SQS Queue for Async Processing
resource "aws_sqs_queue" "order_processing" {
  name                      = "order-processing"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 1209600
  receive_wait_time_seconds = 0
  visibility_timeout_seconds = 300

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.order_processing_dlq.arn
    maxReceiveCount     = 3
  })
}

# Lambda Function for Event Processing
resource "aws_lambda_function" "order_processor" {
  filename         = "order_processor.zip"
  function_name    = "order-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 30

  environment {
    variables = {
      QUEUE_URL = aws_sqs_queue.order_processing.id
    }
  }
}

# SNS Topic for Notifications
resource "aws_sns_topic" "notifications" {
  name = "order-notifications"
}

resource "aws_sns_topic_subscription" "email_notifications" {
  topic_arn = aws_sns_topic.notifications.arn
  protocol  = "email"
  endpoint  = "admin@example.com"
}
```

---

## 🌐 Google Cloud Architecture

### 1. GKE (Google Kubernetes Engine) Architecture

```yaml
# GKE Cluster
apiVersion: container.cnrm.cloud.google.com/v1beta1
kind: ContainerCluster
metadata:
  name: microservices-cluster
  namespace: default
spec:
  location: us-central1
  initialNodeCount: 3
  nodeConfig:
    machineType: e2-medium
    diskSizeGb: 100
    diskType: pd-standard
    imageType: COS_CONTAINERD
    oauthScopes:
      - https://www.googleapis.com/auth/cloud-platform
  network: projects/my-project/global/networks/default
  subnetwork: projects/my-project/regions/us-central1/subnetworks/default
  ipAllocationPolicy:
    useIpAliases: true
  masterAuth:
    clientCertificateConfig:
      issueClientCertificate: false
  addonsConfig:
    httpLoadBalancing:
      disabled: false
    horizontalPodAutoscaling:
      disabled: false
    networkPolicyConfig:
      disabled: false
  networkPolicy:
    enabled: true
    provider: CALICO
```

### 2. Cloud Run Architecture

```yaml
# Cloud Run Service
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: user-service
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/my-project/user-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
          requests:
            cpu: "1"
            memory: "1Gi"
```

### 3. Cloud Functions Architecture

```javascript
// Cloud Function for Event Processing
const functions = require('@google-cloud/functions-framework');
const { PubSub } = require('@google-cloud/pubsub');

const pubsub = new PubSub();

functions.cloudEvent('processOrder', async (cloudEvent) => {
  const order = cloudEvent.data;
  
  try {
    // Process order
    const result = await processOrder(order);
    
    // Publish result
    await pubsub.topic('order-processed').publishMessage({
      data: Buffer.from(JSON.stringify(result))
    });
    
    console.log('Order processed successfully:', result.id);
  } catch (error) {
    console.error('Error processing order:', error);
    
    // Publish to dead letter topic
    await pubsub.topic('order-failed').publishMessage({
      data: Buffer.from(JSON.stringify({
        order,
        error: error.message
      }))
    });
  }
});

async function processOrder(order) {
  // Order processing logic
  return {
    id: order.id,
    status: 'processed',
    processedAt: new Date().toISOString()
  };
}
```

---

## 🔵 Azure Architecture

### 1. Azure Container Instances

```yaml
# Azure Container Instance
apiVersion: 2021-09-01
kind: ContainerGroup
metadata:
  name: user-service
spec:
  location: eastus
  containers:
  - name: user-service
    image: myregistry.azurecr.io/user-service:latest
    ports:
    - port: 8080
      protocol: TCP
    resources:
      requests:
        cpu: 1
        memoryInGb: 2
    environmentVariables:
    - name: DATABASE_URL
      value: "Server=tcp:myserver.database.windows.net,1433;Database=mydb;User ID=admin;Password=password;Encrypt=true;"
    - name: AZURE_STORAGE_CONNECTION_STRING
      valueFrom:
        secretKeyRef:
          name: storage-secret
          key: connection-string
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8080
  imageRegistryCredentials:
  - server: myregistry.azurecr.io
    username: myregistry
    passwordSecretRef:
      name: acr-secret
      key: password
```

### 2. Azure Functions

```csharp
// Azure Function for Event Processing
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.EventGrid;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

public static class OrderProcessor
{
    [FunctionName("ProcessOrder")]
    public static async Task Run(
        [EventGridTrigger] EventGridEvent eventGridEvent,
        [ServiceBus("order-processing", Connection = "ServiceBusConnection")] IAsyncCollector<string> outputEvents,
        ILogger log)
    {
        log.LogInformation($"Event received: {eventGridEvent.Data}");
        
        try
        {
            var order = JsonConvert.DeserializeObject<Order>(eventGridEvent.Data.ToString());
            
            // Process order
            var result = await ProcessOrderAsync(order);
            
            // Send to next stage
            await outputEvents.AddAsync(JsonConvert.SerializeObject(result));
            
            log.LogInformation($"Order {order.Id} processed successfully");
        }
        catch (Exception ex)
        {
            log.LogError(ex, "Error processing order");
            throw;
        }
    }
    
    private static async Task<OrderResult> ProcessOrderAsync(Order order)
    {
        // Order processing logic
        await Task.Delay(1000); // Simulate processing
        
        return new OrderResult
        {
            OrderId = order.Id,
            Status = "Processed",
            ProcessedAt = DateTime.UtcNow
        };
    }
}

public class Order
{
    public string Id { get; set; }
    public string CustomerId { get; set; }
    public decimal Amount { get; set; }
}

public class OrderResult
{
    public string OrderId { get; set; }
    public string Status { get; set; }
    public DateTime ProcessedAt { get; set; }
}
```

---

## 🌍 Multi-Cloud Strategies

### 1. Multi-Cloud Architecture

```yaml
# Multi-cloud deployment configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-cloud-config
data:
  aws-region: "us-west-2"
  gcp-region: "us-central1"
  azure-region: "eastus"
  
  # Service routing
  user-service: "aws"
  payment-service: "gcp"
  notification-service: "azure"
  
  # Database configuration
  primary-db: "aws-rds"
  cache: "gcp-redis"
  analytics: "azure-synapse"
```

### 2. Cloud-Agnostic Deployment

```go
// Cloud-agnostic deployment interface
package main

import (
    "context"
    "fmt"
)

type CloudProvider interface {
    DeployService(ctx context.Context, config ServiceConfig) error
    ScaleService(ctx context.Context, serviceName string, replicas int) error
    GetServiceStatus(ctx context.Context, serviceName string) (*ServiceStatus, error)
    DeleteService(ctx context.Context, serviceName string) error
}

type ServiceConfig struct {
    Name        string
    Image       string
    Replicas    int
    Resources   ResourceConfig
    Environment map[string]string
}

type ResourceConfig struct {
    CPU    string
    Memory string
}

type ServiceStatus struct {
    Name      string
    Replicas  int
    Ready     int
    Status    string
}

// AWS Implementation
type AWSProvider struct {
    region string
}

func (p *AWSProvider) DeployService(ctx context.Context, config ServiceConfig) error {
    // AWS ECS deployment logic
    fmt.Printf("Deploying %s to AWS ECS in %s\n", config.Name, p.region)
    return nil
}

func (p *AWSProvider) ScaleService(ctx context.Context, serviceName string, replicas int) error {
    // AWS ECS scaling logic
    fmt.Printf("Scaling %s to %d replicas in AWS\n", serviceName, replicas)
    return nil
}

func (p *AWSProvider) GetServiceStatus(ctx context.Context, serviceName string) (*ServiceStatus, error) {
    // AWS ECS status check
    return &ServiceStatus{
        Name:     serviceName,
        Replicas: 3,
        Ready:    3,
        Status:   "Running",
    }, nil
}

func (p *AWSProvider) DeleteService(ctx context.Context, serviceName string) error {
    // AWS ECS deletion logic
    fmt.Printf("Deleting %s from AWS\n", serviceName)
    return nil
}

// GCP Implementation
type GCPProvider struct {
    project string
    region  string
}

func (p *GCPProvider) DeployService(ctx context.Context, config ServiceConfig) error {
    // GCP Cloud Run deployment logic
    fmt.Printf("Deploying %s to GCP Cloud Run in %s\n", config.Name, p.region)
    return nil
}

func (p *GCPProvider) ScaleService(ctx context.Context, serviceName string, replicas int) error {
    // GCP Cloud Run scaling logic
    fmt.Printf("Scaling %s to %d replicas in GCP\n", serviceName, replicas)
    return nil
}

func (p *GCPProvider) GetServiceStatus(ctx context.Context, serviceName string) (*ServiceStatus, error) {
    // GCP Cloud Run status check
    return &ServiceStatus{
        Name:     serviceName,
        Replicas: 2,
        Ready:    2,
        Status:   "Running",
    }, nil
}

func (p *GCPProvider) DeleteService(ctx context.Context, serviceName string) error {
    // GCP Cloud Run deletion logic
    fmt.Printf("Deleting %s from GCP\n", serviceName)
    return nil
}

// Multi-cloud deployment manager
type MultiCloudManager struct {
    providers map[string]CloudProvider
}

func NewMultiCloudManager() *MultiCloudManager {
    return &MultiCloudManager{
        providers: map[string]CloudProvider{
            "aws":  &AWSProvider{region: "us-west-2"},
            "gcp":  &GCPProvider{project: "my-project", region: "us-central1"},
            "azure": &AzureProvider{resourceGroup: "my-rg", region: "eastus"},
        },
    }
}

func (m *MultiCloudManager) DeployService(ctx context.Context, config ServiceConfig, provider string) error {
    p, exists := m.providers[provider]
    if !exists {
        return fmt.Errorf("unknown provider: %s", provider)
    }
    
    return p.DeployService(ctx, config)
}

func (m *MultiCloudManager) DeployToAllClouds(ctx context.Context, config ServiceConfig) error {
    for providerName, provider := range m.providers {
        if err := provider.DeployService(ctx, config); err != nil {
            return fmt.Errorf("failed to deploy to %s: %w", providerName, err)
        }
    }
    return nil
}
```

---

## ⚡ Serverless Architecture

### 1. AWS Lambda Architecture

```python
# Lambda function for API Gateway
import json
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')

def lambda_handler(event, context):
    try:
        # Parse request
        http_method = event['httpMethod']
        path = event['path']
        
        if http_method == 'GET' and path == '/users':
            return get_users()
        elif http_method == 'POST' and path == '/users':
            return create_user(json.loads(event['body']))
        elif http_method == 'GET' and path.startswith('/users/'):
            user_id = path.split('/')[-1]
            return get_user(user_id)
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not found'})
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_users():
    response = table.scan()
    users = response['Items']
    
    # Convert Decimal to int for JSON serialization
    for user in users:
        if 'age' in user:
            user['age'] = int(user['age'])
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(users)
    }

def create_user(user_data):
    user_id = str(uuid.uuid4())
    user_data['id'] = user_id
    user_data['created_at'] = datetime.utcnow().isoformat()
    
    # Convert int to Decimal for DynamoDB
    if 'age' in user_data:
        user_data['age'] = Decimal(str(user_data['age']))
    
    table.put_item(Item=user_data)
    
    return {
        'statusCode': 201,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(user_data)
    }

def get_user(user_id):
    response = table.get_item(Key={'id': user_id})
    
    if 'Item' not in response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'User not found'})
        }
    
    user = response['Item']
    if 'age' in user:
        user['age'] = int(user['age'])
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(user)
    }
```

### 2. Serverless Event Processing

```yaml
# Serverless event processing with Step Functions
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  OrderProcessingStateMachine:
    Type: AWS::Serverless::StateMachine
    Properties:
      Name: OrderProcessingStateMachine
      DefinitionUri: statemachine/order-processing.asl.json
      Role: !GetAtt StateMachineRole.Arn
      Events:
        OrderCreated:
          Type: EventBridge
          Properties:
            Pattern:
              source: ["myapp.orders"]
              detail-type: ["Order Created"]

  StateMachineRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: states.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: StateMachineExecutionPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - lambda:InvokeFunction
                  - sqs:SendMessage
                  - sns:Publish
                Resource: "*"

  ProcessOrderFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/process-order/
      Handler: index.handler
      Runtime: python3.9
      Timeout: 30
      Environment:
        Variables:
          INVENTORY_QUEUE_URL: !Ref InventoryQueue
          NOTIFICATION_TOPIC_ARN: !Ref NotificationTopic
      Policies:
        - SQSSendMessagePolicy:
            QueueName: !GetAtt InventoryQueue.QueueName
        - SNSPublishMessagePolicy:
            TopicName: !GetAtt NotificationTopic.TopicName

  InventoryQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: inventory-updates
      VisibilityTimeoutSeconds: 300
      MessageRetentionPeriod: 1209600

  NotificationTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: order-notifications
```

---

## 🗄️ Data Architecture

### 1. Data Lake Architecture

```yaml
# Data Lake on AWS
Resources:
  DataLakeBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${AWS::StackName}-data-lake"
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      LifecycleConfiguration:
        Rules:
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - StorageClass: STANDARD_IA
                TransitionInDays: 30
              - StorageClass: GLACIER
                TransitionInDays: 90

  GlueCrawler:
    Type: AWS::Glue::Crawler
    Properties:
      Name: data-lake-crawler
      Role: !GetAtt GlueServiceRole.Arn
      DatabaseName: !Ref GlueDatabase
      Targets:
        S3Targets:
          - Path: !Sub "s3://${DataLakeBucket}/raw/"
          - Path: !Sub "s3://${DataLakeBucket}/processed/"

  GlueDatabase:
    Type: AWS::Glue::Database
    Properties:
      CatalogId: !Ref AWS::AccountId
      DatabaseInput:
        Name: data_lake_db
        Description: Data Lake Database

  AthenaWorkGroup:
    Type: AWS::Athena::WorkGroup
    Properties:
      Name: data-lake-workgroup
      Description: Workgroup for data lake queries
      WorkGroupConfiguration:
        ResultConfiguration:
          OutputLocation: !Sub "s3://${DataLakeBucket}/athena-results/"
```

### 2. Real-time Data Pipeline

```yaml
# Kinesis Data Streams for real-time processing
Resources:
  OrderStream:
    Type: AWS::Kinesis::Stream
    Properties:
      Name: order-stream
      ShardCount: 2
      RetentionPeriodHours: 24

  OrderProcessorFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/order-processor/
      Handler: index.handler
      Runtime: python3.9
      Timeout: 60
      Events:
        OrderStream:
          Type: Kinesis
          Properties:
            Stream: !GetAtt OrderStream.Arn
            StartingPosition: TRIM_HORIZON
            BatchSize: 100
            MaximumBatchingWindowInSeconds: 5
      Policies:
        - KinesisReadPolicy:
            StreamName: !GetAtt OrderStream.StreamName
        - DynamoDBCrudPolicy:
            TableName: !Ref OrdersTable

  OrdersTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: orders
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: order_id
          AttributeType: S
        - AttributeName: created_at
          AttributeType: S
      KeySchema:
        - AttributeName: order_id
          KeyType: HASH
      GlobalSecondaryIndexes:
        - IndexName: created-at-index
          KeySchema:
            - AttributeName: created_at
              KeyType: HASH
          Projection:
            ProjectionType: ALL
```

---

## 🔒 Security Architecture

### 1. Zero Trust Architecture

```yaml
# Zero Trust Network Architecture
Resources:
  # VPC with private subnets only
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # Private subnets
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: false

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: false

  # VPC Endpoints for AWS services
  S3Endpoint:
    Type: AWS::EC2::VPCEndpoint
    Properties:
      VpcId: !Ref VPC
      ServiceName: !Sub "com.amazonaws.${AWS::Region}.s3"
      VpcEndpointType: Gateway
      RouteTableIds:
        - !Ref PrivateRouteTable1
        - !Ref PrivateRouteTable2

  DynamoDBEndpoint:
    Type: AWS::EC2::VPCEndpoint
    Properties:
      VpcId: !Ref VPC
      ServiceName: !Sub "com.amazonaws.${AWS::Region}.dynamodb"
      VpcEndpointType: Gateway
      RouteTableIds:
        - !Ref PrivateRouteTable1
        - !Ref PrivateRouteTable2

  # Application Load Balancer (internal)
  InternalALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: internal-alb
      Scheme: internal
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for internal ALB
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupId: !Ref AppSecurityGroup
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          SourceSecurityGroupId: !Ref AppSecurityGroup

  AppSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for application instances
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8080
          ToPort: 8080
          SourceSecurityGroupId: !Ref ALBSecurityGroup
      SecurityGroupEgress:
        - IpProtocol: -1
          CidrIp: 0.0.0.0/0
```

### 2. Identity and Access Management

```yaml
# IAM Roles and Policies
Resources:
  # ECS Task Role
  ECSTaskRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: ECSTaskPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:PutItem
                  - dynamodb:UpdateItem
                  - dynamodb:DeleteItem
                  - dynamodb:Query
                  - dynamodb:Scan
                Resource: !Sub "${OrdersTable.Arn}/*"
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                Resource: !Sub "${DataBucket.Arn}/*"

  # ECS Task Execution Role
  ECSTaskExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
      Policies:
        - PolicyName: ECSExecutionPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/ecs/*"

  # Lambda Execution Role
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: LambdaPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:PutItem
                  - dynamodb:UpdateItem
                  - dynamodb:DeleteItem
                Resource: !Sub "${OrdersTable.Arn}/*"
              - Effect: Allow
                Action:
                  - kinesis:DescribeStream
                  - kinesis:GetShardIterator
                  - kinesis:GetRecords
                Resource: !GetAtt OrderStream.Arn
```

---

## 💰 Cost Optimization

### 1. Cost Optimization Strategies

```yaml
# Cost optimization configuration
Resources:
  # Spot Instances for non-critical workloads
  SpotFleetRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: spotfleet.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole

  SpotFleet:
    Type: AWS::EC2::SpotFleet
    Properties:
      SpotFleetRequestConfig:
        IamFleetRole: !GetAtt SpotFleetRole.Arn
        TargetCapacity: 10
        SpotPrice: "0.05"
        LaunchSpecifications:
          - ImageId: ami-0c02fb55956c7d316
            InstanceType: t3.micro
            KeyName: my-key
            SecurityGroups:
              - GroupId: !Ref SecurityGroup
            UserData:
              Fn::Base64: !Sub |
                #!/bin/bash
                yum update -y
                yum install -y docker
                service docker start
                usermod -a -G docker ec2-user

  # Reserved Instances for predictable workloads
  ReservedInstance:
    Type: AWS::EC2::ReservedInstances
    Properties:
      InstanceType: t3.medium
      InstanceCount: 3
      OfferingType: All Upfront
      ProductDescription: Linux/UNIX
      ReservedInstancesOfferingId: !Ref ReservedInstancesOffering

  # Auto Scaling with cost optimization
  CostOptimizedASG:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 1
      MaxSize: 10
      DesiredCapacity: 3
      VPCZoneIdentifier:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      MixedInstancesPolicy:
        InstancesDistribution:
          OnDemandBaseCapacity: 1
          OnDemandPercentageAboveBaseCapacity: 20
          SpotAllocationStrategy: diversified
        LaunchTemplate:
          LaunchTemplateSpecification:
            LaunchTemplateId: !Ref LaunchTemplate
            Version: !GetAtt LaunchTemplate.LatestVersionNumber
          Overrides:
            - InstanceType: t3.small
            - InstanceType: t3.medium
            - InstanceType: t3.large
```

### 2. Cost Monitoring and Alerting

```yaml
# Cost monitoring and alerting
Resources:
  # Billing alarm
  BillingAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: MonthlyBillingAlarm
      AlarmDescription: Alert when monthly costs exceed threshold
      MetricName: EstimatedCharges
      Namespace: AWS/Billing
      Statistic: Maximum
      Period: 86400
      EvaluationPeriods: 1
      Threshold: 1000
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: Currency
          Value: USD
      AlarmActions:
        - !Ref BillingAlarmTopic

  BillingAlarmTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: billing-alerts
      Subscription:
        - Protocol: email
          Endpoint: admin@example.com

  # Cost and Usage Report
  CostUsageReport:
    Type: AWS::CUR::ReportDefinition
    Properties:
      ReportName: cost-usage-report
      TimeUnit: DAILY
      Format: textORcsv
      Compression: GZIP
      S3Bucket: !Ref CostReportBucket
      S3Prefix: cost-reports/
      S3Region: !Ref AWS::Region
      AdditionalSchemaElements:
        - RESOURCES
      RefreshClosedReports: true
      ReportVersioning: CREATE_NEW_REPORT

  CostReportBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${AWS::StackName}-cost-reports"
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
```

---

## 🎯 Best Practices Summary

### 1. Design Principles
- **Scalability**: Design for horizontal scaling
- **Reliability**: Implement fault tolerance
- **Security**: Follow zero trust principles
- **Cost Optimization**: Right-size resources
- **Performance**: Optimize for speed and efficiency

### 2. Implementation Guidelines
- **Infrastructure as Code**: Use Terraform, CloudFormation
- **Automation**: Implement CI/CD pipelines
- **Monitoring**: Set up comprehensive observability
- **Testing**: Implement automated testing
- **Documentation**: Maintain up-to-date documentation

### 3. Operational Excellence
- **Disaster Recovery**: Implement backup and restore
- **Incident Response**: Have runbooks and procedures
- **Change Management**: Implement controlled deployments
- **Capacity Planning**: Monitor and plan for growth
- **Security**: Regular security audits and updates

---

**☁️ Master these cloud architecture patterns to build scalable, reliable, and cost-effective cloud solutions! 🚀**


##  Microservices On Cloud

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-microservices-on-cloud -->

Placeholder content. Please replace with proper section.
