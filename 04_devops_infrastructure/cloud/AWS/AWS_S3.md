# 🗄️ AWS S3: Object Storage, Versioning, and Lifecycle Policies

> **Master Amazon S3 for scalable object storage and data management**

## 📚 Concept

**Detailed Explanation:**
Amazon Simple Storage Service (S3) is a revolutionary cloud storage service that provides highly scalable, durable, and secure object storage. Unlike traditional file systems, S3 treats data as objects rather than files, enabling massive scalability and global accessibility. S3 has become the foundation for modern cloud-native applications, data lakes, backup solutions, and content delivery systems.

**Core Philosophy:**

- **Object-Based Storage**: Data is stored as objects with unique keys, metadata, and version information
- **Unlimited Scalability**: Designed to store and retrieve any amount of data from anywhere
- **High Durability**: 99.999999999% (11 9's) durability with automatic replication
- **Global Accessibility**: Access data from anywhere in the world via HTTP/HTTPS
- **Pay-as-You-Use**: Only pay for the storage you actually use
- **Integration-First**: Seamlessly integrates with other AWS services and third-party applications

**Why S3 Matters:**

- **Scalability**: Handle petabytes of data without performance degradation
- **Durability**: Enterprise-grade data protection with automatic replication
- **Cost-Effectiveness**: Pay only for what you use with no upfront costs
- **Global Reach**: Deploy applications globally with consistent performance
- **Developer-Friendly**: Simple REST API for easy integration
- **Compliance**: Meet regulatory requirements with built-in compliance features
- **Innovation**: Enable new use cases like data lakes, machine learning, and analytics
- **Disaster Recovery**: Built-in backup and recovery capabilities

**Key Features:**

**1. Object Storage:**

- **Definition**: Store files as objects with unique identifiers, metadata, and version information
- **Purpose**: Provide scalable, durable storage for any type of data
- **Benefits**: Unlimited scalability, global accessibility, simple API, cost-effective
- **Use Cases**: Web applications, data lakes, backup, content delivery, analytics
- **Best Practices**: Use meaningful object keys, implement proper naming conventions, leverage metadata

**2. Versioning:**

- **Definition**: Keep multiple versions of objects to protect against accidental deletion and enable rollback
- **Purpose**: Provide data protection and change management capabilities
- **Benefits**: Data protection, rollback capabilities, compliance support, audit trails
- **Use Cases**: Document management, configuration files, backup systems, compliance requirements
- **Best Practices**: Enable versioning for critical data, implement lifecycle policies for old versions, monitor version costs

**3. Lifecycle Policies:**

- **Definition**: Automatically transition objects between storage classes or delete them based on age or other criteria
- **Purpose**: Optimize costs by moving data to appropriate storage classes over time
- **Benefits**: Cost optimization, automated data management, compliance support
- **Use Cases**: Data archiving, cost optimization, compliance requirements, automated cleanup
- **Best Practices**: Design policies based on data access patterns, monitor policy effectiveness, test policies in non-production

**4. Cross-Region Replication:**

- **Definition**: Automatically replicate objects to buckets in different AWS regions
- **Purpose**: Provide disaster recovery, compliance, and global data distribution
- **Benefits**: Disaster recovery, compliance, reduced latency, data sovereignty
- **Use Cases**: Disaster recovery, compliance requirements, global applications, data sovereignty
- **Best Practices**: Consider replication costs, implement proper IAM roles, monitor replication status

**5. Event Notifications:**

- **Definition**: Trigger actions when objects are created, updated, or deleted in S3 buckets
- **Purpose**: Enable event-driven architectures and automated workflows
- **Benefits**: Real-time processing, automated workflows, integration capabilities
- **Use Cases**: Image processing, data validation, backup automation, analytics pipelines
- **Best Practices**: Use appropriate event types, implement proper error handling, monitor event processing

**6. Access Control:**

- **Definition**: Fine-grained permissions using IAM policies, bucket policies, and ACLs
- **Purpose**: Secure data access and ensure compliance with security requirements
- **Benefits**: Security, compliance, audit trails, flexible access control
- **Use Cases**: Multi-tenant applications, compliance requirements, secure data sharing
- **Best Practices**: Use least privilege principle, implement proper access logging, regular access reviews

**Advanced S3 Concepts:**

- **Storage Classes**: Standard, Standard-IA, Intelligent-Tiering, Glacier, Deep Archive for different access patterns
- **Transfer Acceleration**: Use CloudFront edge locations for faster uploads
- **Multipart Upload**: Upload large files in parallel for better performance
- **Presigned URLs**: Generate temporary URLs for secure access without exposing credentials
- **Server-Side Encryption**: Encrypt data at rest using AWS-managed or customer-managed keys
- **Access Logging**: Log all access requests for security and compliance
- **Request Payment**: Require requesters to pay for data transfer costs
- **Object Lock**: Prevent objects from being deleted or overwritten for compliance

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive S3 strategy for a large-scale data lake architecture with multiple data sources and analytics requirements?**

**Answer:** Comprehensive S3 data lake strategy design:

- **Data Organization**: Implement proper partitioning and naming conventions for efficient querying
- **Storage Classes**: Use Intelligent-Tiering for automatic cost optimization based on access patterns
- **Data Formats**: Use columnar formats (Parquet, ORC) for analytics workloads
- **Compression**: Implement appropriate compression algorithms to reduce storage costs
- **Lifecycle Policies**: Design policies based on data access patterns and retention requirements
- **Security**: Implement encryption at rest and in transit, proper IAM policies, and access logging
- **Monitoring**: Use CloudWatch and S3 analytics to monitor usage and performance
- **Integration**: Integrate with analytics services (Athena, Redshift, EMR) for data processing
- **Backup**: Implement cross-region replication for disaster recovery
- **Compliance**: Ensure compliance with regulatory requirements through proper data governance
- **Cost Optimization**: Monitor and optimize costs through storage class transitions and lifecycle policies
- **Documentation**: Maintain comprehensive documentation of data lake architecture and processes

**Q2: What are the key considerations when implementing S3 security and compliance for enterprise applications handling sensitive data?**

**Answer:** S3 security and compliance implementation:

- **Encryption**: Implement server-side encryption using AWS KMS for sensitive data
- **Access Control**: Use IAM policies, bucket policies, and VPC endpoints for secure access
- **Network Security**: Implement VPC endpoints and security groups for network isolation
- **Audit Logging**: Enable CloudTrail and S3 access logging for comprehensive audit trails
- **Data Classification**: Implement proper data classification and tagging for sensitive data
- **Compliance Frameworks**: Ensure compliance with SOC2, PCI DSS, HIPAA, and other relevant frameworks
- **Access Monitoring**: Implement real-time monitoring and alerting for suspicious access patterns
- **Data Residency**: Ensure data stays within required geographic regions for compliance
- **Backup and Recovery**: Implement secure backup and recovery procedures for sensitive data
- **Incident Response**: Have clear procedures for responding to security incidents
- **Regular Audits**: Conduct regular security audits and penetration testing
- **Training**: Provide security training for teams working with sensitive data

**Q3: How do you optimize S3 performance and costs for high-frequency data processing and analytics workloads?**

**Answer:** S3 performance and cost optimization:

- **Storage Class Optimization**: Use appropriate storage classes based on access patterns and cost requirements
- **Lifecycle Policies**: Implement intelligent lifecycle policies to automatically transition data to cost-effective storage classes
- **Data Partitioning**: Partition data properly to enable efficient querying and reduce costs
- **Compression**: Use appropriate compression algorithms to reduce storage costs and improve transfer speeds
- **Multipart Uploads**: Use multipart uploads for large files to improve upload performance
- **Transfer Acceleration**: Use S3 Transfer Acceleration for faster uploads from distant locations
- **CloudFront Integration**: Use CloudFront for frequently accessed data to reduce latency and costs
- **Request Optimization**: Optimize API requests to reduce costs and improve performance
- **Monitoring**: Use S3 analytics and CloudWatch to monitor usage patterns and optimize accordingly
- **Cost Allocation**: Implement proper cost allocation tags to track and optimize spending
- **Data Lifecycle**: Implement proper data lifecycle management to delete unnecessary data
- **Performance Testing**: Conduct regular performance testing to identify optimization opportunities

## 🏗️ S3 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    S3 Bucket                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Object    │  │   Object    │  │   Object    │     │
│  │   Key:      │  │   Key:      │  │   Key:      │     │
│  │   file1.txt │  │   file2.jpg │  │   file3.pdf │     │
│  │   Version:  │  │   Version:  │  │   Version:  │     │
│  │   1, 2, 3   │  │   1, 2      │  │   1         │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Metadata  │  │   Metadata  │  │   Metadata  │     │
│  │   Content-  │  │   Content-  │  │   Content-  │     │
│  │   Type:     │  │   Type:     │  │   Type:     │     │
│  │   text/plain│  │   image/jpeg│  │   application│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### S3 Bucket with CloudFormation

```yaml
# s3-bucket.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "S3 bucket with versioning, lifecycle, and replication"

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

  BucketName:
    Type: String
    Default: my-app-bucket
    Description: Name for the S3 bucket

Resources:
  # S3 Bucket
  S3Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${BucketName}-${Environment}-${AWS::AccountId}"
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      LifecycleConfiguration:
        Rules:
          - Id: DeleteIncompleteMultipartUploads
            Status: Enabled
            AbortIncompleteMultipartUpload:
              DaysAfterInitiation: 7
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - StorageClass: STANDARD_IA
                TransitionInDays: 30
              - StorageClass: GLACIER
                TransitionInDays: 90
              - StorageClass: DEEP_ARCHIVE
                TransitionInDays: 365
          - Id: DeleteOldVersions
            Status: Enabled
            NoncurrentVersionTransitions:
              - StorageClass: STANDARD_IA
                TransitionInDays: 30
              - StorageClass: GLACIER
                TransitionInDays: 90
            NoncurrentVersionExpiration:
              NoncurrentDays: 2555 # 7 years
      NotificationConfiguration:
        LambdaConfigurations:
          - Event: s3:ObjectCreated:*
            Function: !GetAtt ProcessUploadFunction.Arn
            Filter:
              S3Key:
                Rules:
                  - Name: prefix
                    Value: uploads/
        TopicConfigurations:
          - Event: s3:ObjectRemoved:*
            Topic: !Ref S3EventTopic
            Filter:
              S3Key:
                Rules:
                  - Name: suffix
                    Value: .log
      CorsConfiguration:
        CorsRules:
          - AllowedHeaders: ["*"]
            AllowedMethods: [GET, PUT, POST, DELETE, HEAD]
            AllowedOrigins: ["*"]
            MaxAge: 3000
      Tags:
        - Key: Name
          Value: !Sub "${BucketName}-${Environment}"
        - Key: Environment
          Value: !Ref Environment

  # S3 Bucket Policy
  S3BucketPolicy:
    Type: AWS::S3::BucketPolicy
    Properties:
      Bucket: !Ref S3Bucket
      PolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Sid: AllowCloudFrontAccess
            Effect: Allow
            Principal:
              Service: cloudfront.amazonaws.com
            Action: s3:GetObject
            Resource: !Sub "${S3Bucket}/*"
            Condition:
              StringEquals:
                "AWS:SourceArn": !Sub "arn:aws:cloudfront::${AWS::AccountId}:distribution/${CloudFrontDistribution}"
          - Sid: DenyInsecureConnections
            Effect: Deny
            Principal: "*"
            Action: s3:*
            Resource: !Sub "${S3Bucket}/*"
            Condition:
              Bool:
                "aws:SecureTransport": "false"

  # CloudFront Distribution
  CloudFrontDistribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - DomainName: !GetAtt S3Bucket.RegionalDomainName
            Id: S3Origin
            S3OriginConfig:
              OriginAccessIdentity: !Sub "origin-access-identity/cloudfront/${CloudFrontOriginAccessIdentity}"
        Enabled: true
        DefaultRootObject: index.html
        DefaultCacheBehavior:
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: redirect-to-https
          AllowedMethods: [GET, HEAD, OPTIONS]
          CachedMethods: [GET, HEAD]
          ForwardedValues:
            QueryString: false
            Cookies:
              Forward: none
          Compress: true
        PriceClass: PriceClass_100
        ViewerCertificate:
          CloudFrontDefaultCertificate: true
        CustomErrorResponses:
          - ErrorCode: 404
            ResponseCode: 200
            ResponsePagePath: /index.html
        Tags:
          - Key: Name
            Value: !Sub "${BucketName}-${Environment}-cdn"

  # CloudFront Origin Access Identity
  CloudFrontOriginAccessIdentity:
    Type: AWS::CloudFront::CloudFrontOriginAccessIdentity
    Properties:
      CloudFrontOriginAccessIdentityConfig:
        Comment: !Sub "OAI for ${BucketName}-${Environment}"

  # S3 Event Topic
  S3EventTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub "${BucketName}-${Environment}-s3-events"
      DisplayName: S3 Event Notifications

  # Lambda Function for Processing Uploads
  ProcessUploadFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub "${BucketName}-${Environment}-process-upload"
      Runtime: python3.9
      Handler: index.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        ZipFile: |
          import json
          import boto3

          def lambda_handler(event, context):
              s3 = boto3.client('s3')
              
              for record in event['Records']:
                  bucket = record['s3']['bucket']['name']
                  key = record['s3']['object']['key']
                  
                  print(f"Processing upload: s3://{bucket}/{key}")
                  
                  # Add your processing logic here
                  # e.g., image resizing, virus scanning, etc.
                  
              return {
                  'statusCode': 200,
                  'body': json.dumps('Upload processed successfully')
              }
      Environment:
        Variables:
          BUCKET_NAME: !Ref S3Bucket
      Tags:
        - Key: Name
          Value: !Sub "${BucketName}-${Environment}-process-upload"

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
        - PolicyName: S3Access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                Resource: !Sub "${S3Bucket}/*"

  # Lambda Permission
  LambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref ProcessUploadFunction
      Action: lambda:InvokeFunction
      Principal: s3.amazonaws.com
      SourceArn: !Sub "${S3Bucket}/*"

  # Cross-Region Replication (Optional)
  ReplicationRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: s3.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: ReplicationPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObjectVersion
                  - s3:GetObjectVersionAcl
                Resource: !Sub "${S3Bucket}/*"
              - Effect: Allow
                Action:
                  - s3:ReplicateObject
                  - s3:ReplicateDelete
                Resource: !Sub "arn:aws:s3:::${BucketName}-${Environment}-replica-${AWS::AccountId}/*"

  # Replica Bucket
  ReplicaBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${BucketName}-${Environment}-replica-${AWS::AccountId}"
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      Tags:
        - Key: Name
          Value: !Sub "${BucketName}-${Environment}-replica"
        - Key: Environment
          Value: !Ref Environment

Outputs:
  BucketName:
    Description: S3 Bucket Name
    Value: !Ref S3Bucket
    Export:
      Name: !Sub "${Environment}-S3-Bucket-Name"

  BucketDomainName:
    Description: S3 Bucket Domain Name
    Value: !GetAtt S3Bucket.DomainName
    Export:
      Name: !Sub "${Environment}-S3-Bucket-Domain"

  CloudFrontDistributionId:
    Description: CloudFront Distribution ID
    Value: !Ref CloudFrontDistribution
    Export:
      Name: !Sub "${Environment}-CloudFront-Distribution-ID"

  CloudFrontDomainName:
    Description: CloudFront Domain Name
    Value: !GetAtt CloudFrontDistribution.DomainName
    Export:
      Name: !Sub "${Environment}-CloudFront-Domain"
```

### S3 with Terraform

```hcl
# s3.tf
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

variable "bucket_name" {
  description = "S3 bucket name"
  type        = string
  default     = "my-app-bucket"
}

# S3 Bucket
resource "aws_s3_bucket" "main" {
  bucket = "${var.bucket_name}-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.bucket_name}-${var.environment}"
    Environment = var.environment
  }
}

# Random ID for bucket suffix
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket Lifecycle Configuration
resource "aws_s3_bucket_lifecycle_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    id     = "DeleteIncompleteMultipartUploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "TransitionToIA"
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
  }

  rule {
    id     = "DeleteOldVersions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 2555  # 7 years
    }
  }
}

# S3 Bucket CORS Configuration
resource "aws_s3_bucket_cors_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}

# S3 Bucket Notification
resource "aws_s3_bucket_notification" "main" {
  bucket = aws_s3_bucket.main.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.process_upload.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }

  topic {
    topic_arn     = aws_sns_topic.s3_events.arn
    events        = ["s3:ObjectRemoved:*"]
    filter_suffix = ".log"
  }
}

# S3 Bucket Policy
resource "aws_s3_bucket_policy" "main" {
  bucket = aws_s3_bucket.main.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontAccess"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.main.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.main.arn
          }
        }
      },
      {
        Sid    = "DenyInsecureConnections"
        Effect = "Deny"
        Principal = "*"
        Action   = "s3:*"
        Resource = "${aws_s3_bucket.main.arn}/*"
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# CloudFront Origin Access Identity
resource "aws_cloudfront_origin_access_identity" "main" {
  comment = "OAI for ${var.bucket_name}-${var.environment}"
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_s3_bucket.main.bucket_regional_domain_name
    origin_id   = "S3Origin"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main.cloudfront_access_identity_path
    }
  }

  enabled             = true
  default_root_object = "index.html"

  default_cache_behavior {
    target_origin_id       = "S3Origin"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    compress = true
  }

  price_class = "PriceClass_100"

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  tags = {
    Name        = "${var.bucket_name}-${var.environment}-cdn"
    Environment = var.environment
  }
}

# SNS Topic for S3 Events
resource "aws_sns_topic" "s3_events" {
  name = "${var.bucket_name}-${var.environment}-s3-events"

  tags = {
    Name        = "${var.bucket_name}-${var.environment}-s3-events"
    Environment = var.environment
  }
}

# Lambda Function for Processing Uploads
resource "aws_lambda_function" "process_upload" {
  function_name = "${var.bucket_name}-${var.environment}-process-upload"
  runtime       = "python3.9"
  handler       = "index.lambda_handler"
  role          = aws_iam_role.lambda_execution.arn

  filename         = "lambda_function.zip"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      BUCKET_NAME = aws_s3_bucket.main.bucket
    }
  }

  tags = {
    Name        = "${var.bucket_name}-${var.environment}-process-upload"
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

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Processing upload: s3://{bucket}/{key}")

        # Add your processing logic here
        # e.g., image resizing, virus scanning, etc.

    return {
        'statusCode': 200,
        'body': json.dumps('Upload processed successfully')
    }
EOF
    filename = "index.py"
  }
}

# Lambda Execution Role
resource "aws_iam_role" "lambda_execution" {
  name = "${var.bucket_name}-${var.environment}-lambda-execution"

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
  name = "${var.bucket_name}-${var.environment}-lambda-execution"
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
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.main.arn}/*"
      }
    ]
  })
}

# Lambda Permission
resource "aws_lambda_permission" "s3_invoke" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.process_upload.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.main.arn
}

# Outputs
output "bucket_name" {
  description = "S3 Bucket Name"
  value       = aws_s3_bucket.main.bucket
}

output "bucket_domain_name" {
  description = "S3 Bucket Domain Name"
  value       = aws_s3_bucket.main.bucket_domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront Distribution ID"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain_name" {
  description = "CloudFront Domain Name"
  value       = aws_cloudfront_distribution.main.domain_name
}
```

### S3 Operations with Go

```go
package main

import (
    "bytes"
    "context"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"

    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/credentials"
    "github.com/aws/aws-sdk-go-v2/service/s3"
    "github.com/aws/aws-sdk-go-v2/service/s3/types"
)

type S3Service struct {
    client *s3.Client
    bucket string
}

func NewS3Service(bucket string) (*S3Service, error) {
    cfg, err := config.LoadDefaultConfig(context.TODO())
    if err != nil {
        return nil, err
    }

    client := s3.NewFromConfig(cfg)

    return &S3Service{
        client: client,
        bucket: bucket,
    }, nil
}

// Upload file to S3
func (s *S3Service) UploadFile(ctx context.Context, key string, filePath string) error {
    file, err := os.Open(filePath)
    if err != nil {
        return err
    }
    defer file.Close()

    _, err = s.client.PutObject(ctx, &s3.PutObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
        Body:   file,
    })

    return err
}

// Upload data to S3
func (s *S3Service) UploadData(ctx context.Context, key string, data []byte) error {
    _, err := s.client.PutObject(ctx, &s3.PutObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
        Body:   bytes.NewReader(data),
    })

    return err
}

// Download file from S3
func (s *S3Service) DownloadFile(ctx context.Context, key string, filePath string) error {
    result, err := s.client.GetObject(ctx, &s3.GetObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return err
    }
    defer result.Body.Close()

    file, err := os.Create(filePath)
    if err != nil {
        return err
    }
    defer file.Close()

    _, err = io.Copy(file, result.Body)
    return err
}

// Download data from S3
func (s *S3Service) DownloadData(ctx context.Context, key string) ([]byte, error) {
    result, err := s.client.GetObject(ctx, &s3.GetObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    defer result.Body.Close()

    return io.ReadAll(result.Body)
}

// List objects in bucket
func (s *S3Service) ListObjects(ctx context.Context, prefix string) ([]types.Object, error) {
    result, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
        Bucket: aws.String(s.bucket),
        Prefix: aws.String(prefix),
    })
    if err != nil {
        return nil, err
    }

    return result.Contents, nil
}

// Delete object from S3
func (s *S3Service) DeleteObject(ctx context.Context, key string) error {
    _, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })

    return err
}

// Copy object within S3
func (s *S3Service) CopyObject(ctx context.Context, sourceKey, destKey string) error {
    source := fmt.Sprintf("%s/%s", s.bucket, sourceKey)

    _, err := s.client.CopyObject(ctx, &s3.CopyObjectInput{
        Bucket:     aws.String(s.bucket),
        CopySource: aws.String(source),
        Key:        aws.String(destKey),
    })

    return err
}

// Generate presigned URL for upload
func (s *S3Service) GeneratePresignedUploadURL(ctx context.Context, key string, expiration time.Duration) (string, error) {
    request, err := s.client.PutObjectRequest(&s3.PutObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return "", err
    }

    return request.Presign(expiration)
}

// Generate presigned URL for download
func (s *S3Service) GeneratePresignedDownloadURL(ctx context.Context, key string, expiration time.Duration) (string, error) {
    request, err := s.client.GetObjectRequest(&s3.GetObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return "", err
    }

    return request.Presign(expiration)
}

// Set object metadata
func (s *S3Service) SetObjectMetadata(ctx context.Context, key string, metadata map[string]string) error {
    _, err := s.client.CopyObject(ctx, &s3.CopyObjectInput{
        Bucket:     aws.String(s.bucket),
        CopySource: aws.String(fmt.Sprintf("%s/%s", s.bucket, key)),
        Key:        aws.String(key),
        Metadata:   metadata,
        MetadataDirective: types.MetadataDirectiveReplace,
    })

    return err
}

// Get object metadata
func (s *S3Service) GetObjectMetadata(ctx context.Context, key string) (map[string]string, error) {
    result, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }

    return result.Metadata, nil
}

// Enable versioning
func (s *S3Service) EnableVersioning(ctx context.Context) error {
    _, err := s.client.PutBucketVersioning(ctx, &s3.PutBucketVersioningInput{
        Bucket: aws.String(s.bucket),
        VersioningConfiguration: &types.VersioningConfiguration{
            Status: types.BucketVersioningStatusEnabled,
        },
    })

    return err
}

// List object versions
func (s *S3Service) ListObjectVersions(ctx context.Context, prefix string) ([]types.ObjectVersion, error) {
    result, err := s.client.ListObjectVersions(ctx, &s3.ListObjectVersionsInput{
        Bucket: aws.String(s.bucket),
        Prefix: aws.String(prefix),
    })
    if err != nil {
        return nil, err
    }

    return result.Versions, nil
}

// Delete object version
func (s *S3Service) DeleteObjectVersion(ctx context.Context, key, versionID string) error {
    _, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
        Bucket:    aws.String(s.bucket),
        Key:       aws.String(key),
        VersionId: aws.String(versionID),
    })

    return err
}

// Set lifecycle policy
func (s *S3Service) SetLifecyclePolicy(ctx context.Context, rules []types.LifecycleRule) error {
    _, err := s.client.PutBucketLifecycleConfiguration(ctx, &s3.PutBucketLifecycleConfigurationInput{
        Bucket: aws.String(s.bucket),
        LifecycleConfiguration: &types.BucketLifecycleConfiguration{
            Rules: rules,
        },
    })

    return err
}

// Example usage
func main() {
    s3Service, err := NewS3Service("my-bucket")
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Upload a file
    err = s3Service.UploadFile(ctx, "uploads/file.txt", "/path/to/local/file.txt")
    if err != nil {
        log.Printf("Error uploading file: %v", err)
    }

    // Upload data
    data := []byte("Hello, S3!")
    err = s3Service.UploadData(ctx, "data/hello.txt", data)
    if err != nil {
        log.Printf("Error uploading data: %v", err)
    }

    // Download a file
    err = s3Service.DownloadFile(ctx, "uploads/file.txt", "/path/to/download/file.txt")
    if err != nil {
        log.Printf("Error downloading file: %v", err)
    }

    // List objects
    objects, err := s3Service.ListObjects(ctx, "uploads/")
    if err != nil {
        log.Printf("Error listing objects: %v", err)
    } else {
        for _, obj := range objects {
            fmt.Printf("Object: %s, Size: %d, LastModified: %v\n",
                *obj.Key, *obj.Size, *obj.LastModified)
        }
    }

    // Generate presigned URL
    url, err := s3Service.GeneratePresignedUploadURL(ctx, "uploads/presigned.txt", time.Hour)
    if err != nil {
        log.Printf("Error generating presigned URL: %v", err)
    } else {
        fmt.Printf("Presigned URL: %s\n", url)
    }

    // Set lifecycle policy
    rules := []types.LifecycleRule{
        {
            ID:     aws.String("DeleteIncompleteMultipartUploads"),
            Status: types.ExpirationStatusEnabled,
            AbortIncompleteMultipartUpload: &types.AbortIncompleteMultipartUpload{
                DaysAfterInitiation: aws.Int32(7),
            },
        },
        {
            ID:     aws.String("TransitionToIA"),
            Status: types.ExpirationStatusEnabled,
            Transitions: []types.Transition{
                {
                    Days:         aws.Int32(30),
                    StorageClass: types.TransitionStorageClassStandardIa,
                },
                {
                    Days:         aws.Int32(90),
                    StorageClass: types.TransitionStorageClassGlacier,
                },
            },
        },
    }

    err = s3Service.SetLifecyclePolicy(ctx, rules)
    if err != nil {
        log.Printf("Error setting lifecycle policy: %v", err)
    }
}
```

## 🚀 Best Practices

### 1. Security Best Practices

```yaml
# Secure S3 bucket configuration
S3Bucket:
  Type: AWS::S3::Bucket
  Properties:
    PublicAccessBlockConfiguration:
      BlockPublicAcls: true
      BlockPublicPolicy: true
      IgnorePublicAcls: true
      RestrictPublicBuckets: true
    BucketEncryption:
      ServerSideEncryptionConfiguration:
        - ServerSideEncryptionByDefault:
            SSEAlgorithm: AES256
    BucketPolicy:
      Statement:
        - Effect: Deny
          Principal: "*"
          Action: s3:*
          Resource: !Sub "${S3Bucket}/*"
          Condition:
            Bool:
              "aws:SecureTransport": "false"
```

### 2. Cost Optimization

```yaml
# Lifecycle configuration for cost optimization
LifecycleConfiguration:
  Rules:
    - Id: TransitionToIA
      Status: Enabled
      Transitions:
        - StorageClass: STANDARD_IA
          TransitionInDays: 30
        - StorageClass: GLACIER
          TransitionInDays: 90
        - StorageClass: DEEP_ARCHIVE
          TransitionInDays: 365
    - Id: DeleteOldVersions
      Status: Enabled
      NoncurrentVersionExpiration:
        NoncurrentDays: 2555 # 7 years
```

### 3. Performance Optimization

```yaml
# CloudFront distribution for performance
CloudFrontDistribution:
  Type: AWS::CloudFront::Distribution
  Properties:
    DistributionConfig:
      Origins:
        - DomainName: !GetAtt S3Bucket.RegionalDomainName
          Id: S3Origin
          S3OriginConfig:
            OriginAccessIdentity: !Sub "origin-access-identity/cloudfront/${CloudFrontOriginAccessIdentity}"
      DefaultCacheBehavior:
        TargetOriginId: S3Origin
        ViewerProtocolPolicy: redirect-to-https
        Compress: true
        CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad # CachingOptimized
```

## 🏢 Industry Insights

### Netflix's S3 Usage

- **Content Delivery**: Video content storage
- **Data Lake**: Analytics and machine learning
- **Backup**: Disaster recovery
- **Cross-Region Replication**: Global content distribution

### Airbnb's S3 Strategy

- **Image Storage**: Property photos
- **Data Analytics**: User behavior data
- **Backup**: Database backups
- **CDN Integration**: CloudFront for global delivery

### Spotify's S3 Approach

- **Music Storage**: Audio file storage
- **Data Processing**: ETL pipelines
- **Analytics**: User listening data
- **Machine Learning**: Recommendation systems

## 🎯 Interview Questions

### Basic Level

1. **What is S3?**

   - Simple Storage Service
   - Object storage in the cloud
   - Highly scalable and durable
   - Pay-per-use pricing

2. **What are S3 storage classes?**

   - Standard: Frequently accessed data
   - Standard-IA: Infrequently accessed data
   - Glacier: Archive data
   - Deep Archive: Long-term archive

3. **What is S3 versioning?**
   - Keep multiple versions of objects
   - Protect against accidental deletion
   - Enable rollback capabilities
   - Additional storage costs

### Intermediate Level

4. **How do you implement S3 security?**

   ```yaml
   # Secure S3 bucket
   S3Bucket:
     Type: AWS::S3::Bucket
     Properties:
       PublicAccessBlockConfiguration:
         BlockPublicAcls: true
         BlockPublicPolicy: true
         IgnorePublicAcls: true
         RestrictPublicBuckets: true
       BucketEncryption:
         ServerSideEncryptionConfiguration:
           - ServerSideEncryptionByDefault:
               SSEAlgorithm: AES256
   ```

5. **How do you optimize S3 costs?**

   - Lifecycle policies
   - Storage class transitions
   - Intelligent tiering
   - Delete incomplete multipart uploads

6. **How do you handle S3 events?**
   - S3 event notifications
   - Lambda triggers
   - SNS topics
   - SQS queues

### Advanced Level

7. **How do you implement cross-region replication?**

   - Replication configuration
   - IAM roles
   - Source and destination buckets
   - Replication rules

8. **How do you handle S3 performance?**

   - CloudFront integration
   - Transfer acceleration
   - Multipart uploads
   - Parallel processing

9. **How do you implement S3 data lake?**
   - Partitioning strategies
   - Data formats (Parquet, ORC)
   - Query optimization
   - Analytics services integration

---

**Next**: [AWS Lambda](AWS_Lambda.md) - Serverless functions, event-driven architecture
