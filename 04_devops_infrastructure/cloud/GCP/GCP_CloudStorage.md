# 🗄️ GCP Cloud Storage: Object Storage, Versioning, and Lifecycle Policies

> **Master Google Cloud Storage for scalable object storage and data management**

## 📚 Concept

**Detailed Explanation:**
Google Cloud Storage is a revolutionary object storage service that provides highly scalable, durable, and secure storage for any amount of data. Unlike traditional file systems, Cloud Storage treats data as objects with unique names, metadata, and version information, enabling massive scalability and global accessibility. It has become the foundation for modern cloud-native applications, data lakes, backup solutions, and content delivery systems.

**Core Philosophy:**

- **Object-Based Storage**: Data is stored as objects with unique names, metadata, and version information
- **Unlimited Scalability**: Designed to store and retrieve any amount of data from anywhere
- **High Durability**: 99.999999999% (11 9's) durability with automatic replication
- **Global Accessibility**: Access data from anywhere in the world via HTTP/HTTPS
- **Pay-as-You-Use**: Only pay for the storage you actually use
- **Integration-First**: Seamlessly integrates with other Google Cloud services and third-party applications

**Why Cloud Storage Matters:**

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

- **Definition**: Store files as objects with unique names, metadata, and version information
- **Purpose**: Provide scalable, durable storage for any type of data
- **Benefits**: Unlimited scalability, global accessibility, simple API, cost-effective
- **Use Cases**: Web applications, data lakes, backup, content delivery, analytics
- **Best Practices**: Use meaningful object names, implement proper naming conventions, leverage metadata

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

- **Definition**: Automatically replicate objects to buckets in different Google Cloud regions
- **Purpose**: Provide disaster recovery, compliance, and global data distribution
- **Benefits**: Disaster recovery, compliance, reduced latency, data sovereignty
- **Use Cases**: Disaster recovery, compliance requirements, global applications, data sovereignty
- **Best Practices**: Consider replication costs, implement proper IAM roles, monitor replication status

**5. Event Notifications:**

- **Definition**: Trigger actions when objects are created, updated, or deleted in Cloud Storage buckets
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

**Advanced Cloud Storage Concepts:**

- **Storage Classes**: Standard, Nearline, Coldline, Archive for different access patterns
- **Transfer Acceleration**: Use Cloud CDN for faster uploads and downloads
- **Multipart Upload**: Upload large files in parallel for better performance
- **Signed URLs**: Generate temporary URLs for secure access without exposing credentials
- **Server-Side Encryption**: Encrypt data at rest using Google-managed or customer-managed keys
- **Access Logging**: Log all access requests for security and compliance
- **Request Payment**: Require requesters to pay for data transfer costs
- **Object Lock**: Prevent objects from being deleted or overwritten for compliance

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive Cloud Storage strategy for a large-scale data lake architecture with multiple data sources and analytics requirements?**

**Answer:** Comprehensive Cloud Storage data lake strategy design:

- **Data Organization**: Implement proper partitioning and naming conventions for efficient querying
- **Storage Classes**: Use appropriate storage classes based on access patterns and cost requirements
- **Data Formats**: Use columnar formats (Parquet, ORC) for analytics workloads
- **Compression**: Implement appropriate compression algorithms to reduce storage costs
- **Lifecycle Policies**: Design policies based on data access patterns and retention requirements
- **Security**: Implement encryption at rest and in transit, proper IAM policies, and access logging
- **Monitoring**: Use Cloud Monitoring and Cloud Storage analytics to monitor usage and performance
- **Integration**: Integrate with analytics services (BigQuery, Dataflow, Dataproc) for data processing
- **Backup**: Implement cross-region replication for disaster recovery
- **Compliance**: Ensure compliance with regulatory requirements through proper data governance
- **Cost Optimization**: Monitor and optimize costs through storage class transitions and lifecycle policies
- **Documentation**: Maintain comprehensive documentation of data lake architecture and processes

**Q2: What are the key considerations when implementing Cloud Storage security and compliance for enterprise applications handling sensitive data?**

**Answer:** Cloud Storage security and compliance implementation:

- **Encryption**: Implement server-side encryption using Cloud KMS for sensitive data
- **Access Control**: Use IAM policies, bucket policies, and VPC Service Controls for secure access
- **Network Security**: Implement VPC Service Controls and firewall rules for network isolation
- **Audit Logging**: Enable Cloud Audit Logs and Cloud Storage access logging for comprehensive audit trails
- **Data Classification**: Implement proper data classification and tagging for sensitive data
- **Compliance Frameworks**: Ensure compliance with SOC2, PCI DSS, HIPAA, and other relevant frameworks
- **Access Monitoring**: Implement real-time monitoring and alerting for suspicious access patterns
- **Data Residency**: Ensure data stays within required geographic regions for compliance
- **Backup and Recovery**: Implement secure backup and recovery procedures for sensitive data
- **Incident Response**: Have clear procedures for responding to security incidents
- **Regular Audits**: Conduct regular security audits and penetration testing
- **Training**: Provide security training for teams working with sensitive data

**Q3: How do you optimize Cloud Storage performance and costs for high-frequency data processing and analytics workloads?**

**Answer:** Cloud Storage performance and cost optimization:

- **Storage Class Optimization**: Use appropriate storage classes based on access patterns and cost requirements
- **Lifecycle Policies**: Implement intelligent lifecycle policies to automatically transition data to cost-effective storage classes
- **Data Partitioning**: Partition data properly to enable efficient querying and reduce costs
- **Compression**: Use appropriate compression algorithms to reduce storage costs and improve transfer speeds
- **Multipart Uploads**: Use multipart uploads for large files to improve upload performance
- **Cloud CDN Integration**: Use Cloud CDN for frequently accessed data to reduce latency and costs
- **Request Optimization**: Optimize API requests to reduce costs and improve performance
- **Monitoring**: Use Cloud Storage analytics and Cloud Monitoring to monitor usage patterns and optimize accordingly
- **Cost Allocation**: Implement proper cost allocation tags to track and optimize spending
- **Data Lifecycle**: Implement proper data lifecycle management to delete unnecessary data
- **Performance Testing**: Conduct regular performance testing to identify optimization opportunities

## 🏗️ Cloud Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Cloud Storage Bucket               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Object    │  │   Object    │  │   Object    │     │
│  │   Name:     │  │   Name:     │  │   Name:     │     │
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

### Cloud Storage with Deployment Manager

```yaml
# cloud-storage.yaml
imports:
  - path: storage-bucket.jinja

resources:
  # Cloud Storage Bucket
  - name: production-storage-bucket
    type: storage-bucket.jinja
    properties:
      name: production-storage-bucket-${PROJECT_ID}
      location: US
      storageClass: STANDARD
      versioning:
        enabled: true
      lifecycle:
        rule:
          - action:
              type: Delete
            condition:
              age: 365
          - action:
              type: SetStorageClass
              storageClass: NEARLINE
            condition:
              age: 30
          - action:
              type: SetStorageClass
              storageClass: COLDLINE
            condition:
              age: 90
          - action:
              type: SetStorageClass
              storageClass: ARCHIVE
            condition:
              age: 365
      cors:
        - origin: ["*"]
          method: ["GET", "PUT", "POST", "DELETE", "HEAD"]
          responseHeader: ["Content-Type"]
          maxAgeSeconds: 3600
      labels:
        environment: production
        team: backend

  # Cloud Storage Bucket for staging
  - name: staging-storage-bucket
    type: storage-bucket.jinja
    properties:
      name: staging-storage-bucket-${PROJECT_ID}
      location: US
      storageClass: STANDARD
      versioning:
        enabled: true
      lifecycle:
        rule:
          - action:
              type: Delete
            condition:
              age: 30
      labels:
        environment: staging
        team: backend

  # Cloud Storage Bucket for development
  - name: dev-storage-bucket
    type: storage-bucket.jinja
    properties:
      name: dev-storage-bucket-${PROJECT_ID}
      location: US
      storageClass: STANDARD
      versioning:
        enabled: false
      lifecycle:
        rule:
          - action:
              type: Delete
            condition:
              age: 7
      labels:
        environment: dev
        team: backend

  # Cloud Function for processing uploads
  - name: process-upload-function
    type: gcp-types/cloudfunctions-v1:projects.locations.functions
    properties:
      parent: projects/${PROJECT_ID}/locations/us-central1
      function: process-upload-function
      sourceArchiveUrl: gs://${PROJECT_ID}-functions/process-upload.zip
      entryPoint: processUpload
      runtime: python39
      eventTrigger:
        eventType: google.storage.object.finalize
        resource: production-storage-bucket-${PROJECT_ID}
      environmentVariables:
        BUCKET_NAME: production-storage-bucket-${PROJECT_ID}
      labels:
        environment: production
        team: backend

  # Pub/Sub Topic for storage events
  - name: storage-events-topic
    type: gcp-types/pubsub-v1:projects.topics
    properties:
      topic: storage-events
      labels:
        environment: production
        team: backend

  # Pub/Sub Subscription
  - name: storage-events-subscription
    type: gcp-types/pubsub-v1:projects.subscriptions
    properties:
      subscription: storage-events-subscription
      topic: projects/${PROJECT_ID}/topics/storage-events
      ackDeadlineSeconds: 60
      labels:
        environment: production
        team: backend

  # Cloud Storage Notification
  - name: storage-notification
    type: gcp-types/storage-v1:notifications
    properties:
      bucket: production-storage-bucket-${PROJECT_ID}
      topic: projects/${PROJECT_ID}/topics/storage-events
      eventTypes: ["OBJECT_FINALIZE", "OBJECT_DELETE"]
      payloadFormat: JSON_API_V1
      objectNamePrefix: uploads/

  # IAM Policy for bucket access
  - name: storage-bucket-iam-policy
    type: gcp-types/storage-v1:projects.buckets.iam
    properties:
      bucket: production-storage-bucket-${PROJECT_ID}
      bindings:
        - role: roles/storage.objectViewer
          members:
            - serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com
        - role: roles/storage.objectCreator
          members:
            - serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com
        - role: roles/storage.objectAdmin
          members:
            - user:admin@example.com

  # Cloud CDN for static content
  - name: storage-cdn-backend-bucket
    type: compute.v1.backendBucket
    properties:
      name: storage-cdn-backend-bucket
      bucketName: production-storage-bucket-${PROJECT_ID}
      enableCdn: true
      cdnPolicy:
        cacheMode: CACHE_ALL_STATIC
        defaultTtl: 3600
        maxTtl: 86400
        clientTtl: 3600

  # URL Map for CDN
  - name: storage-cdn-url-map
    type: compute.v1.urlMap
    properties:
      name: storage-cdn-url-map
      defaultService: $(ref.storage-cdn-backend-bucket.selfLink)

  # Target HTTP Proxy
  - name: storage-cdn-target-http-proxy
    type: compute.v1.targetHttpProxy
    properties:
      name: storage-cdn-target-http-proxy
      urlMap: $(ref.storage-cdn-url-map.selfLink)

  # Global Forwarding Rule
  - name: storage-cdn-global-forwarding-rule
    type: compute.v1.globalForwardingRule
    properties:
      name: storage-cdn-global-forwarding-rule
      target: $(ref.storage-cdn-target-http-proxy.selfLink)
      portRange: 80
      IPProtocol: TCP
```

### Storage Bucket Template

```yaml
# storage-bucket.jinja
resources:
  - name: {{ properties["name"] }}
    type: storage.v1.bucket
    properties:
      name: {{ properties["name"] }}
      location: {{ properties["location"] }}
      storageClass: {{ properties["storageClass"] }}
      versioning:
        enabled: {{ properties["versioning"]["enabled"] }}
      lifecycle:
        rule: {{ properties["lifecycle"]["rule"] }}
      cors: {{ properties["cors"] }}
      labels: {{ properties["labels"] }}
      defaultEventBasedHold: false
      retentionPolicy:
        retentionPeriod: 0
      iamConfiguration:
        uniformBucketLevelAccess:
          enabled: true
```

### Cloud Storage with Terraform

```hcl
# cloud-storage.tf
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

# Cloud Storage Bucket
resource "google_storage_bucket" "main" {
  name          = "${var.environment}-storage-bucket-${random_id.bucket_suffix.hex}"
  location      = "US"
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"
    }
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Random ID for bucket suffix
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Cloud Storage Bucket IAM
resource "google_storage_bucket_iam_binding" "viewer" {
  bucket = google_storage_bucket.main.name
  role   = "roles/storage.objectViewer"

  members = [
    "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
  ]
}

resource "google_storage_bucket_iam_binding" "creator" {
  bucket = google_storage_bucket.main.name
  role   = "roles/storage.objectCreator"

  members = [
    "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
  ]
}

# Pub/Sub Topic for storage events
resource "google_pubsub_topic" "storage_events" {
  name = "${var.environment}-storage-events"

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Pub/Sub Subscription
resource "google_pubsub_subscription" "storage_events" {
  name  = "${var.environment}-storage-events-subscription"
  topic = google_pubsub_topic.storage_events.name

  ack_deadline_seconds = 60

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Cloud Storage Notification
resource "google_storage_notification" "main" {
  bucket         = google_storage_bucket.main.name
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.storage_events.id
  event_types    = ["OBJECT_FINALIZE", "OBJECT_DELETE"]
  object_name_prefix = "uploads/"
}

# Cloud Function for processing uploads
resource "google_storage_bucket" "functions" {
  name     = "${var.environment}-functions-${random_id.bucket_suffix.hex}"
  location = "US"
}

# Cloud Function source code
data "archive_file" "function_zip" {
  type        = "zip"
  output_path = "function.zip"
  source {
    content = <<EOF
import json
import os
from google.cloud import storage

def process_upload(event, context):
    """Process uploaded files"""
    bucket_name = event['bucket']
    file_name = event['name']

    print(f"Processing file: gs://{bucket_name}/{file_name}")

    # Initialize Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Get file metadata
    blob.reload()
    size = blob.size
    content_type = blob.content_type

    print(f"File size: {size} bytes")
    print(f"Content type: {content_type}")

    # Process based on file type
    if content_type.startswith('image/'):
        process_image(bucket, blob)
    elif content_type == 'text/plain':
        process_text(bucket, blob)
    elif content_type == 'application/json':
        process_json(bucket, blob)

    return f"Processed {file_name}"

def process_image(bucket, blob):
    """Process image files"""
    print(f"Processing image: {blob.name}")
    # Add image processing logic here

def process_text(bucket, blob):
    """Process text files"""
    print(f"Processing text file: {blob.name}")
    # Add text processing logic here

def process_json(bucket, blob):
    """Process JSON files"""
    print(f"Processing JSON file: {blob.name}")
    # Add JSON processing logic here
EOF
    filename = "main.py"
  }
}

# Upload function code to Cloud Storage
resource "google_storage_bucket_object" "function_zip" {
  name   = "process-upload.zip"
  bucket = google_storage_bucket.functions.name
  source = data.archive_file.function_zip.output_path
}

# Cloud Function
resource "google_cloudfunctions_function" "process_upload" {
  name        = "${var.environment}-process-upload"
  description = "Process uploaded files"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions.name
  source_archive_object = google_storage_bucket_object.function_zip.name
  trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.main.name
  }

  environment_variables = {
    BUCKET_NAME = google_storage_bucket.main.name
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Cloud CDN Backend Bucket
resource "google_compute_backend_bucket" "storage_cdn" {
  name        = "${var.environment}-storage-cdn-backend"
  bucket_name = google_storage_bucket.main.name
  enable_cdn  = true

  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 3600
    max_ttl                      = 86400
    client_ttl                   = 3600
    negative_caching             = true
    serve_while_stale            = 86400
  }
}

# URL Map for CDN
resource "google_compute_url_map" "storage_cdn" {
  name            = "${var.environment}-storage-cdn-url-map"
  default_service = google_compute_backend_bucket.storage_cdn.id
}

# Target HTTP Proxy
resource "google_compute_target_http_proxy" "storage_cdn" {
  name    = "${var.environment}-storage-cdn-proxy"
  url_map = google_compute_url_map.storage_cdn.id
}

# Global Forwarding Rule
resource "google_compute_global_forwarding_rule" "storage_cdn" {
  name       = "${var.environment}-storage-cdn-forwarding-rule"
  target     = google_compute_target_http_proxy.storage_cdn.id
  port_range = "80"
  ip_protocol = "TCP"
}

# Outputs
output "bucket_name" {
  description = "Cloud Storage Bucket Name"
  value       = google_storage_bucket.main.name
}

output "bucket_url" {
  description = "Cloud Storage Bucket URL"
  value       = google_storage_bucket.main.url
}

output "cdn_ip" {
  description = "CDN IP Address"
  value       = google_compute_global_forwarding_rule.storage_cdn.ip_address
}
```

### Cloud Storage Operations with Go

```go
package main

import (
    "context"
    "fmt"
    "io"
    "log"
    "os"
    "path/filepath"
    "strings"
    "time"

    "cloud.google.com/go/storage"
    "google.golang.org/api/iterator"
    "google.golang.org/api/option"
)

type CloudStorageService struct {
    client *storage.Client
    bucket string
}

func NewCloudStorageService(bucket string) (*CloudStorageService, error) {
    ctx := context.Background()
    client, err := storage.NewClient(ctx)
    if err != nil {
        return nil, err
    }

    return &CloudStorageService{
        client: client,
        bucket: bucket,
    }, nil
}

// Upload file to Cloud Storage
func (s *CloudStorageService) UploadFile(ctx context.Context, objectName string, filePath string) error {
    file, err := os.Open(filePath)
    if err != nil {
        return err
    }
    defer file.Close()

    obj := s.client.Bucket(s.bucket).Object(objectName)
    writer := obj.NewWriter(ctx)
    defer writer.Close()

    _, err = io.Copy(writer, file)
    return err
}

// Upload data to Cloud Storage
func (s *CloudStorageService) UploadData(ctx context.Context, objectName string, data []byte) error {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    writer := obj.NewWriter(ctx)
    defer writer.Close()

    _, err := writer.Write(data)
    return err
}

// Download file from Cloud Storage
func (s *CloudStorageService) DownloadFile(ctx context.Context, objectName string, filePath string) error {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    reader, err := obj.NewReader(ctx)
    if err != nil {
        return err
    }
    defer reader.Close()

    file, err := os.Create(filePath)
    if err != nil {
        return err
    }
    defer file.Close()

    _, err = io.Copy(file, reader)
    return err
}

// Download data from Cloud Storage
func (s *CloudStorageService) DownloadData(ctx context.Context, objectName string) ([]byte, error) {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    reader, err := obj.NewReader(ctx)
    if err != nil {
        return nil, err
    }
    defer reader.Close()

    return io.ReadAll(reader)
}

// List objects in bucket
func (s *CloudStorageService) ListObjects(ctx context.Context, prefix string) ([]*storage.ObjectAttrs, error) {
    var objects []*storage.ObjectAttrs

    it := s.client.Bucket(s.bucket).Objects(ctx, &storage.Query{
        Prefix: prefix,
    })

    for {
        obj, err := it.Next()
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, err
        }
        objects = append(objects, obj)
    }

    return objects, nil
}

// Delete object from Cloud Storage
func (s *CloudStorageService) DeleteObject(ctx context.Context, objectName string) error {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    return obj.Delete(ctx)
}

// Copy object within Cloud Storage
func (s *CloudStorageService) CopyObject(ctx context.Context, sourceName, destName string) error {
    src := s.client.Bucket(s.bucket).Object(sourceName)
    dst := s.client.Bucket(s.bucket).Object(destName)

    _, err := dst.CopierFrom(src).Run(ctx)
    return err
}

// Generate signed URL for upload
func (s *CloudStorageService) GenerateSignedUploadURL(ctx context.Context, objectName string, expiration time.Duration) (string, error) {
    obj := s.client.Bucket(s.bucket).Object(objectName)

    opts := &storage.SignedURLOptions{
        Scheme:  storage.SigningSchemeV4,
        Method:  "PUT",
        Expires: time.Now().Add(expiration),
    }

    return obj.SignedURL(opts)
}

// Generate signed URL for download
func (s *CloudStorageService) GenerateSignedDownloadURL(ctx context.Context, objectName string, expiration time.Duration) (string, error) {
    obj := s.client.Bucket(s.bucket).Object(objectName)

    opts := &storage.SignedURLOptions{
        Scheme:  storage.SigningSchemeV4,
        Method:  "GET",
        Expires: time.Now().Add(expiration),
    }

    return obj.SignedURL(opts)
}

// Set object metadata
func (s *CloudStorageService) SetObjectMetadata(ctx context.Context, objectName string, metadata map[string]string) error {
    obj := s.client.Bucket(s.bucket).Object(objectName)

    attrs := &storage.ObjectAttrsToUpdate{
        Metadata: metadata,
    }

    _, err := obj.Update(ctx, attrs)
    return err
}

// Get object metadata
func (s *CloudStorageService) GetObjectMetadata(ctx context.Context, objectName string) (*storage.ObjectAttrs, error) {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    return obj.Attrs(ctx)
}

// Enable versioning
func (s *CloudStorageService) EnableVersioning(ctx context.Context) error {
    bucket := s.client.Bucket(s.bucket)

    attrs := &storage.BucketAttrs{
        VersioningEnabled: true,
    }

    _, err := bucket.Update(ctx, attrs)
    return err
}

// List object versions
func (s *CloudStorageService) ListObjectVersions(ctx context.Context, prefix string) ([]*storage.ObjectAttrs, error) {
    var objects []*storage.ObjectAttrs

    it := s.client.Bucket(s.bucket).Objects(ctx, &storage.Query{
        Prefix: prefix,
        Versions: true,
    })

    for {
        obj, err := it.Next()
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, err
        }
        objects = append(objects, obj)
    }

    return objects, nil
}

// Delete object version
func (s *CloudStorageService) DeleteObjectVersion(ctx context.Context, objectName, generation string) error {
    obj := s.client.Bucket(s.bucket).Object(objectName)
    return obj.Generation(parseInt64(generation)).Delete(ctx)
}

// Set lifecycle policy
func (s *CloudStorageService) SetLifecyclePolicy(ctx context.Context, rules []storage.LifecycleRule) error {
    bucket := s.client.Bucket(s.bucket)

    attrs := &storage.BucketAttrs{
        Lifecycle: storage.Lifecycle{
            Rules: rules,
        },
    }

    _, err := bucket.Update(ctx, attrs)
    return err
}

// Example usage
func main() {
    storageService, err := NewCloudStorageService("my-bucket")
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Upload a file
    err = storageService.UploadFile(ctx, "uploads/file.txt", "/path/to/local/file.txt")
    if err != nil {
        log.Printf("Error uploading file: %v", err)
    }

    // Upload data
    data := []byte("Hello, Cloud Storage!")
    err = storageService.UploadData(ctx, "data/hello.txt", data)
    if err != nil {
        log.Printf("Error uploading data: %v", err)
    }

    // Download a file
    err = storageService.DownloadFile(ctx, "uploads/file.txt", "/path/to/download/file.txt")
    if err != nil {
        log.Printf("Error downloading file: %v", err)
    }

    // List objects
    objects, err := storageService.ListObjects(ctx, "uploads/")
    if err != nil {
        log.Printf("Error listing objects: %v", err)
    } else {
        for _, obj := range objects {
            fmt.Printf("Object: %s, Size: %d, Created: %v\n",
                obj.Name, obj.Size, obj.Created)
        }
    }

    // Generate signed URL
    url, err := storageService.GenerateSignedUploadURL(ctx, "uploads/signed.txt", time.Hour)
    if err != nil {
        log.Printf("Error generating signed URL: %v", err)
    } else {
        fmt.Printf("Signed URL: %s\n", url)
    }

    // Set lifecycle policy
    rules := []storage.LifecycleRule{
        {
            Action: storage.LifecycleAction{
                Type: storage.DeleteAction,
            },
            Condition: storage.LifecycleCondition{
                AgeInDays: 365,
            },
        },
        {
            Action: storage.LifecycleAction{
                Type:         storage.SetStorageClassAction,
                StorageClass: "NEARLINE",
            },
            Condition: storage.LifecycleCondition{
                AgeInDays: 30,
            },
        },
    }

    err = storageService.SetLifecyclePolicy(ctx, rules)
    if err != nil {
        log.Printf("Error setting lifecycle policy: %v", err)
    }
}

func parseInt64(s string) int64 {
    // Simple implementation - in production, use strconv.ParseInt
    return 0
}
```

## 🚀 Best Practices

### 1. Security Best Practices

```yaml
# Secure Cloud Storage configuration
StorageBucket:
  type: storage.v1.bucket
  properties:
    name: secure-storage-bucket
    iamConfiguration:
      uniformBucketLevelAccess:
        enabled: true
    cors:
      - origin: ["https://example.com"]
        method: ["GET", "PUT"]
        responseHeader: ["Content-Type"]
        maxAgeSeconds: 3600
```

### 2. Cost Optimization

```yaml
# Lifecycle configuration for cost optimization
lifecycle:
  rule:
    - action:
        type: SetStorageClass
        storageClass: NEARLINE
      condition:
        age: 30
    - action:
        type: SetStorageClass
        storageClass: COLDLINE
      condition:
        age: 90
    - action:
        type: SetStorageClass
        storageClass: ARCHIVE
      condition:
        age: 365
```

### 3. Performance Optimization

```yaml
# Cloud CDN for performance
BackendBucket:
  type: compute.v1.backendBucket
  properties:
    name: storage-cdn-backend
    bucketName: production-storage-bucket
    enableCdn: true
    cdnPolicy:
      cacheMode: CACHE_ALL_STATIC
      defaultTtl: 3600
      maxTtl: 86400
```

## 🏢 Industry Insights

### Google's Cloud Storage Usage

- **Global Infrastructure**: Multi-region storage
- **Data Analytics**: BigQuery integration
- **Machine Learning**: Training data storage
- **Content Delivery**: CDN integration

### Netflix's GCP Strategy

- **Content Storage**: Video content storage
- **Data Pipeline**: ETL operations
- **Analytics**: User behavior data
- **Global Distribution**: Multi-region replication

### Spotify's Cloud Storage Approach

- **Music Storage**: Audio file storage
- **Data Processing**: ETL pipelines
- **Analytics**: User listening data
- **Machine Learning**: Recommendation systems

## 🎯 Interview Questions

### Basic Level

1. **What is Cloud Storage?**

   - Object storage in Google Cloud
   - Highly scalable and durable
   - Global infrastructure
   - Pay-per-use pricing

2. **What are Cloud Storage classes?**

   - Standard: Frequently accessed data
   - Nearline: Infrequently accessed data
   - Coldline: Archive data
   - Archive: Long-term archive

3. **What is Cloud Storage versioning?**
   - Keep multiple versions of objects
   - Protect against accidental deletion
   - Enable rollback capabilities
   - Additional storage costs

### Intermediate Level

4. **How do you implement Cloud Storage security?**

   ```yaml
   # Secure Cloud Storage bucket
   StorageBucket:
     type: storage.v1.bucket
     properties:
       iamConfiguration:
         uniformBucketLevelAccess:
           enabled: true
       cors:
         - origin: ["https://example.com"]
           method: ["GET", "PUT"]
   ```

5. **How do you optimize Cloud Storage costs?**

   - Lifecycle policies
   - Storage class transitions
   - Intelligent tiering
   - Delete old versions

6. **How do you handle Cloud Storage events?**
   - Cloud Storage notifications
   - Cloud Functions triggers
   - Pub/Sub topics
   - Event-driven processing

### Advanced Level

7. **How do you implement cross-region replication?**

   - Replication configuration
   - IAM roles
   - Source and destination buckets
   - Replication rules

8. **How do you handle Cloud Storage performance?**

   - Cloud CDN integration
   - Transfer acceleration
   - Parallel processing
   - Caching strategies

9. **How do you implement Cloud Storage data lake?**
   - Partitioning strategies
   - Data formats (Parquet, ORC)
   - Query optimization
   - Analytics services integration

---

**Next**: [GCP BigQuery](GCP_BigQuery.md/) - Data warehouse, analytics, machine learning
