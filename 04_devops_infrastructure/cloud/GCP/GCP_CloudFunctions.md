# ⚡ GCP Cloud Functions: Serverless Functions and Event-Driven Architecture

> **Master Google Cloud Functions for serverless computing and event-driven applications**

## 📚 Concept

**Detailed Explanation:**
Google Cloud Functions is a powerful serverless compute service that enables developers to build and deploy event-driven applications without managing infrastructure. It automatically handles server provisioning, scaling, and resource management, allowing developers to focus on writing business logic. Cloud Functions integrates seamlessly with Google Cloud services and responds to events from various sources, making it ideal for building modern, reactive applications.

**Core Philosophy:**

- **Serverless First**: Eliminate infrastructure management and focus on business logic
- **Event-Driven Architecture**: Respond to events from Google Cloud services automatically
- **Auto-Scaling**: Scale from zero to thousands of concurrent executions seamlessly
- **Pay-per-Use**: Only pay for the compute time you actually consume
- **Managed Infrastructure**: Google Cloud handles all infrastructure concerns
- **Developer Productivity**: Focus on code, not infrastructure management

**Why Google Cloud Functions Matters:**

- **Cost Efficiency**: Pay only for actual execution time, not idle server time
- **Automatic Scaling**: Handle traffic spikes without manual intervention
- **Reduced Complexity**: No server management, patching, or capacity planning
- **Faster Time to Market**: Deploy code without infrastructure setup
- **Event-Driven**: Perfect for modern, reactive application architectures
- **Microservices**: Ideal for building loosely coupled, event-driven microservices
- **Integration**: Seamless integration with Google Cloud services
- **Global Availability**: Deploy functions across multiple Google Cloud regions

**Key Features:**

**1. Serverless:**

- **No Server Management**: Google Cloud handles all server provisioning, patching, and maintenance
- **Automatic Infrastructure**: Infrastructure is completely managed by Google Cloud
- **Zero Administration**: No need to manage operating systems, containers, or virtual machines
- **Benefits**: Reduced operational overhead, faster development cycles, lower total cost of ownership
- **Use Cases**: Microservices, event processing, data transformation, API backends

**2. Event-Driven:**

- **Event Sources**: Respond to events from Cloud Storage, Pub/Sub, Firestore, Cloud Scheduler, and more
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
- **Free Tier**: 2M free invocations and 400,000 GB-seconds per month
- **Benefits**: Cost optimization, predictable pricing, no upfront costs
- **Use Cases**: Variable workloads, cost-sensitive applications, proof of concepts

**5. Multiple Runtimes:**

- **Supported Languages**: Python, Node.js, Java, Go, .NET, Ruby, PHP
- **Custom Runtimes**: Bring your own runtime for any language
- **Runtime Updates**: Google Cloud manages runtime updates and security patches
- **Benefits**: Language flexibility, reduced maintenance, security updates
- **Use Cases**: Polyglot applications, legacy system integration, specialized workloads

**6. Integration:**

- **Google Cloud Services**: Native integration with Google Cloud services
- **Event Sources**: Cloud Storage, Pub/Sub, Firestore, Cloud Scheduler, Cloud Build
- **Destinations**: Send results to other Google Cloud services automatically
- **Benefits**: Seamless workflows, reduced complexity, event-driven architecture
- **Use Cases**: Data pipelines, serverless applications, IoT processing

**Advanced Cloud Functions Concepts:**

- **Cold Starts**: Initial latency when functions haven't been used recently
- **Warm Starts**: Faster execution when functions are already loaded
- **Memory Optimization**: Configure memory allocation for optimal performance
- **Timeout Configuration**: Set appropriate timeouts for different workloads
- **Environment Variables**: Secure configuration management
- **IAM Integration**: Fine-grained access control and security
- **VPC Connectivity**: Connect to private networks and resources
- **Cloud Build Integration**: Automated deployment and CI/CD pipelines

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive serverless architecture using Google Cloud Functions for a large-scale, event-driven application with complex data processing requirements?**

**Answer:** Comprehensive serverless architecture design:

- **Event-Driven Design**: Use Cloud Storage, Pub/Sub, Firestore, and Cloud Scheduler for event sources
- **Function Decomposition**: Break down complex logic into smaller, focused functions
- **Data Processing Pipeline**: Use Cloud Functions for ETL operations with BigQuery, Firestore, and Cloud Storage
- **API Integration**: Create RESTful APIs with Cloud Functions and Cloud Endpoints
- **State Management**: Use Firestore for stateful operations and Cloud Workflows for complex workflows
- **Error Handling**: Implement retry mechanisms and dead letter queues
- **Monitoring**: Use Cloud Monitoring, Cloud Logging, and Cloud Trace for observability
- **Security**: Implement IAM roles, VPC configuration, and encryption
- **Performance**: Optimize memory allocation and timeout settings
- **Cost Optimization**: Implement proper resource allocation and monitoring
- **Testing**: Use local testing frameworks and Cloud Build for deployment
- **Documentation**: Maintain comprehensive documentation for function purposes and dependencies

**Q2: What are the key considerations when implementing Google Cloud Functions for a production environment with strict performance, security, and compliance requirements?**

**Answer:** Production Cloud Functions implementation considerations:

- **Performance Optimization**: Optimize cold starts, memory allocation, and timeout settings
- **Security Hardening**: Implement least privilege IAM roles, VPC configuration, and encryption
- **Compliance**: Ensure functions meet regulatory requirements (GDPR, HIPAA, SOX)
- **Monitoring**: Implement comprehensive logging, metrics, and alerting
- **Error Handling**: Use retry mechanisms, dead letter queues, and circuit breakers
- **Cost Management**: Monitor costs, implement proper resource allocation, and use committed use discounts
- **Disaster Recovery**: Implement backup and recovery procedures for function code and configuration
- **Testing**: Implement comprehensive testing including unit, integration, and load testing
- **Documentation**: Maintain detailed documentation for operations and troubleshooting
- **Governance**: Establish policies for function deployment, monitoring, and lifecycle management
- **Incident Response**: Have clear procedures for Cloud Functions-related incidents
- **Regular Reviews**: Conduct regular performance and security reviews

**Q3: How do you optimize Google Cloud Functions for performance, cost, and reliability in enterprise environments?**

**Answer:** Enterprise Cloud Functions optimization strategies:

- **Performance Optimization**: Optimize memory allocation, implement connection pooling, and use appropriate timeout settings
- **Cost Optimization**: Right-size memory allocation, implement proper timeout settings, and use committed use discounts
- **Reliability**: Implement retry mechanisms, dead letter queues, and circuit breakers
- **Monitoring**: Use Cloud Monitoring, Cloud Logging, and Cloud Trace for comprehensive observability
- **Security**: Implement proper IAM roles, VPC configuration, and encryption
- **Error Handling**: Use structured error handling and comprehensive logging
- **Testing**: Implement automated testing and deployment pipelines
- **Documentation**: Maintain comprehensive documentation and runbooks
- **Governance**: Establish policies for function lifecycle management
- **Training**: Provide training for teams on Cloud Functions best practices
- **Regular Reviews**: Conduct regular performance and cost reviews
- **Incident Response**: Have clear procedures for Cloud Functions-related incidents

## 🏗️ Cloud Functions Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Event Sources                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │Cloud Storage│  │  Pub/Sub    │  │  Firestore  │     │
│  │   Events    │  │  Messages   │  │   Changes   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Cloud Function                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Runtime   │  │   Handler   │  │   Memory    │ │ │
│  │  │   Python    │  │   Function  │  │   Config    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │Cloud Storage│  │  Firestore  │  │  Pub/Sub    │     │
│  │   Storage   │  │   Database  │  │  Messaging  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Cloud Function with Deployment Manager

```yaml
# cloud-functions.yaml
imports:
  - path: cloud-function.jinja

resources:
  # Cloud Storage Bucket for function code
  - name: functions-bucket
    type: storage.v1.bucket
    properties:
      name: ${PROJECT_ID}-functions
      location: US
      storageClass: STANDARD

  # Cloud Function for processing uploads
  - name: process-upload-function
    type: cloud-function.jinja
    properties:
      name: process-upload-function
      runtime: python39
      entryPoint: process_upload
      sourceArchiveUrl: gs://${PROJECT_ID}-functions/process-upload.zip
      eventTrigger:
        eventType: google.storage.object.finalize
        resource: ${PROJECT_ID}-uploads
      environmentVariables:
        BUCKET_NAME: ${PROJECT_ID}-uploads
        PROJECT_ID: ${PROJECT_ID}
      labels:
        environment: production
        team: backend

  # Cloud Function for API endpoints
  - name: api-function
    type: cloud-function.jinja
    properties:
      name: api-function
      runtime: python39
      entryPoint: api_handler
      sourceArchiveUrl: gs://${PROJECT_ID}-functions/api.zip
      httpsTrigger: {}
      environmentVariables:
        PROJECT_ID: ${PROJECT_ID}
        FIRESTORE_DATABASE: ${PROJECT_ID}
      labels:
        environment: production
        team: backend

  # Cloud Function for Pub/Sub messages
  - name: pubsub-function
    type: cloud-function.jinja
    properties:
      name: pubsub-function
      runtime: python39
      entryPoint: pubsub_handler
      sourceArchiveUrl: gs://${PROJECT_ID}-functions/pubsub.zip
      eventTrigger:
        eventType: google.pubsub.topic.publish
        resource: projects/${PROJECT_ID}/topics/user-events
      environmentVariables:
        PROJECT_ID: ${PROJECT_ID}
      labels:
        environment: production
        team: backend

  # Cloud Function for Firestore changes
  - name: firestore-function
    type: cloud-function.jinja
    properties:
      name: firestore-function
      runtime: python39
      entryPoint: firestore_handler
      sourceArchiveUrl: gs://${PROJECT_ID}-functions/firestore.zip
      eventTrigger:
        eventType: google.firestore.document.write
        resource: projects/${PROJECT_ID}/databases/(default)/documents/users/{userId}
      environmentVariables:
        PROJECT_ID: ${PROJECT_ID}
      labels:
        environment: production
        team: backend

  # Pub/Sub Topic
  - name: user-events-topic
    type: gcp-types/pubsub-v1:projects.topics
    properties:
      topic: user-events
      labels:
        environment: production
        team: backend

  # Pub/Sub Subscription
  - name: user-events-subscription
    type: gcp-types/pubsub-v1:projects.subscriptions
    properties:
      subscription: user-events-subscription
      topic: projects/${PROJECT_ID}/topics/user-events
      ackDeadlineSeconds: 60
      labels:
        environment: production
        team: backend

  # Firestore Database
  - name: firestore-database
    type: gcp-types/firestore-v1:projects.databases
    properties:
      databaseId: (default)
      locationId: us-central1
      type: FIRESTORE_NATIVE
      labels:
        environment: production
        team: backend

  # Cloud Storage Bucket for uploads
  - name: uploads-bucket
    type: storage.v1.bucket
    properties:
      name: ${PROJECT_ID}-uploads
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
        environment: production
        team: backend

  # IAM Policy for Cloud Functions
  - name: cloud-functions-iam-policy
    type: gcp-types/cloudfunctions-v1:projects.locations.functions.iam
    properties:
      resource: projects/${PROJECT_ID}/locations/us-central1/functions/process-upload-function
      bindings:
        - role: roles/cloudfunctions.invoker
          members:
            - serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com
        - role: roles/storage.objectViewer
          members:
            - serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com
        - role: roles/firestore.user
          members:
            - serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com
```

### Cloud Function Template

```yaml
# cloud-function.jinja
resources:
  - name: {{ properties["name"] }}
    type: gcp-types/cloudfunctions-v1:projects.locations.functions
    properties:
      parent: projects/${PROJECT_ID}/locations/us-central1
      function: {{ properties["name"] }}
      sourceArchiveUrl: {{ properties["sourceArchiveUrl"] }}
      entryPoint: {{ properties["entryPoint"] }}
      runtime: {{ properties["runtime"] }}
      {% if properties["eventTrigger"] %}
      eventTrigger:
        eventType: {{ properties["eventTrigger"]["eventType"] }}
        resource: {{ properties["eventTrigger"]["resource"] }}
      {% endif %}
      {% if properties["httpsTrigger"] %}
      httpsTrigger: {{ properties["httpsTrigger"] }}
      {% endif %}
      environmentVariables: {{ properties["environmentVariables"] }}
      labels: {{ properties["labels"] }}
```

### Cloud Functions with Terraform

```hcl
# cloud-functions.tf
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

# Cloud Storage Bucket for function code
resource "google_storage_bucket" "functions" {
  name     = "${var.project_id}-functions"
  location = "US"
}

# Cloud Function source code for upload processing
data "archive_file" "process_upload_zip" {
  type        = "zip"
  output_path = "process-upload.zip"
  source {
    content = <<EOF
import json
import os
from google.cloud import storage
from google.cloud import firestore

def process_upload(event, context):
    """Process uploaded files"""
    bucket_name = event['bucket']
    file_name = event['name']

    print(f"Processing file: gs://{bucket_name}/{file_name}")

    # Initialize clients
    storage_client = storage.Client()
    firestore_client = firestore.Client()

    # Get file metadata
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.reload()

    size = blob.size
    content_type = blob.content_type
    created = blob.time_created

    print(f"File size: {size} bytes")
    print(f"Content type: {content_type}")
    print(f"Created: {created}")

    # Process based on file type
    if content_type.startswith('image/'):
        result = process_image(bucket, blob)
    elif content_type == 'text/plain':
        result = process_text(bucket, blob)
    elif content_type == 'application/json':
        result = process_json(bucket, blob)
    else:
        result = {"status": "skipped", "reason": "unsupported_type"}

    # Store processing result in Firestore
    doc_ref = firestore_client.collection('file_processing').document(file_name)
    doc_ref.set({
        'file_name': file_name,
        'bucket_name': bucket_name,
        'size': size,
        'content_type': content_type,
        'created': created,
        'processed_at': firestore.SERVER_TIMESTAMP,
        'result': result
    })

    return f"Processed {file_name}: {result}"

def process_image(bucket, blob):
    """Process image files"""
    print(f"Processing image: {blob.name}")
    # Add image processing logic here
    return {"status": "processed", "type": "image"}

def process_text(bucket, blob):
    """Process text files"""
    print(f"Processing text file: {blob.name}")
    # Add text processing logic here
    return {"status": "processed", "type": "text"}

def process_json(bucket, blob):
    """Process JSON files"""
    print(f"Processing JSON file: {blob.name}")
    # Add JSON processing logic here
    return {"status": "processed", "type": "json"}
EOF
    filename = "main.py"
  }
}

# Upload function code to Cloud Storage
resource "google_storage_bucket_object" "process_upload_zip" {
  name   = "process-upload.zip"
  bucket = google_storage_bucket.functions.name
  source = data.archive_file.process_upload_zip.output_path
}

# Cloud Function for processing uploads
resource "google_cloudfunctions_function" "process_upload" {
  name        = "${var.environment}-process-upload"
  description = "Process uploaded files"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions.name
  source_archive_object = google_storage_bucket_object.process_upload_zip.name

  trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.uploads.name
  }

  environment_variables = {
    BUCKET_NAME = google_storage_bucket.uploads.name
    PROJECT_ID  = var.project_id
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Cloud Function source code for API
data "archive_file" "api_zip" {
  type        = "zip"
  output_path = "api.zip"
  source {
    content = <<EOF
import json
import os
from google.cloud import firestore
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "api-function"})

@app.route('/users', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()

        if not data or 'name' not in data or 'email' not in data:
            return jsonify({"error": "Missing required fields"}), 400

        # Initialize Firestore client
        firestore_client = firestore.Client()

        # Create user document
        doc_ref = firestore_client.collection('users').document()
        doc_ref.set({
            'name': data['name'],
            'email': data['email'],
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            "message": "User created successfully",
            "user_id": doc_ref.id
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    try:
        firestore_client = firestore.Client()
        doc_ref = firestore_client.collection('users').document(user_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = doc.to_dict()
        user_data['id'] = doc.id

        return jsonify(user_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user by ID"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        firestore_client = firestore.Client()
        doc_ref = firestore_client.collection('users').document(user_id)

        # Check if user exists
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "User not found"}), 404

        # Update user
        data['updated_at'] = firestore.SERVER_TIMESTAMP
        doc_ref.update(data)

        return jsonify({"message": "User updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user by ID"""
    try:
        firestore_client = firestore.Client()
        doc_ref = firestore_client.collection('users').document(user_id)

        # Check if user exists
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "User not found"}), 404

        # Delete user
        doc_ref.delete()

        return jsonify({"message": "User deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def api_handler(request):
    """Cloud Function entry point"""
    return app(request.environ, lambda *args: None)
EOF
    filename = "main.py"
  }
}

# Upload API function code to Cloud Storage
resource "google_storage_bucket_object" "api_zip" {
  name   = "api.zip"
  bucket = google_storage_bucket.functions.name
  source = data.archive_file.api_zip.output_path
}

# Cloud Function for API
resource "google_cloudfunctions_function" "api" {
  name        = "${var.environment}-api"
  description = "API endpoints"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions.name
  source_archive_object = google_storage_bucket_object.api_zip.name

  https_trigger {}

  environment_variables = {
    PROJECT_ID = var.project_id
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Cloud Function source code for Pub/Sub
data "archive_file" "pubsub_zip" {
  type        = "zip"
  output_path = "pubsub.zip"
  source {
    content = <<EOF
import json
import base64
from google.cloud import firestore
from google.cloud import pubsub_v1

def pubsub_handler(event, context):
    """Handle Pub/Sub messages"""
    try:
        # Decode the message
        if 'data' in event:
            message = base64.b64decode(event['data']).decode('utf-8')
        else:
            message = event.get('data', '')

        print(f"Received message: {message}")

        # Parse JSON message
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            data = {"message": message}

        # Initialize Firestore client
        firestore_client = firestore.Client()

        # Store message in Firestore
        doc_ref = firestore_client.collection('pubsub_messages').document()
        doc_ref.set({
            'message': message,
            'data': data,
            'received_at': firestore.SERVER_TIMESTAMP,
            'event_id': context.event_id,
            'event_type': context.event_type
        })

        # Process the message
        result = process_message(data)

        # Update document with result
        doc_ref.update({
            'result': result,
            'processed_at': firestore.SERVER_TIMESTAMP
        })

        print(f"Processed message: {result}")
        return f"Processed message: {result}"

    except Exception as e:
        print(f"Error processing message: {e}")
        raise e

def process_message(data):
    """Process the message data"""
    message_type = data.get('type', 'unknown')

    if message_type == 'user_created':
        return handle_user_created(data)
    elif message_type == 'user_updated':
        return handle_user_updated(data)
    elif message_type == 'user_deleted':
        return handle_user_deleted(data)
    else:
        return {"status": "unknown_type", "type": message_type}

def handle_user_created(data):
    """Handle user created event"""
    user_id = data.get('user_id')
    print(f"User created: {user_id}")
    return {"status": "processed", "action": "user_created", "user_id": user_id}

def handle_user_updated(data):
    """Handle user updated event"""
    user_id = data.get('user_id')
    print(f"User updated: {user_id}")
    return {"status": "processed", "action": "user_updated", "user_id": user_id}

def handle_user_deleted(data):
    """Handle user deleted event"""
    user_id = data.get('user_id')
    print(f"User deleted: {user_id}")
    return {"status": "processed", "action": "user_deleted", "user_id": user_id}
EOF
    filename = "main.py"
  }
}

# Upload Pub/Sub function code to Cloud Storage
resource "google_storage_bucket_object" "pubsub_zip" {
  name   = "pubsub.zip"
  bucket = google_storage_bucket.functions.name
  source = data.archive_file.pubsub_zip.output_path
}

# Pub/Sub Topic
resource "google_pubsub_topic" "user_events" {
  name = "${var.environment}-user-events"

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Pub/Sub Subscription
resource "google_pubsub_subscription" "user_events" {
  name  = "${var.environment}-user-events-subscription"
  topic = google_pubsub_topic.user_events.name

  ack_deadline_seconds = 60

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Cloud Function for Pub/Sub
resource "google_cloudfunctions_function" "pubsub" {
  name        = "${var.environment}-pubsub"
  description = "Handle Pub/Sub messages"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions.name
  source_archive_object = google_storage_bucket_object.pubsub_zip.name

  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.user_events.name
  }

  environment_variables = {
    PROJECT_ID = var.project_id
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Firestore Database
resource "google_firestore_database" "main" {
  project     = var.project_id
  name        = "(default)"
  location_id = "us-central1"
  type        = "FIRESTORE_NATIVE"
}

# Cloud Storage Bucket for uploads
resource "google_storage_bucket" "uploads" {
  name     = "${var.project_id}-uploads"
  location = "US"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    team        = "backend"
  }
}

# Outputs
output "api_function_url" {
  description = "API Function URL"
  value       = google_cloudfunctions_function.api.https_trigger_url
}

output "process_upload_function_name" {
  description = "Process Upload Function Name"
  value       = google_cloudfunctions_function.process_upload.name
}

output "pubsub_function_name" {
  description = "Pub/Sub Function Name"
  value       = google_cloudfunctions_function.pubsub.name
}
```

### Cloud Functions with Go

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"

    "cloud.google.com/go/firestore"
    "cloud.google.com/go/pubsub"
    "cloud.google.com/go/storage"
    "github.com/GoogleCloudPlatform/functions-framework-go/funcframework"
)

type User struct {
    ID        string `json:"id"`
    Name      string `json:"name"`
    Email     string `json:"email"`
    CreatedAt string `json:"created_at"`
}

type Message struct {
    Type   string      `json:"type"`
    UserID string      `json:"user_id"`
    Data   interface{} `json:"data"`
}

type CloudFunctionHandler struct {
    firestoreClient *firestore.Client
    storageClient   *storage.Client
    pubsubClient    *pubsub.Client
    projectID       string
}

func NewCloudFunctionHandler() (*CloudFunctionHandler, error) {
    ctx := context.Background()
    projectID := os.Getenv("PROJECT_ID")

    firestoreClient, err := firestore.NewClient(ctx, projectID)
    if err != nil {
        return nil, err
    }

    storageClient, err := storage.NewClient(ctx)
    if err != nil {
        return nil, err
    }

    pubsubClient, err := pubsub.NewClient(ctx, projectID)
    if err != nil {
        return nil, err
    }

    return &CloudFunctionHandler{
        firestoreClient: firestoreClient,
        storageClient:   storageClient,
        pubsubClient:    pubsubClient,
        projectID:       projectID,
    }, nil
}

// HTTP Cloud Function
func (h *CloudFunctionHandler) APIHandler(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        h.handleGet(w, r)
    case http.MethodPost:
        h.handlePost(w, r)
    case http.MethodPut:
        h.handlePut(w, r)
    case http.MethodDelete:
        h.handleDelete(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func (h *CloudFunctionHandler) handleGet(w http.ResponseWriter, r *http.Request) {
    if r.URL.Path == "/" {
        // Health check
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]string{
            "status":  "healthy",
            "service": "api-function",
        })
        return
    }

    // Get user by ID
    userID := r.URL.Path[1:] // Remove leading slash
    if userID == "" {
        http.Error(w, "User ID required", http.StatusBadRequest)
        return
    }

    ctx := context.Background()
    doc, err := h.firestoreClient.Collection("users").Doc(userID).Get(ctx)
    if err != nil {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    var user User
    if err := doc.DataTo(&user); err != nil {
        http.Error(w, "Error parsing user data", http.StatusInternalServerError)
        return
    }

    user.ID = doc.Ref.ID
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (h *CloudFunctionHandler) handlePost(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    if user.Name == "" || user.Email == "" {
        http.Error(w, "Name and email are required", http.StatusBadRequest)
        return
    }

    ctx := context.Background()
    docRef, _, err := h.firestoreClient.Collection("users").Add(ctx, user)
    if err != nil {
        http.Error(w, "Error creating user", http.StatusInternalServerError)
        return
    }

    // Publish user created event
    h.publishEvent(ctx, Message{
        Type:   "user_created",
        UserID: docRef.ID,
        Data:   user,
    })

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(map[string]string{
        "message": "User created successfully",
        "user_id": docRef.ID,
    })
}

func (h *CloudFunctionHandler) handlePut(w http.ResponseWriter, r *http.Request) {
    userID := r.URL.Path[1:] // Remove leading slash
    if userID == "" {
        http.Error(w, "User ID required", http.StatusBadRequest)
        return
    }

    var updates map[string]interface{}
    if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    ctx := context.Background()
    docRef := h.firestoreClient.Collection("users").Doc(userID)

    // Check if user exists
    doc, err := docRef.Get(ctx)
    if err != nil {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    // Update user
    _, err = docRef.Update(ctx, []firestore.Update{
        {Path: "name", Value: updates["name"]},
        {Path: "email", Value: updates["email"]},
    })
    if err != nil {
        http.Error(w, "Error updating user", http.StatusInternalServerError)
        return
    }

    // Publish user updated event
    h.publishEvent(ctx, Message{
        Type:   "user_updated",
        UserID: userID,
        Data:   updates,
    })

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "message": "User updated successfully",
    })
}

func (h *CloudFunctionHandler) handleDelete(w http.ResponseWriter, r *http.Request) {
    userID := r.URL.Path[1:] // Remove leading slash
    if userID == "" {
        http.Error(w, "User ID required", http.StatusBadRequest)
        return
    }

    ctx := context.Background()
    docRef := h.firestoreClient.Collection("users").Doc(userID)

    // Check if user exists
    doc, err := docRef.Get(ctx)
    if err != nil {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    // Delete user
    _, err = docRef.Delete(ctx)
    if err != nil {
        http.Error(w, "Error deleting user", http.StatusInternalServerError)
        return
    }

    // Publish user deleted event
    h.publishEvent(ctx, Message{
        Type:   "user_deleted",
        UserID: userID,
        Data:   nil,
    })

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "message": "User deleted successfully",
    })
}

func (h *CloudFunctionHandler) publishEvent(ctx context.Context, message Message) {
    topic := h.pubsubClient.Topic("user-events")

    data, err := json.Marshal(message)
    if err != nil {
        log.Printf("Error marshaling message: %v", err)
        return
    }

    result := topic.Publish(ctx, &pubsub.Message{
        Data: data,
    })

    if _, err := result.Get(ctx); err != nil {
        log.Printf("Error publishing message: %v", err)
    }
}

// Pub/Sub Cloud Function
func (h *CloudFunctionHandler) PubSubHandler(ctx context.Context, m *pubsub.Message) error {
    var message Message
    if err := json.Unmarshal(m.Data, &message); err != nil {
        log.Printf("Error unmarshaling message: %v", err)
        return err
    }

    log.Printf("Received message: %+v", message)

    // Process the message
    result := h.processMessage(message)

    // Store result in Firestore
    _, err := h.firestoreClient.Collection("pubsub_messages").Add(ctx, map[string]interface{}{
        "message":     string(m.Data),
        "result":      result,
        "received_at": firestore.ServerTimestamp,
        "event_id":    m.ID,
    })
    if err != nil {
        log.Printf("Error storing message: %v", err)
        return err
    }

    log.Printf("Processed message: %+v", result)
    return nil
}

func (h *CloudFunctionHandler) processMessage(message Message) map[string]interface{} {
    switch message.Type {
    case "user_created":
        return h.handleUserCreated(message)
    case "user_updated":
        return h.handleUserUpdated(message)
    case "user_deleted":
        return h.handleUserDeleted(message)
    default:
        return map[string]interface{}{
            "status": "unknown_type",
            "type":   message.Type,
        }
    }
}

func (h *CloudFunctionHandler) handleUserCreated(message Message) map[string]interface{} {
    log.Printf("User created: %s", message.UserID)
    return map[string]interface{}{
        "status": "processed",
        "action": "user_created",
        "user_id": message.UserID,
    }
}

func (h *CloudFunctionHandler) handleUserUpdated(message Message) map[string]interface{} {
    log.Printf("User updated: %s", message.UserID)
    return map[string]interface{}{
        "status": "processed",
        "action": "user_updated",
        "user_id": message.UserID,
    }
}

func (h *CloudFunctionHandler) handleUserDeleted(message Message) map[string]interface{} {
    log.Printf("User deleted: %s", message.UserID)
    return map[string]interface{}{
        "status": "processed",
        "action": "user_deleted",
        "user_id": message.UserID,
    }
}

func main() {
    handler, err := NewCloudFunctionHandler()
    if err != nil {
        log.Fatal(err)
    }

    // Register HTTP function
    funcframework.RegisterHTTPFunction("/", handler.APIHandler)

    // Register Pub/Sub function
    funcframework.RegisterCloudEventFunction("pubsub-handler", handler.PubSubHandler)

    // Start the server
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    if err := funcframework.Start(port); err != nil {
        log.Fatal(err)
    }
}
```

## 🚀 Best Practices

### 1. Performance Optimization

```yaml
# Optimize Cloud Function configuration
CloudFunction:
  type: gcp-types/cloudfunctions-v1:projects.locations.functions
  properties:
    runtime: python39
    entryPoint: main
    availableMemoryMb: 512
    timeout: 60s
    environmentVariables:
      LOG_LEVEL: INFO
    maxInstances: 10
```

### 2. Error Handling

```python
import json
import logging
from google.cloud import firestore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def cloud_function_handler(event, context):
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
# Secure Cloud Function configuration
CloudFunction:
  type: gcp-types/cloudfunctions-v1:projects.locations.functions
  properties:
    runtime: python39
    entryPoint: main
    environmentVariables:
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
    labels:
      environment: production
      team: backend
```

## 🏢 Industry Insights

### Google's Cloud Functions Usage

- **Event Processing**: Real-time data processing
- **Microservices**: Serverless microservices
- **Data Pipeline**: ETL operations
- **Cost Optimization**: Pay-per-use model

### Netflix's GCP Strategy

- **Content Processing**: Video content processing
- **Data Analytics**: Real-time analytics
- **API Gateway**: Serverless APIs
- **Event-Driven**: Event-driven architecture

### Spotify's Cloud Functions Approach

- **Music Processing**: Audio file processing
- **Recommendations**: Real-time recommendations
- **Data Pipeline**: ETL operations
- **Cost Efficiency**: Serverless computing

## 🎯 Interview Questions

### Basic Level

1. **What are Cloud Functions?**

   - Serverless compute service
   - Event-driven execution
   - Auto-scaling
   - Pay-per-use pricing

2. **What are Cloud Function triggers?**

   - HTTP requests
   - Cloud Storage events
   - Pub/Sub messages
   - Firestore changes

3. **What are Cloud Function limitations?**
   - 9-minute execution timeout
   - 8GB memory limit
   - 512MB temporary storage
   - Cold start latency

### Intermediate Level

4. **How do you optimize Cloud Function performance?**

   ```yaml
   # Performance optimization
   CloudFunction:
     type: gcp-types/cloudfunctions-v1:projects.locations.functions
     properties:
       availableMemoryMb: 512
       timeout: 60s
       maxInstances: 10
       environmentVariables:
         LOG_LEVEL: INFO
   ```

5. **How do you handle Cloud Function errors?**

   - Try-catch blocks
   - Dead letter queues
   - Retry mechanisms
   - Error logging

6. **How do you secure Cloud Functions?**
   - IAM roles with least privilege
   - VPC configuration
   - Environment variables encryption
   - Input validation

### Advanced Level

7. **How do you implement Cloud Function patterns?**

   - Fan-out pattern
   - SAGA pattern
   - Event sourcing
   - CQRS pattern

8. **How do you handle Cloud Function cold starts?**

   - Connection pooling
   - Keep-alive strategies
   - Warming functions
   - Memory optimization

9. **How do you implement Cloud Function testing?**
   - Unit testing
   - Integration testing
   - Local testing
   - Mock Google Cloud services

---

**Next**: [GCP Cloud SQL](GCP_CloudSQL.md/) - Managed databases, read replicas, backups
